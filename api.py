"""LLM scheduling proxy — thin server for multi-client workloads.

Endpoints
---------
POST   /call_types                          – register a reusable (name, system_prompt) pair
POST   /sessions                            – create a session, returns session_id
POST   /sessions/{id}/completions           – submit one LLM call through the scheduler
POST   /sessions/{id}/completions/batch     – fan-out multiple LLM calls concurrently
DELETE /sessions/{id}                       – tear down a session

Simulation / benchmarking endpoints (not for production use)
-------------------------------------------------------------
GET    /sim/stats                           – server-side metrics snapshot
POST   /sim/reset                           – reset state for a fresh benchmark run
GET    /sim/config                          – current server configuration
PATCH  /sim/config                          – update config (max_tokens, rpm, tpm)
GET    /sim/events                          – SSE stream of global scheduler events (all sessions)

The server owns scheduling, rate limiting, and system-prompt caching.
Clients drive their own agent loops and conversation state.

Authentication
--------------
All requests require a Bearer token via the ``Authorization`` header.
Set ``PROXY_API_KEY`` in the environment (or ``.env``); the server refuses
to start if it is missing.  Clients must send::

    Authorization: Bearer <PROXY_API_KEY>

SSE endpoints also accept ``?token=<PROXY_API_KEY>`` as a query parameter
since browser EventSource does not support custom headers.

Configuration (environment variables)
--------------------------------------
PROXY_API_KEY   – required auth token (no default; must be set)
SCHEDULER       – scheduler name (default: "fifo")
RPM             – initial requests per minute (default: 30; updatable via PATCH /sim/config)
TPM             – initial tokens per minute (default: 200000; updatable via PATCH /sim/config)
MAX_TOKENS      – per-completion output cap (default: 2048)
COST_LIMIT      – hard dollar budget (default: 20.0)
MODEL           – OpenAI model identifier (default: "gpt-4.1-nano")
"""

import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from openai import AsyncOpenAI, APIConnectionError, BadRequestError, RateLimitError, APIError
from pydantic import BaseModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
for _quiet in ("openai", "httpx", "httpcore"):
    logging.getLogger(_quiet).setLevel(logging.WARNING)
logger = logging.getLogger("api")

from dotenv import load_dotenv
load_dotenv()

from sim import RateLimiter, init_cost_tracker
from schedulers import get_scheduler
from events import global_bus, session_buses
import sim.cost_tracker as ct

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

SCHEDULER_NAME = os.getenv("SCHEDULER", "fifo")
COST_LIMIT = float(os.getenv("COST_LIMIT", "20.0"))
MODEL = os.getenv("MODEL", "gpt-4.1-nano")

# Mutable rate-limit settings (defaults from env; can be updated via PATCH /sim/config)
_rpm = int(os.getenv("RPM", "30"))
_tpm = int(os.getenv("TPM", "200000"))
API_KEY = os.getenv("PROXY_API_KEY")
if not API_KEY:
    raise RuntimeError("PROXY_API_KEY environment variable is required")

# Mutable max_tokens (default from env; can be updated via PATCH /sim/config)
_max_tokens = int(os.getenv("MAX_TOKENS", "2048"))

# ---------------------------------------------------------------------------
# OpenAI client (shared across all requests)
# ---------------------------------------------------------------------------

_openai = AsyncOpenAI(timeout=1200)

# ---------------------------------------------------------------------------
# Call-type registry: name -> system_prompt
# ---------------------------------------------------------------------------

_call_types: dict[str, str] = {}
_call_types_lock = asyncio.Lock()

# ---------------------------------------------------------------------------
# Session registry: UUID -> {"created_at": str, "int_id": int}
# ---------------------------------------------------------------------------

_sessions: dict[str, dict] = {}
_next_int_id = 0

# ---------------------------------------------------------------------------
# Server-wide scheduler and rate limiter (set during lifespan)
# ---------------------------------------------------------------------------

_scheduler = None
_limiter = None
_active_scheduler_name = SCHEDULER_NAME

# ---------------------------------------------------------------------------
# Token estimation (mirrors llm.py logic)
# ---------------------------------------------------------------------------


def _estimate_tokens(messages: list[dict], call_key: str,
                     max_tokens: int) -> int:
    if _scheduler is not None and hasattr(_scheduler, "estimate_total_tokens"):
        return _scheduler.estimate_total_tokens(
            messages, call_key, max_tokens=max_tokens)
    text = json.dumps(messages, default=str)
    return len(text) // 4 + max_tokens


# ---------------------------------------------------------------------------
# Core dispatch: submit one LLM call through the scheduler
# ---------------------------------------------------------------------------

_DISPATCH_MAX_RETRIES = 3
_DISPATCH_RETRY_BACKOFF = 2.0


async def _dispatch(messages: list[dict], session_int_id: int,
                    call_key: str, max_tokens: int) -> dict:
    """Build a coro_factory, submit to the scheduler, return parsed result.

    Retries on transient OpenAI errors (JSON-parse 400s, 429 rate limits,
    5xx) with exponential backoff.
    """
    est_tokens = _estimate_tokens(messages, call_key, max_tokens)

    if ct.cost_tracker is not None:
        await ct.cost_tracker.check()

    last_exc: Exception | None = None
    for attempt in range(_DISPATCH_MAX_RETRIES):
        try:
            def coro_factory():
                return _openai.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    max_tokens=max_tokens,
                )

            response = await _scheduler.submit(
                coro_factory, est_tokens, session_int_id, call_key,
                label=call_key,
            )
            break
        except BadRequestError as exc:
            last_exc = exc
            if "could not parse the json body" in str(exc).lower():
                delay = _DISPATCH_RETRY_BACKOFF * (2 ** attempt)
                logger.warning(
                    "Transient OpenAI JSON-parse error on %s (attempt %d/%d), "
                    "retrying in %.1fs: %s",
                    call_key, attempt + 1, _DISPATCH_MAX_RETRIES, delay, exc,
                )
                await asyncio.sleep(delay)
                continue
            logger.error("OpenAI BadRequestError on %s: %s", call_key, exc)
            raise
        except RateLimitError as exc:
            last_exc = exc
            retry_after = None
            if exc.response is not None:
                header = exc.response.headers.get("retry-after")
                if header:
                    try:
                        retry_after = float(header)
                    except (ValueError, TypeError):
                        pass
            delay = retry_after or _DISPATCH_RETRY_BACKOFF * (2 ** attempt)
            logger.warning(
                "OpenAI 429 rate limit on %s (attempt %d/%d), "
                "sleeping %.1fs (retry-after=%s)",
                call_key, attempt + 1, _DISPATCH_MAX_RETRIES,
                delay, retry_after,
            )
            await asyncio.sleep(delay)
            continue
        except APIConnectionError as exc:
            last_exc = exc
            delay = _DISPATCH_RETRY_BACKOFF * (2 ** attempt)
            logger.warning(
                "OpenAI connection error on %s (attempt %d/%d), "
                "retrying in %.1fs: %s",
                call_key, attempt + 1, _DISPATCH_MAX_RETRIES, delay, exc,
            )
            await asyncio.sleep(delay)
            continue
        except APIError as exc:
            last_exc = exc
            status = getattr(exc, "status_code", None)
            if status is not None and status >= 500:
                delay = _DISPATCH_RETRY_BACKOFF * (2 ** attempt)
                logger.warning(
                    "OpenAI %d error on %s (attempt %d/%d), retrying in %.1fs",
                    status, call_key,
                    attempt + 1, _DISPATCH_MAX_RETRIES, delay,
                )
                await asyncio.sleep(delay)
                continue
            logger.error("OpenAI APIError on %s (status=%s): %s",
                         call_key, status, exc)
            raise
    else:
        logger.error("OpenAI error on %s: retries exhausted (%d/%d): %s",
                     call_key, _DISPATCH_MAX_RETRIES,
                     _DISPATCH_MAX_RETRIES, last_exc)
        raise last_exc  # type: ignore[misc]

    usage = None
    if response.usage is not None:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        if ct.cost_tracker is not None:
            await ct.cost_tracker.record(
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )

    return {
        "content": response.choices[0].message.content,
        "usage": usage,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _scheduler, _limiter, _active_scheduler_name
    _limiter = RateLimiter(rpm=_rpm, tpm=_tpm)
    _active_scheduler_name = SCHEDULER_NAME
    _scheduler = get_scheduler(_active_scheduler_name, _limiter)
    _scheduler.start()
    init_cost_tracker(COST_LIMIT)
    yield
    await _scheduler.stop()
    global_bus.close()
    session_buses.close_all()


app = FastAPI(title="CS244 LLM Scheduling Proxy", lifespan=lifespan)

# CORS for browser-based frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _check_auth(request: Request) -> bool:
    """Check Bearer header OR query param token."""
    if request.headers.get("Authorization") == f"Bearer {API_KEY}":
        return True
    if request.query_params.get("token") == API_KEY:
        return True
    return False


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if not _check_auth(request):
        return Response(status_code=401, content="Unauthorized")
    return await call_next(request)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class RegisterCallTypeRequest(BaseModel):
    name: str
    system_prompt: str


class SessionResponse(BaseModel):
    session_id: str


class CompletionRequest(BaseModel):
    call_type: str
    messages: list[dict]
    max_tokens: int | None = None
    call_detail: str | None = None


class CompletionResponse(BaseModel):
    content: str
    usage: dict | None = None


class BatchCompletionRequest(BaseModel):
    calls: list[CompletionRequest]


class BatchCompletionResponse(BaseModel):
    completions: list[CompletionResponse]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_session(session_id: str) -> int:
    """Return the int_id for a session UUID, or raise 404."""
    meta = _sessions.get(session_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return meta["int_id"]


def _build_messages(call_type: str, user_messages: list[dict]) -> list[dict]:
    """Prepend the registered system prompt to the caller-supplied messages."""
    system_prompt = _call_types.get(call_type)
    if system_prompt is None:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown call_type: {call_type!r}. Register it first via POST /call_types.",
        )
    return [{"role": "system", "content": system_prompt}] + user_messages


# ---------------------------------------------------------------------------
# SSE event stream helpers
# ---------------------------------------------------------------------------

_SSE_HEARTBEAT_INTERVAL = 15  # seconds


async def _sse_generator(queue: asyncio.Queue):
    """Yield SSE-formatted events from an async queue.

    Sends a heartbeat comment every 15s to keep the connection alive
    through proxies and load balancers.
    """
    try:
        while True:
            try:
                event = await asyncio.wait_for(
                    queue.get(), timeout=_SSE_HEARTBEAT_INTERVAL
                )
            except asyncio.TimeoutError:
                yield ": heartbeat\n\n"
                continue

            if event is None:
                # Sentinel — session closed or server shutting down
                yield f"event: done\ndata: {{}}\n\n"
                return

            event_type = event.get("type", "message")
            yield f"event: {event_type}\ndata: {json.dumps(event)}\n\n"
    except asyncio.CancelledError:
        return


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/call_types", status_code=201)
async def register_call_type(req: RegisterCallTypeRequest):
    async with _call_types_lock:
        exists = req.name in _call_types
        _call_types[req.name] = req.system_prompt
    status = "updated" if exists else "created"
    return {"name": req.name, "status": status}


@app.get("/call_types")
async def list_call_types():
    return {"call_types": list(_call_types.keys())}


@app.post("/sessions", response_model=SessionResponse)
async def create_session():
    global _next_int_id
    sid = str(uuid.uuid4())
    int_id = _next_int_id
    _next_int_id += 1
    _sessions[sid] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "int_id": int_id,
    }
    # Pre-create the session event bus so it's ready before any calls
    session_buses.get_or_create(int_id)
    return SessionResponse(session_id=sid)


@app.get("/sim/events", tags=["simulation"])
async def global_events():
    """SSE stream of all scheduler events across all sessions.

    Includes per-call lifecycle events (enqueued, dispatched, completed,
    failed, rate_limited) and periodic queue_snapshot events showing the
    full pending queue and in-flight calls.

    Intended for the demo frontend to visualize the global queue state.

    Auth: supports ``?token=<key>`` for browser EventSource compatibility.
    """
    queue = global_bus.subscribe()

    async def cleanup_generator():
        try:
            async for chunk in _sse_generator(queue):
                yield chunk
        finally:
            global_bus.unsubscribe(queue)

    return StreamingResponse(
        cleanup_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/sessions/{session_id}/completions", response_model=CompletionResponse)
async def submit_completion(session_id: str, req: CompletionRequest):
    int_id = _resolve_session(session_id)
    messages = _build_messages(req.call_type, req.messages)
    max_tokens = req.max_tokens or _max_tokens
    call_key = f"{req.call_type}:{req.call_detail}" if req.call_detail else req.call_type
    result = await _dispatch(messages, int_id, call_key, max_tokens)
    return CompletionResponse(**result)


@app.post("/sessions/{session_id}/completions/batch",
          response_model=BatchCompletionResponse)
async def submit_batch(session_id: str, req: BatchCompletionRequest):
    """Fan-out multiple LLM calls concurrently within a single session.

    Emits a ``batch_progress`` event to the session bus each time a
    sub-call completes, so SSE listeners can track incremental progress
    (e.g. "3/5 analysts done").

    Error handling
    --------------
    Each sub-call is dispatched through ``_dispatch``, which already retries
    transient OpenAI errors (JSON-parse 400s, 5xx) up to
    ``_DISPATCH_MAX_RETRIES`` times with exponential backoff.

    ``asyncio.gather`` is called with ``return_exceptions=True`` so that a
    failure in one sub-call does not cancel or abort the others — all sub-calls
    run to completion independently.  After gathering:

    * If every sub-call succeeded, the full batch of completions is returned.
    * If any sub-call still failed after its retries were exhausted, the
      endpoint returns **502** with a detail string listing how many sub-calls
      failed out of the total (all-or-nothing: partial successes are discarded).
    """
    if not req.calls:
        return BatchCompletionResponse(completions=[])

    int_id = _resolve_session(session_id)
    total = len(req.calls)
    completed_count = 0
    completed_keys: list[str] = []
    _progress_lock = asyncio.Lock()

    async def _one(call: CompletionRequest) -> dict:
        nonlocal completed_count
        messages = _build_messages(call.call_type, call.messages)
        max_tokens = call.max_tokens or _max_tokens
        call_key = f"{call.call_type}:{call.call_detail}" if call.call_detail else call.call_type
        result = await _dispatch(messages, int_id, call_key, max_tokens)

        # Emit batch_progress event
        async with _progress_lock:
            completed_count += 1
            completed_keys.append(call_key)
            session_buses.emit(int_id, {
                "type": "batch_progress",
                "session_id": int_id,
                "completed": completed_count,
                "total": total,
                "call_keys_done": list(completed_keys),
                "latest": call_key,
                "content": result["content"],
            })
            global_bus.emit({
                "type": "batch_progress",
                "session_id": int_id,
                "completed": completed_count,
                "total": total,
                "latest": call_key,
            })

        return result

    results = await asyncio.gather(*[_one(c) for c in req.calls],
                                   return_exceptions=True)

    first_exc = next((r for r in results if isinstance(r, Exception)), None)
    if first_exc is not None:
        failed = [(i, r) for i, r in enumerate(results) if isinstance(r, Exception)]
        for i, exc in failed:
            logger.error("Batch sub-call %d (%s) failed: %s",
                         i, req.calls[i].call_type, exc)
        raise HTTPException(
            status_code=502,
            detail=f"{len(failed)}/{len(req.calls)} batch sub-calls failed: {first_exc}",
        )

    return BatchCompletionResponse(
        completions=[CompletionResponse(**r) for r in results],
    )


@app.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    int_id = _sessions[session_id]["int_id"]
    del _sessions[session_id]
    # Close the session's event bus (sends sentinel to SSE listeners)
    session_buses.close(int_id)
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Simulation / benchmarking endpoints
# ---------------------------------------------------------------------------


class SimResetRequest(BaseModel):
    scheduler: str | None = None


class SimConfigUpdateRequest(BaseModel):
    max_tokens: int | None = None
    rpm: int | None = None
    tpm: int | None = None


@app.get("/sim/config", tags=["simulation"])
async def sim_config():
    """Return current server configuration (simulation endpoint)."""
    return {
        "scheduler": _active_scheduler_name,
        "rpm": _rpm,
        "tpm": _tpm,
        "max_tokens": _max_tokens,
        "cost_limit": COST_LIMIT,
        "model": MODEL,
    }


@app.patch("/sim/config", tags=["simulation"])
async def sim_config_update(req: SimConfigUpdateRequest):
    """Update server configuration (simulation endpoint).

    Only provided fields are updated.  Supported: max_tokens, rpm, tpm.
    When rpm or tpm change the rate limiter and scheduler are re-created so
    the new limits take effect immediately.
    """
    global _max_tokens, _rpm, _tpm, _limiter, _scheduler
    if req.max_tokens is not None:
        if req.max_tokens < 1:
            raise HTTPException(status_code=422, detail="max_tokens must be >= 1")
        _max_tokens = req.max_tokens

    rebuild_limiter = False
    if req.rpm is not None:
        if req.rpm < 1:
            raise HTTPException(status_code=422, detail="rpm must be >= 1")
        _rpm = req.rpm
        rebuild_limiter = True
    if req.tpm is not None:
        if req.tpm < 1:
            raise HTTPException(status_code=422, detail="tpm must be >= 1")
        _tpm = req.tpm
        rebuild_limiter = True

    if rebuild_limiter and _limiter is not None:
        if _scheduler is not None:
            await _scheduler.stop()
        _limiter = RateLimiter(rpm=_rpm, tpm=_tpm)
        _scheduler = get_scheduler(_active_scheduler_name, _limiter)
        _scheduler.start()
        logger.info("Rate limiter re-created: rpm=%d tpm=%d", _rpm, _tpm)

    return {"max_tokens": _max_tokens, "rpm": _rpm, "tpm": _tpm}


@app.get("/sim/stats", tags=["simulation"])
async def sim_stats():
    """Snapshot of server-side metrics (simulation endpoint)."""
    rl = _limiter.stats if _limiter is not None else {}

    cost = {}
    if ct.cost_tracker is not None:
        cost = {
            "call_count": ct.cost_tracker.call_count,
            "total_input_tokens": ct.cost_tracker.total_input_tokens,
            "total_output_tokens": ct.cost_tracker.total_output_tokens,
            "total_cost": ct.cost_tracker.total_cost,
            "limit": ct.cost_tracker.limit,
        }

    sched_stats = {}
    if _scheduler is not None and hasattr(_scheduler, "stats"):
        sched_stats = _scheduler.stats

    return {
        "scheduler": _active_scheduler_name,
        "rate_limiter": rl,
        "cost": cost,
        "scheduler_stats": sched_stats,
        "active_sessions": len(_sessions),
        "sse_global_subscribers": global_bus.subscriber_count,
    }


@app.post("/sim/reset", tags=["simulation"])
async def sim_reset(req: SimResetRequest | None = None):
    """Reset server state for a fresh benchmark run (simulation endpoint).

    Stops the current scheduler, resets the rate limiter and cost tracker,
    clears all sessions, and (optionally) switches to a different scheduler.
    """
    global _scheduler, _limiter, _active_scheduler_name, _sessions, _next_int_id

    if _scheduler is not None:
        await _scheduler.stop()

    # Close all session event buses (sends sentinels to any SSE listeners)
    session_buses.close_all()

    _limiter = RateLimiter(rpm=_rpm, tpm=_tpm)
    init_cost_tracker(COST_LIMIT)
    _sessions = {}
    _next_int_id = 0

    new_name = (req.scheduler if req and req.scheduler else _active_scheduler_name)
    _active_scheduler_name = new_name
    _scheduler = get_scheduler(_active_scheduler_name, _limiter)
    _scheduler.start()

    return {
        "status": "reset",
        "scheduler": _active_scheduler_name,
    }