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
PATCH  /sim/config                          – update config (e.g. max_tokens)

The server owns scheduling, rate limiting, and system-prompt caching.
Clients drive their own agent loops and conversation state.

Authentication
--------------
All requests require a Bearer token via the ``Authorization`` header.
Set ``PROXY_API_KEY`` in the environment (or ``.env``); the server refuses
to start if it is missing.  Clients must send::

    Authorization: Bearer <PROXY_API_KEY>

Configuration (environment variables)
--------------------------------------
PROXY_API_KEY   – required auth token (no default; must be set)
SCHEDULER       – scheduler name (default: "fifo")
RPM             – requests per minute (default: 30)
TPM             – tokens per minute (default: 200000)
MAX_TOKENS      – per-completion output cap (default: 2048)
COST_LIMIT      – hard dollar budget (default: 20.0)
MODEL           – OpenAI model identifier (default: "gpt-4.1-nano")
"""

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from openai import AsyncOpenAI
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()

from sim import RateLimiter, init_cost_tracker
from schedulers import get_scheduler
import sim.cost_tracker as ct

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

SCHEDULER_NAME = os.getenv("SCHEDULER", "fifo")
RPM = int(os.getenv("RPM", "30"))
TPM = int(os.getenv("TPM", "200000"))
COST_LIMIT = float(os.getenv("COST_LIMIT", "20.0"))
MODEL = os.getenv("MODEL", "gpt-4.1-nano")
API_KEY = os.getenv("PROXY_API_KEY")
if not API_KEY:
    raise RuntimeError("PROXY_API_KEY environment variable is required")

# Mutable max_tokens (default from env; can be updated via PATCH /sim/config)
_max_tokens = int(os.getenv("MAX_TOKENS", "2048"))

# ---------------------------------------------------------------------------
# OpenAI client (shared across all requests)
# ---------------------------------------------------------------------------

_openai = AsyncOpenAI()

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

async def _dispatch(messages: list[dict], session_int_id: int,
                    call_key: str, max_tokens: int) -> dict:
    """Build a coro_factory, submit to the scheduler, return parsed result."""
    est_tokens = _estimate_tokens(messages, call_key, max_tokens)

    if ct.cost_tracker is not None:
        await ct.cost_tracker.check()

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
    _limiter = RateLimiter(rpm=RPM, tpm=TPM)
    _active_scheduler_name = SCHEDULER_NAME
    _scheduler = get_scheduler(_active_scheduler_name, _limiter)
    _scheduler.start()
    init_cost_tracker(COST_LIMIT)
    yield
    await _scheduler.stop()


app = FastAPI(title="CS244 LLM Scheduling Proxy", lifespan=lifespan)


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if request.headers.get("Authorization") != f"Bearer {API_KEY}":
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
    return SessionResponse(session_id=sid)


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
    if not req.calls:
        return BatchCompletionResponse(completions=[])

    int_id = _resolve_session(session_id)

    async def _one(call: CompletionRequest) -> dict:
        messages = _build_messages(call.call_type, call.messages)
        max_tokens = call.max_tokens or _max_tokens
        call_key = f"{call.call_type}:{call.call_detail}" if call.call_detail else call.call_type
        return await _dispatch(messages, int_id, call_key, max_tokens)

    results = await asyncio.gather(*[_one(c) for c in req.calls])
    return BatchCompletionResponse(
        completions=[CompletionResponse(**r) for r in results],
    )


@app.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str):
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del _sessions[session_id]
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Simulation / benchmarking endpoints
#
# These endpoints expose server-side metrics and allow resetting state
# between benchmark runs.  They are intended for simulation use only and
# should NOT be exposed in a production deployment.
# ---------------------------------------------------------------------------


class SimResetRequest(BaseModel):
    scheduler: str | None = None


class SimConfigUpdateRequest(BaseModel):
    max_tokens: int | None = None


@app.get("/sim/config", tags=["simulation"])
async def sim_config():
    """Return current server configuration (simulation endpoint)."""
    return {
        "scheduler": _active_scheduler_name,
        "rpm": RPM,
        "tpm": TPM,
        "max_tokens": _max_tokens,
        "cost_limit": COST_LIMIT,
        "model": MODEL,
    }


@app.patch("/sim/config", tags=["simulation"])
async def sim_config_update(req: SimConfigUpdateRequest):
    """Update server configuration (simulation endpoint).

    Only provided fields are updated.  Supported: max_tokens.
    """
    global _max_tokens
    if req.max_tokens is not None:
        if req.max_tokens < 1:
            raise HTTPException(status_code=422, detail="max_tokens must be >= 1")
        _max_tokens = req.max_tokens
    return {"max_tokens": _max_tokens}


@app.get("/sim/stats", tags=["simulation"])
async def sim_stats():
    """Snapshot of server-side metrics (simulation endpoint).

    Returns rate-limiter counters, cost-tracker totals, scheduler stats,
    and the number of active sessions.
    """
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

    _limiter.reset()
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
