"""FastAPI application exposing the research agent over HTTP.

Endpoints
---------
POST /sessions/start          – create a new session, returns session_id
POST /sessions/{id}/run_agent – run the research agent for a session
DELETE /sessions/{id}         – delete a session

The scheduler, rate limiter, and cost tracker are created at startup via the
FastAPI lifespan and wired into llm.py via set_scheduler().

The /run_agent endpoint delegates directly to agent.run_agent(), which supports
two modes:
  "default" — LLM freely picks which analyst/reviewer tools to call.
  "strict"  — deterministic 5-analyst + 3-reviewer pipeline (11 LLM calls).
"""

import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()

from sim import RateLimiter, init_cost_tracker
from schedulers import get_scheduler
from llm import set_scheduler
from agent import run_agent

# ---------------------------------------------------------------------------
# In-memory session registry
# ---------------------------------------------------------------------------

# Maps UUID string -> {"created_at": str, "int_id": int}
sessions: dict[str, dict] = {}
_next_session_int_id = 0


def _alloc_session() -> tuple[str, int]:
    """Create a new session and return (uuid_str, int_id)."""
    global _next_session_int_id
    sid = str(uuid.uuid4())
    int_id = _next_session_int_id
    _next_session_int_id += 1
    sessions[sid] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "int_id": int_id,
    }
    return sid, int_id


# ---------------------------------------------------------------------------
# Lifespan: start scheduler once, stop on shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    limiter = RateLimiter(rpm=20, tpm=100_000)
    scheduler = get_scheduler("fifo", limiter)
    set_scheduler(scheduler)
    scheduler.start()
    init_cost_tracker(15.0)
    yield
    await scheduler.stop()


app = FastAPI(title="CS244c LLM Scheduler API", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class SessionStartResponse(BaseModel):
    session_id: str


class RunAgentRequest(BaseModel):
    prompt: str
    prompt_mode: str = "default"  # "default" or "strict"


class RunAgentResponse(BaseModel):
    response: str
    llm_calls: int
    tool_calls: int


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/sessions/start", response_model=SessionStartResponse)
async def start_session():
    sid, _ = _alloc_session()
    return SessionStartResponse(session_id=sid)


@app.post("/sessions/{session_id}/run_agent", response_model=RunAgentResponse)
async def run_agent_endpoint(session_id: str, req: RunAgentRequest):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    if req.prompt_mode not in ("default", "strict"):
        raise HTTPException(
            status_code=422,
            detail="prompt_mode must be 'default' or 'strict'",
        )

    int_id = sessions[session_id]["int_id"]
    result = await run_agent(
        prompt=req.prompt,
        session_id=int_id,
        prompt_mode=req.prompt_mode,
    )
    return RunAgentResponse(
        response=result.text,
        llm_calls=result.llm_calls,
        tool_calls=result.tool_calls,
    )


@app.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del sessions[session_id]
    return Response(status_code=204)
