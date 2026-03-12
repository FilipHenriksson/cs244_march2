"""FastAPI application exposing LLM scheduler capabilities over HTTP.

Endpoints
---------
POST /tools/register          – register a tool with its webhook URL and arg schema
POST /agents/register         – register an agent with a system prompt and tool list
POST /sessions/start          – create a new session, returns session_id
POST /sessions/{id}/run_llm   – single LLM call for a session
POST /sessions/{id}/run_agent – agent tool-use loop for a session
DELETE /sessions/{id}         – delete a session

All registries are in-memory (module-level dicts). A single FIFO scheduler +
rate limiter + cost tracker is created at startup via the FastAPI lifespan and
wired into llm.py via set_scheduler().
"""

import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()

from rate_limiter import RateLimiter
from schedulers import get_scheduler
from llm import set_scheduler, llm_call
import cost_tracker as ct
from cost_tracker import init_cost_tracker

# ---------------------------------------------------------------------------
# In-memory registries
# ---------------------------------------------------------------------------

# tool_id -> {"webhook": str, "args": dict (OpenAI parameters JSON schema)}
tools_registry: dict[str, dict] = {}

# agent_id -> {"system_prompt": str, "tools": list[str]}
agents_registry: dict[str, dict] = {}

# session_id -> {"created_at": str}
sessions: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Lifespan: start scheduler once, stop on shutdown
# ---------------------------------------------------------------------------

_scheduler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _scheduler
    limiter = RateLimiter(rpm=20, tpm=100_000)
    _scheduler = get_scheduler("fifo", limiter)
    set_scheduler(_scheduler)
    _scheduler.start()
    init_cost_tracker(15.0)
    yield
    await _scheduler.stop()


app = FastAPI(title="CS244c LLM Scheduler API", lifespan=lifespan)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

MAX_AGENT_TOOL_ROUNDS = 3


class ToolRegisterRequest(BaseModel):
    tool_id: str
    webhook: str
    args: dict  # OpenAI function parameters JSON schema


class ToolRegisterResponse(BaseModel):
    registered: bool


class AgentRegisterRequest(BaseModel):
    agent_id: str
    system_prompt: str
    tools: list[str]


class AgentRegisterResponse(BaseModel):
    registered: bool


class SessionStartResponse(BaseModel):
    session_id: str


class RunLLMRequest(BaseModel):
    prompt: str


class RunLLMResponse(BaseModel):
    response: str


class RunAgentRequest(BaseModel):
    agent_id: str
    prompt: str


class RunAgentResponse(BaseModel):
    response: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/tools/register", response_model=ToolRegisterResponse)
async def register_tool(req: ToolRegisterRequest):
    tools_registry[req.tool_id] = {"webhook": req.webhook, "args": req.args}
    return ToolRegisterResponse(registered=True)


@app.post("/agents/register", response_model=AgentRegisterResponse)
async def register_agent(req: AgentRegisterRequest):
    agents_registry[req.agent_id] = {
        "system_prompt": req.system_prompt,
        "tools": req.tools,
    }
    return AgentRegisterResponse(registered=True)


@app.post("/sessions/start", response_model=SessionStartResponse)
async def start_session():
    session_id = str(uuid.uuid4())
    sessions[session_id] = {"created_at": datetime.now(timezone.utc).isoformat()}
    return SessionStartResponse(session_id=session_id)


@app.post("/sessions/{session_id}/run_llm", response_model=RunLLMResponse)
async def run_llm(session_id: str, req: RunLLMRequest):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    resp = await llm_call(
        messages=[{"role": "user", "content": req.prompt}],
        agent_id=f"session:{session_id}",
        call_type="run_llm",
    )
    content = resp.choices[0].message.content
    return RunLLMResponse(response=content if content is not None else "")


@app.post("/sessions/{session_id}/run_agent", response_model=RunAgentResponse)
async def run_agent(session_id: str, req: RunAgentRequest):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    agent = agents_registry.get(req.agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent not found")

    # Build OpenAI tool definitions from registered tools the agent has access to
    tool_schemas = []
    for tool_id in agent["tools"]:
        tool_def = tools_registry.get(tool_id)
        if tool_def is None:
            continue
        tool_schemas.append({
            "type": "function",
            "function": {
                "name": tool_id,
                "parameters": tool_def["args"],
            },
        })

    messages = [
        {"role": "system", "content": agent["system_prompt"]},
        {"role": "user", "content": req.prompt},
    ]

    aid = f"session:{session_id}:agent:{req.agent_id}"

    async with httpx.AsyncClient(timeout=60.0) as http:
        for round_num in range(MAX_AGENT_TOOL_ROUNDS):
            kwargs = {}
            if tool_schemas:
                kwargs["tools"] = tool_schemas

            resp = await llm_call(
                messages=messages,
                agent_id=aid,
                call_type="agent_turn",
                detail=f"round-{round_num}",
                **kwargs,
            )
            msg = resp.choices[0].message

            # No tool calls — LLM produced final text answer
            if not msg.tool_calls:
                content = msg.content
                return RunAgentResponse(response=content if content is not None else "")

            messages.append(msg)

            for tool_call in msg.tool_calls:
                tool_id = tool_call.function.name
                tool_def = tools_registry.get(tool_id)

                if tool_def is None:
                    tool_result = f"Error: tool '{tool_id}' is not registered"
                else:
                    try:
                        call_args = json.loads(tool_call.function.arguments)
                        webhook_resp = await http.post(
                            tool_def["webhook"], json=call_args
                        )
                        webhook_resp.raise_for_status()
                        tool_result = webhook_resp.text
                    except Exception as exc:
                        tool_result = f"Error calling tool '{tool_id}': {exc}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result,
                })

    # Final LLM call after exhausting tool rounds (no tools offered)
    final_resp = await llm_call(
        messages=messages,
        agent_id=aid,
        call_type="agent_turn",
        detail="final",
    )
    content = final_resp.choices[0].message.content
    return RunAgentResponse(response=content if content is not None else "")


@app.delete("/sessions/{session_id}", status_code=204)
async def delete_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    del sessions[session_id]
    return Response(status_code=204)
