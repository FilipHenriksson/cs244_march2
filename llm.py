"""LLM call dispatch — single entry point for all OpenAI API calls.

All agent and orchestrator code calls llm_call() instead of the OpenAI client
directly.  llm_call() handles:
  - Token estimation (for scheduler prioritization and rate-limiter budgeting)
  - Pre-flight cost checks (via cost_tracker)
  - Routing through the active scheduler (backoff, FIFO, SJF, MapReduce, etc.)
  - Post-call cost recording

The active scheduler is set once at startup by main.py / batch_runner.py via
set_scheduler().  Group lifecycle helpers (register_group / deregister_member)
are forwarded to the scheduler so that callers don't need direct access to the
scheduler instance.
"""

import json
import os
from openai import AsyncOpenAI
import cost_tracker as ct
from cost_tracker import CostLimitExceeded

MODEL = "gpt-4.1-nano"
client = AsyncOpenAI()

_scheduler = None          # set once at startup via set_scheduler()
_max_tokens = 1024         # default output cap; overridable via set_max_tokens()


def set_scheduler(scheduler):
    global _scheduler
    _scheduler = scheduler


def set_max_tokens(max_tokens: int):
    global _max_tokens
    _max_tokens = max_tokens


# ---------------------------------------------------------------------------
# Group lifecycle — forwarded to the active scheduler so that callers
# (orchestrator.py) don't reach into _scheduler directly.
# ---------------------------------------------------------------------------

def register_group(group_id: str, size: int):
    """Declare a fan-out group of *size* members with the active scheduler."""
    _scheduler.register_group(group_id, size)


def deregister_member(group_id: str):
    """Signal that one member of a fan-out group has completed."""
    _scheduler.deregister_member(group_id)


# ---------------------------------------------------------------------------
# Token estimation & LLM dispatch
# ---------------------------------------------------------------------------

def estimate_tokens(messages, **kwargs) -> int:
    """Rough token estimate: input chars/4 + expected output (max_tokens)."""
    text = json.dumps(messages, default=str)
    if "tools" in kwargs:
        text += json.dumps(kwargs["tools"], default=str)
    return len(text) // 4


async def llm_call(messages, agent_id: str, call_type: str,
                   detail: str = "", group_id: str = None, **kwargs):
    """Single entry point for all LLM calls.  Routes through the active scheduler."""
    kwargs.setdefault("max_tokens", _max_tokens)

    if hasattr(_scheduler, "estimate_total_tokens"):
        est_tokens = _scheduler.estimate_total_tokens(
            messages, call_type, detail, **kwargs)
    else:
        est_tokens = estimate_tokens(messages, **kwargs)

    if ct.cost_tracker is not None:
        await ct.cost_tracker.check()

    def coro_factory():
        return client.chat.completions.create(
            model=MODEL,
            messages=messages,
            **kwargs,
        )

    response = await _scheduler.submit(
        coro_factory, est_tokens, agent_id, call_type, detail,
        group_id=group_id,
    )

    if ct.cost_tracker is not None and response.usage is not None:
        await ct.cost_tracker.record(
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )

    return response
