"""LLM call dispatch — single entry point for all OpenAI API calls.

All agent and orchestrator code calls llm_call() instead of the OpenAI client
directly.  llm_call() handles:
  - Token estimation (for scheduler prioritization and rate-limiter budgeting)
  - Pre-flight cost checks (via cost_tracker)
  - Routing through the active scheduler (backoff, FIFO, MapReduce, etc.)
  - Post-call cost recording

The active scheduler is set once at startup by main.py / sim/runner.py via
set_scheduler().
"""

import json
import os
from openai import AsyncOpenAI
import sim.cost_tracker as ct

MODEL = "gpt-4.1-nano"
client = AsyncOpenAI()

_scheduler = None          # set once at startup via set_scheduler()
_max_tokens = 2048         # default output cap; overridable via set_max_tokens()


def set_scheduler(scheduler):
    global _scheduler
    _scheduler = scheduler


def set_max_tokens(max_tokens: int):
    global _max_tokens
    _max_tokens = max_tokens


# ---------------------------------------------------------------------------
# Token estimation & LLM dispatch
# ---------------------------------------------------------------------------

def estimate_tokens(messages, **kwargs) -> int:
    """Rough token estimate: input chars/4 + expected output (max_tokens)."""
    text = json.dumps(messages, default=str)
    if "tools" in kwargs:
        text += json.dumps(kwargs["tools"], default=str)
    return len(text) // 4 + kwargs.get("max_tokens", _max_tokens)


async def llm_call(messages, session_id: int, call_key: str,
                   label: str = "", **kwargs):
    """Single entry point for all LLM calls.  Routes through the active scheduler.

    Parameters
    ----------
    messages : list
        OpenAI chat messages.
    session_id : int
        Which session this call belongs to (used for scheduler priority).
    call_key : str
        Call type identifier, e.g. ``"orchestrator:plan"``,
        ``"analyst_web_research"``.  Used for token-learning EMA and logging.
    label : str, optional
        Human-readable context for log lines (e.g. truncated question text).
    """
    kwargs.setdefault("max_tokens", _max_tokens)

    if hasattr(_scheduler, "estimate_total_tokens"):
        est_tokens = _scheduler.estimate_total_tokens(
            messages, call_key, **kwargs)
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
        coro_factory, est_tokens, session_id, call_key, label,
    )

    if ct.cost_tracker is not None and response.usage is not None:
        await ct.cost_tracker.record(
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )

    return response
