import json
import os
from openai import AsyncOpenAI
import cost_tracker as ct
from cost_tracker import CostLimitExceeded

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")
client = AsyncOpenAI()

# Active scheduler — set by main.py before any calls
_scheduler = None

CALL_TYPE_TO_ROLE = {
    "planner_plan": "planner",
    "executor_node": "executor",
    "executor_finalize": "executor",
}

ROLE_TO_MODEL_ENV = {
    "default": "OPENAI_MODEL",
    "planner": "PLANNER_MODEL",
    "executor": "EXECUTOR_MODEL",
}


def set_scheduler(scheduler):
    global _scheduler
    _scheduler = scheduler


def estimate_tokens(messages, **kwargs) -> int:
    """Rough token estimate: chars / 4."""
    text = json.dumps(messages, default=str)
    if "tools" in kwargs:
        text += json.dumps(kwargs["tools"], default=str)
    return len(text) // 4


def _resolve_model(call_type: str, kwargs: dict) -> tuple[str, dict]:
    """
    Resolve a model without changing the public llm_call() contract.

    Custom routing hints like llm_role are consumed here and never forwarded
    to the provider SDK.
    """
    request_kwargs = dict(kwargs)
    explicit_role = request_kwargs.pop("llm_role", None)
    explicit_model = request_kwargs.pop("model", None)

    if explicit_model is not None:
        return explicit_model, request_kwargs

    role = explicit_role or CALL_TYPE_TO_ROLE.get(call_type, "default")
    env_var = ROLE_TO_MODEL_ENV.get(role, ROLE_TO_MODEL_ENV["default"])
    model = os.getenv(env_var) or MODEL
    return model, request_kwargs


async def llm_call(messages, agent_id: str, call_type: str,
                   detail: str = "", group_id: str = None, **kwargs):
    """Single entry point for all LLM calls. Routes through the active scheduler."""
    model, request_kwargs = _resolve_model(call_type, kwargs)
    est_tokens = estimate_tokens(messages, **request_kwargs)

    # Pre-flight cost check
    if ct.cost_tracker is not None:
        await ct.cost_tracker.check()

    def coro_factory():
        return client.chat.completions.create(
            model=model,
            messages=messages,
            **request_kwargs,
        )

    response = await _scheduler.submit(
        coro_factory, est_tokens, agent_id, call_type, detail,
        group_id=group_id,
    )

    # Post-call: record real token usage and correct rate limiter
    if response.usage is not None:
        actual_total = response.usage.prompt_tokens + response.usage.completion_tokens
        await _scheduler.limiter.report_actual(est_tokens, actual_total)
        if ct.cost_tracker is not None:
            await ct.cost_tracker.record(
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )

    return response
