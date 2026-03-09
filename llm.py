import json
from openai import AsyncOpenAI
import cost_tracker as ct
from cost_tracker import CostLimitExceeded

MODEL = "gpt-4.1-nano"
client = AsyncOpenAI()

# Active scheduler — set by main.py before any calls
_scheduler = None


def set_scheduler(scheduler):
    global _scheduler
    _scheduler = scheduler


def estimate_tokens(messages, **kwargs) -> int:
    """Rough token estimate: chars / 4."""
    text = json.dumps(messages, default=str)
    if "tools" in kwargs:
        text += json.dumps(kwargs["tools"], default=str)
    return len(text) // 4


RATELIMIT_HEADERS = [
    "x-ratelimit-limit-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-requests",
    "x-ratelimit-reset-tokens",
]


async def fetch_ratelimit_headers() -> dict[str, str]:
    """Send a minimal chat completion request and return OpenAI rate limit headers."""
    raw = await client.chat.completions.with_raw_response.create(
        model=MODEL,
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=1,
    )
    return {h: raw.headers[h] for h in RATELIMIT_HEADERS if h in raw.headers}


async def llm_call(messages, agent_id: str, call_type: str,
                   detail: str = "", group_id: str = None, **kwargs):
    """Single entry point for all LLM calls. Routes through the active scheduler."""
    est_tokens = estimate_tokens(messages, **kwargs)

    # Pre-flight cost check
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

    # Post-call: record real token usage
    if ct.cost_tracker is not None and response.usage is not None:
        await ct.cost_tracker.record(
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
        )

    return response
