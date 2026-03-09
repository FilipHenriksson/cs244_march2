import json
from openai import AsyncOpenAI
import cost_tracker as ct
from cost_tracker import CostLimitExceeded

MODEL = "gpt-4.1-nano"
client = AsyncOpenAI()

# Active scheduler — set by main.py before any calls
_scheduler = None
# Default max_tokens for output; set via set_max_tokens() from CLI
_max_tokens = 4096


def set_scheduler(scheduler):
    global _scheduler
    _scheduler = scheduler


def set_max_tokens(max_tokens: int):
    global _max_tokens
    _max_tokens = max_tokens


def estimate_tokens(messages, **kwargs) -> int:
    """Rough token estimate: input (chars/4) + output (max_tokens)."""
    text = json.dumps(messages, default=str)
    if "tools" in kwargs:
        text += json.dumps(kwargs["tools"], default=str)
    input_tokens = len(text) // 4
    est_output = kwargs.get("max_tokens", _max_tokens)
    return input_tokens + est_output


RATELIMIT_HEADERS = [
    "x-ratelimit-limit-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-requests",
    "x-ratelimit-reset-tokens",
]


async def fetch_ratelimit_headers(max_tokens: int = 1) -> dict[str, str]:
    """Send a minimal chat completion request and return OpenAI rate limit headers."""
    raw = await client.chat.completions.with_raw_response.create(
        model=MODEL,
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=max_tokens,
    )
    return {h: raw.headers[h] for h in RATELIMIT_HEADERS if h in raw.headers}


async def llm_call(messages, agent_id: str, call_type: str,
                   detail: str = "", group_id: str = None, **kwargs):
    """Single entry point for all LLM calls. Routes through the active scheduler."""
    kwargs.setdefault("max_tokens", _max_tokens)
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
