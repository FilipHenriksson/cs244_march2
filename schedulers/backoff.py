import asyncio
import random
from rate_limiter import RateLimiter
from trace import trace


class BackoffScheduler:
    """Strategy 1: No global queue. Each agent retries with exponential backoff."""

    def __init__(self, limiter: RateLimiter):
        self.limiter = limiter
        self._rng = random.Random()  # own RNG so global seed doesn't affect jitter

    def start(self):
        pass

    async def stop(self):
        pass

    def register_group(self, group_id: str, size: int):
        pass

    def deregister_member(self, group_id: str):
        pass

    async def submit(self, coro_factory, estimated_tokens: int,
                     agent_id: str, call_type: str, detail: str = "",
                     group_id: str = None):
        """Submit a call. Retries with exponential backoff if rate limited."""
        backoff = 1.0
        retries = 0

        while True:
            allowed, acquire_ts = await self.limiter.try_acquire(estimated_tokens)
            if allowed:
                call = trace.start_call(agent_id, call_type, detail,
                                        retries=retries)
                result = await coro_factory()
                if hasattr(result, "usage") and result.usage is not None:
                    await self.limiter.record_actual_usage(
                        result.usage.prompt_tokens,
                        result.usage.completion_tokens,
                        estimated_tokens,
                        acquire_time=acquire_ts,
                    )
                trace.end_call(call)
                return result

            # Rate limited — backoff with jitter
            jitter = self._rng.uniform(0, backoff * 0.5)
            wait = backoff + jitter
            retries += 1
            print(f"  [BACKOFF] {agent_id}:{call_type} rate limited, "
                  f"retry #{retries} in {wait:.2f}s")
            await asyncio.sleep(wait)
            backoff = min(backoff * 2, 30.0)
