import asyncio
import random
from sim.rate_limiter import RateLimiter, THROTTLE_RPM, THROTTLE_TPM
from sim.trace import trace


class BackoffScheduler:
    """Strategy 1: No global queue. Each agent retries with exponential backoff."""

    def __init__(self, limiter: RateLimiter):
        self.limiter = limiter
        self._rng = random.Random()  # own RNG so global seed doesn't affect jitter

    def start(self):
        pass

    async def stop(self):
        pass

    async def submit(self, coro_factory, estimated_tokens: int,
                     session_id: int, call_key: str, label: str = ""):
        """Submit a call. Retries with exponential backoff if rate limited."""
        backoff = 1.0
        retries = 0
        rpm_waits = 0
        tpm_waits = 0

        while True:
            throttle = await self.limiter.try_acquire(estimated_tokens)
            if throttle is None:
                call = trace.start_call(session_id, call_key, label,
                                        retries=retries,
                                        rpm_waits=rpm_waits,
                                        tpm_waits=tpm_waits)
                try:
                    result = await coro_factory()
                    if hasattr(result, "usage") and result.usage is not None:
                        await self.limiter.record_actual_usage(
                            result.usage.prompt_tokens,
                            result.usage.completion_tokens,
                            estimated_tokens,
                        )
                    trace.end_call(call)
                    return result
                except Exception:
                    await self.limiter.record_actual_usage(
                        0, 0, estimated_tokens)
                    trace.end_call(call)
                    raise

            if throttle == THROTTLE_RPM:
                rpm_waits += 1
            else:
                tpm_waits += 1
            jitter = self._rng.uniform(0, backoff * 0.5)
            wait = backoff + jitter
            retries += 1
            print(f"  [BACKOFF] s{session_id}:{call_key} rate limited ({throttle}), "
                  f"retry #{retries} in {wait:.2f}s")
            await asyncio.sleep(wait)
            backoff = min(backoff * 2, 30.0)
