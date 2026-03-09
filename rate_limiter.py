import asyncio
import logging
import time

log = logging.getLogger("rate_limiter")


class RateLimiter:
    """Token-bucket rate limiter with continuous refill for RPM and TPM."""

    def __init__(self, rpm: int = 20, tpm: int = 100_000):
        self.rpm = rpm
        self.tpm = tpm

        # Bucket levels — start full
        self._rpm_bucket: float = float(rpm)
        self._tpm_bucket: float = float(tpm)

        # Refill rates (units per second)
        self._rpm_rate: float = rpm / 60.0
        self._tpm_rate: float = tpm / 60.0

        self._last_refill: float = time.time()
        self._lock = asyncio.Lock()

    def _refill(self, now: float) -> None:
        """Add tokens for elapsed time since last refill, capped at bucket capacity."""
        elapsed = now - self._last_refill
        if elapsed > 0:
            self._rpm_bucket = min(self.rpm, self._rpm_bucket + elapsed * self._rpm_rate)
            self._tpm_bucket = min(self.tpm, self._tpm_bucket + elapsed * self._tpm_rate)
            self._last_refill = now

    async def try_acquire(self, estimated_tokens: int) -> bool:
        """Try to consume 1 request and estimated_tokens from the buckets.
        Returns True if capacity was available and consumed, False otherwise."""
        async with self._lock:
            self._refill(time.time())
            if self._rpm_bucket < 1.0:
                return False
            if self._tpm_bucket < estimated_tokens:
                return False
            self._rpm_bucket -= 1.0
            self._tpm_bucket -= estimated_tokens
            return True

    async def record_actual_usage(self, prompt_tokens: int, completion_tokens: int,
                                  estimated_tokens: int) -> None:
        """Adjust the token bucket after a request completes.

        If estimated > actual, tokens are returned to the bucket (capped at tpm).
        If estimated < actual, additional tokens are consumed (bucket may go negative).
        On failure (prompt_tokens=0, completion_tokens=0), the full estimate is returned.
        """
        actual = prompt_tokens + completion_tokens
        delta = estimated_tokens - actual  # positive = overestimate = give back
        if delta != 0:
            async with self._lock:
                self._tpm_bucket = min(self.tpm, self._tpm_bucket + delta)
                log.info("[RATELIMIT] actual_usage: prompt=%d completion=%d actual=%d "
                         "est=%d delta=%+d tpm_bucket=%.1f",
                         prompt_tokens, completion_tokens, actual,
                         estimated_tokens, delta, self._tpm_bucket)

    async def wait_time(self, estimated_tokens: int = 0) -> float:
        """Return exact seconds until both buckets can satisfy the request."""
        async with self._lock:
            self._refill(time.time())
            waits = []
            if self._rpm_bucket < 1.0:
                rpm_deficit = 1.0 - self._rpm_bucket
                waits.append(rpm_deficit / self._rpm_rate)
            if self._tpm_bucket < estimated_tokens:
                tpm_deficit = estimated_tokens - self._tpm_bucket
                waits.append(tpm_deficit / self._tpm_rate)
            return max(waits) if waits else 0.0

    def reset(self):
        """Reset buckets to full capacity for a fresh run."""
        self._rpm_bucket = float(self.rpm)
        self._tpm_bucket = float(self.tpm)
        self._last_refill = time.time()
