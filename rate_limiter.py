import asyncio
import logging
import time
from collections import deque

log = logging.getLogger("rate_limiter")


class RateLimiter:
    """Simulates OpenAI-style rate limits with RPM and TPM sliding windows."""

    def __init__(self, rpm: int = 20, tpm: int = 100_000):
        self.rpm = rpm
        self.tpm = tpm
        self._request_times: deque[float] = deque()
        self._token_log: deque[tuple[float, int]] = deque()  # (timestamp, tokens)
        self._lock = asyncio.Lock()

    def _prune(self, now: float):
        """Remove entries older than 60 seconds."""
        cutoff = now - 60.0
        while self._request_times and self._request_times[0] < cutoff:
            self._request_times.popleft()
        # Token log has (est, delta) with possibly out-of-order timestamps (deltas use acquire_ts).
        # Remove ALL entries with ts < cutoff, not just from the left.
        self._token_log = deque((ts, t) for ts, t in self._token_log if ts >= cutoff)

    def _current_rpm(self) -> int:
        return len(self._request_times)

    def _current_tpm(self) -> int:
        return sum(t for _, t in self._token_log)

    async def try_acquire(self, estimated_tokens: int) -> tuple[bool, float | None]:
        """Try to acquire capacity. Returns (True, acquire_time) if allowed, (False, None) if rate limited.
        Pass acquire_time to record_actual_usage so estimate and delta are pruned together."""
        async with self._lock:
            now = time.time()
            self._prune(now)
            rpm_now = self._current_rpm()
            tpm_now = self._current_tpm()
            if rpm_now >= self.rpm:
                log.info("[RATELIMIT] acquire denied (RPM): est=%d, rpm=%d/%d, tpm=%d/%d",
                         estimated_tokens, rpm_now, self.rpm, tpm_now, self.tpm)
                return False, None
            if tpm_now + estimated_tokens > self.tpm:
                log.info("[RATELIMIT] acquire denied (TPM): est=%d, rpm=%d/%d, tpm=%d/%d",
                         estimated_tokens, rpm_now, self.rpm, tpm_now, self.tpm)
                return False, None
            self._request_times.append(now)
            self._token_log.append((now, estimated_tokens))
            return True, now

    async def record_actual_usage(self, prompt_tokens: int, completion_tokens: int,
                                  estimated_tokens: int, acquire_time: float | None = None) -> None:
        """Adjust token count after a request completes with actual usage.
        Use acquire_time (from try_acquire) so estimate and delta share the same timestamp and are pruned together."""
        actual = prompt_tokens + completion_tokens
        delta = actual - estimated_tokens
        if delta != 0:
            async with self._lock:
                ts = acquire_time if acquire_time is not None else time.time()
                self._token_log.append((ts, delta))
                tpm_now = self._current_tpm()
                log.info("[RATELIMIT] actual_usage: prompt=%d completion=%d actual=%d "
                         "est=%d delta=%+d tpm_now=%d",
                         prompt_tokens, completion_tokens, actual, estimated_tokens, delta, tpm_now)

    def reset(self):
        """Clear sliding window state for a fresh run."""
        self._request_times.clear()
        self._token_log.clear()

    async def wait_time(self) -> float:
        """Estimate seconds until capacity might free up."""
        async with self._lock:
            now = time.time()
            self._prune(now)
            waits = []
            if self._current_rpm() >= self.rpm:
                oldest = self._request_times[0]
                waits.append(oldest + 60.0 - now)
            if self._current_tpm() >= self.tpm and self._token_log:
                oldest_ts = min(ts for ts, _ in self._token_log)
                waits.append(oldest_ts + 60.0 - now)
            return max(waits) if waits else 0.0
