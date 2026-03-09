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

    async def try_acquire(self, estimated_tokens: int) -> float | None:
        """Try to acquire capacity. Returns acquire timestamp if allowed, None if rate limited."""
        async with self._lock:
            now = time.time()
            self._prune(now)
            if self._current_rpm() >= self.rpm:
                return None
            if self._current_tpm() + estimated_tokens > self.tpm:
                return None
            self._request_times.append(now)
            self._token_log.append((now, estimated_tokens))
            return now

    async def record_actual_usage(self, prompt_tokens: int, completion_tokens: int,
                                  estimated_tokens: int,
                                  acquire_time: float) -> None:
        """Adjust token count after a request completes with actual usage.

        Uses the original acquire_time so the correction expires together
        with the estimate, preventing negative _current_tpm() windows.
        """
        delta = (prompt_tokens + completion_tokens) - estimated_tokens
        if delta != 0:
            async with self._lock:
                self._token_log.append((acquire_time, delta))

    def reset(self):
        """Clear sliding window state for a fresh run."""
        self._request_times.clear()
        self._token_log.clear()

    async def wait_time(self, estimated_tokens: int = 0) -> float:
        """Estimate seconds until capacity might free up for a request of the given size."""
        async with self._lock:
            now = time.time()
            self._prune(now)
            waits = []
            if self._current_rpm() >= self.rpm:
                oldest = self._request_times[0]
                waits.append(oldest + 60.0 - now)
            if self._current_tpm() + estimated_tokens > self.tpm:
                if self._token_log:
                    oldest_tok = self._token_log[0][0]
                    waits.append(oldest_tok + 60.0 - now)
            return max(waits) if waits else 0.0
