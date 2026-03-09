import asyncio
import time
from collections import deque


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
        while self._token_log and self._token_log[0][0] < cutoff:
            self._token_log.popleft()

    def _current_rpm(self) -> int:
        return len(self._request_times)

    def _current_tpm(self) -> int:
        return sum(t for _, t in self._token_log)

    async def try_acquire(self, estimated_tokens: int) -> bool:
        """Try to acquire capacity. Returns True if allowed, False if rate limited."""
        async with self._lock:
            now = time.time()
            self._prune(now)
            if self._current_rpm() >= self.rpm:
                return False
            if self._current_tpm() + estimated_tokens > self.tpm:
                return False
            self._request_times.append(now)
            self._token_log.append((now, estimated_tokens))
            return True

    async def record_actual_usage(self, prompt_tokens: int, completion_tokens: int,
                                  estimated_tokens: int) -> None:
        """Adjust token count after a request completes with actual usage."""
        delta = (prompt_tokens + completion_tokens) - estimated_tokens
        if delta != 0:
            async with self._lock:
                self._token_log.append((time.time(), delta))

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
            if self._current_tpm() >= self.tpm:
                oldest_tok = self._token_log[0][0]
                waits.append(oldest_tok + 60.0 - now)
            return max(waits) if waits else 0.0
