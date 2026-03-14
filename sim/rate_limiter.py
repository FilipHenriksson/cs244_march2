import asyncio
import logging
import time

log = logging.getLogger("rate_limiter")

# Throttle reasons returned by try_acquire
THROTTLE_RPM = "rpm"
THROTTLE_TPM = "tpm"


class RateLimiter:
    """Async-safe token-bucket rate limiter enforcing requests-per-minute (RPM)
    and tokens-per-minute (TPM) limits simultaneously.

    Public API
    ----------
    try_acquire(estimated_tokens)   – attempt to claim capacity before a request
    record_actual_usage(...)        – reconcile actual token usage after a request
    available_capacity()            – snapshot of current bucket levels
    wait_time(estimated_tokens)     – seconds until capacity is available
    stats                           – cumulative counters since last reset
    reset()                         – refill buckets and clear stats

    Expected usage pattern
    ----------------------
    Before each API call, invoke ``try_acquire(estimated_tokens)``. If it
    returns a throttle reason, wait (``wait_time`` can help) and retry.
    Once the API call completes — whether it succeeded or failed — always
    call ``record_actual_usage(prompt, completion, estimated)`` so the token
    bucket stays accurate. Skipping this step causes the limiter to
    permanently undercount available capacity.
    """

    def __init__(self, rpm: int = 20, tpm: int = 100_000):
        """
        Parameters
        ----------
        rpm : int
            Maximum requests allowed per minute. Defaults to 20.
        tpm : int
            Maximum tokens allowed per minute. Defaults to 100 000.
        """
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

        # Stats
        self._total_acquires: int = 0
        self._rpm_throttles: int = 0
        self._tpm_throttles: int = 0
        self._total_estimated: int = 0
        self._total_actual: int = 0

    def _refill(self, now: float) -> None:
        """Add tokens for elapsed time since last refill, capped at bucket capacity."""
        elapsed = now - self._last_refill
        if elapsed > 0:
            self._rpm_bucket = min(self.rpm, self._rpm_bucket + elapsed * self._rpm_rate)
            self._tpm_bucket = min(self.tpm, self._tpm_bucket + elapsed * self._tpm_rate)
            self._last_refill = now

    async def try_acquire(self, estimated_tokens: int) -> str | None:
        """Attempt to claim capacity for one request.

        Parameters
        ----------
        estimated_tokens : int
            Expected token cost of the request (prompt + completion estimate).

        Returns
        -------
        None
            Capacity was granted; the request may proceed.
        THROTTLE_RPM
            Request rate limit reached; caller should back off.
        THROTTLE_TPM
            Token rate limit reached; caller should back off.
        """
        async with self._lock:
            self._refill(time.time())
            if self._rpm_bucket < 1.0:
                self._rpm_throttles += 1
                log.info("[ACQUIRE] THROTTLE_RPM est=%d rpm_bucket=%.1f tpm_bucket=%.1f",
                         estimated_tokens, self._rpm_bucket, self._tpm_bucket)
                return THROTTLE_RPM
            if self._tpm_bucket < estimated_tokens:
                self._tpm_throttles += 1
                log.info("[ACQUIRE] THROTTLE_TPM est=%d rpm_bucket=%.1f tpm_bucket=%.1f",
                         estimated_tokens, self._rpm_bucket, self._tpm_bucket)
                return THROTTLE_TPM
            self._rpm_bucket -= 1.0
            self._tpm_bucket -= estimated_tokens
            self._total_acquires += 1
            log.info("[ACQUIRE] OK est=%d rpm_bucket=%.1f tpm_bucket=%.1f",
                     estimated_tokens, self._rpm_bucket, self._tpm_bucket)
            return None

    async def record_actual_usage(self, prompt_tokens: int, completion_tokens: int,
                                  estimated_tokens: int) -> None:
        """Reconcile the token bucket after a request completes.

        Should be called once per request, whether it succeeded or failed.
        Overestimates are refunded to the bucket; underestimates are charged.
        Passing (0, 0) for prompt/completion tokens signals a failed request
        and refunds the full estimate.

        Parameters
        ----------
        prompt_tokens : int
            Actual prompt tokens reported by the API response.
        completion_tokens : int
            Actual completion tokens reported by the API response.
        estimated_tokens : int
            The same estimate passed to the corresponding ``try_acquire`` call.
        """
        actual = prompt_tokens + completion_tokens
        if actual > 0:
            self._total_estimated += estimated_tokens
            self._total_actual += actual
        delta = estimated_tokens - actual  # positive = overestimate = give back
        if delta != 0:
            async with self._lock:
                self._tpm_bucket = min(self.tpm, self._tpm_bucket + delta)
                log.info("[RATELIMIT] actual_usage: prompt=%d completion=%d actual=%d "
                         "est=%d delta=%+d tpm_bucket=%.1f",
                         prompt_tokens, completion_tokens, actual,
                         estimated_tokens, delta, self._tpm_bucket)

    async def available_capacity(self) -> tuple[bool, float]:
        """Return the current bucket levels as ``(rpm_available, tpm_budget)``.

        ``rpm_available`` is True when at least one request slot is available.
        ``tpm_budget`` is the number of tokens that can be consumed right now.
        """
        async with self._lock:
            self._refill(time.time())
            return self._rpm_bucket >= 1.0, self._tpm_bucket

    async def wait_time(self, estimated_tokens: int = 0) -> float:
        """Return seconds to wait before ``try_acquire(estimated_tokens)`` will succeed.

        Returns 0.0 if capacity is already available.
        """
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

    @property
    def stats(self) -> dict:
        """Cumulative counters since the last ``reset()``.

        Keys: ``total_acquires``, ``rpm_throttles``, ``tpm_throttles``,
        ``total_estimated``, ``total_actual``, ``token_overestimate``.
        """
        return {
            "total_acquires": self._total_acquires,
            "rpm_throttles": self._rpm_throttles,
            "tpm_throttles": self._tpm_throttles,
            "total_estimated": self._total_estimated,
            "total_actual": self._total_actual,
            "token_overestimate": self._total_estimated - self._total_actual,
        }

    def reset(self):
        """Refill both buckets to capacity and zero all stats counters."""
        self._rpm_bucket = float(self.rpm)
        self._tpm_bucket = float(self.tpm)
        self._last_refill = time.time()
        self._total_acquires = 0
        self._rpm_throttles = 0
        self._tpm_throttles = 0
        self._total_estimated = 0
        self._total_actual = 0
