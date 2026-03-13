import asyncio

INPUT_COST_PER_TOKEN = 0.10 / 1_000_000   # $0.10 per 1M
OUTPUT_COST_PER_TOKEN = 0.40 / 1_000_000  # $0.40 per 1M


class CostLimitExceeded(Exception):
    """Raised by ``check()`` or ``record()`` when the dollar limit is reached."""


class CostTracker:
    """Async-safe running cost accumulator with a hard dollar cap.

    Public API
    ----------
    check()                              – raise CostLimitExceeded if limit hit
    record(prompt_tokens, completion)    – add one call's usage and check limit
    summary()                            – formatted multi-line stats string

    Expected usage pattern
    ----------------------
    Call ``check()`` before submitting a request as a pre-flight guard.
    After the API response arrives, call ``record()`` with the actual token
    counts; this updates the running totals and will raise ``CostLimitExceeded``
    if the cumulative spend has now crossed the limit.

    The module exposes a ``cost_tracker`` singleton (initially ``None``) that
    is created by ``init_cost_tracker(limit_dollars)`` at startup. All callers
    should access the tracker through ``cost_tracker as ct; ct.cost_tracker``.
    """

    def __init__(self, limit_dollars: float):
        """
        Parameters
        ----------
        limit_dollars : float
            Hard cap in USD. Both ``check()`` and ``record()`` raise
            ``CostLimitExceeded`` once cumulative spend reaches this value.
        """
        self.limit = limit_dollars
        self._lock = asyncio.Lock()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.call_count = 0

    async def check(self):
        """Raise ``CostLimitExceeded`` if the spend limit has already been reached.

        Call this before issuing a request to abort early rather than wasting
        tokens on a call whose cost would be discarded.
        """
        async with self._lock:
            if self.total_cost >= self.limit:
                raise CostLimitExceeded(
                    f"Cost limit ${self.limit:.2f} reached "
                    f"(spent ${self.total_cost:.4f})")

    async def record(self, prompt_tokens: int, completion_tokens: int):
        """Accumulate token usage for one completed API call.

        Updates running totals and raises ``CostLimitExceeded`` if the new
        cumulative spend meets or exceeds the limit.

        Parameters
        ----------
        prompt_tokens : int
            Input tokens reported by the API response.
        completion_tokens : int
            Output tokens reported by the API response.
        """
        cost = (prompt_tokens * INPUT_COST_PER_TOKEN +
                completion_tokens * OUTPUT_COST_PER_TOKEN)
        async with self._lock:
            self.total_input_tokens += prompt_tokens
            self.total_output_tokens += completion_tokens
            self.total_cost += cost
            self.call_count += 1
            if self.total_cost >= self.limit:
                raise CostLimitExceeded(
                    f"Cost limit ${self.limit:.2f} reached "
                    f"(spent ${self.total_cost:.4f})")

    def summary(self) -> str:
        """Return a formatted multi-line string of cumulative stats."""
        return (
            f"  Total LLM calls:      {self.call_count}\n"
            f"  Total input tokens:   {self.total_input_tokens:,}\n"
            f"  Total output tokens:  {self.total_output_tokens:,}\n"
            f"  Total cost:           ${self.total_cost:.4f} / ${self.limit:.2f}")


# Module-level singleton — initialised by init_cost_tracker() at startup.
cost_tracker: CostTracker | None = None


def init_cost_tracker(limit_dollars: float):
    """Create (or replace) the module-level ``cost_tracker`` singleton.

    Should be called once per run before any LLM calls are made.
    In multi-scheduler runs this is called once per scheduler so each gets
    its own fresh budget (``limit_dollars`` is typically the total cap divided
    by the number of schedulers).
    """
    global cost_tracker
    cost_tracker = CostTracker(limit_dollars)
