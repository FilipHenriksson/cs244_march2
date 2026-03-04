import asyncio

INPUT_COST_PER_TOKEN = 0.10 / 1_000_000   # $0.10 per 1M
OUTPUT_COST_PER_TOKEN = 0.40 / 1_000_000  # $0.40 per 1M


class CostLimitExceeded(Exception):
    pass


class CostTracker:
    def __init__(self, limit_dollars: float):
        self.limit = limit_dollars
        self._lock = asyncio.Lock()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.call_count = 0

    async def check(self):
        async with self._lock:
            if self.total_cost >= self.limit:
                raise CostLimitExceeded(
                    f"Cost limit ${self.limit:.2f} reached "
                    f"(spent ${self.total_cost:.4f})")

    async def record(self, prompt_tokens: int, completion_tokens: int):
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
        return (
            f"  Total LLM calls:      {self.call_count}\n"
            f"  Total input tokens:   {self.total_input_tokens:,}\n"
            f"  Total output tokens:  {self.total_output_tokens:,}\n"
            f"  Total cost:           ${self.total_cost:.4f} / ${self.limit:.2f}")


# Module-level singleton
cost_tracker: CostTracker | None = None


def init_cost_tracker(limit_dollars: float):
    global cost_tracker
    cost_tracker = CostTracker(limit_dollars)
