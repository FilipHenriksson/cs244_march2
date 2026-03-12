import asyncio
import bisect
import json
import time
from rate_limiter import RateLimiter, THROTTLE_RPM, THROTTLE_TPM
from trace import trace


class TokenSJFSkipScheduler:
    """Token-SJF with TPM-aware skipping.

    Extends TokenSJFScheduler with a key improvement: when the top SJF-priority
    item can't fit the current TPM window, the scheduler searches for a
    lower-priority item that *does* fit, rather than busy-waiting.

    A bisect-sorted index on estimated tokens enables O(log n) identification
    of which pending items fit the available TPM budget.  Among those, the
    best SJF-priority item (shortest predicted output) is dispatched.

    Notification (_notify asyncio.Event):
        The drain loop parks on _notify when idle (no items) or when waiting
        for rate-limit capacity to refill (interruptible sleep).  Two
        producers set the event:
          - submit(): a new request was enqueued, so the drain loop should
            re-evaluate whether something can be dispatched now.
          - _record_output(): a completed request updated the EMA, which may
            change SJF priorities or token estimates, so the drain loop
            should wake and re-pick.
        Interruptible sleeps (used for RPM/TPM waits) wrap _notify.wait()
        in asyncio.wait_for with the sleep duration as timeout, so the loop
        reacts immediately to new work instead of blocking for the full
        refill interval.
    """

    BUFFER_RATIO = 0.2
    MIN_BUFFER = 50

    def __init__(self, limiter: RateLimiter):
        self.limiter = limiter

        # Primary store: item_id -> item tuple
        self._items: dict[int, tuple] = {}
        # Sorted index for TPM budget queries: list of (est_tokens, item_id)
        self._sorted_tokens: list[tuple[int, int]] = []

        self._notify: asyncio.Event = asyncio.Event()
        self._drain_task: asyncio.Task | None = None
        self._next_id: int = 0
        self._enqueue_counter: int = 0

        # Learned output-token tracking
        self._ema: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._alpha = 0.3

        # Skip stats
        self._tpm_skips: int = 0

    # --- Output-token learning ------------------------------------------------

    @staticmethod
    def _tracking_key(call_type: str, detail: str) -> str:
        if call_type == "agent_turn":
            return f"agent_turn:{detail}"
        return call_type

    def _predicted_output(self, key: str) -> float | None:
        return self._ema.get(key, None)

    def _record_output(self, key: str, completion_tokens: int):
        tokens = float(completion_tokens)
        if key not in self._ema:
            self._ema[key] = tokens
            self._counts[key] = 1
        else:
            self._ema[key] = self._alpha * tokens + (1 - self._alpha) * self._ema[key]
            self._counts[key] += 1
        self._notify.set()

    # --- Token estimation (called by llm.py) ----------------------------------

    def estimate_total_tokens(self, messages, call_type: str,
                              detail: str = "", **kwargs) -> int:
        text = json.dumps(messages, default=str)
        if "tools" in kwargs:
            text += json.dumps(kwargs["tools"], default=str)
        input_tokens = len(text) // 4

        key = self._tracking_key(call_type, detail)
        pred = self._predicted_output(key)
        if pred is None:
            return input_tokens + kwargs.get("max_tokens", 1024)
        buffer = max(int(pred * self.BUFFER_RATIO), self.MIN_BUFFER)
        return input_tokens + int(pred) + buffer

    # --- Item management (dict + bisect-sorted index) -------------------------

    def _add_item(self, item: tuple) -> int:
        item_id = self._next_id
        self._next_id += 1
        self._items[item_id] = item
        bisect.insort(self._sorted_tokens, (item[1], item_id))
        return item_id

    def _remove_item(self, item_id: int) -> tuple:
        item = self._items.pop(item_id)
        entry = (item[1], item_id)
        idx = bisect.bisect_left(self._sorted_tokens, entry)
        if idx < len(self._sorted_tokens) and self._sorted_tokens[idx] == entry:
            self._sorted_tokens.pop(idx)
        return item

    # --- Priority helpers -----------------------------------------------------

    def _sjf_key(self, item_id: int) -> tuple[float, int]:
        """Return (predicted_output, enqueue_order) for SJF comparison."""
        item = self._items[item_id]
        pred = self._predicted_output(item[8])  # tracking_key
        val = pred if pred is not None else 0.0
        return (val, item[6])  # (predicted, position)

    def _pick_best_sjf(self) -> int | None:
        """Best SJF-priority item across all pending items."""
        if not self._items:
            return None
        return min(self._items, key=self._sjf_key)

    def _pick_best_fitting(self, budget: float) -> int | None:
        """Best SJF-priority item whose est_tokens <= budget.

        Uses bisect on the sorted token index to narrow candidates to
        O(k) where k = number of items that fit, then picks best SJF
        among those.
        """
        cutoff = bisect.bisect_right(
            self._sorted_tokens, (int(budget), float('inf'))
        )
        if cutoff == 0:
            return None

        best_id = None
        best_key = (float('inf'), float('inf'))
        for i in range(cutoff):
            _, item_id = self._sorted_tokens[i]
            key = self._sjf_key(item_id)
            if key < best_key:
                best_id = item_id
                best_key = key
        return best_id

    # --- Scheduler interface --------------------------------------------------

    def start(self):
        self._drain_task = asyncio.create_task(self._drain())

    async def stop(self):
        if self._drain_task:
            self._drain_task.cancel()
            try:
                await self._drain_task
            except asyncio.CancelledError:
                pass

    def register_group(self, group_id: str, size: int):
        pass

    def deregister_member(self, group_id: str):
        pass

    async def submit(self, coro_factory, estimated_tokens: int,
                     agent_id: str, call_type: str, detail: str = "",
                     group_id: str = None):
        future = asyncio.get_event_loop().create_future()
        self._enqueue_counter += 1
        position = self._enqueue_counter
        tracking_key = self._tracking_key(call_type, detail)
        enqueue_time = time.time()
        item = (
            coro_factory, estimated_tokens, future,
            agent_id, call_type, detail, position, group_id,
            tracking_key, enqueue_time
        )
        self._add_item(item)
        self._notify.set()
        return await future

    # --- Internal drain loop --------------------------------------------------

    async def _interruptible_sleep(self, seconds: float):
        """Sleep up to *seconds*, waking early if _notify fires."""
        self._notify.clear()
        try:
            await asyncio.wait_for(self._notify.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass

    async def _drain(self):
        while True:
            while not self._items:
                self._notify.clear()
                await self._notify.wait()

            rpm_ok, tpm_budget = await self.limiter.available_capacity()

            # ---- RPM blocked: nothing can be dispatched --------------------
            if not rpm_ok:
                wait = await self.limiter.wait_time(0)
                wait = max(wait, 0.1)
                sjf_id = self._pick_best_sjf()
                if sjf_id is not None:
                    item = self._items[sjf_id]
                    pred = self._predicted_output(item[8])
                    pred_str = f"{pred:.0f}tok" if pred is not None else "unseen"
                    print(f"  [TSJFs] rpm wait {wait:.2f}s "
                          f"(next: {item[3]}:{item[4]}, pred={pred_str})")
                await self._interruptible_sleep(wait)
                continue

            # ---- Find best item that fits current TPM window ---------------
            sjf_id = self._pick_best_sjf()
            fit_id = self._pick_best_fitting(tpm_budget)

            if fit_id is None:
                # Nothing fits — wait for the cheapest request to become viable
                min_tokens = (self._sorted_tokens[0][0]
                              if self._sorted_tokens else 0)
                wait = await self.limiter.wait_time(min_tokens)
                wait = max(wait, 0.1)
                print(f"  [TSJFs] tpm wait {wait:.2f}s, nothing fits "
                      f"(budget={tpm_budget:.0f}tok, "
                      f"cheapest={min_tokens}tok, "
                      f"pending={len(self._items)})")
                await self._interruptible_sleep(wait)
                continue

            skipped = (fit_id != sjf_id)
            if skipped:
                self._tpm_skips += 1

            item = self._items[fit_id]
            est_tokens = item[1]

            # Authoritative acquire (budget may have shifted slightly)
            throttle = await self.limiter.try_acquire(est_tokens)
            if throttle is not None:
                await asyncio.sleep(0.05)
                continue

            # ---- Dispatch ------------------------------------------------
            item = self._remove_item(fit_id)
            (coro_factory, est_tokens, future,
             agent_id, call_type, detail, position,
             group_id, tracking_key, enqueue_time) = item

            if skipped and sjf_id in self._items:
                sjf_item = self._items[sjf_id]
                print(f"  [TSJFs] skip: dispatching {agent_id}:{call_type} "
                      f"({est_tokens}tok) ahead of "
                      f"{sjf_item[3]}:{sjf_item[4]} "
                      f"({sjf_item[1]}tok) "
                      f"[budget={tpm_budget:.0f}tok, "
                      f"total_skips={self._tpm_skips}]")

            queue_wait = time.time() - enqueue_time
            call = trace.start_call(agent_id, call_type, detail,
                                    queue_position=position,
                                    queue_wait=queue_wait,
                                    estimated_tokens=est_tokens,
                                    rpm_waits=0,
                                    tpm_waits=0)
            asyncio.create_task(
                self._run(coro_factory, future, call, tracking_key,
                          est_tokens))

    async def _run(self, coro_factory, future, call, tracking_key: str,
                   est_tokens: int):
        try:
            result = await coro_factory()
            if hasattr(result, "usage") and result.usage is not None:
                await self.limiter.record_actual_usage(
                    result.usage.prompt_tokens,
                    result.usage.completion_tokens,
                    est_tokens,
                )
                if result.usage.completion_tokens is not None:
                    self._record_output(tracking_key,
                                        result.usage.completion_tokens)
            trace.end_call(call)
            future.set_result(result)
        except Exception as e:
            await self.limiter.record_actual_usage(0, 0, est_tokens)
            trace.end_call(call)
            future.set_exception(e)
