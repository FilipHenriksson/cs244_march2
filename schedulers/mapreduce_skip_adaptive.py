"""MapReduce Skip Adaptive — adds output-aware secondary priority.

Extends MapReduce Skip with a secondary sort on predicted output tokens:
when multiple pending calls share the same MapReduce straggler priority
(e.g. five analysts from the same session, all at priority 1/5), the call
with the lowest predicted output-token count is dispatched first.

Shorter calls complete sooner, freeing rate-limiter capacity faster and
reducing mean session completion time without sacrificing straggler
prioritization.

Priority key (lower tuple = better):
    (-mapreduce_priority, predicted_output_tokens, enqueue_order)
"""

import asyncio
import bisect
import json
import time
from sim.rate_limiter import RateLimiter, THROTTLE_RPM, THROTTLE_TPM
from sim.trace import trace


class MapReduceSkipAdaptiveScheduler:
    """MapReduce priority + output-token tiebreaking + TPM-aware skipping.

    Priority = 1 / (active calls for session).  Among equal-priority calls,
    the one with the lowest predicted output tokens is dispatched first.
    When the top-priority item can't fit the TPM budget, dispatches the
    highest-priority item that does.  Uses learned output-token EMA for
    tighter rate-limiter reservations.
    """

    BUFFER_RATIO = 0.2
    MIN_BUFFER = 50

    def __init__(self, limiter: RateLimiter):
        self.limiter = limiter

        self._items: dict[int, tuple] = {}
        self._sorted_tokens: list[tuple[int, int]] = []

        self._notify: asyncio.Event = asyncio.Event()
        self._drain_task: asyncio.Task | None = None
        self._next_id: int = 0
        self._enqueue_counter: int = 0

        self._session_active: dict[int, int] = {}

        self._ema: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._alpha = 0.3

        self._tpm_skips: int = 0
        self._rpm_waits: int = 0
        self._tpm_waits: int = 0
        self._default_max_tokens = 1024

    # --- Session tracking ---

    def _track_submit(self, session_id: int):
        self._session_active[session_id] = self._session_active.get(session_id, 0) + 1

    def _track_complete(self, session_id: int):
        count = self._session_active.get(session_id, 1) - 1
        if count <= 0:
            self._session_active.pop(session_id, None)
        else:
            self._session_active[session_id] = count
        self._notify.set()

    # --- Output-token learning ---

    def _predicted_output(self, call_key: str) -> float | None:
        return self._ema.get(call_key, None)

    def _record_output(self, call_key: str, completion_tokens: int):
        tokens = float(completion_tokens)
        if call_key not in self._ema:
            self._ema[call_key] = tokens
            self._counts[call_key] = 1
        else:
            self._ema[call_key] = self._alpha * tokens + (1 - self._alpha) * self._ema[call_key]
            self._counts[call_key] += 1
        self._notify.set()

    # --- Token estimation (called by llm.py) ---

    def estimate_total_tokens(self, messages, call_key: str, **kwargs) -> int:
        text = json.dumps(messages, default=str)
        if "tools" in kwargs:
            text += json.dumps(kwargs["tools"], default=str)
        input_tokens = len(text) // 4

        pred = self._predicted_output(call_key)
        if pred is None:
            return input_tokens + kwargs.get("max_tokens", self._default_max_tokens)
        buffer = max(int(pred * self.BUFFER_RATIO), self.MIN_BUFFER)
        return input_tokens + int(pred) + buffer

    # --- MapReduce priority ---

    def _compute_priority(self, session_id: int) -> float:
        """Higher value = higher priority. Range (0, 1]."""
        active = self._session_active.get(session_id, 1)
        return 1.0 / max(active, 1)

    # --- Item management (dict + bisect-sorted index) ---
    # Tuple layout: (coro_factory, est_tokens, future,
    #                session_id, call_key, label, position, enqueue_time)
    #   indices:      0            1           2
    #                 3           4         5      6         7

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

    # --- Priority helpers ---

    def _mr_key(self, item_id: int) -> tuple[float, float, int]:
        """Return (-priority, predicted_output, enqueue_order).
        Lower tuple = better: highest MR priority, then shortest predicted
        output, then earliest arrival."""
        item = self._items[item_id]
        session_id = item[3]
        call_key = item[4]
        pri = self._compute_priority(session_id)
        pred = self._predicted_output(call_key)
        if pred is None:
            pred_tokens = float(self._default_max_tokens)
        else:
            pred_tokens = pred
        return (-pri, pred_tokens, item[6])

    def _pick_best_mr(self) -> int | None:
        """Best priority item across all pending items."""
        if not self._items:
            return None
        return min(self._items, key=self._mr_key)

    def _pick_best_fitting(self, budget: float) -> int | None:
        """Best priority item whose est_tokens <= budget."""
        cutoff = bisect.bisect_right(
            self._sorted_tokens, (int(budget), float('inf'))
        )
        if cutoff == 0:
            return None

        best_id = None
        best_key = (float('inf'), float('inf'), float('inf'))
        for i in range(cutoff):
            _, item_id = self._sorted_tokens[i]
            key = self._mr_key(item_id)
            if key < best_key:
                best_id = item_id
                best_key = key
        return best_id

    # --- Scheduler interface ---

    def start(self):
        self._drain_task = asyncio.create_task(self._drain())

    async def stop(self):
        if self._drain_task:
            self._drain_task.cancel()
            try:
                await self._drain_task
            except asyncio.CancelledError:
                pass

    async def submit(self, coro_factory, estimated_tokens: int,
                     session_id: int, call_key: str, label: str = ""):
        future = asyncio.get_event_loop().create_future()
        self._enqueue_counter += 1
        position = self._enqueue_counter
        enqueue_time = time.time()
        self._track_submit(session_id)
        item = (
            coro_factory, estimated_tokens, future,
            session_id, call_key, label, position, enqueue_time
        )
        self._add_item(item)
        self._notify.set()
        return await future

    # --- Internal drain loop ---

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

            if not rpm_ok:
                self._rpm_waits += 1
                wait = await self.limiter.wait_time(0)
                wait = max(wait, 0.1)
                mr_id = self._pick_best_mr()
                if mr_id is not None:
                    item = self._items[mr_id]
                    pri = self._compute_priority(item[3])
                    print(f"  [MR-SA] rpm wait {wait:.2f}s "
                          f"(next: s{item[3]}:{item[4]}, pri={pri:.2f})")
                await self._interruptible_sleep(wait)
                continue

            mr_id = self._pick_best_mr()
            fit_id = self._pick_best_fitting(tpm_budget)

            if fit_id is None:
                self._tpm_waits += 1
                min_tokens = (self._sorted_tokens[0][0]
                              if self._sorted_tokens else 0)
                wait = await self.limiter.wait_time(min_tokens)
                wait = max(wait, 0.1)
                print(f"  [MR-SA] tpm wait {wait:.2f}s, nothing fits "
                      f"(budget={tpm_budget:.0f}tok, "
                      f"cheapest={min_tokens}tok, "
                      f"pending={len(self._items)})")
                await self._interruptible_sleep(wait)
                continue

            skipped = (fit_id != mr_id)
            if skipped:
                self._tpm_skips += 1

            item = self._items[fit_id]
            est_tokens = item[1]

            throttle = await self.limiter.try_acquire(est_tokens)
            if throttle is not None:
                await asyncio.sleep(0.05)
                continue

            item = self._remove_item(fit_id)
            (coro_factory, est_tokens, future,
             session_id, call_key, label, position, enqueue_time) = item

            if skipped and mr_id in self._items:
                mr_item = self._items[mr_id]
                print(f"  [MR-SA] skip: dispatching s{session_id}:{call_key} "
                      f"({est_tokens}tok) ahead of "
                      f"s{mr_item[3]}:{mr_item[4]} "
                      f"({mr_item[1]}tok) "
                      f"[budget={tpm_budget:.0f}tok, "
                      f"total_skips={self._tpm_skips}]")

            queue_wait = time.time() - enqueue_time
            call = trace.start_call(session_id, call_key, label,
                                    queue_position=position,
                                    queue_wait=queue_wait,
                                    estimated_tokens=est_tokens,
                                    rpm_waits=0,
                                    tpm_waits=0)
            asyncio.create_task(
                self._run(coro_factory, future, call, call_key,
                          est_tokens, session_id))

    @property
    def stats(self) -> dict:
        return {
            "rpm_waits": self._rpm_waits,
            "tpm_waits": self._tpm_waits,
            "tpm_skips": self._tpm_skips,
        }

    async def _run(self, coro_factory, future, call, call_key: str,
                   est_tokens: int, session_id: int):
        try:
            result = await coro_factory()
            if hasattr(result, "usage") and result.usage is not None:
                await self.limiter.record_actual_usage(
                    result.usage.prompt_tokens,
                    result.usage.completion_tokens,
                    est_tokens,
                )
                if result.usage.completion_tokens is not None:
                    self._record_output(call_key,
                                        result.usage.completion_tokens)
            trace.end_call(call)
            future.set_result(result)
        except Exception as e:
            await self.limiter.record_actual_usage(0, 0, est_tokens)
            trace.end_call(call)
            future.set_exception(e)
        finally:
            self._track_complete(session_id)
