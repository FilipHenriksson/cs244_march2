import asyncio
import time
from rate_limiter import RateLimiter, THROTTLE_RPM, THROTTLE_TPM
from trace import trace


class AdaptiveSJFScheduler:
    """Strategy 5: Shortest Job First using learned completion times.

    Maintains an EMA of actual wall-clock duration per call type.
    Prioritizes calls with the shortest predicted duration.
    Unseen call types get top priority (exploration over exploitation).
    """

    def __init__(self, limiter: RateLimiter):
        self.limiter = limiter
        self._pending: list[tuple] = []
        self._notify: asyncio.Event = asyncio.Event()
        self._drain_task: asyncio.Task | None = None
        self._enqueue_counter = 0

        # Learned duration tracking
        self._ema: dict[str, float] = {}      # tracking_key -> EMA of duration (seconds)
        self._counts: dict[str, int] = {}     # tracking_key -> observation count
        self._alpha = 0.3                      # EMA smoothing factor

    # --- Duration learning ---

    @staticmethod
    def _tracking_key(call_type: str, detail: str) -> str:
        """Compute the key for duration tracking.

        agent_turn varies significantly by round, so we include the detail.
        All other call types are tracked by bare name.
        """
        if call_type == "agent_turn":
            return f"agent_turn:{detail}"
        return call_type

    def _predicted_duration(self, key: str) -> float:
        """Return predicted duration for a tracking key. 0.0 for unseen (top priority)."""
        return self._ema.get(key, 0.0)

    def _record_duration(self, key: str, duration: float):
        """Update EMA with a new observation."""
        if key not in self._ema:
            self._ema[key] = duration
            self._counts[key] = 1
        else:
            self._ema[key] = self._alpha * duration + (1 - self._alpha) * self._ema[key]
            self._counts[key] += 1
        # Wake drain loop — priorities may have changed
        self._notify.set()

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

    def register_group(self, group_id: str, size: int):
        pass

    def deregister_member(self, group_id: str):
        pass

    async def submit(self, coro_factory, estimated_tokens: int,
                     agent_id: str, call_type: str, detail: str = "",
                     group_id: str = None):
        """Enqueue a call. Priority computed dynamically from learned durations."""
        future = asyncio.get_event_loop().create_future()
        self._enqueue_counter += 1
        position = self._enqueue_counter
        tracking_key = self._tracking_key(call_type, detail)
        enqueue_time = time.time()
        self._pending.append((
            coro_factory, estimated_tokens, future,
            agent_id, call_type, detail, position, group_id,
            tracking_key, enqueue_time
        ))
        self._notify.set()
        return await future

    # --- Internal ---

    def _pick_best(self) -> int | None:
        """Return index of pending item with shortest predicted duration."""
        if not self._pending:
            return None
        best_idx = 0
        best_pred = self._predicted_duration(self._pending[0][8])
        best_order = self._pending[0][6]
        for i in range(1, len(self._pending)):
            pred = self._predicted_duration(self._pending[i][8])
            order = self._pending[i][6]
            if pred < best_pred or (pred == best_pred and order < best_order):
                best_idx = i
                best_pred = pred
                best_order = order
        return best_idx

    async def _drain(self):
        while True:
            while not self._pending:
                self._notify.clear()
                await self._notify.wait()

            idx = self._pick_best()
            if idx is None:
                self._notify.clear()
                await self._notify.wait()
                continue

            item = self._pending[idx]
            (coro_factory, est_tokens, future,
             agent_id, call_type, detail, position,
             group_id, tracking_key, enqueue_time) = item

            rpm_waits = 0
            tpm_waits = 0
            acquired = False
            while True:
                throttle = await self.limiter.try_acquire(est_tokens)
                if throttle is None:
                    acquired = True
                    break
                if throttle == THROTTLE_RPM:
                    rpm_waits += 1
                else:
                    tpm_waits += 1
                wait = await self.limiter.wait_time(est_tokens)
                wait = max(wait, 0.1)
                pred = self._predicted_duration(tracking_key)
                print(f"  [ASJF] queue waiting {wait:.2f}s for capacity "
                      f"(next: {agent_id}:{call_type}, pred={pred:.2f}s, reason={throttle})")
                await asyncio.sleep(wait)
                # Re-pick after waiting — EMA may have updated
                new_idx = self._pick_best()
                if new_idx is None:
                    break
                item = self._pending[new_idx]
                (coro_factory, est_tokens, future,
                 agent_id, call_type, detail, position,
                 group_id, tracking_key, enqueue_time) = item
                idx = new_idx

            if not acquired:
                continue

            self._pending.pop(idx)
            queue_wait = time.time() - enqueue_time
            call = trace.start_call(agent_id, call_type, detail,
                                    queue_position=position,
                                    queue_wait=queue_wait,
                                    estimated_tokens=est_tokens,
                                    rpm_waits=rpm_waits,
                                    tpm_waits=tpm_waits)
            asyncio.create_task(
                self._run(coro_factory, future, call, tracking_key,
                          est_tokens))

    async def _run(self, coro_factory, future, call, tracking_key: str,
                   est_tokens: int):
        try:
            t0 = time.time()
            result = await coro_factory()
            if hasattr(result, "usage") and result.usage is not None:
                await self.limiter.record_actual_usage(
                    result.usage.prompt_tokens,
                    result.usage.completion_tokens,
                    est_tokens,
                )
            self._record_duration(tracking_key, time.time() - t0)
            trace.end_call(call)
            future.set_result(result)
        except Exception as e:
            await self.limiter.record_actual_usage(0, 0, est_tokens)
            trace.end_call(call)
            future.set_exception(e)
