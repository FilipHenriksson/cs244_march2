import asyncio
import time
from rate_limiter import RateLimiter
from trace import trace


class CombinedMapReduceAdaptiveSJFScheduler:
    """Combined strategy: group-aware priority (MapReduce) + learned durations (Adaptive SJF).

    Priority score = predicted_duration * active_members_in_group (lower = better).
    - Short jobs in nearly-complete groups get top priority.
    - Singleton (reduce-phase) calls: pred * 1, no penalty.
    - Unseen call types: 0.0 (exploration-first).
    - As group members complete, remaining members' scores decrease (priority rises).
    """

    def __init__(self, limiter: RateLimiter):
        self.limiter = limiter
        self._pending: list[tuple] = []
        self._notify: asyncio.Event = asyncio.Event()
        self._drain_task: asyncio.Task | None = None
        self._enqueue_counter = 0

        # Group tracking (from MapReduce)
        self._group_active: dict[str, int] = {}

        # Duration learning (from Adaptive SJF)
        self._ema: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._alpha = 0.3

    # --- Duration learning (from Adaptive SJF) ---

    @staticmethod
    def _tracking_key(call_type: str, detail: str) -> str:
        if call_type == "agent_turn":
            return f"agent_turn:{detail}"
        return call_type

    def _predicted_duration(self, key: str) -> float:
        return self._ema.get(key, 0.0)

    def _record_duration(self, key: str, duration: float):
        if key not in self._ema:
            self._ema[key] = duration
            self._counts[key] = 1
        else:
            self._ema[key] = self._alpha * duration + (1 - self._alpha) * self._ema[key]
            self._counts[key] += 1
        self._notify.set()

    # --- Group lifecycle (from MapReduce) ---

    def register_group(self, group_id: str, size: int):
        self._group_active[group_id] = size

    def deregister_member(self, group_id: str):
        if group_id in self._group_active:
            self._group_active[group_id] -= 1
            if self._group_active[group_id] <= 0:
                del self._group_active[group_id]
            self._notify.set()

    # --- Combined priority ---

    def _effective_score(self, group_id: str | None, tracking_key: str) -> float:
        """Lower score = higher priority."""
        pred = self._predicted_duration(tracking_key)
        if group_id is None:
            active = 1
        else:
            active = self._group_active.get(group_id, 1)
        return pred * active

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
                     agent_id: str, call_type: str, detail: str = "",
                     group_id: str = None):
        future = asyncio.get_event_loop().create_future()
        self._enqueue_counter += 1
        position = self._enqueue_counter
        tracking_key = self._tracking_key(call_type, detail)
        self._pending.append((
            coro_factory, estimated_tokens, future,
            agent_id, call_type, detail, position, group_id, tracking_key
        ))
        self._notify.set()
        return await future

    # --- Internal ---

    def _pick_best(self) -> int | None:
        if not self._pending:
            return None
        best_idx = 0
        best_score = self._effective_score(self._pending[0][7], self._pending[0][8])
        best_order = self._pending[0][6]
        for i in range(1, len(self._pending)):
            score = self._effective_score(self._pending[i][7], self._pending[i][8])
            order = self._pending[i][6]
            if score < best_score or (score == best_score and order < best_order):
                best_idx = i
                best_score = score
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
             group_id, tracking_key) = item

            while not await self.limiter.try_acquire(est_tokens):
                wait = await self.limiter.wait_time()
                wait = max(wait, 0.1)
                score = self._effective_score(group_id, tracking_key)
                print(f"  [COMBINED] queue waiting {wait:.2f}s for capacity "
                      f"(next: {agent_id}:{call_type}, score={score:.2f})")
                await asyncio.sleep(wait)
                # Re-pick — both EMA and group membership may have changed
                new_idx = self._pick_best()
                if new_idx is None:
                    break
                item = self._pending[new_idx]
                (coro_factory, est_tokens, future,
                 agent_id, call_type, detail, position,
                 group_id, tracking_key) = item
                idx = new_idx

            self._pending.pop(idx)
            call = trace.start_call(agent_id, call_type, detail,
                                    queue_position=position,
                                    estimated_tokens=est_tokens)
            asyncio.create_task(self._run(coro_factory, future, call, tracking_key))

    async def _run(self, coro_factory, future, call, tracking_key: str):
        try:
            t0 = time.time()
            result = await coro_factory()
            self._record_duration(tracking_key, time.time() - t0)
            trace.end_call(call)
            future.set_result(result)
        except Exception as e:
            trace.end_call(call)
            future.set_exception(e)
