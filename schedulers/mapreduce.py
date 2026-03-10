import asyncio
import time
from rate_limiter import RateLimiter, THROTTLE_RPM, THROTTLE_TPM
from trace import trace


class MapReduceScheduler:
    """Strategy 4: Dynamic priority based on fan-out group completion.

    Priority = 1 / (number of active members in the same group).
    Standalone calls (no group) get priority 1.0 (highest).
    As group members complete, remaining members' priority increases.
    """

    def __init__(self, limiter: RateLimiter):
        self.limiter = limiter
        self._pending: list[tuple] = []
        self._notify: asyncio.Event = asyncio.Event()
        self._drain_task: asyncio.Task | None = None
        self._enqueue_counter = 0
        self._group_active: dict[str, int] = {}

    # --- Group lifecycle ---

    def register_group(self, group_id: str, size: int):
        """Declare a fan-out group with `size` members. Call before gather()."""
        self._group_active[group_id] = size

    def deregister_member(self, group_id: str):
        """One member of the group finished. Decrement and re-evaluate."""
        if group_id in self._group_active:
            self._group_active[group_id] -= 1
            if self._group_active[group_id] <= 0:
                del self._group_active[group_id]
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

    async def submit(self, coro_factory, estimated_tokens: int,
                     agent_id: str, call_type: str, detail: str = "",
                     group_id: str = None):
        """Enqueue a call. Priority computed dynamically at drain time."""
        future = asyncio.get_event_loop().create_future()
        self._enqueue_counter += 1
        position = self._enqueue_counter
        enqueue_time = time.time()
        self._pending.append((
            coro_factory, estimated_tokens, future,
            agent_id, call_type, detail, position, group_id, enqueue_time
        ))
        self._notify.set()
        return await future

    # --- Internal ---

    def _compute_priority(self, group_id: str | None) -> float:
        """Higher value = higher priority. Range (0, 1]."""
        if group_id is None:
            return 1.0
        active = self._group_active.get(group_id, 1)
        return 1.0 / active

    def _pick_best(self) -> int | None:
        """Return index of highest-priority pending item."""
        if not self._pending:
            return None
        best_idx = 0
        best_pri = self._compute_priority(self._pending[0][7])
        best_order = self._pending[0][6]
        for i in range(1, len(self._pending)):
            pri = self._compute_priority(self._pending[i][7])
            order = self._pending[i][6]
            if pri > best_pri or (pri == best_pri and order < best_order):
                best_idx = i
                best_pri = pri
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
             group_id, enqueue_time) = item

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
                pri = self._compute_priority(group_id)
                print(f"  [MR] queue waiting {wait:.2f}s for capacity "
                      f"(next: {agent_id}:{call_type}, pri={pri:.2f}, reason={throttle})")
                await asyncio.sleep(wait)
                # Re-pick after waiting — priorities may have changed
                new_idx = self._pick_best()
                if new_idx is None:
                    break
                item = self._pending[new_idx]
                (coro_factory, est_tokens, future,
                 agent_id, call_type, detail, position,
                 group_id, enqueue_time) = item
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
                self._run(coro_factory, future, call, est_tokens))

    async def _run(self, coro_factory, future, call, est_tokens: int):
        try:
            result = await coro_factory()
            if hasattr(result, "usage") and result.usage is not None:
                await self.limiter.record_actual_usage(
                    result.usage.prompt_tokens,
                    result.usage.completion_tokens,
                    est_tokens,
                )
            trace.end_call(call)
            future.set_result(result)
        except Exception as e:
            await self.limiter.record_actual_usage(0, 0, est_tokens)
            trace.end_call(call)
            future.set_exception(e)
