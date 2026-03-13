"""Improved MapReduce scheduler — infers group membership from session_id.

Same priority logic as MapReduceScheduler (priority = 1 / active_in_group),
but the group is derived from the session portion of agent_id rather than
requiring explicit register_group / deregister_member calls from the client.

Active call count is tracked internally:
  - submit()  increments _session_active[session] when a call arrives
  - _run()    decrements it when the call completes (success or failure)

Because pipeline phases within a session are sequential (breakdown -> agents
-> synthesis -> reviewers -> finalize), the active count at any moment
reflects exactly the current fan-out width, giving the same prioritization
signal as the original MapReduceScheduler without any client-side lifecycle
management.
"""

import asyncio
import time
from sim.rate_limiter import RateLimiter, THROTTLE_RPM, THROTTLE_TPM
from sim.trace import trace


class MapReduceImprovedScheduler:
    """MapReduce priority with self-tracked group membership.

    Groups are identified by the session prefix of agent_id (e.g. "s0" from
    "s0:a2").  Priority = 1 / (pending + in-flight calls for that session).
    Standalone calls with no parseable session get priority 1.0.
    """

    def __init__(self, limiter: RateLimiter):
        self.limiter = limiter
        self._pending: list[tuple] = []
        self._notify: asyncio.Event = asyncio.Event()
        self._drain_task: asyncio.Task | None = None
        self._enqueue_counter = 0
        self._session_active: dict[str, int] = {}

    # --- Backward-compatible no-ops ---

    def register_group(self, group_id: str, size: int):
        pass

    def deregister_member(self, group_id: str):
        pass

    # --- Session key extraction ---

    @staticmethod
    def _session_key(agent_id: str) -> str | None:
        """Extract session prefix from agent_id (e.g. "s0" from "s0:a2")."""
        if ":" in agent_id:
            return agent_id.split(":")[0]
        return None

    # --- Internal tracking ---

    def _track_submit(self, session: str | None):
        if session is not None:
            self._session_active[session] = self._session_active.get(session, 0) + 1

    def _track_complete(self, session: str | None):
        if session is not None:
            count = self._session_active.get(session, 1) - 1
            if count <= 0:
                self._session_active.pop(session, None)
            else:
                self._session_active[session] = count
            self._notify.set()

    # --- Priority ---

    def _compute_priority(self, session: str | None) -> float:
        """Higher value = higher priority. Range (0, 1]."""
        if session is None:
            return 1.0
        active = self._session_active.get(session, 1)
        return 1.0 / max(active, 1)

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
        """Enqueue a call.  group_id is accepted for interface compatibility but
        ignored — session is inferred from agent_id."""
        future = asyncio.get_event_loop().create_future()
        self._enqueue_counter += 1
        position = self._enqueue_counter
        enqueue_time = time.time()
        session = self._session_key(agent_id)
        self._track_submit(session)
        self._pending.append((
            coro_factory, estimated_tokens, future,
            agent_id, call_type, detail, position, session, enqueue_time
        ))
        self._notify.set()
        return await future

    # --- Internal ---

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
             session, enqueue_time) = item

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
                pri = self._compute_priority(session)
                print(f"  [MR-I] queue waiting {wait:.2f}s for capacity "
                      f"(next: {agent_id}:{call_type}, pri={pri:.2f}, reason={throttle})")
                await asyncio.sleep(wait)
                new_idx = self._pick_best()
                if new_idx is None:
                    break
                item = self._pending[new_idx]
                (coro_factory, est_tokens, future,
                 agent_id, call_type, detail, position,
                 session, enqueue_time) = item
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
                self._run(coro_factory, future, call, est_tokens, session))

    async def _run(self, coro_factory, future, call, est_tokens: int,
                   session: str | None):
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
        finally:
            self._track_complete(session)
