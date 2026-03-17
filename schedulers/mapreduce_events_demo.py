import asyncio
import time
from sim.rate_limiter import RateLimiter, THROTTLE_RPM, THROTTLE_TPM
from sim.trace import trace
from events import global_bus, session_buses


class MapReduceWithEventsScheduler:
    """MapReduce priority with self-tracked session membership.

    Priority = 1 / (pending + in-flight calls for that session).
    """

    def __init__(self, limiter: RateLimiter):
        self.limiter = limiter
        self._pending: list[tuple] = []
        self._notify: asyncio.Event = asyncio.Event()
        self._drain_task: asyncio.Task | None = None
        self._enqueue_counter = 0
        self._session_active: dict[int, int] = {}
        self._in_flight: dict[str, dict] = {}  # flight_key -> info

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

    # --- Priority ---

    def _compute_priority(self, session_id: int) -> float:
        """Higher value = higher priority. Range (0, 1]."""
        active = self._session_active.get(session_id, 1)
        return 1.0 / max(active, 1)

    def _pick_best(self) -> int | None:
        """Return index of highest-priority pending item."""
        if not self._pending:
            return None
        best_idx = 0
        best_pri = self._compute_priority(self._pending[0][3])
        best_order = self._pending[0][6]
        for i in range(1, len(self._pending)):
            pri = self._compute_priority(self._pending[i][3])
            order = self._pending[i][6]
            if pri > best_pri or (pri == best_pri and order < best_order):
                best_idx = i
                best_pri = pri
                best_order = order
        return best_idx

    # --- Event emission helpers ---

    def _emit(self, session_id: int, event: dict) -> None:
        """Emit to both global and per-session buses."""
        global_bus.emit(event)
        session_buses.emit(session_id, event)

    def _build_queue_snapshot(self) -> dict:
        """Build a snapshot sorted by scheduler dispatch order.

        Sorted by priority descending, then enqueue_order ascending —
        exactly matching _pick_best() logic. Position 0 is the next
        call that will be dispatched.

        Each item includes enqueue_order as a stable identity key so
        the frontend can track items across snapshots and show
        movement arrows when priority reordering occurs.
        """
        now = time.time()
        pending = []
        for item in self._pending:
            session_id = item[3]
            pending.append({
                "call_key": item[4],
                "label": item[5],
                "session_id": session_id,
                "priority": round(self._compute_priority(session_id), 4),
                "wait_ms": int((now - item[7]) * 1000),
                "estimated_tokens": item[1],
                "enqueue_order": item[6],
            })

        # Sort to match _pick_best: highest priority first, then earliest arrival
        pending.sort(key=lambda p: (-p["priority"], p["enqueue_order"]))

        # Assign positions after sorting
        for i, p in enumerate(pending):
            p["position"] = i

        return {
            "type": "queue_snapshot",
            "pending": pending,
            "in_flight": list(self._in_flight.values()),
            "pending_count": len(pending),
            "in_flight_count": len(self._in_flight),
        }

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
        self._pending.append((
            coro_factory, estimated_tokens, future,
            session_id, call_key, label, position, enqueue_time
        ))
        self._notify.set()

        # --- Event: call_enqueued ---
        self._emit(session_id, {
            "type": "call_enqueued",
            "call_key": call_key,
            "label": label,
            "session_id": session_id,
            "queue_position": len(self._pending) - 1,
            "priority": round(self._compute_priority(session_id), 4),
            "estimated_tokens": estimated_tokens,
            "enqueue_order": position,
        })
        # Snapshot after enqueue so all listeners see updated positions
        global_bus.emit(self._build_queue_snapshot())

        return await future

    # --- Internal ---
    # Tuple layout: (coro_factory, est_tokens, future,
    #                session_id, call_key, label, position, enqueue_time)
    #   indices:      0            1           2
    #                 3           4         5      6         7

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
             session_id, call_key, label, position, enqueue_time) = item

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
                pri = self._compute_priority(session_id)
                print(f"  [MR] queue waiting {wait:.2f}s for capacity "
                      f"(next: s{session_id}:{call_key}, pri={pri:.2f}, reason={throttle})")

                # --- Event: rate_limited ---
                reason = "rpm" if throttle == THROTTLE_RPM else "tpm"
                self._emit(session_id, {
                    "type": "rate_limited",
                    "call_key": call_key,
                    "session_id": session_id,
                    "reason": reason,
                    "wait_seconds": round(wait, 2),
                })

                await asyncio.sleep(wait)
                new_idx = self._pick_best()
                if new_idx is None:
                    break
                item = self._pending[new_idx]
                (coro_factory, est_tokens, future,
                 session_id, call_key, label, position, enqueue_time) = item
                idx = new_idx

            if not acquired:
                continue

            self._pending.pop(idx)
            queue_wait = time.time() - enqueue_time

            # Track in-flight
            flight_key = f"{session_id}:{call_key}:{position}"
            self._in_flight[flight_key] = {
                "call_key": call_key,
                "label": label,
                "session_id": session_id,
                "dispatched_at": time.time(),
                "enqueue_order": position,
            }

            # --- Event: call_dispatched ---
            self._emit(session_id, {
                "type": "call_dispatched",
                "call_key": call_key,
                "label": label,
                "session_id": session_id,
                "queue_wait_ms": int(queue_wait * 1000),
                "rpm_waits": rpm_waits,
                "tpm_waits": tpm_waits,
                "enqueue_order": position,
            })
            # Snapshot after dispatch so positions update
            global_bus.emit(self._build_queue_snapshot())

            call = trace.start_call(session_id, call_key, label,
                                    queue_position=position,
                                    queue_wait=queue_wait,
                                    estimated_tokens=est_tokens,
                                    rpm_waits=rpm_waits,
                                    tpm_waits=tpm_waits)
            asyncio.create_task(
                self._run(coro_factory, future, call, est_tokens,
                          session_id, call_key, flight_key))

    async def _run(self, coro_factory, future, call, est_tokens: int,
                   session_id: int, call_key: str, flight_key: str):
        start_time = time.time()
        try:
            result = await coro_factory()
            if hasattr(result, "usage") and result.usage is not None:
                await self.limiter.record_actual_usage(
                    result.usage.prompt_tokens,
                    result.usage.completion_tokens,
                    est_tokens,
                )
            trace.end_call(call)

            latency_ms = int((time.time() - start_time) * 1000)
            usage = None
            if hasattr(result, "usage") and result.usage is not None:
                usage = {
                    "prompt_tokens": result.usage.prompt_tokens,
                    "completion_tokens": result.usage.completion_tokens,
                    "total_tokens": result.usage.total_tokens,
                }

            # --- Event: call_completed ---
            self._emit(session_id, {
                "type": "call_completed",
                "call_key": call_key,
                "session_id": session_id,
                "latency_ms": latency_ms,
                "usage": usage,
            })

            future.set_result(result)
        except Exception as e:
            await self.limiter.record_actual_usage(0, 0, est_tokens)
            trace.end_call(call)

            latency_ms = int((time.time() - start_time) * 1000)

            # --- Event: call_failed ---
            self._emit(session_id, {
                "type": "call_failed",
                "call_key": call_key,
                "session_id": session_id,
                "latency_ms": latency_ms,
                "error": str(e),
            })

            future.set_exception(e)
        finally:
            self._in_flight.pop(flight_key, None)
            self._track_complete(session_id)
            # Snapshot after completion so in-flight list updates
            global_bus.emit(self._build_queue_snapshot())