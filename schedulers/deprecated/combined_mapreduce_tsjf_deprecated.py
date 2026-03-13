import asyncio
import json
import time
from sim.rate_limiter import RateLimiter, THROTTLE_RPM, THROTTLE_TPM
from sim.trace import trace


class CombinedMapReduceTokenSJFScheduler:
    """Combined strategy: group-aware priority (MapReduce) + learned output tokens (Token SJF).

    Priority score = predicted_output_tokens * active_members_in_group (lower = better).
    - Short-output jobs in nearly-complete groups get top priority.
    - Singleton (reduce-phase) calls: pred * 1, no penalty.
    - Unseen call types: 0.0 (exploration-first).
    - As group members complete, remaining members' scores decrease (priority rises).
    - Uses learned output-token EMA for rate-limiter reservations via estimate_total_tokens.
    """

    BUFFER_RATIO = 0.2
    MIN_BUFFER = 50

    def __init__(self, limiter: RateLimiter):
        self.limiter = limiter
        self._pending: list[tuple] = []
        self._notify: asyncio.Event = asyncio.Event()
        self._drain_task: asyncio.Task | None = None
        self._enqueue_counter = 0

        # Group tracking (from MapReduce)
        self._group_active: dict[str, int] = {}

        # Output-token learning (from Token SJF)
        self._ema: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._alpha = 0.3

    # --- Output-token learning (from Token SJF) --------------------------------

    @staticmethod
    def _tracking_key(call_type: str, detail: str) -> str:
        if call_type == "orchestrator":
            return f"orchestrator:{detail}"
        return call_type

    def _predicted_output(self, key: str) -> float | None:
        """Return predicted completion tokens, or None if unseen."""
        return self._ema.get(key, None)

    def _record_output(self, key: str, completion_tokens: int):
        """Update EMA with observed completion tokens."""
        tokens = float(completion_tokens)
        if key not in self._ema:
            self._ema[key] = tokens
            self._counts[key] = 1
        else:
            self._ema[key] = self._alpha * tokens + (1 - self._alpha) * self._ema[key]
            self._counts[key] += 1
        self._notify.set()

    # --- Token estimation (called by llm.py) ------------------------------------

    def estimate_total_tokens(self, messages, call_type: str,
                              detail: str = "", **kwargs) -> int:
        """Compute input + predicted output tokens using learned EMA.

        For unseen call types, falls back to input + max_tokens.
        For known types, uses EMA + a proportional/minimum buffer to
        avoid under-reserving when actual output exceeds the average.
        """
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

    # --- Group lifecycle (from MapReduce) ---------------------------------------

    def register_group(self, group_id: str, size: int):
        self._group_active[group_id] = size

    def deregister_member(self, group_id: str):
        if group_id in self._group_active:
            self._group_active[group_id] -= 1
            if self._group_active[group_id] <= 0:
                del self._group_active[group_id]
            self._notify.set()

    # --- Combined priority ------------------------------------------------------

    def _effective_score(self, group_id: str | None, tracking_key: str) -> float:
        """Lower score = higher priority."""
        pred = self._predicted_output(tracking_key)
        token_estimate = pred if pred is not None else 0.0
        if group_id is None:
            active = 1
        else:
            active = self._group_active.get(group_id, 1)
        return token_estimate * active

    # --- Scheduler interface ----------------------------------------------------

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
        enqueue_time = time.time()
        self._pending.append((
            coro_factory, estimated_tokens, future,
            agent_id, call_type, detail, position, group_id,
            tracking_key, enqueue_time
        ))
        self._notify.set()
        return await future

    # --- Internal ---------------------------------------------------------------

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
                score = self._effective_score(group_id, tracking_key)
                print(f"  [COMBINED-TSJF] queue waiting {wait:.2f}s for capacity "
                      f"(next: {agent_id}:{call_type}, score={score:.0f}tok, reason={throttle})")
                await asyncio.sleep(wait)
                # Re-pick — both EMA and group membership may have changed
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
