import asyncio
import json
import time
from sim.rate_limiter import RateLimiter, THROTTLE_RPM, THROTTLE_TPM
from sim.trace import trace


class TokenSJFScheduler:
    """Shortest Job First by learned output token count.

    Maintains an EMA of actual completion tokens per call type.
    Uses learned estimates (+ safety buffer) for rate-limiter token
    reservations, replacing the static max_tokens worst-case used by
    other schedulers.  Prioritizes calls with the shortest predicted
    output tokens.  Unseen call types get top priority (exploration).
    """

    BUFFER_RATIO = 0.2
    MIN_BUFFER = 50

    def __init__(self, limiter: RateLimiter):
        self.limiter = limiter
        self._pending: list[tuple] = []
        self._notify: asyncio.Event = asyncio.Event()
        self._drain_task: asyncio.Task | None = None
        self._enqueue_counter = 0

        # Learned output-token tracking
        self._ema: dict[str, float] = {}
        self._counts: dict[str, int] = {}
        self._alpha = 0.3

    # --- Output-token learning ------------------------------------------------

    @staticmethod
    def _tracking_key(call_type: str, detail: str) -> str:
        """Orchestrator calls split by round; all others tracked by bare name."""
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

    # --- Token estimation (called by llm.py) ----------------------------------

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
        """Enqueue a call.  Priority is shortest predicted output tokens."""
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

    # --- Internal -------------------------------------------------------------

    def _pick_best(self) -> int | None:
        """Return index of pending item with shortest predicted output tokens."""
        if not self._pending:
            return None
        best_idx = 0
        pred = self._predicted_output(self._pending[0][8])
        best_val = pred if pred is not None else 0.0
        best_order = self._pending[0][6]
        for i in range(1, len(self._pending)):
            pred = self._predicted_output(self._pending[i][8])
            val = pred if pred is not None else 0.0
            order = self._pending[i][6]
            if val < best_val or (val == best_val and order < best_order):
                best_idx = i
                best_val = val
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
                pred = self._predicted_output(tracking_key)
                pred_str = f"{pred:.0f}tok" if pred is not None else "unseen"
                print(f"  [TSJF] queue waiting {wait:.2f}s for capacity "
                      f"(next: {agent_id}:{call_type}, pred={pred_str}, "
                      f"reason={throttle})")
                await asyncio.sleep(wait)
                # Re-pick — EMA may have updated while we waited
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
