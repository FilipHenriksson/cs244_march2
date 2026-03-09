import asyncio
from rate_limiter import RateLimiter
from trace import trace


class SJFScheduler:
    """Strategy 3: Shortest Job First priority queue ordered by input tokens."""

    def __init__(self, limiter: RateLimiter):
        self.limiter = limiter
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self._drain_task: asyncio.Task | None = None
        self._enqueue_counter = 0

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
        """Enqueue a call, prioritized by estimated token count (ascending)."""
        future = asyncio.get_event_loop().create_future()
        self._enqueue_counter += 1
        position = self._enqueue_counter
        # Priority: (estimated_tokens, insertion_order) for stable sorting
        priority = (estimated_tokens, self._enqueue_counter)
        await self._queue.put((priority, coro_factory, future,
                               agent_id, call_type, detail, position))
        return await future

    async def _drain(self):
        while True:
            item = await self._queue.get()
            priority, coro_factory, future, agent_id, call_type, detail, position = item
            est_tokens = priority[0]

            # Wait until rate limiter allows
            allowed, acquire_ts = False, None
            while True:
                allowed, acquire_ts = await self.limiter.try_acquire(est_tokens)
                if allowed:
                    break
                wait = await self.limiter.wait_time()
                wait = max(wait, 0.1)
                print(f"  [SJF] queue waiting {wait:.2f}s for capacity "
                      f"(next: {agent_id}:{call_type}, {est_tokens} tokens)")
                await asyncio.sleep(wait)

            call = trace.start_call(agent_id, call_type, detail,
                                    queue_position=position,
                                    estimated_tokens=est_tokens)
            asyncio.create_task(self._run(coro_factory, future, call, est_tokens, acquire_ts))

    async def _run(self, coro_factory, future, call, est_tokens: int, acquire_ts: float | None):
        try:
            result = await coro_factory()
            if hasattr(result, "usage") and result.usage is not None:
                await self.limiter.record_actual_usage(
                    result.usage.prompt_tokens,
                    result.usage.completion_tokens,
                    est_tokens,
                    acquire_time=acquire_ts,
                )
            trace.end_call(call)
            future.set_result(result)
        except Exception as e:
            trace.end_call(call)
            future.set_exception(e)
