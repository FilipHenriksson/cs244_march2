import asyncio
from rate_limiter import RateLimiter
from trace import trace


class FIFOScheduler:
    """Strategy 2: Global FIFO queue. Drains respecting rate limits."""

    def __init__(self, limiter: RateLimiter):
        self.limiter = limiter
        self._queue: asyncio.Queue = asyncio.Queue()
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
        """Enqueue a call. Returns result when the call completes."""
        future = asyncio.get_event_loop().create_future()
        self._enqueue_counter += 1
        position = self._enqueue_counter
        await self._queue.put((coro_factory, estimated_tokens, future,
                               agent_id, call_type, detail, position))
        return await future

    async def _drain(self):
        while True:
            item = await self._queue.get()
            coro_factory, est_tokens, future, agent_id, call_type, detail, position = item

            # Wait until rate limiter allows
            while not await self.limiter.try_acquire(est_tokens):
                wait = await self.limiter.wait_time()
                wait = max(wait, 0.1)
                print(f"  [FIFO] queue waiting {wait:.2f}s for capacity "
                      f"(next: {agent_id}:{call_type})")
                await asyncio.sleep(wait)

            call = trace.start_call(agent_id, call_type, detail,
                                    queue_position=position)
            asyncio.create_task(self._run(coro_factory, future, call))

    async def _run(self, coro_factory, future, call):
        try:
            result = await coro_factory()
            trace.end_call(call)
            future.set_result(result)
        except Exception as e:
            trace.end_call(call)
            future.set_exception(e)
