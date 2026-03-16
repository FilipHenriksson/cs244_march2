"""Event bus for SSE streaming.

Provides per-session and global event distribution. Scheduler and API
code emit events; SSE endpoints consume them via async queues.

Usage::

    from events import global_bus, session_buses

    # Emit (non-blocking, never awaited)
    global_bus.emit({"type": "call_enqueued", "session_id": 3, ...})
    session_buses.emit(session_id=3, event={...})

    # Subscribe (SSE endpoint)
    queue = global_bus.subscribe()
    try:
        while True:
            event = await queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\\n\\n"
    finally:
        global_bus.unsubscribe(queue)
"""

from __future__ import annotations

import asyncio
import time
from typing import Any


class EventBus:
    """Fan-out event bus backed by asyncio.Queue per subscriber.

    emit() is synchronous (put_nowait) so the scheduler drain loop
    never blocks on slow SSE consumers.
    """

    def __init__(self, max_queue_size: int = 512):
        self._subscribers: list[asyncio.Queue] = []
        self._max_queue_size = max_queue_size

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=self._max_queue_size)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass

    def emit(self, event: dict[str, Any]) -> None:
        if "timestamp" not in event:
            event["timestamp"] = time.time()
        dead: list[asyncio.Queue] = []
        for q in self._subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Drop oldest event to make room (back-pressure)
                try:
                    q.get_nowait()
                    q.put_nowait(event)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    dead.append(q)
        for q in dead:
            self._subscribers.remove(q)

    def close(self) -> None:
        """Send sentinel (None) to all subscribers, then clear."""
        for q in self._subscribers:
            try:
                q.put_nowait(None)
            except asyncio.QueueFull:
                pass
        self._subscribers.clear()

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)


class SessionEventBuses:
    """Registry of per-session event buses, keyed by int session_id."""

    def __init__(self):
        self._buses: dict[int, EventBus] = {}

    def get_or_create(self, session_id: int) -> EventBus:
        if session_id not in self._buses:
            self._buses[session_id] = EventBus()
        return self._buses[session_id]

    def emit(self, session_id: int, event: dict[str, Any]) -> None:
        bus = self._buses.get(session_id)
        if bus is not None:
            bus.emit(event)

    def close(self, session_id: int) -> None:
        bus = self._buses.pop(session_id, None)
        if bus is not None:
            bus.close()

    def close_all(self) -> None:
        for bus in self._buses.values():
            bus.close()
        self._buses.clear()


# ---------------------------------------------------------------------------
# Singleton instances — imported by scheduler and API
# ---------------------------------------------------------------------------

global_bus = EventBus()
session_buses = SessionEventBuses()