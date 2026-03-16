from sim.rate_limiter import RateLimiter
from schedulers.backoff import BackoffScheduler
from schedulers.fifo import FIFOScheduler
from schedulers.mapreduce import MapReduceScheduler
from schedulers.mapreduce_events_demo import MapReduceWithEventsScheduler
from schedulers.mapreduce_skip import MapReduceSkipScheduler
from schedulers.mapreduce_skip_adaptive import MapReduceSkipAdaptiveScheduler


def get_scheduler(name: str, limiter: RateLimiter):
    if name == "backoff":
        return BackoffScheduler(limiter)
    elif name == "fifo":
        return FIFOScheduler(limiter)
    elif name == "mapreduce":
        return MapReduceScheduler(limiter)
    elif name == "mapreduce_events":
        return MapReduceWithEventsScheduler(limiter)
    elif name == "mapreduce_skip":
        return MapReduceSkipScheduler(limiter)
    elif name == "mapreduce_skip_adaptive":
        return MapReduceSkipAdaptiveScheduler(limiter)
    else:
        raise ValueError(
            f"Unknown scheduler: {name}. Use: backoff, fifo, mapreduce, "
            f"mapreduce_events, mapreduce_skip, mapreduce_skip_adaptive"
        )
