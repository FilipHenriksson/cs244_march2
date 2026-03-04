from rate_limiter import RateLimiter
from schedulers.backoff import BackoffScheduler
from schedulers.fifo import FIFOScheduler
from schedulers.sjf import SJFScheduler


def get_scheduler(name: str, limiter: RateLimiter):
    if name == "backoff":
        return BackoffScheduler(limiter)
    elif name == "fifo":
        return FIFOScheduler(limiter)
    elif name == "sjf":
        return SJFScheduler(limiter)
    else:
        raise ValueError(f"Unknown scheduler: {name}. Use: backoff, fifo, sjf")
