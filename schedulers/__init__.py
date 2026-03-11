from rate_limiter import RateLimiter
from schedulers.backoff import BackoffScheduler
from schedulers.fifo import FIFOScheduler
from schedulers.sjf import SJFScheduler
from schedulers.mapreduce import MapReduceScheduler
from schedulers.adaptive_sjf import AdaptiveSJFScheduler
from schedulers.token_sjf import TokenSJFScheduler
from schedulers.combined_map_reduce_adaptive_sjf import CombinedMapReduceAdaptiveSJFScheduler
from schedulers.combined_map_reduce_token_sjf import CombinedMapReduceTokenSJFScheduler
from schedulers.token_sjf_skip import TokenSJFSkipScheduler
from schedulers.mapreduce_improved import MapReduceImprovedScheduler


def get_scheduler(name: str, limiter: RateLimiter):
    if name == "backoff":
        return BackoffScheduler(limiter)
    elif name == "fifo":
        return FIFOScheduler(limiter)
    elif name == "sjf":
        return SJFScheduler(limiter)
    elif name == "mapreduce":
        return MapReduceScheduler(limiter)
    elif name == "mapreduce_improved":
        return MapReduceImprovedScheduler(limiter)
    elif name == "adaptive_sjf":
        return AdaptiveSJFScheduler(limiter)
    elif name == "token_sjf":
        return TokenSJFScheduler(limiter)
    elif name == "token_sjf_skip":
        return TokenSJFSkipScheduler(limiter)
    elif name == "combined_mapreduce_asjf":
        return CombinedMapReduceAdaptiveSJFScheduler(limiter)
    elif name == "combined_mapreduce_tsjf":
        return CombinedMapReduceTokenSJFScheduler(limiter)
    else:
        raise ValueError(
            f"Unknown scheduler: {name}. Use: backoff, fifo, sjf, mapreduce, "
            f"mapreduce_improved, adaptive_sjf, token_sjf, token_sjf_skip, "
            f"combined_mapreduce_asjf, combined_mapreduce_tsjf"
        )
