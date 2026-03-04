import statistics
from dataclasses import dataclass
from typing import Optional


@dataclass
class SessionResult:
    session_id: int
    batch_id: int
    prompt: str
    start_time: float
    end_time: float
    success: bool
    error: Optional[str] = None

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


def compute_stats(durations: list[float]) -> dict:
    if not durations:
        return {}
    sorted_d = sorted(durations)
    return {
        "count": len(durations),
        "mean": statistics.mean(durations),
        "median": statistics.median(durations),
        "p95": sorted_d[int(len(sorted_d) * 0.95)],
        "p99": sorted_d[int(len(sorted_d) * 0.99)] if len(sorted_d) > 1 else sorted_d[-1],
        "min": min(durations),
        "max": max(durations),
        "stdev": statistics.stdev(durations) if len(durations) > 1 else 0.0,
    }


def print_batch_summary(batch_id: int, results: list["SessionResult"]):
    ok = [r for r in results if r.success]
    err = [r for r in results if not r.success]
    print(f"\n{'='*60}")
    print(f" BATCH {batch_id} SUMMARY  ({len(ok)} ok / {len(err)} failed)")
    print(f"{'='*60}")
    if ok:
        stats = compute_stats([r.duration for r in ok])
        print(f"  Session durations (successful):")
        print(f"    count  = {stats['count']}")
        print(f"    mean   = {stats['mean']:.2f}s")
        print(f"    median = {stats['median']:.2f}s")
        print(f"    p95    = {stats['p95']:.2f}s")
        print(f"    min    = {stats['min']:.2f}s")
        print(f"    max    = {stats['max']:.2f}s")
        print(f"    stdev  = {stats['stdev']:.2f}s")
    for r in err:
        print(f"  FAILED session {r.session_id}: {r.error}")


def print_aggregate_summary(all_results: list["SessionResult"], cost_tracker_obj):
    ok = [r for r in all_results if r.success]
    err = [r for r in all_results if not r.success]
    print(f"\n{'='*70}")
    print(f" AGGREGATE SUMMARY  ({len(ok)} ok / {len(err)} failed)")
    print(f"{'='*70}")
    if ok:
        stats = compute_stats([r.duration for r in ok])
        print(f"  Session durations across all batches:")
        print(f"    count  = {stats['count']}")
        print(f"    mean   = {stats['mean']:.2f}s")
        print(f"    median = {stats['median']:.2f}s")
        print(f"    p95    = {stats['p95']:.2f}s")
        print(f"    p99    = {stats['p99']:.2f}s")
        print(f"    min    = {stats['min']:.2f}s")
        print(f"    max    = {stats['max']:.2f}s")
        print(f"    stdev  = {stats['stdev']:.2f}s")
    if cost_tracker_obj:
        print(f"\n  Cost:")
        print(cost_tracker_obj.summary())
    print(f"{'='*70}\n")
