import statistics
from dataclasses import dataclass
from typing import Optional


@dataclass
class SessionResult:
    session_id: int
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


def print_comparison(results_by_scheduler: dict[str, list["SessionResult"]],
                     cost_by_scheduler: dict[str, float]):
    """Print side-by-side comparison table across schedulers."""
    schedulers = list(results_by_scheduler.keys())
    stats_by = {}
    for name, results in results_by_scheduler.items():
        ok = [r for r in results if r.success]
        stats_by[name] = compute_stats([r.duration for r in ok]) if ok else {}

    print(f"\n{'='*70}")
    print(f" SCHEDULER COMPARISON")
    print(f"{'='*70}")

    # Header
    header = f"{'Metric':<12}"
    for s in schedulers:
        header += f" | {s:>12}"
    print(header)
    print("-" * len(header))

    # Rows
    for metric in ["count", "mean", "median", "p95", "min", "max", "stdev"]:
        row = f"{metric:<12}"
        for s in schedulers:
            val = stats_by[s].get(metric, 0)
            if metric == "count":
                row += f" | {val:>12}"
            else:
                row += f" | {val:>11.2f}s"
        print(row)

    # Cost row
    row = f"{'cost':<12}"
    for s in schedulers:
        c = cost_by_scheduler.get(s, 0)
        row += f" | {f'${c:.4f}':>12}"
    print(row)

    print(f"{'='*70}\n")


def print_aggregate_summary(all_results: list["SessionResult"], cost_tracker_obj):
    ok = [r for r in all_results if r.success]
    err = [r for r in all_results if not r.success]
    print(f"\n{'='*70}")
    print(f" SUMMARY  ({len(ok)} ok / {len(err)} failed)")
    print(f"{'='*70}")
    if ok:
        stats = compute_stats([r.duration for r in ok])
        print(f"  Session durations:")
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
    for r in err:
        print(f"  FAILED session {r.session_id}: {r.error}")
    print(f"{'='*70}\n")
