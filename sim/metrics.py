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
    llm_calls: int = 0       # total LLM calls (orchestrator + tools)
    tool_calls: int = 0      # total tool invocations

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
                     cost_by_scheduler: dict[str, float],
                     trace_stats_by_scheduler: dict[str, dict] = None,
                     rl_stats_by_scheduler: dict[str, dict] = None):
    """Print side-by-side comparison table across schedulers."""
    schedulers = list(results_by_scheduler.keys())
    col_w = max(16, *(len(s) + 2 for s in schedulers))

    stats_by = {}
    for name, results in results_by_scheduler.items():
        ok = [r for r in results if r.success]
        stats_by[name] = compute_stats([r.duration for r in ok]) if ok else {}

    trace_stats_by_scheduler = trace_stats_by_scheduler or {}
    rl_stats_by_scheduler = rl_stats_by_scheduler or {}

    print(f"\n{'='*80}")
    print(f" SCHEDULER COMPARISON")
    print(f"{'='*80}")

    # Header
    header = f"{'Metric':<18}"
    for s in schedulers:
        header += f" | {s:>{col_w}}"
    print(header)
    print("-" * len(header))

    def _row(label, values):
        row = f"{label:<18}"
        for v in values:
            row += f" | {v:>{col_w}}"
        print(row)

    # Sessions OK
    _row("Sessions OK", [
        f"{stats_by[s].get('count', 0)}/{len(results_by_scheduler[s])}"
        for s in schedulers
    ])

    # LLM calls
    _row("Total LLM calls", [
        f"{sum(r.llm_calls for r in results_by_scheduler[s] if r.success)}"
        for s in schedulers
    ])
    _row("Mean LLM calls", [
        f"{statistics.mean([r.llm_calls for r in results_by_scheduler[s] if r.success and r.llm_calls > 0]):.1f}"
        if any(r.success and r.llm_calls > 0 for r in results_by_scheduler[s])
        else "—"
        for s in schedulers
    ])

    # Wall time
    _row("Wall time", [
        f"{trace_stats_by_scheduler.get(s, {}).get('wall_time', 0):.0f}s"
        for s in schedulers
    ])

    # Session duration stats
    for label, key in [("Mean session", "mean"), ("Median", "median"),
                       ("p95", "p95"), ("Min", "min"), ("Max", "max"),
                       ("Stdev", "stdev")]:
        _row(label, [
            f"{stats_by[s].get(key, 0):.1f}s" for s in schedulers
        ])

    # Cost
    _row("Cost", [
        f"${cost_by_scheduler.get(s, 0):.4f}" for s in schedulers
    ])

    # Rate limiter stats
    _row("RPM throttles", [
        f"{rl_stats_by_scheduler.get(s, {}).get('rpm_throttles', 0)}"
        for s in schedulers
    ])
    _row("TPM throttles", [
        f"{rl_stats_by_scheduler.get(s, {}).get('tpm_throttles', 0)}"
        for s in schedulers
    ])
    _row("RPM waits", [
        f"{rl_stats_by_scheduler.get(s, {}).get('rpm_waits', 0)}"
        for s in schedulers
    ])
    _row("TPM waits", [
        f"{rl_stats_by_scheduler.get(s, {}).get('tpm_waits', 0)}"
        for s in schedulers
    ])
    _row("TPM skips", [
        f"{rl_stats_by_scheduler.get(s, {}).get('tpm_skips', 0)}"
        for s in schedulers
    ])

    # Bottleneck indicator
    def _bottleneck(s):
        rl = rl_stats_by_scheduler.get(s, {})
        rpm = rl.get("rpm_throttles", 0) + rl.get("rpm_waits", 0)
        tpm = rl.get("tpm_throttles", 0) + rl.get("tpm_waits", 0)
        if rpm == 0 and tpm == 0:
            return "neither"
        if rpm >= tpm:
            return f"RPM ({rpm}v{tpm})"
        return f"TPM ({tpm}v{rpm})"

    _row("Bottleneck", [_bottleneck(s) for s in schedulers])

    # Token overestimation
    _row("Token overest.", [
        f"{rl_stats_by_scheduler.get(s, {}).get('total_estimated', 0) / max(1, rl_stats_by_scheduler.get(s, {}).get('total_actual', 1)):.1f}x"
        for s in schedulers
    ])

    # Parallelism
    _row("Parallelism", [
        f"{trace_stats_by_scheduler.get(s, {}).get('parallelism', 0):.1f}x"
        for s in schedulers
    ])

    # Queue wait
    _row("Mean queue wait", [
        f"{trace_stats_by_scheduler.get(s, {}).get('mean_queue_wait', 0):.1f}s"
        if trace_stats_by_scheduler.get(s, {}).get('mean_queue_wait', 0) > 0.01
        else "—"
        for s in schedulers
    ])
    _row("Max queue wait", [
        f"{trace_stats_by_scheduler.get(s, {}).get('max_queue_wait', 0):.1f}s"
        if trace_stats_by_scheduler.get(s, {}).get('max_queue_wait', 0) > 0.01
        else "—"
        for s in schedulers
    ])

    print(f"{'='*80}\n")


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
