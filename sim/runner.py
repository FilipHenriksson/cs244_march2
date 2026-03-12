"""Batch experiment runner — compare scheduling policies under identical workloads.

Runs one or more scheduling policies back-to-back, each processing the *same*
workload, then prints a side-by-side comparison of latency and cost metrics.

Usage::

    python -m sim.runner --sessions 15 --scheduler fifo
    python -m sim.runner --sessions 10 --all-schedulers --stagger-mode poisson
    python -m sim.runner --sessions 15 --all-schedulers --output-dir results/
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from agent import run_agent
from sim.rate_limiter import RateLimiter
from sim.cost_tracker import CostLimitExceeded, init_cost_tracker
import sim.cost_tracker as ct
from sim.trace import trace
from sim.metrics import SessionResult, print_aggregate_summary, print_comparison
from sim.workload import generate_workload
from schedulers import get_scheduler
from llm import set_scheduler, set_max_tokens

ALL_SCHEDULERS = [
    "backoff", "fifo", "mapreduce", "mapreduce_skip",
]


async def run_session(session_id: int, prompt: str,
                      stagger_delay: float,
                      prompt_mode: str = "default") -> SessionResult:
    """Run one agent session, sleeping *stagger_delay* seconds first."""
    await asyncio.sleep(stagger_delay)
    start = time.time()
    try:
        agent_result = await run_agent(prompt, session_id=session_id, prompt_mode=prompt_mode)
        end = time.time()
        print(f"  [session={session_id}] done in {end - start:.1f}s "
              f"(arrived at {stagger_delay:.1f}s, "
              f"{agent_result.llm_calls} LLM calls, "
              f"{agent_result.tool_calls} tool calls)")
        return SessionResult(session_id=session_id,
                             prompt=prompt, start_time=start, end_time=end,
                             success=True,
                             llm_calls=agent_result.llm_calls,
                             tool_calls=agent_result.tool_calls)
    except CostLimitExceeded as e:
        end = time.time()
        print(f"  [session={session_id}] COST LIMIT: {e}")
        return SessionResult(session_id=session_id,
                             prompt=prompt, start_time=start, end_time=end,
                             success=False, error=str(e))
    except Exception as e:
        end = time.time()
        print(f"  [session={session_id}] ERROR: {e}")
        return SessionResult(session_id=session_id,
                             prompt=prompt, start_time=start, end_time=end,
                             success=False, error=str(e))


async def run_one_scheduler(scheduler_name: str, args, limiter: RateLimiter,
                            workload: dict, num_schedulers: int,
                            sched_index: int = 0):
    """Execute a full batch of sessions under one scheduler, returning results."""
    print(f"\n{'#'*70}")
    print(f" SCHEDULER: {scheduler_name.upper()}  [{sched_index + 1}/{num_schedulers}]")
    print(f"{'#'*70}")

    # Fresh state
    limiter.reset()
    trace.reset()
    per_scheduler_limit = args.cost_limit / num_schedulers
    init_cost_tracker(per_scheduler_limit)

    scheduler = get_scheduler(scheduler_name, limiter)
    set_scheduler(scheduler)
    scheduler.start()

    prompts = workload["prompts"]
    arrivals = workload["arrivals"]

    print(f" Launching {len(prompts)} sessions with continuous arrivals")
    print(f" Arrival times: {', '.join(f'{t:.1f}s' for t in arrivals[:20])}"
          f"{'...' if len(arrivals) > 20 else ''}")

    try:
        prompt_mode = getattr(args, 'prompt_mode', 'default')
        tasks = [
            asyncio.create_task(
                run_session(session_id=i, prompt=p, stagger_delay=arrivals[i],
                            prompt_mode=prompt_mode)
            )
            for i, p in enumerate(prompts)
        ]
        results = list(await asyncio.gather(*tasks))
    finally:
        await scheduler.stop()

    cost = ct.cost_tracker.total_cost if ct.cost_tracker else 0.0
    trace_stats = trace.get_stats()
    rl_stats = limiter.stats
    failed_cost = [r for r in results
                   if not r.success and "Cost limit" in (r.error or "")]
    was_incomplete = len(failed_cost) > 0
    if was_incomplete:
        print(f"\n  *** WARNING: {len(failed_cost)} session(s) hit the cost limit! ***")
        print(f"  *** Results for '{scheduler_name}' are INCOMPLETE. ***\n")
    print_aggregate_summary(results, ct.cost_tracker)
    trace.print_summary(rl_stats=rl_stats)
    return results, cost, was_incomplete, trace_stats, rl_stats


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

class _Tee:
    """Write to both original stream and a file simultaneously."""
    def __init__(self, file, original):
        self._file = file
        self._original = original

    def write(self, data):
        self._original.write(data)
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._original.flush()
        self._file.flush()

    def fileno(self):
        return self._original.fileno()

    def isatty(self):
        return False


def _write_sessions_csv(path, results_by_scheduler):
    """Write per-session results to CSV."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["scheduler", "session_id", "prompt", "arrival_time",
                     "duration_s", "llm_calls", "tool_calls", "success", "error"])
        for sched, results in results_by_scheduler.items():
            for r in results:
                w.writerow([
                    sched, r.session_id, r.prompt[:80],
                    f"{r.start_time:.2f}",
                    f"{r.duration:.2f}",
                    r.llm_calls, r.tool_calls,
                    r.success, r.error or "",
                ])


def _write_comparison_csv(path, results_by_scheduler, cost_by_scheduler,
                          trace_stats_by_scheduler, rl_stats_by_scheduler):
    """Write per-scheduler summary metrics to CSV."""
    import statistics as stats_mod

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "scheduler", "sessions_ok", "sessions_total",
            "total_llm_calls", "mean_llm_calls",
            "wall_time_s", "mean_session_s", "median_session_s",
            "p95_session_s", "p99_session_s", "min_session_s", "max_session_s",
            "stdev_session_s", "cost_usd",
            "rpm_throttles", "tpm_throttles", "token_overestimate_ratio",
            "parallelism", "mean_queue_wait_s", "max_queue_wait_s",
        ])
        for sched in results_by_scheduler:
            results = results_by_scheduler[sched]
            ok = [r for r in results if r.success]
            durations = [r.duration for r in ok]
            llm_counts = [r.llm_calls for r in ok if r.llm_calls > 0]
            ts = trace_stats_by_scheduler.get(sched, {})
            rl = rl_stats_by_scheduler.get(sched, {})

            if durations:
                sorted_d = sorted(durations)
                mean_d = stats_mod.mean(durations)
                median_d = stats_mod.median(durations)
                p95 = sorted_d[int(len(sorted_d) * 0.95)]
                p99 = sorted_d[int(len(sorted_d) * 0.99)] if len(sorted_d) > 1 else sorted_d[-1]
                stdev_d = stats_mod.stdev(durations) if len(durations) > 1 else 0.0
            else:
                mean_d = median_d = p95 = p99 = stdev_d = 0.0
                sorted_d = []

            est = rl.get("total_estimated", 0)
            act = rl.get("total_actual", 1)

            w.writerow([
                sched, len(ok), len(results),
                sum(r.llm_calls for r in ok),
                f"{stats_mod.mean(llm_counts):.1f}" if llm_counts else 0,
                f"{ts.get('wall_time', 0):.1f}",
                f"{mean_d:.2f}", f"{median_d:.2f}",
                f"{p95:.2f}",
                f"{p99:.2f}",
                f"{min(durations):.2f}" if durations else 0,
                f"{max(durations):.2f}" if durations else 0,
                f"{stdev_d:.2f}",
                f"{cost_by_scheduler.get(sched, 0):.6f}",
                rl.get("rpm_throttles", 0),
                rl.get("tpm_throttles", 0),
                f"{est / max(1, act):.2f}",
                f"{ts.get('parallelism', 0):.2f}",
                f"{ts.get('mean_queue_wait', 0):.2f}",
                f"{ts.get('max_queue_wait', 0):.2f}",
            ])


def _write_config_json(path, args, schedulers):
    """Save run parameters for reproducibility."""
    config = {
        "timestamp": datetime.now().isoformat(),
        "sessions": args.sessions,
        "stagger": args.stagger,
        "stagger_mode": args.stagger_mode,
        "schedulers": schedulers,
        "rpm": args.rpm,
        "tpm": args.tpm,
        "cost_limit": args.cost_limit,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "prompt_mode": args.prompt_mode,
    }
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------

def _configure_logging():
    """Suppress noisy HTTP/OpenAI logs."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in ("openai", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)
    rl_log = logging.getLogger("rate_limiter")
    rl_log.propagate = False
    if not rl_log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(message)s"))
        rl_log.addHandler(h)
        rl_log.setLevel(logging.INFO)


def _parse_args():
    parser = argparse.ArgumentParser(description="Batch experiment runner")
    parser.add_argument("--sessions", type=int, default=15,
                        help="Total number of sessions (default: 15)")
    parser.add_argument("--stagger", type=float, default=4.0,
                        help="Mean seconds between session arrivals (default: 4.0)")
    parser.add_argument("--stagger-mode", type=str, default="fixed",
                        choices=["fixed", "poisson", "wave"],
                        help="Arrival mode: fixed, poisson, or wave (default: fixed)")
    parser.add_argument("--scheduler", type=str, default="fifo",
                        choices=ALL_SCHEDULERS)
    parser.add_argument("--schedulers", type=str, nargs="+",
                        choices=ALL_SCHEDULERS,
                        help="Run specific schedulers sequentially and compare")
    parser.add_argument("--all-schedulers", action="store_true",
                        help="Run all schedulers sequentially and compare")
    parser.add_argument("--rpm", type=int, default=30,
                        help="Rate limit: requests per minute (default: 30)")
    parser.add_argument("--tpm", type=int, default=200_000,
                        help="Rate limit: tokens per minute (default: 200000)")
    parser.add_argument("--cost-limit", type=float, default=20.0,
                        help="Hard cost cap in USD (default: 20.0)")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Max tokens per completion (default: 2048)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for workload generation (default: 42)")
    parser.add_argument("--prompt-mode", type=str, default="strict",
                        choices=["default", "strict"],
                        help="Prompt mode: default (flexible) or strict (fixed tool counts, default: strict)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for output files (log, CSVs, config). "
                             "Auto-creates a timestamped subdir.")
    return parser.parse_args()


async def main():
    args = _parse_args()
    _configure_logging()

    if args.all_schedulers:
        schedulers = list(ALL_SCHEDULERS)
    elif args.schedulers:
        schedulers = list(args.schedulers)
    else:
        schedulers = [args.scheduler]

    # Set up output directory
    out_dir = None
    log_file = None
    if args.output_dir:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(args.output_dir, f"run_{ts}")
        os.makedirs(out_dir, exist_ok=True)
        log_path = os.path.join(out_dir, "run.log")
        log_file = open(log_path, "w")
        sys.stdout = _Tee(log_file, sys.__stdout__)
        sys.stderr = _Tee(log_file, sys.__stderr__)

    try:
        results_by, cost_by, trace_by, rl_by = await _run(args, schedulers)

        # Write output files
        if out_dir:
            _write_sessions_csv(
                os.path.join(out_dir, "sessions.csv"), results_by)
            _write_comparison_csv(
                os.path.join(out_dir, "comparison.csv"),
                results_by, cost_by, trace_by, rl_by)
            _write_config_json(
                os.path.join(out_dir, "config.json"), args, schedulers)
    finally:
        if log_file:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            log_file.close()
            print(f"\nResults written to {out_dir}/")
            print(f"  run.log         — full console output")
            print(f"  sessions.csv    — per-session data ({sum(len(v) for v in results_by.values())} rows)")
            print(f"  comparison.csv  — per-scheduler summary ({len(results_by)} rows)")
            print(f"  config.json     — run parameters")


async def _run(args, schedulers):
    set_max_tokens(args.max_tokens)
    limiter = RateLimiter(rpm=args.rpm, tpm=args.tpm)

    workload = generate_workload(args.sessions, args.stagger,
                                 args.stagger_mode, args.seed)

    total_sessions = args.sessions * len(schedulers)
    print(f"Runner: {len(schedulers)} scheduler(s) x {args.sessions} sessions "
          f"= {total_sessions} total")
    print(f"Schedulers: {', '.join(schedulers)} | RPM: {args.rpm} | "
          f"TPM: {args.tpm} | max_tokens: {args.max_tokens}")
    print(f"Arrivals: every {args.stagger}s ({args.stagger_mode}) | "
          f"Cost limit: ${args.cost_limit:.2f} | Mode: {args.prompt_mode}")
    print(f"Seed: {args.seed}")

    print(f"\nWorkload (same for all schedulers):")
    for i, (p, t) in enumerate(zip(workload["prompts"], workload["arrivals"])):
        print(f"  Session {i}: arrives at {t:.1f}s — {p[:60]}...")

    results_by_scheduler: dict[str, list[SessionResult]] = {}
    cost_by_scheduler: dict[str, float] = {}
    trace_stats_by_scheduler: dict[str, dict] = {}
    rl_stats_by_scheduler: dict[str, dict] = {}
    incomplete_schedulers: list[str] = []

    run_start = time.time()
    for i, sched_name in enumerate(schedulers):
        if i > 0:
            cooldown = 30
            elapsed = time.time() - run_start
            print(f"\n--- Cooldown: sleeping {cooldown}s between scheduler runs "
                  f"(elapsed: {elapsed:.0f}s) ---\n")
            await asyncio.sleep(cooldown)
        results, cost, was_incomplete, trace_stats, rl_stats = await run_one_scheduler(
            sched_name, args, limiter, workload, len(schedulers),
            sched_index=i)
        results_by_scheduler[sched_name] = results
        cost_by_scheduler[sched_name] = cost
        trace_stats_by_scheduler[sched_name] = trace_stats
        rl_stats_by_scheduler[sched_name] = rl_stats
        if was_incomplete:
            incomplete_schedulers.append(sched_name)

    total_elapsed = time.time() - run_start
    print(f"\n{'='*70}")
    print(f" ALL SCHEDULERS COMPLETE — total elapsed: {total_elapsed:.0f}s "
          f"({total_elapsed/60:.1f}min)")
    print(f"{'='*70}")

    if incomplete_schedulers:
        print(f"\n*** ERROR: The following schedulers hit the cost limit and "
              f"have INVALID results: {', '.join(incomplete_schedulers)} ***")
        print(f"*** Increase --cost-limit and re-run. ***\n")

    print_comparison(results_by_scheduler, cost_by_scheduler,
                     trace_stats_by_scheduler, rl_stats_by_scheduler)

    return results_by_scheduler, cost_by_scheduler, trace_stats_by_scheduler, rl_stats_by_scheduler


if __name__ == "__main__":
    asyncio.run(main())
