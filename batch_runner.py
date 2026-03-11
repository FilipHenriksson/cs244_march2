"""Batch experiment runner — compare scheduling policies under identical workloads.

Experiment design
-----------------
This script is the primary entry point for generating research results.  It
runs one or more scheduling policies back-to-back, each processing the *same*
workload, then prints a side-by-side comparison of latency and cost metrics.

Sessions and batching
---------------------
A **session** is one call to ``orchestrate()``, which drives the full five-stage
research pipeline (breakdown → agents → synthesis → reviewers → finalize),
producing ~15-25 LLM calls per session.  A **batch** is a group of sessions
launched concurrently against a single shared rate limiter, simulating multiple
users contending for the same API quota.

Workload determinism
--------------------
Before any scheduler runs, ``generate_workload()`` pre-generates the full set
of prompt assignments and arrival times from a fixed random seed.  Every
scheduler therefore sees *exactly* the same session prompts arriving at the same
relative times.  Between schedulers all mutable state (rate-limiter buckets,
trace log, cost tracker) is reset so results are independent.

Arrival modes
-------------
Sessions do not all start at t=0.  ``compute_arrival_times()`` supports three
modes that control how sessions enter the system:
  - **fixed**   — uniform spacing (deterministic)
  - **poisson** — exponentially-distributed inter-arrivals (random, realistic)
  - **wave**    — bursty clusters separated by longer gaps

Execution pseudocode
--------------------
::

  # 1. Generate one deterministic workload (--seed)
  prompts[0..N], arrivals[0..N] = generate_workload(--sessions N, --stagger, --stagger-mode, --seed)

  # 2. Run each scheduler against that same workload
  for scheduler_name in selected_schedulers:        # --scheduler / --schedulers / --all-schedulers

      reset(rate_limiter, trace, cost_tracker)       # fresh state per scheduler
      scheduler = new Scheduler(scheduler_name, rate_limiter(--rpm, --tpm))

      # 3. Launch all N sessions concurrently; each sleeps until its arrival time
      async for i in 0..N-1:
          sleep(arrivals[i])                         # staggered entry into the system
          session_result[i] = orchestrate(prompts[i])
              # orchestrate() issues ~15-25 LLM calls across 5 pipeline stages.
              # Every llm_call() is routed through the shared scheduler, which
              # queues, prioritizes (--scheduler), and rate-limits (--rpm/--tpm)
              # each request before forwarding it to the OpenAI API.

      collect results, cost

  # 4. Compare scheduler metrics side-by-side (if >1 scheduler)
  print_comparison(results_by_scheduler, cost_by_scheduler)
"""

import argparse
import asyncio
import logging
import random
import time

from dotenv import load_dotenv
load_dotenv()

from prompts import RESEARCH_PROMPTS
from orchestrator import orchestrate
from rate_limiter import RateLimiter
from schedulers import get_scheduler
from llm import set_scheduler, set_max_tokens
from trace import trace
import cost_tracker as ct
from cost_tracker import init_cost_tracker, CostLimitExceeded
from metrics import (SessionResult, print_aggregate_summary, print_comparison)

ALL_SCHEDULERS = ["backoff", "fifo", "sjf", "mapreduce", "mapreduce_improved",
                  "adaptive_sjf", "token_sjf",
                  "token_sjf_skip", "combined_mapreduce_asjf", "combined_mapreduce_tsjf"]


async def run_session(session_id: int, prompt: str,
                      stagger_delay: float) -> SessionResult:
    """Run one orchestrate() session, sleeping *stagger_delay* seconds first."""
    await asyncio.sleep(stagger_delay)
    start = time.time()
    try:
        await orchestrate(prompt, session_id=session_id)
        end = time.time()
        print(f"  [session={session_id}] done in {end - start:.1f}s "
              f"(arrived at {stagger_delay:.1f}s)")
        return SessionResult(session_id=session_id,
                             prompt=prompt, start_time=start, end_time=end,
                             success=True)
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


def compute_arrival_times(n: int, stagger: float, mode: str,
                          rng: random.Random = None) -> list[float]:
    """Return *n* cumulative arrival offsets (seconds from t=0).

    *stagger* is the mean inter-arrival gap.  *mode* selects the distribution:
    fixed (deterministic), poisson (memoryless), or wave (bursty clusters).
    """
    if mode == "fixed":
        return [i * stagger for i in range(n)]
    elif mode == "poisson":
        r = rng or random
        times = [0.0]
        for _ in range(n - 1):
            inter_arrival = r.expovariate(1.0 / stagger)
            times.append(times[-1] + inter_arrival)
        return times
    elif mode == "wave":
        r = rng or random
        times = []
        t = 0.0
        remaining = n
        while remaining > 0:
            # Random burst size: 2-5 sessions
            burst_size = min(r.randint(2, 5), remaining)
            # Sessions within a burst arrive 1-2s apart
            for j in range(burst_size):
                times.append(t)
                t += r.uniform(0.5, 2.0)
            remaining -= burst_size
            # Gap between bursts: 15-35s
            if remaining > 0:
                t += r.uniform(15.0, 35.0)
        return times
    else:
        raise ValueError(f"Unknown stagger mode: {mode}")


def generate_workload(sessions: int, stagger: float,
                      stagger_mode: str, seed: int) -> dict:
    """Pre-generate the full workload so every scheduler gets the same one."""
    rng = random.Random(seed)
    prompts = [rng.choice(RESEARCH_PROMPTS) for _ in range(sessions)]
    arrivals = compute_arrival_times(sessions, stagger, stagger_mode, rng)
    return {"prompts": prompts, "arrivals": arrivals}


async def run_one_scheduler(scheduler_name: str, args, limiter: RateLimiter,
                            workload: dict, num_schedulers: int):
    """Execute a full batch of sessions under one scheduler, returning results.

    Resets all shared state (rate limiter, trace, cost tracker) so this
    scheduler's run is independent of any prior one.  Sessions are launched
    concurrently with pre-computed arrival delays; the scheduler and rate
    limiter mediate their contention for the API.
    """
    print(f"\n{'#'*70}")
    print(f" SCHEDULER: {scheduler_name.upper()}")
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
    print(f" Arrival times: {', '.join(f'{t:.1f}s' for t in arrivals)}")

    try:
        tasks = [
            asyncio.create_task(
                run_session(
                    session_id=i,
                    prompt=p,
                    stagger_delay=arrivals[i],
                )
            )
            for i, p in enumerate(prompts)
        ]
        results = list(await asyncio.gather(*tasks))
    finally:
        await scheduler.stop()

    cost = ct.cost_tracker.total_cost if ct.cost_tracker else 0.0
    print_aggregate_summary(results, ct.cost_tracker)
    trace.print_summary(rl_stats=limiter.stats)
    return results, cost


def _configure_logging():
    """Suppress noisy HTTP/OpenAI logs; isolate rate-limiter logger."""
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
    parser = argparse.ArgumentParser(description="Batch research agent runner")
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
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Max tokens per completion (default: 1024)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for workload generation (default: 42)")
    return parser.parse_args()


async def main():
    args = _parse_args()
    _configure_logging()

    if args.all_schedulers:
        schedulers = ALL_SCHEDULERS
    elif args.schedulers:
        schedulers = args.schedulers
    else:
        schedulers = [args.scheduler]
    set_max_tokens(args.max_tokens)
    limiter = RateLimiter(rpm=args.rpm, tpm=args.tpm)

    # Pre-generate workload once — all schedulers get identical prompts & arrivals
    workload = generate_workload(args.sessions, args.stagger,
                                 args.stagger_mode, args.seed)

    total_sessions = args.sessions * len(schedulers)
    print(f"Runner: {len(schedulers)} scheduler(s) x {args.sessions} sessions "
          f"= {total_sessions} total")
    print(f"Schedulers: {', '.join(schedulers)} | RPM: {args.rpm} | TPM: {args.tpm} | max_tokens: {args.max_tokens}")
    print(f"Arrivals: every {args.stagger}s ({args.stagger_mode}) | "
          f"Cost limit: ${args.cost_limit:.2f}")
    print(f"Seed: {args.seed}")

    # Print workload for verification
    print(f"\nWorkload (same for all schedulers):")
    for i, (p, t) in enumerate(zip(workload["prompts"], workload["arrivals"])):
        print(f"  Session {i}: arrives at {t:.1f}s — {p[:60]}...")

    results_by_scheduler: dict[str, list[SessionResult]] = {}
    cost_by_scheduler: dict[str, float] = {}

    for sched_name in schedulers:
        results, cost = await run_one_scheduler(
            sched_name, args, limiter, workload, len(schedulers))
        results_by_scheduler[sched_name] = results
        cost_by_scheduler[sched_name] = cost

    if len(schedulers) > 1:
        print_comparison(results_by_scheduler, cost_by_scheduler)


if __name__ == "__main__":
    asyncio.run(main())
