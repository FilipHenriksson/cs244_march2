import argparse
import asyncio
import random
import time

from dotenv import load_dotenv
load_dotenv()

from prompts import RESEARCH_PROMPTS
from orchestrator import orchestrate
from rate_limiter import RateLimiter
from schedulers import get_scheduler
from llm import set_scheduler
from trace import trace
import cost_tracker as ct
from cost_tracker import init_cost_tracker, CostLimitExceeded
from metrics import (SessionResult, print_batch_summary,
                     print_aggregate_summary, print_comparison)

ALL_SCHEDULERS = ["backoff", "fifo", "sjf", "mapreduce"]


async def run_session(session_id: int, batch_id: int,
                      prompt: str, stagger_delay: float) -> SessionResult:
    """Run a single orchestrate() session with staggered start."""
    await asyncio.sleep(stagger_delay)
    start = time.time()
    try:
        await orchestrate(prompt, session_id=session_id)
        end = time.time()
        print(f"  [batch={batch_id} session={session_id}] done in {end - start:.1f}s")
        return SessionResult(session_id=session_id, batch_id=batch_id,
                             prompt=prompt, start_time=start, end_time=end,
                             success=True)
    except CostLimitExceeded as e:
        end = time.time()
        print(f"  [batch={batch_id} session={session_id}] COST LIMIT: {e}")
        return SessionResult(session_id=session_id, batch_id=batch_id,
                             prompt=prompt, start_time=start, end_time=end,
                             success=False, error=str(e))
    except Exception as e:
        end = time.time()
        print(f"  [batch={batch_id} session={session_id}] ERROR: {e}")
        return SessionResult(session_id=session_id, batch_id=batch_id,
                             prompt=prompt, start_time=start, end_time=end,
                             success=False, error=str(e))


def compute_stagger_delays(n: int, stagger: float, mode: str,
                           rng: random.Random = None) -> list[float]:
    """Compute cumulative stagger delays for n sessions."""
    if mode == "fixed":
        return [i * stagger for i in range(n)]
    elif mode == "poisson":
        r = rng or random
        delays = [0.0]
        for _ in range(n - 1):
            inter_arrival = r.expovariate(1.0 / stagger)
            delays.append(delays[-1] + inter_arrival)
        return delays
    else:
        raise ValueError(f"Unknown stagger mode: {mode}")


async def run_batch(batch_id: int, prompts: list[str],
                    stagger_delays: list[float]) -> list[SessionResult]:
    """Launch all sessions in a batch with pre-computed stagger delays."""
    print(f"\n{'='*60}")
    print(f" BATCH {batch_id}: launching {len(prompts)} sessions")
    print(f" Arrival delays: {', '.join(f'{d:.1f}s' for d in stagger_delays)}")
    print(f"{'='*60}")

    tasks = [
        asyncio.create_task(
            run_session(
                session_id=batch_id * len(prompts) + i,
                batch_id=batch_id,
                prompt=p,
                stagger_delay=stagger_delays[i],
            )
        )
        for i, p in enumerate(prompts)
    ]
    return list(await asyncio.gather(*tasks))


def generate_workload(batches: int, sessions: int, stagger: float,
                      stagger_mode: str, seed: int) -> list[dict]:
    """Pre-generate the full workload so every scheduler gets the same one."""
    rng = random.Random(seed)
    workload = []
    for batch_id in range(batches):
        prompts = [rng.choice(RESEARCH_PROMPTS) for _ in range(sessions)]
        delays = compute_stagger_delays(sessions, stagger, stagger_mode, rng)
        workload.append({"batch_id": batch_id, "prompts": prompts, "delays": delays})
    return workload


async def run_one_scheduler(scheduler_name: str, args, limiter: RateLimiter,
                            workload: list[dict]):
    """Run all batches for a single scheduler. Returns (results, cost)."""
    print(f"\n{'#'*70}")
    print(f" SCHEDULER: {scheduler_name.upper()}")
    print(f"{'#'*70}")

    # Fresh state
    limiter.reset()
    trace.reset()
    per_scheduler_limit = args.cost_limit / len(ALL_SCHEDULERS) if args.all_schedulers else args.cost_limit
    init_cost_tracker(per_scheduler_limit)

    scheduler = get_scheduler(scheduler_name, limiter)
    set_scheduler(scheduler)
    scheduler.start()

    all_results: list[SessionResult] = []
    cost_exceeded = False

    try:
        for batch in workload:
            if cost_exceeded:
                print(f"\n[COST LIMIT] Skipping batch {batch['batch_id']} and beyond.")
                break

            batch_results = await run_batch(
                batch_id=batch["batch_id"],
                prompts=batch["prompts"],
                stagger_delays=batch["delays"],
            )
            all_results.extend(batch_results)
            print_batch_summary(batch["batch_id"], batch_results)

            if any(r.error and "CostLimitExceeded" in r.error
                   for r in batch_results):
                cost_exceeded = True
    finally:
        await scheduler.stop()

    cost = ct.cost_tracker.total_cost if ct.cost_tracker else 0.0
    print_aggregate_summary(all_results, ct.cost_tracker)
    trace.print_summary()
    return all_results, cost


async def main():
    parser = argparse.ArgumentParser(description="Batch research agent runner")
    parser.add_argument("--batches", type=int, default=3,
                        help="Number of sequential batches (default: 3)")
    parser.add_argument("--sessions", type=int, default=5,
                        help="Sessions per batch (default: 5)")
    parser.add_argument("--stagger", type=float, default=4.0,
                        help="Mean seconds between session starts (default: 4.0)")
    parser.add_argument("--stagger-mode", type=str, default="fixed",
                        choices=["fixed", "poisson"],
                        help="Stagger mode: fixed or poisson (default: fixed)")
    parser.add_argument("--scheduler", type=str, default="fifo",
                        choices=ALL_SCHEDULERS)
    parser.add_argument("--all-schedulers", action="store_true",
                        help="Run all schedulers sequentially and compare")
    parser.add_argument("--rpm", type=int, default=30,
                        help="Rate limit: requests per minute (default: 30)")
    parser.add_argument("--tpm", type=int, default=200_000,
                        help="Rate limit: tokens per minute (default: 200000)")
    parser.add_argument("--cost-limit", type=float, default=20.0,
                        help="Hard cost cap in USD (default: 20.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for workload generation (default: 42)")
    args = parser.parse_args()

    schedulers = ALL_SCHEDULERS if args.all_schedulers else [args.scheduler]
    limiter = RateLimiter(rpm=args.rpm, tpm=args.tpm)

    # Pre-generate workload once — all schedulers get identical prompts & delays
    workload = generate_workload(args.batches, args.sessions, args.stagger,
                                 args.stagger_mode, args.seed)

    total_sessions = args.batches * args.sessions * len(schedulers)
    print(f"Batch runner: {len(schedulers)} scheduler(s) x {args.batches} batches "
          f"x {args.sessions} sessions = {total_sessions} total")
    print(f"Schedulers: {', '.join(schedulers)} | RPM: {args.rpm} | TPM: {args.tpm}")
    print(f"Stagger: {args.stagger}s ({args.stagger_mode}) | Cost limit: ${args.cost_limit:.2f}")
    print(f"Seed: {args.seed}")

    # Print workload for verification
    print(f"\nWorkload (same for all schedulers):")
    for batch in workload:
        prompts_short = [p[:50] + "..." if len(p) > 50 else p for p in batch["prompts"]]
        print(f"  Batch {batch['batch_id']}: {prompts_short}")

    results_by_scheduler: dict[str, list[SessionResult]] = {}
    cost_by_scheduler: dict[str, float] = {}

    for sched_name in schedulers:
        results, cost = await run_one_scheduler(sched_name, args, limiter, workload)
        results_by_scheduler[sched_name] = results
        cost_by_scheduler[sched_name] = cost

    if len(schedulers) > 1:
        print_comparison(results_by_scheduler, cost_by_scheduler)


if __name__ == "__main__":
    asyncio.run(main())
