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
from metrics import SessionResult, print_batch_summary, print_aggregate_summary


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


async def run_batch(batch_id: int, prompts: list[str],
                    stagger_secs: float, session_offset: int) -> list[SessionResult]:
    """Launch all sessions in a batch with staggered starts."""
    print(f"\n{'='*60}")
    print(f" BATCH {batch_id}: launching {len(prompts)} sessions "
          f"(stagger={stagger_secs}s)")
    print(f"{'='*60}")

    tasks = [
        asyncio.create_task(
            run_session(
                session_id=session_offset + i,
                batch_id=batch_id,
                prompt=p,
                stagger_delay=i * stagger_secs,
            )
        )
        for i, p in enumerate(prompts)
    ]
    return list(await asyncio.gather(*tasks))


async def main():
    parser = argparse.ArgumentParser(description="Batch research agent runner")
    parser.add_argument("--batches", type=int, default=5,
                        help="Number of sequential batches (default: 5)")
    parser.add_argument("--sessions", type=int, default=30,
                        help="Sessions per batch (default: 30)")
    parser.add_argument("--stagger", type=float, default=1.0,
                        help="Seconds between session starts within a batch (default: 1.0)")
    parser.add_argument("--scheduler", type=str, default="fifo",
                        choices=["backoff", "fifo", "sjf"])
    parser.add_argument("--rpm", type=int, default=60,
                        help="Rate limit: requests per minute (default: 60)")
    parser.add_argument("--tpm", type=int, default=200_000,
                        help="Rate limit: tokens per minute (default: 200000)")
    parser.add_argument("--cost-limit", type=float, default=15.0,
                        help="Hard cost cap in USD (default: 15.0)")
    args = parser.parse_args()

    # Shared rate limiter + scheduler + cost tracker
    limiter = RateLimiter(rpm=args.rpm, tpm=args.tpm)
    scheduler = get_scheduler(args.scheduler, limiter)
    set_scheduler(scheduler)
    scheduler.start()
    init_cost_tracker(args.cost_limit)

    total_sessions = args.batches * args.sessions
    print(f"Batch runner: {args.batches} batches x {args.sessions} sessions "
          f"= {total_sessions} total")
    print(f"Scheduler: {args.scheduler} | RPM: {args.rpm} | TPM: {args.tpm}")
    print(f"Stagger: {args.stagger}s | Cost limit: ${args.cost_limit:.2f}")

    all_results: list[SessionResult] = []
    cost_exceeded = False

    try:
        for batch_id in range(args.batches):
            if cost_exceeded:
                print(f"\n[COST LIMIT] Skipping batch {batch_id} and beyond.")
                break

            prompts = [random.choice(RESEARCH_PROMPTS)
                       for _ in range(args.sessions)]

            batch_results = await run_batch(
                batch_id=batch_id,
                prompts=prompts,
                stagger_secs=args.stagger,
                session_offset=len(all_results),
            )
            all_results.extend(batch_results)
            print_batch_summary(batch_id, batch_results)

            if any(r.error and "CostLimitExceeded" in r.error
                   for r in batch_results):
                cost_exceeded = True

    finally:
        await scheduler.stop()

    print_aggregate_summary(all_results, ct.cost_tracker)


if __name__ == "__main__":
    asyncio.run(main())
