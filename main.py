import argparse
import asyncio
import random

from dotenv import load_dotenv
load_dotenv()

from prompts import RESEARCH_PROMPTS, DEFAULT_PROMPT
from orchestrator import orchestrate
from trace import trace
from rate_limiter import RateLimiter
from schedulers import get_scheduler
from llm import set_scheduler
import cost_tracker as ct
from cost_tracker import init_cost_tracker


async def main():
    parser = argparse.ArgumentParser(description="Research agent loop")
    parser.add_argument("--random", action="store_true", help="Pick a random research prompt")
    parser.add_argument("--prompt", type=str, help="Custom research prompt")
    parser.add_argument("--scheduler", type=str, default="fifo",
                        choices=["backoff", "fifo", "sjf", "mapreduce"],
                        help="Scheduling strategy (default: fifo)")
    parser.add_argument("--rpm", type=int, default=20,
                        help="Rate limit: requests per minute (default: 20)")
    parser.add_argument("--tpm", type=int, default=100_000,
                        help="Rate limit: tokens per minute (default: 100000)")
    parser.add_argument("--cost-limit", type=float, default=15.0,
                        help="Hard cost cap in USD (default: 15.0)")
    args = parser.parse_args()

    if args.prompt:
        prompt = args.prompt
    elif args.random:
        prompt = random.choice(RESEARCH_PROMPTS)
    else:
        prompt = DEFAULT_PROMPT

    # Set up rate limiter + scheduler + cost tracker
    limiter = RateLimiter(rpm=args.rpm, tpm=args.tpm)
    scheduler = get_scheduler(args.scheduler, limiter)
    set_scheduler(scheduler)
    scheduler.start()
    init_cost_tracker(args.cost_limit)

    print(f"Research topic: {prompt}")
    print(f"Scheduler: {args.scheduler} | RPM: {args.rpm} | TPM: {args.tpm}\n")

    try:
        result = await orchestrate(prompt)

        print("=" * 60)
        print("FINAL SYNTHESIS")
        print("=" * 60)
        print(result)
    finally:
        await scheduler.stop()

    print(f"\n  Cost:")
    print(ct.cost_tracker.summary())
    trace.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
