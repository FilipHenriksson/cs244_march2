"""Single-session entry point — run one research agent session.

Usage::

    python main.py
    python main.py --random --scheduler sjf
    python main.py --prompt "What is dark matter?"
"""

import argparse
import asyncio
import logging
import random

from dotenv import load_dotenv
load_dotenv()

from prompts import RESEARCH_PROMPTS, DEFAULT_PROMPT
from agent import run_agent
from sim import RateLimiter, trace, init_cost_tracker
import sim.cost_tracker as ct
from schedulers import get_scheduler
from llm import set_scheduler

ALL_SCHEDULERS = [
    "backoff", "fifo", "sjf", "mapreduce", "mapreduce_improved",
    "adaptive_sjf", "token_sjf", "token_sjf_skip",
    "combined_mapreduce_asjf", "combined_mapreduce_tsjf",
]


async def main():
    parser = argparse.ArgumentParser(description="Run one research agent session")
    parser.add_argument("--random", action="store_true",
                        help="Pick a random research prompt")
    parser.add_argument("--prompt", type=str, help="Custom research prompt")
    parser.add_argument("--scheduler", type=str, default="fifo",
                        choices=ALL_SCHEDULERS,
                        help="Scheduling strategy (default: fifo)")
    parser.add_argument("--rpm", type=int, default=20,
                        help="Rate limit: requests per minute (default: 20)")
    parser.add_argument("--tpm", type=int, default=100_000,
                        help="Rate limit: tokens per minute (default: 100000)")
    parser.add_argument("--cost-limit", type=float, default=15.0,
                        help="Hard cost cap in USD (default: 15.0)")
    parser.add_argument("--prompt-mode", type=str, default="default",
                        choices=["default", "strict"],
                        help="Prompt mode: default (flexible) or strict (fixed tool counts)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for _name in ("openai", "httpx", "httpcore"):
        logging.getLogger(_name).setLevel(logging.WARNING)

    if args.prompt:
        prompt = args.prompt
    elif args.random:
        prompt = random.choice(RESEARCH_PROMPTS)
    else:
        prompt = DEFAULT_PROMPT

    limiter = RateLimiter(rpm=args.rpm, tpm=args.tpm)
    scheduler = get_scheduler(args.scheduler, limiter)
    set_scheduler(scheduler)
    scheduler.start()
    init_cost_tracker(args.cost_limit)

    print(f"Research topic: {prompt}")
    print(f"Scheduler: {args.scheduler} | RPM: {args.rpm} | TPM: {args.tpm} | "
          f"Mode: {args.prompt_mode}\n")

    try:
        result = await run_agent(prompt, prompt_mode=args.prompt_mode)

        print("=" * 60)
        print("FINAL SYNTHESIS")
        print("=" * 60)
        print(result.text)
        print(f"\nLLM calls: {result.llm_calls} (tool calls: {result.tool_calls})")
    finally:
        await scheduler.stop()

    print(f"\n  Cost:")
    print(ct.cost_tracker.summary())
    trace.print_summary(rl_stats=limiter.stats)


if __name__ == "__main__":
    asyncio.run(main())
