"""Two-wave API benchmark — launch n1 sessions, wait t seconds, launch n2 more.

Sets scheduler (default: mapreduce_events), rpm, and tpm once at start.
Does NOT reset between wave 1 and wave 2.

Usage::

    # Terminal 1: start the proxy
    hypercorn api:app --host 0.0.0.0 --port 8000

    # Terminal 2: run two waves
    python -m sim.two_waves --n1 5 --s1 2 --t 10 --n2 5 --s2 2
    python -m sim.two_waves --n1 10 --s1 1 --t 30 --n2 10 --s2 3 --rpm 20 --tpm 200000
"""

import argparse
import asyncio
import random

import httpx
from dotenv import load_dotenv

load_dotenv()

from prompts import RESEARCH_PROMPTS
from sim.api_runner import (
    AUTH_HEADERS,
    print_summary,
    register_call_types,
    run_session,
)


def generate_two_wave_workload(n1: int, s1: float, n2: int, s2: float,
                               t: float, seed: int) -> dict:
    """Generate prompts and arrival times for two waves.

    Wave 1: n1 sessions at 0, s1, 2*s1, ...
    Wave 2: n2 sessions at t, t+s2, t+2*s2, ...
    """
    rng = random.Random(seed)
    total = n1 + n2
    copies = total // len(RESEARCH_PROMPTS)
    remainder = total % len(RESEARCH_PROMPTS)
    prompts = RESEARCH_PROMPTS * copies + RESEARCH_PROMPTS[:remainder]
    rng.shuffle(prompts)

    arrivals = (
        [i * s1 for i in range(n1)] +
        [t + i * s2 for i in range(n2)]
    )
    return {"prompts": prompts, "arrivals": arrivals}


async def main():
    parser = argparse.ArgumentParser(
        description="Two-wave API benchmark — n1 sessions, wait t, n2 sessions")
    parser.add_argument("--n1", type=int, default=5,
                        help="Sessions in wave 1 (default: 5)")
    parser.add_argument("--s1", type=float, default=2.0,
                        help="Stagger (seconds) between wave 1 arrivals (default: 2)")
    parser.add_argument("--t", type=float, default=10.0,
                        help="Seconds to wait between wave 1 launch and wave 2 launch (default: 10)")
    parser.add_argument("--n2", type=int, default=5,
                        help="Sessions in wave 2 (default: 5)")
    parser.add_argument("--s2", type=float, default=2.0,
                        help="Stagger (seconds) between wave 2 arrivals (default: 2)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-url", type=str, default="http://localhost:8000")
    parser.add_argument("--timeout", type=float, default=1200.0,
                        help="HTTP request timeout in seconds (default: 1200)")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Max tokens per completion (default: 2048)")
    parser.add_argument("--rpm", type=int, default=None,
                        help="Requests per minute (sets API server rate limiter)")
    parser.add_argument("--tpm", type=int, default=None,
                        help="Tokens per minute (sets API server rate limiter)")
    parser.add_argument("--scheduler", type=str, default="mapreduce_events",
                        help="Scheduler name (default: mapreduce_events)")
    args = parser.parse_args()

    workload = generate_two_wave_workload(
        args.n1, args.s1, args.n2, args.s2, args.t, args.seed
    )
    prompts = workload["prompts"]
    arrivals = workload["arrivals"]

    # Set max_tokens / rpm / tpm and scheduler once at beginning
    config_update: dict = {"max_tokens": args.max_tokens}
    if args.rpm is not None:
        config_update["rpm"] = args.rpm
    if args.tpm is not None:
        config_update["tpm"] = args.tpm
    async with httpx.AsyncClient(
        base_url=args.base_url, timeout=httpx.Timeout(args.timeout), http2=True,
        headers=AUTH_HEADERS,
    ) as client:
        resp = await client.patch("/sim/config", json=config_update)
        resp.raise_for_status()
        resp = await client.post("/sim/reset", json={"scheduler": args.scheduler})
        resp.raise_for_status()
        resp = await client.get("/sim/config")
        resp.raise_for_status()
        config = resp.json()

    print(f"Server config: scheduler={config['scheduler']}, "
          f"RPM={config['rpm']}, TPM={config['tpm']}, "
          f"max_tokens={config['max_tokens']}")
    print(f"Wave 1: {args.n1} sessions @ stagger {args.s1}s")
    print(f"Wait: {args.t}s")
    print(f"Wave 2: {args.n2} sessions @ stagger {args.s2}s")
    print(f"Seed: {args.seed}")

    await register_call_types(args.base_url, args.timeout)

    # Wave 1: launch n1 sessions
    wave1_tasks = [
        asyncio.create_task(
            run_session(
                session_id=i, prompt=prompts[i], stagger_delay=arrivals[i],
                base_url=args.base_url, timeout=args.timeout,
            )
        )
        for i in range(args.n1)
    ]
    print(f"\nLaunched wave 1 ({args.n1} sessions)")

    # Wait t seconds
    await asyncio.sleep(args.t)
    print(f"Waited {args.t}s — launching wave 2 ({args.n2} sessions)")

    # Wave 2: launch n2 sessions
    wave2_tasks = [
        asyncio.create_task(
            run_session(
                session_id=args.n1 + i, prompt=prompts[args.n1 + i],
                stagger_delay=arrivals[args.n1 + i],
                base_url=args.base_url, timeout=args.timeout,
            )
        )
        for i in range(args.n2)
    ]

    # Wait for all sessions to complete
    all_results = list(await asyncio.gather(*wave1_tasks, *wave2_tasks))

    # Fetch server-side stats
    async with httpx.AsyncClient(
        base_url=args.base_url, timeout=httpx.Timeout(args.timeout), http2=True,
        headers=AUTH_HEADERS,
    ) as client:
        resp = await client.get("/sim/stats")
        resp.raise_for_status()
        server_stats = resp.json()

    print_summary(all_results, server_stats)


if __name__ == "__main__":
    asyncio.run(main())