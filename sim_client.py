"""API-based simulation runner — reproduce sim/runner.py results over HTTP.

Drives the same deterministic workloads through the scheduling proxy (api.py)
using concurrent HTTP clients, each modeled as a separate machine (own TCP
connection).  Reports client-recorded latency and fetches server-side metrics.

Usage::

    # Terminal 1: start the proxy
    SCHEDULER=fifo uvicorn api:app --host 0.0.0.0 --port 8000

    # Terminal 2: run the benchmark
    python sim_client.py --sessions 15 --scheduler fifo
    python sim_client.py --sessions 10 --all-schedulers --stagger-mode bursty
    python sim_client.py --sessions 15 --schedulers fifo mapreduce_skip_adaptive
"""

import argparse
import asyncio
import statistics
import time

import httpx

from prompts.system import ORCHESTRATOR_SYSTEM_PROMPT
from tools.analysts import _ANALYSTS
from tools.reviewers import _REVIEWERS
from sim.workload import generate_workload

ALL_SCHEDULERS = [
    "backoff", "fifo", "mapreduce", "mapreduce_skip", "mapreduce_skip_adaptive",
]

STRICT_ANALYSTS = [
    "analyst_web_research", "analyst_summarizer", "analyst_deep_analysis",
    "analyst_historical_context", "analyst_statistical",
]
STRICT_REVIEWERS = ["review_citations", "review_style", "review_facts"]

ALL_CALL_TYPES: dict[str, str] = {
    "orchestrator": ORCHESTRATOR_SYSTEM_PROMPT,
}
for _name, _desc, _sp in _ANALYSTS:
    ALL_CALL_TYPES[_name] = _sp
for _name, _desc, _sp in _REVIEWERS:
    ALL_CALL_TYPES[_name] = _sp

N_ANALYSTS = len(STRICT_ANALYSTS)
N_REVIEWERS = len(STRICT_REVIEWERS)
LLM_CALLS_PER_SESSION = 3 + N_ANALYSTS + N_REVIEWERS


# ---------------------------------------------------------------------------
# Session result
# ---------------------------------------------------------------------------

class SessionResult:
    __slots__ = ("session_id", "prompt", "elapsed", "success", "error")

    def __init__(self, session_id, prompt, elapsed, success=True, error=None):
        self.session_id = session_id
        self.prompt = prompt
        self.elapsed = elapsed
        self.success = success
        self.error = error


# ---------------------------------------------------------------------------
# Call-type registration (once per server lifetime)
# ---------------------------------------------------------------------------

async def register_call_types(base_url: str, timeout: float):
    async with httpx.AsyncClient(
        base_url=base_url, timeout=httpx.Timeout(timeout),
    ) as client:
        for name, system_prompt in ALL_CALL_TYPES.items():
            resp = await client.post("/call_types", json={
                "name": name,
                "system_prompt": system_prompt,
            })
            resp.raise_for_status()
    print(f"Registered {len(ALL_CALL_TYPES)} call types")


# ---------------------------------------------------------------------------
# Single-session pipeline (strict mode over HTTP)
# ---------------------------------------------------------------------------

async def run_session(session_id: int, prompt: str, stagger_delay: float,
                      base_url: str, timeout: float) -> SessionResult:
    """Run one strict-mode research session as an independent HTTP client."""
    await asyncio.sleep(stagger_delay)
    start = time.time()

    async with httpx.AsyncClient(
        base_url=base_url, timeout=httpx.Timeout(timeout),
    ) as client:
        try:
            resp = await client.post("/sessions")
            resp.raise_for_status()
            sid = resp.json()["session_id"]
            base = f"/sessions/{sid}/completions"
            messages: list[dict] = []

            # Step 1: Orchestrator plans
            messages.append({"role": "user", "content": prompt})
            resp = await client.post(base, json={
                "call_type": "orchestrator",
                "call_detail": "plan",
                "messages": list(messages),
            })
            resp.raise_for_status()
            plan = resp.json()["content"]
            messages.append({"role": "assistant", "content": plan})

            # Step 2: Fan-out analysts
            analyst_calls = [
                {"call_type": name, "messages": [{"role": "user", "content": prompt}]}
                for name in STRICT_ANALYSTS
            ]
            resp = await client.post(f"{base}/batch", json={"calls": analyst_calls})
            resp.raise_for_status()
            analyst_results = resp.json()["completions"]
            analyst_text = "\n\n".join(
                f"--- {name} ---\n{r['content']}"
                for name, r in zip(STRICT_ANALYSTS, analyst_results)
            )
            messages.append({"role": "user", "content":
                f"Here are findings from 5 specialist analysts:\n\n{analyst_text}\n\n"
                f"Write a comprehensive draft synthesis of these findings."
            })

            # Step 3: Orchestrator synthesizes
            resp = await client.post(base, json={
                "call_type": "orchestrator",
                "call_detail": "synthesize",
                "messages": list(messages),
            })
            resp.raise_for_status()
            draft = resp.json()["content"]
            messages.append({"role": "assistant", "content": draft})

            # Step 4: Fan-out reviewers
            reviewer_calls = [
                {"call_type": name, "messages": [{"role": "user", "content": draft}]}
                for name in STRICT_REVIEWERS
            ]
            resp = await client.post(f"{base}/batch", json={"calls": reviewer_calls})
            resp.raise_for_status()
            reviewer_results = resp.json()["completions"]
            reviewer_text = "\n\n".join(
                f"--- {name} ---\n{r['content']}"
                for name, r in zip(STRICT_REVIEWERS, reviewer_results)
            )
            messages.append({"role": "user", "content":
                f"Here is feedback from 3 reviewers:\n\n{reviewer_text}\n\n"
                f"Incorporate the feedback and produce your final polished synthesis."
            })

            # Step 5: Final synthesis
            resp = await client.post(base, json={
                "call_type": "orchestrator",
                "call_detail": "final",
                "messages": list(messages),
            })
            resp.raise_for_status()

            elapsed = time.time() - start
            await client.delete(f"/sessions/{sid}")

            print(f"  [session={session_id}] done in {elapsed:.1f}s "
                  f"(arrived at {stagger_delay:.1f}s)")
            return SessionResult(session_id, prompt, elapsed)

        except Exception as e:
            elapsed = time.time() - start
            print(f"  [session={session_id}] ERROR after {elapsed:.1f}s: {e}")
            return SessionResult(session_id, prompt, elapsed,
                                 success=False, error=str(e))


# ---------------------------------------------------------------------------
# Latency stats
# ---------------------------------------------------------------------------

def compute_latency_stats(results: list[SessionResult]) -> dict:
    ok = [r for r in results if r.success]
    if not ok:
        return {}
    durations = sorted(r.elapsed for r in ok)
    return {
        "count": len(durations),
        "mean": statistics.mean(durations),
        "median": statistics.median(durations),
        "p95": durations[int(len(durations) * 0.95)],
        "p99": durations[int(len(durations) * 0.99)] if len(durations) > 1 else durations[-1],
        "min": min(durations),
        "max": max(durations),
        "stdev": statistics.stdev(durations) if len(durations) > 1 else 0.0,
    }


def print_summary(results: list[SessionResult], server_stats: dict):
    ok = [r for r in results if r.success]
    err = [r for r in results if not r.success]
    stats = compute_latency_stats(results)

    print(f"\n{'='*70}")
    print(f" SUMMARY  ({len(ok)} ok / {len(err)} failed)")
    print(f"{'='*70}")

    if stats:
        print(f"  Client-recorded latency:")
        print(f"    count  = {stats['count']}")
        print(f"    mean   = {stats['mean']:.2f}s")
        print(f"    median = {stats['median']:.2f}s")
        print(f"    p95    = {stats['p95']:.2f}s")
        print(f"    p99    = {stats['p99']:.2f}s")
        print(f"    min    = {stats['min']:.2f}s")
        print(f"    max    = {stats['max']:.2f}s")
        print(f"    stdev  = {stats['stdev']:.2f}s")

    if server_stats:
        rl = server_stats.get("rate_limiter", {})
        cost = server_stats.get("cost", {})
        if cost:
            print(f"\n  Server cost:")
            print(f"    LLM calls:       {cost.get('call_count', 0)}")
            print(f"    Input tokens:    {cost.get('total_input_tokens', 0):,}")
            print(f"    Output tokens:   {cost.get('total_output_tokens', 0):,}")
            print(f"    Total cost:      ${cost.get('total_cost', 0):.4f}"
                  f" / ${cost.get('limit', 0):.2f}")
        if rl:
            print(f"\n  Rate limiter:")
            print(f"    Acquires:        {rl.get('total_acquires', 0)}")
            print(f"    RPM throttles:   {rl.get('rpm_throttles', 0)}")
            print(f"    TPM throttles:   {rl.get('tpm_throttles', 0)}")
            est = rl.get("total_estimated", 0)
            act = rl.get("total_actual", 1) or 1
            print(f"    Token overest.:  {est / act:.1f}x "
                  f"({est:,} est / {act:,} actual)")

    for r in err:
        print(f"  FAILED session {r.session_id}: {r.error}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_comparison(results_by: dict[str, list[SessionResult]],
                     server_stats_by: dict[str, dict]):
    schedulers = list(results_by.keys())
    col_w = max(16, *(len(s) + 2 for s in schedulers))

    latency_by = {s: compute_latency_stats(results_by[s]) for s in schedulers}

    print(f"\n{'='*80}")
    print(f" SCHEDULER COMPARISON (client-recorded latency)")
    print(f"{'='*80}")

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

    _row("Sessions OK", [
        f"{latency_by[s].get('count', 0)}/{len(results_by[s])}"
        for s in schedulers
    ])
    _row("Total LLM calls", [
        f"{latency_by[s].get('count', 0) * LLM_CALLS_PER_SESSION}"
        for s in schedulers
    ])
    _row("Mean latency", [
        f"{latency_by[s].get('mean', 0):.1f}s" for s in schedulers
    ])
    _row("Median", [
        f"{latency_by[s].get('median', 0):.1f}s" for s in schedulers
    ])
    _row("p95", [
        f"{latency_by[s].get('p95', 0):.1f}s" for s in schedulers
    ])
    _row("Min", [
        f"{latency_by[s].get('min', 0):.1f}s" for s in schedulers
    ])
    _row("Max", [
        f"{latency_by[s].get('max', 0):.1f}s" for s in schedulers
    ])
    _row("Stdev", [
        f"{latency_by[s].get('stdev', 0):.1f}s" for s in schedulers
    ])

    # Server-side stats
    _row("Cost", [
        f"${server_stats_by.get(s, {}).get('cost', {}).get('total_cost', 0):.4f}"
        for s in schedulers
    ])
    _row("RPM throttles", [
        f"{server_stats_by.get(s, {}).get('rate_limiter', {}).get('rpm_throttles', 0)}"
        for s in schedulers
    ])
    _row("TPM throttles", [
        f"{server_stats_by.get(s, {}).get('rate_limiter', {}).get('tpm_throttles', 0)}"
        for s in schedulers
    ])

    def _bottleneck(s):
        rl = server_stats_by.get(s, {}).get("rate_limiter", {})
        rpm = rl.get("rpm_throttles", 0)
        tpm = rl.get("tpm_throttles", 0)
        if rpm == 0 and tpm == 0:
            return "neither"
        return f"RPM ({rpm}v{tpm})" if rpm >= tpm else f"TPM ({tpm}v{rpm})"

    _row("Bottleneck", [_bottleneck(s) for s in schedulers])

    def _overest(s):
        rl = server_stats_by.get(s, {}).get("rate_limiter", {})
        est = rl.get("total_estimated", 0)
        act = max(1, rl.get("total_actual", 1))
        return f"{est / act:.1f}x"

    _row("Token overest.", [_overest(s) for s in schedulers])

    print(f"{'='*80}\n")


# ---------------------------------------------------------------------------
# Run one scheduler
# ---------------------------------------------------------------------------

async def run_one_scheduler(scheduler_name: str, workload: dict,
                            base_url: str, timeout: float,
                            sched_index: int, num_schedulers: int):
    """Run the full workload for one scheduler, return results + server stats."""
    print(f"\n{'#'*70}")
    print(f" SCHEDULER: {scheduler_name.upper()}  [{sched_index + 1}/{num_schedulers}]")
    print(f"{'#'*70}")

    prompts = workload["prompts"]
    arrivals = workload["arrivals"]

    print(f" Launching {len(prompts)} sessions with staggered arrivals")
    print(f" Arrival times: {', '.join(f'{t:.1f}s' for t in arrivals[:20])}"
          f"{'...' if len(arrivals) > 20 else ''}")

    tasks = [
        asyncio.create_task(
            run_session(
                session_id=i, prompt=p, stagger_delay=arrivals[i],
                base_url=base_url, timeout=timeout,
            )
        )
        for i, p in enumerate(prompts)
    ]
    results = list(await asyncio.gather(*tasks))

    # Fetch server-side stats
    async with httpx.AsyncClient(
        base_url=base_url, timeout=httpx.Timeout(timeout),
    ) as client:
        resp = await client.get("/sim/stats")
        resp.raise_for_status()
        server_stats = resp.json()

    print_summary(results, server_stats)
    return results, server_stats


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(
        description="API-based simulation runner — benchmark the scheduling proxy")
    parser.add_argument("--sessions", type=int, default=15,
                        help="Total number of sessions (default: 15)")
    parser.add_argument("--stagger", type=float, default=4.0,
                        help="Mean seconds between session arrivals (default: 4.0)")
    parser.add_argument("--stagger-mode", type=str, default="constant",
                        choices=["constant", "bursty"])
    parser.add_argument("--scheduler", type=str, default="fifo",
                        choices=ALL_SCHEDULERS)
    parser.add_argument("--schedulers", type=str, nargs="+",
                        choices=ALL_SCHEDULERS,
                        help="Run specific schedulers sequentially and compare")
    parser.add_argument("--all-schedulers", action="store_true",
                        help="Run all schedulers sequentially and compare")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-url", type=str, default="http://localhost:8000")
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="HTTP request timeout in seconds (default: 600)")
    parser.add_argument("--cooldown", type=float, default=5.0,
                        help="Seconds to sleep between scheduler runs (default: 5)")
    parser.add_argument("--max-tokens", type=int, default=1024,
                        help="Max tokens per completion; sets API server value once (default: 1024)")
    return parser.parse_args()


async def main():
    args = _parse_args()

    if args.all_schedulers:
        schedulers = list(ALL_SCHEDULERS)
    elif args.schedulers:
        schedulers = list(args.schedulers)
    else:
        schedulers = [args.scheduler]

    workload = generate_workload(args.sessions, args.stagger,
                                 args.stagger_mode, args.seed)

    # Set max_tokens and fetch server config
    async with httpx.AsyncClient(
        base_url=args.base_url, timeout=httpx.Timeout(args.timeout),
    ) as client:
        resp = await client.patch("/sim/config", json={"max_tokens": args.max_tokens})
        resp.raise_for_status()
        resp = await client.get("/sim/config")
        resp.raise_for_status()
        config = resp.json()

    print(f"Server config: scheduler={config['scheduler']}, "
          f"RPM={config['rpm']}, TPM={config['tpm']}, "
          f"max_tokens={config['max_tokens']}, model={config['model']}")
    print(f"Runner: {len(schedulers)} scheduler(s) x {args.sessions} sessions")
    print(f"Schedulers: {', '.join(schedulers)}")
    print(f"Arrivals: every {args.stagger}s ({args.stagger_mode}) | "
          f"Seed: {args.seed}")

    print(f"\nWorkload:")
    for i, (p, t) in enumerate(zip(workload["prompts"], workload["arrivals"])):
        print(f"  Session {i}: arrives at {t:.1f}s — {p[:60]}...")

    # Register call types once
    await register_call_types(args.base_url, args.timeout)

    results_by: dict[str, list[SessionResult]] = {}
    server_stats_by: dict[str, dict] = {}

    run_start = time.time()
    for i, sched_name in enumerate(schedulers):
        if i > 0:
            print(f"\n--- Cooldown: sleeping {args.cooldown}s before next scheduler ---\n")
            await asyncio.sleep(args.cooldown)

        # Reset server to this scheduler before each run
        async with httpx.AsyncClient(
            base_url=args.base_url, timeout=httpx.Timeout(args.timeout),
        ) as client:
            resp = await client.post("/sim/reset", json={"scheduler": sched_name})
            resp.raise_for_status()
            print(f"Server reset -> scheduler={resp.json()['scheduler']}")

        results, server_stats = await run_one_scheduler(
            sched_name, workload, args.base_url, args.timeout,
            sched_index=i, num_schedulers=len(schedulers),
        )
        results_by[sched_name] = results
        server_stats_by[sched_name] = server_stats

    total_elapsed = time.time() - run_start
    print(f"\n{'='*70}")
    print(f" ALL SCHEDULERS COMPLETE — total elapsed: {total_elapsed:.0f}s "
          f"({total_elapsed/60:.1f}min)")
    print(f"{'='*70}")

    if len(schedulers) > 1:
        print_comparison(results_by, server_stats_by)


if __name__ == "__main__":
    asyncio.run(main())
