"""RPM experiment: test reliability and predictability of OpenAI's RPM rate limiting.

Protocol
--------
Phase 1 — BURST (t=0 to ~10s):
    Fire N concurrent requests with max_tokens=1 and tiny prompts.
    Token cost per request is ~12 tokens, so 3000 requests ≈ 36K tokens
    (well under 2M TPM). This isolates the RPM limit.

Phase 2 — SUSTAINED (t≈10s to t≈70s):
    Send requests at a controlled rate (e.g. 80/s ≈ 4800 RPM) to maintain
    pressure near the 5000 RPM limit. Record every remaining-requests value.

Phase 3 — PROBE (t≈70s to t≈150s):
    Stop sending and probe every 1s to observe remaining-requests recovery.

What we learn
-------------
- Does remaining-requests decrement by exactly 1 per request?
- Under concurrent load, do simultaneous responses show inconsistent values
  (like remaining-tokens did for TPM)?
- Is recovery linear (token bucket) or step (sliding window)?
- Can a local RPM counter reliably prevent 429s?

Usage
-----
    python experiments/rpm_test.py [--burst-size 2000] [--sustained-rate 80]
"""

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError

load_dotenv()

MODEL = "gpt-4.1-nano"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
OUTPUT_FILE = RESULTS_DIR / "rpm_experiment.jsonl"

TINY_PROMPT = "Say ok."

RL_HEADER_KEYS = (
    "x-ratelimit-limit-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-requests",
    "x-ratelimit-reset-tokens",
)


def parse_reset_time(val: str | None) -> float | None:
    if not val:
        return None
    total = 0.0
    for amount, unit in re.findall(r"([0-9.]+)(ms|m|s)", val):
        amount = float(amount)
        if unit == "ms":
            total += amount / 1000
        elif unit == "s":
            total += amount
        elif unit == "m":
            total += amount * 60
    return total if total > 0 else None


def extract_headers(raw_headers, t0: float) -> dict:
    row = {"t": round(time.time() - t0, 3)}
    for key in RL_HEADER_KEYS:
        val = raw_headers.get(key)
        if val is not None:
            short = key.replace("x-ratelimit-", "")
            if "remaining" in short or "limit" in short:
                try:
                    val = int(val)
                except (ValueError, TypeError):
                    pass
            elif "reset" in short:
                val = parse_reset_time(val)
            row[short] = val
    retry = raw_headers.get("retry-after")
    if retry is not None:
        try:
            row["retry_after"] = float(retry)
        except (ValueError, TypeError):
            row["retry_after"] = retry
    return row


async def tiny_request(client: AsyncOpenAI, t0: float, idx: int,
                       phase: str) -> dict:
    """Send a minimal request (max_tokens=1) and return headers."""
    try:
        raw = await client.chat.completions.with_raw_response.create(
            model=MODEL,
            messages=[{"role": "user", "content": TINY_PROMPT}],
            max_tokens=1,
        )
        headers = extract_headers(raw.headers, t0)
        parsed = raw.parse()
        usage = {
            "prompt_tokens": parsed.usage.prompt_tokens,
            "completion_tokens": parsed.usage.completion_tokens,
            "total_tokens": parsed.usage.total_tokens,
        } if parsed.usage else {}
        return {"phase": phase, "idx": idx, "status": "ok",
                **headers, **usage}
    except RateLimitError as exc:
        headers = {}
        if exc.response is not None:
            headers = extract_headers(exc.response.headers, t0)
        return {"phase": phase, "idx": idx, "status": "429", **headers}
    except Exception as exc:
        return {"phase": phase, "idx": idx, "status": "error",
                "t": round(time.time() - t0, 3), "error": str(exc)[:200]}


async def run_experiment(burst_size: int = 2000,
                         sustained_rate: float = 80,
                         sustained_duration: float = 60,
                         probe_duration: float = 90,
                         probe_interval: float = 1.0):
    client = AsyncOpenAI()
    results = []
    t0 = time.time()
    idx_counter = 0

    def log(row):
        results.append(row)
        with open(OUTPUT_FILE, "a") as f:
            f.write(json.dumps(row) + "\n")

    # --- Phase 0: Initial probe ---
    print(f"[t=0.0] Phase 0: initial probe...")
    r = await tiny_request(client, t0, -1, "init")
    log(r)
    print(f"  remaining-requests = {r.get('remaining-requests', '?')}")
    print(f"  remaining-tokens   = {r.get('remaining-tokens', '?')}")

    # --- Phase 1: BURST ---
    print(f"\n[t={time.time()-t0:.1f}] Phase 1: firing {burst_size} concurrent "
          f"requests (max_tokens=1)...")

    tasks = [tiny_request(client, t0, i, "burst") for i in range(burst_size)]
    burst_results = await asyncio.gather(*tasks)

    ok_burst = [r for r in burst_results if r["status"] == "ok"]
    err_burst = [r for r in burst_results if r["status"] == "429"]
    total_tokens = sum(r.get("total_tokens", 0) for r in ok_burst)

    for r in burst_results:
        log(r)

    burst_end_t = time.time() - t0
    print(f"  Burst complete at t={burst_end_t:.1f}s")
    print(f"  OK: {len(ok_burst)}, 429: {len(err_burst)}")
    print(f"  Total tokens consumed: {total_tokens:,}")

    if ok_burst:
        rr = [r.get("remaining-requests", 5000) for r in ok_burst]
        print(f"  remaining-requests: min={min(rr)} max={max(rr)}")

    idx_counter = burst_size

    # --- Phase 2: SUSTAINED ---
    interval = 1.0 / sustained_rate
    print(f"\n[t={time.time()-t0:.1f}] Phase 2: sustained at {sustained_rate} req/s "
          f"for {sustained_duration}s...")

    phase2_start = time.time()
    sustained_ok = 0
    sustained_429 = 0
    while time.time() - phase2_start < sustained_duration:
        r = await tiny_request(client, t0, idx_counter, "sustained")
        log(r)
        if r["status"] == "ok":
            sustained_ok += 1
        elif r["status"] == "429":
            sustained_429 += 1
        idx_counter += 1

        elapsed_in_phase = time.time() - phase2_start
        expected = elapsed_in_phase * sustained_rate
        actual_sent = sustained_ok + sustained_429
        if actual_sent > expected:
            await asyncio.sleep(interval * 0.8)

        if idx_counter % 100 == 0:
            rr = r.get("remaining-requests", "?")
            print(f"  [t={time.time()-t0:.1f}s] sent={actual_sent} "
                  f"ok={sustained_ok} 429={sustained_429} "
                  f"remaining-requests={rr}")

    print(f"  Sustained phase complete: ok={sustained_ok} 429={sustained_429}")

    # --- Phase 3: PROBE ---
    print(f"\n[t={time.time()-t0:.1f}] Phase 3: probing every {probe_interval}s "
          f"for {probe_duration}s...")

    probe_start = time.time()
    probe_idx = 0
    while time.time() - probe_start < probe_duration:
        await asyncio.sleep(probe_interval)
        r = await tiny_request(client, t0, idx_counter, "probe")
        log(r)
        idx_counter += 1

        rr = r.get("remaining-requests", "?")
        rt = r.get("remaining-tokens", "?")
        reset_r = r.get("reset-requests", "?")
        status = r.get("status", "?")
        sym = "+" if status == "ok" else "X"
        print(f"  [{sym}] t={time.time()-t0:>6.1f}s  "
              f"remaining-requests={str(rr):>6}  "
              f"remaining-tokens={str(rt):>10}  "
              f"reset-requests={reset_r}")
        probe_idx += 1

    total_points = len(results)
    print(f"\n[t={time.time()-t0:.1f}] Experiment complete. {total_points} data points.")
    print(f"Results written to {OUTPUT_FILE}")
    return results


def analyze():
    """Analyze existing results."""
    with open(OUTPUT_FILE) as f:
        results = [json.loads(line) for line in f if line.strip()]

    burst = [r for r in results if r["phase"] == "burst"]
    sustained = [r for r in results if r["phase"] == "sustained"]
    probes = [r for r in results if r["phase"] == "probe"]

    print("=" * 70)
    print("RPM EXPERIMENT ANALYSIS")
    print("=" * 70)

    # --- Burst analysis ---
    burst_ok = [r for r in burst if r["status"] == "ok"]
    burst_429 = [r for r in burst if r["status"] == "429"]
    print(f"\nBurst: {len(burst_ok)} OK, {len(burst_429)} 429")

    if burst_ok:
        rr = [r["remaining-requests"] for r in burst_ok
              if "remaining-requests" in r]
        if rr:
            print(f"  remaining-requests: min={min(rr)} max={max(rr)}")

    # --- Consistency check: do simultaneous responses agree? ---
    print("\n--- Consistency of remaining-requests under concurrency ---")
    from collections import defaultdict
    by_100ms = defaultdict(list)
    all_with_rr = [r for r in results if "remaining-requests" in r and r["status"] == "ok"]
    for r in all_with_rr:
        window = round(r["t"], 1)
        by_100ms[window].append(r["remaining-requests"])

    inconsistent = 0
    total_windows = 0
    max_spread = 0
    for w in sorted(by_100ms):
        vals = by_100ms[w]
        if len(vals) >= 2:
            total_windows += 1
            spread = max(vals) - min(vals)
            if spread > 1:
                inconsistent += 1
                max_spread = max(max_spread, spread)

    print(f"  100ms windows with 2+ readings: {total_windows}")
    print(f"  Windows with spread > 1: {inconsistent} "
          f"({100*inconsistent/total_windows:.1f}% if {total_windows} > 0)")
    print(f"  Max spread in single window: {max_spread}")

    # --- Sustained phase ---
    if sustained:
        sus_ok = [r for r in sustained if r["status"] == "ok"]
        sus_429 = [r for r in sustained if r["status"] == "429"]
        print(f"\nSustained: {len(sus_ok)} OK, {len(sus_429)} 429")

    # --- Probe recovery ---
    if probes:
        probe_ok = [r for r in probes if r["status"] == "ok"
                    and "remaining-requests" in r]
        if probe_ok:
            print(f"\nProbe recovery ({len(probe_ok)} readings):")
            for p in probe_ok[:20]:
                print(f"  t={p['t']:>7.1f}s  remaining-requests={p['remaining-requests']}")

    # --- Compare with TPM unpredictability ---
    print("\n--- Key question: is RPM more predictable than TPM? ---")
    if inconsistent == 0 and total_windows > 10:
        print("  YES: remaining-requests is consistent across concurrent responses")
    elif total_windows > 0:
        pct = 100 * inconsistent / total_windows
        if pct < 5:
            print(f"  MOSTLY: only {pct:.1f}% of windows showed inconsistency")
        else:
            print(f"  NO: {pct:.1f}% of windows showed inconsistent remaining-requests")
    else:
        print("  INSUFFICIENT DATA")


def main():
    parser = argparse.ArgumentParser(
        description="RPM experiment: test OpenAI RPM rate limit reliability")
    parser.add_argument("--burst-size", type=int, default=2000,
                        help="Concurrent requests in burst phase (default: 2000)")
    parser.add_argument("--sustained-rate", type=float, default=80,
                        help="Requests/second in sustained phase (default: 80)")
    parser.add_argument("--sustained-duration", type=float, default=60,
                        help="Duration of sustained phase in seconds (default: 60)")
    parser.add_argument("--probe-duration", type=float, default=90,
                        help="Duration of probe phase in seconds (default: 90)")
    parser.add_argument("--probe-interval", type=float, default=1.0,
                        help="Seconds between probes (default: 1)")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Skip experiment, just analyze existing results")
    args = parser.parse_args()

    if args.analyze_only:
        analyze()
        return

    OUTPUT_FILE.write_text("")

    asyncio.run(run_experiment(
        burst_size=args.burst_size,
        sustained_rate=args.sustained_rate,
        sustained_duration=args.sustained_duration,
        probe_duration=args.probe_duration,
        probe_interval=args.probe_interval,
    ))
    analyze()


if __name__ == "__main__":
    main()
