"""Burst experiment: determine if OpenAI TPM uses a sliding window or token bucket.

Protocol
--------
Phase 1 — BURST (t=0 to ~15s):
    Fire N concurrent requests (max_tokens=2048) to consume ~1-1.5M tokens.
    Record total actual tokens consumed and completion timestamps.

Phase 2 — PROBE (t≈15s to t≈90s):
    Every PROBE_INTERVAL seconds, send a single tiny request (max_tokens=1)
    and record the remaining-tokens header. This traces the recovery curve.

Predictions
-----------
Token bucket (capacity=2M, refill=33,333/s):
    remaining-tokens recovers LINEARLY at 33,333 tokens/s after burst ends.
    30s after consuming 1.5M: remaining ≈ 0.5M + 1M refill = 1.5M

Sliding window (2M per 60s):
    remaining-tokens stays FLAT at ~500K for ~60s after burst, then
    jumps as burst tokens expire from the trailing edge.
    30s after consuming 1.5M: remaining ≈ 500K (unchanged)

Usage
-----
    python experiments/burst_test.py [--burst-size 400] [--probe-duration 90]
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
OUTPUT_FILE = RESULTS_DIR / "burst_experiment.jsonl"

BURST_PROMPT = (
    "Write a long, detailed essay about the history of computing from the 1940s "
    "to the present day. Cover major milestones, key figures, and technological "
    "breakthroughs. Be comprehensive and thorough."
)
PROBE_PROMPT = "Say 'ok'."

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


async def burst_request(client: AsyncOpenAI, t0: float, idx: int,
                        max_tokens: int = 2048) -> dict:
    """Send one burst request and return header + usage info."""
    try:
        raw = await client.chat.completions.with_raw_response.create(
            model=MODEL,
            messages=[{"role": "user", "content": BURST_PROMPT}],
            max_tokens=max_tokens,
        )
        headers = extract_headers(raw.headers, t0)
        parsed = raw.parse()
        usage = {
            "prompt_tokens": parsed.usage.prompt_tokens,
            "completion_tokens": parsed.usage.completion_tokens,
            "total_tokens": parsed.usage.total_tokens,
        } if parsed.usage else {}
        return {"phase": "burst", "idx": idx, "status": "ok",
                **headers, **usage}
    except RateLimitError as exc:
        headers = {}
        if exc.response is not None:
            headers = extract_headers(exc.response.headers, t0)
        return {"phase": "burst", "idx": idx, "status": "429", **headers}
    except Exception as exc:
        return {"phase": "burst", "idx": idx, "status": "error",
                "t": round(time.time() - t0, 3), "error": str(exc)[:200]}


async def probe_request(client: AsyncOpenAI, t0: float, probe_idx: int) -> dict:
    """Send a tiny probe request to read remaining-tokens header."""
    try:
        raw = await client.chat.completions.with_raw_response.create(
            model=MODEL,
            messages=[{"role": "user", "content": PROBE_PROMPT}],
            max_tokens=1,
        )
        headers = extract_headers(raw.headers, t0)
        parsed = raw.parse()
        usage = {
            "prompt_tokens": parsed.usage.prompt_tokens,
            "completion_tokens": parsed.usage.completion_tokens,
            "total_tokens": parsed.usage.total_tokens,
        } if parsed.usage else {}
        return {"phase": "probe", "probe_idx": probe_idx, "status": "ok",
                **headers, **usage}
    except RateLimitError as exc:
        headers = {}
        if exc.response is not None:
            headers = extract_headers(exc.response.headers, t0)
        return {"phase": "probe", "probe_idx": probe_idx, "status": "429",
                **headers}
    except Exception as exc:
        return {"phase": "probe", "probe_idx": probe_idx, "status": "error",
                "t": round(time.time() - t0, 3), "error": str(exc)[:200]}


async def run_experiment(burst_size: int = 400, probe_duration: float = 90,
                         probe_interval: float = 3.0, max_tokens: int = 2048):
    client = AsyncOpenAI()
    results = []
    t0 = time.time()

    def log(row):
        results.append(row)
        with open(OUTPUT_FILE, "a") as f:
            f.write(json.dumps(row) + "\n")

    # --- Phase 0: Initial probe to confirm starting state ---
    print(f"[t=0.0] Phase 0: initial probe...")
    r = await probe_request(client, t0, -1)
    log(r)
    remaining_start = r.get("remaining-tokens", "?")
    print(f"  remaining-tokens = {remaining_start}")

    # --- Phase 1: BURST ---
    print(f"\n[t={time.time()-t0:.1f}] Phase 1: firing {burst_size} concurrent "
          f"requests (max_tokens={max_tokens})...")

    burst_t0 = time.time() - t0
    tasks = [burst_request(client, t0, i, max_tokens) for i in range(burst_size)]
    burst_results = await asyncio.gather(*tasks)

    ok_results = [r for r in burst_results if r["status"] == "ok"]
    err_results = [r for r in burst_results if r["status"] == "429"]
    total_tokens = sum(r.get("total_tokens", 0) for r in ok_results)
    burst_end_t = time.time() - t0

    for r in burst_results:
        log(r)

    last_remaining = None
    if ok_results:
        last_ok = max(ok_results, key=lambda r: r["t"])
        last_remaining = last_ok.get("remaining-tokens")

    print(f"  Burst complete at t={burst_end_t:.1f}s")
    print(f"  OK: {len(ok_results)}, 429: {len(err_results)}")
    print(f"  Total tokens consumed: {total_tokens:,}")
    print(f"  Last remaining-tokens: {last_remaining}")

    # --- Phase 2: PROBE ---
    print(f"\n[t={time.time()-t0:.1f}] Phase 2: probing every {probe_interval}s "
          f"for {probe_duration}s...")

    probe_start = time.time()
    probe_idx = 0
    while time.time() - probe_start < probe_duration:
        await asyncio.sleep(probe_interval)
        elapsed = time.time() - t0
        r = await probe_request(client, t0, probe_idx)
        log(r)
        rem = r.get("remaining-tokens", "?")
        reset = r.get("reset-tokens", "?")
        status = r.get("status", "?")
        sym = "✓" if status == "ok" else "✗"
        print(f"  [{sym}] t={elapsed:>6.1f}s  remaining-tokens={rem!s:>10}  "
              f"reset-tokens={reset!s:>6}")
        probe_idx += 1

    print(f"\n[t={time.time()-t0:.1f}] Experiment complete. {len(results)} data points.")
    print(f"Results written to {OUTPUT_FILE}")
    return results


def analyze(results: list[dict]):
    """Print analysis and verdict."""
    probes = [r for r in results if r["phase"] == "probe" and r["status"] == "ok"
              and "remaining-tokens" in r]
    bursts = [r for r in results if r["phase"] == "burst" and r["status"] == "ok"]

    if not probes or not bursts:
        print("Not enough data for analysis.")
        return

    total_burst_tokens = sum(r.get("total_tokens", 0) for r in bursts)
    burst_end_t = max(r["t"] for r in bursts)

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    print(f"Burst: {len(bursts)} requests, {total_burst_tokens:,} tokens consumed")
    print(f"Burst ended at t={burst_end_t:.1f}s")

    # Model predictions
    capacity = 2_000_000
    refill_rate = capacity / 60  # 33,333/s

    print(f"\n{'Probe t':>8} {'Remaining':>12} {'TB Predict':>12} {'SW Predict':>12} "
          f"{'TB Error':>10} {'SW Error':>10}")
    print("-" * 70)

    tb_errors = []
    sw_errors = []

    for p in probes:
        t = p["t"]
        remaining = p["remaining-tokens"]
        dt = max(0, t - burst_end_t)

        # Token bucket: linear refill from (capacity - total_burst_tokens)
        tb_remaining = min(capacity,
                           capacity - total_burst_tokens + dt * refill_rate)
        tb_remaining = max(0, tb_remaining)

        # Sliding window: burst tokens stay in window for 60s
        # (simplified: all burst tokens consumed at burst_end_t)
        if t - 0 < 60:
            sw_remaining = max(0, capacity - total_burst_tokens)
        else:
            sw_remaining = capacity

        tb_err = remaining - tb_remaining
        sw_err = remaining - sw_remaining

        tb_errors.append(abs(tb_err))
        sw_errors.append(abs(sw_err))

        print(f"{t:>7.1f}s {remaining:>11,} {tb_remaining:>11,.0f} "
              f"{sw_remaining:>11,.0f} {tb_err:>+9,.0f} {sw_err:>+9,.0f}")

    avg_tb = sum(tb_errors) / len(tb_errors)
    avg_sw = sum(sw_errors) / len(sw_errors)

    print(f"\nMean absolute error:")
    print(f"  Token bucket model: {avg_tb:,.0f}")
    print(f"  Sliding window model: {avg_sw:,.0f}")

    if avg_tb < avg_sw * 0.5:
        print(f"\n>>> VERDICT: TOKEN BUCKET (TB error {avg_tb/avg_sw:.1f}x lower)")
    elif avg_sw < avg_tb * 0.5:
        print(f"\n>>> VERDICT: SLIDING WINDOW (SW error {avg_sw/avg_tb:.1f}x lower)")
    else:
        print(f"\n>>> VERDICT: INCONCLUSIVE (errors within 2x of each other)")
        print(f"   Ratio TB/SW = {avg_tb/avg_sw:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Burst experiment for rate limit model")
    parser.add_argument("--burst-size", type=int, default=400,
                        help="Number of concurrent burst requests (default: 400)")
    parser.add_argument("--probe-duration", type=float, default=90,
                        help="Seconds to probe after burst (default: 90)")
    parser.add_argument("--probe-interval", type=float, default=3.0,
                        help="Seconds between probes (default: 3)")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="max_tokens for burst requests (default: 2048)")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Skip experiment, just analyze existing results")
    args = parser.parse_args()

    if args.analyze_only:
        with open(OUTPUT_FILE) as f:
            results = [json.loads(line) for line in f if line.strip()]
        analyze(results)
        return

    # Clear previous results
    OUTPUT_FILE.write_text("")

    results = asyncio.run(run_experiment(
        burst_size=args.burst_size,
        probe_duration=args.probe_duration,
        probe_interval=args.probe_interval,
        max_tokens=args.max_tokens,
    ))
    analyze(results)


if __name__ == "__main__":
    main()
