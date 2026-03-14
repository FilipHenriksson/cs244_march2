"""Measure the token-bucket refill rate of OpenAI's TPM rate limiter.

Protocol
--------
Run K independent trials, each with a different burst size.  Each trial:

  1. CONFIRM bucket is full — probe until remaining-tokens > 0.99 × capacity.
  2. BURST — fire N concurrent requests (max_tokens=2048) to create a
     measurable deficit D.  Record every response to compute D precisely.
  3. PROBE — immediately send tiny requests (max_tokens=1) every PROBE_INTERVAL
     seconds for PROBE_DURATION seconds.  This traces the recovery curve.
  4. WAIT — sleep until the bucket is expected to be full again before the
     next trial.

Across trials we vary N to create different deficit depths (small, medium,
large).  The refill rate R should be consistent across depths — if it isn't,
the model is wrong or there's a nonlinear effect.

From each trial's probe data, we fit:
    remaining(t) = remaining_0 + R × (t − t_burst_end)
to the linear recovery region (before the bucket caps at capacity).
R is the refill rate in tokens/second.

Usage
-----
    python experiments/refill_rate_test.py
    python experiments/refill_rate_test.py --analyze-only
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
OUTPUT_FILE = RESULTS_DIR / "refill_rate_experiment.jsonl"

BURST_PROMPT = (
    "Write a long, detailed essay about the history of computing from the 1940s "
    "to the present day. Cover major milestones, key figures, and technological "
    "breakthroughs. Be comprehensive and thorough."
)
PROBE_PROMPT = "Say ok."

RL_HEADER_KEYS = (
    "x-ratelimit-limit-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-requests",
    "x-ratelimit-reset-tokens",
)

CAPACITY = 2_000_000

# Trial burst sizes: small → large deficit
TRIAL_BURST_SIZES = [100, 250, 500]
PROBE_INTERVAL = 0.5       # seconds between probes
PROBE_DURATION = 90         # seconds of probing per trial
RECOVERY_WAIT  = 120        # seconds between trials


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


async def send_request(client: AsyncOpenAI, t0: float, *,
                       prompt: str, max_tokens: int,
                       phase: str, trial: int, idx: int) -> dict:
    try:
        raw = await client.chat.completions.with_raw_response.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        headers = extract_headers(raw.headers, t0)
        parsed = raw.parse()
        usage = {}
        if parsed.usage:
            usage = {
                "prompt_tokens": parsed.usage.prompt_tokens,
                "completion_tokens": parsed.usage.completion_tokens,
                "total_tokens": parsed.usage.total_tokens,
            }
        return {"trial": trial, "phase": phase, "idx": idx, "status": "ok",
                **headers, **usage}
    except RateLimitError as exc:
        headers = {}
        if exc.response is not None:
            headers = extract_headers(exc.response.headers, t0)
        return {"trial": trial, "phase": phase, "idx": idx, "status": "429",
                **headers}
    except Exception as exc:
        return {"trial": trial, "phase": phase, "idx": idx, "status": "error",
                "t": round(time.time() - t0, 3), "error": str(exc)[:200]}


async def wait_for_full_bucket(client: AsyncOpenAI, t0: float, trial: int,
                               log_fn, threshold: float = 0.99):
    """Poll until remaining-tokens is above threshold × capacity."""
    target = int(CAPACITY * threshold)
    attempt = 0
    while True:
        r = await send_request(client, t0, prompt=PROBE_PROMPT, max_tokens=1,
                               phase="wait", trial=trial, idx=attempt)
        log_fn(r)
        remaining = r.get("remaining-tokens", 0)
        if remaining >= target:
            print(f"    Bucket confirmed full: remaining-tokens={remaining:,}")
            return r
        print(f"    Waiting... remaining-tokens={remaining:,} (need {target:,})")
        attempt += 1
        await asyncio.sleep(5)


async def run_trial(client: AsyncOpenAI, t0: float, trial_idx: int,
                    burst_size: int, log_fn):
    """Run a single burst + probe trial."""
    print(f"\n{'='*60}")
    print(f"TRIAL {trial_idx}: burst_size={burst_size}")
    print(f"{'='*60}")

    # Step 1: Confirm bucket is full
    print(f"  [t={time.time()-t0:.0f}s] Confirming bucket is full...")
    await wait_for_full_bucket(client, t0, trial_idx, log_fn)

    # Step 2: Burst
    print(f"  [t={time.time()-t0:.0f}s] Firing {burst_size} concurrent requests "
          f"(max_tokens=2048)...")
    burst_start = time.time()
    tasks = [
        send_request(client, t0, prompt=BURST_PROMPT, max_tokens=2048,
                     phase="burst", trial=trial_idx, idx=i)
        for i in range(burst_size)
    ]
    burst_results = await asyncio.gather(*tasks)

    ok = [r for r in burst_results if r["status"] == "ok"]
    errs = [r for r in burst_results if r["status"] == "429"]
    total_tokens = sum(r.get("total_tokens", 0) for r in ok)
    burst_end = time.time()
    burst_duration = burst_end - burst_start

    for r in burst_results:
        log_fn(r)

    print(f"  Burst done in {burst_duration:.1f}s: "
          f"ok={len(ok)} 429={len(errs)} tokens={total_tokens:,}")

    if ok:
        last_remaining = min(r.get("remaining-tokens", CAPACITY) for r in ok
                             if "remaining-tokens" in r)
        print(f"  Min remaining-tokens during burst: {last_remaining:,}")

    # Step 3: Probe — dense sampling of recovery curve
    print(f"  [t={time.time()-t0:.0f}s] Probing every {PROBE_INTERVAL}s "
          f"for {PROBE_DURATION}s...")
    probe_start = time.time()
    probe_idx = 0
    while time.time() - probe_start < PROBE_DURATION:
        await asyncio.sleep(PROBE_INTERVAL)
        r = await send_request(client, t0, prompt=PROBE_PROMPT, max_tokens=1,
                               phase="probe", trial=trial_idx, idx=probe_idx)
        log_fn(r)

        remaining = r.get("remaining-tokens", "?")
        elapsed = time.time() - t0
        dt = time.time() - burst_end
        if probe_idx % 10 == 0:
            print(f"    t={elapsed:.0f}s (burst+{dt:.1f}s)  "
                  f"remaining-tokens={remaining}")
        probe_idx += 1

    print(f"  Trial {trial_idx} complete: {probe_idx} probes collected")


async def run_experiment():
    client = AsyncOpenAI()
    t0 = time.time()

    def log_fn(row):
        with open(OUTPUT_FILE, "a") as f:
            f.write(json.dumps(row) + "\n")

    print(f"Refill rate experiment: {len(TRIAL_BURST_SIZES)} trials")
    print(f"Burst sizes: {TRIAL_BURST_SIZES}")
    print(f"Probe interval: {PROBE_INTERVAL}s, duration: {PROBE_DURATION}s")
    print(f"Recovery wait between trials: {RECOVERY_WAIT}s")

    for i, burst_size in enumerate(TRIAL_BURST_SIZES):
        if i > 0:
            print(f"\n  Waiting {RECOVERY_WAIT}s for bucket to recover...")
            await asyncio.sleep(RECOVERY_WAIT)

        await run_trial(client, t0, trial_idx=i, burst_size=burst_size,
                        log_fn=log_fn)

    print(f"\n[t={time.time()-t0:.0f}s] All trials complete.")
    print(f"Results written to {OUTPUT_FILE}")


def analyze():
    """Fit refill rate from each trial's probe data."""
    import numpy as np

    with open(OUTPUT_FILE) as f:
        results = [json.loads(line) for line in f if line.strip()]

    trial_ids = sorted(set(r["trial"] for r in results))

    print("=" * 70)
    print("REFILL RATE ANALYSIS")
    print("=" * 70)

    all_rates = []

    for trial in trial_ids:
        burst = [r for r in results
                 if r["trial"] == trial and r["phase"] == "burst"
                 and r["status"] == "ok"]
        probes = [r for r in results
                  if r["trial"] == trial and r["phase"] == "probe"
                  and r["status"] == "ok" and "remaining-tokens" in r]

        if not burst or not probes:
            print(f"\nTrial {trial}: insufficient data")
            continue

        total_tokens = sum(r.get("total_tokens", 0) for r in burst)
        burst_end_t = max(r["t"] for r in burst)

        print(f"\n--- Trial {trial} ---")
        print(f"  Burst: {len(burst)} requests, {total_tokens:,} tokens")
        print(f"  Burst ended: t={burst_end_t:.1f}s")
        print(f"  Probes: {len(probes)}")

        # Extract recovery curve: (dt_from_burst_end, remaining_tokens)
        pts = []
        for p in probes:
            dt = p["t"] - burst_end_t
            remaining = p["remaining-tokens"]
            pts.append((dt, remaining))

        pts.sort()
        dts = np.array([p[0] for p in pts])
        rems = np.array([p[1] for p in pts])

        # Identify the LINEAR recovery region:
        # - Start: first probe (bucket is depleted)
        # - End: last probe before remaining hits capacity (within 1%)
        cap_threshold = CAPACITY * 0.99
        linear_mask = rems < cap_threshold
        if linear_mask.sum() < 3:
            print(f"  Too few points in linear region ({linear_mask.sum()}). "
                  f"Burst may be too small.")
            # Still estimate from the first few points
            if len(pts) >= 2 and rems[0] < cap_threshold:
                linear_mask[:min(5, len(pts))] = True
            else:
                # Use two-point estimate from deficit to first full reading
                first_below = None
                first_full = None
                for dt_val, rem_val in pts:
                    if rem_val < cap_threshold and first_below is None:
                        first_below = (dt_val, rem_val)
                    if rem_val >= cap_threshold and first_full is None and first_below is not None:
                        first_full = (dt_val, rem_val)
                        break
                if first_below and first_full:
                    R = (first_full[1] - first_below[1]) / (first_full[0] - first_below[0])
                    print(f"  Two-point estimate: R = {R:,.0f} tok/s ({R*60:,.0f} tok/min)")
                    all_rates.append(R)
                continue

        dts_lin = dts[linear_mask]
        rems_lin = rems[linear_mask]

        # Linear regression: remaining = a + R * dt
        if len(dts_lin) >= 2:
            coeffs = np.polyfit(dts_lin, rems_lin, 1)
            R = coeffs[0]  # slope = refill rate
            intercept = coeffs[1]

            # Residuals
            predicted = np.polyval(coeffs, dts_lin)
            residuals = rems_lin - predicted
            rmse = np.sqrt(np.mean(residuals**2))
            r_squared = 1 - np.sum(residuals**2) / np.sum((rems_lin - np.mean(rems_lin))**2)

            print(f"  Linear region: {len(dts_lin)} points, "
                  f"dt=[{dts_lin[0]:.1f}, {dts_lin[-1]:.1f}]s")
            print(f"  Remaining range: [{rems_lin[0]:,.0f}, {rems_lin[-1]:,.0f}]")
            print(f"  Refill rate R = {R:,.0f} tok/s ({R*60:,.0f} tok/min)")
            print(f"  R² = {r_squared:.6f},  RMSE = {rmse:,.0f}")
            print(f"  Intercept = {intercept:,.0f} "
                  f"(implied deficit = {CAPACITY - intercept:,.0f})")

            all_rates.append(R)

            # Print first/last few points for inspection
            print(f"  Recovery curve (first 10 + last 5):")
            for j in list(range(min(10, len(pts)))) + list(range(max(0, len(pts)-5), len(pts))):
                dt_val, rem_val = pts[j]
                in_fit = "←" if dt_val <= dts_lin[-1] and rem_val < cap_threshold else ""
                print(f"    burst+{dt_val:>6.1f}s  remaining={rem_val:>10,}  {in_fit}")

    # Summary across trials
    if len(all_rates) >= 2:
        rates = np.array(all_rates)
        print(f"\n{'='*70}")
        print(f"SUMMARY: {len(rates)} trials")
        print(f"  Refill rates: {[f'{r:,.0f}' for r in rates]} tok/s")
        print(f"  Mean:   {np.mean(rates):>10,.0f} tok/s ({np.mean(rates)*60:>12,.0f} tok/min)")
        print(f"  Std:    {np.std(rates):>10,.0f} tok/s")
        print(f"  95% CI: [{np.mean(rates) - 1.96*np.std(rates):>,.0f}, "
              f"{np.mean(rates) + 1.96*np.std(rates):>,.0f}] tok/s")
        print(f"\n  Compare: naive 'TPM/60' = {CAPACITY/60:,.0f} tok/s "
              f"({CAPACITY:,} tok/min)")
        print(f"  Ratio actual/naive = {np.mean(rates) / (CAPACITY/60):.2f}")
    elif len(all_rates) == 1:
        print(f"\n  Single trial estimate: R = {all_rates[0]:,.0f} tok/s "
              f"({all_rates[0]*60:,.0f} tok/min)")


def main():
    global TRIAL_BURST_SIZES, PROBE_INTERVAL, PROBE_DURATION, RECOVERY_WAIT

    parser = argparse.ArgumentParser(
        description="Measure OpenAI TPM token-bucket refill rate")
    parser.add_argument("--burst-sizes", type=int, nargs="+",
                        default=TRIAL_BURST_SIZES,
                        help=f"Burst sizes for each trial (default: {TRIAL_BURST_SIZES})")
    parser.add_argument("--probe-interval", type=float, default=PROBE_INTERVAL,
                        help=f"Seconds between probes (default: {PROBE_INTERVAL})")
    parser.add_argument("--probe-duration", type=float, default=PROBE_DURATION,
                        help=f"Probe duration per trial (default: {PROBE_DURATION})")
    parser.add_argument("--recovery-wait", type=float, default=RECOVERY_WAIT,
                        help=f"Wait between trials (default: {RECOVERY_WAIT})")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Skip experiment, just analyze existing results")
    args = parser.parse_args()

    if args.analyze_only:
        analyze()
        return

    TRIAL_BURST_SIZES = args.burst_sizes
    PROBE_INTERVAL = args.probe_interval
    PROBE_DURATION = args.probe_duration
    RECOVERY_WAIT = args.recovery_wait

    OUTPUT_FILE.write_text("")
    asyncio.run(run_experiment())
    analyze()


if __name__ == "__main__":
    main()
