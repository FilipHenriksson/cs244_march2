#!/usr/bin/env python3
"""Full test run — all schedulers × 2 sim modes × 2 rate-limit profiles.

Runs every scheduler under all four combinations of arrival pattern and
rate-limit constraint, then merges results into a single comparison CSV.

Rate-limit profiles:
    rpm_heavy  — RPM=20, TPM=200k  (request slots are the bottleneck)
    tpm_heavy  — RPM=60, TPM=40k   (token budget is the bottleneck)

Simulation modes:
    constant   — uniform inter-arrival spacing
    bursty     — clustered wave arrivals with quiet gaps

Total: 5 schedulers × 4 scenarios = 20 scheduler runs.

Usage::

    python test_run.py
    python test_run.py --sessions 15   # faster test with fewer sessions
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime


SIM_MODES = ["constant", "bursty"]

RATE_PROFILES = {
    "rpm_heavy": {"rpm": 20, "tpm": 200_000},
    "tpm_heavy": {"rpm": 60, "tpm": 40_000},
}


def run_scenario(profile_name: str, sim_mode: str, out_dir: str, args) -> str:
    """Run all schedulers for one profile × mode combo, return the run subdir."""
    profile = RATE_PROFILES[profile_name]
    scenario_dir = os.path.join(out_dir, f"{profile_name}_{sim_mode}")
    cmd = [
        sys.executable, "-m", "sim.runner",
        "--all-schedulers",
        "--sessions", str(args.sessions),
        "--stagger-mode", sim_mode,
        "--stagger", str(args.stagger),
        "--rpm", str(profile["rpm"]),
        "--tpm", str(profile["tpm"]),
        "--cost-limit", str(args.cost_limit),
        "--max-tokens", str(args.max_tokens),
        "--seed", str(args.seed),
        "--prompt-mode", args.prompt_mode,
        "--output-dir", scenario_dir,
    ]

    print(f"\n{'#'*70}")
    print(f" SCENARIO: {profile_name} × {sim_mode}")
    print(f" RPM={profile['rpm']}  TPM={profile['tpm']:,}  sessions={args.sessions}")
    print(f" Command: {' '.join(cmd)}")
    print(f"{'#'*70}\n")

    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    if result.returncode != 0:
        print(f"\n*** WARNING: {profile_name}×{sim_mode} exited with "
              f"code {result.returncode} ***\n")

    subdirs = sorted(
        [d for d in os.listdir(scenario_dir)
         if os.path.isdir(os.path.join(scenario_dir, d)) and d.startswith("run_")],
    )
    if not subdirs:
        print(f"*** ERROR: No run directory found in {scenario_dir} ***")
        return ""
    return os.path.join(scenario_dir, subdirs[-1])


def merge_results(run_dirs: dict[str, str], out_dir: str):
    """Merge comparison CSVs from all scenarios into a single CSV + table."""
    merged_rows = []
    header = None

    for scenario_key, run_dir in run_dirs.items():
        if not run_dir:
            continue
        csv_path = os.path.join(run_dir, "comparison.csv")
        if not os.path.exists(csv_path):
            print(f"*** WARNING: {csv_path} not found, skipping ***")
            continue

        profile_name, sim_mode = scenario_key.rsplit("_", 1)
        # Handle profile names that contain underscores
        for pname in RATE_PROFILES:
            if scenario_key.startswith(pname + "_"):
                profile_name = pname
                sim_mode = scenario_key[len(pname) + 1:]
                break

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            rows = list(reader)
            if not rows:
                continue
            if header is None:
                header = ["rate_profile", "sim_mode"] + rows[0]
            for row in rows[1:]:
                merged_rows.append([profile_name, sim_mode] + row)

    if not header or not merged_rows:
        print("*** No results to merge ***")
        return

    merged_path = os.path.join(out_dir, "combined_comparison.csv")
    with open(merged_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(merged_rows)

    _print_summary_table(header, merged_rows, merged_path, run_dirs)


def _print_summary_table(header, merged_rows, merged_path, run_dirs):
    """Print a readable console summary of all scenarios."""
    prof_idx = header.index("rate_profile")
    mode_idx = header.index("sim_mode")
    sched_idx = header.index("scheduler")
    mean_idx = header.index("mean_session_s") if "mean_session_s" in header else None
    median_idx = header.index("median_session_s") if "median_session_s" in header else None
    p95_idx = header.index("p95_session_s") if "p95_session_s" in header else None
    wall_idx = header.index("wall_time_s") if "wall_time_s" in header else None
    bottleneck_idx = header.index("bottleneck") if "bottleneck" in header else None
    rpm_t_idx = header.index("rpm_throttles") if "rpm_throttles" in header else None
    tpm_t_idx = header.index("tpm_throttles") if "tpm_throttles" in header else None
    rpm_w_idx = header.index("rpm_waits") if "rpm_waits" in header else None
    tpm_w_idx = header.index("tpm_waits") if "tpm_waits" in header else None
    skips_idx = header.index("tpm_skips") if "tpm_skips" in header else None
    par_idx = header.index("parallelism") if "parallelism" in header else None

    print(f"\n{'='*140}")
    print(f" COMBINED RESULTS — ALL SCHEDULERS × ALL SCENARIOS")
    print(f"{'='*140}")

    col = 12
    tbl_hdr = (
        f"{'Profile':<12} {'Mode':<10} {'Scheduler':<26}"
        f" {'Mean':>{col}} {'Median':>{col}} {'P95':>{col}}"
        f" {'Wall':>{col}} {'Parall.':>{col}}"
        f" {'RPM pres':>{col}} {'TPM pres':>{col}} {'Skips':>{col}}"
        f" {'Bottleneck':>{col}}"
    )
    print(tbl_hdr)
    print("-" * len(tbl_hdr))

    prev_scenario = None
    for row in merged_rows:
        scenario = (row[prof_idx], row[mode_idx])
        if prev_scenario and scenario != prev_scenario:
            print("-" * len(tbl_hdr))
        prev_scenario = scenario

        def _v(idx, suf=""):
            if idx is None:
                return "—"
            v = row[idx]
            return f"{v}{suf}" if v else "—"

        rpm_p = int(row[rpm_t_idx] or 0) + int(row[rpm_w_idx] or 0) if rpm_t_idx and rpm_w_idx else "—"
        tpm_p = int(row[tpm_t_idx] or 0) + int(row[tpm_w_idx] or 0) if tpm_t_idx and tpm_w_idx else "—"

        print(
            f"{row[prof_idx]:<12} {row[mode_idx]:<10} {row[sched_idx]:<26}"
            f" {_v(mean_idx, 's'):>{col}} {_v(median_idx, 's'):>{col}} {_v(p95_idx, 's'):>{col}}"
            f" {_v(wall_idx, 's'):>{col}} {_v(par_idx, 'x'):>{col}}"
            f" {str(rpm_p):>{col}} {str(tpm_p):>{col}} {_v(skips_idx):>{col}}"
            f" {_v(bottleneck_idx):>{col}}"
        )

    print(f"{'='*140}")
    print(f"\n  Combined CSV: {merged_path}")
    for key, run_dir in run_dirs.items():
        if run_dir:
            print(f"  {key:>25}: {run_dir}/")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Full test run: all schedulers × 2 sim modes × 2 rate profiles")
    parser.add_argument("--sessions", type=int, default=30,
                        help="Sessions per scheduler per scenario (default: 30)")
    parser.add_argument("--stagger", type=float, default=4.0,
                        help="Mean seconds between arrivals (default: 4.0)")
    parser.add_argument("--cost-limit", type=float, default=20.0,
                        help="Cost cap in USD per scenario (default: 20.0)")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Max tokens per completion (default: 2048)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--prompt-mode", type=str, default="strict",
                        choices=["default", "strict"],
                        help="Prompt mode (default: strict)")
    parser.add_argument("--output-dir", type=str, default="results",
                        help="Base output directory (default: results)")
    args = parser.parse_args()

    n_scenarios = len(RATE_PROFILES) * len(SIM_MODES)
    n_schedulers = 5
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_dir, f"test_run_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"{'='*70}")
    print(f" TEST RUN")
    print(f"{'='*70}")
    print(f"  Scenarios:    {n_scenarios} ({len(RATE_PROFILES)} rate profiles × {len(SIM_MODES)} sim modes)")
    print(f"  Schedulers:   {n_schedulers}")
    print(f"  Sessions:     {args.sessions} per scheduler per scenario")
    print(f"  Total runs:   {n_scenarios * n_schedulers}")
    print(f"  Rate profiles:")
    for name, profile in RATE_PROFILES.items():
        print(f"    {name:<12} RPM={profile['rpm']:<6} TPM={profile['tpm']:,}")
    print(f"  Sim modes:    {', '.join(SIM_MODES)}")
    print(f"  Output:       {out_dir}/")
    print(f"{'='*70}")

    run_dirs = {}
    total_start = time.time()
    scenario_num = 0

    for profile_name in RATE_PROFILES:
        for sim_mode in SIM_MODES:
            scenario_num += 1
            key = f"{profile_name}_{sim_mode}"
            elapsed = time.time() - total_start
            print(f"\n>>> Scenario {scenario_num}/{n_scenarios}: {key} "
                  f"(elapsed: {elapsed/60:.1f}min)")
            run_dirs[key] = run_scenario(profile_name, sim_mode, out_dir, args)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f" ALL {n_scenarios} SCENARIOS COMPLETE — "
          f"total elapsed: {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    print(f"{'='*70}")

    merge_results(run_dirs, out_dir)


if __name__ == "__main__":
    main()
