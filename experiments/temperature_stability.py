#!/usr/bin/env python3
"""
Phase 5a -- LLM temperature-stability check.

Runs the llm_planner system (baselines skipped -- they don't use the LLM,
so they're pure noise here) repeatedly at each of several temperature
values, on free-tier providers only (gemini, groq -- NOT anthropic, to
keep this check at zero cost and preserve the budgeted paid run for
Phase 3). For each (provider, temperature) pair, runs N independent
repeats and reports how much num_proposed / num_accepted / combined
speedup vary run-to-run at an IDENTICAL config -- this is the number
that answers "is the instability we saw (num_proposed 4-13 across
identical runs) a temperature artifact or something else?"

Usage (run from project root, same convention as run_demo.py):
    python -m experiments.temperature_stability
    python -m experiments.temperature_stability --providers gemini --temps 0.0 0.4 0.8 --repeats 5

Writes:
    experiments/temperature_stability_results.csv  -- one row per individual run
    Also prints a per-(provider, temperature) summary table (mean/stdev/range)
    to stdout.
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import statistics
import sys
import time

# Same sys.path bootstrap as run_demo.py -- MUST run before `from src...`
# imports below. Do not let an autoformatter reorder these.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # isort:skip

from dotenv import load_dotenv  # isort:skip
import yaml  # isort:skip

from src.benchmark import run_benchmark, write_csv  # isort:skip
from src.workload import load_workload  # isort:skip

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

DEFAULT_PROVIDERS = ["gemini", "groq"]
DEFAULT_TEMPS = [0.0, 0.4, 0.8]
DEFAULT_REPEATS = 3


def load_base_config() -> dict:
    config_path = os.path.join(ROOT, "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Same path resolution run_demo.py does, for consistency (harmless if
    # backend is postgres and these sqlite paths never get touched).
    config["sandbox"]["sqlite"]["source_db"] = os.path.join(
        ROOT, config["sandbox"]["sqlite"]["source_db"]
    )
    config["sandbox"]["sqlite"]["scratch_db"] = os.path.join(
        ROOT, config["sandbox"]["sqlite"]["scratch_db"]
    )
    config["workload"]["path"] = os.path.join(ROOT, config["workload"]["path"])
    return config


def ensure_sandbox_data(config: dict) -> None:
    source_db = config["sandbox"]["sqlite"]["source_db"]
    if not os.path.exists(source_db):
        print(f"{source_db} not found -- generating synthetic sandbox data first...")
        sys.path.insert(0, os.path.join(ROOT, "data"))
        import generate_data

        generate_data.build()


def run_one(base_config: dict, workload, provider: str, temperature: float, run_index: int) -> dict:
    """Runs a single fresh in-process benchmark call at the given
    provider/temperature and returns a flat dict of the metrics we care
    about, plus an 'error' key (None on success) so a single failed API
    call (e.g. a Groq rate limit) doesn't kill the whole sweep."""

    cfg = copy.deepcopy(base_config)
    cfg["llm"]["provider"] = provider
    cfg["llm"]["temperature"] = temperature
    cfg["llm"]["mock_mode"] = False
    # Skip explain_advisor/rule_based entirely -- they're deterministic
    # and don't touch the LLM, so re-running them per temperature value
    # would just burn sandbox-verification time for zero signal.
    cfg["benchmark"]["baselines"] = []
    cfg["benchmark"]["output_csv"] = os.path.join(
        ROOT, "experiments", "_temp_stability_scratch.csv"
    )

    row = {
        "provider": provider,
        "temperature": temperature,
        "run_index": run_index,
        "num_proposed": None,
        "num_accepted": None,
        "num_rejected": None,
        "best_speedup": None,
        "avg_accepted_speedup": None,
        "combined_speedup": None,
        "total_index_storage_mb": None,
        "error": None,
    }

    # Retries are ONLY for transient connection failures -- a genuine
    # CandidateParseError (the LLM returned a schema-noncompliant or
    # malformed candidate) is a real, meaningful outcome for a stability
    # study and must NOT be silently retried away; it's recorded as a
    # failed run on first occurrence, same as before.
    max_connection_retries = 2
    backoff_seconds = 5

    for attempt in range(max_connection_retries + 1):
        try:
            reports = run_benchmark(workload, cfg)
            # baselines == [] means reports has exactly one entry: llm_planner.
            r = reports[0]
            s = r.summary
            row.update(
                num_proposed=s["num_proposed"],
                num_accepted=s["num_accepted"],
                num_rejected=s["num_rejected"],
                best_speedup=s["best_speedup"],
                avg_accepted_speedup=s["avg_accepted_speedup"],
                combined_speedup=(
                    r.combined_effect.combined_speedup if r.combined_effect else None
                ),
                total_index_storage_mb=s["total_index_storage_mb"],
            )
            break
        except Exception as e:  # noqa: BLE001 -- deliberately broad: one bad
            # API call must not abort the whole sweep, and we want the
            # failure recorded in the CSV, not just printed and lost.
            is_connection_error = type(e).__name__ in (
                "APIConnectionError", "ConnectionError", "Timeout", "TimeoutError",
            )
            if is_connection_error and attempt < max_connection_retries:
                wait = backoff_seconds * (attempt + 1)
                print(f"  .. transient {type(e).__name__}, retrying in {wait}s "
                      f"(attempt {attempt + 1}/{max_connection_retries})...")
                time.sleep(wait)
                continue
            row["error"] = f"{type(e).__name__}: {e}"
            print(f"  !! run failed ({provider}, temp={temperature}, run {run_index}): {row['error']}")
            break

    return row


def summarize(rows: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("SUMMARY -- per (provider, temperature), across repeats")
    print("=" * 78)

    keys = sorted({(r["provider"], r["temperature"]) for r in rows})
    for provider, temp in keys:
        group = [r for r in rows if r["provider"] == provider and r["temperature"] == temp]
        ok = [r for r in group if r["error"] is None]
        failed = len(group) - len(ok)

        print(f"\n{provider} @ temperature={temp}  ({len(ok)} ok, {failed} failed)")
        if not ok:
            print("  (no successful runs -- see errors above)")
            continue

        for metric in ("num_proposed", "num_accepted", "combined_speedup"):
            vals = [r[metric] for r in ok if r[metric] is not None]
            if not vals:
                print(f"  {metric}: no data")
                continue
            mean = statistics.mean(vals)
            stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
            print(
                f"  {metric:20s} mean={mean:6.3f}  stdev={stdev:6.3f}  "
                f"range=[{min(vals):.3f}, {max(vals):.3f}]  (n={len(vals)})"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--providers", nargs="+", default=DEFAULT_PROVIDERS,
                         choices=["gemini", "groq"],
                         help="Free-tier providers only -- anthropic deliberately excluded from this check.")
    parser.add_argument("--temps", nargs="+", type=float, default=DEFAULT_TEMPS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS,
                         help="Independent runs per (provider, temperature) pair.")
    parser.add_argument("--out", default=os.path.join(ROOT, "experiments", "temperature_stability_results.csv"))
    args = parser.parse_args()

    load_dotenv(os.path.join(ROOT, ".env"))

    base_config = load_base_config()
    ensure_sandbox_data(base_config)
    workload = load_workload(base_config["workload"]["path"])

    total_runs = len(args.providers) * len(args.temps) * args.repeats
    print(f"Temperature stability sweep: {len(args.providers)} provider(s) x "
          f"{len(args.temps)} temperature(s) x {args.repeats} repeat(s) = {total_runs} runs\n")

    all_rows = []
    run_counter = 0
    for provider in args.providers:
        for temp in args.temps:
            for i in range(args.repeats):
                run_counter += 1
                print(f"[{run_counter}/{total_runs}] {provider} temp={temp} repeat={i + 1}/{args.repeats} ...")
                row = run_one(base_config, workload, provider, temp, i)
                all_rows.append(row)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nPer-run results written to {args.out}")

    summarize(all_rows)


if __name__ == "__main__":
    main()
