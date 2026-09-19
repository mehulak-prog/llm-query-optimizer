#!/usr/bin/env python3
"""
Phase 3 completion -- the real, budgeted Anthropic API evaluation.

Unlike temperature_stability.py, this deliberately runs the FULL benchmark
(llm_planner via Anthropic + explain_advisor + rule_based baselines) every
repeat, matching Review 2's exact methodology, so these numbers slot
directly into the same comparison table (explain_advisor vs llm_planner
vs rule_based, average combined_speedup per system).

This makes REAL, PAID Anthropic API calls. It will not run without you
typing RUN at the confirmation prompt.

Usage (run from project root):
    python -m experiments.anthropic_eval
    python -m experiments.anthropic_eval --repeats 5
    python -m experiments.anthropic_eval --repeats 3 --temperature 0.4
"""

from __future__ import annotations

import argparse
import copy
import csv
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # isort:skip

from dotenv import load_dotenv  # isort:skip
import yaml  # isort:skip

from src.benchmark import run_benchmark, write_csv  # isort:skip
from src.workload import load_workload  # isort:skip

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

DEFAULT_REPEATS = 5


def load_base_config() -> dict:
    config_path = os.path.join(ROOT, "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

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


def confirm_or_exit(repeats: int, model: str, temperature: float) -> None:
    print("=" * 78)
    print("REAL, PAID ANTHROPIC API RUN -- confirm before proceeding")
    print("=" * 78)
    print(f"  Model:        {model}")
    print(f"  Temperature:  {temperature}")
    print(f"  Repeats:      {repeats} (each repeat runs llm_planner + explain_advisor + rule_based)")
    print(f"  Baselines per repeat: explain_advisor, rule_based (no extra API cost -- deterministic)")
    print(f"  Budgeted estimate: ~$0.05-0.10 per llm_planner repeat, ~${0.05 * repeats:.2f}-{0.10 * repeats:.2f} total")
    print(f"  Hard spend cap: $5 credits loaded, auto-reload OFF (per your Console setup)")
    print("=" * 78)
    answer = input("Type RUN to proceed, anything else to cancel: ").strip()
    if answer != "RUN":
        print("Cancelled -- no API calls made.")
        sys.exit(0)


def run_one(base_config: dict, workload, repeat_index: int, out_dir: str) -> list[dict]:
    """Runs one full benchmark (llm_planner + baselines) and returns one
    flat dict per system for the master aggregate CSV. Retries are scoped
    to connection-type errors only -- a CandidateParseError or a billing/
    permission error must never be silently retried."""

    cfg = copy.deepcopy(base_config)
    cfg["llm"]["provider"] = "anthropic"
    cfg["llm"]["mock_mode"] = False
    per_run_csv = os.path.join(out_dir, f"anthropic_run_{repeat_index}.csv")
    cfg["benchmark"]["output_csv"] = per_run_csv

    max_connection_retries = 2
    backoff_seconds = 10  # longer than the free-tier sweep's backoff -- a
    # paid call failing is more expensive to get wrong than a free one.

    for attempt in range(max_connection_retries + 1):
        try:
            reports = run_benchmark(workload, cfg)
            write_csv(reports, per_run_csv)

            rows = []
            for r in reports:
                s = r.summary
                rows.append({
                    "repeat_index": repeat_index,
                    "system": r.system_name,
                    "num_proposed": s["num_proposed"],
                    "num_accepted": s["num_accepted"],
                    "num_rejected": s["num_rejected"],
                    "best_speedup": s["best_speedup"],
                    "avg_accepted_speedup": s["avg_accepted_speedup"],
                    "combined_speedup": (
                        r.combined_effect.combined_speedup if r.combined_effect else None
                    ),
                    "total_index_storage_mb": s["total_index_storage_mb"],
                    "error": None,
                })
            return rows
        except Exception as e:  # noqa: BLE001 -- one bad repeat must not
            # lose the whole eval; a permission/billing error should NOT
            # be retried (retrying a bad key just wastes time), only
            # genuine transient connection failures should be.
            is_connection_error = type(e).__name__ in (
                "APIConnectionError", "ConnectionError", "Timeout", "TimeoutError",
            )
            if is_connection_error and attempt < max_connection_retries:
                wait = backoff_seconds * (attempt + 1)
                print(f"  .. transient {type(e).__name__}, retrying in {wait}s "
                      f"(attempt {attempt + 1}/{max_connection_retries})...")
                time.sleep(wait)
                continue
            print(f"  !! repeat {repeat_index} failed: {type(e).__name__}: {e}")
            return [{
                "repeat_index": repeat_index,
                "system": "ALL (repeat failed)",
                "num_proposed": None, "num_accepted": None, "num_rejected": None,
                "best_speedup": None, "avg_accepted_speedup": None,
                "combined_speedup": None, "total_index_storage_mb": None,
                "error": f"{type(e).__name__}: {e}",
            }]


def summarize(rows: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("SUMMARY -- per system, across repeats (this is your Results-section table)")
    print("=" * 78)

    systems = sorted({r["system"] for r in rows if r["error"] is None})
    for system in systems:
        group = [r for r in rows if r["system"] == system]
        print(f"\n{system}  (n={len(group)})")
        for metric in ("num_proposed", "num_accepted", "best_speedup",
                       "avg_accepted_speedup", "combined_speedup",
                       "total_index_storage_mb"):
            vals = [r[metric] for r in group if r[metric] is not None]
            if not vals:
                print(f"  {metric}: no data")
                continue
            mean = statistics.mean(vals)
            stdev = statistics.stdev(vals) if len(vals) > 1 else 0.0
            print(
                f"  {metric:22s} mean={mean:7.3f}  stdev={stdev:6.3f}  "
                f"range=[{min(vals):.3f}, {max(vals):.3f}]  (n={len(vals)})"
            )

    failed = [r for r in rows if r["error"] is not None]
    if failed:
        print(f"\n{len(failed)} repeat(s) failed entirely:")
        for r in failed:
            print(f"  repeat {r['repeat_index']}: {r['error']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--temperature", type=float, default=None,
                         help="Override config.yaml's llm.temperature. Default: use config as-is.")
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "experiments", "anthropic_eval"))
    parser.add_argument("--out", default=None,
                         help="Master aggregate CSV path. Default: <out-dir>/aggregate.csv")
    args = parser.parse_args()

    load_dotenv(os.path.join(ROOT, ".env"))

    base_config = load_base_config()
    if args.temperature is not None:
        base_config["llm"]["temperature"] = args.temperature
    ensure_sandbox_data(base_config)
    workload = load_workload(base_config["workload"]["path"])

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = args.out or os.path.join(args.out_dir, "aggregate.csv")

    confirm_or_exit(
        args.repeats,
        base_config["llm"]["model"],
        base_config["llm"]["temperature"],
    )

    all_rows = []
    for i in range(args.repeats):
        print(f"\n[{i + 1}/{args.repeats}] Running full benchmark (llm_planner=Anthropic + baselines)...")
        all_rows.extend(run_one(base_config, workload, i, args.out_dir))

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nMaster aggregate written to {out_path}")
    print(f"Per-repeat CSVs written to {args.out_dir}/anthropic_run_<i>.csv")

    summarize(all_rows)


if __name__ == "__main__":
    main()
