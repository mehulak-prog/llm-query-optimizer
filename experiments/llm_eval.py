#!/usr/bin/env python3
"""
Generalized version of anthropic_eval.py -- runs the FULL benchmark
(llm_planner via the chosen provider + explain_advisor + rule_based
baselines) for N repeats, for whichever provider you point it at. Same
methodology as Review 2's 4-run comparison and the Phase 3 Anthropic
run, so results across providers are directly comparable.

The paid-API confirmation gate only fires for --provider anthropic --
gemini and groq are free-tier, so they run immediately.

Usage (run from project root):
    python -m experiments.llm_eval --provider gemini --repeats 5
    python -m experiments.llm_eval --provider groq --repeats 5
    python -m experiments.llm_eval --provider anthropic --repeats 5
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

# Only the paid provider needs a spend confirmation gate.
PAID_PROVIDERS = {"anthropic"}


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


def model_name_for(provider: str, config: dict) -> str:
    if provider == "gemini":
        return config["llm"].get("gemini_model", "gemini-3.5-flash-lite")
    if provider == "groq":
        return config["llm"].get("groq_model", "openai/gpt-oss-120b")
    return config["llm"]["model"]


def confirm_if_paid(provider: str, repeats: int, model: str, temperature: float) -> None:
    if provider not in PAID_PROVIDERS:
        print(f"Provider '{provider}' is free-tier -- running without a spend confirmation.\n")
        return

    print("=" * 78)
    print("REAL, PAID API RUN -- confirm before proceeding")
    print("=" * 78)
    print(f"  Provider:     {provider}")
    print(f"  Model:        {model}")
    print(f"  Temperature:  {temperature}")
    print(f"  Repeats:      {repeats} (each repeat runs llm_planner + explain_advisor + rule_based)")
    print(f"  Budgeted estimate: ~$0.05-0.10 per llm_planner repeat, ~${0.05 * repeats:.2f}-{0.10 * repeats:.2f} total")
    print("=" * 78)
    answer = input("Type RUN to proceed, anything else to cancel: ").strip()
    if answer != "RUN":
        print("Cancelled -- no API calls made.")
        sys.exit(0)


def run_one(base_config: dict, workload, provider: str, repeat_index: int, out_dir: str) -> list[dict]:
    """Runs one full benchmark (llm_planner + baselines) for the given
    provider and returns one flat dict per system. Retries are scoped
    to connection-type errors only."""

    cfg = copy.deepcopy(base_config)
    cfg["llm"]["provider"] = provider
    cfg["llm"]["mock_mode"] = False
    per_run_csv = os.path.join(out_dir, f"{provider}_run_{repeat_index}.csv")
    cfg["benchmark"]["output_csv"] = per_run_csv

    max_connection_retries = 2
    backoff_seconds = 10 if provider in PAID_PROVIDERS else 5

    for attempt in range(max_connection_retries + 1):
        try:
            reports = run_benchmark(workload, cfg)
            write_csv(reports, per_run_csv)

            rows = []
            for r in reports:
                s = r.summary
                rows.append({
                    "provider": provider,
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
        except Exception as e:  # noqa: BLE001
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
                "provider": provider,
                "repeat_index": repeat_index,
                "system": "ALL (repeat failed)",
                "num_proposed": None, "num_accepted": None, "num_rejected": None,
                "best_speedup": None, "avg_accepted_speedup": None,
                "combined_speedup": None, "total_index_storage_mb": None,
                "error": f"{type(e).__name__}: {e}",
            }]


def summarize(rows: list[dict]) -> None:
    print("\n" + "=" * 78)
    print("SUMMARY -- per system, across repeats")
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
    parser.add_argument("--provider", required=True, choices=["anthropic", "gemini", "groq"])
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--temperature", type=float, default=None,
                         help="Override config.yaml's llm.temperature. Default: use config as-is.")
    parser.add_argument("--out-dir", default=None,
                         help="Default: experiments/<provider>_eval")
    parser.add_argument("--out", default=None,
                         help="Master aggregate CSV path. Default: <out-dir>/aggregate.csv")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(ROOT, "experiments", f"{args.provider}_eval")

    load_dotenv(os.path.join(ROOT, ".env"))

    base_config = load_base_config()
    if args.temperature is not None:
        base_config["llm"]["temperature"] = args.temperature
    ensure_sandbox_data(base_config)
    workload = load_workload(base_config["workload"]["path"])

    os.makedirs(out_dir, exist_ok=True)
    out_path = args.out or os.path.join(out_dir, "aggregate.csv")

    model = model_name_for(args.provider, base_config)
    confirm_if_paid(args.provider, args.repeats, model, base_config["llm"]["temperature"])

    all_rows = []
    for i in range(args.repeats):
        print(f"\n[{i + 1}/{args.repeats}] Running full benchmark ({args.provider} + baselines)...")
        all_rows.extend(run_one(base_config, workload, args.provider, i, out_dir))

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nMaster aggregate written to {out_path}")
    print(f"Per-repeat CSVs written to {out_dir}/{args.provider}_run_<i>.csv")

    summarize(all_rows)


if __name__ == "__main__":
    main()
