#!/usr/bin/env python3
"""
End-to-end demo: builds the sandbox DB if missing, runs the LLM planner
(mock mode by default) through the full verification loop, benchmarks it
against the two baseline advisors, and prints/saves a comparison table.

    python experiments/run_demo.py
"""

from __future__ import annotations
from src.metrics import candidate_label  # isort:skip
from src.benchmark import run_benchmark, write_csv  # isort:skip
from src.workload import load_workload  # isort:skip
from dotenv import load_dotenv
from tabulate import tabulate
import yaml

# NOTE: sys.path.insert() below MUST run before the `from src...` imports
# further down -- it's what makes `src` importable at all when this script
# is run directly. Auto-formatters (isort, "organize imports on save") sort
# alphabetically and don't know about this side-effect dependency, so they
# WILL silently break this file by moving the src imports above it. The
# `# isort:skip` markers tell isort to leave these specific lines alone;
# additionally, prefer running this file as `python -m experiments.run_demo`
# from the project root, which adds the cwd to sys.path automatically and
# makes this whole ordering issue moot regardless of import order.
import os  # isort:skip
import sys  # isort:skip

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))  # isort:skip


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    load_dotenv(os.path.join(ROOT, ".env"))

    config_path = os.path.join(ROOT, "config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Resolve relative paths against project root regardless of cwd.
    config["sandbox"]["sqlite"]["source_db"] = os.path.join(
        ROOT, config["sandbox"]["sqlite"]["source_db"]
    )
    config["sandbox"]["sqlite"]["scratch_db"] = os.path.join(
        ROOT, config["sandbox"]["sqlite"]["scratch_db"]
    )
    config["workload"]["path"] = os.path.join(ROOT, config["workload"]["path"])
    config["benchmark"]["output_csv"] = os.path.join(
        ROOT, config["benchmark"]["output_csv"]
    )

    source_db = config["sandbox"]["sqlite"]["source_db"]
    if not os.path.exists(source_db):
        print(f"{source_db} not found -- generating synthetic sandbox data first...")
        sys.path.insert(0, os.path.join(ROOT, "data"))
        import generate_data

        generate_data.build()

    workload = load_workload(config["workload"]["path"])

    if config["llm"]["mock_mode"]:
        llm_mode_str = "MOCK (heuristic placeholder)"
    elif config["llm"].get("provider") == "groq":
        llm_mode_str = f"REAL (Groq API, model={config['llm'].get('groq_model', 'openai/gpt-oss-120b')})"
    elif config["llm"].get("provider") == "gemini":
        llm_mode_str = f"REAL (Gemini API, model={config['llm'].get('gemini_model', 'gemini-3.6-flash')})"
    else:
        llm_mode_str = f"REAL (Anthropic API, model={config['llm']['model']})"
    print(f"\nLLM mode: {llm_mode_str}")
    print(f"Sandbox backend: {config['sandbox']['backend']}")
    print(
        f"Workload: {len(workload.queries)} queries, total weight={workload.total_weight()}\n")
    print("Running system + baselines through the verification loop (this executes "
          "every candidate for real -- it will take a little while)...\n")

    reports = run_benchmark(workload, config)

    write_csv(reports, config["benchmark"]["output_csv"])

    table_rows = []
    for r in reports:
        s = r.summary
        combined_str = (
            f"{r.combined_effect.combined_speedup:.3f}x"
            if r.combined_effect else "n/a"
        )
        table_rows.append(
            [
                r.system_name,
                f"{r.baseline_seconds:.4f}s",
                s["num_proposed"],
                s["num_accepted"],
                s["num_rejected"],
                f"{s['best_speedup']:.3f}x",
                f"{s['avg_accepted_speedup']:.3f}x",
                combined_str,
                f"{s['total_index_storage_mb']:.2f} MB",
            ]
        )

    print(
        tabulate(
            table_rows,
            headers=[
                "System",
                "Baseline runtime",
                "Proposed",
                "Accepted",
                "Rejected",
                "Best speedup (per-candidate)",
                "Avg accepted (per-candidate)",
                "Combined (full accepted set)",
                "Index storage",
            ],
            tablefmt="github",
        )
    )
    print(
        "\nNote: 'per-candidate' columns test each accepted candidate in "
        "isolation and are diluted by workload weighting when a candidate "
        "only helps some queries. 'Combined' applies every accepted "
        "candidate together and measures the real effect on the full "
        "weighted workload -- this is the number that answers 'if I do "
        "everything this system recommends, how much faster is my "
        "workload?'"
    )

    print("\nAccepted candidates by system:\n")
    for r in reports:
        print(f"### {r.system_name}")
        if not r.accepted:
            print("  (none accepted)")
        for sc in r.accepted:
            print(f"  - {candidate_label(sc)}  ->  {sc.reason}")
        print()

    print(f"Full results written to {config['benchmark']['output_csv']}")


if __name__ == "__main__":
    main()
