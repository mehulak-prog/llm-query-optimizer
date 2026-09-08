"""
Runs the full LLM-as-planner-assistant system AND the baseline advisors
on the identical workload/schema/sandbox, verifying every candidate from
every system through the exact same verifier + reward pipeline. This is
the apples-to-apples comparison the paper's evaluation section needs:
nobody's candidates get a numeric advantage from being measured
differently.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from typing import List, Dict

from .optimizer_loop import run_loop, _make_backend_factory, LoopResult
from .workload import Workload
from .schema import extract_schema
from .stats import collect_stats
from .verifier import measure_baseline, verify_index_candidate_with_trials
from .reward import score, ScoredCandidate, enforce_shared_storage_budget
from .baselines import explain_advisor, rule_based
from .metrics import summarize
from .combined_effect import measure_combined_effect, CombinedEffect


@dataclass
class BenchmarkReport:
    system_name: str
    baseline_seconds: float
    accepted: List[ScoredCandidate]
    rejected: List[ScoredCandidate]
    combined_effect: CombinedEffect | None = None

    @property
    def summary(self) -> dict:
        return summarize(self.accepted, self.rejected)


def _run_baseline_system(name: str, config: dict, workload: Workload) -> BenchmarkReport:
    backend_factory = _make_backend_factory(config)
    workload_cfg = config["workload"]
    repeat, warmup = workload_cfg["repeat_runs"], workload_cfg["warmup_runs"]
    max_storage_bytes = config["candidates"]["max_index_storage_mb"] * 1024 * 1024

    intro_backend = backend_factory()
    try:
        schema = extract_schema(intro_backend)
        stats = collect_stats(intro_backend)
    finally:
        intro_backend.close()

    baseline_seconds = measure_baseline(
        backend_factory, workload, repeat, warmup)

    if name == "explain_advisor":
        probe_backend = backend_factory()
        try:
            candidates = explain_advisor.propose(
                probe_backend, schema, stats, workload)
        finally:
            probe_backend.close()
    elif name == "rule_based":
        candidates = rule_based.propose(schema, workload)
    else:
        raise ValueError(f"Unknown baseline: {name}")

    accepted, rejected = [], []
    for idx_candidate in candidates.indexes:
        vr = verify_index_candidate_with_trials(
            backend_factory, idx_candidate, workload, baseline_seconds, repeat, warmup, max_storage_bytes,
            write_bench_rows=config["write_bench"]["rows"],
            write_trials=config["write_bench"].get("trials", 3),
        )
        sc = score(vr, config)
        (accepted if sc.accepted else rejected).append(sc)

    budget_bytes = config["candidates"]["shared_storage_budget_mb"] * 1024 * 1024
    accepted, rejected = enforce_shared_storage_budget(
        accepted, rejected, budget_bytes)

    return BenchmarkReport(
        system_name=name,
        baseline_seconds=baseline_seconds,
        accepted=accepted,
        rejected=rejected,
    )


def run_benchmark(workload: Workload, config: dict) -> List[BenchmarkReport]:
    reports: List[BenchmarkReport] = []

    loop_result: LoopResult = run_loop(workload, config)
    reports.append(
        BenchmarkReport(
            system_name="llm_planner (this system)",
            baseline_seconds=loop_result.baseline_weighted_seconds,
            accepted=loop_result.accepted,
            rejected=loop_result.rejected,
        )
    )

    for baseline_name in config["benchmark"]["baselines"]:
        reports.append(_run_baseline_system(baseline_name, config, workload))

    # Per-candidate speedups above answer "is this one candidate worth
    # recommending?" -- deliberately diluted by workload weighting when a
    # candidate only helps a subset of queries. That's NOT the same as
    # "how much faster is my workload if I apply everything this system
    # is telling me to do?", which needs every accepted candidate applied
    # together in one sandbox. Compute that here for every system so it's
    # never accidentally skipped.
    backend_factory = _make_backend_factory(config)
    workload_cfg = config["workload"]
    repeat, warmup = workload_cfg["repeat_runs"], workload_cfg["warmup_runs"]

    for r in reports:
        if not r.accepted:
            continue
        r.combined_effect = measure_combined_effect(
            backend_factory, r.accepted, workload, r.baseline_seconds, repeat, warmup,
        )

    return reports


def write_csv(reports: List[BenchmarkReport], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "system",
                "num_proposed",
                "num_accepted",
                "num_rejected",
                "best_speedup",
                "avg_accepted_speedup",
                "total_index_storage_mb",
                "combined_workload_speedup",
            ]
        )
        for r in reports:
            s = r.summary
            combined = (
                f"{r.combined_effect.combined_speedup:.3f}"
                if r.combined_effect else "n/a"
            )
            writer.writerow(
                [
                    r.system_name,
                    s["num_proposed"],
                    s["num_accepted"],
                    s["num_rejected"],
                    f"{s['best_speedup']:.3f}",
                    f"{s['avg_accepted_speedup']:.3f}",
                    f"{s['total_index_storage_mb']:.3f}",
                    combined,
                ]
            )
