"""
The closed loop: LLM proposes candidates -> sandbox verifies them against
real execution -> results are scored -> accepted candidates are kept,
rejected ones (with their measured reason) are fed back to the LLM for a
bounded number of refinement rounds.

This is the module implementing the paper's central claim: the LLM is a
candidate generator, not an oracle, and the system's quality comes from
the verification loop, not from the model's confidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Callable

from .backends import get_backend
from .backends.base import SandboxBackend
from .schema import extract_schema, DatabaseSchema
from .stats import collect_stats
from .workload import Workload
from .llm_client import LLMPlanner
from .candidates import CandidateSet
from .verifier import (
    measure_baseline,
    verify_index_candidate_with_trials,
    verify_view_candidate,
    verify_rewrite_candidate,
)
from .reward import score, ScoredCandidate, enforce_shared_storage_budget


@dataclass
class LoopResult:
    rounds_run: int
    baseline_weighted_seconds: float
    accepted: List[ScoredCandidate] = field(default_factory=list)
    rejected: List[ScoredCandidate] = field(default_factory=list)


def _make_backend_factory(config: dict) -> Callable[[], SandboxBackend]:
    def factory() -> SandboxBackend:
        backend = get_backend(config["sandbox"])
        backend.clone_from_source()
        return backend

    return factory


def _format_prior_results(scored: List[ScoredCandidate]) -> str:
    lines = []
    for sc in scored:
        r = sc.result
        if r.candidate_kind == "index":
            c = r.candidate
            desc = f"INDEX on {c.table}({', '.join(c.columns)})"
        elif r.candidate_kind == "materialized_view":
            c = r.candidate
            desc = f"MATERIALIZED VIEW {c.name}"
        else:
            c = r.candidate
            desc = f"REWRITE of {c.target_query_name}"
        status = "ACCEPTED" if sc.accepted else "REJECTED"
        lines.append(
            f"- {desc}: {status} (speedup={r.speedup:.3f}x). {sc.reason}"
        )
    return "\n".join(lines)


def _candidate_identity(candidate) -> tuple:
    """A hashable identity for deduping candidates across rounds. Two
    candidates with the same identity would generate identical DDL (same
    index name, same view name, same rewrite target) -- verifying and
    accepting both wastes a full round's work and, worse, crashes later
    when the combined-effect measurement tries to apply the same DDL
    twice to one sandbox. This isn't hypothetical: the mock planner
    ignores prior_results_text and will re-propose an already-accepted
    index verbatim in the next round; a real LLM at temperature > 0 can
    do the same by coincidence.
    """
    kind = getattr(candidate, "kind", None)
    if kind == "index":
        return ("index", candidate.table, tuple(candidate.columns))
    if kind == "materialized_view":
        return ("materialized_view", candidate.name)
    if kind == "rewrite":
        return ("rewrite", candidate.target_query_name)
    return ("unknown", id(candidate))


def run_loop(workload: Workload, config: dict) -> LoopResult:
    backend_factory = _make_backend_factory(config)
    planner = LLMPlanner(config)

    workload_cfg = config["workload"]
    repeat = workload_cfg["repeat_runs"]
    warmup = workload_cfg["warmup_runs"]
    max_storage_bytes = config["candidates"]["max_index_storage_mb"] * 1024 * 1024

    # Extract schema + stats from a fresh sandbox clone (never touch the
    # source DB directly).
    intro_backend = backend_factory()
    try:
        schema = extract_schema(intro_backend)
        stats = collect_stats(intro_backend)
    finally:
        intro_backend.close()

    baseline_seconds = measure_baseline(
        backend_factory, workload, repeat, warmup)

    result = LoopResult(
        rounds_run=0, baseline_weighted_seconds=baseline_seconds)
    prior_results_text = None
    max_rounds = config["llm"].get(
        "max_refinement_rounds", 1) + 1  # +1 for the initial round
    seen_identities: set = set()

    for round_idx in range(max_rounds):
        candidates: CandidateSet = planner.propose(
            schema, stats, workload, prior_results_text=prior_results_text
        )
        result.rounds_run += 1

        if candidates.is_empty():
            break

        # Drop anything already verified in a prior round (same table+
        # columns / view name / rewrite target) before spending any
        # verification budget on it.
        new_indexes = [c for c in candidates.indexes if _candidate_identity(
            c) not in seen_identities]
        new_views = [c for c in candidates.views if _candidate_identity(
            c) not in seen_identities]
        new_rewrites = [c for c in candidates.rewrites if _candidate_identity(
            c) not in seen_identities]

        round_scored: List[ScoredCandidate] = []
        for idx_candidate in new_indexes[: config["candidates"]["max_indexes_per_round"]]:
            seen_identities.add(_candidate_identity(idx_candidate))
            vr = verify_index_candidate_with_trials(
                backend_factory, idx_candidate, workload, baseline_seconds,
                repeat, warmup, max_storage_bytes,
                write_bench_rows=config["write_bench"]["rows"],
                write_trials=config["write_bench"].get("trials", 3),
            )
            round_scored.append(score(vr, config))

        for view_candidate in new_views[: config["candidates"]["max_views_per_round"]]:
            seen_identities.add(_candidate_identity(view_candidate))
            vr = verify_view_candidate(
                backend_factory, view_candidate, workload, baseline_seconds, repeat, warmup
            )
            round_scored.append(score(vr, config))

        for rewrite_candidate in new_rewrites[: config["candidates"]["max_rewrites_per_round"]]:
            seen_identities.add(_candidate_identity(rewrite_candidate))
            vr = verify_rewrite_candidate(
                backend_factory, rewrite_candidate, workload, baseline_seconds, repeat, warmup
            )
            round_scored.append(score(vr, config))

        for sc in round_scored:
            (result.accepted if sc.accepted else result.rejected).append(sc)

        if round_idx < max_rounds - 1:
            prior_results_text = _format_prior_results(round_scored)

    budget_bytes = config["candidates"]["shared_storage_budget_mb"] * 1024 * 1024
    result.accepted, result.rejected = enforce_shared_storage_budget(
        result.accepted, result.rejected, budget_bytes
    )
    return result
