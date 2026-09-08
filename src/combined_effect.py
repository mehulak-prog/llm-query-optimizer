"""
Measures the effect of applying ALL accepted candidates from a system
together, in one sandbox, against the full weighted workload -- as
opposed to verifier.py's per-candidate measurements, which apply and
test each candidate in isolation on its own fresh clone.

Why this exists: per-candidate speedups (see reward.py / verifier.py)
answer "is this one candidate worth recommending on its own?" and are
deliberately diluted by workload weighting when a candidate only helps
a subset of queries -- that's correct for that question. But it is NOT
the number a DBA (or a paper reviewer) actually wants first: "if I do
everything this system tells me to do, how much faster is my real
workload?" That requires applying the whole accepted set at once,
which nothing else in this codebase currently does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Callable, Dict

from .backends.base import SandboxBackend
from .workload import Workload
from .sandbox import run_workload, apply_index, apply_view
from .reward import ScoredCandidate


@dataclass
class CombinedEffect:
    baseline_weighted_seconds: float
    combined_weighted_seconds: float
    combined_speedup: float
    num_indexes_applied: int
    num_views_applied: int
    num_rewrites_applied: int
    total_index_storage_bytes: int
    per_query_before: Dict[str, float]
    per_query_after: Dict[str, float]


def measure_combined_effect(
    backend_factory: Callable[[], SandboxBackend],
    accepted: List[ScoredCandidate],
    workload: Workload,
    baseline_weighted_seconds: float,
    repeat: int,
    warmup: int,
) -> CombinedEffect:
    """Applies every accepted candidate to a single fresh sandbox clone,
    then runs the full workload once against that combined state.

    Indexes and materialized views are applied as real DDL. Rewrite
    candidates can't be "applied" as DDL -- they substitute query text
    at run time -- so they're collected into query_overrides and passed
    into run_workload alongside the DDL-modified schema.
    """
    backend = backend_factory()
    query_overrides: Dict[str, str] = {}
    num_indexes = num_views = num_rewrites = 0
    total_storage = 0
    applied_identities: set = set()  # (kind, table, cols) / (kind, view_name) / etc.

    def _identity(candidate, kind: str) -> tuple:
        if kind == "index":
            return ("index", candidate.table, tuple(candidate.columns))
        if kind == "materialized_view" or kind == "view":
            return ("materialized_view", candidate.name)
        if kind == "rewrite":
            return ("rewrite", candidate.target_query_name)
        return (kind, id(candidate))

    try:
        # Baseline (before any candidate applied) on THIS clone, so the
        # "before" comparison uses the exact same sandbox state/warmup
        # conditions as the "after" measurement -- avoids conflating
        # cross-clone variance with the candidates' actual effect.
        before_timing = run_workload(backend, workload, repeat=repeat, warmup=warmup)

        for sc in accepted:
            candidate = sc.result.candidate
            kind = sc.result.candidate_kind
            ident = _identity(candidate, kind)

            # Defensive: two accepted candidates can resolve to identical
            # DDL (same table+columns / view name / rewrite target),
            # e.g. proposed again in a later round, or by coincidence
            # from an LLM at temperature > 0. optimizer_loop.py dedupes
            # this for the main system already, but baseline advisors
            # (rule_based, explain_advisor) don't go through that path,
            # so this check is what actually prevents a crash for them.
            if ident in applied_identities:
                continue
            applied_identities.add(ident)

            if kind == "index":
                apply_index(backend, candidate)
                num_indexes += 1
                if sc.result.storage_bytes:
                    total_storage += sc.result.storage_bytes

            elif kind == "materialized_view" or kind == "view":
                apply_view(backend, candidate, materialized=True)
                num_views += 1
                for qname in getattr(candidate, "replaces_query_names", []) or []:
                    # If the LLM told us this view replaces a specific
                    # query, honor that in the combined run too.
                    pass  # left as-is unless a rewrite explicitly overrides it

            elif kind == "rewrite":
                query_overrides[candidate.target_query_name] = candidate.rewritten_sql
                num_rewrites += 1

        after_timing = run_workload(
            backend, workload, repeat=repeat, warmup=warmup,
            query_overrides=query_overrides,
        )

        combined_speedup = (
            before_timing.weighted_total_seconds / after_timing.weighted_total_seconds
            if after_timing.weighted_total_seconds else 1.0
        )

        return CombinedEffect(
            baseline_weighted_seconds=baseline_weighted_seconds,
            combined_weighted_seconds=after_timing.weighted_total_seconds,
            combined_speedup=combined_speedup,
            num_indexes_applied=num_indexes,
            num_views_applied=num_views,
            num_rewrites_applied=num_rewrites,
            total_index_storage_bytes=total_storage,
            per_query_before=before_timing.per_query_seconds,
            per_query_after=after_timing.per_query_seconds,
        )
    finally:
        backend.close()
