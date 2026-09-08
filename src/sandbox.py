"""
Sandbox lifecycle helpers built on top of a SandboxBackend: applying a
candidate, running the full workload and getting per-query + aggregate
timing, and reverting. Kept separate from verifier.py so it can also be
used directly by baselines/benchmark.py for a like-for-like comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .backends.base import SandboxBackend
from .candidates import IndexCandidate, ViewCandidate, RewriteCandidate
from .workload import Workload


@dataclass
class WorkloadTiming:
    per_query_seconds: Dict[str, float]
    weighted_total_seconds: float


def run_workload(
    backend: SandboxBackend,
    workload: Workload,
    repeat: int = 5,
    warmup: int = 1,
    query_overrides: Dict[str, str] | None = None,
) -> WorkloadTiming:
    """Times every query in the workload. `query_overrides` lets a rewrite
    candidate substitute rewritten SQL for a specific query name without
    mutating the original Workload object."""
    query_overrides = query_overrides or {}
    per_query: Dict[str, float] = {}
    weighted_total = 0.0
    total_weight = workload.total_weight() or 1.0

    for q in workload.queries:
        sql = query_overrides.get(q.name, q.sql)
        t = backend.time_query(sql, repeat=repeat, warmup=warmup)
        per_query[q.name] = t
        weighted_total += t * (q.weight / total_weight)

    return WorkloadTiming(per_query_seconds=per_query, weighted_total_seconds=weighted_total)


def apply_index(backend: SandboxBackend, candidate: IndexCandidate) -> None:
    backend.execute_ddl(candidate.to_ddl())


def revert_index(backend: SandboxBackend, candidate: IndexCandidate) -> None:
    backend.execute_ddl(candidate.to_drop_ddl())


def apply_view(backend: SandboxBackend, candidate: ViewCandidate, materialized: bool = True) -> None:
    backend.execute_ddl(candidate.to_ddl(materialized=materialized))


def revert_view(backend: SandboxBackend, candidate: ViewCandidate, materialized: bool = True) -> None:
    backend.execute_ddl(candidate.to_drop_ddl(materialized=materialized))


def result_set_fingerprint(rows: List[tuple]) -> str:
    """Order-insensitive fingerprint of a query result set, used to verify
    a rewrite is actually equivalent to the original query before it's
    ever considered for the speedup it measured."""
    import hashlib

    normalized = sorted(tuple(str(v) for v in row) for row in rows)
    h = hashlib.sha256()
    for row in normalized:
        h.update("|".join(row).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()
