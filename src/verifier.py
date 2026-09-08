"""
The verification loop's core: for a given candidate, clone a fresh
sandbox, apply the candidate, measure real runtime, and (for rewrites)
check the result set is actually equivalent. This is the module that
turns "the LLM said this would help" into "we measured that this helps."

Every candidate gets its OWN fresh clone of the sandbox so trials never
interfere with each other (an index from trial 1 must not still be
present when trial 2 is measured).
"""

from __future__ import annotations

import logging
import statistics
import time
from dataclasses import dataclass, field
from typing import List, Optional, Union

from .backends.base import SandboxBackend
from .candidates import IndexCandidate, ViewCandidate, RewriteCandidate
from .workload import Workload
from .sandbox import run_workload, apply_index, apply_view, result_set_fingerprint


logger = logging.getLogger(__name__)
Candidate = Union[IndexCandidate, ViewCandidate, RewriteCandidate]


@dataclass
class VerificationResult:
    candidate: Candidate
    candidate_kind: str
    baseline_weighted_seconds: float
    candidate_weighted_seconds: float
    speedup: float                     # baseline / candidate ; >1 means faster
    storage_bytes: Optional[int] = None
    write_baseline_seconds: Optional[float] = None
    write_candidate_seconds: Optional[float] = None
    write_regression_trials: Optional[List[float]] = None
    correctness_ok: bool = True
    correctness_note: str = ""
    error: Optional[str] = None


def measure_baseline(
    backend_factory, workload: Workload, repeat: int, warmup: int
) -> float:
    """backend_factory: zero-arg callable returning a fresh, cloned backend."""
    backend = backend_factory()
    try:
        timing = run_workload(backend, workload, repeat=repeat, warmup=warmup)
        return timing.weighted_total_seconds
    finally:
        backend.close()


def measure_write_regression(
    backend,
    candidate: IndexCandidate,
    n_rows: int,
    repeat: int,
    warmup: int,
):
    """Times a synthetic insert/update/delete cycle against candidate.table
    BEFORE candidate's index exists. Caller applies the DDL and calls
    backend.time_write_batch again afterward for the candidate-side number."""

    # strip " ASC"/" DESC" suffix if present
    touch_column = candidate.columns[0].split()[0]

    columns_info = backend.get_columns(candidate.table)
    pk_columns = [c["name"] for c in columns_info if c["is_primary_key"]]
    fk_columns = {fk["column"]
                  for fk in backend.get_foreign_keys(candidate.table)}
    # Shifting a PK column that's also a foreign key invents a parent row
    # that doesn't exist (e.g. lineitem.l_orderkey -> orders), which
    # violates referential integrity. Shift a PK component that ISN'T
    # also a FK instead -- for lineitem that's l_linenumber, whose real
    # values are small (1-7), so offsetting far above the real max still
    # cleanly separates synthetic rows from real ones.
    shiftable_pk_columns = [c for c in pk_columns if c not in fk_columns]
    if not shiftable_pk_columns:
        raise ValueError(
            f"Cannot measure write regression for {candidate.table}: every "
            f"primary-key column ({pk_columns}) is also a foreign key, so no "
            f"PK component can be shifted without either violating a FK "
            f"constraint (if shifted) or the PK uniqueness constraint (if "
            f"left unchanged). This table needs a different write-synthesis "
            f"strategy than the offset-shift approach -- flag it rather than "
            f"guess at a fix."
        )
    pk_col = shiftable_pk_columns[0]

    stats = backend.get_column_stats(candidate.table, pk_col, sample_size=1)
    offset_base = int(stats["max_value"]) + 10_000_000
    baseline_seconds = backend.time_write_batch(
        candidate.table, pk_col, touch_column, n_rows, offset_base,
        repeat=repeat, warmup=warmup,
    )
    return baseline_seconds, pk_col, touch_column, offset_base


def verify_index_candidate(
    backend_factory,
    candidate: IndexCandidate,
    workload: Workload,
    baseline_seconds: float,
    repeat: int,
    warmup: int,
    max_storage_bytes: int,
    write_bench_rows: int = 500,
) -> VerificationResult:
    backend = backend_factory()
    try:
        storage_estimate = backend.estimate_index_storage_bytes(
            candidate.table, candidate.columns
        )
        if storage_estimate > max_storage_bytes:
            return VerificationResult(
                candidate=candidate,
                candidate_kind="index",
                baseline_weighted_seconds=baseline_seconds,
                candidate_weighted_seconds=baseline_seconds,
                speedup=1.0,
                storage_bytes=storage_estimate,
                error=(
                    f"Skipped: estimated storage {storage_estimate} bytes exceeds "
                    f"max_index_storage_mb budget."
                ),
            )

        # Warm the buffer cache with one untimed workload pass before
        # timing the write baseline. Without this, write_baseline_seconds
        # is measured cold (right after clone_from_source) while
        # write_candidate_seconds is measured warm (after the full timed
        # read workload below has already run) -- that cache-state
        # mismatch can swamp or invert the real index-maintenance cost.
        import time as _time
        _t0 = time.perf_counter()
        run_workload(backend, workload, repeat=repeat, warmup=warmup)
        logger.debug(
            "warm-up workload pass: %.1fs (%s on %s)",
            time.perf_counter() - _t0, candidate.columns, candidate.table,
        )

        write_baseline_seconds, pk_col, touch_column, offset_base = measure_write_regression(
            backend, candidate, write_bench_rows, repeat, warmup
        )

        try:
            apply_index(backend, candidate)
        except Exception as e:
            return VerificationResult(
                candidate=candidate,
                candidate_kind="index",
                baseline_weighted_seconds=baseline_seconds,
                candidate_weighted_seconds=baseline_seconds,
                speedup=1.0,
                storage_bytes=storage_estimate,
                error=f"DDL failed: {e}",
            )

        _t0 = time.perf_counter()
        timing = run_workload(backend, workload, repeat=repeat, warmup=warmup)
        logger.debug(
            "read-timing workload pass: %.1fs (%s on %s)",
            time.perf_counter() - _t0, candidate.columns, candidate.table,
        )
        speedup = baseline_seconds / \
            timing.weighted_total_seconds if timing.weighted_total_seconds else 1.0

        _t0 = time.perf_counter()

        write_candidate_seconds = backend.time_write_batch(
            candidate.table, pk_col, touch_column, write_bench_rows,
            offset_base + write_bench_rows * 10 * (repeat + warmup),
            repeat=repeat, warmup=warmup,
        )
        logger.debug(
            "write-candidate timing: %.1fs (%s on %s)",
            time.perf_counter() - _t0, candidate.columns, candidate.table,
        )

        return VerificationResult(
            candidate=candidate,
            candidate_kind="index",
            baseline_weighted_seconds=baseline_seconds,
            candidate_weighted_seconds=timing.weighted_total_seconds,
            speedup=speedup,
            storage_bytes=storage_estimate,
            write_baseline_seconds=write_baseline_seconds,
            write_candidate_seconds=write_candidate_seconds,
        )
    finally:
        backend.close()


def verify_index_candidate_with_trials(
    backend_factory,
    candidate,
    workload,
    baseline_seconds,
    repeat,
    warmup,
    max_storage_bytes,
    write_bench_rows: int = 500,
    write_trials: int = 3,
) -> VerificationResult:
    """Runs verify_index_candidate write_trials times, each on its own
    independent fresh clone, and returns the trial whose write regression
    is closest to the group's median -- confirmed empirically that
    single-clone write-path timing has real between-clone variance large
    enough to flip a candidate's apparent regression sign (see project
    notes), which a single trial can't distinguish from a genuine effect.
    Read-side speedup is NOT re-averaged here -- no similar variance was
    found on that side, and re-using the median-ratio trial's own
    speedup keeps every reported field internally consistent with one
    real measurement rather than mixing values across trials."""
    trials: List[VerificationResult] = []
    for _ in range(write_trials):
        vr = verify_index_candidate(
            backend_factory, candidate, workload, baseline_seconds,
            repeat, warmup, max_storage_bytes, write_bench_rows,
        )
        if vr.error:
            # a failed trial (DDL error, storage skip) surfaces immediately, not averaged away
            return vr
        trials.append(vr)

    ratios = [
        t.write_candidate_seconds / t.write_baseline_seconds
        for t in trials
        if t.write_baseline_seconds and t.write_candidate_seconds
    ]
    if not ratios:
        # no write data on any trial; shouldn't happen absent an error, but don't crash
        return trials[0]

    median_ratio = statistics.median(ratios)
    best = min(
        trials,
        key=lambda t: abs(
            (t.write_candidate_seconds / t.write_baseline_seconds) - median_ratio
        ) if t.write_baseline_seconds and t.write_candidate_seconds else float("inf"),
    )
    best.write_regression_trials = ratios
    return best


def verify_view_candidate(
    backend_factory,
    candidate: ViewCandidate,
    workload: Workload,
    baseline_seconds: float,
    repeat: int,
    warmup: int,
) -> VerificationResult:
    backend = backend_factory()
    try:
        try:
            apply_view(backend, candidate, materialized=True)
        except Exception as e:
            return VerificationResult(
                candidate=candidate,
                candidate_kind="materialized_view",
                baseline_weighted_seconds=baseline_seconds,
                candidate_weighted_seconds=baseline_seconds,
                speedup=1.0,
                error=f"DDL failed: {e}",
            )

        # Only the queries this view claims to replace are re-timed against
        # a version that selects from the view -- the LLM must express the
        # replacement as an override query, provided via
        # candidate.replaces_query_names + the view name. We approximate
        # here by re-running the full workload unchanged (the view existing
        # is enough to show whether the planner opportunistically uses it;
        # for engines without automatic query rewriting -- e.g. SQLite --
        # this will show no change and is an expected limitation, noted in
        # docs/research_framing.md).
        timing = run_workload(backend, workload, repeat=repeat, warmup=warmup)
        speedup = baseline_seconds / \
            timing.weighted_total_seconds if timing.weighted_total_seconds else 1.0
        return VerificationResult(
            candidate=candidate,
            candidate_kind="materialized_view",
            baseline_weighted_seconds=baseline_seconds,
            candidate_weighted_seconds=timing.weighted_total_seconds,
            speedup=speedup,
        )
    finally:
        backend.close()


def verify_rewrite_candidate(
    backend_factory,
    candidate: RewriteCandidate,
    workload: Workload,
    baseline_seconds: float,
    repeat: int,
    warmup: int,
) -> VerificationResult:
    backend = backend_factory()
    try:
        try:
            original_rows = backend.execute_query(candidate.original_sql)
            rewritten_rows = backend.execute_query(candidate.rewritten_sql)
        except Exception as e:
            return VerificationResult(
                candidate=candidate,
                candidate_kind="rewrite",
                baseline_weighted_seconds=baseline_seconds,
                candidate_weighted_seconds=baseline_seconds,
                speedup=1.0,
                correctness_ok=False,
                correctness_note=f"Execution failed: {e}",
                error=str(e),
            )

        fp_original = result_set_fingerprint(original_rows)
        fp_rewritten = result_set_fingerprint(rewritten_rows)
        correctness_ok = fp_original == fp_rewritten

        if not correctness_ok:
            return VerificationResult(
                candidate=candidate,
                candidate_kind="rewrite",
                baseline_weighted_seconds=baseline_seconds,
                candidate_weighted_seconds=baseline_seconds,
                speedup=1.0,
                correctness_ok=False,
                correctness_note=(
                    "REJECTED: rewritten query returns a different result set "
                    "than the original. This candidate is discarded regardless "
                    "of any speed improvement -- correctness is not negotiable."
                ),
            )

        timing = run_workload(
            backend,
            workload,
            repeat=repeat,
            warmup=warmup,
            query_overrides={
                candidate.target_query_name: candidate.rewritten_sql},
        )
        speedup = baseline_seconds / \
            timing.weighted_total_seconds if timing.weighted_total_seconds else 1.0
        return VerificationResult(
            candidate=candidate,
            candidate_kind="rewrite",
            baseline_weighted_seconds=baseline_seconds,
            candidate_weighted_seconds=timing.weighted_total_seconds,
            speedup=speedup,
            correctness_ok=True,
            correctness_note="Result sets match original query.",
        )
    finally:
        backend.close()
