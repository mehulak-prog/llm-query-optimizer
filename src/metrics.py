"""Small formatting helpers shared by benchmark.py and experiments/."""

from __future__ import annotations

from typing import List
from .reward import ScoredCandidate


def summarize(accepted: List[ScoredCandidate], rejected: List[ScoredCandidate]) -> dict:
    total_storage = sum(
        (sc.result.storage_bytes or 0) for sc in accepted if sc.result.candidate_kind == "index"
    )
    best_speedup = max((sc.result.speedup for sc in accepted), default=1.0)
    avg_speedup = (
        sum(sc.result.speedup for sc in accepted) / len(accepted) if accepted else 1.0
    )
    return {
        "num_proposed": len(accepted) + len(rejected),
        "num_accepted": len(accepted),
        "num_rejected": len(rejected),
        "best_speedup": best_speedup,
        "avg_accepted_speedup": avg_speedup,
        "total_index_storage_mb": total_storage / (1024 * 1024),
    }


def candidate_label(sc: ScoredCandidate) -> str:
    r = sc.result
    if r.candidate_kind == "index":
        c = r.candidate
        return f"INDEX {c.table}({', '.join(c.columns)})"
    if r.candidate_kind == "materialized_view":
        c = r.candidate
        return f"VIEW {c.name}"
    c = r.candidate
    return f"REWRITE {c.target_query_name}"
