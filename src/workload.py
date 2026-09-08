"""
Workload loading: a "workload" is a set of representative queries with
relative frequency weights, the thing a real optimizer would get from
pg_stat_statements, a slow-query log, or an app's query logger.

Replace data/workload.json with a real capture. The format is intentionally
simple JSON so exporting from pg_stat_statements is a short script:

    SELECT query, calls FROM pg_stat_statements ORDER BY calls DESC LIMIT 50;

maps directly to {"queries": [{"sql": ..., "weight": calls, "name": ...}]}.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List


@dataclass
class WorkloadQuery:
    name: str
    sql: str
    weight: float = 1.0   # relative frequency; used to weight aggregate runtime


@dataclass
class Workload:
    queries: List[WorkloadQuery]

    def total_weight(self) -> float:
        return sum(q.weight for q in self.queries)

    def to_llm_text(self) -> str:
        lines = []
        for q in self.queries:
            lines.append(f"-- {q.name} (relative frequency weight={q.weight})")
            lines.append(q.sql.strip())
            lines.append("")
        return "\n".join(lines)


def load_workload(path: str) -> Workload:
    with open(path) as f:
        raw = json.load(f)
    queries = [
        WorkloadQuery(
            name=q.get("name", f"q{i}"),
            sql=q["sql"],
            weight=q.get("weight", 1.0),
        )
        for i, q in enumerate(raw["queries"])
    ]
    return Workload(queries=queries)
