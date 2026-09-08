"""
A simplified re-implementation of the "read EXPLAIN, find sequential
scans on large tables, propose an index on the filtering column"
strategy used by Postgres-EXPLAIN-based advisors and the classic
AutoAdmin what-if-index-selection line of work (Chaudhuri & Narasayya).

This is a genuine baseline, not a strawman: it's what most rule-based
advisor tools actually do under the hood, minus AutoAdmin's proper
cost-based what-if optimizer (which requires deep planner hooks this
scaffold doesn't have). The gap between this baseline and the real
AutoAdmin system is itself worth stating precisely in the paper's
related-work section -- see docs/research_framing.md.
"""

from __future__ import annotations

import re
from typing import List

from ..schema import DatabaseSchema
from ..stats import TableStats
from ..workload import Workload
from ..candidates import CandidateSet, IndexCandidate
from ..backends.base import SandboxBackend


SCAN_KEYWORDS = ("SCAN", "Seq Scan")  # SQLite: "SCAN table"; Postgres: "Seq Scan on table"
LARGE_TABLE_ROW_THRESHOLD = 1000


def propose(
    backend: SandboxBackend,
    schema: DatabaseSchema,
    stats: dict,
    workload: Workload,
    max_indexes: int = 6,
) -> CandidateSet:
    existing_indexed_cols = set()
    for table_name, table in schema.tables.items():
        for idx in table.existing_indexes:
            for c in idx.columns:
                existing_indexed_cols.add((table_name, c))
        for col in table.columns:
            if col.is_primary_key:
                existing_indexed_cols.add((table_name, col.name))

    scan_hits: dict[tuple[str, str], int] = {}

    for q in workload.queries:
        plan = backend.explain(q.sql)
        scanned_tables = _tables_from_plan(plan, schema.tables.keys())
        for table_name in scanned_tables:
            table_stats: TableStats = stats.get(table_name)
            if not table_stats or table_stats.row_count < LARGE_TABLE_ROW_THRESHOLD:
                continue
            # naive: any column of this table referenced with a comparison
            # operator in the query text is a candidate
            for col in schema.tables[table_name].columns:
                pattern = rf"\b{re.escape(col.name.lower())}\b\s*(=|>|<|>=|<=|in|between)"
                if re.search(pattern, q.sql.lower()):
                    key = (table_name, col.name)
                    if key in existing_indexed_cols:
                        continue
                    scan_hits[key] = scan_hits.get(key, 0) + int(q.weight)

    ranked = sorted(scan_hits.items(), key=lambda kv: kv[1], reverse=True)
    result = CandidateSet()
    for (table_name, col_name), weight in ranked[:max_indexes]:
        result.indexes.append(
            IndexCandidate(
                table=table_name,
                columns=[col_name],
                rationale=(
                    f"[explain_advisor baseline] query plan showed a full scan on "
                    f"{table_name} (>{LARGE_TABLE_ROW_THRESHOLD} rows) with a filter "
                    f"on {col_name}; weighted hit count={weight}."
                ),
            )
        )
    return result


def _tables_from_plan(plan_text: str, known_tables) -> List[str]:
    hits = []
    for table in known_tables:
        for kw in SCAN_KEYWORDS:
            if f"{kw} {table}".lower() in plan_text.lower() or f"{kw} on {table}".lower() in plan_text.lower():
                hits.append(table)
                break
    return hits
