"""
A re-implementation of the flat-rule heuristics commercial tools like
EverSQL and pganalyze visibly apply (based on their public documentation
and blog posts on index advisor logic, not their proprietary source):

  1. Every foreign-key column without a supporting index gets flagged
     (a near-universal recommendation in these tools, independent of
     actual query frequency).
  2. Every column that appears in an ORDER BY without a supporting index
     gets flagged.
  3. Every column used in a JOIN ... ON without a supporting index gets
     flagged.

Deliberately workload-statistics-blind (these tools mostly are, unless
you pay for their query-log integration tier) -- that's the point of
including it: it's the thing a cost-blind, execution-blind rule engine
gets you, and the paper's evaluation should show where that helps and
where it wastes storage on indexes nothing actually queries often.
"""

from __future__ import annotations

import re

from ..schema import DatabaseSchema
from ..workload import Workload
from ..candidates import CandidateSet, IndexCandidate


def propose(schema: DatabaseSchema, workload: Workload, max_indexes: int = 6) -> CandidateSet:
    existing_indexed_cols = set()
    for table_name, table in schema.tables.items():
        for idx in table.existing_indexes:
            for c in idx.columns:
                existing_indexed_cols.add((table_name, c))
        for col in table.columns:
            if col.is_primary_key:
                existing_indexed_cols.add((table_name, col.name))

    flagged: dict[tuple[str, str], str] = {}

    # Rule 1: unindexed foreign keys
    for table_name, table in schema.tables.items():
        for fk in table.foreign_keys:
            key = (table_name, fk.column)
            if key not in existing_indexed_cols:
                flagged[key] = "unindexed foreign key"

    # Rules 2 & 3: ORDER BY / JOIN..ON columns without an index
    all_sql = "\n".join(q.sql.lower() for q in workload.queries)
    for table_name, table in schema.tables.items():
        for col in table.columns:
            key = (table_name, col.name)
            if key in existing_indexed_cols or key in flagged:
                continue
            order_pattern = rf"order by[^;]*\b{re.escape(col.name.lower())}\b"
            join_pattern = rf"\bon\b[^;]*\b{re.escape(col.name.lower())}\b"
            if re.search(order_pattern, all_sql):
                flagged[key] = "column used in ORDER BY without a supporting index"
            elif re.search(join_pattern, all_sql):
                flagged[key] = "column used in JOIN..ON without a supporting index"

    result = CandidateSet()
    for (table_name, col_name), rule in list(flagged.items())[:max_indexes]:
        result.indexes.append(
            IndexCandidate(
                table=table_name,
                columns=[col_name],
                rationale=f"[rule_based baseline] {rule}.",
            )
        )
    return result
