"""
Sample statistics collection: cardinality estimates, null fractions, and
lightweight histograms per column. These are what a real query planner
uses internally (Postgres's pg_stats, MySQL's InnoDB persistent stats),
and what the LLM needs to reason about selectivity -- without them it can
only pattern-match on column names, which is exactly the failure mode
this system is trying to avoid.

For a real deployment, replace `collect_stats` with a query against the
DB's native stats tables (pg_stats, information_schema, SHOW STATS, etc.)
rather than sampling -- it's cheaper and more accurate. Sampling is kept
here as a backend-agnostic fallback that works identically on SQLite and
Postgres.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any
import json


@dataclass
class ColumnStats:
    column: str
    distinct_count_estimate: int
    null_fraction: float
    min_value: Any = None
    max_value: Any = None
    most_common_values: List[Any] = field(default_factory=list)


@dataclass
class TableStats:
    table: str
    row_count: int
    columns: Dict[str, ColumnStats] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "table": self.table,
            "row_count": self.row_count,
            "columns": {k: asdict(v) for k, v in self.columns.items()},
        }


def collect_stats(backend, sample_size: int = 5000) -> Dict[str, TableStats]:
    """Collects per-table, per-column statistics via the backend interface.

    sample_size caps how many rows are pulled for MCV/distinct estimation
    on large tables, to keep this cheap on real production-scale sandboxes.
    """
    stats: Dict[str, TableStats] = {}
    for table_name in backend.list_tables():
        row_count = backend.get_row_count(table_name)
        col_stats: Dict[str, ColumnStats] = {}
        for col in backend.get_columns(table_name):
            col_name = col["name"]
            raw = backend.get_column_stats(table_name, col_name, sample_size)
            col_stats[col_name] = ColumnStats(
                column=col_name,
                distinct_count_estimate=raw["distinct_count_estimate"],
                null_fraction=raw["null_fraction"],
                min_value=raw.get("min_value"),
                max_value=raw.get("max_value"),
                most_common_values=raw.get("most_common_values", []),
            )
        stats[table_name] = TableStats(
            table=table_name, row_count=row_count, columns=col_stats
        )
    return stats


def stats_to_llm_text(stats: Dict[str, TableStats]) -> str:
    lines = []
    for t in stats.values():
        lines.append(f"TABLE {t.table} (row_count={t.row_count}):")
        for c in t.columns.values():
            selectivity = (
                f"{c.distinct_count_estimate}/{t.row_count} distinct"
                if t.row_count
                else "n/a"
            )
            mcv = f" MCVs={c.most_common_values[:5]}" if c.most_common_values else ""
            lines.append(
                f"  - {c.column}: {selectivity}, "
                f"null_fraction={c.null_fraction:.3f}, "
                f"range=[{c.min_value}, {c.max_value}]{mcv}"
            )
        lines.append("")
    return "\n".join(lines)


def stats_to_json(stats: Dict[str, TableStats]) -> str:
    return json.dumps({k: v.to_dict() for k, v in stats.items()}, indent=2, default=str)
