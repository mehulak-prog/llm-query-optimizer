"""
SQLite sandbox backend. Zero external dependencies (sqlite3 is stdlib),
which is why it's the default in config.yaml -- you can validate the
whole pipeline before standing up a real Postgres sandbox.

Caveats you should know about before citing SQLite results in a paper:
- SQLite's query optimizer is much simpler than Postgres/MySQL's, so
  index-selection wins here are a weaker signal than on a real OLTP/OLAP
  engine. Treat this backend as a correctness/plumbing testbed, not as
  the evaluation backend for the paper.
- SQLite has no materialized views; `execute_ddl` on a MATERIALIZED VIEW
  candidate is downgraded to a regular VIEW with a comment noting this.
"""

from __future__ import annotations

import shutil
import sqlite3
import os
from typing import List, Dict, Any

from .base import SandboxBackend


class SQLiteBackend(SandboxBackend):
    def __init__(self, source_db: str, scratch_db: str):
        self.source_db = source_db
        self.scratch_db = scratch_db
        self.conn: sqlite3.Connection | None = None

    def clone_from_source(self) -> None:
        if self.conn is not None:
            self.conn.close()
        if os.path.exists(self.scratch_db):
            os.remove(self.scratch_db)
        shutil.copy(self.source_db, self.scratch_db)
        self.conn = sqlite3.connect(self.scratch_db)
        self.conn.execute("PRAGMA foreign_keys = ON;")

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None
        if os.path.exists(self.scratch_db):
            os.remove(self.scratch_db)

    def list_tables(self) -> List[str]:
        cur = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        return [r[0] for r in cur.fetchall()]

    def get_columns(self, table: str) -> List[Dict[str, Any]]:
        cur = self.conn.execute(f"PRAGMA table_info({table})")
        out = []
        for cid, name, dtype, notnull, dflt, pk in cur.fetchall():
            out.append(
                {
                    "name": name,
                    "data_type": dtype or "UNKNOWN",
                    "nullable": not bool(notnull),
                    "is_primary_key": bool(pk),
                }
            )
        return out

    def get_foreign_keys(self, table: str) -> List[Dict[str, Any]]:
        cur = self.conn.execute(f"PRAGMA foreign_key_list({table})")
        out = []
        for row in cur.fetchall():
            # id, seq, table, from, to, on_update, on_delete, match
            out.append(
                {
                    "column": row[3],
                    "references_table": row[2],
                    "references_column": row[4],
                }
            )
        return out

    def get_indexes(self, table: str) -> List[Dict[str, Any]]:
        cur = self.conn.execute(f"PRAGMA index_list({table})")
        out = []
        for row in cur.fetchall():
            # seq, name, unique, origin, partial
            idx_name = row[1]
            unique = bool(row[2])
            col_cur = self.conn.execute(f"PRAGMA index_info({idx_name})")
            cols = [c[2] for c in col_cur.fetchall()]
            out.append({"name": idx_name, "columns": cols, "unique": unique})
        return out

    def get_row_count(self, table: str) -> int:
        cur = self.conn.execute(f"SELECT COUNT(*) FROM {table}")
        return cur.fetchone()[0]

    def get_column_stats(self, table: str, column: str, sample_size: int) -> Dict[str, Any]:
        row_count = self.get_row_count(table)
        limit = min(sample_size, row_count) if row_count else 0

        distinct = self.conn.execute(
            f"SELECT COUNT(DISTINCT {column}) FROM "
            f"(SELECT {column} FROM {table} LIMIT {max(limit, 1)})"
        ).fetchone()[0]

        nulls = self.conn.execute(
            f"SELECT COUNT(*) FROM (SELECT {column} FROM {table} LIMIT {max(limit,1)}) "
            f"WHERE {column} IS NULL"
        ).fetchone()[0]
        null_fraction = (nulls / limit) if limit else 0.0

        min_val, max_val = self.conn.execute(
            f"SELECT MIN({column}), MAX({column}) FROM {table}"
        ).fetchone()

        mcv_rows = self.conn.execute(
            f"SELECT {column}, COUNT(*) c FROM "
            f"(SELECT {column} FROM {table} LIMIT {max(limit,1)}) "
            f"GROUP BY {column} ORDER BY c DESC LIMIT 5"
        ).fetchall()
        mcv = [r[0] for r in mcv_rows]

        return {
            "distinct_count_estimate": distinct,
            "null_fraction": null_fraction,
            "min_value": min_val,
            "max_value": max_val,
            "most_common_values": mcv,
        }

    def execute_ddl(self, ddl: str) -> None:
        stmt = ddl.strip()
        if stmt.upper().startswith("CREATE MATERIALIZED VIEW"):
            stmt = "CREATE VIEW" + stmt[len("CREATE MATERIALIZED VIEW"):]
        elif stmt.upper().startswith("DROP MATERIALIZED VIEW"):
            stmt = "DROP VIEW" + stmt[len("DROP MATERIALIZED VIEW"):]
        self.conn.execute(stmt)
        self.conn.commit()

    def execute_query(self, sql: str) -> List[tuple]:
        cur = self.conn.execute(sql)
        return cur.fetchall()

    def explain(self, sql: str) -> str:
        cur = self.conn.execute(f"EXPLAIN QUERY PLAN {sql}")
        return "\n".join(str(row) for row in cur.fetchall())

    def estimate_index_storage_bytes(self, table: str, columns: List[str]) -> int:
        # Crude estimate: row_count * (avg 8 bytes per indexed column + 8 byte rowid pointer)
        row_count = self.get_row_count(table)
        return row_count * (8 * len(columns) + 8)
