"""
Postgres sandbox backend. This is the one you'll actually want for real
experiments -- it's the backend that makes "actual execution as reward"
mean something, since Postgres's cost-based optimizer, real index types
(B-tree, GIN, GiST, BRIN, hash), and MVCC write costs are the thing the
paper is claiming to help with.

REQUIRES a disposable Postgres instance you control. Never point
`template_db` at anything you can't afford to have dropped/rebuilt --
this backend issues `CREATE DATABASE ... TEMPLATE ...` and
`DROP DATABASE` to get a cheap, isolated per-trial clone.

Setup sketch (fill in for your environment):
    docker run -d --name pg-sandbox -e POSTGRES_PASSWORD=postgres -p 5432:5432 postgres:16
    psql -h localhost -U postgres -c "CREATE DATABASE sandbox_template;"
    psql -h localhost -U postgres -d sandbox_template -f data/schema.sql
    python data/generate_data.py --backend postgres

This module is written and believed correct but has not been exercised
against a live server in this environment (no outbound DB access here).
Treat the first real run as a smoke test: check `explain()` output looks
like real Postgres EXPLAIN, and that clone/drop round-trips cleanly.
"""

from __future__ import annotations

from typing import List, Dict, Any
import re

try:
    import psycopg2
    import psycopg2.extras
except ImportError:  # pragma: no cover - only needed when backend=postgres
    psycopg2 = None

from .base import SandboxBackend


class PostgresBackend(SandboxBackend):
    def __init__(self, dsn: str, template_db: str, scratch_db: str = "sandbox_trial"):
        if psycopg2 is None:
            raise ImportError(
                "psycopg2-binary is required for the Postgres backend. "
                "pip install psycopg2-binary"
            )
        self.dsn = dsn
        self.template_db = template_db
        self.scratch_db = scratch_db
        self.conn = None
        self._admin_conn = None

    def _admin_connect(self, dbname: str = "postgres"):
        params = dict(kv.split("=", 1) for kv in self.dsn.split())
        params["dbname"] = dbname
        conn = psycopg2.connect(**params)
        conn.autocommit = True
        return conn

    def clone_from_source(self) -> None:
        if self.conn is not None:
            self.conn.close()
        admin = self._admin_connect()
        with admin.cursor() as cur:
            cur.execute(
                f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                f"WHERE datname = %s;",
                (self.scratch_db,),
            )
            cur.execute(f"DROP DATABASE IF EXISTS {self.scratch_db};")
            cur.execute(
                f"CREATE DATABASE {self.scratch_db} TEMPLATE {self.template_db};"
            )
        admin.close()

        params = dict(kv.split("=", 1) for kv in self.dsn.split())
        params["dbname"] = self.scratch_db
        self.conn = psycopg2.connect(**params)
        self.conn.autocommit = True

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None
        admin = self._admin_connect()
        with admin.cursor() as cur:
            cur.execute(
                f"SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                f"WHERE datname = %s;",
                (self.scratch_db,),
            )
            cur.execute(f"DROP DATABASE IF EXISTS {self.scratch_db};")
        admin.close()

    def list_tables(self) -> List[str]:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema='public' AND table_type='BASE TABLE';"
            )
            return [r[0] for r in cur.fetchall()]

    def get_columns(self, table: str) -> List[Dict[str, Any]]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.column_name, c.data_type, c.is_nullable,
                       COALESCE(pk.is_pk, false) AS is_pk
                FROM information_schema.columns c
                LEFT JOIN (
                    SELECT kcu.column_name, true AS is_pk
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu
                      ON tc.constraint_name = kcu.constraint_name
                    WHERE tc.table_name = %s AND tc.constraint_type = 'PRIMARY KEY'
                ) pk ON pk.column_name = c.column_name
                WHERE c.table_name = %s
                ORDER BY c.ordinal_position;
                """,
                (table, table),
            )
            return [
                {
                    "name": r[0],
                    "data_type": r[1],
                    "nullable": r[2] == "YES",
                    "is_primary_key": bool(r[3]),
                }
                for r in cur.fetchall()
            ]

    def get_foreign_keys(self, table: str) -> List[Dict[str, Any]]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT kcu.column_name, ccu.table_name AS foreign_table,
                       ccu.column_name AS foreign_column
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                  ON tc.constraint_name = kcu.constraint_name
                JOIN information_schema.constraint_column_usage ccu
                  ON tc.constraint_name = ccu.constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_name = %s;
                """,
                (table,),
            )
            return [
                {"column": r[0], "references_table": r[1],
                    "references_column": r[2]}
                for r in cur.fetchall()
            ]

    def get_indexes(self, table: str) -> List[Dict[str, Any]]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT i.relname AS index_name, ix.indisunique,
                       array_agg(a.attname ORDER BY x.n)
                FROM pg_index ix
                JOIN pg_class t ON t.oid = ix.indrelid
                JOIN pg_class i ON i.oid = ix.indexrelid
                JOIN unnest(ix.indkey) WITH ORDINALITY AS x(attnum, n) ON true
                JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = x.attnum
                WHERE t.relname = %s
                GROUP BY i.relname, ix.indisunique;
                """,
                (table,),
            )
            return [
                {"name": r[0], "unique": r[1], "columns": list(r[2])}
                for r in cur.fetchall()
            ]

    def get_row_count(self, table: str) -> int:
        with self.conn.cursor() as cur:
            # identifier came from list_tables(), safe to interpolate
            cur.execute(f"SELECT COUNT(*) FROM {table};")
            return cur.fetchone()[0]

    def get_column_stats(self, table: str, column: str, sample_size: int) -> Dict[str, Any]:
        with self.conn.cursor() as cur:
            # Prefer Postgres's own planner statistics when available -- far
            # cheaper than sampling and what the real optimizer trusts.
            cur.execute(
                "SELECT n_distinct, null_frac, most_common_vals "
                "FROM pg_stats WHERE tablename=%s AND attname=%s;",
                (table, column),
            )
            row = cur.fetchone()
            if row and row[0] is not None:
                n_distinct_raw, null_frac, mcv = row
                row_count = self.get_row_count(table)
                n_distinct = (
                    int(-n_distinct_raw * row_count)
                    if n_distinct_raw < 0
                    else int(n_distinct_raw)
                )
                cur.execute(
                    f"SELECT MIN({column}), MAX({column}) FROM {table};")
                min_v, max_v = cur.fetchone()
                return {
                    "distinct_count_estimate": n_distinct,
                    "null_fraction": float(null_frac or 0.0),
                    "min_value": min_v,
                    "max_value": max_v,
                    "most_common_values": list(mcv) if mcv else [],
                }

            # Fallback: sample directly (e.g. right after data load, before ANALYZE)
            cur.execute(f"SELECT COUNT(DISTINCT {column}) FROM "
                        f"(SELECT {column} FROM {table} LIMIT %s) s;", (sample_size,))
            distinct = cur.fetchone()[0]
            cur.execute(f"SELECT MIN({column}), MAX({column}) FROM {table};")
            min_v, max_v = cur.fetchone()
            return {
                "distinct_count_estimate": distinct,
                "null_fraction": 0.0,
                "min_value": min_v,
                "max_value": max_v,
                "most_common_values": [],
            }

    def execute_ddl(self, ddl: str) -> None:
        with self.conn.cursor() as cur:
            cur.execute(ddl)

    def execute_dml(self, sql: str) -> int:
        """INSERT/UPDATE/DELETE for write-path regression timing (Phase 4).
        autocommit is already on (see clone_from_source), so no explicit
        commit is needed here -- same pattern as execute_ddl/execute_query."""
        with self.conn.cursor() as cur:
            cur.execute(sql)
            return cur.rowcount

    def execute_query(self, sql: str) -> List[tuple]:
        with self.conn.cursor() as cur:
            cur.execute(sql)
            return cur.fetchall()

    def explain(self, sql: str) -> str:
        with self.conn.cursor() as cur:
            cur.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) {sql}")
            return "\n".join(r[0] for r in cur.fetchall())

    def estimate_index_storage_bytes(self, table: str, columns: List[str]) -> int:
        # Ask Postgres directly via hypothetical index size if HypoPG is
        # installed; otherwise fall back to a rough heuristic.
        with self.conn.cursor() as cur:
            try:
                cur.execute(
                    "SELECT 1 FROM pg_extension WHERE extname='hypopg';")
                if cur.fetchone():
                    cols = ", ".join(columns)
                    cur.execute(
                        f"SELECT * FROM hypopg_create_index('CREATE INDEX ON {table} ({cols})');"
                    )
                    indexrelid = cur.fetchone()[0]
                    cur.execute(f"SELECT hypopg_relation_size({indexrelid});")
                    size = cur.fetchone()[0]
                    cur.execute(f"SELECT hypopg_drop_index({indexrelid});")
                    return size
            except Exception:
                self.conn.rollback()
        row_count = self.get_row_count(table)
        return row_count * (8 * len(columns) + 8)
