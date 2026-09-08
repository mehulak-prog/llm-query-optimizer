"""
Abstract interface every sandbox backend must implement. This is the
seam that lets the rest of the system (schema extraction, stats
collection, the sandbox lifecycle, the verifier) be backend-agnostic.

To add a new backend (MySQL, DuckDB, Snowflake, ...), implement this
interface and register it in src/backends/__init__.py's `get_backend`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import time


class SandboxBackend(ABC):
    """A live connection to a (disposable, isolated) sandbox database."""

    # ---- lifecycle -------------------------------------------------
    @abstractmethod
    def clone_from_source(self) -> None:
        """Create/refresh the scratch sandbox from the source database.
        Must be cheap enough to call once per trial (candidate)."""

    @abstractmethod
    def close(self) -> None:
        ...

    # ---- introspection ----------------------------------------------
    @abstractmethod
    def list_tables(self) -> List[str]:
        ...

    @abstractmethod
    def get_columns(self, table: str) -> List[Dict[str, Any]]:
        """Returns list of dicts matching ColumnInfo fields:
        name, data_type, nullable, is_primary_key"""

    @abstractmethod
    def get_foreign_keys(self, table: str) -> List[Dict[str, Any]]:
        """Returns list of dicts: column, references_table, references_column"""

    @abstractmethod
    def get_indexes(self, table: str) -> List[Dict[str, Any]]:
        """Returns list of dicts: name, columns, unique"""

    @abstractmethod
    def get_row_count(self, table: str) -> int:
        ...

    @abstractmethod
    def get_column_stats(self, table: str, column: str, sample_size: int) -> Dict[str, Any]:
        """Returns dict: distinct_count_estimate, null_fraction, min_value,
        max_value, most_common_values"""

    # ---- execution / mutation ----------------------------------------
    @abstractmethod
    def execute_ddl(self, ddl: str) -> None:
        """Executes a DDL statement (CREATE INDEX, CREATE VIEW, DROP ...)."""

    @abstractmethod
    def execute_query(self, sql: str) -> List[tuple]:
        """Executes a read query and returns all rows (for correctness checks)."""

    @abstractmethod
    def execute_dml(self, sql: str) -> int:
        """Executes a non-DDL mutation (INSERT/UPDATE/DELETE) and returns
        the affected row count. Separate from execute_query (SELECT-only,
        returns rows) and execute_ddl (schema changes, no return value) --
        this is the primitive Phase 4's write-regression timing needs."""

    @abstractmethod
    def explain(self, sql: str) -> str:
        """Returns the query plan as text."""

    # ---- timing -------------------------------------------------------
    def time_query(self, sql: str, repeat: int = 5, warmup: int = 1) -> float:
        """Runs `sql` `warmup` times (discarded) then `repeat` times,
        returns the median wall-clock seconds. Shared default
        implementation; backends can override for native timing."""
        for _ in range(warmup):
            self.execute_query(sql)
        durations = []
        for _ in range(repeat):
            start = time.perf_counter()
            self.execute_query(sql)
            durations.append(time.perf_counter() - start)
        durations.sort()
        mid = len(durations) // 2
        if len(durations) % 2 == 0:
            return (durations[mid - 1] + durations[mid]) / 2
        return durations[mid]

    # ---- estimation -----------------------------------------------------
    @abstractmethod
    def estimate_index_storage_bytes(self, table: str, columns: List[str]) -> int:
        """Rough estimate of on-disk size an index would take, used to
        enforce candidates.max_index_storage_mb before wasting a trial."""
        # ---- write timing (Phase 4: real write-path regression) -----------

    def time_write_batch(
        self,
        table: str,
        pk_column: str,
        touch_column: str,
        n_rows: int,
        offset_base: int,
        repeat: int = 5,
        warmup: int = 1,
    ) -> float:
        """Times `warmup` discarded then `repeat` measured cycles of
        insert(n_rows) -> update(n_rows, self-assign touch_column) ->
        delete(n_rows), each cycle using a fresh non-overlapping
        synthetic PK range so cycles never collide. Returns median
        wall-clock seconds for one full cycle. `touch_column` should be
        a column covered by the index candidate under test -- see
        verifier.measure_write_regression for why the self-assignment
        still forces real index maintenance in Postgres."""
        columns = [c["name"] for c in self.get_columns(table)]
        select_exprs = [
            f"{c} + {{offset}}" if c == pk_column else c for c in columns
        ]
        cols_csv = ", ".join(columns)

        def run_cycle(offset: int) -> None:
            select_csv = ", ".join(e.format(offset=offset)
                                   for e in select_exprs)
            self.execute_dml(
                f"INSERT INTO {table} ({cols_csv}) "
                f"SELECT {select_csv} FROM {table} ORDER BY {pk_column} LIMIT {n_rows}"
            )
            self.execute_dml(
                f"UPDATE {table} SET {touch_column} = {touch_column} "
                f"WHERE {pk_column} > {offset}"
            )
            self.execute_dml(
                f"DELETE FROM {table} WHERE {pk_column} > {offset}")

        step = n_rows * 10  # generous gap so no cycle's shifted PKs overlap another's
        cycle = 0
        for _ in range(warmup):
            run_cycle(offset_base + cycle * step)
            cycle += 1

        durations = []
        for _ in range(repeat):
            offset = offset_base + cycle * step
            start = time.perf_counter()
            run_cycle(offset)
            durations.append(time.perf_counter() - start)
            cycle += 1

        durations.sort()
        mid = len(durations) // 2
        if len(durations) % 2 == 0:
            return (durations[mid - 1] + durations[mid]) / 2
        return durations[mid]
