"""
Schema extraction: turns a live sandbox connection into a structured,
LLM-friendly representation of tables, columns, types, primary/foreign
keys, and existing indexes.

This is backend-agnostic: it only calls methods on the SandboxBackend
interface (see src/backends/base.py), so the same code works whether the
underlying connection is SQLite or Postgres.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional


@dataclass
class ColumnInfo:
    name: str
    data_type: str
    nullable: bool = True
    is_primary_key: bool = False


@dataclass
class ForeignKeyInfo:
    column: str
    references_table: str
    references_column: str


@dataclass
class IndexInfo:
    name: str
    columns: List[str]
    unique: bool = False


@dataclass
class TableSchema:
    name: str
    columns: List[ColumnInfo] = field(default_factory=list)
    foreign_keys: List[ForeignKeyInfo] = field(default_factory=list)
    existing_indexes: List[IndexInfo] = field(default_factory=list)
    row_count: Optional[int] = None


@dataclass
class DatabaseSchema:
    tables: Dict[str, TableSchema] = field(default_factory=dict)

    def to_llm_text(self) -> str:
        """Render a compact, unambiguous textual schema for the LLM prompt.

        Deliberately not just json.dumps: a DDL-like rendering uses fewer
        tokens and is closer to what the model has seen at pretraining
        time (real CREATE TABLE statements), which improves candidate
        quality in practice.
        """
        lines = []
        for table in self.tables.values():
            lines.append(f"TABLE {table.name} (rows ~= {table.row_count}):")
            for col in table.columns:
                pk = " PRIMARY KEY" if col.is_primary_key else ""
                null = "" if col.nullable else " NOT NULL"
                lines.append(f"  - {col.name} {col.data_type}{pk}{null}")
            for fk in table.foreign_keys:
                lines.append(
                    f"  FK: {fk.column} -> {fk.references_table}.{fk.references_column}"
                )
            if table.existing_indexes:
                for idx in table.existing_indexes:
                    uniq = "UNIQUE " if idx.unique else ""
                    lines.append(
                        f"  EXISTING {uniq}INDEX {idx.name} ({', '.join(idx.columns)})"
                    )
            else:
                lines.append("  EXISTING INDEXES: none besides primary key")
            lines.append("")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            name: {
                "columns": [asdict(c) for c in t.columns],
                "foreign_keys": [asdict(f) for f in t.foreign_keys],
                "existing_indexes": [asdict(i) for i in t.existing_indexes],
                "row_count": t.row_count,
            }
            for name, t in self.tables.items()
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def extract_schema(backend) -> DatabaseSchema:
    """backend: an object implementing SandboxBackend (see backends/base.py)."""
    schema = DatabaseSchema()
    for table_name in backend.list_tables():
        columns = [
            ColumnInfo(**c) for c in backend.get_columns(table_name)
        ]
        fks = [ForeignKeyInfo(**f) for f in backend.get_foreign_keys(table_name)]
        idxs = [IndexInfo(**i) for i in backend.get_indexes(table_name)]
        row_count = backend.get_row_count(table_name)
        schema.tables[table_name] = TableSchema(
            name=table_name,
            columns=columns,
            foreign_keys=fks,
            existing_indexes=idxs,
            row_count=row_count,
        )
    return schema
