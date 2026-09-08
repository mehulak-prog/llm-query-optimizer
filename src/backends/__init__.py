from __future__ import annotations

from typing import Any, Dict

from .base import SandboxBackend
from .sqlite_backend import SQLiteBackend


def get_backend(config: Dict[str, Any]) -> SandboxBackend:
    """Factory: builds the configured sandbox backend from config.yaml's
    `sandbox` section."""
    kind = config["backend"]
    if kind == "sqlite":
        cfg = config["sqlite"]
        return SQLiteBackend(source_db=cfg["source_db"], scratch_db=cfg["scratch_db"])
    elif kind == "postgres":
        from .postgres_backend import PostgresBackend  # lazy import (needs psycopg2)

        cfg = config["postgres"]
        return PostgresBackend(dsn=cfg["dsn"], template_db=cfg["template_db"])
    else:
        raise ValueError(f"Unknown sandbox backend: {kind}")
