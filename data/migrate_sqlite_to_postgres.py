#!/usr/bin/env python3
"""
One-time migration: copies the data already generated in data/sandbox.db
(SQLite) into a Postgres database -- normally sandbox_template, the DB
that PostgresBackend.clone_from_source() clones per trial.

This exists because data/generate_data.py only writes SQLite; there is
no --backend postgres flag (yet). Loading schema.sql once via psql and
then running this script is the fastest path to a working Postgres
template without duplicating the whole generator.

Usage:
    python data/migrate_sqlite_to_postgres.py

Assumes:
    - data/sandbox.db already exists (run generate_data.py first if not)
    - The target Postgres database already has schema.sql applied
      (see the psql command in the README/step instructions -- this
      script does NOT create tables, only inserts rows)
"""

from __future__ import annotations

import os
import sqlite3

import psycopg2

HERE = os.path.dirname(os.path.abspath(__file__))
SQLITE_PATH = os.path.join(HERE, "sandbox.db")

# Order matters: parents before children, to satisfy foreign keys.
TABLES_IN_LOAD_ORDER = ["nation", "customer", "part", "supplier", "orders", "lineitem"]

PG_DSN = dict(
    host="localhost",
    port=5432,
    dbname="sandbox_template",
    user="postgres",
    password="postgres",
)


def migrate():
    if not os.path.exists(SQLITE_PATH):
        raise SystemExit(
            f"{SQLITE_PATH} not found -- run `python data/generate_data.py` first."
        )

    sqlite_conn = sqlite3.connect(SQLITE_PATH)
    pg_conn = psycopg2.connect(**PG_DSN)
    pg_conn.autocommit = False

    try:
        pg_cur = pg_conn.cursor()

        for table in TABLES_IN_LOAD_ORDER:
            sqlite_cur = sqlite_conn.cursor()
            sqlite_cur.execute(f"SELECT * FROM {table};")
            rows = sqlite_cur.fetchall()
            col_names = [d[0] for d in sqlite_cur.description]

            if not rows:
                print(f"{table}: 0 rows, skipping.")
                continue

            placeholders = ", ".join(["%s"] * len(col_names))
            col_list = ", ".join(col_names)
            insert_sql = f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})"

            psycopg2.extras = __import__("psycopg2.extras", fromlist=["extras"])
            psycopg2.extras.execute_batch(pg_cur, insert_sql, rows, page_size=1000)

            print(f"{table}: inserted {len(rows)} rows.")

        pg_conn.commit()
        print("\nMigration complete. Running ANALYZE so Postgres's planner "
              "has real statistics...")

        pg_cur.execute("ANALYZE;")
        pg_conn.commit()
        print("Done.")

    except Exception:
        pg_conn.rollback()
        raise
    finally:
        sqlite_conn.close()
        pg_conn.close()


if __name__ == "__main__":
    migrate()
