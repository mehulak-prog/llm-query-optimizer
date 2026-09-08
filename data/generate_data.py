#!/usr/bin/env python3
"""
Builds data/sandbox.db (SQLite) from schema.sql and populates it with
synthetic TPC-H-shaped data using only the standard library `random`
module (no numpy/faker dependency required for the base demo).

Usage:
    python data/generate_data.py [--scale 1.0]

Row counts scale roughly with --scale (default 1.0 -> ~5k orders,
~20k lineitems). Bump this to make the index-selection problem more
realistic; SQLite will start to genuinely benefit from good indexes
somewhere around 50k-500k rows depending on the query.
"""

from __future__ import annotations

import argparse
import os
import random
import sqlite3

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA_PATH = os.path.join(HERE, "schema.sql")
DB_PATH = os.path.join(HERE, "sandbox.db")

NATIONS = [
    (0, "ALGERIA", 0), (1, "ARGENTINA", 1), (2, "BRAZIL", 1), (3, "CANADA", 1),
    (4, "EGYPT", 4), (5, "FRANCE", 3), (6, "GERMANY", 3), (7, "INDIA", 2),
    (8, "INDONESIA", 2), (9, "JAPAN", 2), (10, "KENYA", 0), (11, "MOROCCO", 0),
    (12, "RUSSIA", 3), (13, "UNITED KINGDOM", 3), (14, "UNITED STATES", 1),
]
MKTSEGMENTS = ["AUTOMOBILE", "BUILDING", "FURNITURE", "MACHINERY", "HOUSEHOLD"]
ORDER_STATUS = ["O", "F", "P"]
ORDER_PRIORITY = ["1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"]
RETURN_FLAG = ["N", "R", "A"]
CATEGORIES = ["ELECTRONICS", "TOYS", "CLOTHING", "GROCERY", "TOOLS", "BOOKS"]


def build(scale: float = 1.0, seed: int = 42) -> None:
    random.seed(seed)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    conn = sqlite3.connect(DB_PATH)
    with open(SCHEMA_PATH) as f:
        conn.executescript(f.read())

    n_customers = int(2000 * scale)
    n_orders = int(5000 * scale)
    n_parts = int(1500 * scale)
    n_suppliers = int(200 * scale)

    conn.executemany(
        "INSERT INTO nation VALUES (?, ?, ?)", NATIONS
    )

    conn.executemany(
        "INSERT INTO customer VALUES (?, ?, ?, ?, ?, ?)",
        (
            (
                i,
                f"Customer#{i:09d}",
                random.randrange(len(NATIONS)),
                f"{random.randint(10,99)}-{random.randint(100,999)}-{random.randint(1000,9999)}",
                round(random.uniform(-999.99, 9999.99), 2),
                random.choice(MKTSEGMENTS),
            )
            for i in range(n_customers)
        ),
    )

    conn.executemany(
        "INSERT INTO supplier VALUES (?, ?, ?, ?)",
        (
            (
                i,
                f"Supplier#{i:09d}",
                random.randrange(len(NATIONS)),
                round(random.uniform(-999.99, 9999.99), 2),
            )
            for i in range(n_suppliers)
        ),
    )

    conn.executemany(
        "INSERT INTO part VALUES (?, ?, ?, ?, ?)",
        (
            (
                i,
                f"Part#{i:09d}",
                f"Brand#{random.randint(1,5)}{random.randint(1,5)}",
                random.choice(CATEGORIES),
                round(random.uniform(1.0, 2000.0), 2),
            )
            for i in range(n_parts)
        ),
    )

    order_rows = []
    for i in range(n_orders):
        year = random.randint(1992, 1998)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        order_rows.append(
            (
                i,
                random.randrange(n_customers),
                random.choice(ORDER_STATUS),
                round(random.uniform(100.0, 500000.0), 2),
                f"{year:04d}-{month:02d}-{day:02d}",
                random.choice(ORDER_PRIORITY),
            )
        )
    conn.executemany("INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?)", order_rows)

    lineitem_rows = []
    for orderkey, *_ in order_rows:
        n_lines = random.randint(1, 7)
        for line_no in range(1, n_lines + 1):
            year = random.randint(1992, 1998)
            month = random.randint(1, 12)
            day = random.randint(1, 28)
            lineitem_rows.append(
                (
                    orderkey,
                    line_no,
                    random.randrange(n_parts),
                    float(random.randint(1, 50)),
                    round(random.uniform(100.0, 100000.0), 2),
                    round(random.uniform(0.0, 0.1), 2),
                    f"{year:04d}-{month:02d}-{day:02d}",
                    random.choice(RETURN_FLAG),
                )
            )
    conn.executemany(
        "INSERT INTO lineitem VALUES (?, ?, ?, ?, ?, ?, ?, ?)", lineitem_rows
    )

    conn.commit()
    conn.close()
    print(
        f"Built {DB_PATH}: {n_customers} customers, {n_orders} orders, "
        f"{len(lineitem_rows)} lineitems, {n_parts} parts, {n_suppliers} suppliers."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build(scale=args.scale, seed=args.seed)
