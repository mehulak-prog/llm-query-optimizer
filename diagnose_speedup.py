"""
Diagnostic: runs EXPLAIN (ANALYZE, BUFFERS) on the same query before and
after adding an index, directly against sandbox_template (disposable --
fine to mutate, we drop the index at the end). This is to see WHAT is
actually happening, rather than guessing why the measured speedups at
scale look smaller than expected.

Usage:
    python diagnose_speedup.py
"""

import psycopg2

DSN = dict(
    host="localhost", port=5432, dbname="sandbox_template",
    user="postgres", password="postgres",
)

# Highest-weight query in the workload -- filters orders by o_custkey.
QUERY = "SELECT o_orderkey, o_totalprice, o_orderdate FROM orders WHERE o_custkey = 137"
INDEX_DDL = "CREATE INDEX IF NOT EXISTS idx_diag_orders_custkey ON orders (o_custkey);"
DROP_DDL = "DROP INDEX IF EXISTS idx_diag_orders_custkey;"


def explain(cur, label):
    cur.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) {QUERY};")
    print(f"\n=== {label} ===")
    for row in cur.fetchall():
        print(row[0])


def main():
    conn = psycopg2.connect(**DSN)
    conn.autocommit = True
    cur = conn.cursor()

    cur.execute(DROP_DDL)  # ensure clean starting state

    # Run the query a few times BEFORE the index, to mimic what the
    # verifier's repeat_runs/warmup_runs already does, and to check if
    # caching alone (not the index) is what's shrinking the gap.
    print("Running query 3x with NO index (checking for cache warm-up effects)...")
    for i in range(3):
        cur.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) {QUERY};")
        rows = cur.fetchall()
        exec_line = [r[0] for r in rows if "Execution Time" in r[0]]
        print(f"  run {i+1}: {exec_line[0] if exec_line else '(not found)'}")

    explain(cur, "BEFORE INDEX (full plan)")

    print("\nCreating index...")
    cur.execute(INDEX_DDL)

    print("Running query 3x WITH index...")
    for i in range(3):
        cur.execute(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) {QUERY};")
        rows = cur.fetchall()
        exec_line = [r[0] for r in rows if "Execution Time" in r[0]]
        print(f"  run {i+1}: {exec_line[0] if exec_line else '(not found)'}")

    explain(cur, "AFTER INDEX (full plan)")

    # How many rows actually match this filter? If very few or very
    # many, that changes whether an index should even help.
    cur.execute("SELECT COUNT(*) FROM orders WHERE o_custkey = 137;")
    match_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM orders;")
    total_count = cur.fetchone()[0]
    print(f"\nSelectivity: {match_count} matching rows out of {total_count} total "
          f"({100*match_count/total_count:.4f}%)")

    print("\nCleaning up (dropping diagnostic index)...")
    cur.execute(DROP_DDL)
    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
