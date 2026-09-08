"""
Smoke test for PostgresBackend -- exercises the actual project code
(not just raw psql) to confirm clone_from_source / introspection /
EXPLAIN / cleanup all round-trip correctly against a live Postgres.

Usage:
    python test_postgres_backend.py
"""

from src.backends.postgres_backend import PostgresBackend

DSN = "host=localhost port=5432 dbname=postgres user=postgres password=postgres"

backend = PostgresBackend(
    dsn=DSN,
    template_db="sandbox_template",
    scratch_db="sandbox_trial_smoketest",
)

print("Cloning scratch DB from sandbox_template...")
backend.clone_from_source()
print("  OK -- clone created.\n")

print("Listing tables...")
tables = backend.list_tables()
print(f"  {tables}\n")
assert set(tables) == {"nation", "customer", "part", "supplier", "orders", "lineitem"}, \
    f"Unexpected table set: {tables}"

print("Row counts:")
for t in tables:
    print(f"  {t}: {backend.get_row_count(t)}")
print()

print("Columns for 'orders':")
for col in backend.get_columns("orders"):
    print(f"  {col}")
print()

print("Foreign keys for 'orders':")
print(f"  {backend.get_foreign_keys('orders')}\n")

print("Real EXPLAIN ANALYZE on a join query...")
plan = backend.explain(
    "SELECT o_orderkey, o_totalprice FROM orders "
    "WHERE o_custkey = 42 ORDER BY o_orderdate DESC LIMIT 10;"
)
print(plan)
print()

print("Creating a test index via execute_ddl...")
backend.execute_ddl("CREATE INDEX ON orders (o_custkey);")
print("  OK -- index created.\n")

print("Estimating index storage size...")
size = backend.estimate_index_storage_bytes("orders", ["o_custkey"])
print(f"  ~{size} bytes\n")

print("Cleaning up (closing + dropping scratch DB)...")
backend.close()
print("  OK -- scratch DB dropped.\n")

print("ALL CHECKS PASSED.")
