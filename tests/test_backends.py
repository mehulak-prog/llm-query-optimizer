import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backends.sqlite_backend import SQLiteBackend

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


class TestSQLiteBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = os.path.join(ROOT, "data", "sandbox.db")
        if not os.path.exists(source):
            sys.path.insert(0, os.path.join(ROOT, "data"))
            import generate_data

            generate_data.build(scale=0.1)
        cls.source_db = source

    def setUp(self):
        self.scratch = os.path.join(ROOT, "data", "test_scratch.db")
        self.backend = SQLiteBackend(source_db=self.source_db, scratch_db=self.scratch)
        self.backend.clone_from_source()

    def tearDown(self):
        self.backend.close()

    def test_list_tables(self):
        tables = self.backend.list_tables()
        self.assertIn("orders", tables)
        self.assertIn("customer", tables)

    def test_get_columns(self):
        cols = self.backend.get_columns("orders")
        names = {c["name"] for c in cols}
        self.assertIn("o_custkey", names)
        pk_cols = [c for c in cols if c["is_primary_key"]]
        self.assertTrue(any(c["name"] == "o_orderkey" for c in pk_cols))

    def test_row_count(self):
        self.assertGreater(self.backend.get_row_count("orders"), 0)

    def test_create_and_drop_index(self):
        self.backend.execute_ddl("CREATE INDEX idx_test ON orders (o_custkey)")
        idxs = self.backend.get_indexes("orders")
        self.assertTrue(any(i["name"] == "idx_test" for i in idxs))
        self.backend.execute_ddl("DROP INDEX IF EXISTS idx_test")
        idxs = self.backend.get_indexes("orders")
        self.assertFalse(any(i["name"] == "idx_test" for i in idxs))

    def test_column_stats(self):
        stats = self.backend.get_column_stats("orders", "o_custkey", sample_size=100)
        self.assertIn("distinct_count_estimate", stats)
        self.assertGreaterEqual(stats["distinct_count_estimate"], 1)

    def test_time_query_returns_positive_float(self):
        t = self.backend.time_query("SELECT * FROM orders LIMIT 10", repeat=2, warmup=1)
        self.assertGreaterEqual(t, 0.0)


if __name__ == "__main__":
    unittest.main()
