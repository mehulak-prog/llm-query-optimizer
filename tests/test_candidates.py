import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.candidates import parse_llm_response, CandidateParseError, IndexCandidate


class TestCandidateParsing(unittest.TestCase):
    def test_valid_json(self):
        raw = """
        {
          "indexes": [{"table": "orders", "columns": ["o_custkey"], "unique": false, "rationale": "x"}],
          "materialized_views": [],
          "rewrites": []
        }
        """
        cs = parse_llm_response(raw, known_tables=["orders", "customer"])
        self.assertEqual(len(cs.indexes), 1)
        self.assertEqual(cs.indexes[0].table, "orders")

    def test_strips_markdown_fences(self):
        raw = '```json\n{"indexes": [], "materialized_views": [], "rewrites": []}\n```'
        cs = parse_llm_response(raw, known_tables=["orders"])
        self.assertTrue(cs.is_empty())

    def test_rejects_unknown_table(self):
        raw = '{"indexes": [{"table": "not_a_table", "columns": ["x"]}]}'
        with self.assertRaises(CandidateParseError):
            parse_llm_response(raw, known_tables=["orders"])

    def test_rejects_invalid_json(self):
        with self.assertRaises(CandidateParseError):
            parse_llm_response("not json at all", known_tables=["orders"])

    def test_index_ddl_generation(self):
        idx = IndexCandidate(table="orders", columns=["o_custkey", "o_orderdate"], rationale="x")
        ddl = idx.to_ddl()
        self.assertIn("CREATE INDEX", ddl)
        self.assertIn("orders", ddl)
        self.assertIn("o_custkey", ddl)
        self.assertIn("o_orderdate", ddl)


if __name__ == "__main__":
    unittest.main()
