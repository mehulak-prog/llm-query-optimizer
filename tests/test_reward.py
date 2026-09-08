import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.reward import score
from src.verifier import VerificationResult
from src.candidates import IndexCandidate

CONFIG = {
    "reward": {
        "weight_runtime": 1.0,
        "weight_storage_penalty": 0.15,
        "weight_write_amplification_penalty": 0.2,
        "min_speedup_to_recommend": 1.05,
    }
}


class TestReward(unittest.TestCase):
    def test_accepts_good_speedup(self):
        idx = IndexCandidate(table="orders", columns=["o_custkey"], rationale="x")
        vr = VerificationResult(
            candidate=idx,
            candidate_kind="index",
            baseline_weighted_seconds=1.0,
            candidate_weighted_seconds=0.5,
            speedup=2.0,
            storage_bytes=1024,
        )
        sc = score(vr, CONFIG)
        self.assertTrue(sc.accepted)

    def test_rejects_below_threshold_speedup(self):
        idx = IndexCandidate(table="orders", columns=["o_custkey"], rationale="x")
        vr = VerificationResult(
            candidate=idx,
            candidate_kind="index",
            baseline_weighted_seconds=1.0,
            candidate_weighted_seconds=0.99,
            speedup=1.01,
            storage_bytes=1024,
        )
        sc = score(vr, CONFIG)
        self.assertFalse(sc.accepted)

    def test_rejects_incorrect_rewrite_regardless_of_speed(self):
        from src.candidates import RewriteCandidate

        rw = RewriteCandidate(
            target_query_name="q1", original_sql="SELECT 1", rewritten_sql="SELECT 2",
            rationale="x",
        )
        vr = VerificationResult(
            candidate=rw,
            candidate_kind="rewrite",
            baseline_weighted_seconds=1.0,
            candidate_weighted_seconds=0.1,
            speedup=10.0,
            correctness_ok=False,
            correctness_note="mismatch",
        )
        sc = score(vr, CONFIG)
        self.assertFalse(sc.accepted)

    def test_error_result_rejected(self):
        idx = IndexCandidate(table="orders", columns=["o_custkey"], rationale="x")
        vr = VerificationResult(
            candidate=idx,
            candidate_kind="index",
            baseline_weighted_seconds=1.0,
            candidate_weighted_seconds=1.0,
            speedup=1.0,
            error="DDL failed",
        )
        sc = score(vr, CONFIG)
        self.assertFalse(sc.accepted)


if __name__ == "__main__":
    unittest.main()
