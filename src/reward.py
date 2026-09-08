"""
Turns a VerificationResult (measured, not predicted, runtime) into a
scalar reward and an accept/reject decision. This is deliberately a
separate, small, inspectable module -- the reward function is a first-
class research object in this line of work (see docs/research_framing.md
on evaluation metrics), and you'll want to swap weightings, ablate the
storage penalty, etc. without touching the verifier or the loop.
"""

from __future__ import annotations

from dataclasses import dataclass

from .verifier import VerificationResult


@dataclass
class ScoredCandidate:
    result: VerificationResult
    reward: float
    accepted: bool
    reason: str


def score(result: VerificationResult, config: dict) -> ScoredCandidate:
    rc = config["reward"]
    min_speedup = rc["min_speedup_to_recommend"]

    if result.error:
        return ScoredCandidate(result=result, reward=0.0, accepted=False, reason=result.error)

    if result.candidate_kind == "rewrite" and not result.correctness_ok:
        return ScoredCandidate(
            result=result, reward=0.0, accepted=False, reason=result.correctness_note
        )

    runtime_component = rc["weight_runtime"] * (result.speedup - 1.0)

    storage_penalty = 0.0
    if result.storage_bytes:
        storage_mb = result.storage_bytes / (1024 * 1024)
        storage_penalty = rc["weight_storage_penalty"] * (storage_mb / 100.0)

    write_penalty = 0.0
    if (
        result.candidate_kind == "index"
        and result.write_baseline_seconds
        and result.write_candidate_seconds
    ):
        write_regression = result.write_candidate_seconds / result.write_baseline_seconds
        write_penalty = rc["weight_write_amplification_penalty"] * \
            max(0.0, write_regression - 1.0)

    reward = runtime_component - storage_penalty - write_penalty

    if result.speedup < min_speedup:
        return ScoredCandidate(
            result=result,
            reward=reward,
            accepted=False,
            reason=(
                f"Measured speedup {result.speedup:.3f}x is below the "
                f"min_speedup_to_recommend threshold ({min_speedup}x)."
            ),
        )

    # Acceptance must agree with reward: a candidate that clears the raw
    # speedup bar can still net negative once storage/write penalties are
    # applied (e.g. a marginal index that clears min_speedup but isn't worth
    # its write-path cost). Gating on reward > 0 here, rather than on speedup
    # alone, keeps "accepted" and "reward" from contradicting each other.
    if reward <= 0:
        return ScoredCandidate(
            result=result,
            reward=reward,
            accepted=False,
            reason=(
                f"Measured speedup {result.speedup:.3f}x clears the threshold, "
                f"but net reward {reward:.3f} is not positive after storage/"
                f"write penalties -- not worth recommending."
            ),
        )

    return ScoredCandidate(
        result=result,
        reward=reward,
        accepted=True,
        reason=f"Verified speedup {result.speedup:.3f}x, reward={reward:.3f}.",
    )


def enforce_shared_storage_budget(
    accepted: list, rejected: list, budget_bytes: int
) -> tuple:
    """Caps total accepted index storage to the SAME budget across every
    system being compared. Without this, per-candidate accept/reject
    (score() above) lets each system accumulate as much total storage as
    it likes -- and a system that's simply less storage-conservative can
    look like it "wins" on combined workload speedup for no reason other
    than spending more disk space than what it's being compared against.
    That's not a fair comparison, so this makes it one: candidates are
    kept greedily by reward (best first) until the shared budget would be
    exceeded; everything past that point is moved to rejected with a
    reason explaining why, so it's visible in output rather than silently
    dropped.

    Views/rewrites have no storage_bytes (not applicable) and are never
    excluded by this -- only index storage counts against the budget,
    since that's the resource actually being compared unfairly above.
    """
    indexed = [sc for sc in accepted if sc.result.candidate_kind == "index"]
    non_indexed = [
        sc for sc in accepted if sc.result.candidate_kind != "index"]

    indexed_by_reward = sorted(indexed, key=lambda sc: sc.reward, reverse=True)

    kept: list = []
    dropped: list = []
    running_total = 0
    for sc in indexed_by_reward:
        size = sc.result.storage_bytes or 0
        if running_total + size <= budget_bytes:
            kept.append(sc)
            running_total += size
        else:
            dropped.append(
                ScoredCandidate(
                    result=sc.result,
                    reward=sc.reward,
                    accepted=False,
                    reason=(
                        f"{sc.reason} [EXCLUDED: shared storage budget "
                        f"({budget_bytes / (1024*1024):.1f} MB) would be "
                        f"exceeded -- kept higher-reward candidates instead, "
                        f"to keep this comparison fair across systems.]"
                    ),
                )
            )

    return kept + non_indexed, rejected + dropped
