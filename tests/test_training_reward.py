"""Tests for the pure intrinsic reward reducer (golden-locked formula)."""

from __future__ import annotations

from daydream.training.reward import REWARD_VERSION, ScoringInputs, score_trajectory


def test_intrinsic_composite_is_golden():
    rb = score_trajectory(
        ScoringInputs(
            verifier_verdicts=[
                {"issue_id": 1, "verdict": "consistent"},
                {"issue_id": 2, "verdict": "uncertain"},
            ],
            grounding_rate=0.5,
            format_valid=True,
            length=4000,
        )
    )
    assert rb.reward_version == REWARD_VERSION
    assert rb.correctness_per_finding == [1.0, 0.5]
    assert rb.composite == 0.6  # credit (0.6·0.75+0.4·0.5)=0.65 − 0.2·len_norm(0.25)=0.05


def test_format_invalid_floors_composite():
    rb = score_trajectory(
        ScoringInputs(verifier_verdicts=None, grounding_rate=0.9, format_valid=False, length=50)
    )
    assert rb.composite == 0.0  # dominating gate


def test_missing_correctness_axis_renormalizes_over_grounding():
    rb = score_trajectory(
        ScoringInputs(verifier_verdicts=None, grounding_rate=0.8, format_valid=True, length=None)
    )
    assert rb.correctness_per_finding is None
    assert rb.axes_present["correctness"] is False
    assert rb.composite == 0.8  # renormalized: grounding alone, NOT 0.4·0.8=0.32


def test_weights_are_overridable_and_change_composite_predictably():
    from daydream.training.reward import RewardWeights
    base_len = ScoringInputs(verifier_verdicts=[{"verdict": "consistent"}],
                             grounding_rate=None, format_valid=True, length=10000)
    # length=10000 → len_norm saturates at 1.0; only w_len differs between calls.
    assert score_trajectory(base_len).composite == 0.8                          # default w_len=0.2
    assert score_trajectory(base_len, weights=RewardWeights(w_len=0.5)).composite == 0.5
