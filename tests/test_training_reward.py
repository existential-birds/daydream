"""Pin the intrinsic formula, separate posterior penalty axis, and weight overrides."""
from __future__ import annotations

from daydream.training.reward import (
    RewardWeights,
    ScoringInputs,
    score_trajectory,
)


def test_weights_are_overridable_and_change_composite_predictably() -> None:
    base_len = ScoringInputs(verifier_verdicts=[{"verdict": "consistent"}], format_valid=True, length=10000)
    # At length 10000, len_norm saturates at 1.0; only w_len changes.
    assert score_trajectory(base_len).composite == 0.8                          # default w_len=0.2
    assert score_trajectory(base_len, weights=RewardWeights(w_len=0.5)).composite == 0.5
