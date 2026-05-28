"""Tests for the intrinsic reward reducer — covers golden-locked formula, posterior false-positive axis, and weight overrides."""

from __future__ import annotations

import pytest

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


def test_rejected_outcome_applies_posterior_penalty_golden():
    rb = score_trajectory(
        ScoringInputs(verifier_verdicts=[{"verdict": "consistent"}, {"verdict": "uncertain"}],
                      grounding_rate=0.5, format_valid=True, length=4000),
        pr_feedback="rejected")
    assert rb.false_positive_penalty == 1.0
    assert rb.axes_present["false_positive"] is True
    assert rb.composite == 0.30   # 0.65 credit − 0.2·0.25 len − 0.3·1.0 fp


def test_contested_outcome_applies_intermediate_penalty_golden():
    rb = score_trajectory(
        ScoringInputs(verifier_verdicts=[{"verdict": "consistent"}, {"verdict": "uncertain"}],
                      grounding_rate=0.5, format_valid=True, length=4000),
        pr_feedback="contested")
    assert rb.false_positive_penalty == 0.5
    assert rb.axes_present["false_positive"] is True
    assert rb.composite == 0.45   # 0.65 credit − 0.2·0.25 len − 0.3·0.5 fp


def test_accepted_outcome_has_zero_penalty_and_all_six_fields():
    rb = score_trajectory(
        ScoringInputs(verifier_verdicts=[{"verdict": "consistent"}],
                      grounding_rate=0.8, format_valid=True, length=3000),
        pr_feedback="accepted")
    assert rb.false_positive_penalty == 0.0
    assert all(v is not None for v in
               (rb.correctness_per_finding, rb.grounding, rb.length_penalty,
                rb.false_positive_penalty, rb.composite)) and rb.format_valid is True


def test_unknown_or_absent_posterior_leaves_axis_none_and_score_unchanged():
    args = ScoringInputs(verifier_verdicts=[{"verdict": "consistent"}],
                         grounding_rate=0.5, format_valid=True, length=4000)
    assert score_trajectory(args, pr_feedback="unknown").false_positive_penalty is None
    assert score_trajectory(args, pr_feedback="unknown").composite == score_trajectory(args).composite


def test_posterior_penalty_cannot_outrank_correctness_signal():
    # KD2 drown-out guard: a high-correctness REJECTED run still scores above a
    # zero-correctness ACCEPTED run — reject deducts but never inverts the order.
    good_rejected = score_trajectory(ScoringInputs([{"verdict": "consistent"}], 0.9, True, None),
                                     pr_feedback="rejected")
    bad_accepted = score_trajectory(ScoringInputs([{"verdict": "contradicts"}], 0.0, True, None),
                                    pr_feedback="accepted")
    assert good_rejected.composite > bad_accepted.composite


def test_score_trajectory_does_no_io(monkeypatch):
    import builtins
    monkeypatch.setattr(builtins, "open", lambda *a, **k: (_ for _ in ()).throw(AssertionError("I/O!")))
    rb = score_trajectory(ScoringInputs([{"verdict": "consistent"}], 0.5, True, 100),
                          pr_feedback="rejected")
    assert rb.composite is not None   # ran purely, no file access


def test_zero_sum_present_credit_weights_raises_value_error():
    from daydream.training.reward import RewardWeights
    weights = RewardWeights(w_correctness=0.0, w_grounding=0.0)
    inputs = ScoringInputs(
        verifier_verdicts=[{"verdict": "consistent"}],
        grounding_rate=0.5,
        format_valid=True,
        length=None,
    )
    with pytest.raises(ValueError, match="sum of present credit weights"):
        score_trajectory(inputs, weights=weights)


def test_same_function_scores_producer_and_eval_caller_paths():
    # Guard that harvest.py's bound score_trajectory is the same object as the
    # canonical one from daydream.training.reward — not a stale copy or wrapper.
    import daydream.training.harvest as harvest_mod
    from daydream.training.reward import score_trajectory as canonical_fn
    assert harvest_mod.score_trajectory is canonical_fn
    inp = ScoringInputs([{"verdict": "consistent"}], 0.7, True, 500)
    assert canonical_fn(inp, pr_feedback="accepted").composite == harvest_mod.score_trajectory(inp, pr_feedback="accepted").composite
