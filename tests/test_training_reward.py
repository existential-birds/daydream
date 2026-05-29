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
    from daydream.training.reward import PosteriorBreakdown
    rb = score_trajectory(
        ScoringInputs(verifier_verdicts=[{"verdict": "consistent"}, {"verdict": "uncertain"}],
                      grounding_rate=0.5, format_valid=True, length=4000),
        pr_feedback="rejected")
    assert isinstance(rb, PosteriorBreakdown)
    assert rb.false_positive_penalty == 1.0
    assert rb.axes_present["false_positive"] is True
    assert rb.composite == 0.6        # pure intrinsic: 0.65 credit − 0.2·0.25 len; posterior NOT folded in
    assert rb.posterior_cost == 0.5   # sibling field: max(0, 1.0 − 0.5 default prior)


def test_contested_outcome_applies_intermediate_penalty_golden():
    from daydream.training.reward import PosteriorBreakdown
    rb = score_trajectory(
        ScoringInputs(verifier_verdicts=[{"verdict": "consistent"}, {"verdict": "uncertain"}],
                      grounding_rate=0.5, format_valid=True, length=4000),
        pr_feedback="contested")
    assert isinstance(rb, PosteriorBreakdown)
    assert rb.false_positive_penalty == 0.5
    assert rb.axes_present["false_positive"] is True
    assert rb.composite == 0.6        # pure intrinsic, unchanged by the posterior label
    assert rb.posterior_cost == 0.0   # sibling field: max(0, 0.5 − 0.5 default prior)


def test_accepted_outcome_has_zero_penalty_and_all_six_fields():
    from daydream.training.reward import PosteriorBreakdown
    rb = score_trajectory(
        ScoringInputs(verifier_verdicts=[{"verdict": "consistent"}],
                      grounding_rate=0.8, format_valid=True, length=3000),
        pr_feedback="accepted")
    assert isinstance(rb, PosteriorBreakdown)
    assert rb.false_positive_penalty == 0.0
    assert rb.posterior_cost == 0.0   # max(0, 0.0 − 0.5) clamps to 0.0
    assert all(v is not None for v in
               (rb.correctness_per_finding, rb.grounding, rb.length_penalty,
                rb.false_positive_penalty, rb.composite)) and rb.format_valid is True


def test_unknown_or_absent_posterior_leaves_axis_none_and_score_unchanged():
    from daydream.training.reward import PosteriorBreakdown, RewardBreakdown
    args = ScoringInputs(verifier_verdicts=[{"verdict": "consistent"}],
                         grounding_rate=0.5, format_valid=True, length=4000)
    unknown = score_trajectory(args, pr_feedback="unknown")
    # Unmapped label ⇒ base type, no posterior axis, composite unchanged.
    assert type(unknown) is RewardBreakdown and not isinstance(unknown, PosteriorBreakdown)
    assert "false_positive" not in unknown.axes_present
    assert unknown.composite == score_trajectory(args).composite


def test_posterior_penalty_cannot_outrank_correctness_signal():
    # KD2 drown-out guard, now structural under C5: the composite is pure
    # intrinsic, so the posterior label cannot perturb the ordering at all. A
    # high-correctness REJECTED run still scores above a zero-correctness
    # ACCEPTED run, and its composite is identical to the unlabeled score.
    good_rejected = score_trajectory(ScoringInputs([{"verdict": "consistent"}], 0.9, True, None),
                                     pr_feedback="rejected")
    bad_accepted = score_trajectory(ScoringInputs([{"verdict": "contradicts"}], 0.0, True, None),
                                    pr_feedback="accepted")
    assert good_rejected.composite > bad_accepted.composite
    # Composite is unaffected by the posterior — sibling, not subtracted.
    good_unlabeled = score_trajectory(ScoringInputs([{"verdict": "consistent"}], 0.9, True, None))
    assert good_rejected.composite == good_unlabeled.composite
    assert good_rejected.posterior_cost == 0.5  # max(0, 1.0 − 0.5); lives beside the composite


def test_composite_is_pure_intrinsic_posterior_is_sibling():
    from daydream.training.reward import PosteriorBreakdown, RewardBreakdown
    inp = ScoringInputs([{"verdict": "consistent"}, {"verdict": "uncertain"}], 0.5, True, 4000)
    base = score_trajectory(inp)                       # no label → intrinsic
    labeled = score_trajectory(inp, pr_feedback="rejected")
    assert type(base) is RewardBreakdown and not isinstance(base, PosteriorBreakdown)
    assert isinstance(labeled, PosteriorBreakdown)
    assert base.composite == 0.6                        # unchanged golden intrinsic (0.65 − 0.2·0.25)
    assert labeled.composite == 0.6                     # composite IDENTICAL despite rejected label
    assert labeled.posterior_cost == 0.5                # max(0, 1.0 − 0.5 default prior)
    assert labeled.false_positive_penalty == 1.0        # raw observed penalty retained
    assert "posterior_cost" in labeled.to_dict() and "posterior_cost" not in base.to_dict()


def test_unmapped_label_returns_base_type():
    inp = ScoringInputs([{"verdict": "consistent"}], 0.5, True, 4000)
    assert type(score_trajectory(inp, pr_feedback="unknown")).__name__ == "RewardBreakdown"


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


def test_default_weights_flagged_and_overrides_fingerprint_stably():
    from daydream.training.reward import DEFAULT_WEIGHTS, RewardWeights, _weights_fingerprint
    assert DEFAULT_WEIGHTS.is_default is True
    assert RewardWeights(w_fp=0.5).is_default is False
    assert _weights_fingerprint(RewardWeights(w_fp=0.5)) == _weights_fingerprint(RewardWeights(w_fp=0.5))
    assert _weights_fingerprint(RewardWeights(w_fp=0.5)) != _weights_fingerprint(RewardWeights(w_fp=0.6))
    assert len(_weights_fingerprint(RewardWeights(w_fp=0.5))) == 8


def test_posterior_cost_penalizes_only_surprise_above_prior():
    inp = ScoringInputs([{"verdict": "consistent"}], 0.5, True, 4000)
    chronic = score_trajectory(inp, pr_feedback="rejected", outcome_prior=0.8, outcome_prior_n=12)
    generous = score_trajectory(inp, pr_feedback="rejected", outcome_prior=0.2, outcome_prior_n=12)
    assert chronic.posterior_cost == pytest.approx(0.2)   # max(0, 1.0 − 0.8)
    assert generous.posterior_cost == pytest.approx(0.8)  # max(0, 1.0 − 0.2)
    assert generous.posterior_cost > chronic.posterior_cost          # de-bias direction
    assert chronic.outcome_prior == 0.8 and chronic.outcome_prior_n == 12
    acc = score_trajectory(inp, pr_feedback="accepted", outcome_prior=0.5, outcome_prior_n=10)
    assert acc.posterior_cost == 0.0                       # accepted below prior clamps to 0
    none_prior = score_trajectory(inp, pr_feedback="rejected")
    assert none_prior.outcome_prior is None                # audit shows uncalibrated
    assert none_prior.posterior_cost == 0.5                # falls back to 0.5 default


def test_reward_version_stamp_default_vs_custom():
    from daydream.training.reward import RewardWeights, REWARD_VERSION, _weights_fingerprint
    inp = ScoringInputs([{"verdict": "consistent"}], 0.5, True, 4000)
    assert score_trajectory(inp).reward_version == REWARD_VERSION
    custom = RewardWeights(w_fp=0.5)
    rb = score_trajectory(inp, weights=custom)
    assert rb.reward_version == f"{REWARD_VERSION}+custom-{_weights_fingerprint(custom)}"
    assert rb.to_dict()["reward_version"].startswith(f"{REWARD_VERSION}+custom-")
