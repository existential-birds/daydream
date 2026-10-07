"""Pin the intrinsic formula, separate posterior penalty axis, and weight overrides."""
from __future__ import annotations

import builtins
from typing import cast

import pytest

from daydream.training.reward import (
    REWARD_VERSION,
    PosteriorBreakdown,
    RewardBreakdown,
    RewardWeights,
    ScoringInputs,
    _weights_fingerprint,
    score_trajectory,
)


def test_intrinsic_composite_is_golden() -> None:
    rb = score_trajectory(ScoringInputs(
            verifier_verdicts=[{"issue_id": 1, "verdict": "consistent"}, {"issue_id": 2, "verdict": "uncertain"}],

            format_valid=True, length=4000,
        )
    )
    assert rb.reward_version == REWARD_VERSION
    assert rb.correctness_per_finding == [1.0, 0.5]
    assert rb.composite == 0.7  # correctness 0.75 − 0.2·len_norm(0.25)=0.05

def test_format_invalid_floors_composite() -> None:
    rb = score_trajectory(ScoringInputs(verifier_verdicts=None,  format_valid=False, length=50))
    assert rb.composite == 0.0  # dominating gate

def test_missing_correctness_axis_has_no_credit() -> None:
    rb = score_trajectory(ScoringInputs(verifier_verdicts=None,  format_valid=True, length=None))
    assert rb.correctness_per_finding is None
    assert rb.axes_present["correctness"] is False
    assert rb.composite is None
    assert "grounding" not in rb.to_dict()

def test_weights_are_overridable_and_change_composite_predictably() -> None:
    base_len = ScoringInputs(verifier_verdicts=[{"verdict": "consistent"}], format_valid=True, length=10000)
    # At length 10000, len_norm saturates at 1.0; only w_len changes.
    assert score_trajectory(base_len).composite == 0.8                          # default w_len=0.2
    assert score_trajectory(base_len, weights=RewardWeights(w_len=0.5)).composite == 0.5

@pytest.mark.parametrize(("pr_feedback", "expected_penalty", "expected_posterior_cost"),
    [pytest.param("rejected", 1.0, 0.5, id="rejected"), pytest.param("contested", 0.5, 0.0, id="contested")],
)
def test_outcome_applies_posterior_penalty_golden(
    pr_feedback: str, expected_penalty: float, expected_posterior_cost: float,
) -> None:
    """Expose posterior penalties separately from the intrinsic composite score."""
    inputs = ScoringInputs(verifier_verdicts=[{"verdict": "consistent"}, {"verdict": "uncertain"}],
        format_valid=True, length=4000)
    base = score_trajectory(inputs)
    assert type(base) is RewardBreakdown and not isinstance(base, PosteriorBreakdown)
    assert base.composite == 0.7
    assert "posterior_cost" not in base.to_dict()
    rb = score_trajectory(inputs, pr_feedback=pr_feedback)
    assert isinstance(rb, PosteriorBreakdown)
    assert rb.false_positive_penalty == expected_penalty
    assert rb.axes_present["false_positive"] is True
    assert rb.composite == 0.7        # correctness minus length; posterior remains separate
    assert rb.posterior_cost == expected_posterior_cost
    assert "posterior_cost" in rb.to_dict()

def test_accepted_outcome_has_zero_penalty_and_all_six_fields() -> None:
    rb = score_trajectory(ScoringInputs(verifier_verdicts=[{"verdict": "consistent"}], format_valid=True, length=3000),
        pr_feedback="accepted")
    assert isinstance(rb, PosteriorBreakdown)
    assert rb.false_positive_penalty == 0.0
    assert rb.posterior_cost == pytest.approx(0.5)   # abs(0.0 − 0.5 default prior)
    assert all(v is not None for v in
               (rb.correctness_per_finding, rb.length_penalty,
                rb.false_positive_penalty, rb.composite)) and rb.format_valid is True

def test_unknown_or_absent_posterior_leaves_axis_none_and_score_unchanged() -> None:
    args = ScoringInputs(verifier_verdicts=[{"verdict": "consistent"}], format_valid=True, length=4000)
    unknown = score_trajectory(args, pr_feedback="unknown")
    assert type(unknown) is RewardBreakdown and not isinstance(unknown, PosteriorBreakdown)
    assert "false_positive" not in unknown.axes_present
    assert unknown.composite == score_trajectory(args).composite

def test_posterior_penalty_cannot_outrank_correctness_signal() -> None:
    # Posterior labels cannot change intrinsic ordering: high-correctness rejected work still
    # outranks zero-correctness accepted work.
    good_rejected = cast(PosteriorBreakdown,
                         score_trajectory(ScoringInputs([{"verdict": "consistent"}], True, None),
                                          pr_feedback="rejected"))
    bad_accepted = cast(PosteriorBreakdown,
                        score_trajectory(ScoringInputs([{"verdict": "contradicts"}], True, None),
                                         pr_feedback="accepted"))
    assert good_rejected.composite is not None and bad_accepted.composite is not None
    assert good_rejected.composite > bad_accepted.composite
    good_unlabeled = score_trajectory(ScoringInputs([{"verdict": "consistent"}], True, None))
    assert good_rejected.composite == good_unlabeled.composite
    assert good_rejected.posterior_cost == 0.5  # max(0, 1.0 − 0.5); lives beside the composite


def test_score_trajectory_does_no_io(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(builtins, "open", lambda *a, **k: (_ for _ in ()).throw(AssertionError("I/O!")))
    rb = score_trajectory(ScoringInputs([{"verdict": "consistent"}], True, 100), pr_feedback="rejected")
    assert rb.composite is not None   # ran purely, no file access


def test_overrides_fingerprint_stably() -> None:
    assert _weights_fingerprint(RewardWeights(w_fp=0.5)) == _weights_fingerprint(RewardWeights(w_fp=0.5))
    assert _weights_fingerprint(RewardWeights(w_fp=0.5)) != _weights_fingerprint(RewardWeights(w_fp=0.6))
    assert len(_weights_fingerprint(RewardWeights(w_fp=0.5))) == 8

def test_posterior_cost_is_absolute_surprise_from_prior() -> None:
    inp = ScoringInputs([{"verdict": "consistent"}], True, 4000)
    chronic = cast(PosteriorBreakdown,
                   score_trajectory(inp, pr_feedback="rejected", outcome_prior=0.8, outcome_prior_n=12))
    generous = cast(PosteriorBreakdown,
                    score_trajectory(inp, pr_feedback="rejected", outcome_prior=0.2, outcome_prior_n=12))
    assert chronic.posterior_cost == pytest.approx(0.2)   # abs(1.0 − 0.8)
    assert generous.posterior_cost == pytest.approx(0.8)  # abs(1.0 − 0.2)
    assert generous.posterior_cost > chronic.posterior_cost          # de-bias direction
    assert chronic.outcome_prior == 0.8 and chronic.outcome_prior_n == 12
    # Two-sided: accepted (0.0) with prior 0.5 is also a surprise — harsh reviewer being lenient
    acc = cast(PosteriorBreakdown, score_trajectory(inp, pr_feedback="accepted", outcome_prior=0.5, outcome_prior_n=10))
    assert acc.posterior_cost == pytest.approx(0.5)        # abs(0.0 − 0.5)
    none_prior = cast(PosteriorBreakdown, score_trajectory(inp, pr_feedback="rejected"))
    assert none_prior.outcome_prior is None                # audit shows uncalibrated
    assert none_prior.posterior_cost == pytest.approx(0.5) # abs(1.0 − 0.5 default prior)

def test_reward_version_stamp_default_vs_custom() -> None:
    inp = ScoringInputs([{"verdict": "consistent"}], True, 4000)
    assert score_trajectory(inp).reward_version == REWARD_VERSION
    custom = RewardWeights(w_fp=0.5)
    rb = score_trajectory(inp, weights=custom)
    assert rb.reward_version == f"{REWARD_VERSION}+custom-{_weights_fingerprint(custom)}"
    assert rb.to_dict()["reward_version"].startswith(f"{REWARD_VERSION}+custom-")
