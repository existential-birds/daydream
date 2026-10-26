"""Pin the intrinsic formula, separate posterior penalty axis, and weight overrides."""
from __future__ import annotations

import builtins
import json
from pathlib import Path
from typing import cast

import pytest

from daydream.dataset_scoring import assemble_scoring_inputs
from daydream.deep.render import insert_review_context
from daydream.training.reward import (
    REWARD_VERSION,
    PosteriorBreakdown,
    RewardBreakdown,
    RewardWeights,
    ScoringInputs,
    score_trajectory,
)


@pytest.mark.parametrize('location', ['root', 'deep'])
def test_archived_host_context_does_not_change_review_length_or_reward(tmp_path: Path, location: str) -> None:
    """The archive acquisition used by capture and sealed RL scores findings, retaining human bytes."""
    legacy = ('⚠️ Existing warning.\n\n# Review\n\n## Diagrams\n```mermaid\nA-->B\n```\n\n'
              '## Issues\n1. [api.py:1] ' + 'Existing finding. ' * 240 +
              '\n\n## Held Findings\n2. [api.py:2] Existing held finding.\n')
    deep = tmp_path / 'deep'
    deep.mkdir()
    verdicts = [{'issue_id': 1, 'verdict': 'consistent'}]
    (deep / 'recommendation-verdicts.json').write_text(json.dumps({'verdicts': verdicts}))
    report = (tmp_path if location == 'root' else deep) / 'review-output.md'
    report.write_text(legacy)
    baseline = assemble_scoring_inputs(tmp_path)
    assert baseline.length == len(legacy)
    baseline_reward = score_trajectory(baseline)
    assert baseline_reward.composite is not None and baseline_reward.length_penalty is not None
    sizes = []
    for amount in (16, 3000):
        context = {'scopes': [], 'scopes_omitted': 0,
            'verification': {'status': 'available', 'value': {'remote-ci-verdict.json': {'notes': 'c' * amount}}},
            'author_reported': {'source': 'pull_request_body', 'head_sha': None, 'body': 'a' * amount,
                                'truncated': False}}
        enriched = insert_review_context(legacy, context)
        report.write_text(enriched)
        acquired = assemble_scoring_inputs(tmp_path)
        assert acquired.length == len(legacy) and acquired == baseline
        assert score_trajectory(acquired).to_dict() == baseline_reward.to_dict()
        assert report.read_text() == enriched and '## Review outcome' in enriched
        assert '## Diagrams' in enriched and '## Held Findings' in enriched
        sizes.append(len(enriched))
    assert sizes[0] != sizes[1]


def test_weights_are_overridable_and_change_composite_predictably() -> None:
    base_len = ScoringInputs(verifier_verdicts=[{"verdict": "consistent"}], format_valid=True, length=10000)
    # At length 10000, len_norm saturates at 1.0; only w_len changes.
    assert score_trajectory(base_len).composite == 0.8                          # default w_len=0.2
    assert score_trajectory(base_len, weights=RewardWeights(w_len=0.5)).composite == 0.5

def test_outcome_applies_posterior_penalty_golden() -> None:
    """Expose posterior penalties separately from the intrinsic composite score."""
    inputs = ScoringInputs(verifier_verdicts=[{"verdict": "consistent"}, {"verdict": "uncertain"}],
        format_valid=True, length=4000)
    base = score_trajectory(inputs)
    assert type(base) is RewardBreakdown and not isinstance(base, PosteriorBreakdown)
    assert base.composite == 0.7
    assert "posterior_cost" not in base.to_dict()
    rb = score_trajectory(inputs, pr_feedback="contested")
    assert isinstance(rb, PosteriorBreakdown)
    assert rb.false_positive_penalty == 0.5
    assert rb.axes_present["false_positive"] is True
    assert rb.composite == 0.7        # correctness minus length; posterior remains separate
    assert rb.posterior_cost == 0.0
    assert "posterior_cost" in rb.to_dict()

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
    assert isinstance(acc, PosteriorBreakdown)
    assert (acc.correctness_per_finding, acc.format_valid, acc.length_penalty,
            acc.false_positive_penalty, acc.composite) == ([1.0], True, 0.25, 0.0, 0.95)
    assert acc.posterior_cost == pytest.approx(0.5)        # abs(0.0 − 0.5)
    none_prior = cast(PosteriorBreakdown, score_trajectory(inp, pr_feedback="rejected"))
    assert isinstance(none_prior, PosteriorBreakdown)
    assert (none_prior.false_positive_penalty, none_prior.axes_present["false_positive"],
            none_prior.composite) == (1.0, True, 0.95)
    assert "posterior_cost" in none_prior.to_dict()
    assert none_prior.outcome_prior is None                # audit shows uncalibrated
    assert none_prior.posterior_cost == pytest.approx(0.5) # abs(1.0 − 0.5 default prior)

def test_reward_version_stamp_default_vs_custom() -> None:
    inp = ScoringInputs([{"verdict": "consistent"}], True, 4000)
    assert score_trajectory(inp).reward_version == REWARD_VERSION
    custom = RewardWeights(w_fp=0.5)
    rb = score_trajectory(inp, weights=custom)
    assert rb.reward_version == f"{REWARD_VERSION}+custom-3d5fd246"
    assert rb.to_dict()["reward_version"] == rb.reward_version
    assert score_trajectory(inp, weights=RewardWeights(w_fp=0.5)).reward_version == rb.reward_version
    assert score_trajectory(inp, weights=RewardWeights(w_fp=0.6)).reward_version == f"{REWARD_VERSION}+custom-67af4ab5"
