"""Test the core rubric, learned Stage-0 outcome term, and CR-Bench false-positive penalty."""

from __future__ import annotations

from typing import Any, Literal, cast

import pytest

from daydream.training.labeler_signals import (
    CommentResolutionSignal,
    FixAppliedSignal,
    LocalCommitAppliedSignal,
    PerFindingDisposition,
    PerFindingResolution,
    PRMergeSignal,
)
from daydream.training.rubric import (
    REWARD_VERSION_RUBRIC,
    PosteriorSource,
    Rubric,
    RubricV2Breakdown,
    RubricV2Weights,
    _rubric_fingerprint,
    derive_outcome_label,
    derive_per_finding_labels,
    score_review,
)


def _fp_rubric(pr_merge: PRMergeSignal, resolutions: list[PerFindingResolution], source: PosteriorSource = "pr_review"
) -> Rubric:
    # CommentResolutionSignal invariant: unresolved = total - replied.
    replied = sum(r.disposition == "accepted" for r in resolutions)
    return Rubric(pr_merge=pr_merge, fix_applied=FixAppliedSignal("unknown", 0, 0, []),
        comment_resolution=CommentResolutionSignal(len(resolutions), replied, len(resolutions) - replied),
        local_commit_applied=None, posterior_source=source, per_finding_resolutions=resolutions,
    )


def _res(disp: PerFindingDisposition) -> PerFindingResolution:
    return PerFindingResolution(
        fingerprint="a" * 64, comment_id=1 if disp != "missing" else None, disposition=disp, evidence=[],
        evidence_digest="x",
    )

MERGED = PRMergeSignal(True, "2026-01-01T00:00:00Z", state="merged")


def test_run_label_all_accepted_any_merge_state() -> None:
    for pr in (MERGED, PRMergeSignal(False, None, state="open"), PRMergeSignal(False, None, state="closed")):
        assert derive_outcome_label(_fp_rubric(pr, [_res("accepted")])) == "accepted"

def test_run_label_all_rejected_any_merge_state() -> None:
    for pr in (MERGED, PRMergeSignal(False, None, state="closed")):
        assert derive_outcome_label(_fp_rubric(pr, [_res("rejected")])) == "rejected"

def test_run_label_mixed_decisive_is_contested() -> None:
    rub = _fp_rubric(MERGED, [_res("accepted"), _res("rejected")])
    assert derive_outcome_label(rub) == "contested"

def test_run_label_decisive_with_non_decisive_is_contested() -> None:
    rub = _fp_rubric(MERGED, [_res("accepted"), _res("unanswered"), _res("missing")])
    assert derive_outcome_label(rub) == "contested"

@pytest.mark.parametrize(("verdict", "expected"),
    [pytest.param("applied", "accepted", id="applied-is-accepted"),
        pytest.param("rejected", "rejected", id="rejected-is-rejected"),
        pytest.param("unknown", "unknown", id="other-verdict-is-unknown"),
    ],
)
def test_run_label_local_branch(verdict: Literal["applied", "rejected", "unknown"], expected: str) -> None:
    assert derive_outcome_label(_local_rubric(verdict)) == expected


def _local_rubric(verdict: Literal["applied", "rejected", "unknown"]) -> Rubric:
    """Local-branch posterior: the commit-applied verdict is the only varied signal."""
    return Rubric(pr_merge=PRMergeSignal(False, None), fix_applied=FixAppliedSignal("unknown", 0, 0, []),
        comment_resolution=CommentResolutionSignal(0, 0, 0), local_commit_applied=LocalCommitAppliedSignal(verdict),
        posterior_source="local_branch",
    )


def test_run_label_no_signal_is_unknown() -> None:
    rub = _fp_rubric(MERGED, [_res("accepted")], source="none")
    assert derive_outcome_label(rub) == "unknown"


def test_per_finding_non_pr_source_stays_unknown() -> None:
    rub = _fp_rubric(MERGED, [], source="local_branch")
    assert derive_per_finding_labels(rub, [_res("accepted")]) == ["unknown"]

class _StubModel:

    def score_comment(self, text: str) -> float:  # noqa: ARG002
        return 0.5


@pytest.fixture
def model() -> _StubModel:
    return _StubModel()


def _finding(text: str = "nit", fid: str | None = None, **extra: Any) -> dict[str, Any]:
    f: dict[str, Any] = {"id": fid or text, "text": text}
    f.update(extra)
    return f


def test_all_noise_scores_below_clean_with_same_recall(model: _StubModel) -> None:
    noise = cast(
        float, score_review(model, findings=[_finding("vague") for _ in range(5)], fp_count=5, total_findings=5),
    )
    clean = cast(float,
        score_review(model, findings=[_finding("real bug, line 3") for _ in range(5)], fp_count=0, total_findings=5),
    )
    assert noise < clean  # M5: strictly below, same recall, no noise

def test_fp_penalty_term_present_and_dominant_direction(model: _StubModel) -> None:
    w = RubricV2Weights()
    assert w.w_false_positive > 0
    b = cast(
        RubricV2Breakdown, score_review(model, findings=[_finding()], fp_count=3, total_findings=3, breakdown=True),
    )
    fp_term = b.terms["fp_penalty"]
    assert b.false_positive_penalty is not None and b.false_positive_penalty > 0
    assert fp_term is not None and fp_term < 0  # explicit subtractive term

def test_version_fingerprint_changes_on_weight_change() -> None:
    f1 = _rubric_fingerprint(RubricV2Weights())
    f2 = _rubric_fingerprint(RubricV2Weights(w_false_positive=0.5))
    assert f1 != f2  # M7 discipline: formula identity detectable

def test_missing_correctness_is_none_not_zero(model: _StubModel) -> None:
    b = cast(
        RubricV2Breakdown, score_review(model, findings=[_finding()], fp_count=0, total_findings=1, breakdown=True),
    )
    assert b.terms["intrinsic_composite"] is None
    assert b.reward_version.startswith(REWARD_VERSION_RUBRIC)
    assert "localization" not in b.terms
    assert "tool_grounded" not in b.terms

def test_zero_total_findings_guards_fp_and_snr_terms(model: _StubModel) -> None:
    # At zero findings, ratio terms stay None without dividing by zero.
    b = cast(
        RubricV2Breakdown, score_review(model, findings=[_finding()], fp_count=0, total_findings=0, breakdown=True),
    )
    assert b.false_positive_penalty is None
    assert b.terms["fp_penalty"] is None  # renormalized out, not imputed 0.0
    assert b.signal_to_noise is None
    assert b.composite == 0.5  # learned term alone renormalized

def test_malformed_finding_raises_with_id(model: _StubModel) -> None:
    with pytest.raises(ValueError, match="f-9"):
        score_review(model, findings=[{"id": "f-9"}], fp_count=0, total_findings=1)
