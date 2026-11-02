"""Test the core rubric, learned Stage-0 outcome term, and CR-Bench false-positive penalty."""

from __future__ import annotations

from typing import Any, cast

import pytest

from daydream.training.labeler_signals import (
    CommentResolutionSignal,
    FixAppliedSignal,
    PerFindingDisposition,
    PerFindingResolution,
    PRMergeSignal,
)
from daydream.training.rubric import (
    PosteriorSource,
    Rubric,
    RubricV2Breakdown,
    RubricV2Weights,
    _rubric_fingerprint,
    derive_outcome_label,
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




def test_run_label_is_contested_by_non_decisive_and_unknown_without_signal() -> None:
    rub = _fp_rubric(MERGED, [_res("accepted"), _res("unanswered"), _res("missing")])
    assert derive_outcome_label(rub) == "contested"
    rub = _fp_rubric(MERGED, [_res("accepted")], source="none")
    assert derive_outcome_label(rub) == "unknown"


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


def test_version_fingerprint_changes_on_weight_change() -> None:
    f1 = _rubric_fingerprint(RubricV2Weights())
    f2 = _rubric_fingerprint(RubricV2Weights(w_false_positive=0.5))
    assert f1 != f2  # M7 discipline: formula identity detectable

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
