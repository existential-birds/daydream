"""Test the run outcome label the rubric derives from posterior evidence."""

from __future__ import annotations

from daydream.training.labeler_signals import (
    CommentResolutionSignal,
    FixAppliedSignal,
    PerFindingDisposition,
    PerFindingResolution,
    PRMergeSignal,
)
from daydream.training.rubric import PosteriorSource, Rubric, derive_outcome_label


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
