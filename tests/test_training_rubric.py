"""Tests for :mod:`daydream.training.rubric`."""

from __future__ import annotations

from daydream.training.labeler_signals import (
    CommentResolutionSignal,
    FixAppliedSignal,
    LocalCommitAppliedSignal,
    PerFindingDisposition,
    PerFindingResolution,
    PRMergeSignal,
)
from daydream.training.rubric import (
    PosteriorSource,
    Rubric,
    derive_outcome_label,
    derive_per_finding_labels,
)


def test_rubric_serializes_to_dict_with_pr_source() -> None:
    rub = Rubric(
        pr_merge=PRMergeSignal(True, "2026-01-01T00:00:00Z", state="merged", draft=False),
        fix_applied=FixAppliedSignal("applied", 2, 2, ["c1", "c2"]),
        comment_resolution=CommentResolutionSignal(1, 1, 0),
        local_commit_applied=None,
        posterior_source="pr_review",
    )
    d = rub.to_dict()
    assert d["posterior_source"] == "pr_review"
    assert d["pr_merge"]["merged"] is True
    assert d["pr_merge"]["merged_at"] == "2026-01-01T00:00:00Z"
    assert d["pr_merge"]["state"] == "merged"
    assert d["pr_merge"]["draft"] is False
    assert d["fix_applied"]["hunks_applied"] == 2


def test_rubric_serializes_with_local_source() -> None:
    rub = Rubric(
        pr_merge=PRMergeSignal(False, None),
        fix_applied=FixAppliedSignal("unknown", 0, 0, []),
        comment_resolution=CommentResolutionSignal(0, 0, 0),
        local_commit_applied=LocalCommitAppliedSignal("applied"),
        posterior_source="local_branch",
    )
    assert rub.to_dict()["posterior_source"] == "local_branch"
    assert rub.to_dict()["local_commit_applied"] == {"verdict": "applied"}


def _fp_rubric(
    pr_merge: PRMergeSignal, resolutions: list[PerFindingResolution], source: PosteriorSource = "pr_review"
) -> Rubric:
    # CommentResolutionSignal invariant: unresolved = total - replied.
    replied = sum(r.disposition == "accepted" for r in resolutions)
    return Rubric(
        pr_merge=pr_merge,
        fix_applied=FixAppliedSignal("unknown", 0, 0, []),
        comment_resolution=CommentResolutionSignal(
            len(resolutions),
            replied,
            len(resolutions) - replied,
        ),
        local_commit_applied=None,
        posterior_source=source,
        per_finding_resolutions=resolutions,
    )


def _res(disp: PerFindingDisposition) -> PerFindingResolution:
    return PerFindingResolution(
        fingerprint="a" * 64,
        comment_id=1 if disp != "missing" else None,
        disposition=disp,
        evidence=[],
        evidence_digest="x",
    )


MERGED = PRMergeSignal(True, "2026-01-01T00:00:00Z", state="merged")


def test_run_label_all_accepted_any_merge_state() -> None:
    """All-accepted maps to accepted regardless of merge state (M22/M10)."""
    for pr in (MERGED, PRMergeSignal(False, None, state="open"), PRMergeSignal(False, None, state="closed")):
        assert derive_outcome_label(_fp_rubric(pr, [_res("accepted")])) == "accepted"


def test_run_label_all_rejected_any_merge_state() -> None:
    for pr in (MERGED, PRMergeSignal(False, None, state="closed")):
        assert derive_outcome_label(_fp_rubric(pr, [_res("rejected")])) == "rejected"


def test_run_label_mixed_decisive_is_contested() -> None:
    """Mixed decisive evidence → contested (M9); the amelia#626 shape (M10)."""
    rub = _fp_rubric(MERGED, [_res("accepted"), _res("rejected")])
    assert derive_outcome_label(rub) == "contested"


def test_run_label_decisive_with_non_decisive_is_contested() -> None:
    rub = _fp_rubric(MERGED, [_res("accepted"), _res("unanswered"), _res("missing")])
    assert derive_outcome_label(rub) == "contested"


def test_run_label_no_decisive_evidence_is_unknown() -> None:
    for pr in (MERGED, PRMergeSignal(False, None, state="open")):
        for disps in ([], [_res("ambiguous")], [_res("unanswered")], [_res("missing")]):
            assert derive_outcome_label(_fp_rubric(pr, disps)) == "unknown"


def test_run_label_local_branch_unchanged() -> None:
    """The local-branch posterior semantics are byte-stable (Out of Scope)."""
    rub = Rubric(
        pr_merge=PRMergeSignal(False, None),
        fix_applied=FixAppliedSignal("unknown", 0, 0, []),
        comment_resolution=CommentResolutionSignal(0, 0, 0),
        local_commit_applied=LocalCommitAppliedSignal("applied"),
        posterior_source="local_branch",
    )
    assert derive_outcome_label(rub) == "accepted"


def test_run_label_local_branch_rejected() -> None:
    """local_branch with ``rejected`` verdict maps to ``rejected`` (rubric.py:113-116)."""
    rub = Rubric(
        pr_merge=PRMergeSignal(False, None),
        fix_applied=FixAppliedSignal("unknown", 0, 0, []),
        comment_resolution=CommentResolutionSignal(0, 0, 0),
        local_commit_applied=LocalCommitAppliedSignal("rejected"),
        posterior_source="local_branch",
    )
    assert derive_outcome_label(rub) == "rejected"


def test_run_label_local_branch_unknown() -> None:
    """local_branch with any other verdict maps to ``unknown`` (rubric.py:116-117)."""
    rub = Rubric(
        pr_merge=PRMergeSignal(False, None),
        fix_applied=FixAppliedSignal("unknown", 0, 0, []),
        comment_resolution=CommentResolutionSignal(0, 0, 0),
        local_commit_applied=LocalCommitAppliedSignal("unknown"),
        posterior_source="local_branch",
    )
    assert derive_outcome_label(rub) == "unknown"


def test_run_label_no_signal_is_unknown() -> None:
    """posterior_source ``none`` always maps to ``unknown`` (rubric.py:118)."""
    rub = _fp_rubric(MERGED, [_res("accepted")], source="none")
    assert derive_outcome_label(rub) == "unknown"


def test_per_finding_labels_come_from_dispositions() -> None:
    """Per-finding labels pass dispositions through; merge state is irrelevant (M10)."""
    rub = _fp_rubric(PRMergeSignal(False, None, state="closed"), [])
    per = [_res("accepted"), _res("rejected"), _res("ambiguous"), _res("unanswered"), _res("missing")]
    assert derive_per_finding_labels(rub, per) == ["accepted", "rejected", "ambiguous", "unanswered", "missing"]


def test_per_finding_non_pr_source_stays_unknown() -> None:
    rub = _fp_rubric(MERGED, [], source="local_branch")
    assert derive_per_finding_labels(rub, [_res("accepted")]) == ["unknown"]
