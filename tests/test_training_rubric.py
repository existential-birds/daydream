"""Tests for :mod:`daydream.training.rubric`."""

from __future__ import annotations

from daydream.training.labeler_signals import (
    CommentResolutionSignal,
    FixAppliedSignal,
    LocalCommitAppliedSignal,
    PRMergeSignal,
)
from daydream.training.rubric import Rubric, derive_outcome_label


def test_rubric_serializes_to_dict_with_pr_source() -> None:
    rub = Rubric(
        pr_merge=PRMergeSignal(True, "2026-01-01T00:00:00Z"),
        fix_applied=FixAppliedSignal("applied", 2, 2, ["c1", "c2"]),
        comment_resolution=CommentResolutionSignal(1, 1, 0),
        local_commit_applied=None,
        posterior_source="pr_review",
    )
    d = rub.to_dict()
    assert d["posterior_source"] == "pr_review"
    assert d["pr_merge"]["merged"] is True
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


def test_outcome_label_accepted_when_pr_merged_and_no_unresolved() -> None:
    rub = Rubric(
        pr_merge=PRMergeSignal(True, "2026-01-01T00:00:00Z"),
        fix_applied=FixAppliedSignal("applied", 1, 1, ["c1"]),
        comment_resolution=CommentResolutionSignal(2, 2, 0),
        local_commit_applied=None,
        posterior_source="pr_review",
    )
    assert derive_outcome_label(rub) == "accepted"


def test_outcome_label_contested_when_merged_but_unresolved() -> None:
    rub = Rubric(
        pr_merge=PRMergeSignal(True, "2026-01-01T00:00:00Z"),
        fix_applied=FixAppliedSignal("applied", 1, 1, ["c1"]),
        comment_resolution=CommentResolutionSignal(3, 1, 2),
        local_commit_applied=None,
        posterior_source="pr_review",
    )
    assert derive_outcome_label(rub) == "contested"


def test_outcome_label_rejected_when_pr_closed_unmerged() -> None:
    rub = Rubric(
        pr_merge=PRMergeSignal(False, None),
        fix_applied=FixAppliedSignal("not_applied", 0, 1, []),
        comment_resolution=CommentResolutionSignal(1, 0, 1),
        local_commit_applied=None,
        posterior_source="pr_review",
    )
    assert derive_outcome_label(rub) == "rejected"


def test_outcome_label_accepted_via_local_branch() -> None:
    rub = Rubric(
        pr_merge=PRMergeSignal(False, None),
        fix_applied=FixAppliedSignal("unknown", 0, 0, []),
        comment_resolution=CommentResolutionSignal(0, 0, 0),
        local_commit_applied=LocalCommitAppliedSignal("applied"),
        posterior_source="local_branch",
    )
    assert derive_outcome_label(rub) == "accepted"


def test_outcome_label_rejected_via_local_branch() -> None:
    rub = Rubric(
        pr_merge=PRMergeSignal(False, None),
        fix_applied=FixAppliedSignal("unknown", 0, 0, []),
        comment_resolution=CommentResolutionSignal(0, 0, 0),
        local_commit_applied=LocalCommitAppliedSignal("rejected"),
        posterior_source="local_branch",
    )
    assert derive_outcome_label(rub) == "rejected"


def test_outcome_label_unknown_when_no_signal_at_all() -> None:
    rub = Rubric(
        pr_merge=PRMergeSignal(False, None),
        fix_applied=FixAppliedSignal("unknown", 0, 0, []),
        comment_resolution=CommentResolutionSignal(0, 0, 0),
        local_commit_applied=LocalCommitAppliedSignal("unknown"),
        posterior_source="none",
    )
    assert derive_outcome_label(rub) == "unknown"
