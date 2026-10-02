"""Finding, placement, and submission value objects."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Literal


class PostStatus(Enum):
    """Deep review warns on non-POSTED outcomes; comment mode fails on NO_PR or FAILED."""

    POSTED = "posted"
    NOTHING_TO_POST = "nothing-to-post"
    NO_PR = "no-pr"
    FAILED = "failed"


class ReviewEvent(StrEnum):
    """GitHub review event authorized by the source-specific caller."""

    COMMENT = "COMMENT"
    APPROVE = "APPROVE"


class SubmissionStatus(StrEnum):
    """Whether the final classified review was posted."""

    POSTED = "posted"
    FAILED = "failed"


@dataclass
class ParsedIssue:
    """A finding with a potentially stale line hint and stable cross-run fingerprint.

    Location demotion retains the original severity for the approval gate.
    Off-vocabulary severity is tracked separately from an omitted severity."""

    path: str
    line: int | None
    title: str
    body: str
    is_cross_stack: bool = False
    confidence: str | None = None
    severity: str | None = None
    fingerprint: str | None = None
    location_distrust: bool = False
    severity_before_demotion: str | None = None
    severity_off_vocabulary: bool = False


@dataclass(frozen=True)
class InlineReviewComment:
    """One immutable inline comment in a final GitHub review payload."""

    path: str
    line: int
    side: Literal["RIGHT"]
    body: str


@dataclass(frozen=True, kw_only=True)
class FileCommentPayload:
    """Payload for one top-level file review comment."""

    commit_id: str
    path: str
    subject_type: Literal["file"] = "file"
    body: str


@dataclass(frozen=True)
class ReviewPayload:
    """Payload for the single final GitHub review."""

    event: ReviewEvent
    commit_id: str
    body: str
    comments: tuple[InlineReviewComment, ...]


@dataclass(frozen=True)
class ReviewPostResult:
    """Transport-level result for the final review write."""

    review_url: str | None
    safe_error: str | None


@dataclass(frozen=True)
class ClassifiedReviewResult:
    """Confirmed write results from the shared submitter.

    ``final_review_posted=False`` means no successful response was confirmed;
    it does not prove that an ambiguous failed request had no remote effect.
    """

    status: SubmissionStatus
    review_url: str | None
    posted_file_level: tuple[ParsedIssue, ...]
    folded_file_level: tuple[ParsedIssue, ...]
    final_review_posted: bool
    safe_error: str | None


@dataclass(frozen=True)
class PRInfo:
    """Base repository posting target and immutable PR head.

    head_repo identifies the fork holding the reviewed commit; it only affects links."""

    number: int
    head_sha: str
    base_sha: str
    base_ref: str
    head_ref: str
    owner: str
    repo: str
    url: str
    # Slug of the PR's head repository (``owner/repo``) — the fork when the
    # PR head lives in a fork. Used ONLY for the reviewed-commit link, which
    # must point at the repo that actually holds the head commit; comments
    # always post to the base repo via ``owner``/``repo``.
    head_repo: str | None = None
    # GitHub base tip at initial lookup, distinct from base_sha (diff merge-base).
    pr_base_sha: str | None = None


@dataclass(frozen=True)
class ItemFields:
    """Normalized canonical finding fields, before comment rendering."""

    path: str
    line_int: int | None
    description: str
    rationale: str
    severity: str | None
    confidence: str | None
    is_cross_stack: bool
    location_distrust: bool = False
    severity_before_demotion: str | None = None
    severity_off_vocabulary: bool = False


@dataclass
class ClassifiedIssues:
    inline: list[InlineReviewComment] = field(default_factory=list)
    body_only: list[ParsedIssue] = field(default_factory=list)
    # Parallel list to `inline`: the original ParsedIssue for each inline
    # comment. Used to roll severity/confidence into the summary body.
    inline_issues: list[ParsedIssue] = field(default_factory=list)
    # Findings with no diff-line home whose file is still part of the PR
    # diff. Posted as file-level review comments so they land in
    # `/pulls/{n}/comments` as repliable threads the labeler can read back.
    file_level: list[ParsedIssue] = field(default_factory=list)

    def is_empty(self) -> bool:
        """Check rendered comments as well as file and body placements."""
        return not (self.inline or self.file_level or self.body_only)

    def total(self) -> int:
        """Count of findings across every placement."""
        return len(self.inline) + len(self.file_level) + len(self.body_only)

    def all_issues(self) -> list[ParsedIssue]:
        """Every classified :class:`ParsedIssue`, for severity/confidence rollups."""
        return [*self.inline_issues, *self.file_level, *self.body_only]
