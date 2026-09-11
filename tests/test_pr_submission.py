"""Typed unit coverage for classified PR review submission."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from daydream import pr_review
from daydream.pr_review import (
    ClassifiedReviewPlan,
    FileCommentPayload,
    InlineReviewComment,
    ParsedIssue,
    PRInfo,
    ReviewEvent,
    ReviewPayload,
    ReviewPostResult,
    SubmissionStatus,
    parse_finding_markers,
    post_classified_review,
)


def _pr() -> PRInfo:
    return PRInfo(
        number=42,
        head_sha="head123",
        base_sha="base456",
        base_ref="main",
        head_ref="feature",
        owner="acme",
        repo="widgets",
        url="https://github.com/acme/widgets/pull/42",
    )


def _finding(
    path: str,
    title: str,
    fingerprint: str,
    *,
    line: int | None = None,
    location_distrust: bool = False,
    severity_before_demotion: str | None = None,
    severity_off_vocabulary: bool = False,
) -> ParsedIssue:
    return ParsedIssue(
        path=path,
        line=line,
        title=title,
        body=f"{title} rationale",
        is_cross_stack=True,
        confidence="HIGH",
        severity="low",
        fingerprint=fingerprint,
        location_distrust=location_distrust,
        severity_before_demotion=severity_before_demotion,
        severity_off_vocabulary=severity_off_vocabulary,
    )


def _plan(
    *,
    inline: list[dict[str, object]] | None = None,
    inline_issues: list[ParsedIssue] | None = None,
    file_level: list[ParsedIssue] | None = None,
    body_only: list[ParsedIssue] | None = None,
    event: ReviewEvent = ReviewEvent.COMMENT,
) -> ClassifiedReviewPlan:
    classified = pr_review._ClassifiedIssues(
        inline=[] if inline is None else inline,
        inline_issues=[] if inline_issues is None else inline_issues,
        file_level=[] if file_level is None else file_level,
        body_only=[] if body_only is None else body_only,
    )
    return ClassifiedReviewPlan.from_classified(
        _pr(),
        classified,
        event=event,
        run_info="Run information from the authorized caller.",
        renderers=pr_review.ReviewRenderers(
            pr_review.default_render_finding,
            pr_review.default_render_summary,
        ),
    )


class _FakeTransport:
    """A typed write boundary that records each non-idempotent request."""

    def __init__(
        self,
        file_results: Sequence[bool],
        review_result: ReviewPostResult,
    ) -> None:
        self._file_results = iter(file_results)
        self._review_result = review_result
        self.file_requests: list[tuple[PRInfo, FileCommentPayload]] = []
        self.review_requests: list[tuple[PRInfo, ReviewPayload]] = []

    def post_file_comment(self, pr: PRInfo, payload: FileCommentPayload) -> bool:
        self.file_requests.append((pr, payload))
        return next(self._file_results)

    def post_review(self, pr: PRInfo, payload: ReviewPayload) -> ReviewPostResult:
        self.review_requests.append((pr, payload))
        return self._review_result


def test_submission_posts_file_comments_in_order_folds_failure_and_posts_once() -> None:
    first = _finding("first.py", "First file finding", "1" * 64)
    failed = _finding("failed.py", "Folded file finding", "2" * 64)
    plan = _plan(file_level=[first, failed])
    transport = _FakeTransport(
        [True, False],
        ReviewPostResult("https://github.com/acme/widgets/pull/42#review", None),
    )

    result = post_classified_review(plan, transport=transport)

    assert [request.path for _pr, request in transport.file_requests] == ["first.py", "failed.py"]
    assert [request.subject_type for _pr, request in transport.file_requests] == ["file", "file"]
    assert len(transport.review_requests) == 1
    final = transport.review_requests[0][1]
    assert final.event is ReviewEvent.COMMENT
    assert final.commit_id == plan.pr.head_sha
    assert final.comments == ()
    assert "Folded file finding" in final.body
    assert parse_finding_markers(final.body) == ["2" * 64]
    assert result.status is SubmissionStatus.POSTED
    assert result.review_url == "https://github.com/acme/widgets/pull/42#review"
    assert result.posted_file_level == (plan.file_level[0],)
    assert result.folded_file_level == (plan.file_level[1],)
    assert result.final_review_posted is True
    assert result.safe_error is None


@pytest.mark.parametrize(
    ("file_result", "expected_posted", "expected_folded"),
    [(False, 0, 1), (True, 1, 0)],
    ids=["zero-external-file-writes", "partial-file-write"],
)
def test_submission_reports_zero_and_partial_writes_when_final_review_fails(
    file_result: bool,
    expected_posted: int,
    expected_folded: int,
) -> None:
    plan = _plan(file_level=[_finding("file.py", "File finding", "3" * 64)])
    transport = _FakeTransport(
        [file_result],
        ReviewPostResult(None, "GitHub review submission failed (request payload preserved at /tmp/review.json)"),
    )

    result = post_classified_review(plan, transport=transport)

    assert len(transport.file_requests) == 1
    assert len(transport.review_requests) == 1
    assert result.status is SubmissionStatus.FAILED
    assert result.final_review_posted is False
    assert len(result.posted_file_level) == expected_posted
    assert len(result.folded_file_level) == expected_folded
    assert result.safe_error == "GitHub review submission failed (request payload preserved at /tmp/review.json)"


def test_submission_plan_is_detached_from_mutable_classification_and_preserves_provenance() -> None:
    issue = _finding(
        "original.py",
        "Original finding",
        "4" * 64,
        location_distrust=True,
        severity_before_demotion="high",
        severity_off_vocabulary=True,
    )
    inline = {"path": "inline.py", "line": 7, "side": "RIGHT", "body": "original inline"}
    plan = _plan(inline=[inline], file_level=[issue])

    issue.path = "mutated.py"
    issue.title = "Mutated finding"
    issue.location_distrust = False
    issue.severity_before_demotion = None
    issue.severity_off_vocabulary = False
    inline["body"] = "mutated inline"

    assert plan.inline == (InlineReviewComment("inline.py", 7, "RIGHT", "original inline"),)
    retained = plan.file_level[0]
    assert retained.path == "original.py"
    assert retained.title == "Original finding"
    assert retained.location_distrust is True
    assert retained.severity_before_demotion == "high"
    assert retained.severity_off_vocabulary is True


def test_submission_uses_caller_authorized_approve_event() -> None:
    plan = _plan(
        inline=[{"path": "inline.py", "line": 4, "side": "RIGHT", "body": "inline"}],
        inline_issues=[_finding("inline.py", "Inline finding", "5" * 64, line=4)],
        event=ReviewEvent.APPROVE,
    )
    transport = _FakeTransport(
        [],
        ReviewPostResult("https://github.com/acme/widgets/pull/42#review", None),
    )

    result = post_classified_review(plan, transport=transport)

    assert transport.review_requests[0][1].event is ReviewEvent.APPROVE
    assert transport.review_requests[0][1].comments == (
        InlineReviewComment("inline.py", 4, "RIGHT", "inline"),
    )
    assert result.status is SubmissionStatus.POSTED
