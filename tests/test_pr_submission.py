"""Typed unit coverage for classified PR review submission."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

import daydream.reviews.rendering as review_rendering
from daydream import pr_review
from daydream.pr_review import (
    InlineReviewComment,
    ParsedIssue,
    PRInfo,
    ReviewEvent,
    SubmissionStatus,
    post_classified_review,
)
from daydream.reviews.identity import parse_finding_markers
from daydream.reviews.models import ClassifiedReviewResult, FileCommentPayload, ReviewPayload, ReviewPostResult
from tests.harness.review_profile import sample_pr


def _finding(path: str, title: str, fingerprint: str, *, line: int | None = None, location_distrust: bool = False,
    severity_before_demotion: str | None = None, severity_off_vocabulary: bool = False,
) -> ParsedIssue:
    return ParsedIssue(
        path=path, line=line, title=title, body=f"{title} rationale", is_cross_stack=True, confidence="HIGH",
        severity="low", fingerprint=fingerprint, location_distrust=location_distrust,
        severity_before_demotion=severity_before_demotion, severity_off_vocabulary=severity_off_vocabulary,
    )


def _classified(*, inline: list[InlineReviewComment] | None = None, inline_issues: list[ParsedIssue] | None = None,
    file_level: list[ParsedIssue] | None = None, body_only: list[ParsedIssue] | None = None,
) -> pr_review.ClassifiedIssues:
    return pr_review.ClassifiedIssues(
        inline=[] if inline is None else inline, inline_issues=[] if inline_issues is None else inline_issues,
        file_level=[] if file_level is None else file_level, body_only=[] if body_only is None else body_only,
    )


def _submit(classified: pr_review.ClassifiedIssues, transport: _FakeTransport,
    *, event: ReviewEvent = ReviewEvent.COMMENT,
) -> ClassifiedReviewResult:
    return post_classified_review(
        sample_pr(), classified, event=event,
        run_info="Run information from the authorized caller.",
        renderers=pr_review.ReviewRenderers(
            review_rendering.default_render_finding, review_rendering.default_render_summary,
        ), transport=transport,
    )


class _FakeTransport:
    """A typed write boundary that records each non-idempotent request."""

    def __init__(self, file_results: Sequence[bool], review_result: ReviewPostResult,) -> None:
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
    classified = _classified(file_level=[first, failed])
    transport = _FakeTransport([True, False],
        ReviewPostResult("https://github.com/acme/widgets/pull/42#review", None),
    )
    result = _submit(classified, transport)

    assert [request.path for _pr, request in transport.file_requests] == ["first.py", "failed.py"]
    assert [request.subject_type for _pr, request in transport.file_requests] == ["file", "file"]
    assert len(transport.review_requests) == 1
    final = transport.review_requests[0][1]
    assert final.event is ReviewEvent.COMMENT
    assert final.commit_id == sample_pr().head_sha
    assert final.comments == ()
    assert "Folded file finding" in final.body
    assert parse_finding_markers(final.body) == ["2" * 64]
    assert result.status is SubmissionStatus.POSTED
    assert result.review_url == "https://github.com/acme/widgets/pull/42#review"
    assert result.posted_file_level == (classified.file_level[0],)
    assert result.folded_file_level == (classified.file_level[1],)
    assert result.final_review_posted is True
    assert result.safe_error is None

@pytest.mark.parametrize(("file_result", "expected_posted", "expected_folded"), [(False, 0, 1), (True, 1, 0)],
    ids=["zero-external-file-writes", "partial-file-write"],
)
def test_submission_reports_zero_and_partial_writes_when_final_review_fails(
    file_result: bool, expected_posted: int, expected_folded: int,
) -> None:
    classified = _classified(file_level=[_finding("file.py", "File finding", "3" * 64)])
    transport = _FakeTransport([file_result],
        ReviewPostResult(None, "GitHub review submission failed (request payload preserved at /tmp/review.json)"),
    )
    result = _submit(classified, transport)

    assert len(transport.file_requests) == 1
    assert len(transport.review_requests) == 1
    assert result.status is SubmissionStatus.FAILED
    assert result.final_review_posted is False
    assert len(result.posted_file_level) == expected_posted
    assert len(result.folded_file_level) == expected_folded
    assert result.safe_error == "GitHub review submission failed (request payload preserved at /tmp/review.json)"

def test_submission_captures_before_writes_and_preserves_provenance() -> None:
    issue = _finding(
        "original.py", "Original finding", "4" * 64, location_distrust=True,
        severity_before_demotion="high", severity_off_vocabulary=True,
    )
    inline = InlineReviewComment("inline.py", 7, "RIGHT", "original inline")
    classified = _classified(inline=[inline], file_level=[issue, _finding("second.py", "Second", "6" * 64)])

    class MutatingTransport(_FakeTransport):
        def post_file_comment(self, pr: PRInfo, payload: FileCommentPayload) -> bool:
            if not self.file_requests:
                issue.path = "mutated.py"
                issue.title = "Mutated finding"
                issue.location_distrust = False
                issue.severity_before_demotion = None
                issue.severity_off_vocabulary = False
                classified.inline.clear()
                classified.file_level.clear()
            return super().post_file_comment(pr, payload)

    transport = MutatingTransport(
        [True, True], ReviewPostResult("https://github.com/acme/widgets/pull/42#review", None),
    )
    result = _submit(classified, transport)
    assert [payload.path for _, payload in transport.file_requests] == ["original.py", "second.py"]
    assert transport.review_requests[0][1].comments == (inline,)
    retained = result.posted_file_level[0]
    assert retained.path == "original.py"
    assert retained.title == "Original finding"
    assert retained.location_distrust is True
    assert retained.severity_before_demotion == "high"
    assert retained.severity_off_vocabulary is True

def test_submission_uses_caller_authorized_approve_event() -> None:
    classified = _classified(inline=[InlineReviewComment("inline.py", 4, "RIGHT", "inline")],
        inline_issues=[_finding("inline.py", "Inline finding", "5" * 64, line=4)],
    )
    transport = _FakeTransport([], ReviewPostResult("https://github.com/acme/widgets/pull/42#review", None),)
    result = _submit(classified, transport, event=ReviewEvent.APPROVE)

    assert transport.review_requests[0][1].event is ReviewEvent.APPROVE
    assert transport.review_requests[0][1].comments == (InlineReviewComment("inline.py", 4, "RIGHT", "inline"),)
    assert result.status is SubmissionStatus.POSTED
