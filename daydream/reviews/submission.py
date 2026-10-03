"""Submit an authorized review through an explicit GitHub write capability.

File comments are posted in order; failed comments are folded into the final
review body. The operation captures findings before any network write.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from daydream import git_ops
from daydream.git_ops import GitError, GitHubAuth
from daydream.reviews.models import (
    ClassifiedIssues,
    ClassifiedReviewResult,
    FileCommentPayload,
    ParsedIssue,
    PRInfo,
    ReviewEvent,
    ReviewPayload,
    ReviewPostResult,
    SubmissionStatus,
)
from daydream.reviews.rendering import ReviewRenderers, build_payload_for_event, format_comment_body


class ReviewTransport(Protocol):
    """Explicit capability for the two kinds of GitHub review writes."""

    def post_file_comment(self, pr: PRInfo, payload: FileCommentPayload) -> bool: ...

    def post_review(self, pr: PRInfo, payload: ReviewPayload) -> ReviewPostResult: ...



def _file_comment_payload_dict(payload: FileCommentPayload) -> dict[str, Any]:
    return asdict(payload)



def _review_payload_dict(payload: ReviewPayload) -> dict[str, Any]:
    return {
        "event": payload.event.value,
        "commit_id": payload.commit_id,
        "body": payload.body,
        "comments": [asdict(comment) for comment in payload.comments],
    }



@dataclass(frozen=True)
class GitHubReviewTransport:
    """GitHub review writes bound to one repository checkout and auth source."""

    target_dir: Path
    auth: GitHubAuth = field(repr=False)

    def post_file_comment(self, pr: PRInfo, payload: FileCommentPayload) -> bool:
        endpoint = f"/repos/{pr.owner}/{pr.repo}/pulls/{pr.number}/comments"
        try:
            git_ops.gh_api(
                self.target_dir,
                endpoint,
                method="POST",
                input_data=_file_comment_payload_dict(payload),
                auth=self.auth,
            )
        except GitError:
            return False
        return True

    def post_review(self, pr: PRInfo, payload: ReviewPayload) -> ReviewPostResult:
        endpoint = f"/repos/{pr.owner}/{pr.repo}/pulls/{pr.number}/reviews"
        try:
            data = git_ops.gh_api(
                self.target_dir,
                endpoint,
                method="POST",
                input_data=_review_payload_dict(payload),
                auth=self.auth,
            )
        except GitError as exc:
            safe_error = "GitHub review submission failed"
            if exc.preserved_payload_path is not None:
                safe_error += (
                    " (request payload preserved at "
                    f"{exc.preserved_payload_path})"
                )
            return ReviewPostResult(review_url=None, safe_error=safe_error)
        if not isinstance(data, dict):
            return ReviewPostResult(review_url=None, safe_error=None)
        url = data.get("html_url")
        return ReviewPostResult(
            review_url=str(url) if url else None,
            safe_error=None,
        )



def post_classified_review(
    pr: PRInfo,
    classified: ClassifiedIssues,
    *,
    event: ReviewEvent,
    run_info: str,
    renderers: ReviewRenderers,
    transport: ReviewTransport,
    diagram_blocks: str | None = None,
    review_warnings: tuple[str, ...] = (),
) -> ClassifiedReviewResult:
    """Capture authorized findings before ordered writes, fold failures, then post one review."""
    classified = ClassifiedIssues(
        inline=list(classified.inline),
        inline_issues=[replace(issue) for issue in classified.inline_issues],
        file_level=[replace(issue) for issue in classified.file_level],
        body_only=[replace(issue) for issue in classified.body_only],
    )
    posted: list[ParsedIssue] = []
    folded: list[ParsedIssue] = []
    for finding in classified.file_level:
        payload = FileCommentPayload(
            commit_id=pr.head_sha,
            path=finding.path,
            subject_type="file",
            body=format_comment_body(finding, "file_level", renderers),
        )
        if transport.post_file_comment(pr, payload):
            posted.append(finding)
        else:
            folded.append(finding)

    classified.file_level = posted
    classified.body_only.extend(folded)
    review_payload = build_payload_for_event(
        pr,
        classified,
        event=event,
        run_info=run_info,
        renderers=renderers,
        diagram_blocks=diagram_blocks,
        review_warnings=review_warnings,
    )
    review_result = transport.post_review(pr, review_payload)
    posted_review = review_result.review_url is not None
    return ClassifiedReviewResult(
        status=SubmissionStatus.POSTED if posted_review else SubmissionStatus.FAILED,
        review_url=review_result.review_url,
        posted_file_level=tuple(posted),
        folded_file_level=tuple(folded),
        final_review_posted=posted_review,
        safe_error=None if posted_review else review_result.safe_error,
    )
