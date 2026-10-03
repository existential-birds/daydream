"""Submit an authorized review through an explicit GitHub write capability.

File comments are posted in order; failed comments are folded into the final
review body. A plan snapshots findings before any network write.
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
    InlineReviewComment,
    ParsedIssue,
    PRInfo,
    ReviewEvent,
    ReviewPayload,
    ReviewPostResult,
    SubmissionStatus,
)
from daydream.reviews.rendering import ReviewRenderers, build_payload_for_event, format_comment_body


@dataclass(frozen=True)
class ClassifiedReviewPlan:
    """Immutable, authorized input to the shared review write operation."""

    pr: PRInfo
    inline: tuple[InlineReviewComment, ...]
    inline_issues: tuple[ParsedIssue, ...]
    file_level: tuple[ParsedIssue, ...]
    body_only: tuple[ParsedIssue, ...]
    event: ReviewEvent
    run_info: str
    renderers: ReviewRenderers
    diagram_blocks: str | None
    review_warnings: tuple[str, ...] = ()

    @classmethod
    def from_classified(
        cls,
        pr: PRInfo,
        classified: ClassifiedIssues,
        *,
        event: ReviewEvent,
        run_info: str,
        renderers: ReviewRenderers,
        diagram_blocks: str | None = None,
        review_warnings: tuple[str, ...] = (),
    ) -> ClassifiedReviewPlan:
        """Snapshot a mutable classified review after the caller authorizes it."""
        return cls(
            pr=pr,
            inline=tuple(classified.inline),
            inline_issues=tuple(replace(issue) for issue in classified.inline_issues),
            file_level=tuple(replace(issue) for issue in classified.file_level),
            body_only=tuple(replace(issue) for issue in classified.body_only),
            event=event,
            run_info=run_info,
            renderers=renderers,
            diagram_blocks=diagram_blocks,
            review_warnings=review_warnings,
        )



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
    plan: ClassifiedReviewPlan,
    *,
    transport: ReviewTransport,
) -> ClassifiedReviewResult:
    """Submit ordered file comments, fold failures, then post one final review."""
    posted: list[ParsedIssue] = []
    folded: list[ParsedIssue] = []
    for finding in plan.file_level:
        payload = FileCommentPayload(
            commit_id=plan.pr.head_sha,
            path=finding.path,
            subject_type="file",
            body=format_comment_body(finding, "file_level", plan.renderers),
        )
        if transport.post_file_comment(plan.pr, payload):
            posted.append(finding)
        else:
            folded.append(finding)

    final_classified = ClassifiedIssues(
        inline=list(plan.inline),
        inline_issues=list(plan.inline_issues),
        file_level=list(posted),
        body_only=[*plan.body_only, *folded],
    )
    review_payload = build_payload_for_event(
        plan.pr,
        final_classified,
        event=plan.event,
        run_info=plan.run_info,
        renderers=plan.renderers,
        diagram_blocks=plan.diagram_blocks,
        review_warnings=plan.review_warnings,
    )
    review_result = transport.post_review(plan.pr, review_payload)
    posted_review = review_result.review_url is not None
    return ClassifiedReviewResult(
        status=SubmissionStatus.POSTED if posted_review else SubmissionStatus.FAILED,
        review_url=review_result.review_url,
        posted_file_level=tuple(posted),
        folded_file_level=tuple(folded),
        final_review_posted=posted_review,
        safe_error=None if posted_review else review_result.safe_error,
    )
