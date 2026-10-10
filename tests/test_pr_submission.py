"""Typed unit coverage for classified PR review submission."""

from __future__ import annotations

from daydream import pr_review
from daydream.pr_review import (
    ClassifiedReviewPlan,
    InlineReviewComment,
    ParsedIssue,
    ReviewEvent,
)
from tests.harness.review_profile import sample_pr


def _finding(path: str, title: str, fingerprint: str, *, line: int | None = None, location_distrust: bool = False,
    severity_before_demotion: str | None = None, severity_off_vocabulary: bool = False,
) -> ParsedIssue:
    return ParsedIssue(
        path=path, line=line, title=title, body=f"{title} rationale", is_cross_stack=True, confidence="HIGH",
        severity="low", fingerprint=fingerprint, location_distrust=location_distrust,
        severity_before_demotion=severity_before_demotion, severity_off_vocabulary=severity_off_vocabulary,
    )


def _plan(*, inline: list[InlineReviewComment] | None = None, inline_issues: list[ParsedIssue] | None = None,
    file_level: list[ParsedIssue] | None = None, body_only: list[ParsedIssue] | None = None,
    event: ReviewEvent = ReviewEvent.COMMENT,
) -> ClassifiedReviewPlan:
    classified = pr_review.ClassifiedIssues(
        inline=[] if inline is None else inline, inline_issues=[] if inline_issues is None else inline_issues,
        file_level=[] if file_level is None else file_level, body_only=[] if body_only is None else body_only,
    )
    return ClassifiedReviewPlan.from_classified(
        sample_pr(), classified, event=event, run_info="Run information from the authorized caller.",
        renderers=pr_review.ReviewRenderers(pr_review.default_render_finding, pr_review.default_render_summary,),
    )





def test_submission_plan_is_detached_from_mutable_classification_and_preserves_provenance() -> None:
    issue = _finding(
        "original.py", "Original finding", "4" * 64, location_distrust=True, severity_before_demotion="high",
        severity_off_vocabulary=True,
    )
    inline = InlineReviewComment("inline.py", 7, "RIGHT", "original inline")
    plan = _plan(inline=[inline], file_level=[issue])
    issue.path = "mutated.py"
    issue.title = "Mutated finding"
    issue.location_distrust = False
    issue.severity_before_demotion = None
    issue.severity_off_vocabulary = False

    assert plan.inline == (InlineReviewComment("inline.py", 7, "RIGHT", "original inline"),)
    retained = plan.file_level[0]
    assert retained.path == "original.py"
    assert retained.title == "Original finding"
    assert retained.location_distrust is True
    assert retained.severity_before_demotion == "high"
    assert retained.severity_off_vocabulary is True
