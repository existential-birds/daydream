"""Render findings and authorized review payloads with captured extension functions."""
from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from daydream.extensions import CommentFinding, FindingRenderContext, Registry, SummaryContext, SummaryFinding
from daydream.review_budget import render_review_warnings
from daydream.reviews.identity import DAYDREAM_FOOTER, finding_marker
from daydream.reviews.models import ClassifiedIssues, ParsedIssue, PRInfo, ReviewEvent, ReviewPayload
from daydream.severity import model_facing_levels

_logger = logging.getLogger(__name__)


_SEVERITY_EMOJI: dict[str, str] = {
    "high": "⚠️",
    "medium": "🔵",
    "low": "💡",
}


def _severity_emoji(severity: str | None) -> str:
    """Map a severity level to an emoji prefix."""
    if not severity:
        return ""
    return _SEVERITY_EMOJI.get(severity.lower(), "")


def _issue_header(issue: CommentFinding, *, prefix: str = "", always_bold: bool = False) -> str:
    """Compose the emoji/title/tag header line for one issue."""
    emoji = _severity_emoji(issue.severity)
    title_prefix = f"{emoji} " if emoji else ""
    header = f"{title_prefix}**{prefix}{issue.title}**" if issue.title or always_bold else ""
    tags = _format_tag_line(issue)
    if header and tags:
        return f"{header} | {tags}"
    return header or tags


def default_render_finding(finding: CommentFinding, ctx: FindingRenderContext) -> str:
    """Render a finding's human block; the host adds identity markers and footer."""
    if ctx.placement == "inline":
        parts = [p for p in (_issue_header(finding), finding.body) if p]
        parts.append(_build_agent_prompt(finding))
        return "\n\n".join(parts)
    prefix = "[cross-stack] " if finding.is_cross_stack else ""
    if ctx.placement == "summary":
        summary_parts = [_issue_header(finding, prefix=prefix, always_bold=True)]
        if finding.body:
            summary_parts.append(f"\n{finding.body}\n")
        summary_parts.append(_build_agent_prompt(finding))
        return "\n".join(summary_parts)
    # file_level (default placement).
    header = _issue_header(finding, prefix=prefix, always_bold=True)
    parts = [p for p in (header, finding.body) if p]
    parts.append(_build_agent_prompt(finding))
    return "\n\n".join(parts)


def _comment_finding(issue: ParsedIssue) -> CommentFinding:
    """Map an internal :class:`ParsedIssue` to the public :class:`CommentFinding`."""
    return CommentFinding(
        path=issue.path,
        line=issue.line,
        title=issue.title,
        body=issue.body,
        is_cross_stack=issue.is_cross_stack,
        severity=issue.severity,
        confidence=issue.confidence,
        fingerprint=issue.fingerprint,
    )


def _render_component(render: Callable[[], object], fallback: Callable[[], str], label: str) -> str:
    """Use the fallback when a custom renderer fails or produces empty/non-string output."""
    try:
        result = render()
    except Exception as exc:  # noqa: BLE001 - any fork error degrades to the default
        _logger.warning("%s renderer failed (%s); using default", label, exc)
        return fallback()
    if not isinstance(result, str) or not result:
        _logger.warning("%s renderer failed (returned %r); using default", label, result)
        return fallback()
    return result


def _render_finding(issue: ParsedIssue, placement: str, renderers: ReviewRenderers) -> str:
    """Render a finding through the captured extension, with its built-in fallback."""
    cf = _comment_finding(issue)
    ctx = FindingRenderContext(placement=placement)
    _fn = renderers.finding
    _label = "builtin" if _fn is default_render_finding else "custom"
    return _render_component(
        lambda: _fn(cf, ctx),
        lambda: renderers.fallback_finding(cf, ctx),
        f"{_label} 'finding'",
    )


def format_comment_body(issue: ParsedIssue, kind: str, renderers: ReviewRenderers) -> str:
    """Append host-owned finding identity to rendered inline or file comments."""
    parts = [_render_finding(issue, kind, renderers), DAYDREAM_FOOTER]
    if issue.fingerprint:
        parts.append(finding_marker(issue.fingerprint))
    return "\n\n".join(parts).strip()


def _format_tag_line(issue: CommentFinding) -> str:
    """Render severity/confidence badges for a single issue, if set."""
    bits: list[str] = []
    if issue.severity:
        bits.append(f"severity: `{issue.severity}`")
    if issue.confidence:
        bits.append(f"confidence: `{issue.confidence}`")
    return " · ".join(bits)


def _build_agent_prompt(issue: CommentFinding) -> str:
    """Build a collapsible AI-agent-friendly prompt for a single issue."""
    loc = f"`{issue.path}`"
    if issue.line:
        loc += f" around line {issue.line}"
    instruction = issue.title
    if issue.body:
        # First meaningful body line as added context.
        first_line = issue.body.strip().split("\n")[0].strip()
        if first_line and first_line != issue.title:
            instruction = f"{instruction}: {first_line}" if instruction else first_line
    return (
        "<details>\n"
        "<summary>🔮 Prompt for AI Agents</summary>\n\n"
        "```\n"
        "Verify each finding against the current code and only fix it if needed.\n\n"
        f"In {loc}, {instruction}\n"
        "```\n\n"
        "</details>"
    )


def _summary_body_block(issue: ParsedIssue, renderers: ReviewRenderers) -> str:
    """Append host-owned identity to the summary renderer's finding block."""
    block = _render_finding(issue, "summary", renderers)
    if issue.fingerprint:
        block = f"{block}\n{finding_marker(issue.fingerprint)}"
    return block


def _render_body_section(findings: tuple[SummaryFinding, ...]) -> str:
    """Group intact finding blocks by file in collapsible sections."""
    if not findings:
        return ""
    grouped: dict[str, list[SummaryFinding]] = {}
    for sf in findings:
        grouped.setdefault(sf.finding.path, []).append(sf)
    total = len(findings)
    parts: list[str] = [
        "<details>",
        f"<summary>📋 Non-inline findings ({total})</summary><blockquote>\n",
    ]
    for filepath, file_findings in grouped.items():
        parts.append("<details>")
        parts.append(
            f"<summary>{filepath} ({len(file_findings)})</summary><blockquote>\n"
        )
        for i, sf in enumerate(file_findings):
            parts.append(sf.body_block)
            if i < len(file_findings) - 1:
                parts.append("\n---\n")
        parts.append("\n</blockquote></details>")
    parts.append("\n</blockquote></details>")
    return "\n".join(parts)


def _summary_findings(body_only: list[ParsedIssue], renderers: ReviewRenderers) -> tuple[SummaryFinding, ...]:
    """Map non-inline :class:`ParsedIssue` objects to public :class:`SummaryFinding`s."""
    return tuple(
        SummaryFinding(finding=_comment_finding(issue), body_block=_summary_body_block(issue, renderers))
        for issue in body_only
    )


def default_render_summary(ctx: SummaryContext) -> str:
    """Render summary, diagrams, grouped findings, agent prompt, and review info.

    The host owns the approval line, review event, and final footer."""
    chunks: list[str] = ["**Code Review Summary**"]
    if ctx.diagrams:
        chunks.append(ctx.diagrams)
    section = _render_body_section(ctx.findings)
    if section:
        chunks.append(section)
    if ctx.agent_prompt:
        chunks.append(ctx.agent_prompt)
    chunks.append(ctx.review_info)
    return "\n\n".join(chunks)


@dataclass(frozen=True)
class ReviewRenderers:
    """Resolved comment renderers and their explicit built-in fallbacks."""

    finding: Callable[[CommentFinding, FindingRenderContext], str]
    summary: Callable[[SummaryContext], str]
    fallback_finding: Callable[[CommentFinding, FindingRenderContext], str] = default_render_finding
    fallback_summary: Callable[[SummaryContext], str] = default_render_summary


def resolve_review_renderers(registry: Registry) -> ReviewRenderers:
    """Capture the run's renderer selection before payload assembly."""
    return ReviewRenderers(
        finding=registry.renderer("finding"),
        summary=registry.renderer("summary"),
    )


def _render_summary(ctx: SummaryContext, renderers: ReviewRenderers) -> str:
    """Render the summary through the captured extension, with its built-in fallback."""
    return _render_component(
        lambda: renderers.summary(ctx),
        lambda: renderers.fallback_summary(ctx),
        "custom 'summary'",
    )


def _count_labels(
    issues: list[ParsedIssue], attr: str, order: tuple[str, ...]
) -> list[str]:
    """Return ordered `N LABEL` strings for non-empty counts."""
    counts = Counter(getattr(issue, attr) for issue in issues)
    return [f"{counts[key]} {key}" for key in order if counts[key]]


def _build_consolidated_prompt(
    classified: ClassifiedIssues,
    pr: PRInfo,
) -> str:
    """Build a single collapsible prompt block that tells AI agents to fetch and fix review comments."""
    total = classified.total()

    prompt_body = (
        f"Fix the {total} review comment(s) posted on this PR.\n"
        "\n"
        "Fetch the comments manually:\n"
        f"1. gh api repos/{pr.owner}/{pr.repo}/pulls/{pr.number}/comments\n"
        f"2. gh api repos/{pr.owner}/{pr.repo}/issues/{pr.number}/comments\n"
        "\n"
        "These endpoints return all comments on the PR. Focus on the most\n"
        "recent review — ignore older review threads that have already been\n"
        "addressed. For each comment: read the referenced file, verify the\n"
        "finding against the current code, and fix it if valid. Skip false\n"
        "positives. Commit all fixes when done."
    )

    return (
        "<details>\n"
        "<summary>🔮 Prompt for all review comments with AI agents</summary>\n\n"
        f"```\n{prompt_body}\n```\n\n"
        "</details>"
    )


_REVIEWED_COMMIT_MARKER = "- **Reviewed commit:**"


def _strip_forged_reviewed_commit_lines(run_info: str) -> str:
    """Remove artifact-supplied commit markers; only validated PRInfo may author them."""
    return "\n".join(
        line
        for line in run_info.splitlines()
        if not line.lstrip().startswith(_REVIEWED_COMMIT_MARKER)
    )


def build_payload_for_event(
    pr: PRInfo,
    classified: ClassifiedIssues,
    *,
    event: ReviewEvent,
    run_info: str,
    renderers: ReviewRenderers,
    diagram_blocks: str | None = None,
    review_warnings: tuple[str, ...] = (),
) -> ReviewPayload:
    """Render the authorized review event with host-owned identity, warnings, and footer.

    Diagram blocks pass through SummaryContext. The linked reviewed commit comes only
    from validated PRInfo, using the fork repository when available."""
    all_issues_with_inline_meta = classified.all_issues()

    approved = event is ReviewEvent.APPROVE

    # Consolidated AI agent prompt (host-built; empty means "omit").
    agent_prompt = (
        _build_consolidated_prompt(classified, pr) if not classified.is_empty() else ""
    )

    # Collapsible review info: a linked reviewed-commit line (from
    # ``pr.head_sha``, so readers can tell whether findings apply to the
    # PR's current head), then enriched run-info (rollup + per-phase
    # breakdown + version footer, owned by the renderer), then the
    # conditional severity/confidence breakdown. The renderer emits its own
    # ``<sub>Generated by daydream...</sub>`` footer, so don't double it.
    # The commit link targets the repo that holds the head commit: the fork
    # for fork-head PRs (``head_repo``), else the base repo from
    # ``owner``/``repo``. ``owner``/``repo`` themselves stay the base repo —
    # that is where the review comment posts.
    commit_slug = pr.head_repo or f"{pr.owner}/{pr.repo}"
    commit_line = (
        f"{_REVIEWED_COMMIT_MARKER} [`{pr.head_sha[:7]}`]"
        f"(https://github.com/{commit_slug}/commit/{pr.head_sha})"
    )
    extra_info_lines: list[str] = []
    severity_parts = _count_labels(
        all_issues_with_inline_meta, "severity", model_facing_levels()
    )
    if severity_parts:
        extra_info_lines.append("- **Severity:** " + ", ".join(severity_parts))
    confidence_parts = _count_labels(
        all_issues_with_inline_meta, "confidence", ("HIGH", "MEDIUM", "LOW")
    )
    if confidence_parts:
        extra_info_lines.append("- **Confidence:** " + ", ".join(confidence_parts))
    review_info = f"{commit_line}\n\n{_strip_forged_reviewed_commit_lines(run_info)}"
    if extra_info_lines:
        review_info = f"{review_info}\n\n" + "\n".join(extra_info_lines)
    review_info_block = (
        "<details>\n"
        "<summary>ℹ️ Review info</summary>\n\n"
        f"{review_info}\n\n"
        "</details>"
    )

    summary_ctx = SummaryContext(
        findings=_summary_findings(classified.body_only, renderers),
        agent_prompt=agent_prompt,
        review_info=review_info_block,
        diagrams=diagram_blocks or None,
    )
    summary_body = _render_summary(summary_ctx, renderers)

    body_chunks: list[str] = []
    if review_warnings:
        body_chunks.append(render_review_warnings(review_warnings))
    if approved:
        body_chunks.append("✅ **Deep review passed with no high/medium findings.**")
    body_chunks.append(summary_body)
    # DAYDREAM_FOOTER is the bottom-of-comment "🧙 Posted by daydream"
    # badge — distinct from the renderer's "Generated by daydream" line
    # inside the review-info block.
    body_chunks.append(DAYDREAM_FOOTER)

    return ReviewPayload(
        event=event,
        commit_id=pr.head_sha,
        body="\n\n".join(body_chunks),
        comments=tuple(classified.inline),
    )
