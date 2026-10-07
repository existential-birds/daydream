"""Review prompts for review and fix phases."""

import json
from pathlib import Path

from daydream import review_profile as _rp
from daydream.phases.inputs import _PR_BODY_MAX_CHARS
from daydream.prompt_budget import (
    inline_context_file,
)
from daydream.prompts.authorial_intent import (
    AUTHORITATIVE_INTENT_BLOCK,
)
from daydream.prompts.grounding import UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY
from daydream.severity import SEVERITY_RUBRIC


def _confidence_and_convention_instructions(*, stage_scoped: bool = False) -> str:
    """Shared confidence, convention, and error-handling rules, appended after Exploration Context."""
    confidence = (
        "For a confirmed candidate's finding, set `confidence` and `rationale`:\n"
        "- HIGH: directly verified by completed source evidence in this review or relevant admitted "
        "evidence. Name the precise source location and applicable convention or dependency.\n"
        "- MEDIUM: supported by concrete source evidence with remaining uncertainty about impact.\n\n"
        if stage_scoped else
        "For every issue you report, you MUST set `confidence` and `rationale`:\n"
        "- HIGH: directly verified by a specific entry in the Exploration Context above. "
        "Your rationale MUST name the specific Dependency edge, Convention entry, or "
        "affected file that supports the issue.\n"
        "- MEDIUM: consistent with the Exploration Context but not pinned to a specific entry.\n\n"
    )
    grounding = (
        "Only report candidates grounded in completed source evidence for the assigned work. "
        "Supporting context can explain intent and conventions, but does not replace source evidence. "
        if stage_scoped else
        "You are reviewing AI-generated code. Be strict. Only report an issue you can ground "
        "in evidence — the diff itself or a specific Exploration Context entry. "
    )
    scope = (
        "Apply the following judgment rules only to this stage's assigned targets or candidates. "
        "Convention and canonical-helper checks are bounded support for those concrete concerns; "
        "they do not assign a new audit.\n\n" if stage_scoped else ""
    )
    return (
        scope + "## Confidence and Convention Rules\n\n" + confidence +
        "Convention handling has TWO distinct cases — do not conflate them:\n"
        "1. Before proposing a fix, check it against the Codebase Conventions section. "
        "If your fix would violate a convention, DROP IT — do not include it.\n"
        "2. If the reviewed code itself violates a convention, that IS the issue. "
        "flag it as HIGH confidence and cite the convention by name in `rationale`.\n\n"
        + grounding + "If you cannot "
        "point to what proves the issue is real, do not emit it. Do not pad the review with "
        "speculative or 'might-be' findings.\n\n"
        "## Error Handling Semantics (QUAL-04)\n\n"
        "Not all caught-and-logged errors are bugs. Before flagging error handling as\n"
        "an issue, classify the operation's criticality:\n\n"
        "- **Critical path**: The caller NEEDS this result to proceed (e.g., loading\n"
        "  config, connecting to database, parsing user input). Swallowing errors here\n"
        "  IS a bug — flag it.\n"
        "- **Best-effort / diagnostic**: The operation is non-essential (e.g., writing\n"
        "  telemetry, flushing debug traces, updating timestamps, sending analytics).\n"
        "  Logging a warning and continuing is the CORRECT pattern — it prevents a\n"
        "  secondary failure from masking or killing the primary operation.\n\n"
        "A `warn!()` + continue after a non-critical operation is intentional graceful\n"
        "degradation, not a 'silent failure.' Changing it to error propagation (`?`,\n"
        "`return Err`, `unwrap`) would make the system MORE fragile, not less.\n\n"
        "When reporting an error handling issue:\n"
        "- State whether the operation is critical-path or best-effort\n"
        "- If best-effort, explain why propagation would be better than logging\n"
        "- If you cannot articulate why the caller benefits from receiving the error,\n"
        "  DROP the finding\n\n"
        "## Refactoring Recommendations\n\n"
        "Before recommending extraction or deduplication (e.g. 'extract a shared\n"
        "helper', 'consolidate duplicated logic'), check the same directory for\n"
        "existing shared modules (shared.ts, utils.ts, helpers.py, common.go, etc.).\n"
        "If shared utilities already exist, the author likely made a deliberate\n"
        "factoring choice. Refactoring recommendations without evidence that shared\n"
        "code doesn't already exist should be MEDIUM confidence at most. Focus on\n"
        "concrete sub-findings (bugs, correctness issues) rather than structural\n"
        "opinions about code organization."
    )


def _dependency_impact_instructions(*, stage_scoped: bool = False) -> str:
    """Prompt language for QUAL-01 cross-file dependency surfacing during review.

    This is an investigation method, not an output section: asking the model to
    prepend a prose section produced incidental text outside the required JSON
    object and was directly implicated in the empty-result extraction defect
    (issue #1445). The heading is kept only as a stable capability label.
    """
    scope = (
        "Apply dependency-impact analysis only to changed symbols in the current assigned files/hunks. "
        "Other paths are supporting context only for concrete candidates in that assigned work:\n"
        if stage_scoped else
        "Apply dependency-impact analysis to every changed symbol listed in the Exploration "
        "Context dependencies above:\n"
    )
    return (
        "## Dependency Impact\n\n"
        + scope +
        "  1. Trace the call chain from each changed symbol through its dependents, so a "
        "defect is judged by what it actually breaks downstream rather than by how its own "
        "body reads.\n"
        "  2. When an individual issue's rationale cites a dependency, include the "
        "file:symbol reference inline within that issue.\n"
        "  This is an investigation method, not extra output: report only substantiated "
        "findings inside the required schema."
    )


def _exploration_pointer(exploration_dir: Path | None, *, fixer: bool = False) -> str:
    """Name exact untrusted exploration files: reviewer summary/index or fixer index only.

    Absent exploration returns an empty string; never expose a directory-wide pointer.
    """
    if exploration_dir is None:
        return ""
    if fixer:
        return (
            f"\n{UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY}\n\n"
            f"Pre-scan exploration indexed this repo — Read {exploration_dir / 'affected_files.md'} "
            "for the structural/import file map before fixing."
        )

    summary = inline_context_file(exploration_dir / "summary.md")
    affected = inline_context_file(exploration_dir / "affected_files.md")
    if summary is not None and affected is not None:
        return (
            f"{UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY}\n\n"
            "Shared exploration context (complete captured artifacts; do not re-read these files):\n"
            + json.dumps({"summary": summary, "affected_files": affected}, ensure_ascii=False)
            + "\nAssigned source files still require same-review reads; this context does not establish clean coverage."
        )
    return (
        f"{UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY}\n\n"
        f"Read the pre-scan summary at {exploration_dir / 'summary.md'} and the "
        f"deterministic structural/import map at {exploration_dir / 'affected_files.md'} "
        "as bounded context for this review. Do not infer or enumerate sibling "
        "artifact files.\n"
        "Assigned source files are different: read the changed hunks in all assigned source files "
        "with the full enclosing symbol or configuration section; expand only as needed "
        "to resolve concrete candidates.\n"
    )


def _settled_decisions_block(prior_commits: str | None) -> str:
    """Mark prior Daydream oneline commits as settled; absent/empty history produces no block."""
    if not prior_commits:
        return ""
    return (
        "Prior automated-review commits on this branch — treat as settled "
        "decisions unless they introduce bugs or security issues:\n"
        f"{prior_commits}"
    )


def build_intent_prompt(
    *,
    strategy: str,
    diff_path: str = "",
    branch: str = "",
    log: str = "",
    exploration_dir: Path | None = None,
    pr_description: str | None = None,
    inline_diff: str | None = None,
    inline_exploration_summary: str | None = None,
) -> str:
    """Ask for author intent using the diff and exploration context."""
    parts: list[str] = []
    if inline_exploration_summary is not None:
        parts.append(
            f"{UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY}\n\n"
            "Pre-scan exploration summary (inlined because the on-disk "
            "exploration files are not guaranteed to exist in this execution "
            "context — treat it as the complete exploration context):\n"
            f"{inline_exploration_summary.rstrip()}\n"
        )
    else:
        pointer = _exploration_pointer(exploration_dir)
        if pointer:
            parts.append(pointer)
        elif inline_diff is not None:
            # No exploration pointer to carry the untrusted boundary; the inlined
            # diff is itself repository-controlled content, so guard it directly.
            parts.append(UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY)
    if pr_description and pr_description.strip():
        body_text = pr_description.strip()
        if len(body_text) > _PR_BODY_MAX_CHARS:
            body_text = body_text[:_PR_BODY_MAX_CHARS] + "\n[PR description truncated]"
        # Neutralize the delimiter so a body containing it can't break containment.
        safe_body = body_text.replace("<pr_description>", "&lt;pr_description>").replace(
            "</pr_description>", "&lt;/pr_description>"
        )
        parts.append(
            f"The author supplied the following pull-request description. "
            f"{AUTHORITATIVE_INTENT_BLOCK}\n\n"
            "Pull request description:\n"
            "<pr_description>\n"
            f"{safe_body}\n"
            "</pr_description>\n"
        )
    # The host replaces diff-reading instructions when inlining while preserving
    # the strategy's judgment tail.
    non_inline_body = strategy.format(diff_path=diff_path)
    if inline_diff is not None:
        inline_prefix = (
            "The complete diff under review is inlined below (do NOT re-Read "
            "it from disk — it is already here):\n\n"
            f"{inline_diff.rstrip()}\n\n"
            "You have full access to explore the codebase. Examine it alongside "
            "the diff above to understand the intent of these changes. "
        )
        _, _, judgment = non_inline_body.partition(_rp.INTENT_STRATEGY_JUDGMENT_MARKER)
        body = inline_prefix + (judgment or non_inline_body)
    else:
        body = non_inline_body
    body += (
        f"\n\nBranch: {branch}\n\n"
        f"Commit log:\n{log}\n"
    )
    parts.append(body)
    return "\n".join(parts)


def build_alternative_review_prompt(
    *,
    strategy: str,
    intent_summary: str = "",
    diff_path: str = "",
    exploration_dir: Path | None = None,
    inline_diff: str | None = None,
) -> str:
    """Fill strategy placeholders and use inline diff when supplied, otherwise its file pointer."""
    parts: list[str] = []
    pointer = _exploration_pointer(exploration_dir)
    if pointer:
        parts.append(pointer)
    elif inline_diff is not None:
        # No exploration pointer to carry the untrusted boundary; the inlined
        # diff is itself repository-controlled content, so guard it directly.
        parts.append(UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY)
    parts.append(_confidence_and_convention_instructions())
    strategy_filled = strategy.format(intent_summary=intent_summary, diff_path=diff_path)
    if inline_diff is not None:
        # Replace only the non-inline diff clause, preserving the judgment tail.
        prefix = (
            f"The intent of this PR has been confirmed as:\n\n"
            f"{intent_summary}\n\n"
            f"Given this intent, explore the codebase and evaluate the implementation "
            f"in the diff inlined below (do NOT re-Read "
            f"the diff from disk — it is already here):\n\n"
            f"{inline_diff.rstrip()}\n\n"
        )
        _, marker, tail = strategy_filled.partition(_rp.ALTERNATIVES_STRATEGY_JUDGMENT_MARKER)
        body = prefix + (marker + tail if marker else strategy_filled)
    else:
        body = strategy_filled
    parts.append(body + "\n")
    parts.append(SEVERITY_RUBRIC)
    return "\n".join(parts)
