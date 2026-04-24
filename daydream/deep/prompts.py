"""Prompt builders for deep-review mode.

Pure keyword-only functions that assemble prompt strings from context pointers.
All context passes via filesystem paths (D-09) -- no full file contents embedded in
prompts. Per-stack agents only see their own stack's files + TTT context (D-10).
"""

from __future__ import annotations

from pathlib import Path

from daydream.phases import (
    _confidence_and_convention_instructions,
    _dependency_impact_instructions,
    _exploration_pointer,
)

DOC_REVIEW_NOTICE = (
    "[Notice] Dedicated documentation review (beagle-docs) is planned but not yet "
    "implemented.\nThese documentation files are currently being reviewed by the "
    "generic-fallback agent (D-20)."
)


def _context_pointers(
    *,
    intent_path: Path,
    alternatives_path: Path,
) -> str:
    """Reference pointers for TTT stage outputs (D-09/D-19 context bus)."""
    return (
        f"TTT intent summary is at {intent_path}. Read it before starting your review "
        f"so your findings align with the author's stated intent.\n"
        f"TTT alternative-review findings are at {alternatives_path}. Use them as a "
        f"starting point -- you may deepen, confirm, or dismiss each finding with "
        f"language-specific evidence."
    )


def _stack_scope_instruction(stack_name: str, files: list[str]) -> str:
    joined = ", ".join(files)
    return (
        f"You are reviewing the {stack_name} stack. Focus ONLY on these files:\n"
        f"  {joined}\n"
        f"Do NOT review files from other stacks -- their reviews are running in "
        f"parallel and will be merged afterwards."
    )


def _diff_instruction(diff_path: Path, files: list[str]) -> str:
    joined = ", ".join(files)
    # Point agents at diff_path directly. A bare `git diff -- <files>` command
    # only surfaces uncommitted workspace changes; on a clean PR branch it
    # would return empty and hide every committed change. diff_path already
    # contains the full base..HEAD diff.
    return (
        f"The full PR diff (base..HEAD) is at {diff_path}. Read it directly; "
        f"do NOT run `git diff` without a base ref -- on a clean branch that "
        f"returns empty and hides committed changes.\n"
        f"Focus on hunks that touch your stack's files: {joined}."
    )


def build_per_stack_prompt(
    *,
    skill_invocation: str,
    stack_name: str,
    files: list[str],
    diff_path: Path,
    intent_path: Path,
    alternatives_path: Path,
    output_path: Path,
    exploration_dir: Path | None = None,
) -> str:
    """Assemble the per-stack review prompt.

    Args:
        skill_invocation: Beagle skill invocation, e.g. "/beagle-python:review-python".
        stack_name: Lower-case stack key for scope messaging.
        files: Files this stack owns.
        diff_path: Path to the full diff on disk.
        intent_path: Path to TTT intent.md.
        alternatives_path: Path to TTT alternatives.json.
        output_path: Where the agent must write its review.
        exploration_dir: Pre-scan exploration directory (if available).

    Returns:
        Assembled prompt string.
    """
    parts: list[str] = []
    pointer = _exploration_pointer(exploration_dir)
    if pointer:
        parts.append(pointer)
    parts.append(_context_pointers(intent_path=intent_path, alternatives_path=alternatives_path))
    parts.append(_confidence_and_convention_instructions())
    parts.append(_dependency_impact_instructions())
    parts.append(_stack_scope_instruction(stack_name, files))
    parts.append(_diff_instruction(diff_path, files))
    parts.append(skill_invocation)
    parts.append(f"Write your full review to {output_path}.")
    return "\n\n".join(parts)


def build_merge_prompt(
    *,
    per_stack_records_paths: list[Path],
    intent_path: Path,
    alternatives_path: Path,
    dedup_candidates_path: Path,
    output_path: Path,
    exploration_dir: Path | None = None,
    failed_stacks: dict[str, str] | None = None,
) -> str:
    """Assemble the cross-stack merge prompt (D-23..D-27).

    The merged report MUST:
      - live at output_path (single-file contract, D-24/D-42)
      - carry a flat globally-numbered ## Issues list (D-25)
      - have a ## Cross-Stack Issues subsection that CONTINUES the numbering (D-25)
      - prefix every cross-stack title with the literal "[cross-stack]" (D-26, normative)
      - collapse duplicates per dedup candidate adjudication (D-27)

    Args:
        per_stack_records_paths: Parsed per-stack record JSON paths (D-22 inputs).
        intent_path: Path to TTT intent.md.
        alternatives_path: Path to TTT alternatives.json.
        dedup_candidates_path: Path to dedup-candidates.json (D-27 pre-filter output).
        output_path: Where the merge agent must write the unified report (D-24).
        exploration_dir: Pre-scan exploration directory (if available).
        failed_stacks: Optional stack_name -> failure reason for stacks whose
            per-stack agent raised. The merge prompt includes an explicit
            "Uncovered stacks" block so the merge report can call out missing
            coverage instead of silently pretending the run was complete.

    Returns:
        Assembled prompt string.
    """
    records_block = "\n".join(f"  - {p}" for p in per_stack_records_paths)
    parts: list[str] = []
    pointer = _exploration_pointer(exploration_dir)
    if pointer:
        parts.append(pointer)
    parts.append(
        f"TTT intent summary: {intent_path}\n"
        f"TTT alternative-review findings: {alternatives_path}\n"
        f"Dedup pre-filter candidate pairs: {dedup_candidates_path}\n"
        f"Per-stack parsed records:\n{records_block}"
    )
    if failed_stacks:
        failed_block = "\n".join(
            f"  - {name}: {reason}" for name, reason in sorted(failed_stacks.items())
        )
        parts.append(
            "Uncovered stacks (per-stack agent raised; no records available):\n"
            f"{failed_block}\n"
            "In the merged report add a '## Uncovered Stacks' section listing each "
            "stack above. Do NOT silently omit them -- downstream readers must be "
            "able to tell 'no findings' apart from 'this stack never ran'."
        )
    parts.append(
        "You are the cross-stack merge agent. Read every artifact above by path -- "
        "do NOT re-run any reviews. Your only output is the merged markdown report."
    )
    parts.append(
        "Dedup adjudication:\n"
        "  dedup-candidates.json has two sections:\n\n"
        "  record_alt_pairs (record ↔ TTT alt-review):\n"
        "  - For each candidate pair, decide whether the two findings describe the\n"
        "    same concern. If yes, emit ONE entry citing both sources as combined\n"
        "    evidence. If no, emit both entries independently.\n\n"
        "  record_duplicate_pairs (record ↔ record):\n"
        "  - These are per-stack records with near-identical descriptions across\n"
        "    different files (e.g. the same architectural concern reported once per\n"
        "    affected file). When two records describe the same conceptual finding,\n"
        "    emit ONE entry listing all affected files rather than repeating the\n"
        "    finding verbatim for each file.\n\n"
        "  - Concerns that span multiple stacks (contract drift, shared-type "
        "mismatches, API-contract misalignment) go in ## Cross-Stack Issues."
    )
    parts.append(
        "Report format (MANDATORY):\n\n"
        "# Review\n\n"
        "## Per-Stack Context\n"
        "(optional human-readable per-stack summaries; phase_parse_feedback ignores "
        "this section)\n\n"
        "## Issues\n"
        "1. [FILE:LINE] TITLE\n"
        "   rationale / recommendation\n"
        "2. [FILE:LINE] TITLE\n"
        "   ...\n\n"
        "## Cross-Stack Issues\n"
        "<continues the SAME numbering -- do NOT reset to 1>\n"
        "N. [cross-stack] [FILE:LINE] TITLE\n"
        "   rationale citing each per-stack source that contributed\n\n"
        "Rules:\n"
        "  - Every cross-stack title MUST begin with the literal prefix [cross-stack].\n"
        "  - Numbering is flat and global; cross-stack continues per-stack numbering.\n"
        "  - The numbered head line is plain text -- do NOT wrap it in `**...**` or "
        "`__...__` bold markers. Markdown emphasis is fine INSIDE the rationale, "
        "but the `N. [FILE:LINE] TITLE` line itself stays unbolded.\n"
        "  - Each `[FILE:LINE]` bracket contains EXACTLY ONE file path. For an issue "
        "that spans multiple files AND was NOT flagged as a duplicate in "
        "record_duplicate_pairs, emit a separate numbered entry per file (repeat "
        "the title and rationale as needed) instead of listing paths comma-separated "
        "inside one bracket. For deduplicated findings (same concern across files), "
        "emit ONE entry with the primary file in the bracket and list the other "
        "affected files in the rationale body.\n"
        "  - FILE must be the FULL repo-relative path exactly as it appears in the "
        "per-stack records (e.g. `services/my-svc/handler.py`, not just `handler.py`). "
        "Downstream tooling uses `git show <sha>:<FILE>` to resolve lines, so "
        "abbreviated paths will fail to post as inline comments.\n"
        "  - Per-stack human-readable context may appear above ## Issues under "
        "## Per-Stack Context, but all actionable issues live in the two lists above.\n"
        "  - Do not invent issues not supported by the source records."
    )
    parts.append(f"Write the complete report to {output_path}.")
    return "\n\n".join(parts)


def build_generic_fallback_prompt(
    *,
    files: list[str],
    diff_path: Path,
    intent_path: Path,
    alternatives_path: Path,
    output_path: Path,
    exploration_dir: Path | None = None,
    is_docs_only: bool = False,
) -> str:
    """Assemble the generic-fallback review prompt (no skill invocation).

    When is_docs_only=True, prepends the D-20 documentation-review notice.

    Args:
        files: Files this bucket owns.
        diff_path: Path to the full diff on disk.
        intent_path: Path to TTT intent.md.
        alternatives_path: Path to TTT alternatives.json.
        output_path: Where the agent must write its review.
        exploration_dir: Pre-scan exploration directory (if available).
        is_docs_only: Whether the whole diff is docs-only (D-20).

    Returns:
        Assembled prompt string.
    """
    parts: list[str] = []
    if is_docs_only:
        parts.append(DOC_REVIEW_NOTICE)
    pointer = _exploration_pointer(exploration_dir)
    if pointer:
        parts.append(pointer)
    parts.append(_context_pointers(intent_path=intent_path, alternatives_path=alternatives_path))
    parts.append(_confidence_and_convention_instructions())
    parts.append(_dependency_impact_instructions())
    parts.append(_stack_scope_instruction("generic-fallback", files))
    parts.append(_diff_instruction(diff_path, files))
    parts.append(
        "Review these files for correctness, clarity, and consistency with the "
        "author's intent. Apply language-agnostic review practices."
    )
    parts.append(f"Write your full review to {output_path}.")
    return "\n\n".join(parts)
