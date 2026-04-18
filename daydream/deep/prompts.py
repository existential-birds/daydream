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
    joined = " ".join(files)
    return (
        f"The full diff is at {diff_path}. Focus on changes in your stack's files "
        f"using:\n"
        f"  git diff --no-color -- {joined}\n"
        f"from within the repository root."
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
