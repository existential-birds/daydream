"""Prompt builders for deep-review mode.

Pure keyword-only functions that assemble prompt strings from context pointers.
All context passes via filesystem paths (D-09) -- no full file contents embedded in
prompts. Per-stack agents only see their own stack's files + TTT context (D-10).

Public builders:
    - build_per_stack_prompt: per-language stack scoped review.
    - build_structural_prompt: repo-wide structural-maintainability meta-stack.
    - build_merge_prompt: cross-stack merge into a unified report.
    - build_verification_prompt: recommendation-verifier agent prompt.
    - build_generic_fallback_prompt: fallback for files without a dedicated stack.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from daydream.config import STRUCTURE_SKILL
from daydream.phases import (
    RECOMMENDATION_VERDICTS_SCHEMA,
    _confidence_and_convention_instructions,
    _dependency_impact_instructions,
    _exploration_pointer,
    _settled_decisions_block,
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
    prior_commits: str | None = None,
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
        prior_commits: Oneline log of prior daydream commits on this branch.

    Returns:
        Assembled prompt string.
    """
    parts: list[str] = []
    pointer = _exploration_pointer(exploration_dir)
    if pointer:
        parts.append(pointer)
    settled = _settled_decisions_block(prior_commits)
    if settled:
        parts.append(settled)
    parts.append(_context_pointers(intent_path=intent_path, alternatives_path=alternatives_path))
    parts.append(_confidence_and_convention_instructions())
    parts.append(_dependency_impact_instructions())
    parts.append(_stack_scope_instruction(stack_name, files))
    parts.append(_diff_instruction(diff_path, files))
    parts.append(skill_invocation)
    parts.append(f"Write your full review to {output_path}.")
    return "\n\n".join(parts)


def build_structural_prompt(
    *,
    files: list[str],
    diff_path: Path,
    intent_path: Path,
    alternatives_path: Path,
    output_path: Path,
    exploration_dir: Path | None = None,
    prior_commits: str | None = None,
) -> str:
    """Assemble the structural-maintainability meta-stack prompt.

    Mirrors ``build_per_stack_prompt`` but covers the full PR rather than a
    single language's files. The structural rubric judges repo-wide concerns
    (canonical helpers, file-size budgets, layering, branching shape), so the
    reviewer must be free to read any file in the codebase via Read/Grep/Bash
    instead of being scoped to a stack subset.

    Args:
        files: Full union of changed files across every stack. Used to anchor
            the scope statement; the reviewer is still free to read beyond.
        diff_path: Path to the full diff on disk.
        intent_path: Path to TTT intent.md.
        alternatives_path: Path to TTT alternatives.json.
        output_path: Where the agent must write its review.
        exploration_dir: Accepted for signature symmetry with
            ``build_per_stack_prompt``; intentionally ignored in the prompt
            body because the structural reviewer discovers context via tool
            calls, not pre-injected pointers.
        prior_commits: Oneline log of prior daydream commits on this branch.

    Returns:
        Assembled prompt string.
    """
    del exploration_dir  # accepted for signature symmetry; intentionally unused.
    joined = ", ".join(files)
    parts: list[str] = []
    settled = _settled_decisions_block(prior_commits)
    if settled:
        parts.append(settled)
    parts.append(_context_pointers(intent_path=intent_path, alternatives_path=alternatives_path))
    parts.append(
        f"You are the structural reviewer. The full change spans: {joined}. "
        f"The structural rubric applies repo-wide -- read any file in the "
        f"codebase as needed (Read/Grep/Bash) to judge whether canonical "
        f"helpers exist, file-size budgets are honored, and the change makes "
        f"the codebase easier or harder to live with."
    )
    parts.append(
        f"The full PR diff (base..HEAD) is at {diff_path}. Read it directly; "
        f"do NOT run `git diff` without a base ref -- on a clean branch that "
        f"returns empty and hides committed changes."
    )
    parts.append(f"/{STRUCTURE_SKILL}")
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
    structural_records_path: Path | None = None,
) -> str:
    """Assemble the cross-stack merge prompt (D-23..D-27).

    The merge agent returns a schema-validated JSON item list
    (``MERGED_ITEMS_SCHEMA``) -- NOT markdown. Each item is one actionable
    finding tagged with ``lens`` (``per-stack`` | ``cross-stack``) and
    ``severity``. The host (``phase_cross_stack_merge``) appends structural
    findings tagged ``lens="structural"`` in Python, normalizes ids, writes the
    canonical ``merged-items.json``, and renders ``review-output.md`` from it.
    This prompt therefore does NOT ask the agent for markdown, a structural
    section, or a write-to-file step.

    Each emitted item MUST:
      - carry a ``lens`` of ``per-stack`` or ``cross-stack`` (D-26 — cross-stack
        for concerns spanning multiple stacks)
      - carry a ``severity`` of ``high`` | ``medium`` | ``low`` (D-25 ordering)
      - collapse duplicates per dedup candidate adjudication (D-27)

    Args:
        per_stack_records_paths: Parsed per-stack record JSON paths (D-22 inputs).
        intent_path: Path to TTT intent.md.
        alternatives_path: Path to TTT alternatives.json.
        dedup_candidates_path: Path to dedup-candidates.json (D-27 pre-filter output).
        output_path: Deep-dir report path. Retained for call-site compatibility;
            the rendered report is written by ``phase_cross_stack_merge``, so the
            prompt no longer instructs the agent to write a file here.
        exploration_dir: Pre-scan exploration directory (if available).
        failed_stacks: Optional stack_name -> failure reason for stacks whose
            per-stack agent raised. The merge prompt includes an explicit
            "Uncovered stacks" block so missing coverage is surfaced instead of
            silently pretending the run was complete.
        structural_records_path: Optional path to the parsed structural-stack
            records JSON. Retained for call-site compatibility; structural
            findings are appended by ``phase_cross_stack_merge`` in Python (not
            via this prompt), so the agent is never pointed at this file.

    Returns:
        Assembled prompt string.
    """
    del output_path, structural_records_path  # appended/rendered by the host, not the prompt
    records_block = "\n".join(f"  - {p}" for p in per_stack_records_paths)
    parts: list[str] = []
    pointer = _exploration_pointer(exploration_dir)
    if pointer:
        parts.append(pointer)
    context_lines = [
        f"TTT intent summary: {intent_path}",
        f"TTT alternative-review findings: {alternatives_path}",
        f"Dedup pre-filter candidate pairs: {dedup_candidates_path}",
    ]
    context_lines.append(f"Per-stack parsed records:\n{records_block}")
    parts.append("\n".join(context_lines))
    if failed_stacks:
        failed_block = "\n".join(
            f"  - {name}: {reason}" for name, reason in sorted(failed_stacks.items())
        )
        parts.append(
            "Uncovered stacks (per-stack agent raised; no records available):\n"
            f"{failed_block}\n"
            "Note these uncovered stacks in your reasoning. Do NOT silently omit "
            "them -- downstream readers must be able to tell 'no findings' apart "
            "from 'this stack never ran'."
        )
    parts.append(
        "You are the cross-stack merge agent. Read every artifact above by path -- "
        "do NOT re-run any reviews. Return a single JSON object matching the "
        "structured-output schema: {\"items\": [ ... ]}. Each item is one "
        "actionable finding. Emit nothing else."
    )
    parts.append(
        "Dedup adjudication:\n"
        "  dedup-candidates.json has two sections:\n\n"
        "  record_alt_pairs (record ↔ TTT alt-review):\n"
        "  - For each candidate pair, decide whether the two findings describe the\n"
        "    same concern. If yes, emit ONE item citing both sources as combined\n"
        "    evidence. If no, emit both items independently.\n\n"
        "  record_duplicate_pairs (record ↔ record):\n"
        "  - These are per-stack records with near-identical descriptions across\n"
        "    different files (e.g. the same architectural concern reported once per\n"
        "    affected file). When two records describe the same conceptual finding,\n"
        "    emit ONE item listing all affected files rather than repeating the\n"
        "    finding verbatim for each file.\n\n"
        "  - Concerns that span multiple stacks (contract drift, shared-type "
        "mismatches, API-contract misalignment) are cross-stack findings."
    )
    parts.append(
        "Item fields (MANDATORY):\n"
        "  - id: integer; any value -- the host renumbers contiguously.\n"
        "  - lens: \"per-stack\" for a single-stack finding, \"cross-stack\" for a "
        "concern spanning multiple stacks. (Structural findings are appended by the "
        "host -- do NOT emit them yourself.)\n"
        "  - severity: \"high\" | \"medium\" | \"low\".\n"
        "  - confidence: \"HIGH\" | \"MEDIUM\" | \"LOW\".\n"
        "  - file: the FULL repo-relative path exactly as it appears in the per-stack "
        "records (e.g. `services/my-svc/handler.py`, not just `handler.py`). "
        "Downstream tooling uses `git show <sha>:<FILE>` to resolve lines, so "
        "abbreviated paths will fail to post as inline comments.\n"
        "  - line: integer line number for the finding.\n"
        "  - description: the finding title / one-line summary, plain text.\n"
        "  - rationale: why it matters; cite the actual records filename or stack "
        "name -- e.g. `(Sources: python-records item 6, alternatives item 4)`. "
        "NEVER use the `#N` notation (e.g. `#6`); GitHub auto-links `#N` to "
        "repository issues/PRs, creating misleading links.\n\n"
        "Rules:\n"
        "  - Each item's `file` contains EXACTLY ONE path. For a concern that spans "
        "multiple files AND was NOT flagged as a duplicate in "
        "record_duplicate_pairs, emit a separate item per file. For deduplicated "
        "findings (same concern across files), emit ONE item with the primary file "
        "and list the other affected files in the rationale.\n"
        "  - Do not invent findings not supported by the source records."
    )
    return "\n\n".join(parts)


def build_verification_prompt(
    *,
    items: list[dict[str, Any]],
    target_dir: Path,
    output_path: Path,
) -> str:
    """Assemble the recommendation-verifier prompt.

    The verifier audits each numbered language-lens item against the codebase:
    trait/interface specs, sibling implementations, and any transitive
    properties the recommendation asserts about functions it does not modify.
    Verdicts are advisory -- the verifier does not block fixes; it warns the fix
    agent inline and surfaces a count to the user.

    Structural items are filtered out by the caller
    (``phase_verify_recommendations``) before this builder runs, so the rendered
    item list embedded here is non-structural by construction.

    Hard contract:
      - Read-only tools only: Read, Grep, Glob, and Bash restricted to
        non-mutating commands (git, cat, ls).
      - The non-structural finding list is rendered inline below.
      - Empty issue list yields an empty verdict list (no error).

    Args:
        items: The non-structural (per-stack / cross-stack) canonical items to
            verify. Rendered inline into the prompt; verdicts are keyed by each
            item's canonical ``id`` (the verdict ``issue_id``).
        target_dir: Repository root the verifier runs against.
        output_path: Where the verifier must write its JSON verdicts file.

    Returns:
        Assembled prompt string.
    """
    from daydream.deep.render import render_report

    parts: list[str] = []
    parts.append(
        "You are the recommendation-verifier agent. Your job is to audit each "
        "numbered issue in the finding list below against the actual codebase "
        "and decide whether its recommendation is consistent with trait/interface "
        "specs and sibling implementations.\n\n"
        f"Repository root: {target_dir}\n"
        "The numbered findings to verify (each `issue_id` in your output MUST "
        "match the leading number `N.` of the finding it verifies):\n\n"
        + render_report(items)
        + "\nDo NOT re-run any reviews."
    )
    parts.append(
        "Read-only contract (MANDATORY):\n"
        "  - Allowed tools: Read, Grep, Glob, Bash.\n"
        "  - Bash is restricted to non-mutating commands only: `git`, `cat`, `ls`.\n"
        "  - Do NOT write, edit, or move files anywhere except the JSON output "
        "path specified below. Do NOT run `git commit`, `git add`, `git checkout`, "
        "`git reset`, `git stash`, or any other state-changing command."
    )
    parts.append(
        "Turn budget: cap your investigation at 25 turns total. Prefer Grep/Glob "
        "to narrow the search before opening files with Read."
    )
    parts.append(
        "For EACH numbered issue in the merged report, perform these five steps:\n\n"
        "  1. Locate the `impl` / interface / protocol declaration the changed "
        "code participates in. If absent, set `verdict=consistent` only if no "
        "sibling implementations exist.\n"
        "  2. Locate every sibling implementation "
        "(`grep -rn \"impl <Trait> for\"` / `class X(<Iface>)`).\n"
        "  3. Locate the trait/interface doc-comment that specifies the behavior "
        "being changed.\n"
        "  4. Compare the recommendation against those. Verdicts:\n"
        "     - `consistent` -- recommendation aligns with the trait doc and at "
        "least one sibling. Cite one line of evidence.\n"
        "     - `contradicts` -- recommendation would make this impl diverge "
        "from the trait doc OR from a sibling that the trait doc agrees with. "
        "Cite the conflicting line.\n"
        "     - `uncertain` -- cannot decide from the codebase. List the "
        "assumption that would need to hold.\n"
        "  5. Additionally: list any *transitive properties* the recommendation "
        "asserts about functions it does not modify (`unverified_assumptions`). "
        'Example: "assumes `osprey_home()` always returns an absolute path."'
    )
    parts.append(
        "Empty-input rule: if the merged report contains no numbered issues "
        "under `## Issues` or `## Cross-Stack Issues`, emit an empty `verdicts` "
        "array. This is NOT an error."
    )
    parts.append(
        "Output JSON conforming EXACTLY to this schema. Every verdict entry "
        "MUST include all four required fields, even when "
        "`unverified_assumptions` is an empty array.\n\n"
        "RECOMMENDATION_VERDICTS_SCHEMA = "
        + json.dumps(RECOMMENDATION_VERDICTS_SCHEMA, indent=4)
    )
    parts.append(f"Write your JSON verdicts to {output_path}.")
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
    prior_commits: str | None = None,
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
        prior_commits: Oneline log of prior daydream commits on this branch.

    Returns:
        Assembled prompt string.
    """
    parts: list[str] = []
    if is_docs_only:
        parts.append(DOC_REVIEW_NOTICE)
    pointer = _exploration_pointer(exploration_dir)
    if pointer:
        parts.append(pointer)
    settled = _settled_decisions_block(prior_commits)
    if settled:
        parts.append(settled)
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
