"""Prompt builders for deep-review mode.

Pure keyword-only functions that assemble prompt strings from context pointers.
All context passes via filesystem paths (D-09) -- no full file contents embedded in
prompts. Per-stack agents only see their own stack's files + TTT context (D-10).

Public builders:
    - build_per_stack_prompt: per-language stack scoped review.
    - build_structural_prompt: repo-wide structural-maintainability meta-stack.
    - build_arbiter_prompt: scoped Opus arbiter for cross-stack conflict resolution.
    - build_merge_prompt: cross-stack merge into a unified report.
    - build_verification_prompt: recommendation-verifier agent prompt.
    - build_generic_fallback_prompt: fallback for files without a dedicated stack.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from daydream.phases import (
    RECOMMENDATION_VERDICTS_SCHEMA,
    _confidence_and_convention_instructions,
    _dependency_impact_instructions,
    _exploration_pointer,
    _settled_decisions_block,
)
from daydream.prompts.grounding import CWD_GROUNDING_INSTRUCTION

DOC_REVIEW_NOTICE = (
    "[Notice] Dedicated documentation review (beagle-docs) is planned but not yet "
    "implemented.\nThese documentation files are currently being reviewed by the "
    "generic-fallback agent (D-20)."
)

# Shared verification-protocol instruction for structural and generic-fallback
# builders (issue #229). Embedded as instruction text, not routed through
# ``Backend.format_skill_invocation``.
VERIFICATION_PROTOCOL_INSTRUCTION = (
    "Before writing findings, load the review-verification-protocol skill "
    "(read review-verification-protocol/SKILL.md) and apply its "
    "anchor-evidence-severity gates (gate 1: anchor file:line, gate 2: produce "
    "evidence artifacts, gate 3: calibrate severity). Do NOT report a finding "
    "that fails any gate."
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


# Issue #172 — Read-once diff hunks (Fix B). Upper byte bound for the inlined
# diff section in per-stack / generic prompts. Above this bound the helper
# returns ``None`` and the prompt falls back to the diff_path pointer so the
# prompt size stays bounded. ~12 KiB comfortably fits a few hundred lines of
# unified diff without bloating the per-stack review prompts.
INLINE_DIFF_BUDGET_BYTES = 12_288

# Per-file block splitter (splits the unified diff at each `diff --git` header).
_DIFF_BLOCK_SPLIT = re.compile(r"^(?=diff --git )", re.MULTILINE)
# `+++ ` and `--- ` file headers inside a single block.
_DIFF_PLUS_HEADER = re.compile(r"^\+\+\+ (.+)$", re.MULTILINE)
_DIFF_MINUS_HEADER = re.compile(r"^--- (.+)$", re.MULTILINE)
# Fallback header for binary / mode-only diffs that lack `--- / +++`.
_DIFF_GIT_HEADER = re.compile(r"^diff --git a/(\S+) b/(\S+)")


def _diff_block_path(block: str) -> str | None:
    """Resolve the single changed path for one ``diff --git`` block.

    Shared unified-diff block-parsing contract used by both
    ``_diff_blocks_for_files`` (here) and ``orchestrator._diff_changed_files``
    so the post-state / pre-state / header fallback order and ``/dev/null``
    handling live in exactly one place.

    Prefers the post-state path (``+++ b/<path>``) so renames produce only the
    destination. Falls back to the pre-state path for deletions
    (``+++ /dev/null``) and to the ``diff --git`` header for binary / mode-only
    diffs that lack ``---``/``+++`` lines. ``/dev/null`` sentinels are skipped
    at every layer. Returns ``None`` for blocks that are not ``diff --git``
    headers or where no path can be resolved.
    """

    def _strip_prefix(path: str, prefix: str) -> str:
        return path[len(prefix) :] if path.startswith(prefix) else path

    if not block.startswith("diff --git "):
        return None
    plus = _DIFF_PLUS_HEADER.search(block)
    if plus and plus.group(1) != "/dev/null":
        return _strip_prefix(plus.group(1), "b/")
    minus = _DIFF_MINUS_HEADER.search(block)
    if minus and minus.group(1) != "/dev/null":
        return _strip_prefix(minus.group(1), "a/")
    git = _DIFF_GIT_HEADER.match(block)
    if git:
        return git.group(2)
    return None


def _diff_blocks_for_files(diff: str, files: list[str]) -> str | None:
    """Return the concatenated diff blocks for ``files`` (issue #172, Fix B).

    Reuses the existing per-file block splitter (``_DIFF_BLOCK_SPLIT`` regex)
    plus ``_diff_block_path`` (which applies the post-state header regexes
    ``_DIFF_PLUS_HEADER`` / ``_DIFF_MINUS_HEADER`` / ``_DIFF_GIT_HEADER``) to
    select the ``diff --git`` blocks whose post-state path matches a file in
    ``files``. The blocks are concatenated as-is (unified-diff text, including
    headers / hunks).

    Byte-bounded: when the concatenated result would exceed
    ``INLINE_DIFF_BUDGET_BYTES`` the helper returns ``None`` so the caller
    falls back to the diff_path pointer (keeps prompt size bounded). Also
    returns ``None`` when no blocks match (e.g. files absent from the diff).

    Args:
        diff: Full unified diff text.
        files: Repo-relative paths to select blocks for.

    Returns:
        The concatenated diff blocks (with a trailing newline), or ``None``
        when the result would exceed the byte budget or no blocks match.
    """
    wanted = set(files)
    if not wanted:
        return None

    selected: list[str] = []
    for block in _DIFF_BLOCK_SPLIT.split(diff):
        if _diff_block_path(block) in wanted:
            selected.append(block if block.endswith("\n") else block + "\n")

    if not selected:
        return None
    result = "".join(selected)
    if len(result.encode("utf-8")) > INLINE_DIFF_BUDGET_BYTES:
        return None
    return result


def _diff_instruction(
    diff_path: Path,
    files: list[str],
    *,
    inline_diff: str | None = None,
) -> str:
    """Diff context for a per-stack / generic-fallback reviewer.

    Issue #172 Fix B (read-once):
      - When ``inline_diff`` is supplied (the relevant hunks already extracted
        by ``_diff_blocks_for_files`` and under the byte bound), the hunks are
        inlined and the ``Read it directly`` instruction is DROPPED. The agent
        has what it needs without a tool-call round-trip for the static
        ``diff.patch`` file.
      - When ``inline_diff`` is ``None`` (byte budget exceeded / no matching
        blocks / caller had no diff text), today's path-pointer text is used
        unchanged so the agent can still locate the full diff for whole-file
        context. ``diff_path`` stays a required param either way.

    Args:
        diff_path: Path to the full diff on disk.
        files: Files this stack owns (used in the fallback path-pointer text).
        inline_diff: Pre-extracted hunks to inline, or ``None`` for the fallback.

    Returns:
        The diff-context section for the prompt.
    """
    if inline_diff:
        return (
            "Relevant diff hunks for your stack (inlined; do NOT re-Read "
            "diff.patch for these — the hunks are already here):\n\n"
            f"{inline_diff.rstrip()}\n\n"
            "Focus on hunks that touch your stack's files. For whole-file "
            "context beyond these hunks you MAY Read the source files directly."
        )
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
    cwd: Path,
    exploration_dir: Path | None = None,
    prior_commits: str | None = None,
    inline_diff: str | None = None,
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
        cwd: Absolute working directory the agent runs in (grounds path resolution).
        exploration_dir: Pre-scan exploration directory (if available).
        prior_commits: Oneline log of prior daydream commits on this branch.
        inline_diff: Issue #172 Fix B. Pre-extracted diff hunks for ``files``
            to inline (skips the ``Read it directly`` instruction). ``None``
            falls back to the diff_path pointer.

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
    parts.append(CWD_GROUNDING_INSTRUCTION.format(cwd=cwd))
    parts.append(_context_pointers(intent_path=intent_path, alternatives_path=alternatives_path))
    parts.append(_confidence_and_convention_instructions())
    parts.append(_dependency_impact_instructions())
    parts.append(_stack_scope_instruction(stack_name, files))
    parts.append(_diff_instruction(diff_path, files, inline_diff=inline_diff))
    parts.append(skill_invocation)
    parts.append(f"Write your full review to {output_path}.")
    return "\n\n".join(parts)


def build_structural_prompt(
    *,
    skill_invocation: str,
    files: list[str],
    diff_path: Path,
    intent_path: Path,
    alternatives_path: Path,
    output_path: Path,
    cwd: Path,
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
        skill_invocation: Backend-formatted invocation for the structural skill.
        files: Full union of changed files across every stack. Used to anchor
            the scope statement; the reviewer is still free to read beyond.
        diff_path: Path to the full diff on disk.
        intent_path: Path to TTT intent.md.
        alternatives_path: Path to TTT alternatives.json.
        output_path: Where the agent must write its review.
        cwd: Absolute working directory the agent runs in (grounds path resolution).
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
    parts.append(CWD_GROUNDING_INSTRUCTION.format(cwd=cwd))
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
    parts.append(skill_invocation)
    parts.append(VERIFICATION_PROTOCOL_INSTRUCTION)
    parts.append(f"Write your full review to {output_path}.")
    return "\n\n".join(parts)


def build_arbiter_prompt(
    *,
    arbiter_input_path: Path,
    diff_path: Path,
    intent_path: Path,
    alternatives_path: Path,
    cwd: Path,
    exploration_dir: Path | None = None,
) -> str:
    """Assemble the scoped Opus arbiter prompt (issue #168).

    The arbiter re-reviews ONLY the high-severity / contested findings that the
    cheaper Sonnet per-stack reviewers surfaced. It is an adjudicator, not a
    discoverer: it may downgrade, confirm, sharpen, or reject each finding, but
    it must not invent new ones (new discovery is the per-stack reviewers' job;
    the arbiter can only re-rank what they found).

    Args:
        arbiter_input_path: JSON file of the selected findings. Each entry
            carries an ``arb_id`` the arbiter must echo back, plus the original
            ``file``/``line``/``severity``/``confidence``/``description``.
        diff_path: Path to the full diff on disk.
        intent_path: Path to TTT intent.md.
        alternatives_path: Path to TTT alternatives.json.
        cwd: Absolute working directory the agent runs in (grounds path resolution).
        exploration_dir: Pre-scan exploration directory (if available).

    Returns:
        Assembled prompt string.
    """
    parts: list[str] = []
    pointer = _exploration_pointer(exploration_dir)
    if pointer:
        parts.append(pointer)
    parts.append(CWD_GROUNDING_INSTRUCTION.format(cwd=cwd))
    parts.append(_context_pointers(intent_path=intent_path, alternatives_path=alternatives_path))
    parts.append(
        f"The full PR diff (base..HEAD) is at {diff_path}. Read it directly; "
        f"do NOT run `git diff` without a base ref -- on a clean branch that "
        f"returns empty and hides committed changes."
    )
    parts.append(
        "You are the arbiter. The cheaper per-stack reviewers flagged the "
        f"high-severity and contested findings listed in {arbiter_input_path}. "
        "Re-review each one against the actual code (Read/Grep/Bash) and the "
        "diff. You are adjudicating their work, NOT starting a fresh review: do "
        "not introduce findings that are not in the input list."
    )
    parts.append(
        "Return a single JSON object matching the structured-output schema: "
        '{"findings": [ ... ]}. Emit exactly one entry per input finding, echoing '
        "its `arb_id` unchanged. For each:\n"
        "  - keep: true if the finding is real and actionable; false to reject a "
        "false positive or a non-issue (rejected findings are dropped entirely).\n"
        "  - severity: your adjudicated high | medium | low (you may change it).\n"
        "  - confidence: your adjudicated HIGH | MEDIUM | LOW.\n"
        "  - description: a sharpened one-line summary (keep it about the same "
        "finding; do not repurpose the slot for a different issue).\n"
        "  - rationale: why it matters, grounded in what you actually read."
    )
    return "\n\n".join(parts)


def build_suppression_prompt(
    *,
    suppression_input_path: Path,
    diff_path: Path,
    intent_path: Path,
    alternatives_path: Path,
    cwd: Path,
    exploration_dir: Path | None = None,
) -> str:
    """Assemble the skeptical precision-mode suppression prompt (issue #232).

    The suppression reviewer re-examines ONLY the borderline (LOW-confidence /
    low-severity uncontested) findings the arbiter never scrutinizes. Its default
    stance is the inverse of the arbiter's: a finding is DROPPED unless the
    reviewer can point at confirming evidence in the actual code. This trims
    evidenced-but-immaterial false positives on precision-sensitive runs without
    the arbiter's fail-open protection (which exists to guard high-severity /
    contested findings -- exactly the ones this pass never sees).

    Like the arbiter it is an adjudicator, not a discoverer: it may confirm or
    reject each input finding, but must not invent new ones.

    Args:
        suppression_input_path: JSON file of the selected borderline findings.
            Each entry carries a ``sup_id`` the reviewer must echo back, plus the
            original ``file``/``line``/``severity``/``confidence``/``description``.
        diff_path: Path to the full diff on disk.
        intent_path: Path to TTT intent.md.
        alternatives_path: Path to TTT alternatives.json.
        cwd: Absolute working directory the agent runs in (grounds path resolution).
        exploration_dir: Pre-scan exploration directory (if available).

    Returns:
        Assembled prompt string.
    """
    parts: list[str] = []
    pointer = _exploration_pointer(exploration_dir)
    if pointer:
        parts.append(pointer)
    parts.append(CWD_GROUNDING_INSTRUCTION.format(cwd=cwd))
    parts.append(_context_pointers(intent_path=intent_path, alternatives_path=alternatives_path))
    parts.append(
        f"The full PR diff (base..HEAD) is at {diff_path}. Read it directly; "
        f"do NOT run `git diff` without a base ref -- on a clean branch that "
        f"returns empty and hides committed changes."
    )
    parts.append(
        "You are the suppression reviewer. The cheaper per-stack reviewers "
        "flagged the borderline, low-confidence / low-severity findings listed "
        f"in {suppression_input_path}. These were NOT contested and NOT "
        "high-severity, so no heavyweight arbiter looked at them. Your job is to "
        "cut false positives: re-examine each one against the actual code "
        "(Read/Grep/Bash) and the diff. You are adjudicating their work, NOT "
        "starting a fresh review: do not introduce findings that are not in the "
        "input list."
    )
    parts.append(
        "Default to DROPPING each finding. Keep one ONLY when you can point at "
        "confirming evidence in the code that it is a real, actionable problem. "
        "Absence of evidence is a drop, not a keep -- a merely plausible or "
        "stylistic nit with no concrete grounding must be dropped."
    )
    parts.append(
        "Return a single JSON object matching the structured-output schema: "
        '{"findings": [ ... ]}. Emit exactly one entry per input finding, echoing '
        "its `sup_id` unchanged. For each:\n"
        "  - keep: true ONLY if you cite confirming evidence that the finding is "
        "real and actionable; false to drop an unconfirmed / immaterial finding.\n"
        "  - severity: your adjudicated high | medium | low.\n"
        "  - confidence: your adjudicated HIGH | MEDIUM | LOW.\n"
        "  - description: a sharpened one-line summary of the SAME finding.\n"
        "  - rationale: for a keep, the concrete evidence you found; for a drop, "
        "why it is not confirmable.\n"
        "  - evidence: the grounded `file:line` citation backing a kept finding."
    )
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
    cwd: Path,
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
        cwd: Absolute working directory the verifier runs in (grounds path resolution).
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
        f"{CWD_GROUNDING_INSTRUCTION.format(cwd=cwd)}\n"
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
        "Gate-0 anti-confabulation (MANDATORY — applies before any verdict):\n"
        "  Before issuing ANY verdict (consistent/contradicts/uncertain), you MUST "
        "echo the exact artifact you are judging, quoted from a source read in THIS "
        "turn:\n"
        "    - The file:line plus the cited code, read freshly now (not recalled "
        "from earlier in the session).\n"
        "  The artifact is the only source of truth. A verdict issued without a "
        "same-turn echo of its target is INVALID — emit the echo first, or do not "
        "emit the verdict."
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
    cwd: Path,
    exploration_dir: Path | None = None,
    is_docs_only: bool = False,
    prior_commits: str | None = None,
    inline_diff: str | None = None,
) -> str:
    """Assemble the generic-fallback review prompt (no skill invocation).

    When is_docs_only=True, prepends the D-20 documentation-review notice.

    Args:
        files: Files this bucket owns.
        diff_path: Path to the full diff on disk.
        intent_path: Path to TTT intent.md.
        alternatives_path: Path to TTT alternatives.json.
        output_path: Where the agent must write its review.
        cwd: Absolute working directory the agent runs in (grounds path resolution).
        exploration_dir: Pre-scan exploration directory (if available).
        is_docs_only: Whether the whole diff is docs-only (D-20).
        prior_commits: Oneline log of prior daydream commits on this branch.
        inline_diff: Issue #172 Fix B. Pre-extracted diff hunks for ``files``
            to inline (skips the ``Read it directly`` instruction). ``None``
            falls back to the diff_path pointer.

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
    parts.append(CWD_GROUNDING_INSTRUCTION.format(cwd=cwd))
    parts.append(_context_pointers(intent_path=intent_path, alternatives_path=alternatives_path))
    parts.append(_confidence_and_convention_instructions())
    parts.append(_dependency_impact_instructions())
    parts.append(_stack_scope_instruction("generic-fallback", files))
    parts.append(_diff_instruction(diff_path, files, inline_diff=inline_diff))
    parts.append(
        "Review these files for correctness, clarity, and consistency with the "
        "author's intent. Apply language-agnostic review practices."
    )
    parts.append(VERIFICATION_PROTOCOL_INSTRUCTION)
    parts.append(f"Write your full review to {output_path}.")
    return "\n\n".join(parts)
