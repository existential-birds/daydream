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

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from daydream.phases import (
    _confidence_and_convention_instructions,
    _dependency_impact_instructions,
    _exploration_pointer,
    _render_bash_allowlist,
    _settled_decisions_block,
)
from daydream.prompt_budget import INLINE_DIFF_BUDGET_BYTES, fits_inline_diff_budget  # noqa: F401
from daydream.prompts.authorial_intent import AUTHORITATIVE_INTENT_BLOCK
from daydream.prompts.grounding import CWD_GROUNDING_INSTRUCTION
from daydream.prompts.wire_contract import (
    WIRE_CONTRACT_GENERIC_INSTRUCTION,
    WIRE_CONTRACT_RUST_INSTRUCTION,
)

DOC_REVIEW_NOTICE = (
    "[Notice] Dedicated documentation review (beagle-docs) is planned but not yet "
    "implemented.\nThese documentation files are currently being reviewed by the "
    "generic-fallback agent (D-20)."
)

# Repo-wide cross-file symbol existence check (issue #310). Embedded inline as
# instruction text for the same reason as ``ANTI_SLOP_RUBRIC_INSTRUCTION``: the
# structural reviewer runs with cwd set to the reviewed repo, so a bare
# skill-file read resolves against that repo and silently drops the gate.
# Demands Gate-2 evidence (``rg`` the definition) before flagging any symbol
# referenced outside the diff, so cross-file bug classes -- a subcommand invoked
# by a CLI wrapper that does not exist, a trait method implemented by generated
# code whose contract differs from its call site, a config field read by a
# different module than the one that writes it -- are verified against the repo
# rather than asserted from the call site alone.
CROSS_FILE_SYMBOL_EXISTENCE_INSTRUCTION = (
    "Cross-file symbol existence check (apply before flagging anything about a "
    "symbol defined OUTSIDE the diff):\n"
    "  1. Every referenced symbol not defined in this diff -- a function, a "
    "subcommand invoked by a CLI wrapper, a trait method implemented by "
    "generated code, a config field, a CLI flag -- must be verified to exist "
    "in the checked-out repo before you report a finding about it.\n"
    "  2. Evidence (Gate-2): `rg` for the definition in the repo and cite the "
    "file:line where it is declared. Never assert a symbol's behavior from the "
    "call site alone.\n"
    "  3. If no definition can be found, say so explicitly and downgrade the "
    "finding's confidence -- an unresolved reference is reportable only when "
    "the missing definition is real, never when you simply failed to locate it."
)

# Per-stack config-flow trace (issue #310). Embedded inline as instruction text
# for the same reason as ``TEST_QUALITY_RUBRIC_INSTRUCTION``: per-stack
# reviewers run with cwd set to the reviewed repo, so a bare skill-file read
# resolves against that repo and silently drops the gate. Targets the
# plumbed-config bug class: a field parsed in a config struct but silently
# dropped before it reaches the request, or the same value read twice at
# different points with the source able to change between reads (TOCTOU).
CONFIG_FLOW_TRACE_INSTRUCTION = (
    "Config/env flow trace (apply to every config field or env var plumbed "
    "through layers):\n"
    "  1. Trace the full path of each plumbed field: config struct -> driver "
    "config -> request construction.\n"
    "  2. Emit a one-line trace statement per field naming where it is parsed, "
    "where it is forwarded, and where (if anywhere) it reaches the request.\n"
    "  3. Flag silent drops -- a field parsed but never forwarded to the next "
    "layer.\n"
    "  4. Flag double-resolves -- the same value read twice at different points "
    "with the source able to change between reads (TOCTOU)."
)

# Trust-model check (issue #310). Embedded inline as instruction text for the
# same reason as the other rubrics: reviewers run with cwd set to the reviewed
# repo, so a bare skill-file read resolves against that repo and drops the
# gate. Targets security-relevant markers -- cache-control injection across a
# trust boundary, escaping, credential forwarding -- by demanding an explicit
# trust-model sentence before a finding is reported.
TRUST_MODEL_INSTRUCTION = (
    "Trust-model check (apply to every security-relevant marker: cache-control "
    "injection, trust boundaries, escaping, credential forwarding):\n"
    "  For each marker, state the trust model in one sentence: who is the "
    "untrusted party here, and does this path honor the boundary?\n"
    "  Flag any path that instructs an untrusted party to retain or forward "
    "sensitive content -- e.g. an edge proxy echoing an untrusted response's "
    "cache-control directive, or credentials passed through an intermediate hop."
)

# Shared verification-protocol instruction for structural and generic-fallback
# builders (issue #229). The gates are embedded inline as instruction text, not
# routed through ``Backend.format_skill_invocation`` and NOT loaded from a skill
# file: these two reviewers run with cwd set to the reviewed repo, so a bare
# ``read review-verification-protocol/SKILL.md`` resolves against that repo and
# fails ("skill doesn't exist as a file"), silently dropping the gates. Both
# reviewers are language-agnostic (repo-wide structural / non-stack fallback), so
# the protocol's language-specific valid-pattern tables add little here — the
# gate discipline is what matters, and it is self-contained below. Mirrors the
# inline gate-0 embedding in ``build_verification_prompt``.
VERIFICATION_PROTOCOL_INSTRUCTION = (
    "Before writing findings, apply the review-verification-protocol gates "
    "(stated inline here — no skill file read is required):\n"
    "  Gate-0 anti-confabulation (before ANY finding): echo the exact artifact "
    "you are judging — file:line plus the cited code, read freshly in THIS turn, "
    "not recalled. The source is the only truth; never infer a finding from the "
    "branch name, cwd, or memory. A finding without a same-turn echo of its "
    "target is INVALID.\n"
    "  A `clean` verdict for a file also requires a same-turn read of that file: "
    "absent the read, mark the file `not reviewed`, never `clean`.\n"
    "  Gate 1 (anchor): read the full enclosing symbol/module, not just the diff "
    "hunk; state the file path and line range you are judging.\n"
    "  Gate 2 (evidence): produce an artifact for the finding's type — pasted "
    'tool output, a file:line citation, or an explicit "none" / "N matches" '
    'after a repo search. Never claim you "looked" without an artifact.\n'
    "  Gate 3 (severity): calibrate severity to impact; a request for net-new "
    "code that did not exist in scope is Informational only.\n"
    "Do NOT report a finding that fails any gate."
)

# Per-stack test-quality rubric (issue #308). Embedded inline as instruction text
# for the same reason as ``VERIFICATION_PROTOCOL_INSTRUCTION``: per-stack
# reviewers run with cwd set to the reviewed repo, so a bare skill-file read
# resolves against that repo and silently drops the gates. The rubric targets
# test hunks in the diff: vacuous assertions, internal-field/pointer-identity
# assertions, nondeterminism, canonical-path bypasses, and portability breaks.
TEST_QUALITY_RUBRIC_INSTRUCTION = (
    "Apply the test-quality rubric to every test hunk in the diff "
    "(stated inline here — no skill file read is required):\n"
    "  1. Would this test fail if the behavior under test were wrong? Scan for "
    "vacuous assertions — e.g. `read_to_string(...).unwrap_or_default()` "
    "returning empty on failure, expected values built with the same helper "
    "under test, a wait/retry helper returning the last nonmatching frame.\n"
    "  2. Does it assert observable consequences (output, filesystem, exit code, "
    "store state) rather than internal fields/pointers/dispatch plumbing "
    "(`context as *const _ as usize`, dispatch internals, event payloads with no "
    "observable check)?\n"
    "  3. Is it deterministic (no sleeps, no `yield_now()` reaping assumptions, "
    "no environment leaks — require restore guards for any env mutation)?\n"
    "  4. Does it exercise the new behavior through the canonical public path (no "
    "raw `system_prompt` copies, no bypassing the public API the behavior lives "
    "behind)?\n"
    "  5. Does it compile on all platforms (`#[cfg]` gates)?\n"
    "Layering awareness: legitimate pure-function seams are fine — a unit test of "
    "a pure `build_driver_request` or driver-boundary propagation helper is NOT an "
    "internal-field assertion. Flag a seam ONLY when it bypasses the observable "
    "behavior the test claims to cover."
)

# Per-stack + structural anti-slop review rubric (issue #314). Embedded inline as
# instruction text for the same reason as ``TEST_QUALITY_RUBRIC_INSTRUCTION``:
# per-stack and structural reviewers run with cwd set to the reviewed repo, so a
# bare skill-file read resolves against that repo and silently drops the rubric.
# Targets the SlopCodeBench degradation patterns -- structural erosion, verbosity,
# duplication -- in the code hunks, with severity calibrated so it flags
# maintainability regressions without over-applying to legitimate structure.
ANTI_SLOP_RUBRIC_INSTRUCTION = (
    "Apply the anti-slop rubric to every code hunk in the diff "
    "(stated inline here -- no skill file read is required). It targets the "
    "SlopCodeBench degradation patterns -- structural erosion, verbosity, "
    "duplication:\n"
    "  1. Flag complexity concentration: when a hunk adds logic to a function "
    "that is already large/high-complexity (cyclomatic complexity > ~10, or > ~80 "
    "lines), require extraction into focused callables -- especially when the "
    "same pattern (flag pair, branch ladder, error guard) is repeated verbatim.\n"
    "  2. Verbosity: flag redundant code -- identity comprehensions instead of "
    "filter/map, empty-list guards inside loops, single-use intermediate "
    "variables, casts to dodge type checking, trivial wrapper functions, "
    "nested ladders.\n"
    "  3. Duplication: flag the same hunk structure repeated (e.g. N flags x 2 "
    "branches) that should be a loop/helper/template.\n"
    "  4. Severity: maintainability findings are medium/low -- never high -- "
    "under this rubric, full stop. The structural lens may flag real erosion, "
    "but anti-slop findings never escalate to high.\n"
    "  5. Scope: when erosion is pre-existing-and-growing, flag the growth, not "
    "the whole function -- report only the newly introduced growth, scoped to "
    "this diff's contribution."
)


def _context_pointers(
    *,
    intent_path: Path,
    alternatives_path: Path,
    intent_authoritative: bool = False,
    include_alternatives: bool = True,
) -> str:
    """Reference pointers for TTT stage outputs (D-09/D-19 context bus).

    When ``intent_authoritative`` is True, the intent pointer is upgraded: the
    pointer to ``intent_path`` is accompanied by a provenance sentence and the
    ``AUTHORITATIVE_INTENT_RULE`` precedence rule, since the intent was grounded
    by a fresh, head-matched PR description (issue #279).

    ``include_alternatives=False`` omits exactly the alternatives paragraph and
    nothing else — for callers running concurrently with the wonder pass, whose
    ``alternatives.json`` does not exist yet.
    """
    alternatives_paragraph = (
        f"TTT alternative-review findings are at {alternatives_path}. Use them as a "
        f"starting point -- you may deepen, confirm, or dismiss each finding with "
        f"language-specific evidence."
    )
    if intent_authoritative:
        head = (
            f"TTT intent summary is at {intent_path}. Read it before starting your "
            f"review -- it records the author's stated intent from the pull-request "
            f"description."  # provenance sentence
            f"\n{AUTHORITATIVE_INTENT_BLOCK}"
        )
        return f"{head}\n{alternatives_paragraph}" if include_alternatives else head
    head = (
        f"TTT intent summary is at {intent_path}. Read it before starting your review "
        f"so your findings align with the author's stated intent."
    )
    return f"{head}\n{alternatives_paragraph}" if include_alternatives else head


def _stack_scope_instruction(stack_name: str, files: list[str]) -> str:
    joined = ", ".join(files)
    return (
        f"You are reviewing the {stack_name} stack. Focus ONLY on these files:\n"
        f"  {joined}\n"
        f"Do NOT review files from other stacks -- their reviews are running in "
        f"parallel and will be merged afterwards."
    )


# Per-file block splitter (splits the unified diff at each `diff --git` header).
_DIFF_BLOCK_SPLIT = re.compile(r"^(?=diff --git )", re.MULTILINE)
# `+++ ` and `--- ` file headers inside a single block.
_DIFF_PLUS_HEADER = re.compile(r"^\+\+\+ (.+)$", re.MULTILINE)
_DIFF_MINUS_HEADER = re.compile(r"^--- (.+)$", re.MULTILINE)
# Fallback header for binary / mode-only diffs that lack `--- / +++`.
_DIFF_GIT_HEADER = re.compile(r"^diff --git a/(\S+) b/(\S+)")
# Truncation marker emitted by ``bound_deep_diff``; carries the same dropped
# names as ``DeepDiffBoundInfo.dropped_paths`` so consumers of the bounded text
# alone (``_diff_blocks_for_files``) can tell a dropped block from a file
# simply absent from the diff. Parse-safe for every block consumer: the line
# carries no ``diff --git`` / ``---`` / ``+++`` header, so ``_diff_block_path``
# returns None and the block splitters skip it.
_DIFF_TRUNCATION_MARKER = re.compile(
    r"^# daydream: deep diff truncated: \d+ -> \d+ bytes "
    r"\(\d+/\d+ blocks retained(?:; dropped: (?P<dropped>[^)]+))?\)\n"
)


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
        when the result would exceed the byte budget, no blocks match, or the
        ``diff`` is a bounded value whose truncation marker names one of
        ``files`` as dropped whole (a stack that mixes retained and dropped
        blocks must fall back to the diff_path pointer rather than inline a
        silently partial hunk set).
    """
    wanted = set(files)
    if not wanted:
        return None

    # Issue #644 follow-up: ``bound_deep_diff`` drops whole blocks over budget
    # and names them in the leading truncation marker. A wanted file absent
    # from the bounded text is only a problem when the marker names it as
    # dropped -- a scope file never changed in this PR has no hunks to inline.
    marker = _DIFF_TRUNCATION_MARKER.match(diff)
    if marker is not None and marker.group("dropped") is not None:
        dropped = {p.strip() for p in marker.group("dropped").split(",") if p.strip()}
        if wanted & dropped:
            return None

    selected: list[str] = []
    for block in _DIFF_BLOCK_SPLIT.split(diff):
        if _diff_block_path(block) in wanted:
            selected.append(block if block.endswith("\n") else block + "\n")

    if not selected:
        return None
    result = "".join(selected)
    if not fits_inline_diff_budget(result):
        return None
    return result


def inline_grounded_files(diff: str, files: list[str]) -> set[str]:
    """Return the set of ``files`` inline-grounded in ``diff`` (issue #731).

    A shard's inline grounding is all-or-nothing: ``_diff_blocks_for_files``
    returns the concatenated blocks or ``None`` (over the byte budget / no
    matching blocks). Returns ``set(files)`` when the blocks fit, else
    ``set()``. Pure and deterministic; feeds the coverage-evidence receipt.
    """
    if _diff_blocks_for_files(diff, files) is not None:
        return set(files)
    return set()


@dataclass
class DeepDiffBoundInfo:
    """Truncation statistics for one ``bound_deep_diff`` call.

    ``retained_bytes`` excludes the marker line; ``oversize_paths`` names the
    files whose single block exceeded the cap and was kept whole;
    ``dropped_paths`` names the files whose blocks were dropped whole by the
    bound (``marker`` carries the same names inline, so a consumer of the
    bounded text alone -- e.g. ``_diff_blocks_for_files`` -- can tell a
    dropped block from a file simply absent from the diff). Only a LEADING
    oversize block (one that arrives before any retained block) is kept
    whole and recorded; an oversize block arriving after a retained block is
    dropped whole, so it is absent from ``oversize_paths`` but present in
    ``dropped_paths``.
    """

    truncated: bool
    original_bytes: int = 0
    retained_bytes: int = 0
    total_blocks: int = 0
    retained_blocks: int = 0
    oversize_paths: list[str] = field(default_factory=list)
    dropped_paths: list[str] = field(default_factory=list)
    marker: str | None = None


def bound_deep_diff(diff: str, budget: int = INLINE_DIFF_BUDGET_BYTES) -> tuple[str, DeepDiffBoundInfo]:
    """Bound ``diff`` to ``budget`` bytes using whole ``diff --git``-block retention.

    Issue #644: the deep-flow gather stores this bounded value in
    ``ctx.data["diff"]`` so a pathological PR never feeds an oversized
    in-memory diff into the prompt pipeline. Reuses the shared
    ``_DIFF_BLOCK_SPLIT`` / ``_diff_block_path`` parse contract -- a retained
    block is byte-identical to its source block and no block is ever split
    mid-stream.

    At/under ``budget`` the input is returned unchanged with no marker
    (byte-for-byte identical to today, Must-have #4). Over ``budget`` whole
    blocks are retained while ``retained_bytes + block_bytes <= budget``; a
    single block that alone exceeds the cap is kept whole with its path
    recorded in ``oversize_paths`` -- but only when it is the leading block
    (no retained block yet). An oversize block arriving after a retained
    block is dropped whole and omitted from ``oversize_paths`` (block
    integrity outranks the soft bound; the prompt-inline budget already keeps
    it out of an oversized prompt). The
    returned value carries a leading ``# daydream: deep diff truncated:``
    marker line only when truncated; ``retained_bytes`` excludes the marker.
    When blocks were dropped the marker names them (``; dropped: <paths>``),
    mirroring ``dropped_paths`` in the info object.

    Blocks that fail to resolve a path via ``_diff_block_path`` (the leading
    empty split fragment, non-``diff --git`` elements) are skipped exactly as
    ``_diff_changed_files`` / ``_diff_blocks_for_files`` do.

    Returns:
        ``(bounded_diff, DeepDiffBoundInfo)``.
    """
    original_bytes = len(diff.encode("utf-8"))
    if original_bytes <= budget:
        return diff, DeepDiffBoundInfo(truncated=False, original_bytes=original_bytes, marker=None)

    retained: list[str] = []
    retained_bytes = 0
    total_blocks = 0
    oversize_paths: list[str] = []
    dropped_paths: list[str] = []
    for block in _DIFF_BLOCK_SPLIT.split(diff):
        block_path = _diff_block_path(block)
        if block_path is None:
            # Leading empty fragment / non-``diff --git`` element: never a block.
            continue
        total_blocks += 1
        block_bytes = len(block.encode("utf-8"))
        if retained_bytes + block_bytes <= budget:
            retained.append(block)
            retained_bytes += block_bytes
        elif not retained:
            # A single block larger than the cap is kept whole (block
            # integrity outranks the soft byte bound); the prompt-inline
            # budget keeps it out of an oversized prompt.
            retained.append(block)
            retained_bytes += block_bytes
            oversize_paths.append(block_path)
        else:
            # Block dropped whole; never split mid-stream. The path is
            # recorded so per-stack extraction can refuse a silently partial
            # inline when a stack mixes retained and dropped blocks.
            dropped_paths.append(block_path)

    dropped_clause = f"; dropped: {', '.join(dropped_paths)}" if dropped_paths else ""
    marker = (
        f"# daydream: deep diff truncated: {original_bytes} -> {retained_bytes} bytes "
        f"({len(retained)}/{total_blocks} blocks retained{dropped_clause})\n"
    )
    bounded = marker + "".join(retained)
    return bounded, DeepDiffBoundInfo(
        truncated=True,
        original_bytes=original_bytes,
        retained_bytes=retained_bytes,
        total_blocks=total_blocks,
        retained_blocks=len(retained),
        oversize_paths=oversize_paths,
        dropped_paths=dropped_paths,
        marker=marker,
    )


def _full_diff_pointer(diff_path: Path) -> str:
    """Shared paragraph pointing agents at the on-disk full PR diff."""
    return (
        f"The full PR diff (base..HEAD) is at {diff_path}. Read it directly; "
        "do NOT run `git diff` without a base ref -- on a clean branch that "
        "returns empty and hides committed changes."
    )


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
    return f"{_full_diff_pointer(diff_path)}\nFocus on hunks that touch your stack's files: {joined}."


def _frontier_read_instruction(frontier_files: list[str]) -> str:
    """Cross-shard interface read instruction for a sharded stack (issue #731).

    Names the sibling-shard files a shard's review depends on and instructs the
    agent to Read them for cross-shard context; these become
    ``dependency_frontier_read`` coverage-evidence candidates.
    """
    joined = ", ".join(frontier_files)
    return (
        f"Cross-shard interface file(s): this shard's review targets reference "
        f"files assigned to sibling shards. Read the following cross-shard "
        f"interface file(s) for context (they are NOT part of this shard's "
        f"review targets): {joined}."
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
    intent_authoritative: bool = False,
    include_alternatives: bool = True,
    frontier_files: list[str] | None = None,
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
        intent_authoritative: Issue #279. When True, the context pointers
            include the ``AUTHORITATIVE_INTENT_RULE`` precedence rule, because
            the intent phase was grounded by a fresh, head-matched PR description.
    """
    parts: list[str] = []
    pointer = _exploration_pointer(exploration_dir)
    if pointer:
        parts.append(pointer)
    settled = _settled_decisions_block(prior_commits)
    if settled:
        parts.append(settled)
    parts.append(CWD_GROUNDING_INSTRUCTION.format(cwd=cwd))
    parts.append(
        _context_pointers(
            intent_path=intent_path,
            alternatives_path=alternatives_path,
            intent_authoritative=intent_authoritative,
            include_alternatives=include_alternatives,
        )
    )
    parts.append(_confidence_and_convention_instructions())
    parts.append(_dependency_impact_instructions())
    parts.append(_stack_scope_instruction(stack_name, files))
    if frontier_files:
        parts.append(_frontier_read_instruction(frontier_files))
    parts.append(_diff_instruction(diff_path, files, inline_diff=inline_diff))
    parts.append(skill_invocation)
    parts.append(TEST_QUALITY_RUBRIC_INSTRUCTION)
    parts.append(ANTI_SLOP_RUBRIC_INSTRUCTION)
    parts.append(VERIFICATION_PROTOCOL_INSTRUCTION)
    parts.append(CONFIG_FLOW_TRACE_INSTRUCTION)
    parts.append(TRUST_MODEL_INSTRUCTION)
    if stack_name == "rust":
        parts.append(WIRE_CONTRACT_RUST_INSTRUCTION)
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
    intent_authoritative: bool = False,
    include_alternatives: bool = True,
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
        exploration_dir: When present, points the reviewer at the deterministic
            structural/import index in ``affected_files.md``.
        prior_commits: Oneline log of prior daydream commits on this branch.
        intent_authoritative: Issue #279. When True, the context pointers
            include the ``AUTHORITATIVE_INTENT_RULE`` precedence rule, because
            the intent phase was grounded by a fresh, head-matched PR description.
    """
    joined = ", ".join(files)
    parts: list[str] = []
    settled = _settled_decisions_block(prior_commits)
    if settled:
        parts.append(settled)
    parts.append(CWD_GROUNDING_INSTRUCTION.format(cwd=cwd))
    if exploration_dir is not None:
        parts.append(_exploration_pointer(exploration_dir))
    parts.append(
        _context_pointers(
            intent_path=intent_path,
            alternatives_path=alternatives_path,
            intent_authoritative=intent_authoritative,
            include_alternatives=include_alternatives,
        )
    )
    parts.append(
        f"You are the structural reviewer. The full change spans: {joined}. "
        f"The structural rubric applies repo-wide -- read any file in the "
        f"codebase as needed (Read/Grep/Bash) to judge whether canonical "
        f"helpers exist, file-size budgets are honored, and the change makes "
        f"the codebase easier or harder to live with."
    )
    parts.append(_full_diff_pointer(diff_path))
    parts.append(skill_invocation)
    parts.append(VERIFICATION_PROTOCOL_INSTRUCTION)
    parts.append(ANTI_SLOP_RUBRIC_INSTRUCTION)
    parts.append(CROSS_FILE_SYMBOL_EXISTENCE_INSTRUCTION)
    parts.append(TRUST_MODEL_INSTRUCTION)
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
    intent_authoritative: bool = False,
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
        intent_authoritative: Issue #279. When True, the context pointers
            include the ``AUTHORITATIVE_INTENT_RULE`` precedence rule, because
            the intent phase was grounded by a fresh, head-matched PR description.
    """
    parts: list[str] = []
    pointer = _exploration_pointer(exploration_dir)
    if pointer:
        parts.append(pointer)
    parts.append(CWD_GROUNDING_INSTRUCTION.format(cwd=cwd))
    parts.append(
        _context_pointers(
            intent_path=intent_path,
            alternatives_path=alternatives_path,
            intent_authoritative=intent_authoritative,
        )
    )
    parts.append(_full_diff_pointer(diff_path))
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


def build_supervise_prompt(
    *,
    supervise_input_path: Path,
    diff_path: Path,
    intent_path: Path,
    alternatives_path: Path,
    cwd: Path,
    exploration_dir: Path | None = None,
) -> str:
    """Assemble the batched canonical findings supervisor prompt."""
    parts: list[str] = []
    pointer = _exploration_pointer(exploration_dir)
    if pointer:
        parts.append(pointer)
    parts.append(CWD_GROUNDING_INSTRUCTION.format(cwd=cwd))
    parts.append(_context_pointers(intent_path=intent_path, alternatives_path=alternatives_path))
    parts.append(_full_diff_pointer(diff_path))
    parts.append(
        "Supervisor adjudication: review the canonical findings listed in "
        f"{supervise_input_path}. This is an adjudication pass, not a fresh "
        "review: do not invent findings or change their file, line, or id."
    )
    parts.append(
        "Return one JSON object matching the structured-output schema with one "
        "verdict per finding when possible. Each verdict must echo the canonical "
        "id and choose exactly one action: allow, drop, edit, or hold. Explain "
        "the decision in reason. For edit, revise only severity, confidence, "
        "description, rationale, or evidence; never file, line, or id. Missing "
        "verdicts are treated as allow by the host."
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
    """
    parts: list[str] = []
    pointer = _exploration_pointer(exploration_dir)
    if pointer:
        parts.append(pointer)
    parts.append(CWD_GROUNDING_INSTRUCTION.format(cwd=cwd))
    parts.append(_context_pointers(intent_path=intent_path, alternatives_path=alternatives_path))
    parts.append(_full_diff_pointer(diff_path))
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
    intent_authoritative: bool = False,
    resumed_from_arbiter: bool = False,
) -> str:
    """Assemble the cross-stack merge prompt (D-23..D-27).

    ``resumed_from_arbiter`` appends a stale-context warning: the resumed
    session replays pre-adjudication records, but the files on disk were
    rewritten after that turn. Every cold path leaves it False and produces
    today's prompt byte-identically — the prompt is fully self-sufficient
    without a resumed session.

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
        intent_authoritative: Issue #279. When True, the context lines include
            the ``AUTHORITATIVE_INTENT_RULE`` precedence rule immediately after
            the TTT intent summary line, because the intent phase was grounded
            by a fresh, head-matched PR description.
    """
    del output_path, structural_records_path  # appended/rendered by the host, not the prompt
    records_block = "\n".join(f"  - {p}" for p in per_stack_records_paths)
    parts: list[str] = []
    pointer = _exploration_pointer(exploration_dir)
    if pointer:
        parts.append(pointer)
    context_lines: list[str] = [
        f"TTT intent summary: {intent_path}",
        f"TTT alternative-review findings: {alternatives_path}",
        f"Dedup pre-filter candidate pairs: {dedup_candidates_path}",
    ]
    if intent_authoritative:
        context_lines.insert(1, AUTHORITATIVE_INTENT_BLOCK)  # right after the intent line
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
    if resumed_from_arbiter:
        # Resuming the arbiter's session replays ITS context, which holds the
        # PRE-adjudication records. The files on disk were rewritten after that
        # turn, so the resumed context is stale for exactly the inputs that
        # matter most.
        parts.append(
            "NOTE: this conversation is resumed from the arbitration turn. The "
            "per-stack record files listed above were REWRITTEN on disk after "
            "that turn (arbiter verdicts, and possibly suppression verdicts, "
            "were applied). You MUST re-read every record file from disk — the "
            "records held in the resumed context are pre-adjudication and are "
            "no longer authoritative."
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
        non-mutating commands (see `_render_bash_allowlist()`). The verifier writes nothing —
        the host persists the verdicts it returns as structured output.
      - The non-structural finding list is rendered inline below.
      - Empty issue list yields an empty verdict list (no error).

    The verdict schema is NOT dumped into the prompt: it reaches every backend
    through ``output_schema`` (claude natively, codex via a temp file, pi by
    appending its own instruction), so an inline copy is a duplicate the model
    pays for twice.

    Args:
        items: The non-structural (per-stack / cross-stack) canonical items to
            verify. Rendered inline into the prompt; verdicts are keyed by each
            item's canonical ``id`` (the verdict ``issue_id``).
        cwd: Absolute working directory the verifier runs in (grounds path resolution).
        output_path: Accepted and ignored — kept because dropping a prompt kwarg
            is a breaking extension change. The host writes the verdicts file.
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
        f"  - Bash is restricted to non-mutating commands only: {_render_bash_allowlist()}.\n"
        "  - Do NOT write, edit, or move files. Do NOT run `git commit`, "
        "`git add`, `git checkout`, `git reset`, `git stash`, or any other "
        "state-changing command."
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
        "  2. Locate every sibling implementation using the Grep tool "
        "(e.g. search for `impl <Trait> for` or `class X(<Iface>)`).\n"
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
        "Every verdict entry MUST include all four required fields, even when "
        "`unverified_assumptions` is an empty array."
    )
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
    intent_authoritative: bool = False,
    include_alternatives: bool = True,
    frontier_files: list[str] | None = None,
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
        intent_authoritative: Issue #279. When True, the context pointers
            include the ``AUTHORITATIVE_INTENT_RULE`` precedence rule, because
            the intent phase was grounded by a fresh, head-matched PR description.
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
    parts.append(
        _context_pointers(
            intent_path=intent_path,
            alternatives_path=alternatives_path,
            intent_authoritative=intent_authoritative,
            include_alternatives=include_alternatives,
        )
    )
    parts.append(_confidence_and_convention_instructions())
    parts.append(_dependency_impact_instructions())
    parts.append(_stack_scope_instruction("generic-fallback", files))
    if frontier_files:
        parts.append(_frontier_read_instruction(frontier_files))
    parts.append(_diff_instruction(diff_path, files, inline_diff=inline_diff))
    parts.append(
        "Review these files for correctness, clarity, and consistency with the "
        "author's intent. Apply language-agnostic review practices."
    )
    parts.append(VERIFICATION_PROTOCOL_INSTRUCTION)
    parts.append(CONFIG_FLOW_TRACE_INSTRUCTION)
    parts.append(TRUST_MODEL_INSTRUCTION)
    parts.append(WIRE_CONTRACT_GENERIC_INSTRUCTION)
    parts.append(f"Write your full review to {output_path}.")
    return "\n\n".join(parts)
