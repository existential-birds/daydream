"""Uncovered-diff-file sweep helpers (issue #309).

After per-stack reviews + parse, the deep flow computes which diff files NO
reviewer read (``analyze_coverage``), then dispatches a cheap second-pass
reviewer per uncovered file above a small budget threshold. This module holds
the pure, deterministic pieces: coverage computation, budget filtering
(hunk-size + capacity cap), and the sweep prompt builder.

The sweep's coverage set is computed HERE, not via ``analyzer.analyze_coverage``:
a read covers a diff file only at a path-component boundary (a read of
``notapi.py`` never covers ``api.py``) and only when the read tool call
carries a completed observation (an interrupted read never covers anything).
``daydream.eval.analyzer`` is imported read-only -- issue #316 owns that
module. The sweep prompt builder lives here, NOT in ``daydream.deep.prompts``
(issue #314 owns that module).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from daydream.deep.artifacts import per_stack_records_path
from daydream.deep.prompts import (
    _DIFF_BLOCK_SPLIT,
    VERIFICATION_PROTOCOL_INSTRUCTION,
    _diff_block_path,
)
from daydream.eval.analyzer import (
    _agent_label,
    _files_from_diff,
    _read_paths_for_call,
    load_trajectories,
)
from daydream.phases import (
    _confidence_and_convention_instructions,
    _dependency_impact_instructions,
    _exploration_pointer,
)


def _path_component_matches(absolute: str, relative: str) -> bool:
    """Whether an absolute read path corresponds to ``relative`` at a path boundary.

    ``absolute == relative`` (the read path is already repo-relative) or
    ``absolute`` ends with ``"/" + relative`` — the basename boundary. A bare
    ``endswith(relative)`` would let a read of ``/repo/notapi.py`` cover the
    changed file ``api.py``; the component boundary excludes suffix
    collisions. The sweep uses this matcher instead of ``analyzer._path_matches``
    so it never inherits that false positive (issue #316 owns the analyzer).
    """
    return absolute == relative or absolute.endswith("/" + relative)


def coverage_receipt_path(deep_dir: Path) -> Path:
    """Path to the run's structured coverage receipts (issue #731).

    Written at prompt-build time by ``phase_per_stack_reviews`` when sharding
    is enabled and consumed by ``compute_uncovered_files`` (Task 9/10).
    """
    return deep_dir / "coverage-receipts.json"


def write_coverage_receipts(
    deep_dir: Path, receipts: dict[str, dict[str, list[str]]]
) -> None:
    """Write the deterministic per-shard coverage receipts (issue #731).

    ``receipts`` maps ``stack_name`` -> ``{"assigned_files", "inline_files",
    "frontier_files"}``. JSON with sorted keys so the receipt is stable across
    runs (deterministic evidence for the sweep gate).
    """
    deep_dir.mkdir(parents=True, exist_ok=True)
    coverage_receipt_path(deep_dir).write_text(
        json.dumps(receipts, sort_keys=True), encoding="utf-8"
    )


def _completed_read_paths(
    trajectory: dict[str, Any], phases: set[str] | None = None
) -> set[str]:
    """Read paths from ``trajectory`` whose tool call carries a completed observation.

    A Read only covers a diff file when the read tool call is paired with a
    ToolResult in the SAME step's observation:
    ``observation.results[].source_call_id`` must equal the tool call's
    ``tool_call_id``. Tool-call IDs are scoped to individual invocations and
    are NOT required to be trajectory-global, so the completed set is built per
    step and never leaks across steps: an interrupted read whose ID collides
    with a completed read in another step stays uncovered (fail-open: the file
    gets swept, never skipped). An interrupted read (ToolStartEvent with no
    ToolResultEvent) returns no content, so it must NOT count as coverage.
    ``phases``, when given, restricts the steps considered to those whose
    ``extra.daydream_phase`` is in the set.
    """
    paths: set[str] = set()
    for step in trajectory.get("steps", []):
        if phases is not None and ((step.get("extra") or {}).get("daydream_phase")) not in phases:
            continue
        completed_call_ids: set[str] = set()
        for result in (step.get("observation") or {}).get("results") or []:
            if not isinstance(result, dict):
                continue
            call_id = result.get("source_call_id")
            if isinstance(call_id, str):
                completed_call_ids.add(call_id)
        for tc in step.get("tool_calls") or []:
            if tc.get("tool_call_id") not in completed_call_ids:
                continue
            paths.update(_read_paths_for_call(tc))
    return paths


def _parsed_finding_files(records_path: Path) -> set[str] | None:
    """Set of ``file`` fields across a completed shard's parsed records.

    Returns ``None`` when the records file is absent or unreadable -- an
    incomplete shard contributes ZERO inline/frontier coverage (fail-open: the
    reviewer failed/omitted, so its files stay uncovered and get swept).
    """
    try:
        records = json.loads(records_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(records, dict):
        # Issue #742: fresh-run per-stack records carry the dict shape
        # {"issues": [...], "verdicts": [...]}; the issues list is the
        # findings evidence either way (legacy bare lists pass through).
        findings = records.get("issues")
    elif isinstance(records, list):
        findings = records
    else:
        findings = None
    if not isinstance(findings, list):
        return None
    files: set[str] = set()
    for finding in findings:
        if isinstance(finding, dict) and isinstance(finding.get("file"), str):
            file = finding["file"]
            # A leading ``./`` is a legal path spelling since the grammar
            # relaxed (#572/#573); the reviewed-diff file set and the receipt
            # lists are always bare. Normalize once here (mirroring the fix
            # gate at orchestrator.py) so a ``./x`` finding contributes
            # inline/frontier evidence instead of failing every path-component
            # match and getting swept.
            if file.startswith("./"):
                file = file[2:]
            files.add(file)
    return files


def _receipt_covered_files(
    diff_files: list[str], receipts: dict[str, Any], deep_dir_path: Path
) -> tuple[set[str], dict[str, int]]:
    """Coverage from completed shards' inline/frontier evidence (issue #731).

    A diff file is ``inline_hunk_reviewed``-covered when it is in a shard's
    ``inline_files`` AND that shard's parsed records exist AND at least one
    parsed finding ``_path_component_matches`` it. ``dependency_frontier_read``
    is the same check over the shard's ``frontier_files``. Assignment/grounding
    alone never counts; a shard without a records file contributes zero.

    Returns the covered set plus per-evidence-type counts (a file covered by
    multiple types counts once per type it satisfies).
    """
    covered: set[str] = set()
    # Per-evidence-type covered sets so a file present in N shards' lists is
    # counted once per type, not once per (shard, type) -- the "once per type
    # it satisfies" contract in the docstring (finding 2).
    covered_by_type: dict[str, set[str]] = {
        "inline_hunk_reviewed": set(),
        "dependency_frontier_read": set(),
    }
    diff_set = set(diff_files)
    for stack_name, receipt in receipts.items():
        records_path = per_stack_records_path(deep_dir_path, stack_name)
        finding_files = _parsed_finding_files(records_path)
        if finding_files is None:
            continue  # incomplete shard contributes zero inline/frontier evidence
        for evidence_key, files_key in (
            ("inline_hunk_reviewed", "inline_files"),
            ("dependency_frontier_read", "frontier_files"),
        ):
            for f in receipt.get(files_key, []) or []:
                if f in diff_set and any(
                    _path_component_matches(ff, f) for ff in finding_files
                ):
                    covered.add(f)
                    covered_by_type[evidence_key].add(f)
    counts = {key: len(files) for key, files in covered_by_type.items()}
    return covered, counts


def resolve_per_stack_verdicts(
    *,
    assigned_files: list[str],
    declared_verdicts: list[dict[str, Any]],
    completed_read_paths: set[str],
    finding_files: set[str],
) -> list[dict[str, Any]]:
    """Reconcile declared per-file verdicts against completed-read evidence (issue #742).

    A per-stack reviewer's declared verdict is NOT the recorded truth: a
    ``clean`` verdict for an assigned file with no completed read of that file
    in the same review is downgraded to ``not_reviewed`` (routing the file to
    the uncovered sweep), never recorded as a pass. The gate reads evidence
    (``_completed_read_paths`` output + parsed finding files), never reviewer
    self-report.

    The final verdict per assigned file is resolved by evidence, in order:
    a parsed finding that path-component-matches the file wins (``has_findings``
    beats a read, beats ``clean``); otherwise a completed read that
    path-component-matches the file yields ``clean``; otherwise the file is
    ``not_reviewed``. This mirrors the read-detection the sweep already uses
    (``_path_component_matches``), so a clean shard that read its file is
    covered regardless of findings and an unread file is never recorded clean.

    Args:
        assigned_files: The stack's assigned files, in order.
        declared_verdicts: The reviewer-declared per-file verdict entries
            (``path`` / ``lines_read`` / ``verdict``), surfaced through the
            per-stack parse; evidence, not authority.
        completed_read_paths: Completed-read paths from the stack's own fork
            trajectory (``_completed_read_paths``).
        finding_files: ``file`` fields from the stack's parsed issues.

    Returns:
        Exactly one verdict dict per ``assigned_files`` path, in the given
        order: ``{"path", "lines_read", "verdict"}``, plus ``n_findings``
        when the verdict is ``has_findings``. The declared ``lines_read`` is
        preserved even when the verdict is downgraded to ``not_reviewed`` (the
        reviewer said it read N lines; the gate records it was not read). Pure
        and total: a declared verdict whose path is not in ``assigned_files``
        is ignored (never fabricated), and missing keys default safely.
    """
    declared_by_path: dict[str, dict[str, Any]] = {}
    for entry in declared_verdicts:
        if not isinstance(entry, dict):
            continue
        path = entry.get("path")
        if not isinstance(path, str) or path not in assigned_files:
            continue  # non-assigned declared paths are ignored, never fabricated
        declared_by_path[path] = entry

    out: list[dict[str, Any]] = []
    for path in assigned_files:
        declared = declared_by_path.get(path, {})
        lines_read = declared.get("lines_read", 0)
        matching_findings = [
            ff for ff in finding_files if _path_component_matches(ff, path)
        ]
        if matching_findings:
            # A finding beats a read and beats a declared clean.
            out.append(
                {
                    "path": path,
                    "lines_read": lines_read,
                    "verdict": "has_findings",
                    "n_findings": len(matching_findings),
                }
            )
        elif any(_path_component_matches(r, path) for r in completed_read_paths):
            out.append({"path": path, "lines_read": lines_read, "verdict": "clean"})
        else:
            out.append(
                {"path": path, "lines_read": lines_read, "verdict": "not_reviewed"}
            )
    return out


def compute_uncovered_files(
    daydream_dir: Path,
    session_id: str | None,
    *,
    receipts: dict[str, Any] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Return the diff files no reviewer read, plus the coverage stats.

    Mirrors ``analyzer.analyze_coverage``'s shape but applies the sweep's own
    matching rules: reads are counted only when the tool call is completed
    (paired ToolResult observation) and a read path covers a diff file only at
    a path-component boundary. A read of ``/repo/notapi.py`` therefore never
    covers ``api.py``, and an interrupted read never covers anything. Both
    rules fail open — a genuinely unread file is swept, never skipped.

    Issue #731: when ``receipts`` (the structured coverage receipts written at
    prompt-build time when sharding is enabled) is provided, a diff file is
    additionally covered by ``inline_hunk_reviewed`` / ``dependency_frontier_read``
    evidence per ``_receipt_covered_files``. When ``receipts`` is ``None`` (or
    the file is absent) behavior is byte-identical to today (Reads only).

    Args:
        daydream_dir: The run's ``.daydream`` directory (parent of the
            ``deep/`` artifact dir).
        session_id: The run's recorder session id, or ``None`` to resolve the
            most recent trajectory.
        receipts: Optional structured coverage receipts (issue #731); ``None``
            keeps the forensic (Reads-only) path byte-identical.

    Returns:
        ``(uncovered_files, stats)`` where ``stats`` is the coverage dict
        (``files_in_diff`` / ``files_read_by_reviewers`` / ``coverage_ratio`` /
        ``uncovered_files``) plus ``coverage_by_evidence`` only when ``receipts``
        is provided, and ``uncovered_files`` is its sorted list of diff files
        no review agent covered.
    """
    trajectories = load_trajectories(daydream_dir, session_id=session_id)
    diff_files = _files_from_diff(daydream_dir / "diff.patch")

    review_reads: set[str] = set()
    for traj in trajectories["forked"]:
        if _agent_label(traj["_source_file"]).startswith("deep-"):
            review_reads.update(_completed_read_paths(traj))
    if trajectories["main"]:
        review_reads.update(
            _completed_read_paths(trajectories["main"], phases={"deep", "alternatives"})
        )

    covered = {df for df in diff_files if any(_path_component_matches(r, df) for r in review_reads)}
    source_covered = len(covered)
    if receipts:
        receipt_covered, receipt_counts = _receipt_covered_files(
            diff_files, receipts, daydream_dir / "deep"
        )
        covered |= receipt_covered
    uncovered = sorted(set(diff_files) - covered)

    stats: dict[str, Any] = {
        "files_in_diff": len(diff_files),
        "files_read_by_reviewers": len(covered),
        "coverage_ratio": round(len(covered) / len(diff_files), 4) if diff_files else 1.0,
        "uncovered_files": uncovered,
    }
    if receipts:
        # Issue #731: per-evidence-type counts surface ONLY when sharding was
        # enabled this run (receipts provided); absent otherwise (finding 3), so
        # the Reads-only path stays byte-identical to today's stats artifact.
        stats["coverage_by_evidence"] = {
            "source_read": source_covered,
            **receipt_counts,
        }
    return uncovered, stats


def diff_block_for_file(diff: str, file: str) -> str | None:
    """Return the unified-diff block for ``file``, or ``None`` when absent.

    Reuses the shared ``diff --git`` block splitter and post-state path
    resolution from ``daydream.deep.prompts`` so the unified-diff parse
    contract is not duplicated here.
    """
    for block in _DIFF_BLOCK_SPLIT.split(diff):
        if _diff_block_path(block) == file:
            return block if block.endswith("\n") else block + "\n"
    return None


def hunk_change_line_count(hunks: str) -> int:
    """Count added/removed lines (``+``/``-``) in a diff block, headers excluded.

    ``+++``/``---`` file headers carry leading plus/minus signs but are not
    content changes, so they are excluded from the count.
    """
    count = 0
    for line in hunks.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+") or line.startswith("-"):
            count += 1
    return count


def filter_sweepable_files(
    uncovered_files: list[str],
    diff: str,
    *,
    min_hunk_lines: int,
    max_files: int,
) -> tuple[list[str], list[str], list[str]]:
    """Budget-filter the uncovered list into the files actually swept.

    A file is sweepable only when its hunks contain at least ``min_hunk_lines``
    added/removed lines -- a trivially small hunk does not justify a second
    pass. The sweepable set is capped at ``max_files`` in diff order (the
    uncovered list arrives sorted); the remainder is reported as
    skipped-for-capacity rather than silently dropped.

    Returns:
        ``(swept_files, skipped_small_hunk_files, skipped_capacity_files)``.
        The two skip lists name the omitted files in diff order (mirroring
        ``swept_files``) so consumers can audit exactly which files the
        hunk-size floor and the capacity cap left out; the integer skip counts
        are ``len(...)`` of these lists.
    """
    swept: list[str] = []
    skipped_small: list[str] = []
    for file in uncovered_files:
        block = diff_block_for_file(diff, file)
        if block is None or hunk_change_line_count(block) < min_hunk_lines:
            skipped_small.append(file)
            continue
        swept.append(file)
    skipped_capacity = swept[max_files:]
    return swept[:max_files], skipped_small, skipped_capacity


def build_uncovered_sweep_prompt(
    *,
    file: str,
    hunks: str,
    intent_path: Path,
    cwd: Path,
    output_path: Path,
    exploration_dir: Path | None = None,
) -> str:
    """Build the second-pass sweep reviewer prompt for one uncovered file.

    The reviewer scopes itself to ``file``'s hunks only (no per-stack reviewer
    read this file), uses the TTT intent for authorial context, and writes its
    findings to ``output_path``. The sweep reviewer is held to the same standard
    as ordinary per-stack reviewers -- its findings are parsed into
    ``PER_STACK_RECORD_SCHEMA`` and merged as ordinary findings -- so the prompt
    is composed from the CANONICAL deep prompt primitives (imported from
    ``daydream.phases`` / ``daydream.deep.prompts``, read-only): the exploration
    pointer, the Confidence and Convention Rules (incl. QUAL-04 error
    semantics), the Dependency Impact instructions, and the full
    ``VERIFICATION_PROTOCOL_INSTRUCTION``. The gates are embedded inline -- not
    routed through ``Backend.format_skill_invocation`` and NOT loaded from a
    skill file -- because the reviewer runs with cwd set to the reviewed repo,
    where a bare ``read review-verification-protocol/SKILL.md`` resolves against
    that repo and silently drops the gates (same rationale as the canonical
    constant).
    """
    parts: list[str] = []
    pointer = _exploration_pointer(exploration_dir)
    if pointer:
        parts.append(pointer)
    parts.append(
        "You are the uncovered file sweep reviewer for the deep-review "
        "pipeline (issue #309).\n"
        f"The changed file {file} was NOT read by any per-stack reviewer, "
        "so you are the second pass that covers it. Review ONLY this "
        "file's hunks below -- correctness, error handling, test quality, "
        "and maintainability. Do NOT review other files."
    )
    parts.append(f"TTT author intent is at {intent_path}. Read it before starting.")
    parts.append(
        f"Relevant diff hunks for {file} (inlined; do NOT re-read "
        f"diff.patch for these):\n{hunks.rstrip()}"
    )
    parts.append(
        "Read the source file FIRST; you may only comment on hunks you have "
        "read. The inlined hunks are not a substitute for reading the file."
    )
    parts.append(_confidence_and_convention_instructions())
    parts.append(_dependency_impact_instructions())
    parts.append(VERIFICATION_PROTOCOL_INSTRUCTION)
    parts.append(f"Work in {cwd}. Write your full review to {output_path}.")
    return "\n\n".join(parts)
