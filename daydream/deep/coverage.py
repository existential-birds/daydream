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

from pathlib import Path
from typing import Any

from daydream.deep.prompts import _DIFF_BLOCK_SPLIT, _diff_block_path
from daydream.eval.analyzer import (
    _agent_label,
    _files_from_diff,
    _read_paths_for_call,
    load_trajectories,
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


def _completed_read_paths(
    trajectory: dict[str, Any], phases: set[str] | None = None
) -> set[str]:
    """Read paths from ``trajectory`` whose tool call carries a completed observation.

    A Read only covers a diff file when the read tool call is paired with a
    ToolResult in the same step's observation: ``observation.results[].source_call_id``
    must equal the tool call's ``tool_call_id``. An interrupted read
    (ToolStartEvent with no ToolResultEvent) returns no content, so it must NOT
    count as coverage — the sweep treats it as uncovered (fail-open: the file
    gets swept, never skipped). ``phases``, when given, restricts the steps
    considered to those whose ``extra.daydream_phase`` is in the set.
    """
    completed_call_ids: set[str] = set()
    for step in trajectory.get("steps", []):
        for result in (step.get("observation") or {}).get("results") or []:
            if not isinstance(result, dict):
                continue
            call_id = result.get("source_call_id")
            if isinstance(call_id, str):
                completed_call_ids.add(call_id)
    paths: set[str] = set()
    for step in trajectory.get("steps", []):
        if phases is not None and ((step.get("extra") or {}).get("daydream_phase")) not in phases:
            continue
        for tc in step.get("tool_calls") or []:
            if tc.get("tool_call_id") not in completed_call_ids:
                continue
            paths.update(_read_paths_for_call(tc))
    return paths


def compute_uncovered_files(
    daydream_dir: Path, session_id: str | None
) -> tuple[list[str], dict[str, Any]]:
    """Return the diff files no reviewer read, plus the coverage stats.

    Mirrors ``analyzer.analyze_coverage``'s shape but applies the sweep's own
    matching rules: reads are counted only when the tool call is completed
    (paired ToolResult observation) and a read path covers a diff file only at
    a path-component boundary. A read of ``/repo/notapi.py`` therefore never
    covers ``api.py``, and an interrupted read never covers anything. Both
    rules fail open — a genuinely unread file is swept, never skipped.

    Args:
        daydream_dir: The run's ``.daydream`` directory (parent of the
            ``deep/`` artifact dir).
        session_id: The run's recorder session id, or ``None`` to resolve the
            most recent trajectory.

    Returns:
        ``(uncovered_files, stats)`` where ``stats`` is the coverage dict
        (``files_in_diff`` / ``files_read_by_reviewers`` / ``coverage_ratio`` /
        ``uncovered_files``) and ``uncovered_files`` is its sorted list of diff
        files no review agent read.
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
    uncovered = sorted(set(diff_files) - covered)

    return uncovered, {
        "files_in_diff": len(diff_files),
        "files_read_by_reviewers": len(covered),
        "coverage_ratio": round(len(covered) / len(diff_files), 4) if diff_files else 1.0,
        "uncovered_files": uncovered,
    }


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
) -> tuple[list[str], int, int]:
    """Budget-filter the uncovered list into the files actually swept.

    A file is sweepable only when its hunks contain at least ``min_hunk_lines``
    added/removed lines -- a trivially small hunk does not justify a second
    pass. The sweepable set is capped at ``max_files`` in diff order (the
    uncovered list arrives sorted); the remainder is reported as
    skipped-for-capacity rather than silently dropped.

    Returns:
        ``(swept_files, skipped_small_hunks, skipped_capacity)``.
    """
    swept: list[str] = []
    skipped_small = 0
    for file in uncovered_files:
        block = diff_block_for_file(diff, file)
        if block is None or hunk_change_line_count(block) < min_hunk_lines:
            skipped_small += 1
            continue
        swept.append(file)
    total_sweepable = len(swept)
    skipped_capacity = max(0, total_sweepable - max_files)
    return swept[:max_files], skipped_small, skipped_capacity


_SWEEP_VERIFICATION_GATES = (
    "Before writing findings, apply the review-verification gates (stated "
    "inline here -- no skill file read is required):\n"
    "  Gate-0 anti-confabulation: echo the exact artifact you are judging -- "
    "file:line plus the cited code, read freshly in THIS turn, not recalled.\n"
    "  Gate 1 (anchor): read the full enclosing symbol/module, not just the "
    "hunk; state the file path and line range you are judging.\n"
    "  Gate 2 (evidence): produce an artifact for the finding's type -- pasted "
    'tool output, a file:line citation, or an explicit "none" after a search.\n'
    "  Gate 3 (severity): calibrate severity to impact; a request for net-new "
    "code outside the diff is Informational only.\n"
    "Do NOT report a finding that fails any gate."
)


def build_uncovered_sweep_prompt(
    *,
    file: str,
    hunks: str,
    intent_path: Path,
    cwd: Path,
    output_path: Path,
) -> str:
    """Build the second-pass sweep reviewer prompt for one uncovered file.

    The reviewer scopes itself to ``file``'s hunks only (no per-stack reviewer
    read this file), uses the TTT intent for authorial context, and writes its
    findings to ``output_path``. The verification gates are embedded inline --
    not routed through ``Backend.format_skill_invocation`` and NOT loaded from
    a skill file -- because the reviewer runs with cwd set to the reviewed
    repo, where a bare ``read review-verification-protocol/SKILL.md`` resolves
    against that repo and silently drops the gates (same rationale as
    ``VERIFICATION_PROTOCOL_INSTRUCTION`` in ``daydream.deep.prompts``).
    """
    return "\n\n".join(
        [
            "You are the uncovered file sweep reviewer for the deep-review "
            "pipeline (issue #309).\n"
            f"The changed file {file} was NOT read by any per-stack reviewer, "
            "so you are the second pass that covers it. Review ONLY this "
            "file's hunks below -- correctness, error handling, test quality, "
            "and maintainability. Do NOT review other files.",
            f"TTT author intent is at {intent_path}. Read it before starting.",
            f"Relevant diff hunks for {file} (inlined; do NOT re-read "
            f"diff.patch for these):\n{hunks.rstrip()}",
            "For whole-file context beyond these hunks you MAY Read the source "
            "file directly.",
            _SWEEP_VERIFICATION_GATES,
            f"Work in {cwd}. Write your full review to {output_path}.",
        ]
    )
