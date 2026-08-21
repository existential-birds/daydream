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
    _read_paths_for_call,
    _records_issues,
    load_trajectories,
)
from daydream.hunk_index import (
    change_line_count,
    files_in_index,
    hunk_index_path,
    load_hunk_index,
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


def _same_repo_relative(a: str, b: str) -> bool:
    """Exact dir-aware match between two repo-relative paths.

    ``_path_component_matches`` pairs an absolute read path (from a tool call)
    with a repo-relative diff file, where a basename-boundary fallback is
    needed. When BOTH operands are repo-relative (a parsed finding ``file`` vs
    an assigned or receipt file), that one-directional basename fallback
    misattributes: ``lib/util.py`` ``endswith("/util.py")`` matches the
    assigned top-level ``util.py``. Repo-relative operands share the same
    normalization, so exact equality is the only correct comparison.
    """
    return a == b


def coverage_receipt_path(deep_dir: Path) -> Path:
    """Path to the run's structured coverage receipts (issue #731).

    Written at prompt-build time by ``phase_per_stack_reviews`` on every deep
    run (decoupled from sharding, #740) and consumed by
    ``compute_uncovered_files`` (Task 9/10).
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


def _strip_dot_slash(path: str) -> str:
    """Normalize a leading ``./`` off a repo-relative path (issue #740).

    A leading ``./`` is a legal path spelling since the grammar relaxed
    (#572/#573); the reviewed-diff file set and the receipt lists are always
    bare, so stripping once in a single canonical location keeps a ``./x``
    finding matching its assigned file rather than failing every
    path-component match and getting swept. The findings fallback, the verdict
    path, and the orchestrator reconciliation all route through here so a
    future normalization change is applied in one place.
    """
    if path.startswith("./"):
        return path[2:]
    return path


def _finding_files_from_records(findings: list[Any]) -> set[str]:
    """Normalized ``file`` fields across parsed finding records (issue #742).

    Shared between the findings-only fallback (:func:`_parsed_finding_files`)
    and the per-stack verdict reconciliation in the orchestrator (in-memory
    parsed records) so the ``./`` strip lives in one place
    (:func:`_strip_dot_slash`): a leading ``./`` is a legal path spelling
    since the grammar relaxed (#572/#573), and the reviewed-diff file set and
    the receipt lists are always bare, so normalizing once keeps a ``./x``
    finding matching its assigned file rather than failing every
    path-component match and getting swept.
    """
    files: set[str] = set()
    for finding in findings:
        if isinstance(finding, dict) and isinstance(finding.get("file"), str):
            file = finding["file"]
            files.add(_strip_dot_slash(file))
    return files


def _load_records_or_none(records_path: Path) -> Any | None:
    """Load and parse a per-stack records file, or ``None`` when absent/unreadable.

    The single ``json.loads`` / ``(OSError, ValueError)`` opener shared by
    :func:`_parsed_finding_files` and :func:`_parsed_covered_files` so the
    fail-open loader lives in one place instead of being copy-pasted. A missing
    or malformed records file degrades to ``None``; callers treat that as an
    incomplete shard contributing ZERO coverage (fail-open: swept, never
    skipped).
    """
    try:
        return json.loads(records_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _parsed_finding_files(records: Any) -> set[str] | None:
    """Set of ``file`` fields across a completed shard's parsed records.

    Operates on an already-loaded records object (:func:`_load_records_or_none`)
    so the findings-only extraction is shared rather than copy-pasted, and used
    as the fallback by :func:`_parsed_covered_files` when a shard's records
    predate the evidence-gated ``verdicts`` array. Returns ``None`` when the
    records shape carries no parseable findings (a non-list load, or a dict
    without an ``issues`` list); the caller keeps its own fail-open for that
    case.
    """
    findings = _records_issues(records)
    if findings is None:
        return None
    return _finding_files_from_records(findings)


def _parsed_covered_files(records_path: Path) -> set[str] | None:
    """Set of files a completed shard's evidence-gated verdicts mark covered.

    A diff file is covered when the shard's persisted ``verdicts`` array
    records it as ``clean`` or ``has_findings`` (the reviewer read it and the
    verdict is evidence-backed, never raw declared self-report). An unread
    file records ``not_reviewed`` and never enters the set. Legacy record
    shapes -- a bare findings list, a dict without a ``verdicts`` key, or an
    empty ``verdicts`` list -- fall back to the findings-only set (:func:`_parsed_finding_files`).

    Returns ``None`` when the records file is absent or unreadable -- an
    incomplete shard contributes ZERO inline/frontier coverage (fail-open: the
    reviewer failed/omitted, so its files stay uncovered and get swept).
    """
    records = _load_records_or_none(records_path)
    if records is None:
        return None
    if isinstance(records, dict):
        verdicts = records.get("verdicts")
        if isinstance(verdicts, list) and verdicts:
            covered: set[str] = set()
            for entry in verdicts:
                if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                    continue
                if entry.get("verdict") not in {"clean", "has_findings"}:
                    continue  # not_reviewed (or any other) never credits
                path = _strip_dot_slash(entry["path"])
                covered.add(path)
            return covered
    return _parsed_finding_files(records)


def _receipt_covered_files(
    diff_files: list[str], receipts: dict[str, Any], deep_dir_path: Path
) -> tuple[set[str], dict[str, int]]:
    """Coverage from completed shards' inline/frontier evidence (issue #731).

    A diff file is ``inline_hunk_reviewed``-covered when it is in a shard's
    ``inline_files`` AND that shard's parsed records exist AND the shard's
    evidence-gated ``verdicts`` array (or, for legacy records, at least one
    parsed finding) marks it covered. A ``clean`` verdict -- a reviewed file
    with no findings -- credits the file, so a clean review is never swept;
    a ``not_reviewed`` verdict never credits. ``dependency_frontier_read``
    credits a shard's ``frontier_files`` when ANY completed shard's evidence
    covers the file: a frontier file lives in a SIBLING shard, so its read
    evidence is recorded in the sibling's records, never the shard that merely
    lists it as a frontier. Assignment/grounding alone never counts; a shard
    without a records file contributes zero inline evidence.

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

    # A shard's ``frontier_files`` are files in OTHER shards of the same
    # language (_assign_frontiers, sharding.py), so the read evidence backing a
    # frontier entry lives in the SIBLING shard's records, never the shard that
    # lists it as a frontier. Merge every shard's completed covered set once so
    # the frontier branch credits a file when its owning (or any) shard actually
    # read it -- else a frontier file never appears in a shard's own records and
    # ``dependency_frontier_read`` can never fire (issue #740 regression).
    shard_covered: dict[str, set[str]] = {}
    frontier_evidence: set[str] = set()
    for stack_name, _ in receipts.items():
        loaded = _parsed_covered_files(per_stack_records_path(deep_dir_path, stack_name))
        if loaded is not None:
            shard_covered[stack_name] = loaded
            frontier_evidence |= loaded

    for stack_name, receipt in receipts.items():
        inline_covered = shard_covered.get(stack_name)
        # Inline evidence is gated on THIS shard's own records: a shard without
        # a records file contributes zero inline evidence (fail-open).
        if inline_covered is not None:
            for f in receipt.get("inline_files", []) or []:
                if f in diff_set and any(
                    _same_repo_relative(ff, f) for ff in inline_covered
                ):
                    covered.add(f)
                    covered_by_type["inline_hunk_reviewed"].add(f)
        # Frontier evidence is gated on the SIBLING union, NOT this shard's own
        # records: a frontier file lives in a sibling shard (its read evidence
        # is recorded in the sibling's records, never the shard that merely
        # lists it as a frontier). So frontier credit fires whenever ANY
        # completed shard read the file, independent of whether the listing
        # shard itself completed (issue #740). Otherwise a shard with missing
        # records suppresses frontier credit for the files it merely lists.
        for f in receipt.get("frontier_files", []) or []:
            if f in diff_set and any(
                _same_repo_relative(ff, f) for ff in frontier_evidence
            ):
                covered.add(f)
                covered_by_type["dependency_frontier_read"].add(f)
    counts = {key: len(files) for key, files in covered_by_type.items()}
    return covered, counts


def _verdict(path: str, lines_read: int, verdict: str, n_findings: int) -> dict[str, Any]:
    """Build one conformant per-file verdict dict (issue #742).

    Collapses the three near-identical ``out.append({...})`` blocks in
    :func:`resolve_per_stack_verdicts`, which differ only in ``verdict`` and
    ``n_findings``. Every returned dict conforms to ``PER_STACK_RECORD_SCHEMA``'s
    required ``path`` / ``lines_read`` / ``verdict`` / ``n_findings`` keys.
    """
    return {
        "path": path,
        "lines_read": lines_read,
        "verdict": verdict,
        "n_findings": n_findings,
    }


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
    a parsed finding that exactly matches the file (``_same_repo_relative``) wins (``has_findings``
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
        order: ``{"path", "lines_read", "verdict", "n_findings"}``. ``n_findings``
        is the count of parsed findings matching the path (0 for ``clean`` /
        ``not_reviewed``), so every recorded verdict conforms to
        ``PER_STACK_RECORD_SCHEMA``'s required ``n_findings`` key. The declared
        ``lines_read`` is preserved even when the verdict is downgraded to
        ``not_reviewed`` (the reviewer said it did read N lines; the gate
        records it was not read). Pure and total: a declared verdict whose path
        is not in ``assigned_files`` is ignored (never fabricated), and missing
        keys default safely.
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
            ff for ff in finding_files if _same_repo_relative(ff, path)
        ]
        if matching_findings:
            # A finding beats a read and beats a declared clean.
            out.append(_verdict(path, lines_read, "has_findings", len(matching_findings)))
        elif any(_path_component_matches(r, path) for r in completed_read_paths):
            # A completed read that matches the file yields clean.
            out.append(_verdict(path, lines_read, "clean", 0))
        else:
            # No finding and no completed read: never recorded as a pass.
            out.append(_verdict(path, lines_read, "not_reviewed", 0))
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
    prompt-build time on every deep run, decoupled from sharding) is provided,
    a diff file is
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
    # The changed-file set comes from the persisted hunk index (written at
    # gather right after diff materialization) rather than re-reading
    # ``diff.patch``. Fail-open: ``load_hunk_index`` never raises and degrades
    # a missing index to no changed files -- but that is NOT evidence of full
    # coverage. An absent index leaves the changed-file set unenumerated, so
    # reporting it as ``diff_files == []`` (ratio 1.0, uncovered []) would let
    # a genuine coverage gap masquerade as a clean pass. Surfaced below instead.
    hunk_index_missing = not hunk_index_path(daydream_dir).is_file()
    diff_files = files_in_index(load_hunk_index(daydream_dir))

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
    if hunk_index_missing:
        # Issue #336: a missing index is NOT a clean empty diff. Fail-open is
        # preserved (``load_hunk_index`` still never raises); only the
        # reporting changes so the unenumerated changed-file set is surfaced as
        # a gap rather than silently rendered as full coverage.
        stats["coverage_ratio"] = None
        stats["hunk_index_missing"] = True
    if receipts:
        # Issue #731: per-evidence-type counts surface whenever receipts are
        # provided (written and loaded on every deep run, decoupled from
        # sharding -- #740); absent otherwise, so the Reads-only path stays
        # byte-identical to its receipts-free stats artifact.
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


def filter_sweepable_files(
    uncovered_files: list[str],
    index: dict[str, Any],
    *,
    min_hunk_lines: int,
    max_files: int,
) -> tuple[list[str], list[str], list[str]]:
    """Budget-filter the uncovered list into the files actually swept.

    ``index`` is the persisted hunk index (as loaded by
    ``daydream.hunk_index.load_hunk_index``) -- the run-time authority for
    changed-file hunk sizes. A file is sweepable only when its index entry's
    ``added_total + removed_total`` is at least ``min_hunk_lines`` -- a
    trivially small hunk does not justify a second pass. A file absent from
    the index is treated as too small (``skipped_small``), mirroring today's
    ``block is None`` behavior. The sweepable set is capped at ``max_files``
    in diff order (the uncovered list arrives sorted); the remainder is
    reported as skipped-for-capacity rather than silently dropped.

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
        info = index.get(file)
        if info is None or change_line_count(index, file) < min_hunk_lines:
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
