"""Structured source-read evidence, verdict reconciliation, and sweep targeting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from daydream.deep.artifacts import per_stack_records_path
from daydream.deep.prompts import (
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
    files_in_index,
    hunk_index_path,
    load_hunk_index,
)
from daydream.phases import (
    _confidence_and_convention_instructions,
    _dependency_impact_instructions,
    _exploration_pointer,
)
from daydream.prompt_budget import INLINE_DIFF_BUDGET_BYTES, truncate_utf8_to_budget
from daydream.repository_paths import strip_dot_slash
from daydream.severity import SEVERITY_RUBRIC


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


def _result_credits_coverage(result: dict[str, Any]) -> bool:
    """Whether a paired ``ToolResult`` observation establishes read coverage.

    Reads ``result["extra"]`` when it is a dict and returns ``False`` iff the
    observation is damaged: ``is_error`` is ``True``, ``cancelled`` is ``True``,
    ``status`` is ``"interrupted"``, ``truncated`` is ``True``, or ``exit_code``
    is an ``int`` and non-zero. Any other outcome credits the read. Absent or
    non-dict ``extra`` is not failure -- Claude/Pi report no structured
    metadata, and treating absence as failure would zero their read credit.
    """
    extra = result.get("extra")
    if not isinstance(extra, dict):
        return True
    if extra.get("is_error") is True:
        return False
    if extra.get("cancelled") is True:
        return False
    if extra.get("status") == "interrupted":
        return False
    if extra.get("truncated") is True:
        return False
    exit_code = extra.get("exit_code")
    if isinstance(exit_code, int) and exit_code != 0:
        return False
    return True


def _completed_read_paths(
    trajectory: dict[str, Any], phases: set[str] | None = None
) -> set[str]:
    """Read paths from ``trajectory`` whose tool call carries a completed observation.

    A Read only covers a diff file when the read tool call is paired with a
    ToolResult in the SAME step's observation:
    ``observation.results[].source_call_id`` must equal the tool call's
    ``tool_call_id``. The paired result must also establish a successful
    observation (:func:`_result_credits_coverage`): a result marked failed,
    cancelled, interrupted, truncated, or with a non-zero exit code credits
    nothing. Tool-call IDs are scoped to individual invocations and are NOT
    required to be trajectory-global, so the completed set is built per step
    and never leaks across steps: an interrupted read whose ID collides with a
    completed read in another step stays uncovered (fail-open: the file gets
    swept, never skipped). ``phases``, when given, restricts the steps
    considered to those whose ``extra.daydream_phase`` is in the set.
    """
    paths: set[str] = set()
    for step in trajectory.get("steps", []):
        if phases is not None and ((step.get("extra") or {}).get("daydream_phase")) not in phases:
            continue
        completed_call_ids: set[str] = set()
        for result in (step.get("observation") or {}).get("results") or []:
            if not isinstance(result, dict):
                continue
            if not _result_credits_coverage(result):
                continue
            call_id = result.get("source_call_id")
            if isinstance(call_id, str):
                completed_call_ids.add(call_id)
        for tc in step.get("tool_calls") or []:
            if tc.get("tool_call_id") not in completed_call_ids:
                continue
            paths.update(_read_paths_for_call(tc))
    return paths


def _read_coverage_unverifiable(
    trajectory: dict[str, Any], phases: set[str] | None = None
) -> bool:
    """Opaque shell calls cannot establish per-file read outcomes."""
    return any(
        str(call.get("function_name", "")).casefold() in {"shell", "bash", "exec_command"}
        for step in trajectory.get("steps", [])
        if phases is None or (step.get("extra") or {}).get("daydream_phase") in phases
        for call in step.get("tool_calls") or []
    )


def _finding_files_from_records(findings: list[Any]) -> set[str]:
    """Normalized ``file`` fields across parsed finding records (issue #742).

    Shared between the findings-only fallback in :func:`_parsed_covered_files`
    and the per-stack verdict reconciliation in the orchestrator (in-memory
    parsed records) so the ``./`` strip lives in one place
    (:func:`strip_dot_slash`): a leading ``./`` is a legal path spelling
    since the grammar relaxed (#572/#573), and the reviewed-diff file set and
    the receipt lists are always bare, so normalizing once keeps a ``./x``
    finding matching its assigned file rather than failing every
    path-component match and getting swept.
    """
    files: set[str] = set()
    for finding in findings:
        if isinstance(finding, dict) and isinstance(finding.get("file"), str):
            file = finding["file"]
            files.add(strip_dot_slash(file))
    return files


def _parsed_covered_files(records_path: Path, *, require_read_evidence: bool = False) -> set[str] | None:
    """Set of files a completed shard's evidence-gated verdicts mark covered.

    A diff file is covered when the shard's persisted ``verdicts`` array
    records it as ``clean`` or ``has_findings`` (the reviewer read it and the
    verdict is evidence-backed, never raw declared self-report). An unread
    file records ``not_reviewed`` and never enters the set. Legacy record
    shapes -- a bare findings list, a dict without a ``verdicts`` key, or an
    empty ``verdicts`` list -- fall back to the findings-only set below.

    Returns ``None`` when the records file is absent or unreadable -- an
    incomplete shard contributes ZERO inline/frontier coverage (fail-open: the
    reviewer failed/omitted, so its files stay uncovered and get swept).
    """
    try:
        records = json.loads(records_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
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
                if require_read_evidence and entry.get("source_read_status") == "unverifiable":
                    continue  # retained findings do not prove an opaque source read
                path = strip_dot_slash(entry["path"])
                covered.add(path)
            return covered
    findings = _records_issues(records)
    if findings is None:
        return None
    return _finding_files_from_records(findings)


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
        loaded = _parsed_covered_files(
            per_stack_records_path(deep_dir_path, stack_name), require_read_evidence=True
        )
        if loaded is not None:
            shard_covered[stack_name] = loaded
            frontier_evidence |= loaded

    for stack_name, receipt in receipts.items():
        if not isinstance(receipt, dict):
            continue
        inline_covered = shard_covered.get(stack_name)
        packet_files = receipt.get("source_packet_files", [])
        if isinstance(packet_files, list) and packet_files:
            packet_verdicts = _parsed_covered_files(per_stack_records_path(deep_dir_path, stack_name))
            packet_covered = covered_by_type.setdefault("source_packet_reviewed", set())
            for path in packet_files:
                if (
                    isinstance(path, str)
                    and path in diff_set
                    and packet_verdicts is not None
                    and path in packet_verdicts
                ):
                    covered.add(path)
                    packet_covered.add(path)
        # Inline evidence is gated on THIS shard's own records: a shard without
        # a records file contributes zero inline evidence (fail-open).
        if inline_covered is not None:
            for f in receipt.get("inline_files", []) or []:
                if f in diff_set and f in inline_covered:
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
            if f in diff_set and f in frontier_evidence:
                covered.add(f)
                covered_by_type["dependency_frontier_read"].add(f)
    counts = {key: len(files) for key, files in covered_by_type.items()}
    return covered, counts


def _verdict(path: str, lines_read: int | None, verdict: str, n_findings: int) -> dict[str, Any]:
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
    source_packet_paths: set[str] | None = None,
    read_coverage_unverifiable: bool = False,
) -> list[dict[str, Any]]:
    """Retain findings and credit successful reads or completed host source packets.

    Opaque reads leave missing tool evidence unknown, rather than unread or clean.
    A finite receipt still requires a completed declared verdict.
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
        n_findings = 1 if path in finding_files else 0
        if read_coverage_unverifiable and source_packet_paths is None and not any(
            _path_component_matches(read, path) for read in completed_read_paths
        ):
            lines_read = None
        finite_covered = (
            source_packet_paths is not None
            and path in source_packet_paths
            and declared.get("verdict") in {"clean", "has_findings"}
        )
        if source_packet_paths is not None and not finite_covered:
            out.append(_verdict(path, lines_read, "not_reviewed", n_findings))
        elif n_findings:
            # A finding beats a read and beats a declared clean.
            entry = _verdict(path, lines_read, "has_findings", n_findings)
            if read_coverage_unverifiable and source_packet_paths is None and not any(
                _path_component_matches(read, path) for read in completed_read_paths
            ):
                entry["source_read_status"] = "unverifiable"
            out.append(entry)
        elif finite_covered or any(_path_component_matches(r, path) for r in completed_read_paths):
            # A completed read that matches the file yields clean.
            out.append(_verdict(path, lines_read, "clean", 0))
        elif read_coverage_unverifiable:
            out.append(_verdict(path, None, "unknown", 0))
        else:
            out.append(_verdict(path, lines_read, "not_reviewed", 0))
    return out


def load_source_packet_paths(
    deep_dir: Path, stack_name: str, assigned_files: list[str]
) -> set[str] | None:
    """Load finite completion evidence; ``None`` means no finite receipt exists."""
    try:
        receipts = json.loads(coverage_receipt_path(deep_dir).read_text())
        paths = receipts[stack_name]["source_packet_files"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if not isinstance(paths, list):
        return set()
    return {path for path in paths if isinstance(path, str) and path in assigned_files}


def compute_uncovered_files(
    daydream_dir: Path,
    session_id: str | None,
    *,
    receipts: dict[str, Any] | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Return sweep candidates and honest coverage accounting.

    Verified reads and independently completed receipts credit coverage. Opaque
    review phases leave their remaining assigned files unknown and unavailable
    for targeting; other phases' observed gaps can still be swept.
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
    unknown_files: set[str] = set()
    unknown_agents: list[str] = []
    for traj in trajectories["forked"]:
        if _agent_label(traj["_source_file"]).startswith("deep-"):
            review_reads.update(_completed_read_paths(traj))
            if _read_coverage_unverifiable(traj):
                label = _agent_label(traj["_source_file"])
                unknown_agents.append(label)
                assigned = set(diff_files)
                if receipts:
                    for stack, receipt in receipts.items():
                        slug = "deep-" + stack.replace("#", "-")
                        if label == slug or label.startswith(slug + "--"):
                            assigned = set(receipt.get("assigned_files", diff_files))
                            break
                unknown_files.update(assigned)
    if trajectories["main"]:
        main = trajectories["main"]
        review_reads.update(_completed_read_paths(main, phases={"deep", "alternatives"}))
        primary_forks = any(
            (label := _agent_label(traj["_source_file"])).startswith("deep-")
            and not label.startswith("deep-uncovered-")
            for traj in trajectories["forked"]
        )
        for source_phase in ("deep", "alternatives"):
            if _read_coverage_unverifiable(main, phases={source_phase}):
                unknown_agents.append(source_phase)
                if not primary_forks:
                    unknown_files.update(diff_files)

    covered = {df for df in diff_files if any(_path_component_matches(r, df) for r in review_reads)}
    source_covered = len(covered)
    if receipts:
        receipt_covered, receipt_counts = _receipt_covered_files(
            diff_files, receipts, daydream_dir / "deep"
        )
        covered |= receipt_covered
    unknown_files &= set(diff_files) - covered
    if receipts:
        # Restored records retain their unavailable evidence state without a new fork.
        for stack, receipt in receipts.items():
            try:
                records = json.loads(per_stack_records_path(daydream_dir / "deep", stack).read_text())
            except (OSError, ValueError):
                continue
            if isinstance(records, dict) and any(
                isinstance(entry, dict)
                and (entry.get("verdict") == "unknown"
                     or entry.get("source_read_status") == "unverifiable")
                and entry.get("path") in diff_files and entry["path"] not in covered
                for entry in records.get("verdicts", [])
            ):
                unknown_files.update(set(receipt.get("assigned_files", diff_files)) & set(diff_files) - covered)
    uncovered = sorted(set(diff_files) - covered - unknown_files)

    stats: dict[str, Any] = {
        "files_in_diff": len(diff_files),
        "files_read_by_reviewers": len(covered),
        "coverage_ratio": round(len(covered) / len(diff_files), 4) if diff_files else 1.0,
        "uncovered_files": None if unknown_files else uncovered,
        "coverage_status": "unverifiable" if unknown_files else "verified",
        "verified_files": sorted(covered),
        "verified_files_read_by_reviewers": len(covered),
        "unverifiable_files": sorted(unknown_files),
        "unverifiable_agents": sorted(set(unknown_agents)),
    }
    if unknown_files:
        stats["files_read_by_reviewers"] = None
        stats["coverage_ratio"] = None
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


def bounded_diff_block_for_file(path: Path, file: str) -> str:
    """Stream a bounded file excerpt for the non-Pi sweep compatibility path.

    Both each read and each retained block prefix are bounded, including for
    diffs containing multi-megabyte single lines. Use the shared path parser
    so renames and deletions resolve like ordinary in-memory diff blocks.
    """
    budget = INLINE_DIFF_BUDGET_BYTES
    prefix = bytearray()
    line_start = True
    with path.open("rb") as stream:
        while chunk := stream.readline(budget + 1):
            if line_start and chunk.startswith(b"diff --git "):
                if _diff_block_path(prefix.decode("utf-8", errors="ignore")) == file:
                    break
                prefix.clear()
            prefix.extend(chunk[:max(0, budget + 1 - len(prefix))])
            line_start = chunk.endswith(b"\n")
    text = prefix.decode("utf-8", errors="ignore")
    if _diff_block_path(text) != file:
        return "[Diff excerpt unavailable; missing hunks do not establish coverage.]"
    return truncate_utf8_to_budget(text, budget, "\n[diff excerpt truncated; missing hunks remain unreviewed]")


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
    ``block is None`` behavior. The sweepable set is capped at
    ``max_files`` in path order (the uncovered list arrives path-sorted from
    ``compute_uncovered_files``); the remainder is reported as
    skipped-for-capacity rather than silently dropped.

    Returns:
        ``(swept_files, skipped_small_hunk_files, skipped_capacity_files)``.
        The two skip lists name the omitted files in path order (mirroring
        ``swept_files``) so consumers can audit exactly which files the
        hunk-size floor and the capacity cap left out; the integer skip counts
        are ``len(...)`` of these lists.
    """
    swept: list[str] = []
    skipped_small: list[str] = []
    for file in uncovered_files:
        info = index.get(file)
        if info is None or info["added_total"] + info["removed_total"] < min_hunk_lines:
            skipped_small.append(file)
            continue
        swept.append(file)
    skipped_capacity = swept[max_files:]
    return swept[:max_files], skipped_small, skipped_capacity


def build_uncovered_sweep_prompt(
    *,
    strategy: str,
    file: str,
    diff_path: Path,
    intent_path: Path,
    cwd: Path,
    output_path: Path,
    exploration_dir: Path | None = None,
    inline_diff: str | None = None,
) -> str:
    """Build the second-pass sweep reviewer prompt for one uncovered file.

    The reviewer scopes itself to ``file``'s hunks only (no per-stack reviewer
    read this file), uses the TTT intent for authorial context, and returns structured
    findings that the host writes to ``output_path``. The sweep reviewer is held to the same standard
    as ordinary per-stack reviewers -- its findings are parsed into
    ``PER_STACK_RECORD_SCHEMA`` and merged as ordinary findings -- so the prompt
    is composed from the CANONICAL deep prompt primitives (imported from
    ``daydream.phases`` / ``daydream.deep.prompts``, read-only): the exploration
    pointer, the Confidence and Convention Rules (incl. QUAL-04 error
    semantics), the Dependency Impact instructions, and the full
    ``VERIFICATION_PROTOCOL_INSTRUCTION``. The gates are embedded inline and no
    skill-file read is required because the reviewer runs with cwd set to the
    reviewed repo, where a bare ``read`` of the protocol skill file resolves
    and silently drops the gates (same rationale as the canonical constant).

    Args:
        strategy: The profile-owned ``uncovered_review`` strategy content,
            rendered with the runtime ``file`` placeholder filled.
    """
    parts: list[str] = []
    pointer = _exploration_pointer(exploration_dir)
    if pointer:
        parts.append(pointer)
    parts.append(truncate_utf8_to_budget(strategy.format(file=file), 8192, "\n[strategy truncated]"))
    parts.append(f"TTT author intent is at {intent_path}. Read it before starting.")
    if inline_diff is not None:
        # Non-Pi compatibility: no host-private diff pointer is exposed to an
        # isolated transport, and no whole-diff admission is introduced.
        excerpt = truncate_utf8_to_budget(inline_diff, INLINE_DIFF_BUDGET_BYTES, "\n[diff excerpt truncated]")
        parts.append(f"Relevant diff excerpt for {file}:\n{excerpt}")
    else:
        parts.append(
            f"Changed file: {file}\nThe full PR diff is available at {diff_path}.\n"
            "Use your read-only tools to inspect the relevant section and the source checkout. "
            "Diff contents are not embedded in this prompt."
        )
    parts.append(
        "Read the source file FIRST; you may only comment on hunks you have "
        "read. A diff reference is not a substitute for reading the file."
    )
    parts.append(_confidence_and_convention_instructions())
    parts.append(_dependency_impact_instructions())
    parts.append(VERIFICATION_PROTOCOL_INSTRUCTION)
    parts.append(SEVERITY_RUBRIC)
    parts.append(
        f"Work in {cwd}. Return your findings in the requested structured output. "
        f"The host writes the review to {output_path}; do not write files yourself."
    )
    return "\n\n".join(parts)
