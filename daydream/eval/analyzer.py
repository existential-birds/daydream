"""Quantitative trajectory, finding, and post-fix quality analysis for archived runs."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from daydream._tree_sitter_safety import (
    TreeSitterBadVersionError,
)
from daydream.artifact_visibility import ArtifactEvidenceProvenance, ArtifactSourceUnavailableError
from daydream.artifacts.ownership import CodeWorkspace
from daydream.eval.quality import _QUALITY_CALIBRATION, analyze_quality as analyze_quality
from daydream.hunk_index import load_hunk_index, parse_hunks, range_distance
from daydream.observability.runtime import current_session
from daydream.observability.spans import SpanScope
from daydream.trajectory import (
    DOCUMENT_SUFFIX,
    RUN_DOCUMENT_NAME,
    RUNS_DIRNAME,
    DocumentSummary,
    RunSummary,
    StepFacts,
    run_document_path,
    siblings_directory,
)

# Trajectory loading

def _latest_main_trajectory(daydream_dir: Path) -> Path | None:
    """Find the newest runs/<session>/ root evidence document by mtime."""
    candidates = list(daydream_dir.glob(f"{RUNS_DIRNAME}/*/{RUN_DOCUMENT_NAME}"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _run_dir_trajectory_paths(run_dir: Path) -> list[Path]:
    """The root evidence document plus its sorted sibling documents directly under *run_dir*."""
    paths: list[Path] = []
    main_path = run_document_path(run_dir)
    if main_path.is_file():
        paths.append(main_path)
    siblings_dir = siblings_directory(run_dir)
    if siblings_dir.is_dir():
        paths.extend(f for f in sorted(siblings_dir.glob(f"*{DOCUMENT_SUFFIX}")) if f.is_file())
    return paths


def collect_trajectory_paths(run_dir: Path) -> list[Path]:
    """Collect evidence documents for *run_dir*, the root document first.

    Looks for the root document plus its sibling documents directly under
    *run_dir*; when neither exists, falls back to the most recently modified
    ``runs/<session_id>/`` run beneath it.
    """
    paths = _run_dir_trajectory_paths(run_dir)
    if not paths:
        latest = _latest_main_trajectory(run_dir)
        if latest:
            paths = _run_dir_trajectory_paths(latest.parent)
    return paths


def _agent_label(filename: str) -> str:
    """Label an evidence document by its descriptor, removing the fork identity digest."""
    if filename == RUN_DOCUMENT_NAME:
        return "main"
    return re.sub(r"--[0-9a-f]{64}$", "", filename.removesuffix(DOCUMENT_SUFFIX))


_WRITE_TOOL_ALIASES = frozenset(
    {"write", "edit", "multiedit", "notebookedit", "patch", "apply_patch"}
)


# Analysis functions

def _root_first(summary: RunSummary) -> tuple[DocumentSummary, ...]:
    return summary.root, *summary.forked


_BILLING_FIELDS = {
    "cost_usd": ("total_cost_usd", 0.0),
    "prompt_tokens": ("total_prompt_tokens", 0),
    "completion_tokens": ("total_completion_tokens", 0),
    "cached_tokens": ("total_cached_tokens", 0),
}


def analyze_costs(summary: RunSummary) -> dict[str, Any]:
    """Report invocation-local agent rows and authoritative whole-run totals.

    Marked roots include fork billing already; legacy documents aggregate rows.
    Cache tokens are a subset of prompt tokens and are never added again.
    """
    agents: list[dict[str, Any]] = []
    forks_by_source = {fork.source_file: fork.metadata for fork in summary.forked}
    for document in _root_first(summary):
        traj = document.metadata
        metrics = traj.get("final_metrics") or {}
        child_sources = {
            Path(ref).name
            for child in (traj.get("extra") or {}).get("subtrajectories") or []
            if isinstance(child, dict)
            for ref in [child.get("sibling_trajectory_ref")]
            if isinstance(ref, str)
        }
        child_metrics = [
            forks_by_source[source].get("final_metrics") or {}
            for source in child_sources if source in forks_by_source
        ]
        agents.append({
            "agent": _agent_label(document.source_file),
            **{
                name: (metrics.get(key) or default) - sum(child.get(key) or default for child in child_metrics)
                for name, (key, default) in _BILLING_FIELDS.items()
            },
            "steps": metrics.get("total_steps") or document.steps.require().count,
            "model": traj.get("agent", {}).get("model_name", "unknown"),
        })

    root_metrics = summary.root.metadata.get("final_metrics") or {}
    root_includes_forks = (root_metrics.get("extra") or {}).get("daydream_metric_scope") == "whole_run_including_forks"
    totals = {
        name: root_metrics.get(key) or default if root_includes_forks else sum(row[name] for row in agents)
        for name, (key, default) in _BILLING_FIELDS.items()
    }
    total_input = totals["prompt_tokens"]
    return {
        "total_cost_usd": totals["cost_usd"],
        "total_input_tokens": total_input,
        "total_prompt_tokens_raw": total_input,
        "total_completion_tokens": totals["completion_tokens"],
        "total_cached_tokens": totals["cached_tokens"],
        "cache_hit_rate": round(totals["cached_tokens"] / total_input, 4) if total_input > 0 else 0.0,
        "by_agent": sorted(agents, key=lambda a: a["cost_usd"], reverse=True),
    }


def analyze_tools(summary: RunSummary) -> dict[str, Any]:
    """Tool call counts and per-agent breakdown."""
    total_counts: Counter[str] = Counter()
    by_agent: dict[str, dict[str, Any]] = {}

    for document in _root_first(summary):
        counts = Counter(document.steps.require().tool_names)
        by_agent[_agent_label(document.source_file)] = dict(counts)
        total_counts.update(counts)

    total = sum(total_counts.values())
    write_count = sum(
        count
        for function_name, count in total_counts.items()
        if function_name.casefold() in _WRITE_TOOL_ALIASES
    )

    return {
        "total_calls": total,
        "by_type": dict(total_counts.most_common()),
        "by_agent": by_agent,
        "write_ratio": round(write_count / total, 4) if total > 0 else 0,
    }


def _load_shipped_items(deep_dir: Path) -> list[Any] | None:
    """Load the canonical shipped set: None means absent, [] means shipped nothing.

    Corrupt JSON or a missing/non-list items field raises a data-integrity error.
    Return items unchanged, including non-dict entries, so counts remain faithful.
    """
    merged_items_file = deep_dir / "merged-items.json"
    if not merged_items_file.exists():
        return None
    merged = json.loads(merged_items_file.read_text())
    # Shape-check before use so a well-formed-but-wrong-shape file (top-level
    # non-object, or ``items`` not a list) is treated as the documented
    # data-integrity error instead of silently yielding a bogus count. The
    # writer always emits the ``items`` key, so a missing ``items`` is a
    # wrong shape too -- only ``items": []`` is "shipped nothing".
    if not isinstance(merged, dict) or not isinstance(merged.get("items"), list):
        raise ValueError(
            "merged-items.json must be an object whose ``items`` is a list; "
            f"got top-level type {type(merged).__name__}"
        )
    items: list[Any] = merged["items"]
    return items


def analyze_findings(daydream_dir: Path) -> dict[str, Any]:
    """Report shipped totals and confidence from one evidence source.

    Canonical merged items win, including an empty list ("shipped nothing"). Otherwise use
    the rendered review's count, which carries no confidence attribution.
    """
    deep_dir = daydream_dir / "deep"
    merged_review: dict[str, Any] = {}
    review_path = deep_dir / "review-output.md"
    if review_path.exists():
        merged_review["merged_finding_count"] = len(re.findall(r"^\d+\.\s+\[", review_path.read_text(), re.MULTILINE))
    items = _load_shipped_items(deep_dir)
    if items is None:
        return {"total": merged_review.get("merged_finding_count", 0), "by_confidence": {},
                "merged_review": merged_review}
    by_confidence = dict(Counter(i.get("confidence", "UNKNOWN") for i in items))
    return {"total": len(items), "by_confidence": by_confidence, "merged_review": merged_review}


# Location accuracy + shipped duplication (issue #1106) ---------------------

_LOCATION_TIERS = ("in_hunk", "within_tolerance", "beyond_tolerance", "file_absent")
"""The four tiers a scored ``file:line`` citation can land in."""

_LOCATION_ITEM_CAP = 200
"""Max per-item rows carried in ``analyze_location``'s ``items`` list.

``evaluation.json`` is archived per run and read whole; a pathological run with
thousands of shipped items must not turn the eval artifact into a dump. The
counters and rates are always complete -- only the per-item detail is capped.
"""

_DUPLICATION_PAIR_CAP = 20
"""Max pair rows carried in ``analyze_shipped_duplication``'s ``pairs`` list."""

_DUPLICATION_INPUT_CAP = 200
"""Max shipped items compared in ``analyze_shipped_duplication``.

Unlike ``_LOCATION_ITEM_CAP``, which only bounds an output list while its
counters stay complete, the pairwise similarity scan itself is O(n^2): a
pathological run with thousands of shipped items must not turn a single eval
pass into an unbounded time/memory sink. ``shipped_items`` still reports the
true total; only the comparison is capped.
"""


def _hunk_ranges(daydream_dir: Path) -> tuple[dict[str, list[tuple[int, int]]], str]:
    """Read inclusive head-side ranges from hunk-index.json, falling back to diff.patch.

    Return ranges and source; malformed/missing inputs degrade to ({}, "none") so
    optional location analysis cannot prevent archive finalization.
    """
    index: dict[str, Any] = load_hunk_index(daydream_dir)
    source = "hunk-index.json"
    if not index:
        diff_path = daydream_dir / "diff.patch"
        try:
            index = parse_hunks(diff_path.read_text()) if diff_path.is_file() else {}
        except (OSError, ValueError):
            index = {}
        source = "diff.patch" if index else "none"

    ranges: dict[str, list[tuple[int, int]]] = {}
    for path, info in index.items():
        if not isinstance(info, dict):
            continue
        hunks = info.get("hunks")
        file_ranges: list[tuple[int, int]] = []
        for hunk in hunks if isinstance(hunks, list) else []:
            if not isinstance(hunk, dict):
                continue
            start = _int_or_none(hunk.get("new_start"))
            end = _int_or_none(hunk.get("new_end"))
            if start is None or end is None:
                continue
            file_ranges.append((start, end))
        ranges[str(path)] = file_ranges
    if not ranges:
        return {}, "none"
    return ranges, source


def _int_or_none(value: Any) -> int | None:
    """Accept integer citations, excluding bool so true cannot mean line one."""
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _cited_line(record: dict[str, Any]) -> Any:
    """Read the original citation, before the location validator snapped or demoted it."""
    if "location_cited_line" in record:
        return record["location_cited_line"]
    return record.get("line")


def _location_tier(ranges: list[tuple[int, int]] | None, line: int) -> tuple[str, int | None]:
    """Classify distance from changed lines using the shared posting tolerance.

    Missing/empty ranges mean file_absent; zero distance is in_hunk, then
    within_tolerance or beyond_tolerance.
    """
    from daydream.pr_review import HUNK_TOLERANCE

    if not ranges:
        return "file_absent", None
    distance = min(range_distance(line, start, end) for start, end in ranges)
    if distance == 0:
        return "in_hunk", 0
    if distance <= HUNK_TOLERANCE:
        return "within_tolerance", distance
    return "beyond_tolerance", distance


def analyze_location(daydream_dir: Path) -> dict[str, Any]:
    """Score original citations in the shipped review set against the run's hunks.

    Missing/noninteger/bool lines and non-dict items are unscorable. Without hunks, nothing is scored.
    Rates over empty populations are None, never evidence of perfect accuracy.
    Only per-item detail is capped; counters/rates cover all shipped items.
    """
    deep_dir = daydream_dir / "deep"
    items = _load_shipped_items(deep_dir)
    ranges, hunk_source = _hunk_ranges(daydream_dir)

    tiers = dict.fromkeys(_LOCATION_TIERS, 0)
    detail: list[dict[str, Any]] = []
    unscorable_items = 0
    distrusted_items = 0
    relocated_items = 0
    scored_items = 0

    for item in items or []:
        if not isinstance(item, dict):
            unscorable_items += 1
            continue
        if item.get("location_distrust") is True:
            distrusted_items += 1
        elif "location_cited_line" in item:
            # ``location_cited_line`` is stamped on both the snap (relocate)
            # and the demote branch of ``location_validator.validate_records``
            # (its own docstring: the key means "relocated OR distrusted").
            # Demoted items are already counted above, so only count this as
            # a relocation when it was NOT a demotion (issue #1106 R2).
            relocated_items += 1

        line = _int_or_none(_cited_line(item))
        if line is None:
            unscorable_items += 1
            continue
        if hunk_source == "none":
            continue

        cited_file = str(item.get("file", ""))
        tier, distance = _location_tier(ranges.get(cited_file), line)
        tiers[tier] += 1
        scored_items += 1
        if len(detail) < _LOCATION_ITEM_CAP:
            detail.append({
                "id": item.get("id"),
                "file": cited_file,
                "line": item.get("line"),
                "cited_line": item.get("location_cited_line"),
                "tier": tier,
                "distance": distance,
                "location_distrust": item.get("location_distrust") is True,
            })

    tier_rates = (
        {name: round(count / scored_items, 4) for name, count in tiers.items()}
        if scored_items > 0
        else {}
    )
    return {
        "hunk_source": hunk_source,
        "shipped_items": len(items or []),
        "scored_items": scored_items,
        "unscorable_items": unscorable_items,
        "tiers": tiers,
        "tier_rates": tier_rates,
        "in_hunk_rate": (
            round(tiers["in_hunk"] / scored_items, 4) if scored_items > 0 else None
        ),
        "distrusted_items": distrusted_items,
        "relocated_items": relocated_items,
        "items": detail,
    }


def analyze_shipped_duplication(daydream_dir: Path) -> dict[str, Any]:
    """Measure duplicate findings that survived merge.

    Compare descriptions across files and report the full similarity distribution,
    not only threshold crossings. Provenance lists retain every source uid; empty
    attribution remains empty. Missing shipped sets have an unknown duplicate count;
    empty sets have zero. Similarity is None without comparable pairs.
    Scan at most _DUPLICATION_INPUT_CAP items and report truncation; retain the true
    shipped total and at most _DUPLICATION_PAIR_CAP highest-similarity detail rows.
    """
    # Function-local import: ``daydream.deep.orchestrator`` imports this module
    # at module level, so a module-level ``daydream.deep.dedup`` import here
    # would close an import cycle through ``daydream/deep/__init__.py``. Same
    # no-cycle pattern as ``daydream/phases.py:1091``. ``daydream.deep.records``
    # is imported here for the same reason and not at module level: it has no
    # daydream imports of its own, but importing it still executes the
    # ``daydream.deep`` package ``__init__``, which pulls in the orchestrator
    # and closes the very same cycle.
    from daydream.deep.dedup import _SIM_THRESHOLD, build_record_dedup_candidates
    from daydream.deep.records import item_source_uids

    deep_dir = daydream_dir / "deep"
    raw_items = _load_shipped_items(deep_dir)
    shipped_set_produced = raw_items is not None
    items = raw_items or []
    dict_items = [item for item in items if isinstance(item, dict)]
    records = dict_items[:_DUPLICATION_INPUT_CAP]
    input_truncated = len(dict_items) > _DUPLICATION_INPUT_CAP
    # ``sources`` round-trips the item index; ``build_record_dedup_candidates``
    # never inspects it, so pair ordering is unchanged.
    sources = [str(index) for index in range(len(records))]
    pairs = build_record_dedup_candidates(records, sources, threshold=0.0)

    def _same_file(pair: Any) -> bool:
        return bool(pair.record_a_file) and pair.record_a_file == pair.record_b_file

    def _item_at(source_index: str) -> dict[str, Any]:
        """Resolve the source index minted for this scan back to its shipped item."""
        return records[int(source_index)]

    def _pair_row(pair: Any) -> dict[str, Any]:
        """Attach shipped-item provenance to a reported duplicate pair."""
        item_a = _item_at(pair.record_a_source)
        item_b = _item_at(pair.record_b_source)
        return {
            "a_id": pair.record_a_id,
            "a_source_uids": item_source_uids(item_a),
            "a_file": pair.record_a_file,
            "b_id": pair.record_b_id,
            "b_source_uids": item_source_uids(item_b),
            "b_file": pair.record_b_file,
            "similarity": round(pair.similarity, 4),
            "same_file": _same_file(pair),
        }

    similarities = [pair.similarity for pair in pairs]
    near = [pair for pair in pairs if pair.similarity >= _SIM_THRESHOLD]
    same_file = [pair for pair in pairs if _same_file(pair)]
    same_file_near = [pair for pair in same_file if pair.similarity >= _SIM_THRESHOLD]
    top = sorted(
        pairs, key=lambda pair: (-pair.similarity, pair.record_a_id, pair.record_b_id)
    )[:_DUPLICATION_PAIR_CAP]

    return {
        "shipped_items": len(items),
        "input_truncated": input_truncated,
        "comparable_pairs": len(pairs),
        "near_duplicate_pairs": len(near) if shipped_set_produced else None,
        "same_file_pairs": len(same_file),
        "same_file_near_duplicate_pairs": len(same_file_near),
        "max_similarity": round(max(similarities), 4) if similarities else None,
        "mean_similarity": (
            round(sum(similarities) / len(similarities), 4) if similarities else None
        ),
        "pairs": [_pair_row(pair) for pair in top],
    }


def analyze_timing(summary: RunSummary) -> dict[str, Any]:
    """Project the shared lifecycle-first timing reducer into evaluation JSON."""
    agent_timings: list[dict[str, Any]] = []

    for document in _root_first(summary):
        duration = document.steps.require().duration_seconds
        if duration is not None:
            agent_timings.append({"agent": _agent_label(document.source_file), "duration_seconds": round(duration, 1)})

    timing = summary.timing
    return {
        **(timing.to_dict() if timing is not None else {"total_wall_clock_seconds": None}),
        "by_agent": sorted(agent_timings, key=lambda item: item["duration_seconds"], reverse=True),
    }


_TOOL_OUTCOME_FLAG_ORDER = (
    "failed_tool_result",
    "incomplete_tool_call",
    "unmatched_tool_result",
    "incomplete_telemetry",
    "parser_coverage_gap",
)
_DIAGNOSTIC_TRAINING_FLAGS = {
    "codex_transport_coverage": "incomplete_telemetry",
    "codex_parser_coverage": "parser_coverage_gap",
}


def _tool_outcome_flags(facts: StepFacts) -> list[str]:
    """Classify step-local tool pairing and backend diagnostics as training outcome flags."""
    found = {flag for flag, present in (
        ("failed_tool_result", facts.error_result),
        ("incomplete_tool_call", facts.interrupted_result or facts.unpaired_call),
        ("unmatched_tool_result", facts.orphan_result),
    ) if present}
    found.update(flag for code, flag in _DIAGNOSTIC_TRAINING_FLAGS.items() if code in facts.diagnostic_codes)
    return [flag for flag in _TOOL_OUTCOME_FLAG_ORDER if flag in found]


def analyze_training_signals(summary: RunSummary) -> dict[str, Any]:
    """Assess forked training trajectories for content and evidence quality."""
    signals: list[dict[str, Any]] = []

    for document in summary.forked:
        facts = document.steps.require()
        # Reasoning token fraction (approximation from char length)
        total_output_chars = facts.reasoning_chars + facts.agent_message_chars
        reasoning_fraction = (
            round(facts.reasoning_chars / total_output_chars, 4)
            if total_output_chars > 0
            else 0
        )
        noise_flags = ["empty_tool_result"] if facts.blank_result else []
        noise_flags.extend(_tool_outcome_flags(facts))

        signals.append({
            "trajectory": _agent_label(document.source_file),
            "source_file": document.source_file,
            "steps": facts.count,
            "has_reasoning": facts.has_reasoning,
            "tool_calls": facts.tool_calls,
            "reasoning_fraction": reasoning_fraction,
            "noise_flags": noise_flags,
            "training_quality": "clean" if not noise_flags else "review",
        })

    clean = sum(1 for s in signals if s["training_quality"] == "clean")

    return {
        "total_trajectories": len(signals),
        "clean_for_training": clean,
        "needs_review": len(signals) - clean,
        "trajectories": signals,
    }


# Top-level entry point

def analyze_session(
    daydream_dir: str | Path,
    *,
    summary: RunSummary,
    artifact_provenance: ArtifactEvidenceProvenance | None = None,
    code_workspace: Path | CodeWorkspace | None = None,
) -> dict[str, Any]:
    """Combine trajectory, findings, timing, training, and quality analysis.

    The frozen run summary supplies checked trajectory metadata, step facts and lifecycle timing. Artifact
    provenance supplies the public display path; only quality reads the live code_workspace. Unavailable code
    or unsafe tree-sitter versions preserve the remaining frozen-only evaluation.
    """
    daydream_dir = Path(daydream_dir)
    display_daydream_dir = (
        daydream_dir if artifact_provenance is None else artifact_provenance.public_daydream_dir
    )
    main = summary.root.metadata
    session_id = main.get("session_id", "unknown")
    agent_info = main.get("agent", {})

    # Extract PR metadata from the root document's run identity
    traj_extra = main.get("extra") or {}
    pr_number = traj_extra.get("pr_number")
    pr_repo = traj_extra.get("pr_repo")

    costs = analyze_costs(summary)
    tools = analyze_tools(summary)
    findings_data = analyze_findings(daydream_dir)
    location = analyze_location(daydream_dir)
    shipped_duplication = analyze_shipped_duplication(daydream_dir)
    timing = analyze_timing(summary)
    training = analyze_training_signals(summary)
    try:
        # Omit optional labels to retain the enclosing host scope's explicit attribution.
        with SpanScope(
            current_session(), "daydream.step.quality-analysis", "step",
            {"daydream.step": "quality-analysis", "daydream.phase": "quality-analysis"},
        ) as observed:
            quality = analyze_quality(daydream_dir, code_workspace=code_workspace)
            observed.finish()
    except (ArtifactSourceUnavailableError, TreeSitterBadVersionError) as exc:
        # Frozen evidence remains valid when live quality cannot be measured.
        # Other Git, analyzer, and workspace safety failures still propagate.
        quality = {
            "erosion": None,
            "verbosity": None,
            "per_file": {},
            "calibration": dict(_QUALITY_CALIBRATION),
            "scoped_files": 0,
            "error": str(exc),
            "unavailable": True,
        }

    finding_count = findings_data["total"]
    cost_per_finding = (
        round(costs["total_cost_usd"] / finding_count, 4)
        if finding_count > 0
        else None
    )

    result: dict[str, Any] = {
        "session_id": session_id,
        "agent": agent_info,
        "daydream_dir": str(display_daydream_dir),
        "trajectory_count": len(summary.documents),
        "cost": costs,
        "timing": timing,
        "tools": tools,
        "findings": {
            "total": finding_count,
            "by_confidence": findings_data["by_confidence"],
            "shipped_duplication": shipped_duplication,
            "merged_review": findings_data["merged_review"],
        },
        "location": location,
        "training_signals": training,
        "quality": quality,
        "derived": {
            "cost_per_finding_usd": cost_per_finding,
        },
    }

    if pr_number is not None or pr_repo is not None:
        result["pr"] = {"pr_number": pr_number, "pr_repo": pr_repo}

    return result
