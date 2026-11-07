"""Quantitative trajectory, finding, and post-fix quality analysis for archived runs."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from daydream._tree_sitter_safety import (
    TreeSitterBadVersionError,
)
from daydream.artifact_visibility import ArtifactEvidenceProvenance, ArtifactSourceUnavailableError
from daydream.artifacts.ownership import CodeWorkspace
from daydream.deep.records import record_issues_or_empty
from daydream.deep.routing_record import read_routing_record
from daydream.eval.quality import _QUALITY_CALIBRATION, analyze_quality as analyze_quality
from daydream.hunk_index import load_hunk_index, parse_hunks, range_distance
from daydream.observability.runtime import current_session
from daydream.observability.spans import SpanScope
from daydream.trajectory import (
    RUN_DOCUMENT_NAME,
    RUNS_DIRNAME,
    DocumentSummary,
    RunSummary,
    StepFacts,
    compute_timing_summary,
    run_document_path,
    siblings_directory,
)

# Trajectory loading

def _latest_main_trajectory(daydream_dir: Path) -> Path | None:
    """Find the newest runs/<session>/trajectory.json by mtime."""
    candidates = list(daydream_dir.glob(f"{RUNS_DIRNAME}/*/{RUN_DOCUMENT_NAME}"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _run_dir_trajectory_paths(run_dir: Path) -> list[Path]:
    """``trajectory.json`` plus sorted ``trajectories/*.json`` directly under *run_dir*."""
    paths: list[Path] = []
    main_path = run_document_path(run_dir)
    if main_path.is_file():
        paths.append(main_path)
    siblings_dir = siblings_directory(run_dir)
    if siblings_dir.is_dir():
        paths.extend(f for f in sorted(siblings_dir.glob("*.json")) if f.is_file())
    return paths


def collect_trajectory_paths(run_dir: Path) -> list[Path]:
    """Collect trajectory files for *run_dir*, main trajectory first.

    Looks for ``trajectory.json`` plus ``trajectories/*.json`` directly under
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
    """Label current and legacy trajectory filenames, removing host collision suffixes."""
    if filename.startswith("trajectory"):
        return "main"
    # Legacy ``trajectory-<timestamp>.json`` / ``<hash>.deep-python.json`` shapes.
    parts = filename.rsplit(".", 2)
    if len(parts) >= 3:
        return parts[1]
    label = filename.replace(".json", "")
    return re.sub(r"--[0-9a-f]{64}$", "", label)


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
            "steps": metrics.get("total_steps") or document.steps.require("count").count,
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
        counts = Counter(document.steps.require("tools").tool_names)
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


def _iter_stack_records(deep_dir: Path) -> Iterator[tuple[str, list[Any]]]:
    for f in sorted(deep_dir.glob("stack-*-records.json")):
        if f.name == "stack-uncovered-records.json":
            continue
        yield f.stem.replace("stack-", "").replace("-records", ""), record_issues_or_empty(json.loads(f.read_text()))


_EMPTY_PER_LENS = {"wonder": 0, "per-stack": 0, "structure": 0}


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


def _shipped_counts(
    deep_dir: Path, all_findings: list[dict[str, Any]], merged_review: dict[str, Any]
) -> tuple[int, dict[str, int]]:
    """Return shipped total and confidence counts from the same evidence source.

    Canonical merged items win, including an empty list. Otherwise use the rendered
    review count (without confidence attribution), then raw per-stack records.
    """
    merged_items = _load_shipped_items(deep_dir)
    if merged_items is not None:
        # Every merged item is shipped: the renderer now emits wonder-lens
        # findings too (issue #741), so the shipped set equals the posted
        # review. A present-but-empty list means "shipped nothing".
        by_confidence = dict(
            Counter(i.get("confidence", "UNKNOWN") for i in merged_items)
        )
        return len(merged_items), by_confidence
    if merged_review.get("merged_finding_count"):
        # The review is the shipped rendering but carries no confidence values,
        # so no confidence attribution is possible without another source. Return
        # an empty rather than cross-mix the pre-merge per-stack confidences.
        return merged_review["merged_finding_count"], {}
    return len(all_findings), dict(
        Counter(f.get("confidence", "UNKNOWN") for f in all_findings)
    )


def analyze_findings(daydream_dir: Path) -> dict[str, Any]:
    """Report shipped totals/confidence, raw lens attribution, and merge/dedup evidence.

    _shipped_counts owns source precedence. Per-lens counts describe pre-merge inputs
    and need not sum to the shipped total.
    """
    deep_dir = daydream_dir / "deep"
    if not deep_dir.is_dir():
        return {
            "total": 0,
            "by_confidence": {},
            "findings": [],
            "stacks": [],
            "dedup": {},
            "merged_review": {},
            "per_lens": dict(_EMPTY_PER_LENS),
        }

    all_findings: list[dict[str, Any]] = []
    stacks: list[dict[str, Any]] = []
    per_lens = dict(_EMPTY_PER_LENS)

    # ``wonder`` reads the bare-list ``alternatives.json`` (the canonical wonder
    # artifact); the ``stack-*-records.json`` walk below buckets each stack into
    # per-stack / structure by name. These lens counts are raw,
    # *pre-merge* attribution -- they are not derived from the shipped
    # ``merged-items.json`` set, so they need not sum to, or relate to, the
    # shipped ``total`` reported by ``_shipped_counts``. Reconciling a lens to
    # the shipped set would hide how many items survived merge/dedup, so report
    # readers should treat the lens as raw attribution rather than a partition.
    alts_path = deep_dir / "alternatives.json"
    if alts_path.exists():
        try:
            alternatives = json.loads(alts_path.read_text())
        except json.JSONDecodeError:
            # A present-but-malformed alternatives.json must not take down
            # analyze_findings / analyze_session; leave wonder attribution at 0.
            alternatives = None
        if isinstance(alternatives, list):
            per_lens["wonder"] = len(alternatives)

    for stack_name, records in _iter_stack_records(deep_dir):
        stacks.append({"name": stack_name, "finding_count": len(records)})
        if stack_name == "structure":
            per_lens["structure"] += len(records)
        else:
            per_lens["per-stack"] += len(records)
        for r in records:
            r["_stack"] = stack_name
            all_findings.append(r)

    dedup_stats: dict[str, Any] = {}
    dedup_path = deep_dir / "dedup-candidates.json"
    if dedup_path.exists():
        dedup = json.loads(dedup_path.read_text())
        pairs = dedup.get("record_alt_pairs", [])
        dupes = dedup.get("record_duplicate_pairs", [])
        avg_sim = sum(p.get("similarity", 0) for p in pairs) / len(pairs) if pairs else 0
        # INPUT counters, not escapes: ``dedup-candidates.json`` holds the
        # pre-merge candidate pairs the pre-filter handed the merge agent.
        dedup_stats = {
            "record_alt_overlaps": len(pairs),
            "record_duplicate_candidates": len(dupes),
            "avg_overlap_similarity": round(avg_sim, 4),
        }

    merged_review: dict[str, Any] = {}
    review_path = deep_dir / "review-output.md"
    if review_path.exists():
        text = review_path.read_text()
        merged_review["merged_finding_count"] = len(
            re.findall(r"^\d+\.\s+\[", text, re.MULTILINE)
        )

    total, by_confidence = _shipped_counts(deep_dir, all_findings, merged_review)

    return {
        "total": total,
        "by_confidence": by_confidence,
        "findings": all_findings,
        "stacks": stacks,
        "dedup": dedup_stats,
        "merged_review": merged_review,
        "per_lens": per_lens,
    }


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

    Structural line-zero anchors are whole-file exemptions. Missing/noninteger/bool
    lines and non-dict items are unscorable. Without hunks, nothing is scored.
    Rates over empty populations are None, never evidence of perfect accuracy.
    Only per-item detail is capped; counters/rates cover all shipped items.
    """
    deep_dir = daydream_dir / "deep"
    items = _load_shipped_items(deep_dir)
    ranges, hunk_source = _hunk_ranges(daydream_dir)

    tiers = dict.fromkeys(_LOCATION_TIERS, 0)
    detail: list[dict[str, Any]] = []
    whole_file_anchors = 0
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

        lens = item.get("lens")
        cited = _cited_line(item)
        if lens == "structural" and cited == 0:
            whole_file_anchors += 1
            continue
        line = _int_or_none(cited)
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
                "lens": lens,
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
        "whole_file_anchors": whole_file_anchors,
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
    """Measure duplicate findings that survived merge, including structural/language pairs.

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
        """Attach shipped-item provenance and lens to a reported duplicate pair."""
        item_a = _item_at(pair.record_a_source)
        item_b = _item_at(pair.record_b_source)
        return {
            "a_id": pair.record_a_id,
            "a_source_uids": item_source_uids(item_a),
            "a_file": pair.record_a_file,
            "a_lens": str(item_a.get("lens", "")),
            "b_id": pair.record_b_id,
            "b_source_uids": item_source_uids(item_b),
            "b_file": pair.record_b_file,
            "b_lens": str(item_b.get("lens", "")),
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
        duration = document.steps.require("timing").duration_seconds
        if duration is not None:
            agent_timings.append({"agent": _agent_label(document.source_file), "duration_seconds": round(duration, 1)})

    timing = compute_timing_summary(summary)
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
        facts = document.steps.require("content")
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


def analyze_routing(daydream_dir: str | Path) -> dict[str, Any]:
    """Read recorded latency decisions; missing/malformed/old evidence yields an empty report."""
    record = read_routing_record(Path(daydream_dir) / "deep")
    profile_block = record.get("profile")
    selected = profile_block.get("selected") if isinstance(profile_block, dict) else None
    decisions: dict[str, Any] = {}
    for name in ("wonder", "arbiter"):
        slice_ = record.get(name)
        if isinstance(slice_, dict):
            decisions[name] = slice_
    return {
        "profile": selected if isinstance(selected, str) else None,
        "decisions": decisions,
    }


# Top-level entry point

def analyze_session(
    daydream_dir: str | Path,
    *,
    summary: RunSummary,
    artifact_provenance: ArtifactEvidenceProvenance | None = None,
    code_workspace: Path | CodeWorkspace | None = None,
) -> dict[str, Any]:
    """Combine trajectory, findings, timing, routing, training, and quality analysis.

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

    # Extract PR metadata from trajectory extra (set by TrajectoryRecorder)
    traj_extra = main.get("extra") or {}
    pr_number = traj_extra.get("pr_number")
    pr_repo = traj_extra.get("pr_repo")

    costs = analyze_costs(summary)
    tools = analyze_tools(summary)
    findings_data = analyze_findings(daydream_dir)
    location = analyze_location(daydream_dir)
    routing = analyze_routing(daydream_dir)
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
            "stacks": findings_data["stacks"],
            "dedup": findings_data["dedup"],
            "shipped_duplication": shipped_duplication,
            "merged_review": findings_data.get("merged_review", {}),
            "per_lens": findings_data.get("per_lens", {}),
        },
        "location": location,
        "latency_profile": routing["profile"],
        "routing": routing,
        "training_signals": training,
        "quality": quality,
        "derived": {
            "cost_per_finding_usd": cost_per_finding,
        },
    }

    if pr_number is not None or pr_repo is not None:
        result["pr"] = {"pr_number": pr_number, "pr_repo": pr_repo}

    return result
