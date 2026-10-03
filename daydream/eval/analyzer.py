"""Quantitative trajectory, finding, and post-fix quality analysis for archived runs."""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from daydream._tree_sitter_safety import (
    TreeSitterBadVersionError,
)
from daydream.artifact_visibility import ArtifactEvidenceProvenance
from daydream.deep.records import record_issues_or_empty
from daydream.deep.routing_record import read_routing_record
from daydream.eval.quality import _QUALITY_CALIBRATION, analyze_quality as analyze_quality
from daydream.hunk_index import load_hunk_index, parse_hunks, range_distance
from daydream.timeutil import parse_iso_timestamp
from daydream.trajectory import (
    RUN_DOCUMENT_NAME,
    RUNS_DIRNAME,
    run_document_path,
    siblings_directory,
)
from daydream.trajectory.timing import compute_payload_timing_summary

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
    """Collect root/sibling documents, falling back to the newest nested run if absent."""
    paths = _run_dir_trajectory_paths(run_dir)
    if not paths:
        latest = _latest_main_trajectory(run_dir)
        if latest:
            paths = _run_dir_trajectory_paths(latest.parent)
    return paths


def load_trajectories(daydream_dir: Path, session_id: str | None = None) -> dict[str, Any]:
    """Load main and sibling trajectories for an exact session or unique prefix.

    Without a session, select the newest run. Ambiguous prefixes raise ValueError.
    """
    main = None
    forked: list[dict[str, Any]] = []
    runs_dir = daydream_dir / RUNS_DIRNAME

    run_dir: Path | None = None
    if session_id:
        # Exact match first, then prefix match on run directory names
        exact = runs_dir / session_id
        if exact.is_dir():
            run_dir = exact
        elif runs_dir.is_dir():
            matches = sorted(d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith(session_id))
            if len(matches) == 1:
                run_dir = matches[0]
            elif len(matches) > 1:
                raise ValueError(f"Session prefix '{session_id}' matches multiple runs")
    else:
        latest = _latest_main_trajectory(daydream_dir)
        if latest:
            # latest is runs/<session_id>/trajectory.json — parent is the run dir
            run_dir = latest.parent

    if run_dir:
        for path in _run_dir_trajectory_paths(run_dir):
            data = json.loads(path.read_text())
            data["_source_file"] = path.name
            if path.name == RUN_DOCUMENT_NAME:
                main = data
            else:
                forked.append(data)

    return {"main": main, "forked": forked}


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


_WRITE_TOOL_ALIASES = frozenset({"write", "edit", "multiedit", "notebookedit", "patch", "apply_patch"})


# Analysis functions


def _all_trajectories(trajectories: dict[str, Any]) -> list[dict[str, Any]]:
    all_trajs: list[dict[str, Any]] = []
    if trajectories["main"]:
        all_trajs.append(trajectories["main"])
    all_trajs.extend(trajectories["forked"])
    return all_trajs


_BILLING_FIELDS = {
    "cost_usd": ("total_cost_usd", 0.0),
    "prompt_tokens": ("total_prompt_tokens", 0),
    "completion_tokens": ("total_completion_tokens", 0),
    "cached_tokens": ("total_cached_tokens", 0),
}


def analyze_costs(trajectories: dict[str, Any]) -> dict[str, Any]:
    """Report local agent rows and whole-run billing; marked roots include forks, legacy rows
    aggregate. Cache tokens remain a prompt subset.
    """
    agents: list[dict[str, Any]] = []
    forks_by_source = {fork["_source_file"]: fork for fork in trajectories.get("forked") or []}
    for traj in _all_trajectories(trajectories):
        metrics = traj.get("final_metrics") or {}
        child_sources = {
            Path(ref).name
            for child in (traj.get("extra") or {}).get("subtrajectories") or []
            if isinstance(child, dict)
            for ref in [child.get("sibling_trajectory_ref")]
            if isinstance(ref, str)
        }
        child_metrics = [
            forks_by_source[source].get("final_metrics") or {} for source in child_sources if source in forks_by_source
        ]
        agents.append(
            {
                "agent": _agent_label(traj["_source_file"]),
                **{
                    name: (metrics.get(key) or default) - sum(child.get(key) or default for child in child_metrics)
                    for name, (key, default) in _BILLING_FIELDS.items()
                },
                "steps": metrics.get("total_steps") or len(traj.get("steps", [])),
                "model": traj.get("agent", {}).get("model_name", "unknown"),
            }
        )

    root = trajectories.get("main")
    root_metrics = (root or {}).get("final_metrics") or {}
    root_includes_forks = (root_metrics.get("extra") or {}).get("daydream_metric_scope") == "whole_run_including_forks"
    totals = {
        name: root_metrics.get(key) or default if root and root_includes_forks else sum(row[name] for row in agents)
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


def analyze_tools(trajectories: dict[str, Any]) -> dict[str, Any]:
    """Tool call counts and per-agent breakdown."""
    total_counts: Counter[str] = Counter()
    by_agent: dict[str, dict[str, Any]] = {}

    for traj in _all_trajectories(trajectories):
        label = _agent_label(traj["_source_file"])
        counts = Counter(
            call["function_name"] for step in traj.get("steps", []) for call in step.get("tool_calls") or []
        )
        by_agent[label] = dict(counts)
        total_counts.update(counts)

    total = sum(total_counts.values())
    write_count = sum(
        count for function_name, count in total_counts.items() if function_name.casefold() in _WRITE_TOOL_ALIASES
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
    """Load shipped items unchanged: absent is None, empty is []; corrupt/missing/non-list items raise."""
    merged_items_file = deep_dir / "merged-items.json"
    if not merged_items_file.exists():
        return None
    merged = json.loads(merged_items_file.read_text())
    # Missing/non-list items are corrupt evidence; only an explicit [] means shipped nothing.
    if not isinstance(merged, dict) or not isinstance(merged.get("items"), list):
        raise ValueError(
            f"merged-items.json must be an object whose ``items`` is a list; got top-level type {type(merged).__name__}"
        )
    items: list[Any] = merged["items"]
    return items


def _shipped_counts(
    deep_dir: Path, all_findings: list[dict[str, Any]], merged_review: dict[str, Any]
) -> tuple[int, dict[str, int]]:
    """Prefer canonical shipped items (including []), then rendered review counts, then raw stack records."""
    merged_items = _load_shipped_items(deep_dir)
    if merged_items is not None:
        # Every merged item is shipped, including wonder; an explicit [] means shipped nothing.
        by_confidence = dict(Counter(i.get("confidence", "UNKNOWN") for i in merged_items))
        return len(merged_items), by_confidence
    if merged_review.get("merged_finding_count"):
        # Rendered reviews have no confidence values; never borrow attribution from pre-merge records.
        return merged_review["merged_finding_count"], {}
    return len(all_findings), dict(Counter(f.get("confidence", "UNKNOWN") for f in all_findings))


def analyze_findings(daydream_dir: Path) -> dict[str, Any]:
    """Report shipped counts, raw pre-merge lens attribution, and merge/dedup evidence."""
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

    # Raw wonder/stack/structure counts precede merge and need not partition shipped totals.
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
        merged_review["merged_finding_count"] = len(re.findall(r"^\d+\.\s+\[", text, re.MULTILINE))

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
    """Read head-side hunk-index ranges, falling back to diff.patch; malformed inputs yield ({}, "none")."""
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
    """Classify shared posting distance: absent file, in hunk, within tolerance, or beyond tolerance."""
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
    """Score original shipped citations against hunks; structural line-zero anchors are exempt.

    Malformed citations are unscorable; missing hunks produce no scores. Empty rates
    are None. Only detail rows are capped; counters cover all items.
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
            # Cited-line metadata marks relocation or demotion; demotions must not count twice.
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
            detail.append(
                {
                    "id": item.get("id"),
                    "file": cited_file,
                    "line": item.get("line"),
                    "cited_line": item.get("location_cited_line"),
                    "lens": lens,
                    "tier": tier,
                    "distance": distance,
                    "location_distrust": item.get("location_distrust") is True,
                }
            )

    tier_rates = {name: round(count / scored_items, 4) for name, count in tiers.items()} if scored_items > 0 else {}
    return {
        "hunk_source": hunk_source,
        "shipped_items": len(items or []),
        "scored_items": scored_items,
        "whole_file_anchors": whole_file_anchors,
        "unscorable_items": unscorable_items,
        "tiers": tiers,
        "tier_rates": tier_rates,
        "in_hunk_rate": (round(tiers["in_hunk"] / scored_items, 4) if scored_items > 0 else None),
        "distrusted_items": distrusted_items,
        "relocated_items": relocated_items,
        "items": detail,
    }


def analyze_shipped_duplication(daydream_dir: Path) -> dict[str, Any]:
    """Measure shipped description similarity, including cross-file and cross-lens pairs.

    Preserve all source UIDs. Missing shipped sets have unknown duplicate counts; empty
    sets have zero. No comparable pairs yield None similarity. Bound scan/detail counts
    while retaining true shipped totals and explicit truncation.
    """
    # Local imports avoid a deep package/orchestrator/analyzer import cycle.
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
    top = sorted(pairs, key=lambda pair: (-pair.similarity, pair.record_a_id, pair.record_b_id))[:_DUPLICATION_PAIR_CAP]

    return {
        "shipped_items": len(items),
        "input_truncated": input_truncated,
        "comparable_pairs": len(pairs),
        "near_duplicate_pairs": len(near) if shipped_set_produced else None,
        "same_file_pairs": len(same_file),
        "same_file_near_duplicate_pairs": len(same_file_near),
        "max_similarity": round(max(similarities), 4) if similarities else None,
        "mean_similarity": (round(sum(similarities) / len(similarities), 4) if similarities else None),
        "pairs": [_pair_row(pair) for pair in top],
    }


def analyze_timing(trajectories: dict[str, Any]) -> dict[str, Any]:
    """Project the shared lifecycle-first timing reducer into evaluation JSON."""
    all_timestamps: list[datetime] = []
    agent_timings: list[dict[str, Any]] = []

    for traj in _all_trajectories(trajectories):
        label = _agent_label(traj["_source_file"])
        ts_list = [parse_iso_timestamp(s["timestamp"]) for s in traj.get("steps", []) if s.get("timestamp")]
        if len(ts_list) >= 2:
            duration = (ts_list[-1] - ts_list[0]).total_seconds()
            agent_timings.append({"agent": label, "duration_seconds": round(duration, 1)})
        all_timestamps.extend(ts_list)

    main = trajectories.get("main")
    if isinstance(main, dict):
        payloads = [
            main,
            *[item for item in trajectories.get("forked", []) if isinstance(item, dict)],
        ]
        payloads = [
            payload if isinstance(payload.get("trajectory_id"), str) else {
                **payload, "trajectory_id": str(payload.get("session_id", f"legacy-{index}")),
            }
            for index, payload in enumerate(payloads)
        ]
        raw_extra = main.get("extra")
        extra: dict[str, Any] = raw_extra if isinstance(raw_extra, dict) else {}
        partial = bool(extra.get("partial")) and isinstance(extra.get("snapshot_at"), str)
        cutoff = extra.get("snapshot_at") if partial else extra.get("run_ended_at")
        if not isinstance(cutoff, str):
            cutoff = ""
        summary = compute_payload_timing_summary(
            payloads,
            status="partial" if partial else "complete",
            cutoff_at=cutoff,
            root_trajectory_id=payloads[0]["trajectory_id"],
        )
        if summary is not None:
            return {
                **summary.to_dict(),
                "by_agent": sorted(
                    agent_timings,
                    key=lambda item: item["duration_seconds"],
                    reverse=True,
                ),
            }

    total_duration = (max(all_timestamps) - min(all_timestamps)).total_seconds() if len(all_timestamps) >= 2 else 0.0
    return {
        "total_wall_clock_seconds": round(total_duration, 1),
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


def _tool_outcome_flags(steps: list[dict[str, Any]]) -> list[str]:
    """Classify tool outcomes and diagnostics with step-local correlation."""
    found: set[str] = set()
    for step in steps:
        if not isinstance(step, dict):
            continue
        call_counts: Counter[str] = Counter()
        malformed_calls = 0
        raw_calls = step.get("tool_calls")
        if isinstance(raw_calls, list):
            for call in raw_calls:
                call_id = call.get("tool_call_id") if isinstance(call, dict) else None
                if isinstance(call_id, str) and call_id:
                    call_counts[call_id] += 1
                else:
                    malformed_calls += 1

        paired_counts: Counter[str] = Counter()
        observation = step.get("observation")
        raw_results = observation.get("results") if isinstance(observation, dict) else None
        if isinstance(raw_results, list):
            for result in raw_results:
                if not isinstance(result, dict):
                    continue
                source_call_id = result.get("source_call_id")
                raw_extra = result.get("extra")
                extra = raw_extra if isinstance(raw_extra, dict) else {}
                interrupted = extra.get("status") == "interrupted" or extra.get("cancelled") is True
                if isinstance(source_call_id, str) and source_call_id:
                    if source_call_id not in call_counts:
                        found.add("unmatched_tool_result")
                        continue
                    paired_counts[source_call_id] += 1
                    if interrupted:
                        found.add("incomplete_tool_call")
                    elif extra.get("is_error") is True:
                        found.add("failed_tool_result")
                elif interrupted and (call_counts or malformed_calls):
                    # Null-ID interruption markers describe unpaired calls, not orphan results.
                    found.add("incomplete_tool_call")

        if malformed_calls or any(paired_counts[call_id] < count for call_id, count in call_counts.items()):
            found.add("incomplete_tool_call")

        step_extra = step.get("extra")
        if not isinstance(step_extra, dict):
            continue
        unmatched = step_extra.get("unmatched_tool_results")
        if isinstance(unmatched, list) and unmatched:
            found.add("unmatched_tool_result")
        diagnostics = step_extra.get("backend_diagnostics")
        if isinstance(diagnostics, list):
            for diagnostic in diagnostics:
                code = diagnostic.get("code") if isinstance(diagnostic, dict) else None
                flag = _DIAGNOSTIC_TRAINING_FLAGS.get(code) if isinstance(code, str) else None
                if flag is not None:
                    found.add(flag)

    return [flag for flag in _TOOL_OUTCOME_FLAG_ORDER if flag in found]


def analyze_training_signals(
    trajectories: dict[str, Any],
) -> dict[str, Any]:
    """Assess forked training trajectories for content and evidence quality."""
    signals: list[dict[str, Any]] = []

    for traj in trajectories["forked"]:
        label = _agent_label(traj["_source_file"])
        steps = traj.get("steps", [])

        has_reasoning = any(s.get("reasoning_content") for s in steps)
        total_tool_calls = sum(len(s.get("tool_calls") or []) for s in steps)

        # Reasoning token fraction (approximation from char length)
        reasoning_chars = sum(len(s.get("reasoning_content") or "") for s in steps)
        message_chars = sum(
            len(s["message"]) for s in steps if s.get("source") == "agent" and isinstance(s.get("message"), str)
        )
        total_output_chars = reasoning_chars + message_chars
        reasoning_fraction = round(reasoning_chars / total_output_chars, 4) if total_output_chars > 0 else 0

        noise_flags: list[str] = []
        for s in steps:
            obs = s.get("observation")
            if obs:
                for r in obs.get("results", []):
                    content = r.get("content", "")
                    if isinstance(content, str) and content.strip() == "":
                        noise_flags.append("empty_tool_result")
                        break

        noise_flags.extend(_tool_outcome_flags(steps))

        # Keep the documented category order while ensuring repeated evidence
        # never repeats a training-review flag.
        noise_flags = list(dict.fromkeys(noise_flags))

        signals.append(
            {
                "trajectory": label,
                "source_file": traj["_source_file"],
                "steps": len(steps),
                "has_reasoning": has_reasoning,
                "tool_calls": total_tool_calls,
                "reasoning_fraction": reasoning_fraction,
                "noise_flags": noise_flags,
                "training_quality": "clean" if not noise_flags else "review",
            }
        )

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
    session_id: str | None = None,
    *,
    frozen_trajectories: dict[str, Any] | None = None,
    artifact_provenance: ArtifactEvidenceProvenance | None = None,
    code_workspace: Path | None = None,
) -> dict[str, Any]:
    """Combine archived analysis; frozen trajectories bypass files and provenance controls display paths.

    Only quality reads code_workspace. Unsafe tree-sitter marks quality unavailable
    without dropping the remaining evaluation.
    """
    daydream_dir = Path(daydream_dir)
    display_daydream_dir = daydream_dir if artifact_provenance is None else artifact_provenance.public_daydream_dir
    trajectories = (
        frozen_trajectories
        if frozen_trajectories is not None
        else load_trajectories(daydream_dir, session_id=session_id)
    )

    if not trajectories["main"] and not trajectories["forked"]:
        return {"error": f"No trajectory files found in {daydream_dir}"}

    main = trajectories["main"] or trajectories["forked"][0]
    session_id = main.get("session_id", "unknown")
    agent_info = main.get("agent", {})

    # Extract PR metadata from trajectory extra (set by TrajectoryRecorder)
    traj_extra = main.get("extra") or {}
    pr_number = traj_extra.get("pr_number")
    pr_repo = traj_extra.get("pr_repo")

    costs = analyze_costs(trajectories)
    tools = analyze_tools(trajectories)
    findings_data = analyze_findings(daydream_dir)
    location = analyze_location(daydream_dir)
    routing = analyze_routing(daydream_dir)
    shipped_duplication = analyze_shipped_duplication(daydream_dir)
    timing = analyze_timing(trajectories)
    training = analyze_training_signals(trajectories)
    try:
        quality = analyze_quality(daydream_dir, code_workspace=code_workspace)
    except TreeSitterBadVersionError as exc:
        # Unsafe native parsing marks only quality unavailable, preserving pure-Python archival analysis.
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
    cost_per_finding = round(costs["total_cost_usd"] / finding_count, 4) if finding_count > 0 else None

    result: dict[str, Any] = {
        "session_id": session_id,
        "agent": agent_info,
        "daydream_dir": str(display_daydream_dir),
        "trajectory_count": len(_all_trajectories(trajectories)),
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
