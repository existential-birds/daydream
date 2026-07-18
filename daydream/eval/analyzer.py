"""Quantitative analysis of ATIF v1.7 trajectory data from daydream runs.

Parses trajectory JSON files and deep review artifacts, computes metrics
across cost, tool usage, file coverage, finding quality, grounding, and
training signal dimensions.  Output is a JSON-serializable dict consumed
by the evaluate-trajectory Claude Code skill.

Usage::

    from daydream.eval.analyzer import analyze_session
    report = analyze_session(Path("/path/to/.daydream"))
"""

import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from daydream.timeutil import parse_iso_timestamp

# Trajectory loading

def _latest_main_trajectory(daydream_dir: Path) -> Path | None:
    """Return the most recent main trajectory by mtime, or None.

    New layout: ``runs/<session_id>/trajectory.json``.
    """
    candidates = list(daydream_dir.glob("runs/*/trajectory.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _run_dir_trajectory_paths(run_dir: Path) -> list[Path]:
    """``trajectory.json`` plus sorted ``trajectories/*.json`` directly under *run_dir*."""
    paths: list[Path] = []
    main_path = run_dir / "trajectory.json"
    if main_path.is_file():
        paths.append(main_path)
    siblings_dir = run_dir / "trajectories"
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


def load_trajectories(daydream_dir: Path, session_id: str | None = None) -> dict:
    """Load trajectory files for a single session from a .daydream directory.

    New layout: ``runs/<session_id>/trajectory.json`` plus
    ``runs/<session_id>/trajectories/<descriptor>.json``.

    Args:
        daydream_dir: Path to the ``.daydream`` directory.
        session_id: Optional session ID (or prefix) to filter to. When None,
            the most recent main trajectory is used.

    Returns:
        Dict with ``main`` (root trajectory or None) and ``forked`` (list of
        subagent trajectories belonging to the same session).
    """
    main = None
    forked: list[dict] = []
    runs_dir = daydream_dir / "runs"

    # --- Resolve the run directory ------------------------------------------
    run_dir: Path | None = None
    if session_id:
        # Exact match first, then prefix match on run directory names
        exact = runs_dir / session_id
        if exact.is_dir():
            run_dir = exact
        elif runs_dir.is_dir():
            matches = sorted(
                d for d in runs_dir.iterdir()
                if d.is_dir() and d.name.startswith(session_id)
            )
            if len(matches) == 1:
                run_dir = matches[0]
            elif len(matches) > 1:
                raise ValueError(f"Session prefix '{session_id}' matches multiple runs")
    else:
        latest = _latest_main_trajectory(daydream_dir)
        if latest:
            # latest is runs/<session_id>/trajectory.json — parent is the run dir
            run_dir = latest.parent

    # --- Resolve the main and forked trajectories ---------------------------
    if run_dir:
        for path in _run_dir_trajectory_paths(run_dir):
            data = json.loads(path.read_text())
            data["_source_file"] = path.name
            if path.name == "trajectory.json":
                main = data
            else:
                forked.append(data)

    return {"main": main, "forked": forked}


# Helpers

def _parse_iso_ts(ts: str) -> datetime:
    return parse_iso_timestamp(ts)


def _agent_label(filename: str) -> str:
    """Human-readable label from trajectory filename.

    New layout::

        ``trajectory.json`` → ``main``
        ``deep-python.json`` → ``deep-python``

    Legacy layout::

        ``trajectory-20260429-121816-5f0088a9.json`` → ``main``
        ``10073f9b.deep-python.json`` → ``deep-python``
    """
    if filename.startswith("trajectory"):
        return "main"
    parts = filename.rsplit(".", 2)
    if len(parts) >= 3:
        return parts[1]
    return filename.replace(".json", "")


def _extract_tool_calls(trajectory: dict) -> list[dict]:
    calls: list[dict] = []
    for step in trajectory.get("steps", []):
        for tc in step.get("tool_calls") or []:
            calls.append({
                "step_id": step["step_id"],
                "function_name": tc["function_name"],
                "arguments": tc.get("arguments", {}),
                "phase": (step.get("extra") or {}).get("daydream_phase", "unknown"),
            })
    return calls


def _files_from_diff(diff_path: Path) -> list[str]:
    if not diff_path.exists():
        return []
    text = diff_path.read_text()
    files: set[str] = set()
    for m in re.finditer(r"^diff --git a/.+? b/(.+?)$", text, re.MULTILINE):
        files.add(m.group(1))
    return sorted(files)


def _files_read(tool_calls: list[dict]) -> set[str]:
    paths: set[str] = set()
    for tc in tool_calls:
        if tc["function_name"] == "Read":
            p = tc["arguments"].get("file_path", "")
            if p:
                paths.add(p)
        elif tc["function_name"] == "Grep":
            p = tc["arguments"].get("path", "")
            if p:
                paths.add(p)
    return paths


def _path_matches(absolute: str, relative: str) -> bool:
    """Check if an absolute tool-call path corresponds to a relative diff path."""
    return absolute.endswith(relative) or absolute.endswith("/" + relative)


# Analysis functions

def _all_trajectories(trajectories: dict) -> list[dict]:
    all_trajs: list[dict] = []
    if trajectories["main"]:
        all_trajs.append(trajectories["main"])
    all_trajs.extend(trajectories["forked"])
    return all_trajs


def analyze_costs(trajectories: dict) -> dict:
    """Cost and token breakdown across all agents.

    Sums costs across all trajectory files.  The ATIF spec says
    ``total_cost_usd`` should include subagent costs, but in practice the
    daydream recorder emits per-agent costs separately in forked files, so we
    sum everything to get the true total.

    The recorder also stores only *non-cached* tokens in ``prompt_tokens``
    (despite the spec saying it should include cached).  We detect this when
    ``cached_tokens > prompt_tokens`` and compute ``total_input_tokens`` as
    ``prompt + cached`` in that case.
    """
    agents: list[dict] = []
    for traj in _all_trajectories(trajectories):
        label = _agent_label(traj["_source_file"])
        fm = traj.get("final_metrics") or {}
        agents.append({
            "agent": label,
            "cost_usd": fm.get("total_cost_usd") or 0.0,
            "prompt_tokens": fm.get("total_prompt_tokens") or 0,
            "completion_tokens": fm.get("total_completion_tokens") or 0,
            "cached_tokens": fm.get("total_cached_tokens") or 0,
            "steps": fm.get("total_steps") or len(traj.get("steps", [])),
            "model": traj.get("agent", {}).get("model_name", "unknown"),
        })

    total_cost = sum(a["cost_usd"] for a in agents)
    total_prompt = sum(a["prompt_tokens"] for a in agents)
    total_completion = sum(a["completion_tokens"] for a in agents)
    total_cached = sum(a["cached_tokens"] for a in agents)

    # Detect recorder quirk: prompt_tokens = non-cached only
    if total_cached > total_prompt:
        total_input = total_prompt + total_cached
        cache_hit_rate = total_cached / total_input if total_input > 0 else 0.0
    else:
        total_input = total_prompt
        cache_hit_rate = total_cached / total_prompt if total_prompt > 0 else 0.0

    return {
        "total_cost_usd": round(total_cost, 4),
        "total_input_tokens": total_input,
        "total_prompt_tokens_raw": total_prompt,
        "total_completion_tokens": total_completion,
        "total_cached_tokens": total_cached,
        "cache_hit_rate": round(cache_hit_rate, 4),
        "by_agent": sorted(agents, key=lambda a: a["cost_usd"], reverse=True),
    }


def analyze_tools(trajectories: dict) -> dict:
    """Tool call counts, per-agent breakdown, and redundancy detection."""
    total_counts: Counter = Counter()
    by_agent: dict[str, dict] = {}
    redundant_reads: list[dict] = []

    for traj in _all_trajectories(trajectories):
        label = _agent_label(traj["_source_file"])
        calls = _extract_tool_calls(traj)
        counts = Counter(tc["function_name"] for tc in calls)
        by_agent[label] = dict(counts)
        total_counts.update(counts)

        read_paths = [
            tc["arguments"].get("file_path", "")
            for tc in calls
            if tc["function_name"] == "Read"
        ]
        for path, count in Counter(read_paths).items():
            if count > 1 and path:
                redundant_reads.append({"agent": label, "file": path, "read_count": count})

    total = sum(total_counts.values())
    write_count = total_counts.get("Write", 0)

    return {
        "total_calls": total,
        "by_type": dict(total_counts.most_common()),
        "by_agent": by_agent,
        "write_ratio": round(write_count / total, 4) if total > 0 else 0,
        "redundant_reads": redundant_reads,
    }


def analyze_coverage(trajectories: dict, daydream_dir: Path) -> dict:
    """File review coverage: files in diff vs files read by review agents."""
    diff_files = _files_from_diff(daydream_dir / "diff.patch")

    review_reads: set[str] = set()
    for traj in trajectories["forked"]:
        label = _agent_label(traj["_source_file"])
        if label.startswith("deep-"):
            review_reads.update(_files_read(_extract_tool_calls(traj)))

    if trajectories["main"]:
        for tc in _extract_tool_calls(trajectories["main"]):
            if tc["phase"] in ("deep", "alternatives"):
                if tc["function_name"] in ("Read", "Grep"):
                    p = tc["arguments"].get("file_path") or tc["arguments"].get("path", "")
                    if p:
                        review_reads.add(p)

    covered = {df for df in diff_files if any(_path_matches(r, df) for r in review_reads)}
    uncovered = sorted(set(diff_files) - covered)

    return {
        "files_in_diff": len(diff_files),
        "files_read_by_reviewers": len(covered),
        "coverage_ratio": round(len(covered) / len(diff_files), 4) if diff_files else 1.0,
        "uncovered_files": uncovered,
    }


def analyze_findings(daydream_dir: Path) -> dict:
    """Parse per-stack records, dedup stats, and merged review."""
    deep_dir = daydream_dir / "deep"
    if not deep_dir.is_dir():
        return {
            "total": 0,
            "by_confidence": {},
            "findings": [],
            "stacks": [],
            "dedup": {},
            "merged_review": {},
        }

    all_findings: list[dict] = []
    stacks: list[dict] = []

    for f in sorted(deep_dir.glob("stack-*-records.json")):
        stack_name = f.stem.replace("stack-", "").replace("-records", "")
        records = json.loads(f.read_text())
        stacks.append({"name": stack_name, "finding_count": len(records)})
        for r in records:
            r["_stack"] = stack_name
            all_findings.append(r)

    confidence_counts = Counter(f.get("confidence", "UNKNOWN") for f in all_findings)

    dedup_stats: dict = {}
    dedup_path = deep_dir / "dedup-candidates.json"
    if dedup_path.exists():
        dedup = json.loads(dedup_path.read_text())
        pairs = dedup.get("record_alt_pairs", [])
        dupes = dedup.get("record_duplicate_pairs", [])
        avg_sim = sum(p.get("similarity", 0) for p in pairs) / len(pairs) if pairs else 0
        dedup_stats = {
            "record_alt_overlaps": len(pairs),
            "record_duplicates": len(dupes),
            "avg_overlap_similarity": round(avg_sim, 4),
        }

    merged_review: dict = {}
    review_path = deep_dir / "review-output.md"
    if review_path.exists():
        text = review_path.read_text()
        merged_review["merged_finding_count"] = len(
            re.findall(r"^\d+\.\s+\[", text, re.MULTILINE)
        )

    return {
        "total": len(all_findings),
        "by_confidence": dict(confidence_counts),
        "findings": all_findings,
        "stacks": stacks,
        "dedup": dedup_stats,
        "merged_review": merged_review,
    }


def analyze_grounding(trajectories: dict, findings: list[dict]) -> dict:
    """Tier 1 grounding: verify cited files were actually read by the agent."""
    agent_reads: dict[str, set[str]] = {}
    for traj in trajectories["forked"]:
        label = _agent_label(traj["_source_file"])
        agent_reads[label] = _files_read(_extract_tool_calls(traj))

    grounded: list[dict] = []
    ungrounded: list[dict] = []

    for finding in findings:
        stack = finding.get("_stack", "")
        reads = agent_reads.get(f"deep-{stack}", set())

        cited_file = finding.get("file", "")
        rationale = finding.get("rationale", "")

        file_was_read = any(_path_matches(r, cited_file) for r in reads)

        rationale_refs = re.findall(r"[\w/.:-]+\.(?:md|json|py|ts|tsx|js|txt|yaml|yml|in|toml|cfg)", rationale)
        unread_refs = [
            ref for ref in rationale_refs
            if not any(_path_matches(r, ref) for r in reads)
        ]

        entry = {
            "id": finding.get("id"),
            "stack": stack,
            "file": cited_file,
            "confidence": finding.get("confidence", "UNKNOWN"),
            "file_was_read": file_was_read,
            "unread_rationale_refs": unread_refs,
            "grounded": file_was_read and len(unread_refs) == 0,
        }
        (grounded if entry["grounded"] else ungrounded).append(entry)

    total = len(findings)
    return {
        "total_findings": total,
        "grounded_count": len(grounded),
        "ungrounded_count": len(ungrounded),
        "grounding_rate": round(len(grounded) / total, 4) if total > 0 else 1.0,
        "grounded": grounded,
        "ungrounded": ungrounded,
    }


def analyze_exploration_utilization(trajectories: dict) -> dict:
    """Check whether review agents referenced exploration artifacts."""
    results: list[dict] = []

    for traj in trajectories["forked"]:
        label = _agent_label(traj["_source_file"])
        if not label.startswith("deep-"):
            continue

        calls = _extract_tool_calls(traj)
        exploration_refs: list[str] = []
        total_reads = 0

        for tc in calls:
            if tc["function_name"] in ("Read", "Grep"):
                total_reads += 1
                path = tc["arguments"].get("file_path") or tc["arguments"].get("path", "")
                if "/exploration/" in path:
                    exploration_refs.append(path)

        results.append({
            "agent": label,
            "total_reads": total_reads,
            "exploration_reads": len(exploration_refs),
            "utilized": len(exploration_refs) > 0,
        })

    utilized = sum(1 for r in results if r["utilized"])
    total_reviewers = len(results)

    return {
        "reviewers_utilizing_exploration": utilized,
        "total_reviewers": total_reviewers,
        "utilization_rate": round(utilized / total_reviewers, 4) if total_reviewers > 0 else 0,
        "by_agent": results,
    }


def analyze_timing(trajectories: dict) -> dict:
    """Wall-clock timing from step timestamps."""
    all_timestamps: list[datetime] = []
    agent_timings: list[dict] = []

    for traj in _all_trajectories(trajectories):
        label = _agent_label(traj["_source_file"])
        ts_list = [
            _parse_iso_ts(s["timestamp"])
            for s in traj.get("steps", [])
            if s.get("timestamp")
        ]
        if len(ts_list) >= 2:
            duration = (ts_list[-1] - ts_list[0]).total_seconds()
            agent_timings.append({"agent": label, "duration_seconds": round(duration, 1)})
        all_timestamps.extend(ts_list)

    total_duration = 0.0
    if len(all_timestamps) >= 2:
        total_duration = (max(all_timestamps) - min(all_timestamps)).total_seconds()

    return {
        "total_wall_clock_seconds": round(total_duration, 1),
        "by_agent": sorted(agent_timings, key=lambda a: a["duration_seconds"], reverse=True),
    }


def analyze_training_signals(
    trajectories: dict,
    findings: list[dict],
    grounding: dict,
) -> dict:
    """Assess trajectory quality for ML training purposes."""
    signals: list[dict] = []

    for traj in trajectories["forked"]:
        label = _agent_label(traj["_source_file"])
        steps = traj.get("steps", [])

        has_reasoning = any(s.get("reasoning_content") for s in steps)
        total_tool_calls = sum(len(s.get("tool_calls") or []) for s in steps)

        # Reasoning token fraction (approximation from char length)
        reasoning_chars = sum(len(s.get("reasoning_content") or "") for s in steps)
        message_chars = sum(
            len(s["message"]) for s in steps
            if s.get("source") == "agent" and isinstance(s.get("message"), str)
        )
        total_output_chars = reasoning_chars + message_chars
        reasoning_fraction = (
            round(reasoning_chars / total_output_chars, 4)
            if total_output_chars > 0
            else 0
        )

        noise_flags: list[str] = []
        for s in steps:
            obs = s.get("observation")
            if obs:
                for r in obs.get("results", []):
                    content = r.get("content", "")
                    if isinstance(content, str) and content.strip() == "":
                        noise_flags.append("empty_tool_result")
                        break

        # Extract stack name from agent label (e.g. "deep-python" → "python")
        agent_stack = label.removeprefix("deep-").removeprefix("explore-")
        agent_ungrounded = [
            g for g in grounding.get("ungrounded", [])
            if g.get("stack", "") == agent_stack
        ]
        if agent_ungrounded:
            noise_flags.append(f"ungrounded_findings:{len(agent_ungrounded)}")

        signals.append({
            "trajectory": label,
            "source_file": traj["_source_file"],
            "steps": len(steps),
            "has_reasoning": has_reasoning,
            "tool_calls": total_tool_calls,
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

def analyze_session(daydream_dir: str | Path, session_id: str | None = None) -> dict:
    """Run full quantitative analysis on a .daydream directory.

    Args:
        daydream_dir: Path to the ``.daydream`` directory from a completed run.
        session_id: Optional session ID (or prefix) to analyze. Defaults to the
            most recent session.
    """
    daydream_dir = Path(daydream_dir)
    trajectories = load_trajectories(daydream_dir, session_id=session_id)

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
    coverage = analyze_coverage(trajectories, daydream_dir)
    grounding = analyze_grounding(trajectories, findings_data["findings"])
    exploration = analyze_exploration_utilization(trajectories)
    timing = analyze_timing(trajectories)
    training = analyze_training_signals(
        trajectories, findings_data["findings"], grounding,
    )

    finding_count = findings_data["total"]
    cost_per_finding = (
        round(costs["total_cost_usd"] / finding_count, 4)
        if finding_count > 0
        else None
    )

    result: dict = {
        "session_id": session_id,
        "agent": agent_info,
        "daydream_dir": str(daydream_dir),
        "trajectory_count": len(_all_trajectories(trajectories)),
        "cost": costs,
        "timing": timing,
        "tools": tools,
        "coverage": coverage,
        "findings": {
            "total": finding_count,
            "by_confidence": findings_data["by_confidence"],
            "stacks": findings_data["stacks"],
            "dedup": findings_data["dedup"],
            "merged_review": findings_data.get("merged_review", {}),
        },
        "grounding": grounding,
        "exploration_utilization": exploration,
        "training_signals": training,
        "derived": {
            "cost_per_finding_usd": cost_per_finding,
        },
    }

    if pr_number is not None or pr_repo is not None:
        result["pr"] = {"pr_number": pr_number, "pr_repo": pr_repo}

    return result
