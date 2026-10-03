"""Pure timing and invocation-coverage reduction over frozen trajectory bytes."""
from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from daydream.timeutil import parse_iso_timestamp
from daydream.trajectory.layout import PARTIAL_SUFFIX, RUN_DOCUMENT_NAME
from daydream.trajectory.types import LifecycleStatus, RunWriteSnapshot


@dataclass(frozen=True)
class TimingSummary:
    """Overlap-aware timing and invocation coverage for one frozen run."""

    wall_clock_seconds: float
    phase_timings: dict[str, dict[str, int | float]]
    attributed_wall_clock_seconds: float
    unattributed_wall_clock_seconds: float
    coverage_ratio: float | None
    agent_completeness: dict[str, int]
    diagnostics: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON projection used by eval and manifests."""
        return {
            "total_wall_clock_seconds": self.wall_clock_seconds,
            "phase_timings": self.phase_timings,
            "attributed_wall_clock_seconds": self.attributed_wall_clock_seconds,
            "unattributed_wall_clock_seconds": self.unattributed_wall_clock_seconds,
            "coverage_ratio": self.coverage_ratio,
            "agent_completeness": self.agent_completeness,
            "diagnostics": self.diagnostics,
        }


_TIMING_DIAGNOSTIC_KEYS = (
    "malformed_interval",
    "duplicate_interval",
    "orphaned_interval",
    "malformed_invocation",
    "duplicate_invocation",
    "legacy_fork_proxy_used",
)


def _timing_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return parse_iso_timestamp(value)
    except ValueError:
        return None


def _timing_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _union_intervals(
    intervals: Sequence[tuple[datetime, datetime]],
) -> list[tuple[datetime, datetime]]:
    merged: list[tuple[datetime, datetime]] = []
    for start, end in sorted(intervals, key=lambda pair: (pair[0], pair[1])):
        if merged and start <= merged[-1][1]:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def _interval_seconds(intervals: Sequence[tuple[datetime, datetime]]) -> float:
    return sum((end - start).total_seconds() for start, end in intervals)


def _clip_intervals(
    intervals: Sequence[tuple[datetime, datetime]],
    wall: tuple[datetime, datetime],
) -> list[tuple[datetime, datetime]]:
    wall_start, wall_end = wall
    return [
        (max(start, wall_start), min(end, wall_end))
        for start, end in intervals
        if max(start, wall_start) < min(end, wall_end)
    ]


def _snapshot_payloads(write_snapshot: RunWriteSnapshot) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for document in write_snapshot.documents:
        try:
            value = json.loads(document.json_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if isinstance(value, dict):
            payloads.append(value)
    return payloads


def snapshot_trajectories(write_snapshot: RunWriteSnapshot) -> dict[str, Any]:
    """Return analyzer-shaped trajectory data from immutable document bytes."""
    main: dict[str, Any] | None = None
    forked: list[dict[str, Any]] = []
    for document in write_snapshot.documents:
        try:
            payload = json.loads(document.json_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid frozen trajectory JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("frozen trajectory document must be an object")
        if payload.get("trajectory_id") != document.trajectory_id:
            raise ValueError("frozen trajectory identity mismatch")
        copied = dict(payload)
        copied["_source_file"] = (
            RUN_DOCUMENT_NAME
            if document.trajectory_id == write_snapshot.root_trajectory_id
            else document.path.name.removesuffix(PARTIAL_SUFFIX)
        )
        if document.trajectory_id == write_snapshot.root_trajectory_id:
            if main is not None:
                raise ValueError("duplicate frozen root trajectory")
            main = copied
        else:
            forked.append(copied)
    if main is None and write_snapshot.documents:
        raise ValueError("frozen root trajectory is missing")
    return {"main": main, "forked": forked}


def compute_timing_summary(write_snapshot: RunWriteSnapshot) -> TimingSummary | None:
    """Reduce one immutable run snapshot without reading live recorder state."""
    return compute_payload_timing_summary(
        _snapshot_payloads(write_snapshot),
        root_trajectory_id=write_snapshot.root_trajectory_id,
        status=write_snapshot.status,
        cutoff_at=write_snapshot.cutoff_at,
    )


def compute_payload_timing_summary(
    payloads: Sequence[dict[str, Any]],
    *,
    root_trajectory_id: str,
    status: Literal["complete", "partial"] = "complete",
    cutoff_at: str = "",
) -> TimingSummary | None:
    """Reduce parsed immutable evidence for archive/eval readers without reconstructing write state."""
    root = next(
        (payload for payload in payloads if payload.get("trajectory_id") == root_trajectory_id),
        None,
    )
    if root is None:
        return None
    diagnostics = {key: 0 for key in _TIMING_DIAGNOSTIC_KEYS}
    root_extra = _timing_mapping(root.get("extra"))
    wall_start = _timing_timestamp(root_extra.get("run_started_at"))
    wall_end_key = "snapshot_at" if status == "partial" else "run_ended_at"
    if status == "partial" and root_extra.get("snapshot_at") != cutoff_at:
        return None
    if (
        status == "complete"
        and cutoff_at
        and root_extra.get("run_ended_at") != cutoff_at
    ):
        return None
    wall_end = _timing_timestamp(root_extra.get(wall_end_key))
    wall: tuple[datetime, datetime] | None = None
    if wall_start is not None and wall_end is not None and wall_start <= wall_end:
        wall = (wall_start, wall_end)
    elif "run_started_at" not in root_extra:
        legacy_times: list[datetime] = []
        for payload in payloads:
            for step in payload.get("steps", []):
                if isinstance(step, dict):
                    timestamp = _timing_timestamp(step.get("timestamp"))
                    if timestamp is not None:
                        legacy_times.append(timestamp)
        if len(legacy_times) >= 2:
            wall = (min(legacy_times), max(legacy_times))
    if wall is None:
        return None

    identified: dict[tuple[str, str, str], dict[str, list[datetime]]] = {}
    legacy_intervals: list[tuple[str, datetime, datetime]] = []
    for payload in payloads:
        extra = _timing_mapping(payload.get("extra"))
        pending_legacy: dict[str, list[datetime]] = {}
        events = extra.get("phase_events")
        if not isinstance(events, list):
            continue
        for event in events:
            if not isinstance(event, dict):
                continue
            phase = event.get("phase")
            kind = event.get("event")
            timestamp = _timing_timestamp(event.get("timestamp"))
            if not isinstance(phase, str) or kind not in {"phase_start", "phase_end"}:
                continue
            session_id = event.get("session_id")
            scope_id = event.get("scope_id")
            if session_id is not None or scope_id is not None:
                if not isinstance(session_id, str) or not session_id or not isinstance(scope_id, str) or not scope_id:
                    diagnostics["malformed_interval"] += 1
                    continue
                if kind == "phase_end" and event.get("status") not in {status.value for status in LifecycleStatus}:
                    diagnostics["malformed_interval"] += 1
                    continue
                key = (session_id, scope_id, phase)
                bucket = identified.setdefault(key, {"phase_start": [], "phase_end": []})
                if timestamp is None:
                    diagnostics["malformed_interval"] += 1
                else:
                    bucket[kind].append(timestamp)
                continue
            if timestamp is None:
                diagnostics["malformed_interval"] += 1
                continue
            stack = pending_legacy.setdefault(phase, [])
            if kind == "phase_start":
                stack.append(timestamp)
            elif stack:
                legacy_intervals.append((phase, stack.pop(), timestamp))
            else:
                diagnostics["orphaned_interval"] += 1
        diagnostics["orphaned_interval"] += sum(len(stack) for stack in pending_legacy.values())

    phase_intervals: dict[str, list[tuple[datetime, datetime]]] = {}
    for (_session_id, _scope_id, phase), pair in identified.items():
        starts = pair["phase_start"]
        ends = pair["phase_end"]
        if len(starts) != 1 or len(ends) != 1:
            if len(starts) > 1 or len(ends) > 1:
                diagnostics["duplicate_interval"] += 1
            else:
                diagnostics["orphaned_interval"] += 1
            continue
        start, end = starts[0], ends[0]
        if end < start:
            diagnostics["malformed_interval"] += 1
            continue
        phase_intervals.setdefault(phase, []).append((start, end))
    for phase, start, end in legacy_intervals:
        if end < start:
            diagnostics["malformed_interval"] += 1
        else:
            phase_intervals.setdefault(phase, []).append((start, end))

    clipped_by_phase = {
        phase: _union_intervals(_clip_intervals(intervals, wall)) for phase, intervals in phase_intervals.items()
    }
    phase_timings = {
        phase: {
            "wall_clock_seconds": round(_interval_seconds(intervals), 3),
            "occurrences": len(phase_intervals[phase]),
        }
        for phase, intervals in clipped_by_phase.items()
        if intervals
    }
    all_intervals = _union_intervals([interval for intervals in clipped_by_phase.values() for interval in intervals])
    wall_seconds = max(0.0, (wall[1] - wall[0]).total_seconds())
    attributed_seconds = _interval_seconds(all_intervals)

    invocation_rows: dict[tuple[str, str], list[dict[str, Any]]] = {}
    malformed_invocations = 0
    for payload in payloads:
        trajectory_id = payload.get("trajectory_id")
        extra = _timing_mapping(payload.get("extra"))
        summaries = extra.get("subtrajectories")
        direct_invocations = (
            [
                item
                for item in summaries
                if isinstance(item, dict) and "invocation_id" in item
            ]
            if isinstance(summaries, list)
            else []
        )
        if isinstance(summaries, list):
            malformed_invocations += sum(
                1
                for item in summaries
                if isinstance(item, dict)
                and "invocation_id" not in item
                and "invocations" not in item
                and "started_at" in item
                and "ended_at" in item
            )
        for invocation in direct_invocations:
            invocation_id = invocation.get("invocation_id")
            row_trajectory_id = invocation.get("trajectory_id")
            if not isinstance(row_trajectory_id, str) or not isinstance(invocation_id, str):
                malformed_invocations += 1
                continue
            invocation_rows.setdefault((row_trajectory_id, invocation_id), []).append(invocation)
        if payload is root or direct_invocations or "run_started_at" in extra:
            continue
        step_times = [
            parsed
            for step in payload.get("steps", [])
            if isinstance(step, dict)
            if (parsed := _timing_timestamp(step.get("timestamp"))) is not None
        ]
        if len(step_times) < 2:
            continue
        legacy_start, legacy_end = min(step_times), max(step_times)
        phase = next(
            (
                str((step.get("extra") or {}).get("daydream_phase"))
                for step in payload.get("steps", [])
                if isinstance(step, dict) and (step.get("extra") or {}).get("daydream_phase")
            ),
            "unknown",
        )
        invocation_rows[(str(trajectory_id), "legacy_fork_proxy")] = [
            {
                "phase": phase,
                "started_at": legacy_start.isoformat(),
                "ended_at": legacy_end.isoformat(),
            }
        ]
        diagnostics["legacy_fork_proxy_used"] += 1

    diagnostics["malformed_invocation"] += malformed_invocations
    attributed_invocations = 0
    unattributed_invocations = malformed_invocations
    for rows in invocation_rows.values():
        if len(rows) != 1:
            diagnostics["duplicate_invocation"] += 1
            unattributed_invocations += 1
            continue
        invocation = rows[0]
        phase = invocation.get("phase")
        invocation_start = _timing_timestamp(invocation.get("started_at"))
        invocation_end = _timing_timestamp(invocation.get("ended_at"))
        intervals = clipped_by_phase.get(phase, []) if isinstance(phase, str) else []
        if invocation_start is None or invocation_end is None or invocation_end < invocation_start:
            diagnostics["malformed_invocation"] += 1
            unattributed_invocations += 1
        elif any(
            interval_start <= invocation_start and invocation_end <= interval_end
            for interval_start, interval_end in intervals
        ):
            attributed_invocations += 1
        else:
            unattributed_invocations += 1

    return TimingSummary(
        wall_clock_seconds=round(wall_seconds, 3),
        phase_timings=phase_timings,
        attributed_wall_clock_seconds=round(attributed_seconds, 3),
        unattributed_wall_clock_seconds=round(max(0.0, wall_seconds - attributed_seconds), 3),
        coverage_ratio=(round(attributed_seconds / wall_seconds, 4) if wall_seconds > 0 else None),
        agent_completeness={
            "total": attributed_invocations + unattributed_invocations,
            "attributed": attributed_invocations,
            "unattributed": unattributed_invocations,
        },
        diagnostics=diagnostics,
    )
