"""Tests for trajectory phase timing events."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import datetime
from pathlib import Path
from typing import Any

import anyio
import pytest

import daydream.trajectory as trajectory_module
from daydream.atif import Step
from daydream.atif import validate as atif_validate
from daydream.backends import (
    AgentEvent,
    ResultEvent,
    TextEvent,
)
from daydream.trajectory import (
    DaydreamPhase,
    PhaseEvent,
    RunWriteSnapshot,
    TrajectoryDocumentSnapshot,
    compute_timing_summary,
    get_current_recorder,
    phase_scope,
)
from tests.harness.git_helpers import bare_remote, git
from tests.harness.phase_backend import PhaseDispatchBackend
from tests.harness.stub_backend import StubBackend
from tests.harness.trajectory import make_recorder, read_trajectory


def _snapshot_document(path: Path, payload: dict[str, Any]) -> TrajectoryDocumentSnapshot:
    return TrajectoryDocumentSnapshot(
        trajectory_id=str(payload["trajectory_id"]),
        path=path,
        json_bytes=json.dumps(payload, sort_keys=True).encode(),
    )


def test_overlap_coverage_and_recursive_invocation_identity(tmp_path: Path) -> None:
    """Identified intervals are unioned and wrappers never double-count calls."""
    session = "timing-session"
    root = {
        "session_id": session,
        "trajectory_id": session,
        "steps": [],
        "extra": {
            "run_started_at": "2026-01-01T00:00:00Z",
            "run_ended_at": "2026-01-01T00:00:10Z",
            "phase_events": [
                {
                    "phase": "deep",
                    "event": "phase_start",
                    "timestamp": "2026-01-01T00:00:01Z",
                    "session_id": session,
                    "scope_id": "deep-1",
                },
                {
                    "phase": "deep",
                    "event": "phase_end",
                    "timestamp": "2026-01-01T00:00:05Z",
                    "session_id": session,
                    "scope_id": "deep-1",
                    "status": "succeeded",
                },
                {
                    "phase": "deep",
                    "event": "phase_start",
                    "timestamp": "2026-01-01T00:00:03Z",
                    "session_id": session,
                    "scope_id": "deep-2",
                },
                {
                    "phase": "deep",
                    "event": "phase_end",
                    "timestamp": "2026-01-01T00:00:07Z",
                    "session_id": session,
                    "scope_id": "deep-2",
                    "status": "partial",
                },
                {
                    "phase": "diagram",
                    "event": "phase_start",
                    "timestamp": "2026-01-01T00:00:06Z",
                    "session_id": session,
                    "scope_id": "diagram-1",
                },
                {
                    "phase": "diagram",
                    "event": "phase_end",
                    "timestamp": "2026-01-01T00:00:09Z",
                    "session_id": session,
                    "scope_id": "diagram-1",
                    "status": "succeeded",
                },
            ],
            "subtrajectories": [
                {
                    "trajectory_id": "child",
                    "phase": "deep",
                    "invocations": [
                        {
                            "trajectory_id": "child",
                            "invocation_id": "child-call",
                            "phase": "deep",
                            "started_at": "2026-01-01T00:00:04Z",
                            "ended_at": "2026-01-01T00:00:06Z",
                        },
                        {
                            "trajectory_id": "child",
                            "invocation_id": "uncovered-call",
                            "phase": "fix",
                            "started_at": "2026-01-01T00:00:07Z",
                            "ended_at": "2026-01-01T00:00:08Z",
                        },
                    ],
                },
            ],
        },
    }
    nested = {
        "session_id": session,
        "trajectory_id": "nested",
        "steps": [],
        "extra": {
            "run_started_at": "2026-01-01T00:00:06Z",
            "run_ended_at": "2026-01-01T00:00:09Z",
            "subtrajectories": [
                {
                    "trajectory_id": "nested",
                    "invocation_id": "nested-call",
                    "phase": "diagram",
                    "started_at": "2026-01-01T00:00:06Z",
                    "ended_at": "2026-01-01T00:00:09Z",
                }
            ],
        },
    }
    child = {
        "session_id": session,
        "trajectory_id": "child",
        "steps": [],
        "extra": {
            "run_started_at": "2026-01-01T00:00:03Z",
            "run_ended_at": "2026-01-01T00:00:08Z",
            "subtrajectories": [
                {
                    "trajectory_id": "child",
                    "invocation_id": "child-call",
                    "phase": "deep",
                    "started_at": "2026-01-01T00:00:04Z",
                    "ended_at": "2026-01-01T00:00:06Z",
                },
                {
                    "trajectory_id": "child",
                    "invocation_id": "uncovered-call",
                    "phase": "fix",
                    "started_at": "2026-01-01T00:00:07Z",
                    "ended_at": "2026-01-01T00:00:08Z",
                },
            ],
        },
    }
    snapshot = RunWriteSnapshot(
        status="complete",
        cutoff_at="2026-01-01T00:00:10Z",
        root_trajectory_id=session,
        documents=(
            _snapshot_document(tmp_path / "trajectory.json", root),
            _snapshot_document(tmp_path / "trajectories" / "child.json", child),
            _snapshot_document(tmp_path / "trajectories" / "nested.json", nested),
        ),
    )

    summary = compute_timing_summary(snapshot)

    assert summary is not None
    assert summary.wall_clock_seconds == 10.0
    assert summary.phase_timings == {
        "deep": {"wall_clock_seconds": 6.0, "occurrences": 2},
        "diagram": {"wall_clock_seconds": 3.0, "occurrences": 1},
    }
    assert summary.attributed_wall_clock_seconds == 8.0
    assert summary.unattributed_wall_clock_seconds == 2.0
    assert summary.coverage_ratio == 0.8
    assert summary.agent_completeness == {
        "total": 3,
        "attributed": 2,
        "unattributed": 1,
    }


def test_malformed_identified_interval_is_diagnosed_not_coerced(tmp_path: Path) -> None:
    session = "bad-timing"
    payload = {
        "session_id": session,
        "trajectory_id": session,
        "steps": [],
        "extra": {
            "run_started_at": "2026-01-01T00:00:00Z",
            "run_ended_at": "2026-01-01T00:00:02Z",
            "phase_events": [
                {
                    "phase": "review",
                    "event": "phase_start",
                    "timestamp": "not-a-time",
                    "session_id": session,
                    "scope_id": "scope",
                },
                {
                    "phase": "review",
                    "event": "phase_end",
                    "timestamp": "2026-01-01T00:00:01Z",
                    "session_id": session,
                    "scope_id": "scope",
                    "status": "succeeded",
                },
            ],
        },
    }
    snapshot = RunWriteSnapshot(
        status="complete",
        cutoff_at="2026-01-01T00:00:02Z",
        root_trajectory_id=session,
        documents=(_snapshot_document(tmp_path / "trajectory.json", payload),),
    )

    summary = compute_timing_summary(snapshot)

    assert summary is not None
    assert summary.attributed_wall_clock_seconds == 0.0
    assert summary.diagnostics["malformed_interval"] == 1


# --- PhaseEvent.to_dict ----------------------------------------------------


def test_phase_event_to_dict_basic() -> None:
    """PhaseEvent serializes phase value, event, and timestamp."""
    ev = PhaseEvent(
        phase=DaydreamPhase.REVIEW,
        event="phase_start",
        timestamp="2026-01-01T00:00:00Z",
    )
    d = ev.to_dict()
    assert d == {
        "phase": "review",
        "event": "phase_start",
        "timestamp": "2026-01-01T00:00:00Z",
    }


def test_phase_event_to_dict_includes_metadata() -> None:
    """Metadata appears when non-empty."""
    ev = PhaseEvent(
        phase=DaydreamPhase.DEEP,
        event="phase_start",
        timestamp="2026-01-01T00:00:00Z",
        metadata={"stage": "review"},
    )
    d = ev.to_dict()
    assert d["metadata"] == {"stage": "review"}


# --- emit_phase_start / emit_phase_end -------------------------------------


async def test_emit_phase_start_end_appends_events(tmp_path: Path) -> None:
    """emit_phase_start/emit_phase_end append PhaseEvents in order."""
    rec = make_recorder(tmp_path)
    rec.emit_phase_start(DaydreamPhase.REVIEW)
    rec.emit_phase_end(DaydreamPhase.REVIEW)
    assert len(rec._phase_events) == 2
    assert rec._phase_events[0].event == "phase_start"
    assert rec._phase_events[0].phase is DaydreamPhase.REVIEW
    assert rec._phase_events[1].event == "phase_end"


async def test_emit_phase_carries_metadata(tmp_path: Path) -> None:
    """Keyword metadata is stored on the PhaseEvent."""
    rec = make_recorder(tmp_path)
    rec.emit_phase_start(DaydreamPhase.DEEP, stage="arbiter")
    assert rec._phase_events[0].metadata == {"stage": "arbiter"}


async def test_emit_supervisor_and_tool_veto_events(tmp_path: Path) -> None:
    """Supervisor decisions and tool vetoes are recorded as phase events."""
    rec = make_recorder(tmp_path)

    rec.emit_supervisor_verdict(7, "drop", "duplicate")
    rec.emit_tool_veto("Write", "protected path", phase=DaydreamPhase.FIX)

    assert rec._phase_events[0].event == "supervisor_verdict"
    assert rec._phase_events[0].phase is DaydreamPhase.DEEP
    assert rec._phase_events[0].metadata == {
        "finding_id": 7,
        "action": "drop",
        "reason": "duplicate",
    }
    assert rec._phase_events[1].event == "tool_veto"
    assert rec._phase_events[1].phase is DaydreamPhase.FIX
    assert rec._phase_events[1].metadata == {
        "tool_name": "Write",
        "reason": "protected path",
    }


async def test_emit_command_validation_summary_is_structured_and_redacted(
    tmp_path: Path,
) -> None:
    rec = make_recorder(tmp_path)

    rec.emit_command_validation_summary(
        total_candidates=33,
        accepted=4,
        rejected=29,
        reasons={"RECON_EVIDENCE_MISMATCH": 28, "RECON_MALFORMED_COMMAND": 1},
    )

    event = rec._phase_events[0]
    assert event.event == "command_validation"
    assert event.phase is DaydreamPhase.RECON
    assert event.metadata == {
        "counts": {
            "total_candidates": 33,
            "accepted": 4,
            "rejected": 29,
        },
        "reasons": {
            "RECON_EVIDENCE_MISMATCH": 28,
            "RECON_MALFORMED_COMMAND": 1,
        },
    }


async def test_phase_events_serialize_into_trajectory_extra(tmp_path: Path) -> None:
    """Phase events appear in Trajectory.extra["phase_events"] when present."""
    rec = make_recorder(tmp_path)
    async with rec:
        rec.emit_phase_start(DaydreamPhase.REVIEW)
        rec.emit_phase_end(DaydreamPhase.REVIEW)
        # Need at least one step so _write doesn't skip.
        async with rec.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="x"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))
    traj = read_trajectory(rec.path)
    assert atif_validate(traj, validate_images=False) is True
    events = traj["extra"]["phase_events"]
    assert len(events) == 2
    assert events[0]["phase"] == "review"
    assert events[0]["event"] == "phase_start"


async def test_no_phase_events_omits_key(tmp_path: Path) -> None:
    """When no phase events emitted, extra has no phase_events key."""
    rec = make_recorder(tmp_path)
    async with rec:
        async with rec.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="x"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))
    traj = read_trajectory(rec.path)
    assert "phase_events" not in traj["extra"]


# --- phase_scope -----------------------------------------------------------


async def test_phase_scope_emits_events_when_recorder_active(tmp_path: Path) -> None:
    """phase_scope emits start/end when a recorder is active via ContextVar."""
    rec = make_recorder(tmp_path)
    async with rec:
        async with phase_scope(DaydreamPhase.FIX):
            assert len(rec._phase_events) == 1  # start emitted
            assert rec._phase_events[0].event == "phase_start"
        assert len(rec._phase_events) == 2
        assert rec._phase_events[1].event == "phase_end"


async def test_phase_scope_noop_without_recorder() -> None:
    """phase_scope is a no-op when no recorder is active (no crash)."""
    assert get_current_recorder() is None
    async with phase_scope(DaydreamPhase.REVIEW):
        pass  # must not raise


async def test_phase_scope_emits_end_even_on_exception(tmp_path: Path) -> None:
    """phase_end fires even when the body raises (finally clause)."""
    rec = make_recorder(tmp_path)
    async with rec:
        with pytest.raises(RuntimeError, match="boom"):
            async with phase_scope(DaydreamPhase.TEST):
                raise RuntimeError("boom")
        assert len(rec._phase_events) == 2
        assert rec._phase_events[1].event == "phase_end"


async def test_phase_scope_id_pairs_concurrent_same_phase(tmp_path: Path) -> None:
    """Overlapping equal-valued phases close by identity, not phase-name LIFO."""
    rec = make_recorder(tmp_path)
    entered = {name: anyio.Event() for name in ("first", "second")}
    release = {name: anyio.Event() for name in entered}

    async def scoped(name: str) -> None:
        async with phase_scope(DaydreamPhase.DEEP, stage=name) as handle:
            assert handle is not None
            entered[name].set()
            await release[name].wait()

    async with rec:
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(scoped, "first")
            await entered["first"].wait()
            task_group.start_soon(scoped, "second")
            await entered["second"].wait()
            release["second"].set()
            await anyio.sleep(0)
            release["first"].set()
        async with rec.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="seed"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))

    events = [event for event in read_trajectory(rec.path)["extra"]["phase_events"] if event["phase"] == "deep"]
    assert [event["event"] for event in events] == [
        "phase_start",
        "phase_start",
        "phase_end",
        "phase_end",
    ]
    starts = {event["metadata"]["stage"]: event for event in events[:2]}
    ends = {event["metadata"]["stage"]: event for event in events[2:]}
    assert set(starts) == set(ends) == {"first", "second"}
    assert starts["first"]["scope_id"] == ends["first"]["scope_id"]
    assert starts["second"]["scope_id"] == ends["second"]["scope_id"]
    assert starts["first"]["scope_id"] != starts["second"]["scope_id"]
    assert all(event["session_id"] == rec.session_id for event in events)
    assert all(event["status"] == "succeeded" for event in ends.values())


async def test_phase_scope_id_rejects_second_or_post_close_decision(
    tmp_path: Path,
) -> None:
    """A phase handle accepts exactly one caller terminal decision."""
    assert hasattr(trajectory_module, "LifecycleStatus")
    rec = make_recorder(tmp_path)
    async with rec:
        async with phase_scope(DaydreamPhase.REVIEW) as handle:
            handle.finish(
                trajectory_module.LifecycleStatus.PARTIAL,
                trajectory_module.LifecycleReasonCode.SOME_CHILDREN_FAILED,
            )
            with pytest.raises(RuntimeError, match="terminal decision"):
                handle.finish(trajectory_module.LifecycleStatus.FAILED)
        with pytest.raises(RuntimeError, match="closed"):
            handle.finish(trajectory_module.LifecycleStatus.FAILED)
        async with rec.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="seed"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))

    terminal = read_trajectory(rec.path)["extra"]["phase_events"][1]
    assert terminal["status"] == "partial"
    assert terminal["reason_code"] == "some_children_failed"


# --- Per-Invocation subtrajectory timestamps -------------------------------


async def test_invocation_records_started_at_ended_at(tmp_path: Path) -> None:
    """An Invocation scope registers started_at/ended_at timestamps."""
    rec = make_recorder(tmp_path)
    async with rec:
        async with rec.invocation(phase=DaydreamPhase.REVIEW) as inv:
            assert inv.started_at != ""
            assert inv.ended_at == ""  # not set until exit
            inv.observe(TextEvent(text="hi"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))
        assert inv.ended_at != ""
    traj = read_trajectory(rec.path)
    subs = traj["extra"]["subtrajectories"]
    assert len(subs) == 1
    assert subs[0]["phase"] == "review"
    assert subs[0]["started_at"]
    assert subs[0]["ended_at"]
    assert subs[0]["step_ids"] == [1]


async def test_invocation_ended_at_not_before_final_step(tmp_path: Path) -> None:
    """ended_at is stamped after finish() flushes the still-open final step.

    A lone TextEvent leaves the step open, so finish() materializes it during
    __aexit__ with a fresh timestamp. ended_at must be stamped after that flush,
    otherwise it predates its own last step and underreports timing (#203).
    """
    rec = make_recorder(tmp_path)
    async with rec:
        async with rec.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="open step, never closed before exit"))
    traj = read_trajectory(rec.path)
    sub = traj["extra"]["subtrajectories"][0]
    step_ts = [s["timestamp"] for s in traj["steps"] if s["step_id"] in sub["step_ids"]]
    assert step_ts, "expected the open step to be flushed by finish()"
    assert sub["ended_at"] >= max(step_ts), f"ended_at {sub['ended_at']!r} predates final step {max(step_ts)!r}"


async def test_no_invocations_omits_subtrajectories_key(tmp_path: Path) -> None:
    """Zero invocations → extra has no subtrajectories key, even with steps present."""
    rec = make_recorder(tmp_path)
    async with rec:
        # Seed a step so _write does not take its empty-steps early return,
        # but open NO invocation — so _register_subtrajectory never fires.
        rec._extend_steps([Step(step_id=1, source="user", message="seed")])
    assert rec.path.exists()
    data = read_trajectory(rec.path)
    assert "subtrajectories" not in data["extra"]


async def test_subtrajectory_step_ids_track_multiple_invocations(
    tmp_path: Path,
) -> None:
    """Multiple invocations produce multiple subtrajectory entries with sequential step_ids."""
    rec = make_recorder(tmp_path)
    async with rec:
        async with rec.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="a"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))
        async with rec.invocation(phase=DaydreamPhase.PARSE) as inv:
            inv.observe(TextEvent(text="b"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))
    traj = read_trajectory(rec.path)
    subs = traj["extra"]["subtrajectories"]
    assert len(subs) == 2
    assert subs[0]["step_ids"] == [1]
    assert subs[1]["step_ids"] == [2]


async def test_fork_subtrajectory_entries_have_timestamps(tmp_path: Path) -> None:
    """Fork siblings register subtrajectory entries on the parent (issue #212)."""
    from daydream.trajectory import maybe_fork

    rec = make_recorder(tmp_path)
    async with rec:
        # Seed a parent step so _write does not take its empty-steps early
        # return; in a real run the parent always has prior-phase steps.
        rec._extend_steps([Step(step_id=1, source="user", message="seed")])
        async with maybe_fork(rec, "fix-src-foo-py") as child:
            async with child.invocation(phase=DaydreamPhase.FIX) as inv:
                inv.observe(TextEvent(text="fixing foo"))
                inv.observe(ResultEvent(structured_output=None, continuation=None))
    traj = read_trajectory(rec.path)
    subs = traj["extra"].get("subtrajectories", [])
    assert len(subs) == 1, f"expected 1 fork subtrajectory, got {len(subs)}: {subs}"
    sub = subs[0]
    assert sub["phase"] == "fix", f"expected phase 'fix' for descriptor 'fix-src-foo-py', got {sub['phase']!r}"
    assert sub["descriptor"] == "fix-src-foo-py", f"descriptor missing/incorrect: {sub}"
    assert sub["started_at"], "started_at must be non-empty"
    assert sub["ended_at"], "ended_at must be non-empty"
    assert sub["sibling_trajectory_ref"], "sibling_trajectory_ref must be non-empty"
    assert "step_ids" not in sub, "step_ids should be replaced by sibling_trajectory_ref"
    assert ".json" in sub["sibling_trajectory_ref"], (
        f"sibling_trajectory_ref should be a .json path, got {sub['sibling_trajectory_ref']!r}"
    )


# --- compute_phase_timings -------------------------------------------------


async def test_compute_phase_timings_returns_none_when_empty(tmp_path: Path) -> None:
    """No phase events → None (backward compat)."""
    rec = make_recorder(tmp_path)
    assert rec.compute_phase_timings() is None


async def test_compute_phase_timings_pairs_start_end(tmp_path: Path) -> None:
    """A matched start/end pair yields wall_clock_seconds and occurrences=1."""
    rec = make_recorder(tmp_path)
    rec.emit_phase_start(DaydreamPhase.REVIEW)
    rec.emit_phase_end(DaydreamPhase.REVIEW)
    timings = rec.compute_phase_timings()
    assert timings is not None
    assert "review" in timings
    assert timings["review"]["occurrences"] == 1
    assert timings["review"]["wall_clock_seconds"] >= 0.0


async def test_compute_phase_timings_sums_repeated_phase(tmp_path: Path) -> None:
    """Two occurrences of the same phase sum into one bucket with occurrences=2."""
    rec = make_recorder(tmp_path)
    for _ in range(2):
        rec.emit_phase_start(DaydreamPhase.FIX)
        rec.emit_phase_end(DaydreamPhase.FIX)
    timings = rec.compute_phase_timings()
    assert timings is not None
    assert timings["fix"]["occurrences"] == 2


async def test_compute_phase_timings_deep_stages_fold_into_one_bucket(
    tmp_path: Path,
) -> None:
    """DEEP stage='review' and stage='arbiter' fold into the 'deep' bucket."""
    rec = make_recorder(tmp_path)
    rec.emit_phase_start(DaydreamPhase.DEEP, stage="review")
    rec.emit_phase_end(DaydreamPhase.DEEP, stage="review")
    rec.emit_phase_start(DaydreamPhase.DEEP, stage="arbiter")
    rec.emit_phase_end(DaydreamPhase.DEEP, stage="arbiter")
    timings = rec.compute_phase_timings()
    assert timings is not None
    assert timings["deep"]["occurrences"] == 2


async def test_compute_phase_timings_orphaned_end_skipped(tmp_path: Path) -> None:
    """An end with no matching start contributes zero occurrences."""
    rec = make_recorder(tmp_path)
    rec.emit_phase_start(DaydreamPhase.REVIEW)
    rec.emit_phase_end(DaydreamPhase.REVIEW)
    rec.emit_phase_end(DaydreamPhase.FIX)  # orphaned — no start
    timings = rec.compute_phase_timings()
    assert timings is not None
    assert "review" in timings
    # FIX had an end but no start → not in timings (no completed pair).
    assert "fix" not in timings


async def test_compute_phase_timings_orphaned_start_pruned(tmp_path: Path) -> None:
    """A start with no matching end is pruned — no zero-occurrence bucket (symmetric to orphaned ends)."""
    rec = make_recorder(tmp_path)
    rec.emit_phase_start(DaydreamPhase.FIX)  # orphaned — no end
    rec.emit_phase_start(DaydreamPhase.REVIEW)
    rec.emit_phase_end(DaydreamPhase.REVIEW)
    timings = rec.compute_phase_timings()
    assert timings is not None
    assert "review" in timings
    # FIX had a start but no end → pruned as a zero-occurrence bucket.
    assert "fix" not in timings


# --- Real-path: deep run via runner.run ------------------------------------


class _OverlappingReviewBackend(StubBackend):
    """Hold the external wonder/reviewer calls until both are in flight."""

    def __init__(self, target: Path) -> None:
        super().__init__(target)
        self.entered = {"wonder": anyio.Event(), "review": anyio.Event()}
        self.finished: set[str] = set()

    async def execute(
        self, cwd: Path, prompt: str, output_schema: Any = None,
        continuation: Any = None, agents: Any = None,
        max_turns: Any = None, read_only: bool = False,
    ) -> AsyncIterator[AgentEvent]:
        lowered = prompt.lower()
        role = None
        if "would you have done this differently" in lowered or "evaluate the implementation" in lowered:
            role = "wonder"
        elif "you are reviewing the" in lowered or "you are the structural reviewer" in lowered:
            role = "review"
        if role is not None:
            self.entered[role].set()
            with anyio.fail_after(10):
                await self.entered["wonder"].wait()
                await self.entered["review"].wait()
        async for event in super().execute(cwd, prompt, output_schema, continuation, agents, max_turns, read_only):
            yield event
        if role is not None:
            self.finished.add(role)


def _seconds(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _phase_union_seconds(events: list[dict[str, Any]]) -> float:
    starts = {event["scope_id"]: event for event in events if event["event"] == "phase_start"}
    intervals = sorted(
        (_seconds(starts[event["scope_id"]]["timestamp"]), _seconds(event["timestamp"]))
        for event in events if event["event"] == "phase_end"
    )
    union: list[tuple[float, float]] = []
    for start, end in intervals:
        if union and start <= union[-1][1]:
            union[-1] = (union[-1][0], max(union[-1][1], end))
        else:
            union.append((start, end))
    return sum(end - start for start, end in union)


async def test_complete_overlapping_deep_run_timing_completeness(
    tmp_path: Path, archive_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One actual run proves fan-out, overlap, diagrams and frozen archive totals."""
    from daydream.runner import RunConfig, run
    from tests.harness import diagram_repos as dr

    target = dr.build_both_signals_repo(tmp_path)
    backend = _OverlappingReviewBackend(target)
    backend.diagram_specs = {
        "sequence": [dr.sequence_spec()],
        "flowchart": [dr.flowchart_spec(root_file="pkg_b/client.py", offset=10)],
    }
    backend.diagram_emit_reads = True
    backend.per_stack_emit_reads = True
    monkeypatch.setattr("daydream.runner.create_backend", lambda *args, **kwargs: backend)

    assert await run(RunConfig(target=str(target), cleanup=False, non_interactive=True)) == 0
    assert backend.finished == {"wonder", "review"}
    assert not any("Fix this issue" in call["prompt"] for call in backend.calls)
    roots = list((target / ".daydream" / "runs").glob("*/trajectory.json"))
    assert len(roots) == 1
    root = json.loads(roots[0].read_bytes())
    assert atif_validate(root, validate_images=False)
    events = root["extra"]["phase_events"]
    expected_metadata = {
        "merge": {"stage": "cross-stack-agent"},
        "diagram": {"stage": "diagram"},
    }
    for phase, metadata in expected_metadata.items():
        pair = [event for event in events if event["phase"] == phase]
        assert [event["event"] for event in pair] == ["phase_start", "phase_end"]
        assert pair[0]["scope_id"] == pair[1]["scope_id"]
        assert pair[1]["status"] == "succeeded"
        assert pair[0]["metadata"] == pair[1]["metadata"] == metadata
    assert not any(event["phase"] == "fix" for event in events)
    diagram = json.loads((target / ".daydream/deep/diagram.json").read_bytes())
    assert {kind: value["status"] for kind, value in diagram["results"].items()} == {
        "sequence": "rendered", "flowchart": "rendered",
    }

    documents = {root["trajectory_id"]: root}
    dispatches = [step for step in root["steps"] if "dispatch_id" in step.get("extra", {})]
    assert {
        step["extra"]["daydream_phase"]: [
            result["content"] for result in step["observation"]["results"]
        ]
        for step in dispatches
    } == {
        "exploration": ["Dispatched to explore-dependency_tracer"],
        "deep": ["Dispatched to deep-python", "Dispatched to deep-structure"],
        "diagram": [
            "Dispatched to diagram-sequence",
            "Dispatched to diagram-flowchart",
        ],
    }
    for step in dispatches:
        extra = step["extra"]
        refs = [ref for result in step["observation"]["results"] for ref in result["subagent_trajectory_ref"]]
        assert len(refs) == extra["planned_count"] == extra["attempted_count"] == extra["completed_count"]
        assert len({ref["trajectory_id"] for ref in refs}) == len(refs)
        assert extra["dispatch_status"] == "succeeded"
        for ref in refs:
            child = json.loads((target / ".daydream" / ref["trajectory_path"]).read_bytes())
            assert child["trajectory_id"] == ref["trajectory_id"]
            assert child["session_id"] == root["session_id"]
            assert extra["dispatch_started_at"] <= child["extra"]["run_started_at"]
            assert child["extra"]["run_ended_at"] <= extra["dispatch_completed_at"]
            documents[child["trajectory_id"]] = child

    invocation_keys: list[tuple[str, str]] = []
    for document_id, document in documents.items():
        for invocation in document["extra"].get("subtrajectories", []):
            if "invocation_id" in invocation:
                assert invocation["trajectory_id"] == document_id
                invocation_keys.append((document_id, invocation["invocation_id"]))
            else:
                assert "sibling_trajectory_ref" in invocation
    assert len(set(invocation_keys)) == len(invocation_keys) > len(documents)

    archive = archive_dir / "runs" / root["session_id"]
    manifest = json.loads((archive / "manifest.json").read_bytes())
    evaluation = json.loads((archive / "evaluation.json").read_bytes())
    coverage = manifest["metrics"]["timing_coverage"]
    assert coverage["agent_completeness"] == {
        "total": len(invocation_keys), "attributed": len(invocation_keys), "unattributed": 0,
    }
    assert all(value == 0 for value in coverage["diagnostics"].values())
    union = _phase_union_seconds(events)
    wall = _seconds(root["extra"]["run_ended_at"]) - _seconds(root["extra"]["run_started_at"])
    assert coverage["attributed_wall_clock_seconds"] == pytest.approx(union, abs=0.001)
    assert coverage["unattributed_wall_clock_seconds"] == pytest.approx(wall - union, abs=0.001)
    assert 0 <= coverage["coverage_ratio"] <= 1
    assert evaluation["timing"]["agent_completeness"] == coverage["agent_completeness"]
    assert evaluation["timing"]["total_wall_clock_seconds"] == manifest["metrics"]["wall_clock_seconds"]
    # Concurrent wonder/review intervals overlap, so summing their widths is not wall time.
    overlapping = [event for event in events if event["phase"] in {"alternatives", "deep"}]
    starts = {event["scope_id"]: event for event in overlapping if event["event"] == "phase_start"}
    summed = sum(
        _seconds(event["timestamp"]) - _seconds(starts[event["scope_id"]]["timestamp"])
        for event in overlapping if event["event"] == "phase_end"
    )
    assert _phase_union_seconds(overlapping) < summed
    assert not any(
        call["prompt"].lower().startswith(("fix this issue", "fix these"))
        for call in backend.calls
    )
    assert not (target / ".daydream-fix-applied").exists()
    assert not list(target.glob(".fixed-*"))


async def test_real_fix_fallback_records_multiple_invocations_in_one_fork(
    multi_stack_target: Path,
    tmp_path: Path,
    archive_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: Any,
) -> None:
    """A failed batch plus serial fallback stays one exact FIX child document."""
    from daydream.runner import run

    target = multi_stack_target
    origin = bare_remote(tmp_path / "origin.git")
    git(target, "remote", "add", "origin", str(origin))
    backend = StubBackend(target)
    backend.fail_batched_fix_file = "api.py"
    backend.fix_edit_line = "\n"
    monkeypatch.setattr(
        "daydream.runner.create_backend", lambda *args, **kwargs: backend
    )

    with anyio.fail_after(30):
        exit_code = await run(
            make_config(
                target,
                archive=True,
                assume="yes",
                cleanup=False,
                output_mode="loop",
                run_eval=True,
            )
        )

    assert exit_code == 0
    assert git(origin, "rev-parse", "refs/heads/feature") == git(
        target, "rev-parse", "HEAD"
    )
    fix_calls = [
        call
        for call in backend.calls
        if call["prompt"].lower().startswith(("fix this issue", "fix these"))
    ]
    api_batch = [
        call
        for call in fix_calls
        if call["prompt"].startswith("Fix these 3 issues in ")
        and "api.py:" in call["prompt"].splitlines()[0]
    ]
    api_serial = [
        call
        for call in fix_calls
        if call["prompt"].startswith("Fix this issue:")
        and any(
            line.startswith("File: ") and Path(line.removeprefix("File: ")).name == "api.py"
            for line in call["prompt"].splitlines()
        )
    ]
    assert len(api_batch) == 1
    assert len(api_serial) == 3

    roots = list((target / ".daydream" / "runs").glob("*/trajectory.json"))
    assert len(roots) == 1
    root = json.loads(roots[0].read_bytes())
    fix_dispatches = [
        step
        for step in root["steps"]
        if step.get("extra", {}).get("daydream_phase") == "fix"
        and "dispatch_id" in step.get("extra", {})
    ]
    assert len(fix_dispatches) == 1
    dispatch = fix_dispatches[0]
    assert dispatch["extra"]["dispatch_status"] == "succeeded"
    assert dispatch["extra"]["planned_count"] == 2
    assert dispatch["extra"]["attempted_count"] == 2
    assert dispatch["extra"]["completed_count"] == 2
    assert [
        result["content"] for result in dispatch["observation"]["results"]
    ] == ["Dispatched to fix-api.py", "Dispatched to fix-App.tsx"]
    api_result = dispatch["observation"]["results"][0]
    assert len(api_result["subagent_trajectory_ref"]) == 1
    api_ref = api_result["subagent_trajectory_ref"][0]
    assert Path(api_ref["trajectory_path"]).name.startswith("fix-api-py--")
    api_child = json.loads(
        (target / ".daydream" / api_ref["trajectory_path"]).read_bytes()
    )
    assert api_child["trajectory_id"] == api_ref["trajectory_id"]
    assert api_child["session_id"] == root["session_id"]

    api_summaries = [
        summary
        for summary in root["extra"]["subtrajectories"]
        if summary.get("dispatch_id") == dispatch["extra"]["dispatch_id"]
        and summary.get("descriptor") == "fix-api.py"
    ]
    assert len(api_summaries) == 1
    api_summary = api_summaries[0]
    assert "invocation_id" not in api_summary
    assert api_summary["trajectory_id"] == api_ref["trajectory_id"]
    assert api_summary["sibling_trajectory_ref"] == api_ref["trajectory_path"]
    assert api_summary["invocations"] == api_child["extra"]["subtrajectories"]
    api_invocations = api_summary["invocations"]
    assert len(api_invocations) == 4
    assert all(invocation["phase"] == "fix" for invocation in api_invocations)
    assert all(
        invocation["trajectory_id"] == api_child["trajectory_id"]
        for invocation in api_invocations
    )
    assert len(
        {
            (invocation["trajectory_id"], invocation["invocation_id"])
            for invocation in api_invocations
        }
    ) == 4
    assert all(
        api_child["extra"]["run_started_at"]
        <= invocation["started_at"]
        <= invocation["ended_at"]
        <= api_child["extra"]["run_ended_at"]
        for invocation in api_invocations
    )

    fix_events = [
        event for event in root["extra"]["phase_events"] if event["phase"] == "fix"
    ]
    assert [event["event"] for event in fix_events] == ["phase_start", "phase_end"]
    assert fix_events[0]["scope_id"] == fix_events[1]["scope_id"]
    assert fix_events[1]["status"] == "succeeded"
    assert dispatch["extra"]["dispatch_started_at"] <= api_child["extra"]["run_started_at"]
    assert api_child["extra"]["run_ended_at"] <= dispatch["extra"]["dispatch_completed_at"]
    assert all(
        fix_events[0]["timestamp"]
        <= invocation["started_at"]
        <= invocation["ended_at"]
        <= fix_events[1]["timestamp"]
        for invocation in api_invocations
    )

    documents = {root["trajectory_id"]: root}
    document_descriptors = {root["trajectory_id"]: "root"}
    pending = [root]
    wrapper_count = 0
    while pending:
        document = pending.pop()
        for entry in document["extra"].get("subtrajectories", []):
            sibling_path = entry.get("sibling_trajectory_ref")
            if sibling_path is None:
                continue
            wrapper_count += 1
            child = json.loads((target / ".daydream" / sibling_path).read_bytes())
            if child["trajectory_id"] not in documents:
                documents[child["trajectory_id"]] = child
                document_descriptors[child["trajectory_id"]] = Path(
                    sibling_path
                ).name.split("--", 1)[0]
                pending.append(child)
    invocation_keys = {
        (document_id, invocation["invocation_id"])
        for document_id, document in documents.items()
        for invocation in document["extra"].get("subtrajectories", [])
        if "invocation_id" in invocation
    }
    assert wrapper_count == len(documents) - 1 > 0
    assert len(invocation_keys) == len(backend.calls)
    assert len(invocation_keys) + wrapper_count > len(invocation_keys)
    qualified_counts: dict[tuple[str, str], int] = {}
    for document_id, document in documents.items():
        for invocation in document["extra"].get("subtrajectories", []):
            if "invocation_id" not in invocation:
                continue
            key = (document_descriptors[document_id], invocation["phase"])
            qualified_counts[key] = qualified_counts.get(key, 0) + 1
    assert qualified_counts == {
        ("root", "alternatives"): 1,
        ("root", "intent"): 1,
        ("root", "merge"): 1,
        ("root", "test"): 1,
        ("root", "verify"): 2,
        ("deep-generic", "deep"): 1,
        ("deep-python", "deep"): 1,
        ("deep-react", "deep"): 1,
        ("deep-structure", "deep"): 1,
        ("explore-dependency-tracer", "exploration"): 1,
        ("fix-api-py", "fix"): 4,
        ("fix-app-tsx", "fix"): 1,
    }

    verify_invocations = [
        invocation
        for invocation in root["extra"]["subtrajectories"]
        if invocation.get("phase") == "verify" and "invocation_id" in invocation
    ]
    verify_events = [
        event
        for event in root["extra"]["phase_events"]
        if event["phase"] == "verify"
    ]
    assert len(verify_invocations) == 2
    assert [event["event"] for event in verify_events] == [
        "phase_start",
        "phase_end",
        "phase_start",
        "phase_end",
    ]
    for start, end, invocation in zip(
        verify_events[::2], verify_events[1::2], verify_invocations, strict=True
    ):
        assert start["scope_id"] == end["scope_id"]
        assert end["status"] == "succeeded"
        assert start["timestamp"] <= invocation["started_at"]
        assert invocation["ended_at"] <= end["timestamp"]

    archive = archive_dir / "runs" / root["session_id"]
    manifest = json.loads((archive / "manifest.json").read_bytes())
    evaluation = json.loads((archive / "evaluation.json").read_bytes())
    completeness = {
        "total": len(invocation_keys),
        "attributed": len(invocation_keys),
        "unattributed": 0,
    }
    assert manifest["metrics"]["timing_coverage"]["agent_completeness"] == completeness
    assert evaluation["timing"]["agent_completeness"] == completeness


async def test_shallow_run_emits_phase_events_and_subtrajectories(
    feature_branch_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    silence_console: Any,
    mute_side_effects: Any,
) -> None:
    """Shallow single-pass run writes trajectory and manifest timing data."""
    from daydream.runner import RunConfig, run

    issue = {
        "id": 1,
        "description": "Add type hints",
        "file": "main.py",
        "line": 1,
    }

    monkeypatch.setattr(
        "daydream.runner.create_backend",
        lambda name, model=None, **kwargs: PhaseDispatchBackend(parse_results=[[issue]]),
    )

    mute_side_effects()
    silence_console("daydream.runner")
    silence_console("daydream.deep.orchestrator")

    traj = tmp_path / "trajectory.json"
    config = RunConfig(
        target=str(feature_branch_repo),
        stack="python",
        shallow=True,
        cleanup=False,
        non_interactive=True,
        assume="yes",  # accept the fix gate so the fix/test cycle runs
        trajectory_path=traj,
    )
    exit_code = await run(config)
    assert exit_code == 0

    assert traj.exists(), "shallow run must write the trajectory to disk"
    data = json.loads(traj.read_text(encoding="utf-8"))
    assert atif_validate(data, validate_images=False) is True

    # Phase events: the deep-shallow spine's review and the fix's test must
    # appear (the parse-<stack> stage was removed with issue #745).
    events = data["extra"].get("phase_events", [])
    event_phases = [e["phase"] for e in events]
    assert "test" in event_phases, f"test phase event missing; got {event_phases!r}"

    # Subtrajectories: the review invocation registered one with timestamps.
    subs = data["extra"].get("subtrajectories", [])
    assert subs, "subtrajectories missing from trajectory extra"
    assert all(s["started_at"] and s["ended_at"] for s in subs), "subtrajectory missing complete timestamps"

    # Manifest: phase_timings appears in the metrics block.
    archive_dir = tmp_path / "archive"
    manifest_files = list(archive_dir.rglob("manifest.json"))
    assert manifest_files, "manifest.json not written"
    manifest = json.loads(manifest_files[0].read_text())
    assert manifest["metrics"]["phase_timings"] is not None


async def test_deep_run_emits_phase_events_and_manifest_timings(
    multi_stack_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mute_side_effects: Any,
) -> None:
    """Real-path: deep run writes trajectory with DEEP review phase_events + phase_timings.

    Drives ``runner.run`` → ``_run_loop_deep`` → ``run_deep`` (the production
    default deep pipeline) with the stub backend from ``test_deep_orchestrator``.
    The orchestrator wraps the per-stack review fan-out in
    ``phase_scope(DaydreamPhase.DEEP, stage="review")``, so
    the trajectory's ``extra["phase_events"]`` must carry the deep review
    boundary. Asserts the on-disk trajectory JSON + manifest.
    """
    from tests.test_deep_orchestrator import _install_stub_backend, _silence

    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target)

    mute_side_effects()

    from daydream.runner import RunConfig, run

    traj = tmp_path / "trajectory.json"
    config = RunConfig(
        target=str(multi_stack_target),
        non_interactive=True,  # decline the apply-fixes gate
        trajectory_path=traj,
        cleanup=False,
    )
    exit_code = await run(config)
    assert exit_code == 0

    assert traj.exists(), "deep run must write the trajectory to disk"
    data = json.loads(traj.read_text(encoding="utf-8"))
    assert atif_validate(data, validate_images=False) is True

    # Phase events: the per-stack review stage (DEEP) must appear.
    events = data["extra"].get("phase_events", [])
    deep_events = [e for e in events if e["phase"] == "deep"]
    assert deep_events, f"deep phase events missing; got phases: {[e['phase'] for e in events]!r}"
    # The stage metadata should carry "review".
    deep_review_starts = [
        e for e in deep_events if e["event"] == "phase_start" and e.get("metadata", {}).get("stage") == "review"
    ]
    assert deep_review_starts, f"deep review stage start event missing; got: {deep_events!r}"

    # Subtrajectories: TTT invocations registered timing entries.
    subs = data["extra"].get("subtrajectories", [])
    assert subs, "subtrajectories missing from deep trajectory extra"
    assert all(s["started_at"] and s["ended_at"] for s in subs), f"subtrajectory missing timestamps: {subs!r}"

    # Manifest: phase_timings carries the deep bucket.
    archive_dir = tmp_path / "archive"
    manifest_files = list(archive_dir.rglob("manifest.json"))
    assert manifest_files, "manifest.json not written"
    manifest = json.loads(manifest_files[0].read_text())
    phase_timings = manifest["metrics"]["phase_timings"]
    assert phase_timings is not None
    assert "deep" in phase_timings, f"deep missing from manifest phase_timings: {phase_timings!r}"
    # Declined gate still records the phases reached before fix/test/verify. The
    # parse-<stack> stage was removed (issue #745), so it is not expected here.
    for phase in ("intent", "alternatives"):
        assert phase in phase_timings, f"{phase} missing from deep phase_timings: {phase_timings!r}"


async def test_deep_run_accept_gate_wraps_fix_test_verify(
    multi_stack_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mute_side_effects: Any,
) -> None:
    """Accepted deep fix gate records fix/test/verify timing events."""
    from tests.test_deep_orchestrator import _install_stub_backend, _silence

    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target)

    mute_side_effects()

    from daydream.runner import RunConfig, run

    traj = tmp_path / "trajectory.json"
    config = RunConfig(
        target=str(multi_stack_target),
        assume="yes",  # accept the apply-fixes gate -> verify/fix/test run
        trajectory_path=traj,
        cleanup=False,
    )
    exit_code = await run(config)
    assert exit_code == 0

    assert traj.exists(), "deep run must write the trajectory to disk"
    data = json.loads(traj.read_text(encoding="utf-8"))
    assert atif_validate(data, validate_images=False) is True

    # The longest phases -- fix/test/verify -- only run past an accepted gate.
    events = data["extra"].get("phase_events", [])
    event_phases = {e["phase"] for e in events}
    for phase in ("verify", "fix", "test"):
        assert phase in event_phases, f"{phase} phase_events missing; got phases: {sorted(event_phases)!r}"

    # Manifest: phase_timings must carry every wrapped deep phase.
    manifest_files = list((tmp_path / "archive").rglob("manifest.json"))
    assert manifest_files, "manifest.json not written"
    manifest = json.loads(manifest_files[0].read_text())
    phase_timings = manifest["metrics"]["phase_timings"]
    assert phase_timings is not None
    for phase in ("intent", "alternatives", "verify", "fix", "test", "deep"):
        assert phase in phase_timings, f"{phase} missing from deep phase_timings: {phase_timings!r}"


async def test_parallel_fix_registers_subtrajectories(
    multi_stack_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mute_side_effects: Any,
) -> None:
    """Real-path: parallel fix phase registers per-file subtrajectory entries.

    Drives ``runner.run`` through the deep pipeline with the stub backend,
    producing >=2 file findings that exercise ``phase_fix_parallel`` and the
    ``recorder.fork()`` path. Asserts multiple ``fix`` entries appear in
    ``extra["subtrajectories"]``.
    """
    from tests.test_deep_orchestrator import (
        _install_stub_backend,
        _merge_item,
        _silence,
    )

    _silence(monkeypatch)
    stub = _install_stub_backend(monkeypatch, multi_stack_target)
    # Override merge items to produce 2 findings on distinct files.
    stub.merge_items = [
        _merge_item(1, "api.py", "high"),
        _merge_item(2, "App.tsx", "medium"),
    ]

    mute_side_effects()

    from daydream.runner import RunConfig, run

    traj = tmp_path / "trajectory.json"
    config = RunConfig(
        target=str(multi_stack_target),
        assume="yes",
        trajectory_path=traj,
        cleanup=False,
    )
    exit_code = await run(config)
    assert exit_code == 0

    assert traj.exists(), "deep run must write the trajectory to disk"
    data = json.loads(traj.read_text(encoding="utf-8"))
    assert atif_validate(data, validate_images=False) is True

    subs = data["extra"].get("subtrajectories", [])
    fix_subs = [s for s in subs if s["phase"] == "fix"]
    assert len(fix_subs) >= 2, f"expected >=2 fix subtrajectories from parallel forks, got {len(fix_subs)}: {fix_subs}"
    for sub in fix_subs:
        assert sub["descriptor"].startswith("fix-"), f"fix subtrajectory descriptor must start with 'fix-': {sub}"
        assert sub["started_at"], f"fix subtrajectory missing started_at: {sub}"
        assert sub["ended_at"], f"fix subtrajectory missing ended_at: {sub}"
        assert sub["sibling_trajectory_ref"], f"fix subtrajectory missing sibling_trajectory_ref: {sub}"
        assert "step_ids" not in sub, f"step_ids should be replaced by sibling_trajectory_ref: {sub}"


async def test_review_flow_emits_phase_events_and_manifest_timings(
    multi_stack_target: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mute_side_effects: Any,
) -> None:
    """Review-only mode records review-spine timings and stops before fix/test.

    ``--review`` (a mode of the single deep flow, #330) runs the review spine —
    intent, alternatives, per-stack parse — and stops at ``findings-out``, so
    the fix cycle's fix/test/verify must never run (and must not appear in the
    recorded phase events or manifest timings).
    """
    from tests.test_deep_orchestrator import (
        _install_stub_backend,
        _pin_findings_pr,
        _silence,
    )

    _silence(monkeypatch)
    mute_side_effects()
    _pin_findings_pr(monkeypatch, multi_stack_target)
    _install_stub_backend(monkeypatch, multi_stack_target)

    from daydream.runner import RunConfig, run

    findings_out = tmp_path / "findings.json"
    traj = tmp_path / "trajectory.json"
    config = RunConfig(
        target=str(multi_stack_target),
        output_mode="review",
        pr_number=7,
        findings_out=str(findings_out),
        trajectory_path=traj,
        non_interactive=True,
        cleanup=False,
    )

    exit_code = await run(config)
    assert exit_code == 0
    assert findings_out.is_file(), "review mode must emit the findings artifact"

    assert traj.exists(), "review run must write the trajectory to disk"
    data = json.loads(traj.read_text(encoding="utf-8"))
    assert atif_validate(data, validate_images=False) is True

    # Review-only mode records intent + alternatives phases (the parse-<stack>
    # stage was removed with issue #745).
    events = data["extra"].get("phase_events", [])
    event_phases = {e["phase"] for e in events}
    for phase in ("intent", "alternatives"):
        assert phase in event_phases, f"{phase} phase_events missing; got phases: {sorted(event_phases)!r}"
    # The fix cycle must never run in review mode.
    for phase in ("fix", "test", "verify"):
        assert phase not in event_phases, (
            f"review mode ran the fix cycle phase {phase!r}; got: {sorted(event_phases)!r}"
        )

    # Manifest: phase_timings must be non-null (was null before the fix) and
    # carry the wrapped review phases.
    manifest_files = list((tmp_path / "archive").rglob("manifest.json"))
    assert manifest_files, "manifest.json not written"
    manifest = json.loads(manifest_files[0].read_text())
    phase_timings = manifest["metrics"]["phase_timings"]
    assert phase_timings is not None, "review flow phase_timings must not be null"
    for phase in ("intent", "alternatives"):
        assert phase in phase_timings, f"{phase} missing from review phase_timings: {phase_timings!r}"
