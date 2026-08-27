"""Tests for trajectory phase timing events."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from daydream.atif import Step
from daydream.atif import validate as atif_validate
from daydream.backends import (
    ResultEvent,
    TextEvent,
)
from daydream.trajectory import (
    DaydreamPhase,
    PhaseEvent,
    get_current_recorder,
    phase_scope,
)
from tests.harness.phase_backend import PhaseDispatchBackend
from tests.harness.trajectory import make_recorder, read_trajectory

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
    assert sub["ended_at"] >= max(step_ts), (
        f"ended_at {sub['ended_at']!r} predates final step {max(step_ts)!r}"
    )


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


async def test_subtrajectory_step_ids_track_multiple_invocations(tmp_path: Path) -> None:
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


async def test_compute_phase_timings_deep_stages_fold_into_one_bucket(tmp_path: Path) -> None:
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
    assert all(s["started_at"] and s["ended_at"] for s in subs), (
        "subtrajectory missing complete timestamps"
    )

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
    ``phase_scope(DaydreamPhase.DEEP, stage="review")`` (issue #203 Task 6), so
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
        e for e in deep_events
        if e["event"] == "phase_start" and e.get("metadata", {}).get("stage") == "review"
    ]
    assert deep_review_starts, (
        f"deep review stage start event missing; got: {deep_events!r}"
    )

    # Subtrajectories: TTT invocations registered timing entries.
    subs = data["extra"].get("subtrajectories", [])
    assert subs, "subtrajectories missing from deep trajectory extra"
    assert all(s["started_at"] and s["ended_at"] for s in subs), (
        f"subtrajectory missing timestamps: {subs!r}"
    )

    # Manifest: phase_timings carries the deep bucket.
    archive_dir = tmp_path / "archive"
    manifest_files = list(archive_dir.rglob("manifest.json"))
    assert manifest_files, "manifest.json not written"
    manifest = json.loads(manifest_files[0].read_text())
    phase_timings = manifest["metrics"]["phase_timings"]
    assert phase_timings is not None
    assert "deep" in phase_timings, (
        f"deep missing from manifest phase_timings: {phase_timings!r}"
    )
    # Declined gate still records the phases reached before fix/test/verify. The
    # parse-<stack> stage was removed (issue #745), so it is not expected here.
    for phase in ("intent", "alternatives"):
        assert phase in phase_timings, (
            f"{phase} missing from deep phase_timings: {phase_timings!r}"
        )


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
        assert phase in event_phases, (
            f"{phase} phase_events missing; got phases: {sorted(event_phases)!r}"
        )

    # Manifest: phase_timings must carry every wrapped deep phase.
    manifest_files = list((tmp_path / "archive").rglob("manifest.json"))
    assert manifest_files, "manifest.json not written"
    manifest = json.loads(manifest_files[0].read_text())
    phase_timings = manifest["metrics"]["phase_timings"]
    assert phase_timings is not None
    for phase in ("intent", "alternatives", "verify", "fix", "test", "deep"):
        assert phase in phase_timings, (
            f"{phase} missing from deep phase_timings: {phase_timings!r}"
        )


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
    from tests.test_deep_orchestrator import _install_stub_backend, _merge_item, _silence

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
    assert len(fix_subs) >= 2, (
        f"expected >=2 fix subtrajectories from parallel forks, got {len(fix_subs)}: {fix_subs}"
    )
    for sub in fix_subs:
        assert sub["descriptor"].startswith("fix-"), (
            f"fix subtrajectory descriptor must start with 'fix-': {sub}"
        )
        assert sub["started_at"], f"fix subtrajectory missing started_at: {sub}"
        assert sub["ended_at"], f"fix subtrajectory missing ended_at: {sub}"
        assert sub["sibling_trajectory_ref"], f"fix subtrajectory missing sibling_trajectory_ref: {sub}"
        assert "step_ids" not in sub, (
            f"step_ids should be replaced by sibling_trajectory_ref: {sub}"
        )


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
    from tests.test_deep_orchestrator import _install_stub_backend, _pin_findings_pr, _silence

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
        assert phase in event_phases, (
            f"{phase} phase_events missing; got phases: {sorted(event_phases)!r}"
        )
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
        assert phase in phase_timings, (
            f"{phase} missing from review phase_timings: {phase_timings!r}"
        )
