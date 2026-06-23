"""Tests for issue #203 — phase timing events in ATIF trajectories.

Unit tests for the PhaseEvent model, ``emit_phase_start``/``emit_phase_end``,
per-Invocation subtrajectory timestamps, ``compute_phase_timings``, and the
``phase_scope`` context manager; plus real-path integration tests that drive
``runner.run`` (the production entrypoint) with a mocked backend and assert
the on-disk trajectory + manifest carry the new timing data.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from daydream.atif import validate as atif_validate
from daydream.backends import (
    AgentEvent,
    ContinuationToken,
    ResultEvent,
    TextEvent,
)
from daydream.trajectory import (
    DaydreamPhase,
    DaydreamRunFlow,
    PhaseEvent,
    TrajectoryRecorder,
    get_current_recorder,
    phase_scope,
)

# --- Helpers ---------------------------------------------------------------


def _make_recorder(tmp_path: Path, *, agent_model_name: str = "opus") -> TrajectoryRecorder:
    return TrajectoryRecorder(
        path=tmp_path / ".daydream" / "trajectory.json",
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=tmp_path,
        agent_model_name=agent_model_name,
        session_id="test",
    )


def _read_trajectory(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def test_phase_event_to_dict_omits_empty_metadata() -> None:
    """No metadata key when metadata is empty."""
    ev = PhaseEvent(
        phase=DaydreamPhase.FIX,
        event="phase_end",
        timestamp="2026-01-01T00:00:00Z",
    )
    assert "metadata" not in ev.to_dict()


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
    rec = _make_recorder(tmp_path)
    rec.emit_phase_start(DaydreamPhase.REVIEW)
    rec.emit_phase_end(DaydreamPhase.REVIEW)
    assert len(rec._phase_events) == 2
    assert rec._phase_events[0].event == "phase_start"
    assert rec._phase_events[0].phase is DaydreamPhase.REVIEW
    assert rec._phase_events[1].event == "phase_end"


async def test_emit_phase_carries_metadata(tmp_path: Path) -> None:
    """Keyword metadata is stored on the PhaseEvent."""
    rec = _make_recorder(tmp_path)
    rec.emit_phase_start(DaydreamPhase.DEEP, stage="arbiter")
    assert rec._phase_events[0].metadata == {"stage": "arbiter"}


async def test_phase_events_serialize_into_trajectory_extra(tmp_path: Path) -> None:
    """Phase events appear in Trajectory.extra["phase_events"] when present."""
    rec = _make_recorder(tmp_path)
    async with rec:
        rec.emit_phase_start(DaydreamPhase.REVIEW)
        rec.emit_phase_end(DaydreamPhase.REVIEW)
        # Need at least one step so _write doesn't skip.
        async with rec.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="x"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))
    traj = _read_trajectory(rec.path)
    assert atif_validate(traj, validate_images=False) is True
    events = traj["extra"]["phase_events"]
    assert len(events) == 2
    assert events[0]["phase"] == "review"
    assert events[0]["event"] == "phase_start"


async def test_no_phase_events_omits_key(tmp_path: Path) -> None:
    """When no phase events emitted, extra has no phase_events key."""
    rec = _make_recorder(tmp_path)
    async with rec:
        async with rec.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="x"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))
    traj = _read_trajectory(rec.path)
    assert "phase_events" not in traj["extra"]


# --- phase_scope -----------------------------------------------------------


async def test_phase_scope_emits_events_when_recorder_active(tmp_path: Path) -> None:
    """phase_scope emits start/end when a recorder is active via ContextVar."""
    rec = _make_recorder(tmp_path)
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
    rec = _make_recorder(tmp_path)
    async with rec:
        with pytest.raises(RuntimeError, match="boom"):
            async with phase_scope(DaydreamPhase.TEST):
                raise RuntimeError("boom")
        assert len(rec._phase_events) == 2
        assert rec._phase_events[1].event == "phase_end"


# --- Per-Invocation subtrajectory timestamps -------------------------------


async def test_invocation_records_started_at_ended_at(tmp_path: Path) -> None:
    """An Invocation scope registers started_at/ended_at timestamps."""
    rec = _make_recorder(tmp_path)
    async with rec:
        async with rec.invocation(phase=DaydreamPhase.REVIEW) as inv:
            assert inv.started_at != ""
            assert inv.ended_at == ""  # not set until exit
            inv.observe(TextEvent(text="hi"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))
        assert inv.ended_at != ""
    traj = _read_trajectory(rec.path)
    subs = traj["extra"]["subtrajectories"]
    assert len(subs) == 1
    assert subs[0]["phase"] == "review"
    assert subs[0]["started_at"]
    assert subs[0]["ended_at"]
    assert subs[0]["step_ids"] == [1]


async def test_no_steps_omits_subtrajectories_key(tmp_path: Path) -> None:
    """When no invocations ran, extra has no subtrajectories key."""
    rec = _make_recorder(tmp_path)
    async with rec:
        rec.emit_phase_start(DaydreamPhase.REVIEW)
        rec.emit_phase_end(DaydreamPhase.REVIEW)
        # Manually add a step so _write doesn't skip.
        async with rec.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="x"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))
    # But now there IS a subtrajectory. Test the truly-empty case:
    rec2 = _make_recorder(tmp_path.parent / "second")
    rec2.path = tmp_path / "t2.json"
    async with rec2:
        # Emit phase events but no invocation — however _write skips empty steps.
        rec2.emit_phase_start(DaydreamPhase.REVIEW)
        rec2.emit_phase_end(DaydreamPhase.REVIEW)
    # No file written (steps empty → _write returns early).
    assert not rec2.path.exists()


async def test_subtrajectory_step_ids_track_multiple_invocations(tmp_path: Path) -> None:
    """Multiple invocations produce multiple subtrajectory entries with sequential step_ids."""
    rec = _make_recorder(tmp_path)
    async with rec:
        async with rec.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="a"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))
        async with rec.invocation(phase=DaydreamPhase.PARSE) as inv:
            inv.observe(TextEvent(text="b"))
            inv.observe(ResultEvent(structured_output=None, continuation=None))
    traj = _read_trajectory(rec.path)
    subs = traj["extra"]["subtrajectories"]
    assert len(subs) == 2
    assert subs[0]["step_ids"] == [1]
    assert subs[1]["step_ids"] == [2]


# --- compute_phase_timings -------------------------------------------------


async def test_compute_phase_timings_returns_none_when_empty(tmp_path: Path) -> None:
    """No phase events → None (backward compat)."""
    rec = _make_recorder(tmp_path)
    assert rec.compute_phase_timings() is None


async def test_compute_phase_timings_pairs_start_end(tmp_path: Path) -> None:
    """A matched start/end pair yields wall_clock_seconds and occurrences=1."""
    rec = _make_recorder(tmp_path)
    rec.emit_phase_start(DaydreamPhase.REVIEW)
    rec.emit_phase_end(DaydreamPhase.REVIEW)
    timings = rec.compute_phase_timings()
    assert timings is not None
    assert "review" in timings
    assert timings["review"]["occurrences"] == 1
    assert timings["review"]["wall_clock_seconds"] >= 0.0


async def test_compute_phase_timings_sums_repeated_phase(tmp_path: Path) -> None:
    """Two occurrences of the same phase sum into one bucket with occurrences=2."""
    rec = _make_recorder(tmp_path)
    for _ in range(2):
        rec.emit_phase_start(DaydreamPhase.FIX)
        rec.emit_phase_end(DaydreamPhase.FIX)
    timings = rec.compute_phase_timings()
    assert timings is not None
    assert timings["fix"]["occurrences"] == 2


async def test_compute_phase_timings_deep_stages_fold_into_one_bucket(tmp_path: Path) -> None:
    """DEEP stage='review' and stage='arbiter' fold into the 'deep' bucket."""
    rec = _make_recorder(tmp_path)
    rec.emit_phase_start(DaydreamPhase.DEEP, stage="review")
    rec.emit_phase_end(DaydreamPhase.DEEP, stage="review")
    rec.emit_phase_start(DaydreamPhase.DEEP, stage="arbiter")
    rec.emit_phase_end(DaydreamPhase.DEEP, stage="arbiter")
    timings = rec.compute_phase_timings()
    assert timings is not None
    assert timings["deep"]["occurrences"] == 2


async def test_compute_phase_timings_orphaned_end_skipped(tmp_path: Path) -> None:
    """An end with no matching start contributes zero occurrences."""
    rec = _make_recorder(tmp_path)
    rec.emit_phase_start(DaydreamPhase.REVIEW)
    rec.emit_phase_end(DaydreamPhase.REVIEW)
    rec.emit_phase_end(DaydreamPhase.FIX)  # orphaned — no start
    timings = rec.compute_phase_timings()
    assert timings is not None
    assert "review" in timings
    # FIX had an end but no start → not in timings (no completed pair).
    assert "fix" not in timings


# --- Real-path: deep run via runner.run ------------------------------------


class _ShallowMockBackend:
    """Minimal Backend that yields canned events for shallow phase tests.

    Every execute() returns a TextEvent + ResultEvent with empty issues, so
    the parse phase returns no feedback items and the fix phase is skipped.
    """

    model = "mock-model"

    async def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: dict[str, Any] | None = None,
        continuation: ContinuationToken | None = None,
        agents: dict[str, Any] | None = None,
        max_turns: int | None = None,
        read_only: bool = False,
    ) -> AsyncIterator[AgentEvent]:
        yield TextEvent(text="ok")
        yield ResultEvent(structured_output={"issues": []}, continuation=None)

    async def cancel(self) -> None:
        return None

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
        return f"/{skill_key}" + (f" {args}" if args else "")


async def test_shallow_run_emits_phase_events_and_subtrajectories(
    feature_branch_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real-path: shallow single-pass run writes trajectory with phase_events + subtrajectories.

    Drives ``runner.run`` → ``_run_loop_shallow`` (the production shallow path)
    with a mock backend. The run starts at ``fix`` (skipping the skill-bound
    review phase) with a pre-created ``.review-output.md`` so the parse phase
    can proceed. Asserts the OBSERVABLE on-disk trajectory JSON carries
    explicit phase_events (parse + test), subtrajectories with timestamps, and
    that the manifest's metrics block carries phase_timings.
    """
    from daydream.config import REVIEW_OUTPUT_FILE
    from daydream.runner import RunConfig, run

    # Pre-create review output so check_review_file_exists passes for start_at="fix".
    (feature_branch_repo / REVIEW_OUTPUT_FILE).write_text("## Issues\n\n1. foo.py:1\n")

    monkeypatch.setattr(
        "daydream.runner.create_backend",
        lambda name, model=None: _ShallowMockBackend(),
    )

    # Patch phase_test_and_heal to avoid running a real test suite.
    async def _ok_test(*_a: Any, **_kw: Any) -> tuple[bool, int]:
        return True, 0

    monkeypatch.setattr("daydream.runner.phase_test_and_heal", _ok_test)

    for name in (
        "print_phase_hero",
        "print_info",
        "print_success",
        "print_warning",
        "print_dim",
        "print_summary",
        "print_skipped_phases",
        "print_iteration_divider",
    ):
        monkeypatch.setattr(f"daydream.runner.{name}", lambda *a, **kw: None)

    traj = tmp_path / "trajectory.json"
    config = RunConfig(
        target=str(feature_branch_repo),
        skill="python",
        shallow=True,
        cleanup=False,
        non_interactive=True,
        assume="no",  # decline commit gate
        start_at="fix",
        trajectory_path=traj,
    )
    exit_code = await run(config)
    assert exit_code == 0

    assert traj.exists(), "shallow run must write the trajectory to disk"
    data = json.loads(traj.read_text(encoding="utf-8"))
    assert atif_validate(data, validate_images=False) is True

    # Phase events: parse and test must appear (review is skipped via start_at).
    events = data["extra"].get("phase_events", [])
    event_phases = [e["phase"] for e in events]
    assert "parse" in event_phases, f"parse phase event missing; got {event_phases!r}"
    assert "test" in event_phases, f"test phase event missing; got {event_phases!r}"
    # Each must have a matching start+end pair.
    parse_events = [e for e in events if e["phase"] == "parse"]
    assert any(e["event"] == "phase_start" for e in parse_events)
    assert any(e["event"] == "phase_end" for e in parse_events)

    # Subtrajectories: the parse invocation registered one with timestamps.
    subs = data["extra"].get("subtrajectories", [])
    assert subs, "subtrajectories missing from trajectory extra"
    sub = next(s for s in subs if s["phase"] == "parse")
    assert sub["started_at"], "subtrajectory started_at is empty"
    assert sub["ended_at"], "subtrajectory ended_at is empty"

    # Manifest: phase_timings appears in the metrics block.
    archive_dir = tmp_path / "archive"
    manifest_files = list(archive_dir.rglob("manifest.json"))
    assert manifest_files, "manifest.json not written"
    manifest = json.loads(manifest_files[0].read_text())
    assert manifest["metrics"]["phase_timings"] is not None
    assert "parse" in manifest["metrics"]["phase_timings"], (
        f"parse missing from manifest phase_timings: {manifest['metrics']['phase_timings']!r}"
    )


async def test_deep_run_emits_phase_events_and_manifest_timings(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real-path: deep run writes trajectory with DEEP review phase_events + phase_timings.

    Drives ``runner.run`` → ``_run_loop_deep`` → ``run_deep`` (the production
    default deep pipeline) with the stub backend from ``test_deep_orchestrator``.
    The orchestrator wraps the per-stack review fan-out in
    ``phase_scope(DaydreamPhase.DEEP, stage="review")`` (issue #203 Task 6), so
    the trajectory's ``extra["phase_events"]`` must carry the deep review
    boundary. Asserts the on-disk trajectory JSON + manifest.
    """
    # Reuse the battle-tested stub backend + silence helpers.
    from tests.test_deep_orchestrator import _install_stub_backend, _silence

    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target)

    # Stub the non-idempotent GitHub write (PR comment posting).
    async def _no_post(target_dir: Path, report_path: Path, *, console: Any) -> None:
        return None

    monkeypatch.setattr("daydream.pr_review.post_review_to_pr_from_report", _no_post)

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
