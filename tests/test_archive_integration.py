# tests/test_archive_integration.py
"""Integration tests for the TrajectoryRecorder on_write callback and archive pipeline.

Verifies that the on_write callback fires at the right times, that the full
archive round-trip produces valid bundles, and that CLI flags for --no-archive
and --eval are parsed correctly into RunConfig.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pytest

from daydream.atif import Step
from daydream.trajectory import (
    DaydreamPhase,
    DaydreamRunFlow,
    TrajectoryRecorder,
    now_iso,
)


def _make_recorder(
    tmp_path: Path,
    *,
    on_write: Any = None,
) -> TrajectoryRecorder:
    """Construct a TrajectoryRecorder with an optional on_write callback."""
    return TrajectoryRecorder(
        path=tmp_path / ".daydream" / "trajectory.json",
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=tmp_path,
        agent_model_name="test-model",
        session_id="test",
        on_write=on_write,
    )


def _add_user_step(recorder: TrajectoryRecorder) -> None:
    """Append a minimal user Step so the recorder has at least one step and won't skip _write."""
    step = Step(
        step_id=recorder._next_step_id(),
        timestamp=now_iso(),
        source="user",
        message="test prompt",
        extra={
            "daydream_phase": DaydreamPhase.REVIEW.value,
            "daydream_run_flow": DaydreamRunFlow.NORMAL.value,
        },
    )
    recorder.steps.append(step)


# on_write callback fires on normal write
async def test_on_write_fires_on_normal_write(tmp_path: Path) -> None:
    """on_write is called with (recorder, 'complete') when trajectory has steps."""
    callback_calls: list[tuple[str, str]] = []

    def on_write(recorder: TrajectoryRecorder, status: str) -> None:
        callback_calls.append((recorder.session_id, status))

    recorder = _make_recorder(tmp_path, on_write=on_write)
    async with recorder:
        _add_user_step(recorder)

    assert len(callback_calls) == 1
    assert callback_calls[0] == (recorder.session_id, "complete")


# on_write does NOT fire on empty trajectory
async def test_on_write_does_not_fire_on_empty_trajectory(tmp_path: Path) -> None:
    """Empty trajectories skip _write entirely, so on_write must not be called."""
    callback_calls: list[tuple[str, str]] = []

    def on_write(recorder: TrajectoryRecorder, status: str) -> None:
        callback_calls.append((recorder.session_id, status))

    recorder = _make_recorder(tmp_path, on_write=on_write)
    async with recorder:
        pass

    assert len(callback_calls) == 0
    assert not (tmp_path / ".daydream" / "trajectory.json").exists()


# Full archive round-trip via on_write
async def test_full_archive_round_trip(tmp_path: Path, archive_dir: Path) -> None:
    """_make_archive_callback wires archive_run through on_write, producing manifest + SQLite row."""
    from daydream.runner import RunConfig, _make_archive_callback

    # Set up a minimal .daydream/ structure the archive copier expects
    target_dir = tmp_path / "project"
    target_dir.mkdir()
    daydream_dir = target_dir / ".daydream"
    daydream_dir.mkdir()
    (target_dir / ".review-output.md").write_text("# Review\nLooks good.\n")

    config = RunConfig(
        target=str(target_dir),
        skill="python",
        backend="claude",
        archive=True,
        run_eval=False,
    )

    callback = _make_archive_callback(config, target_dir)
    assert callback is not None

    recorder = TrajectoryRecorder(
        path=daydream_dir / "trajectory.json",
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=target_dir,
        agent_model_name="opus",
        session_id="test",
        on_write=callback,
    )

    async with recorder:
        _add_user_step(recorder)

    assert (daydream_dir / "trajectory.json").exists()

    run_dir = archive_dir / "runs" / recorder.session_id
    assert run_dir.is_dir()

    manifest_path = run_dir / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["session_id"] == recorder.session_id
    assert manifest["status"] == "complete"
    assert manifest["run"]["flow"] == "normal"
    assert manifest["run"]["skill"] == "python"

    db_path = archive_dir / "index.db"
    assert db_path.exists()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM runs WHERE session_id = ?",
            (recorder.session_id,),
        ).fetchone()
        assert row is not None
        assert row["status"] == "complete"
        assert row["run_flow"] == "normal"
    finally:
        conn.close()


async def test_archive_populates_wall_clock_without_eval(tmp_path: Path, archive_dir: Path) -> None:
    """Real-path: manifest.json carries wall_clock_seconds even when --eval did not run."""
    from daydream.runner import RunConfig, _make_archive_callback

    target_dir = tmp_path / "project"
    target_dir.mkdir()
    (target_dir / ".daydream").mkdir()
    (target_dir / ".review-output.md").write_text("# Review\nLooks good.\n")

    config = RunConfig(
        target=str(target_dir),
        skill="python",
        backend="claude",
        archive=True,
        run_eval=False,  # the path that was previously leaving wall_clock null
    )
    callback = _make_archive_callback(config, target_dir)
    assert callback is not None

    recorder = TrajectoryRecorder(
        path=target_dir / ".daydream" / "trajectory.json",
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=target_dir,
        agent_model_name="opus",
        session_id="test",
        on_write=callback,
    )

    async with recorder:
        # Two steps spaced 8.5s apart so the derived span is deterministic.
        for ts in ("2026-05-31T10:00:00.000000Z", "2026-05-31T10:00:08.500000Z"):
            recorder.steps.append(
                Step(
                    step_id=recorder._next_step_id(),
                    timestamp=ts,
                    source="agent",
                    message="step",
                    extra={
                        "daydream_phase": DaydreamPhase.REVIEW.value,
                        "daydream_run_flow": DaydreamRunFlow.NORMAL.value,
                    },
                )
            )

    manifest_path = archive_dir / "runs" / recorder.session_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["metrics"]["wall_clock_seconds"] == 8.5


# on_write failure does not raise
async def test_on_write_failure_does_not_raise(tmp_path: Path) -> None:
    """If on_write raises, the context manager exits cleanly and trajectory is still written."""

    def on_write_boom(recorder: TrajectoryRecorder, status: str) -> None:
        raise RuntimeError("archive exploded")

    recorder = _make_recorder(tmp_path, on_write=on_write_boom)
    async with recorder:
        _add_user_step(recorder)

    # Trajectory should still be on disk despite the callback failure
    traj_path = tmp_path / ".daydream" / "trajectory.json"
    assert traj_path.exists()
    data = json.loads(traj_path.read_text(encoding="utf-8"))
    assert data["session_id"] == recorder.session_id
    assert len(data["steps"]) == 1


# CLI --no-archive flag
def test_cli_no_archive_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """--no-archive sets config.archive to False."""
    from daydream.cli import _parse_args

    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/fake", "--no-archive"])
    config = _parse_args()
    assert config.archive is False


# CLI --eval flag
def test_cli_eval_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    """--eval sets config.run_eval to True."""
    from daydream.cli import _parse_args

    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/fake", "--eval"])
    config = _parse_args()
    assert config.run_eval is True


# CLI defaults for archive and eval
def test_cli_defaults_archive_and_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without --no-archive or --eval, archive=True and run_eval=False."""
    from daydream.cli import _parse_args

    monkeypatch.setattr(sys, "argv", ["daydream", "/tmp/fake"])
    config = _parse_args()
    assert config.archive is True
    assert config.run_eval is False
