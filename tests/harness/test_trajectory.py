"""Self-test for the shared trajectory/manifest builders.

Drives the real ``TrajectoryRecorder`` through the harness builders and asserts
the observable outcome (a schema-valid trajectory on disk carrying the observed
text), plus the manifest and diff builders the archive/training tests rely on.
"""

from __future__ import annotations

from pathlib import Path

from daydream.atif import validate as atif_validate
from daydream.trajectory import DaydreamPhase, DaydreamRunFlow
from tests.harness.trajectory import (
    make_recorder,
    observe_text_and_result,
    read_trajectory,
)


async def test_make_recorder_writes_schema_valid_trajectory(tmp_path: Path) -> None:
    """The builders produce a real on-disk trajectory carrying the observed text."""
    recorder = make_recorder(tmp_path, run_flow=DaydreamRunFlow.DEEP, agent_model_name="sonnet")
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.DEEP) as inv:
            observe_text_and_result(inv, "harness-output")

    traj = read_trajectory(recorder.path)
    assert atif_validate(traj, validate_images=False) is True
    assert traj["agent"]["model_name"] == "sonnet"
    agent_steps = [s for s in traj["steps"] if s["source"] == "agent"]
    assert len(agent_steps) == 1
    assert agent_steps[0]["message"] == "harness-output"
    assert agent_steps[0]["extra"]["daydream_run_flow"] == "deep"


async def test_make_recorder_forwards_on_write(tmp_path: Path) -> None:
    """The on_write callback reaches the recorder and fires on a completed write."""
    calls: list[tuple[str, str]] = []
    recorder = make_recorder(
        tmp_path, on_write=lambda rec, status: calls.append((rec.session_id, status))
    )
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            observe_text_and_result(inv)

    assert calls == [("test", "complete")]
