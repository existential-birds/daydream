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
    diff_adding,
    make_manifest,
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


def test_make_manifest_defaults_and_overrides() -> None:
    m = make_manifest()
    assert m.session_id == "sess-0001"
    assert m.status == "complete"
    assert m.skill == "python"

    pr_row = make_manifest("other", pr_number=7, pr_repo="o/r")
    assert pr_row.session_id == "other"
    assert (pr_row.pr_number, pr_row.pr_repo) == (7, "o/r")


def test_diff_adding_is_a_one_hunk_unified_diff() -> None:
    patch = diff_adding("new_line", file="pkg/mod.py")
    assert patch.startswith("diff --git a/pkg/mod.py b/pkg/mod.py\n")
    assert "@@ -1,1 +1,2 @@\n existing\n+new_line\n" in patch
