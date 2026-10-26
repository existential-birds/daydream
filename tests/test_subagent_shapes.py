"""Validate real root/sibling recorder output for deep review and exploration fan-out."""

from __future__ import annotations

from pathlib import Path

from daydream.atif import validate as atif_validate
from daydream.trajectory import (
    DaydreamPhase,
    TrajectoryRecorder,
    dispatch_scope,
)
from tests.harness.trajectory import (
    make_recorder,
    observe_text_and_result,
    read_trajectory,
)


async def test_exploration_produces_per_specialist_siblings(tmp_path: Path) -> None:
    """Three specialists produce valid siblings in the per-run directory, sharing the parent session."""
    recorder = make_recorder(tmp_path)
    children: list[TrajectoryRecorder] = []
    descriptors = ("explore-pattern-scanner", "explore-dependency-tracer", "explore-test-mapper")

    async with recorder:
        async with dispatch_scope(recorder, phase=DaydreamPhase.EXPLORATION, descriptors=descriptors) as dispatch:
            for desc in descriptors:
                async with recorder.fork(desc, dispatch=dispatch) as child:
                    children.append(child)
                    async with child.invocation(phase=DaydreamPhase.EXPLORATION) as inv:
                        observe_text_and_result(inv, f"{desc}-output")

    parent_traj = read_trajectory(recorder.path)
    assert atif_validate(parent_traj) is True

    dispatch_steps = [s for s in parent_traj["steps"]
        if s["source"] == "agent" and "Dispatching" in s.get("message", "")
    ]
    assert len(dispatch_steps) == 1
    results = dispatch_steps[0]["observation"]["results"]
    assert len(results) == 3
    # Each result carries exactly one sibling trajectory ref naming its child.
    descriptors_found = set()
    for r in results:
        refs = r["subagent_trajectory_ref"]
        assert len(refs) == 1
        path_str = refs[0]["trajectory_path"]
        for desc in descriptors:
            if desc in path_str:
                descriptors_found.add(desc)
    assert descriptors_found == set(descriptors)

    for child in children:
        child_traj = read_trajectory(child.path)
        assert atif_validate(child_traj) is True
        assert child_traj["session_id"] == parent_traj["session_id"]

    # The 3 sibling files land in the per-run trajectories/ subdir.
    traj_dir = tmp_path / ".daydream" / "runs" / recorder.session_id / "trajectories"
    sibling_files = sorted(traj_dir.iterdir())
    assert len(sibling_files) == 3
