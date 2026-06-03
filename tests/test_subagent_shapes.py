"""TEST-07: Subagent trajectory shape validation.

Uses MockBackend to drive the recorder's fork() path across the parallel
fan-out phases it must support: deep-mode per-stack reviews and exploration
pre_scan specialists. Validates the resulting root + sibling trajectory file
sets against the vendored ATIF validator.

Per D-03, these tests exercise the real recorder code with fake backends.
No pre-recorded fixture files.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from daydream.atif import validate as atif_validate
from daydream.backends import (
    AgentEvent,
    ContinuationToken,
    MetricsEvent,
    ResultEvent,
    TextEvent,
)
from daydream.trajectory import (
    DaydreamPhase,
    DaydreamRunFlow,
    Invocation,
    TrajectoryRecorder,
)


@dataclass
class MockBackend:
    """Minimal Backend replaying a canned event list."""

    model = "mock-model"
    events: list[AgentEvent]

    def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: dict[str, Any] | None = None,
        continuation: ContinuationToken | None = None,
        agents: dict[str, Any] | None = None,
        max_turns: int | None = None,
        read_only: bool = False,
    ) -> AsyncIterator[AgentEvent]:
        events = self.events

        async def _gen() -> AsyncIterator[AgentEvent]:
            for event in events:
                yield event

        return _gen()

    async def cancel(self) -> None:
        return None

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
        return f"/{skill_key}"


def _make_recorder(
    tmp_path: Path,
    *,
    run_flow: DaydreamRunFlow = DaydreamRunFlow.NORMAL,
    agent_model_name: str = "opus",
) -> TrajectoryRecorder:
    """Construct a TrajectoryRecorder rooted in tmp_path."""
    return TrajectoryRecorder(
        path=tmp_path / ".daydream" / "trajectory.json",
        run_flow=run_flow,
        target_dir=tmp_path,
        agent_model_name=agent_model_name,
        session_id="test",
    )


def _read_trajectory(path: Path) -> dict[str, Any]:
    """Load the produced trajectory JSON from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def _observe_text_and_result(inv: Invocation, text: str = "output") -> None:
    """Observe a TextEvent + ResultEvent to produce a minimal agent step."""
    inv.observe(TextEvent(text=text))
    inv.observe(ResultEvent(structured_output=None, continuation=None))


async def test_deep_mode_produces_per_stack_siblings(tmp_path: Path) -> None:
    """Deep-mode fork: 2 per-stack children with run_flow=DEEP."""
    recorder = _make_recorder(tmp_path, run_flow=DaydreamRunFlow.DEEP)
    children: list[TrajectoryRecorder] = []

    async with recorder:
        for desc in ("deep-python", "deep-typescript"):
            async with recorder.fork(desc) as child:
                children.append(child)
                async with child.invocation(phase=DaydreamPhase.DEEP) as inv:
                    _observe_text_and_result(inv, f"{desc}-output")
        recorder.create_dispatch_step(phase=DaydreamPhase.DEEP)

    parent_traj = _read_trajectory(recorder.path)

    # Dispatch step has 2 refs
    dispatch_steps = [
        s for s in parent_traj["steps"]
        if s["source"] == "agent" and "Dispatching" in s.get("message", "")
    ]
    assert len(dispatch_steps) == 1
    results = dispatch_steps[0]["observation"]["results"]
    assert len(results) == 2
    ref_paths = [r["subagent_trajectory_ref"][0]["trajectory_path"] for r in results]
    assert any("deep-python" in p for p in ref_paths)
    assert any("deep-typescript" in p for p in ref_paths)

    assert atif_validate(parent_traj) is True

    # Both children valid and inherit run_flow "deep"
    for child in children:
        child_traj = _read_trajectory(child.path)
        assert atif_validate(child_traj) is True
        agent_steps = [s for s in child_traj["steps"] if s["source"] == "agent"]
        assert len(agent_steps) >= 1
        assert agent_steps[0]["extra"]["daydream_run_flow"] == "deep"


async def test_exploration_produces_per_specialist_siblings(tmp_path: Path) -> None:
    """Exploration fork: 3 specialist children produce 3 valid sibling files.

    Also covers what the deleted fix-parallel test uniquely asserted: sibling
    trajectory files land in the per-run trajectories/ subdir, and each child
    inherits the parent's session_id.
    """
    recorder = _make_recorder(tmp_path)
    children: list[TrajectoryRecorder] = []
    descriptors = ("explore-pattern-scanner", "explore-dependency-tracer", "explore-test-mapper")

    async with recorder:
        for desc in descriptors:
            async with recorder.fork(desc) as child:
                children.append(child)
                async with child.invocation(phase=DaydreamPhase.EXPLORATION) as inv:
                    _observe_text_and_result(inv, f"{desc}-output")
        recorder.create_dispatch_step(phase=DaydreamPhase.EXPLORATION)

    parent_traj = _read_trajectory(recorder.path)
    assert atif_validate(parent_traj) is True

    # Dispatch step has 3 refs
    dispatch_steps = [
        s for s in parent_traj["steps"]
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

    # All 3 children pass validation and inherit the parent's session_id.
    for child in children:
        child_traj = _read_trajectory(child.path)
        assert atif_validate(child_traj) is True
        assert child_traj["session_id"] == parent_traj["session_id"]

    # The 3 sibling files land in the per-run trajectories/ subdir.
    traj_dir = tmp_path / ".daydream" / "runs" / recorder.session_id / "trajectories"
    sibling_files = sorted(traj_dir.iterdir())
    assert len(sibling_files) == 3


async def test_step_id_isolation_across_concurrent_siblings(tmp_path: Path) -> None:
    """SUBA-08: Concurrent siblings have independent step_id sequences starting at 1."""
    recorder = _make_recorder(tmp_path)
    children: list[TrajectoryRecorder] = []

    async with recorder:
        # Parent gets a step first
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            _observe_text_and_result(inv, "parent-step-1")

        # Fork 2 children, each producing 3 steps
        for desc in ("child-a", "child-b"):
            async with recorder.fork(desc) as child:
                children.append(child)
                for j in range(3):
                    async with child.invocation(phase=DaydreamPhase.FIX) as inv:
                        _observe_text_and_result(inv, f"{desc}-step-{j}")

        recorder.create_dispatch_step(phase=DaydreamPhase.FIX)

    parent_traj = _read_trajectory(recorder.path)
    parent_step_ids = [s["step_id"] for s in parent_traj["steps"]]
    assert parent_step_ids == list(range(1, len(parent_step_ids) + 1))

    # Each child has step_ids starting at 1
    all_child_ids: list[list[int]] = []
    for child in children:
        child_traj = _read_trajectory(child.path)
        child_ids = [s["step_id"] for s in child_traj["steps"]]
        assert child_ids[0] == 1
        assert child_ids == list(range(1, len(child_ids) + 1))
        all_child_ids.append(child_ids)

    # Both children independently number from 1 (they overlap, which is correct
    # since they're in separate files)
    assert all_child_ids[0][0] == 1
    assert all_child_ids[1][0] == 1


async def test_parent_final_metrics_excludes_sibling_steps(tmp_path: Path) -> None:
    """SUBA-09: Parent FinalMetrics.total_prompt_tokens excludes child contributions."""
    recorder = _make_recorder(tmp_path)

    async with recorder:
        # Parent invocation with known metrics
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="parent-text"))
            inv.observe(MetricsEvent(
                message_id="m-parent",
                prompt_tokens=100,
                completion_tokens=10,
                cached_tokens=5,
                cost_usd=0.001,
            ))
            inv.observe(ResultEvent(structured_output=None, continuation=None))

        # Child with much larger metrics
        async with recorder.fork("fix-0") as child:
            async with child.invocation(phase=DaydreamPhase.FIX) as inv:
                inv.observe(TextEvent(text="child-text"))
                inv.observe(MetricsEvent(
                    message_id="m-child",
                    prompt_tokens=500,
                    completion_tokens=50,
                    cached_tokens=25,
                    cost_usd=0.005,
                ))
                inv.observe(ResultEvent(structured_output=None, continuation=None))

        recorder.create_dispatch_step(phase=DaydreamPhase.FIX)

    parent_traj = _read_trajectory(recorder.path)
    child_traj = _read_trajectory(child.path)

    assert parent_traj["final_metrics"]["total_prompt_tokens"] == 100
    assert child_traj["final_metrics"]["total_prompt_tokens"] == 500


async def test_continuation_appends_to_same_trajectory_no_sibling(tmp_path: Path) -> None:
    """SUBA-05: Sequential invocations (continuation) stay in one file, no siblings."""
    recorder = _make_recorder(tmp_path)

    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.FIX) as inv:
            _observe_text_and_result(inv, "first-continuation")
        async with recorder.invocation(phase=DaydreamPhase.TEST) as inv:
            _observe_text_and_result(inv, "second-continuation")

    assert recorder.path.exists()
    traj = _read_trajectory(recorder.path)

    # No sibling files
    traj_dir = tmp_path / ".daydream" / "trajectories"
    assert not traj_dir.exists() or len(list(traj_dir.iterdir())) == 0

    # All steps in one file with sequential step_ids
    step_ids = [s["step_id"] for s in traj["steps"]]
    assert step_ids == list(range(1, len(step_ids) + 1))

    # 2 invocations each producing 1 agent step = 2 steps
    # (user steps only appear when run_agent calls observe_user_step)
    assert len(step_ids) == 2
