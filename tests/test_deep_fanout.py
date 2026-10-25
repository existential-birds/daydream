"""phase_per_stack_reviews concurrency + correctness tests (D-17, D-18, D-38)."""
from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import anyio
import pytest

from daydream.backends import AgentEvent
from tests.deep_orchestrator.empty_synthesis_support import EmptyReviewBackend, empty_review_config
from tests.harness.git_helpers import commit, write_and_stage
from tests.harness.review_result import saved_coverage
from tests.harness.stub_backend import review_stage_state
from tests.harness.trajectory import (
    dispatch_descriptors as _dispatch_descriptors,
    dispatch_encloses_children as _dispatch_encloses_children,
    read_trajectory,
)


def _deep_dispatch(trajectory: dict[str, Any]) -> dict[str, Any]:
    steps = [step
        for step in trajectory["steps"]
        if step.get("llm_call_count") == 0
        and step.get("extra", {}).get("daydream_phase") == "deep"
        and "dispatch_id" in step.get("extra", {})
    ]
    assert len(steps) == 1
    step = steps[0]
    assert isinstance(step, dict)
    return step

@pytest.mark.parametrize(
    ("fanout_concurrency", "expected", "automatic_shards"),
    [(None, 4, False), (2, 2, False), (2, 2, True), (100, 10, True)],
    ids=["default_concurrency", "low_concurrency", "automatic_shards", "automatic_shards_host_ceiling"],
)
async def test_fanout_concurrency_limiter(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    fanout_concurrency: int | None, expected: int, automatic_shards: bool,
) -> None:
    """The real runner enforces the backend's ceiling while completing every scope."""
    from daydream.runner import run

    if automatic_shards:
        for index in range(55 if fanout_concurrency == 100 else 12):
            write_and_stage(multi_stack_target, f"module_{index:02}.py", f"VALUE = {index}\n")
    write_and_stage(multi_stack_target, "main.go", "package main\n\nfunc main() {}\n")
    commit(multi_stack_target, "add Go review scope")
    ready, release = anyio.Event(), anyio.Event()

    class BlockingReviewBackend(EmptyReviewBackend):
        active = 0
        peak = 0
        started_scopes: set[str]

        def __init__(self, repo: Path) -> None:
            super().__init__(repo)
            self.started_scopes = set()

        async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
            reviewing = "Host review stage:\n" in prompt
            try:
                if reviewing:
                    stage = review_stage_state(prompt)
                    assert stage is not None
                    self.started_scopes.add(stage["scope_id"])
                    self.active += 1
                    self.peak = max(self.peak, self.active)
                    if not automatic_shards or stage["scope_id"].startswith("python#"):
                        if self.active >= expected:
                            ready.set()
                        await release.wait()
                async for event in super().execute(cwd, prompt, *args, **kwargs):
                    yield event
            finally:
                if reviewing:
                    self.active -= 1

    backend = BlockingReviewBackend(multi_stack_target)
    if fanout_concurrency is None:
        del backend.fanout_concurrency
    else:
        backend.fanout_concurrency = fanout_concurrency
    monkeypatch.setattr("daydream.runner.create_backend", lambda *_args, **_kwargs: backend)
    async def drive() -> None:
        assert await run(empty_review_config(multi_stack_target, tmp_path / "trajectory.json")) == 0

    # This is a concurrency assertion, not a pipeline latency benchmark.
    # Real source preparation/publication must fit under full-suite contention.
    with anyio.fail_after(60):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(drive)
            await ready.wait()
            await anyio.wait_all_tasks_blocked()
            assert backend.active == expected
            if automatic_shards:
                assert len([scope for scope in backend.started_scopes if scope.startswith("python#")]) == expected
            release.set()

    assert backend.peak == expected and backend.active == 0
    coverage = saved_coverage(multi_stack_target / ".daydream/deep")
    if automatic_shards:
        shard_count = 12 if fanout_concurrency == 100 else 3
        scopes = ["structure", "go", "react", "generic", *[f"python#{index}" for index in range(shard_count)]]
    else:
        scopes = ["go", "python", "react", "generic", "structure"]
    assert set(coverage.scopes) == set(scopes)
    assert backend.started_scopes == set(coverage.scopes)
    assert all(scope["status"] == "complete" for scope in coverage.scopes.values())
    step = _deep_dispatch(read_trajectory(tmp_path / "trajectory.json"))
    assert _dispatch_descriptors(step) == [f"deep-{scope}" for scope in scopes]
    assert _dispatch_encloses_children(step, multi_stack_target)
    assert step["extra"]["dispatch_status"] == "succeeded"
    assert step["extra"]["planned_count"] == len(scopes)
    assert step["extra"]["attempted_count"] == len(scopes)
    assert step["extra"]["completed_count"] == len(scopes)
