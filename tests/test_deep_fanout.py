"""phase_per_stack_reviews concurrency + correctness tests (D-17, D-18, D-38)."""
from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import anyio
import pytest

from daydream.backends import AgentEvent, Backend, ResultEvent
from daydream.deep import sharding
from daydream.deep.detection import StackAssignment, detect_stacks
from daydream.hunk_index import write_hunk_index
from daydream.workspace import WorkContext
from tests.deep_orchestrator.empty_synthesis_support import EmptyReviewBackend, empty_review_config
from tests.harness.backend import ScriptedBackend
from tests.harness.git_helpers import commit, git, seed_feature_branch, write_and_stage
from tests.harness.review_result import review_scopes, saved_coverage
from tests.harness.stub_backend import review_stage_result
from tests.harness.trajectory import (
    dispatch_descriptors as _dispatch_descriptors,
    dispatch_encloses_children as _dispatch_encloses_children,
    make_recorder,
    read_trajectory,
)


def _review_backend(**attrs: Any) -> ScriptedBackend:
    def respond(cwd: Any, prompt: str, *args: Any) -> list[Any]:
        return [ResultEvent(structured_output=review_stage_result(prompt, []), continuation=None)]
    return ScriptedBackend(responder=respond, model="mock-model", **attrs)

def _mk_stacks() -> list[StackAssignment]:
    return [StackAssignment(stack_name="python", files=["api.py"], is_docs_only=False,),
        StackAssignment(stack_name="react", files=["App.tsx"], is_docs_only=False,),
        StackAssignment(stack_name="generic", files=["README.md"], is_docs_only=True,),
    ]

def _mk_context_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    diff = tmp_path / "diff.patch"
    sources = {name: 'VALUE = 1\n' for name in
               ('api.py', 'a.py', 'App.tsx', 'README.md', 'main.go', 'lib.rs', 'app.ex', 'notes.txt')}
    seed_feature_branch(tmp_path, base={name: 'VALUE = 0\n' for name in sources}, feature=sources)
    diff.write_text(''.join(f'diff --git a/{name} b/{name}\n--- a/{name}\n+++ b/{name}\n'
                            '@@ -1 +1 @@\n-VALUE = 0\n+VALUE = 1\n' for name in sources))
    write_hunk_index(tmp_path, diff.read_text())
    intent = tmp_path / "intent.md"
    intent.write_text("x")
    alts = tmp_path / "alts.json"
    alts.write_text("[]")
    return diff, intent, alts


async def _run_per_stack(
    tmp_path: Path,
    make_work: Callable[..., WorkContext],
    backend: Backend,
    stacks: list[StackAssignment],
) -> tuple[dict[str, Path], dict[str, str]]:
    diff, intent, alts = _mk_context_files(tmp_path)
    results, failures = await review_scopes(
        backend,
        make_work(tmp_path, base_sha=git(tmp_path, "rev-parse", "main"),
                  head_sha=git(tmp_path, "rev-parse", "HEAD"), head_branch="feature"),
        stacks,
        diff_path=diff,
        intent_path=intent,
        alternatives_path=alts,
        allow_standalone=True,
    )
    return results, failures


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

async def test_phase_per_stack_reviews_dispatch_interval_success(tmp_path: Path, make_work: Callable[..., WorkContext],
) -> None:
    """Successful per-stack reviews retain declared order and enclosure."""
    recorder = make_recorder(tmp_path)

    async with recorder:
        results, failures = await _run_per_stack(tmp_path, make_work, _review_backend(), _mk_stacks())

    assert set(results) == {"python", "react", "generic"}
    assert failures == {}
    step = _deep_dispatch(read_trajectory(recorder.path))
    assert _dispatch_descriptors(step) == ["deep-python", "deep-react", "deep-generic"]
    assert _dispatch_encloses_children(step, recorder.target_dir)
    assert step["extra"]["dispatch_status"] == "succeeded"
    assert step["extra"]["planned_count"] == 3
    assert step["extra"]["attempted_count"] == 3
    assert step["extra"]["completed_count"] == 3


@pytest.mark.parametrize(
    ("fanout_concurrency", "expected"), [(None, 4), (2, 2)], ids=["default_concurrency", "low_concurrency"],
)
async def test_fanout_concurrency_limiter(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    fanout_concurrency: int | None, expected: int,
) -> None:
    """The real runner enforces the backend's ceiling while completing every scope."""
    from daydream.runner import run

    write_and_stage(multi_stack_target, "main.go", "package main\n\nfunc main() {}\n")
    commit(multi_stack_target, "add Go review scope")
    ready, release = anyio.Event(), anyio.Event()

    class BlockingReviewBackend(EmptyReviewBackend):
        active = 0
        peak = 0

        async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
            reviewing = "Host review stage:\n" in prompt
            try:
                if reviewing:
                    self.active += 1
                    self.peak = max(self.peak, self.active)
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
            release.set()

    assert backend.peak == expected and backend.active == 0
    coverage = saved_coverage(multi_stack_target / ".daydream/deep")
    assert set(coverage.scopes) == {"python", "react", "go", "generic", "structure"}
    assert all(scope["status"] == "complete" for scope in coverage.scopes.values())

def test_shards_carry_scope_not_skill() -> None:
    """M2: shards inherit stack name / files / frontier, never a skill field."""

    files = ["a.py", "b.py", "c.py", "d.py"]
    stacks = detect_stacks(files)
    python = next(s for s in stacks if s.stack_name == "python")
    assert not hasattr(python, "skill_invocation")

    shards = sharding.shard_stacks([python],
        # Synthetic diff so every file has 1 changed byte.
        "index 0..1 100644\n--- a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+x\n"
        "index 0..1 100644\n--- a.py\n+++ b/b.py\n@@ -1 +1 @@\n-x\n+x\n"
        "index 0..1 100644\n--- a.py\n+++ b/c.py\n@@ -1 +1 @@\n-x\n+x\n"
        "index 0..1 100644\n--- a.py\n+++ b/d.py\n@@ -1 +1 @@\n-x\n+x\n",
        max_files=2, max_bytes=1_000_000, fanout_cap=4, frontier_max=2,
    )
    assert len(shards) >= 2  # forced split -> shard path exercised
    for shard in shards:
        assert shard.stack_name.startswith("python")  # stack identity preserved
        assert not hasattr(shard, "skill_invocation")
        assert shard.files and shard.frontier_files is not None
