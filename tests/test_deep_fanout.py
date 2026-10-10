"""phase_per_stack_reviews concurrency + correctness tests (D-17, D-18, D-38)."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import anyio
import pytest

from daydream.backends import Backend, ResultEvent, TextEvent
from daydream.deep import sharding
from daydream.deep.detection import StackAssignment, detect_stacks
from daydream.workspace import WorkContext
from tests.harness.backend import ScriptedBackend, Turn
from tests.harness.review_result import review_scopes
from tests.harness.trajectory import (
    dispatch_descriptors as _dispatch_descriptors,
    dispatch_encloses_children as _dispatch_encloses_children,
    make_recorder,
    read_trajectory,
)

# The minimal turn a per-stack review agent has to emit to satisfy run_agent.
# Issue #745 (AC4): the reviewer emits PER_STACK_RECORD_SCHEMA structured
# output directly (issues) -- there is no separate parse stage.
_REVIEW_TURN: Turn = [TextEvent(text="done"), ResultEvent(structured_output={"issues": []}, continuation=None),]

def _review_backend(**attrs: Any) -> ScriptedBackend:
    return ScriptedBackend(events=_REVIEW_TURN, model="mock-model", **attrs)

def _mk_stacks() -> list[StackAssignment]:
    return [StackAssignment(stack_name="python", files=["api.py"], is_docs_only=False,),
        StackAssignment(stack_name="react", files=["App.tsx"], is_docs_only=False,),
        StackAssignment(stack_name="generic", files=["README.md"], is_docs_only=True,),
    ]

def _mk_context_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    diff = tmp_path / "diff.patch"
    diff.write_text("")
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
        make_work(tmp_path),
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
    ("fanout_concurrency", "expected"), [(None, [4]), (2, [2])], ids=["default_concurrency", "low_concurrency"],
)
async def test_fanout_concurrency_limiter(
    tmp_path: Path, make_work: Callable[..., WorkContext], monkeypatch: pytest.MonkeyPatch,
    fanout_concurrency: int | None, expected: list[int],
) -> None:
    """Backend fanout_concurrency selects the limiter width (absent → 4)."""
    captured: list[int] = []
    real_limiter = anyio.CapacityLimiter

    def patched_limiter(n: int) -> anyio.CapacityLimiter:
        captured.append(n)
        return real_limiter(n)

    monkeypatch.setattr(anyio, "CapacityLimiter", patched_limiter)

    if fanout_concurrency is None:
        # The default-limiter path is only reached when the attribute is
        # ABSENT, so drop the one ScriptedBackend always sets → 4.
        backend = _review_backend()
        del backend.fanout_concurrency
        assert not hasattr(backend, "fanout_concurrency")
    else:
        backend = _review_backend(fanout_concurrency=fanout_concurrency)
    await _run_per_stack(tmp_path, make_work, backend, _mk_stacks())

    assert captured == expected

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
