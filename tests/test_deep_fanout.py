"""phase_per_stack_reviews concurrency + correctness tests (D-17, D-18, D-38)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anyio

from daydream.backends import ResultEvent, TextEvent
from daydream.deep.detection import StackAssignment
from daydream.phases import phase_per_stack_reviews
from tests.harness.backend import ScriptedBackend, Turn

# The minimal turn a per-stack review agent has to emit to satisfy run_agent.
# Issue #745 (AC4): the reviewer emits PER_STACK_RECORD_SCHEMA structured
# output directly (issues + verdicts) -- there is no separate parse stage.
_REVIEW_TURN: Turn = [
    TextEvent(text="done"),
    ResultEvent(structured_output={"issues": [], "verdicts": []}, continuation=None),
]


def _review_backend(**attrs: Any) -> ScriptedBackend:
    return ScriptedBackend(events=_REVIEW_TURN, model="mock-model", **attrs)


def _mk_stacks() -> list[StackAssignment]:
    return [
        StackAssignment(
            stack_name="python",
            skill_invocation="beagle-python:review-python",
            files=["api.py"],
            is_docs_only=False,
        ),
        StackAssignment(
            stack_name="react",
            skill_invocation="beagle-react:review-frontend",
            files=["App.tsx"],
            is_docs_only=False,
        ),
        StackAssignment(
            stack_name="generic",
            skill_invocation=None,
            files=["README.md"],
            is_docs_only=True,
        ),
    ]


def _mk_context_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    diff = tmp_path / "diff.patch"
    diff.write_text("")
    intent = tmp_path / "intent.md"
    intent.write_text("x")
    alts = tmp_path / "alts.json"
    alts.write_text("[]")
    return diff, intent, alts


async def test_fan_out_invokes_each_stack(tmp_path: Path, make_work) -> None:
    """D-17/D-18/D-38: fan-out preserves per-stack calls, paths, prompts, and isolation."""
    backend = _review_backend()
    diff, intent, alts = _mk_context_files(tmp_path)

    results, failures = await phase_per_stack_reviews(
        backend,  # type: ignore[arg-type]
        make_work(tmp_path),
        _mk_stacks(),
        diff_path=diff,
        intent_path=intent,
        alternatives_path=alts,
    )

    assert set(results.keys()) == {"python", "react", "generic"}
    assert failures == {}
    assert len(backend.prompts) == 3
    assert all(c["agents"] is None for c in backend.calls)
    paths = set(results.values())
    assert len(paths) == 3
    for p in paths:
        assert p.name.startswith("stack-") and p.name.endswith("-review.md")

    # Issue #745 (AC4): the reviewer emits PER_STACK_RECORD_SCHEMA structured
    # output directly and the fan-out persists each stack's records to
    # ``stack-<name>-records.json`` -- the on-disk input ``_step_per_stack_parse``
    # / merge consume. A regression that stops persisting records.json would
    # silently break merge while these md-path assertions still pass, so assert
    # the records artifact exists and carries the declared issues/verdicts.
    from daydream.deep.artifacts import deep_dir as _deep_dir
    from daydream.deep.artifacts import per_stack_records_path

    deep_dir_path = _deep_dir(tmp_path)
    declared: dict[str, list[Any]] = {"issues": [], "verdicts": []}
    for name in results:
        records = per_stack_records_path(deep_dir_path, name)
        assert records.is_file(), f"missing {records.name} for {name}"
        assert json.loads(records.read_text()) == declared
    prompts = backend.prompts
    assert any("python" in p for p in prompts)
    assert any("react" in p for p in prompts)
    assert any("generic-fallback" in p for p in prompts)


async def test_phase_per_stack_reviews_uses_structural_prompt_for_structure_stack(
    tmp_path: Path, make_work, monkeypatch
) -> None:
    """Structural stack flows through build_structural_prompt; language stacks do not."""
    from daydream.config import STRUCTURE_STACK_NAME
    from daydream.deep import prompts as _prompts
    from daydream.phases import phase_per_stack_reviews as _phase

    structural_calls: list[dict[str, Any]] = []
    per_stack_calls: list[dict[str, Any]] = []

    def _capture_structural(**kwargs: Any) -> str:
        structural_calls.append(kwargs)
        return "STRUCTURAL_PROMPT"

    def _capture_per_stack(**kwargs: Any) -> str:
        per_stack_calls.append(kwargs)
        return "PER_STACK_PROMPT"

    # phase_per_stack_reviews late-imports these symbols from daydream.deep.prompts;
    # patch on the module so the late-import re-binding picks up the stubs.
    monkeypatch.setattr(_prompts, "build_structural_prompt", _capture_structural)
    monkeypatch.setattr(_prompts, "build_per_stack_prompt", _capture_per_stack)

    backend = _review_backend()
    diff, intent, alts = _mk_context_files(tmp_path)

    stacks = [
        StackAssignment(
            stack_name="python",
            skill_invocation="beagle-python:review-python",
            files=["a.py"],
            is_docs_only=False,
        ),
        StackAssignment(
            stack_name=STRUCTURE_STACK_NAME,
            skill_invocation=None,
            files=["a.py"],
            is_docs_only=False,
        ),
    ]

    await _phase(
        backend,  # type: ignore[arg-type]
        make_work(tmp_path),
        stacks,
        diff_path=diff,
        intent_path=intent,
        alternatives_path=alts,
    )

    assert len(structural_calls) == 1
    assert len(per_stack_calls) == 1
    assert structural_calls[0]["files"] == ["a.py"]
    assert "skill_invocation" not in structural_calls[0]  # skill-free (M2)
    assert structural_calls[0]["strategy"]  # profile-owned structural strategy (M4)
    assert "stack_name" not in structural_calls[0]
    assert per_stack_calls[0]["stack_name"] == "python"


async def test_fan_out_continues_after_one_failure(tmp_path: Path, make_work) -> None:
    """A single stack failure does not abort the whole fan-out, and is reported."""

    # Prompt-conditional, so it stays a dispatch fake: ScriptedBackend scripts by
    # call index and the fan-out completion order is not fixed.
    class _FlakyBackend(ScriptedBackend):
        async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any):
            if "react" in prompt.lower():
                raise RuntimeError("simulated react failure")
            async for event in super().execute(cwd, prompt, *args, **kwargs):
                yield event

    backend = _FlakyBackend(events=_REVIEW_TURN)
    diff, intent, alts = _mk_context_files(tmp_path)

    results, failures = await phase_per_stack_reviews(
        backend,
        make_work(tmp_path),
        _mk_stacks(),
        diff_path=diff,
        intent_path=intent,
        alternatives_path=alts,
    )

    assert "python" in results
    assert "generic" in results
    assert "react" not in results
    # Failure surfaces in the returned failures dict with the exception reason.
    assert "react" in failures
    assert "simulated react failure" in failures["react"]


class _PiShapeBackend(ScriptedBackend):
    """Mock backend that uses PiBackend's real skill formatter."""

    def __init__(self) -> None:
        super().__init__(events=_REVIEW_TURN)
        from daydream.backends.pi import PiBackend

        self._pi = PiBackend(model="glm-5.2")

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
        return self._pi.format_skill_invocation(skill_key, args)


async def test_per_stack_prompts_are_skill_free(
    tmp_path: Path, make_work
) -> None:
    """M12: built-in stacks dispatch native per-stack prompts with no /skill: token."""
    from daydream.config import STRUCTURE_STACK_NAME

    backend = _PiShapeBackend()
    diff, intent, alts = _mk_context_files(tmp_path)

    # Built-in stacks carry no skill_invocation (M2); every dispatch is native.
    stacks = [
        StackAssignment(stack_name="python", skill_invocation=None,
            files=["api.py"], is_docs_only=False),
        StackAssignment(stack_name="react", skill_invocation=None,
            files=["App.tsx"], is_docs_only=False),
        StackAssignment(stack_name="go", skill_invocation=None,
            files=["main.go"], is_docs_only=False),
        StackAssignment(stack_name="rust", skill_invocation=None,
            files=["lib.rs"], is_docs_only=False),
        StackAssignment(stack_name="elixir", skill_invocation=None,
            files=["app.ex"], is_docs_only=False),
        StackAssignment(stack_name="generic", skill_invocation=None,
            files=["notes.txt"], is_docs_only=False),
        StackAssignment(stack_name=STRUCTURE_STACK_NAME, skill_invocation=None,
            files=["api.py", "App.tsx"], is_docs_only=False),
    ]

    results, failures = await phase_per_stack_reviews(
        backend,
        make_work(tmp_path),
        stacks,
        diff_path=diff,
        intent_path=intent,
        alternatives_path=alts,
    )

    assert failures == {}
    assert len(backend.prompts) == 7
    joined = "\n\n".join(backend.prompts)

    # No skill token or raw Beagle key may appear anywhere.
    for token in ("/skill:", "/beagle-", "beagle-", "$review-"):
        assert token not in joined, f"skill token {token} leaked into a per-stack prompt"

    # Every language stack still gets a per-stack (non-generic) prompt.
    for stack_name in ("python", "react", "go", "rust", "elixir"):
        assert stack_name in joined

    # Generic fallback stack injects no skill command at all.
    generic_prompt = next(p for p in backend.prompts if "generic-fallback" in p)
    assert "/skill:" not in generic_prompt


async def test_fanout_default_concurrency(tmp_path: Path, make_work, monkeypatch) -> None:
    """Backend without fanout_concurrency attribute → limiter defaults to 4."""
    captured: list[int] = []
    real_limiter = anyio.CapacityLimiter

    def patched_limiter(n: int) -> anyio.CapacityLimiter:
        captured.append(n)
        return real_limiter(n)

    monkeypatch.setattr(anyio, "CapacityLimiter", patched_limiter)

    # The default-limiter path is only reached when the attribute is ABSENT, so
    # drop the one ScriptedBackend always sets → getattr(..., 4) returns 4.
    backend = _review_backend()
    del backend.fanout_concurrency
    assert not hasattr(backend, "fanout_concurrency")
    diff, intent, alts = _mk_context_files(tmp_path)

    await phase_per_stack_reviews(
        backend,  # type: ignore[arg-type]
        make_work(tmp_path),
        _mk_stacks(),
        diff_path=diff,
        intent_path=intent,
        alternatives_path=alts,
    )

    assert 4 in captured


async def test_fanout_low_concurrency(tmp_path: Path, make_work, monkeypatch) -> None:
    """Backend with fanout_concurrency=2 → limiter uses 2."""
    captured: list[int] = []
    real_limiter = anyio.CapacityLimiter

    def patched_limiter(n: int) -> anyio.CapacityLimiter:
        captured.append(n)
        return real_limiter(n)

    monkeypatch.setattr(anyio, "CapacityLimiter", patched_limiter)

    backend = _review_backend(fanout_concurrency=2)
    diff, intent, alts = _mk_context_files(tmp_path)

    await phase_per_stack_reviews(
        backend,
        make_work(tmp_path),
        _mk_stacks(),
        diff_path=diff,
        intent_path=intent,
        alternatives_path=alts,
    )

    assert captured == [2]


def test_shards_carry_scope_not_skill() -> None:
    """M2: shards inherit stack name / files / frontier, never a skill field."""
    from daydream.deep import sharding
    from daydream.deep.detection import detect_stacks

    files = ["a.py", "b.py", "c.py", "d.py"]
    stacks = detect_stacks(files)
    python = next(s for s in stacks if s.stack_name == "python")
    assert python.skill_invocation is None  # built-in stack is skill-free

    shards = sharding.shard_stacks(
        [python],
        # Synthetic diff so every file has 1 changed byte.
        "index 0..1 100644\n--- a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x\n+x\n"
        "index 0..1 100644\n--- a.py\n+++ b/b.py\n@@ -1 +1 @@\n-x\n+x\n"
        "index 0..1 100644\n--- a.py\n+++ b/c.py\n@@ -1 +1 @@\n-x\n+x\n"
        "index 0..1 100644\n--- a.py\n+++ b/d.py\n@@ -1 +1 @@\n-x\n+x\n",
        max_files=2,
        max_bytes=1_000_000,
        fanout_cap=4,
        frontier_max=2,
    )
    assert len(shards) >= 2  # forced split -> shard path exercised
    for shard in shards:
        assert shard.stack_name.startswith("python")  # stack identity preserved
        assert shard.skill_invocation is None  # no skill field copied
        assert shard.files and shard.frontier_files is not None
