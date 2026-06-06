"""phase_per_stack_reviews concurrency + correctness tests (D-17, D-18, D-38)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from daydream.backends import ResultEvent, TextEvent
from daydream.deep.detection import StackAssignment
from daydream.phases import phase_per_stack_reviews


class _RecordingBackend:
    """Records every execute call; verifies no `agents` kwarg was passed."""

    model = "mock-model"  # satisfies the Backend protocol's `model: str` member

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.agents_seen: list[Any] = []

    async def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: Any = None,
        continuation: Any = None,
        agents: Any = None,
        max_turns: Any = None,
        read_only: bool = False,
    ):
        self.prompts.append(prompt)
        self.agents_seen.append(agents)
        # Emit minimal events to satisfy run_agent.
        yield TextEvent(text="done")
        yield ResultEvent(structured_output=None, continuation=None)

    async def cancel(self) -> None:
        pass

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
        return f"/{skill_key}"


def _mk_stacks() -> list[StackAssignment]:
    return [
        StackAssignment(
            stack_name="python",
            skill_invocation="/beagle-python:review-python",
            files=["api.py"],
            is_docs_only=False,
        ),
        StackAssignment(
            stack_name="react",
            skill_invocation="/beagle-react:review-frontend",
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
    """D-17: each stack gets exactly one backend.execute call."""
    backend = _RecordingBackend()
    diff, intent, alts = _mk_context_files(tmp_path)

    results, failures = await phase_per_stack_reviews(
        backend,
        make_work(tmp_path),
        _mk_stacks(),
        diff_path=diff,
        intent_path=intent,
        alternatives_path=alts,
    )

    assert set(results.keys()) == {"python", "react", "generic"}
    assert failures == {}
    assert len(backend.prompts) == 3


async def test_fan_out_never_passes_agents_kwarg(tmp_path: Path, make_work) -> None:
    """D-38 (Codex parity): the `agents` kwarg to backend.execute must be None."""
    backend = _RecordingBackend()
    diff, intent, alts = _mk_context_files(tmp_path)

    await phase_per_stack_reviews(
        backend,
        make_work(tmp_path),
        _mk_stacks(),
        diff_path=diff,
        intent_path=intent,
        alternatives_path=alts,
    )

    assert all(a is None for a in backend.agents_seen)


async def test_fan_out_unique_output_paths(tmp_path: Path, make_work) -> None:
    """D-18: per-stack output paths are unique and deterministic."""
    backend = _RecordingBackend()
    diff, intent, alts = _mk_context_files(tmp_path)

    results, _ = await phase_per_stack_reviews(
        backend,
        make_work(tmp_path),
        _mk_stacks(),
        diff_path=diff,
        intent_path=intent,
        alternatives_path=alts,
    )

    paths = set(results.values())
    assert len(paths) == 3
    for p in paths:
        assert p.name.startswith("stack-") and p.name.endswith("-review.md")


async def test_fan_out_closure_capture(tmp_path: Path, make_work) -> None:
    """Pitfall 2: no late-binding bug -- each task gets its own prompt."""
    backend = _RecordingBackend()
    diff, intent, alts = _mk_context_files(tmp_path)

    await phase_per_stack_reviews(
        backend,
        make_work(tmp_path),
        _mk_stacks(),
        diff_path=diff,
        intent_path=intent,
        alternatives_path=alts,
    )

    prompts = backend.prompts
    assert any("python" in p for p in prompts)
    assert any("react" in p for p in prompts)
    assert any("generic-fallback" in p for p in prompts)


async def test_phase_per_stack_reviews_uses_structural_prompt_for_structure_stack(
    tmp_path: Path, make_work, monkeypatch
) -> None:
    """Structural stack flows through build_structural_prompt; language stacks do not."""
    from daydream.config import STRUCTURE_SKILL, STRUCTURE_STACK_NAME
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

    backend = _RecordingBackend()
    diff, intent, alts = _mk_context_files(tmp_path)

    stacks = [
        StackAssignment(
            stack_name="python",
            skill_invocation="/beagle-python:review-python",
            files=["a.py"],
            is_docs_only=False,
        ),
        StackAssignment(
            stack_name=STRUCTURE_STACK_NAME,
            skill_invocation=STRUCTURE_SKILL,
            files=["a.py"],
            is_docs_only=False,
        ),
    ]

    await _phase(
        backend,
        make_work(tmp_path),
        stacks,
        diff_path=diff,
        intent_path=intent,
        alternatives_path=alts,
    )

    assert len(structural_calls) == 1
    assert len(per_stack_calls) == 1
    assert structural_calls[0]["files"] == ["a.py"]
    assert "skill_invocation" not in structural_calls[0]
    assert "stack_name" not in structural_calls[0]
    assert per_stack_calls[0]["stack_name"] == "python"


async def test_fan_out_continues_after_one_failure(tmp_path: Path, make_work) -> None:
    """A single stack failure does not abort the whole fan-out, and is reported."""

    class _FlakyBackend(_RecordingBackend):
        async def execute(
            self,
            cwd: Path,
            prompt: str,
            output_schema: Any = None,
            continuation: Any = None,
            agents: Any = None,
            max_turns: Any = None,
            read_only: bool = False,
        ):
            self.prompts.append(prompt)
            self.agents_seen.append(agents)
            if "react" in prompt.lower():
                raise RuntimeError("simulated react failure")
            yield TextEvent(text="ok")
            yield ResultEvent(structured_output=None, continuation=None)

    backend = _FlakyBackend()
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
