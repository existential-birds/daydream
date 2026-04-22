"""phase_per_stack_reviews concurrency + correctness tests (D-17, D-18, D-38)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from daydream.backends import ResultEvent, TextEvent
from daydream.deep.detection import StackAssignment
from daydream.phases import phase_per_stack_reviews


class _RecordingBackend:
    """Records every execute call; verifies no `agents` kwarg was passed."""

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


async def test_fan_out_invokes_each_stack(tmp_path: Path) -> None:
    """D-17: each stack gets exactly one backend.execute call."""
    backend = _RecordingBackend()
    diff, intent, alts = _mk_context_files(tmp_path)

    results, failures = await phase_per_stack_reviews(
        backend,
        tmp_path,
        _mk_stacks(),
        diff_path=diff,
        intent_path=intent,
        alternatives_path=alts,
    )

    assert set(results.keys()) == {"python", "react", "generic"}
    assert failures == {}
    assert len(backend.prompts) == 3


async def test_fan_out_never_passes_agents_kwarg(tmp_path: Path) -> None:
    """D-38 (Codex parity): the `agents` kwarg to backend.execute must be None."""
    backend = _RecordingBackend()
    diff, intent, alts = _mk_context_files(tmp_path)

    await phase_per_stack_reviews(
        backend,
        tmp_path,
        _mk_stacks(),
        diff_path=diff,
        intent_path=intent,
        alternatives_path=alts,
    )

    assert all(a is None for a in backend.agents_seen)


async def test_fan_out_unique_output_paths(tmp_path: Path) -> None:
    """D-18: per-stack output paths are unique and deterministic."""
    backend = _RecordingBackend()
    diff, intent, alts = _mk_context_files(tmp_path)

    results, _ = await phase_per_stack_reviews(
        backend,
        tmp_path,
        _mk_stacks(),
        diff_path=diff,
        intent_path=intent,
        alternatives_path=alts,
    )

    paths = set(results.values())
    assert len(paths) == 3
    for p in paths:
        assert p.name.startswith("stack-") and p.name.endswith("-review.md")


async def test_fan_out_closure_capture(tmp_path: Path) -> None:
    """Pitfall 2: no late-binding bug -- each task gets its own prompt."""
    backend = _RecordingBackend()
    diff, intent, alts = _mk_context_files(tmp_path)

    await phase_per_stack_reviews(
        backend,
        tmp_path,
        _mk_stacks(),
        diff_path=diff,
        intent_path=intent,
        alternatives_path=alts,
    )

    prompts = backend.prompts
    assert any("python" in p for p in prompts)
    assert any("react" in p for p in prompts)
    assert any("generic-fallback" in p for p in prompts)


async def test_fan_out_continues_after_one_failure(tmp_path: Path) -> None:
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
        tmp_path,
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
