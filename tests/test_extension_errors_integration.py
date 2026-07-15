"""Real-path tests: broken extensions fail fast with named, actionable errors.

Enters from the production entrypoint (``runner.run``) with a real temp git
repo, mocking ONLY the backend seam (``daydream.runner.create_backend``) per
the testing standard. A ``daydream_ext`` package written by the ``ext_dir``
fixture is deliberately broken (dangling flow reference / wrong API version);
assertions pin the CLI-visible outcome: exit code 1, zero agents run, and
error output naming the broken piece (Task 16 of the extension-seam plan).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from daydream import runner
from daydream.backends import ResultEvent, TextEvent
from daydream.runner import RunConfig
from tests.conftest import ExtDir


class RecordingBackend:
    """Minimal prompt-recording stub: a broken extension must keep ``prompts`` empty."""

    model = "mock-model"
    fanout_concurrency = 4

    def __init__(self) -> None:
        self.prompts: list[str] = []

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
        yield TextEvent(text="")
        yield ResultEvent(structured_output=None, continuation=None)

    async def cancel(self) -> None:
        pass

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
        result = f"/{skill_key}"
        if args:
            result = f"{result} {args}"
        return result


async def test_broken_flow_ref_fails_before_any_agent(
    ext_dir: ExtDir,
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A flow entry naming an unregistered phase aborts before any agent runs.

    ``insert_after`` accepts the dangling name at mutation time (entry names
    resolve at ``run_flow``'s pre-flight pass), so this pins the CLI-visible
    outcome through ``runner.run``: exit 1, zero agents, named broken piece.
    """
    ext_dir.write_module(
        "DAYDREAM_EXT_API = 2\n"
        "def register(r):\n"
        "    r.insert_after('deep', anchor='intent', step='ghost_phase')\n"
    )
    backend = RecordingBackend()
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None: backend)
    monkeypatch.delenv("DAYDREAM_APP_ID", raising=False)
    monkeypatch.delenv("DAYDREAM_APP_PRIVATE_KEY", raising=False)

    rc = await runner.run(RunConfig(target=str(multi_stack_target), non_interactive=True, archive=False))

    assert rc == 1
    assert backend.prompts == []                      # zero agents ran
    assert "ghost_phase" in capsys.readouterr().out   # error names the broken piece


async def test_version_mismatch_exits_1_naming_versions(
    ext_dir: ExtDir,
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A DAYDREAM_EXT_API mismatch exits 1 naming both versions, before any git work."""
    # 99 is above the ceiling.
    ext_dir.write_module("DAYDREAM_EXT_API = 99\ndef register(r): ...\n")
    backend = RecordingBackend()
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None: backend)
    monkeypatch.delenv("DAYDREAM_APP_ID", raising=False)
    monkeypatch.delenv("DAYDREAM_APP_PRIVATE_KEY", raising=False)

    rc = await runner.run(RunConfig(target=str(multi_stack_target), non_interactive=True, archive=False))

    assert rc == 1
    out = capsys.readouterr().out
    assert "99" in out and "2" in out
