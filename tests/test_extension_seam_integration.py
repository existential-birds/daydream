"""Real-path tests: fork flow mutations reach the registered flow definitions.

Drives the production entrypoints (``runner.run_feedback`` / ``runner.run``)
against a real temp git repo, mocking ONLY the backend seam
(``daydream.runner.create_backend``) per the testing standard. A
``daydream_ext`` package written by the ``ext_dir`` fixture mutates the flow
definitions (remove/insert steps); assertions are on the prompts the backend
actually received and the exit code. Grows across Tasks 9-15 of the
extension-seam plan, one flow migration at a time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from daydream import runner
from daydream.backends import ResultEvent, TextEvent
from daydream.runner import RunConfig
from tests.conftest import ExtDir
from tests.harness.phase_backend import PhaseDispatchBackend


class RecordingBackend:
    """Prompt-recording stub modelled on ``_PRFeedbackStubBackend``.

    Dispatches on prompt content just enough to drive the pr-feedback flow
    past every gate: writes the review-output file for the fetch prompt,
    yields ONE parseable feedback item for the parse prompt, and no-ops
    everything else (fix, commit, respond).
    """

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
        pl = prompt.lower()

        if "fetch-pr-feedback" in pl:
            (cwd / ".review-output.md").write_text(
                "# PR Feedback\n\n"
                "## x[bot]\n\n"
                "1. [api.py:1] `hello()` returns 'universe' but the docstring "
                "says 'world' — align the return value.\n"
            )
            yield TextEvent(text="")
            yield ResultEvent(structured_output=None, continuation=None)
            return

        if "extract only actionable issues" in pl:
            yield TextEvent(text="")
            yield ResultEvent(
                structured_output={
                    "issues": [
                        {
                            "id": 1,
                            "description": "Align hello() return value with docstring",
                            "file": "api.py",
                            "line": 1,
                            "confidence": "HIGH",
                            "rationale": "return value diverges from docstring",
                            "evidence": "api.py:1",
                        }
                    ]
                },
                continuation=None,
            )
            return

        yield TextEvent(text="")
        yield ResultEvent(structured_output=None, continuation=None)

    async def cancel(self) -> None:
        pass

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
        # Mirror ClaudeBackend: append args so the test can read the slot from the prompt.
        result = f"/{skill_key}"
        if args:
            result = f"{result} {args}"
        return result


async def test_fork_disables_respond_step(
    ext_dir: ExtDir, multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A daydream_ext removal of ``respond-feedback`` skips only that step.

    Observable outcomes: exit 0, the flow still ran (fetch prompt reached the
    backend), and the removed respond step never invoked its skill.
    """
    ext_dir.write_module(
        "DAYDREAM_EXT_API = 1\n"
        "def register(r):\n"
        "    r.remove('pr-feedback', 'respond-feedback')\n"
    )
    backend = RecordingBackend()
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None: backend)

    rc = await runner.run_feedback(
        RunConfig(target=str(multi_stack_target), bot="x[bot]", non_interactive=True), pr=1
    )

    assert rc == 0
    assert any("fetch" in p.lower() for p in backend.prompts)  # flow still ran
    assert not any("respond-pr-feedback" in p for p in backend.prompts)  # removed step never invoked


async def test_fork_inserts_custom_phase_into_review_flow(
    ext_dir: ExtDir, multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A daydream_ext phase inserted after ``alternatives`` runs in ``--review``.

    Observable outcomes: exit 0 and the custom phase's prompt reached the
    backend through the registered ``review`` flow.
    """
    ext_dir.write_module(
        "from daydream.extensions import FlowStep\n"
        "DAYDREAM_EXT_API = 1\n"
        "async def _ro(ctx):\n"
        "    from daydream.agent import run_agent\n"
        "    from daydream.trajectory import DaydreamPhase\n"
        "    await run_agent(ctx.backend_for('ro_audit'), ctx.work.repo, 'RO-AUDIT-PROMPT',\n"
        "                    phase=DaydreamPhase.REVIEW)\n"
        "def register(r):\n"
        "    r.register_phase(FlowStep(name='ro_audit', run=_ro))\n"
        "    r.insert_after('review', anchor='alternatives', step='ro_audit')\n"
    )
    backend = RecordingBackend()
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None: backend)
    monkeypatch.delenv("DAYDREAM_APP_ID", raising=False)
    monkeypatch.delenv("DAYDREAM_APP_PRIVATE_KEY", raising=False)

    rc = await runner.run(
        RunConfig(
            target=str(multi_stack_target),
            output_mode="review",
            non_interactive=True,
            archive=False,
        )
    )

    idx = [i for i, p in enumerate(backend.prompts) if p == "RO-AUDIT-PROMPT"]
    assert idx, "custom phase never reached the backend"
    assert rc == 0


class ShallowRecordingBackend(PhaseDispatchBackend):
    """The shared shallow phase-dispatch fake, plus full-prompt recording.

    ``PhaseDispatchBackend`` drives the shallow review-parse-fix-test flow
    past every gate; ``prompts`` records the exact prompt each ``execute``
    call received so the test can assert the fork phase's prompt arrived.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
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
        async for event in super().execute(
            cwd, prompt, output_schema, continuation, agents, max_turns, read_only
        ):
            yield event


async def test_fork_inserts_phase_before_summary_in_shallow(
    ext_dir: ExtDir, multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A daydream_ext phase inserted before ``summary`` runs in ``--shallow``.

    Observable outcomes: exit 0 and the custom phase's prompt reached the
    backend through the registered ``shallow`` flow (after the iterate loop
    group, before the summary step).
    """
    ext_dir.write_module(
        "from daydream.extensions import FlowStep\n"
        "DAYDREAM_EXT_API = 1\n"
        "async def _ro(ctx):\n"
        "    from daydream.agent import run_agent\n"
        "    from daydream.trajectory import DaydreamPhase\n"
        "    await run_agent(ctx.backend_for('ro_shallow'), ctx.work.repo, 'RO-SHALLOW-PROMPT',\n"
        "                    phase=DaydreamPhase.REVIEW)\n"
        "def register(r):\n"
        "    r.register_phase(FlowStep(name='ro_shallow', run=_ro))\n"
        "    r.insert_before('shallow', anchor='summary', step='ro_shallow')\n"
    )
    backend = ShallowRecordingBackend(
        parse_results=[[{"id": 1, "description": "Align hello() return value", "file": "api.py", "line": 1}]]
    )
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None: backend)
    monkeypatch.delenv("DAYDREAM_APP_ID", raising=False)
    monkeypatch.delenv("DAYDREAM_APP_PRIVATE_KEY", raising=False)

    rc = await runner.run(
        RunConfig(
            target=str(multi_stack_target),
            shallow=True,
            skill="python",
            non_interactive=True,
            cleanup=False,
            archive=False,
        )
    )

    assert "RO-SHALLOW-PROMPT" in backend.prompts, "fork phase never reached the backend"
    assert rc == 0
