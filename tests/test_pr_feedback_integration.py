"""Real-path integration test for the PR-feedback flow.

Drives the production entrypoint ``runner.run_feedback(config, pr)`` through the
full five-phase PR-feedback body (fetch -> parse -> fix -> commit -> respond)
with real dependencies: a real temp git worktree (``multi_stack_target``), the
real filesystem, real git, and the real anyio event loop. The ONLY thing mocked
is the external SDK backend (via ``daydream.runner.create_backend``) — exactly
the seam used by the deep-orchestrator exemplar in ``test_deep_orchestrator.py``.

The mock backend dispatches on prompt content to behave like the skill agents
would: it writes ``.review-output.md``, returns parsed issues, applies a REAL
edit to ``api.py``, runs REAL ``git add``/``git commit``, and records the
respond-pr-feedback invocation. Assertions are on observable outcomes only:
exit code, the marker written into the real file, a new git commit carrying the
Daydream trailer, and the recorded respond invocation carrying the right pr/bot.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

from daydream.backends import ResultEvent, TextEvent
from tests.harness.git_helpers import git as _git

FIX_MARKER = "# daydream-pr-feedback-fix-applied\n"
PR_NUMBER = 4242
BOT = "coderabbitai[bot]"


class _PRFeedbackStubBackend:
    """Prompt-dispatching stub mirroring the five PR-feedback skill agents.

    Records every call so the test can assert ordering and observables.
    ``respond_calls`` captures the prompt for the respond phase so the test can
    confirm the reply path ran with the correct pr/bot.
    """

    model = "mock-model"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.respond_calls: list[str] = []

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
        self.calls.append({"cwd": cwd, "prompt": prompt})
        pl = prompt.lower()

        # Phase 1: fetch-pr-feedback -> write the bot-comment markdown that
        # names a real changed file (api.py).
        if "fetch-pr-feedback" in pl:
            (cwd / ".review-output.md").write_text(
                "# PR Feedback\n\n"
                "## coderabbitai[bot]\n\n"
                "1. [api.py:1] `hello()` returns 'universe' but the docstring "
                "says 'world' — align the return value.\n"
            )
            yield TextEvent(text="")
            yield ResultEvent(structured_output=None, continuation=None)
            return

        # Phase 5: respond-pr-feedback -> the reply observable. Checked before
        # the parse branch since both mention "feedback".
        if "respond-pr-feedback" in pl:
            self.respond_calls.append(prompt)
            yield TextEvent(text="")
            yield ResultEvent(structured_output=None, continuation=None)
            return

        # Phase 2: parse feedback -> structured issues list.
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

        # Phase 4: commit -> run REAL git in cwd. Include the two Daydream
        # trailers so the prompt's amend path is not exercised (we assert on
        # what the agent itself committed).
        if "stage all changes and commit" in pl:
            run_id = self._extract_trailer(prompt, "Daydream-Run")
            version = self._extract_trailer(prompt, "Daydream-Version")
            _git(cwd, "add", "-A")
            _git(
                cwd,
                "commit",
                "-m",
                "fix: align hello() return value with docstring\n\n"
                f"Daydream-Run: {run_id}\n"
                f"Daydream-Version: {version}\n",
            )
            yield TextEvent(text="")
            yield ResultEvent(structured_output=None, continuation=None)
            return

        # Phase 3: fix -> REAL edit appending a marker to api.py.
        if pl.startswith("fix this issue:"):
            target = cwd / "api.py"
            target.write_text(target.read_text() + FIX_MARKER)
            yield TextEvent(text="")
            yield ResultEvent(structured_output=None, continuation=None)
            return

        yield TextEvent(text="")
        yield ResultEvent(structured_output=None, continuation=None)

    @staticmethod
    def _extract_trailer(prompt: str, key: str) -> str:
        m = re.search(rf"{re.escape(key)}:\s*(\S+)", prompt)
        return m.group(1) if m else "unknown"

    async def cancel(self) -> None:
        pass

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
        # Mirror ClaudeBackend: append args so the test can read --pr/--bot from the prompt.
        result = f"/{skill_key}"
        if args:
            result = f"{result} {args}"
        return result


def _silence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Silence interactive prompts so the flow runs unattended."""
    monkeypatch.setattr("daydream.phases.prompt_user", lambda *a, **kw: "n")
    monkeypatch.setattr("daydream.runner.prompt_user", lambda *a, **kw: "n")


async def test_pr_feedback_real_path(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real-path: run_feedback drives fetch->parse->fix->commit->respond.

    Mocks ONLY the Backend. Asserts observable outcomes: exit 0, the fix marker
    landed in the real api.py, a new commit carrying the Daydream trailer
    exists, and respond-pr-feedback was invoked with the right pr/bot.
    """
    from daydream.runner import RunConfig, run_feedback

    _silence(monkeypatch)
    stub = _PRFeedbackStubBackend()
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: stub)

    head_before = _git(multi_stack_target, "rev-parse", "HEAD")

    config = RunConfig(target=str(multi_stack_target), bot=BOT, cleanup=False)
    exit_code = await run_feedback(config, PR_NUMBER)

    # Observable 1: success exit code.
    assert exit_code == 0

    # Observable 2: the fix was applied to the REAL file on disk.
    api_text = (multi_stack_target / "api.py").read_text()
    assert FIX_MARKER.strip() in api_text

    # Observable 3: a NEW commit exists carrying the Daydream-Run trailer.
    head_after = _git(multi_stack_target, "rev-parse", "HEAD")
    assert head_after != head_before
    commit_msg = _git(multi_stack_target, "log", "-1", "--format=%B")
    assert "Daydream-Run:" in commit_msg

    # Observable 4: the reply path ran with the right pr/bot.
    assert len(stub.respond_calls) == 1
    respond_prompt = stub.respond_calls[0]
    assert f"--pr {PR_NUMBER}" in respond_prompt
    assert f"--bot {BOT}" in respond_prompt
