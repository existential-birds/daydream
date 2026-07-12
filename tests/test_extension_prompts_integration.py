"""Real-path test: a fork prompt override reaches the backend wholesale.

Drives the production entrypoint (``runner.run``) over a real temp git repo,
mocking ONLY the backend seam (``daydream.runner.create_backend``) per the
testing standard — the same shape as ``tests/test_extension_skills_integration.py``.
A ``daydream_ext`` package written by the ``ext_dir`` fixture overrides the
``review`` prompt; assertions are on the prompts the backend actually received
and the exit code.
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
    """Prompt-recording stub modelled on the skills-integration variant.

    Dispatches on prompt content just enough to drive the shallow flow to a
    clean exit: writes the review-output file for review prompts (built-in OR
    the fork-overridden ``RO-REVIEW`` shape), returns an empty issues list for
    parse prompts, and reports passing tests.
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

        if "ro-review" in pl or "review the changes" in pl:
            (cwd / ".review-output.md").write_text("# Review\n\nNo issues found.\n")
            yield TextEvent(text="")
            yield ResultEvent(structured_output=None, continuation=None)
            return

        if "extract only actionable issues" in pl:
            yield TextEvent(text="")
            yield ResultEvent(structured_output={"issues": []}, continuation=None)
            return

        if "test suite" in pl:
            yield TextEvent(text="All tests passed")
            yield ResultEvent(structured_output=None, continuation=None)
            return

        yield TextEvent(text="")
        yield ResultEvent(structured_output=None, continuation=None)

    async def cancel(self) -> None:
        pass

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
        # Mirror ClaudeBackend: append args so the test can read the skill from the prompt.
        result = f"/{skill_key}"
        if args:
            result = f"{result} {args}"
        return result


async def test_fork_prompt_override_reaches_backend(
    ext_dir: ExtDir, feature_branch_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A daydream_ext override of the ``review`` prompt replaces the prompt wholesale.

    The kwarg assertion (``kw['skill_invocation']`` echoed back — the real
    parameter name per ``build_review_prompt``) pins that overrides receive
    the exact built-in kwargs — the wholesale-override contract.
    """
    ext_dir.write_module(
        "DAYDREAM_EXT_API = 2\n"
        "def register(r):\n"
        "    r.override_prompt('review', lambda **kw: f\"RO-REVIEW {kw['skill_invocation']}\")\n"
    )
    backend = RecordingBackend()
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None: backend)

    rc = await runner.run(
        RunConfig(
            target=str(feature_branch_repo),
            shallow=True,
            skill="python",
            non_interactive=True,
            archive=False,
        )
    )

    assert rc == 0
    review_prompts = [p for p in backend.prompts if p.startswith("RO-REVIEW")]
    assert review_prompts and "beagle-python:review-python" in review_prompts[0]
