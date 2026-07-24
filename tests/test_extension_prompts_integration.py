"""Real-path test: a fork prompt override reaches the backend wholesale.

Drives the production entrypoint (``runner.run``) over a real temp git repo,
mocking ONLY the backend seam (``daydream.runner.create_backend``) per the
testing standard — the same shape as ``tests/test_extension_skills_integration.py``.
A ``daydream_ext`` package written by the ``ext_dir`` fixture overrides the
``review`` prompt; assertions are on the prompts the backend actually received
and the exit code.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from daydream import runner
from daydream.backends import ResultEvent, TextEvent
from daydream.improve.prompts import PLAN_AUTHOR_SCHEMA
from daydream.runner import RunConfig
from tests.conftest import ExtDir
from tests.harness.improve_backend import ImproveStubBackend, improve_artifact


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
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: backend)

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


def _plan_writer_override(*, raises_on_first_call: bool = False) -> str:
    failure = (
        "    if _calls == 1:\n"
        "        raise RuntimeError('PRIVATE_PROMPT_EXCEPTION_SECRET')\n"
        if raises_on_first_call
        else ""
    )
    return (
        "import json\n"
        "DAYDREAM_EXT_API = 3\n"
        "_calls = 0\n"
        "def _plan_writer(*, finding, recon_summary, verification_commands, cwd):\n"
        "    global _calls\n"
        "    _calls += 1\n"
        "    joined = '\\n'.join(verification_commands)\n"
        "    assert all(isinstance(command, str) for command in verification_commands)\n"
        f"{failure}"
        "    return (\n"
        "        'You are writing a self-contained implementation plan.\\n'\n"
        "        'EXTENSION_TYPED_PLAN_WRITER\\n'\n"
        "        f'Legacy commands:\\n{joined}\\n'\n"
        "        'Selected vetted finding:\\n```json\\n'\n"
        "        + json.dumps(finding)\n"
        "        + '\\n```\\nRecon:\\n'\n"
        "        + recon_summary\n"
        "    )\n"
        "def register(r):\n"
        "    r.override_prompt('plan-writer', _plan_writer)\n"
    )


@pytest.mark.anyio
async def test_plan_writer_override_receives_legacy_string_commands_and_typed_output_succeeds(
    ext_dir: ExtDir,
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ext_dir.write_module(_plan_writer_override())
    backend = ImproveStubBackend(improve_monorepo_target, n_findings=1)
    monkeypatch.setattr(
        "daydream.runner.create_backend",
        lambda name, model=None, **kwargs: backend,
    )

    rc = await runner.run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        )
    )

    assert rc == 0
    plan_calls = [
        call
        for call in backend.calls
        if "EXTENSION_TYPED_PLAN_WRITER" in call["prompt"]
    ]
    assert plan_calls
    assert all(call["output_schema"] == PLAN_AUTHOR_SCHEMA for call in plan_calls)
    assert all("uv run pytest" in call["prompt"] for call in plan_calls)
    assert all('"id": "test-suite"' in call["prompt"] for call in plan_calls)
    assert all('"working_directory": "."' in call["prompt"] for call in plan_calls)
    assert list(
        (improve_monorepo_target / "daydream_plans").glob(
            "[0-9][0-9][0-9]-*.md"
        )
    )


@pytest.mark.anyio
async def test_legacy_markdown_plan_writer_override_blocks_with_sanitized_diagnostics(
    ext_dir: ExtDir,
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A legacy markdown-blob payload blocks on missing authored content.

    Under the valid-by-construction contract the stray ``markdown`` key is
    host-stripped, never rejected: the block is diagnosed as authoring issues
    (``AUTHOR_SCHEMA_INVALID`` at the missing sections), with no
    ``LEGACY_MARKDOWN_OUTPUT`` code and no pointer at ``/markdown``.
    """
    ext_dir.write_module(_plan_writer_override())
    backend = ImproveStubBackend(improve_monorepo_target, n_findings=1)
    backend.return_legacy_plan = True
    monkeypatch.setattr(
        "daydream.runner.create_backend",
        lambda name, model=None, **kwargs: backend,
    )

    rc = await runner.run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        )
    )

    plans_dir = improve_monorepo_target / "daydream_plans"
    diagnostics = improve_artifact(
        improve_monorepo_target,
        "plan-write-diagnostics.json",
    ).read_text(encoding="utf-8")
    index = (plans_dir / "README.md").read_text(encoding="utf-8")
    assert rc == 1
    assert not list(plans_dir.glob("[0-9][0-9][0-9]-*.md"))
    assert "BLOCKED (PLAN_VALIDATION_FAILED: " in index
    assert "AUTHOR_SCHEMA_INVALID" in index
    assert "LEGACY_MARKDOWN_OUTPUT" not in index
    errors = [
        error
        for attempt in json.loads(diagnostics)["attempts"]
        for error in attempt["errors"]
    ]
    codes = {error["code"] for error in errors}
    pointers = {error["pointer"] for error in errors}
    assert "AUTHOR_SCHEMA_INVALID" in codes
    assert "LEGACY_MARKDOWN_OUTPUT" not in codes
    assert {"/scope", "/steps", "/done_criteria"} <= pointers
    assert "/markdown" not in pointers
    assert "Make the change." not in diagnostics


@pytest.mark.anyio
async def test_plan_writer_prompt_exception_blocks_only_that_plan(
    ext_dir: ExtDir,
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ext_dir.write_module(_plan_writer_override(raises_on_first_call=True))
    backend = ImproveStubBackend(improve_monorepo_target, n_findings=2)
    monkeypatch.setattr(
        "daydream.runner.create_backend",
        lambda name, model=None, **kwargs: backend,
    )

    rc = await runner.run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        )
    )

    plans_dir = improve_monorepo_target / "daydream_plans"
    index = (plans_dir / "README.md").read_text(encoding="utf-8")
    diagnostics = improve_artifact(
        improve_monorepo_target,
        "plan-write-diagnostics.json",
    ).read_text(encoding="utf-8")
    assert rc == 0
    assert list(plans_dir.glob("[0-9][0-9][0-9]-*.md"))
    assert "BLOCKED (PLAN_WRITER_FAILED: PROMPT_CONSTRUCTION_FAILED)" in index
    assert "PROMPT_CONSTRUCTION_FAILED" in diagnostics
    assert "PRIVATE_PROMPT_EXCEPTION_SECRET" not in diagnostics
