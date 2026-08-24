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
from collections.abc import Callable
from pathlib import Path

import pytest

from daydream import runner
from daydream.backends import ResultEvent, TextEvent
from daydream.improve.prompts import PLAN_AUTHOR_SCHEMA
from daydream.runner import RunConfig
from tests.conftest import ExtDir
from tests.harness.backend import ScriptedBackend
from tests.harness.improve_backend import ImproveStubBackend, improve_artifact


async def test_fork_prompt_override_reaches_backend(
    ext_dir: ExtDir,
    feature_branch_repo: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
    mute_side_effects: Callable[..., None],
) -> None:
    """A daydream_ext override of the ``per-stack`` prompt replaces it wholesale.

    The ``review`` prompt slot was deleted with the shallow flow (#330): shallow
    mode now runs the deep flow, whose per-stack reviewer resolves the
    ``per-stack`` slot. The kwarg assertion (``kw['strategy']`` echoed
    back — the real parameter name per ``build_per_stack_prompt``) pins that
    overrides receive the exact built-in kwargs — the wholesale-override contract.
    """
    ext_dir.write_module(
        "def register(r):\n"
        "    r.override_prompt('per-stack', lambda **kw: f\"RO-STACK {kw['strategy']}\")\n"
    )
    backend = ScriptedBackend(
        events=(
            TextEvent(text=""),
            ResultEvent(structured_output={"issues": []}, continuation=None),
        )
    )
    install_backend(backend)
    mute_side_effects("daydream.deep.orchestrator")

    rc = await runner.run(make_config(feature_branch_repo, shallow=True, skill="python"))

    assert rc == 0
    review_prompts = [p for p in backend.prompts if p.startswith("RO-STACK")]
    assert review_prompts  # the wholesale override replaced the per-stack builder
    # The override received the built-in strategy kwarg (the profile-owned
    # per-stack strategy, now that built-in stacks are skill-free, M2) -- the
    # wholesale-override contract.
    assert "RO-STACK " in review_prompts[0]


async def test_shallow_without_skill_keeps_detected_language_skill(
    feature_branch_repo: Path,
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
    mute_side_effects: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--shallow <repo>`` without ``--stack`` preserves the detected language scope (#6).

    The diff is `main.py` only, so ``detect_stacks`` routes it to the python
    stack. Shallow collapse must preserve that stack's scope (python) when no
    explicit ``--stack`` is given, instead of downgrading to the generic-fallback
    reviewer -- and must emit no skill token (M2/M12).
    Observable outcome: the backend receives a per-stack python prompt with no skill.
    """
    backend = ScriptedBackend(
        events=(
            TextEvent(text=""),
            ResultEvent(structured_output={"issues": []}, continuation=None),
        )
    )
    install_backend(backend)
    mute_side_effects("daydream.deep.orchestrator")

    rc = await runner.run(make_config(feature_branch_repo, shallow=True))

    assert rc == 0
    per_stack = [p for p in backend.prompts if "python" in p]
    assert per_stack, (
        "shallow with no --skill must keep the detected python scope "
        "rather than fall back to the generic-fallback reviewer"
    )
    joined = "\n".join(backend.prompts)
    assert "/beagle-" not in joined and "beagle-python" not in joined


def _plan_writer_override(*, raises_on_first_call: bool = False) -> str:
    failure = (
        "    if _calls == 1:\n"
        "        raise RuntimeError('PRIVATE_PROMPT_EXCEPTION_SECRET')\n"
        if raises_on_first_call
        else ""
    )
    return (
        "import json\n"
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
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
) -> None:
    ext_dir.write_module(_plan_writer_override())
    backend = ImproveStubBackend(improve_monorepo_target, n_findings=1)
    install_backend(backend)

    rc = await runner.run(make_config(improve_monorepo_target, flow_name="improve"))

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
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
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
    install_backend(backend)

    rc = await runner.run(make_config(improve_monorepo_target, flow_name="improve"))

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
    make_config: Callable[..., RunConfig],
    install_backend: Callable[[object], object],
) -> None:
    ext_dir.write_module(_plan_writer_override(raises_on_first_call=True))
    backend = ImproveStubBackend(improve_monorepo_target, n_findings=2)
    install_backend(backend)

    rc = await runner.run(make_config(improve_monorepo_target, flow_name="improve"))

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
