"""Extension prompt overrides through runner.run with real Git and a stub backend.

ext_dir supplies daydream_ext; assertions inspect received prompts and exit status.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from daydream import runner
from daydream.backends import ResultEvent, TextEvent
from daydream.run_config import RunConfig
from tests.conftest import ExtDir
from tests.harness.backend import ScriptedBackend
from tests.harness.improve_backend import (
    ImproveStubBackend,
    improve_artifact,
    install_capable_improve_backend,
)
from tests.harness.stub_backend import force_interactive, silence


async def test_shallow_without_skill_keeps_detected_language_skill(
    feature_branch_repo: Path, make_config: Callable[..., RunConfig], install_backend: Callable[[object], object],
    mute_side_effects: Callable[..., None], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Shallow review without --stack retains detected Python scope and emits no skill token."""
    silence(monkeypatch)
    force_interactive(monkeypatch)
    backend = ScriptedBackend(
        events=(TextEvent(text="Intent and review complete."),
                ResultEvent(structured_output={"issues": []}, continuation=None))
    )
    install_backend(backend)
    mute_side_effects("daydream.deep.fix_steps")

    rc = await runner.run(make_config(feature_branch_repo, output_mode="loop", non_interactive=False, shallow=True))

    assert rc == 0
    per_stack = [p for p in backend.prompts if "python" in p]
    assert per_stack, ("shallow with no --skill must keep the detected python scope "
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
async def test_legacy_markdown_plan_writer_override_blocks_with_sanitized_diagnostics(
    ext_dir: ExtDir, improve_monorepo_target: Path, make_config: Callable[..., RunConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raising override or a legacy markdown-blob payload blocks with sanitized diagnostics.

    Under the valid-by-construction contract the stray ``markdown`` key is
    host-stripped, never rejected: the block is diagnosed as authoring issues
    (``AUTHOR_SCHEMA_INVALID`` at the missing sections), with no
    ``LEGACY_MARKDOWN_OUTPUT`` code and no pointer at ``/markdown``.
    """
    ext_dir.write_module(_plan_writer_override(raises_on_first_call=True))
    backend = ImproveStubBackend(improve_monorepo_target, n_findings=2)
    backend.return_legacy_plan = True
    install_capable_improve_backend(monkeypatch, backend)

    rc = await runner.run(make_config(improve_monorepo_target, flow_name="improve"))

    plans_dir = improve_monorepo_target / "daydream_plans"
    diagnostics = improve_artifact(improve_monorepo_target, "plan-write-diagnostics.json",).read_text(encoding="utf-8")
    index = (plans_dir / "README.md").read_text(encoding="utf-8")
    assert rc == 1
    assert not list(plans_dir.glob("[0-9][0-9][0-9]-*.md"))
    assert "BLOCKED (PLAN_VALIDATION_FAILED: " in index
    assert "AUTHOR_SCHEMA_INVALID" in index
    assert "LEGACY_MARKDOWN_OUTPUT" not in index
    errors = [error for attempt in json.loads(diagnostics)["attempts"] for error in attempt["errors"]]
    codes = {error["code"] for error in errors}
    pointers = {error["pointer"] for error in errors}
    assert "AUTHOR_SCHEMA_INVALID" in codes
    assert "LEGACY_MARKDOWN_OUTPUT" not in codes
    assert {"/scope", "/steps", "/done_criteria"} <= pointers
    assert "/markdown" not in pointers
    assert "Make the change." not in diagnostics
    # The override raising for the first finding blocks only that plan, without leaking its exception text.
    assert "PROMPT_CONSTRUCTION_FAILED" in codes
    assert "PRIVATE_PROMPT_EXCEPTION_SECRET" not in diagnostics + index
    assert backend.plan_writer_calls == 2
    report = improve_artifact(improve_monorepo_target, "report.md").read_text(encoding="utf-8")
    assert "Plans written: 0" in report
