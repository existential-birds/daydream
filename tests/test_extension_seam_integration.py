"""Run fork-mutated flows through runner.run in real temporary repositories.

Only the backend seam is mocked; temporary extension packages modify registered
steps. Assert delivered prompts and exit codes."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from daydream import git_ops, pr_review, runner
from daydream.backends import ResultEvent, TextEvent
from daydream.config_file import load_file_config
from daydream.deep.merge_steps import _step_post_review
from daydream.extensions.registry import Registry
from daydream.flows.engine import FlowContext
from daydream.run_config import RunConfig
from daydream.run_context import InteractionPolicy, RunContext
from daydream.workspace import WorkContext
from tests.conftest import ExtDir
from tests.harness.backend import ScriptedBackend
from tests.harness.review_result import review_coverage

KEEP_ME = "KEEP_ME"
DROP_ME = "DROP_ME"

MakeConfig = Callable[..., RunConfig]
InstallBackend = Callable[[object], object]

_EMPTY_TURN = (TextEvent(text=""), ResultEvent(structured_output=None, continuation=None))


async def _no_post(*_args: object, **_kwargs: object) -> None:
    return None


def _post_context(*, dd: Path, items_file: Path) -> FlowContext:
    """Build the smallest context needed by the post-review step."""
    registry = Registry()
    registry.override_renderer("finding", pr_review.default_render_finding)
    registry.override_renderer("summary", pr_review.default_render_summary)
    coverage = review_coverage()
    for scope in coverage.scopes:
        coverage.record_scope(scope, "complete")
    coverage.record_phase("merge", "complete", noop=True)
    coverage.finalize("completed")
    return FlowContext(
        config=RunConfig(),
        work=WorkContext(
            repo=dd.parent, source=dd.parent, base_branch="main", base_sha="base", head_branch="feature",
            head_sha="head", is_ephemeral=False, run_id="test-run",
        ), registry=registry, data={"dd": dd, "items_file": items_file, "review_coverage": coverage},
        run_context=RunContext(InteractionPolicy()),
    )


async def test_post_review_uses_published_items_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The post step passes through the canonical path published by load-items."""
    private_dir = tmp_path / "private-deep"
    stable_items = tmp_path / "stable-merged-items.json"
    ctx = _post_context(dd=private_dir, items_file=stable_items)
    posted_paths: list[Path] = []
    async def _record_post(
        target_dir: Path, merged_items_path: Path, *, console: Any, run_info: str, renderers: pr_review.ReviewRenderers,
        post: bool, approve_on_clean: bool = False, pr_number: int | None = None, diagram_blocks: str | None = None,
        run_context: RunContext | None = None, auth: git_ops.GitHubAuth = git_ops.INHERIT_GITHUB_AUTH,
        review_summary: str | None = None, review_complete: bool = True,
    ) -> None:
        assert run_context is ctx.run_context
        assert isinstance(run_info, str)
        assert isinstance(renderers, pr_review.ReviewRenderers)
        assert review_complete is False  # Fixture has no sealed terminal result authorizing approval.
        posted_paths.append(merged_items_path)

    monkeypatch.setattr("daydream.pr_review.post_review_to_pr_from_report", _record_post)

    await _step_post_review(ctx)

    assert posted_paths == [stable_items]

async def test_bound_custom_flow_rejects_public_artifact_before_backend_entry(
    ext_dir: ExtDir, multi_stack_target: Path, install_backend: InstallBackend, make_config: MakeConfig,
) -> None:
    ext_dir.write_module(
        "from daydream.extensions import FlowStep\n"
        "async def _leak(ctx):\n"
        "    from daydream.agent import run_agent\n"
        "    from daydream.trajectory import DaydreamPhase\n"
        "    public = ctx.work.repo / '.daydream'\n"
        "    public.mkdir()\n"
        "    (public / 'leak.txt').write_text('generated')\n"
        "    await run_agent(ctx.backend_for('review'), ctx.work.repo, 'SECOND-CALL',\n"
        "                    phase=DaydreamPhase.REVIEW)\n"
        "def register(r):\n"
        "    r.register_phase(FlowStep(name='leak', run=_leak))\n"
        "    r.set_flow('leak', ['leak'])\n"
    )
    backend = ScriptedBackend(events=_EMPTY_TURN)
    install_backend(backend)

    rc = await runner.run(make_config(multi_stack_target, flow_name="leak"))

    assert rc == 1
    assert backend.call_count == 0


async def test_builtin_and_fork_tool_supervisor_conflict_fails_loud(
    ext_dir: ExtDir, multi_stack_target: Path, make_config: MakeConfig, capsys: pytest.CaptureFixture[str],
) -> None:
    """Config-enabled built-in and fork supervisors cannot silently compose."""

    ext_dir.write_module(
        "from daydream.extensions import ToolDecision\n"
        "def _fork_supervisor(name, tool_input, *, phase):\n"
        "    return ToolDecision(veto=False)\n"
        "def register(r):\n"
        "    r.register_tool_supervisor(_fork_supervisor)\n"
    )
    (multi_stack_target / ".daydream.toml").write_text('tool_supervisor = "rules"\n')

    rc = await runner.run(make_config(multi_stack_target, file_config=load_file_config(multi_stack_target)))

    assert rc == 1
    output = capsys.readouterr().out.lower()
    assert "tool supervisor conflict" in output
    assert "config-enabled built-in" in output
    assert "extension-registered" in output

async def test_pr_feedback_not_selectable_via_flow(
    multi_stack_target: Path, install_backend: InstallBackend, make_config: MakeConfig,
) -> None:
    """--flow pr-feedback errors (needs PR number + bot)."""
    install_backend(ScriptedBackend(events=_EMPTY_TURN, model="mock-model"))

    rc = await runner.run(make_config(multi_stack_target, flow_name="pr-feedback"))
    assert rc == 1
