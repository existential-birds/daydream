"""Interaction isolation through real runner, extension, and agent boundaries."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from pathlib import Path
from typing import Any

import anyio
import pytest

from daydream import runner
from daydream.agent import console
from daydream.backends import AgentEvent, ResultEvent
from daydream.phases import phase_alternative_review
from daydream.run_context import (
    InteractionPolicy,
    RunContext,
    active_backends,
    bind_run_context,
    current_run_context,
)
from daydream.runner import RunConfig
from daydream.ui import prompt_user
from daydream.workspace import WorkContext
from tests.conftest import ExtDir
from tests.harness.backend import ScriptedBackend
from tests.test_runner import _feature_repo


def _write_policy_flow(ext_dir: ExtDir) -> None:
    ext_dir.write_module(
        "from daydream.agent import run_agent\n"
        "from daydream.extensions import FlowStep\n"
        "from daydream.trajectory import DaydreamPhase\n"
        "async def probe(ctx):\n"
        "    data = ctx.data\n"
        "    data['extension_marker'] = 'retained'\n"
        "    runtime = ctx.run_context\n"
        "    assert runtime is not None\n"
        "    assert ctx.artifacts is not None\n"
        "    await run_agent(ctx.backend_for('probe'), ctx.work.repo, 'PROBE',\n"
        "        phase=DaydreamPhase.REVIEW, run_context=runtime)\n"
        "    assert ctx.data is data\n"
        "    assert data['extension_marker'] == 'retained'\n"
        "    assert ctx.run_context is runtime\n"
        "def register(registry):\n"
        "    registry.register_phase(FlowStep(name='probe', run=probe))\n"
        "    registry.set_flow('policy-probe', ['probe'])\n"
    )


async def test_standalone_phase_binds_explicit_policy_for_surrounding_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_work: Callable[..., WorkContext],
) -> None:
    """The phase's own output uses its runtime before and after the agent."""
    outer = RunContext(InteractionPolicy(log_mode=False))
    explicit = RunContext(InteractionPolicy(log_mode=True))
    sentinel = "ghp_" + "x" * 16
    backend = ScriptedBackend(
        model=sentinel,
        events=[ResultEvent(structured_output={"issues": []}, continuation=None)],
    )
    diff_path = tmp_path / "diff.patch"
    diff_path.write_text("diff --git a/app.py b/app.py\n")
    observed: dict[str, RunContext | None] = {}
    original_print = console.print

    def record_output(*objects: Any, **kwargs: Any) -> None:
        for value in objects:
            if isinstance(value, str):
                if "Agent is evaluating" in value:
                    observed["before"] = current_run_context()
                if "No issues found" in value:
                    observed["after"] = current_run_context()
        original_print(*objects, **kwargs)

    monkeypatch.setattr(console, "print", record_output)
    with bind_run_context(outer), console.capture() as captured:
        assert await phase_alternative_review(
            backend, make_work(tmp_path), diff_path, "Update app",
            run_context=explicit,
        ) == []
        assert current_run_context() is outer

    assert observed == {"before": explicit, "after": explicit}
    assert sentinel not in captured.get()
    assert "REDACTED_API_KEY" in captured.get()
    assert current_run_context() is None


async def test_overlapping_runs_keep_prompt_and_console_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ext_dir: ExtDir,
    make_config: Callable[..., RunConfig],
) -> None:
    """An interactive sibling cannot enable stdin or disable log redaction."""
    logged_repo = _feature_repo(tmp_path / "logged")
    plain_repo = _feature_repo(tmp_path / "plain")
    _write_policy_flow(ext_dir)
    monkeypatch.setattr(runner, "_stdin_isatty", lambda: True)
    monkeypatch.delenv("CI", raising=False)
    input_calls: list[str] = []

    def answer() -> str:
        input_calls.append("read")
        return "interactive-answer"

    monkeypatch.setattr("builtins.input", answer)
    logged_entered = anyio.Event()
    plain_entered = anyio.Event()
    logged_observed = anyio.Event()
    release_plain = anyio.Event()
    logged_finished = anyio.Event()
    choices: dict[str, str] = {}
    contexts: dict[str, RunContext] = {}
    results: dict[str, int] = {}
    sentinel = "ghp_" + "x" * 16

    class SharedBackend(ScriptedBackend):
        async def execute(
            self, cwd: Path, prompt: str, *args: Any, **kwargs: Any,
        ) -> AsyncGenerator[AgentEvent, None]:
            label = "logged" if cwd == logged_repo else "plain"
            runtime = current_run_context()
            assert runtime is not None
            contexts[label] = runtime
            if label == "logged":
                logged_entered.set()
                await plain_entered.wait()
            else:
                plain_entered.set()
                await logged_observed.wait()
            choices[label] = prompt_user(console, f"{label} choice", "safe-default")
            console.print(f"{label}-token={sentinel}", markup=False, highlight=False)
            if label == "logged":
                logged_observed.set()
            else:
                await release_plain.wait()
            yield ResultEvent(structured_output=None, continuation=None)

    backend = SharedBackend()
    monkeypatch.setattr(runner, "create_backend", lambda *args, **kwargs: backend)
    logged_config = make_config(
        logged_repo, flow_name="policy-probe", backend="claude",
        log_mode=True, quiet=True, non_interactive=True,
    )
    plain_config = make_config(
        plain_repo, flow_name="policy-probe", backend="claude",
        log_mode=False, quiet=False, non_interactive=False, assume="yes",
    )

    async def run_logged() -> None:
        try:
            results["logged"] = await runner.run(logged_config)
        finally:
            logged_finished.set()

    async def run_plain() -> None:
        results["plain"] = await runner.run(plain_config)

    with console.capture() as captured, anyio.fail_after(30):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(run_logged)
            await logged_entered.wait()
            tasks.start_soon(run_plain)
            await logged_finished.wait()
            assert contexts["logged"].active_backends() == ()
            assert contexts["plain"].active_backends() == (backend,)
            assert active_backends() == (backend,)
            release_plain.set()

    assert results == {"logged": 0, "plain": 0}
    assert choices == {"logged": "safe-default", "plain": "interactive-answer"}
    assert input_calls == ["read"]
    output = captured.get()
    assert f"logged-token={sentinel}" not in output
    assert "logged-token=[REDACTED_API_KEY]" in output
    assert f"plain-token={sentinel}" in output
    assert contexts["logged"] is not contexts["plain"]
    assert contexts["logged"].policy.quiet is True
    assert contexts["logged"].policy.log_mode is True
    assert contexts["logged"].policy.interactive is False
    assert contexts["plain"].policy.assume == "yes"
    assert contexts["plain"].policy.log_mode is False
    assert contexts["plain"].policy.interactive is True
    assert active_backends() == ()
    assert current_run_context() is None


async def test_failed_run_releases_policy_and_backends_before_later_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    ext_dir: ExtDir,
    make_config: Callable[..., RunConfig],
) -> None:
    """A failed logged run cannot leave policy or active backends behind."""
    failed_repo = _feature_repo(tmp_path / "failed")
    later_repo = _feature_repo(tmp_path / "later")
    _write_policy_flow(ext_dir)
    monkeypatch.setattr(runner, "_stdin_isatty", lambda: True)
    monkeypatch.delenv("CI", raising=False)
    choices: list[str] = []
    sentinel = "ghp_" + "y" * 16
    monkeypatch.setattr("builtins.input", lambda: "later-answer")

    class FailingThenSuccessfulBackend(ScriptedBackend):
        async def execute(
            self, cwd: Path, prompt: str, *args: Any, **kwargs: Any,
        ) -> AsyncGenerator[AgentEvent, None]:
            if cwd == failed_repo:
                console.print(f"failed-token={sentinel}", markup=False, highlight=False)
                raise ValueError("intentional backend failure")
            choices.append(prompt_user(console, "Later choice", "safe-default"))
            console.print(f"later-token={sentinel}", markup=False, highlight=False)
            yield ResultEvent(structured_output=None, continuation=None)

    backend = FailingThenSuccessfulBackend()
    monkeypatch.setattr(runner, "create_backend", lambda *args, **kwargs: backend)
    with console.capture() as captured:
        with pytest.raises(ValueError, match="intentional backend failure"):
            await runner.run(make_config(
                failed_repo, flow_name="policy-probe", backend="claude",
                log_mode=True, non_interactive=True,
            ))
        assert current_run_context() is None
        assert active_backends() == ()
        assert await runner.run(make_config(
            later_repo, flow_name="policy-probe", backend="claude",
            log_mode=False, non_interactive=False,
        )) == 0
    assert choices == ["later-answer"]
    assert f"failed-token={sentinel}" not in captured.get()
    assert "failed-token=[REDACTED_API_KEY]" in captured.get()
    assert f"later-token={sentinel}" in captured.get()
    assert current_run_context() is None
    assert active_backends() == ()
