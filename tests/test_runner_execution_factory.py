"""Execution factories exercised through real custom and Improve flows."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from daydream import git_ops, runner
from daydream.artifact_visibility import private_root_locations
from daydream.backends import AUDIT_ROOT_ISOLATION, BackendExecutionInput
from daydream.config_file import DaydreamFileConfig
from daydream.github_app import GitHubExecutionInput
from daydream.run_config import RunConfig
from tests.conftest import ExtDir
from tests.harness.backend import ScriptedBackend
from tests.harness.improve_backend import ImproveStubBackend, improve_artifact


@pytest.mark.parametrize(("flow", "injected"), [("factory-probe", True), ("improve", True), ("factory-probe", False)])
async def test_runner_factory_reaches_real_flows_and_preserves_ordinary_fallback(
    flow: str, injected: bool, improve_monorepo_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ext_dir: ExtDir,
) -> None:
    repo = improve_monorepo_target
    login = "execution-owner" if injected else "ambient-owner"
    ext_dir.write_module(
        "from daydream.agent import run_agent\n"
        "from daydream.extensions import FlowStep\n"
        "from daydream.trajectory import DaydreamPhase\n"
        "async def probe(ctx):\n"
        "    phase = 'recon' if ctx.config.flow_name == 'improve' else 'probe'\n"
        "    first = ctx.backend_for(phase)\n"
        "    assert first is ctx.backend_for(phase)\n"
        f"    assert ctx.config.identity == ctx.run_context.github_identity.login == {login!r}\n"
        f"    assert (ctx.backend_execution is not None) is {injected!r}\n"
        "    assert 'backend_execution' not in repr(ctx)\n"
        "    assert 'github_execution' not in repr(ctx)\n"
        "    if ctx.audit_workspace is not None:\n"
        "        assert first.audit_root == ctx.audit_workspace.repo\n"
        "        assert first.audit_root != ctx.work.repo\n"
        "    else:\n"
        "        await run_agent(first, ctx.work.repo, 'FACTORY_PROBE', "
        "phase=DaydreamPhase.REVIEW, run_context=ctx.run_context)\n"
        "def register(registry):\n"
        "    registry.register_phase(FlowStep(name='factory-probe', run=probe))\n"
        "    registry.set_flow('factory-probe', ['factory-probe'])\n"
        "    registry.insert_before('improve', anchor='recon', step='factory-probe')\n"
    )
    source = {
        "PATH": os.environ["PATH"], "HOME": str(tmp_path / "execution-home"),
        "ANTHROPIC_API_KEY": "factory-owned-private-key", "DAYDREAM_PI_RETRY_ATTEMPTS": "0",
    }
    backend_input = BackendExecutionInput.from_environment(source, backend="claude")
    github_input = GitHubExecutionInput(
        git_ops.StaticGitHubAuth({ "PATH": os.environ["PATH"], "HOME": source["HOME"], })
    )
    execution = runner.RunnerExecutionInput(backend_input, github_input) if injected else None
    created: list[Any] = []
    github_environments: list[dict[str, str] | None] = []
    real_subprocess_run = subprocess.run
    def github_subprocess(args: list[str], *pargs: Any, **kwargs: Any) -> Any:
        if args[0] != "gh":
            return real_subprocess_run(args, *pargs, **kwargs)
        assert args[1:3] == ["api", "/user"]
        github_environments.append(kwargs["env"])
        return subprocess.CompletedProcess(args, 0, json.dumps({"login": login}), "")
    def create_backend(
        name: str, model: str | None = None, *, execution_input: Any = None, audit_root: Path | None = None,
        audit_outward_symlinks: frozenset[Path] = frozenset(), **kwargs: Any,
    ) -> Any:
        assert name == "claude"
        assert model == "factory-model"
        assert execution_input is (backend_input if injected else None)
        if flow == "improve":
            assert audit_root is not None and audit_root != repo
            backend: Any = ImproveStubBackend(repo, n_findings=0)
            backend.audit_root = audit_root
            backend.audit_root_isolation = AUDIT_ROOT_ISOLATION
            backend.audit_outward_symlinks = audit_outward_symlinks
        else:
            assert audit_root is None
            backend = ScriptedBackend(model=model)
        backend.model = model
        if execution_input is not None:
            backend.retry_policy = execution_input.retry_policy
        created.append(backend)
        return backend
    monkeypatch.setattr(subprocess, "run", github_subprocess)
    monkeypatch.setattr(runner, "create_backend", create_backend)
    if injected:
        # An injected GitHub capability must bypass ambient App validation/mint.
        monkeypatch.setenv("DAYDREAM_APP_ID", "malformed-parent-app-id")
        monkeypatch.setenv("DAYDREAM_APP_PRIVATE_KEY", "parent-private-key")
    config = RunConfig(
        target=str(repo), flow_name=flow, base="main", backend="claude", model="factory-model", archive=False,
        run_eval=False, non_interactive=True, file_config=DaydreamFileConfig(),
        trajectory_path=tmp_path / "trajectory.json",
    )
    result = await runner.run(
        config, execution=execution, private_roots=private_root_locations(base=tmp_path / "private"),
    )
    assert result == 0
    assert created and any(backend.calls for backend in created)
    assert len(github_environments) == 1
    if injected:
        assert github_environments[0] == github_input.auth.environment_for_request()
    else:
        assert github_environments == [None]
    if flow == "improve":
        assert improve_artifact(repo, "recon.json").is_file()
        for backend in created:
            assert all(call["cwd"] == backend.audit_root for call in backend.calls)
    else:
        assert len(created) == 1
        assert created[0].prompts == ["FACTORY_PROBE"]
        assert created[0].calls[0]["cwd"] == repo
    trajectory = (tmp_path / "trajectory.json").read_text()
    assert "factory-owned-private-key" not in trajectory
    assert "parent-private-key" not in trajectory


async def test_runner_captured_execution_preserves_native_effort_overrides_and_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Embedded arbiter groups use their own native effort with the same private capability."""
    from daydream.backends.codex import CodexBackend
    from daydream.extensions import Registry
    from daydream.flows.engine import FlowContext
    from daydream.workspace import WorkContext

    execution = BackendExecutionInput.from_environment(
        {"HOME": str(tmp_path), "PRIVATE_NATIVE_CANARY": "private-owned-value"}, backend="codex",
    )
    config = RunConfig(backend="codex", model="captured-model", non_interactive=True)

    async def capture(config: RunConfig, **kwargs: Any) -> int:
        assert kwargs["backend_execution"] is execution
        work = WorkContext(
            repo=tmp_path, source=tmp_path, base_branch="main", base_sha="0" * 40,
            head_branch="main", head_sha="0" * 40, is_ephemeral=False, run_id="execution-effort",
        )
        ctx = FlowContext(config, work, Registry(), backend_execution=kwargs["backend_execution"])
        medium = ctx.backend_for_effort("arbiter", "medium")
        high = ctx.backend_for_effort("arbiter", "xhigh")
        assert isinstance(medium, CodexBackend) and isinstance(high, CodexBackend)
        assert (medium.reasoning_effort, high.reasoning_effort) == ("medium", "xhigh")
        assert medium.model == high.model == "captured-model"
        assert medium._execution_input is high._execution_input is execution
        assert medium is ctx.backend_for_effort("arbiter", "medium")
        assert medium is not high and len(ctx._backend_cache) == 2
        assert "backend_execution" not in repr(ctx) and "private-owned-value" not in repr(ctx)
        return 0

    monkeypatch.setattr(runner, "_run_with_context", capture)
    assert await runner.run(config, execution=runner.RunnerExecutionInput(execution, GitHubExecutionInput())) == 0
