"""Captured manifest identity exercised through the real runner and archive."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator, Callable
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from daydream import git_ops, runner
from daydream.backends import AgentEvent, BackendExecutionInput, MetricsEvent, ResultEvent, TextEvent
from daydream.backends.pi import PiBackend
from daydream.config import DEFAULT_PI_MODEL
from daydream.config_file import DaydreamFileConfig
from daydream.github_app import GitHubExecutionInput
from daydream.review_profile import ProfileError
from daydream.run_snapshot import ManifestRunIdentity
from tests.conftest import ExtDir
from tests.harness.backend import ScriptedBackend
from tests.harness.fake_gh import FakeGh


def _install_probe(ext_dir: ExtDir) -> None:
    """A real extension runs two model calls, then changes the live registry."""
    ext_dir.write_module(
        "from daydream.agent import run_agent\n"
        "from daydream.extensions import FlowStep\n"
        "from daydream.trajectory import DaydreamPhase, get_current_recorder\n"
        "async def probe(ctx):\n"
        "    backend = ctx.backend_for('per_stack_review')\n"
        "    recorder = get_current_recorder()\n"
        "    assert recorder is not None\n"
        "    for name in ('first', 'second'):\n"
        "        async with recorder.fork(name):\n"
        "            await run_agent(backend, ctx.work.repo, name,\n"
        "                phase=DaydreamPhase.REVIEW, run_context=ctx.run_context)\n"
        "    ctx.registry.set_flow('deep', ['fix', 'test', 'commit', 'remote-ci'])\n"
        "def register(registry):\n"
        "    registry.register_phase(FlowStep(name='manifest-probe', run=probe))\n"
        "    registry.set_flow('deep', ['manifest-probe'])\n"
        "    registry.set_flow('manifest-probe', ['manifest-probe'])\n"
    )


def _events(model: str, prompt: str) -> list[AgentEvent]:
    return [
        TextEvent(text="Reviewed the supplied diff."),
        MetricsEvent(
            message_id=prompt, prompt_tokens=30, completion_tokens=7,
            cached_tokens=5, cost_usd=0.125, model_name=model,
        ),
        ResultEvent(structured_output=None, continuation=None, model_name=model),
    ]


class _MutatingBackend(ScriptedBackend):
    def __init__(self, model: str, mutate: Callable[[], None]) -> None:
        super().__init__(model=model)
        self.mutate = mutate

    async def execute(
        self, cwd: Path, prompt: str, *args: Any, **kwargs: Any,
    ) -> AsyncGenerator[AgentEvent, None]:
        self.mutate()
        for event in _events(self.model, prompt):
            yield event


def _manifest(archive_dir: Path) -> tuple[dict[str, Any], Path]:
    paths = list((archive_dir / "runs").glob("*/manifest.json"))
    assert len(paths) == 1
    return json.loads(paths[0].read_bytes()), paths[0].parent


@pytest.mark.parametrize("cli_override", [False, True], ids=["file-phase", "cli-global"])
async def test_runner_archives_identity_before_model_mutates_live_policy(
    cli_override: bool,
    multi_stack_target: Path,
    archive_dir: Path,
    ext_dir: ExtDir,
    fake_gh: FakeGh,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_probe(ext_dir)
    monkeypatch.delenv("DAYDREAM_TRAJECTORY_HUB_REPO", raising=False)
    config = runner.RunConfig(
        target=str(multi_stack_target), base="main", cleanup=False,
        non_interactive=True, archive=True, run_eval=False,
        backend="codex" if cli_override else None,
        model="cli-model" if cli_override else None,
        pr_number=7, pr_repo="Owner/Repo",
        file_config=DaydreamFileConfig(
            backend="claude", model="file-global-model",
            phases={
                "review": {"backend": "pi"},
                "per_stack_review": {"backend": "pi", "model": "file-phase-model"},
            },
        ),
    )
    expected_backend = "codex" if cli_override else "pi"
    expected_model = "cli-model" if cli_override else "file-phase-model"
    observed_profile: dict[str, Any] = {}
    calls: list[tuple[str, str | None]] = []
    captured: list[ManifestRunIdentity] = []
    capture_identity = runner.capture_manifest_run_identity

    def observe_capture(*args: Any, **kwargs: Any) -> ManifestRunIdentity:
        identity = capture_identity(*args, **kwargs)
        captured.append(identity)
        return identity

    monkeypatch.setattr(runner, "capture_manifest_run_identity", observe_capture)

    def mutate() -> None:
        if observed_profile:
            return
        assert len(captured) == 1
        profile = config.review_profile
        assert profile is not None
        observed_profile.update(
            profile_schema_version=profile.profile.schema_version,
            profile_name=profile.name,
            profile_source_kind=profile.source_kind,
            profile_digest=profile.digest,
        )
        config.backend = "claude"
        config.model = "changed-after-model-start"
        config.stack = "changed-skill"
        config.shallow = True
        config.output_mode = "review"
        config.review_profile = None
        config.pr_number = 99
        config.pr_repo = "Changed/Repo"
        config.file_config = DaydreamFileConfig()

    def create_backend(name: str, model: str | None = None, **kwargs: Any) -> _MutatingBackend:
        calls.append((name, model))
        assert (name, model) == (expected_backend, expected_model)
        return _MutatingBackend(expected_model, mutate)

    monkeypatch.setattr(runner, "create_backend", create_backend)
    assert await runner.run(config) == 0
    assert calls == [(expected_backend, expected_model)]
    assert len(captured) == 1
    identity = captured[0]
    assert identity.per_stack_review_model == expected_model
    assert identity.phases.fix is False
    assert identity.profile is not None
    for value, field, replacement in (
        (identity, "backend", "changed"),
        (identity.phases, "fix", True),
        (identity.profile, "name", "changed"),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(value, field, replacement)
    manifest, archived_run = _manifest(archive_dir)
    assert manifest["run"] == {
        "flow": "deep", "skill": None, "model": None,
        "backend": "codex" if cli_override else "claude",
        "review_backend": "pi", "per_stack_review_backend": expected_backend,
        "per_stack_review_model": expected_model, "review_only": False, "deep": True,
    }
    for key, value in observed_profile.items():
        assert manifest[key] == value
    assert manifest["pr"] == {"number": 7, "repo": "Owner/Repo"}
    for phase in ("fix", "test", "push", "remote_ci"):
        assert manifest["phase_states"][phase]["ran"] is False
    assert manifest["metrics"]["total_prompt_tokens"] == 60
    assert manifest["metrics"]["total_completion_tokens"] == 14
    assert manifest["metrics"]["total_cached_tokens"] == 10
    assert manifest["metrics"]["total_cost_usd"] == 0.25
    trajectories = [json.loads(path.read_bytes()) for path in archived_run.rglob("*.json")
                    if path.name != "manifest.json"]
    roots = [document for document in trajectories
             if document.get("trajectory_id") == manifest["session_id"]]
    assert len(roots) == 1
    assert roots[0]["extra"]["pr_number"] == 7
    assert roots[0]["extra"]["pr_repo"] == "Owner/Repo"
    forks = [document for document in trajectories
             if document.get("trajectory_id") and document.get("trajectory_id") != manifest["session_id"]]
    assert len(forks) == 2


@pytest.mark.parametrize("owned_agent_dir", [True, False], ids=["owned-settings", "no-owned-home"])
async def test_runner_captures_pi_default_from_owned_execution_before_settings_change(
    owned_agent_dir: bool,
    multi_stack_target: Path,
    tmp_path: Path,
    archive_dir: Path,
    ext_dir: ExtDir,
    fake_gh: FakeGh,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_probe(ext_dir)
    monkeypatch.delenv("DAYDREAM_TRAJECTORY_HUB_REPO", raising=False)
    ambient = tmp_path / "ambient-pi"
    ambient.mkdir()
    ambient_settings = ambient / "settings.json"
    ambient_settings.write_text(json.dumps({"defaultModel": "ambient-model"}))
    monkeypatch.setenv("PI_CODING_AGENT_DIR", str(ambient))
    owned = tmp_path / "owned-pi"
    owned.mkdir()
    owned_settings = owned / "settings.json"
    owned_settings.write_text(json.dumps({"defaultModel": "owned-model"}))
    environment = {"PI_CODING_AGENT_DIR": str(owned)} if owned_agent_dir else {}
    backend_input = BackendExecutionInput.from_environment(environment, backend="pi")
    execution = runner.RunnerExecutionInput(
        backend_input, GitHubExecutionInput(git_ops.StaticGitHubAuth({})),
    )
    expected_model = "owned-model" if owned_agent_dir else DEFAULT_PI_MODEL
    observed_models: list[str] = []

    async def execute(
        self: PiBackend, cwd: Path, prompt: str, *args: Any, **kwargs: Any,
    ) -> AsyncGenerator[AgentEvent, None]:
        observed_models.append(self.model)
        owned_settings.write_text(json.dumps({"defaultModel": "changed-owned-model"}))
        ambient_settings.write_text(json.dumps({"defaultModel": "changed-ambient-model"}))
        for event in _events(self.model, prompt):
            yield event

    monkeypatch.setattr(PiBackend, "execute", execute)
    assert await runner.run(
        runner.RunConfig(
            target=str(multi_stack_target), base="main", backend="pi",
            cleanup=False, non_interactive=True, archive=True, run_eval=False,
            file_config=DaydreamFileConfig(),
        ),
        execution=execution,
    ) == 0
    assert observed_models == [expected_model, expected_model]
    manifest, _ = _manifest(archive_dir)
    assert manifest["run"]["per_stack_review_backend"] == "pi"
    assert manifest["run"]["per_stack_review_model"] == expected_model


@pytest.mark.parametrize("flow", ["improve", "manifest-probe"], ids=["improve", "custom"])
async def test_runner_rejects_invalid_profile_before_recorder_or_backend(
    flow: str,
    multi_stack_target: Path,
    tmp_path: Path,
    archive_dir: Path,
    ext_dir: ExtDir,
    fake_gh: FakeGh,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Invalid policy has no run identity and must not publish an empty run."""
    _install_probe(ext_dir)
    bad_profile = tmp_path / "invalid-profile.toml"
    bad_profile.write_text('schema_version = 1\nname = "invalid"\nunknown = true\n')
    backend_calls: list[str] = []

    def unexpected_backend(name: str, **kwargs: Any) -> Any:
        backend_calls.append(name)
        raise AssertionError("invalid profile must fail before backend construction")

    monkeypatch.setattr(runner, "create_backend", unexpected_backend)
    archives_before = set((archive_dir / "runs").glob("*/manifest.json"))
    trajectory_path = tmp_path / "invalid-profile-trajectory.json"
    with pytest.raises(ProfileError, match="invalid review profile") as failure:
        await runner.run(runner.RunConfig(
            target=str(multi_stack_target), base="main", flow_name=flow,
            cleanup=False, non_interactive=True, archive=True, run_eval=False,
            file_config=DaydreamFileConfig(), review_profile_path=bad_profile,
            trajectory_path=trajectory_path,
        ))
    assert str(bad_profile) in str(failure.value)
    assert not backend_calls
    assert not trajectory_path.exists()
    assert set((archive_dir / "runs").glob("*/manifest.json")) == archives_before
