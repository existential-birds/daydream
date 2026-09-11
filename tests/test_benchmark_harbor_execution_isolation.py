"""Concurrent Harbor execution through real runners and artifact publication."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from daydream import runner
from daydream.backends import AgentEvent
from daydream.benchmark.harbor import entrypoint, verifier_core
from tests.harness.git_helpers import commit, git, init_repo
from tests.harness.stub_backend import StubBackend


def _review_repository(root: Path) -> Path:
    repo = root / "repo"
    repo.mkdir(parents=True)
    (repo / "api.py").write_text("def hello():\n    return 'world'\n")
    init_repo(repo)
    git(repo, "add", ".")
    commit(repo, "base")
    git(repo, "branch", "base")
    git(repo, "checkout", "-b", "feature")
    (repo / "api.py").write_text("def hello():\n    return 'universe'\n")
    git(repo, "add", ".")
    commit(repo, "change")
    return repo


async def test_concurrent_harbor_reviews_keep_execution_and_artifacts_private(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repositories = {name: _review_repository(tmp_path / name) for name in ("pi", "claude")}
    environments: dict[str, dict[str, str]] = {}
    for name, repo in repositories.items():
        root = repo.parent
        profile = root / "profile.toml"
        profile.write_text(
            f'schema_version = 1\nname = "{name}-candidate"\n'
            "[strategies.intent]\n"
            f'content = "Understand the intent of these changes. PROFILE-{name}."\n'
            'source = "copied: isolation-test"\n'
        )
        environments[name] = {
            "PATH": os.environ["PATH"],
            "HOME": str(root / "home"),
            "LANG": "C.UTF-8",
            "DAYDREAM_REVIEW_BACKEND": name,
            "DAYDREAM_REVIEW_MODEL": f"isolation-{name}-model",
            "DAYDREAM_REVIEW_REPO_DIR": str(repo),
            "DAYDREAM_REVIEW_ARTIFACT_PATH": str(root / "review.json"),
            "DAYDREAM_REVIEW_TRAJECTORY_PATH": str(root / "trajectory.json"),
            "DAYDREAM_REVIEW_CASE_ID": f"case-{name}",
            "DAYDREAM_REVIEW_PROFILE_CANDIDATE": str(profile),
            "DAYDREAM_PI_RETRY_ATTEMPTS": "1" if name == "pi" else "2",
            "DAYDREAM_PI_RETRY_BASE_DELAY_S": "0",
            "DAYDREAM_PI_RETRY_MAX_DELAY_S": "0",
            "DAYDREAM_PI_FANOUT_CONCURRENCY": "2",
            "DAYDREAM_FANOUT_CONCURRENCY": "3",
            "GH_TOKEN": f"forbidden-{name}-github-token",
            "GH_HOST": f"forbidden-{name}.example",
            "DAYDREAM_JUDGE_API_KEY": f"forbidden-{name}-judge-key",
            "ZAI_API_KEY": f"forbidden-{name}-zai-key",
            "NOUS_API_KEY": f"forbidden-{name}-nous-key",
        }
        if name == "pi":
            environments[name].update(
                {
                    "DAYDREAM_REVIEW_API_KEY": "reviewer-pi-private-key",
                    "DAYDREAM_REVIEW_BASE_URL": "https://openrouter.ai/api/v1",
                    "PI_THINKING": "high",
                    "PI_CODING_AGENT_DIR": str(root / "pi-agent"),
                }
            )
        else:
            environments[name].update(
                {
                    "ANTHROPIC_API_KEY": "reviewer-claude-private-key",
                    "ANTHROPIC_BASE_URL": "https://claude-review.example/v1",
                }
            )

    monkeypatch.setenv("DAYDREAM_APP_ID", "not-an-integer")
    monkeypatch.setenv("DAYDREAM_APP_PRIVATE_KEY", "parent-private-app-key")
    monkeypatch.setenv("GH_TOKEN", "parent-github-private-token")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "parent-anthropic-private-token")
    monkeypatch.setenv("DAYDREAM_PI_RETRY_ATTEMPTS", "0")
    parent_before = dict(os.environ)
    caller_before = {name: dict(env) for name, env in environments.items()}
    entered = {name: asyncio.Event() for name in repositories}
    release = asyncio.Event()
    failed_once: set[str] = set()
    created: dict[str, list[Any]] = {name: [] for name in repositories}
    prompts: dict[str, list[str]] = {name: [] for name in repositories}
    github_environments: dict[Path, list[dict[str, str]]] = {}
    real_subprocess_run = subprocess.run

    def github_subprocess(args: list[str], *pargs: Any, **kwargs: Any) -> Any:
        if args[0] != "gh":
            return real_subprocess_run(args, *pargs, **kwargs)
        assert args[1:3] == ["api", "/user"]
        environment = kwargs["env"]
        assert environment is not None
        github_environments.setdefault(Path(kwargs["cwd"]), []).append(dict(environment))
        return subprocess.CompletedProcess(args, 0, '{"login":"reviewer"}', "")

    class RetryableError(Exception):
        retryable = True

    class ReviewBackend(StubBackend):
        def __init__(self, name: str, model: str, execution: Any) -> None:
            super().__init__(repositories[name], model=model)
            self.name = name
            self.execution = execution
            self.retry_policy = execution.retry_policy
            self.fanout_concurrency = execution.fanout_concurrency

        async def execute(
            self,
            cwd: Path,
            prompt: str,
            *args: Any,
            **kwargs: Any,
        ) -> AsyncIterator[AgentEvent]:
            assert cwd == repositories[self.name]
            prompts[self.name].append(prompt)
            entered[self.name].set()
            await release.wait()
            if self.name not in failed_once:
                failed_once.add(self.name)
                raise RetryableError("retry the owning review once")
            async for event in super().execute(cwd, prompt, *args, **kwargs):
                yield event

    def create_backend(
        name: str,
        model: str | None = None,
        *,
        execution_input: Any = None,
        **kwargs: Any,
    ) -> ReviewBackend:
        assert execution_input is not None, "Harbor lost its explicit backend execution"
        assert model == f"isolation-{name}-model"
        backend = ReviewBackend(name, model, execution_input)
        created[name].append(backend)
        return backend

    monkeypatch.setattr(subprocess, "run", github_subprocess)
    monkeypatch.setattr(runner, "create_backend", create_backend)
    results: dict[str, int] = {}

    async def review(name: str) -> None:
        results[name] = await entrypoint.main(environments[name])

    async with asyncio.timeout(45):
        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(review("pi"))
            tasks.create_task(review("claude"))
            await asyncio.gather(*(event.wait() for event in entered.values()))
            assert dict(os.environ) == parent_before
            release.set()

    assert results == {"pi": 0, "claude": 0}
    assert failed_once == {"pi", "claude"}
    assert environments == caller_before
    assert dict(os.environ) == parent_before
    forbidden = (
        "parent-private-app-key",
        "parent-github-private-token",
        "parent-anthropic-private-token",
        "reviewer-pi-private-key",
        "reviewer-claude-private-key",
        "forbidden-pi-github-token",
        "forbidden-claude-github-token",
        "forbidden-pi-judge-key",
        "forbidden-claude-judge-key",
        "forbidden-pi-zai-key",
        "forbidden-pi-nous-key",
        "forbidden-claude-zai-key",
        "forbidden-claude-nous-key",
    )
    for name, repo in repositories.items():
        assert created[name]
        assert len({id(backend.execution) for backend in created[name]}) == 1
        assert any(f"PROFILE-{name}" in prompt for prompt in prompts[name])
        other = "claude" if name == "pi" else "pi"
        assert all(f"PROFILE-{other}" not in prompt for prompt in prompts[name])
        for backend in created[name]:
            environment = backend.execution.child_environment()
            assert environment["HOME"] == str(repo.parent / "home")
            assert backend.fanout_concurrency == (2 if name == "pi" else 3)
            expected_agent_dir = (
                repo.parent / "pi-agent" if name == "pi"
                else repo.parent / "home" / ".pi" / "agent"
            )
            assert backend.execution.pi_agent_dir == expected_agent_dir
            assert backend.retry_policy.attempts == (1 if name == "pi" else 2)
            assert backend.retry_policy.base_delay_s == backend.retry_policy.max_delay_s == 0
            if name == "pi":
                assert backend.execution.pi_provider == "openrouter"
                assert environment["PI_API_KEY"] == "reviewer-pi-private-key"
                assert environment["PI_THINKING"] == "high"
                assert environment["PI_CODING_AGENT_DIR"] == str(repo.parent / "pi-agent")
                assert backend.execution.pi_thinking == "high"
                assert "ZAI_API_KEY" not in environment and "NOUS_API_KEY" not in environment
                assert not any(key.startswith("ANTHROPIC_") for key in environment)
            else:
                assert environment["ANTHROPIC_API_KEY"] == "reviewer-claude-private-key"
                assert environment["ANTHROPIC_BASE_URL"] == "https://claude-review.example/v1"
                assert "PI_API_KEY" not in environment
                assert "PI_THINKING" not in environment and "PI_CODING_AGENT_DIR" not in environment
                assert "ZAI_API_KEY" not in environment and "NOUS_API_KEY" not in environment
        assert len(github_environments[repo]) == 1
        for environment in github_environments[repo]:
            assert environment["HOME"] == str(repo.parent / "home")
            assert "GH_TOKEN" not in environment and "GH_HOST" not in environment
            assert not any(key.startswith("DAYDREAM_APP_") for key in environment)
            assert not any(key in environment for key in (
                "PI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
                "ZAI_API_KEY", "NOUS_API_KEY",
            ))
        artifact_path = repo.parent / "review.json"
        trajectory_path = repo.parent / "trajectory.json"
        artifact = json.loads(artifact_path.read_text())
        assert artifact["case_id"] == f"case-{name}"
        assert verifier_core.validate_candidate_artifact(artifact)
        assert trajectory_path.is_file()
        for path in (artifact_path, trajectory_path):
            payload = path.read_text()
            assert all(secret not in payload for secret in forbidden)
