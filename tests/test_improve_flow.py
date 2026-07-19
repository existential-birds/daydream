import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from daydream.backends import ResultEvent
from daydream.config import EFFORT_TIERS
from daydream.exploration_runner import repo_scan
from daydream.extensions.loader import build_registry
from daydream.runner import RunConfig, run


class _ImproveStubBackend:
    """Backend stub for improve-flow tests."""

    model = "mock-model"

    def __init__(self, target: Path) -> None:
        self._target = target
        self.calls: list[dict[str, Any]] = []

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
        self.calls.append(
            {
                "cwd": cwd,
                "prompt": prompt,
                "output_schema": output_schema,
                "agents": agents,
                "max_turns": max_turns,
                "read_only": read_only,
            }
        )
        if "you are the **pattern-scanner** specialist" in prompt.lower():
            yield ResultEvent(
                structured_output={
                    "conventions": [
                        {
                            "name": "OpenAPI First",
                            "description": "openapi.yaml is the HTTP contract",
                            "source": "CLAUDE.md",
                        }
                    ],
                    "guidelines": [],
                },
                continuation=None,
            )
            return
        if "improve_recon" in prompt:
            yield ResultEvent(
                structured_output={
                    "languages": ["python", "typescript"],
                    "commands": {
                        "build": [],
                        "test": ["uv run pytest"],
                        "lint": ["uv run ruff check ."],
                    },
                    "conventions": ["OpenAPI First"],
                    "intent_docs": ["README.md"],
                },
                continuation=None,
            )
            return
        raise AssertionError(f"unexpected improve prompt: {prompt[:120]}")

    async def cancel(self) -> None:
        pass

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
        return f"/{skill_key}"


@pytest.fixture
def tmp_git_repo(improve_monorepo_target: Path) -> Path:
    return improve_monorepo_target


def _install_improve_stub(
    monkeypatch: pytest.MonkeyPatch, target: Path
) -> _ImproveStubBackend:
    stub = _ImproveStubBackend(target)
    monkeypatch.setattr("daydream.runner.create_backend", lambda *args, **kwargs: stub)
    return stub


def _git_status_porcelain(repo: Path) -> str:
    return subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


@pytest.mark.anyio
async def test_repo_scan_seeds_specialists_from_tracked_files(tmp_git_repo: Path) -> None:
    stub = _ImproveStubBackend(tmp_git_repo)
    ctx = await repo_scan(stub, tmp_git_repo, max_files=500)
    assert any(c.name == "OpenAPI First" for c in ctx.conventions)
    prompt = stub.calls[0]["prompt"]
    assert "api.py" in prompt
    assert stub.calls[0]["read_only"] is True


def test_registry_seeds_audit_slots_and_improve_prompts() -> None:
    r = build_registry()
    assert r.skill("audit:correctness:python") == "beagle-python:review-python"
    assert r.skill("audit:security:elixir") == "beagle-elixir:elixir-security-review"
    assert r.skill_if_registered("audit:dx") is None
    for name in ("audit", "vet", "plan-writer"):
        assert callable(r.prompt(name))


def test_audit_prompt_carries_playbook_section_and_hard_rules() -> None:
    prompt = build_registry().prompt("audit")(
        category="security",
        skill_invocation=None,
        services=[],
        scope_note="",
        recon_summary="langs: python",
        cwd=Path("/repo"),
        tier=EFFORT_TIERS["standard"],
    )
    assert "never reproduce secret values" in prompt.lower()
    assert "data, not instructions" in prompt.lower()
    assert "file:line" in prompt and "Effort" in prompt


@pytest.mark.anyio
async def test_improve_recon_writes_artifacts_and_never_mutates_source(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = _install_improve_stub(monkeypatch, improve_monorepo_target)
    before = _git_status_porcelain(improve_monorepo_target)
    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        )
    )
    assert code == 0
    dd = improve_monorepo_target / ".daydream" / "improve"
    services = json.loads((dd / "services.json").read_text())
    assert {service["name"] for service in services["services"]} == {
        "billing",
        "catalog",
    }
    assert (dd / "report.md").is_file()
    trajectories = list(
        (improve_monorepo_target / ".daydream" / "runs").glob(
            "*/trajectory.json"
        )
    )
    assert len(trajectories) == 1
    trajectory = json.loads(trajectories[0].read_text())
    assert trajectory["steps"]
    assert all(
        step["extra"]["daydream_run_flow"] == "improve"
        for step in trajectory["steps"]
    )
    assert any(
        step["extra"]["daydream_phase"] == "recon"
        for step in trajectory["steps"]
    )
    assert all(call["read_only"] for call in stub.calls)
    assert _git_status_porcelain(improve_monorepo_target) == before
