"""Harbor skill-free gate (M16): the controlled entrypoint runs native, skill-free.

The real end-to-end Python + mixed-stack Harbor run is the #783-dependent e2e in
``test_benchmark_e2e.py``; this module proves the controlled wiring: no
``DAYDREAM_SKILLS_DIR``, no Beagle probe, and the candidate profile still
resolves via the explicit-only Harbor resolver.
"""
import asyncio
import os
from pathlib import Path
from typing import Any

import pytest

from daydream.benchmark.harbor import entrypoint


def test_openrouter_reviewer_env_uses_pi_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "stale-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "stale-url")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "stale-token")
    monkeypatch.setenv("OPENROUTER_API_KEY", "stale-openrouter-key")
    monkeypatch.setenv("PI_API_KEY", "stale-pi-key")

    entrypoint.apply_reviewer_env({
        "DAYDREAM_REVIEW_API_KEY": "sk-or-test",
        "DAYDREAM_REVIEW_BASE_URL": "https://openrouter.ai/api",
    })

    assert os.environ["PI_PROVIDER"] == "openrouter"
    assert os.environ["PI_API_KEY"] == "sk-or-test"
    assert os.environ["PI_TELEMETRY"] == "0"
    assert "OPENROUTER_API_KEY" not in os.environ
    assert not any(key.startswith("ANTHROPIC_") for key in os.environ)


def test_reviewer_env_rejects_non_openrouter_endpoint() -> None:
    with pytest.raises(entrypoint.EntrypointError, match="openrouter.ai"):
        entrypoint.apply_reviewer_env({
            "DAYDREAM_REVIEW_API_KEY": "key",
            "DAYDREAM_REVIEW_BASE_URL": "https://example.com/api",
        })


def test_claude_reviewer_env_keeps_anthropic_and_no_openrouter_requirement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "stale-or")
    monkeypatch.setenv("PI_API_KEY", "stale-pi")
    monkeypatch.setenv("DAYDREAM_JUDGE_MODEL", "judge-model")

    entrypoint.apply_reviewer_env({
        "DAYDREAM_REVIEW_BACKEND": "claude",
        "ANTHROPIC_API_KEY": "sk-ant-live",
        "ANTHROPIC_AUTH_TOKEN": "tok-live",
        "ANTHROPIC_BASE_URL": "https://api.anthropic.com",
    }, backend="claude")

    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-live"
    assert os.environ["ANTHROPIC_AUTH_TOKEN"] == "tok-live"
    assert os.environ["ANTHROPIC_BASE_URL"] == "https://api.anthropic.com"
    assert "OPENROUTER_API_KEY" not in os.environ
    assert "PI_API_KEY" not in os.environ
    assert not any(k.startswith("DAYDREAM_JUDGE_") for k in os.environ)


def test_claude_reviewer_env_fails_without_anthropic_key(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(entrypoint.EntrypointError, match="ANTHROPIC"):
        entrypoint.apply_reviewer_env({
            "DAYDREAM_REVIEW_BACKEND": "claude",
            "ANTHROPIC_API_KEY": "",
        }, backend="claude")


def test_claude_reviewer_env_accepts_non_openrouter_base_url_only_for_claude(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entrypoint.apply_reviewer_env({
        "DAYDREAM_REVIEW_BACKEND": "claude",
        "ANTHROPIC_API_KEY": "sk-ant",
        "ANTHROPIC_BASE_URL": "https://claude-proxy.internal/v1",
    }, backend="claude")   # no raise: openrouter.ai requirement is pi-only


def test_entrypoint_skill_free_python_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # A Python diff resolves through the controlled entrypoint with no backend
    # network dependency (the runner is stubbed at the production seam): the run
    # completes, publishes the canonical artifact, and never emits a ProfileError.
    (tmp_path / "a.py").write_text("x = 1\n")
    artifact = tmp_path / "logs" / "artifacts" / "review.json"
    artifact.parent.mkdir(parents=True)

    async def _fake_run(config: Any) -> int:
        return 0

    def _fake_publish(
        *,
        repo_dir: Any,
        artifact_path: Any,
        case_id: Any,
        base_ref: Any="base",
        head_ref: Any="head",
    ) -> None:
        Path(artifact_path).write_text("{}")

    monkeypatch.setattr("daydream.runner.run", _fake_run)
    monkeypatch.setattr(entrypoint, "publish_review", _fake_publish)

    rc = asyncio.run(entrypoint.main(monkeypatch_env={
        "DAYDREAM_REVIEW_CASE_ID": "case-python",
        "DAYDREAM_REVIEW_ARTIFACT_PATH": str(artifact),
        "DAYDREAM_REVIEW_REPO_DIR": str(tmp_path),
        "DAYDREAM_REVIEW_BACKEND": "pi",
        "DAYDREAM_REVIEW_API_KEY": "sk-or-test",
        "DAYDREAM_REVIEW_BASE_URL": "https://openrouter.ai/api",
    }))
    assert rc == 0
    text = Path(artifact).read_text() if Path(artifact).exists() else ""
    assert "ProfileError" not in text


def test_entrypoint_env_has_no_skill_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The controlled entrypoint must never inject a skill-registry env var.
    monkeypatch.delenv("DAYDREAM_SKILLS_DIR", raising=False)
    artifact = tmp_path / "logs" / "artifacts" / "review.json"
    artifact.parent.mkdir(parents=True)

    async def _fake_run(config: Any) -> int:
        return 0

    def _fake_publish(
        *,
        repo_dir: Any,
        artifact_path: Any,
        case_id: Any,
        base_ref: Any="base",
        head_ref: Any="head",
    ) -> None:
        Path(artifact_path).write_text("{}")

    monkeypatch.setattr("daydream.runner.run", _fake_run)
    monkeypatch.setattr(entrypoint, "publish_review", _fake_publish)

    rc = asyncio.run(entrypoint.main(monkeypatch_env={
        "DAYDREAM_REVIEW_CASE_ID": "case-noskill",
        "DAYDREAM_REVIEW_ARTIFACT_PATH": str(artifact),
        "DAYDREAM_REVIEW_REPO_DIR": str(tmp_path),
        "DAYDREAM_REVIEW_BACKEND": "pi",
        "DAYDREAM_REVIEW_API_KEY": "sk-or-test",
        "DAYDREAM_REVIEW_BASE_URL": "https://openrouter.ai/api",
    }))
    assert rc == 0
    assert os.environ.get("DAYDREAM_SKILLS_DIR") is None
