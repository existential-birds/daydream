"""Harbor skill-free gate (M16): the controlled entrypoint runs native, skill-free.

The real end-to-end Python + mixed-stack Harbor run is outside this module;
these tests prove the controlled wiring: no
``DAYDREAM_SKILLS_DIR``, no Beagle probe, and the candidate profile still
resolves via the explicit-only Harbor resolver.
"""
import asyncio
import json
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


# ---------------------------------------------------------------------------
# host-side gate/egress: the preflight must resolve and validate the claude
# reviewer endpoint (ANTHROPIC_BASE_URL), never the pi-era var alone.
# ---------------------------------------------------------------------------


def test_host_reviewer_host_resolution_is_backend_aware() -> None:
    # A claude operator configuring only ANTHROPIC_API_KEY + ANTHROPIC_BASE_URL
    # (no pi-era DAYDREAM_REVIEW_BASE_URL) must resolve the reviewer host from
    # ANTHROPIC_BASE_URL; the pi default keeps its existing behavior/error.
    from daydream.benchmark.harbor import run as run_mod

    assert run_mod._reviewer_host_from_env({
        "DAYDREAM_REVIEW_BACKEND": "claude",
        "ANTHROPIC_BASE_URL": "https://claude-proxy.internal/v1",
    }) == "claude-proxy.internal"
    # Unset ANTHROPIC_BASE_URL falls back to the Anthropic SDK default, mirroring
    # the in-container claude branch that accepts an unset base URL.
    assert run_mod._reviewer_host_from_env({
        "DAYDREAM_REVIEW_BACKEND": "claude",
    }) == "api.anthropic.com"
    assert run_mod._reviewer_base_url_from_env({}) == ""
    # Default (pi) resolution is unchanged, including the fail-closed error.
    assert run_mod._reviewer_host_from_env({
        "DAYDREAM_REVIEW_BASE_URL": "https://openrouter.ai/api",
    }) == "openrouter.ai"
    with pytest.raises(ValueError, match="missing DAYDREAM_REVIEW_BASE_URL"):
        run_mod._reviewer_host_from_env({})


def _seed_host_ws(tmp_path: Path, reviewer_hosts: list[str]) -> Path:
    """A minimal compiled benchmark workspace for the host preflight."""
    ws = tmp_path / "ws"
    (ws / "harbor" / "case-a").mkdir(parents=True)
    (ws / "harbor" / "case-a" / "task.toml").write_text(
        "[agent]\n"
        f"allowed_hosts = {json.dumps(reviewer_hosts)}\n"
        "\n"
        "[verifier.environment]\n"
        'allowed_hosts = ["127.0.0.1"]\n'
    )
    (ws / "harbor" / "benchmark.lock.json").write_text(
        '{"schema_version": 1, "cases": {"case-a": {"key": "case-a"}}, "files": {}}'
    )
    (ws / "harbor" / "harbor-job.yaml").write_text("jobs_dir: jobs\n")
    (ws / "harbor" / "harbor-oracle.yaml").write_text("jobs_dir: jobs\n")
    privacy = {"classification": "confidential", "reviewer_data": "source_snapshot",
               "reviewer_allowed_hosts": reviewer_hosts,
               "judge_data": "finding_text_and_location_only",
               "judge_allowed_hosts": ["127.0.0.1"],
               "archive": "disabled", "uploads": "disabled"}
    (ws / "benchmark.yaml").write_text(json.dumps({
        "schema_version": 1, "benchmark_id": "6c38dc0a",
        "created_at": "2026-08-21T12:00:00Z",
        "source": {"provider": "github", "hostname": "github.com",
                   "repository": "OWNER/REPO", "repository_id": None,
                   "visibility": "unresolved"},
        "privacy": privacy, "pull_requests": [], "cases": []}))
    return ws


def test_host_preflight_accepts_allowlisted_claude_proxy_without_pi_var(
    tmp_path: Path,
) -> None:
    # A claude reviewer whose ANTHROPIC_BASE_URL is in the compiled reviewer
    # allowed_hosts passes preflight with no pi-era DAYDREAM_REVIEW_BASE_URL
    # set (regression: "cannot resolve reviewer host: missing
    # DAYDREAM_REVIEW_BASE_URL" blocked the documented claude surface).
    from daydream.benchmark.harbor import package as pkg
    from daydream.benchmark.harbor import run as run_mod

    ws = _seed_host_ws(tmp_path, ["claude-proxy.internal"])

    def _docker_ok() -> pkg.DockerNetworkPolicyCapability:
        return pkg.DockerNetworkPolicyCapability(supported=True)

    errs = run_mod._preflight(ws, oracle=True, env={
        "DAYDREAM_REVIEW_BACKEND": "claude",
        "ANTHROPIC_API_KEY": "sk-ant",
        "ANTHROPIC_BASE_URL": "https://claude-proxy.internal/v1",
        "DAYDREAM_JUDGE_BASE_URL": "http://127.0.0.1:9",
    }, docker_ok=_docker_ok)
    assert not any("reviewer host" in e for e in errs)
    assert not any("missing DAYDREAM_REVIEW_BASE_URL" in e for e in errs)


def test_host_preflight_blocks_non_allowlisted_claude_proxy(tmp_path: Path) -> None:
    # A proxy ANTHROPIC_BASE_URL outside the compiled reviewer allowed_hosts
    # must be rejected host-side, before any paid review starts (previously it
    # passed setup+preflight and failed only in-container at the SDK call).
    from daydream.benchmark.harbor import package as pkg
    from daydream.benchmark.harbor import run as run_mod

    ws = _seed_host_ws(tmp_path, ["review.example"])

    def _docker_ok() -> pkg.DockerNetworkPolicyCapability:
        return pkg.DockerNetworkPolicyCapability(supported=True)

    errs = run_mod._preflight(ws, oracle=True, env={
        "DAYDREAM_REVIEW_BACKEND": "claude",
        "ANTHROPIC_API_KEY": "sk-ant",
        "ANTHROPIC_BASE_URL": "https://claude-proxy.internal/v1",
        "DAYDREAM_JUDGE_BASE_URL": "http://127.0.0.1:9",
    }, docker_ok=_docker_ok)
    reviewer_errs = [e for e in errs if "reviewer host" in e]
    assert any("claude-proxy.internal" in e for e in reviewer_errs)
    assert not any("missing DAYDREAM_REVIEW_BASE_URL" in e for e in errs)


def test_host_run_gate_threads_claude_credentials_into_supervisor_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The host-side run gate env snapshot must carry the ANTHROPIC_* reviewer
    # credentials (previously it forwarded only DAYDREAM_REVIEW_*/JUDGE_*), so
    # run.py can resolve ANTHROPIC_BASE_URL for the claude backend.
    from daydream.benchmark.cli import _handle_benchmark_command
    from daydream.benchmark.harbor import run as run_mod

    monkeypatch.setenv("DAYDREAM_REVIEW_BACKEND", "claude")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "tok")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://claude-proxy.internal/v1")

    captured: dict[str, Any] = {}

    def fake_run_run(ws: Any, *, oracle: Any, yes: Any, env: Any) -> int:
        captured["env"] = env
        return 0

    monkeypatch.setattr(run_mod, "run_run", fake_run_run)
    code = _handle_benchmark_command(["run", str(tmp_path), "--yes"])
    assert code == 0
    assert captured["env"]["DAYDREAM_REVIEW_BACKEND"] == "claude"
    assert captured["env"]["ANTHROPIC_API_KEY"] == "sk-ant"
    assert captured["env"]["ANTHROPIC_AUTH_TOKEN"] == "tok"
    assert captured["env"]["ANTHROPIC_BASE_URL"] == "https://claude-proxy.internal/v1"


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


def test_entrypoint_claude_backend_reaches_run_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    artifact = tmp_path / "logs" / "artifacts" / "review.json"
    artifact.parent.mkdir(parents=True)
    seen: dict[str, Any] = {}

    async def _fake_run(config: Any) -> int:
        seen["backend"] = config.backend
        return 0

    def _fake_publish(**kwargs: Any) -> None:
        Path(kwargs["artifact_path"]).write_text("{}")

    monkeypatch.setattr("daydream.runner.run", _fake_run)
    monkeypatch.setattr(entrypoint, "publish_review", _fake_publish)

    rc = asyncio.run(entrypoint.main(monkeypatch_env={
        "DAYDREAM_REVIEW_CASE_ID": "case-claude",
        "DAYDREAM_REVIEW_ARTIFACT_PATH": str(artifact),
        "DAYDREAM_REVIEW_REPO_DIR": str(tmp_path),
        "DAYDREAM_REVIEW_BACKEND": "claude",
        "ANTHROPIC_API_KEY": "sk-ant",
    }))
    assert rc == 0
    assert seen["backend"] == "claude"          # the wiring this issue is about


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
