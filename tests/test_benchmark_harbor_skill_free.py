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


def test_parse_reviewer_environment_maps_pi_without_mutating_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PI_API_KEY", "ambient-pi-key")
    parent = {
        "PATH": "/usr/bin",
        "DAYDREAM_REVIEW_CASE_ID": "case-parser-pi",
        "DAYDREAM_REVIEW_REPO_DIR": "/review/repo",
        "DAYDREAM_REVIEW_ARTIFACT_PATH": "/review/out.json",
        "DAYDREAM_REVIEW_TRAJECTORY_PATH": "/review/trajectory.json",
        "DAYDREAM_REVIEW_BASE_REF": "frozen-base",
        "DAYDREAM_REVIEW_HEAD_REF": "frozen-head",
        "DAYDREAM_REVIEW_MODEL": "review-model",
        "DAYDREAM_REVIEW_PROFILE_CANDIDATE": "/review/profile.toml",
        "DAYDREAM_REVIEW_API_KEY": "sk-or-test",
        "DAYDREAM_REVIEW_BASE_URL": "https://openrouter.ai/api",
        "ANTHROPIC_API_KEY": "stale-key",
        "OPENROUTER_API_KEY": "stale-openrouter-key",
        "PI_API_KEY": "stale-pi-key",
        "PI_THINKING": "high",
        "PI_CODING_AGENT_DIR": "/review/pi-agent",
        "PI_UNRELATED_CONTROL": "must-scrub",
        "ZAI_API_KEY": "stale-zai-key",
        "NOUS_API_KEY": "stale-nous-key",
        "DAYDREAM_JUDGE_MODEL": "judge-model",
        "GH_TOKEN": "stale-gh-token",
        "GITHUB_TOKEN": "stale-github-token",
        "GH_HOST": "github.example",
        "DAYDREAM_APP_FUTURE_SECRET": "stale-app-secret",
    }

    parsed = entrypoint.parse_reviewer_environment(parent)
    child = parsed.execution.backend.child_environment()

    assert child["PI_PROVIDER"] == "openrouter"
    assert child["PI_API_KEY"] == "sk-or-test"
    assert child["PI_TELEMETRY"] == "0"
    assert child["PI_THINKING"] == "high"
    assert child["PI_CODING_AGENT_DIR"] == "/review/pi-agent"
    assert parsed.execution.backend.pi_thinking == "high"
    assert parsed.execution.backend.pi_agent_dir == Path("/review/pi-agent")
    assert "PI_UNRELATED_CONTROL" not in child
    assert "OPENROUTER_API_KEY" not in child
    assert "ZAI_API_KEY" not in child and "NOUS_API_KEY" not in child
    assert not any(key.startswith("ANTHROPIC_") for key in child)
    assert not any(key.startswith("DAYDREAM_JUDGE_") for key in child)
    assert "GH_TOKEN" not in child
    github_environment = parsed.execution.github.auth.environment_for_request()
    assert github_environment is not None
    for forbidden in (
        "GH_TOKEN", "GITHUB_TOKEN", "GH_HOST", "DAYDREAM_APP_FUTURE_SECRET",
        "PI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
        "ZAI_API_KEY", "NOUS_API_KEY",
    ):
        assert forbidden not in github_environment
    assert parsed.repo_dir == Path("/review/repo")
    assert parsed.artifact_path == Path("/review/out.json")
    assert parsed.trajectory_path == Path("/review/trajectory.json")
    assert (parsed.case_id, parsed.base_ref, parsed.head_ref) == (
        "case-parser-pi", "frozen-base", "frozen-head"
    )
    assert parsed.model == "review-model"
    assert parsed.profile_candidate == "/review/profile.toml"
    assert parent["PI_API_KEY"] == "stale-pi-key"
    assert parent["PI_THINKING"] == "high"
    assert parent["PI_CODING_AGENT_DIR"] == "/review/pi-agent"
    assert parent["ANTHROPIC_API_KEY"] == "stale-key"
    assert os.environ["PI_API_KEY"] == "ambient-pi-key"


def test_parse_reviewer_environment_rejects_non_openrouter_endpoint() -> None:
    with pytest.raises(entrypoint.EntrypointError, match="openrouter.ai"):
        entrypoint.parse_reviewer_environment({
            "DAYDREAM_REVIEW_API_KEY": "key",
            "DAYDREAM_REVIEW_BASE_URL": "https://example.com/api",
        })


def test_parse_reviewer_environment_maps_claude_without_openrouter_requirement() -> None:
    parent = {
        "DAYDREAM_REVIEW_BACKEND": "claude",
        "DAYDREAM_REVIEW_CASE_ID": "case-parser-claude",
        "ANTHROPIC_API_KEY": "sk-ant-live",
        "ANTHROPIC_AUTH_TOKEN": "tok-live",
        "ANTHROPIC_BASE_URL": "https://api.anthropic.com",
        "OPENROUTER_API_KEY": "stale-or",
        "PI_API_KEY": "stale-pi",
        "PI_THINKING": "must-scrub",
        "PI_CODING_AGENT_DIR": "/review/must-scrub",
        "ZAI_API_KEY": "stale-zai-key",
        "NOUS_API_KEY": "stale-nous-key",
        "DAYDREAM_JUDGE_MODEL": "judge-model",
    }

    parsed = entrypoint.parse_reviewer_environment(parent)
    child = parsed.execution.backend.child_environment()

    assert child["ANTHROPIC_API_KEY"] == "sk-ant-live"
    assert child["ANTHROPIC_AUTH_TOKEN"] == "tok-live"
    assert child["ANTHROPIC_BASE_URL"] == "https://api.anthropic.com"
    assert "OPENROUTER_API_KEY" not in child
    assert "PI_API_KEY" not in child
    assert "PI_THINKING" not in child and "PI_CODING_AGENT_DIR" not in child
    assert "ZAI_API_KEY" not in child and "NOUS_API_KEY" not in child
    assert not any(key.startswith("DAYDREAM_JUDGE_") for key in child)
    assert parent["PI_API_KEY"] == "stale-pi"


@pytest.mark.parametrize("environment", [
    {"DAYDREAM_REVIEW_BACKEND": "claude", "ANTHROPIC_API_KEY": ""},
    {"DAYDREAM_REVIEW_BACKEND": "claude", "ANTHROPIC_BASE_URL": "http://api.anthropic.com"},
])
def test_parse_reviewer_environment_rejects_invalid_claude_credentials(
    environment: dict[str, str],
) -> None:
    with pytest.raises(entrypoint.EntrypointError, match="ANTHROPIC"):
        entrypoint.parse_reviewer_environment(environment)


def test_parse_reviewer_environment_accepts_claude_proxy() -> None:
    parsed = entrypoint.parse_reviewer_environment({
        "DAYDREAM_REVIEW_BACKEND": "claude",
        "DAYDREAM_REVIEW_CASE_ID": "case-parser-proxy",
        "ANTHROPIC_API_KEY": "sk-ant",
        "ANTHROPIC_BASE_URL": "https://claude-proxy.internal/v1",
    })
    assert parsed.execution.backend.child_environment()["ANTHROPIC_BASE_URL"] == (
        "https://claude-proxy.internal/v1"
    )


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
        "DAYDREAM_REVIEW_CASE_ID": "case-parser-proxy",
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
        "DAYDREAM_REVIEW_CASE_ID": "case-parser-proxy",
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
    # The runner is stubbed at its production seam, but the entrypoint still
    # builds its controlled review config and runs the real publisher against
    # the runner's canonical merged-items output.
    (tmp_path / "a.py").write_text("x = 1\n")
    artifact = tmp_path / "logs" / "artifacts" / "review.json"
    seen: dict[str, Any] = {}

    async def _fake_run(config: Any, *, execution: Any) -> int:
        seen["config"] = config
        merged = Path(config.target) / ".daydream" / "deep" / "merged-items.json"
        merged.parent.mkdir(parents=True)
        merged.write_text(json.dumps({
            "items": [{
                "file": "a.py",
                "line": 1,
                "description": "The reviewed Python file contains an actionable issue.",
                "rationale": "The issue is reproducible in the frozen review snapshot.",
                "severity": "medium",
                "confidence": "HIGH",
            }],
        }))
        return 0

    monkeypatch.setattr("daydream.runner.run", _fake_run)

    rc = asyncio.run(entrypoint.main({
        "DAYDREAM_REVIEW_CASE_ID": "case-python",
        "DAYDREAM_REVIEW_ARTIFACT_PATH": str(artifact),
        "DAYDREAM_REVIEW_REPO_DIR": str(tmp_path),
        "DAYDREAM_REVIEW_BACKEND": "pi",
        "DAYDREAM_REVIEW_API_KEY": "sk-or-test",
        "DAYDREAM_REVIEW_BASE_URL": "https://openrouter.ai/api",
    }))
    assert rc == 0
    config = seen["config"]
    assert config.target == str(tmp_path)
    assert config.backend == "pi"
    assert config.output_mode == "review"
    assert config.non_interactive is True
    assert config.archive is False
    assert config.run_eval is False
    assert config.file_config is not None

    from daydream.benchmark.harbor import verifier_core as vc

    loaded = json.loads(artifact.read_text())
    assert loaded["case_id"] == "case-python"
    assert loaded["base_ref"] == "base"
    assert loaded["head_ref"] == "head"
    assert [finding["path"] for finding in loaded["findings"]] == ["a.py"]
    assert vc.validate_candidate_artifact(loaded)


def test_entrypoint_claude_backend_reaches_run_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "a.py").write_text("x = 1\n")
    artifact = tmp_path / "logs" / "artifacts" / "review.json"
    artifact.parent.mkdir(parents=True)
    seen: dict[str, Any] = {}

    async def _fake_run(config: Any, *, execution: Any) -> int:
        seen["backend"] = config.backend
        return 0

    def _fake_publish(**kwargs: Any) -> None:
        Path(kwargs["artifact_path"]).write_text("{}")

    monkeypatch.setattr("daydream.runner.run", _fake_run)
    monkeypatch.setattr(entrypoint, "publish_review", _fake_publish)

    rc = asyncio.run(entrypoint.main({
        "DAYDREAM_REVIEW_CASE_ID": "case-claude",
        "DAYDREAM_REVIEW_ARTIFACT_PATH": str(artifact),
        "DAYDREAM_REVIEW_REPO_DIR": str(tmp_path),
        "DAYDREAM_REVIEW_BACKEND": "claude",
        "ANTHROPIC_API_KEY": "sk-ant",
    }))
    assert rc == 0
    assert seen["backend"] == "claude"          # the wiring this issue is about


def test_entrypoint_env_has_no_skill_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The map passed to main is the only execution environment; a hostile
    # skill-dir value must not reach the run-owned backend carrier.
    artifact = tmp_path / "logs" / "artifacts" / "review.json"
    seen: dict[str, Any] = {}

    async def _fake_run(config: Any, *, execution: Any) -> int:
        seen["skill_dir"] = execution.backend.child_environment().get("DAYDREAM_SKILLS_DIR")
        merged = Path(config.target) / ".daydream" / "deep" / "merged-items.json"
        merged.parent.mkdir(parents=True)
        merged.write_text('{"items": []}')
        return 0

    monkeypatch.setattr("daydream.runner.run", _fake_run)

    rc = asyncio.run(entrypoint.main({
        "DAYDREAM_REVIEW_CASE_ID": "case-noskill",
        "DAYDREAM_REVIEW_ARTIFACT_PATH": str(artifact),
        "DAYDREAM_REVIEW_REPO_DIR": str(tmp_path),
        "DAYDREAM_REVIEW_BACKEND": "pi",
        "DAYDREAM_REVIEW_API_KEY": "sk-or-test",
        "DAYDREAM_REVIEW_BASE_URL": "https://openrouter.ai/api",
        "DAYDREAM_SKILLS_DIR": "/host/skills",
    }))
    assert rc == 0
    assert seen["skill_dir"] is None

    from daydream.benchmark.harbor import verifier_core as vc

    loaded = json.loads(artifact.read_text())
    assert loaded["case_id"] == "case-noskill"
    assert loaded["findings"] == []
    assert vc.validate_candidate_artifact(loaded) == []
