"""Controlled Harbor wiring: no skill directory or Beagle probe, explicit profiles only.

Full Python and mixed-stack Harbor runs are tested separately.
"""
import json
from pathlib import Path
from typing import Any

import pytest

from daydream.benchmark.cli import _handle_benchmark_command
from daydream.benchmark.harbor import entrypoint, package as pkg, run as run_mod


def test_parse_reviewer_environment_rejects_non_openrouter_endpoint() -> None:
    with pytest.raises(entrypoint.EntrypointError, match="openrouter.ai"):
        entrypoint.parse_reviewer_environment({
            "DAYDREAM_REVIEW_API_KEY": "key", "DAYDREAM_REVIEW_BASE_URL": "https://example.com/api",
        })

def test_parse_reviewer_environment_refuses_unsupported_claude_before_credentials() -> None:
    with pytest.raises(entrypoint.EntrypointError, match="unsupported.*claude"):
        entrypoint.parse_reviewer_environment({"DAYDREAM_REVIEW_BACKEND": "claude"})

def test_host_reviewer_host_resolution_is_backend_aware() -> None:
    assert run_mod._reviewer_host_from_env({
        "DAYDREAM_REVIEW_BACKEND": "claude", "ANTHROPIC_BASE_URL": "https://claude-proxy.internal/v1",
    }) == "claude-proxy.internal"
    # Match the Anthropic SDK default when no base URL is configured.
    assert run_mod._reviewer_host_from_env({"DAYDREAM_REVIEW_BACKEND": "claude"}) == "api.anthropic.com"
    assert run_mod._reviewer_base_url_from_env({}) == ""
    assert run_mod._reviewer_host_from_env({"DAYDREAM_REVIEW_BASE_URL": "https://openrouter.ai/api"}) == "openrouter.ai"
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
               "reviewer_allowed_hosts": reviewer_hosts, "judge_data": "finding_text_and_location_only",
               "judge_allowed_hosts": ["127.0.0.1"], "archive": "disabled", "uploads": "disabled"}
    (ws / "benchmark.yaml").write_text(json.dumps({
        "schema_version": 1, "benchmark_id": "6c38dc0a", "created_at": "2026-08-21T12:00:00Z",
        "source": {"provider": "github", "hostname": "github.com",
                   "repository": "OWNER/REPO", "repository_id": None, "visibility": "unresolved"},
        "privacy": privacy, "pull_requests": [], "cases": []}))
    return ws


@pytest.mark.parametrize(("reviewer_hosts", "blocked"),
    [pytest.param(["claude-proxy.internal"], False, id="allowlisted"),
        pytest.param(["review.example"], True, id="non-allowlisted"),
    ],
)
def test_host_preflight_resolves_claude_proxy_against_reviewer_allowlist(
    tmp_path: Path, reviewer_hosts: list[str], blocked: bool
) -> None:
    # Reject non-allowlisted Claude proxies on the host, before a paid review.
    ws = _seed_host_ws(tmp_path, reviewer_hosts)

    def _docker_ok() -> pkg.DockerNetworkPolicyCapability:
        return pkg.DockerNetworkPolicyCapability(supported=True)

    errs = run_mod._preflight(ws, oracle=True, env={
        "DAYDREAM_REVIEW_BACKEND": "claude", "DAYDREAM_REVIEW_CASE_ID": "case-parser-proxy",
        "ANTHROPIC_API_KEY": "sk-ant", "ANTHROPIC_BASE_URL": "https://claude-proxy.internal/v1",
        "DAYDREAM_JUDGE_BASE_URL": "http://127.0.0.1:9",
    }, docker_ok=_docker_ok)
    reviewer_errs = [e for e in errs if "reviewer host" in e]
    if blocked:
        assert any("claude-proxy.internal" in e for e in reviewer_errs)
    else:
        assert not reviewer_errs
    assert not any("missing DAYDREAM_REVIEW_BASE_URL" in e for e in errs)

def test_host_run_gate_threads_claude_credentials_into_supervisor_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
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

def test_entrypoint_env_has_no_skill_dirs() -> None:
    parsed = entrypoint.parse_reviewer_environment({
        "DAYDREAM_REVIEW_CASE_ID": "case-noskill", "DAYDREAM_REVIEW_API_KEY": "sk-or-test",
        "DAYDREAM_REVIEW_BASE_URL": "https://openrouter.ai/api", "DAYDREAM_SKILLS_DIR": "/host/skills",
    })
    assert "DAYDREAM_SKILLS_DIR" not in parsed.execution.child_environment()
