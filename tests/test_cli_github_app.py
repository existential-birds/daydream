"""CLI bootstrap uses local identity before any authenticated GitHub request.

Real Git, CLI event loop, workspace, JWT signing and artifacts; only GitHub's
subprocess transport and the model Backend are fake.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import jwt as pyjwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from daydream import cli, github_app
from tests.harness.fake_gh import FakeGh
from tests.harness.git_helpers import git
from tests.harness.rsa import generate_rsa_pem
from tests.test_github_app_integration import _minimal_backend

_TOKEN = "ghs_scoped_installation_test_token"
_AMBIENT = "ghp_unusable_ambient_test_token"


@dataclass
class _Transport:
    gh: FakeGh
    requests: list[tuple[Path, list[str], dict[str, str]]] = field(default_factory=list, repr=False)
    require_app: bool = True
    resolved_session: github_app.ResolvedGitHubSession | None = field(default=None, repr=False)


@pytest.fixture
def transport(fake_gh: FakeGh, feature_branch_repo: Path, monkeypatch: pytest.MonkeyPatch) -> _Transport:
    boundary = _Transport(fake_gh)
    fake_gh.set_response("GET", "app/installations", [
        {"id": 99, "account": {"login": "another-owner"}, "app_slug": "unrelated-app"},
        {"id": 71, "account": {"login": "AcMe"}, "app_slug": "review-app"},
    ])
    fake_gh.set_response("POST", "app/installations/71/access_tokens", {
        "token": _TOKEN, "expires_at": "2099-01-01T00:00:00Z",
    })
    fake_gh.set_response("GET", "user", {"login": "ambient-user"})
    fake_gh.set_response("repo-view", value="acme/widgets")
    fake_gh.serve_open_pr(feature_branch_repo)
    real_run = subprocess.run

    def request(args: Any, *positional: Any, **kwargs: Any) -> Any:
        if isinstance(args, (list, tuple)) and args and args[0] == "gh":
            environment = dict(kwargs.get("env") or os.environ)
            boundary.requests.append((Path(kwargs.get("cwd") or Path.cwd()), list(args), environment))
            if boundary.require_app and environment.get("GH_TOKEN") in (None, _AMBIENT):
                return subprocess.CompletedProcess(args, 1, stdout="", stderr="ambient gh authentication unavailable")
        return real_run(args, *positional, **kwargs)

    monkeypatch.setattr(subprocess, "run", request)
    for name in ("GH_REPO", "GH_HOST", "DAYDREAM_APP_ID", "DAYDREAM_APP_PRIVATE_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GH_TOKEN", _AMBIENT)
    monkeypatch.setenv("GITHUB_TOKEN", "ambient-alternate-token")
    monkeypatch.setenv("DAYDREAM_APP_ID", "42")
    monkeypatch.setenv("DAYDREAM_APP_PRIVATE_KEY", generate_rsa_pem())
    resolve_session = github_app.resolve_run_identity
    def observe_session(*args: Any, **kwargs: Any) -> github_app.ResolvedGitHubSession:
        session = resolve_session(*args, **kwargs)
        boundary.resolved_session = session
        return session
    monkeypatch.setattr(github_app, "resolve_run_identity", observe_session)
    monkeypatch.setattr("daydream.runner.create_backend", lambda *args, **kwargs: _minimal_backend())
    return boundary


def _run(repo: Path, trajectory: Path, *flags: str) -> int | str | None:
    with pytest.raises(SystemExit) as result:
        cli.main([str(repo), "--shallow", "--stack", "python", "--non-interactive",
                  "--trajectory", str(trajectory), "--no-capture-data", *flags])
    return result.value.code


def _assert_app_requests(transport: _Transport) -> None:
    listing, exchange = transport.requests[:2]
    jwt_token = listing[2]["GH_TOKEN"]
    authorization = f"Authorization: Bearer {jwt_token}"
    assert listing[1] == ["gh", "api", "-H", authorization, "--paginate", "--jq", "(.[]) | @json", "/app/installations"]
    assert exchange[1][:8] == ["gh", "api", "-H", authorization, "/app/installations/71/access_tokens",
                               "--method", "POST", "--input"]
    assert len(exchange[1]) == 9 and exchange[0] == listing[0]
    assert exchange[2]["GH_TOKEN"] == jwt_token
    private_key = serialization.load_pem_private_key(os.environ["DAYDREAM_APP_PRIVATE_KEY"].encode(), password=None)
    decoded = pyjwt.decode(jwt_token, cast(rsa.RSAPublicKey, private_key.public_key()), algorithms=["RS256"])
    assert decoded["iss"] == "42" and decoded["exp"] - decoded["iat"] == 600
    for _cwd, argv, environment in transport.requests:
        assert environment["GH_TOKEN"] != _AMBIENT, argv
        assert environment["GH_REPO"] == "acme/widgets"
        assert environment["GH_HOST"] == "github.com"
        assert "GITHUB_TOKEN" not in environment
        assert "DAYDREAM_APP_PRIVATE_KEY" not in environment
    assert os.environ["GH_TOKEN"] == _AMBIENT
    assert os.environ["GITHUB_TOKEN"] == "ambient-alternate-token"


def test_cli_app_failure_is_redacted_without_ambient_fallback(
    feature_branch_repo: Path, tmp_path: Path, transport: _Transport, capsys: pytest.CaptureFixture[str],
) -> None:
    git(feature_branch_repo, "remote", "add", "origin", "https://github.com/acme/widgets.git")
    secret = "ghs_sensitive_installation_response_token"
    transport.gh.set_response("POST", "app/installations/71/access_tokens", {"__error__": secret})

    assert _run(feature_branch_repo, tmp_path / "trajectory.json", "--comment", "--verbose") == 1

    captured = capsys.readouterr()
    assert "App token resolution failed" in captured.out + captured.err
    assert secret not in captured.out + captured.err
    _assert_app_requests(transport)
    assert not (feature_branch_repo / ".review-output.md").exists()


def test_cli_invalid_app_key_aborts_without_exposing_secret(
    feature_branch_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    transport: _Transport, capsys: pytest.CaptureFixture[str],
) -> None:
    git(feature_branch_repo, "remote", "add", "origin", "https://github.com/acme/widgets.git")
    secret = "malformed-private-key-secret"
    monkeypatch.setenv("DAYDREAM_APP_PRIVATE_KEY", secret)

    assert _run(feature_branch_repo, tmp_path / "trajectory.json", "--comment", "--verbose") == 1

    captured = capsys.readouterr()
    assert "App token resolution failed" in captured.out + captured.err
    assert secret not in captured.out + captured.err
    assert transport.requests == []
    assert not (feature_branch_repo / ".review-output.md").exists()
