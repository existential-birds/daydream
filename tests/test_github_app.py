"""Unit tests for :mod:`daydream.github_app`.

Covers credential resolution, JWT minting, installation token exchange,
identity resolution, and gh token-env propagation through
:mod:`daydream.git_ops`, with mocked environment and GitHub API calls.
"""

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from daydream import git_ops, github_app
from daydream.git_ops import process as git_process
from daydream.github_app import (
    AppCredentials,
    GitHubAppError,
    _mint_installation_token,
    exchange_manifest_code,
    get_app_metadata,
    resolve_credentials,
    resolve_run_identity,
)
from tests.harness.fake_gh import block_real_gh
from tests.harness.rsa import generate_rsa_pem


@pytest.fixture(autouse=True)
def _block_real_gh(monkeypatch: pytest.MonkeyPatch) -> None:
    block_real_gh(monkeypatch)


def test_run_gh_resolves_auth_once_for_the_complete_retry_sequence() -> None:
    calls = 0
    class CountingAuth:
        def environment_for_request(self) -> dict[str, str]:
            nonlocal calls
            calls += 1
            return {"PATH": "/tools", "GH_TOKEN": "ghs_one_token_1234567890"}
    with patch("subprocess.run",
        side_effect=[subprocess.TimeoutExpired(["gh", "api", "/user"], 1),
            subprocess.CompletedProcess(["gh", "api", "/user"], 0, "{}", ""),
        ],
    ):
        git_process._run_gh(Path("/tmp"), ["api", "/user"], auth=CountingAuth(), timeout=1, retries=1,)

    assert calls == 1


def test_refresh_failure_is_redacted_and_never_replaces_the_prior_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 800.0
    monkeypatch.setattr("daydream.git_ops.auth.time.time", lambda: now)
    secret = "ghs_refresh_failure_secret_1234567890"
    def fail_refresh() -> tuple[Any, float]:
        raise RuntimeError(f"transport rejected {secret}")
    auth = git_ops.RefreshingGitHubAuth(
        git_ops.StaticGitHubAuth({"PATH": "/tools", "GH_TOKEN": "ghs_last_good_token_1234567890"}), expires_at=1_000,
        refresh=fail_refresh,
    )
    changed = git_ops.RefreshingGitHubAuth(
        git_ops.StaticGitHubAuth({"PATH": "/original", "GH_TOKEN": "ghs_prior_token_1234567890"}), expires_at=1_000,
        refresh=lambda: (git_ops.StaticGitHubAuth({"PATH": "/changed", "GH_TOKEN": "ghs_new_token_1234567890"}), 2_000),
    )
    with pytest.raises(git_ops.GitError) as excinfo:
        auth.environment_for_request()
    assert secret not in str(excinfo.value)
    with pytest.raises(git_ops.GitError, match="base environment"):
        changed.environment_for_request()
    now = 0.0
    assert auth.environment_for_request() == {"PATH": "/tools", "GH_TOKEN": "ghs_last_good_token_1234567890"}
    assert changed.environment_for_request()["PATH"] == "/original"


def test_app_credentials_fail_closed_and_hide_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Partial config, unmatched installations and gh failures abort typed; secrets never reach repr."""
    for env, match in [
        ({"DAYDREAM_APP_PRIVATE_KEY": "x"}, "DAYDREAM_APP_ID"),
        ({"DAYDREAM_APP_ID": "not-an-int", "DAYDREAM_APP_PRIVATE_KEY": "x"}, "DAYDREAM_APP_ID"),
    ]:
        with monkeypatch.context() as scoped:
            scoped.delenv("DAYDREAM_APP_ID", raising=False)
            scoped.delenv("DAYDREAM_APP_PRIVATE_KEY", raising=False)
            for key, value in env.items():
                scoped.setenv(key, value)
            with pytest.raises(ValueError, match=match):
                resolve_credentials()
    private_key = "unstructured-private-key-secret"
    token = "unstructured-installation-token-secret"
    assert private_key not in repr(AppCredentials(app_id=12345, private_key=private_key))
    assert token not in repr(github_app._InstallationToken(token=token, identity="app[bot]", expires_at=1.0))
    identity = github_app.GitHubIdentity("app[bot]")
    sessions = [github_app.ResolvedGitHubSession(identity, github_app.GitHubExecutionInput(
        git_ops.StaticGitHubAuth({"GH_TOKEN": f"ghs_{name}_secret_1234567890"}))) for name in ("first", "second")]
    assert sessions[0] == sessions[1]
    assert "secret" not in repr(sessions[0]) and "secret" not in repr(sessions[0].execution)
    with patch("daydream.git_ops.gh_api", return_value=[{"id": 1, "account": {"login": "other"}}]):
        with pytest.raises(ValueError, match="installation"):
            _mint_installation_token(Path("/tmp"), 12345, _TEST_PEM, "myorg", "myrepo")
    with patch("daydream.git_ops.gh_api", side_effect=git_ops.GitError("HTTP 401")):
        with pytest.raises(ValueError, match="failed to list App installations"):
            _mint_installation_token(Path("/tmp"), 12345, _TEST_PEM, "myorg", "myrepo")
        with pytest.raises(GitHubAppError):
            get_app_metadata(Path("."), 42, _TEST_PEM)
        with pytest.raises(GitHubAppError):
            exchange_manifest_code(Path("."), "abc123")
    # An arbitrary initial mint failure is replaced by a fixed, cause-free diagnostic.
    monkeypatch.setenv("DAYDREAM_APP_ID", "12345")
    monkeypatch.setenv("DAYDREAM_APP_PRIVATE_KEY", _TEST_PEM)
    with patch("daydream.github_app._mint_installation_token",
               side_effect=RuntimeError("transport exposed unstructured-new-credential-secret")):
        with pytest.raises(GitHubAppError) as excinfo:
            resolve_run_identity(Path("/tmp"), "myorg/myrepo", is_posting=True)
    assert str(excinfo.value) == "App token resolution failed" and excinfo.value.__cause__ is None
    # A conversion missing the id or PEM aborts naming the field; no placeholder.
    for payload, match in [({"pem": "x", "slug": "acme-bot"}, "id"), ({"id": 42, "slug": "acme-bot"}, "pem")]:
        with patch("daydream.git_ops.gh_api", return_value=payload):
            with pytest.raises(GitHubAppError, match=match):
                exchange_manifest_code(Path("."), "abc123")


def test_resolve_run_identity_refreshes_installation_token_after_expiry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """A long-running review remints before its next GitHub request."""
    monkeypatch.setenv("DAYDREAM_APP_ID", "12345")
    monkeypatch.setenv("DAYDREAM_APP_PRIVATE_KEY", _TEST_PEM)
    base = {
        "DAYDREAM_APP_ID": "12345",
        "DAYDREAM_APP_PRIVATE_KEY": _TEST_PEM,
        "GH_REPO": "conflicting/repository",
        "GH_HOST": "github.com",
        "PATH": "/captured/tools",
        "GH_TOKEN": "ambient-gh",
        "GITHUB_TOKEN": "ambient-github",
        "GH_ENTERPRISE_TOKEN": "ambient-enterprise",
        "GITHUB_ENTERPRISE_TOKEN": "ambient-github-enterprise",
    }
    minted = 0
    captured: dict[str, Any] = {}

    def fake_gh_api(repo: Any, endpoint: str, **kwargs: Any) -> Any:
        nonlocal minted
        environment = kwargs["auth"].environment_for_request()
        assert environment["GH_REPO"] == "myorg/myrepo"
        assert environment["GH_HOST"] == "github.com"
        assert environment["PATH"] == "/captured/tools"
        assert set(environment) == {"PATH", "GH_REPO", "GH_HOST", "DAYDREAM_APP_ID", "GH_TOKEN"}
        assert environment["GH_TOKEN"].startswith("ey")
        if endpoint == "/app/installations":
            return [{"id": 999, "account": {"login": "myorg"}, "app_slug": "daydream-bot"}]
        assert endpoint == "/app/installations/999/access_tokens"
        assert kwargs["input_data"] == {"repositories": ["myrepo"]}
        minted += 1
        if minted == 1:
            return {"token": "ghs_expired", "expires_at": "1970-01-01T00:00:00Z"}
        return {"token": "ghs_fresh", "expires_at": "9999-01-01T00:00:00Z"}

    def spy_run(*args: list[Any], **kwargs: Any) -> Any:
        captured.update(kwargs)
        return subprocess.CompletedProcess(args[0], 0, stdout="{}", stderr="")

    with patch("daydream.git_ops.gh_api", side_effect=fake_gh_api):
        session = resolve_run_identity(tmp_path, "myorg/myrepo", is_posting=True, base_environment=base)
        base["GH_REPO"] = "mutated/repository"
        base["GH_HOST"] = "foreign.invalid"
        monkeypatch.setenv("GH_REPO", "ambient/repository")
        monkeypatch.setenv("GH_HOST", "foreign.invalid")
        with patch("subprocess.run", side_effect=spy_run):
            git_process._run_gh(tmp_path, ["api", "/user"], auth=session.execution.auth)

    assert session.identity == github_app.GitHubIdentity("daydream-bot[bot]")
    assert minted == 2
    assert captured["env"]["GH_TOKEN"] == "ghs_fresh"

    assert captured["env"]["GH_REPO"] == "myorg/myrepo"
    assert captured["env"]["GH_HOST"] == "github.com"
    assert captured["env"]["PATH"] == "/captured/tools"


def test_resolve_run_identity_validates_explicit_credentials_before_read_only_fallback(tmp_path: Path,) -> None:
    """Partial App config fails even read-only; complete config never mints for a read-only run."""
    with pytest.raises(GitHubAppError, match="DAYDREAM_APP_PRIVATE_KEY"):
        resolve_run_identity(
            tmp_path, None, is_posting=False, base_environment={"PATH": "/tools", "DAYDREAM_APP_ID": "12345"},
        )
    base = {"PATH": "/tools", "DAYDREAM_APP_ID": "12345", "DAYDREAM_APP_PRIVATE_KEY": _TEST_PEM, "GH_TOKEN": "ambient"}
    with patch("daydream.github_app._mint_installation_token", side_effect=AssertionError("read-only run minted")):
        session = resolve_run_identity(tmp_path, "acme/widgets", is_posting=False, base_environment=base)
    environment = session.execution.auth.environment_for_request()
    assert environment is not None and environment["GH_TOKEN"] == "ambient"


_TEST_PEM = generate_rsa_pem()


def test_exchange_manifest_code_returns_credentials_and_slug() -> None:
    """POST the manifest code conversion and read id/pem/slug into AppCredentials."""
    auth = git_ops.StaticGitHubAuth({"PATH": "/tools"})
    def fake_gh_api(repo: Any, endpoint: str, **kw: Any) -> dict[str, Any]:
        assert endpoint == "/app-manifests/abc123/conversions" and kw["method"] == "POST"
        assert kw["auth"] is auth
        return {"id": 42, "pem": "-----BEGIN RSA PRIVATE KEY-----\nx\n", "slug": "acme-bot"}
    with patch("daydream.git_ops.gh_api", side_effect=fake_gh_api):
        creds, slug = exchange_manifest_code(Path("."), "abc123", auth=auth)
    assert creds.app_id == 42 and "BEGIN RSA" in creds.private_key and slug == "acme-bot"
