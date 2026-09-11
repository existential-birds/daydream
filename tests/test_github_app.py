"""Unit tests for :mod:`daydream.github_app`.

Covers credential resolution, JWT minting, installation token exchange,
identity resolution, and gh token-env propagation through
:mod:`daydream.git_ops`, with mocked environment and GitHub API calls.
"""
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from daydream import git_ops, github_app
from daydream.github_app import (
    AppCredentials,
    GitHubAppError,
    _mint_installation_token,
    build_gh_env,
    exchange_manifest_code,
    get_app_metadata,
    mint_jwt,
    resolve_credentials,
    resolve_run_identity,
    resolve_user_identity,
)


@pytest.fixture(autouse=True)
def _block_real_gh(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep this suite hermetic even when the developer is authenticated."""
    real_run = subprocess.run

    def guarded_run(args: list[Any], *pargs: Any, **kwargs: Any) -> Any:
        if args and args[0] == "gh":
            raise AssertionError("test attempted to execute the real gh CLI")
        return real_run(args, *pargs, **kwargs)

    monkeypatch.setattr(subprocess, "run", guarded_run)


def test_static_auth_copies_environment_and_returns_fresh_mappings() -> None:
    source = {"PATH": "/tools", "GH_TOKEN": "ghs_static_token_1234567890"}

    auth = git_ops.StaticGitHubAuth(source)
    source["GH_TOKEN"] = "mutated"
    first = auth.environment_for_request()
    assert isinstance(first, dict)
    first["GH_TOKEN"] = "caller-mutated"

    assert auth.environment_for_request() == {
        "PATH": "/tools",
        "GH_TOKEN": "ghs_static_token_1234567890",
    }
    assert "ghs_static_token" not in repr(auth)


def test_run_gh_passes_exact_static_environment_without_ambient_merge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def spy_run(*args: list[Any], **kwargs: Any) -> Any:
        captured.update(kwargs)
        return subprocess.CompletedProcess(args[0], 0, stdout="", stderr="")

    monkeypatch.setenv("GITHUB_TOKEN", "github_pat_ambient_secret_1234567890")
    auth = git_ops.StaticGitHubAuth(
        {"GH_TOKEN": "ghs_explicit_token_1234567890", "PATH": "/usr/bin"}
    )
    with patch("subprocess.run", side_effect=spy_run):
        git_ops._run_gh(Path("/tmp"), ["version"], auth=auth)

    assert captured["env"] == {
        "GH_TOKEN": "ghs_explicit_token_1234567890",
        "PATH": "/usr/bin",
    }


def test_refreshing_auth_serializes_refresh_across_callers() -> None:
    refresh_calls = 0

    def refresh() -> tuple[Any, float]:
        nonlocal refresh_calls
        refresh_calls += 1
        return (
            git_ops.StaticGitHubAuth(
                {"PATH": "/tools", "GH_TOKEN": "ghs_fresh_token_1234567890"}
            ),
            float("inf"),
        )

    auth = git_ops.RefreshingGitHubAuth(
        git_ops.StaticGitHubAuth(
            {"PATH": "/tools", "GH_TOKEN": "ghs_expired_token_1234567890"}
        ),
        expires_at=0,
        refresh=refresh,
    )

    with ThreadPoolExecutor(max_workers=8) as executor:
        environments = list(executor.map(lambda _index: auth.environment_for_request(), range(8)))

    assert refresh_calls == 1
    assert all(env is not None and env["GH_TOKEN"] == "ghs_fresh_token_1234567890" for env in environments)


def test_run_gh_passes_none_for_inherited_auth() -> None:
    captured = {}

    def spy_run(*args: list[Any], **kwargs: Any) -> Any:
        captured.update(kwargs)
        return subprocess.CompletedProcess(args[0], 0, stdout="", stderr="")

    with patch("subprocess.run", side_effect=spy_run):
        git_ops._run_gh(
            Path("/tmp"), ["version"], auth=git_ops.INHERIT_GITHUB_AUTH
        )

    assert captured.get("env") is None


def test_run_gh_resolves_auth_once_for_the_complete_retry_sequence() -> None:
    calls = 0

    class CountingAuth:
        def environment_for_request(self) -> dict[str, str]:
            nonlocal calls
            calls += 1
            return {"PATH": "/tools", "GH_TOKEN": "ghs_one_token_1234567890"}

    with patch(
        "subprocess.run",
        side_effect=[
            subprocess.TimeoutExpired(["gh", "api", "/user"], 1),
            subprocess.CompletedProcess(["gh", "api", "/user"], 0, "{}", ""),
        ],
    ):
        git_ops._run_gh(
            Path("/tmp"),
            ["api", "/user"],
            auth=CountingAuth(),
            timeout=1,
            retries=1,
        )

    assert calls == 1


def test_refresh_failure_is_redacted_and_retains_last_good_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 800.0
    monkeypatch.setattr("daydream.git_ops.time.time", lambda: now)
    secret = "ghs_refresh_failure_secret_1234567890"

    def fail_refresh() -> tuple[Any, float]:
        raise RuntimeError(f"transport rejected {secret}")

    auth = git_ops.RefreshingGitHubAuth(
        git_ops.StaticGitHubAuth(
            {"PATH": "/tools", "GH_TOKEN": "ghs_last_good_token_1234567890"}
        ),
        expires_at=1_000,
        refresh=fail_refresh,
    )

    with pytest.raises(git_ops.GitError) as excinfo:
        auth.environment_for_request()
    assert secret not in str(excinfo.value)

    now = 0.0
    assert auth.environment_for_request() == {
        "PATH": "/tools",
        "GH_TOKEN": "ghs_last_good_token_1234567890",
    }


def test_refresh_rejects_changed_base_environment_and_keeps_prior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = 800.0
    monkeypatch.setattr("daydream.git_ops.time.time", lambda: now)
    auth = git_ops.RefreshingGitHubAuth(
        git_ops.StaticGitHubAuth(
            {"PATH": "/original", "GH_TOKEN": "ghs_prior_token_1234567890"}
        ),
        expires_at=1_000,
        refresh=lambda: (
            git_ops.StaticGitHubAuth(
                {"PATH": "/changed", "GH_TOKEN": "ghs_new_token_1234567890"}
            ),
            2_000,
        ),
    )

    with pytest.raises(git_ops.GitError, match="base environment"):
        auth.environment_for_request()

    now = 0.0
    assert auth.environment_for_request()["PATH"] == "/original"


def test_two_refreshing_sessions_keep_subprocess_credentials_isolated() -> None:
    captured: list[dict[str, str] | None] = []
    refresh_calls = {"a": 0, "b": 0}

    def session(name: str) -> Any:
        def refresh() -> tuple[Any, float]:
            refresh_calls[name] += 1
            return (
                git_ops.StaticGitHubAuth(
                    {
                        "PATH": f"/{name}/tools",
                        "GH_TOKEN": f"ghs_{name}_fresh_token_1234567890",
                    }
                ),
                float("inf"),
            )

        return git_ops.RefreshingGitHubAuth(
            git_ops.StaticGitHubAuth(
                {
                    "PATH": f"/{name}/tools",
                    "GH_TOKEN": f"ghs_{name}_expired_token_1234567890",
                }
            ),
            expires_at=0,
            refresh=refresh,
        )

    def spy_run(*args: list[Any], **kwargs: Any) -> Any:
        captured.append(kwargs["env"])
        return subprocess.CompletedProcess(args[0], 0, stdout="{}", stderr="")

    first = session("a")
    second = session("b")
    with patch("subprocess.run", side_effect=spy_run):
        git_ops._run_gh(Path("/tmp"), ["api", "/user"], auth=first)
        git_ops._run_gh(Path("/tmp"), ["api", "/user"], auth=second)
        git_ops._run_gh(Path("/tmp"), ["api", "/user"], auth=first)

    assert [env["GH_TOKEN"] for env in captured if env is not None] == [
        "ghs_a_fresh_token_1234567890",
        "ghs_b_fresh_token_1234567890",
        "ghs_a_fresh_token_1234567890",
    ]
    assert refresh_calls == {"a": 1, "b": 1}


def test_resolve_credentials_returns_none_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DAYDREAM_APP_ID", raising=False)
    monkeypatch.delenv("DAYDREAM_APP_PRIVATE_KEY", raising=False)
    assert resolve_credentials() is None


def test_resolve_credentials_parses_both(monkeypatch: pytest.MonkeyPatch) -> None:
    pem = "-----BEGIN RSA PRIVATE KEY-----\nx\n-----END RSA PRIVATE KEY-----"
    monkeypatch.setenv("DAYDREAM_APP_ID", "12345")
    monkeypatch.setenv("DAYDREAM_APP_PRIVATE_KEY", pem)
    creds = resolve_credentials()
    assert creds == AppCredentials(app_id=12345, private_key=pem)


def test_secret_credential_values_are_hidden_from_repr() -> None:
    private_key = "unstructured-private-key-secret"
    token = "unstructured-installation-token-secret"

    credentials = AppCredentials(app_id=12345, private_key=private_key)
    installation = github_app._InstallationToken(
        token=token,
        identity="app[bot]",
        expires_at=1.0,
    )

    assert private_key not in repr(credentials)
    assert token not in repr(installation)


def test_resolve_credentials_raises_on_partial_id_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DAYDREAM_APP_ID", "12345")
    monkeypatch.delenv("DAYDREAM_APP_PRIVATE_KEY", raising=False)
    with pytest.raises(ValueError, match="DAYDREAM_APP_PRIVATE_KEY"):
        resolve_credentials()


def test_resolve_credentials_raises_on_partial_key_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DAYDREAM_APP_ID", raising=False)
    monkeypatch.setenv("DAYDREAM_APP_PRIVATE_KEY", "x")
    with pytest.raises(ValueError, match="DAYDREAM_APP_ID"):
        resolve_credentials()


def test_resolve_credentials_raises_on_non_integer_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DAYDREAM_APP_ID", "not-an-int")
    monkeypatch.setenv("DAYDREAM_APP_PRIVATE_KEY", "x")
    with pytest.raises(ValueError, match="DAYDREAM_APP_ID"):
        resolve_credentials()


def test_build_gh_env_returns_complete_sanitized_environment() -> None:
    base = {
        "PATH": "/tools",
        "GH_HOST": "github.example.test",
        "GH_TOKEN": "ambient-gh",
        "GITHUB_TOKEN": "ambient-github",
        "GH_ENTERPRISE_TOKEN": "ambient-enterprise",
        "GITHUB_ENTERPRISE_TOKEN": "ambient-github-enterprise",
        "DAYDREAM_APP_PRIVATE_KEY": "private-key",
    }

    env = build_gh_env("ghs_explicit", base_environment=base)

    assert env == {
        "PATH": "/tools",
        "GH_HOST": "github.example.test",
        "GH_TOKEN": "ghs_explicit",
    }


def test_session_dtos_hide_and_ignore_execution_credentials() -> None:
    identity = github_app.GitHubIdentity("app[bot]")
    first = github_app.ResolvedGitHubSession(
        identity,
        github_app.GitHubExecutionInput(
            git_ops.StaticGitHubAuth({"GH_TOKEN": "ghs_first_secret_1234567890"})
        ),
    )
    second = github_app.ResolvedGitHubSession(
        identity,
        github_app.GitHubExecutionInput(
            git_ops.StaticGitHubAuth({"GH_TOKEN": "ghs_second_secret_1234567890"})
        ),
    )

    assert first == second
    assert "secret" not in repr(first)
    assert "secret" not in repr(first.execution)


def test_mint_jwt_is_rs256_with_expected_claims() -> None:
    import jwt as pyjwt
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pem = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()

    token = mint_jwt(12345, pem)
    decoded = pyjwt.decode(token, key.public_key(), algorithms=["RS256"])
    assert decoded["iss"] == "12345" or decoded["iss"] == 12345
    assert decoded["exp"] - decoded["iat"] <= 600


def test_build_app_jwt_auth_uses_static_sanitized_environment() -> None:
    auth, headers = github_app.build_app_jwt_auth(
        12345,
        _TEST_PEM,
        base_environment={
            "PATH": "/tools",
            "GH_HOST": "github.example.test",
            "GITHUB_TOKEN": "ambient-secret",
        },
    )

    environment = auth.environment_for_request()
    assert environment["PATH"] == "/tools"
    assert environment["GH_HOST"] == "github.example.test"
    assert environment["GH_TOKEN"].startswith("ey")
    assert "GITHUB_TOKEN" not in environment
    assert headers == {"Authorization": f"Bearer {environment['GH_TOKEN']}"}


def _real_pem() -> str:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()


def test_mint_installation_token_happy_path() -> None:
    pem = _real_pem()
    calls = []

    def fake_gh_api(repo: Any, endpoint: Any, **kwargs: Any) -> Any:
        calls.append((endpoint, kwargs))
        if "access_tokens" in endpoint:
            return {"token": "ghs_minted", "expires_at": "2099-01-01T00:00:00Z"}
        return [{"id": 999, "account": {"login": "MyOrg"}, "app_slug": "daydream-bot"}]

    with patch("daydream.git_ops.gh_api", side_effect=fake_gh_api):
        minted = _mint_installation_token(Path("/tmp"), 12345, pem, "myorg", "myrepo")

    assert minted.token == "ghs_minted"
    assert minted.identity == "daydream-bot[bot]"
    # Two API calls: list installations, then exchange. No extra GET /app.
    assert len(calls) == 2
    # GitHub only accepts App JWTs via the Bearer scheme, so both calls must
    # carry an explicit Authorization header (gh's GH_TOKEN uses token scheme).
    for _, kwargs in calls:
        assert kwargs["headers"]["Authorization"].startswith("Bearer ey")
        assert isinstance(kwargs["auth"], git_ops.StaticGitHubAuth)
    exchange_endpoint, exchange_kwargs = calls[1]
    assert exchange_endpoint == "/app/installations/999/access_tokens"
    assert exchange_kwargs["method"] == "POST"
    assert exchange_kwargs["input_data"] == {"repositories": ["myrepo"]}


def test_mint_installation_token_missing_app_slug_yields_unknown_identity() -> None:
    """A missing/empty app_slug is cosmetic: the mint succeeds, identity is 'unknown'."""
    pem = _real_pem()

    def fake_gh_api(repo: Any, endpoint: Any, **kwargs: Any) -> Any:
        if "access_tokens" in endpoint:
            return {"token": "ghs_minted", "expires_at": "2099-01-01T00:00:00Z"}
        return [{"id": 999, "account": {"login": "myorg"}}]

    with patch("daydream.git_ops.gh_api", side_effect=fake_gh_api):
        minted = _mint_installation_token(Path("/tmp"), 12345, pem, "myorg", "myrepo")

    assert minted.token == "ghs_minted"
    assert minted.identity == "unknown"


def test_mint_installation_token_no_matching_installation() -> None:
    pem = _real_pem()

    with patch("daydream.git_ops.gh_api", return_value=[{"id": 1, "account": {"login": "other"}}]):
        with pytest.raises(ValueError, match="installation"):
            _mint_installation_token(Path("/tmp"), 12345, pem, "myorg", "myrepo")


def test_mint_installation_token_wraps_gh_api_failure() -> None:
    """A gh api failure (GitError) surfaces as the module's ValueError abort channel."""
    pem = _real_pem()

    with patch("daydream.git_ops.gh_api", side_effect=git_ops.GitError("HTTP 401")):
        with pytest.raises(ValueError, match="failed to list App installations"):
            _mint_installation_token(Path("/tmp"), 12345, pem, "myorg", "myrepo")


def test_mint_installation_token_uses_one_explicit_jwt_auth() -> None:
    """Both App calls use one static JWT auth without ambient state."""
    pem = _real_pem()
    seen_auth: list[Any] = []

    def fake_gh_api(repo: Any, endpoint: Any, **kwargs: Any) -> Any:
        seen_auth.append(kwargs["auth"])
        if "access_tokens" in endpoint:
            return {"token": "ghs_x", "expires_at": "2099-01-01T00:00:00Z"}
        return [{"id": 7, "account": {"login": "myorg"}, "app_slug": "daydream-bot"}]

    with patch("daydream.git_ops.gh_api", side_effect=fake_gh_api):
        _mint_installation_token(Path("/tmp"), 12345, pem, "myorg", "myrepo")

    assert len(seen_auth) == 2
    assert seen_auth[0] is seen_auth[1]
    jwt_environment = seen_auth[0].environment_for_request()
    assert jwt_environment["GH_TOKEN"].startswith("ey")
    assert "ghs_x" not in jwt_environment["GH_TOKEN"]


def test_resolve_run_identity_refreshes_installation_token_after_expiry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A long-running review remints before its next GitHub request."""
    monkeypatch.setenv("DAYDREAM_APP_ID", "12345")
    monkeypatch.setenv("DAYDREAM_APP_PRIVATE_KEY", _TEST_PEM)
    minted = 0
    captured = {}

    def fake_gh_api(repo: Any, endpoint: str, **kwargs: Any) -> Any:
        nonlocal minted
        if endpoint == "/app/installations":
            return [{"id": 999, "account": {"login": "myorg"}, "app_slug": "daydream-bot"}]
        assert endpoint == "/app/installations/999/access_tokens"
        minted += 1
        if minted == 1:
            return {"token": "ghs_expired", "expires_at": "1970-01-01T00:00:00Z"}
        return {"token": "ghs_fresh", "expires_at": "9999-01-01T00:00:00Z"}

    def spy_run(*args: list[Any], **kwargs: Any) -> Any:
        captured.update(kwargs)
        return subprocess.CompletedProcess(args[0], 0, stdout="{}", stderr="")

    with patch("daydream.git_ops.gh_api", side_effect=fake_gh_api):
        session = resolve_run_identity(tmp_path, "myorg/myrepo", is_posting=True)

        with patch("subprocess.run", side_effect=spy_run):
            git_ops._run_gh(tmp_path, ["api", "/user"], auth=session.execution.auth)

    assert session.identity == github_app.GitHubIdentity("daydream-bot[bot]")
    assert minted == 2
    assert captured["env"]["GH_TOKEN"] == "ghs_fresh"


def test_resolve_run_identity_skips_minting_when_not_posting(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """With App credentials configured and owner/repo resolvable, is_posting=False
    must never attempt minting — a read-only run has no need for a scoped token."""
    monkeypatch.setenv("DAYDREAM_APP_ID", "12345")
    monkeypatch.setenv("DAYDREAM_APP_PRIVATE_KEY", _TEST_PEM)

    def fake_gh_api(repo: Any, endpoint: str, **kwargs: Any) -> dict[str, Any]:
        if endpoint == "/user":
            return {"login": "personal-user"}
        raise AssertionError(f"minting must not be attempted when is_posting=False (hit {endpoint})")

    with patch("daydream.git_ops.gh_api", side_effect=fake_gh_api):
        session = resolve_run_identity(tmp_path, "myorg/myrepo", is_posting=False)

    assert session.identity == github_app.GitHubIdentity("personal-user")
    assert session.execution.auth is git_ops.INHERIT_GITHUB_AUTH


def test_resolve_run_identity_binds_explicit_base_without_app_credentials(
    tmp_path: Path,
) -> None:
    base = {"PATH": "/isolated/tools", "GH_HOST": "github.example.test"}
    seen_auth: list[Any] = []

    def identity(_repo: Path, *, auth: Any) -> str:
        seen_auth.append(auth)
        return "isolated-user"

    with patch("daydream.github_app.resolve_user_identity", side_effect=identity):
        session = resolve_run_identity(
            tmp_path,
            None,
            is_posting=False,
            base_environment=base,
        )

    assert session.identity == github_app.GitHubIdentity("isolated-user")
    assert seen_auth == [session.execution.auth]
    assert session.execution.auth.environment_for_request() == base


def test_resolve_run_identity_validates_explicit_credentials_before_read_only_fallback(
    tmp_path: Path,
) -> None:
    with pytest.raises(GitHubAppError, match="DAYDREAM_APP_PRIVATE_KEY"):
        resolve_run_identity(
            tmp_path,
            None,
            is_posting=False,
            base_environment={"PATH": "/tools", "DAYDREAM_APP_ID": "12345"},
        )


def test_resolve_run_identity_redacts_arbitrary_initial_mint_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    secret = "unstructured-new-credential-secret"
    monkeypatch.setenv("DAYDREAM_APP_ID", "12345")
    monkeypatch.setenv("DAYDREAM_APP_PRIVATE_KEY", _TEST_PEM)

    with patch(
        "daydream.github_app._mint_installation_token",
        side_effect=RuntimeError(f"transport exposed {secret}"),
    ):
        with pytest.raises(GitHubAppError) as excinfo:
            resolve_run_identity(tmp_path, "myorg/myrepo", is_posting=True)

    assert str(excinfo.value) == "App token resolution failed"
    assert secret not in str(excinfo.value)
    assert excinfo.value.__cause__ is None


def test_resolve_user_identity_returns_login(tmp_path: Path) -> None:
    """resolve_user_identity reads the current gh-authenticated user."""
    with patch("daydream.git_ops.gh_api", return_value={"login": "personal-user"}):
        assert resolve_user_identity(tmp_path) == "personal-user"


def test_resolve_user_identity_returns_unknown_on_failure(tmp_path: Path) -> None:
    """A failed user lookup is non-fatal — return 'unknown', never raise."""
    with patch("daydream.git_ops.gh_api", side_effect=git_ops.GitError("boom")):
        assert resolve_user_identity(tmp_path) == "unknown"


_TEST_PEM = _real_pem()


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


def test_exchange_manifest_code_raises_when_id_missing() -> None:
    """A conversion missing the App id aborts naming the field — no placeholder."""
    with patch("daydream.git_ops.gh_api", return_value={"pem": "x", "slug": "acme-bot"}):
        with pytest.raises(GitHubAppError, match="id"):
            exchange_manifest_code(Path("."), "abc123")


def test_exchange_manifest_code_raises_when_id_not_int() -> None:
    """A non-integer App id aborts naming the field — never coerce a placeholder."""
    with patch("daydream.git_ops.gh_api", return_value={"id": "not-int", "pem": "x", "slug": "s"}):
        with pytest.raises(GitHubAppError, match="id"):
            exchange_manifest_code(Path("."), "abc123")


def test_exchange_manifest_code_raises_when_pem_missing() -> None:
    """A conversion missing the PEM aborts naming the field — no placeholder key."""
    with patch("daydream.git_ops.gh_api", return_value={"id": 42, "slug": "acme-bot"}):
        with pytest.raises(GitHubAppError, match="pem"):
            exchange_manifest_code(Path("."), "abc123")


def test_exchange_manifest_code_wraps_gh_api_failure() -> None:
    """A gh api failure (GitError) surfaces as GitHubAppError, never silent."""
    with patch("daydream.git_ops.gh_api", side_effect=git_ops.GitError("HTTP 422")):
        with pytest.raises(GitHubAppError):
            exchange_manifest_code(Path("."), "abc123")


def test_get_app_metadata_returns_permissions() -> None:
    """get_app_metadata mints a JWT and returns the parsed /app object."""
    with patch(
        "daydream.git_ops.gh_api",
        side_effect=lambda *a, **k: {"permissions": {"pull_requests": "write"}, "slug": "acme-bot"},
    ):
        meta = get_app_metadata(Path("."), 42, _TEST_PEM)
    assert meta["permissions"]["pull_requests"] == "write"


def test_get_app_metadata_uses_bearer_jwt_and_explicit_auth() -> None:
    """The /app call carries matching Bearer headers and static JWT auth."""
    seen = {}

    def fake_gh_api(repo: Any, endpoint: Any, **kw: dict[str, Any]) -> dict[str, Any]:
        seen["endpoint"] = endpoint
        seen["headers"] = kw.get("headers")
        seen["auth"] = kw.get("auth")
        return {"permissions": {"pull_requests": "write"}, "slug": "acme-bot"}

    with patch("daydream.git_ops.gh_api", side_effect=fake_gh_api):
        get_app_metadata(Path("."), 42, _TEST_PEM)

    assert seen["endpoint"] == "/app"
    assert seen["headers"]["Authorization"].startswith("Bearer ey")
    environment = seen["auth"].environment_for_request()
    assert seen["headers"]["Authorization"] == f"Bearer {environment['GH_TOKEN']}"


def test_get_app_metadata_does_not_mutate_refreshing_installation_auth() -> None:
    """A separate App JWT call leaves the installation session refreshable."""
    captured = {}
    refresh_calls = 0

    def refresh() -> tuple[Any, float]:
        nonlocal refresh_calls
        refresh_calls += 1
        return (
            git_ops.StaticGitHubAuth(
                {"PATH": "/tools", "GH_TOKEN": "ghs_fresh_token_1234567890"}
            ),
            float("inf"),
        )

    def spy_run(*args: list[Any], **kwargs: Any) -> Any:
        captured.update(kwargs)
        return subprocess.CompletedProcess(args[0], 0, stdout="", stderr="")

    installation_auth = git_ops.RefreshingGitHubAuth(
        git_ops.StaticGitHubAuth(
            {"PATH": "/tools", "GH_TOKEN": "ghs_expired_token_1234567890"}
        ),
        expires_at=0,
        refresh=refresh,
    )
    with patch("daydream.git_ops.gh_api", return_value={"permissions": {}, "slug": "acme-bot"}):
        get_app_metadata(Path("."), 42, _TEST_PEM)
    with patch("subprocess.run", side_effect=spy_run):
        git_ops._run_gh(
            Path("/tmp"),
            ["api", "/user"],
            auth=installation_auth,
        )

    assert refresh_calls == 1
    assert captured["env"]["GH_TOKEN"] == "ghs_fresh_token_1234567890"


def test_get_app_metadata_wraps_gh_api_failure() -> None:
    """A gh api failure (GitError) surfaces as GitHubAppError, never silent."""
    with patch("daydream.git_ops.gh_api", side_effect=git_ops.GitError("HTTP 401")):
        with pytest.raises(GitHubAppError):
            get_app_metadata(Path("."), 42, _TEST_PEM)
