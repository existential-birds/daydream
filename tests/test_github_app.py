"""Unit tests for :mod:`daydream.github_app`.

Covers credential resolution, JWT minting, installation token exchange,
identity resolution, and gh token-env propagation through
:mod:`daydream.git_ops`, with mocked environment and GitHub API calls.
"""

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from daydream import git_ops
from daydream.github_app import (
    AppCredentials,
    GitHubAppError,
    build_gh_env,
    exchange_manifest_code,
    get_app_metadata,
    mint_installation_token,
    mint_jwt,
    resolve_credentials,
    resolve_run_identity,
    resolve_user_identity,
)


@pytest.fixture(autouse=True)
def _block_real_gh(monkeypatch):
    """Keep this suite hermetic even when the developer is authenticated."""
    real_run = subprocess.run

    def guarded_run(args, *pargs, **kwargs):
        if args and args[0] == "gh":
            raise AssertionError("test attempted to execute the real gh CLI")
        return real_run(args, *pargs, **kwargs)

    monkeypatch.setattr(subprocess, "run", guarded_run)


def test_run_gh_injects_token_env_when_set():
    """When the token-env singleton is set, _run_gh passes it to subprocess.run."""
    captured = {}

    def spy_run(*args, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(args[0], 0, stdout="", stderr="")

    git_ops.set_gh_token_env({"GH_TOKEN": "ghs_test123", "PATH": "/usr/bin"})
    try:
        with patch("subprocess.run", side_effect=spy_run):
            git_ops._run_gh(Path("/tmp"), ["version"])
    finally:
        git_ops.reset_gh_token_env()

    assert captured["env"]["GH_TOKEN"] == "ghs_test123"


def test_run_gh_refreshes_expired_token_before_request():
    """An expired App token is replaced before the gh subprocess starts."""
    captured = {}
    refresh_calls = 0

    def refresh():
        nonlocal refresh_calls
        refresh_calls += 1
        return {"GH_TOKEN": "ghs_fresh"}, float("inf")

    def spy_run(*args, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(args[0], 0, stdout="", stderr="")

    git_ops.set_gh_token_env(
        {"GH_TOKEN": "ghs_expired"},
        expires_at=0,
        refresh=refresh,
    )
    try:
        with patch("subprocess.run", side_effect=spy_run):
            git_ops._run_gh(Path("/tmp"), ["api", "/user"])
    finally:
        git_ops.reset_gh_token_env()

    assert refresh_calls == 1
    assert captured["env"]["GH_TOKEN"] == "ghs_fresh"


def test_run_gh_passes_none_env_when_unset():
    """With no token-env set, _run_gh passes env=None (parent inheritance)."""
    captured = {}

    def spy_run(*args, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(args[0], 0, stdout="", stderr="")

    git_ops.reset_gh_token_env()
    with patch("subprocess.run", side_effect=spy_run):
        git_ops._run_gh(Path("/tmp"), ["version"])

    assert captured.get("env") is None


def test_token_env_accessors_roundtrip():
    """set/get/reset behave as a simple module singleton."""
    git_ops.reset_gh_token_env()
    assert git_ops.get_gh_token_env() is None
    git_ops.set_gh_token_env({"GH_TOKEN": "x"})
    assert git_ops.get_gh_token_env() == {"GH_TOKEN": "x"}
    git_ops.reset_gh_token_env()
    assert git_ops.get_gh_token_env() is None


def test_token_env_does_not_leak_across_tests_part1():
    """Set the singleton; the autouse fixture must clear it before part2 runs."""
    git_ops.set_gh_token_env({"GH_TOKEN": "leaky"})
    assert git_ops.get_gh_token_env() == {"GH_TOKEN": "leaky"}


def test_token_env_does_not_leak_across_tests_part2():
    """If the fixture works, this test sees a clean singleton regardless of order."""
    assert git_ops.get_gh_token_env() is None


def test_resolve_credentials_returns_none_when_unset(monkeypatch):
    monkeypatch.delenv("DAYDREAM_APP_ID", raising=False)
    monkeypatch.delenv("DAYDREAM_APP_PRIVATE_KEY", raising=False)
    assert resolve_credentials() is None


def test_resolve_credentials_parses_both(monkeypatch):
    pem = "-----BEGIN RSA PRIVATE KEY-----\nx\n-----END RSA PRIVATE KEY-----"
    monkeypatch.setenv("DAYDREAM_APP_ID", "12345")
    monkeypatch.setenv("DAYDREAM_APP_PRIVATE_KEY", pem)
    creds = resolve_credentials()
    assert creds == AppCredentials(app_id=12345, private_key=pem)


def test_resolve_credentials_raises_on_partial_id_only(monkeypatch):
    monkeypatch.setenv("DAYDREAM_APP_ID", "12345")
    monkeypatch.delenv("DAYDREAM_APP_PRIVATE_KEY", raising=False)
    with pytest.raises(ValueError, match="DAYDREAM_APP_PRIVATE_KEY"):
        resolve_credentials()


def test_resolve_credentials_raises_on_partial_key_only(monkeypatch):
    monkeypatch.delenv("DAYDREAM_APP_ID", raising=False)
    monkeypatch.setenv("DAYDREAM_APP_PRIVATE_KEY", "x")
    with pytest.raises(ValueError, match="DAYDREAM_APP_ID"):
        resolve_credentials()


def test_resolve_credentials_raises_on_non_integer_id(monkeypatch):
    monkeypatch.setenv("DAYDREAM_APP_ID", "not-an-int")
    monkeypatch.setenv("DAYDREAM_APP_PRIVATE_KEY", "x")
    with pytest.raises(ValueError, match="DAYDREAM_APP_ID"):
        resolve_credentials()


def test_build_gh_env_returns_token_override_only():
    # build_gh_env returns only the token override; os.environ is merged at
    # subprocess call time by git_ops._run_gh so callers see the live env.
    env = build_gh_env("ghs_tok")
    assert env == {"GH_TOKEN": "ghs_tok"}


def test_mint_jwt_is_rs256_with_expected_claims():
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


def _real_pem() -> str:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()


def test_mint_installation_token_happy_path():
    pem = _real_pem()
    calls = []

    def fake_gh_api(repo, endpoint, **kwargs):
        calls.append((endpoint, kwargs))
        if "access_tokens" in endpoint:
            return {"token": "ghs_minted", "expires_at": "2099-01-01T00:00:00Z"}
        return [{"id": 999, "account": {"login": "MyOrg"}, "app_slug": "daydream-bot"}]

    with patch("daydream.git_ops.gh_api", side_effect=fake_gh_api):
        token, identity = mint_installation_token(Path("/tmp"), 12345, pem, "myorg", "myrepo")

    assert token == "ghs_minted"
    assert identity == "daydream-bot[bot]"
    # Two API calls: list installations, then exchange. No extra GET /app.
    assert len(calls) == 2
    # GitHub only accepts App JWTs via the Bearer scheme, so both calls must
    # carry an explicit Authorization header (gh's GH_TOKEN uses token scheme).
    for _, kwargs in calls:
        assert kwargs["headers"]["Authorization"].startswith("Bearer ey")
    exchange_endpoint, exchange_kwargs = calls[1]
    assert exchange_endpoint == "/app/installations/999/access_tokens"
    assert exchange_kwargs["method"] == "POST"
    assert exchange_kwargs["input_data"] == {"repositories": ["myrepo"]}


def test_mint_installation_token_missing_app_slug_yields_unknown_identity():
    """A missing/empty app_slug is cosmetic: the mint succeeds, identity is 'unknown'."""
    pem = _real_pem()

    def fake_gh_api(repo, endpoint, **kwargs):
        if "access_tokens" in endpoint:
            return {"token": "ghs_minted", "expires_at": "2099-01-01T00:00:00Z"}
        return [{"id": 999, "account": {"login": "myorg"}}]

    with patch("daydream.git_ops.gh_api", side_effect=fake_gh_api):
        token, identity = mint_installation_token(Path("/tmp"), 12345, pem, "myorg", "myrepo")

    assert token == "ghs_minted"
    assert identity == "unknown"


def test_mint_installation_token_no_matching_installation():
    pem = _real_pem()

    with patch("daydream.git_ops.gh_api", return_value=[{"id": 1, "account": {"login": "other"}}]):
        with pytest.raises(ValueError, match="installation"):
            mint_installation_token(Path("/tmp"), 12345, pem, "myorg", "myrepo")


def test_mint_installation_token_wraps_gh_api_failure():
    """A gh api failure (GitError) surfaces as the module's ValueError abort channel."""
    pem = _real_pem()

    with patch("daydream.git_ops.gh_api", side_effect=git_ops.GitError("HTTP 401")):
        with pytest.raises(ValueError, match="failed to list App installations"):
            mint_installation_token(Path("/tmp"), 12345, pem, "myorg", "myrepo")


def test_mint_installation_token_sets_and_clears_jwt_env():
    """The JWT env must be active during the API calls and cleared afterward."""
    pem = _real_pem()
    seen_during = {}

    def fake_gh_api(repo, endpoint, **kwargs):
        seen_during["env"] = git_ops.get_gh_token_env()
        if "access_tokens" in endpoint:
            return {"token": "ghs_x", "expires_at": "2099-01-01T00:00:00Z"}
        return [{"id": 7, "account": {"login": "myorg"}, "app_slug": "daydream-bot"}]

    git_ops.reset_gh_token_env()
    with patch("daydream.git_ops.gh_api", side_effect=fake_gh_api):
        mint_installation_token(Path("/tmp"), 12345, pem, "myorg", "myrepo")

    assert seen_during["env"] is not None
    assert "ghs_x" not in (seen_during["env"].get("GH_TOKEN") or "")  # JWT, not installation token
    assert git_ops.get_gh_token_env() is None  # restored after minting


def test_resolve_run_identity_refreshes_installation_token_after_expiry(monkeypatch):
    """A long-running review remints before its next GitHub request."""
    monkeypatch.setenv("DAYDREAM_APP_ID", "12345")
    monkeypatch.setenv("DAYDREAM_APP_PRIVATE_KEY", _TEST_PEM)
    minted = 0
    captured = {}

    def fake_gh_api(repo, endpoint, **kwargs):
        nonlocal minted
        if endpoint == "/app/installations":
            return [{"id": 999, "account": {"login": "myorg"}, "app_slug": "daydream-bot"}]
        assert endpoint == "/app/installations/999/access_tokens"
        minted += 1
        if minted == 1:
            return {"token": "ghs_expired", "expires_at": "1970-01-01T00:00:00Z"}
        return {"token": "ghs_fresh", "expires_at": "9999-01-01T00:00:00Z"}

    def spy_run(*args, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(args[0], 0, stdout="{}", stderr="")

    with patch("daydream.git_ops.gh_api", side_effect=fake_gh_api):
        identity = resolve_run_identity(Path("/tmp"), "myorg/myrepo", is_posting=True)

        with patch("subprocess.run", side_effect=spy_run):
            git_ops._run_gh(Path("/tmp"), ["api", "/user"])

    assert identity == "daydream-bot[bot]"
    assert minted == 2
    assert captured["env"]["GH_TOKEN"] == "ghs_fresh"


def test_resolve_user_identity_returns_login():
    """resolve_user_identity reads the current gh-authenticated user."""
    with patch("daydream.git_ops.gh_api", return_value={"login": "personal-user"}):
        assert resolve_user_identity(Path("/tmp")) == "personal-user"


def test_resolve_user_identity_returns_unknown_on_failure():
    """A failed user lookup is non-fatal — return 'unknown', never raise."""
    with patch("daydream.git_ops.gh_api", side_effect=git_ops.GitError("boom")):
        assert resolve_user_identity(Path("/tmp")) == "unknown"


_TEST_PEM = _real_pem()


def test_exchange_manifest_code_returns_credentials_and_slug():
    """POST the manifest code conversion and read id/pem/slug into AppCredentials."""

    def fake_gh_api(repo, endpoint, **kw):
        assert endpoint == "/app-manifests/abc123/conversions" and kw["method"] == "POST"
        return {"id": 42, "pem": "-----BEGIN RSA PRIVATE KEY-----\nx\n", "slug": "acme-bot"}

    with patch("daydream.git_ops.gh_api", side_effect=fake_gh_api):
        creds, slug = exchange_manifest_code(Path("."), "abc123")
    assert creds.app_id == 42 and "BEGIN RSA" in creds.private_key and slug == "acme-bot"


def test_exchange_manifest_code_raises_when_id_missing():
    """A conversion missing the App id aborts naming the field — no placeholder."""
    with patch("daydream.git_ops.gh_api", return_value={"pem": "x", "slug": "acme-bot"}):
        with pytest.raises(GitHubAppError, match="id"):
            exchange_manifest_code(Path("."), "abc123")


def test_exchange_manifest_code_raises_when_id_not_int():
    """A non-integer App id aborts naming the field — never coerce a placeholder."""
    with patch("daydream.git_ops.gh_api", return_value={"id": "not-int", "pem": "x", "slug": "s"}):
        with pytest.raises(GitHubAppError, match="id"):
            exchange_manifest_code(Path("."), "abc123")


def test_exchange_manifest_code_raises_when_pem_missing():
    """A conversion missing the PEM aborts naming the field — no placeholder key."""
    with patch("daydream.git_ops.gh_api", return_value={"id": 42, "slug": "acme-bot"}):
        with pytest.raises(GitHubAppError, match="pem"):
            exchange_manifest_code(Path("."), "abc123")


def test_exchange_manifest_code_wraps_gh_api_failure():
    """A gh api failure (GitError) surfaces as GitHubAppError, never silent."""
    with patch("daydream.git_ops.gh_api", side_effect=git_ops.GitError("HTTP 422")):
        with pytest.raises(GitHubAppError):
            exchange_manifest_code(Path("."), "abc123")


def test_get_app_metadata_returns_permissions():
    """get_app_metadata mints a JWT and returns the parsed /app object."""
    with patch(
        "daydream.git_ops.gh_api",
        side_effect=lambda *a, **k: {"permissions": {"pull_requests": "write"}, "slug": "acme-bot"},
    ):
        meta = get_app_metadata(Path("."), 42, _TEST_PEM)
    assert meta["permissions"]["pull_requests"] == "write"


def test_get_app_metadata_uses_bearer_jwt_and_clears_env():
    """The /app call carries an explicit Bearer JWT, and the token env is restored."""
    seen = {}

    def fake_gh_api(repo, endpoint, **kw):
        seen["endpoint"] = endpoint
        seen["headers"] = kw.get("headers")
        seen["env"] = git_ops.get_gh_token_env()
        return {"permissions": {"pull_requests": "write"}, "slug": "acme-bot"}

    git_ops.reset_gh_token_env()
    with patch("daydream.git_ops.gh_api", side_effect=fake_gh_api):
        get_app_metadata(Path("."), 42, _TEST_PEM)

    assert seen["endpoint"] == "/app"
    assert seen["headers"]["Authorization"].startswith("Bearer ey")
    assert seen["env"] is not None  # JWT scoped during the call
    assert git_ops.get_gh_token_env() is None  # restored after


def test_get_app_metadata_restores_refreshable_token_state():
    """A scoped App JWT preserves the installation token's refresh behavior."""
    captured = {}
    refresh_calls = 0

    def refresh():
        nonlocal refresh_calls
        refresh_calls += 1
        return {"GH_TOKEN": "ghs_fresh"}, float("inf")

    def spy_run(*args, **kwargs):
        captured.update(kwargs)
        return subprocess.CompletedProcess(args[0], 0, stdout="", stderr="")

    git_ops.set_gh_token_env({"GH_TOKEN": "ghs_expired"}, expires_at=0, refresh=refresh)
    try:
        with patch("daydream.git_ops.gh_api", return_value={"permissions": {}, "slug": "acme-bot"}):
            get_app_metadata(Path("."), 42, _TEST_PEM)
        with patch("subprocess.run", side_effect=spy_run):
            git_ops._run_gh(Path("/tmp"), ["api", "/user"])
    finally:
        git_ops.reset_gh_token_env()

    assert refresh_calls == 1
    assert captured["env"]["GH_TOKEN"] == "ghs_fresh"


def test_get_app_metadata_wraps_gh_api_failure():
    """A gh api failure (GitError) surfaces as GitHubAppError, never silent."""
    with patch("daydream.git_ops.gh_api", side_effect=git_ops.GitError("HTTP 401")):
        with pytest.raises(GitHubAppError):
            get_app_metadata(Path("."), 42, _TEST_PEM)
