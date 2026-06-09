import subprocess
from pathlib import Path
from unittest.mock import patch

from daydream import git_ops


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


import pytest

from daydream.github_app import (
    AppCredentials,
    build_gh_env,
    mint_jwt,
    resolve_credentials,
)


def test_resolve_credentials_returns_none_when_unset(monkeypatch):
    monkeypatch.delenv("DAYDREAM_APP_ID", raising=False)
    monkeypatch.delenv("DAYDREAM_APP_PRIVATE_KEY", raising=False)
    assert resolve_credentials() is None


def test_resolve_credentials_parses_both(monkeypatch):
    monkeypatch.setenv("DAYDREAM_APP_ID", "12345")
    monkeypatch.setenv("DAYDREAM_APP_PRIVATE_KEY", "-----BEGIN RSA PRIVATE KEY-----\nx\n-----END RSA PRIVATE KEY-----")
    creds = resolve_credentials()
    assert creds == AppCredentials(app_id=12345, private_key="-----BEGIN RSA PRIVATE KEY-----\nx\n-----END RSA PRIVATE KEY-----")


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


def test_build_gh_env_injects_token_and_inherits_path(monkeypatch):
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    env = build_gh_env("ghs_tok")
    assert env["GH_TOKEN"] == "ghs_tok"
    assert env["PATH"] == "/usr/bin:/bin"


def test_mint_jwt_is_rs256_with_expected_claims():
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization
    import jwt as pyjwt

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


import json

from daydream.github_app import mint_installation_token


def _real_pem() -> str:
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()


def test_mint_installation_token_happy_path():
    pem = _real_pem()
    calls = []

    def fake_run_gh(repo, args, *, timeout=60):
        calls.append(args)
        endpoint = args[-1]
        if "access_tokens" in " ".join(args):
            return subprocess.CompletedProcess(args, 0, stdout=json.dumps({"token": "ghs_minted"}), stderr="")
        if "app/installations" in endpoint:
            return subprocess.CompletedProcess(
                args, 0,
                stdout=json.dumps([{"id": 999, "account": {"login": "MyOrg"}}]),
                stderr="",
            )
        return subprocess.CompletedProcess(args, 1, stdout="", stderr="unexpected")

    with patch("daydream.github_app._run_gh", side_effect=fake_run_gh):
        token = mint_installation_token(12345, pem, "myorg", "myrepo")

    assert token == "ghs_minted"
    # Two API calls: list installations, then exchange.
    assert len(calls) == 2


def test_mint_installation_token_no_matching_installation():
    pem = _real_pem()

    def fake_run_gh(repo, args, *, timeout=60):
        return subprocess.CompletedProcess(
            args, 0, stdout=json.dumps([{"id": 1, "account": {"login": "other"}}]), stderr=""
        )

    with patch("daydream.github_app._run_gh", side_effect=fake_run_gh):
        with pytest.raises(ValueError, match="installation"):
            mint_installation_token(12345, pem, "myorg", "myrepo")


def test_mint_installation_token_sets_and_clears_jwt_env():
    """The JWT env must be active during the API calls and cleared afterward."""
    pem = _real_pem()
    seen_during = {}

    def fake_run_gh(repo, args, *, timeout=60):
        from daydream import git_ops
        seen_during["env"] = git_ops.get_gh_token_env()
        if "access_tokens" in " ".join(args):
            return subprocess.CompletedProcess(args, 0, stdout=json.dumps({"token": "ghs_x"}), stderr="")
        return subprocess.CompletedProcess(
            args, 0, stdout=json.dumps([{"id": 7, "account": {"login": "myorg"}}]), stderr=""
        )

    from daydream import git_ops
    git_ops.reset_gh_token_env()
    with patch("daydream.github_app._run_gh", side_effect=fake_run_gh):
        mint_installation_token(12345, pem, "myorg", "myrepo")

    assert seen_during["env"] is not None
    assert "ghs_x" not in (seen_during["env"].get("GH_TOKEN") or "")  # JWT, not installation token
    assert git_ops.get_gh_token_env() is None  # restored after minting
