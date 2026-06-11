"""Real-path tests for :mod:`daydream.bot_setup`.

Task 5 covers the App-from-manifest localhost registration seam. The live
browser/manifest leg (binding a port, opening a browser, blocking on the
callback) is isolated behind :class:`daydream.bot_setup._ManifestListener` so
the code-exchange behavior is testable without real GitHub: the test drives
``_handle_code`` directly, monkeypatching only the manifest-code exchange.
"""

from pathlib import Path

import pytest

from daydream import bot_setup
from daydream.github_app import AppCredentials, GitHubAppError


def test_callback_listener_captures_code_then_exchanges(monkeypatch):
    """The callback seam exchanges the manifest code for creds + slug."""
    monkeypatch.setattr(
        "daydream.bot_setup.exchange_manifest_code",
        lambda repo, code: (AppCredentials(7, "-----BEGIN-----\n"), "acme-bot"),
    )
    listener = bot_setup._ManifestListener(repo_dir=Path("."), org=None)
    creds, slug = listener._handle_code("codeXYZ")
    assert creds.app_id == 7
    assert creds.private_key == "-----BEGIN-----\n"
    assert slug == "acme-bot"


def test_callback_listener_passes_repo_dir_and_code_through(monkeypatch):
    """The seam threads the listener's repo_dir and the callback code unchanged."""
    captured: dict[str, object] = {}

    def fake_exchange(repo, code):
        captured["repo"] = repo
        captured["code"] = code
        return AppCredentials(42, "pem"), "slug-x"

    monkeypatch.setattr("daydream.bot_setup.exchange_manifest_code", fake_exchange)
    listener = bot_setup._ManifestListener(repo_dir=Path("/tmp/repo"), org="acme")
    creds, slug = listener._handle_code("codeABC")
    assert captured["repo"] == Path("/tmp/repo")
    assert captured["code"] == "codeABC"
    assert creds.app_id == 42 and slug == "slug-x"


def test_missing_code_raises_cancelled_and_never_exchanges(monkeypatch):
    """An empty/missing callback code (user declined) aborts with a clear error."""
    called = False

    def fake_exchange(repo, code):
        nonlocal called
        called = True
        return AppCredentials(1, "pem"), "slug"

    monkeypatch.setattr("daydream.bot_setup.exchange_manifest_code", fake_exchange)
    listener = bot_setup._ManifestListener(repo_dir=Path("."), org=None)
    with pytest.raises(GitHubAppError, match="App registration was cancelled"):
        listener._handle_code("")
    with pytest.raises(GitHubAppError, match="App registration was cancelled"):
        listener._handle_code(None)
    assert called is False
