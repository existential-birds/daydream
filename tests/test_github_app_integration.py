"""Real-path test for GitHub App identity (01-app-identity).

Enters from runner.run() with a real temp git repo, real filesystem, real event
loop. Mocks only the Backend (no real AI) and the github_app network helpers
(no real GitHub). Asserts observable banner output and run-owned auth behavior.
"""
from __future__ import annotations

from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from rich.console import Console

from daydream import git_ops
from daydream.backends import ResultEvent, TextEvent
from daydream.runner import RunConfig, run
from tests.harness.backend import ScriptedBackend
from tests.harness.fake_gh import block_real_gh


@pytest.fixture(autouse=True)
def _block_real_gh(monkeypatch: pytest.MonkeyPatch) -> None:
    block_real_gh(monkeypatch)


def _minimal_backend() -> ScriptedBackend:
    """Responder-backed fake: structured calls get an empty issue list, else prose."""

    def respond(cwd: Any, prompt: str, output_schema: Any = None, *args: Any) -> list[Any]:
        # Alternative-review structured call → emit empty issue list so the
        # review-only flow reports "no issues" and exits 0 fast.
        if output_schema is not None:
            return [
                TextEvent(text='{"issues": []}'),
                ResultEvent(structured_output={"issues": []}, continuation=None),
            ]
        return [TextEvent(text="No issues found."), ResultEvent(structured_output=None, continuation=None)]

    return ScriptedBackend(responder=respond, model="mock")


@pytest.fixture(autouse=True)
def _fake_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("daydream.runner.create_backend", lambda *args, **kwargs: _minimal_backend())


def _app_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DAYDREAM_APP_ID", "12345")
    monkeypatch.setenv(
        "DAYDREAM_APP_PRIVATE_KEY",
        "-----BEGIN RSA PRIVATE KEY-----\nx\n-----END RSA PRIVATE KEY-----",
    )


def _no_app_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DAYDREAM_APP_ID", raising=False)
    monkeypatch.delenv("DAYDREAM_APP_PRIVATE_KEY", raising=False)


async def test_app_identity_shown_and_token_injected(
    feature_branch_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _app_creds(monkeypatch)

    # ``pr_repo`` supplies owner/repo so installation-token minting resolves
    # without a real git remote (the temp repo has none).
    config = RunConfig(target=str(feature_branch_repo), non_interactive=True,
                       output_mode="comment", shallow=True, stack="python", quiet=False,
                       pr_repo="myorg/myrepo")

    # Pin a wide recording console so the identity line is captured at the
    # rendered content level, independent of terminal width / TTY / Live state.
    # The identity banner prints from the deep orchestrator's info block (#330).
    rec = Console(file=StringIO(), record=True, force_terminal=True, width=200, height=25)
    monkeypatch.setattr("daydream.runner.console", rec)
    monkeypatch.setattr("daydream.deep.orchestrator.console", rec)

    with patch("daydream.github_app._mint_installation_token",
               return_value=SimpleNamespace(
                   token="ghs_injected", identity="my-app[bot]", expires_at=float("inf")
               )) as mock_mint:
        exit_code = await run(config)

    out = rec.export_text()
    assert "my-app[bot]" in out            # identity surfaced in banner
    assert config.identity == "my-app[bot]"  # raw login on config, escaping only at print
    assert mock_mint.called                 # token minted from App creds
    assert exit_code == 0


async def test_fallback_identity_without_app_creds(
    feature_branch_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _no_app_creds(monkeypatch)

    config = RunConfig(target=str(feature_branch_repo), non_interactive=True,
                       output_mode="review", shallow=True, stack="python", quiet=False)

    with patch("daydream.github_app.resolve_user_identity", return_value="personal-user"), \
         patch("daydream.github_app._mint_installation_token") as mock_mint:
        exit_code = await run(config)

    out = capsys.readouterr().out
    assert "personal-user" in out
    assert not mock_mint.called             # no App creds → no minting
    assert config.identity == "personal-user"
    assert exit_code == 0


async def test_fallback_run_cannot_replace_an_existing_session_auth(
    feature_branch_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _no_app_creds(monkeypatch)

    existing_auth = git_ops.StaticGitHubAuth(
        {"GH_TOKEN": "ghs_existing_session_1234567890"}
    )

    config = RunConfig(target=str(feature_branch_repo), non_interactive=True,
                       output_mode="review", shallow=True, stack="python", quiet=False)

    with patch("daydream.github_app.resolve_user_identity", return_value="personal-user"):
        exit_code = await run(config)

    out = capsys.readouterr().out
    assert "personal-user" in out
    assert existing_auth.environment_for_request()["GH_TOKEN"] == (
        "ghs_existing_session_1234567890"
    )
    assert config.identity == "personal-user"
    assert exit_code == 0


async def test_posting_aborts_when_owner_repo_undeterminable(
    feature_branch_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _app_creds(monkeypatch)

    # Posting mode (--comment) with no pr_repo and no resolvable remote: the
    # run must abort rather than let gh fall back to ambient auth.
    config = RunConfig(target=str(feature_branch_repo), non_interactive=True,
                       output_mode="comment", shallow=True, stack="python", quiet=False)

    with patch("daydream.git_ops.gh_repo_view", return_value=None), \
         patch("daydream.github_app._mint_installation_token") as mock_mint:
        exit_code = await run(config)

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "Cannot determine owner/repo" in out
    assert not mock_mint.called


async def test_minting_failure_aborts_run(
    feature_branch_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _app_creds(monkeypatch)

    config = RunConfig(target=str(feature_branch_repo), non_interactive=True,
                       output_mode="comment", shallow=True, stack="python", quiet=False,
                       pr_repo="myorg/myrepo")

    with patch("daydream.github_app._mint_installation_token",
               side_effect=ValueError("no App installation found for owner 'myorg'")):
        exit_code = await run(config)

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "App token resolution failed" in out
