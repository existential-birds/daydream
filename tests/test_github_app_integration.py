"""Real-path test for GitHub App identity (01-app-identity).

Enters from runner.run() with a real temp git repo, real filesystem, real event
loop. Mocks only the Backend (no real AI) and the github_app network helpers
(no real GitHub). Asserts observable banner output and singleton state.
"""
from __future__ import annotations

from unittest.mock import patch

from rich.console import Console

from daydream import git_ops
from daydream.backends import ResultEvent, TextEvent
from daydream.runner import RunConfig, run


class _MinimalBackend:
    model = "mock"

    async def execute(self, cwd, prompt, output_schema=None, continuation=None,
                       agents=None, max_turns=None, read_only=False):
        # Alternative-review structured call → emit empty issue list so the
        # review-only flow reports "no issues" and exits 0 fast.
        if output_schema is not None:
            yield TextEvent(text='{"issues": []}')
            yield ResultEvent(structured_output={"issues": []}, continuation=None)
        else:
            yield TextEvent(text="No issues found.")
            yield ResultEvent(structured_output=None, continuation=None)

    def cancel(self) -> None: ...
    def format_skill_invocation(self, key: str) -> str: return f"/{key}"


async def test_app_identity_shown_and_token_injected(feature_branch_repo, monkeypatch):
    monkeypatch.setenv("DAYDREAM_APP_ID", "12345")
    monkeypatch.setenv("DAYDREAM_APP_PRIVATE_KEY", "-----BEGIN RSA PRIVATE KEY-----\nx\n-----END RSA PRIVATE KEY-----")

    # ``pr_repo`` supplies owner/repo so installation-token minting resolves
    # without a real git remote (the temp repo has none).
    config = RunConfig(target=str(feature_branch_repo), non_interactive=True,
                       output_mode="review", shallow=True, skill="python", quiet=False,
                       pr_repo="myorg/myrepo")

    # Pin a wide recording console so the identity line is captured at the
    # rendered content level, independent of terminal width / TTY / Live state.
    rec = Console(record=True, force_terminal=True, width=200)
    monkeypatch.setattr("daydream.runner.console", rec)

    with patch("daydream.github_app.mint_installation_token",
               return_value=("ghs_injected", "my-app[bot]")) as mock_mint, \
         patch("daydream.runner.create_backend", return_value=_MinimalBackend()):
        exit_code = await run(config)

    out = rec.export_text()
    assert "my-app[bot]" in out            # identity surfaced in banner
    assert config.identity == "my-app[bot]"  # raw login on config, escaping only at print
    assert mock_mint.called                 # token minted from App creds
    assert exit_code == 0


async def test_fallback_identity_without_app_creds(feature_branch_repo, monkeypatch, capsys):
    monkeypatch.delenv("DAYDREAM_APP_ID", raising=False)
    monkeypatch.delenv("DAYDREAM_APP_PRIVATE_KEY", raising=False)

    config = RunConfig(target=str(feature_branch_repo), non_interactive=True,
                       output_mode="review", shallow=True, skill="python", quiet=False)

    with patch("daydream.github_app.resolve_user_identity", return_value="personal-user"), \
         patch("daydream.github_app.mint_installation_token") as mock_mint, \
         patch("daydream.runner.create_backend", return_value=_MinimalBackend()):
        exit_code = await run(config)

    out = capsys.readouterr().out
    assert "personal-user" in out
    assert not mock_mint.called             # no App creds → no minting
    assert git_ops.get_gh_token_env() is None  # singleton never set in fallback
    assert exit_code == 0


async def test_fallback_clears_stale_token_from_previous_run(feature_branch_repo, monkeypatch, capsys):
    monkeypatch.delenv("DAYDREAM_APP_ID", raising=False)
    monkeypatch.delenv("DAYDREAM_APP_PRIVATE_KEY", raising=False)

    # Simulate a prior run() in the same process that injected an App token.
    git_ops.set_gh_token_env({"GH_TOKEN": "stale"})

    config = RunConfig(target=str(feature_branch_repo), non_interactive=True,
                       output_mode="review", shallow=True, skill="python", quiet=False)

    with patch("daydream.github_app.resolve_user_identity", return_value="personal-user"), \
         patch("daydream.runner.create_backend", return_value=_MinimalBackend()):
        exit_code = await run(config)

    out = capsys.readouterr().out
    assert "personal-user" in out
    assert git_ops.get_gh_token_env() is None  # stale token cleared, not reused
    assert exit_code == 0


async def test_posting_aborts_when_owner_repo_undeterminable(feature_branch_repo, monkeypatch, capsys):
    monkeypatch.setenv("DAYDREAM_APP_ID", "12345")
    monkeypatch.setenv("DAYDREAM_APP_PRIVATE_KEY", "-----BEGIN RSA PRIVATE KEY-----\nx\n-----END RSA PRIVATE KEY-----")

    # Posting mode (--comment) with no pr_repo and no resolvable remote: the
    # run must abort rather than let gh fall back to ambient auth.
    config = RunConfig(target=str(feature_branch_repo), non_interactive=True,
                       output_mode="comment", shallow=True, skill="python", quiet=False)

    with patch("daydream.git_ops.gh_repo_view", return_value=None), \
         patch("daydream.github_app.mint_installation_token") as mock_mint, \
         patch("daydream.runner.create_backend", return_value=_MinimalBackend()):
        exit_code = await run(config)

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "Cannot determine owner/repo" in out
    assert not mock_mint.called
    assert git_ops.get_gh_token_env() is None


async def test_minting_failure_aborts_run(feature_branch_repo, monkeypatch, capsys):
    monkeypatch.setenv("DAYDREAM_APP_ID", "12345")
    monkeypatch.setenv("DAYDREAM_APP_PRIVATE_KEY", "-----BEGIN RSA PRIVATE KEY-----\nx\n-----END RSA PRIVATE KEY-----")

    config = RunConfig(target=str(feature_branch_repo), non_interactive=True,
                       output_mode="review", shallow=True, skill="python", quiet=False,
                       pr_repo="myorg/myrepo")

    with patch("daydream.github_app.mint_installation_token",
               side_effect=ValueError("no App installation found for owner 'myorg'")), \
         patch("daydream.runner.create_backend", return_value=_MinimalBackend()):
        exit_code = await run(config)

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "App token resolution failed" in out
    assert git_ops.get_gh_token_env() is None  # failed minting never injects a token
