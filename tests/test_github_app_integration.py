"""Real-path test for GitHub App identity (01-app-identity).

Enters from runner.run() with a real temp git repo, real filesystem, real event
loop. Mocks only the Backend (no real AI) and the github_app network helpers
(no real GitHub). Asserts observable banner output and singleton state.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

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


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_COMMITTER_NAME": "t",
           "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_EMAIL": "t@t"}
    subprocess.run(["git", "init", "-b", "main", str(tmp_path)], capture_output=True, check=True)
    (tmp_path / "f.py").write_text("x = 1\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "init"], capture_output=True, env=env, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "checkout", "-b", "feat"], capture_output=True, check=True)
    (tmp_path / "f.py").write_text("x = 2\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], capture_output=True, check=True)
    subprocess.run(["git", "-C", str(tmp_path), "commit", "-m", "change"], capture_output=True, env=env, check=True)
    return tmp_path


async def test_app_identity_shown_and_token_injected(git_repo, monkeypatch, capsys):
    monkeypatch.setenv("DAYDREAM_APP_ID", "12345")
    monkeypatch.setenv("DAYDREAM_APP_PRIVATE_KEY", "-----BEGIN RSA PRIVATE KEY-----\nx\n-----END RSA PRIVATE KEY-----")

    # ``pr_repo`` supplies owner/repo so installation-token minting resolves
    # without a real git remote (the temp repo has none).
    config = RunConfig(target=str(git_repo), non_interactive=True,
                       output_mode="review", shallow=True, skill="python", quiet=False,
                       pr_repo="myorg/myrepo")

    with patch("daydream.github_app.mint_installation_token", return_value="ghs_injected") as mock_mint, \
         patch("daydream.github_app.resolve_identity", return_value="my-app[bot]"), \
         patch("daydream.runner.create_backend", return_value=_MinimalBackend()):
        exit_code = await run(config)

    out = capsys.readouterr().out
    assert "my-app[bot]" in out            # identity surfaced in banner
    assert mock_mint.called                 # token minted from App creds
    assert exit_code == 0


async def test_fallback_identity_without_app_creds(git_repo, monkeypatch, capsys):
    monkeypatch.delenv("DAYDREAM_APP_ID", raising=False)
    monkeypatch.delenv("DAYDREAM_APP_PRIVATE_KEY", raising=False)

    config = RunConfig(target=str(git_repo), non_interactive=True,
                       output_mode="review", shallow=True, skill="python", quiet=False)

    with patch("daydream.github_app.resolve_identity", return_value="personal-user"), \
         patch("daydream.github_app.mint_installation_token") as mock_mint, \
         patch("daydream.runner.create_backend", return_value=_MinimalBackend()):
        exit_code = await run(config)

    out = capsys.readouterr().out
    assert "personal-user" in out
    assert not mock_mint.called             # no App creds → no minting
    assert git_ops.get_gh_token_env() is None  # singleton never set in fallback
    assert exit_code == 0
