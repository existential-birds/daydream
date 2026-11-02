"""Real-path test for GitHub App identity (01-app-identity).

Enters from runner.run() with a real temp git repo, real filesystem, real event
loop. Uses the fake gh transport and Backend (no real AI), with token minting
stubbed. Asserts observable banner output and run-owned auth behavior.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from daydream.backends import ResultEvent, TextEvent
from daydream.run_config import RunConfig
from daydream.runner import run
from tests.harness.backend import ScriptedBackend
from tests.harness.fake_gh import FakeGh
from tests.harness.stub_backend import review_stage_result


@pytest.fixture(autouse=True)
def _fake_github(fake_gh: FakeGh, feature_branch_repo: Path) -> None:
    fake_gh.serve_open_pr(feature_branch_repo)


def _minimal_backend() -> ScriptedBackend:
    """Responder-backed fake: structured calls get an empty issue list, else prose."""

    def respond(cwd: Any, prompt: str, output_schema: Any = None, *args: Any) -> list[Any]:
        # Alternative-review structured call → emit empty issue list so the
        # review-only flow reports "no issues" and exits 0 fast.
        if output_schema is not None:
            return [TextEvent(text="Review complete."),
                ResultEvent(structured_output=review_stage_result(prompt, []), continuation=None),
            ]
        return [TextEvent(text="No issues found."), ResultEvent(structured_output=None, continuation=None)]

    return ScriptedBackend(responder=respond, model="mock")


@pytest.fixture(autouse=True)
def _fake_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("daydream.runner.create_backend", lambda *args, **kwargs: _minimal_backend())


def _app_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DAYDREAM_APP_ID", "12345")
    monkeypatch.setenv("DAYDREAM_APP_PRIVATE_KEY",
        "-----BEGIN RSA PRIVATE KEY-----\nx\n-----END RSA PRIVATE KEY-----",
    )


async def test_posting_aborts_when_owner_repo_undeterminable(
    feature_branch_repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    _app_creds(monkeypatch)

    # Posting mode (--comment) with no pr_repo and no resolvable remote: the
    # run must abort rather than let gh fall back to ambient auth.
    config = RunConfig(target=str(feature_branch_repo), non_interactive=True,
                       output_mode="comment", shallow=True, stack="python", quiet=False)
    with patch("daydream.github_app._mint_installation_token") as mock_mint:
        exit_code = await run(config)
    out = capsys.readouterr().out
    assert exit_code == 1
    assert "Cannot determine owner/repo" in out
    assert not mock_mint.called
