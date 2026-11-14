"""Posting with GitHub App credentials aborts before minting when owner/repo is undeterminable."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from daydream.run_config import RunConfig
from daydream.runner import run
from tests.harness.fake_gh import FakeGh
from tests.harness.stub_backend import StubBackend


def _app_creds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DAYDREAM_APP_ID", "12345")
    monkeypatch.setenv("DAYDREAM_APP_PRIVATE_KEY",
        "-----BEGIN RSA PRIVATE KEY-----\nx\n-----END RSA PRIVATE KEY-----",
    )


async def test_posting_aborts_when_owner_repo_undeterminable(
    feature_branch_repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], fake_gh: FakeGh,
) -> None:
    _app_creds(monkeypatch)

    # Posting mode (--comment) with no pr_repo and no resolvable remote: the
    # run must abort rather than let gh fall back to ambient auth.
    config = RunConfig(target=str(feature_branch_repo), non_interactive=True,
                       output_mode="comment", quiet=False)
    with patch("daydream.github_app._mint_installation_token") as mock_mint:
        exit_code = await run(config)
    out = capsys.readouterr().out
    assert exit_code == 1
    assert "Cannot determine owner/repo" in out
    assert not mock_mint.called
    assert fake_gh.process_calls() == []  # no gh request can fall back to ambient authentication

    # A deep-flow review posts nothing, so the same checkout needs no installation token.
    monkeypatch.setattr("daydream.runner.create_backend", lambda *_args, **_kwargs: StubBackend(feature_branch_repo))
    review = RunConfig(target=str(feature_branch_repo), non_interactive=True, flow_name="deep", archive=False)
    assert await run(review) == 0
