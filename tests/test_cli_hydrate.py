"""CLI wiring tests for `daydream corpus hydrate-hub` (#982 M1/M2/M17)."""
from __future__ import annotations

import pathlib
from typing import Any

import pytest

from daydream import cli


def test_hydrate_hub_requires_explicit_args(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli._handle_hydrate_hub_command([])
    assert rc == 1
    assert "--source-repo" in capsys.readouterr().out


def test_hydrate_hub_rejects_moving_branch_without_optin(capsys: pytest.CaptureFixture[str]) -> None:
    rc = cli._handle_hydrate_hub_command(
        ["--source-repo", "org/ds", "--source-revision", "main",
         "--destination-repo", "org/ds", "--stage-dir", "/tmp/x"])
    assert rc == 1
    assert "exploratory" in capsys.readouterr().out


def test_hydrate_hub_missing_token_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    rc = cli._handle_hydrate_hub_command(
        ["--source-repo", "org/ds", "--source-revision", "a" * 40,
         "--destination-repo", "org/ds", "--stage-dir", "/tmp/x"])
    assert rc == 1


def test_help_lists_hydrate_hub(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        cli.main(["corpus", "hydrate-hub", "--help"])
    assert "hydrate-hub" in capsys.readouterr().out


def test_success_path_drives_orchestrator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    calls: list[Any] = []

    class FakeSummary:
        curation_id = "cur-" + "0" * 16
        output_commit_sha = "b" * 40
        verified = True
        dry_run_admitted = 1
        verify_admitted = 1

    def fake_run(config: Any) -> FakeSummary:
        calls.append(config)
        return FakeSummary()

    monkeypatch.setenv("HF_TOKEN", "t")
    monkeypatch.setattr(cli, "_run_hydrate_hub", fake_run, raising=False)
    rc = cli._handle_hydrate_hub_command(
        ["--source-repo", "org/ds", "--source-revision", "a" * 40,
         "--destination-repo", "org/ds", "--stage-dir", str(tmp_path)])
    assert rc == 0
    assert calls and calls[0].source_revision == "a" * 40
