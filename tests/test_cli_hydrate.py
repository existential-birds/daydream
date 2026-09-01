"""CLI wiring tests for `daydream corpus hydrate-hub` (#982 M1/M2/M17)."""
from __future__ import annotations

import pathlib
from typing import Any

import pytest

from daydream import cli
from daydream.archive import hydrate
from daydream.archive.hydrate_client import FakeHub
from tests.fixtures.training.build_hub_snapshot import SNAPSHOT_REVISION, build_snapshot


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
        dry_run_discovered = 1
        dry_run_admitted = 1
        dry_run_rejected = 0
        dry_run_incomplete_manifests: tuple[str, ...] = ()
        verify_admitted = 1
        license_admission: dict[str, int] = {}

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


def test_cli_wires_license_policy_into_hydration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """--license-policy/--allow-copyleft reach HydrateHubConfig, and the
    previously unreachable license admission summary prints (issue #1080)."""
    calls: list[Any] = []

    class FakeSummary:
        curation_id = "cur-" + "0" * 16
        output_commit_sha = "b" * 40
        verified = True
        dry_run_discovered = 2
        dry_run_admitted = 1
        dry_run_rejected = 1
        dry_run_incomplete_manifests: tuple[str, ...] = ()
        verify_admitted = 1
        license_admission = {
            "admitted": 1, "c5_excluded": 1,
            "c8_copyleft_unopted": 0, "license_evidence_missing": 0,
        }

    def fake_run(config: Any) -> FakeSummary:
        calls.append(config)
        return FakeSummary()

    monkeypatch.setenv("HF_TOKEN", "t")
    monkeypatch.setattr(cli, "_run_hydrate_hub", fake_run, raising=False)
    policy = tmp_path / "license-policy.json"
    policy.write_text('{"policy_version": "1", "spdx_decisions": {}}')
    rc = cli._handle_hydrate_hub_command(
        ["--source-repo", "org/ds", "--source-revision", "a" * 40,
         "--destination-repo", "org/ds", "--stage-dir", str(tmp_path),
         "--license-policy", str(policy),
         "--allow-copyleft", "Owner/Gpl-Repo"])
    assert rc == 0
    assert calls and calls[0].license_policy_path == str(policy)
    assert calls[0].allow_copyleft == frozenset({"owner/gpl-repo"})
    output = " ".join(capsys.readouterr().out.split())
    assert "license admission: admitted 1; c5-excluded 1" in output


def test_success_path_surfaces_incomplete_manifests(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    class FakeSummary:
        curation_id = "cur-" + "0" * 16
        output_commit_sha = "b" * 40
        verified = True
        dry_run_discovered = 2
        dry_run_admitted = 1
        dry_run_rejected = 0
        dry_run_incomplete_manifests = ("sess-a (missing trajectory.json)",)
        verify_admitted = 1
        license_admission: dict[str, int] = {}

    monkeypatch.setenv("HF_TOKEN", "t")
    monkeypatch.setattr(cli, "_run_hydrate_hub", lambda _config: FakeSummary())
    rc = cli._handle_hydrate_hub_command(
        ["--source-repo", "org/ds", "--source-revision", "a" * 40,
         "--destination-repo", "org/ds", "--stage-dir", str(tmp_path)])
    assert rc == 0
    output = " ".join(capsys.readouterr().out.split())
    assert "hydration yield reduced" in output
    assert "sess-a (missing trajectory.json)" in output


def test_dry_run_reports_discovery_accounting_without_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    hub = build_snapshot()
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(hydrate, "_make_client", lambda _repo: hub)

    rc = cli._handle_hydrate_hub_command(
        [
            "--source-repo",
            "org/private-ds",
            "--source-revision",
            SNAPSHOT_REVISION,
            "--destination-repo",
            "org/private-ds",
            "--stage-dir",
            str(tmp_path),
            "--dry-run",
        ]
    )

    assert rc == 0
    output = " ".join(capsys.readouterr().out.split())
    assert "discovered 3 candidate(s) of 3 run-shaped manifest(s)" in output
    assert "admitted 3" in output
    assert "rejected 0 batch(es)" in output
    assert "accounted 3 candidate(s)" in output
    assert "yield reduced" not in output
    assert hub.uploaded_paths == []


def test_dry_run_surfaces_incomplete_manifests_with_reduced_yield(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    complete = {
        "manifest.json": b'{"session_id": "complete"}',
        "trajectory.json": b'{"messages": []}',
    }
    incomplete = {"manifest.json": b'{"session_id": "incomplete"}'}
    hub = FakeHub(
        repo_id="org/private-ds",
        private=True,
        files={
            "sess-complete/manifest.json": complete["manifest.json"],
            "sess-complete/trajectory.json": complete["trajectory.json"],
            "sess-incomplete/manifest.json": incomplete["manifest.json"],
        },
    )
    hub.commit_revision(SNAPSHOT_REVISION)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(hydrate, "_make_client", lambda _repo: hub)

    rc = cli._handle_hydrate_hub_command(
        [
            "--source-repo",
            "org/private-ds",
            "--source-revision",
            SNAPSHOT_REVISION,
            "--destination-repo",
            "org/private-ds",
            "--stage-dir",
            str(tmp_path),
            "--dry-run",
        ]
    )

    assert rc == 0
    output = " ".join(capsys.readouterr().out.split())
    assert "discovered 1 candidate(s) of 2 run-shaped manifest(s)" in output
    assert "accounted 1 candidate(s)" in output
    assert "dry-run yield reduced" in output
    assert "sess-incomplete (missing trajectory.json)" in output
    assert hub.uploaded_paths == []


def test_dry_run_fails_closed_on_run_shaped_zero_discovery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    hub = FakeHub(
        repo_id="org/private-ds",
        private=True,
        files={"incomplete/manifest.json": b'{"session_id": "incomplete"}'},
    )
    hub.commit_revision(SNAPSHOT_REVISION)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    monkeypatch.setattr(hydrate, "_make_client", lambda _repo: hub)

    rc = cli._handle_hydrate_hub_command(
        [
            "--source-repo",
            "org/private-ds",
            "--source-revision",
            SNAPSHOT_REVISION,
            "--destination-repo",
            "org/private-ds",
            "--stage-dir",
            str(tmp_path),
            "--dry-run",
        ]
    )

    assert rc == 1
    output = capsys.readouterr().out
    assert "zero candidates" in output
    assert "trajectory.json" in output
    assert hub.uploaded_paths == []
