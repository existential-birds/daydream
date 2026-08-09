# tests/test_hub_upload.py
from __future__ import annotations

import os
from pathlib import Path

import pytest

from daydream.archive import hub


def test_resolve_hub_repo_precedence_cli_over_env_over_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from daydream.runner import RunConfig
    from daydream.config_file import DaydreamFileConfig

    monkeypatch.delenv("DAYDREAM_TRAJECTORY_HUB_REPO", raising=False)
    file_cfg = DaydreamFileConfig(trajectory_hub_repo="file/repo")
    cli_cfg = RunConfig(trajectory_hub_repo="cli/repo", file_config=file_cfg)
    assert hub.resolve_hub_repo(cli_cfg) == "cli/repo"
    monkeypatch.setenv("DAYDREAM_TRAJECTORY_HUB_REPO", "env/repo")
    assert hub.resolve_hub_repo(cli_cfg) == "cli/repo"
    env_only = RunConfig(trajectory_hub_repo=None, file_config=file_cfg)
    assert hub.resolve_hub_repo(env_only) == "env/repo"
    file_only = RunConfig(trajectory_hub_repo=None, file_config=file_cfg)
    monkeypatch.delenv("DAYDREAM_TRAJECTORY_HUB_REPO")
    assert hub.resolve_hub_repo(file_only) == "file/repo"
    assert hub.resolve_hub_repo(RunConfig()) is None


def test_upload_run_bundle_creates_private_repo_and_uploads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    run_dir = tmp_path / "runs" / "session-123"
    run_dir.mkdir(parents=True)
    (run_dir / "trajectory.json").write_text("{}", encoding="utf-8")

    calls: dict = {}
    class FakeApi:
        def __init__(self, token: str | None = None):
            calls["token"] = token
        def create_repo(self, repo_id: str, *, repo_type: str, private: bool, exist_ok: bool) -> None:
            calls["create_repo"] = (repo_id, repo_type, private, exist_ok)
        def upload_folder(self, *, folder_path: str, repo_id: str, repo_type: str, path_in_repo: str,
                          commit_message: str) -> None:
            calls["upload_folder"] = (folder_path, repo_id, repo_type, path_in_repo, commit_message)

    monkeypatch.setattr(hub, "HfApi", FakeApi)
    assert hub.upload_run_bundle(run_dir, "acme/dd-trajectories", "session-123") is True
    assert calls["create_repo"] == ("acme/dd-trajectories", "dataset", True, True)
    assert calls["upload_folder"][:4] == (str(run_dir), "acme/dd-trajectories", "dataset", "session-123")


def test_upload_run_bundle_skips_without_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    run_dir = tmp_path / "runs" / "s"
    run_dir.mkdir(parents=True)
    (run_dir / "trajectory.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(hub, "HfApi", _BoomApi)  # would raise if instantiated
    assert hub.upload_run_bundle(run_dir, "acme/dd", "s") is False


def test_upload_run_bundle_retries_commit_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    run_dir = tmp_path / "runs" / "s2"
    run_dir.mkdir(parents=True)
    (run_dir / "trajectory.json").write_text("{}", encoding="utf-8")

    attempts: list[int] = []

    class ConflictApi:
        def create_repo(self, repo_id: str, **kw) -> None:
            pass
        def upload_folder(self, **kw) -> None:
            attempts.append(1)
            if len(attempts) < 3:
                raise RuntimeError("Commit failed: concurrent update to refs/heads/main")

    monkeypatch.setattr(hub, "HfApi", ConflictApi)
    assert hub.upload_run_bundle(run_dir, "acme/dd", "s2") is True
    assert len(attempts) == 3


class _BoomApi:
    def __init__(self, token: str | None = None) -> None:
        raise AssertionError("HfApi must not be instantiated without HF_TOKEN")
