# tests/test_hub_upload.py
from __future__ import annotations

from pathlib import Path

import pytest

from daydream.archive import hub


class _RepoInfo:
    """Minimal stand-in for huggingface_hub's ``RepoInfo`` (visibility only)."""

    def __init__(self, private: bool) -> None:
        self.private = private


class _BaseFakeApi:
    """Fake ``HfApi`` defaults shared by upload tests: a private target repo."""

    def create_repo(self, repo_id: str, **kw) -> None:
        pass

    def repo_info(self, repo_id: str, *, repo_type: str) -> _RepoInfo:
        return _RepoInfo(private=True)


@pytest.fixture
def hf_run_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A populated run-bundle dir with ``HF_TOKEN`` set (shared upload scaffold)."""
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    run_dir = tmp_path / "runs" / "session-123"
    run_dir.mkdir(parents=True)
    (run_dir / "trajectory.json").write_text("{}", encoding="utf-8")
    return run_dir


def test_resolve_hub_repo_precedence_cli_over_env_over_file(monkeypatch: pytest.MonkeyPatch) -> None:
    from daydream.config_file import DaydreamFileConfig
    from daydream.runner import RunConfig

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


def test_resolve_hub_repo_empty_strings_treated_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    from daydream.config_file import DaydreamFileConfig
    from daydream.runner import RunConfig

    file_cfg = DaydreamFileConfig(trajectory_hub_repo="file/repo")
    # empty CLI tier falls through to env
    monkeypatch.setenv("DAYDREAM_TRAJECTORY_HUB_REPO", "env/repo")
    assert hub.resolve_hub_repo(RunConfig(trajectory_hub_repo="", file_config=file_cfg)) == "env/repo"
    # empty env tier falls through to file
    monkeypatch.setenv("DAYDREAM_TRAJECTORY_HUB_REPO", "")
    assert hub.resolve_hub_repo(RunConfig(file_config=file_cfg)) == "file/repo"
    # empty file tier is unset
    empty_file_cfg = DaydreamFileConfig(trajectory_hub_repo="")
    assert hub.resolve_hub_repo(RunConfig(file_config=empty_file_cfg)) is None


def test_upload_run_bundle_creates_private_repo_and_uploads(
    hf_run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict = {}

    class FakeApi:
        def __init__(self, token: str | None = None):
            calls["token"] = token

        def create_repo(self, repo_id: str, *, repo_type: str, private: bool, exist_ok: bool) -> None:
            calls["create_repo"] = (repo_id, repo_type, private, exist_ok)

        def repo_info(self, repo_id: str, *, repo_type: str) -> _RepoInfo:
            calls["repo_info"] = (repo_id, repo_type)
            return _RepoInfo(private=True)

        def upload_folder(
            self, *, folder_path: str, repo_id: str, repo_type: str, path_in_repo: str, commit_message: str
        ) -> None:
            calls["upload_folder"] = (folder_path, repo_id, repo_type, path_in_repo, commit_message)

    monkeypatch.setattr(hub, "HfApi", FakeApi)
    assert hub.upload_run_bundle(hf_run_dir, "acme/dd-trajectories", "session-123") is True
    assert calls["create_repo"] == ("acme/dd-trajectories", "dataset", True, True)
    assert calls["repo_info"] == ("acme/dd-trajectories", "dataset")
    assert calls["upload_folder"][:4] == (str(hf_run_dir), "acme/dd-trajectories", "dataset", "session-123")


def test_upload_run_bundle_warns_on_existing_public_repo(
    hf_run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict = {}

    class PublicRepoApi(_BaseFakeApi):
        def create_repo(self, repo_id: str, **kw) -> None:
            calls["create_repo"] = True

        def repo_info(self, repo_id: str, *, repo_type: str) -> _RepoInfo:
            calls["repo_info"] = (repo_id, repo_type)
            return _RepoInfo(private=False)

        def upload_folder(self, **kw) -> None:
            calls["upload_folder"] = True

    monkeypatch.setattr(hub, "HfApi", PublicRepoApi)
    # A pre-existing public repo is reused with its current visibility (documented
    # behavior) — the operator is warned, but the upload still proceeds.
    assert hub.upload_run_bundle(hf_run_dir, "acme/dd", "s") is True
    assert calls["repo_info"] == ("acme/dd", "dataset")
    assert calls.get("upload_folder") is True


def test_upload_run_bundle_skips_without_token(
    hf_run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(hub, "HfApi", _BoomApi)  # would raise if instantiated
    assert hub.upload_run_bundle(hf_run_dir, "acme/dd", "s") is False


def test_upload_run_bundle_hfapi_instantiation_failure(
    hf_run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(hub, "HfApi", _BoomApi)  # HfApi() raises
    assert hub.upload_run_bundle(hf_run_dir, "acme/dd", "s3") is False


def test_upload_run_bundle_create_repo_failure(
    hf_run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CreateRepoBoomApi(_BaseFakeApi):
        def create_repo(self, repo_id: str, **kw) -> None:
            raise RuntimeError("401 Unauthorized: invalid token")

    monkeypatch.setattr(hub, "HfApi", CreateRepoBoomApi)
    assert hub.upload_run_bundle(hf_run_dir, "acme/dd", "s4") is False


def test_upload_run_bundle_non_conflict_upload_error(
    hf_run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[int] = []

    class NonConflictApi(_BaseFakeApi):
        def upload_folder(self, **kw) -> None:
            attempts.append(1)
            raise RuntimeError("some unrelated error")

    monkeypatch.setattr(hub, "HfApi", NonConflictApi)
    assert hub.upload_run_bundle(hf_run_dir, "acme/dd", "s5") is False
    assert len(attempts) == 1


def test_upload_run_bundle_retry_exhaustion(
    hf_run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[int] = []
    monkeypatch.setattr(hub, "_UPLOAD_RETRY_BASE_DELAY_S", 0.0)

    class AlwaysConflictApi(_BaseFakeApi):
        def upload_folder(self, **kw) -> None:
            attempts.append(1)
            raise RuntimeError("Commit failed: concurrent update to refs/heads/main")

    monkeypatch.setattr(hub, "HfApi", AlwaysConflictApi)
    assert hub.upload_run_bundle(hf_run_dir, "acme/dd", "s6") is False
    assert len(attempts) == 3


def test_upload_run_bundle_retries_commit_conflict(
    hf_run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[int] = []
    delays: list[float] = []
    monkeypatch.setattr(hub, "_UPLOAD_RETRY_BASE_DELAY_S", 0.01)
    monkeypatch.setattr(hub.time, "sleep", lambda seconds: delays.append(seconds))

    class ConflictApi(_BaseFakeApi):
        def upload_folder(self, **kw) -> None:
            attempts.append(1)
            if len(attempts) < 3:
                raise RuntimeError("Commit failed: concurrent update to refs/heads/main")

    monkeypatch.setattr(hub, "HfApi", ConflictApi)
    assert hub.upload_run_bundle(hf_run_dir, "acme/dd", "s2") is True
    assert len(attempts) == 3
    # Exponential backoff between the three attempts: 0.01s then 0.02s.
    assert delays == [0.01, 0.02]


class _BoomApi:
    def __init__(self, token: str | None = None) -> None:
        raise AssertionError("HfApi must not be instantiated without HF_TOKEN")
