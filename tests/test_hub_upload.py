from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from daydream.archive import hub
from tests.harness.config import write_target_hub_key


class _RepoInfo:
    """Minimal stand-in for huggingface_hub's ``RepoInfo`` (visibility only)."""

    def __init__(self, private: bool) -> None:
        self.private = private


class _BaseFakeApi:
    """Fake ``HfApi`` defaults shared by upload tests: a private target repo."""

    def create_repo(self, repo_id: str, **kw: Any) -> None:
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


@pytest.mark.parametrize(
    ("cli_repo", "env_repo", "expected"),
    [
        ("cli/repo", "env/repo", "cli/repo"),  # CLI wins over env
        ("cli/repo", None, "cli/repo"),
        ("", "env/repo", "env/repo"),          # empty CLI falls through to env
        (None, "env/repo", "env/repo"),        # env wins when CLI unset
        (None, "", None),                     # empty env treated as unset
        (None, None, None),                    # neither source set -> unset
    ],
)
def test_resolve_hub_repo_prefers_cli_then_environment(
    monkeypatch: pytest.MonkeyPatch,
    cli_repo: str | None,
    env_repo: str | None,
    expected: str | None,
) -> None:
    from daydream.config_file import DaydreamFileConfig
    from daydream.runner import RunConfig

    if env_repo is None:
        monkeypatch.delenv("DAYDREAM_TRAJECTORY_HUB_REPO", raising=False)
    else:
        monkeypatch.setenv("DAYDREAM_TRAJECTORY_HUB_REPO", env_repo)
    # A target checkout's file config is present but can never select a destination
    file_cfg = DaydreamFileConfig(model="target-file-marker")
    assert hub.resolve_hub_repo(
        RunConfig(trajectory_hub_repo=cli_repo, file_config=file_cfg)
    ) == expected


def test_target_file_config_never_selects_hub_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A target pyproject.toml setting trajectory_hub_repo must resolve to None:
    the file-config tier is gone and contributes nothing."""
    from daydream.config_file import load_file_config
    from daydream.runner import RunConfig

    monkeypatch.delenv("DAYDREAM_TRAJECTORY_HUB_REPO", raising=False)
    write_target_hub_key(tmp_path)
    file_cfg = load_file_config(tmp_path)  # contains the key, must be ignored
    assert hub.resolve_hub_repo(RunConfig(file_config=file_cfg)) is None


def test_upload_run_bundle_creates_private_repo_and_uploads(
    hf_run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, Any] = {}

    class FakeApi:
        def __init__(self, token: str | None = None) -> None:
            calls["token"] = token

        def create_repo(self, repo_id: str, *, repo_type: str, private: bool, exist_ok: bool) -> None:
            calls["create_repo"] = (repo_id, repo_type, private, exist_ok)

        def repo_info(self, repo_id: str, *, repo_type: str) -> _RepoInfo:
            calls["repo_info"] = (repo_id, repo_type)
            return _RepoInfo(private=True)

        def upload_folder(
            self,
            *,
            folder_path: str,
            repo_id: str,
            repo_type: str,
            path_in_repo: str,
            commit_message: str,
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
    calls: dict[str, Any] = {}

    class PublicRepoApi(_BaseFakeApi):
        def create_repo(self, repo_id: str, **kw: Any) -> None:
            calls["create_repo"] = True

        def repo_info(self, repo_id: str, *, repo_type: str) -> _RepoInfo:
            calls["repo_info"] = (repo_id, repo_type)
            return _RepoInfo(private=False)

        def upload_folder(self, **kw: Any) -> None:
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
        def create_repo(self, repo_id: str, **kw: Any) -> None:
            raise RuntimeError("401 Unauthorized: invalid token")

    monkeypatch.setattr(hub, "HfApi", CreateRepoBoomApi)
    assert hub.upload_run_bundle(hf_run_dir, "acme/dd", "s4") is False


def test_upload_run_bundle_non_conflict_upload_error(
    hf_run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts: list[int] = []

    class NonConflictApi(_BaseFakeApi):
        def upload_folder(self, **kw: Any) -> None:
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
        def upload_folder(self, **kw: Any) -> None:
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
    monkeypatch.setattr(time, "sleep", lambda seconds: delays.append(seconds))

    class ConflictApi(_BaseFakeApi):
        def upload_folder(self, **kw: Any) -> None:
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


def _write_manifest_with_remote(run_dir: Path, remote_url: str) -> None:
    """Write a manifest.json whose git context carries *remote_url*."""
    (run_dir / "manifest.json").write_text(
        json.dumps({"git": {"remote_url": remote_url}}), encoding="utf-8"
    )


def _install_fake_hfapi(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Swap hub.HfApi for a fake; return the list of upload_folder invocations."""
    calls: list[dict[str, Any]] = []

    class FakeApi(_BaseFakeApi):
        def upload_folder(self, **kw: Any) -> None:
            calls.append(kw)

    monkeypatch.setattr(hub, "HfApi", FakeApi)
    return calls


def test_upload_refused_when_bundle_contains_credential(
    hf_run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_manifest_with_remote(hf_run_dir, "https://user:ghp_canaryfake123@github.com/o/r")
    uploads = _install_fake_hfapi(monkeypatch)
    assert hub.upload_run_bundle(hf_run_dir, "org/ds", "s1") is False
    assert uploads == []  # upload_folder never invoked


def test_upload_canary_never_echoed_in_warning(
    hf_run_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _write_manifest_with_remote(hf_run_dir, "https://user:ghp_canaryfake123@github.com/o/r")
    _install_fake_hfapi(monkeypatch)
    hub.upload_run_bundle(hf_run_dir, "org/ds", "s1")
    out = capsys.readouterr().out + capsys.readouterr().err
    assert "ghp_canaryfake123" not in out


def test_upload_proceeds_for_clean_bundle(
    hf_run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_manifest_with_remote(hf_run_dir, "https://github.com/o/r")
    uploads = _install_fake_hfapi(monkeypatch)
    assert hub.upload_run_bundle(hf_run_dir, "org/ds", "s1") is True
    assert len(uploads) == 1
