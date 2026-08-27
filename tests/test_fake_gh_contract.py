import os
from pathlib import Path

import pytest

from daydream import git_ops
from tests.harness.fake_gh import install_fake_gh


def test_fake_gh_rejects_credential_helper_without_operation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_gh(tmp_path, monkeypatch)
    proc = git_ops._run_gh(tmp_path, ["auth", "git-credential"])   # no get/store/erase
    assert proc.returncode != 0


def test_fake_gh_rejects_credential_helper_missing_stdin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_gh(tmp_path, monkeypatch)
    proc = git_ops._run_gh(tmp_path, ["auth", "git-credential", "get"], input_text="")
    assert proc.returncode != 0


def test_fake_gh_accepts_valid_credential_helper_get(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_gh(tmp_path, monkeypatch)
    proc = git_ops._run_gh(tmp_path, ["auth", "git-credential", "get"],
                           input_text="protocol=https\nhost=github.com\n")
    assert proc.returncode == 0 and "password=" in proc.stdout


def test_fake_gh_requires_helper_fragment_in_git_ls_remote(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install_fake_gh(tmp_path, monkeypatch)
    proc = git_ops._run_git(tmp_path, ["ls-remote", "https://github.com/o/r.git"],
                            env_cmd={**os.environ, "GIT_TERMINAL_PROMPT": "0"})
    assert proc.returncode != 0
