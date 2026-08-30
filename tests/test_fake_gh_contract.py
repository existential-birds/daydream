from pathlib import Path

import pytest

from daydream import git_ops
from tests.harness.fake_gh import install_fake_gh


def test_fake_gh_requires_helper_fragment_in_git_ls_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_gh = install_fake_gh(tmp_path, monkeypatch)
    expected_refs = "c" * 40 + "\trefs/heads/main\n"
    fake_gh.set_response("git-ls-remote", value=expected_refs)

    refs = git_ops.git_ls_remote(tmp_path, "https://github.com/o/r.git")

    assert refs == expected_refs
    calls = fake_gh.command_calls("git ls-remote")
    assert len(calls) == 1
    call = calls[0]
    assert "-c" in call.argv
    assert any(arg == "credential.helper=!gh auth git-credential" for arg in call.argv)
    assert call.env is not None
    assert call.env["GIT_TERMINAL_PROMPT"] == "0"
