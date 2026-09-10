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


def test_fake_gh_serves_an_absent_resource_as_a_recognizable_404(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A valueless canned response must be a 404 production can classify.

    Callers key absence off gh's own ``(HTTP 404)`` stderr token, so a fake
    that renders the miss any other way turns the suite's "this resource is
    absent" idiom into an unclassifiable failure.
    """
    fake_gh = install_fake_gh(tmp_path, monkeypatch)
    fake_gh.set_response("GET", "repos/o/r/contents/a.py")

    with pytest.raises(git_ops.PathAbsentError) as excinfo:
        git_ops.gh_file_at_ref(tmp_path, "o/r", "0" * 40, "a.py")

    assert "HTTP 404" in str(excinfo.value)
