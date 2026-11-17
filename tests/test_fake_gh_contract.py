import asyncio
import subprocess
from pathlib import Path

import pytest

from daydream import git_ops
from daydream.git_ops import process as git_process
from daydream.github_app import resolve_user_identity
from tests.harness import github_schema
from tests.harness.fake_gh import install_fake_gh, install_gh_guard


def _budget() -> git_ops.GitHubRequestBudget:
    return git_ops.GitHubRequestBudget(deadline=float("inf"), per_request_seconds=10.0, monotonic=lambda: 0.0)


def test_gh_guard_fails_unowned_calls_while_fake_gh_owns_classifiable_absence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No route reaches a gh no fake owns, and the owning test fails even when production swallows the call.

    Fakes under the owned root take over. A valueless canned response must be a 404 production can classify:
    callers key absence off gh's own ``(HTTP 404)`` stderr token."""
    host = tmp_path / "host"
    host.mkdir()
    (host / "gh").write_text(f"#!/bin/sh\ntouch {host / 'ran'}\necho '{{\"login\": \"host\"}}'\n", encoding="utf-8")
    (host / "gh").chmod(0o755)
    host_auth = git_ops.StaticGitHubAuth({"PATH": str(host)})
    guard = install_gh_guard(tmp_path / "guard", tmp_path / "owned", monkeypatch)

    assert resolve_user_identity(tmp_path, auth=host_auth) == "unknown"
    stray = asyncio.run(git_process._run_gh_async(tmp_path, ["pr", "view", "7"], auth=host_auth, budget=_budget()))
    assert stray.returncode == 1
    assert subprocess.run(["sh", "-c", "gh auth status"], check=False).returncode == 1

    assert not (host / "ran").exists()
    with pytest.raises(AssertionError, match="no fake owns") as failure:
        guard.finish()
    assert failure.value.args[0].splitlines()[1:] == ["gh api /user", "gh pr view 7", "gh auth status"]

    fake_gh = install_fake_gh(tmp_path / "owned" / "fake-gh", monkeypatch)
    fake_gh.set_response("GET", "repos/o/r/contents/a.py")
    fake_gh.set_response("GET", "user", {"login": "fake"})
    with pytest.raises(git_ops.PathAbsentError) as excinfo:
        git_ops.gh_file_at_ref(tmp_path, "o/r", "0" * 40, "a.py")
    assert "HTTP 404" in str(excinfo.value)
    owned = asyncio.run(git_process._run_gh_async(tmp_path, ["api", "/user"], budget=_budget()))
    assert owned.returncode == 0 and "fake" in owned.stdout
    assert len(guard.calls()) == 3

    # The GraphQL schema subset the fake validates against must flag fields GitHub does not define.
    bad = """
    query X($o:String!,$n:String!,$p:Int!) {
      repository(owner:$o,name:$n) { pullRequest(number:$p) { reviewThreads(first:50) {
        nodes { id isResolved isResolvedBy side startSide
                comments(first:100) { nodes { author { login type isBot } } } }
      } } }
    }
    """
    assert {"side", "startSide", "isResolvedBy", "isBot", "type"} <= github_schema.unknown_query_fields(bad)
