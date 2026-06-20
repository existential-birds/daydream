"""Tests for :mod:`daydream.git_ops`.

These tests use real ``git`` (and optionally ``gh``) against ``tmp_path``
fixtures.  No subprocess mocking — every code path is exercised against an
actual repository.  ``gh``-dependent tests are skipped when the binary is
unavailable.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from conftest import (
    _bare_remote,
    _commit,
    _configure_identity,
    _git,
    _init_repo,
    _make_repo_with_main,
)

from daydream import git_ops
from daydream.git_ops import (
    BranchNotFoundError,
    GitError,
    NotAWorktreeError,
    WrongBranchError,
)

# --- assert_is_worktree / is_inside_worktree --------------------------------


def test_assert_is_worktree_passes_for_real_repo(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    git_ops.assert_is_worktree(repo)
    assert git_ops.is_inside_worktree(repo) is True


def test_assert_is_worktree_rejects_non_repo(tmp_path: Path) -> None:
    with pytest.raises(NotAWorktreeError):
        git_ops.assert_is_worktree(tmp_path)
    assert git_ops.is_inside_worktree(tmp_path) is False


def test_assert_is_worktree_rejects_org_dir(tmp_path: Path) -> None:
    """An "org" dir that contains repos is NOT itself a worktree."""
    org = tmp_path / "org"
    org.mkdir()
    _make_repo_with_main(org, name="child-repo")

    with pytest.raises(NotAWorktreeError):
        git_ops.assert_is_worktree(org)
    assert git_ops.is_inside_worktree(org) is False


def test_assert_is_worktree_rejects_subdir_of_repo(tmp_path: Path) -> None:
    """A subdir inside a repo is "inside a worktree" but not its top-level."""
    repo = _make_repo_with_main(tmp_path)
    sub = repo / "src"
    sub.mkdir()
    (sub / "x.txt").write_text("x\n")

    with pytest.raises(NotAWorktreeError):
        git_ops.assert_is_worktree(sub)
    assert git_ops.is_inside_worktree(sub) is False


def test_assert_is_worktree_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(NotAWorktreeError):
        git_ops.assert_is_worktree(tmp_path / "does-not-exist")


# --- Read-only queries ------------------------------------------------------


def test_head_sha_returns_full_sha(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    expected = _git(repo, "rev-parse", "HEAD")
    assert git_ops.head_sha(repo) == expected
    assert len(expected) == 40


def test_head_sha_raises_on_empty_repo(tmp_path: Path) -> None:
    repo = tmp_path / "empty"
    _init_repo(repo)
    with pytest.raises(GitError):
        git_ops.head_sha(repo)


def test_current_branch_on_named_branch(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    assert git_ops.current_branch(repo) == "main"


def test_current_branch_returns_none_when_detached(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "--detach", sha)
    assert git_ops.current_branch(repo) is None


def test_default_branch_uses_origin_head(tmp_path: Path) -> None:
    bare = _bare_remote(tmp_path / "remote.git")
    repo = _make_repo_with_main(tmp_path, name="repo")
    _git(repo, "remote", "add", "origin", str(bare))
    _git(repo, "push", "-u", "origin", "main")
    _git(repo, "remote", "set-head", "origin", "main")
    assert git_ops.default_branch(repo) == "main"


def test_default_branch_falls_back_to_main(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    # No origin/HEAD — must fall back to local main.
    assert git_ops.default_branch(repo) == "main"


def test_default_branch_falls_back_to_master(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "master")
    _configure_identity(repo)
    (repo / "f.txt").write_text("hi\n")
    _git(repo, "add", "f.txt")
    _commit(repo, "first")
    assert git_ops.default_branch(repo) == "master"


def test_default_branch_raises_when_none_present(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "trunk")
    _configure_identity(repo)
    (repo / "f.txt").write_text("hi\n")
    _git(repo, "add", "f.txt")
    _commit(repo, "first")
    with pytest.raises(BranchNotFoundError):
        git_ops.default_branch(repo)


def test_remote_url_returns_url_when_remote_configured(tmp_path: Path) -> None:
    bare = _bare_remote(tmp_path / "remote.git")
    repo = _make_repo_with_main(tmp_path, name="repo")
    _git(repo, "remote", "add", "origin", str(bare))
    assert git_ops.remote_url(repo) == str(bare)


def test_remote_url_returns_none_when_remote_missing(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    assert git_ops.remote_url(repo) is None


def test_remote_url_returns_none_for_unknown_remote_name(tmp_path: Path) -> None:
    bare = _bare_remote(tmp_path / "remote.git")
    repo = _make_repo_with_main(tmp_path, name="repo")
    _git(repo, "remote", "add", "origin", str(bare))
    assert git_ops.remote_url(repo, "upstream") is None


def test_branch_exists_local(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    _git(repo, "checkout", "-b", "feat-local")
    assert git_ops.branch_exists(repo, "feat-local") is True


def test_branch_exists_origin_only(tmp_path: Path) -> None:
    bare = _bare_remote(tmp_path / "remote.git")
    repo = _make_repo_with_main(tmp_path, name="repo")
    _git(repo, "remote", "add", "origin", str(bare))
    _git(repo, "push", "-u", "origin", "main")
    _git(repo, "checkout", "-b", "remote-only")
    (repo / "r.txt").write_text("r\n")
    _git(repo, "add", "r.txt")
    _commit(repo, "remote-only commit")
    _git(repo, "push", "-u", "origin", "remote-only")
    _git(repo, "checkout", "main")
    _git(repo, "branch", "-D", "remote-only")
    assert git_ops.branch_exists(repo, "remote-only") is True


def test_branch_exists_missing(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    assert git_ops.branch_exists(repo, "nonexistent") is False


# --- ref_exists -------------------------------------------------------------


def test_ref_exists_raw_sha(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    sha = _git(repo, "rev-parse", "HEAD")
    assert git_ops.ref_exists(repo, sha) is True


def test_ref_exists_abbreviated_sha(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    short = _git(repo, "rev-parse", "--short", "HEAD")
    assert git_ops.ref_exists(repo, short) is True


def test_ref_exists_tag(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    _git(repo, "tag", "v1.0")
    assert git_ops.ref_exists(repo, "v1.0") is True


def test_ref_exists_relative_commit_ish(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    (repo / "two.txt").write_text("two\n")
    _git(repo, "add", "two.txt")
    _commit(repo, "second")
    assert git_ops.ref_exists(repo, "HEAD~1") is True


def test_ref_exists_named_branch_still_resolves(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    _git(repo, "checkout", "-b", "feat-local")
    assert git_ops.ref_exists(repo, "feat-local") is True


def test_ref_exists_missing(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    assert git_ops.ref_exists(repo, "nonexistent") is False
    # A tree-ish that is not a commit must be rejected, not accepted.
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    assert git_ops.ref_exists(repo, tree) is False


def test_ref_exists_rejects_leading_dash(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    assert git_ops.ref_exists(repo, "-not-a-ref") is False
    assert git_ops.ref_exists(repo, "--exec=evil") is False


# --- merge_base -------------------------------------------------------------


def test_merge_base_returns_shared_commit(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)

    _git(repo, "checkout", "-b", "feat")
    (repo / "feat.txt").write_text("feat\n")
    _git(repo, "add", "feat.txt")
    _commit(repo, "feat commit")

    _git(repo, "checkout", "main")
    (repo / "main2.txt").write_text("more main\n")
    _git(repo, "add", "main2.txt")
    _commit(repo, "main commit")

    _git(repo, "checkout", "feat")
    expected = _git(repo, "merge-base", "HEAD", "main")
    assert git_ops.merge_base(repo, "main") == expected


def test_merge_base_prefers_upstream_when_remote_ahead(tmp_path: Path) -> None:
    """Port of codex's merge_base_prefers_upstream_when_remote_ahead test."""
    bare = _bare_remote(tmp_path / "remote.git")
    repo = _make_repo_with_main(tmp_path, name="repo")
    _git(repo, "remote", "add", "origin", str(bare))
    _git(repo, "push", "-u", "origin", "main")

    _git(repo, "checkout", "-b", "feature")
    (repo / "feature.txt").write_text("feature\n")
    _git(repo, "add", "feature.txt")
    _commit(repo, "feature commit")

    # Rewrite local main as an unrelated history; track origin/main so the
    # upstream is "ahead" of the rewritten local main.
    _git(repo, "checkout", "--orphan", "rewrite")
    _git(repo, "rm", "-rf", ".")
    (repo / "new-main.txt").write_text("rewritten\n")
    _git(repo, "add", "new-main.txt")
    _commit(repo, "rewrite main")
    _git(repo, "branch", "-M", "rewrite", "main")
    _git(repo, "branch", "--set-upstream-to=origin/main", "main")

    _git(repo, "checkout", "feature")
    _git(repo, "fetch", "origin")

    expected = _git(repo, "merge-base", "HEAD", "origin/main")
    assert git_ops.merge_base(repo, "main") == expected


def test_merge_base_returns_none_for_missing_branch(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    assert git_ops.merge_base(repo, "missing-branch") is None


def test_merge_base_returns_none_when_head_missing(tmp_path: Path) -> None:
    repo = tmp_path / "empty"
    _init_repo(repo)
    # No commits → no HEAD.
    assert git_ops.merge_base(repo, "main") is None


def test_merge_base_returns_none_for_leading_dash_base(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    assert git_ops.merge_base(repo, "-no-such-ref") is None


def test_merge_base_returns_none_for_leading_dash_head(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    assert git_ops.merge_base(repo, "main", "-no-such-ref") is None


# --- diff / log / show / grep / status / upstream_ahead_count ---------------


def test_diff_returns_changes(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    _git(repo, "checkout", "-b", "topic")
    (repo / "added.txt").write_text("hello\n")
    _git(repo, "add", "added.txt")
    _commit(repo, "topic commit")
    out = git_ops.diff(repo, "main")
    assert "added.txt" in out
    assert "+hello" in out


def test_diff_excludes_paths(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    _git(repo, "checkout", "-b", "topic")
    (repo / "keep.txt").write_text("keep\n")
    (repo / "drop.txt").write_text("drop\n")
    _git(repo, "add", "keep.txt", "drop.txt")
    _commit(repo, "topic")
    out = git_ops.diff(repo, "main", exclude=["drop.txt"])
    assert "keep.txt" in out
    assert "drop.txt" not in out


def test_diff_prefers_origin_when_on_default_branch(tmp_path: Path) -> None:
    """When HEAD is on main, diff against origin/main shows unpushed commits."""
    remote_dir = tmp_path / "remote.git"
    remote_dir.mkdir()
    _git(remote_dir, "init", "--bare", "-b", "main")

    repo = _make_repo_with_main(tmp_path)
    _git(repo, "remote", "add", "origin", str(remote_dir))
    _git(repo, "push", "-u", "origin", "main")

    # Local commit on main — not pushed.
    (repo / "local.txt").write_text("local change\n")
    _git(repo, "add", "local.txt")
    _commit(repo, "local only")

    out = git_ops.diff(repo, "main")
    assert "local.txt" in out, "diff should show unpushed changes vs origin/main"


# --- diff_name_only ---------------------------------------------------------


def test_diff_name_only_returns_changed_files(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    _git(repo, "checkout", "-b", "topic")
    (repo / "added.txt").write_text("hello\n")
    _git(repo, "add", "added.txt")
    _commit(repo, "add file")
    result = git_ops.diff_name_only(repo, "main", "HEAD")
    assert result == ["added.txt"]


def test_diff_name_only_returns_multiple_files_in_order(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    _git(repo, "checkout", "-b", "topic")
    (repo / "alpha.txt").write_text("a\n")
    (repo / "beta.txt").write_text("b\n")
    _git(repo, "add", "alpha.txt", "beta.txt")
    _commit(repo, "add two files")
    result = git_ops.diff_name_only(repo, "main", "HEAD")
    assert sorted(result) == ["alpha.txt", "beta.txt"]


def test_diff_name_only_filters_empty_lines(tmp_path: Path) -> None:
    """Ensure blank lines in git output are stripped (empty-line filtering)."""
    repo = _make_repo_with_main(tmp_path)
    _git(repo, "checkout", "-b", "topic")
    (repo / "file.txt").write_text("x\n")
    _git(repo, "add", "file.txt")
    _commit(repo, "add file")
    result = git_ops.diff_name_only(repo, "main", "HEAD")
    assert all(line != "" for line in result)
    assert "file.txt" in result


def test_diff_name_only_returns_empty_list_on_bad_ref(tmp_path: Path) -> None:
    """Soft-failure: unresolvable ref yields [] rather than raising."""
    repo = _make_repo_with_main(tmp_path)
    result = git_ops.diff_name_only(repo, "nonexistent-ref", "HEAD")
    assert result == []


def test_log_returns_oneline_commits(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    _git(repo, "checkout", "-b", "topic")
    (repo / "a.txt").write_text("a\n")
    _git(repo, "add", "a.txt")
    _commit(repo, "topic-msg")
    out = git_ops.log(repo, "main")
    assert "topic-msg" in out


def test_show_returns_file_bytes_at_ref(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    out = git_ops.show(repo, "HEAD", "base.txt")
    assert out == b"base\n"


def test_show_raises_on_missing_path(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    with pytest.raises(GitError):
        git_ops.show(repo, "HEAD", "nope.txt")


def test_grep_returns_matching_paths(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    (repo / "needle.txt").write_text("findme\n")
    (repo / "miss.txt").write_text("nope\n")
    _git(repo, "add", "needle.txt", "miss.txt")
    _commit(repo, "add files")
    matches = git_ops.grep(repo, "findme")
    assert "needle.txt" in matches
    assert "miss.txt" not in matches


def test_grep_returns_empty_when_no_matches(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    assert git_ops.grep(repo, "doesnotexistanywhere") == []


def test_status_porcelain_clean_and_dirty(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    assert git_ops.status_porcelain(repo) == ""
    (repo / "untracked.txt").write_text("u\n")
    out = git_ops.status_porcelain(repo)
    assert "untracked.txt" in out


def test_upstream_ahead_count_no_upstream(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    assert git_ops.upstream_ahead_count(repo, "main") == 0


def test_upstream_ahead_count_when_remote_ahead(tmp_path: Path) -> None:
    bare = _bare_remote(tmp_path / "remote.git")
    repo = _make_repo_with_main(tmp_path, name="repo")
    _git(repo, "remote", "add", "origin", str(bare))
    _git(repo, "push", "-u", "origin", "main")

    # Push two extra commits to origin/main via a sidecar clone, then fetch.
    sidecar = tmp_path / "sidecar"
    _git(tmp_path, "clone", str(bare), str(sidecar))
    _configure_identity(sidecar)
    (sidecar / "x.txt").write_text("x\n")
    _git(sidecar, "add", "x.txt")
    _commit(sidecar, "x")
    (sidecar / "y.txt").write_text("y\n")
    _git(sidecar, "add", "y.txt")
    _commit(sidecar, "y")
    _git(sidecar, "push", "origin", "main")

    _git(repo, "fetch", "origin")
    assert git_ops.upstream_ahead_count(repo, "main") == 2


# --- Mutating ---------------------------------------------------------------


def test_fetch_pulls_new_commits(tmp_path: Path) -> None:
    bare = _bare_remote(tmp_path / "remote.git")
    repo = _make_repo_with_main(tmp_path, name="repo")
    _git(repo, "remote", "add", "origin", str(bare))
    _git(repo, "push", "-u", "origin", "main")

    sidecar = tmp_path / "sidecar"
    _git(tmp_path, "clone", str(bare), str(sidecar))
    _configure_identity(sidecar)
    (sidecar / "z.txt").write_text("z\n")
    _git(sidecar, "add", "z.txt")
    new_sha = _commit(sidecar, "z")
    _git(sidecar, "push", "origin", "main")

    git_ops.fetch(repo)
    assert _git(repo, "rev-parse", "origin/main") == new_sha


def test_checkout_paths_restores_working_tree(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    (repo / "base.txt").write_text("MUTATED\n")
    assert (repo / "base.txt").read_text() == "MUTATED\n"
    git_ops.checkout_paths(repo, [Path(".")])
    assert (repo / "base.txt").read_text() == "base\n"


def test_clean_untracked_removes_files(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    (repo / "junk.txt").write_text("trash\n")
    (repo / "junkdir").mkdir()
    (repo / "junkdir" / "x.txt").write_text("x\n")
    git_ops.clean_untracked(repo)
    assert not (repo / "junk.txt").exists()
    assert not (repo / "junkdir").exists()


def test_worktree_add_and_remove_round_trip(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    head = _git(repo, "rev-parse", "HEAD")
    wt = tmp_path / "wt"

    git_ops.worktree_add(repo, wt, head)
    assert wt.exists()
    assert (wt / "base.txt").read_text() == "base\n"
    # The worktree itself should pass the assertion.
    git_ops.assert_is_worktree(wt)

    git_ops.worktree_remove(repo, wt)
    assert not wt.exists()


# --- branch / commit / push primitives (Task 3) -----------------------------


def test_commit_paths_on_new_branch_pushes_to_origin(repo_with_origin: Path) -> None:
    (repo_with_origin / ".github/workflows").mkdir(parents=True)
    (repo_with_origin / ".github/workflows/daydream-review.yml").write_text("name: x\n")
    git_ops.create_branch(repo_with_origin, "daydream/setup")
    git_ops.commit_paths(
        repo_with_origin,
        [Path(".github/workflows/daydream-review.yml")],
        "add bot workflows",
    )
    git_ops.push_branch(repo_with_origin, "daydream/setup")
    assert git_ops.ref_exists(repo_with_origin, "origin/daydream/setup")


def test_create_branch_raises_when_branch_exists(repo_with_origin: Path) -> None:
    git_ops.create_branch(repo_with_origin, "daydream/dup")
    _git(repo_with_origin, "checkout", "main")
    with pytest.raises(GitError):
        git_ops.create_branch(repo_with_origin, "daydream/dup")


def test_commit_paths_commits_only_named_paths(repo_with_origin: Path) -> None:
    """commit_paths stages only the named files — never a blanket ``-A``."""
    git_ops.create_branch(repo_with_origin, "daydream/selective")
    (repo_with_origin / "tracked.txt").write_text("staged\n")
    (repo_with_origin / "untouched.txt").write_text("left behind\n")
    git_ops.commit_paths(repo_with_origin, [Path("tracked.txt")], "add tracked only")
    # The committed tree contains tracked.txt but not untouched.txt.
    committed = _git(repo_with_origin, "show", "--name-only", "--format=", "HEAD").split()
    assert committed == ["tracked.txt"]
    # untouched.txt is still an uncommitted, untracked file.
    assert "untouched.txt" in _git(repo_with_origin, "status", "--porcelain")


def test_push_branch_failure_raises_git_error(git_repo: Path) -> None:
    """push_branch propagates a push failure (no ``origin`` remote) as GitError."""
    git_ops.create_branch(git_repo, "daydream/no-remote")
    (git_repo / "f.txt").write_text("x\n")
    git_ops.commit_paths(git_repo, [Path("f.txt")], "add f")
    with pytest.raises(GitError):
        git_ops.push_branch(git_repo, "daydream/no-remote")


# --- Error type identity ----------------------------------------------------


def test_error_hierarchy_is_consistent() -> None:
    assert issubclass(NotAWorktreeError, GitError)
    assert issubclass(BranchNotFoundError, GitError)
    assert issubclass(WrongBranchError, GitError)
    assert issubclass(git_ops.GitTimeoutError, GitError)


# --- _run_git timeout retry (issue #120) ------------------------------------


def test_run_git_timeout_retry_behavior(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`_run_git` retries transient timeouts but not other failures (#120).

    One test, three cases:
      * a timeout on the first attempt is retried and the later success returns;
      * timeouts that exhaust every attempt raise the distinct GitTimeoutError;
      * a non-timeout subprocess error raises a plain GitError immediately, with
        no wasted retries.
    """
    repo = _make_repo_with_main(tmp_path)
    ok = subprocess.CompletedProcess(args=["git"], returncode=0, stdout="true\n", stderr="")

    # Case 1: transient timeout (1st attempt) then success on the retry.
    calls = {"n": 0}

    def flaky_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise subprocess.TimeoutExpired(cmd=["git"], timeout=5)
        return ok

    monkeypatch.setattr("daydream.git_ops.subprocess.run", flaky_run)
    assert git_ops._run_git(repo, ["rev-parse", "HEAD"]).returncode == 0
    assert calls["n"] == 2  # timed out once, then succeeded

    # Case 2: every attempt times out -> GitTimeoutError after retries+1 tries.
    calls["n"] = 0

    def always_timeout(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        calls["n"] += 1
        raise subprocess.TimeoutExpired(cmd=["git"], timeout=5)

    monkeypatch.setattr("daydream.git_ops.subprocess.run", always_timeout)
    with pytest.raises(git_ops.GitTimeoutError):
        git_ops._run_git(repo, ["rev-parse", "HEAD"], retries=2)
    assert calls["n"] == 3  # 1 initial + 2 retries

    # Case 3: non-timeout failure raises plain GitError immediately (no retry).
    calls["n"] = 0

    def os_error(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        calls["n"] += 1
        raise OSError("git binary missing")

    monkeypatch.setattr("daydream.git_ops.subprocess.run", os_error)
    with pytest.raises(GitError) as exc:
        git_ops._run_git(repo, ["rev-parse", "HEAD"], retries=2)
    assert not isinstance(exc.value, git_ops.GitTimeoutError)
    assert calls["n"] == 1  # no retries for non-timeout failures


def test_mutating_wrapper_does_not_retry_on_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Mutating wrappers pass `retries=0`, so a timeout is not re-run (#120).

    Re-running a non-idempotent git command after a timeout could land on top of
    partial repo changes. `fetch` drives the production path through `_run_git`
    and must raise GitTimeoutError after exactly one attempt.
    """
    repo = _make_repo_with_main(tmp_path)
    calls = {"n": 0}

    def always_timeout(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        calls["n"] += 1
        raise subprocess.TimeoutExpired(cmd=["git"], timeout=30)

    monkeypatch.setattr("daydream.git_ops.subprocess.run", always_timeout)
    with pytest.raises(git_ops.GitTimeoutError):
        git_ops.fetch(repo)
    assert calls["n"] == 1  # no retries for mutating operations


# --- _run_gh timeout retry (fake-gh flake under load) -----------------------


def test_run_gh_read_wrapper_retries_then_succeeds_and_exhausts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Read-only ``gh`` wrappers ride out transient host-load timeouts.

    A `gh` subprocess that times out under CPU starvation is not a hung command,
    so read-only wrappers retry. Two cases through the production path:
      * a timeout on the first attempt is retried and the later success returns;
      * timeouts that exhaust every attempt raise GitTimeoutError after exactly
        ``_gh_retries() + 1`` tries.
    """
    repo = _make_repo_with_main(tmp_path)
    ok = subprocess.CompletedProcess(args=["gh"], returncode=0, stdout="octocat/hello\n", stderr="")

    # Case 1: transient timeout (1st attempt) then success on the retry.
    calls = {"n": 0}

    def flaky_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        calls["n"] += 1
        if calls["n"] == 1:
            raise subprocess.TimeoutExpired(cmd=["gh"], timeout=60)
        return ok

    monkeypatch.setattr("daydream.git_ops.subprocess.run", flaky_run)
    assert git_ops.gh_repo_view(repo) == ("octocat", "hello")
    assert calls["n"] == 2  # timed out once, then succeeded

    # Case 2: every attempt times out -> GitTimeoutError after retries+1 tries.
    calls["n"] = 0

    def always_timeout(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        calls["n"] += 1
        raise subprocess.TimeoutExpired(cmd=["gh"], timeout=60)

    monkeypatch.setattr("daydream.git_ops.subprocess.run", always_timeout)
    with pytest.raises(git_ops.GitTimeoutError):
        git_ops.gh_pr_diff(repo, 7)
    assert calls["n"] == git_ops._gh_retries() + 1


def test_run_gh_mutation_does_not_retry_on_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Mutating ``gh`` wrappers default to ``retries=0`` so a timeout is not re-run.

    Re-running a non-idempotent ``gh`` call (here ``pr create``) after a timeout
    could open a duplicate PR. The production path must raise GitTimeoutError
    after exactly one attempt.
    """
    repo = _make_repo_with_main(tmp_path)
    calls = {"n": 0}

    def always_timeout(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        calls["n"] += 1
        raise subprocess.TimeoutExpired(cmd=["gh"], timeout=60)

    monkeypatch.setattr("daydream.git_ops.subprocess.run", always_timeout)
    with pytest.raises(git_ops.GitTimeoutError):
        git_ops.gh_pr_create(repo, head="feature", base="main", title="t", body="b")
    assert calls["n"] == 1  # no retries for mutating operations


def test_gh_api_retries_only_when_idempotent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``gh_api`` retries reads but never mutations — incl. GraphQL.

    HTTP method cannot tell a GraphQL query from a mutation (both POST), so the
    retry decision is the caller's ``idempotent`` flag. A read (``idempotent=True``)
    rides out timeouts; a mutation (the ``input_data`` POST path, default flag)
    raises after a single attempt so a comment is never double-posted.
    """
    repo = _make_repo_with_main(tmp_path)
    calls = {"n": 0}

    def always_timeout(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        calls["n"] += 1
        raise subprocess.TimeoutExpired(cmd=["gh"], timeout=60)

    monkeypatch.setattr("daydream.git_ops.subprocess.run", always_timeout)

    # Read: idempotent=True -> retried to exhaustion.
    with pytest.raises(git_ops.GitTimeoutError):
        git_ops.gh_api(repo, "/user", idempotent=True)
    assert calls["n"] == git_ops._gh_retries() + 1

    # Mutation: a GraphQL-shaped POST with a body, default flag -> no retry.
    calls["n"] = 0
    with pytest.raises(git_ops.GitTimeoutError):
        git_ops.gh_api(repo, "graphql", method="POST", input_data={"query": "mutation { x }"})
    assert calls["n"] == 1


def test_wrong_branch_error_is_raisable() -> None:
    with pytest.raises(WrongBranchError):
        raise WrongBranchError("expected feat, got main")
    with pytest.raises(GitError):
        raise WrongBranchError("subclass check")


# --- gh wrappers (skipped when gh missing) ----------------------------------


_gh_available = shutil.which("gh") is not None
gh_required = pytest.mark.skipif(not _gh_available, reason="gh CLI not installed")


@gh_required
def test_gh_repo_view_returns_none_outside_github_repo(tmp_path: Path) -> None:
    """A local-only repo with no GitHub remote yields ``None``."""
    repo = _make_repo_with_main(tmp_path)
    assert git_ops.gh_repo_view(repo) is None


@gh_required
def test_gh_pr_view_returns_none_for_missing_pr(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    # No GitHub remote → gh fails → wrapper returns None instead of raising.
    assert git_ops.gh_pr_view(repo, 999999) is None


@gh_required
def test_gh_pr_list_for_branch_returns_empty_without_remote(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    assert git_ops.gh_pr_list_for_branch(repo, "main") == []


@gh_required
def test_gh_pr_diff_raises_without_remote(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    with pytest.raises(GitError):
        git_ops.gh_pr_diff(repo, 1)


@gh_required
def test_gh_api_raises_without_auth(tmp_path: Path) -> None:
    """``gh api`` against a relative endpoint with no GitHub remote fails."""
    repo = _make_repo_with_main(tmp_path)
    with pytest.raises(GitError):
        git_ops.gh_api(repo, "repos/{owner}/{repo}")


# --- diff_paths -------------------------------------------------------------


def _make_divergent_history(tmp_path: Path) -> tuple[Path, str, str]:
    """Build a repo where main has advanced after `feat` branched off.

    Returns the repo path plus the names of the two branches (`main`, `feat`).
    The same file (`shared.txt`) is modified on both branches so two-dot vs
    three-dot diffs differ in content.
    """
    repo = _make_repo_with_main(tmp_path)
    (repo / "shared.txt").write_text("line one\nline two\nline three\n")
    _git(repo, "add", "shared.txt")
    _commit(repo, "shared baseline")

    _git(repo, "checkout", "-b", "feat")
    (repo / "shared.txt").write_text("line one\nline two FEAT\nline three\n")
    _git(repo, "add", "shared.txt")
    _commit(repo, "feat edit")

    _git(repo, "checkout", "main")
    (repo / "shared.txt").write_text("line one\nline two MAIN\nline three\n")
    _git(repo, "add", "shared.txt")
    _commit(repo, "main edit after branch")

    _git(repo, "checkout", "feat")
    return repo, "main", "feat"


def test_diff_paths_direct_vs_merge_base_differ_on_divergent_history(
    tmp_path: Path,
) -> None:
    """Direct diff includes main's later commit; merge-base diff does not. Pin the diff."""
    repo, base, head = _make_divergent_history(tmp_path)
    direct = git_ops.diff_paths(repo, base, head, ["shared.txt"], merge_base_diff=False)
    since_merge_base = git_ops.diff_paths(repo, base, head, ["shared.txt"], merge_base_diff=True)
    assert direct != since_merge_base
    # Direct diff shows main's "MAIN" line as the - side; merge-base diff doesn't.
    assert "MAIN" in direct
    assert "MAIN" not in since_merge_base


def test_diff_paths_restricts_to_paths(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    _git(repo, "checkout", "-b", "feat")
    (repo / "keep.txt").write_text("keep\n")
    (repo / "drop.txt").write_text("drop\n")
    _git(repo, "add", "keep.txt", "drop.txt")
    _commit(repo, "two files")
    out = git_ops.diff_paths(repo, "main", "feat", ["keep.txt"])
    assert "keep.txt" in out
    assert "drop.txt" not in out


def test_diff_paths_unified_context_lines(tmp_path: Path) -> None:
    """Larger --unified yields a longer diff for the same change."""
    repo = _make_repo_with_main(tmp_path)
    (repo / "ctx.txt").write_text("\n".join(f"line {i}" for i in range(1, 31)) + "\n")
    _git(repo, "add", "ctx.txt")
    _commit(repo, "ctx baseline")

    _git(repo, "checkout", "-b", "feat")
    lines = [f"line {i}" for i in range(1, 31)]
    lines[14] = "line 15 CHANGED"
    (repo / "ctx.txt").write_text("\n".join(lines) + "\n")
    _git(repo, "add", "ctx.txt")
    _commit(repo, "ctx edit")

    small = git_ops.diff_paths(repo, "main", "feat", ["ctx.txt"], unified=1)
    big = git_ops.diff_paths(repo, "main", "feat", ["ctx.txt"], unified=10)
    assert len(big) > len(small)


def test_diff_paths_raises_on_invalid_ref(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    with pytest.raises(GitError):
        git_ops.diff_paths(repo, "definitely-not-a-ref", "HEAD", ["base.txt"])


# --- gh_api(input_data=...) and gh_pr_view(pr=None) -------------------------
# These tests exercise wrapper logic, not gh itself: subprocess is monkeypatched
# to capture argv and drive success/failure paths deterministically.


def test_gh_api_input_data_passes_tempfile_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _make_repo_with_main(tmp_path)
    captured: dict[str, Any] = {}

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        idx = cmd.index("--input")
        captured["input_path"] = cmd[idx + 1]
        # Confirm the tempfile exists at call time and holds our payload.
        captured["payload"] = Path(cmd[idx + 1]).read_text(encoding="utf-8")
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout='{"ok": true}', stderr=""
        )

    monkeypatch.setattr("daydream.git_ops.subprocess.run", fake_run)

    result = git_ops.gh_api(
        repo,
        "repos/owner/repo/pulls/1/reviews",
        method="POST",
        input_data={"event": "COMMENT", "body": "hi"},
    )

    assert result == {"ok": True}
    cmd = captured["cmd"]
    assert cmd[:2] == ["gh", "api"]
    assert "--input" in cmd
    assert "--method" in cmd
    method_idx = cmd.index("--method")
    assert cmd[method_idx + 1] == "POST"
    assert json.loads(captured["payload"]) == {"event": "COMMENT", "body": "hi"}
    # Success path: tempfile must have been deleted.
    assert not Path(captured["input_path"]).exists()


def test_gh_api_input_data_preserves_tempfile_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _make_repo_with_main(tmp_path)
    captured: dict[str, Any] = {}

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        idx = cmd.index("--input")
        captured["input_path"] = cmd[idx + 1]
        return subprocess.CompletedProcess(
            args=cmd, returncode=1, stdout="", stderr="HTTP 422: Validation failed"
        )

    monkeypatch.setattr("daydream.git_ops.subprocess.run", fake_run)

    with pytest.raises(GitError) as excinfo:
        git_ops.gh_api(
            repo,
            "repos/owner/repo/pulls/1/reviews",
            method="POST",
            input_data={"bad": "payload"},
        )

    msg = str(excinfo.value)
    assert "payload preserved at" in msg
    # The tempfile path mentioned in the error must still exist on disk.
    preserved = Path(captured["input_path"])
    assert str(preserved) in msg
    assert preserved.exists()
    # Cleanup so the test doesn't leave debris behind.
    preserved.unlink(missing_ok=True)


def test_gh_pr_view_omits_pr_arg_when_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _make_repo_with_main(tmp_path)
    captured: dict[str, list[str]] = {}

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout='{"number": 7}', stderr=""
        )

    monkeypatch.setattr("daydream.git_ops.subprocess.run", fake_run)

    result = git_ops.gh_pr_view(repo)
    assert result == {"number": 7}
    cmd = captured["cmd"]
    assert cmd[:3] == ["gh", "pr", "view"]
    # No PR number anywhere in the argv.
    assert all(not part.isdigit() for part in cmd)


# --- daydream_commits ---------------------------------------------------------


def test_daydream_commits_returns_tagged_commits(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    _git(repo, "checkout", "-b", "feat/x")
    (repo / "a.py").write_text("a\n")
    _git(repo, "add", "a.py")
    _git(repo, "commit", "-m", "fix: something\n\nDaydream-Run: test-123\nDaydream-Version: 0.14.0")
    result = git_ops.daydream_commits(repo, "main")
    assert result is not None
    assert "fix: something" in result


def test_daydream_commits_excludes_untagged(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    _git(repo, "checkout", "-b", "feat/x")
    (repo / "b.py").write_text("b\n")
    _git(repo, "add", "b.py")
    _commit(repo, "chore: unrelated change")
    result = git_ops.daydream_commits(repo, "main")
    assert result is None


def test_daydream_commits_none_when_no_commits(tmp_path: Path) -> None:
    repo = _make_repo_with_main(tmp_path)
    _git(repo, "checkout", "-b", "feat/x")
    result = git_ops.daydream_commits(repo, "main")
    assert result is None


def test_gh_pr_view_includes_pr_arg_when_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _make_repo_with_main(tmp_path)
    captured: dict[str, list[str]] = {}

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(
            args=cmd, returncode=0, stdout='{"number": 42}', stderr=""
        )

    monkeypatch.setattr("daydream.git_ops.subprocess.run", fake_run)

    result = git_ops.gh_pr_view(repo, 42)
    assert result == {"number": 42}
    cmd = captured["cmd"]
    assert cmd[:4] == ["gh", "pr", "view", "42"]


# --- clone -------------------------------------------------------------------


def _make_bare_remote(tmp_path: Path) -> Path:
    """Create a bare remote repo with one committed file."""
    repo = _make_repo_with_main(tmp_path / "src")
    bare = tmp_path / "bare.git"
    subprocess.run(["git", "clone", "--bare", str(repo), str(bare)], check=True, capture_output=True)  # noqa: S603, S607 - arguments are not user-controlled
    return bare


def test_clone_creates_working_tree(tmp_path: Path) -> None:
    """clone() creates a functional git working tree from a bare remote."""
    bare = _make_bare_remote(tmp_path)
    target = tmp_path / "cloned"
    git_ops.clone(str(bare), target)
    assert (target / ".git").is_dir()
    assert (target / "base.txt").read_text() == "base\n"


def test_clone_raises_on_invalid_remote(tmp_path: Path) -> None:
    """clone() raises GitError when the remote URL is invalid."""
    target = tmp_path / "nope"
    with pytest.raises(git_ops.GitError, match="git clone .* failed"):
        git_ops.clone("file:///nonexistent/repo.git", target)


def test_clone_blobless_passes_filter_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """clone(blobless=True) includes --filter=blob:none in the git invocation."""
    captured: dict[str, list[str]] = {}

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("daydream.git_ops.subprocess.run", fake_run)

    git_ops.clone("https://example.com/repo.git", tmp_path / "out", blobless=True)
    assert "--filter=blob:none" in captured["cmd"]


def test_clone_default_no_filter_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """clone() without blobless does not pass --filter."""
    captured: dict[str, list[str]] = {}

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr("daydream.git_ops.subprocess.run", fake_run)

    git_ops.clone("https://example.com/repo.git", tmp_path / "out")
    assert "--filter=blob:none" not in captured["cmd"]


def test_gh_api_raises_rate_limit_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _Proc:
        returncode = 1
        stdout = ""
        stderr = "gh: API rate limit exceeded for user (HTTP 403)"

    monkeypatch.setattr(git_ops, "_run_gh", lambda *a, **k: _Proc())
    with pytest.raises(git_ops.RateLimitError):
        git_ops.gh_api(tmp_path, "repos/o/r/pulls/1")


def test_gh_api_non_ratelimit_raises_plain_git_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _Proc:
        returncode = 1
        stdout = ""
        stderr = "gh: Not Found (HTTP 404)"

    monkeypatch.setattr(git_ops, "_run_gh", lambda *a, **k: _Proc())
    with pytest.raises(git_ops.GitError):
        git_ops.gh_api(tmp_path, "repos/o/r/pulls/1")
    # RateLimitError subclasses GitError; assert the 404 is NOT classified as one:
    with pytest.raises(git_ops.GitError) as exc:
        git_ops.gh_api(tmp_path, "repos/o/r/pulls/1")
    assert not isinstance(exc.value, git_ops.RateLimitError)


def test_gh_api_jq_parses_ndjson_into_list(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """jq mode passes --jq and parses one JSON value per stdout line."""
    captured: dict[str, list[str]] = {}

    class _Proc:
        returncode = 0
        stdout = '{"id": 1}\n{"id": 2}\n'
        stderr = ""

    def fake_run_gh(repo: Path, args: list[str], **kwargs: Any) -> _Proc:
        captured["args"] = args
        return _Proc()

    monkeypatch.setattr(git_ops, "_run_gh", fake_run_gh)
    result = git_ops.gh_api(tmp_path, "/app/installations", paginate=True, jq=".[]")
    assert result == [{"id": 1}, {"id": 2}]
    assert captured["args"] == ["api", "--paginate", "--jq", ".[]", "/app/installations"]


def test_gh_api_headers_pass_dash_h_args(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Explicit headers reach gh as -H args, in both the plain and --input branches."""
    captured: list[list[str]] = []

    class _Proc:
        returncode = 0
        stdout = "{}"
        stderr = ""

    def fake_run_gh(repo: Path, args: list[str], **kwargs: Any) -> _Proc:
        captured.append(args)
        return _Proc()

    monkeypatch.setattr(git_ops, "_run_gh", fake_run_gh)
    headers = {"Authorization": "Bearer jwt-abc"}
    git_ops.gh_api(tmp_path, "/app/installations", headers=headers)
    git_ops.gh_api(tmp_path, "/app/installations/1/access_tokens", method="POST",
                   input_data={"repositories": ["r"]}, headers=headers)

    assert captured[0][:3] == ["api", "-H", "Authorization: Bearer jwt-abc"]
    assert captured[1][:3] == ["api", "-H", "Authorization: Bearer jwt-abc"]
    assert "--input" in captured[1]


def test_gh_api_error_message_redacts_authorization_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A subprocess failure must not leak the Bearer token into the GitError.

    Real-path: ``gh_api`` builds ``-H "Authorization: Bearer <jwt>"`` and calls the
    real ``_run_gh``; only ``subprocess.run`` is faked (the external boundary).
    """
    repo = _make_repo_with_main(tmp_path)

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise OSError("gh executable failed to spawn")

    monkeypatch.setattr("daydream.git_ops.subprocess.run", fake_run)

    with pytest.raises(GitError) as excinfo:
        git_ops.gh_api(repo, "/app/installations", headers={"Authorization": "Bearer jwt-super-secret-xyz"})

    msg = str(excinfo.value)
    assert "jwt-super-secret-xyz" not in msg
    assert "Authorization: ***" in msg


def test_gh_api_timeout_redacts_authorization_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Timeout must redact the Bearer token in both the retry warning and the error."""
    import logging

    repo = _make_repo_with_main(tmp_path)

    def fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=1)

    monkeypatch.setattr("daydream.git_ops.subprocess.run", fake_run)
    # Force one retry so the warning-log branch fires.
    monkeypatch.setattr(git_ops, "_gh_retries", lambda: 1)

    with caplog.at_level(logging.WARNING, logger="daydream.git_ops"):
        with pytest.raises(git_ops.GitTimeoutError) as excinfo:
            git_ops.gh_api(
                repo,
                "/app/installations",
                headers={"Authorization": "Bearer jwt-super-secret-xyz"},
                idempotent=True,
            )

    msg = str(excinfo.value)
    assert "jwt-super-secret-xyz" not in msg
    assert "Authorization: ***" in msg
    # The retry warning fired and must also be redacted.
    warnings = [r.getMessage() for r in caplog.records]
    assert warnings, "expected a retry warning to be logged"
    assert all("jwt-super-secret-xyz" not in w for w in warnings)
    assert any("Authorization: ***" in w for w in warnings)


def test_gh_api_jq_invalid_line_raises_git_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _Proc:
        returncode = 0
        stdout = '{"id": 1}\nnot-json\n'
        stderr = ""

    monkeypatch.setattr(git_ops, "_run_gh", lambda *a, **k: _Proc())
    with pytest.raises(git_ops.GitError, match="invalid JSON"):
        git_ops.gh_api(tmp_path, "/app/installations", jq=".[]")


# --- gh secret/variable/PR primitives (Task 2) ------------------------------

from tests.harness.fake_gh import FakeGh  # noqa: E402


def test_gh_secret_set_passes_value_on_stdin_never_argv(fake_gh: FakeGh, git_repo: Path) -> None:
    git_ops.gh_secret_set(
        git_repo,
        "DAYDREAM_APP_PRIVATE_KEY",
        "-----BEGIN KEY-----\nx\n",
        repo_slug="o/r",
    )
    call = fake_gh.secret_set_calls()[-1]
    assert call.name == "DAYDREAM_APP_PRIVATE_KEY" and call.repo == "o/r"
    assert "BEGIN KEY" not in " ".join(call.argv)  # PEM never in process args
    assert "BEGIN KEY" in call.stdin


def test_gh_secret_set_org_scope_threads_org_flag(fake_gh: FakeGh, git_repo: Path) -> None:
    git_ops.gh_secret_set(git_repo, "DAYDREAM_APP_ID", "42", org="acme")
    call = fake_gh.secret_set_calls()[-1]
    assert call.name == "DAYDREAM_APP_ID" and call.org == "acme" and call.repo is None
    assert "--org" in call.argv and "acme" in call.argv
    assert "42" in call.stdin


def test_gh_secret_set_requires_exactly_one_scope(fake_gh: FakeGh, git_repo: Path) -> None:
    with pytest.raises(GitError):
        git_ops.gh_secret_set(git_repo, "X", "v")
    with pytest.raises(GitError):
        git_ops.gh_secret_set(git_repo, "X", "v", org="acme", repo_slug="o/r")


def test_gh_variable_set_uses_body_flag(fake_gh: FakeGh, git_repo: Path) -> None:
    git_ops.gh_variable_set(git_repo, "DAYDREAM_BOT_HANDLE", "acme-bot", repo_slug="o/r")
    call = fake_gh.variable_set_calls()[-1]
    assert call.name == "DAYDREAM_BOT_HANDLE" and call.repo == "o/r"
    assert "--body" in call.argv and "acme-bot" in call.argv


def test_gh_secret_list_returns_names(fake_gh: FakeGh, git_repo: Path) -> None:
    fake_gh.serve_secret_list(["DAYDREAM_APP_ID", "ANTHROPIC_API_KEY"])
    assert git_ops.gh_secret_list(git_repo, repo_slug="o/r") == [
        "DAYDREAM_APP_ID",
        "ANTHROPIC_API_KEY",
    ]


def test_gh_variable_list_returns_names(fake_gh: FakeGh, git_repo: Path) -> None:
    fake_gh.serve_variable_list(["DAYDREAM_BOT_HANDLE"])
    assert git_ops.gh_variable_list(git_repo, org="acme") == ["DAYDREAM_BOT_HANDLE"]


def test_gh_pr_create_returns_url(fake_gh: FakeGh, git_repo: Path) -> None:
    fake_gh.set_response("pr-create", value="https://github.com/o/r/pull/9")
    url = git_ops.gh_pr_create(git_repo, head="b", base="main", title="t", body="b")
    assert url == "https://github.com/o/r/pull/9"


def test_gh_pr_create_failure_raises_git_error(fake_gh: FakeGh, git_repo: Path) -> None:
    # No "pr-create" response configured → the shim exits non-zero.
    with pytest.raises(GitError):
        git_ops.gh_pr_create(git_repo, head="b", base="main", title="t", body="b")
