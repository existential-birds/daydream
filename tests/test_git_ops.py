"""Tests for :mod:`daydream.git_ops`.

These tests use real ``git`` (and optionally ``gh``) against ``tmp_path``
fixtures.  No subprocess mocking — every code path is exercised against an
actual repository.  ``gh``-dependent tests are skipped when the binary is
unavailable.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from daydream import git_ops
from daydream.git_ops import (
    BranchNotFoundError,
    GitError,
    NotAWorktreeError,
    WrongBranchError,
)

# --- Helpers ----------------------------------------------------------------


def _git(repo: Path, *args: str, check: bool = True, env: dict[str, str] | None = None) -> str:
    """Run a git command in *repo* and return stripped stdout (test helper)."""
    proc = subprocess.run(  # noqa: S603 - arguments are not user-controlled
        ["git", *args],  # noqa: S607 - git is a trusted command
        cwd=repo,
        capture_output=True,
        text=True,
        check=check,
        env=env,
    )
    return proc.stdout.strip()


def _configure_identity(repo: Path) -> None:
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Tester")


def _commit(repo: Path, message: str) -> str:
    """Create a commit and return its SHA."""
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _init_repo(repo: Path) -> None:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-b", "main")
    _configure_identity(repo)


def _bare_remote(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git(path, "init", "--bare", "-b", "main")
    return path


def _make_repo_with_main(tmp_path: Path, name: str = "repo") -> Path:
    repo = tmp_path / name
    _init_repo(repo)
    (repo / "base.txt").write_text("base\n")
    _git(repo, "add", "base.txt")
    _commit(repo, "initial")
    return repo


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


# --- Error type identity ----------------------------------------------------


def test_error_hierarchy_is_consistent() -> None:
    assert issubclass(NotAWorktreeError, GitError)
    assert issubclass(BranchNotFoundError, GitError)
    assert issubclass(WrongBranchError, GitError)


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
