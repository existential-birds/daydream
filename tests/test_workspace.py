"""Tests for :mod:`daydream.workspace`.

These tests build real git repositories with a real bare-origin remote and
exercise :func:`daydream.workspace.open_workspace` end-to-end.  No subprocess
mocking — every code path runs against actual git.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from daydream import git_ops
from daydream.git_ops import BranchNotFoundError
from daydream.workspace import (
    WorkContext,
    copy_files_into_ephemeral,
    open_workspace,
)

# --- Helpers (mirrors tests/test_git_ops.py for isolation) ------------------


def _git(repo: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(  # noqa: S603 - arguments are not user-controlled
        ["git", *args],  # noqa: S607 - git is a trusted command
        cwd=repo,
        capture_output=True,
        text=True,
        check=check,
    )
    return proc.stdout.strip()


def _configure_identity(repo: Path) -> None:
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Tester")


def _commit(repo: Path, message: str) -> str:
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


def _make_repo_with_origin(tmp_path: Path) -> tuple[Path, Path]:
    """Return (repo, bare_remote) — repo has one initial commit pushed to origin."""
    bare = _bare_remote(tmp_path / "remote.git")
    repo = tmp_path / "repo"
    _init_repo(repo)
    (repo / "base.txt").write_text("base\n")
    _git(repo, "add", "base.txt")
    _commit(repo, "initial")
    _git(repo, "remote", "add", "origin", str(bare))
    _git(repo, "push", "-u", "origin", "main")
    _git(repo, "remote", "set-head", "origin", "main")
    return repo, bare


def _push_origin_commit_via_sidecar(tmp_path: Path, bare: Path, branch: str = "main") -> str:
    """Add a commit to *branch* on *bare* via a fresh sidecar clone; return its SHA."""
    token = _secrets_token()
    sidecar = tmp_path / f"sidecar-{branch}-{token}"
    _git(tmp_path, "clone", str(bare), str(sidecar))
    _configure_identity(sidecar)
    # Determine whether the branch already exists on origin.
    ls = subprocess.run(  # noqa: S603
        ["git", "ls-remote", "--heads", "origin", branch],  # noqa: S607
        cwd=sidecar,
        capture_output=True,
        text=True,
        check=True,
    )
    if ls.stdout.strip():
        # Branch exists on origin -- check it out as a tracking branch.
        _git(sidecar, "checkout", "-B", branch, f"origin/{branch}")
    elif branch != "main":
        _git(sidecar, "checkout", "-b", branch)
    new_file = sidecar / f"{branch}-{token}.txt"
    new_file.write_text("payload\n")
    _git(sidecar, "add", new_file.name)
    sha = _commit(sidecar, f"sidecar commit on {branch}")
    _git(sidecar, "push", "origin", branch)
    return sha


def _secrets_token() -> str:
    import secrets

    return secrets.token_hex(3)


# --- 1. In-place mode -------------------------------------------------------


async def test_in_place_no_branch_no_force(tmp_path: Path) -> None:
    repo, bare = _make_repo_with_origin(tmp_path)
    # Push a new commit to origin from a sidecar so we can prove no fetch ran.
    new_sha = _push_origin_commit_via_sidecar(tmp_path, bare)

    async with open_workspace(
        repo, branch=None, base=None, force_ephemeral=False, skip_tests=False
    ) as ctx:
        assert isinstance(ctx, WorkContext)
        assert ctx.repo == repo
        assert ctx.source == repo
        assert ctx.is_ephemeral is False
        assert ctx.is_in_place is True
        assert ctx.head_branch == "main"
        # No fetch should have run -> the new commit on origin is not visible.
        proc = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "--verify", "origin/main"],  # noqa: S607
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
        # origin/main still points at the original push, not new_sha.
        assert proc.stdout.strip() != new_sha

    # Source remains untouched after exit (no cleanup paths to assert).
    assert repo.exists()


# --- 2. Ephemeral with no branch --------------------------------------------


async def test_ephemeral_with_no_branch_uses_head(tmp_path: Path) -> None:
    repo, _ = _make_repo_with_origin(tmp_path)
    expected_head = git_ops.head_sha(repo)

    captured_path: Path | None = None
    async with open_workspace(
        repo, branch=None, base=None, force_ephemeral=True, skip_tests=False
    ) as ctx:
        assert ctx.is_ephemeral is True
        assert ctx.repo != repo
        assert ctx.repo.is_dir()
        assert ctx.head_sha == expected_head
        assert ctx.head_branch is None  # detached
        captured_path = ctx.repo

    assert captured_path is not None
    assert not captured_path.exists()


# --- 3. Ephemeral with branch (local + origin) ------------------------------


async def test_ephemeral_uses_origin_branch_tip(tmp_path: Path) -> None:
    repo, bare = _make_repo_with_origin(tmp_path)
    _git(repo, "checkout", "-b", "feature")
    (repo / "feat.txt").write_text("local\n")
    _git(repo, "add", "feat.txt")
    _commit(repo, "local feat")
    _git(repo, "push", "-u", "origin", "feature")
    # Add an extra commit on origin/feature via sidecar — origin tip diverges
    # from the local feature branch.
    new_sha = _push_origin_commit_via_sidecar(tmp_path, bare, branch="feature")
    _git(repo, "checkout", "main")

    async with open_workspace(
        repo,
        branch="feature",
        base="main",
        force_ephemeral=False,
        skip_tests=False,
    ) as ctx:
        assert ctx.is_ephemeral is True
        # head_sha should be the origin tip (post-fetch), not local feature tip.
        assert ctx.head_sha == new_sha
        assert ctx.base_branch == "main"


# --- 4. Ephemeral with branch (only origin) ---------------------------------


async def test_ephemeral_branch_only_on_origin(tmp_path: Path) -> None:
    repo, bare = _make_repo_with_origin(tmp_path)
    new_sha = _push_origin_commit_via_sidecar(tmp_path, bare, branch="origin-only")
    # Branch does NOT exist locally.

    async with open_workspace(
        repo,
        branch="origin-only",
        base="main",
        force_ephemeral=False,
        skip_tests=False,
    ) as ctx:
        assert ctx.is_ephemeral is True
        assert ctx.head_sha == new_sha


# --- 5. Branch not found anywhere -------------------------------------------


async def test_unknown_branch_raises(tmp_path: Path) -> None:
    repo, _ = _make_repo_with_origin(tmp_path)
    with pytest.raises(BranchNotFoundError):
        async with open_workspace(
            repo,
            branch="nope-not-here",
            base="main",
            force_ephemeral=False,
            skip_tests=False,
        ):
            pass  # pragma: no cover


# --- 6. copy_files_into_ephemeral default list (gitignored only) ------------


def test_copy_default_only_copies_gitignored(tmp_path: Path) -> None:
    repo, _ = _make_repo_with_origin(tmp_path)
    (repo / ".gitignore").write_text(".env\n.env.local\n")
    _git(repo, "add", ".gitignore")
    _commit(repo, "ignore env")

    # Two files: one gitignored, one tracked-but-named .env-style.
    (repo / ".env").write_text("SECRET=1\n")
    # Tracked .env file with a different name is NOT in the default glob, so
    # use the same default name but track it via add+commit before copy.
    (repo / ".env.committed").write_text("PUBLIC=1\n")
    _git(repo, "add", ".env.committed")
    _commit(repo, "tracked env-style")

    dest = tmp_path / "ephemeral"
    dest.mkdir()

    copied = copy_files_into_ephemeral(repo, dest, extra=None, skip=False)
    rel = {str(p) for p in copied}
    assert ".env" in rel
    # .env.committed is tracked => not gitignored => not copied.
    assert ".env.committed" not in rel
    assert (dest / ".env").read_text() == "SECRET=1\n"


def test_copy_default_skips_tracked_env(tmp_path: Path) -> None:
    """A tracked ``.env`` file is not copied (already in the worktree)."""
    repo, _ = _make_repo_with_origin(tmp_path)
    (repo / ".env").write_text("TRACKED=1\n")
    _git(repo, "add", ".env")
    _commit(repo, "track env")

    dest = tmp_path / "ephemeral"
    dest.mkdir()

    copied = copy_files_into_ephemeral(repo, dest, extra=None, skip=False)
    assert copied == []
    assert not (dest / ".env").exists()


# --- 7. pyproject override --------------------------------------------------


def test_copy_pyproject_override(tmp_path: Path) -> None:
    repo, _ = _make_repo_with_origin(tmp_path)
    (repo / ".gitignore").write_text(".env\nlocal/\n")
    _git(repo, "add", ".gitignore")
    _commit(repo, "ignore env+local")

    (repo / "pyproject.toml").write_text(
        '[tool.daydream.workspace]\ncopy = ["custom.cfg", "local/secrets.toml"]\n'
    )
    (repo / "custom.cfg").write_text("k=v\n")
    (repo / "local").mkdir()
    (repo / "local" / "secrets.toml").write_text("token = 'x'\n")
    # And a .env that the override should *not* pull in.
    (repo / ".env").write_text("SHOULD_BE_SKIPPED=1\n")

    dest = tmp_path / "ephemeral"
    dest.mkdir()

    copied = copy_files_into_ephemeral(repo, dest, extra=None, skip=False)
    rel = {str(p) for p in copied}
    assert "custom.cfg" in rel
    assert str(Path("local/secrets.toml")) in rel
    # Override replaces the default list entirely.
    assert ".env" not in rel
    assert (dest / "custom.cfg").read_text() == "k=v\n"
    assert (dest / "local" / "secrets.toml").read_text() == "token = 'x'\n"


# --- 8. extra paths combine -------------------------------------------------


def test_copy_extra_paths_additive(tmp_path: Path) -> None:
    repo, _ = _make_repo_with_origin(tmp_path)
    (repo / ".gitignore").write_text(".env\nworkspace.json\n")
    _git(repo, "add", ".gitignore")
    _commit(repo, "ignore env+workspace")

    (repo / ".env").write_text("E=1\n")
    (repo / "workspace.json").write_text("{}\n")

    dest = tmp_path / "ephemeral"
    dest.mkdir()

    copied = copy_files_into_ephemeral(
        repo,
        dest,
        extra=[Path("workspace.json")],
        skip=False,
    )
    rel = {str(p) for p in copied}
    assert ".env" in rel
    assert "workspace.json" in rel


# --- 9. skip flag -----------------------------------------------------------


def test_copy_skip_returns_empty(tmp_path: Path) -> None:
    repo, _ = _make_repo_with_origin(tmp_path)
    (repo / ".gitignore").write_text(".env\n")
    _git(repo, "add", ".gitignore")
    _commit(repo, "ignore env")
    (repo / ".env").write_text("E=1\n")

    dest = tmp_path / "ephemeral"
    dest.mkdir()

    copied = copy_files_into_ephemeral(
        repo, dest, extra=[Path("anything.cfg")], skip=True
    )
    assert copied == []
    assert not (dest / ".env").exists()


# --- 10. Cleanup runs even on exception -------------------------------------


async def test_cleanup_runs_on_exception(tmp_path: Path) -> None:
    repo, _ = _make_repo_with_origin(tmp_path)
    captured_path: Path | None = None

    with pytest.raises(RuntimeError, match="boom"):
        async with open_workspace(
            repo, branch=None, base=None, force_ephemeral=True, skip_tests=False
        ) as ctx:
            captured_path = ctx.repo
            assert captured_path.exists()
            raise RuntimeError("boom")

    assert captured_path is not None
    assert not captured_path.exists()


# --- 11. is_inside_worktree on the ephemeral path ---------------------------


async def test_ephemeral_is_a_real_worktree(tmp_path: Path) -> None:
    repo, _ = _make_repo_with_origin(tmp_path)
    async with open_workspace(
        repo, branch=None, base=None, force_ephemeral=True, skip_tests=False
    ) as ctx:
        assert git_ops.is_inside_worktree(ctx.repo) is True


# --- 12. is_in_place property ----------------------------------------------


def test_is_in_place_property_inverse_of_ephemeral() -> None:
    in_place = WorkContext(
        repo=Path("/tmp/x"),
        source=Path("/tmp/x"),
        base_branch="main",
        base_sha="0" * 40,
        head_branch="main",
        head_sha="0" * 40,
        is_ephemeral=False,
        run_id="20260101000000-deadbeef",
    )
    assert in_place.is_in_place is True

    ephemeral = WorkContext(
        repo=Path("/tmp/y"),
        source=Path("/tmp/x"),
        base_branch="main",
        base_sha="0" * 40,
        head_branch=None,
        head_sha="0" * 40,
        is_ephemeral=True,
        run_id="20260101000000-cafef00d",
    )
    assert ephemeral.is_in_place is False


# --- 13. Stale-local warning fires ------------------------------------------


async def test_stale_local_warning_fires(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo, bare = _make_repo_with_origin(tmp_path)
    # Create + push the feature branch, then add commits on origin so the
    # local copy is behind.
    _git(repo, "checkout", "-b", "topic")
    (repo / "topic.txt").write_text("local\n")
    _git(repo, "add", "topic.txt")
    _commit(repo, "local topic")
    _git(repo, "push", "-u", "origin", "topic")
    # Push two more commits on origin/topic via sidecar.
    _push_origin_commit_via_sidecar(tmp_path, bare, branch="topic")
    _push_origin_commit_via_sidecar(tmp_path, bare, branch="topic")
    # 'topic' is currently checked out in repo and now lags origin/topic.

    async with open_workspace(
        repo, branch="topic", base="main", force_ephemeral=False, skip_tests=False
    ) as ctx:
        assert ctx.is_ephemeral is True

    captured = capsys.readouterr()
    # Rich wraps the panel — collapse whitespace and panel borders before
    # asserting the message is present.
    raw = captured.out + captured.err
    flat = " ".join(raw.replace("│", " ").split())
    assert "topic is checked out in cwd" in flat
    assert "2 commits behind origin/topic" in flat
    assert "reviewing origin/topic" in flat
