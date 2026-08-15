"""Tests for :mod:`daydream.workspace`.

These tests build real git repositories with a real bare-origin remote and
exercise :func:`daydream.workspace.open_workspace` end-to-end.  No subprocess
mocking — every code path runs against actual git.
"""

from __future__ import annotations

import os
import subprocess
import time
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from daydream import git_ops
from daydream.git_ops import BranchNotFoundError, GitError
from daydream.workspace import (
    WorkContext,
    WorkspaceCopyPathError,
    copy_files_into_ephemeral,
    open_audit_workspace,
    open_workspace,
    prune_stale_audit_worktrees,
)
from tests.harness.git_helpers import bare_remote as _bare_remote
from tests.harness.git_helpers import commit as _commit
from tests.harness.git_helpers import configure_identity as _configure_identity
from tests.harness.git_helpers import git as _git
from tests.harness.git_helpers import init_repo as _init_repo

# --- Helpers (workspace-specific: bare-origin push plumbing) ----------------


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

    async with open_workspace(repo, branch=None, base=None, force_ephemeral=False, skip_tests=False) as ctx:
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
    async with open_workspace(repo, branch=None, base=None, force_ephemeral=True, skip_tests=False) as ctx:
        assert ctx.is_ephemeral is True
        assert ctx.repo != repo
        assert ctx.repo.is_dir()
        assert git_ops.is_inside_worktree(ctx.repo) is True
        assert ctx.head_sha == expected_head
        assert ctx.head_branch is None  # detached
        assert ctx.is_in_place is False
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


# --- 5b. --base accepts any commit-ish --------------------------------------


async def test_base_accepts_raw_sha(tmp_path: Path) -> None:
    repo, _ = _make_repo_with_origin(tmp_path)
    base_sha = git_ops.head_sha(repo)  # pin base to current commit
    # Advance HEAD so base != HEAD and a merge-base exists.
    (repo / "next.txt").write_text("next\n")
    _git(repo, "add", "next.txt")
    _git(repo, "commit", "-m", "advance head")

    async with open_workspace(repo, branch=None, base=base_sha, force_ephemeral=False, skip_tests=False) as ctx:
        assert isinstance(ctx, WorkContext)
        assert ctx.base_branch == base_sha
        assert ctx.base_sha == base_sha  # merge-base of HEAD and its parent


async def test_base_unknown_ref_raises_reworded(tmp_path: Path) -> None:
    repo, _ = _make_repo_with_origin(tmp_path)
    with pytest.raises(BranchNotFoundError, match="base ref 'deadbeef' not found"):
        async with open_workspace(repo, branch=None, base="deadbeef", force_ephemeral=False, skip_tests=False):
            pass


# --- 6. copy_files_into_ephemeral default list (gitignored only) ------------


def test_copy_default_only_copies_gitignored(tmp_path: Path) -> None:
    repo, _ = _make_repo_with_origin(tmp_path)
    (repo / ".gitignore").write_text(".env\n.env.local\n")
    _git(repo, "add", ".gitignore")
    _commit(repo, "ignore env")

    (repo / ".env").write_text("SECRET=1\n")
    # Tracked, non-default-glob name: must not be copied.
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

    (repo / "pyproject.toml").write_text('[tool.daydream.workspace]\ncopy = ["custom.cfg", "local/secrets.toml"]\n')
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


def test_copy_pyproject_non_table_tool_falls_back_to_defaults(tmp_path: Path) -> None:
    """A valid TOML with a non-table ``tool`` value must not raise; defaults apply."""
    repo, _ = _make_repo_with_origin(tmp_path)
    (repo / ".gitignore").write_text(".env\n")
    _git(repo, "add", ".gitignore")
    _commit(repo, "ignore env")

    # `tool` is a scalar string here — a chained .get() would raise AttributeError.
    (repo / "pyproject.toml").write_text('tool = "not-a-table"\n')
    (repo / ".env").write_text("SECRET=1\n")

    dest = tmp_path / "ephemeral"
    dest.mkdir()

    copied = copy_files_into_ephemeral(repo, dest, extra=None, skip=False)
    rel = {str(p) for p in copied}
    assert ".env" in rel
    assert (dest / ".env").read_text() == "SECRET=1\n"


# --- 7b. fail-closed copy entry validation --------------------------------


@pytest.mark.parametrize("source_kind", ["config", "extra"])
@pytest.mark.parametrize("escape_kind", ["parent", "absolute"])
def test_copy_rejects_absolute_and_parent_entries_before_copy(
    tmp_path: Path, source_kind: str, escape_kind: str
) -> None:
    repo, _ = _make_repo_with_origin(tmp_path)
    (repo / ".gitignore").write_text(".env\n")
    _git(repo, "add", ".gitignore")
    _commit(repo, "ignore env")
    (repo / ".env").write_text("E=1\n")
    outside = tmp_path / "outside.txt"
    outside.write_text("KEEP\n")
    entry = "../outside.txt" if escape_kind == "parent" else str(outside)

    if source_kind == "config":
        (repo / "pyproject.toml").write_text(f'[tool.daydream.workspace]\ncopy = [".env", "{entry}"]\n')
        extra = None
    else:
        (repo / "pyproject.toml").write_text('[tool.daydream.workspace]\ncopy = [".env"]\n')
        extra = [Path(entry)]

    dest = tmp_path / "ephemeral"
    dest.mkdir()

    with pytest.raises(WorkspaceCopyPathError, match="must be relative and must not contain"):
        copy_files_into_ephemeral(repo, dest, extra=extra, skip=False)

    # Fail-closed: the valid earlier ".env" entry was NOT copied.
    assert not (dest / ".env").exists()
    # The external file was never read or modified.
    assert outside.read_text() == "KEEP\n"


@pytest.mark.parametrize("root_kind", ["source", "destination"])
def test_copy_rejects_resolved_symlink_escape(tmp_path: Path, root_kind: str) -> None:
    repo, _ = _make_repo_with_origin(tmp_path)
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    secret = outside_dir / "secret.txt"
    secret.write_text("KEEP\n")
    (repo / "pyproject.toml").write_text('[tool.daydream.workspace]\ncopy = ["sub/leak.txt"]\n')
    (repo / "sub").mkdir()

    dest = tmp_path / "ephemeral"
    dest.mkdir()
    (dest / "sub").mkdir()

    if root_kind == "source":
        # The SOURCE-side "sub" is a symlink escaping the checkout. Do not
        # write the real nested file first -- the directory is replaced by the
        # symlink below.
        (repo / "sub").rmdir()
        (repo / "sub").symlink_to(outside_dir, target_is_directory=True)
        root_label = "source"
    else:
        # The source has a real nested file (passes source containment); the
        # DESTINATION-side "sub" is a symlink escaping the worktree.
        (repo / "sub" / "leak.txt").write_text("real\n")
        (dest / "sub").rmdir()
        (dest / "sub").symlink_to(outside_dir, target_is_directory=True)
        root_label = "destination"

    with pytest.raises(
        WorkspaceCopyPathError,
        match=f"resolves outside the {root_label} worktree",
    ):
        copy_files_into_ephemeral(repo, dest, extra=None, skip=False)

    # Nothing was written into the escaped directory.
    assert not (outside_dir / "leak.txt").exists()
    assert secret.read_text() == "KEEP\n"


def test_copy_allows_source_symlink_resolving_inside_source(
    tmp_path: Path,
) -> None:
    repo, _ = _make_repo_with_origin(tmp_path)
    (repo / "actual.cfg").write_text("inside\n")
    # A RELATIVE symlink whose target stays inside the source root.
    (repo / "inside-link.cfg").symlink_to("actual.cfg")

    dest = tmp_path / "ephemeral"
    dest.mkdir()

    copied = copy_files_into_ephemeral(repo, dest, extra=[Path("inside-link.cfg")], skip=False)

    assert copied == [Path("inside-link.cfg")]
    assert (dest / "inside-link.cfg").read_text() == "inside\n"
    assert not (dest / "inside-link.cfg").is_symlink()


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

    copied = copy_files_into_ephemeral(repo, dest, extra=[Path("anything.cfg")], skip=True)
    assert copied == []
    assert not (dest / ".env").exists()


# --- 10. Cleanup runs even on exception -------------------------------------


async def test_cleanup_runs_on_exception(tmp_path: Path) -> None:
    repo, _ = _make_repo_with_origin(tmp_path)
    captured_path: Path | None = None

    with pytest.raises(RuntimeError, match="boom"):
        async with open_workspace(repo, branch=None, base=None, force_ephemeral=True, skip_tests=False) as ctx:
            captured_path = ctx.repo
            assert captured_path.exists()
            raise RuntimeError("boom")

    assert captured_path is not None
    assert not captured_path.exists()


async def test_open_workspace_rejects_escape_without_persistent_copy(tmp_path: Path) -> None:
    repo, _ = _make_repo_with_origin(tmp_path)
    # Seed the source file the traversal entry reads. `../retained.cfg` resolves
    # from the source root to tmp_path/retained.cfg; without it the copy loop
    # would silently skip (not a file) and the fail-closed assertion below could
    # never detect a regressed guard.
    (tmp_path / "retained.cfg").write_text("secret\n")
    with pytest.raises(WorkspaceCopyPathError, match="must be relative and must not contain"):
        async with open_workspace(
            repo,
            branch=None,
            base=None,
            force_ephemeral=True,
            extra_copy=[Path("../retained.cfg")],
            skip_tests=False,
        ):
            pass  # never reached — the copy boundary rejects before yielding
    # Fail-closed: a regressed guard would copy the seeded source into
    # dest/../retained.cfg == repo/.daydream/worktrees/retained.cfg (the escape
    # destination). Assert nothing was written there.
    assert not (repo / ".daydream" / "worktrees" / "retained.cfg").exists()
    # Cleanup ran: the ephemeral worktree was removed -> worktrees dir is empty.
    worktrees = repo / ".daydream" / "worktrees"
    assert not any(worktrees.iterdir())


# --- 13. Stale-local warning fires ------------------------------------------


async def test_stale_local_warning_fires(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Warn on a stale local branch and review the fresher remote snapshot."""
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

    # Pin a wide recording console so the warning text is captured intact,
    # independent of terminal width (the warning renders through the lazily
    # imported ``daydream.agent.console``).
    rec = Console(file=StringIO(), record=True, force_terminal=True, width=200, height=25)
    monkeypatch.setattr("daydream.agent.console", rec)

    async with open_workspace(repo, branch="topic", base="main", force_ephemeral=False, skip_tests=False) as ctx:
        assert ctx.is_ephemeral is True

    out = rec.export_text()
    assert "topic is checked out in cwd" in out
    assert "2 commits behind origin/topic" in out
    assert "reviewing origin/topic" in out


# --- 14. open_audit_workspace (detached audit snapshot) ---------------------


@pytest.mark.anyio
async def test_audit_workspace_preserves_target_state_when_audit_commits(tmp_path: Path) -> None:
    repo, _ = _make_repo_with_origin(tmp_path)
    _configure_identity(repo)
    # tracked staged + unstaged state the audit must reproduce
    (repo / "staged.txt").write_text("v2")
    _git(repo, "add", "staged.txt")
    (repo / "unstaged.txt").write_text("v1")
    _git(repo, "add", "unstaged.txt")
    _commit(repo, "add unstaged tracked")
    (repo / "unstaged.txt").write_text("v2")
    before_head = git_ops.head_sha(repo)
    before_status = git_ops.status_porcelain(repo)
    before_refs = _git(repo, "show-ref")

    captured: Path | None = None
    async with open_audit_workspace(repo, run_id="audit-test") as audit:
        captured = audit
        assert git_ops.is_inside_worktree(audit) is True
        # the audit worktree reproduces the staged + unstaged tracked state
        assert (audit / "staged.txt").read_text() == "v2"
        assert (audit / "unstaged.txt").read_text() == "v2"
        # a model commit inside the audit worktree...
        (audit / "model-note.txt").write_text("model wrote this")
        _git(audit, "add", "-A")
        _git(audit, "commit", "-m", "model commit")

    # ...must leave the target untouched, and the audit worktree cleaned up.
    assert captured is not None and not captured.exists()
    assert git_ops.head_sha(repo) == before_head
    assert git_ops.status_porcelain(repo) == before_status
    assert _git(repo, "show-ref") == before_refs
    assert _git(repo, "stash", "list") == ""


@pytest.mark.anyio
async def test_audit_workspace_reproduces_clean_head_when_no_tracked_changes(tmp_path: Path) -> None:
    repo, _ = _make_repo_with_origin(tmp_path)
    _configure_identity(repo)
    async with open_audit_workspace(repo, run_id="audit-clean") as audit:
        assert git_ops.head_sha(audit) == git_ops.head_sha(repo)


@pytest.mark.anyio
async def test_audit_workspace_cleanup_runs_on_exception(tmp_path: Path) -> None:
    repo, _ = _make_repo_with_origin(tmp_path)
    _configure_identity(repo)
    captured: Path | None = None
    with pytest.raises(RuntimeError, match="boom"):
        async with open_audit_workspace(repo, run_id="audit-exc") as audit:
            captured = audit
            raise RuntimeError("boom")
    assert captured is not None and not captured.exists()


@pytest.mark.anyio
async def test_prune_stale_audit_worktrees_reclaims_crashed_run_leftover(
    tmp_path: Path,
) -> None:
    """A hard-killed improve run's locked audit worktree is reclaimed.

    ``open_audit_workspace`` creates the worktree locked and only the owning
    run's exit removes it, so a hard-killed run leaves a locked audit worktree
    behind with no ``*-reanchor`` name for ``prune_stale_reanchor_worktrees``
    to match — this prune is its reclamation path.
    """
    repo, _ = _make_repo_with_origin(tmp_path)
    _configure_identity(repo)
    stale_dir = repo / ".daydream" / "audit" / "run-crashed"
    _git(
        repo,
        "worktree",
        "add",
        "--detach",
        "--lock",
        "--reason",
        "run-crashed",
        str(stale_dir),
        "HEAD",
    )
    (stale_dir / "marker.txt").write_text("leftover", encoding="utf-8")
    # Age the lock beyond the staleness window so it reads as a crashed session.
    common = Path(_git(repo, "rev-parse", "--git-common-dir"))
    if not common.is_absolute():
        common = repo / common
    old = time.time() - 48 * 3600
    os.utime(common / "worktrees" / stale_dir.name / "locked", (old, old))

    removed = prune_stale_audit_worktrees(repo)

    assert removed == 1
    assert not stale_dir.exists()
    assert "run-crashed" not in _git(repo, "worktree", "list")


@pytest.mark.anyio
async def test_prune_stale_audit_worktrees_skips_live_locked_worktree(
    tmp_path: Path,
) -> None:
    """A live audit worktree (fresh lock) is never destroyed by the prune."""
    repo, _ = _make_repo_with_origin(tmp_path)
    _configure_identity(repo)
    live_dir = repo / ".daydream" / "audit" / "run-live"
    _git(
        repo,
        "worktree",
        "add",
        "--detach",
        "--lock",
        "--reason",
        "run-live",
        str(live_dir),
        "HEAD",
    )

    removed = prune_stale_audit_worktrees(repo)

    assert removed == 0
    assert live_dir.is_dir()  # a concurrent run mid-write is never destroyed


@pytest.mark.anyio
async def test_audit_workspace_yields_source_when_head_unborn(tmp_path: Path) -> None:
    """A repo with no initial commit runs without a snapshot worktree.

    ``git stash create`` (and ``git worktree add``) fail on an unborn HEAD —
    there is no commit to snapshot or materialize — so the source itself is
    yielded and the improve run proceeds without worktree isolation.
    """
    repo = tmp_path / "unborn"
    _init_repo(repo)
    (repo / "staged.txt").write_text("v1")
    _git(repo, "add", "staged.txt")

    async with open_audit_workspace(repo, run_id="audit-unborn") as audit:
        assert audit == repo
        # No worktree was created, and the source's staged state was untouched.
        assert not (repo / ".daydream" / "audit").exists()
        proc = subprocess.run(  # noqa: S603
            ["git", "rev-parse", "--verify", "HEAD"],  # noqa: S607
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode != 0  # still unborn while the workspace is open
    assert (repo / "staged.txt").read_text() == "v1"


@pytest.mark.anyio
async def test_audit_workspace_does_not_warn_when_worktree_add_fails(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A failed ``git worktree add`` must not trigger cleanup of a never-created worktree.

    The pre-existing run path makes ``git worktree add`` fail after the parent
    ``mkdir`` succeeded; cleanup must not then attempt to remove a worktree
    that was never created, which would surface a spurious "Failed to remove
    audit worktree" warning over the primary add error.
    """
    repo, _ = _make_repo_with_origin(tmp_path)
    _configure_identity(repo)
    doomed = repo / ".daydream" / "audit" / "audit-fail"
    doomed.mkdir(parents=True, exist_ok=True)
    (doomed / "marker.txt").write_text("occupied", encoding="utf-8")

    with pytest.raises(GitError, match="already exists"):
        async with open_audit_workspace(repo, run_id="audit-fail"):
            pytest.fail("the audit body must not run")

    assert "Failed to remove audit worktree" not in capsys.readouterr().out
