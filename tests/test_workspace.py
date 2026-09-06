"""Tests for :mod:`daydream.workspace`.

These tests build real git repositories with a real bare-origin remote and
exercise :func:`daydream.workspace.open_workspace` end-to-end.  No subprocess
mocking — every code path runs against actual git.
"""

from __future__ import annotations

import json
import os
import subprocess
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from daydream import artifact_visibility, git_ops
from daydream.artifact_visibility import (
    ArtifactVisibilityError,
    private_root_locations,
    resolve_private_workspace_owner,
)
from daydream.git_ops import BranchNotFoundError, GitError
from daydream.workspace import (
    WorkContext,
    WorkspaceCopyPathError,
    _resolve_base,
    copy_files_into_ephemeral,
    open_audit_workspace,
    open_workspace,
)
from tests.harness.git_helpers import bare_remote as _bare_remote
from tests.harness.git_helpers import commit as _commit
from tests.harness.git_helpers import configure_identity as _configure_identity
from tests.harness.git_helpers import git as _git
from tests.harness.git_helpers import init_repo as _init_repo

# --- Helpers (workspace-specific: bare-origin push plumbing) ----------------


def test_resolve_base_falls_back_when_pr_lookup_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("daydream.workspace.shutil.which", lambda _name: "/bin/gh")
    monkeypatch.setattr(
        git_ops,
        "gh_pr_list_for_branch",
        lambda *_args: (_ for _ in ()).throw(GitError("gh auth failed")),
    )
    monkeypatch.setattr(git_ops, "default_branch", lambda _repo: "trunk")

    assert _resolve_base(tmp_path, "feature", None) == "trunk"


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


async def test_external_worktrees_use_supplied_private_workspace_owner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo, _ = _make_repo_with_origin(tmp_path)
    locations = private_root_locations(base=tmp_path / "private")
    owner = resolve_private_workspace_owner(repo, locations=locations)
    artifact_owner = owner.artifact_state_root / "owner.json"
    operational_owner = owner.operational_state_root / "owner.json"
    assert artifact_owner.read_bytes() == operational_owner.read_bytes()
    payload = json.loads(artifact_owner.read_text(encoding="utf-8"))
    assert payload["source"] == str(repo.resolve())
    assert payload["git_common_dir"] == str(git_ops.git_common_dir(repo))

    def unexpected_default_lookup() -> Path:
        raise AssertionError("a supplied owner must not consult the default provider")

    monkeypatch.setattr(artifact_visibility, "_default_private_base", unexpected_default_lookup)
    captured: list[Path] = []
    async with open_workspace(
        repo,
        branch=None,
        base="main",
        force_ephemeral=True,
        extra_copy=[],
        skip_tests=True,
        private_owner=owner,
    ) as first:
        captured.append(first.repo)
        async with open_workspace(
            repo,
            branch=None,
            base="main",
            force_ephemeral=True,
            extra_copy=[],
            skip_tests=True,
            private_owner=owner,
        ) as second:
            captured.append(second.repo)
            for work in (first, second):
                assert work.source == repo.resolve()
                assert work.repo.parent == owner.operational_state_root / "operational"
                assert work.repo.is_relative_to(locations.operational_workspaces)
                assert not work.repo.is_relative_to(repo)
                assert not work.repo.is_relative_to(locations.artifact_runtime)
                assert not locations.artifact_runtime.is_relative_to(work.repo)
                assert work.repo not in repo.rglob("*")
                assert git_ops.git_common_dir(work.repo) == owner.git_common_dir
            assert first.repo != second.repo
            assert all(path.exists() for path in captured)

    assert all(not path.exists() for path in captured)
    assert not (repo / ".daydream" / "worktrees").exists()


async def test_open_workspace_without_owner_resolves_default_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo, _ = _make_repo_with_origin(tmp_path)
    private_base = tmp_path / "standalone-private"
    lookups = 0

    def default_private_base() -> Path:
        nonlocal lookups
        lookups += 1
        return private_base

    monkeypatch.setattr(artifact_visibility, "_default_private_base", default_private_base)
    async with open_workspace(
        repo,
        branch=None,
        base="main",
        force_ephemeral=True,
        extra_copy=[],
        skip_tests=True,
    ) as work:
        assert work.repo.is_relative_to(private_base / "workspaces")
        assert not work.repo.is_relative_to(private_base / "runtime")

    assert lookups == 1


async def test_open_workspace_rejects_wrong_supplied_owner_before_git_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    first_repo, _ = _make_repo_with_origin(tmp_path / "first")
    second_repo, _ = _make_repo_with_origin(tmp_path / "second")
    locations = private_root_locations(base=tmp_path / "private")
    wrong_owner = resolve_private_workspace_owner(first_repo, locations=locations)
    before = _git(second_repo, "worktree", "list", "--porcelain")

    def unexpected_default_lookup() -> Path:
        raise AssertionError("a supplied owner must not consult the default provider")

    monkeypatch.setattr(artifact_visibility, "_default_private_base", unexpected_default_lookup)
    with pytest.raises(ArtifactVisibilityError, match="owner identity mismatch"):
        async with open_workspace(
            second_repo,
            branch=None,
            base="main",
            force_ephemeral=True,
            extra_copy=[],
            skip_tests=True,
            private_owner=wrong_owner,
        ):
            pass

    assert _git(second_repo, "worktree", "list", "--porcelain") == before
    assert not (wrong_owner.operational_state_root / "operational").exists()


async def test_unsafe_operational_root_rejects_before_fetch_mutation(
    tmp_path: Path,
) -> None:
    repo, bare = _make_repo_with_origin(tmp_path)
    owner = resolve_private_workspace_owner(
        repo,
        locations=private_root_locations(base=tmp_path / "private"),
    )
    new_origin_sha = _push_origin_commit_via_sidecar(tmp_path, bare)
    assert _git(repo, "rev-parse", "origin/main") != new_origin_sha
    outside = tmp_path / "outside"
    outside.mkdir()
    (owner.operational_state_root / "operational").symlink_to(
        outside,
        target_is_directory=True,
    )

    with pytest.raises(ArtifactVisibilityError, match="real directory"):
        async with open_workspace(
            repo,
            branch=None,
            base="main",
            force_ephemeral=True,
            skip_tests=True,
            private_owner=owner,
        ):
            pass

    assert _git(repo, "rev-parse", "origin/main") != new_origin_sha
    assert not any("worktree " in line for line in _git(repo, "worktree", "list", "--porcelain").splitlines()[1:])


async def test_open_workspace_migrates_unlocked_legacy_reanchor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo, _ = _make_repo_with_origin(tmp_path)
    locations = private_root_locations(base=tmp_path / "private")
    owner = resolve_private_workspace_owner(repo, locations=locations)
    legacy = repo / ".daydream" / "worktrees" / "run-old-reanchor"
    git_ops.worktree_add(repo, legacy, "main", detach=True)

    monkeypatch.setattr(
        artifact_visibility,
        "_default_private_base",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected default lookup")),
    )
    async with open_workspace(
        repo,
        branch=None,
        base="main",
        force_ephemeral=False,
        skip_tests=True,
        private_owner=owner,
    ):
        migrated = owner.operational_state_root / "operational" / legacy.name
        assert migrated.is_dir()
        assert not legacy.exists()
        assert git_ops.git_common_dir(migrated) == owner.git_common_dir


@pytest.mark.parametrize(
    "namespace_shape",
    ["ancestor-symlink", "terminal-symlink", "ancestor-file", "terminal-file"],
)
async def test_open_workspace_rejects_linked_legacy_namespace_before_mutation(
    tmp_path: Path,
    namespace_shape: str,
) -> None:
    repo, _ = _make_repo_with_origin(tmp_path)
    owner = resolve_private_workspace_owner(
        repo,
        locations=private_root_locations(base=tmp_path / "private"),
    )
    outside = tmp_path / "outside"
    name = "run-outside-reanchor"
    if namespace_shape.startswith("ancestor"):
        external = outside / "worktrees" / name
    else:
        external = outside / name
    git_ops.worktree_add(repo, external, "main", detach=True)
    canary = external / "operator.bin"
    canary.write_bytes(b"outside operator bytes\x00")
    if namespace_shape == "ancestor-symlink":
        (repo / ".daydream").symlink_to(outside, target_is_directory=True)
    elif namespace_shape == "terminal-symlink":
        (repo / ".daydream").mkdir()
        (repo / ".daydream" / "worktrees").symlink_to(
            outside,
            target_is_directory=True,
        )
    elif namespace_shape == "ancestor-file":
        (repo / ".daydream").write_bytes(b"operator namespace bytes\x00")
    else:
        (repo / ".daydream").mkdir()
        (repo / ".daydream" / "worktrees").write_bytes(
            b"operator namespace bytes\x00"
        )
    before = _git(repo, "worktree", "list", "--porcelain")

    with pytest.raises(ArtifactVisibilityError, match="legacy operational"):
        async with open_workspace(
            repo,
            branch=None,
            base="main",
            force_ephemeral=False,
            skip_tests=True,
            private_owner=owner,
        ):
            pass

    assert canary.read_bytes() == b"outside operator bytes\x00"
    if namespace_shape.endswith("file"):
        unsafe_file = repo / ".daydream"
        if namespace_shape == "terminal-file":
            unsafe_file /= "worktrees"
        assert unsafe_file.read_bytes() == b"operator namespace bytes\x00"
    assert _git(repo, "worktree", "list", "--porcelain") == before
    assert not (owner.operational_state_root / "operational" / name).exists()


@pytest.mark.parametrize("entry_kind", ["live", "unknown"])
async def test_open_workspace_refuses_unsafe_legacy_entry_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    entry_kind: str,
) -> None:
    repo, _ = _make_repo_with_origin(tmp_path)
    locations = private_root_locations(base=tmp_path / "private")
    owner = resolve_private_workspace_owner(repo, locations=locations)
    legacy = repo / ".daydream" / "worktrees" / "run-old-reanchor"
    if entry_kind == "live":
        git_ops.worktree_add(
            repo,
            legacy,
            "main",
            detach=True,
            lock_reason="still-running",
        )
    else:
        legacy.mkdir(parents=True)
        (legacy / "retained.txt").write_text("operator bytes\n", encoding="utf-8")
    before = _git(repo, "worktree", "list", "--porcelain")

    monkeypatch.setattr(
        artifact_visibility,
        "_default_private_base",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected default lookup")),
    )
    with pytest.raises(ArtifactVisibilityError, match="legacy operational"):
        async with open_workspace(
            repo,
            branch=None,
            base="main",
            force_ephemeral=False,
            skip_tests=True,
            private_owner=owner,
        ):
            pass

    assert legacy.is_dir()
    assert _git(repo, "worktree", "list", "--porcelain") == before
    assert not (owner.operational_state_root / "operational").exists()
    if entry_kind == "unknown":
        assert (legacy / "retained.txt").read_bytes() == b"operator bytes\n"


async def test_open_workspace_retires_stale_legacy_audit_worktree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo, _ = _make_repo_with_origin(tmp_path)
    locations = private_root_locations(base=tmp_path / "private")
    owner = resolve_private_workspace_owner(repo, locations=locations)
    legacy = repo / ".daydream" / "audit" / "run-crashed"
    git_ops.worktree_add(
        repo,
        legacy,
        "main",
        detach=True,
        lock_reason="run-crashed",
    )
    locked = owner.git_common_dir / "worktrees" / legacy.name / "locked"
    old = 1_600_000_000
    os.utime(locked, (old, old))

    monkeypatch.setattr(
        artifact_visibility,
        "_default_private_base",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected default lookup")),
    )
    async with open_workspace(
        repo,
        branch=None,
        base="main",
        force_ephemeral=False,
        skip_tests=True,
        private_owner=owner,
    ):
        assert not legacy.exists()
        assert legacy.name not in _git(repo, "worktree", "list", "--porcelain")


async def test_legacy_preflight_rejects_unknown_before_moving_registered_entry(
    tmp_path: Path,
) -> None:
    repo, _ = _make_repo_with_origin(tmp_path)
    owner = resolve_private_workspace_owner(
        repo,
        locations=private_root_locations(base=tmp_path / "private"),
    )
    registered = repo / ".daydream" / "worktrees" / "run-known-reanchor"
    unknown = repo / ".daydream" / "worktrees" / "operator-notes"
    git_ops.worktree_add(repo, registered, "main", detach=True)
    unknown.mkdir()
    (unknown / "retained.txt").write_bytes(b"retain me")
    before = _git(repo, "worktree", "list", "--porcelain")

    with pytest.raises(ArtifactVisibilityError, match="unknown or unsafe"):
        async with open_workspace(
            repo,
            branch=None,
            base="main",
            force_ephemeral=False,
            skip_tests=True,
            private_owner=owner,
        ):
            pass

    assert registered.is_dir()
    assert (unknown / "retained.txt").read_bytes() == b"retain me"
    assert _git(repo, "worktree", "list", "--porcelain") == before
    assert not (owner.operational_state_root / "operational").exists()


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
    tmp_path: Path,
    source_kind: str,
    escape_kind: str,
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


async def test_open_workspace_rejects_escape_without_persistent_copy(
    artifact_runtime_root: Path, tmp_path: Path
) -> None:
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
    # Fail-closed: nothing is written into the retired source-local namespace.
    assert not (repo / ".daydream" / "worktrees" / "retained.cfg").exists()
    assert not (repo / ".daydream" / "worktrees").exists()
    # Cleanup ran: the source-owned operational directory retains no worktree.
    operational_dirs = list(
        (artifact_runtime_root.parent / "workspaces").glob("*/operational")
    )
    assert len(operational_dirs) == 1
    assert not any(operational_dirs[0].iterdir())


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


# --- 14. independent audit snapshots ---------------------------------------


def _audit_source_signature(repo: Path) -> tuple[object, ...]:
    status = git_ops.status_porcelain(repo)
    return (
        _git(repo, "rev-parse", "--verify", "HEAD", check=False),
        _git(repo, "symbolic-ref", "--quiet", "HEAD", check=False),
        _git(repo, "show-ref", check=False),
        git_ops.staged_patch(repo),
        status,
        (Path(_git(repo, "rev-parse", "--path-format=absolute", "--git-path", "index"))).read_bytes(),
        _git(repo, "worktree", "list", "--porcelain"),
        _git(repo, "remote", "-v"),
    )


@pytest.mark.anyio
async def test_audit_workspace_binds_diff_base_to_recorded_head(tmp_path: Path) -> None:
    repo, _ = _make_repo_with_origin(tmp_path)
    common = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-b", "feature")
    (repo / "feature.py").write_text("feature\n")
    _git(repo, "add", "feature.py")
    head = _commit(repo, "feature")

    async with open_audit_workspace(
        repo,
        run_id="branch-base",
        branch_base_ref="main",
        expected_head_sha=head,
    ) as audit:
        assert audit.branch_base_sha == common
        assert git_ops.head_sha(audit.repo) == head
        assert git_ops.commit_exists(audit.repo, common)
        assert git_ops.list_remotes(audit.repo, strict=True) == []
        assert _git(audit.repo, "for-each-ref", "refs/remotes") == ""


@pytest.mark.anyio
async def test_audit_workspace_rejects_recorded_head_race_before_snapshot(
    tmp_path: Path,
) -> None:
    repo, _ = _make_repo_with_origin(tmp_path)
    recorded_head = _git(repo, "rev-parse", "HEAD")
    (repo / "advanced.py").write_text("advanced\n")
    _git(repo, "add", "advanced.py")
    _commit(repo, "advance after workspace open")
    before = _audit_source_signature(repo)

    with pytest.raises(git_ops.SnapshotPreparationError, match="HEAD changed") as exc_info:
        async with open_audit_workspace(
            repo,
            run_id="head-race",
            branch_base_ref="main",
            expected_head_sha=recorded_head,
        ):
            pytest.fail("raced source yielded an audit workspace")
    assert str(repo) not in str(exc_info.value)
    assert _audit_source_signature(repo) == before


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["malformed-head", "remove-after-preference"])
async def test_audit_workspace_redacts_diff_base_probe_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    import shutil
    import sys

    repo, _ = _make_repo_with_origin(tmp_path)
    head = _git(repo, "rev-parse", "HEAD")
    before = _audit_source_signature(repo)
    real_git = shutil.which("git")
    assert real_git is not None
    shim_dir = tmp_path / "private git shim"
    shim_dir.mkdir()
    shim = shim_dir / "git"
    shim.write_text(
        f"#!{sys.executable}\n"
        "import os, subprocess, sys\n"
        "args = sys.argv[1:]\n"
        "real = os.environ['DAYDREAM_TEST_REAL_GIT']\n"
        "mode = os.environ['DAYDREAM_TEST_GIT_SHIM_MODE']\n"
        "head = os.environ['DAYDREAM_TEST_HEAD']\n"
        "is_preference = args[:2] == ['rev-parse', '--verify'] and "
        "len(args) == 3 and args[2].startswith('refs/remotes/origin/')\n"
        "is_head = args[:2] == ['rev-parse', '--verify'] and "
        "len(args) == 3 and args[2] == head + '^{commit}'\n"
        "if mode == 'malformed-head' and is_head:\n"
        "    print('PRIVATE_STDOUT_SENTINEL')\n"
        "    raise SystemExit(0)\n"
        "if mode == 'remove-after-preference' and is_preference:\n"
        "    result = subprocess.run([real, *args])\n"
        "    os.unlink(sys.argv[0])\n"
        "    raise SystemExit(result.returncode)\n"
        "raise SystemExit(subprocess.run([real, *args]).returncode)\n",
        encoding="utf-8",
    )
    shim.chmod(0o755)

    with monkeypatch.context() as shim_env:
        shim_env.setenv("DAYDREAM_TEST_REAL_GIT", real_git)
        shim_env.setenv("DAYDREAM_TEST_GIT_SHIM_MODE", mode)
        shim_env.setenv("DAYDREAM_TEST_HEAD", head)
        shim_env.setenv("PATH", str(shim_dir))
        with pytest.raises(
            git_ops.SnapshotPreparationError,
            match=r"^cannot resolve branch-focus diff merge-base$",
        ) as exc_info:
            async with open_audit_workspace(
                repo,
                run_id="redacted-base-probe",
                branch_base_ref="main",
                expected_head_sha=head,
            ):
                pytest.fail("failed source probe yielded an audit workspace")

    message = str(exc_info.value)
    assert "PRIVATE_STDOUT_SENTINEL" not in message
    assert "private git shim" not in message
    assert str(repo) not in message
    assert head not in message
    assert _audit_source_signature(repo) == before


@pytest.mark.anyio
async def test_audit_workspace_rejects_missing_cloned_diff_base_object(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An external Git fault removes only the cloned merge-base object."""
    import shutil
    import sys

    repo, _ = _make_repo_with_origin(tmp_path)
    common = _git(repo, "rev-parse", "HEAD")
    (repo / "main.py").write_text("main\n")
    _git(repo, "add", "main.py")
    _commit(repo, "main advancement")
    _git(repo, "checkout", "-b", "feature", common)
    (repo / "feature.py").write_text("feature\n")
    _git(repo, "add", "feature.py")
    head = _commit(repo, "feature")
    before = _audit_source_signature(repo)

    real_git = shutil.which("git")
    assert real_git is not None
    shim_dir = tmp_path / "git shim"
    shim_dir.mkdir()
    shim = shim_dir / "git"
    shim.write_text(
        f"#!{sys.executable}\n"
        "import os, pathlib, subprocess, sys\n"
        "real = os.environ['DAYDREAM_TEST_REAL_GIT']\n"
        "result = subprocess.run([real, *sys.argv[1:]])\n"
        "if result.returncode == 0 and len(sys.argv) > 2 and sys.argv[1] == 'clone':\n"
        "    destination = pathlib.Path(sys.argv[-1])\n"
        "    objects = destination / '.git' / 'objects'\n"
        "    packs = list((objects / 'pack').glob('*.pack'))\n"
        "    payloads = [pack.read_bytes() for pack in packs]\n"
        "    for packed in list((objects / 'pack').iterdir()):\n"
        "        packed.unlink()\n"
        "    for payload in payloads:\n"
        "        unpack = subprocess.run([real, 'unpack-objects', '-r'], cwd=destination, input=payload)\n"
        "        if unpack.returncode != 0:\n"
        "            sys.exit(unpack.returncode)\n"
        "    oid = os.environ['DAYDREAM_TEST_REMOVE_CLONED_OID']\n"
        "    (objects / oid[:2] / oid[2:]).unlink()\n"
        "sys.exit(result.returncode)\n"
    )
    shim.chmod(0o755)
    monkeypatch.setenv("DAYDREAM_TEST_REAL_GIT", real_git)
    monkeypatch.setenv("DAYDREAM_TEST_REMOVE_CLONED_OID", common)
    monkeypatch.setenv("PATH", f"{shim_dir}{os.pathsep}{os.environ['PATH']}")

    with pytest.raises(
        git_ops.SnapshotPreparationError,
        match="diff base object is missing from audit snapshot",
    ) as exc_info:
        async with open_audit_workspace(
            repo,
            run_id="missing-base-object",
            branch_base_ref="main",
            expected_head_sha=head,
        ):
            pytest.fail("snapshot with missing base object yielded")
    assert str(repo) not in str(exc_info.value)
    assert git_ops.commit_exists(repo, common)
    assert _audit_source_signature(repo) == before


@pytest.mark.anyio
@pytest.mark.parametrize("linked", [False, True])
async def test_audit_workspace_independent_of_source_storage(tmp_path: Path, linked: bool) -> None:
    repo, bare = _make_repo_with_origin(tmp_path)
    if linked:
        source = tmp_path / "linked source"
        _git(repo, "worktree", "add", "-b", "feature", str(source))
        repo = source
    for name in ("layered.txt", "deleted.txt", "recreated.txt"):
        (repo / name).write_text("committed\n")
    (repo / "binary.bin").write_bytes(b"committed\x00\xff")
    (repo / ".gitignore").write_text("secret.txt\n")
    (repo / "inside-link").symlink_to("layered.txt")
    outside = tmp_path / "outside-secret"
    outside.write_text("must not materialize")
    (repo / "outside-link").symlink_to(outside)
    _git(repo, "add", "-A")
    _commit(repo, "snapshot fixture")
    _git(repo, "branch", "another-branch")
    _git(repo, "tag", "independent-tag")
    (repo / "layered.txt").write_text("staged\n")
    (repo / "binary.bin").write_bytes(b"staged\x00\xfe")
    (repo / "new.bin").write_bytes(b"new\x00\xfd")
    _git(repo, "add", "layered.txt", "binary.bin", "new.bin")
    _git(repo, "rm", "recreated.txt")
    (repo / "recreated.txt").write_text("recreated but not indexed\n")
    (repo / "layered.txt").write_text("working\n")
    (repo / "deleted.txt").unlink()
    (repo / "secret.txt").write_text("ignored secret")
    (repo / "scratch.txt").write_text("untracked scratch")
    before = _audit_source_signature(repo)
    before_origin_refs = _git(bare, "show-ref")
    captured: Path | None = None
    async with open_audit_workspace(repo, run_id="independent-storage") as audit:
        captured = audit.repo
        assert not audit.repo.is_relative_to(repo.resolve())
        assert not repo.resolve().is_relative_to(audit.repo)
        assert audit.source == repo.resolve()
        assert audit.repo_git_common_dir != audit.source_git_common_dir
        assert not audit.repo_git_common_dir.is_relative_to(audit.source_git_common_dir)
        assert not audit.source_git_common_dir.is_relative_to(audit.repo_git_common_dir)
        assert git_ops.object_alternates(audit.repo, strict=True) == ()
        assert git_ops.list_remotes(audit.repo, strict=True) == []
        assert _git(audit.repo, "for-each-ref", "refs/remotes") == ""
        assert git_ops.list_local_branches(audit.repo) == git_ops.list_local_branches(repo)
        assert _git(audit.repo, "rev-parse", "independent-tag") == _git(repo, "rev-parse", "independent-tag")
        assert git_ops.staged_patch(audit.repo) == before[3]
        for name in ("layered.txt", "binary.bin", "new.bin", "recreated.txt"):
            assert (audit.repo / name).read_bytes() == (repo / name).read_bytes()
        assert not (audit.repo / "deleted.txt").exists()
        assert not (audit.repo / "secret.txt").exists()
        assert not (audit.repo / "scratch.txt").exists()
        assert (audit.repo / "inside-link").is_symlink()
        assert (audit.repo / "outside-link").is_symlink()
        assert os.readlink(audit.repo / "outside-link") == str(outside)
        assert audit.outward_symlinks == frozenset({audit.repo / "outside-link"})
        _configure_identity(audit.repo)
        (audit.repo / "model-note.txt").write_text("only in snapshot")
        _git(audit.repo, "add", "-A")
        _commit(audit.repo, "model commit")
        _git(audit.repo, "update-ref", "refs/heads/audit-only", "HEAD")
        assert _audit_source_signature(repo) == before
        assert _git(bare, "show-ref") == before_origin_refs
    assert captured is not None and not captured.exists()
    assert _audit_source_signature(repo) == before
    assert outside.read_text() == "must not materialize"


@pytest.mark.anyio
async def test_audit_workspace_unborn_is_independent(tmp_path: Path) -> None:
    repo = tmp_path / "unborn-trunk"
    _init_repo(repo)
    _git(repo, "symbolic-ref", "HEAD", "refs/heads/trunk")
    (repo / "staged.bin").write_bytes(b"staged\x00bytes\xff")
    _git(repo, "add", "staged.bin")
    (repo / "staged.bin").write_bytes(b"working\x00bytes\xfe")
    (repo / "scratch.txt").write_text("private")
    before = _audit_source_signature(repo)
    captured: Path | None = None
    async with open_audit_workspace(repo, run_id="independent-unborn") as audit:
        captured = audit.repo
        assert audit.repo != repo.resolve()
        assert git_ops.is_unborn_head(audit.repo)
        assert git_ops.symbolic_head(audit.repo, strict=True) == "trunk"
        assert git_ops.staged_patch(audit.repo) == before[3]
        assert (audit.repo / "staged.bin").read_bytes() == b"working\x00bytes\xfe"
        assert not (audit.repo / "scratch.txt").exists()
        _configure_identity(audit.repo)
        _commit(audit.repo, "snapshot initial commit")
        assert not git_ops.is_unborn_head(audit.repo)
        assert git_ops.is_unborn_head(repo)
    assert captured is not None and not captured.exists()
    assert _audit_source_signature(repo) == before


@pytest.mark.anyio
@pytest.mark.parametrize("failure", [RuntimeError, SystemExit])
async def test_audit_workspace_cleanup_runs_on_exception(tmp_path: Path, failure: type[BaseException]) -> None:
    repo, _ = _make_repo_with_origin(tmp_path)
    before = _audit_source_signature(repo)
    captured: Path | None = None
    with pytest.raises(failure, match="primary"):
        async with open_audit_workspace(repo, run_id="cleanup-error") as audit:
            captured = audit.repo
            raise failure("primary")
    assert captured is not None and not captured.exists()
    assert _audit_source_signature(repo) == before


@pytest.mark.anyio
async def test_audit_workspace_cleanup_runs_on_cancellation(tmp_path: Path) -> None:
    import anyio

    repo, _ = _make_repo_with_origin(tmp_path)
    captured: Path | None = None
    with anyio.CancelScope() as scope:
        async with open_audit_workspace(repo, run_id="cleanup-cancel") as audit:
            captured = audit.repo
            scope.cancel()
            await anyio.lowlevel.checkpoint()
    assert captured is not None and not captured.exists()


@pytest.mark.anyio
@pytest.mark.parametrize("primary", [False, True])
@pytest.mark.parametrize("cleanup_error", [OSError, RuntimeError])
async def test_audit_workspace_cleanup_failure_preserves_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, primary: bool,
    cleanup_error: type[Exception],
) -> None:
    import tempfile

    repo, _ = _make_repo_with_origin(tmp_path)
    cleanups: list[Path] = []
    real_cleanup = tempfile.TemporaryDirectory.cleanup

    def failing_cleanup(instance: tempfile.TemporaryDirectory[str]) -> None:
        cleanups.append(Path(instance.name))
        real_cleanup(instance)
        raise cleanup_error("injected cleanup failure")

    monkeypatch.setattr(tempfile.TemporaryDirectory, "cleanup", failing_cleanup)
    expected = RuntimeError if primary else GitError
    with pytest.raises(expected, match="primary" if primary else "cleanup failed"):
        async with open_audit_workspace(repo, run_id="cleanup-failure"):
            if primary:
                raise RuntimeError("primary")
    assert len(cleanups) == 1
    if primary:
        assert "cleanup failed during primary error" in caplog.text


@pytest.mark.anyio
async def test_audit_workspace_preparation_failure_cleans_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tempfile

    repo, _ = _make_repo_with_origin(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "child.txt").write_text("secret")
    (repo / "tracked").mkdir()
    (repo / "tracked" / "child.txt").write_text("tracked")
    _git(repo, "add", "tracked")
    _commit(repo, "directory")
    (repo / "tracked" / "child.txt").unlink()
    (repo / "tracked").rmdir()
    (repo / "tracked").symlink_to(outside, target_is_directory=True)
    before = _audit_source_signature(repo)
    created: list[Path] = []
    real_temp = tempfile.TemporaryDirectory

    def capture_temp(*args: object, **kwargs: object) -> tempfile.TemporaryDirectory[str]:
        assert not args
        prefix = kwargs["prefix"]
        assert isinstance(prefix, str)
        temporary = real_temp(prefix=prefix)
        created.append(Path(temporary.name))
        return temporary

    monkeypatch.setattr("daydream.workspace.tempfile.TemporaryDirectory", capture_temp)
    with pytest.raises(git_ops.SnapshotPreparationError, match="symlinked parent"):
        async with open_audit_workspace(repo, run_id="preparation-error"):
            pytest.fail("unsafe snapshot yielded")
    assert len(created) == 1 and not created[0].exists()
    assert (outside / "child.txt").read_text() == "secret"
    assert _audit_source_signature(repo) == before


@pytest.mark.anyio
async def test_open_workspace_unborn_requires_explicit_opt_in(tmp_path: Path) -> None:
    repo = tmp_path / "unborn"
    _init_repo(repo)
    _git(repo, "symbolic-ref", "HEAD", "refs/heads/trunk")
    with pytest.raises(GitError):
        async with open_workspace(repo, branch=None, base=None, force_ephemeral=False, skip_tests=True):
            pytest.fail("ordinary workspace admitted an unborn repository")
    async with open_workspace(
        repo, branch=None, base=None, force_ephemeral=False, skip_tests=True, allow_unborn=True,
    ) as work:
        assert work.is_unborn and work.head_sha is None and work.base_sha is None
        assert work.head_branch == work.base_branch == "trunk"
        assert work.repo == repo


@pytest.mark.anyio
@pytest.mark.parametrize("options", [{"branch": "trunk"}, {"base": "trunk"}, {"force_ephemeral": True}])
async def test_open_workspace_unborn_rejects_commit_anchored_options(
    tmp_path: Path, options: dict[str, object],
) -> None:
    repo = tmp_path / "unborn"
    _init_repo(repo)
    branch = options.get("branch")
    base = options.get("base")
    assert branch is None or isinstance(branch, str)
    assert base is None or isinstance(base, str)
    with pytest.raises(GitError, match="unborn improve"):
        async with open_workspace(
            repo, branch=branch, base=base, force_ephemeral=bool(options.get("force_ephemeral")),
            skip_tests=True, allow_unborn=True,
        ):
            pytest.fail("commit-anchored mode admitted unborn repository")
