"""Exercise workspace opening against real Git repositories and bare remotes."""

from __future__ import annotations

import secrets
import subprocess
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
import pytest

from daydream import git_ops
from daydream.artifact_visibility import (
    ArtifactVisibilityError,
    PrivateWorkspaceOwner,
    private_root_locations,
    resolve_private_workspace_owner,
)
from daydream.artifacts import ownership as artifact_ownership
from daydream.git_ops import BranchNotFoundError, GitError
from daydream.workspace import (
    WorkContext,
    WorkspaceCopyPathError,
    _resolve_base,
    _resolve_ref,
    copy_files_into_ephemeral,
    open_audit_workspace,
    open_workspace,
)
from tests.harness.git_helpers import (
    bare_remote as _bare_remote,
    commit as _commit,
    configure_identity as _configure_identity,
    git as _git,
    init_repo as _init_repo,
    install_diff_base_git_shim,
)


def _forbid_default_private_base(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(artifact_ownership, "_default_private_base",
        lambda: (_ for _ in ()).throw(AssertionError("unexpected default lookup")),
    )


def _private_owner(repo: Path, tmp_path: Path) -> PrivateWorkspaceOwner:
    """Resolve the private workspace owner against a per-test private root."""
    return resolve_private_workspace_owner(repo, locations=private_root_locations(base=tmp_path / "private"))


@asynccontextmanager
async def _open(repo: Path, owner: PrivateWorkspaceOwner, *, base: str = "main", ephemeral: bool = False,
) -> AsyncIterator[WorkContext]:
    """Open a workspace with an explicit private owner and the shared test defaults."""
    async with open_workspace(
        repo, branch=None, base=base, force_ephemeral=ephemeral, skip_tests=True, private_owner=owner,
    ) as work:
        yield work


def test_resolve_base_and_ref_prefer_pr_base_and_warn_on_checked_out_branch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr("daydream.workspace.shutil.which", lambda _name: "/bin/gh")
    monkeypatch.setattr(
        git_ops, "gh_pr_list_for_branch", lambda *_args, **_kwargs: (_ for _ in ()).throw(GitError("gh auth failed")),
    )
    monkeypatch.setattr(git_ops, "default_branch", lambda _repo: "trunk")

    assert _resolve_base(tmp_path, "feature", None) == "trunk"
    # An open PR's base wins over the default branch; rows without a usable base are skipped.
    monkeypatch.setattr(git_ops, "gh_pr_list_for_branch",
                        lambda *_args, **_kwargs: ["invalid", {"baseRefName": ""}, {"baseRefName": "develop"}])
    assert _resolve_base(tmp_path, "feature", None) == "develop"
    # Reviewing the checked-out branch warns that origin's copy is what gets reviewed.
    warnings: list[str] = []
    monkeypatch.setattr("daydream.ui.print_warning", lambda _console, message: warnings.append(message))
    monkeypatch.setattr(git_ops, "branch_exists", lambda *_a: True)
    monkeypatch.setattr(git_ops, "current_branch", lambda _repo: "feature")
    monkeypatch.setattr(git_ops, "upstream_ahead_count", lambda *_a: 2)
    assert _resolve_ref(tmp_path, "feature") == "origin/feature"
    assert warnings == [
        "feature is checked out in cwd and is 2 commits behind origin/feature — reviewing origin/feature."]


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
    token = secrets.token_hex(3)
    branch_name = branch.replace("/", "_")
    sidecar = tmp_path / f"sidecar-{branch_name}-{token}"
    _git(tmp_path, "clone", str(bare), str(sidecar))
    _configure_identity(sidecar)
    # Determine whether the branch already exists on origin.
    ls = subprocess.run(  # noqa: S603
        ["git", "ls-remote", "--heads", "origin", branch],  # noqa: S607
        cwd=sidecar, capture_output=True, text=True, check=True,
    )
    if ls.stdout.strip():
        # Branch exists on origin -- check it out as a tracking branch.
        _git(sidecar, "checkout", "-B", branch, f"origin/{branch}")
    elif branch != "main":
        _git(sidecar, "checkout", "-b", branch)
    new_file = sidecar / f"{branch_name}-{token}.txt"
    new_file.write_text("payload\n")
    _git(sidecar, "add", new_file.name)
    sha = _commit(sidecar, f"sidecar commit on {branch}")
    _git(sidecar, "push", "origin", branch)
    return sha


async def test_unsafe_operational_root_rejects_before_fetch_mutation(tmp_path: Path,) -> None:
    repo, bare = _make_repo_with_origin(tmp_path)
    owner = _private_owner(repo, tmp_path)
    new_origin_sha = _push_origin_commit_via_sidecar(tmp_path, bare)
    assert _git(repo, "rev-parse", "origin/main") != new_origin_sha
    outside = tmp_path / "outside"
    outside.mkdir()
    (owner.operational_state_root / "operational").symlink_to(outside, target_is_directory=True,)

    with pytest.raises(ArtifactVisibilityError, match="real directory"):
        async with _open(repo, owner, ephemeral=True):
            pass

    assert _git(repo, "rev-parse", "origin/main") != new_origin_sha
    assert not any("worktree " in line for line in _git(repo, "worktree", "list", "--porcelain").splitlines()[1:])


@pytest.mark.parametrize("namespace_shape", ["ancestor-file", ],
)
async def test_open_workspace_rejects_linked_public_namespace_before_mutation(tmp_path: Path, namespace_shape: str,
) -> None:
    repo, _ = _make_repo_with_origin(tmp_path)
    owner = _private_owner(repo, tmp_path)
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
        (repo / ".daydream" / "worktrees").symlink_to(outside, target_is_directory=True,)
    elif namespace_shape == "ancestor-file":
        (repo / ".daydream").write_bytes(b"operator namespace bytes\x00")
    else:
        (repo / ".daydream").mkdir()
        (repo / ".daydream" / "worktrees").write_bytes(b"operator namespace bytes\x00")
    before = _git(repo, "worktree", "list", "--porcelain")

    with pytest.raises(ArtifactVisibilityError, match="public operational storage"):
        async with _open(repo, owner):
            pass

    assert canary.read_bytes() == b"outside operator bytes\x00"
    if namespace_shape.endswith("file"):
        unsafe_file = repo / ".daydream"
        if namespace_shape == "terminal-file":
            unsafe_file /= "worktrees"
        assert unsafe_file.read_bytes() == b"operator namespace bytes\x00"
    assert _git(repo, "worktree", "list", "--porcelain") == before
    assert not (owner.operational_state_root / "operational" / name).exists()


async def test_unknown_branch_or_base_ref_raises_named_error(tmp_path: Path) -> None:
    repo, _ = _make_repo_with_origin(tmp_path)
    with pytest.raises(BranchNotFoundError):
        async with open_workspace(repo, branch="nope-not-here", base="main", force_ephemeral=False, skip_tests=False):
            pass
    with pytest.raises(BranchNotFoundError, match="base ref 'deadbeef' not found"):
        async with open_workspace(repo, branch=None, base="deadbeef", force_ephemeral=False, skip_tests=False):
            pass


def _dest(tmp_path: Path) -> Path:
    """Create the empty ``ephemeral`` copy destination under *tmp_path*."""
    dest = tmp_path / "ephemeral"
    dest.mkdir()
    return dest


def _ignore(repo: Path, *patterns: str, message: str = "ignore env") -> None:
    """Write *patterns* to the repo's ``.gitignore`` and commit it."""
    (repo / ".gitignore").write_text("".join(f"{pattern}\n" for pattern in patterns))
    _git(repo, "add", ".gitignore")
    _commit(repo, message)


@pytest.mark.parametrize("root_kind", ["destination"])
def test_copy_rejects_resolved_symlink_escape(tmp_path: Path, root_kind: str) -> None:
    repo, _ = _make_repo_with_origin(tmp_path)
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    secret = outside_dir / "secret.txt"
    secret.write_text("KEEP\n")
    (repo / "pyproject.toml").write_text('[tool.daydream.workspace]\ncopy = ["sub/leak.txt"]\n')
    (repo / "sub").mkdir()

    dest = _dest(tmp_path)
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

    with pytest.raises(WorkspaceCopyPathError, match=f"resolves outside the {root_label} worktree",):
        copy_files_into_ephemeral(repo, dest, extra=None, skip=False)

    # Nothing was written into the escaped directory.
    assert not (outside_dir / "leak.txt").exists()
    assert secret.read_text() == "KEEP\n"


def _audit_source_signature(repo: Path) -> tuple[object, ...]:
    status = git_ops.status_porcelain(repo)
    return (_git(repo, "rev-parse", "--verify", "HEAD", check=False),
        _git(repo, "symbolic-ref", "--quiet", "HEAD", check=False), _git(repo, "show-ref", check=False),
        git_ops.staged_patch(repo), status,
        (Path(_git(repo, "rev-parse", "--path-format=absolute", "--git-path", "index"))).read_bytes(),
        _git(repo, "worktree", "list", "--porcelain"), _git(repo, "remote", "-v"),
    )


@pytest.mark.anyio
async def test_audit_workspace_rejects_recorded_head_race_before_snapshot(tmp_path: Path,) -> None:
    repo, _ = _make_repo_with_origin(tmp_path)
    recorded_head = _git(repo, "rev-parse", "HEAD")
    (repo / "advanced.py").write_text("advanced\n")
    _git(repo, "add", "advanced.py")
    _commit(repo, "advance after workspace open")
    before = _audit_source_signature(repo)

    with pytest.raises(git_ops.SnapshotPreparationError, match="HEAD changed") as exc_info:
        async with open_audit_workspace(
            repo, run_id="head-race", branch_base_ref="main", expected_head_sha=recorded_head,
        ):
            pytest.fail("raced source yielded an audit workspace")
    assert str(repo) not in str(exc_info.value)
    assert _audit_source_signature(repo) == before


@pytest.mark.anyio
@pytest.mark.parametrize("mode", ["remove-after-preference"])
async def test_audit_workspace_redacts_diff_base_probe_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:

    repo, _ = _make_repo_with_origin(tmp_path)
    head = _git(repo, "rev-parse", "HEAD")
    before = _audit_source_signature(repo)

    with install_diff_base_git_shim(tmp_path, monkeypatch, mode=mode, head=head) as shim_dir:
        with pytest.raises(git_ops.SnapshotPreparationError, match=r"^cannot resolve branch-focus diff merge-base$",
        ) as exc_info:
            async with open_audit_workspace(
                repo, run_id="redacted-base-probe", branch_base_ref="main", expected_head_sha=head,
            ):
                pytest.fail("failed source probe yielded an audit workspace")

    message = str(exc_info.value)
    assert "PRIVATE_STDOUT_SENTINEL" not in message
    assert shim_dir.name not in message
    assert str(repo) not in message
    assert head not in message
    assert _audit_source_signature(repo) == before


@pytest.mark.anyio
async def test_audit_workspace_cleanup_failure_preserves_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """A cleanup failure surfaces alone, but never replaces a primary error; cancellation still cleans up."""
    repo, _ = _make_repo_with_origin(tmp_path)
    captured: Path | None = None
    with anyio.CancelScope() as scope:
        async with open_audit_workspace(repo, run_id="cleanup-cancel") as audit:
            captured = audit.repo
            scope.cancel()
            await anyio.lowlevel.checkpoint()
    assert captured is not None and not captured.exists()
    cleanups: list[Path] = []
    real_cleanup = tempfile.TemporaryDirectory.cleanup
    for primary, cleanup_error in ((False, OSError), (True, RuntimeError)):
        def failing_cleanup(instance: tempfile.TemporaryDirectory[str], error: type[Exception] = cleanup_error) -> None:
            cleanups.append(Path(instance.name))
            real_cleanup(instance)
            raise error("injected cleanup failure")

        monkeypatch.setattr(tempfile.TemporaryDirectory, "cleanup", failing_cleanup)
        expected = RuntimeError if primary else GitError
        with pytest.raises(expected, match="primary" if primary else "cleanup failed"):
            async with open_audit_workspace(repo, run_id=f"cleanup-failure-{primary}"):
                if primary:
                    raise RuntimeError("primary")
    assert len(cleanups) == 2
    assert "cleanup failed during primary error" in caplog.text


@pytest.mark.anyio
@pytest.mark.parametrize("options", [{"branch": "trunk"}, ])
async def test_open_workspace_unborn_rejects_commit_anchored_options(tmp_path: Path, options: dict[str, object],
) -> None:
    repo = tmp_path / "unborn"
    _init_repo(repo)
    branch = options.get("branch")
    base = options.get("base")
    assert branch is None or isinstance(branch, str)
    assert base is None or isinstance(base, str)
    with pytest.raises(GitError, match="unborn improve"):
        async with open_workspace(repo, branch=branch, base=base, force_ephemeral=bool(options.get("force_ephemeral")),
            skip_tests=True, allow_unborn=True,
        ):
            pytest.fail("commit-anchored mode admitted unborn repository")


async def test_ephemeral_copy_configuration_overrides_then_falls_back_to_defaults(tmp_path: Path) -> None:
    repo, _ = _make_repo_with_origin(tmp_path)
    _ignore(repo, ".env", ".env.local", "local/")
    (repo / ".env").write_text("DEFAULT=1\n")
    (repo / "custom.cfg").write_text("k=v\n")
    (repo / "local").mkdir()
    (repo / "local/secrets.toml").write_text("token = 'x'\n")
    config = repo / "pyproject.toml"
    config.write_text('[tool.daydream.workspace]\ncopy = ["custom.cfg", "local/secrets.toml"]\n')
    owner = _private_owner(repo, tmp_path)
    async with open_workspace(
        repo,
        branch=None,
        base="main",
        force_ephemeral=True,
        skip_tests=False,
        private_owner=owner,
    ) as configured:
        assert (configured.repo / "custom.cfg").read_text() == "k=v\n"
        assert (configured.repo / "local/secrets.toml").read_text() == "token = 'x'\n"
        assert not (configured.repo / ".env").exists()
    config.write_text('tool = "not-a-table"\n')
    async with open_workspace(
        repo,
        branch=None,
        base="main",
        force_ephemeral=True,
        skip_tests=False,
        private_owner=owner,
    ) as fallback:
        assert (fallback.repo / ".env").read_text() == "DEFAULT=1\n"
        assert not (fallback.repo / "custom.cfg").exists()
        assert not (fallback.repo / ".env.local").exists()
    assert not configured.repo.exists() and not fallback.repo.exists()
    assert (repo / "custom.cfg").read_text() == "k=v\n" and (repo / ".env").read_text() == "DEFAULT=1\n"
