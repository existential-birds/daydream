"""Open in-place, ephemeral worktree, and independent audit workspaces.

All Git operations pass through git_ops.
"""

from __future__ import annotations

import logging
import secrets
import shutil
import stat
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import AsyncIterator, Iterable

from daydream import git_ops
from daydream.artifact_visibility import (
    ArtifactVisibilityError,
    PrivateWorkspaceOwner,
    operational_worktree_root,
    private_root_locations,
    resolve_private_workspace_owner,
    validate_private_workspace_owner,
)
from daydream.config_file import load_toml_or_empty
from daydream.git_ops import BranchNotFoundError, GitError


def reject_public_operational_storage(source: Path) -> None:
    """Refuse unsupported source-local worktrees without following or changing operator data."""
    root = source / ".daydream"
    try:
        metadata = root.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ArtifactVisibilityError("public operational storage is inaccessible") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise ArtifactVisibilityError("public operational storage ancestor must be a real directory")
    for namespace in ("worktrees", "audit"):
        try:
            (root / namespace).lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ArtifactVisibilityError("public operational storage is inaccessible") from exc
        raise ArtifactVisibilityError(
            f"Unsupported public operational storage .daydream/{namespace}; relocate it before opening a workspace"
        )


_logger = logging.getLogger(__name__)

# Default copy list for ephemeral worktrees when ``pyproject.toml`` does not
# specify ``[tool.daydream.workspace] copy``.  Files are only copied when
# they are gitignored in the source -- tracked files come along with the
# worktree checkout itself.
_DEFAULT_COPY_PATHS: tuple[str, ...] = (".env", ".env.local")
_DEFAULT_COPY_GLOB = ".env.*"
class WorkspaceCopyPathError(GitError):
    """A copy entry escapes either workspace; all entries are checked before any copy."""


class UnbornWorkspaceError(GitError):
    """An unborn checkout was requested in a mode requiring commit anchors."""


@dataclass(frozen=True)
class WorkContext:
    """Resolved working repository and its source, with commit anchors captured at open.

    ``base_sha`` is the merge-base, ``head_branch`` is None for detached HEAD,
    and both SHA anchors are None only for explicitly admitted unborn Improve.
    ``run_id`` identifies ephemeral paths and intent files.
    """

    repo: Path
    source: Path
    base_branch: str
    base_sha: str | None
    head_branch: str | None
    head_sha: str | None
    is_ephemeral: bool
    run_id: str

    def __post_init__(self) -> None:
        if (self.base_sha is None) != (self.head_sha is None):
            raise ValueError("workspace commit anchors must both exist or both be absent")

    @property
    def is_unborn(self) -> bool:
        """True only for an explicitly admitted improve-only unborn checkout."""
        return self.head_sha is None




@asynccontextmanager
async def open_workspace(
    source: Path,
    *,
    branch: str | None,
    base: str | None,
    force_ephemeral: bool,
    extra_copy: list[Path] | None = None,
    skip_tests: bool,
    allow_unborn: bool = False,
    private_owner: PrivateWorkspaceOwner | None = None,
    auth: git_ops.GitHubAuth = git_ops.INHERIT_GITHUB_AUTH,
) -> AsyncIterator[WorkContext]:
    """Open in-place unless ``branch`` or ``force_ephemeral`` requests a detached worktree.

    Ephemeral runs fetch first: a branch selects origin/<branch>, otherwise HEAD.
    The base defaults to an open PR's base, then the repository default branch.
    Ignored support files are copied unless skip_tests; cleanup runs on every exit.

    Validate/resolve the source's private owner before any mutation. Unborn Improve
    requires explicit opt-in, a symbolic HEAD, and no branch/base/ephemeral override.
    Mode-specific wrong-branch checks belong to the runner.
    """
    git_ops.assert_is_worktree(source)
    if private_owner is None:
        private_owner = resolve_private_workspace_owner(
            source,
            locations=private_root_locations(),
        )
    else:
        validate_private_workspace_owner(private_owner, source=source)
    source = private_owner.source
    reject_public_operational_storage(source)

    if allow_unborn and git_ops.is_unborn_head(source):
        if branch is not None or base is not None or force_ephemeral:
            raise UnbornWorkspaceError("unborn improve requires an in-place checkout without branch/base overrides")
        symbolic = git_ops.symbolic_head(source, strict=True)
        if symbolic is None:
            raise UnbornWorkspaceError("unborn improve requires a symbolic HEAD")
        yield WorkContext(
            repo=source, source=source, base_branch=symbolic, base_sha=None,
            head_branch=symbolic, head_sha=None, is_ephemeral=False, run_id=_make_run_id(),
        )
        return

    is_ephemeral = force_ephemeral or branch is not None
    operational_root = (
        operational_worktree_root(private_owner) if is_ephemeral else None
    )

    if is_ephemeral:
        # Fetch from the source -- the ephemeral worktree does not exist yet.
        git_ops.fetch(source)

    resolved_ref = _resolve_ref(source, branch) if is_ephemeral else None
    base_branch = _resolve_base(source, branch, base, auth=auth)

    run_id = _make_run_id()
    worktree_path: Path | None = None

    try:
        if is_ephemeral:
            assert resolved_ref is not None  # narrows for the type-checker
            assert operational_root is not None
            worktree_path = operational_root / run_id
            git_ops.worktree_add(source, worktree_path, resolved_ref, detach=True)
            copy_files_into_ephemeral(
                source,
                worktree_path,
                extra=extra_copy,
                skip=skip_tests,
            )
            repo = worktree_path
        else:
            repo = source

        # Validate the resolved base exists relative to the working repo.
        # --base accepts any commit-ish (SHA, tag, relative expr), so use
        # ref_exists rather than the named-ref-only branch_exists.
        if not git_ops.ref_exists(repo, base_branch):
            raise BranchNotFoundError(f"base ref '{base_branch}' not found in {repo}")

        base_sha = git_ops.merge_base(repo, base_branch)
        if base_sha is None:
            raise BranchNotFoundError(f"could not resolve merge-base for '{base_branch}' in {repo}")

        head_sha = git_ops.head_sha(repo)
        head_branch = git_ops.current_branch(repo)

        ctx = WorkContext(
            repo=repo,
            source=source,
            base_branch=base_branch,
            base_sha=base_sha,
            head_branch=head_branch,
            head_sha=head_sha,
            is_ephemeral=is_ephemeral,
            run_id=run_id,
        )

        yield ctx
    finally:
        if worktree_path is not None:
            try:
                git_ops.worktree_remove(source, worktree_path, force=True)
            except GitError as exc:
                from daydream.agent import console
                from daydream.ui import print_warning

                print_warning(console, f"Failed to remove ephemeral worktree {worktree_path}: {exc}")


@dataclass(frozen=True)
class AuditWorkspace:
    """Independent audit repository plus the boundaries its backend must enforce.

    ``branch_base_sha`` is the source-resolved immutable merge-base for a
    branch-focus run. It is absent for full-repository and unborn improve.
    """

    repo: Path
    source: Path
    repo_git_common_dir: Path
    source_git_common_dir: Path
    outward_symlinks: frozenset[Path]
    branch_base_sha: str | None = None


@asynccontextmanager
async def open_audit_workspace(
    source: Path,
    *,
    run_id: str,
    branch_base_ref: str | None = None,
    expected_head_sha: str | None = None,
) -> AsyncIterator[AuditWorkspace]:
    """Open an outside-source snapshot with independent objects, index and refs.

    Only tracked worktree/index state is copied; ignored and untracked data is
    excluded. This isolates Git storage, not arbitrary host filesystem access.
    The improve backend must separately enforce the returned tool-root boundary.
    A genuine unborn checkout remains unborn in a separate repository.

    When ``branch_base_ref`` and ``expected_head_sha`` are supplied together,
    resolve the source's remote-preferred branch diff base before cloning and
    attest both immutable commits in the clone before yielding. Supplying only
    one is invalid. No symbolic base or remote metadata crosses the boundary.

    The process-owned temporary directory is cleaned on every exit. Cleanup
    errors surface unless a body/preparation/cancellation error is already
    active, in which case that primary error is retained and cleanup is warned.
    """
    if (branch_base_ref is None) != (expected_head_sha is None):
        raise git_ops.SnapshotPreparationError(
            "audit snapshot branch-base inputs must be paired"
        )
    source = source.resolve(strict=True)
    git_ops.assert_is_worktree(source)
    temporary = tempfile.TemporaryDirectory(prefix=f"daydream-audit-{run_id}-")
    primary_error = False
    try:
        branch_base_sha: str | None = None
        if branch_base_ref is not None and expected_head_sha is not None:
            try:
                current_head = git_ops.head_sha(source)
                if current_head != expected_head_sha:
                    raise git_ops.SnapshotPreparationError(
                        "source HEAD changed after workspace resolution"
                    )
                branch_base_sha = git_ops.resolve_diff_merge_base(
                    source, branch_base_ref, expected_head_sha
                )
            except git_ops.SnapshotPreparationError:
                raise
            except git_ops.GitError as exc:
                raise git_ops.SnapshotPreparationError(
                    "cannot resolve branch-focus diff merge-base"
                ) from exc
        temporary_root = Path(temporary.name).resolve()
        if source.is_relative_to(temporary_root) or temporary_root.is_relative_to(source):
            raise git_ops.SnapshotPreparationError("audit temporary directory must be outside source")
        snapshot = git_ops.prepare_independent_snapshot(
            source, temporary_root / "repo", include_untracked=False,
        )
        if expected_head_sha is not None and branch_base_sha is not None:
            try:
                snapshot_head = git_ops.head_sha(snapshot.repo)
                base_present = git_ops.commit_exists(snapshot.repo, branch_base_sha)
                base_is_ancestor = base_present and git_ops.is_ancestor(
                    snapshot.repo, branch_base_sha, expected_head_sha
                )
            except git_ops.GitError as exc:
                raise git_ops.SnapshotPreparationError(
                    "cannot verify branch-focus commits in audit snapshot"
                ) from exc
            if snapshot_head != expected_head_sha:
                raise git_ops.SnapshotPreparationError(
                    "audit snapshot HEAD does not match the recorded source HEAD"
                )
            if not base_present:
                raise git_ops.SnapshotPreparationError(
                    "branch-focus diff base object is missing from audit snapshot"
                )
            if not base_is_ancestor:
                raise git_ops.SnapshotPreparationError(
                    "branch-focus diff base is not an ancestor of audit snapshot HEAD"
                )
        yield AuditWorkspace(
            repo=snapshot.repo,
            source=source,
            repo_git_common_dir=git_ops.git_common_dir(snapshot.repo),
            source_git_common_dir=git_ops.git_common_dir(source),
            outward_symlinks=snapshot.outward_symlinks,
            branch_base_sha=branch_base_sha,
        )
    except BaseException:
        primary_error = True
        raise
    finally:
        try:
            temporary.cleanup()
        except Exception as exc:
            if not primary_error:
                raise GitError(f"audit snapshot cleanup failed: {type(exc).__name__}") from exc
            _logger.warning("audit snapshot cleanup failed during primary error: %s", type(exc).__name__)


def copy_files_into_ephemeral(
    source: Path,
    dest: Path,
    *,
    extra: list[Path] | None = None,
    skip: bool = False,
) -> list[Path]:
    """Copy configured support files, then additive ``extra`` entries, in first-seen order.

    The pyproject workspace.copy list replaces defaults; defaults include only
    ignored .env/.env.local/.env.* files. Missing/non-files are skipped.
    Validate every entry against both roots before copying anything: absolute,
    parent-traversing, unresolved or outward-symlink paths abort the whole copy.
    ``skip`` returns immediately without inspecting either workspace.
    """
    if skip:
        return []

    entries = _resolve_copy_entries(source)

    if extra:
        entries.extend(extra)

    # De-duplicate while preserving first-occurrence order.
    unique = _dedupe_ordered(entries)

    # Fail-closed validation against BOTH the source and destination roots,
    # before any copy runs. An entry that escapes either root aborts the
    # whole copy, leaving nothing partially copied.
    for rel in unique:
        for root, root_label in ((source, "source"), (dest, "destination")):
            _resolve_workspace_copy_path(rel, root, root_label)

    copied: list[Path] = []
    for rel_path in unique:
        src_file = source / rel_path
        if not src_file.is_file():
            continue

        dest_file = dest / rel_path
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dest_file)
        copied.append(rel_path)

    return copied


def _dedupe_ordered(entries: Iterable[str | Path]) -> list[Path]:
    """De-duplicate path-like *entries*, preserving first-occurrence order."""
    return list(dict.fromkeys(Path(entry) for entry in entries))


def _resolve_workspace_copy_path(entry: Path, root: Path, root_label: str) -> None:
    """Require a relative, parent-free path whose resolved target stays within root.

    Inward symlinks are allowed. Resolution errors and escapes raise
    WorkspaceCopyPathError naming the affected root.
    """
    if entry.is_absolute() or ".." in entry.parts:
        raise WorkspaceCopyPathError(f"workspace copy path must be relative and must not contain '..': {entry}")

    try:
        resolved = (root / entry).resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise WorkspaceCopyPathError(
            f"workspace copy path could not be resolved in the {root_label} worktree: {entry}"
        ) from exc

    if not resolved.is_relative_to(root.resolve()):
        raise WorkspaceCopyPathError(f"workspace copy path resolves outside the {root_label} worktree: {entry}")


def _make_run_id() -> str:
    """Return a unique ``<UTC YYYYMMDDHHMMSS>-<hex8>`` identifier."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    return f"{timestamp}-{secrets.token_hex(4)}"


def _resolve_ref(source: Path, branch: str | None) -> str:
    """Resolve the git ref to check out in the ephemeral worktree."""
    if branch is None:
        return git_ops.head_sha(source)

    if not git_ops.branch_exists(source, branch):
        raise BranchNotFoundError(f"branch '{branch}' not found locally or on origin in {source}")

    # Emit the staleness warning when the requested branch is also the
    # currently checked out branch in source. ``upstream_ahead_count`` returns
    # 0 (rather than raising) when no upstream is configured -- the message
    # still mentions the count for transparency.
    current = git_ops.current_branch(source)
    if current == branch:
        ahead = git_ops.upstream_ahead_count(source, branch)
        from daydream.agent import console
        from daydream.ui import print_warning

        print_warning(
            console,
            f"{branch} is checked out in cwd and is {ahead} commits behind "
            f"origin/{branch} — reviewing origin/{branch}.",
        )

    return f"origin/{branch}"


def _resolve_base(
    source: Path, branch: str | None, base: str | None, *,
    auth: git_ops.GitHubAuth = git_ops.INHERIT_GITHUB_AUTH,
) -> str:
    """Pick the base branch per the locked resolution rules."""
    if base is not None:
        return base

    if branch is not None and shutil.which("gh") is not None:
        try:
            prs = git_ops.gh_pr_list_for_branch(source, branch, auth=auth)
        except GitError as exc:
            _logger.debug("PR base lookup failed for branch %r: %s", branch, exc)
            prs = []
        if not prs:
            _logger.debug("gh_pr_list_for_branch returned empty for branch %r", branch)
        for pr in prs:
            base_ref = pr.get("baseRefName") if isinstance(pr, dict) else None
            if isinstance(base_ref, str) and base_ref:
                return base_ref

    return git_ops.default_branch(source)


def _resolve_copy_entries(source: Path) -> list[Path]:
    """A configured workspace.copy replaces defaults; use ignored .env* defaults only when absent."""
    pyproject = source / "pyproject.toml"
    if pyproject.is_file():
        data = load_toml_or_empty(pyproject)
        tool = data.get("tool")
        daydream_cfg = tool.get("daydream") if isinstance(tool, dict) else None
        workspace_cfg = daydream_cfg.get("workspace") if isinstance(daydream_cfg, dict) else None
        override = workspace_cfg.get("copy") if isinstance(workspace_cfg, dict) else None
        if isinstance(override, list):
            return [Path(p) for p in override if isinstance(p, str)]

    candidates: list[str] = list(_DEFAULT_COPY_PATHS)
    candidates.extend(p.name for p in source.glob(_DEFAULT_COPY_GLOB))

    unique = _dedupe_ordered(candidates)
    return [p for p in unique if git_ops.check_ignore(source, str(p))]
