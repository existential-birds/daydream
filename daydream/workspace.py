"""WorkContext abstraction for in-place vs ephemeral worktree execution.

This module is the single entry point daydream uses to *open* the directory
it operates on for a single run.  Two modes are supported:

* **In-place** -- daydream operates on the user's checked-out worktree.
* **Ephemeral** -- daydream creates a detached worktree under
  ``<source>/.daydream/worktrees/<run_id>`` and removes it on exit.

The resolution rules and ordering live in :func:`open_workspace` and are
deliberately fixed.

The module shells out via :mod:`daydream.git_ops` only.
"""

from __future__ import annotations

import logging
import secrets
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import AsyncIterator, Iterable

from daydream import git_ops
from daydream.config_file import load_toml_or_empty
from daydream.git_ops import BranchNotFoundError, GitError

_logger = logging.getLogger(__name__)

# Default copy list for ephemeral worktrees when ``pyproject.toml`` does not
# specify ``[tool.daydream.workspace] copy``.  Files are only copied when
# they are gitignored in the source -- tracked files come along with the
# worktree checkout itself.
_DEFAULT_COPY_PATHS: tuple[str, ...] = (".env", ".env.local")
_DEFAULT_COPY_GLOB = ".env.*"


# --- Public types ------------------------------------------------------------


class WorkspaceCopyPathError(GitError):
    """Raised when a ``[tool.daydream.workspace] copy`` / ``--copy`` entry
    escapes the source checkout or the ephemeral destination worktree.

    This is the fail-closed boundary of :func:`copy_files_into_ephemeral`:
    no workspace-copy entry may read a file outside the source checkout or
    write outside the ephemeral worktree. Raised before any file is copied.
    """


class UnbornWorkspaceError(GitError):
    """An unborn checkout was requested in a mode requiring commit anchors."""


@dataclass(frozen=True)
class WorkContext:
    """Resolved working environment for a daydream run.

    Attributes:
        repo: The directory daydream operates on (the source for in-place
            runs, the ephemeral worktree path otherwise).
        source: The original ``cwd`` the user invoked daydream from. Equal to
            :attr:`repo` for in-place runs.
        base_branch: Resolved base ref name (e.g. ``"main"``).
        base_sha: Merge-base SHA between :attr:`base_branch` and the working
            ``HEAD``, captured at workspace open time; None only for unborn improve.
        head_branch: Branch name at :attr:`repo`'s ``HEAD``, or ``None`` when
            ``HEAD`` is detached (e.g. ephemeral worktrees).
        head_sha: Full SHA of :attr:`repo`'s ``HEAD``, or None for unborn improve.
        is_ephemeral: True when :attr:`repo` is an ephemeral worktree.
        run_id: ``<UTC YYYYMMDDHHMMSS>-<hex8>`` identifier used for the
            ephemeral path and intent files.
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

    @property
    def is_in_place(self) -> bool:
        """Return True when this context runs on the user's source worktree."""
        return not self.is_ephemeral


# --- Public API --------------------------------------------------------------


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
) -> AsyncIterator[WorkContext]:
    """Open a workspace for a daydream run, yielding a :class:`WorkContext`.

    Resolution rules (locked):

    * ``branch is None`` and not ``force_ephemeral`` -> in-place at *source*;
      no fetch; no cleanup.
    * ``branch is None`` and ``force_ephemeral`` -> ephemeral at *source*'s
      current ``HEAD``.
    * ``branch`` provided -> ALWAYS ephemeral, detached at ``origin/<branch>``
      after a fetch. When ``branch`` is also currently checked out in
      *source*, a staleness warning is emitted.

    Args:
        source: User's worktree (the directory daydream was invoked from).
        branch: Optional branch name to review.
        base: Optional base branch name. When ``None``, resolved via the open
            PR head (if any) or :func:`git_ops.default_branch`.
        force_ephemeral: Run ephemerally even when no branch is given.
        extra_copy: Additional paths supplied via ``--copy`` flags.
        skip_tests: When True, suppress copying gitignored files into the
            ephemeral worktree (used by ``--comment`` / ``--review`` flows).
        allow_unborn: Improve-only opt-in to an in-place unborn context with
            both SHA anchors absent. Other modes keep their born-HEAD requirement.

    Yields:
        A :class:`WorkContext` describing the resolved working environment.

    Raises:
        NotAWorktreeError: If *source* is not the top-level of a worktree.
        BranchNotFoundError: If *branch* or the resolved *base* cannot be
            located locally or on ``origin``.
        GitError: For other unexpected git failures.

    Note:
        The design doc (``2026-04-30-worktree-isolation-and-mode-consolidation.md``)
        specifies a ``WrongBranchError`` check here when ``branch is None`` and
        ``current_branch == base_branch``. That check lives in
        :func:`daydream.runner._dispatch` instead because it must fire only for
        ``output_mode="loop"`` (not ``--comment`` or ``--review``), and this
        function is deliberately mode-agnostic.
    """
    git_ops.assert_is_worktree(source)

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

    if is_ephemeral:
        # Fetch from the source -- the ephemeral worktree does not exist yet.
        git_ops.fetch(source)

    resolved_ref = _resolve_ref(source, branch) if is_ephemeral else None
    base_branch = _resolve_base(source, branch, base)

    run_id = _make_run_id()
    worktree_path: Path | None = None

    try:
        if is_ephemeral:
            assert resolved_ref is not None  # narrows for the type-checker
            worktree_path = source / ".daydream" / "worktrees" / run_id
            worktree_path.parent.mkdir(parents=True, exist_ok=True)
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
                _warn_removal_failed(worktree_path, exc, kind="ephemeral worktree")


@dataclass(frozen=True)
class AuditWorkspace:
    """Independent audit repository plus the boundaries its backend must enforce."""

    repo: Path
    source: Path
    repo_git_common_dir: Path
    source_git_common_dir: Path
    outward_symlinks: frozenset[Path]


@asynccontextmanager
async def open_audit_workspace(
    source: Path,
    *,
    run_id: str,
) -> AsyncIterator[AuditWorkspace]:
    """Open an outside-source snapshot with independent objects, index and refs.

    Only tracked worktree/index state is copied; ignored and untracked data is
    excluded. This isolates Git storage, not arbitrary host filesystem access.
    The improve backend must separately enforce the returned tool-root boundary.
    A genuine unborn checkout remains unborn in a separate repository.

    The process-owned temporary directory is cleaned on every exit. Cleanup
    errors surface unless a body/preparation/cancellation error is already
    active, in which case that primary error is retained and cleanup is warned.
    """
    source = source.resolve(strict=True)
    git_ops.assert_is_worktree(source)
    temporary = tempfile.TemporaryDirectory(prefix=f"daydream-audit-{run_id}-")
    primary_error = False
    try:
        temporary_root = Path(temporary.name).resolve()
        if source.is_relative_to(temporary_root) or temporary_root.is_relative_to(source):
            raise git_ops.SnapshotPreparationError("audit temporary directory must be outside source")
        snapshot = git_ops.prepare_independent_snapshot(
            source, temporary_root / "repo", include_untracked=False,
        )
        yield AuditWorkspace(
            repo=snapshot.repo,
            source=source,
            repo_git_common_dir=git_ops.git_common_dir(snapshot.repo),
            source_git_common_dir=git_ops.git_common_dir(source),
            outward_symlinks=snapshot.outward_symlinks,
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


def _prune_stale_locked_worktrees(
    repo: Path,
    paths: Iterable[Path],
    *,
    stale_after_s: int,
) -> int:
    """Remove stale locked worktrees from a discovery iterable, tolerating failures.

    Single source of truth for the lock-aware prune policy shared by the
    audit prune (this module) and the re-anchor prune
    (``daydream.improve.plans.prune_stale_reanchor_worktrees``), so the
    staleness window, live-lock skip rule, and unlock-before-remove ordering
    live in one place and cannot silently drift. A live worktree (lock age
    near zero) is skipped without any removal attempt, so a concurrent run
    mid-write is never destroyed. A worktree whose lock is older than
    ``stale_after_s`` is a crashed run's leftover: it is reclaimed via
    ``git_ops.worktree_remove_unlocked`` (unlock, then force-remove) rather
    than wedged forever. Unlocked worktrees are removed the same way, the
    unlock being a no-op for them. Individual failures are tolerated so one
    stale worktree never blocks a plan run.
    """
    removed = 0
    for path in paths:
        try:
            locked_at = git_ops.worktree_lock_mtime(repo, path)
            if locked_at is not None and time.time() - locked_at <= stale_after_s:
                # Live worktree (lock age near zero): never unlock or remove
                # it, so a concurrent run mid-write is not destroyed.
                continue
            git_ops.worktree_remove_unlocked(repo, path)
        except git_ops.GitError:
            continue
        removed += 1
    return removed


def copy_files_into_ephemeral(
    source: Path,
    dest: Path,
    *,
    extra: list[Path] | None = None,
    skip: bool = False,
) -> list[Path]:
    """Copy gitignored support files (e.g. ``.env``) into an ephemeral worktree.

    The list of files to copy is, in order:

    1. ``[tool.daydream.workspace] copy`` from ``source/pyproject.toml`` if set.
    2. Otherwise, the default list (``.env``, ``.env.local``) plus any
       ``.env.*`` siblings -- restricted to gitignored files only.
    3. Any *extra* paths supplied (e.g. via ``--copy``) -- additive.

    Files that do not exist in *source* (or are not regular files) are
    skipped silently.

    Before anything is copied, every entry (de-duplicated, first-occurrence
    order) is validated fail-closed against BOTH *source* and *dest* roots by
    :func:`_resolve_workspace_copy_path`. An absolute/``..`` entry or one that
    resolves outside either root raises :class:`WorkspaceCopyPathError` and
    aborts the whole copy, leaving nothing partially copied.

    Args:
        extra: Optional additional relative paths from CLI flags.
        skip: When True, return ``[]`` immediately (no-op for read-only
            review flows that do not need test fixtures).

    Returns:
        The list of paths actually copied (relative to *source*).
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


# --- Internal helpers --------------------------------------------------------


def _warn_removal_failed(path: Path, exc: GitError, *, kind: str = "worktree") -> None:
    """Warn that cleanup of *path* failed, without raising.

    Best-effort cleanup contract shared by every worktree-teardown path:
    a removal failure must never mask the primary outcome of the run.
    """
    from daydream.agent import console
    from daydream.ui import print_warning

    print_warning(console, f"Failed to remove {kind} {path}: {exc}")




def _dedupe_ordered(entries: Iterable[str | Path]) -> list[Path]:
    """De-duplicate path-like *entries*, preserving first-occurrence order."""
    unique: list[Path] = []
    seen: set[Path] = set()
    for rel in entries:
        rel_path = Path(rel)
        if rel_path in seen:
            continue
        seen.add(rel_path)
        unique.append(rel_path)
    return unique


def _resolve_workspace_copy_path(entry: Path, root: Path, root_label: str) -> None:
    """Validate a workspace-copy entry against *root*, fail-closed.

    Unlike ``daydream/improve/command_contract.py::path_is_confined`` -- which
    walks each path part and rejects symlink edges up-front before its final
    canonical containment check -- this helper relies solely on
    ``resolve(strict=False)`` + ``is_relative_to(root)`` to detect containment
    violations. On a violation it raises a :class:`WorkspaceCopyPathError`
    naming *root_label* instead of returning a bool.

    Raises:
        WorkspaceCopyPathError: If *entry* is absolute or contains ``..``, or
            resolves outside *root*, or cannot be resolved at all.
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


def _resolve_base(source: Path, branch: str | None, base: str | None) -> str:
    """Pick the base branch per the locked resolution rules."""
    if base is not None:
        return base

    if branch is not None and shutil.which("gh") is not None:
        try:
            prs = git_ops.gh_pr_list_for_branch(source, branch)
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
    """Return the configured copy list (pyproject override or defaults).

    The pyproject override -- when present -- replaces the default ``.env*``
    list entirely.  Defaults apply only when ``[tool.daydream.workspace]
    copy`` is missing.
    """
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
    return [p for p in unique if _is_gitignored(source, str(p))]


def _is_gitignored(repo: Path, relative_path: str) -> bool:
    """Return True iff ``git check-ignore`` says *relative_path* is ignored."""
    return git_ops.check_ignore(repo, relative_path)
