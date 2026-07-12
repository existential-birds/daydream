"""WorkContext abstraction for in-place vs ephemeral worktree execution.

This module is the single entry point daydream uses to *open* the directory
it operates on for a single run.  Two modes are supported:

* **In-place** -- daydream operates on the user's checked-out worktree.
* **Ephemeral** -- daydream creates a detached worktree under
  ``<source>/.daydream/worktrees/<run_id>`` and removes it on exit.

The resolution rules and ordering live in :func:`open_workspace` and are
deliberately fixed (matching the design captured in
``docs/plans/2026-04-30-worktree-isolation-and-mode-consolidation.md``).

The module shells out via :mod:`daydream.git_ops` only.
"""

from __future__ import annotations

import logging
import secrets
import shutil
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import AsyncIterator

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
            ``HEAD``, captured at workspace open time.
        head_branch: Branch name at :attr:`repo`'s ``HEAD``, or ``None`` when
            ``HEAD`` is detached (e.g. ephemeral worktrees).
        head_sha: Full SHA of :attr:`repo`'s ``HEAD``.
        is_ephemeral: True when :attr:`repo` is an ephemeral worktree.
        run_id: ``<UTC YYYYMMDDHHMMSS>-<hex8>`` identifier used for the
            ephemeral path and intent files.
    """

    repo: Path
    source: Path
    base_branch: str
    base_sha: str
    head_branch: str | None
    head_sha: str
    is_ephemeral: bool
    run_id: str

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
            raise BranchNotFoundError(
                f"base ref '{base_branch}' not found in {repo}"
            )

        base_sha = git_ops.merge_base(repo, base_branch)
        if base_sha is None:
            raise BranchNotFoundError(
                f"could not resolve merge-base for '{base_branch}' in {repo}"
            )

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
                # Best-effort cleanup -- never let removal failure mask the
                # primary outcome of the run.
                from daydream.agent import console
                from daydream.ui import print_warning

                print_warning(
                    console,
                    f"Failed to remove ephemeral worktree {worktree_path}: {exc}",
                )


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

    copied: list[Path] = []
    seen: set[Path] = set()
    for rel in entries:
        rel_path = Path(rel)
        if rel_path in seen:
            continue
        seen.add(rel_path)

        src_file = source / rel_path
        if not src_file.is_file():
            continue

        dest_file = dest / rel_path
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dest_file)
        copied.append(rel_path)

    return copied


# --- Internal helpers --------------------------------------------------------


def _make_run_id() -> str:
    """Return a unique ``<UTC YYYYMMDDHHMMSS>-<hex8>`` identifier."""
    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    return f"{timestamp}-{secrets.token_hex(4)}"


def _resolve_ref(source: Path, branch: str | None) -> str:
    """Resolve the git ref to check out in the ephemeral worktree."""
    if branch is None:
        return git_ops.head_sha(source)

    if not git_ops.branch_exists(source, branch):
        raise BranchNotFoundError(
            f"branch '{branch}' not found locally or on origin in {source}"
        )

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
            f"origin/{branch}.\nreviewing origin/{branch}.",
        )

    return f"origin/{branch}"


def _resolve_base(source: Path, branch: str | None, base: str | None) -> str:
    """Pick the base branch per the locked resolution rules."""
    if base is not None:
        return base

    if branch is not None and shutil.which("gh") is not None:
        prs = git_ops.gh_pr_list_for_branch(source, branch)
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

    # De-duplicate while preserving order.
    seen: set[str] = set()
    ordered: list[str] = []
    for name in candidates:
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)

    return [Path(name) for name in ordered if _is_gitignored(source, name)]


def _is_gitignored(repo: Path, relative_path: str) -> bool:
    """Return True iff ``git check-ignore`` says *relative_path* is ignored."""
    return git_ops.check_ignore(repo, relative_path)
