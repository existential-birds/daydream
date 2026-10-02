"""Discovery and lock-aware pruning of source-owned re-anchor worktrees."""

from __future__ import annotations

import re
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from daydream import git_ops
from daydream.artifact_visibility import (
    PrivateWorkspaceOwner,
    operational_worktree_path,
    private_root_locations,
    resolve_private_workspace_owner,
    validate_private_directory,
    validate_private_workspace_owner,
)
from daydream.workspace import reject_public_operational_storage

_OPERATIONAL_LOCK_STALE_AFTER_S = 24 * 3600


def _prune_stale_locked_worktrees(
    repo: Path,
    paths: Iterable[Path],
    *,
    stale_after_s: int,
) -> int:
    """Remove unlocked or stale-locked worktrees, tolerating individual Git failures.

    Locks no older than stale_after_s belong to active runs and remain untouched.
    Removal unlocks first; return the number successfully removed.
    """
    removed = 0
    for path in paths:
        try:
            locked_at = git_ops.worktree_lock_mtime(path)
            if locked_at is not None and time.time() - locked_at <= stale_after_s:
                # Live worktree (lock age near zero): never unlock or remove
                # it, so a concurrent run mid-write is not destroyed.
                continue
            git_ops.worktree_remove_unlocked(repo, path)
        except git_ops.GitError:
            continue
        removed += 1
    return removed


# Re-anchor worktree directory names are built from the run session id, so only
# a filesystem-safe run id may reach the path; anything else falls back to an
# anchor-derived name.
_SAFE_DIRNAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")


# Re-anchor worktree dirnames end in this suffix; the prune pass and the
# re-anchor write share it so a rename can never silently stop pruning.
_REANCHOR_DIR_SUFFIX = "-reanchor"


def _iter_reanchor_worktrees(root: Path) -> list[Path]:
    """List real current re-anchor directories without following linked entries."""
    return sorted(path for path in root.glob(f"*{_REANCHOR_DIR_SUFFIX}")
                  if not path.is_symlink() and path.is_dir())


def _private_reanchor_root(repo: Path, owner: PrivateWorkspaceOwner | None) -> Path:
    if owner is None:
        owner = resolve_private_workspace_owner(repo, locations=private_root_locations())
    else:
        validate_private_workspace_owner(owner, source=owner.source, repo=repo)
    reject_public_operational_storage(owner.source)
    root = operational_worktree_path(owner)
    validate_private_directory(root, label="operational re-anchor root", allow_absent=True)
    return root


def prune_stale_reanchor_worktrees(repo: Path, *, private_workspace_owner: PrivateWorkspaceOwner | None = None) -> int:
    """Prune current private re-anchors, preserving live locks and tolerating individual Git failures."""
    return _prune_stale_locked_worktrees(
        repo,
        _iter_reanchor_worktrees(_private_reanchor_root(repo, private_workspace_owner)),
        stale_after_s=_OPERATIONAL_LOCK_STALE_AFTER_S,
    )


# Verdicts for a named prune of a single re-anchor worktree. Distinct outcomes
# let the caller report precisely what happened instead of a bare success/fail.
PRUNE_REMOVED = "removed"
PRUNE_NOT_FOUND = "not-found"
PRUNE_NOT_REANCHOR = "not-reanchor"
PRUNE_UNSAFE_NAME = "unsafe-name"
PRUNE_GIT_FAILURE = "git-failure"


@dataclass(frozen=True)
class NamedPruneOutcome:
    """Named-prune verdict with a best-effort plan count for operator notices."""

    verdict: str
    plan_count: int = 0


def prune_named_reanchor_worktree(
    repo: Path, name: str, *, private_workspace_owner: PrivateWorkspaceOwner | None = None
) -> NamedPruneOutcome:
    """Remove a safely named private re-anchor and report its Markdown plan count."""
    if _SAFE_DIRNAME.fullmatch(name) is None:
        return NamedPruneOutcome(PRUNE_UNSAFE_NAME)
    if not name.endswith(_REANCHOR_DIR_SUFFIX):
        return NamedPruneOutcome(PRUNE_NOT_REANCHOR)
    path = _private_reanchor_root(repo, private_workspace_owner) / name
    if not path.exists() and not path.is_symlink():
        return NamedPruneOutcome(PRUNE_NOT_FOUND)
    if path.is_symlink() or not path.is_dir():
        return NamedPruneOutcome(PRUNE_GIT_FAILURE)
    plans = path / "daydream_plans"
    if plans.is_dir():
        plan_count = sum(1 for _ in plans.glob("*.md"))
    else:
        plan_count = 0
    try:
        git_ops.worktree_remove(repo, path, force=True)
    except git_ops.GitError:
        return NamedPruneOutcome(PRUNE_GIT_FAILURE, plan_count)
    return NamedPruneOutcome(PRUNE_REMOVED, plan_count)


def list_reanchor_worktrees(repo: Path, *, private_workspace_owner: PrivateWorkspaceOwner | None = None) -> list[Path]:
    """List the same real, unambiguous directories considered by automatic pruning."""
    return _iter_reanchor_worktrees(_private_reanchor_root(repo, private_workspace_owner))
