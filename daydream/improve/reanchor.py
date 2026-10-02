"""Discovery and lock-aware pruning of source-owned re-anchor worktrees."""

from __future__ import annotations

import re
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
from daydream.workspace_legacy import (
    _OPERATIONAL_LOCK_STALE_AFTER_S,
    _legacy_operational_root,
    _prune_stale_locked_worktrees,
)

# Re-anchor worktree directory names are built from the run session id, so only
# a filesystem-safe run id may reach the path; anything else falls back to an
# anchor-derived name.
_SAFE_DIRNAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")


# Re-anchor worktree dirnames end in this suffix; the prune pass and the
# re-anchor write share it so a rename can never silently stop pruning.
_REANCHOR_DIR_SUFFIX = "-reanchor"


def _iter_reanchor_worktrees(roots: Iterable[Path]) -> list[Path]:
    """Find real re-anchor directories; reject ambiguous names across roots and skip symlinks."""
    found = (
        path
        for root in roots
        for path in root.glob(f"*{_REANCHOR_DIR_SUFFIX}")
        if not path.is_symlink() and path.is_dir()
    )
    paths = sorted(found, key=lambda path: (path.name, path.as_posix()))
    if len({path.name for path in paths}) != len(paths):
        raise git_ops.GitError("ambiguous re-anchor worktree name across storage roots")
    return paths


def _public_reanchor_roots(repo: Path, owner: PrivateWorkspaceOwner | None) -> tuple[Path, Path]:
    """Resolve the operational root once while retaining legacy discovery."""
    if owner is None:
        owner = resolve_private_workspace_owner(repo, locations=private_root_locations())
    else:
        validate_private_workspace_owner(owner, source=owner.source, repo=repo)
    operational = operational_worktree_path(owner)
    legacy = _legacy_operational_root(owner.source, "worktrees", label="legacy re-anchor root")
    validate_private_directory(operational, label="operational re-anchor root", allow_absent=True)
    return operational, legacy


def prune_stale_reanchor_worktrees(repo: Path, *, private_workspace_owner: PrivateWorkspaceOwner | None = None) -> int:
    """Prune private and legacy re-anchor worktrees with the shared lock policy.

    Live locks survive; stale removals unlock first and tolerate individual Git
    failures. Both namespaces remain discoverable.
    """
    return _prune_stale_locked_worktrees(
        repo,
        _iter_reanchor_worktrees(_public_reanchor_roots(repo, private_workspace_owner)),
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
    """Remove one safe re-anchor name from private or legacy storage.

    Reject unsafe names before storage access and ambiguous roots before mutation.
    Capture the Markdown count before removal for reporting, never as a gate.
    """
    if _SAFE_DIRNAME.fullmatch(name) is None:
        return NamedPruneOutcome(PRUNE_UNSAFE_NAME)
    if not name.endswith(_REANCHOR_DIR_SUFFIX):
        return NamedPruneOutcome(PRUNE_NOT_REANCHOR)
    roots = _public_reanchor_roots(repo, private_workspace_owner)
    found = [root / name for root in roots if (root / name).exists() or (root / name).is_symlink()]
    if not found:
        return NamedPruneOutcome(PRUNE_NOT_FOUND)
    path = found[0]
    if len(found) != 1 or path.is_symlink() or not path.is_dir():
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
    return _iter_reanchor_worktrees(_public_reanchor_roots(repo, private_workspace_owner))
