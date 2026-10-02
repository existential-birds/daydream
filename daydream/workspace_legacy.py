"""Migrate legacy operational worktrees and prune abandoned locked worktrees."""

import logging
import re
import shutil
import stat
import time
from contextlib import suppress
from pathlib import Path
from typing import Iterable

from daydream import git_ops
from daydream.artifact_visibility import (
    ArtifactVisibilityError,
    PrivateWorkspaceOwner,
    operational_worktree_path,
    operational_worktree_root,
)
from daydream.json_utils import _fsync_directory

_logger = logging.getLogger(__name__)
_LEGACY_REANCHOR_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}-reanchor$")
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


def _legacy_operational_root(
    source: Path,
    name: str,
    *,
    label: str = "legacy operational namespace",
) -> Path:
    """Return one legacy root after no-follow lexical directory validation."""
    if name not in {"worktrees", "audit"}:
        raise ArtifactVisibilityError(f"{label} name is invalid")
    ancestor = source / ".daydream"
    root = ancestor / name
    for path, description in ((ancestor, f"{label} ancestor"), (root, label)):
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            return root
        except OSError as exc:
            raise ArtifactVisibilityError(f"{label} is inaccessible") from exc
        if not stat.S_ISDIR(metadata.st_mode):
            raise ArtifactVisibilityError(f"{description} must be a real directory")
    return root


def _retire_legacy_operational_worktrees(
    source: Path,
    owner: PrivateWorkspaceOwner,
) -> None:
    """Preflight all legacy entries, then move recognized worktrees or retire residue.

    Never follow symlinks. A live lock, different Git owner, failed probe, or
    registry-listed broken worktree aborts before mutation. Unlocked reanchor
    worktrees move to the private owner; stale/audit worktrees are removed.
    Unknown residue is pruned with warnings and empty legacy roots are removed.
    """
    actions: list[tuple[str, Path, Path | None]] = []
    destinations: set[Path] = set()
    for root, kind in (
        (_legacy_operational_root(source, "worktrees"), "reanchor"),
        (_legacy_operational_root(source, "audit"), "audit"),
    ):
        if not root.exists():
            continue
        try:
            entries = tuple(root.iterdir())
        except OSError as exc:
            raise ArtifactVisibilityError("legacy operational namespace is inaccessible") from exc
        for entry in entries:
            try:
                metadata = entry.lstat()
            except OSError as exc:
                raise ArtifactVisibilityError("legacy operational entry is inaccessible") from exc
            if not stat.S_ISDIR(metadata.st_mode):
                # lstat never follows links; non-directories cannot be worktrees.
                actions.append(("retire-entry", entry, None))
                continue
            # Probe ownership and lock before classifying by name: an unfamiliar
            # name can still belong to a live worktree holding operator changes.
            try:
                git_ops.assert_is_worktree(entry)
                if git_ops.git_common_dir(entry) != owner.git_common_dir:
                    raise ArtifactVisibilityError(
                        "legacy operational worktree has different Git ownership"
                    )
                locked_at = git_ops.worktree_lock_mtime(entry)
            except ArtifactVisibilityError:
                raise
            except git_ops.NotAWorktreeError as exc:
                # A broken .git chain can hide a registered worktree. Require
                # registry absence before treating the directory as residue.
                if git_ops.registered_worktree_containing(source, entry) is not None:
                    raise ArtifactVisibilityError(
                        "legacy operational entry is registry-listed but unprobeable"
                    ) from exc
                _logger.warning(
                    "retiring unrecognized legacy operational entry %s (%s)",
                    entry,
                    exc,
                )
                actions.append(("retire-entry", entry, None))
                continue
            except git_ops.GitError as exc:
                raise ArtifactVisibilityError(
                    "legacy operational worktree could not be probed safely"
                ) from exc
            if locked_at is not None and time.time() - locked_at <= _OPERATIONAL_LOCK_STALE_AFTER_S:
                # Live and locked: a concurrent run is mid-write. Fail closed.
                raise ArtifactVisibilityError("legacy operational worktree is live and locked")
            if kind == "reanchor" and _LEGACY_REANCHOR_NAME.fullmatch(entry.name) is not None and locked_at is None:
                target = operational_worktree_path(owner) / entry.name
                if target in destinations or target.exists() or target.is_symlink():
                    raise ArtifactVisibilityError("legacy operational migration destination is occupied")
                destinations.add(target)
                actions.append(("move", entry, target))
            else:
                actions.append(("retire-worktree", entry, None))

    if any(action == "move" for action, _, _ in actions):
        operational_worktree_root(owner)
    retired_worktrees = [entry for action, entry, _ in actions if action == "retire-worktree"]
    if retired_worktrees and _prune_stale_locked_worktrees(
        source,
        retired_worktrees,
        stale_after_s=_OPERATIONAL_LOCK_STALE_AFTER_S,
    ) != len(retired_worktrees):
        raise ArtifactVisibilityError("legacy operational worktree could not be retired")
    for action, entry, action_destination in actions:
        if action == "move":
            assert action_destination is not None
            git_ops.worktree_move(source, entry, action_destination)
        elif action == "retire-entry":
            _logger.warning("retiring unrecognized legacy operational entry %s", entry)
            if entry.is_symlink() or not entry.is_dir():
                entry.unlink(missing_ok=True)
            else:
                shutil.rmtree(entry, ignore_errors=True)
            if entry.exists() or entry.is_symlink():
                # Leave partial removal for the session's visibility gate to
                # refuse with a diagnostic naming the residue.
                _logger.warning(
                    "residue remains at %s after retirement attempt; the next "
                    "session open will refuse it naming the leftover",
                    entry,
                )
    _remove_emptied_legacy_roots(source)


def _remove_emptied_legacy_roots(source: Path) -> None:
    """Remove empty legacy roots without following links; leave residue for session validation."""
    for name in ("worktrees", "audit"):
        root = source / ".daydream" / name
        try:
            metadata = root.lstat()
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ArtifactVisibilityError("legacy operational namespace is inaccessible") from exc
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            continue
        try:
            root.rmdir()
        except OSError:
            # Residue remains: leave the root for _validate_legacy_public to
            # refuse with its actionable diagnostic.
            continue
        with suppress(OSError):
            _fsync_directory(root.parent)
