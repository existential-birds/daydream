"""Disposable fixer repositories and a join-before-publication write boundary."""

from __future__ import annotations

import os
import stat
import tempfile
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path

from daydream import git_ops
from daydream.fix_footprint import AuthorizedFixFootprint
from daydream.git_ops import GitPathState, WorktreeRollbackSnapshot
from daydream.workspace import WorkContext


class FixIsolationRound:
    """Keep every fix group away from shared source and index state.

    This is not a filesystem sandbox: a shell may explicitly address the
    original checkout. Consequently the parent is restored after *all* agents
    have joined, before any isolated group's authorized result is imported.
    Failed or cancelled groups never publish their disposable checkout.
    """

    def __init__(
        self, work: WorkContext, footprint: AuthorizedFixFootprint, *, capture_discarded_edits: bool = False,
    ) -> None:
        self.work = work
        self.footprint = footprint
        self._temporary = tempfile.TemporaryDirectory(prefix="daydream-fix-")
        self.root = Path(self._temporary.name).resolve()
        self.template = self.root / "baseline"
        self._published: list[tuple[Path, frozenset[str]]] = []
        self._counter = 0
        self.capture_discarded_edits = capture_discarded_edits
        self.discarded_edits: list[tuple[str, str]] = []
        try:
            self._raw_index = git_ops.snapshot_raw_index(work.repo)
            untracked = git_ops.snapshot_untracked_paths(work.repo, include_runtime_artifacts=False)
            untracked.update(git_ops.snapshot_ignored_owner_paths(work.repo))
            paths = {*git_ops.ls_files(work.repo, strict=True), *git_ops.ls_tree_files(work.repo, "HEAD", strict=True)}
            self.parent = WorktreeRollbackSnapshot(
                ref=git_ops.head_sha(work.repo),
                index=git_ops.snapshot_index(work.repo),
                path_states=git_ops.snapshot_worktree_paths(work.repo, paths, allow_leaf_symlink=True),
                untracked=untracked,
            )
            self._directories: dict[str, int] = {}
            for path in (*paths, *untracked):
                for ancestor in (work.repo / path).parents:
                    if ancestor == work.repo:
                        break
                    relative = str(ancestor.relative_to(work.repo))
                    try:
                        metadata = ancestor.lstat()
                    except FileNotFoundError:
                        continue
                    if not stat.S_ISDIR(metadata.st_mode):
                        raise git_ops.GitError("fix baseline ancestor is not a real directory")
                    self._directories[relative] = stat.S_IMODE(metadata.st_mode)
            git_ops.prepare_independent_snapshot(work.repo, self.template, include_untracked=True)
        except BaseException:
            self._temporary.cleanup()
            raise

    def create_group(self, paths: Iterable[str] | None = None) -> tuple[WorkContext, WorktreeRollbackSnapshot]:
        """Create an independently writable repository at the round baseline."""
        self._counter += 1
        destination = self.root / str(self._counter)
        git_ops.prepare_independent_snapshot(self.template, destination, include_untracked=True)
        snapshot = WorktreeRollbackSnapshot(
            ref=git_ops.head_sha(destination),
            index=git_ops.snapshot_index(destination),
            path_states=git_ops.snapshot_worktree_paths(
                destination, paths if paths is not None else self.footprint.run_allowed_paths,
                allow_leaf_symlink=True,
            ),
            untracked=git_ops.snapshot_untracked_paths(destination, include_runtime_artifacts=False),
        )
        return replace(self.work, repo=destination, is_ephemeral=True), snapshot

    def retain_group(self, repo: Path, paths: frozenset[str], snapshot: WorktreeRollbackSnapshot) -> None:
        """Queue successful assigned paths; audit everything else as discarded."""
        protected = set(self.parent.untracked)
        admitted = paths - protected
        self.audit_group(repo, snapshot, admitted=admitted)
        self._published.append((repo, admitted))

    def audit_group(
        self, repo: Path, snapshot: WorktreeRollbackSnapshot, *, admitted: frozenset[str] = frozenset(),
    ) -> None:
        """Record discarded writes even when a group never succeeds."""
        baseline = {
            state.path: state for state in (
                *self.parent.path_states, *snapshot.path_states, *snapshot.untracked.values(),
            )
        }
        changed = git_ops.changed_paths_z(repo, snapshot.ref, include_runtime_artifacts=False)
        for state in git_ops.snapshot_worktree_paths(repo, changed, allow_leaf_symlink=True):
            path = state.path
            if path not in admitted and state != baseline.get(path, GitPathState(path, "missing", None, None)):
                self._audit(path, "discarded isolated fixer write outside its assigned footprint")
                self._capture_discarded(repo, snapshot.ref, path)

    def _capture_discarded(self, repo: Path, ref: str, path: str) -> None:
        if not self.capture_discarded_edits or path in self.parent.untracked:
            return
        if path in self.footprint.run_allowed_paths:
            return
        try:
            patch = git_ops.diff_worktree_against(repo, ref, [path])
        except Exception:
            return  # Optional issue evidence never blocks recovery.
        if patch and (path, patch) not in self.discarded_edits:
            self.discarded_edits.append((path, patch))

    def _audit(self, path: str, reason: str) -> None:
        self.footprint.record_git_event(
            action="restore", path=path, origin="guard", phase="fix_group",
            round_number=None, reason=reason,
        )

    def restore_parent(self) -> None:
        """Restore direct parent writes without touching host runtime evidence."""
        try:
            self._restore_parent_worktree()
        finally:
            git_ops.restore_raw_index(self.work.repo, self._raw_index)

    def _restore_parent_worktree(self) -> None:
        # Rebuild original parent directories without following replacement
        # symlinks. Otherwise a shell replacing pkg/ would make leaf recovery
        # fail confinement or write into the link's external destination.
        for relative, mode in sorted(self._directories.items(), key=lambda item: (item[0].count("/"), item[0])):
            directory = self.work.repo / relative
            try:
                metadata = directory.lstat()
            except FileNotFoundError:
                metadata = None
            if metadata is not None and not stat.S_ISDIR(metadata.st_mode):
                directory.unlink()
                metadata = None
                self._audit(relative, "removed replacement of a parent directory without following it")
            if metadata is None:
                directory.mkdir(mode=mode)
            os.chmod(directory, mode)
        paths = set(git_ops.changed_paths_z(
            self.work.repo, self.parent.ref, include_runtime_artifacts=False,
        )) | set(self.parent.untracked) | {state.path for state in self.parent.path_states}
        paths.update(git_ops.snapshot_ignored_owner_paths(self.work.repo))
        baseline = {state.path: state for state in (*self.parent.path_states, *self.parent.untracked.values())}
        # New paths are absent at the baseline, including files staged by an
        # escaping shell. Every protected untracked path is inspected even when
        # the fixer deleted or staged it.
        current = git_ops.snapshot_worktree_paths(self.work.repo, paths, allow_leaf_symlink=True)
        changed = {
            state.path for state in current
            if state != baseline.get(state.path, GitPathState(state.path, "missing", None, None))
        }
        if changed:
            for path in changed:
                self._capture_discarded(self.work.repo, self.parent.ref, path)
            git_ops.restore_group_worktree_from_snapshot(
                self.work.repo, self.parent, changed, allow_leaf_type_replacement=True,
            )
            for path in sorted(changed):
                self._audit(path, "restored a direct parent write before importing isolated fixes")

    def publish(self) -> None:
        """Import only retained assigned states, after writers have all joined."""
        for source, paths in self._published:
            git_ops.copy_worktree_paths(source, self.work.repo, paths)

    def close(self) -> None:
        """Remove only the temporary repositories owned by this round."""
        self._temporary.cleanup()
