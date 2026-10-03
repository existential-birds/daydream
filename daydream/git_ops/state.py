"""Exact index and path snapshots with confined restoration and patch generation."""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from daydream.git_ops import process, queries
from daydream.git_ops.models import (
    GitError,
    GitPathState,
    IndexSnapshot,
    NotAWorktreeError,
    RawIndexSnapshot,
    WorktreeRollbackSnapshot,
)


def snapshot_raw_index(repo: Path) -> RawIndexSnapshot:
    """Capture an index without reducing it to a Git tree."""
    path = queries._snapshot_git_path(repo, "index")
    try:
        return RawIndexSnapshot(path.read_bytes(), stat.S_IMODE(path.stat().st_mode))
    except FileNotFoundError:
        return RawIndexSnapshot(None, None)


def restore_raw_index(repo: Path, snapshot: RawIndexSnapshot) -> None:
    """Restore exact bytes under Git's ordinary exclusive index lock."""
    path = queries._snapshot_git_path(repo, "index")
    lock = path.with_name(path.name + ".lock")
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, snapshot.mode or 0o644)
    except OSError as exc:
        raise GitError("could not acquire index lock for exact restoration") from exc
    try:
        with os.fdopen(descriptor, "wb") as stream:
            if snapshot.content is not None:
                stream.write(snapshot.content)
            stream.flush()
            os.fsync(stream.fileno())
        if snapshot.content is None:
            path.unlink(missing_ok=True)
        else:
            os.replace(lock, path)
    finally:
        lock.unlink(missing_ok=True)


def _write_git_blob(repo: Path, content: bytes) -> str:
    proc = process._run_git(
        repo,
        ["hash-object", "-w", "--stdin"],
        timeout=30,
        capture_bytes=True,
        input_bytes=content,
        error_context=f"git hash-object failed in {repo}",
    )
    oid = os.fsdecode(proc.stdout).strip()
    if not oid:
        raise GitError("git hash-object returned no object id")
    return oid


def _read_git_blob(repo: Path, oid: str) -> bytes:
    proc = process._run_git(
        repo,
        ["cat-file", "blob", oid],
        timeout=30,
        capture_bytes=True,
        error_context=f"git cat-file blob failed in {repo}",
    )
    return proc.stdout


def _snapshot_worktree_path(
    repo: Path,
    path: str,
    *,
    allow_leaf_symlink: bool,
) -> GitPathState:
    queries._require_git_path_confined(repo, path, allow_leaf_symlink=allow_leaf_symlink)
    mode_from_index = _snapshot_git_tree_paths(repo, ["ls-files", "--stage"], [path])[0].mode
    absolute_bytes = os.fsencode(repo) + b"/" + os.fsencode(path)
    try:
        metadata = os.lstat(absolute_bytes)
    except FileNotFoundError:
        return GitPathState(path=path, state="missing", mode=None, digest=None)
    except OSError as exc:
        raise GitError("could not inspect a confined worktree path") from exc

    if mode_from_index == 0o160000 and stat.S_ISDIR(metadata.st_mode):
        nested = repo / path
        try:
            queries.assert_is_worktree(nested)
        except NotAWorktreeError as exc:
            raise GitError("gitlink working tree is unavailable for exact evidence") from exc
        dirty = process._run_git(
            nested,
            ["status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignore-submodules=none"],
            capture_bytes=True,
        )
        if dirty.returncode != 0:
            raise GitError("could not inspect gitlink working tree for exact evidence")
        if dirty.stdout:
            # A commit OID cannot identify additional worktree content. Refuse
            # stale test evidence instead of recursively snapshotting submodules.
            raise GitError("dirty gitlink cannot provide commit-only test evidence")
        oid = queries.head_sha(nested)
        return GitPathState(path=path, state="gitlink", mode=0o160000, digest=oid)
    if stat.S_ISLNK(metadata.st_mode):
        target = os.readlink(absolute_bytes)
        content = target if isinstance(target, bytes) else os.fsencode(target)
        return GitPathState(path=path, state="symlink", mode=0o120000, digest=_write_git_blob(repo, content))
    if stat.S_ISREG(metadata.st_mode):
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(absolute_bytes, flags)
            with os.fdopen(descriptor, "rb") as stream:
                opened = os.fstat(descriptor)
                if not stat.S_ISREG(opened.st_mode):
                    raise GitError("captured worktree path changed type during read")
                content = stream.read()
        except OSError as exc:
            raise GitError("could not read a confined worktree path") from exc
        # Worktree evidence preserves actual permissions, including private
        # owner files. Its identity must not change merely because a formerly
        # untracked path enters the index. Git-mode projection belongs only at
        # the worktree-to-index comparison boundary.
        mode = stat.S_IFREG | stat.S_IMODE(metadata.st_mode)
        return GitPathState(path=path, state="regular", mode=mode, digest=_write_git_blob(repo, content))
    raise GitError("unsupported worktree path type")


def snapshot_untracked_paths(
    repo: Path,
    *,
    include_runtime_artifacts: bool = True,
) -> dict[str, GitPathState]:
    """Capture actual untracked content/type/mode, optionally omitting runtime output."""
    paths = queries.list_untracked(repo, strict=True)
    return {
        path: _snapshot_worktree_path(repo, path, allow_leaf_symlink=True)
        for path in paths
        if include_runtime_artifacts or not queries._is_untracked_runtime_artifact(path)
    }


def snapshot_ignored_owner_paths(repo: Path) -> dict[str, GitPathState]:
    """Protect ignored source/owner files, excluding generated runtime trees.

    These bytes remain private to parent recovery and never enter fixer
    clones. Dependency installations and tool caches are runtime output,
    rather than source state to import or snapshot for each fixer round.
    """
    runtime_directories = {
        ".daydream",
        ".git",
        ".venv",
        "venv",
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".cache",
        ".tox",
        ".nox",
    }
    proc = process._run_git(
        repo,
        ["ls-files", "--others", "--ignored", "--exclude-standard", "-z"],
        capture_bytes=True,
        timeout=30,
        error_context="could not enumerate ignored owner files",
    )
    return {
        path: _snapshot_worktree_path(repo, path, allow_leaf_symlink=True)
        for path in queries._decode_nul_paths(proc.stdout)
        if not runtime_directories.intersection(path.split("/")[:-1])
        and not queries._is_untracked_runtime_artifact(path)
    }


def snapshot_worktree_paths(
    repo: Path,
    paths: Iterable[str],
    *,
    allow_leaf_symlink: bool = False,
) -> tuple[GitPathState, ...]:
    """Capture binary-safe worktree states for exact confined paths."""
    unique = sorted(set(paths), key=queries._path_sort_key)
    return tuple(_snapshot_worktree_path(repo, path, allow_leaf_symlink=allow_leaf_symlink) for path in unique)


def snapshot_worktree_gitlinks(repo: Path) -> tuple[GitPathState, ...]:
    """Capture every tracked gitlink's actual clean checked-out commit."""
    proc = process._run_git(
        repo, ["ls-files", "--stage", "-z"], capture_bytes=True, error_context=f"git ls-files --stage failed in {repo}"
    )
    paths: list[str] = []
    for record in (record for record in proc.stdout.split(b"\0") if record):
        metadata, separator, raw_path = record.partition(b"\t")
        if not separator:
            raise GitError("git returned malformed staged path data")
        mode, _oid, stage = metadata.decode("ascii").split(" ")
        if mode == "160000" and stage == "0":
            paths.append(os.fsdecode(raw_path))
    return snapshot_worktree_paths(repo, paths)


def snapshot_worktree_delta(
    repo: Path,
    ref: str,
    *,
    preexisting_untracked: dict[str, GitPathState],
    preexisting_gitlinks: tuple[GitPathState, ...] = (),
) -> tuple[GitPathState, ...]:
    """Capture source delta plus protected paths, not changing runtime output.

    Explicitly protected paths remain visible regardless of namespace. This
    makes user-file changes invalidate evidence without audit/trace writes
    recursively invalidating the evidence they describe.
    """
    paths = (
        set(queries.changed_paths_z(repo, ref, include_runtime_artifacts=False))
        | set(preexisting_untracked)
        | {state.path for state in preexisting_gitlinks}
    )
    return tuple(
        _snapshot_worktree_path(
            repo,
            path,
            allow_leaf_symlink=path in preexisting_untracked,
        )
        for path in sorted(paths, key=queries._path_sort_key)
    )


def _snapshot_git_tree_paths(
    repo: Path,
    args: list[str],
    paths: Iterable[str],
) -> tuple[GitPathState, ...]:
    unique = sorted(set(paths), key=queries._path_sort_key)
    for path in unique:
        queries._require_git_path_confined(repo, path, allow_leaf_symlink=True)
    if not unique:
        return ()
    proc = process._run_git(
        repo,
        [*args, "-z", "--", *(queries._literal_pathspec(path) for path in unique)],
        capture_bytes=True,
        error_context=f"git tree-state query failed in {repo}",
    )
    found: dict[str, GitPathState] = {}
    for record in (record for record in proc.stdout.split(b"\0") if record):
        metadata, separator, raw_path = record.partition(b"\t")
        if not separator:
            raise GitError("git returned malformed tree-state output")
        path = os.fsdecode(raw_path)
        if path not in unique:
            raise GitError("git returned an unexpected tree-state path")
        fields = metadata.decode("ascii").split(" ")
        if args[0] == "ls-files":  # ls-files --stage: mode oid stage
            if len(fields) != 3:
                raise GitError("git returned malformed index-state metadata")
            mode_text, oid, stage_text = fields
            if stage_text != "0":
                raise GitError("index contains unresolved entries")
        elif len(fields) == 3:  # ls-tree: mode type oid
            mode_text, _object_type, oid = fields
        else:
            raise GitError("git returned malformed tree-state metadata")
        mode = int(mode_text, 8)
        state: Literal["regular", "symlink", "gitlink"]
        if mode == 0o120000:
            state = "symlink"
        elif mode == 0o160000:
            state = "gitlink"
        else:
            state = "regular"
        found[path] = GitPathState(path=path, state=state, mode=mode, digest=oid)
    return tuple(found.get(path, GitPathState(path=path, state="missing", mode=None, digest=None)) for path in unique)


def snapshot_index(repo: Path) -> IndexSnapshot:
    """Capture the complete index tree without changing the worktree."""
    tree = process._run_git(repo, ["write-tree"], timeout=30, error_context=f"git write-tree failed in {repo}")
    changed = process._run_git(
        repo,
        ["diff", "--cached", "--name-only", "-z", "HEAD"],
        capture_bytes=True,
        error_context=f"git diff --cached failed in {repo}",
    )
    paths = tuple(sorted(set(queries._decode_nul_paths(changed.stdout)), key=queries._path_sort_key))
    return IndexSnapshot(tree_sha=tree.stdout.strip(), paths=paths)


def snapshot_index_paths(repo: Path, paths: Iterable[str]) -> tuple[GitPathState, ...]:
    """Capture exact path states from the current index."""
    return _snapshot_git_tree_paths(repo, ["ls-files", "--stage"], paths)


def snapshot_commit_paths(repo: Path, ref: str, paths: Iterable[str]) -> tuple[GitPathState, ...]:
    """Capture exact path states from a commit/tree ref."""
    return _snapshot_git_tree_paths(repo, ["ls-tree", ref], paths)


def tree_key(states: Iterable[GitPathState]) -> str:
    """Hash a canonical binary-safe sequence of content-only path states."""
    ordered = sorted(states, key=lambda state: queries._path_sort_key(state.path))
    digest = hashlib.sha256()
    for state in ordered:
        path = os.fsencode(state.path)
        state_bytes = state.state.encode("ascii")
        mode = b"-" if state.mode is None else format(state.mode, "o").encode("ascii")
        object_digest = b"-" if state.digest is None else state.digest.encode("ascii")
        for field_value in (path, state_bytes, mode, object_digest):
            digest.update(len(field_value).to_bytes(8, "big"))
            digest.update(field_value)
    return digest.hexdigest()


def _remove_confined_leaf(repo: Path, path: str) -> None:
    queries._require_git_path_confined(repo, path, allow_leaf_symlink=True)
    absolute = os.fsencode(repo) + b"/" + os.fsencode(path)
    try:
        metadata = os.lstat(absolute)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise GitError("could not inspect path before confined removal") from exc
    try:
        if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
            shutil.rmtree(absolute)
        else:
            os.unlink(absolute)
    except OSError as exc:
        raise GitError("could not remove confined worktree path") from exc


def _restore_path_state(
    repo: Path,
    state: GitPathState,
    *,
    allow_leaf_type_replacement: bool,
) -> None:
    if state.state == "missing":
        _remove_confined_leaf(repo, state.path)
        return
    queries._require_git_path_confined(
        repo,
        state.path,
        allow_leaf_symlink=allow_leaf_type_replacement or state.state == "symlink",
    )
    if state.state == "gitlink":
        nested = _preflight_gitlink_restore(repo, state)
        assert state.digest is not None
        process._run_git(
            nested,
            ["checkout", "--detach", state.digest],
            timeout=30,
            retries=0,
            error_context=f"could not restore gitlink {state.path!r} to its captured commit",
        )
        if queries.head_sha(nested) != state.digest:
            raise GitError(f"gitlink {state.path!r} did not reach its captured commit")
        _preflight_gitlink_restore(repo, state)
        return
    if state.digest is None or state.mode is None:
        raise GitError("restorable path state is incomplete")
    content = _read_git_blob(repo, state.digest)
    _remove_confined_leaf(repo, state.path)
    absolute = os.fsencode(repo) + b"/" + os.fsencode(state.path)
    parent = os.path.dirname(absolute)
    try:
        os.makedirs(parent, exist_ok=True)
        if state.state == "symlink":
            os.symlink(content, absolute)
            return
        descriptor = os.open(absolute, os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat.S_IMODE(state.mode))
        try:
            view = memoryview(content)
            while view:
                written = os.write(descriptor, view)
                view = view[written:]
        finally:
            os.close(descriptor)
        os.chmod(absolute, stat.S_IMODE(state.mode))
    except OSError as exc:
        raise GitError("could not restore confined worktree path") from exc


def _preflight_gitlink_restore(repo: Path, state: GitPathState) -> Path:
    """Prove a gitlink can be restored without discarding nested user state."""
    if state.digest is None or state.mode != 0o160000:
        raise GitError("restorable gitlink state is incomplete")
    queries._require_git_path_confined(repo, state.path, allow_leaf_symlink=False)
    nested = repo / state.path
    try:
        queries.assert_is_worktree(nested)
    except NotAWorktreeError as exc:
        raise GitError("gitlink working tree is unavailable for exact restoration") from exc
    dirty = process._run_git(
        nested,
        ["status", "--porcelain=v1", "-z", "--untracked-files=all", "--ignore-submodules=none"],
        capture_bytes=True,
    )
    if dirty.returncode != 0:
        raise GitError("could not inspect gitlink before exact restoration")
    if dirty.stdout:
        raise GitError("dirty gitlink cannot be restored without discarding nested user state")
    target = process._run_git(
        nested,
        ["cat-file", "-e", f"{state.digest}^{{commit}}"],
        timeout=30,
        retries=0,
    )
    if target.returncode != 0:
        raise GitError(f"captured gitlink commit is unavailable for {state.path!r}")
    return nested


def restore_group_worktree_from_snapshot(
    repo: Path,
    snapshot: WorktreeRollbackSnapshot,
    paths: Iterable[str],
    *,
    allow_leaf_type_replacement: bool = False,
) -> None:
    """Restore requested group paths without mutating the parent index."""
    requested = sorted(set(paths), key=queries._path_sort_key)
    tracked = {state.path: state for state in snapshot.path_states}
    committed = {
        state.path: state
        for state in snapshot_commit_paths(
            repo,
            snapshot.ref,
            [path for path in requested if path not in tracked and path not in snapshot.untracked],
        )
    }
    restore_states: list[tuple[GitPathState, bool]] = []
    for path in requested:
        if path in snapshot.untracked:
            state = snapshot.untracked[path]
            replace_type = True
        elif path in tracked:
            state = tracked[path]
            replace_type = False
        else:
            state = committed[path]
            replace_type = state.state == "missing"
        restore_states.append((state, replace_type))

    # Preflight every nested repository before changing any worktree path. A
    # dirty or unavailable gitlink is a fail-closed condition, never grounds
    # for a forced checkout that could destroy user content.
    for state, _replace_type in restore_states:
        if state.state == "gitlink":
            _preflight_gitlink_restore(repo, state)
    for state, replace_type in restore_states:
        _restore_path_state(
            repo,
            state,
            allow_leaf_type_replacement=replace_type or allow_leaf_type_replacement,
        )


def restore_group_from_snapshot(
    repo: Path,
    snapshot: WorktreeRollbackSnapshot,
    paths: Iterable[str],
) -> None:
    """Restore every requested group path and the supplied complete index."""
    try:
        restore_group_worktree_from_snapshot(repo, snapshot, paths)
    finally:
        restore_index(repo, snapshot.index)


def copy_worktree_paths(source: Path, destination: Path, paths: Iterable[str]) -> None:
    """Import exact file/type/mode states without staging destination paths.

    Transfer captured blobs into the destination object store first, so no
    shared Git storage or surviving disposable checkout is required. Validate
    every path before the first destination write.
    """
    states = snapshot_worktree_paths(source, paths, allow_leaf_symlink=True)
    for state in states:
        queries._require_git_path_confined(destination, state.path, allow_leaf_symlink=True)
        if state.state == "gitlink":
            raise GitError("isolated fixer cannot publish a gitlink mutation")
        if state.digest is not None:
            if _write_git_blob(destination, _read_git_blob(source, state.digest)) != state.digest:
                raise GitError("isolated fixer blob transfer changed identity")
    for state in states:
        _restore_path_state(destination, state, allow_leaf_type_replacement=True)


def restore_index(repo: Path, snapshot: IndexSnapshot) -> None:
    """Restore a complete index tree without modifying the worktree."""
    process._run_git(
        repo, ["read-tree", snapshot.tree_sha], timeout=30, retries=0, error_context=f"git read-tree failed in {repo}"
    )


def build_recommended_patch_strict(
    repo: Path,
    base_ref: str,
    retained_paths: Iterable[str],
) -> bytes:
    """Build deterministic binary-capable presentation output for exact paths."""
    unique = sorted(set(retained_paths), key=queries._path_sort_key)
    if not unique:
        return b""
    base_states = {state.path: state for state in snapshot_commit_paths(repo, base_ref, unique)}
    current_states = {state.path: state for state in snapshot_worktree_paths(repo, unique)}
    chunks: list[bytes] = []
    for path in unique:
        if base_states[path].state == "missing" and current_states[path].state != "missing":
            proc = process._run_git(
                repo,
                ["diff", "--no-index", "--binary", "--full-index", "--", "/dev/null", path],
                timeout=30,
                capture_bytes=True,
            )
            if proc.returncode not in {0, 1}:
                raise GitError(f"git diff --no-index --binary failed in {repo}: {os.fsdecode(proc.stderr).strip()}")
        else:
            proc = process._run_git(
                repo,
                [
                    "diff",
                    "--binary",
                    "--full-index",
                    "--no-ext-diff",
                    base_ref,
                    "--",
                    queries._literal_pathspec(path),
                ],
                timeout=30,
                capture_bytes=True,
            )
            process._require_ok(proc, f"git diff --binary failed in {repo}")
        chunks.append(proc.stdout)
    return b"".join(chunks)
