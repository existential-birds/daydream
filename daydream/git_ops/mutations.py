"""Explicit Git changes: refs, clones, worktrees, staging, commits, and pushes."""

from __future__ import annotations

import base64
import os
import re
import stat
from collections.abc import Iterable
from pathlib import Path

from daydream.git_ops import process, queries, state
from daydream.git_ops.models import GitError

# --- Mutating ----------------------------------------------------------------


def fetch(repo: Path, remote: str = "origin") -> None:
    """Fetch the named remote once; propagate failures without retrying mutations."""
    process._run_git(
        repo, ["fetch", remote], timeout=30, retries=0, error_context=f"git fetch {remote} failed in {repo}"
    )


def remove_remote(repo: Path, remote: str = "origin") -> None:
    """Remove a configured remote once; missing remotes and other failures raise."""
    process._run_git(
        repo,
        ["remote", "remove", remote],
        timeout=10,
        retries=0,
        error_context=f"git remote remove {remote} failed in {repo}",
    )


_OBJECT_ID_RE = re.compile(r"[0-9a-fA-F]{40}")


def _validate_ref_oid_pair(ref: str, oid: str) -> None:
    """Require a full OID and a ref safe for ``update-ref --stdin``.

    Reject leading dashes, whitespace/control characters, and a literal lowercase
    ``.lock`` final suffix. Case variants such as ``topic.LOCK`` remain valid.
    """
    if _OBJECT_ID_RE.fullmatch(oid) is None:
        raise GitError(f"invalid OID: {oid}")
    if ref.startswith("-"):
        raise GitError(f"invalid ref name: {ref}")
    if ref.rsplit("/", 1)[-1].endswith(".lock"):
        raise GitError(f"invalid ref name: {ref}")
    if any(ord(ch) <= 0x20 or ord(ch) == 0x7F for ch in ref):
        raise GitError(f"invalid ref name: {ref}")


def update_refs(repo: Path, ref_oids: dict[str, str]) -> None:
    """Validate all ref/OID pairs, then write one atomic ``update-ref --stdin`` batch.

    Git performs final ref-format validation for the whole transaction. Empty input is
    a no-op; failed or timed-out writes are never retried.
    """
    if not ref_oids:
        return
    for ref, oid in ref_oids.items():
        _validate_ref_oid_pair(ref, oid)
    lines = "".join(f"update {ref} {oid}\n" for ref, oid in ref_oids.items())
    process._run_git(
        repo,
        ["update-ref", "--stdin"],
        timeout=30,
        retries=0,
        input_text=lines,
        error_context=f"git update-ref --stdin failed in {repo}",
    )


def apply_staged_patch(repo: Path, patch: bytes) -> None:
    """Apply binary patch bytes to the index through stdin, without retrying writes."""
    process._run_git(
        repo,
        ["apply", "--cached", "--binary"],
        timeout=30,
        retries=0,
        capture_bytes=True,
        input_bytes=patch,
        error_context=f"git apply --cached --binary failed in {repo}",
    )


def worktree_patch_applies(repo: Path, patch: bytes, *, reverse: bool = False) -> bool:
    """Report whether ``patch`` applies to the worktree, without touching it.

    ``git apply --check`` is the worktree counterpart of the index check, and it
    is how a restoration stays idempotent: a candidate that is already present
    fails the forward check and passes the reverse one, so a second recovery
    attempt reads "already there" instead of failing to apply. ``reverse`` swaps
    the direction for exactly that question. A patch that does not apply is a
    question with the answer ``False``, not an error.
    """
    args = ["apply", "--check", "--binary"]
    if reverse:
        args.append("--reverse")
    try:
        process._run_git(
            repo,
            args,
            timeout=30,
            retries=0,
            capture_bytes=True,
            input_bytes=patch,
            error_context=f"git apply --check failed in {repo}",
        )
    except GitError:
        return False
    return True


def apply_worktree_patch(repo: Path, patch: bytes) -> None:
    """Apply binary patch bytes to the worktree, without retrying writes.

    Restoration is the only caller: it re-applies a captured candidate to a tree
    whose identity the host checked first. Retries are off because a write that
    is not idempotent must never be repeated on a transport failure.
    """
    process._run_git(
        repo,
        ["apply", "--binary"],
        timeout=30,
        retries=0,
        capture_bytes=True,
        input_bytes=patch,
        error_context=f"git apply --binary failed in {repo}",
    )


def checkout_detach(repo: Path, sha: str, *, timeout: int = 300) -> None:
    """Detach HEAD at the requested commit without retries.

    The longer default timeout allows lazy blob fetches in partial clones.
    """
    process._run_git(
        repo,
        ["checkout", "--detach", sha],
        timeout=timeout,
        retries=0,
        error_context=f"git checkout --detach {sha} failed in {repo}",
    )


def clone_with_token(
    remote_url: str,
    target: Path,
    token: str | None = None,
    *,
    blobless: bool = False,
    timeout: int = 300,
) -> None:
    """Clone a credential-free identity URL with optional out-of-band authorization.

    The token travels through ``GIT_CONFIG_*`` environment values, never argv, URL, or
    config files. Without a token, inherit credential helpers; disable terminal prompts.
    """
    cmd = ["git"]
    env: dict[str, str] = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    if token:
        # Issue #981: auth travels out-of-band via git config environment
        # variables, never on argv. The base64 Authorization header is
        # trivially recoverable, so putting it on argv (via -c
        # http.extraHeader) would leak the token; GIT_CONFIG_* keeps it out of
        # the command line, honoring the contract that no token lands on argv.
        basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        env.update(
            {
                "GIT_CONFIG_COUNT": "1",
                "GIT_CONFIG_KEY_0": "http.extraHeader",
                "GIT_CONFIG_VALUE_0": f"Authorization: Basic {basic}",
            }
        )
    cmd.append("clone")
    if blobless:
        cmd.append("--filter=blob:none")
    cmd += [remote_url, str(target)]
    process._run_clone(remote_url, cmd, timeout, env=env)


def clone(
    remote_url: str,
    target: Path,
    *,
    blobless: bool = False,
    no_local: bool = False,
    timeout: int = 300,
) -> None:
    """Clone a remote URL or local repository, raising on failure.

    ``blobless`` requests deferred blob fetching (requires server support); ``no_local``
    disables local hardlinks/copy optimizations. The timeout allows initial lazy fetches.
    """
    cmd = ["git", "clone"]
    if blobless:
        cmd.append("--filter=blob:none")
    if no_local:
        cmd.append("--no-local")
    cmd += [remote_url, str(target)]
    process._run_clone(remote_url, cmd, timeout)


def restore_paths_from_ref(repo: Path, ref: str, paths: list[str]) -> None:
    """Replace and stage exactly the named paths from a ref; empty input is a no-op.

    A missing path at that ref raises ``GitError`` so callers can handle newly created files.
    """
    if not paths:
        return
    args = ["checkout", ref, "--", *(str(p) for p in paths)]
    process._run_git(repo, args, timeout=30, retries=0, error_context=f"git checkout {ref} -- {paths} failed in {repo}")


def restore_worktree_paths_from_ref(repo: Path, ref: str, paths: Iterable[str]) -> None:
    """Restore exact paths from *ref* without changing the index."""
    unique = sorted(set(paths), key=queries._path_sort_key)
    if not unique:
        return
    expected = state.snapshot_commit_paths(repo, ref, unique)
    for path_state in expected:
        queries._require_git_path_confined(repo, path_state.path, allow_leaf_symlink=path_state.state == "symlink")
    process._run_git(
        repo,
        [
            "restore",
            f"--source={ref}",
            "--worktree",
            "--no-overlay",
            "--",
            *(queries._literal_pathspec(path) for path in unique),
        ],
        timeout=30,
        retries=0,
        error_context=f"git restore --worktree from ref failed in {repo}",
    )


def worktree_add(
    repo: Path,
    path: Path,
    ref: str,
    *,
    detach: bool = True,
    lock_reason: str | None = None,
) -> None:
    """Create a worktree at a new path, optionally detached and atomically locked.

    ``lock_reason`` arms the lock in the same Git command, closing the concurrent-prune
    window between creation and a separate lock call.
    """
    args = ["worktree", "add"]
    if detach:
        args.append("--detach")
    if lock_reason is not None:
        args.append("--lock")
        args.extend(["--reason", lock_reason])
    args.extend([str(path), ref])
    process._run_git(repo, args, timeout=30, retries=0, error_context=f"git worktree add {path} {ref} failed")


def worktree_remove(repo: Path, path: Path, *, force: bool = True) -> None:
    """Remove a linked worktree, allowing dirty content only with ``force``; failures raise."""
    args = ["worktree", "remove"]
    if force:
        args.append("--force")
    args.append(str(path))
    process._run_git(repo, args, timeout=30, retries=0, error_context=f"git worktree remove {path} failed")


def worktree_move(repo: Path, source: Path, destination: Path) -> None:
    """Move one registered worktree without bypassing Git's bookkeeping."""
    process._run_git(
        repo,
        ["worktree", "move", str(source), str(destination)],
        timeout=30,
        retries=0,
        error_context=f"git worktree move {source} {destination} failed",
    )


def worktree_remove_unlocked(repo: Path, path: Path, *, force: bool = True) -> None:
    """Best-effort unlock, then remove with authoritative error handling.

    Git refuses locked worktrees even with force. Already-unlocked errors are expected;
    removal failures propagate.
    """
    try:
        worktree_unlock(repo, path)
    except GitError:
        pass
    worktree_remove(repo, path, force=force)


def worktree_unlock(repo: Path, path: Path) -> None:
    """Release Git's worktree removal guard, raising ``GitError`` on failure."""
    process._run_git(
        repo,
        ["worktree", "unlock", str(path)],
        timeout=30,
        retries=0,
        error_context=f"git worktree unlock {path} failed",
    )


def registered_worktree_containing(repo: Path, path: Path) -> Path | None:
    """Find the checkout's entry in Git's authoritative worktree registry.

    A broken ``.git`` link or admin chain can fail ordinary worktree discovery while
    still registered; such entries must never be treated as disposable residue.
    """
    proc = process._run_git(
        repo, ["worktree", "list", "--porcelain"], timeout=30, retries=0, error_context="git worktree list failed"
    )
    wanted = path.resolve()
    for line in proc.stdout.splitlines():
        if line.startswith("worktree "):
            try:
                candidate = Path(line.split(maxsplit=1)[1].strip()).resolve()
            except (IndexError, OSError):
                continue
            if candidate == wanted:
                return candidate
    return None


def worktree_lock_mtime(path: Path) -> float | None:
    """Return the lock-armed time of the worktree at *path*, or None if unlocked.

    Git names each linked worktree's administrative directory itself, so the
    lock file is resolved from the exact worktree rather than assembled from
    its basename. ``None`` means a genuinely absent lock file; a worktree or
    lock whose metadata cannot be resolved safely raises :class:`GitError`.
    """
    try:
        locked = queries.git_dir(path) / "locked"
    except (GitError, OSError) as exc:
        raise GitError(f"cannot resolve Git directory for exact worktree {path}") from exc
    try:
        metadata = locked.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise GitError(f"cannot inspect worktree lock for {path}") from exc
    if not stat.S_ISREG(metadata.st_mode):
        raise GitError(f"worktree lock metadata is unsafe for {path}")
    return metadata.st_mtime


def create_branch(repo: Path, name: str) -> None:
    """Create and check out a branch, refusing to overwrite an existing name."""
    process._run_git(
        repo, ["checkout", "-b", name], timeout=30, retries=0, error_context=f"git checkout -b {name} failed in {repo}"
    )


def checkout_branch(repo: Path, name: str) -> None:
    """Check out an existing local branch or track ``origin/<name>``; missing names raise."""
    local = process._run_git(repo, ["rev-parse", "--verify", f"refs/heads/{name}"], timeout=5)
    if local.returncode == 0:
        proc = process._run_git(repo, ["checkout", name], timeout=30, retries=0)
    else:
        proc = process._run_git(repo, ["checkout", "-b", name, f"origin/{name}"], timeout=30, retries=0)
    process._require_ok(proc, f"git checkout {name} failed in {repo}")


def stage_paths(repo: Path, paths: list[Path]) -> None:
    """Stage the nonempty explicit path set, preserving all unrelated working-tree changes."""
    if not paths:
        raise GitError("stage_paths requires at least one path")
    normalized = [p.as_posix() for p in paths]
    for path in normalized:
        queries._require_git_path_confined(repo, path)
    process._run_git(
        repo,
        ["--literal-pathspecs", "add", "--", *normalized],
        timeout=30,
        retries=0,
        error_context=f"git add {paths} failed in {repo}",
    )


def commit_staged(repo: Path, message: str) -> None:
    """Commit the already-validated index without staging again."""
    identity_ok = (
        process._run_git(repo, ["config", "user.email"], timeout=5).returncode == 0
        and process._run_git(repo, ["config", "user.name"], timeout=5).returncode == 0
    )
    commit_args = ["commit", "-m", message]
    if not identity_ok:
        commit_args = [
            "-c",
            "user.email=daydream@localhost",
            "-c",
            "user.name=daydream",
            *commit_args,
        ]
    process._run_git(repo, commit_args, timeout=30, retries=0, error_context=f"git commit failed in {repo}")


def commit_paths(repo: Path, paths: list[Path], message: str) -> None:
    """Stage only the nonempty explicit path set, then commit with ordinary hooks enabled."""
    # Empty-path guard lives in stage_paths (identical GitError).
    stage_paths(repo, paths)
    commit_staged(repo, message)


def push_branch(repo: Path, branch: str, *, remote: str = "origin") -> None:
    """Push the branch with upstream tracking and ordinary hooks; propagate failures."""
    process._run_git(
        repo,
        ["push", "-u", remote, branch],
        timeout=60,
        retries=0,
        error_context=f"git push -u {remote} {branch} failed in {repo}",
    )
