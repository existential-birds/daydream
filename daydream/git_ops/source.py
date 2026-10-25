"""Bounded retrieval of regular-file content from a captured Git revision."""

from __future__ import annotations

import os
import re
import stat
from pathlib import Path, PurePosixPath

from daydream.git_ops import process, queries
from daydream.git_ops.models import GitError


class FrozenSourceRejected(GitError):
    """A selected path/blob violates the frozen-source policy; another selection can resolve it."""


def _validate_commit(repo: Path, revision: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", revision):
        raise GitError("committed source requires a full commit ID")
    resolved = process._run_git(
        repo, ["rev-parse", "--verify", f"{revision}^{{commit}}"],
        error_context="committed source revision unavailable",
    )
    if resolved.stdout.strip() != revision:
        raise GitError("committed source revision is not a commit ID")


def committed_diff(repo: Path, base_sha: str, head_sha: str) -> str:
    """Diff exact captured commits without checkout bytes or configured diff transforms."""
    _validate_commit(repo, base_sha)
    _validate_commit(repo, head_sha)
    return process._run_git(
        repo, ["diff", "--no-ext-diff", "--no-textconv", "--src-prefix=a/", "--dst-prefix=b/",
               base_sha, head_sha, "--"],
        timeout=30, error_context="committed source diff unavailable",
    ).stdout


def validate_committed_worktree(repo: Path, revision: str) -> None:
    """Refuse native regular-file reads that would differ from captured Git blobs.

    A clean index/status is insufficient: reversible checkout filters and EOL
    conversion can leave different native bytes while Git reports no changes.
    Symlinks and gitlinks retain their nonregular identity; this validation does
    not grant them regular-file source authority.
    """
    _validate_commit(repo, revision)
    tree = process._run_git(
        repo, ["ls-tree", "-r", "-z", revision], capture_bytes=True,
        timeout=30, error_context="committed source inventory unavailable",
    )
    regular: list[tuple[str, bytes]] = []
    for entry in tree.stdout.split(b"\0"):
        if not entry:
            continue
        identity, _, raw_path = entry.partition(b"\t")
        mode, kind, oid = identity.split()
        if (mode, kind) in {(b"120000", b"blob"), (b"160000", b"commit")}:
            continue
        if mode not in {b"100644", b"100755"} or kind != b"blob":
            raise FrozenSourceRejected("committed source inventory has unsupported entry metadata")
        path = os.fsdecode(raw_path)
        relative = PurePosixPath(path)
        if relative.is_absolute() or ".." in relative.parts or not path:
            raise FrozenSourceRejected("committed source inventory has an invalid path")
        current = repo
        try:
            for index, part in enumerate(relative.parts):
                current /= part
                mode_bits = current.lstat().st_mode
                expected = stat.S_ISREG if index == len(relative.parts) - 1 else stat.S_ISDIR
                if not expected(mode_bits):
                    raise FrozenSourceRejected(f"committed source {path!r} is not a native regular file")
        except OSError as exc:
            raise FrozenSourceRejected(f"committed source {path!r} is unavailable in the captured worktree") from exc
        regular.append((path, oid))
    # Bound argv size/count and never invoke a clean filter or write an object.
    for offset in range(0, len(regular), 32):
        batch = regular[offset:offset + 32]
        hashed = process._run_git(
            repo, ["hash-object", "--no-filters", "--", *(path for path, _ in batch)],
            capture_bytes=True, timeout=30, error_context="committed native source identity unavailable",
        ).stdout.splitlines()
        if len(hashed) != len(batch):
            raise FrozenSourceRejected("committed native source identities are incomplete")
        for (path, expected_oid), actual_oid in zip(batch, hashed, strict=True):
            if actual_oid != expected_oid:
                raise FrozenSourceRejected(
                    f"committed source {path!r} differs from its captured Git blob; "
                    "checkout filters or EOL conversion must preserve raw bytes for committed review"
                )


def frozen_source(repo: Path, revision: str, path: str) -> tuple[str, bytes]:
    """Resolve one regular blob at a full commit ID, refusing path/ref expansion."""
    relative = PurePosixPath(path)
    if not re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', revision):
        raise GitError('invalid frozen source revision')
    if (not path or relative.is_absolute() or '..' in relative.parts or '\\' in path
            or '\x00' in path or path != relative.as_posix()):
        raise FrozenSourceRejected('invalid frozen source selector')
    entry = process._run_git(repo, ['ls-tree', '-z', revision, '--', path], capture_bytes=True,
                             timeout=30, error_context='frozen source identity unavailable')
    raw = entry.stdout if isinstance(entry.stdout, bytes) else entry.stdout.encode()
    records = [record for record in raw.split(b'\0') if record]
    if len(records) != 1:
        # show retains the existing typed absence/error distinction.
        queries.show(repo, revision, path)
        raise FrozenSourceRejected('frozen source is not a regular blob')
    identity, _, name = records[0].partition(b'\t')
    mode, kind, oid = identity.split()
    if mode not in {b'100644', b'100755'} or kind != b'blob' or name.decode() != path:
        raise FrozenSourceRejected('frozen source is not a regular blob')
    size = process._run_git(repo, ['cat-file', '-s', oid.decode()], timeout=30,
                            error_context='frozen source size unavailable')
    try:
        blob_size = int(size.stdout.strip())
    except ValueError as exc:
        raise GitError('frozen source size is malformed') from exc
    if blob_size > 8 * 1024 * 1024:
        raise FrozenSourceRejected('captured revision file exceeds the bounded retrieval limit')
    body = queries.show(repo, revision, path)
    if len(body) > 8 * 1024 * 1024:
        raise FrozenSourceRejected('captured revision file exceeds the bounded retrieval limit')
    return oid.decode(), body
