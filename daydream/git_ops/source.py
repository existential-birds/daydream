"""Bounded retrieval of regular-file content from a captured Git revision."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath

from daydream.git_ops import process, queries
from daydream.git_ops.models import GitError


def frozen_source(repo: Path, revision: str, path: str) -> tuple[str, bytes]:
    """Resolve one regular blob at a full commit ID, refusing path/ref expansion."""
    relative = PurePosixPath(path)
    if (not re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', revision)
            or not path or relative.is_absolute() or '..' in relative.parts or '\\' in path
            or '\x00' in path or path != relative.as_posix()):
        raise GitError('invalid frozen source selector')
    entry = process._run_git(repo, ['ls-tree', '-z', revision, '--', path], capture_bytes=True,
                             timeout=30, error_context='frozen source identity unavailable')
    raw = entry.stdout if isinstance(entry.stdout, bytes) else entry.stdout.encode()
    records = [record for record in raw.split(b'\0') if record]
    if len(records) != 1:
        # show retains the existing typed absence/error distinction.
        queries.show(repo, revision, path)
        raise GitError('frozen source is not a regular blob')
    identity, _, name = records[0].partition(b'\t')
    mode, kind, oid = identity.split()
    if mode not in {b'100644', b'100755'} or kind != b'blob' or name.decode() != path:
        raise GitError('frozen source is not a regular blob')
    size = process._run_git(repo, ['cat-file', '-s', oid.decode()], timeout=30,
                            error_context='frozen source size unavailable')
    try:
        blob_size = int(size.stdout.strip())
    except ValueError as exc:
        raise GitError('frozen source size is malformed') from exc
    if blob_size > 8 * 1024 * 1024:
        raise GitError('captured revision file exceeds the bounded retrieval limit')
    body = queries.show(repo, revision, path)
    if len(body) > 8 * 1024 * 1024:
        raise GitError('captured revision file exceeds the bounded retrieval limit')
    return oid.decode(), body
