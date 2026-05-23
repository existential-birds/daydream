"""File-backed cache + JSONL resume log for the labeler backfill loop.

Backfilling labels for historical archive runs means making one or more
``gh_api`` calls per session — PR state, files-changed, review comments,
etc. The labeler orchestrator (Task 13) is restart-safe: if the process
dies partway through a 10k-session sweep, it must resume without
re-paying for completed work.

This module provides two cooperating pieces:

* :class:`BackfillCache` — a callable wrapping ``gh_api`` that memoizes
  responses to JSON files under ``cache_dir``. Historical PRs are
  immutable, so no TTL is needed (see ``/tmp/research-backfill-sla.md``).
* ``progress.jsonl`` — an append-only JSONL log of completed
  ``session_id`` rows, written by :meth:`BackfillCache.mark_session_done`
  and read back by :meth:`BackfillCache.completed_sessions`.

The cache is intentionally process-local and lock-free: each cache key
maps to one file, and the labeler runs single-process. The atomic write
pattern mirrors :mod:`daydream.trajectory` (tempfile + ``os.replace``)
so a crash mid-write leaves either the prior file or nothing — never a
truncated read.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from contextlib import suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from daydream.ui import create_console, print_warning

GHApiFn = Callable[..., Any]
"""Signature of the wrapped ``gh_api`` callable: ``(repo, endpoint, **kwargs) -> Any``."""


def _now_iso_utc() -> str:
    """Return the current UTC time as an ISO-8601 string with a ``Z`` suffix."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _slug_endpoint(repo: str, endpoint: str) -> str:
    """Collapse an endpoint into a short filename-safe slug.

    Replaces ``/`` with ``__`` and collapses any leading
    ``repos__<owner>__<repo>__`` prefix so the resulting filename stays
    a reasonable length even for nested endpoints like
    ``repos/<owner>/<repo>/pulls/<n>/files``.
    """
    raw = endpoint.replace("/", "__")
    owner_repo_prefix = f"repos__{repo.replace('/', '__')}__"
    if raw.startswith(owner_repo_prefix):
        raw = raw[len(owner_repo_prefix):]
    # Strip any leftover characters that are unfriendly in filenames.
    return re.sub(r"[^A-Za-z0-9_.-]", "_", raw)


def _cache_key(repo: str, endpoint: str, kwargs: dict[str, Any]) -> str:
    """Compute the SHA-256 hex digest of ``(repo, endpoint, sorted(kwargs))``."""
    payload = json.dumps(
        {"repo": repo, "endpoint": endpoint, "kwargs": dict(sorted(kwargs.items()))},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, data: Any) -> None:
    """Atomically write ``data`` as JSON to ``path``.

    Mirrors the tempfile + ``os.replace`` pattern in
    :mod:`daydream.trajectory` (lines 893-916).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, indent=2, default=str)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp)
        raise


class BackfillCache:
    """File-backed memoizer for ``gh_api`` + resume log for the labeler.

    Attributes:
        cache_dir: Directory where per-call JSON cache files and
            ``progress.jsonl`` are written.
        inner: The underlying ``gh_api(repo, endpoint, **kwargs)``
            callable that is invoked on cache misses.
    """

    def __init__(self, cache_dir: Path, inner: GHApiFn) -> None:
        self.cache_dir = cache_dir
        self.inner = inner

    def __call__(self, repo: str, endpoint: str, **kwargs: Any) -> Any:
        """Return the cached response for ``(repo, endpoint, **kwargs)``.

        On a cache hit, the JSON cache file is read and returned. On a
        miss (or on a corrupt cache read), ``inner`` is called and the
        result is written through to the cache before being returned.
        """
        digest = _cache_key(repo, endpoint, kwargs)
        owner, _, name = repo.partition("/")
        slug = _slug_endpoint(repo, endpoint)
        filename = f"{owner}__{name}__{slug}__{digest[:8]}.json"
        path = self.cache_dir / filename

        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as exc:
                print_warning(
                    create_console(),
                    f"BackfillCache: corrupt cache file {path.name} ({exc}); "
                    "refetching from inner gh_api.",
                )
                # Fall through to refetch.

        result = self.inner(repo, endpoint, **kwargs)
        _atomic_write_json(path, result)
        return result

    def mark_session_done(self, session_id: str) -> None:
        """Append a completion row for ``session_id`` to ``progress.jsonl``.

        Creates ``cache_dir`` if it does not already exist.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            {"session_id": session_id, "completed_at": _now_iso_utc()},
            sort_keys=True,
        )
        progress = self.cache_dir / "progress.jsonl"
        with progress.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def completed_sessions(self) -> set[str]:
        """Return the set of ``session_id``s recorded in ``progress.jsonl``.

        Returns an empty set if the log does not exist. Malformed lines
        are skipped (the log is append-only and a partial last-line
        write is the only realistic failure mode).
        """
        progress = self.cache_dir / "progress.jsonl"
        if not progress.exists():
            return set()
        out: set[str] = set()
        with progress.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = row.get("session_id")
                if isinstance(sid, str):
                    out.add(sid)
        return out
