"""File-backed cache + JSONL resume log for the labeler backfill loop.

Backfilling labels for historical archive runs means making one or more
``gh_api`` calls per session — PR state, files-changed, review comments,
etc. The labeler orchestrator (Task 13) is restart-safe: if the process
dies partway through a 10k-session sweep, it must resume without
re-paying for completed work.

This module provides two cooperating pieces:

* :class:`BackfillCache` — a callable wrapping ``gh_api`` that memoizes
  responses to JSON files under ``cache_dir``. GitHub state is only
  immutable within a freshness window — replies can be edited and PRs
  can merge — so a memoized entry older than ``CACHE_TTL_SECONDS`` is
  refetched rather than served (M14): the reply-evidence digest is
  recomputed from live data, so an edited reply appends a fresh
  generation instead of being masked by a stale memoized response.
* ``progress.jsonl`` — an append-only JSONL log of completed
  ``session_id`` rows, written by :meth:`BackfillCache.mark_session_done`
  and read back by :meth:`BackfillCache.completed_sessions`. Each row is
  stamped with the labeler policy version in force at completion time, so
  a policy bump wholesale-invalidates the resume markers (M15) and forces
  a re-fetch rather than silently resuming stale labels. Markers also
  age out past ``CACHE_TTL_SECONDS`` (on their ``completed_at``), so a
  re-run outside the freshness window re-processes the session and can
  observe a reply edit.

The cache is intentionally process-local and lock-free: each cache key
maps to one file, and the labeler runs single-process. Cache files are
written via :func:`daydream.json_utils.atomic_write_json` so a crash
mid-write leaves either the prior file or nothing — never a truncated
read.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from daydream.json_utils import atomic_write_json
from daydream.training import labeler_versions
from daydream.ui import create_console, print_warning

GHApiFn = Callable[..., Any]
"""Signature of the wrapped ``gh_api`` callable: ``(repo, endpoint, **kwargs) -> Any``."""

CACHE_TTL_SECONDS = 24 * 60 * 60
"""Freshness window (seconds) for memoized responses and resume markers.

GitHub state is not truly immutable — replies can be edited, PRs can merge —
so an entry older than this window is refetched (and a completion marker
older than this window does not resume the session), letting a re-run observe
a reply edit and append a fresh generation (M14) instead of being masked by a
stale memoized response.
"""


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

    @property
    def progress_path(self) -> Path:
        """Absolute path of the JSONL resume log (``<cache_dir>/progress.jsonl``)."""
        return self.cache_dir / "progress.jsonl"

    def __call__(self, repo: str, endpoint: str, **kwargs: Any) -> Any:
        """Return the cached response for ``(repo, endpoint, **kwargs)``.

        On a cache hit within the freshness window
        (:data:`CACHE_TTL_SECONDS`), the JSON cache file is read and
        returned. On a miss, a stale cache file, or a corrupt cache read,
        ``inner`` is called and the result is written through to the cache
        before being returned.
        """
        digest = _cache_key(repo, endpoint, kwargs)
        owner, _, name = repo.partition("/")
        slug = _slug_endpoint(repo, endpoint)
        filename = f"{owner}__{name}__{slug}__{digest[:8]}.json"
        path = self.cache_dir / filename

        if path.exists():
            try:
                fresh = time.time() - path.stat().st_mtime < CACHE_TTL_SECONDS
            except OSError:
                fresh = False
            if fresh:
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
        atomic_write_json(path, result, default=str)
        return result

    def mark_session_done(self, session_id: str) -> None:
        """Append a completion row for ``session_id`` to ``progress.jsonl``.

        The row records the labeler policy version in force at completion
        time (M15), so a later policy bump invalidates the marker wholesale.
        Creates ``cache_dir`` if it does not already exist.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            {
                "session_id": session_id,
                "labeler_policy_version": labeler_versions.LABELER_POLICY_VERSION,
                "completed_at": _now_iso_utc(),
            },
            sort_keys=True,
        )
        with self.progress_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def completed_sessions(self) -> set[str]:
        """Return sessions recorded in ``progress.jsonl`` for the current policy.

        Only rows whose stored ``labeler_policy_version`` equals the *current*
        :data:`~daydream.training.labeler_versions.LABELER_POLICY_VERSION`
        (read at call time) count as done — a policy bump re-fetches every
        previously completed session (M15 wholesale invalidation). Rows from
        before the version field existed never match. A marker also ages out
        once its ``completed_at`` is older than
        :data:`CACHE_TTL_SECONDS`, so a re-run outside the freshness window
        re-processes the session — letting an edited reply append a fresh
        generation (M14) rather than being masked by a stale marker. Returns
        an empty set if the log does not exist. Malformed lines are skipped
        (the log is append-only and a partial last-line write is the only
        realistic failure mode).
        """
        if not self.progress_path.exists():
            return set()
        current_version = labeler_versions.LABELER_POLICY_VERSION
        cutoff = time.time() - CACHE_TTL_SECONDS
        out: set[str] = set()
        with self.progress_path.open("r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sid = row.get("session_id")
                if not (isinstance(sid, str) and row.get("labeler_policy_version") == current_version):
                    continue
                completed_at = row.get("completed_at")
                if not isinstance(completed_at, str):
                    continue
                try:
                    stamped = datetime.fromisoformat(completed_at)
                except ValueError:
                    continue
                if stamped.timestamp() < cutoff:
                    continue
                out.add(sid)
        return out
