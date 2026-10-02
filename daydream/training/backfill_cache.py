"""File-backed GitHub response cache and append-only session completion log.

Responses and completion markers expire after CACHE_TTL_SECONDS so reply edits
and PR changes are re-observed. Markers also require the current labeler policy
version; policy changes invalidate the whole resume set. Cache writes are atomic.
This single-process cache is intentionally lock-free, with one file per key.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from daydream.json_utils import atomic_write_json
from daydream.timeutil import now_iso_utc
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


def _slug_endpoint(repo: str, endpoint: str) -> str:
    """Make an endpoint filename-safe, replacing / with __ and removing its repo prefix."""
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
    """Memoize inner gh_api calls and persist session completion under cache_dir."""

    def __init__(self, cache_dir: Path, inner: GHApiFn) -> None:
        self.cache_dir = cache_dir
        self.inner = inner

    @property
    def progress_path(self) -> Path:
        """Absolute path of the JSONL resume log (``<cache_dir>/progress.jsonl``)."""
        return self.cache_dir / "progress.jsonl"

    def __call__(self, repo: str, endpoint: str, **kwargs: Any) -> Any:
        """Return a fresh cached response; fetch and atomically cache misses, stale, or corrupt entries."""
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
        """Append the current policy and completion time to progress.jsonl, creating cache_dir."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            {
                "session_id": session_id,
                "labeler_policy_version": labeler_versions.LABELER_POLICY_VERSION,
                "completed_at": now_iso_utc(),
            },
            sort_keys=True,
        )
        with self.progress_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def completed_sessions(self) -> set[str]:
        """Return sessions whose markers match the current policy and freshness window.

        Read the policy at call time. Missing logs return an empty set; legacy,
        stale, or malformed lines do not count, including partial append tails.
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
