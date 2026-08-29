"""Tests for :mod:`daydream.training.backfill_cache`.

Covers:
* File-backed memoization of ``gh_api(repo, endpoint, **kwargs)`` calls.
* Cache key isolation by endpoint.
* Freshness window: memoized responses older than ``CACHE_TTL_SECONDS``
  are refetched (M14 edited-reply freshness).
* JSONL ``progress.jsonl`` resume log — append + read.
* Policy-version stamping: a bump wholesale-invalidates resume markers (M15).
* Marker aging: a completion older than ``CACHE_TTL_SECONDS`` does not resume.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from daydream.training import labeler_versions
from daydream.training.backfill_cache import CACHE_TTL_SECONDS, BackfillCache


def test_cache_refetches_stale_response(tmp_path: Path) -> None:
    """A memoized response older than CACHE_TTL_SECONDS is refetched (M14).

    An edited reply must not be masked by a stale memoized /comments payload:
    once the freshness window expires the inner gh_api is called again, so the
    reply-evidence digest is recomputed from live data.
    """
    calls: list[tuple[str, str]] = []

    def real_gh(repo: str, endpoint: str, **kwargs: object) -> dict[str, object]:
        calls.append((repo, endpoint))
        return {"merged": True, "n": len(calls)}

    cache = BackfillCache(cache_dir=tmp_path, inner=real_gh)
    first = cache("org/repo", "repos/org/repo/pulls/42")
    assert first == {"merged": True, "n": 1}
    assert calls == [("org/repo", "repos/org/repo/pulls/42")]
    # Age the cache file beyond the freshness window (tmp_path also holds the
    # autouse ``archive_dir`` fixture dir, so target the ``.json`` entry).
    path = next(p for p in tmp_path.iterdir() if p.suffix == ".json")
    old = time.time() - CACHE_TTL_SECONDS - 60
    os.utime(path, (old, old))
    second = cache("org/repo", "repos/org/repo/pulls/42")
    assert second == {"merged": True, "n": 2}
    assert calls == [("org/repo", "repos/org/repo/pulls/42")] * 2


def test_cache_returns_cached_response_on_second_call(tmp_path: Path) -> None:
    calls: list[tuple[str, str]] = []

    def real_gh(repo: str, endpoint: str, **kwargs: object) -> dict[str, object]:
        calls.append((repo, endpoint))
        return {"merged": True, "n": len(calls)}

    cache = BackfillCache(cache_dir=tmp_path, inner=real_gh)
    first = cache("org/repo", "repos/org/repo/pulls/42")
    second = cache("org/repo", "repos/org/repo/pulls/42")
    assert first == second == {"merged": True, "n": 1}
    assert calls == [("org/repo", "repos/org/repo/pulls/42")]


def test_cache_misses_for_different_endpoints(tmp_path: Path) -> None:
    calls: list[tuple[str, str]] = []

    def real_gh(repo: str, endpoint: str, **kwargs: object) -> dict[str, object]:
        calls.append((repo, endpoint))
        return {"endpoint": endpoint}

    cache = BackfillCache(cache_dir=tmp_path, inner=real_gh)
    cache("o/r", "pulls/1")
    cache("o/r", "pulls/2")
    assert len(calls) == 2


def test_progress_log_appends_one_line_per_session(tmp_path: Path) -> None:
    """BackfillCache writes a versioned JSONL line per session_id processed."""
    cache = BackfillCache(cache_dir=tmp_path, inner=lambda r, e, **kw: {})
    cache.mark_session_done("session-abc")
    cache.mark_session_done("session-xyz")
    lines = (tmp_path / "progress.jsonl").read_text().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["session_id"] == "session-abc"
    # M15: the row is stamped with the labeler policy version at completion time.
    assert first["labeler_policy_version"] == labeler_versions.LABELER_POLICY_VERSION


def test_completed_sessions_resume(tmp_path: Path) -> None:
    """On startup, BackfillCache exposes the set of already-completed sessions.

    Only rows stamped with the *current* labeler policy version and completed
    inside the freshness window count; legacy rows without the version field
    (or from an older policy) never match, so a policy bump re-fetches
    previously completed sessions (M15).
    """
    current = labeler_versions.LABELER_POLICY_VERSION
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    (tmp_path / "progress.jsonl").write_text(
        json.dumps(
            {"session_id": "s1", "labeler_policy_version": current,
             "completed_at": now},
        ) + "\n"
        + json.dumps(
            {"session_id": "s-legacy", "completed_at": now},
        ) + "\n"
        + json.dumps(
            {"session_id": "s-old", "labeler_policy_version": "980-policy-r0",
             "completed_at": now},
        ) + "\n"
    )
    cache = BackfillCache(cache_dir=tmp_path, inner=lambda r, e, **kw: {})
    assert cache.completed_sessions() == {"s1"}


def test_completed_sessions_age_out_after_freshness_window(tmp_path: Path) -> None:
    """A marker older than CACHE_TTL_SECONDS does not resume the session (M14).

    A completion marker from a previous run must not permanently filter the
    queue: once the freshness window passes, the session is re-processed so a
    reply edit appends a fresh generation through the standard pipeline.
    """
    current = labeler_versions.LABELER_POLICY_VERSION
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    stale_ts = time.time() - CACHE_TTL_SECONDS - 60
    stale = datetime.fromtimestamp(stale_ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    (tmp_path / "progress.jsonl").write_text(
        json.dumps(
            {"session_id": "s-fresh", "labeler_policy_version": current,
             "completed_at": now},
        ) + "\n"
        + json.dumps(
            {"session_id": "s-stale", "labeler_policy_version": current,
             "completed_at": stale},
        ) + "\n"
    )
    cache = BackfillCache(cache_dir=tmp_path, inner=lambda r, e, **kw: {})
    assert cache.completed_sessions() == {"s-fresh"}
