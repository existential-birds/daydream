"""Tests for :mod:`daydream.training.backfill_cache`.

Covers:
* File-backed memoization of ``gh_api(repo, endpoint, **kwargs)`` calls.
* Cache key isolation by endpoint.
* JSONL ``progress.jsonl`` resume log — append + read.
* Policy-version stamping: a bump wholesale-invalidates resume markers (M15).
"""

from __future__ import annotations

import json
from pathlib import Path

from daydream.training import labeler_versions
from daydream.training.backfill_cache import BackfillCache


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

    Only rows stamped with the *current* labeler policy version count; legacy
    rows without the version field (or from an older policy) never match, so a
    policy bump re-fetches previously completed sessions (M15).
    """
    current = labeler_versions.LABELER_POLICY_VERSION
    (tmp_path / "progress.jsonl").write_text(
        json.dumps(
            {"session_id": "s1", "labeler_policy_version": current,
             "completed_at": "2026-05-22T00:00:00Z"},
        ) + "\n"
        + json.dumps(
            {"session_id": "s-legacy", "completed_at": "2026-05-22T00:00:00Z"},
        ) + "\n"
        + json.dumps(
            {"session_id": "s-old", "labeler_policy_version": "980-policy-r0",
             "completed_at": "2026-05-22T00:00:00Z"},
        ) + "\n"
    )
    cache = BackfillCache(cache_dir=tmp_path, inner=lambda r, e, **kw: {})
    assert cache.completed_sessions() == {"s1"}
