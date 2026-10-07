"""Pin cache endpoint isolation, response TTL, versioned completion markers, and marker aging."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from daydream.training import labeler_versions
from daydream.training.backfill_cache import CACHE_TTL_SECONDS, BackfillCache, GHApiFn


def _counting_gh(calls: list[tuple[str, str]]) -> GHApiFn:
    def real_gh(repo: str, endpoint: str, **kwargs: object) -> dict[str, object]:
        calls.append((repo, endpoint))
        return {"merged": True, "n": len(calls)}

    return real_gh


def test_cache_refetches_stale_response(tmp_path: Path) -> None:
    """Expired comment responses must refetch so reply edits can change the evidence digest."""
    calls: list[tuple[str, str]] = []

    cache = BackfillCache(cache_dir=tmp_path, inner=_counting_gh(calls))
    first = cache("org/repo", "repos/org/repo/pulls/42")
    assert first == {"merged": True, "n": 1}
    assert calls == [("org/repo", "repos/org/repo/pulls/42")]
    cached = cache("org/repo", "repos/org/repo/pulls/42")
    assert first == cached == {"merged": True, "n": 1}
    assert calls == [("org/repo", "repos/org/repo/pulls/42")]
    # Select the JSON cache beside the archive fixture before aging its timestamp.
    path = next(p for p in tmp_path.iterdir() if p.suffix == ".json")
    old = time.time() - CACHE_TTL_SECONDS - 60
    os.utime(path, (old, old))
    second = cache("org/repo", "repos/org/repo/pulls/42")
    assert second == {"merged": True, "n": 2}
    assert calls == [("org/repo", "repos/org/repo/pulls/42")] * 2


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
    cache = BackfillCache(cache_dir=tmp_path, inner=lambda r, e, **kw: {})
    cache.mark_session_done("session-abc")
    cache.mark_session_done("session-xyz")
    lines = (tmp_path / "progress.jsonl").read_text().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["session_id"] == "session-abc"
    assert first["labeler_policy_version"] == labeler_versions.LABELER_POLICY_VERSION

def test_completed_sessions_resume(tmp_path: Path) -> None:
    """Resume only fresh completion markers stamped with the current policy version; legacy or older-
    version markers must refetch.
    """
    current = labeler_versions.LABELER_POLICY_VERSION
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    (tmp_path / "progress.jsonl").write_text(
        json.dumps({"session_id": "s1", "labeler_policy_version": current, "completed_at": now}) + "\n"
        + json.dumps({"session_id": "s-legacy", "completed_at": now}) + "\n"
        + json.dumps({"session_id": "s-old", "labeler_policy_version": "980-policy-r0", "completed_at": now}) + "\n"
    )
    cache = BackfillCache(cache_dir=tmp_path, inner=lambda r, e, **kw: {})
    assert cache.completed_sessions() == {"s1"}

def test_completed_sessions_age_out_after_freshness_window(tmp_path: Path) -> None:
    """Expired completion markers must reprocess the session so edited replies can append a fresh
    generation.
    """
    current = labeler_versions.LABELER_POLICY_VERSION
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    stale_ts = time.time() - CACHE_TTL_SECONDS - 60
    stale = datetime.fromtimestamp(stale_ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    (tmp_path / "progress.jsonl").write_text(
        json.dumps({"session_id": "s-fresh", "labeler_policy_version": current, "completed_at": now}) + "\n"
        + json.dumps({"session_id": "s-stale", "labeler_policy_version": current, "completed_at": stale}) + "\n"
    )
    cache = BackfillCache(cache_dir=tmp_path, inner=lambda r, e, **kw: {})
    assert cache.completed_sessions() == {"s-fresh"}
