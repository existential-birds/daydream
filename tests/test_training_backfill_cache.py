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
    """A real response cache separates endpoints, reuses fresh bytes and refetches stale evidence."""
    calls: list[tuple[str, str]] = []
    cache = BackfillCache(cache_dir=tmp_path, inner=_counting_gh(calls))
    first = cache("org/repo", "repos/org/repo/pulls/42")
    path = next(tmp_path.glob("*.json"))
    assert first == cache("org/repo", "repos/org/repo/pulls/42") == {"merged": True, "n": 1}
    assert calls == [("org/repo", "repos/org/repo/pulls/42")]
    assert cache("org/repo", "pulls/43") == {"merged": True, "n": 2}
    relative_path = next(candidate for candidate in tmp_path.glob("*.json") if candidate != path)
    assert relative_path.name.startswith("org__repo__pulls__43__")
    assert len(list(tmp_path.glob("*.json"))) == 2
    old = time.time() - CACHE_TTL_SECONDS - 60
    os.utime(path, (old, old))
    assert cache("org/repo", "repos/org/repo/pulls/42") == {"merged": True, "n": 3}
    assert calls == [("org/repo", "repos/org/repo/pulls/42"), ("org/repo", "pulls/43"),
                     ("org/repo", "repos/org/repo/pulls/42")]




def test_completed_sessions_resume(tmp_path: Path) -> None:
    """One progress population admits only fresh current-policy records, then appends new completions."""
    current = labeler_versions.LABELER_POLICY_VERSION
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    stale = datetime.fromtimestamp(time.time() - CACHE_TTL_SECONDS - 60, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    path = tmp_path / "progress.jsonl"
    path.write_text(json.dumps({"session_id": "s1", "labeler_policy_version": current, "completed_at": now}) + "\n"
        + json.dumps({"session_id": "s-legacy", "completed_at": now}) + "\n"
        + json.dumps({"session_id": "s-old", "labeler_policy_version": "980-policy-r0", "completed_at": now}) + "\n"
        + json.dumps({"session_id": "s-stale", "labeler_policy_version": current, "completed_at": stale}) + "\n")
    cache = BackfillCache(cache_dir=tmp_path, inner=lambda r, e, **kw: {})
    assert cache.completed_sessions() == {"s1"}
    cache.mark_session_done("session-abc")
    cache.mark_session_done("session-xyz")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 6
    assert [row["session_id"] for row in rows[-2:]] == ["session-abc", "session-xyz"]
    assert all(set(row) == {"session_id", "labeler_policy_version", "completed_at"}
               and row["labeler_policy_version"] == current for row in rows[-2:])
    assert cache.completed_sessions() == {"s1", "session-abc", "session-xyz"}
