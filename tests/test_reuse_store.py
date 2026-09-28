"""Tests for the crash-safe deep review reuse entry store.

The store owns ``.daydream/review-cache/``: one directory per content key
holding the cached payload files, then a manifest recording the payload digests
*and* the grounding digests the result was produced under (MH16), then a
completion marker written last. A lookup is a hit only when all three agree.
"""

from __future__ import annotations

from pathlib import Path

from daydream.deep import reuse_key, reuse_store


def _identity() -> reuse_key.PhaseIdentity:
    return reuse_key.PhaseIdentity(
        backend="claude",
        model="claude-sonnet-4-5",
        effort="high",
        profile_digest="d" * 64,
    )


def _cache(
    tmp_path: Path, *, budget: reuse_store.ReuseBudget | None = None
) -> reuse_store.ReuseCache:
    return reuse_store.ReuseCache(tmp_path / "review-cache", budget=budget)


def entry_dir(tmp_path: Path, key: str) -> Path:
    return reuse_store.entry_dir(tmp_path / "review-cache", key)


def test_entry_is_a_hit_only_when_payload_and_marker_agree(tmp_path: Path) -> None:
    store = _cache(tmp_path)
    grounding = {
        "exploration": "a" * 64,
        "intent": "b" * 64,
        "alternatives": "absent",
        "settled_decisions": "c" * 64,
    }
    store.store(
        "a" * 64,
        unit="shard:python#0",
        payload={"stack-python#0-records.json": b"{}"},
        components={"profile": "p"},
        identity=_identity(),
        grounding=grounding,
        grounding_status={
            "exploration": "regenerated",
            "intent": "regenerated",
            "alternatives": "absent",
            "settled_decisions": "regenerated",
        },
    )
    hit = store.lookup("a" * 64)
    assert isinstance(hit, reuse_store.ReuseHit)
    assert hit.manifest["grounding"] == grounding  # MH16: produced-under digests persist
    payload_file = entry_dir(tmp_path, "a" * 64) / "stack-python#0-records.json"
    payload_file.write_bytes(b"{")  # truncated payload
    miss = store.lookup("a" * 64)
    assert isinstance(miss, reuse_store.ReuseMiss) and "payload" in miss.reason
    payload_file.write_bytes(b"{}")
    reuse_store.entry_marker_path(tmp_path / "review-cache", "a" * 64).unlink()
    miss = store.lookup("a" * 64)
    assert isinstance(miss, reuse_store.ReuseMiss) and "marker" in miss.reason
    assert isinstance(store.lookup("b" * 64), reuse_store.ReuseMiss)  # unknown key
