"""Tests for the crash-safe deep review reuse entry store.

The store owns ``.daydream/review-cache/``: one directory per content key
holding the cached payload files, then a manifest recording the payload digests
*and* the grounding digests the result was produced under (MH16), then a
completion marker written last. A lookup is a hit only when all three agree.
"""

from __future__ import annotations

import json
from pathlib import Path

from daydream.deep import reuse_store
from tests.test_reuse_key import _identity


def _cache(tmp_path: Path, *, budget: reuse_store.ReuseBudget | None = None) -> reuse_store.ReuseCache:
    return reuse_store.ReuseCache(tmp_path / "review-cache", budget=budget, run_id="s1", session_id="s1")


def _hit(store: reuse_store.ReuseCache, *, key: str, grounding: dict[str, str]) -> reuse_store.ReuseHit:
    """A verified hit-shaped manifest without writing an entry to the store."""
    return reuse_store.ReuseHit(key, reuse_store.entry_dir(store.store_dir, key), {"grounding": grounding})


def entry_dir(tmp_path: Path, key: str) -> Path:
    return reuse_store.entry_dir(tmp_path / "review-cache", key)


def _seed_entry(store: reuse_store.ReuseCache, key: str, *, last_used_at: float) -> None:
    store.store(key,
        unit="shard:python#0",
        payload={"x.json": b"{}"}, components={}, identity=_identity(), grounding={}, grounding_status={},
        now=last_used_at,
    )

def test_entry_is_a_hit_only_when_payload_and_marker_agree(tmp_path: Path) -> None:
    store = _cache(tmp_path)
    grounding = {"exploration": "a" * 64, "intent": "b" * 64, "alternatives": "absent", "settled_decisions": "c" * 64}
    store.store("a" * 64,
        unit="shard:python#0",
        payload={"stack-python#0-records.json": b"{}"},
        components={"profile": "p"}, identity=_identity(), grounding=grounding,
        grounding_status={"exploration": "regenerated", "intent": "regenerated", "alternatives": "absent",
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
    manifest_path = reuse_store.entry_manifest_path(store.store_dir, hit.key)
    original_manifest = manifest_path.read_bytes()
    manifest_path.write_text(json.dumps({**hit.manifest, "format": 1}))
    miss = store.lookup(hit.key)
    assert isinstance(miss, reuse_store.ReuseMiss) and miss.reason == "manifest format mismatch"
    manifest_path.write_bytes(original_manifest)
    reuse_store.entry_marker_path(tmp_path / "review-cache", "a" * 64).unlink()
    miss = store.lookup("a" * 64)
    assert isinstance(miss, reuse_store.ReuseMiss) and "marker" in miss.reason
    assert isinstance(store.lookup("b" * 64), reuse_store.ReuseMiss)  # unknown key

def test_prune_evicts_oldest_last_used_and_spares_the_fresh_entry(tmp_path: Path) -> None:
    store = _cache(tmp_path, budget=reuse_store.ReuseBudget(max_entries=2, max_bytes=10**9, max_age_seconds=3600))
    for key, used in (("a" * 64, 1_000), ("b" * 64, 2_000), ("c" * 64, 3_000)):
        _seed_entry(store, key, last_used_at=used)
    store.store("d" * 64, unit="shard:python#0", payload={"x.json": b"{}"}, components={},
                identity=_identity(), grounding={}, grounding_status={})
    live = {p.name for p in reuse_store.entries_dir(store.store_dir).iterdir()}
    assert "d" * 64 in live and "a" * 64 not in live
    assert isinstance(store.lookup("d" * 64), reuse_store.ReuseHit)
    # An entry evicted by age is a plain miss on the next run, never a truncated hit.
    store2 = _cache(tmp_path, budget=reuse_store.ReuseBudget(max_entries=8, max_bytes=10**9, max_age_seconds=1))
    _seed_entry(store2, "e" * 64, last_used_at=1)
    store2.store("f" * 64, unit="shard:python#0", payload={"x.json": b"{}"}, components={},
                 identity=_identity(), grounding={}, grounding_status={})
    assert isinstance(store2.lookup("e" * 64), reuse_store.ReuseMiss)

def test_provenance_records_the_grounding_delta_of_a_reused_unit(tmp_path: Path) -> None:
    store = _cache(tmp_path)
    produced = {"exploration": "a" * 64, "intent": "b" * 64, "alternatives": "absent", "settled_decisions": "c" * 64}
    current = {"exploration": "d" * 64, "intent": "b" * 64, "alternatives": "e" * 64, "settled_decisions": "f" * 64}
    hit = _hit(store, key="k" * 64, grounding=produced)
    delta = store.grounding_delta(hit, current)
    assert delta["exploration"] == {"produced": "a" * 64, "current": "d" * 64, "moved": True}
    assert delta["intent"]["moved"] is False
    assert delta["alternatives"] == {"produced": "absent", "current": "e" * 64, "moved": True}
    store.record("exploration", outcome="regenerated", reason="pre-scan key moved (head changed)")
    store.record(
        "shard:python#0",
        outcome="hit", reason="exact match", key="k" * 64, origin_run_id="run-77",
        detail={"grounding": delta, "grounding_status": {"exploration": "regenerated"}},
    )
    record = store.provenance()
    assert record["units"]["shard:python#0"]["outcome"] == "hit"
    assert record["units"]["shard:python#0"]["origin_run_id"] == "run-77"  # MH5
    # MH16: the whole grounding delta is durable evidence, not just a moved flag.
    assert record["units"]["shard:python#0"]["grounding"]["settled_decisions"]["moved"] is True
    assert record["units"]["shard:python#0"]["grounding_status"]["exploration"] == "regenerated"
    assert record["store"]["entries"] == 0
    assert record["store"]["oldest_last_used_age_s"] is None
    # The record is durable inside the store, so it survives the deep-dir wipe (MH5).
    assert reuse_store.provenance_path(store.store_dir, "s1").is_file()
