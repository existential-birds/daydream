"""Tests for the crash-safe deep review reuse entry store.

The store owns ``.daydream/review-cache/``: one directory per content key
holding the cached payload files, then a manifest recording the payload digests
*and* the grounding digests the result was produced under (MH16), then a
completion marker written last. A lookup is a hit only when all three agree.
"""

from __future__ import annotations

from pathlib import Path

from daydream.config_file import DaydreamFileConfig
from daydream.deep import reuse_key, reuse_store
from daydream.runner import RunConfig


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


def _seed_entry(
    store: reuse_store.ReuseCache, key: str, *, last_used_at: float
) -> None:
    store.store(
        key,
        unit="shard:python#0",
        payload={"x.json": b"{}"},
        components={},
        identity=_identity(),
        grounding={},
        grounding_status={},
        now=last_used_at,
    )


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


def test_review_cache_enablement_and_budget_resolve_cli_then_file_then_default(tmp_path: Path) -> None:
    cfg = RunConfig(target=str(tmp_path))
    assert reuse_store.review_cache_enabled(cfg) is True          # MH13 default
    budget = reuse_store.review_cache_budget(cfg)
    assert (budget.max_entries, budget.max_bytes, budget.max_age_seconds) == (1024, 1024**3, 30 * 86400)
    file_cfg = DaydreamFileConfig(review_cache_enabled=False, review_cache_max_entries=7)
    fcfg = RunConfig(target=str(tmp_path), file_config=file_cfg)
    assert reuse_store.review_cache_enabled(fcfg) is False
    assert reuse_store.review_cache_budget(fcfg).max_entries == 7
    cli = RunConfig(target=str(tmp_path), file_config=file_cfg, review_cache_enabled=True)
    assert reuse_store.review_cache_enabled(cli) is True          # CLI tier wins
    assert reuse_store.review_cache_budget(cli).max_entries == 7  # budget tiers are independent
