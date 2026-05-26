"""Re-score reproducibility gate over the harvest + build-corpus pipeline.

This is a pure contract test over Tasks 5-9: harvest is append-only (a
``REWARD_VERSION`` bump + re-harvest writes a *new* annotation generation, never
mutates the old one) and build-corpus pins to the ``as_of``-resolved generation
(``latest_label_observation(as_of=)`` orders by ``observed_at DESC`` and resolves
the most recent annotation at-or-before the pin). Together they guarantee that an
old ``as_of`` reproduces a prior corpus byte-for-byte even after the reward
formula changes.

Drives the production ``run_harvest`` + ``run_build_corpus`` against a real
SQLite index (no archive-layer mocking). The PR posterior fetch is stubbed via
the ``daydream.training.harvest._gh_api`` monkeypatch — the same seam the other
harvest tests use — so the run never touches the network or the ``gh`` CLI.
"""

from __future__ import annotations

from datetime import datetime, timezone

from daydream.training.corpus import BuildCorpusConfig, CorpusFilters, run_build_corpus
from daydream.training.harvest import HarvestConfig, run_harvest
from tests.test_training_harvest import _fake_gh_merged, _seed_archived_deep_run


async def test_rescore_preserves_old_as_of_byte_for_byte(tmp_path, archive_dir, monkeypatch):
    _seed_archived_deep_run(archive_dir, "s1", merged_at="2026-02-01T00:00:00+00:00")
    monkeypatch.setattr("daydream.training.harvest._gh_api", _fake_gh_merged("2026-02-01T00:00:00+00:00"))
    monkeypatch.setattr("daydream.training.reward.REWARD_VERSION", "r1")
    await run_harvest(HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c1"))
    pin = datetime.now(timezone.utc).isoformat()
    out_a = tmp_path / "a.jsonl"
    run_build_corpus(BuildCorpusConfig(out_path=out_a, archive_dir=archive_dir,
                                       filters=CorpusFilters(include_all_labels=True), as_of=pin))
    bytes_a = out_a.read_bytes()
    monkeypatch.setattr("daydream.training.reward.REWARD_VERSION", "r2")  # bump + re-harvest
    await run_harvest(HarvestConfig(archive_dir=archive_dir, cache_dir=tmp_path / "c2"))
    out_a2 = tmp_path / "a2.jsonl"
    run_build_corpus(BuildCorpusConfig(out_path=out_a2, archive_dir=archive_dir,
                                       filters=CorpusFilters(include_all_labels=True), as_of=pin))
    assert out_a2.read_bytes() == bytes_a   # old as_of reproduces prior corpus exactly
