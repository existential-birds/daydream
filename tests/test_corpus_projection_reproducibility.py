"""Offline pinned membership, deterministic bytes, and chronological evidence eligibility."""
import json
from dataclasses import replace
from pathlib import Path

import pytest

from daydream.dataset import semantic_evidence_digest
from daydream.training.corpus_projection.projector import build_frozen_corpus
from daydream.training.corpus_projection.splits import assign_split
from tests.harness.dataset import observation
from tests.harness.record_projection import (
    add_projection_run,
    projection_config,
    seed_projection_store,
)
from tests.test_corpus_projection import _read_jsonl


def _memberships(out_dir: Path) -> dict[str, set[str]]:
    return {name: {row["record_id"] for row in _read_jsonl(out_dir / f"{name}.jsonl")}
            for name in ("train", "validation", "holdout")}


def test_assign_split_is_deterministic_and_salted() -> None:
    rid = "ab" * 32
    assert assign_split(rid, holdout_rate=0.1, val_rate=0.1, salt="s") == assign_split(
        rid, holdout_rate=0.1, val_rate=0.1, salt="s")
    assert assign_split(rid, holdout_rate=0, val_rate=0, salt="s") == "train"
    assert assign_split(rid, holdout_rate=1, val_rate=0, salt="s") == "holdout"


@pytest.mark.parametrize("capped", [False, True])
def test_reprojection_is_byte_identical_with_frozen_disjoint_membership(tmp_path: Path, capped: bool) -> None:
    store = seed_projection_store(tmp_path, dispositions=("accepted", "accepted", "rejected"), siblings=4)
    add_projection_run(store, run_id="sess-b", dispositions=("accepted", "rejected"), stack="rust",
                       profile="quick-review", repo_slug="owner/repo-b")
    caps = {"max_stack_share": 0.5, "max_repo_share": 0.5, "max_profile_share": 0.5} if capped else {}
    config = projection_config(store, tmp_path, **caps)
    build_frozen_corpus(config)
    first = {path.name: path.read_bytes() for path in config.out_dir.iterdir()}
    # Later records are outside the selected immutable snapshot and must not alter any byte.
    add_projection_run(store, run_id="later", repo_slug="owner/later")
    build_frozen_corpus(replace(config, out_dir=tmp_path / "replay"))
    assert {path.name: path.read_bytes() for path in (tmp_path / "replay").iterdir()} == first
    build_frozen_corpus(config)
    assert {path.name: path.read_bytes() for path in config.out_dir.iterdir()} == first
    memberships = _memberships(config.out_dir)
    assert memberships["train"].isdisjoint(memberships["validation"])
    assert memberships["train"].isdisjoint(memberships["holdout"])
    assert memberships["validation"].isdisjoint(memberships["holdout"])
    assert set.union(*memberships.values()) == {
        row["record_id"] for row in _read_jsonl(config.out_dir / "corpus.jsonl")}
    for row in _read_jsonl(config.out_dir / "corpus.jsonl"):
        assert row["record_id"] in memberships[row["lineage"]["split"]]


def test_observation_temporal_eligibility_and_embedded_posterior_leak_are_distinct(tmp_path: Path) -> None:
    store = seed_projection_store(tmp_path, dispositions=("accepted",))
    run = store.read_records()["runs"][0]
    item = run["findings"]["value"]["items"][0]
    evidence = [{"valid_at": "2026-10-06T00:00:00Z", "body": "late"}]
    # An observation whose own valid time is future cannot replace eligible human evidence.
    store.append_observation(observation("late-valid", run_id=run["run_id"], item_uid=item["item_uid"],
        role="adjudicator", valid_at="2026-10-06T00:00:00Z", observed_at="2026-10-04T13:00:00Z",
        semantic_evidence=evidence, evidence_digest=semantic_evidence_digest(evidence),
        payload={"type": "finding-judgment", "disposition": "rejected", "rationale": "late"}))
    config = projection_config(store, tmp_path)
    build_frozen_corpus(config)
    assert _read_jsonl(config.out_dir / "corpus.jsonl")[0]["disposition"] == "accepted"
    # Eligible envelopes still must not smuggle posterior reply evidence through the pin.
    store.append_observation(observation("leaking-valid", run_id=run["run_id"], item_uid=item["item_uid"],
        role="automatic", observed_at="2026-10-04T14:00:00Z", semantic_evidence=evidence,
        evidence_digest=semantic_evidence_digest(evidence),
        payload={"type": "finding-judgment", "disposition": "rejected", "rationale": "invalid pin"}))
    with pytest.raises(ValueError, match="posterior"):
        build_frozen_corpus(projection_config(store, tmp_path, out_dir=tmp_path / "leaking"))
    assert not (tmp_path / "leaking").exists()


def test_snapshot_pins_download_provenance_and_replays_after_source_changes(tmp_path: Path) -> None:
    store = seed_projection_store(tmp_path)
    store.record_download_source({"repository": "owner/private", "revision": "a" * 40,
                                  "manifest_sha256": "c" * 64}, **store.read_records())
    config = projection_config(store, tmp_path)
    build_frozen_corpus(config)
    first = {path.name: path.read_bytes() for path in config.out_dir.iterdir()}
    lineage = json.loads((config.out_dir / "lineage.json").read_text())
    assert lineage["hub_commit"] == "a" * 40
    assert lineage["source_repository"] == "owner/private"
    assert lineage["snapshot"]["source"]["manifest_sha256"] == "c" * 64
    assert all(row["lineage"]["hub_commit"] == "a" * 40 for row in _read_jsonl(config.out_dir / "corpus.jsonl"))
    store.record_download_source({"repository": "owner/other", "revision": "b" * 40,
                                  "manifest_sha256": "d" * 64}, **store.read_records())
    build_frozen_corpus(config)
    assert {path.name: path.read_bytes() for path in config.out_dir.iterdir()} == first
