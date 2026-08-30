"""Frozen deterministic splits + byte-for-byte reproducibility (Req 12, 13, D5).

Mirrors ``tests/test_corpus_reproducibility.py``'s determinism standard and
``tests/test_corpus_leakage.py``'s as_of boundary standard.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from daydream.training.corpus_v2.projector import BuildCorpusV2Config, run_build_corpus_v2
from daydream.training.corpus_v2.splits import assign_split
from tests.test_corpus_v2 import _write_annotations_snapshot, _write_bundle


def _cfg(out_dir: Path, bundle_dir: Path, snapshot: Path, **kw: Any) -> BuildCorpusV2Config:
    return BuildCorpusV2Config(out_dir=out_dir, bundle_dir=bundle_dir, annotations_snapshot=snapshot, **kw)


def _read_split_memberships(out_dir: Path) -> tuple[list[str], list[str], list[str]]:
    def ids(name: str) -> list[str]:
        path = out_dir / name
        if not path.is_file():
            return []
        return [json.loads(line)["record_id"] for line in path.read_text().splitlines() if line]

    return ids("train.jsonl"), ids("validation.jsonl"), ids("holdout.jsonl")


def test_assign_split_is_deterministic_and_salted() -> None:
    rid = "ab" * 32
    assert assign_split(rid, holdout_rate=0.1, val_rate=0.1, salt="s") == assign_split(
        rid, holdout_rate=0.1, val_rate=0.1, salt="s"
    )
    assert assign_split(rid, holdout_rate=0.0, val_rate=0.0, salt="s") == "train"
    assert assign_split(rid, holdout_rate=1.0, val_rate=0.0, salt="s") == "holdout"
    # a different salt is allowed to reorder membership — it must merely be stable per salt
    assert assign_split(rid, holdout_rate=0.5, val_rate=0.0, salt="s1") == assign_split(
        rid, holdout_rate=0.5, val_rate=0.0, salt="s1"
    )


def test_reprojection_is_byte_for_byte_deterministic(tmp_path: Path) -> None:
    bundle_dir = _write_bundle(tmp_path)
    snap = _write_annotations_snapshot(bundle_dir)
    run_build_corpus_v2(_cfg(tmp_path / "a", bundle_dir, snap))
    run_build_corpus_v2(_cfg(tmp_path / "b", bundle_dir, snap))
    assert (tmp_path / "b" / "corpus.jsonl").read_bytes() == (tmp_path / "a" / "corpus.jsonl").read_bytes()
    assert (tmp_path / "b" / "lineage.json").read_bytes() == (tmp_path / "a" / "lineage.json").read_bytes()
    for name in ("train.jsonl", "validation.jsonl", "holdout.jsonl"):
        assert (tmp_path / "b" / name).read_bytes() == (tmp_path / "a" / name).read_bytes()


def test_splits_are_disjoint_and_frozen(tmp_path: Path) -> None:
    bundle_dir = _write_bundle(tmp_path)
    snap = _write_annotations_snapshot(bundle_dir)
    out_a = tmp_path / "o"
    run_build_corpus_v2(_cfg(out_a, bundle_dir, snap))
    train, val, holdout = _read_split_memberships(out_a)
    assert not (set(train) & set(val))
    assert not (set(train) & set(holdout))
    assert not (set(val) & set(holdout))
    assert train + val + holdout  # the fixture projected records
    # frozen: same membership again on re-run into a fresh directory
    run_build_corpus_v2(_cfg(tmp_path / "b", bundle_dir, snap))
    train2, _, _ = _read_split_memberships(tmp_path / "b")
    assert train2 == train
    # frozen: same membership under re-run in place (overwrite is stable)
    run_build_corpus_v2(_cfg(out_a, bundle_dir, snap))
    train3, _, _ = _read_split_memberships(out_a)
    assert train3 == train


def test_split_membership_recorded_in_record_lineage(tmp_path: Path) -> None:
    bundle_dir = _write_bundle(tmp_path)
    snap = _write_annotations_snapshot(bundle_dir)
    out = tmp_path / "o"
    run_build_corpus_v2(_cfg(out, bundle_dir, snap))
    for line in (out / "corpus.jsonl").read_text().splitlines():
        record = json.loads(line)
        assert record["lineage"]["split"] in {"train", "validation", "holdout"}


def test_late_outcome_evidence_is_refused(tmp_path: Path) -> None:
    bundle_dir = _write_bundle(tmp_path)
    snap = _write_annotations_snapshot(bundle_dir, valid_at="2030-01-01T00:00:00+00:00")
    cfg = _cfg(tmp_path / "late", bundle_dir, snap, as_of="2026-06-01T00:00:00+00:00")
    with pytest.raises(ValueError, match="valid_at"):
        run_build_corpus_v2(cfg)
    # refusal, not drop: nothing was written
    assert not (tmp_path / "late" / "corpus.jsonl").exists()
    assert not (tmp_path / "late" / "lineage.json").exists()
