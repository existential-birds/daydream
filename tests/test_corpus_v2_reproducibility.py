"""Frozen deterministic splits + byte-for-byte reproducibility (Req 12, 13, D5).

Mirrors ``tests/test_corpus_reproducibility.py``'s determinism standard and
``tests/test_corpus_leakage.py``'s as_of boundary standard.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from daydream.training import corpus_v2 as corpus_v2_pkg
from daydream.training.corpus import BuildCorpusConfig, CorpusFilters, run_build_corpus
from daydream.training.corpus_v2.projector import BuildCorpusV2Config, run_build_corpus_v2
from daydream.training.corpus_v2.splits import assign_split
from tests.test_corpus_v2 import _cfg, _write_annotations_snapshot, _write_bundle
from tests.test_training_corpus import _seed_run_with_annotation


def _read_split_memberships(out_dir: Path) -> tuple[list[str], list[str], list[str]]:
    def ids(name: str) -> list[str]:
        path = out_dir / name
        # Fail-closed: a missing split manifest must not read as an empty
        # membership — vacuous intersection/disjointness checks would mask a
        # projector regression that skipped a split write.
        if not path.is_file():
            raise FileNotFoundError(f"missing split manifest {path}")
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


def test_v1_and_v2_projection_paths_are_independent(tmp_path: Path, archive_dir: Any) -> None:
    """Run BOTH projectors over the same pinned inputs; assert (a) v1 bytes
    are identical before/after any v2 build, and (b) v2 reprojection is
    byte-identical — the two paths must agree on inputs and diverge only in
    their documented, schema-pinned output shape."""
    _seed_run_with_annotation(archive_dir, "s1", label="accepted",
                              reward_version="r1", observed_at="2026-03-01T00:00:00+00:00",
                              valid_at="2026-03-01T00:00:00+00:00")
    as_of = "2026-04-01T00:00:00+00:00"

    def _run_v1_build(out_path: Path) -> Path:
        run_build_corpus(BuildCorpusConfig(out_path=out_path, archive_dir=archive_dir,
                                           filters=CorpusFilters(), as_of=as_of))
        return out_path

    def _run_v2_build(out_dir: Path) -> Path:
        bundle_dir = _write_bundle(out_dir)
        snap = _write_annotations_snapshot(bundle_dir)
        run_build_corpus_v2(BuildCorpusV2Config(out_dir=out_dir / "out", bundle_dir=bundle_dir,
                                                annotations_snapshot=snap))
        return out_dir / "out" / "corpus.jsonl"

    v1_before = _run_v1_build(tmp_path / "v1a" / "corpus.jsonl").read_bytes()
    assert v1_before  # non-empty checkpoint: a regression that empties v1 must fail here
    _run_v2_build(tmp_path / "v2")
    v1_after = _run_v1_build(tmp_path / "v1b" / "corpus.jsonl").read_bytes()
    assert v1_before == v1_after           # v1 untouched by v2 runs
    v2a = _run_v2_build(tmp_path / "v2a")
    v2b = _run_v2_build(tmp_path / "v2b")
    assert v2a.read_bytes() == v2b.read_bytes()   # v2 determinism
    # the documented, schema-pinned output shape: the v2 build ships schema/v2.json verbatim
    projector_dir = Path(corpus_v2_pkg.__file__).parent
    assert (tmp_path / "v2a" / "out" / "schema.json").read_bytes() == (
        (projector_dir.parent / "schema" / "v2.json").read_bytes()
    )


def test_late_outcome_evidence_is_refused(tmp_path: Path) -> None:
    bundle_dir = _write_bundle(tmp_path)
    snap = _write_annotations_snapshot(bundle_dir, valid_at="2030-01-01T00:00:00+00:00")
    cfg = _cfg(tmp_path / "late", bundle_dir, snap, as_of="2026-06-01T00:00:00+00:00")
    with pytest.raises(ValueError, match="valid_at"):
        run_build_corpus_v2(cfg)
    # refusal, not drop: the full fail-closed file set was never written
    # (every artifact the projector emits, plus the _SUCCESS completeness
    # marker — a regression that wrote any of them before raising fails)
    late_dir = tmp_path / "late"
    for name in ("corpus.jsonl", "corpus-v2.jsonl", "train.jsonl", "validation.jsonl",
                 "holdout.jsonl", "adjudication-report.json", "schema.json",
                 "lineage.json", "_SUCCESS"):
        assert not (late_dir / name).exists(), name
