"""Tests for the per-snapshot lineage manifest emitted by build-corpus.

Each ``run_build_corpus`` invocation writes ``lineage.json`` beside the JSONL
output, pinning the provenance of the snapshot: the content-addressed
``trajectory_set_hash`` of the included sessions, the observed labeler/reward
versions, the ``as_of`` pin, and a wall-clock ``created_at``. These tests drive
:func:`run_build_corpus` against a real SQLite index (no archive-layer mocking)
and assert on the manifest the user would inspect on disk.
"""

from __future__ import annotations

import hashlib
import json

from daydream.training.corpus import BuildCorpusConfig, CorpusFilters, run_build_corpus
from tests.test_training_corpus import _seed_run_with_annotation


def test_build_corpus_emits_lineage_manifest(tmp_path, archive_dir):
    _seed_run_with_annotation(archive_dir, "s1", label="accepted",
                              reward_version="r1", observed_at="2026-03-01T00:00:00+00:00",
                              valid_at="2026-03-01T00:00:00+00:00")
    out = tmp_path / "c.jsonl"
    run_build_corpus(BuildCorpusConfig(out_path=out, archive_dir=archive_dir,
                                       filters=CorpusFilters(), as_of="2026-04-01T00:00:00+00:00"))
    man = json.loads((tmp_path / "lineage.json").read_text())
    assert set(man) >= {"trajectory_set_hash", "labeler_version", "reward_version",
                        "as_of", "created_at"}
    assert man["trajectory_set_hash"] == hashlib.sha256(b"s1").hexdigest()
    assert man["as_of"] == "2026-04-01T00:00:00+00:00"
