"""Temporal-leakage guard tests for the build-corpus projection.

These exercise the valid-time exclusion: an annotation whose outcome only
became true *after* the ``as_of`` pin must not leak its posterior-derived
``outcome_label`` into a corpus pinned to that ``as_of``. The guard is a
lexical comparison on ISO-8601 strings (already UTC, lexically ordered),
mirroring the ``observed_at <= as_of`` pin in ``latest_label_observation``.

Drives :func:`run_build_corpus` against a real SQLite index built with the
production ``upsert_run`` + ``append_label_observation`` helpers — reusing the
``_seed_run_with_annotation`` helper and ``archive_dir`` fixture established in
``tests/test_training_corpus.py``.
"""

from __future__ import annotations

import json

from daydream.training.corpus import BuildCorpusConfig, CorpusFilters, run_build_corpus
from tests.test_training_corpus import _seed_run_with_annotation


def test_posterior_dated_after_pin_is_excluded(tmp_path, archive_dir):
    # annotation recorded before the pin, but its outcome became true AFTER it
    _seed_run_with_annotation(archive_dir, "s1", label="accepted",
                              observed_at="2026-03-01T00:00:00+00:00",
                              valid_at="2026-09-01T00:00:00+00:00")  # valid_at > as_of
    out = tmp_path / "c.jsonl"
    run_build_corpus(BuildCorpusConfig(out_path=out, archive_dir=archive_dir,
                                       filters=CorpusFilters(), as_of="2026-04-01T00:00:00+00:00"))
    recs = [json.loads(line) for line in out.read_text().splitlines()]
    # posterior label leaked from the future is not admitted
    assert all(r["outcome_label"] != "accepted" for r in recs)
