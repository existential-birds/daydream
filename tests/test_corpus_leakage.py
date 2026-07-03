"""Temporal-leakage guard tests for the build-corpus projection.

These exercise the valid-time exclusion: an annotation whose outcome only
became true *after* the ``as_of`` pin must not leak its posterior-derived
``outcome_label`` into a corpus pinned to that ``as_of``. The guard compares
parsed datetimes chronologically, so ``Z``/``+00:00`` spellings, sub-second
precision, and non-UTC offsets can never mis-order it; ``as_of`` itself is
validated and canonicalized once at the :class:`BuildCorpusConfig` boundary.

Drives :func:`run_build_corpus` against a real SQLite index built with the
production ``upsert_run`` + ``append_label_observation`` helpers — reusing the
``_seed_run_with_annotation`` helper and ``archive_dir`` fixture established in
``tests/test_training_corpus.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from daydream.training.corpus import (
    BuildCorpusConfig,
    CorpusFilters,
    _is_posterior_leak,
    run_build_corpus,
)
from tests.test_training_corpus import _seed_run_with_annotation


def test_posterior_dated_after_pin_is_excluded(tmp_path, archive_dir):
    # annotation recorded before the pin, but its outcome became true AFTER it
    _seed_run_with_annotation(archive_dir, "s1", label="accepted",
                              observed_at="2026-03-01T00:00:00+00:00",
                              valid_at="2026-09-01T00:00:00+00:00")  # valid_at > as_of
    out = tmp_path / "c.jsonl"
    run_build_corpus(BuildCorpusConfig(out_path=out, archive_dir=archive_dir,
                                       filters=CorpusFilters(include_all_labels=True),
                                       as_of="2026-04-01T00:00:00+00:00"))
    recs = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(recs) == 1
    assert recs[0]["outcome_label"] is None


# _is_posterior_leak boundary semantics (unit): valid_at == as_of is in-time,
# strictly greater is a leak, and no ISO-8601 spelling difference mis-orders it.

AS_OF = "2026-04-01T00:00:00+00:00"


def _ann(valid_at: str | None) -> dict:
    return {"valid_at": valid_at}


def test_leak_guard_equal_instant_is_not_a_leak():
    assert _is_posterior_leak(_ann(AS_OF), AS_OF) is False


def test_leak_guard_strictly_greater_is_a_leak():
    assert _is_posterior_leak(_ann("2026-04-01T00:00:00.000001+00:00"), AS_OF) is True
    assert _is_posterior_leak(_ann("2026-03-31T23:59:59.999999+00:00"), AS_OF) is False


def test_leak_guard_none_inputs_never_leak():
    assert _is_posterior_leak(None, AS_OF) is False
    assert _is_posterior_leak(_ann(None), AS_OF) is False
    assert _is_posterior_leak(_ann("2026-09-01T00:00:00+00:00"), None) is False


def test_leak_guard_mixed_z_and_offset_spellings_compare_chronologically():
    # Same instant spelled "Z" vs "+00:00", both directions: never a leak.
    assert _is_posterior_leak(_ann("2026-04-01T00:00:00Z"), AS_OF) is False
    assert _is_posterior_leak(_ann(AS_OF), "2026-04-01T00:00:00Z") is False
    # One second later, spelled "Z": still detected as a leak.
    assert _is_posterior_leak(_ann("2026-04-01T00:00:01Z"), AS_OF) is True


def test_leak_guard_subsecond_precision_compares_chronologically():
    # ".000000" and no-fraction are the same instant — not a leak in either
    # direction. (A suffix-only lexical normalisation would have called the
    # fractional spelling "greater" and leaked a false exclusion.)
    assert _is_posterior_leak(_ann("2026-04-01T00:00:00.000000Z"), AS_OF) is False
    assert _is_posterior_leak(_ann("2026-04-01T00:00:00.000000+00:00"), AS_OF) is False
    # Half a second after the pin is a leak; half a second before is not.
    assert _is_posterior_leak(_ann("2026-04-01T00:00:00.500000+00:00"), "2026-04-01T00:00:00Z") is True
    assert _is_posterior_leak(_ann("2026-04-01T00:00:00Z"), "2026-04-01T00:00:00.500000+00:00") is False


def test_leak_guard_non_utc_valid_at_converts_chronologically():
    # A stored non-UTC offset is an unambiguous instant: +05:00 at 05:00 == the
    # pin instant (not a leak); one second later leaks.
    assert _is_posterior_leak(_ann("2026-04-01T05:00:00+05:00"), AS_OF) is False
    assert _is_posterior_leak(_ann("2026-04-01T05:00:01+05:00"), AS_OF) is True


# as_of entry boundary: BuildCorpusConfig validates and canonicalizes ONCE,
# before the pin reaches the SQL cutoff or the leak guard.


def _cfg(tmp_path: Path, as_of: str) -> BuildCorpusConfig:
    return BuildCorpusConfig(out_path=tmp_path / "c.jsonl", filters=CorpusFilters(), as_of=as_of)


def test_config_boundary_canonicalizes_z_spelling(tmp_path):
    assert _cfg(tmp_path, "2026-04-01T00:00:00Z").as_of == "2026-04-01T00:00:00+00:00"


def test_config_boundary_is_idempotent_for_canonical_input(tmp_path):
    assert _cfg(tmp_path, AS_OF).as_of == AS_OF


def test_config_boundary_rejects_non_utc_offset(tmp_path):
    with pytest.raises(ValueError, match="must be a UTC timestamp"):
        _cfg(tmp_path, "2026-04-01T05:00:00+05:00")


def test_config_boundary_rejects_naive_timestamp(tmp_path):
    with pytest.raises(ValueError, match="must be a UTC timestamp"):
        _cfg(tmp_path, "2026-04-01T00:00:00")


def test_config_boundary_rejects_unparseable_timestamp(tmp_path):
    with pytest.raises(ValueError, match="not a valid ISO-8601"):
        _cfg(tmp_path, "yesterday-ish")
