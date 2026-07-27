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


@pytest.mark.parametrize(
    ("first_valid_at", "first_expected", "second_valid_at", "second_expected"),
    [
        pytest.param(
            "2026-04-01T00:00:00.000001+00:00",
            True,
            "2026-03-31T23:59:59.999999+00:00",
            False,
            id="strict-boundary",
        ),
        pytest.param(
            "2026-04-01T05:00:00+05:00",
            False,
            "2026-04-01T05:00:01+05:00",
            True,
            id="non-utc-offset",
        ),
    ],
)
def test_leak_guard_chronological_comparison(
    first_valid_at: str,
    first_expected: bool,
    second_valid_at: str,
    second_expected: bool,
) -> None:
    assert _is_posterior_leak(_ann(first_valid_at), AS_OF) is first_expected
    assert _is_posterior_leak(_ann(second_valid_at), AS_OF) is second_expected


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


# as_of entry boundary: BuildCorpusConfig validates and canonicalizes ONCE,
# before the pin reaches the SQL cutoff or the leak guard.


def _cfg(tmp_path: Path, as_of: str) -> BuildCorpusConfig:
    return BuildCorpusConfig(out_path=tmp_path / "c.jsonl", filters=CorpusFilters(), as_of=as_of)


@pytest.mark.parametrize(
    ("as_of", "expected"),
    [
        pytest.param("2026-04-01T00:00:00Z", AS_OF, id="z-spelling"),
        pytest.param(AS_OF, AS_OF, id="already-canonical"),
    ],
)
def test_config_boundary_canonicalizes_timestamp(tmp_path: Path, as_of: str, expected: str) -> None:
    assert _cfg(tmp_path, as_of).as_of == expected


@pytest.mark.parametrize(
    ("as_of", "error"),
    [
        pytest.param(
            "2026-04-01T05:00:00+05:00",
            "must be a UTC timestamp",
            id="non-utc-offset",
        ),
        pytest.param(
            "2026-04-01T00:00:00",
            "must be a UTC timestamp",
            id="naive",
        ),
        pytest.param(
            "yesterday-ish",
            "not a valid ISO-8601",
            id="unparseable",
        ),
    ],
)
def test_config_boundary_rejects_invalid_timestamp(
    tmp_path: Path,
    as_of: str,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        _cfg(tmp_path, as_of)
