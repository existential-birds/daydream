"""Unit tests for the corpus projection's temporal-leakage guard.

Exercises the ``_is_posterior_leak`` boundary semantics in isolation: an
annotation whose outcome only became true *after* the ``as_of`` pin must not
leak its posterior-derived ``outcome_label`` into a corpus pinned to that
``as_of``. The guard compares parsed datetimes chronologically, so ``Z``/
``+00:00`` spellings, sub-second precision, and non-UTC offsets can never
mis-order it.

(#1093: the legacy emission path these guards fed is deleted; the guard
itself is canonical shared infrastructure for the corpus projection.)
"""
from __future__ import annotations

from typing import Any

import pytest

from daydream.training.corpus import _is_posterior_leak

# _is_posterior_leak boundary semantics (unit): valid_at == as_of is in-time,
# strictly greater is a leak, and no ISO-8601 spelling difference mis-orders it.

AS_OF = "2026-04-01T00:00:00+00:00"


def _ann(valid_at: str | None) -> dict[str, Any]:
    return {"valid_at": valid_at}


def test_leak_guard_equal_instant_is_not_a_leak() -> None:
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
    """Compare posterior timestamps across offsets and subsecond precision."""
    assert _is_posterior_leak(_ann(first_valid_at), AS_OF) is first_expected
    assert _is_posterior_leak(_ann(second_valid_at), AS_OF) is second_expected


def test_leak_guard_none_inputs_never_leak() -> None:
    assert _is_posterior_leak(None, AS_OF) is False
    assert _is_posterior_leak(_ann(None), AS_OF) is False
    assert _is_posterior_leak(_ann("2026-09-01T00:00:00+00:00"), None) is False


def test_leak_guard_mixed_z_and_offset_spellings_compare_chronologically() -> None:
    # Same instant spelled "Z" vs "+00:00", both directions: never a leak.
    assert _is_posterior_leak(_ann("2026-04-01T00:00:00Z"), AS_OF) is False
    assert _is_posterior_leak(_ann(AS_OF), "2026-04-01T00:00:00Z") is False
    # One second later, spelled "Z": still detected as a leak.
    assert _is_posterior_leak(_ann("2026-04-01T00:00:01Z"), AS_OF) is True


def test_leak_guard_subsecond_precision_compares_chronologically() -> None:
    # ".000000" and no-fraction are the same instant — not a leak in either
    # direction. (A suffix-only lexical normalisation would have called the
    # fractional spelling "greater" and leaked a false exclusion.)
    assert _is_posterior_leak(_ann("2026-04-01T00:00:00.000000Z"), AS_OF) is False
    assert _is_posterior_leak(_ann("2026-04-01T00:00:00.000000+00:00"), AS_OF) is False
    # Half a second after the pin is a leak; half a second before is not.
    assert _is_posterior_leak(_ann("2026-04-01T00:00:00.500000+00:00"), "2026-04-01T00:00:00Z") is True
    assert _is_posterior_leak(_ann("2026-04-01T00:00:00Z"), "2026-04-01T00:00:00.500000+00:00") is False
