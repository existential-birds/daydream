"""Posterior-derived labels must not leak into a corpus pinned before their outcome.

The shared projection guard compares parsed datetimes, including equivalent zone
spellings, fractional seconds, and non-UTC offsets.
"""
from __future__ import annotations

from collections.abc import Sequence
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

def test_leak_guard_chronological_comparison(subtests: pytest.Subtests) -> None:
    """Compare posterior timestamps across offsets and subsecond precision."""
    cases: Sequence[tuple[str, bool, str, bool]] = [
        ("2026-04-01T00:00:00.000001+00:00", True, "2026-03-31T23:59:59.999999+00:00", False),
        ("2026-04-01T05:00:00+05:00", False, "2026-04-01T05:00:01+05:00", True),
    ]
    case_ids_0 = ["strict-boundary", "non-utc-offset"]
    for case_index, (first_valid_at, first_expected, second_valid_at, second_expected) in enumerate(cases):
        with subtests.test(
            msg=case_ids_0[case_index],
            first_valid_at=first_valid_at,
            first_expected=first_expected,
            second_valid_at=second_valid_at,
            second_expected=second_expected,
        ):
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
