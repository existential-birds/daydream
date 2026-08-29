"""Tests for daydream.ui.summary severity rendering policy (issue #972 R3)."""

from typing import Any

from daydream.ui import summary


def test_summary_missing_severity_is_none_not_medium() -> None:
    """R3.1: missing severity renders as absent (``None``), never a fabricated
    ``"medium"`` — the unified fallback policy at the UI boundary."""
    assert summary._display_severity({}) is None


def test_summary_known_severity_normalizes() -> None:
    assert summary._display_severity({"severity": "HIGH"}) == "high"


def test_summary_unknown_severity_is_none() -> None:
    assert summary._display_severity({"severity": "bogus"}) is None


def test_display_severity_is_none_safe() -> None:
    issue: dict[str, Any] = {"severity": None}
    assert summary._display_severity(issue) is None
