# tests/test_cli_help_tiers.py
"""Tests for the two-tier help surface (``--help`` vs ``--help-all``)."""

import pytest

from daydream.cli import _parse_args
from daydream.runner import RunConfig


def _parse_argv_for_test(argv: list[str]) -> RunConfig:
    """Drive the real CLI parser with an explicit argv (no sys.argv mutation)."""
    return _parse_args(argv)


def _cfg(argv: list[str]) -> RunConfig:
    """Parse ``daydream <argv>`` into a RunConfig via the real CLI parser."""
    return _parse_args(argv)


def test_default_help_hides_advanced(capsys):
    with pytest.raises(SystemExit):
        _parse_argv_for_test(["--help"])
    out = capsys.readouterr().out
    assert "--comment" in out and "--start-at" not in out and "--ignore-path" not in out
    assert "--findings-out" not in out and "--pr-number" not in out
    assert "--precision" not in out  # #232: opt-in precision mode is an advanced flag


def test_help_all_shows_advanced(capsys):
    with pytest.raises(SystemExit):
        _parse_argv_for_test(["--help-all"])
    out = capsys.readouterr().out
    assert "--start-at" in out
    assert "--findings-out" in out and "--pr-number" in out
    assert "--precision" in out  # #232: reachable from --help-all


def test_advanced_flags_still_parse():
    assert _cfg(["--start-at", "fix", "/t"]).start_at == "fix"


def test_precision_flag_activates_precision_mode():
    """#232: ``--precision`` is the activation path into RunConfig.precision_mode.

    Absent the flag the field stays ``False`` (byte-identical default); with it,
    the field is ``True`` so the deep orchestrator's ``_precision_mode`` resolver
    runs the suppression pass.
    """
    assert _cfg(["/t"]).precision_mode is False
    assert _cfg(["--precision", "/t"]).precision_mode is True
