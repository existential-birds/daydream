# tests/test_cli_help_tiers.py
"""Tests for the two-tier help surface (``--help`` vs ``--help-all``)."""

import pytest

from daydream.cli import _parse_args


def test_default_help_hides_advanced(capsys):
    with pytest.raises(SystemExit):
        _parse_args(["--help"])
    out = capsys.readouterr().out
    assert "--comment" in out and "--start-at" not in out and "--ignore-path" not in out
    assert "--findings-out" not in out and "--pr-number" not in out
    assert "--precision" not in out  # #232: opt-in precision mode is an advanced flag
    assert "--approve-on-clean" not in out  # #343: opt-in auto-approval is an advanced flag


def test_help_all_shows_advanced(capsys):
    with pytest.raises(SystemExit):
        _parse_args(["--help-all"])
    out = capsys.readouterr().out
    assert "--start-at" in out
    assert "--findings-out" in out and "--pr-number" in out
    assert "--precision" in out  # #232: reachable from --help-all
    assert "--approve-on-clean" in out  # #343: reachable from --help-all
    assert "--log" in out                # #438: --log is an advanced flag
    assert "redacted agent events" in out   # #438: exact phrase
    assert "raw agent events" not in out    # #438: raw wording removed


def test_advanced_flags_still_parse():
    assert _parse_args(["--start-at", "fix", "/t"]).start_at == "fix"


def test_precision_flag_activates_precision_mode():
    """#232: ``--precision`` is the activation path into RunConfig.precision_mode.

    Absent the flag the field stays ``False`` (byte-identical default); with it,
    the field is ``True`` so the deep orchestrator's ``_precision_mode`` resolver
    runs the suppression pass.
    """
    assert _parse_args(["/t"]).precision_mode is False
    assert _parse_args(["--precision", "/t"]).precision_mode is True
