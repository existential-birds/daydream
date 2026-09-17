"""Tests for the two-tier help surface (``--help`` vs ``--help-all``)."""


import pytest

from daydream.cli import _build_improve_parser, _parse_args, _parse_improve_args


def test_advanced_flags_still_parse() -> None:
    assert _parse_args(["--start-at", "fix", "/t"]).start_at == "fix"


def test_precision_flag_activates_precision_mode() -> None:
    """#232: ``--precision`` is the activation path into RunConfig.precision_mode.

    Absent the flag the field stays ``False`` (byte-identical default); with it,
    the field is ``True`` so the deep orchestrator's ``_precision_mode`` resolver
    runs the suppression pass.
    """
    assert _parse_args(["/t"]).precision_mode is False
    assert _parse_args(["--precision", "/t"]).precision_mode is True


def test_diagram_flags_parse_from_both_tiers() -> None:
    """#1113: both flags reach ``RunConfig`` through the production parser."""
    assert _parse_args(["--diagram", "off", "/t"]).diagram == "off"
    only = _parse_args(["--diagram-only", "both", "/t"])
    assert only.diagram == "both"
    assert only.output_mode == "diagram"


def test_verbose_flag_activates_log_mode() -> None:
    assert _parse_args(["/t"]).log_mode is False
    assert _parse_args(["--verbose", "/t"]).log_mode is True


def test_verbose_flag_activates_log_mode_improve() -> None:
    assert _parse_improve_args(["improve", "/t"]).log_mode is False
    assert _parse_improve_args(["improve", "/t", "--verbose"]).log_mode is True


def test_plain_help_hides_both_diagnostic_flags(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        _parse_args(["--help"])
    out = capsys.readouterr().out
    assert "--log" not in out
    assert "verbose" not in out


def test_help_all_shows_verbose_and_not_log(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        _parse_args(["--help-all"])
    out = capsys.readouterr().out
    assert "--verbose" in out
    assert "--log" not in out


def test_improve_help_shows_verbose_and_not_log(capsys: pytest.CaptureFixture[str]) -> None:
    _build_improve_parser().print_help()
    out = capsys.readouterr().out
    assert "--verbose" in out
    assert "--log" not in out
