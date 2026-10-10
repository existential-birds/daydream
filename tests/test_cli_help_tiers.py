"""Tests for the two-tier help surface (``--help`` vs ``--help-all``)."""


import pytest

from daydream.commands.improve import _build_improve_parser, _parse_improve_args
from daydream.commands.review import _parse_args


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
