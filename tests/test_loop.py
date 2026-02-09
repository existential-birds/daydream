"""Tests for continuous loop mode."""

import pytest

from daydream.runner import RunConfig


def test_runconfig_loop_defaults():
    config = RunConfig()
    assert config.loop is False
    assert config.max_iterations == 5


import re
from io import StringIO

from rich.console import Console

from daydream.ui import NEON_THEME, print_iteration_divider

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE.sub("", text)


def test_print_iteration_divider():
    output = StringIO()
    test_console = Console(file=output, force_terminal=True, width=80, theme=NEON_THEME)
    print_iteration_divider(test_console, 2, 5)
    plain = strip_ansi(output.getvalue())
    assert "Iteration 2 of 5" in plain
    assert "━" in plain


from daydream.ui import SummaryData, print_summary


def test_summary_data_loop_fields_default():
    data = SummaryData(
        skill="python", target="/tmp", feedback_count=3,
        fixes_applied=3, test_retries=0, tests_passed=True,
    )
    assert data.loop_mode is False
    assert data.iterations_used == 1


def test_summary_loop_mode_shows_iterations():
    output = StringIO()
    test_console = Console(file=output, force_terminal=True, width=80, theme=NEON_THEME)
    data = SummaryData(
        skill="python", target="/tmp", feedback_count=8,
        fixes_applied=8, test_retries=1, tests_passed=True,
        loop_mode=True, iterations_used=3,
    )
    print_summary(test_console, data)
    plain = strip_ansi(output.getvalue())
    assert "Iterations" in plain
    assert "3" in plain
