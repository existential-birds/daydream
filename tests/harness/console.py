"""Console-capture helpers for tests asserting operator-visible sentences."""

from __future__ import annotations

import pytest

# Box-drawing glyphs rich emits as panel borders and separators.
_PANEL_GLYPHS = "│╭╮╰╯─║╔╗╚╝═"


def collapse_panel_text(capsys: pytest.CaptureFixture[str]) -> str:
    """Captured stdout with rich panel framing and line wrapping normalized away.

    ``print_warning`` renders inside a bordered panel, so a wrapped message
    arrives with box-drawing gutters and newlines spliced into the middle of
    it. Dropping the border glyphs and collapsing whitespace lets a test assert
    the sentence the operator reads rather than the width it happened to wrap at.
    """
    out = capsys.readouterr().out
    return " ".join(out.translate({ord(char): " " for char in _PANEL_GLYPHS}).split())
