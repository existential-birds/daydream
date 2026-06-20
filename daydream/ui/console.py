"""Console construction and phase-hero banner components.

Themed ``Console`` factory, the gradient interpolation helpers, and the
pyfiglet phase-hero banner.
"""

import pyfiglet
from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.panel import Panel
from rich.style import Style
from rich.text import Text

from daydream.ui.theme import (
    ASCII_GRADIENT_COLORS,
    NEON_COLORS,
    NEON_THEME,
)


def create_console() -> Console:
    """Create a Rich Console with neon theme applied.

    Returns:
        Console: A new Rich Console instance configured with the neon theme.

    """
    return Console(theme=NEON_THEME)


def _interpolate_color(color1: str, color2: str, t: float) -> str:
    """Interpolate between two hex colors.

    Args:
        color1: Starting hex color (e.g., "#8BE9FD").
        color2: Ending hex color (e.g., "#FF79C6").
        t: Interpolation factor (0.0 = color1, 1.0 = color2).

    Returns:
        Interpolated hex color string.

    """
    r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
    r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def _get_gradient_color(position: float) -> str:
    """Get a color from the gradient based on position (0.0 to 1.0).

    The gradient smoothly transitions: cyan -> pink -> purple.

    Args:
        position: Position in gradient (0.0 = start, 1.0 = end).

    Returns:
        Hex color string at the given position.

    """
    position = max(0.0, min(1.0, position))

    index = position * (len(ASCII_GRADIENT_COLORS) - 1)
    lower_idx = int(index)
    upper_idx = min(lower_idx + 1, len(ASCII_GRADIENT_COLORS) - 1)

    t = index - lower_idx
    return _interpolate_color(
        ASCII_GRADIENT_COLORS[lower_idx],
        ASCII_GRADIENT_COLORS[upper_idx],
        t,
    )


def print_phase_hero(
    console: Console,
    title: str,
    description: str,
) -> None:
    """Print a visually striking ASCII art banner with neon gradient.

    Uses pyfiglet to generate ASCII art and applies a horizontal
    gradient from cyan -> pink -> purple character-by-character.
    Includes a decorative subtitle below the ASCII art.

    Args:
        console: Rich Console instance for output.
        title: The ASCII art text (e.g., "DAYDREAM", "BREATHE", "HEAL").
        description: Subtitle text displayed below the ASCII art.

    """
    # Generate ASCII art using pyfiglet with 'ansi_shadow' font for readability
    try:
        ascii_art = pyfiglet.figlet_format(title, font="ansi_shadow")
    except pyfiglet.FigletError:
        ascii_art = pyfiglet.figlet_format(title, font="standard")

    lines = ascii_art.rstrip("\n").split("\n")

    max_width = max(len(line) for line in lines) if lines else 1

    gradient_text = Text()

    for line_idx, line in enumerate(lines):
        if not line.strip():
            gradient_text.append(line + "\n")
            continue

        for char_idx, char in enumerate(line):
            if char == " ":
                gradient_text.append(char)
            else:
                position = char_idx / max_width if max_width > 0 else 0
                color = _get_gradient_color(position)
                gradient_text.append(char, style=Style(color=color, bold=True))

        if line_idx < len(lines) - 1:
            gradient_text.append("\n")

    tagline = Text()
    tagline.append("~", style=Style(color=NEON_COLORS["purple"], dim=True))
    tagline.append(f" {description} ", style=Style(color=NEON_COLORS["pink"], dim=True, italic=True))
    tagline.append("~", style=Style(color=NEON_COLORS["purple"], dim=True))

    full_content = Group(
        Align.center(gradient_text),
        Align.center(tagline),
    )

    panel = Panel(
        full_content,
        box=box.DOUBLE_EDGE,
        border_style=Style(color=NEON_COLORS["purple"], dim=True),
        padding=(0, 2),
    )

    console.print()
    console.print(panel)
