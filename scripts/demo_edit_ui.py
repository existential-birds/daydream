#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "rich>=13.0.0",
# ]
# ///
"""Demo script showcasing 4 psychedelic/dreamlike UI concepts for Edit tool visualization.

Usage:
    uv run scripts/demo_edit_ui.py

Each visualization presents a different aesthetic for showing code transformations,
drawing from themes of metamorphosis, ritual magic, dream surgery, and cosmic patterns.
"""

import random
import time
from typing import Callable

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.style import Style
from rich.table import Table
from rich.text import Text

# =============================================================================
# Color Theme (Dracula-based, matching daydream/ui.py)
# =============================================================================

NEON_COLORS = {
    "background": "#282A36",
    "foreground": "#F8F8F2",
    "red": "#FF5555",
    "green": "#50FA7B",
    "yellow": "#F1FA8C",
    "purple": "#BD93F9",
    "pink": "#FF79C6",
    "cyan": "#8BE9FD",
    "orange": "#FFB86C",
}

# Transformation gradient: red -> purple -> cyan -> green
METAMORPHOSIS_GRADIENT = [
    "#FF5555",  # red
    "#E05575",
    "#C15595",
    "#A255B5",
    "#8355D5",
    "#7465E5",
    "#6575F5",
    "#5695E5",
    "#47B5D5",
    "#38D5C5",
    "#50E5A5",
    "#50FA7B",  # green
]

# Fire colors for ritual burning
FIRE_COLORS = ["#FF5555", "#FF7744", "#FFB86C", "#F1FA8C", "#FFFFFF"]

# Ethereal glow colors
ETHEREAL_COLORS = ["#8BE9FD", "#BD93F9", "#FF79C6"]

# Star twinkle characters
STAR_CHARS = [".", "*", "+", "x", "*", ".", "+"]
CONSTELLATION_CHARS = [" ", ".", "*", "+", "x", "X", "#", "X", "x", "+", "*", ".", " "]

# Mystical runes and glyphs
RUNES = "ᚠᚢᚦᚨᚱᚲᚷᚹᚺᚾᛁᛃᛇᛈᛉᛊᛋᛏᛒᛖᛗᛚᛜᛞᛟ"
MOON_PHASES = "☽★☾✦✧⋆"
MYSTICAL_SYMBOLS = "◈◇◆●○◐◑☯⚛⚡✺✹✸✷✶✵"

# Chakra/Energy symbols
CHAKRA_SYMBOLS = ["◉", "◎", "●", "○", "◐", "◑", "◒"]
ENERGY_FLOW = ["~", "≈", "∿", "〜", "⌇", "⌁", "⚡"]

console = Console()


def interpolate_color(color1: str, color2: str, t: float) -> str:
    """Interpolate between two hex colors."""
    r1, g1, b1 = int(color1[1:3], 16), int(color1[3:5], 16), int(color1[5:7], 16)
    r2, g2, b2 = int(color2[1:3], 16), int(color2[3:5], 16), int(color2[5:7], 16)
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def get_metamorphosis_color(progress: float) -> str:
    """Get color from metamorphosis gradient based on transformation progress."""
    progress = max(0.0, min(1.0, progress))
    index = progress * (len(METAMORPHOSIS_GRADIENT) - 1)
    lower_idx = int(index)
    upper_idx = min(lower_idx + 1, len(METAMORPHOSIS_GRADIENT) - 1)
    t = index - lower_idx
    return interpolate_color(METAMORPHOSIS_GRADIENT[lower_idx], METAMORPHOSIS_GRADIENT[upper_idx], t)


def print_option_header(number: int, name: str, description: str) -> None:
    """Print a styled header for each visualization option."""
    console.print()
    console.print()

    header_text = Text()
    header_text.append(f"  OPTION {number}  ", style=Style(color="black", bgcolor=NEON_COLORS["pink"], bold=True))
    header_text.append(f"  {name}", style=Style(color=NEON_COLORS["cyan"], bold=True))

    console.print(Panel(
        Group(
            Align.center(header_text),
            Text(),
            Align.center(Text(description, style=Style(color=NEON_COLORS["purple"], italic=True))),
        ),
        box=box.DOUBLE_EDGE,
        border_style=Style(color=NEON_COLORS["pink"]),
        padding=(1, 2),
    ))
    console.print()
    time.sleep(0.5)


# =============================================================================
# Option 1: Metamorphosis Visualization
# =============================================================================

def demo_metamorphosis(old_string: str, new_string: str, duration: float = 3.0) -> None:
    """Chrysalis/butterfly transformation - old code dissolves while new code crystallizes."""

    print_option_header(
        1,
        "METAMORPHOSIS",
        "The old code dissolves character by character while new code crystallizes"
    )

    # Dissolution characters representing decay states
    dissolution_chars = ["█", "▓", "▒", "░", "·", " "]
    crystallization_chars = [" ", "·", "░", "▒", "▓", "█"]

    steps = 30
    step_duration = duration / steps

    def render_frame(frame: int) -> Panel:
        progress = frame / steps

        # Create the transformation display
        content = Text()

        # Title with progress indicator
        progress_bar_width = 40
        filled = int(progress * progress_bar_width)
        bar = "█" * filled + "░" * (progress_bar_width - filled)

        content.append("  TRANSFORMATION PROGRESS\n", style=Style(color=NEON_COLORS["purple"], bold=True))
        content.append(f"  [{bar}] {int(progress * 100):3d}%\n\n", style=Style(color=get_metamorphosis_color(progress)))

        # OLD STRING - dissolving
        content.append("  DISSOLVING:\n", style=Style(color=NEON_COLORS["red"], bold=True))
        content.append("  ")

        for i, char in enumerate(old_string):
            char_progress = (progress * len(old_string) - i) / 3
            char_progress = max(0, min(1, char_progress))

            if char_progress >= 1:
                # Fully dissolved
                content.append(" ", style=Style(color=NEON_COLORS["red"]))
            else:
                # Dissolving - pick appropriate decay character
                decay_idx = int(char_progress * (len(dissolution_chars) - 1))
                if char == " ":
                    content.append(" ")
                else:
                    color = interpolate_color(NEON_COLORS["red"], "#282A36", char_progress)
                    content.append(dissolution_chars[decay_idx], style=Style(color=color))

        content.append("\n\n")

        # NEW STRING - crystallizing
        content.append("  CRYSTALLIZING:\n", style=Style(color=NEON_COLORS["green"], bold=True))
        content.append("  ")

        for i, char in enumerate(new_string):
            char_progress = (progress * len(new_string) - i + len(new_string) * 0.3) / 3
            char_progress = max(0, min(1, char_progress))

            if char_progress >= 1:
                # Fully crystallized
                color = get_metamorphosis_color(1.0)
                content.append(char, style=Style(color=color, bold=True))
            elif char_progress > 0:
                # Crystallizing
                crystal_idx = int(char_progress * (len(crystallization_chars) - 1))
                color = get_metamorphosis_color(char_progress)
                if char == " ":
                    content.append(" ")
                else:
                    content.append(crystallization_chars[crystal_idx], style=Style(color=color))
            else:
                content.append(" ")

        content.append("\n\n")

        # Morphing state indicator
        if progress < 0.3:
            state = "~ chrysalis forming ~"
            state_color = NEON_COLORS["red"]
        elif progress < 0.7:
            state = "~ metamorphosis in progress ~"
            state_color = NEON_COLORS["purple"]
        else:
            state = "~ emergence complete ~"
            state_color = NEON_COLORS["green"]

        content.append(f"  {state}", style=Style(color=state_color, italic=True))

        return Panel(
            content,
            title="[bold cyan]METAMORPHOSIS[/bold cyan]",
            border_style=Style(color=get_metamorphosis_color(progress)),
            box=box.ROUNDED,
            padding=(1, 2),
        )

    with Live(render_frame(0), refresh_per_second=20, console=console) as live:
        for frame in range(steps + 1):
            live.update(render_frame(frame))
            time.sleep(step_duration)


# =============================================================================
# Option 2: Ritual/Spell Casting
# =============================================================================

def demo_ritual(old_string: str, new_string: str, duration: float = 3.0) -> None:
    """Frame the edit as a magical incantation with mystical glyphs."""

    print_option_header(
        2,
        "RITUAL CASTING",
        "A magical incantation where old code burns away and new code manifests"
    )

    steps = 40
    step_duration = duration / steps

    def render_frame(frame: int) -> Panel:
        progress = frame / steps

        content = Text()

        # Spinning rune circle
        rune_circle_size = 12
        rune_offset = int(frame * 0.5) % len(RUNES)

        content.append("  ")
        for i in range(rune_circle_size):
            rune_idx = (rune_offset + i) % len(RUNES)
            color = ETHEREAL_COLORS[i % len(ETHEREAL_COLORS)]
            content.append(RUNES[rune_idx], style=Style(color=color, bold=True))
            content.append(" ")
        content.append("\n\n")

        # Incantation phase indicator
        if progress < 0.25:
            phase = "INVOCATION"
            phase_symbol = "☽"
            phase_color = NEON_COLORS["purple"]
        elif progress < 0.5:
            phase = "IMMOLATION"
            phase_symbol = "🔥"
            phase_color = NEON_COLORS["orange"]
        elif progress < 0.75:
            phase = "TRANSMUTATION"
            phase_symbol = "✦"
            phase_color = NEON_COLORS["pink"]
        else:
            phase = "MANIFESTATION"
            phase_symbol = "★"
            phase_color = NEON_COLORS["cyan"]

        content.append(f"  {phase_symbol} {phase} {phase_symbol}\n\n", style=Style(color=phase_color, bold=True))

        # OLD STRING - burning away with fire colors
        content.append("  ☽ THE OLD WAY ☾\n", style=Style(color=NEON_COLORS["orange"], bold=True))
        content.append("  ")

        if progress < 0.5:
            burn_progress = progress * 2
            for i, char in enumerate(old_string):
                char_burn = (burn_progress * len(old_string) - i) / 4
                char_burn = max(0, min(1, char_burn))

                if char_burn >= 1:
                    # Burned away
                    content.append("·", style=Style(color="#444444"))
                elif char_burn > 0:
                    # Burning - cycle through fire colors
                    fire_idx = int((char_burn + frame * 0.1) * len(FIRE_COLORS)) % len(FIRE_COLORS)
                    content.append(char, style=Style(color=FIRE_COLORS[fire_idx], bold=True))
                else:
                    content.append(char, style=Style(color=NEON_COLORS["red"]))
        else:
            # All burned - show ashes
            content.append("·" * len(old_string), style=Style(color="#444444"))

        content.append("\n\n")

        # Mystical separator with animated symbols
        separator_width = max(len(old_string), len(new_string)) + 4
        sep_symbols = ""
        for i in range(separator_width):
            sym_idx = (frame + i) % len(MYSTICAL_SYMBOLS)
            sep_symbols += MYSTICAL_SYMBOLS[sym_idx]
        content.append(f"  {sep_symbols}\n\n", style=Style(color=NEON_COLORS["purple"]))

        # NEW STRING - manifesting with ethereal glow
        content.append("  ★ THE NEW WAY ★\n", style=Style(color=NEON_COLORS["cyan"], bold=True))
        content.append("  ")

        if progress > 0.4:
            manifest_progress = (progress - 0.4) / 0.6
            for i, char in enumerate(new_string):
                char_manifest = (manifest_progress * len(new_string) - i + len(new_string) * 0.5) / 4
                char_manifest = max(0, min(1, char_manifest))

                if char_manifest >= 1:
                    # Fully manifested with pulsing glow
                    glow_intensity = 0.5 + 0.5 * ((frame + i) % 3) / 2
                    color = interpolate_color(NEON_COLORS["cyan"], "#FFFFFF", glow_intensity * 0.3)
                    content.append(char, style=Style(color=color, bold=True))
                elif char_manifest > 0:
                    # Manifesting - ethereal shimmer
                    ethereal_idx = int((char_manifest + frame * 0.15) * len(ETHEREAL_COLORS)) % len(ETHEREAL_COLORS)
                    content.append(char if random.random() > 0.3 else "✧", style=Style(color=ETHEREAL_COLORS[ethereal_idx]))
                else:
                    content.append("·", style=Style(color="#333333"))
        else:
            # Not yet manifesting
            content.append("·" * len(new_string), style=Style(color="#333333"))

        content.append("\n\n")

        # Bottom rune circle (spinning opposite direction)
        content.append("  ")
        for i in range(rune_circle_size):
            rune_idx = (len(RUNES) - rune_offset + i) % len(RUNES)
            color = ETHEREAL_COLORS[(i + 1) % len(ETHEREAL_COLORS)]
            content.append(RUNES[rune_idx], style=Style(color=color, bold=True))
            content.append(" ")

        border_color = interpolate_color(NEON_COLORS["purple"], NEON_COLORS["cyan"], progress)

        return Panel(
            content,
            title=f"[bold pink]{MOON_PHASES[frame % len(MOON_PHASES)]} RITUAL CASTING {MOON_PHASES[(frame + 3) % len(MOON_PHASES)]}[/bold pink]",
            border_style=Style(color=border_color),
            box=box.DOUBLE,
            padding=(1, 2),
        )

    with Live(render_frame(0), refresh_per_second=15, console=console) as live:
        for frame in range(steps + 1):
            live.update(render_frame(frame))
            time.sleep(step_duration)


# =============================================================================
# Option 3: Dream Surgery
# =============================================================================

def demo_surgery(old_string: str, new_string: str, duration: float = 3.0) -> None:
    """Visualize as precise energy work on code - the file as a living organism."""

    print_option_header(
        3,
        "DREAM SURGERY",
        "The file is a living organism - we perform precise energy work at the surgical site"
    )

    steps = 35
    step_duration = duration / steps

    # Simulated file context
    before_lines = [
        "class User:",
        "    '''User model with authentication.'''",
        "    ",
    ]
    after_lines = [
        "    ",
        "    def validate(self):",
        "        return True",
    ]

    def render_frame(frame: int) -> Panel:
        progress = frame / steps

        content = Text()

        # Header with energy flow indicator
        energy_flow = ENERGY_FLOW[frame % len(ENERGY_FLOW)]
        content.append(f"  {energy_flow} ENERGY ALIGNMENT {energy_flow}\n\n", style=Style(color=NEON_COLORS["cyan"], bold=True))

        # Surgical phase
        if progress < 0.3:
            phase = "LOCATING MERIDIAN"
            phase_color = NEON_COLORS["yellow"]
        elif progress < 0.6:
            phase = "CLEARING BLOCKAGE"
            phase_color = NEON_COLORS["red"]
        elif progress < 0.85:
            phase = "CHANNELING FLOW"
            phase_color = NEON_COLORS["purple"]
        else:
            phase = "HARMONY RESTORED"
            phase_color = NEON_COLORS["green"]

        content.append(f"  [{phase}]\n\n", style=Style(color=phase_color))

        # File as organism view
        line_num = 42  # Starting line number

        # Context before
        for i, line in enumerate(before_lines):
            chakra = CHAKRA_SYMBOLS[i % len(CHAKRA_SYMBOLS)]
            content.append(f"  {chakra} ", style=Style(color="#666666"))
            content.append(f"{line_num + i:3d} ", style=Style(color=NEON_COLORS["yellow"]))
            content.append(f"{line}\n", style=Style(color="#888888"))

        # The surgical site - OLD (blocked energy / red)
        surgical_line = line_num + len(before_lines)

        # Pulsing attention indicator
        pulse = abs((frame % 10) - 5) / 5

        if progress < 0.5:
            # Show old string with blocked energy visualization
            attention_char = "▶" if frame % 2 == 0 else "►"
            chakra_color = interpolate_color(NEON_COLORS["red"], NEON_COLORS["orange"], pulse)
            content.append(f"  ", style=Style(color=chakra_color))
            content.append(attention_char, style=Style(color=NEON_COLORS["red"], bold=True))
            content.append(f" {surgical_line:3d} ", style=Style(color=NEON_COLORS["red"], bold=True))

            # Old string with "blocked" visualization
            block_intensity = 1.0 - progress * 2
            for i, char in enumerate(old_string):
                if progress > 0.25 and i < int((progress - 0.25) * 4 * len(old_string)):
                    # Being cleared
                    content.append("~", style=Style(color=NEON_COLORS["purple"]))
                else:
                    color = interpolate_color(NEON_COLORS["red"], NEON_COLORS["orange"], pulse)
                    content.append(char, style=Style(color=color, bold=True))

            # Blocked indicator
            content.append(" ⊗ BLOCKED", style=Style(color=NEON_COLORS["red"]))
        else:
            # Show new string with flowing energy visualization
            attention_char = "◆" if frame % 2 == 0 else "◇"
            chakra_color = interpolate_color(NEON_COLORS["cyan"], NEON_COLORS["green"], pulse)
            content.append(f"  ", style=Style(color=chakra_color))
            content.append(attention_char, style=Style(color=NEON_COLORS["green"], bold=True))
            content.append(f" {surgical_line:3d} ", style=Style(color=NEON_COLORS["green"], bold=True))

            # New string with "flowing" visualization
            flow_progress = (progress - 0.5) * 2
            for i, char in enumerate(new_string):
                if i < int(flow_progress * len(new_string)):
                    # Energy flowing through
                    glow = abs(((frame + i) % 6) - 3) / 3
                    color = interpolate_color(NEON_COLORS["green"], NEON_COLORS["cyan"], glow)
                    content.append(char, style=Style(color=color, bold=True))
                else:
                    content.append(char, style=Style(color="#666666"))

            if flow_progress >= 1:
                content.append(" ✓ FLOWING", style=Style(color=NEON_COLORS["green"]))
            else:
                content.append(" ⟳ ALIGNING", style=Style(color=NEON_COLORS["cyan"]))

        content.append("\n")

        # Context after
        for i, line in enumerate(after_lines):
            chakra = CHAKRA_SYMBOLS[(i + 4) % len(CHAKRA_SYMBOLS)]
            content.append(f"  {chakra} ", style=Style(color="#666666"))
            content.append(f"{surgical_line + 1 + i:3d} ", style=Style(color=NEON_COLORS["yellow"]))
            content.append(f"{line}\n", style=Style(color="#888888"))

        content.append("\n")

        # Energy meter
        meter_width = 30
        if progress < 0.5:
            blocked = int((1 - progress * 2) * meter_width)
            cleared = meter_width - blocked
            content.append("  BLOCKAGE: ", style=Style(color=NEON_COLORS["red"]))
            content.append("█" * blocked, style=Style(color=NEON_COLORS["red"]))
            content.append("░" * cleared, style=Style(color="#444444"))
        else:
            flow = int((progress - 0.5) * 2 * meter_width)
            remaining = meter_width - flow
            content.append("  FLOW:     ", style=Style(color=NEON_COLORS["green"]))
            content.append("█" * flow, style=Style(color=NEON_COLORS["green"]))
            content.append("░" * remaining, style=Style(color="#444444"))

        border_color = interpolate_color(NEON_COLORS["red"], NEON_COLORS["green"], progress)

        return Panel(
            content,
            title="[bold cyan]DREAM SURGERY[/bold cyan]",
            subtitle=f"[dim]surgical site: line {surgical_line}[/dim]",
            border_style=Style(color=border_color),
            box=box.ROUNDED,
            padding=(1, 2),
        )

    with Live(render_frame(0), refresh_per_second=15, console=console) as live:
        for frame in range(steps + 1):
            live.update(render_frame(frame))
            time.sleep(step_duration)


# =============================================================================
# Option 4: Cosmic Rewriting
# =============================================================================

def demo_cosmic(old_string: str, new_string: str, duration: float = 3.0) -> None:
    """Display edit as constellation patterns being rearranged among the stars."""

    print_option_header(
        4,
        "COSMIC REWRITING",
        "Characters become stars that shift position as constellations rearrange"
    )

    steps = 40
    step_duration = duration / steps

    # Star field characters
    star_brightness = [" ", ".", "·", "+", "*", "✦", "✧", "⋆", "★"]

    # Create a starfield
    field_width = 50
    field_height = 12

    def generate_starfield(frame: int) -> list[list[str]]:
        """Generate a twinkling starfield."""
        field = []
        random.seed(42)  # Consistent base pattern
        for y in range(field_height):
            row = []
            for x in range(field_width):
                if random.random() < 0.08:
                    # Twinkle effect
                    brightness = (random.randint(0, 3) + frame) % len(star_brightness)
                    row.append(star_brightness[brightness])
                else:
                    row.append(" ")
            field.append(row)
        return field

    def render_frame(frame: int) -> Panel:
        progress = frame / steps

        content = Text()

        # Phase indicator
        if progress < 0.3:
            phase = "~ mapping old constellation ~"
            phase_color = NEON_COLORS["purple"]
        elif progress < 0.7:
            phase = "~ stars in transit ~"
            phase_color = NEON_COLORS["pink"]
        else:
            phase = "~ new constellation emerges ~"
            phase_color = NEON_COLORS["cyan"]

        content.append(f"  {phase}\n\n", style=Style(color=phase_color, italic=True))

        # Generate starfield
        field = generate_starfield(frame)

        # Calculate positions for old and new strings
        old_y = 3
        new_y = 8
        old_x_start = (field_width - len(old_string)) // 2
        new_x_start = (field_width - len(new_string)) // 2

        # Place old string characters (fading)
        old_opacity = max(0, 1 - progress * 1.5)
        if old_opacity > 0:
            for i, char in enumerate(old_string):
                if char != " ":
                    x = old_x_start + i
                    # Stars drift upward as they fade
                    drift = int(progress * 3)
                    y = max(0, old_y - drift)
                    if 0 <= y < field_height and 0 <= x < field_width:
                        field[y][x] = char

        # Place new string characters (emerging)
        new_opacity = max(0, (progress - 0.3) / 0.7)
        if new_opacity > 0:
            for i, char in enumerate(new_string):
                if char != " ":
                    x = new_x_start + i
                    # Stars drift down into position
                    start_y = 0
                    drift = int(new_opacity * (new_y - start_y))
                    y = min(field_height - 1, start_y + drift)
                    if 0 <= y < field_height and 0 <= x < field_width:
                        field[y][x] = char

        # Render the starfield
        for y, row in enumerate(field):
            content.append("  ")
            for x, char in enumerate(row):
                # Determine if this is part of old or new string
                is_old_char = (
                    y == max(0, old_y - int(progress * 3)) and
                    old_x_start <= x < old_x_start + len(old_string) and
                    old_string[x - old_x_start] != " " and
                    old_opacity > 0
                )

                is_new_char = (
                    y == min(field_height - 1, int(max(0, (progress - 0.3) / 0.7) * new_y)) and
                    new_x_start <= x < new_x_start + len(new_string) and
                    new_string[x - new_x_start] != " " and
                    new_opacity > 0
                )

                if is_old_char:
                    # Dying star - fading red/purple
                    fade_color = interpolate_color(NEON_COLORS["red"], "#333333", progress * 1.5)
                    content.append(char, style=Style(color=fade_color, bold=old_opacity > 0.5))
                elif is_new_char:
                    # New star - brightening cyan/white
                    bright_progress = (frame + x) % 6 / 6
                    bright_color = interpolate_color(NEON_COLORS["cyan"], "#FFFFFF", bright_progress * new_opacity)
                    content.append(char, style=Style(color=bright_color, bold=True))
                elif char in star_brightness[3:]:
                    # Background star - twinkle
                    twinkle = (frame + x + y) % 4
                    colors = ["#444444", "#666666", "#888888", "#AAAAAA"]
                    content.append(char, style=Style(color=colors[twinkle]))
                else:
                    content.append(char, style=Style(color="#333333"))
            content.append("\n")

        content.append("\n")

        # Constellation labels
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style=Style(color=NEON_COLORS["red"]))
        table.add_column(style=Style(color=NEON_COLORS["purple"]))
        table.add_column(style=Style(color=NEON_COLORS["cyan"]))

        old_label = f"✧ {old_string} ✧" if old_opacity > 0.3 else "  (faded)  "
        new_label = f"★ {new_string} ★" if new_opacity > 0.5 else "  (forming)  "

        table.add_row(
            old_label if old_opacity > 0.3 else Text("(faded)", style=Style(color="#444444")),
            "→" if 0.3 < progress < 0.8 else "·",
            new_label if new_opacity > 0.5 else Text("(forming)", style=Style(color="#444444")),
        )

        content.append("  ")

        border_color = interpolate_color(NEON_COLORS["purple"], NEON_COLORS["cyan"], progress)

        return Panel(
            Group(content, Align.center(table)),
            title="[bold purple]★ COSMIC REWRITING ★[/bold purple]",
            border_style=Style(color=border_color),
            box=box.DOUBLE,
            padding=(1, 2),
        )

    with Live(render_frame(0), refresh_per_second=12, console=console) as live:
        for frame in range(steps + 1):
            live.update(render_frame(frame))
            time.sleep(step_duration)


# =============================================================================
# Main Demo Runner
# =============================================================================

def main() -> None:
    """Run all four visualization demos."""

    # Sample edit to demonstrate
    old_string = "def hello():"
    new_string = "def greet(name: str):"

    # Welcome banner
    console.print()
    console.print(Panel(
        Group(
            Align.center(Text("EDIT TOOL VISUALIZATIONS", style=Style(color=NEON_COLORS["pink"], bold=True))),
            Text(),
            Align.center(Text("Four psychedelic concepts for transforming code", style=Style(color=NEON_COLORS["purple"]))),
            Text(),
            Align.center(Text(f'"{old_string}" → "{new_string}"', style=Style(color=NEON_COLORS["cyan"]))),
        ),
        box=box.DOUBLE_EDGE,
        border_style=Style(color=NEON_COLORS["pink"]),
        padding=(1, 4),
    ))

    time.sleep(1)

    # Run each demo
    demo_metamorphosis(old_string, new_string, duration=3.0)
    time.sleep(1)

    demo_ritual(old_string, new_string, duration=3.0)
    time.sleep(1)

    demo_surgery(old_string, new_string, duration=3.0)
    time.sleep(1)

    demo_cosmic(old_string, new_string, duration=3.0)
    time.sleep(1)

    # Farewell
    console.print()
    console.print(Panel(
        Align.center(Text("~ the code has been transformed ~", style=Style(color=NEON_COLORS["green"], italic=True))),
        box=box.ROUNDED,
        border_style=Style(color=NEON_COLORS["green"]),
        padding=(1, 4),
    ))
    console.print()


if __name__ == "__main__":
    main()
