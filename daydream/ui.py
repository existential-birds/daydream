"""Neon terminal UI components for review_fix_loop.py.

Implements a 1980s neon terminal aesthetic using the Rich library,
with a Dracula-based color theme and animated elements.
"""

import random
import re
import time
from dataclasses import dataclass
from typing import Any

import pyfiglet
from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.style import Style
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# =============================================================================
# Color Theme (Dracula-based)
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

NEON_THEME = Theme({
    "neon.bg": NEON_COLORS["background"],
    "neon.fg": NEON_COLORS["foreground"],
    "neon.red": NEON_COLORS["red"],
    "neon.green": NEON_COLORS["green"],
    "neon.yellow": NEON_COLORS["yellow"],
    "neon.purple": NEON_COLORS["purple"],
    "neon.pink": NEON_COLORS["pink"],
    "neon.cyan": NEON_COLORS["cyan"],
    "neon.orange": NEON_COLORS["orange"],
    # Semantic styles
    "neon.error": f"bold {NEON_COLORS['red']}",
    "neon.success": f"bold {NEON_COLORS['green']}",
    "neon.warning": f"bold {NEON_COLORS['yellow']}",
    "neon.info": NEON_COLORS["cyan"],
    "neon.dim": f"dim {NEON_COLORS['foreground']}",
    "neon.tool": f"bold {NEON_COLORS['pink']}",
    "neon.path": NEON_COLORS["cyan"],
    "neon.number": NEON_COLORS["yellow"],
    "neon.string": NEON_COLORS["orange"],
})

# =============================================================================
# Reusable Style Constants
# =============================================================================
# Pre-defined Style objects for use with Text.append() and other Rich components
# that require Style objects rather than theme strings.

STYLE_CYAN = Style(color=NEON_COLORS["cyan"])
STYLE_PURPLE = Style(color=NEON_COLORS["purple"])
STYLE_PINK = Style(color=NEON_COLORS["pink"])
STYLE_GREEN = Style(color=NEON_COLORS["green"])
STYLE_YELLOW = Style(color=NEON_COLORS["yellow"])
STYLE_ORANGE = Style(color=NEON_COLORS["orange"])
STYLE_RED = Style(color=NEON_COLORS["red"])
STYLE_FG = Style(color=NEON_COLORS["foreground"])

# Bold variants
STYLE_BOLD_PINK = Style(color=NEON_COLORS["pink"], bold=True)
STYLE_BOLD_CYAN = Style(color=NEON_COLORS["cyan"], bold=True)
STYLE_BOLD_PURPLE = Style(color=NEON_COLORS["purple"], bold=True)
STYLE_BOLD_GREEN = Style(color=NEON_COLORS["green"], bold=True)
STYLE_BOLD_YELLOW = Style(color=NEON_COLORS["yellow"], bold=True)
STYLE_BOLD_RED = Style(color=NEON_COLORS["red"], bold=True)

# Panel styles
STYLE_PANEL_BG = Style(bgcolor="#1e1e2e")
STYLE_AGENT_BG = Style(bgcolor="#051208")

# Dim style
STYLE_DIM = Style(dim=True)

# Mystical action terms for tool displays
MYSTICAL_TERMS = {
    "Glob": ["scrying", "divining", "seeking", "wandering"],
    "Grep": ["channeling", "attuning", "resonating", "listening"],
    "Read": ["beholding", "absorbing", "dreaming into", "perceiving"],
    "Edit": ["healing", "realigning", "restoring flow to", "mending"],
    "Write": ["inscribing", "etching", "conjuring", "forging"],
    "Skill": ["invoking", "channeling", "summoning", "awakening"],
}

# Dream Surgery symbols for Edit tool visualization
SURGERY_CHAKRA_SYMBOLS = ["◉", "◎", "●", "○", "◐", "◑", "◒"]
SURGERY_ENERGY_FLOW = ["~", "≈", "∿", "〜", "⌇", "⌁", "⚡"]
SURGERY_PHASES = [
    ("LOCATING MERIDIAN", "yellow"),
    ("CLEARING BLOCKAGE", "red"),
    ("CHANNELING FLOW", "purple"),
    ("HARMONY RESTORED", "green"),
]


def mystical_term(tool: str) -> str:
    """Return a random mystical action term for the given tool."""
    return random.choice(MYSTICAL_TERMS.get(tool, ["processing"]))


# Dreamlike phase subtitles
PHASE_SUBTITLES = {
    "DAYDREAM": [
        "the mind wanders, the code improves",
        "where code dreams itself awake",
        "the quiet work of becoming",
        "drift into clarity",
    ],
    "REFLECT": [
        "gathering scattered thoughts",
        "seeing what surfaces",
        "listening to the echoes",
        "patterns emerge from stillness",
    ],
    "HEAL": [
        "mending what was found",
        "gentle corrections",
        "restoring harmony",
        "the wounds close softly",
    ],
    "AWAKEN": [
        "returning to waking life",
        "does the dream hold?",
        "testing the vision",
        "reality reasserts itself",
    ],
}


def phase_subtitle(phase: str) -> str:
    """Return a random dreamlike subtitle for the given phase."""
    return random.choice(PHASE_SUBTITLES.get(phase, ["..."]))


GRADIENT_COLORS = [
    "#881177",
    "#aa3355",
    "#cc6666",
    "#ee9944",
    "#eedd00",
    "#99dd55",
    "#44dd88",
    "#22ccbb",
    "#00bbcc",
    "#0099cc",
    "#3366bb",
    "#663399",
]

# Status configuration mapping
STATUS_CONFIG = {
    "pending": {"icon": "\u23f2", "color": NEON_COLORS["yellow"]},  # ⏲
    "in_progress": {"icon": "\u2b95", "color": NEON_COLORS["cyan"]},  # ⮕
    "completed": {"icon": "\u2714", "color": NEON_COLORS["green"]},  # ✔
    "failed": {"icon": "\u2717", "color": NEON_COLORS["red"]},  # ✗
}


# =============================================================================
# NeonConsole Class
# =============================================================================


class NeonConsole:
    """Wrapper around Rich Console providing themed output methods.

    This class encapsulates all the neon-styled output functionality,
    providing a consistent visual theme across all terminal output.

    Args:
        console: Optional Rich Console instance. If not provided,
            a new Console with the neon theme will be created.

    """

    def __init__(self, console: Console | None = None) -> None:
        """Initialize the NeonConsole.

        Args:
            console: Optional Rich Console instance. If not provided,
                    a new Console with the neon theme will be created.

        """
        self.console = console or Console(theme=NEON_THEME)
        self._throbber = NeonThrobber()

    def print(self, *args: Any, **kwargs: Any) -> None:
        """Pass through to underlying console.print().

        Args:
            *args: Positional arguments to pass to console.print().
            **kwargs: Keyword arguments to pass to console.print().

        Returns:
            None

        """
        self.console.print(*args, **kwargs)

    def clear(self) -> None:
        """Clear the terminal screen.

        Returns:
            None

        """
        self.console.clear()


def create_console() -> Console:
    """Create a Rich Console with neon theme applied.

    Returns:
        Console: A new Rich Console instance configured with the neon theme.

    """
    return Console(theme=NEON_THEME)


# =============================================================================
# Header Component
# =============================================================================


def print_header(console: Console, text: str) -> None:
    """Print a neon-bordered header panel.

    Creates a prominent header with pink border and purple text,
    using double-edge box styling for a retro terminal look.

    Args:
        console: Rich Console instance for output.
        text: The header text to display.

    """
    header_panel = Panel(
        Text(text, style=STYLE_BOLD_PURPLE),
        box=box.DOUBLE_EDGE,
        border_style=STYLE_PINK,
        padding=(0, 2),
    )
    console.print(header_panel)


# =============================================================================
# ASCII Art Header Component
# =============================================================================

# Gradient colors for ASCII art header (cyan -> pink -> purple)
ASCII_GRADIENT_COLORS = [
    "#8BE9FD",  # Cyan (Dracula)
    "#7DDFFA",
    "#6FD5F7",
    "#61CBF4",
    "#53C1F1",
    "#45B7EE",
    "#5AABE5",
    "#6F9FDC",
    "#8493D3",
    "#9987CA",
    "#AE7BC1",
    "#C36FB8",
    "#D863AF",
    "#ED57A6",
    "#FF79C6",  # Pink (Dracula)
    "#F07DC1",
    "#E181BC",
    "#D285B7",
    "#C389B2",
    "#B48DAD",
    "#A591A8",
    "#9695A3",
    "#87999E",
    "#A08FAB",
    "#B985B8",
    "#BD93F9",  # Purple (Dracula)
]


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
    # Clamp position
    position = max(0.0, min(1.0, position))

    # Map position to gradient array index
    index = position * (len(ASCII_GRADIENT_COLORS) - 1)
    lower_idx = int(index)
    upper_idx = min(lower_idx + 1, len(ASCII_GRADIENT_COLORS) - 1)

    # Interpolate between the two nearest colors
    t = index - lower_idx
    return _interpolate_color(
        ASCII_GRADIENT_COLORS[lower_idx],
        ASCII_GRADIENT_COLORS[upper_idx],
        t,
    )


# =============================================================================
# Phase Hero Component (ASCII Art Banners)
# =============================================================================


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
        # Fallback to standard font
        ascii_art = pyfiglet.figlet_format(title, font="standard")

    lines = ascii_art.rstrip("\n").split("\n")

    # Find the maximum line width for gradient calculation
    max_width = max(len(line) for line in lines) if lines else 1

    # Build the gradient text
    gradient_text = Text()

    for line_idx, line in enumerate(lines):
        if not line.strip():
            # Empty or whitespace-only line
            gradient_text.append(line + "\n")
            continue

        # Apply gradient character by character
        for char_idx, char in enumerate(line):
            if char == " ":
                gradient_text.append(char)
            else:
                # Calculate position in gradient based on character position
                position = char_idx / max_width if max_width > 0 else 0
                color = _get_gradient_color(position)
                gradient_text.append(char, style=Style(color=color, bold=True))

        if line_idx < len(lines) - 1:
            gradient_text.append("\n")

    # Create decorative subtitle
    tagline = Text()
    tagline.append("~", style=Style(color=NEON_COLORS["purple"], dim=True))
    tagline.append(f" {description} ", style=Style(color=NEON_COLORS["pink"], dim=True, italic=True))
    tagline.append("~", style=Style(color=NEON_COLORS["purple"], dim=True))

    # Combine ASCII art and tagline as separate centered elements
    full_content = Group(
        Align.center(gradient_text),
        Align.center(tagline),
    )

    # Create panel with dim purple border for subtle framing
    panel = Panel(
        full_content,
        box=box.DOUBLE_EDGE,
        border_style=Style(color=NEON_COLORS["purple"], dim=True),
        padding=(0, 2),
    )

    console.print()  # Add spacing before
    console.print(panel)


# =============================================================================
# Phase Component (Simple)
# =============================================================================


def print_phase(
    console: Console,
    phase_num: int,
    description: str,
    status: str = "in_progress",
) -> None:
    """Print a phase indicator with status.

    Displays a numbered phase with an appropriate status icon and
    color-coded border based on the current status.

    Args:
        console: Rich Console instance for output.
        phase_num: The phase number to display.
        description: Description of the phase.
        status: One of "pending", "in_progress", "completed", "failed".

    """
    config = STATUS_CONFIG.get(status, STATUS_CONFIG["pending"])
    icon = config["icon"]
    color = config["color"]

    phase_text = Text()
    phase_text.append(f"{icon} ", style=Style(color=color))
    phase_text.append(f"Phase {phase_num}: ", style=Style(color=color, bold=True))
    phase_text.append(description, style=STYLE_FG)

    panel = Panel(
        phase_text,
        box=box.ROUNDED,
        border_style=Style(color=color),
        padding=(0, 1),
    )
    console.print(panel)


# =============================================================================
# Pill Badge Component
# =============================================================================


def pill(text: str, bg_color: str, fg_color: str) -> Text:
    """Create a pill-shaped badge with the given colors.

    Uses the pattern: [bg]foreground[/bg] to create a badge effect
    with half-block characters on the edges.

    Args:
        text: The text to display inside the pill.
        bg_color: Background color hex code.
        fg_color: Foreground (text) color hex code.

    Returns:
        Rich Text object containing the styled pill.

    """
    result = Text()
    bg_style = Style(color=bg_color)
    # Left edge
    result.append("\u258c", style=bg_style)
    # Center text with background
    result.append(text, style=Style(color=fg_color, bgcolor=bg_color, bold=True))
    # Right edge
    result.append("\u2590", style=bg_style)
    return result


# =============================================================================
# Tool Call Component
# =============================================================================

# Pattern for tool argument colorization
_TOOL_ARG_PATTERN = re.compile(r"(\w+)=((?:'[^']*'|\"[^\"]*\"|[^,\s]+))")


def _colorize_tool_args(args: dict[str, object]) -> Text:
    """Apply neon syntax highlighting to tool call arguments.

    Colorizes parameter names and values based on their types:
    - Keys in cyan
    - Strings in orange (paths in cyan)
    - Numbers in yellow
    - Booleans in purple

    Args:
        args: Dictionary of argument key-value pairs.

    Returns:
        Rich Text with neon styling applied.

    """
    result = Text()

    for i, (key, value) in enumerate(args.items()):
        if i > 0:
            result.append(", ", style=STYLE_FG)

        # Key in cyan
        result.append(str(key), style=STYLE_CYAN)
        result.append("=", style=STYLE_PURPLE)

        # Value styling based on type
        if isinstance(value, bool):
            # Boolean in purple
            result.append(str(value), style=STYLE_PURPLE)
        elif isinstance(value, (int, float)):
            # Numeric value in yellow
            result.append(str(value), style=STYLE_YELLOW)
        elif isinstance(value, str):
            # String value - check for file paths
            if "/" in value or value.endswith((".py", ".js", ".ts", ".md", ".json", ".yaml", ".yml")):
                result.append(value, style=STYLE_CYAN)
            else:
                result.append(value, style=STYLE_ORANGE)
        elif value is None:
            result.append("None", style=STYLE_PURPLE)
        else:
            # Other values (dicts, lists, etc.) in foreground
            result.append(str(value), style=STYLE_FG)

    return result


# =============================================================================
# Shared Tool Display Helpers
# =============================================================================


def _build_tool_header(
    name: str,
    args: dict[str, object],
    quiet_mode: bool = False,
) -> Text:
    """Build styled tool header content.

    Shared by print_tool_call() and LiveToolPanel._build_tool_header().

    Args:
        name: Name of the tool being called.
        args: Dictionary of arguments passed to the tool.
        quiet_mode: If True, hide command details for Bash tools.

    Returns:
        Rich Text containing the styled tool header.

    """
    content = Text()

    # Special handling for Skill tool calls - Gradient Whisper style
    if name == "Skill":
        header_line = Text()
        header_line.append("\u2728 ", style=STYLE_PURPLE)  # ✨
        header_line.append(f"{mystical_term('Skill')} ", style=Style(color=NEON_COLORS["pink"], italic=True))
        header_line.append("Skill", style=STYLE_BOLD_CYAN)
        content.append_text(header_line)
        content.append("\n  ")

        # Apply character-by-character gradient to skill name
        skill_name = str(args.get("skill", ""))
        for i, char in enumerate(skill_name):
            position = i / max(len(skill_name) - 1, 1)
            color = _get_gradient_color(position)
            content.append(char, style=Style(color=color, bold=True))

        # Show args in non-quiet mode (if present)
        if not quiet_mode:
            skill_args = args.get("args")
            if skill_args:
                content.append("\n  ")
                content.append("args=", style=STYLE_PURPLE)
                content.append(str(skill_args), style=STYLE_ORANGE)

        return content

    # Special handling for TodoWrite tool calls
    if name == "TodoWrite":
        header_line = Text()
        header_line.append("\U0001f527 ", style=STYLE_ORANGE)  # 🔧
        header_line.append("TodoWrite", style=STYLE_BOLD_PINK)
        content.append_text(header_line)

        # Parse and display todos list
        todos = args.get("todos", [])
        if isinstance(todos, list):
            for todo in todos:
                if isinstance(todo, dict):
                    todo_content = todo.get("content", "")
                    if not todo_content:
                        continue
                    status = todo.get("status", "pending")
                    config = STATUS_CONFIG.get(status, STATUS_CONFIG["pending"])
                    content.append("\n")
                    content.append(f"{config['icon']} ", style=Style(color=config["color"]))
                    content.append(todo_content, style=Style(color=config["color"]))
        else:
            # Fallback for non-list todos
            content.append("\n")
            content.append_text(_colorize_tool_args(args))

        return content

    # Special handling for Bash tool calls
    if name == "Bash":
        header_line = Text()
        header_line.append("\U0001f528 ", style=STYLE_ORANGE)  # 🔨
        header_line.append("Bash", style=STYLE_BOLD_PINK)
        content.append_text(header_line)

        # Add description if present
        description = str(args.get("description", ""))
        if description:
            content.append("\n")
            content.append(description, style=STYLE_CYAN)

        # Add command if not in quiet mode
        if not quiet_mode:
            command = str(args.get("command", ""))
            if command:
                content.append("\n")
                content.append("$ ", style=STYLE_DIM)
                content.append(command, style=STYLE_DIM)

        return content

    # Special handling for Write tool (header only, content preview handled separately)
    if name == "Write":
        file_path = str(args.get("file_path", ""))

        header_line = Text()
        header_line.append("\u26CF\uFE0F ", style=STYLE_ORANGE)  # ⛏️
        header_line.append("Write", style=STYLE_BOLD_PINK)
        content.append_text(header_line)
        content.append(f" {mystical_term('Write')}... ", style=f"{STYLE_PURPLE} italic")
        content.append(file_path, style=STYLE_CYAN)

        return content

    # Special handling for Glob tool calls
    if name == "Glob":
        pattern = str(args.get("pattern", ""))
        search_path = str(args.get("path", ""))

        header_line = Text()
        header_line.append("\U0001f52e ", style=STYLE_PURPLE)  # 🔮 crystal ball
        header_line.append("Glob", style=STYLE_BOLD_PINK)
        content.append_text(header_line)
        content.append(f" {mystical_term('Glob')}... ", style=f"{STYLE_PURPLE} italic")
        content.append(pattern, style=STYLE_ORANGE)

        # Search path if provided
        if search_path:
            content.append("\n")
            content.append("path=", style=STYLE_PURPLE)
            content.append(search_path, style=STYLE_CYAN)

        return content

    # Special handling for Grep tool calls
    if name == "Grep":
        pattern = str(args.get("pattern", ""))
        search_path = str(args.get("path", ""))
        glob_filter = str(args.get("glob", ""))
        file_type = str(args.get("type", ""))

        header_line = Text()
        header_line.append("\U0001f9d9 ", style=STYLE_PURPLE)  # 🧙 wizard
        header_line.append("Grep", style=STYLE_BOLD_PINK)
        content.append_text(header_line)
        content.append(f" {mystical_term('Grep')}... ", style=f"{STYLE_PURPLE} italic")
        content.append(pattern, style=STYLE_ORANGE)

        # Search path if provided
        if search_path:
            content.append("\n")
            content.append("path=", style=STYLE_PURPLE)
            content.append(search_path, style=STYLE_CYAN)

        # Glob filter if provided
        if glob_filter:
            content.append("\n")
            content.append("glob=", style=STYLE_PURPLE)
            content.append(glob_filter, style=STYLE_YELLOW)

        # File type if provided
        if file_type:
            content.append("\n")
            content.append("type=", style=STYLE_PURPLE)
            content.append(file_type, style=STYLE_YELLOW)

        return content

    # Special handling for Read tool calls
    if name == "Read":
        file_path = str(args.get("file_path", ""))
        offset = args.get("offset")
        limit = args.get("limit")

        header_line = Text()
        header_line.append("\U0001f4dc ", style=STYLE_ORANGE)  # 📜 scroll
        header_line.append("Read", style=STYLE_BOLD_PINK)
        content.append_text(header_line)
        content.append(f" {mystical_term('Read')}... ", style=f"{STYLE_PURPLE} italic")
        content.append(file_path, style=STYLE_CYAN)

        # Line range if specified
        if offset is not None or limit is not None:
            content.append("\n")
            if offset is not None:
                content.append("offset=", style=STYLE_PURPLE)
                content.append(str(offset), style=STYLE_YELLOW)
            if limit is not None:
                if offset is not None:
                    content.append(", ", style=STYLE_FG)
                content.append("limit=", style=STYLE_PURPLE)
                content.append(str(limit), style=STYLE_YELLOW)

        return content

    # Special handling for Edit tool calls - Dream Surgery visualization
    if name == "Edit":
        file_path = str(args.get("file_path", ""))
        old_string = str(args.get("old_string", ""))
        new_string = str(args.get("new_string", ""))
        replace_all = args.get("replace_all", False)

        # Header with surgery/healing theme
        header_line = Text()
        header_line.append("⚕ ", style=STYLE_CYAN)  # Medical symbol
        header_line.append("Edit", style=STYLE_BOLD_PINK)
        content.append_text(header_line)
        content.append(f" {mystical_term('Edit')}... ", style=Style(color=NEON_COLORS["purple"], italic=True))
        content.append(file_path, style=STYLE_CYAN)

        # Replace all flag if true
        if replace_all:
            content.append("\n")
            content.append("replace_all=", style=STYLE_PURPLE)
            content.append("True", style=STYLE_PURPLE)

        content.append("\n")

        # BLOCKED energy - old string with red→orange gradient
        content.append("\n")
        content.append("  ⊗ ", style=STYLE_RED)
        content.append("BLOCKED", style=Style(color=NEON_COLORS["red"], bold=True))
        content.append("\n")
        if old_string:
            lines = old_string.split("\n")[:30]  # Up to 30 lines
            preview = "\n".join(lines)
            if len(old_string.split("\n")) > 30:
                preview += "\n..."
            preview_len = max(len(preview) - 1, 1)
            # Show with gradient from red to orange
            for i, char in enumerate(preview):
                if char == "\n":
                    content.append("\n  ")  # Indent continuation lines
                else:
                    t = i / preview_len
                    color = _interpolate_color(NEON_COLORS["red"], NEON_COLORS["orange"], t)
                    content.append(char, style=Style(color=color))

        # FLOWING energy - new string with cyan→green gradient
        content.append("\n\n")
        content.append("  ✓ ", style=STYLE_GREEN)
        content.append("FLOWING", style=Style(color=NEON_COLORS["green"], bold=True))
        content.append("\n")
        if new_string:
            lines = new_string.split("\n")[:30]  # Up to 30 lines
            preview = "\n".join(lines)
            if len(new_string.split("\n")) > 30:
                preview += "\n..."
            preview_len = max(len(preview) - 1, 1)
            # Show with gradient from cyan to green
            for i, char in enumerate(preview):
                if char == "\n":
                    content.append("\n  ")  # Indent continuation lines
                else:
                    t = i / preview_len
                    color = _interpolate_color(NEON_COLORS["cyan"], NEON_COLORS["green"], t)
                    content.append(char, style=Style(color=color, bold=True))

        return content

    # Standard tool call display (other tools)
    header_line = Text()
    header_line.append("\U0001f3a0 ", style=STYLE_ORANGE)  # 🎠
    header_line.append(name, style=STYLE_BOLD_PINK)
    content.append_text(header_line)

    # Add formatted key=value pairs
    if args:
        content.append("\n")
        args_text = _colorize_tool_args(args)
        content.append_text(args_text)

    return content


def _build_result_content(
    content: str,
    is_error: bool = False,
    max_lines: int = 20,
) -> tuple[Text | Syntax | Group, bool]:
    """Build styled result content with syntax highlighting.

    Shared by print_tool_result() and LiveToolPanel._build_result_content().

    Args:
        content: The result content to display.
        is_error: Whether this is an error result.
        max_lines: Maximum number of lines to display.

    Returns:
        Tuple of (renderable content, was_truncated).

    """
    lines = content.split("\n")
    truncated = False
    total_lines = len(lines)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        truncated = True

    # For non-error shell content, use Rich Syntax highlighting
    if not is_error and _detect_shell_syntax(content):
        display_content = "\n".join(lines)
        syntax = Syntax(
            display_content,
            "bash",
            theme="dracula",
            line_numbers=False,
            word_wrap=True,
        )

        if truncated:
            truncation_text = Text()
            truncation_text.append(
                f"\n... ({total_lines - max_lines} more lines)",
                style=Style(color=NEON_COLORS["yellow"], italic=True),
            )
            return Group(syntax, truncation_text), truncated
        return syntax, truncated

    # Build colorized text content using manual colorization
    result_text = Text()
    for i, line in enumerate(lines):
        result_text.append_text(_colorize_line(line, is_error))
        if i < len(lines) - 1:
            result_text.append("\n")

    # Add truncation indicator if needed
    if truncated:
        result_text.append("\n")
        result_text.append(
            f"... ({total_lines - max_lines} more lines)",
            style=Style(color=NEON_COLORS["yellow"], italic=True),
        )

    return result_text, truncated


def print_tool_call(
    console: Console,
    name: str,
    args: dict[str, object],
    quiet_mode: bool = False,
) -> None:
    """Print a tool call with its arguments.

    Displays the tool name with a wrench icon, followed by the arguments
    in a styled panel. All tool calls are wrapped in a Panel with dark
    background and purple border.

    Special handling for Skill tool calls: uses ✨ icon and displays
    the skill name as a pink pill badge.

    Special handling for Bash tool calls: shows description and optionally
    the command (hidden in quiet mode).

    Args:
        console: Rich Console instance for output.
        name: Name of the tool being called.
        args: Dictionary of arguments passed to the tool.
        quiet_mode: If True, hide command details for Bash tools.

    """
    # Newline before tool call for separation
    console.print()

    # Build header using shared helper
    content = _build_tool_header(name, args, quiet_mode)

    # Handle Write tool content preview (unique to static display)
    panel_content: Group | Text = content
    if name == "Write" and not quiet_mode:
        file_path = str(args.get("file_path", ""))
        content_str = str(args.get("content", ""))

        if file_path.endswith(".md") and content_str:
            # Render markdown content
            md_content = Markdown(content_str)
            panel_content = Group(content, Text("\n"), md_content)
        elif content_str:
            # Show truncated content preview for non-markdown files
            content.append("\n")
            preview = content_str[:200] + "..." if len(content_str) > 200 else content_str
            content.append(preview, style=Style(color=NEON_COLORS["foreground"], dim=True))

    # Determine border style
    border_style = STYLE_CYAN if name == "Skill" else STYLE_PURPLE

    panel = Panel(
        panel_content,
        box=box.ROUNDED,
        border_style=border_style,
        style=STYLE_PANEL_BG,
        padding=(0, 1),
    )
    console.print(panel)


# =============================================================================
# Tool Result Component
# =============================================================================

# Patterns for syntax highlighting in tool output
_FILE_PATH_PATTERN = re.compile(r"(\.?/?(?:[\w.-]+/)*[\w.-]+\.\w+)")
_LINE_NUMBER_PATTERN = re.compile(r"^(\s*)(\d+)([:\-\|])")
_ERROR_KEYWORDS = re.compile(r"\b(error|Error|ERROR|failed|Failed|FAILED|exception|Exception|EXCEPTION|Traceback)\b")
_SUCCESS_KEYWORDS = re.compile(r"\b(passed|Passed|PASSED|success|Success|SUCCESS|ok|OK|done|Done|DONE)\b")
_WARNING_KEYWORDS = re.compile(r"\b(warning|Warning|WARNING|warn|Warn|WARN|skip|Skip|SKIP|skipped|Skipped)\b")
_NUMBER_PATTERN = re.compile(r"\b(\d+)\b")
_STRING_PATTERN = re.compile(r"(['\"])(.*?)\1")
_ARROW_PATTERN = re.compile(r"(->|=>|-->|==>)")
_BRACKET_PATTERN = re.compile(r"([\[\]{}()])")

# Git-specific colorization patterns
_GIT_ADDED_PATTERN = re.compile(r"^(\+(?!\+\+).*|A[ \t].*)$")
_GIT_DELETED_PATTERN = re.compile(r"^(-(?!--).*|D[ \t].*)$")
_GIT_MODIFIED_PATTERN = re.compile(r"^M[ \t].*$")
_GIT_DIFF_HEADER_PATTERN = re.compile(r"^(@@.*|diff --git.*|index .*)$")
_GIT_FILE_HEADER_PATTERN = re.compile(r"^(\+\+\+|---).*$")

# Bash-specific patterns for manual colorization
_SHELL_VAR_PATTERN = re.compile(r"(\$\{?\w+\}?)")
_SHELL_PROMPT_PATTERN = re.compile(r"^([$#>] )")
_PIPE_REDIRECT_PATTERN = re.compile(r"(\||>>|2>&1|>&2|&>|<|>)")
_COMMAND_PATTERN = re.compile(
    r"^(git|npm|pytest|python|uv|cd|ls|cat|grep|ruff|mypy|pip|docker|make|curl|wget|rm|mv|cp|mkdir|echo|export|source|bash|sh|zsh)\b"
)

# Patterns for shell syntax detection (for Rich Syntax highlighting)
_SHELL_DETECT_PROMPT_PATTERN = re.compile(r"^[$#]\s+", re.MULTILINE)
_SHEBANG_PATTERN = re.compile(r"^#!.*(?:ba)?sh", re.MULTILINE)
_SHELL_DETECT_COMMAND_PATTERN = re.compile(
    r"^\s*(?:export|alias|source|cd|ls|cat|grep|awk|sed|echo|printf|"
    r"mkdir|rm|cp|mv|chmod|chown|sudo|apt|yum|brew|pip|npm|git|docker|"
    r"curl|wget|tar|zip|unzip|find|xargs|sort|uniq|wc|head|tail|tee|"
    r"for\s+\w+\s+in|while\s+|if\s+\[|elif\s+\[|fi$|done$|esac$)\b",
    re.MULTILINE,
)


def _detect_shell_syntax(content: str) -> bool:
    """Detect if content looks like shell script or command output.

    Checks for common shell indicators such as:
    - Lines starting with $ or # (shell prompts)
    - Shebang lines (#!/bin/bash, etc.)
    - Multiple lines with common shell command patterns

    Args:
        content: The content to analyze.

    Returns:
        True if the content appears to be shell script or output.

    """
    # Check for shebang - strong indicator
    if _SHEBANG_PATTERN.search(content):
        return True

    # Count shell prompt lines ($ or # at start)
    prompt_matches = len(_SHELL_DETECT_PROMPT_PATTERN.findall(content))
    if prompt_matches >= 2:
        return True

    # Count shell command patterns
    command_matches = len(_SHELL_DETECT_COMMAND_PATTERN.findall(content))
    if command_matches >= 3:
        return True

    # Check combination: at least 1 prompt + 1 command
    return bool(prompt_matches >= 1 and command_matches >= 1)


def _colorize_git_line(line: str) -> Text | None:
    """Check if a line matches git output patterns and return styled Text.

    Args:
        line: The line to check.

    Returns:
        Rich Text with git-specific styling, or None if not git output.

    """
    # File headers (+++/---) - cyan bold
    if _GIT_FILE_HEADER_PATTERN.match(line):
        return Text(line, style=STYLE_BOLD_CYAN)

    # Diff headers (@@, diff --git, index) - purple
    if _GIT_DIFF_HEADER_PATTERN.match(line):
        return Text(line, style=STYLE_PURPLE)

    # Added lines (+, A ) - green
    if _GIT_ADDED_PATTERN.match(line):
        return Text(line, style=STYLE_GREEN)

    # Deleted lines (-, D ) - red
    if _GIT_DELETED_PATTERN.match(line):
        return Text(line, style=STYLE_RED)

    # Modified lines (M ) - yellow
    if _GIT_MODIFIED_PATTERN.match(line):
        return Text(line, style=STYLE_YELLOW)

    return None


def _colorize_line(line: str, is_error: bool = False) -> Text:
    """Apply neon syntax highlighting to a single line of output.

    Args:
        line: The line to colorize.
        is_error: Whether this is error output.

    Returns:
        Rich Text with neon styling applied.

    """
    # Check for git-specific patterns first
    git_result = _colorize_git_line(line)
    if git_result is not None:
        return git_result

    result = Text()

    if is_error:
        # Error output - still colorize but with red base
        # Highlight file paths in orange for visibility
        pos = 0
        for match in _FILE_PATH_PATTERN.finditer(line):
            if match.start() > pos:
                result.append(line[pos:match.start()], style=STYLE_RED)
            result.append(match.group(1), style=Style(color=NEON_COLORS["orange"], bold=True))
            pos = match.end()
        if pos < len(line):
            result.append(line[pos:], style=STYLE_RED)
        return result

    # Check for line number prefix (e.g., "  42:" or "123|")
    line_num_match = _LINE_NUMBER_PATTERN.match(line)
    if line_num_match:
        indent, num, sep = line_num_match.groups()
        result.append(indent, style=Style())
        result.append(num, style=STYLE_YELLOW)
        result.append(sep, style=STYLE_PURPLE)
        line = line[line_num_match.end():]

    # Process the rest of the line for patterns
    segments: list[tuple[int, int, str, Style]] = []

    # Find all file paths (highest priority - cyan)
    for match in _FILE_PATH_PATTERN.finditer(line):
        segments.append((match.start(), match.end(), match.group(1),
                        STYLE_CYAN))

    # Find error keywords (red bold)
    for match in _ERROR_KEYWORDS.finditer(line):
        segments.append((match.start(), match.end(), match.group(1),
                        STYLE_BOLD_RED))

    # Find success keywords (green bold)
    for match in _SUCCESS_KEYWORDS.finditer(line):
        segments.append((match.start(), match.end(), match.group(1),
                        STYLE_BOLD_GREEN))

    # Find warning keywords (yellow bold)
    for match in _WARNING_KEYWORDS.finditer(line):
        segments.append((match.start(), match.end(), match.group(1),
                        STYLE_BOLD_YELLOW))

    # Find strings (orange)
    for match in _STRING_PATTERN.finditer(line):
        segments.append((match.start(), match.end(), match.group(0),
                        STYLE_ORANGE))

    # Find arrows (pink)
    for match in _ARROW_PATTERN.finditer(line):
        segments.append((match.start(), match.end(), match.group(1),
                        STYLE_BOLD_PINK))

    # Find brackets (purple)
    for match in _BRACKET_PATTERN.finditer(line):
        segments.append((match.start(), match.end(), match.group(1),
                        STYLE_PURPLE))

    # Find shell variables $VAR and ${VAR} (yellow)
    for match in _SHELL_VAR_PATTERN.finditer(line):
        segments.append((match.start(), match.end(), match.group(1),
                        STYLE_YELLOW))

    # Find shell prompts at line start (cyan)
    for match in _SHELL_PROMPT_PATTERN.finditer(line):
        segments.append((match.start(), match.end(), match.group(1),
                        STYLE_CYAN))

    # Find pipe and redirect operators (pink)
    for match in _PIPE_REDIRECT_PATTERN.finditer(line):
        segments.append((match.start(), match.end(), match.group(1),
                        STYLE_PINK))

    # Find common commands at line start (orange bold)
    for match in _COMMAND_PATTERN.finditer(line):
        segments.append((match.start(), match.end(), match.group(1),
                        Style(color=NEON_COLORS["orange"], bold=True)))

    # Sort segments by start position, then by length (longer matches first)
    segments.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    # Build the result, avoiding overlaps
    last_end = 0
    used_ranges: list[tuple[int, int]] = []

    for start, end, text, style in segments:
        # Check if this segment overlaps with any used range
        overlaps = any(not (end <= used_start or start >= used_end)
                      for used_start, used_end in used_ranges)
        if overlaps:
            continue

        if start > last_end:
            # Add default styled text before this segment
            result.append(line[last_end:start],
                         style=STYLE_FG)
        result.append(text, style=style)
        used_ranges.append((start, end))
        last_end = end

    # Add remaining text
    if last_end < len(line):
        result.append(line[last_end:],
                     style=STYLE_FG)

    return result


def print_tool_result(
    console: Console,
    content: str,
    is_error: bool = False,
    max_lines: int = 20,
) -> None:
    """Print a tool result with neon syntax highlighting.

    Displays the result content with colorized file paths, line numbers,
    and keyword highlighting. Uses red styling for errors. For shell scripts
    and command output, uses Rich's Syntax class for proper highlighting.

    Args:
        console: Rich Console instance for output.
        content: The result content to display.
        is_error: Whether this is an error result.
        max_lines: Maximum number of lines to display.

    """
    if not content or not content.strip():
        return

    # Build result content using shared helper
    result_content, _ = _build_result_content(content, is_error, max_lines)

    # Determine border style and title
    if is_error:
        border_style = STYLE_RED
        title = "[bold red]❌ Error[/bold red]"
    else:
        border_style = STYLE_PURPLE
        title = f"[bold {NEON_COLORS['cyan']}]Output[/bold {NEON_COLORS['cyan']}]"

    panel = Panel(
        result_content,
        box=box.ROUNDED,
        border_style=border_style,
        title=title,
        title_align="left",
        style=STYLE_PANEL_BG,
        expand=False,
        padding=(0, 1),
    )
    console.print(panel)


# =============================================================================
# Code Result Component
# =============================================================================


def print_code_result(
    console: Console,
    content: str,
    filename: str = "",
    max_lines: int = 30,
) -> None:
    """Print code content with syntax highlighting.

    Uses Rich's Syntax component for proper highlighting and
    automatic line wrapping that respects terminal width.

    Args:
        console: Rich Console instance for output.
        content: The code content to display.
        filename: Optional filename to detect language (e.g., "test.py").
        max_lines: Maximum number of lines to display.

    """
    lines = content.split("\n")
    truncated = len(lines) > max_lines
    display_content = "\n".join(lines[:max_lines]) if truncated else content

    # Detect language from filename extension
    if "." in filename:
        lang = filename.rsplit(".", 1)[-1]
        # Map common extensions
        lang_map = {"py": "python", "js": "javascript", "ts": "typescript", "md": "markdown"}
        lang = lang_map.get(lang, lang)
    else:
        lang = "text"

    syntax = Syntax(
        display_content,
        lang,
        theme="dracula",
        line_numbers=True,
        word_wrap=True,
    )
    console.print(syntax)

    if truncated:
        console.print(
            f"[neon.dim]... ({len(lines) - max_lines} more lines)[/]"
        )


# =============================================================================
# Thinking Component
# =============================================================================


class LiveThinkingPanel:
    """Animated thinking panel with crazy spinners in title.

    Displays the AI's thought process in a purple-styled panel
    with animated spinners alongside the thought bubble icon.

    Args:
        console: Rich Console instance for output.
        content: The thinking content to display.
        max_length: Maximum content length before truncation.

    """

    def __init__(self, console: Console, content: str, max_length: int = 300) -> None:
        """Initialize the panel.

        Args:
            console: Rich Console instance for output.
            content: The thinking content to display.
            max_length: Maximum content length before truncation.

        """
        self._console = console
        self._content = content if len(content) <= max_length else content[:max_length] + "..."
        self._spinner = CrazySpinner(num_spinners=3)

    def __rich__(self) -> Panel:
        """Render panel with animated title.

        Returns:
            Rich Panel with animated spinners in title.

        """
        # Build title with spinners
        title = Text()
        title.append("\U0001f4ad Thinking")  # 💭
        title.append_text(self._spinner.render())

        return Panel(
            Text(self._content, style=Style(color=NEON_COLORS["purple"], italic=True)),
            title=title,
            title_align="left",
            box=box.ROUNDED,
            border_style=STYLE_PURPLE,
            padding=(0, 1),
        )

    def show(self, duration: float = 1.0) -> None:
        """Show animated panel for duration, then persist final state.

        Args:
            duration: How long to show animation before settling.

        """
        self._console.print()
        with Live(self, console=self._console, refresh_per_second=10, transient=True):
            time.sleep(duration)
        # Print final static panel
        self._console.print(
            Panel(
                Text(self._content, style=Style(color=NEON_COLORS["purple"], italic=True)),
                title="\U0001f4ad Thinking",  # 💭
                title_align="left",
                box=box.ROUNDED,
                border_style=STYLE_PURPLE,
                padding=(0, 1),
            )
        )


def print_thinking(console: Console, content: str, max_length: int = 300) -> None:
    """Print a thinking panel with brief animation.

    Displays the AI's thought process in a purple-styled panel
    with animated spinners in the title that settle after a brief duration.

    Args:
        console: Rich Console instance for output.
        content: The thinking content to display.
        max_length: Maximum content length before truncation.

    """
    panel = LiveThinkingPanel(console, content, max_length)
    panel.show(duration=0.5)  # Brief animation before settling


# =============================================================================
# Feedback Table Component
# =============================================================================


def print_feedback_table(console: Console, items: list[dict[str, object]]) -> None:
    """Print a table of feedback items/issues.

    Creates a styled table with columns for issue number, status,
    description, file, and line number.

    Args:
        console: Rich Console instance for output.
        items: List of dicts with keys: status, description, file, line.

    """
    table = Table(
        title="\U0001f4cb Issues to Fix",  # 📋
        title_style=STYLE_BOLD_CYAN,
        box=box.ROUNDED,
        border_style=STYLE_PURPLE,
        header_style=STYLE_BOLD_PINK,
        show_lines=True,
    )

    table.add_column("#", justify="right", style=STYLE_CYAN)
    table.add_column("Status", justify="center")
    table.add_column("Description", style=STYLE_FG)
    table.add_column("File", style=STYLE_ORANGE)
    table.add_column("Line", justify="right", style=STYLE_YELLOW)

    for i, item in enumerate(items, 1):
        status = str(item.get("status", "pending"))
        config = STATUS_CONFIG.get(status, STATUS_CONFIG["pending"])

        status_pill = pill(
            f" {config['icon']} {status.upper()} ",
            config["color"],
            NEON_COLORS["background"],
        )

        table.add_row(
            str(i),
            status_pill,
            str(item.get("description", "")),
            str(item.get("file", "")),
            str(item.get("line", "")),
        )

    console.print(table)


# =============================================================================
# Error Component
# =============================================================================


def print_error(console: Console, title: str, message: str) -> None:
    """Print an error panel with red styling.

    Creates a prominent error display with a warning icon
    and double-edge border.

    Args:
        console: Rich Console instance for output.
        title: Error title text.
        message: Detailed error message.

    """
    panel = Panel(
        Text(message, style=STYLE_RED),
        title=f"\u26a0\ufe0f  {title}",  # ⚠️
        title_align="left",
        box=box.DOUBLE_EDGE,
        border_style=STYLE_RED,
        padding=(0, 1),
    )
    console.print(panel)


# =============================================================================
# Warning Component
# =============================================================================


def print_warning(console: Console, message: str) -> None:
    """Print a warning panel with yellow styling.

    Args:
        console: Rich Console instance for output.
        message: Warning message to display.

    """
    panel = Panel(
        Text(message, style=STYLE_YELLOW),
        box=box.ROUNDED,
        border_style=STYLE_YELLOW,
        padding=(0, 1),
    )
    console.print(panel)


# =============================================================================
# Success Component
# =============================================================================


def print_success(console: Console, message: str) -> None:
    """Print a success message with green styling.

    Args:
        console: Rich Console instance for output.
        message: Success message to display.

    """
    console.print(f"[neon.success]✔[/] [neon.green]{message}[/]")


# =============================================================================
# Cost Component
# =============================================================================


def print_cost(console: Console, cost_usd: float) -> None:
    """Print a cost indicator with cyan styling.

    Args:
        console: Rich Console instance for output.
        cost_usd: The cost in USD.

    """
    console.print(f"[neon.cyan]💰[/] [neon.dim]${cost_usd:.4f}[/]")


# =============================================================================
# Info Component
# =============================================================================


def print_info(console: Console, message: str) -> None:
    """Print an info message with cyan styling.

    Args:
        console: Rich Console instance for output.
        message: Info message to display.

    """
    console.print(f"[neon.cyan]ℹ[/] [neon.fg]{message}[/]")


def print_skipped_phases(console: Console, start_at: str) -> None:
    """Print message about skipped phases when starting at a non-default phase.

    Args:
        console: Rich Console instance for output.
        start_at: The phase to start at ("parse", "fix", or "test").

    """
    phase_order = ["review", "parse", "fix", "test"]
    skipped = []
    for phase in phase_order:
        if phase == start_at:
            break
        skipped.append(phase)

    if skipped:
        skipped_str = ", ".join(skipped)
        console.print(f"[neon.yellow]⏭[/] [neon.fg]Starting at phase: {start_at} (skipping {skipped_str})[/]")


def print_dim(console: Console, message: str) -> None:
    """Print a dimmed message for secondary information.

    Args:
        console: Rich Console instance for output.
        message: Message to display in dimmed style.

    """
    console.print(f"[neon.dim]{message}[/]")


# =============================================================================
# Agent Text Component
# =============================================================================

# Track state for agent text blocks (gutter display and markdown detection)
_agent_text_line_started = False
_agent_text_has_markdown = False


def _highlight_agent_text(text: str, base_style: Style | None = None) -> Text:
    """Apply syntax highlighting to agent text.

    Highlights:
    - Inline code (`code`)
    - File paths
    - Numbers
    - URLs
    - Bold (**text**) and italic (*text*)

    Args:
        text: Raw agent text to highlight.
        base_style: Style for non-highlighted text. Defaults to STYLE_GREEN.

    Returns:
        Rich Text with neon styling applied.

    """
    if base_style is None:
        base_style = STYLE_GREEN

    result = Text()

    # Patterns for highlighting
    code_pattern = re.compile(r"`([^`]+)`")
    bold_pattern = re.compile(r"\*\*([^*]+)\*\*")
    italic_pattern = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")
    url_pattern = re.compile(r"https?://[^\s\])<>]+")
    file_path_pattern = re.compile(r"(?:^|[\s(])([./]?(?:[\w.-]+/)+[\w.-]+\.\w+)")

    # Process text character by character with pattern matching
    pos = 0
    segments: list[tuple[int, int, str, Style]] = []

    # Find all patterns
    for match in code_pattern.finditer(text):
        segments.append((
            match.start(),
            match.end(),
            match.group(1),  # Just the code, not backticks
            Style(color=NEON_COLORS["orange"], bgcolor="#3a3a3a"),
        ))

    for match in bold_pattern.finditer(text):
        segments.append((
            match.start(),
            match.end(),
            match.group(1),
            Style(color=NEON_COLORS["green"], bold=True),
        ))

    for match in italic_pattern.finditer(text):
        segments.append((
            match.start(),
            match.end(),
            match.group(1),
            Style(color=NEON_COLORS["green"], italic=True),
        ))

    for match in url_pattern.finditer(text):
        segments.append((
            match.start(),
            match.end(),
            match.group(0),
            Style(color=NEON_COLORS["cyan"], underline=True),
        ))

    for match in file_path_pattern.finditer(text):
        segments.append((
            match.start(1),
            match.end(1),
            match.group(1),
            STYLE_CYAN,
        ))

    # Sort segments by position, prefer longer matches
    segments.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    # Build result avoiding overlaps
    pos = 0
    used_ranges: list[tuple[int, int]] = []

    for start, end, display_text, style in segments:
        overlaps = any(
            not (end <= used_start or start >= used_end)
            for used_start, used_end in used_ranges
        )
        if overlaps:
            continue

        if start > pos:
            # Add default styled text with base style
            result.append(text[pos:start], style=base_style)

        result.append(display_text, style=style)
        used_ranges.append((start, end))
        pos = end

    # Add remaining text with base style
    if pos < len(text):
        result.append(text[pos:], style=base_style)

    return result


# =============================================================================
# Agent Text Helpers
# =============================================================================

# Constants for agent text styling
AGENT_TEXT_BG = "#051208"  # Very dark green background
AGENT_TEXT_FG = NEON_COLORS["green"]  # Neon green text

# Pattern to detect markdown headers anywhere in text (not just at line start)
# Matches 1-6 consecutive hashes followed by whitespace and a non-whitespace char
# This handles both proper line-start headers AND inline headers from streaming
_MARKDOWN_HEADER_PATTERN = re.compile(r"#{1,6}\s+\S")


def _has_markdown_headers(text: str) -> bool:
    """Check if text contains markdown headers.

    Args:
        text: Text to check for markdown headers.

    Returns:
        True if text contains markdown headers (e.g., ## Summary).

    """
    return bool(_MARKDOWN_HEADER_PATTERN.search(text))


def _render_agent_lines_with_gradient(
    lines: list[str],
    use_italic: bool = True,
) -> Text:
    """Render agent text lines with vertical cyan-to-green gradient.

    Args:
        lines: List of text lines to render.
        use_italic: Whether to apply italic styling (False for markdown content).

    Returns:
        Rich Text with vertical gradient styling applied.

    """
    highlighted = Text()
    num_lines = max(len(lines), 1)

    for i, line in enumerate(lines):
        if line:
            # Calculate vertical gradient position (0.0 = top/cyan, 1.0 = bottom/green)
            t = i / max(num_lines - 1, 1)
            gradient_color = _interpolate_color(
                NEON_COLORS["cyan"],
                NEON_COLORS["green"],
                t,
            )
            line_style = Style(color=gradient_color, italic=use_italic)
            highlighted.append_text(_highlight_agent_text(line, base_style=line_style))
        if i < len(lines) - 1:
            highlighted.append("\n")

    return highlighted


# =============================================================================
# AgentTextRenderer Class (Live Panel with buffering)
# =============================================================================


class AgentTextRenderer:
    """Renderer for agent text using Rich Live Panel with buffering.

    This class buffers incoming text chunks and displays them in a
    Live-updating Panel that provides proper word wrapping. This solves
    the issue of broken line wrapping when streaming text character-by-character.

    Args:
        console: Rich Console instance for output.

    Usage:
        renderer = AgentTextRenderer(console)
        renderer.start()
        for chunk in stream:
            renderer.append(chunk)
        renderer.finish()

    """

    def __init__(self, console: Console) -> None:
        """Initialize the renderer.

        Args:
            console: Rich Console instance for output.

        """
        self._console = console
        self._buffer: list[str] = []
        self._live: Live | None = None
        self._started = False
        self._spinner = CrazySpinner(num_spinners=3)

    def _render_panel(self, show_spinner: bool = True) -> Panel:
        """Render the current buffer as a styled Panel with vertical gradient.

        Applies a cyan-to-green vertical gradient. Uses italic styling for
        regular text, but not for markdown content (detected by headers).

        Args:
            show_spinner: If True, append animated spinner at end (cursor effect).

        Returns:
            Rich Panel with gradient styling applied.

        """
        full_text = "".join(self._buffer)
        lines = full_text.split("\n")

        # Detect markdown - use gradient but no italic for markdown content
        use_italic = not _has_markdown_headers(full_text)

        # Render with vertical gradient
        content = _render_agent_lines_with_gradient(lines, use_italic=use_italic)

        # Append spinner at end (like a cursor) while streaming
        if show_spinner:
            content.append_text(self._spinner.render())

        return Panel(
            content,
            box=box.ROUNDED,
            border_style=STYLE_GREEN,
            style=STYLE_AGENT_BG,
            padding=(0, 1),
        )

    def __rich__(self) -> Panel:
        """Render for Live refresh cycle - enables continuous spinner animation."""
        return self._render_panel(show_spinner=True)

    def start(self) -> None:
        """Start the Live context for real-time updates.

        Call this before appending any text chunks.

        Returns:
            None

        """
        if self._started:
            return

        # Add newline for separation from previous output
        self._console.print()

        self._live = Live(
            self,  # Pass self so Live calls __rich__() on each refresh
            console=self._console,
            refresh_per_second=10,
            transient=True,  # Remove the live display when done
        )
        self._live.start()
        self._started = True

    def append(self, text: str) -> None:
        """Append text to the buffer and update the display.

        Args:
            text: Text chunk to append.

        Returns:
            None

        """
        if not text:
            return

        # Auto-start if not already started
        if not self._started:
            self.start()

        self._buffer.append(text)
        # Live will pick up buffer changes via __rich__() on next refresh

    def finish(self) -> None:
        """Stop the Live context and print the final Panel.

        Call this when all text has been received.

        Returns:
            None

        """
        if self._live is not None:
            self._live.stop()
            self._live = None

        # Print the final panel (non-transient, without spinner)
        if self._buffer:
            self._console.print(self._render_panel(show_spinner=False))

        # Reset state
        self._buffer = []
        self._started = False

    @property
    def has_content(self) -> bool:
        """Check if the buffer has any content.

        Returns:
            bool: True if the buffer contains text, False otherwise.

        """
        return bool(self._buffer)


def print_agent_text(console: Console, text: str) -> None:
    """Print agent text output with streaming-friendly left gutter.

    Displays agent text with a neon-styled left border and
    markdown-aware syntax highlighting. The gutter prefix is
    shown on the first line, and the text flows with word wrapping.

    Uses italic for regular narration, but not for markdown content.

    Args:
        console: Rich Console instance for output.
        text: The agent text to display.

    Returns:
        None

    """
    global _agent_text_line_started, _agent_text_has_markdown

    if not text:
        return

    # Check for markdown headers in this chunk
    if _has_markdown_headers(text):
        _agent_text_has_markdown = True

    # Split into lines to handle multiline text
    lines = text.split("\n")

    for i, line in enumerate(lines):
        is_last = i == len(lines) - 1

        if not _agent_text_line_started:
            # Start of a new agent text block - add separation and gutter prefix
            console.print()  # Newline for separation from tool calls
            gutter = Text()
            gutter.append("│ ", style=STYLE_GREEN)
            console.print(gutter, end="")
            _agent_text_line_started = True

        # Highlight and print the line content with dark green background
        # Use italic only for non-markdown content
        if line:
            use_italic = not _agent_text_has_markdown
            base_style = Style(color=NEON_COLORS["green"], italic=use_italic)
            highlighted = _highlight_agent_text(line, base_style=base_style)
            highlighted.stylize(STYLE_AGENT_BG)
            console.print(highlighted, end="")

        # Handle newlines - reset gutter state for next line
        if not is_last:
            console.print()  # Complete the current line
            _agent_text_line_started = False


def reset_agent_text_state() -> None:
    """Reset the agent text line state.

    Call this at the end of an agent response to ensure
    the next agent response starts with a fresh gutter.

    Returns:
        None

    """
    global _agent_text_line_started, _agent_text_has_markdown
    _agent_text_line_started = False
    _agent_text_has_markdown = False


def print_fix_progress(
    console: Console, item_num: int, total: int, description: str
) -> None:
    """Print fix progress indicator.

    Args:
        console: Rich Console instance for output.
        item_num: Current item number (1-indexed).
        total: Total number of items.
        description: Description of the fix.

    Returns:
        None

    """
    text = Text()
    text.append("  ", style=Style())
    text.append(f"[{item_num}/{total}] ", style=STYLE_BOLD_CYAN)
    text.append("Fixing: ", style=STYLE_PINK)
    # Truncate description
    desc = description[:60] + "..." if len(description) > 60 else description
    text.append(desc, style=STYLE_FG)
    console.print(text)


def print_fix_complete(console: Console, item_num: int, total: int) -> None:
    """Print fix completion indicator.

    Args:
        console: Rich Console instance for output.
        item_num: Current item number (1-indexed).
        total: Total number of items.

    Returns:
        None

    """
    text = Text()
    text.append("  ", style=Style())
    text.append(f"[{item_num}/{total}] ", style=STYLE_BOLD_CYAN)
    text.append("✔ Fix applied", style=STYLE_GREEN)
    console.print(text)


# =============================================================================
# Summary Component
# =============================================================================


@dataclass
class SummaryData:
    """Data class for summary information.

    Attributes:
        skill: Name of the skill that was executed.
        target: Target file or directory that was reviewed.
        feedback_count: Number of issues found during review.
        fixes_applied: Number of fixes that were applied.
        test_retries: Number of times tests were retried.
        tests_passed: Whether all tests passed after fixes.
        review_only: If True, only review was performed (no fixes).

    """

    skill: str
    target: str
    feedback_count: int
    fixes_applied: int
    test_retries: int
    tests_passed: bool
    review_only: bool = False


def print_summary(console: Console, data: SummaryData) -> None:
    """Print a summary table with neon styling.

    Displays a comprehensive summary of the review/fix session
    with status badges for pass/fail.

    Args:
        console: Rich Console instance for output.
        data: SummaryData containing all summary fields.

    """
    table = Table(
        title="✨ Review Summary",
        title_style=STYLE_BOLD_GREEN,
        box=box.ROUNDED,
        border_style=STYLE_PURPLE,
        show_header=False,
        padding=(0, 1),
    )

    table.add_column("Field", style=STYLE_CYAN)
    table.add_column("Value", style=STYLE_FG)

    table.add_row("Skill", data.skill)
    table.add_row("Target", data.target)
    table.add_row("Issues Found", str(data.feedback_count))

    if data.review_only:
        # Review-only mode: show mode badge instead of fix/test stats
        mode_badge = pill(" REVIEW ONLY ", NEON_COLORS["cyan"], NEON_COLORS["background"])
        table.add_row("Mode", mode_badge)
    else:
        # Full mode: show fix and test stats
        table.add_row("Fixes Applied", str(data.fixes_applied))
        table.add_row("Test Retries", str(data.test_retries))

        # Create pass/fail pill
        if data.tests_passed:
            status_badge = pill(" PASSED ", NEON_COLORS["green"], NEON_COLORS["background"])
        else:
            status_badge = pill(" FAILED ", NEON_COLORS["red"], NEON_COLORS["background"])
        table.add_row("Tests", status_badge)

    console.print(table)


# =============================================================================
# Menu Component
# =============================================================================


def print_menu(console: Console, title: str, options: list[tuple[str, str]]) -> None:
    """Print a styled menu for user selection.

    Displays numbered options with descriptions in a neon-styled panel.

    Args:
        console: Rich Console instance for output.
        title: Menu title.
        options: List of (key, description) tuples.

    """
    menu_text = Text()
    for key, description in options:
        menu_text.append(f"  [{key}] ", style=STYLE_BOLD_CYAN)
        menu_text.append(f"{description}\n", style=STYLE_FG)

    panel = Panel(
        menu_text,
        title=title,
        title_align="left",
        box=box.ROUNDED,
        border_style=STYLE_PINK,
        padding=(0, 1),
    )
    console.print(panel)


# =============================================================================
# Prompt Component
# =============================================================================


def prompt_user(console: Console, message: str, default: str = "") -> str:
    """Display a styled input prompt and get user input.

    Args:
        console: Rich Console instance for output.
        message: Prompt message to display.
        default: Default value if user enters nothing.

    Returns:
        User's input string, or default if empty.

    """
    prompt_text = Text()
    prompt_text.append("\u25b6 ", style=STYLE_CYAN)  # ▶
    prompt_text.append(message, style=STYLE_CYAN)
    if default:
        prompt_text.append(f" [{default}]", style=Style(color=NEON_COLORS["foreground"], dim=True))
    prompt_text.append(": ", style=STYLE_CYAN)

    console.print(prompt_text, end="")
    user_input = input()
    return user_input if user_input else default


# =============================================================================
# NeonThrobber Class
# =============================================================================


class NeonThrobber:
    """Animated gradient bar for progress indication.

    Creates a scrolling rainbow effect using horizontal line characters
    and the gradient color palette.

    Attributes:
        _offset: Current animation offset for color cycling.
        _char: Character used to render the throbber bar.

    """

    def __init__(self) -> None:
        """Initialize the throbber with offset tracking."""
        self._offset = 0
        self._char = "\u2501"  # ━

    def render(self, width: int = 40) -> Text:
        """Render the throbber bar with current animation frame.

        Args:
            width: Width of the throbber bar in characters.

        Returns:
            Rich Text object containing the styled throbber.

        """
        result = Text()
        num_colors = len(GRADIENT_COLORS)

        for i in range(width):
            color_index = (i + self._offset) % num_colors
            color = GRADIENT_COLORS[color_index]
            result.append(self._char, style=Style(color=color))

        # Advance offset for next frame
        self._offset = (self._offset + 1) % num_colors

        return result

    def reset(self) -> None:
        """Reset the animation offset to the beginning.

        Returns:
            None

        """
        self._offset = 0


class CrazySpinner:
    """Wild multi-pattern spinner with gradient colors.

    Displays multiple spinner characters simultaneously, each with its own
    animation pattern and gradient color cycling. Creates a chaotic but
    visually striking loading indicator.

    """

    # Multiple spinner pattern sets - each runs independently
    SPINNERS = [
        # Braille dots - vertical bounce
        ["⠁", "⠂", "⠄", "⡀", "⡀", "⠄", "⠂", "⠁"],
        # Braille dots - horizontal sweep
        ["⠈", "⠐", "⠠", "⢀", "⢀", "⠠", "⠐", "⠈"],
        # Quarter blocks - rotation
        ["◴", "◷", "◶", "◵"],
        # Arrows - spinning
        ["←", "↖", "↑", "↗", "→", "↘", "↓", "↙"],
        # Box drawing - morphing
        ["┤", "┘", "┴", "└", "├", "┌", "┬", "┐"],
        # Stars - twinkling
        ["✶", "✷", "✸", "✹", "✺", "✹", "✸", "✷"],
        # Geometric - pulsing
        ["◯", "◎", "◉", "●", "◉", "◎"],
        # Dice - rolling
        ["⚀", "⚁", "⚂", "⚃", "⚄", "⚅"],
    ]

    def __init__(self, num_spinners: int = 3) -> None:
        """Initialize with multiple independent spinner states.

        Args:
            num_spinners: Number of spinner characters to display.

        """
        self._num_spinners = num_spinners
        self._frame = 0
        # Each spinner gets a random-ish starting pattern and offset
        self._spinner_indices = [i % len(self.SPINNERS) for i in range(num_spinners)]
        self._offsets = [i * 2 for i in range(num_spinners)]

    def render(self) -> Text:
        """Render the current frame of all spinners with gradient colors.

        Returns:
            Rich Text with multiple animated spinner characters.

        """
        result = Text()
        result.append(" ")  # Leading space

        num_colors = len(GRADIENT_COLORS)

        for i in range(self._num_spinners):
            # Get this spinner's pattern
            pattern_idx = (self._spinner_indices[i] + self._frame // 8) % len(self.SPINNERS)
            pattern = self.SPINNERS[pattern_idx]

            # Get current character from pattern
            char_idx = (self._frame + self._offsets[i]) % len(pattern)
            char = pattern[char_idx]

            # Color cycles through gradient, offset per spinner
            color_idx = (self._frame + i * 3) % num_colors
            color = GRADIENT_COLORS[color_idx]

            result.append(char, style=Style(color=color, bold=True))

        # Advance frame
        self._frame += 1

        return result

    def reset(self) -> None:
        """Reset animation to beginning."""
        self._frame = 0


# =============================================================================
# LiveToolPanel Class
# =============================================================================


class LiveToolPanel:
    """Manage a single tool call panel with Live updates.

    Consolidates tool call display and result into a single live-updating panel.
    Shows an animated throbber while waiting for the result, then replaces it
    with the actual result content.

    In quiet mode, delegates to print_tool_call and skips result display.

    Args:
        console: Rich Console instance for output.
        tool_use_id: Unique identifier for the tool use.
        name: Name of the tool being called.
        args: Dictionary of arguments passed to the tool.
        quiet_mode: If True, use static display instead of Live updates.

    Usage:
        panel = LiveToolPanel(console, "tool-123", "Bash", {"command": "ls"})
        panel.start()
        # ... tool executes ...
        panel.set_result("file1.txt\nfile2.txt", is_error=False)
        panel.finish()

    """

    def __init__(
        self,
        console: Console,
        tool_use_id: str,
        name: str,
        args: dict[str, object],
        quiet_mode: bool = False,
    ) -> None:
        """Initialize the LiveToolPanel.

        Args:
            console: Rich Console instance for output.
            tool_use_id: Unique identifier for the tool use.
            name: Name of the tool being called.
            args: Dictionary of arguments passed to the tool.
            quiet_mode: If True, use static display instead of Live updates.

        """
        self._console = console
        self._tool_use_id = tool_use_id
        self._name = name
        self._args = args
        self._result: str | None = None
        self._is_error: bool = False
        self._live: Live | None = None
        self._spinner = CrazySpinner(num_spinners=3)
        self._quiet_mode = quiet_mode
        self._frame = 0  # Animation frame counter for Edit surgery visualization

    def _build_tool_header_content(self) -> Text:
        """Build the tool call header content.

        Delegates to shared _build_tool_header() helper for consistent styling.

        Returns:
            Rich Text containing the styled tool header.

        """
        return _build_tool_header(self._name, self._args, self._quiet_mode)

    def _build_result_content_internal(self, max_lines: int = 20) -> Text | Syntax | Group:
        """Build the result content with syntax highlighting.

        Delegates to shared _build_result_content() helper for consistent styling.
        Special handling for Glob and Grep tools to show file counts and formatted lists.

        Args:
            max_lines: Maximum number of lines to display.

        Returns:
            Rich renderable containing the styled result.

        """
        if self._result is None:
            return Text()
        if not self._result.strip():
            return Text()

        # Special handling for Glob results - show file count and formatted list
        if self._name == "Glob" and not self._is_error:
            return self._build_glob_result(max_lines)

        # Special handling for Grep results - show match count
        if self._name == "Grep" and not self._is_error:
            return self._build_grep_result(max_lines)

        # Special handling for Edit results - show harmony restored message
        if self._name == "Edit" and not self._is_error:
            return self._build_edit_result()

        result, _ = _build_result_content(self._result, self._is_error, max_lines)
        return result

    def _build_glob_result(self, max_lines: int = 15) -> Text:
        """Build formatted Glob result showing file count and paths.

        Args:
            max_lines: Maximum number of files to display.

        Returns:
            Rich Text with formatted file list.

        """
        assert self._result is not None  # Caller ensures this
        lines = [line for line in self._result.strip().split("\n") if line.strip()]
        total_files = len(lines)

        result = Text()

        # Show count with sparkles
        result.append("\u2728 ", style=STYLE_YELLOW)  # ✨
        result.append(f"Found {total_files} file{'s' if total_files != 1 else ''}", style=STYLE_BOLD_CYAN)
        result.append("\n")

        # Show files (truncated if needed)
        display_lines = lines[:max_lines]
        for i, filepath in enumerate(display_lines):
            # Extract just the filename for compact display
            filename = filepath.rsplit("/", 1)[-1] if "/" in filepath else filepath
            # Get parent directory for context
            parent = filepath.rsplit("/", 1)[0] if "/" in filepath else ""

            result.append("  ")
            result.append(filename, style=STYLE_CYAN)
            if parent:
                # Show truncated parent path
                if len(parent) > 40:
                    parent = "..." + parent[-37:]
                result.append(f"  {parent}", style=STYLE_DIM)
            if i < len(display_lines) - 1:
                result.append("\n")

        # Show truncation indicator
        if total_files > max_lines:
            result.append("\n")
            result.append(
                f"  ... and {total_files - max_lines} more",
                style=Style(color=NEON_COLORS["yellow"], italic=True),
            )

        return result

    def _build_grep_result(self, max_lines: int = 20) -> Text | Syntax | Group:
        """Build formatted Grep result showing match count.

        Args:
            max_lines: Maximum number of lines to display.

        Returns:
            Rich renderable with formatted grep output.

        """
        assert self._result is not None  # Caller ensures this
        lines = [line for line in self._result.strip().split("\n") if line.strip()]
        total_matches = len(lines)

        result = Text()

        # Show count with sparkles
        result.append("\u2728 ", style=STYLE_YELLOW)  # ✨
        result.append(f"Found {total_matches} match{'es' if total_matches != 1 else ''}", style=STYLE_BOLD_CYAN)
        result.append("\n")

        # Use standard result formatting for the actual content
        content, _ = _build_result_content(self._result, self._is_error, max_lines)

        return Group(result, content)

    def _build_edit_result(self) -> Text:
        """Build formatted Edit result with surgery completion message.

        Returns:
            Rich Text with harmony restored visualization.

        """
        result = Text()

        # Harmony restored header with energy symbols
        result.append("  ")
        result.append("✓ ", style=STYLE_BOLD_GREEN)
        result.append("HARMONY RESTORED", style=Style(color=NEON_COLORS["green"], bold=True))
        result.append("\n\n")

        # Energy flow visualization
        flow_width = 30
        for i in range(flow_width):
            # Gradient from cyan through green
            t = i / flow_width
            color = _interpolate_color(NEON_COLORS["cyan"], NEON_COLORS["green"], t)
            char = SURGERY_ENERGY_FLOW[i % len(SURGERY_ENERGY_FLOW)]
            result.append(char, style=Style(color=color))

        result.append("\n\n")

        # Show the actual result if present
        if self._result and self._result.strip():
            result.append("  ", style=STYLE_DIM)
            result.append(self._result.strip(), style=STYLE_DIM)

        return result

    def _build_surgery_phase_indicator(self) -> Text:
        """Build animated surgery phase indicator for Edit tool.

        Returns:
            Rich Text with animated energy flow and phase status.

        """
        # Advance frame
        self._frame += 1

        result = Text()

        # Determine current phase based on frame (cycles through phases)
        # At 10fps refresh, 4 frames per phase = 0.4s per phase, full cycle in 1.6s
        phase_duration = 4
        phase_idx = (self._frame // phase_duration) % len(SURGERY_PHASES)
        phase_name, phase_color_key = SURGERY_PHASES[phase_idx]
        phase_color = NEON_COLORS[phase_color_key]

        # Energy flow animation - cycles every frame
        energy_idx = self._frame % len(SURGERY_ENERGY_FLOW)
        energy_char = SURGERY_ENERGY_FLOW[energy_idx]

        # Chakra symbol animation - cycles every 2 frames
        chakra_idx = (self._frame // 2) % len(SURGERY_CHAKRA_SYMBOLS)
        chakra_char = SURGERY_CHAKRA_SYMBOLS[chakra_idx]

        result.append("\n\n  ")
        result.append(energy_char, style=Style(color=phase_color))
        result.append(" ", style=Style(color=phase_color))
        result.append(chakra_char, style=Style(color=phase_color, bold=True))
        result.append(f" {phase_name} ", style=Style(color=phase_color, bold=True))
        result.append(chakra_char, style=Style(color=phase_color, bold=True))
        result.append(" ", style=Style(color=phase_color))
        result.append(energy_char, style=Style(color=phase_color))

        return result

    def _render_panel(self) -> Panel:
        """Render the current state as a Panel.

        Shows tool call header + either throbber (if waiting) or result.

        Returns:
            Rich Panel containing the consolidated tool call display.

        """
        # Build header
        header = self._build_tool_header_content()

        # Determine border color based on tool type and error state
        if self._name == "Skill":
            border_color = NEON_COLORS["cyan"]
        elif self._is_error:
            border_color = NEON_COLORS["red"]
        elif self._name == "Edit" and self._result is None:
            # Animated border color for Edit surgery - pulse between purple and cyan
            # At 10fps, modulo 10 = 1 second full cycle
            pulse = abs((self._frame % 10) - 5) / 5
            border_color = _interpolate_color(NEON_COLORS["purple"], NEON_COLORS["cyan"], pulse)
        else:
            border_color = NEON_COLORS["purple"]

        # Build content: header + inline spinner (if waiting) or result
        if self._result is None:
            # Special handling for Edit - show surgery phase animation
            if self._name == "Edit":
                surgery_indicator = self._build_surgery_phase_indicator()
                header_with_surgery = Text()
                header_with_surgery.append_text(header)
                header_with_surgery.append_text(surgery_indicator)
                header_with_surgery.append_text(self._spinner.render())
                content = Group(header_with_surgery)
            else:
                # Show spinner inline with header
                header_with_spinner = Text()
                header_with_spinner.append_text(header)
                header_with_spinner.append_text(self._spinner.render())
                content = Group(header_with_spinner)
        elif self._name == "Skill":
            # Skip output section for Skill calls - the header already shows skill name
            content = Group(header)
        elif self._quiet_mode:
            # Quiet mode complete: header only
            content = Group(header)
        else:
            # Normal mode: show result
            result_content = self._build_result_content_internal()

            # Add title for result section
            if self._is_error:
                result_title = Text()
                result_title.append("\u274c Error", style=STYLE_BOLD_RED)
            else:
                result_title = Text()
                result_title.append("Output", style=STYLE_BOLD_CYAN)

            if isinstance(result_content, Text) and not result_content.plain.strip():
                # Empty result - just show header (wrap in Group for type consistency)
                content = Group(header)
            else:
                content = Group(
                    header,
                    Text("\n"),
                    result_title,
                    Text("\n"),
                    result_content,
                )

        return Panel(
            content,
            box=box.ROUNDED,
            border_style=Style(color=border_color),
            style=STYLE_PANEL_BG,
            padding=(0, 1),
        )

    def __rich__(self) -> Panel:
        """Return renderable for Rich Live refresh."""
        return self._render_panel()

    def start(self) -> None:
        """Start Live context and show tool call with animated throbber."""
        self._console.print()

        self._live = Live(
            self,
            console=self._console,
            refresh_per_second=10,
            transient=False,
        )
        self._live.start()

    def set_result(self, content: str, is_error: bool = False) -> None:
        """Store result and update the display.

        Args:
            content: The result content from the tool.
            is_error: Whether this is an error result.

        """
        self._result = content
        self._is_error = is_error

        if self._live is not None:
            self._live.update(self._render_panel())

    def finish(self) -> None:
        """Stop Live context. Final panel state persists on screen."""
        if self._live is not None:
            self._live.stop()
            self._live = None


# =============================================================================
# LiveToolPanelRegistry Class
# =============================================================================


class LiveToolPanelRegistry:
    """Registry for tracking multiple concurrent tool panels.

    Provides a central registry for LiveToolPanel instances, indexed by
    tool_use_id. This enables correlation between ToolUseBlock and
    ToolResultBlock events when processing streaming responses.

    Args:
        console: Rich Console instance for creating panels.
        quiet_mode: If True, panels use static display instead of Live updates.

    Usage:
        registry = LiveToolPanelRegistry(console, quiet_mode=False)
        panel = registry.create("tool-123", "Bash", {"command": "ls"})
        # ... tool executes ...
        panel.set_result("file1.txt", is_error=False)
        panel.finish()
        registry.remove("tool-123")

    """

    def __init__(self, console: Console, quiet_mode: bool = False) -> None:
        """Initialize the registry.

        Args:
            console: Rich Console instance for creating panels.
            quiet_mode: If True, panels use static display instead of Live updates.

        """
        self._console = console
        self._quiet_mode = quiet_mode
        self._panels: dict[str, LiveToolPanel] = {}

    def create(
        self,
        tool_use_id: str,
        name: str,
        args: dict[str, object],
    ) -> LiveToolPanel:
        """Create and register a new panel, then start it.

        Args:
            tool_use_id: Unique identifier for the tool use.
            name: Name of the tool being called.
            args: Dictionary of arguments passed to the tool.

        Returns:
            The created and started LiveToolPanel.

        """
        # Finish and remove existing panel if duplicate tool_use_id
        if tool_use_id in self._panels:
            self._panels[tool_use_id].finish()
            del self._panels[tool_use_id]

        panel = LiveToolPanel(
            console=self._console,
            tool_use_id=tool_use_id,
            name=name,
            args=args,
            quiet_mode=self._quiet_mode,
        )
        self._panels[tool_use_id] = panel
        panel.start()
        return panel

    def get(self, tool_use_id: str) -> LiveToolPanel | None:
        """Get a panel by its tool_use_id.

        Args:
            tool_use_id: The unique identifier of the tool use.

        Returns:
            The LiveToolPanel if found, None otherwise.

        """
        return self._panels.get(tool_use_id)

    def remove(self, tool_use_id: str) -> None:
        """Remove a panel from the registry.

        Call this after finishing a panel to clean up the registry.

        Args:
            tool_use_id: The unique identifier of the tool use to remove.

        Returns:
            None

        """
        self._panels.pop(tool_use_id, None)

    def finish_all(self) -> None:
        """Finalize all remaining panels.

        Calls finish() on each panel and clears the registry.
        Use this for cleanup when a response ends unexpectedly.

        Returns:
            None

        """
        for panel in self._panels.values():
            panel.finish()
        self._panels.clear()


# =============================================================================
# ShutdownPanel Class
# =============================================================================


@dataclass
class ShutdownStep:
    """A step in the shutdown process.

    Attributes:
        message: Description of the shutdown step.
        status: Current status of the step ("pending", "in_progress", or "completed").

    """

    message: str
    status: str = "pending"  # pending, in_progress, completed


class ShutdownPanel:
    """Live-updating panel for showing shutdown progress.

    Consolidates all shutdown messages into a single panel that
    updates in place, showing the progression of shutdown steps.

    Args:
        console: Rich Console instance for output.

    Usage:
        panel = ShutdownPanel(console)
        panel.start("Received SIGINT, shutting down")
        panel.add_step("Terminating running agent...")
        panel.complete_step(0)  # Mark first step as completed
        panel.finish()

    """

    def __init__(self, console: Console) -> None:
        """Initialize the ShutdownPanel.

        Args:
            console: Rich Console instance for output.

        """
        self._console = console
        self._steps: list[ShutdownStep] = []
        self._live: Live | None = None

    def _render_panel(self) -> Panel:
        """Render the current state as a Panel.

        Returns:
            Rich Panel containing all shutdown steps with status icons.

        """
        content = Text()

        for i, step in enumerate(self._steps):
            config = STATUS_CONFIG.get(step.status, STATUS_CONFIG["pending"])
            icon = config["icon"]
            color = config["color"]

            content.append(f"{icon} ", style=Style(color=color))
            content.append(step.message, style=STYLE_FG)

            if i < len(self._steps) - 1:
                content.append("\n")

        return Panel(
            content,
            box=box.ROUNDED,
            border_style=STYLE_YELLOW,
            padding=(0, 1),
        )

    def start(self, initial_message: str) -> None:
        """Start the live panel with an initial message.

        Args:
            initial_message: The first message to display (e.g., "Received SIGINT").

        Returns:
            None

        """
        self._steps.append(ShutdownStep(message=initial_message, status="completed"))

        self._console.print()  # Add spacing before panel
        self._live = Live(
            self._render_panel(),
            console=self._console,
            refresh_per_second=10,
            transient=True,
        )
        self._live.start()

    def add_step(self, message: str, status: str = "in_progress") -> int:
        """Add a new step to the shutdown sequence.

        Args:
            message: The step message to display.
            status: Initial status ("pending", "in_progress", "completed").

        Returns:
            Index of the added step for later updates.

        """
        self._steps.append(ShutdownStep(message=message, status=status))
        if self._live is not None:
            self._live.update(self._render_panel())
        return len(self._steps) - 1

    def complete_step(self, index: int) -> None:
        """Mark a step as completed.

        Args:
            index: Index of the step to mark as completed.

        Returns:
            None

        """
        if 0 <= index < len(self._steps):
            self._steps[index].status = "completed"
            if self._live is not None:
                self._live.update(self._render_panel())

    def complete_last_step(self) -> None:
        """Mark the last step as completed.

        Returns:
            None

        """
        if self._steps:
            self.complete_step(len(self._steps) - 1)

    def finish(self) -> None:
        """Stop the live context and print the final panel.

        Call this when all shutdown steps are complete.

        Returns:
            None

        """
        if self._live is not None:
            self._live.stop()
            self._live = None

        # Print the final panel (non-transient)
        if self._steps:
            self._console.print(self._render_panel())


# Global shutdown panel instance for cross-module access
_shutdown_panel: ShutdownPanel | None = None


def get_shutdown_panel() -> ShutdownPanel | None:
    """Get the current shutdown panel instance.

    Returns:
        ShutdownPanel | None: The current shutdown panel, or None if not set.

    """
    return _shutdown_panel


def set_shutdown_panel(panel: ShutdownPanel | None) -> None:
    """Set the global shutdown panel instance.

    Args:
        panel: The ShutdownPanel instance to set, or None to clear.

    Returns:
        None

    """
    global _shutdown_panel
    _shutdown_panel = panel


# =============================================================================
# Convenience Functions
# =============================================================================


def create_neon_console() -> NeonConsole:
    """Create and return a new NeonConsole instance.

    Returns:
        Configured NeonConsole ready for use.

    """
    return NeonConsole()


def get_status_style(status: str) -> Style:
    """Get the Rich Style for a given status.

    Args:
        status: One of "pending", "in_progress", "completed", "failed".

    Returns:
        Rich Style object with appropriate color.

    """
    config = STATUS_CONFIG.get(status, STATUS_CONFIG["pending"])
    return Style(color=config["color"])
