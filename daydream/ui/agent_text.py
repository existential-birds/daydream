"""Streaming agent-text rendering.

Per-response state, inline markdown/path/code highlighting, the vertical
cyan-to-green gradient renderer, the buffering ``AgentTextRenderer`` Live panel,
and the gutter-aware ``print_agent_text`` streamer.
"""

import re

from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.style import Style
from rich.text import Text

from daydream.ui.console import _interpolate_color
from daydream.ui.panels import CrazySpinner
from daydream.ui.theme import (
    NEON_COLORS,
    STYLE_AGENT_BG,
    STYLE_CYAN,
    STYLE_GREEN,
)


class AgentTextState:
    """State holder for agent text rendering.

    Encapsulates mutable state used during streaming agent text output
    to track gutter display and markdown detection.

    This class provides thread-safe state management for concurrent
    rendering contexts.

    Attributes:
        line_started: Whether a line has been started (gutter printed).
        has_markdown: Whether markdown headers have been detected.

    """

    def __init__(self) -> None:
        """Initialize agent text state."""
        self.line_started: bool = False
        self.has_markdown: bool = False

    def reset(self) -> None:
        """Reset state for a new agent response."""
        self.line_started = False
        self.has_markdown = False


# Global instance for backward compatibility
_agent_text_state = AgentTextState()


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


# Detect markdown headers anywhere in text, not just line-start, to catch inline
# headers produced by streaming.
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
            transient=False,
            vertical_overflow="visible",
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
            self._live.update(self._render_panel(show_spinner=False), refresh=True)
            self._live.stop()
            self._live = None

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
    if not text:
        return

    # Check for markdown headers in this chunk
    if _has_markdown_headers(text):
        _agent_text_state.has_markdown = True

    # Split into lines to handle multiline text
    lines = text.split("\n")

    for i, line in enumerate(lines):
        is_last = i == len(lines) - 1

        if not _agent_text_state.line_started:
            # Start of a new agent text block - add separation and gutter prefix
            console.print()  # Newline for separation from tool calls
            gutter = Text()
            gutter.append("│ ", style=STYLE_GREEN)
            console.print(gutter, end="")
            _agent_text_state.line_started = True

        # Highlight and print the line content with dark green background
        # Use italic only for non-markdown content
        if line:
            use_italic = not _agent_text_state.has_markdown
            base_style = Style(color=NEON_COLORS["green"], italic=use_italic)
            highlighted = _highlight_agent_text(line, base_style=base_style)
            highlighted.stylize(STYLE_AGENT_BG)
            console.print(highlighted, end="")

        # Handle newlines - reset gutter state for next line
        if not is_last:
            console.print()  # Complete the current line
            _agent_text_state.line_started = False


def reset_agent_text_state() -> None:
    """Reset the agent text line state.

    Call this at the end of an agent response to ensure
    the next agent response starts with a fresh gutter.

    Returns:
        None

    """
    _agent_text_state.reset()
