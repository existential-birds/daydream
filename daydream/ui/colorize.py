"""Output colorization for tool results.

File/git/shell regex patterns, shell-syntax detection, per-line neon
colorization, and the static ``print_tool_result`` renderer.
"""

import re

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.style import Style
from rich.text import Text

from daydream.ui.theme import (
    _RESULT_MAX_LINES,
    NEON_COLORS,
    STYLE_BOLD_CYAN,
    STYLE_BOLD_GREEN,
    STYLE_BOLD_PINK,
    STYLE_BOLD_RED,
    STYLE_BOLD_YELLOW,
    STYLE_CYAN,
    STYLE_FG,
    STYLE_GREEN,
    STYLE_ORANGE,
    STYLE_PANEL_BG,
    STYLE_PINK,
    STYLE_PURPLE,
    STYLE_RED,
    STYLE_YELLOW,
)

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
    max_lines: int = _RESULT_MAX_LINES,
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
    # Local import to keep the module DAG acyclic: tools.py imports colorize
    # helpers, so colorize.py must not import tools at module load time.
    from daydream.ui.tools import _build_result_content

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
