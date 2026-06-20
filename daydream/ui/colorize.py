"""Output colorization for tool results.

File/git/shell regex patterns, shell-syntax detection, and per-line neon
colorization.
"""

import re

from rich.style import Style
from rich.text import Text

from daydream.ui.theme import (
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
    STYLE_PINK,
    STYLE_PURPLE,
    STYLE_RED,
    STYLE_YELLOW,
)

_FILE_PATH_PATTERN = re.compile(r"(\.?/?(?:[\w.-]+/)*[\w.-]+\.\w+)")
_LINE_NUMBER_PATTERN = re.compile(r"^(\s*)(\d+)([:\-\|])")
_ERROR_KEYWORDS = re.compile(r"\b(error|Error|ERROR|failed|Failed|FAILED|exception|Exception|EXCEPTION|Traceback)\b")
_SUCCESS_KEYWORDS = re.compile(r"\b(passed|Passed|PASSED|success|Success|SUCCESS|ok|OK|done|Done|DONE)\b")
_WARNING_KEYWORDS = re.compile(r"\b(warning|Warning|WARNING|warn|Warn|WARN|skip|Skip|SKIP|skipped|Skipped)\b")
_NUMBER_PATTERN = re.compile(r"\b(\d+)\b")
_STRING_PATTERN = re.compile(r"(['\"])(.*?)\1")
_ARROW_PATTERN = re.compile(r"(->|=>|-->|==>)")
_BRACKET_PATTERN = re.compile(r"([\[\]{}()])")

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
    if _SHEBANG_PATTERN.search(content):
        return True

    prompt_matches = len(_SHELL_DETECT_PROMPT_PATTERN.findall(content))
    if prompt_matches >= 2:
        return True

    command_matches = len(_SHELL_DETECT_COMMAND_PATTERN.findall(content))
    if command_matches >= 3:
        return True

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
    git_result = _colorize_git_line(line)
    if git_result is not None:
        return git_result

    result = Text()

    if is_error:
        # Highlight file paths in orange so they stand out against the red base.
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

    segments: list[tuple[int, int, str, Style]] = []

    for match in _FILE_PATH_PATTERN.finditer(line):
        segments.append((match.start(), match.end(), match.group(1),
                        STYLE_CYAN))

    for match in _ERROR_KEYWORDS.finditer(line):
        segments.append((match.start(), match.end(), match.group(1),
                        STYLE_BOLD_RED))

    for match in _SUCCESS_KEYWORDS.finditer(line):
        segments.append((match.start(), match.end(), match.group(1),
                        STYLE_BOLD_GREEN))

    for match in _WARNING_KEYWORDS.finditer(line):
        segments.append((match.start(), match.end(), match.group(1),
                        STYLE_BOLD_YELLOW))

    for match in _STRING_PATTERN.finditer(line):
        segments.append((match.start(), match.end(), match.group(0),
                        STYLE_ORANGE))

    for match in _ARROW_PATTERN.finditer(line):
        segments.append((match.start(), match.end(), match.group(1),
                        STYLE_BOLD_PINK))

    for match in _BRACKET_PATTERN.finditer(line):
        segments.append((match.start(), match.end(), match.group(1),
                        STYLE_PURPLE))

    for match in _SHELL_VAR_PATTERN.finditer(line):
        segments.append((match.start(), match.end(), match.group(1),
                        STYLE_YELLOW))

    for match in _SHELL_PROMPT_PATTERN.finditer(line):
        segments.append((match.start(), match.end(), match.group(1),
                        STYLE_CYAN))

    for match in _PIPE_REDIRECT_PATTERN.finditer(line):
        segments.append((match.start(), match.end(), match.group(1),
                        STYLE_PINK))

    for match in _COMMAND_PATTERN.finditer(line):
        segments.append((match.start(), match.end(), match.group(1),
                        Style(color=NEON_COLORS["orange"], bold=True)))

    # Sort by start, then longest-first, so wider matches win overlaps.
    segments.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    last_end = 0
    used_ranges: list[tuple[int, int]] = []

    for start, end, text, style in segments:
        overlaps = any(not (end <= used_start or start >= used_end)
                      for used_start, used_end in used_ranges)
        if overlaps:
            continue

        if start > last_end:
            result.append(line[last_end:start],
                         style=STYLE_FG)
        result.append(text, style=style)
        used_ranges.append((start, end))
        last_end = end

    if last_end < len(line):
        result.append(line[last_end:],
                     style=STYLE_FG)

    return result
