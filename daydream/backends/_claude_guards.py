# daydream/backends/claude.py
"""Claude tool guards: audit confinement, read-only commands, and foreground execution."""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path, PureWindowsPath
from typing import Any

from claude_agent_sdk import HookJSONOutput
from claude_agent_sdk.types import (
    HookCallback,
)

from daydream.config import TEST_WALL_BUDGET_S

# Shared read-only Bash families; shell controls are rejected separately.
# _render_bash_allowlist renders this same list into inspection prompts.
READ_ONLY_BASH_ALLOWLIST: tuple[str, ...] = (
    "ls",
    "cat",
    "git status",
    "git log",
    "git show",
    "git blame",
    "git diff",
)

# Non-POSIX shlex exposes bare controls; leading parentheses deny and mid-word parentheses are Bash errors.
# Quoted | ; & < > are inert, but $/backticks in double quotes still expand: this is not a full shell sandbox.
_SHELL_CONTROL_TOKENS: frozenset[str] = frozenset({"|", ";", "&", "`", "$", "<", ">"})

# Git options that write the command's output to a file. Scanned only after a
# matched ``git …`` allowlist family, so ``ls``/``cat`` never hit it.
_GIT_WRITE_OPTIONS: tuple[str, ...] = ("--output",)

# ``.*`` fires the guard for EVERY tool call so it can fail-closed (allow only
# the safe set); a deny-list of mutating tools was fail-open.
_READ_ONLY_HOOK_MATCHER = ".*"

# Always deny catastrophic filesystem-root scans/wipes by raw command pattern.
_DANGEROUS_COMMAND_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*find\s+/(\s|$)"),  # find / ...  (root-anchored scan)
    re.compile(r"^\s*grep\b.*\s/\s*$"),  # grep ... /  (root is the sole trailing path)
    # Match recursive rm flags in any order plus a standalone / or /* target.
    # Subpaths such as /home are outside this catastrophic-wipe backstop.
    re.compile(r"^\s*rm\b(?=.*(?:^|\s)(?:-\w*[rR]\w*|--recursive)\b)(?=.*(?:^|\s)/\*?(?:\s|$)).*$"),
)

# Tools unconditionally permitted under the read-only profile (Bash handled
# separately via the command allowlist).
_READ_ONLY_ALLOWED_TOOLS: frozenset[str] = frozenset({"Read", "Grep", "Glob", "StructuredOutput"})

# Match the largest host wall budget so foreground test suites can exceed
# Claude's default 600s shell timeout without needing background execution.
_BASH_TIMEOUT_MS = int(TEST_WALL_BUDGET_S * 1000)

# Disable background work whose results CLI turn-end cleanup would kill; reinforce with the tool guard.
_CLI_ENV: dict[str, str] = {
    "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
    "BASH_DEFAULT_TIMEOUT_MS": str(_BASH_TIMEOUT_MS),
    "BASH_MAX_TIMEOUT_MS": str(_BASH_TIMEOUT_MS),
}

_AUDIT_TOOLS: tuple[str, ...] = ("Read", "Grep", "Glob", "StructuredOutput")
_AUDIT_GREP_OUTPUT_MODES = frozenset({"content", "files_with_matches", "count"})
_AUDIT_GREP_BOOL_FIELDS = frozenset({"-n", "-i", "multiline"})
_AUDIT_GREP_INT_FIELDS = frozenset({"head_limit", "offset"})


def _audit_cli_env(root: Path, *, base_environment: dict[str, str] | None = None) -> dict[str, str]:
    """Return SDK environment overrides bound to one standalone audit repo."""
    git_dir = root / ".git"
    return {
        **(base_environment or {}),
        **_CLI_ENV,
        "PWD": str(root),
        "OLDPWD": str(root),
        "GIT_DIR": str(git_dir),
        "GIT_WORK_TREE": str(root),
        "GIT_INDEX_FILE": str(git_dir / "index"),
        "GIT_OBJECT_DIRECTORY": str(git_dir / "objects"),
        "GIT_COMMON_DIR": str(git_dir),
        "GIT_ALTERNATE_OBJECT_DIRECTORIES": "",
        "GIT_CEILING_DIRECTORIES": str(root.parent),
        "GIT_PREFIX": "",
    }


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _valid_relative_audit_path(value: str) -> bool:
    """Reject path spellings whose meaning differs across supported hosts."""
    if not value or "\x00" in value or "\\" in value:
        return False
    windows = PureWindowsPath(value)
    path = Path(value)
    return not path.is_absolute() and not windows.drive and ".." not in path.parts


def _audit_regular_file(root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        return None
    if PureWindowsPath(value).drive:
        return None
    candidate = Path(value)
    if ".." in candidate.parts:
        return None
    try:
        resolved = (candidate if candidate.is_absolute() else root / candidate).resolve(strict=True)
        if not resolved.is_relative_to(root) or not resolved.is_file():
            return None
    except (OSError, RuntimeError, ValueError):
        return None
    return resolved


def _audit_directory(root: Path, value: Any) -> Path | None:
    if value is None:
        return root
    if not isinstance(value, str) or not _valid_relative_audit_path(value):
        return None
    try:
        resolved = (root / value).resolve(strict=True)
        if not resolved.is_relative_to(root) or not resolved.is_dir():
            return None
    except (OSError, RuntimeError, ValueError):
        return None
    return resolved


def _audit_path_crosses_lexical_symlink(root: Path, value: str) -> bool:
    """Return whether any component of one validated relative path is a link."""
    candidate = root
    try:
        for part in Path(value).parts:
            candidate /= part
            if candidate.is_symlink():
                return True
    except OSError:
        return True
    return False


def _audit_symlink_inventory(root: Path) -> frozenset[Path]:
    """Inventory lexical links below *root* without traversing link targets."""
    links: set[Path] = set()
    pending = [root]
    try:
        while pending:
            directory = pending.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    path = directory / entry.name
                    if entry.is_symlink():
                        links.add(path)
                    elif entry.is_dir(follow_symlinks=False):
                        pending.append(path)
    except OSError as exc:
        raise ValueError("cannot inventory audit-root symlinks") from exc
    return frozenset(links)


def _denies_git_output_option(argv: list[str], start: int) -> bool:
    """Reject Git --output[=] before the standalone -- separator; subsequent tokens are literal paths."""
    for tok in argv[start:]:
        if tok == "--":
            return False
        if tok in _GIT_WRITE_OPTIONS or any(tok.startswith(opt + "=") for opt in _GIT_WRITE_OPTIONS):
            return True
    return False


def _shlex_tokens(cmd: str, *, posix: bool, words: bool) -> list[str] | None:
    """Lex without comments so mid-word # cannot hide controls; malformed quotes return None.
    words=True preserves argv words; False exposes bare control characters.
    """
    lexer = shlex.shlex(cmd, posix=posix)
    lexer.commenters = ""  # '#' is never a comment; keep every character.
    lexer.whitespace_split = words
    try:
        return list(lexer)
    except ValueError:
        return None


def _is_read_only_command(cmd: str) -> bool:
    """Admit an allowlisted argv family without raw line breaks, shell controls, or Git output writes.
    Malformed quoting and leading parentheses fail closed; mid-word parentheses remain Bash errors.
    Quoted metacharacters retain the double-quote limitation documented at _SHELL_CONTROL_TOKENS.
    """
    stripped = cmd.strip()
    if not stripped:
        return False
    if "\n" in cmd or "\r" in cmd:
        return False
    # Expose individual bare controls, including both characters of &&.
    tokens = _shlex_tokens(stripped, posix=False, words=False)
    if tokens is None:
        return False  # Malformed quoting -- deny (fail-closed).
    for tok in tokens:
        if tok in _SHELL_CONTROL_TOKENS:
            return False
    if tokens[0] in ("(", ")"):
        # Deny leading subshell groups; mid-command unquoted parentheses are
        # Bash syntax errors and retain their previously allowed treatment.
        return False
    # Match whole argv families (never git logfoo), then reject Git output-write options.
    argv = _shlex_tokens(stripped, posix=True, words=True)
    if argv is None:
        return False  # Malformed quoting -- deny (fail-closed).
    for family in READ_ONLY_BASH_ALLOWLIST:
        words = family.split()
        if argv[: len(words)] == words:
            if family.startswith("git ") and _denies_git_output_option(argv, len(words)):
                return False
            return True
    return False


def _tool_input(input_data: Any) -> dict[str, Any]:
    """Defensively extract ``tool_input`` from a PreToolUse payload ({} when malformed)."""
    if isinstance(input_data, dict):
        tool_input = input_data.get("tool_input")
        if isinstance(tool_input, dict):
            return tool_input
    return {}


def _bash_command(input_data: Any) -> str | None:
    """Extract Bash command text: None means non-Bash/malformed payload; malformed Bash commands
    return an empty string so the read-only guard fails closed.
    """
    if not isinstance(input_data, dict) or input_data.get("tool_name") != "Bash":
        return None
    raw = _tool_input(input_data).get("command")
    return raw if isinstance(raw, str) else ""


def _read_only_deny(reason: str) -> HookJSONOutput:
    """Build a PreToolUse deny output (``permissionDecision="deny"``)."""
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def _build_audit_root_guard(
    root: Path,
    lexical_symlinks: frozenset[Path],
) -> HookCallback:
    """Build a deny-by-default guard for one immutable audit snapshot root."""

    async def _guard(
        input_data: Any,
        tool_use_id: Any,
        context: Any,
    ) -> HookJSONOutput:
        del tool_use_id, context
        if not isinstance(input_data, dict):
            return _read_only_deny("audit isolation denied malformed tool input")
        tool_name = input_data.get("tool_name")
        tool_input = input_data.get("tool_input")
        if not isinstance(tool_name, str) or not isinstance(tool_input, dict):
            return _read_only_deny("audit isolation denied malformed tool input")
        if tool_name == "StructuredOutput":
            return {}
        if tool_name == "Read":
            if not set(tool_input).issubset({"file_path", "offset", "limit"}):
                return _read_only_deny("audit isolation denied unsupported Read options")
            if any(field in tool_input and not _nonnegative_int(tool_input[field]) for field in ("offset", "limit")):
                return _read_only_deny("audit isolation denied malformed Read options")
            if _audit_regular_file(root, tool_input.get("file_path")) is None:
                return _read_only_deny("audit isolation denied Read outside its root")
            return {}
        if tool_name == "Grep":
            allowed_fields = {
                "pattern",
                "path",
                "output_mode",
                *_AUDIT_GREP_BOOL_FIELDS,
                *_AUDIT_GREP_INT_FIELDS,
            }
            if not set(tool_input).issubset(allowed_fields):
                return _read_only_deny("audit isolation denied unsupported Grep options")
            pattern = tool_input.get("pattern")
            if not isinstance(pattern, str) or not pattern or "\x00" in pattern:
                return _read_only_deny("audit isolation denied malformed Grep pattern")
            output_mode = tool_input.get("output_mode")
            if output_mode is not None and (
                not isinstance(output_mode, str) or output_mode not in _AUDIT_GREP_OUTPUT_MODES
            ):
                return _read_only_deny("audit isolation denied malformed Grep options")
            if any(
                field in tool_input and not isinstance(tool_input[field], bool) for field in _AUDIT_GREP_BOOL_FIELDS
            ) or any(
                field in tool_input and not _nonnegative_int(tool_input[field]) for field in _AUDIT_GREP_INT_FIELDS
            ):
                return _read_only_deny("audit isolation denied malformed Grep options")
            if _audit_regular_file(root, tool_input.get("path")) is None:
                return _read_only_deny("audit isolation denied Grep outside its root")
            return {}
        if tool_name == "Glob":
            if not set(tool_input).issubset({"pattern", "path"}):
                return _read_only_deny("audit isolation denied unsupported Glob options")
            pattern = tool_input.get("pattern")
            if not isinstance(pattern, str) or not _valid_relative_audit_path(pattern):
                return _read_only_deny("audit isolation denied malformed Glob pattern")
            raw_base = tool_input.get("path")
            if isinstance(raw_base, str) and _valid_relative_audit_path(raw_base):
                if _audit_path_crosses_lexical_symlink(root, raw_base):
                    return _read_only_deny("audit isolation denied Glob across a symlink")
            base = _audit_directory(root, raw_base)
            if base is None:
                return _read_only_deny("audit isolation denied Glob outside its root")
            if any(link == base or link.is_relative_to(base) for link in lexical_symlinks):
                return _read_only_deny("audit isolation denied Glob across a symlink")
            return {}
        return _read_only_deny("audit isolation denied an unsupported tool")

    return _guard


async def _finalization_guard(
    input_data: Any,
    tool_use_id: Any,
    context: Any,
) -> HookJSONOutput:
    """Deny investigation while preserving the SDK schema serialization tool."""
    if isinstance(input_data, dict) and input_data.get("tool_name") == "StructuredOutput":
        return {}
    return _read_only_deny("finalization allows only structured output serialization")


async def _read_only_guard(input_data: Any, tool_use_id: Any, context: Any) -> HookJSONOutput:
    """Allow inspection tools/allowlisted Bash; malformed and unknown tools deny.
    The .* matcher must cover every tool under bypassPermissions.
    """
    command = _bash_command(input_data)
    if command is not None:
        if _is_read_only_command(command):
            return {}
        return _read_only_deny(f"read-only guard: non-read-only Bash command blocked: {command!r}")
    tool_name = input_data.get("tool_name") if isinstance(input_data, dict) else None
    if tool_name in _READ_ONLY_ALLOWED_TOOLS:
        return {}
    return _read_only_deny(f"read-only guard: tool {tool_name!r} is blocked (non-mutating contract)")


def _is_dangerous_command(cmd: str) -> bool:
    """Match catastrophic root scans/wipes; scoped paths and unmatched commands pass."""
    return any(pattern.search(cmd) for pattern in _DANGEROUS_COMMAND_PATTERNS)


async def _dangerous_command_guard(input_data: Any, tool_use_id: Any, context: Any) -> HookJSONOutput:
    """Deny catastrophic Bash commands in every phase; allow other calls."""
    command = _bash_command(input_data)
    if command is None:
        return {}
    if _is_dangerous_command(command):
        return _read_only_deny(f"dangerous command blocked (always-on guard): {command!r}")
    return {}


def _is_background_bash(input_data: Any) -> bool:
    """Only a Bash call with truthy run_in_background is background work; malformed payloads are not Bash."""
    if _bash_command(input_data) is None:
        return False
    return bool(_tool_input(input_data).get("run_in_background"))


async def _background_bash_guard(input_data: Any, tool_use_id: Any, context: Any) -> HookJSONOutput:
    """Deny background Bash in every phase: CLI turn-end cleanup prevents its results reaching the host."""
    if not _is_background_bash(input_data):
        return {}
    return _read_only_deny(
        "background Bash blocked (always-on guard): daydream reads this turn's final text as the "
        "result and the CLI kills background tasks when the turn ends, so a backgrounded command "
        f"never reports. Run it in the foreground (timeout up to {_BASH_TIMEOUT_MS} ms) and wait "
        "for it to finish."
    )
