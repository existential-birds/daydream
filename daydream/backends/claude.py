# daydream/backends/claude.py
"""Claude Agent SDK backend for daydream."""

from __future__ import annotations

import json
import os
import re
import shlex
from collections.abc import AsyncGenerator
from pathlib import Path, PureWindowsPath
from typing import Any, cast

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, HookJSONOutput, HookMatcher
from claude_agent_sdk.types import (
    AgentDefinition,
    AssistantMessage,
    EffortLevel,
    HookCallback,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from daydream.backends import (
    AUDIT_ROOT_ISOLATION_V1,
    AgentEvent,
    ClaudeRequestConfig,
    ContinuationToken,
    CostEvent,
    MetricsEvent,
    ModelUsageTotals,
    RequestEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
    TurnEndEvent,
    resolve_fanout_concurrency,
)
from daydream.config import TEST_WALL_BUDGET_S

# Read-only Bash allowlist shared by every agent that runs under the read-only
# guard (setup-investigator, failure summarizer, exploration specialists,
# verification agent): permitted only if the command begins with one of these
# prefixes AND has no shell-chaining metacharacter that could smuggle in a
# mutation. Mirrored via _render_bash_allowlist() in the sibling prompts that
# advertise it (phases.py, deep/prompts.py).
READ_ONLY_BASH_ALLOWLIST: tuple[str, ...] = (
    "ls",
    "cat",
    "git status",
    "git log",
    "git show",
    "git blame",
    "git diff",
)

# Shell-control tokens checked against shlex output: shlex (non-posix)
# splits multi-char sequences like ``&&``/``$(`` into single chars, so we check
# per-char. ``<``/``>`` are included so redirection can never write to or
# truncate a file in the caller's tree.
#
# ``(``/``)`` are deliberately NOT here: bash rejects an unquoted paren glued to
# a word (``--format=%C(red)%h``, ``foo(1).txt``) as a syntax error, so nothing
# executes and no file is touched; a subshell only executes when ``(`` begins
# the command, which the command-leading check in _is_read_only_command()
# denies. A subshell reached after an operator (``a | (rm x)``,
# ``a && (rm x)``, ``a; (rm x)``, ``$(rm x)``) is already caught by that
# operator token in this set.
#
# The quote-safety claim holds only inside single quotes and for *literal*
# characters inside double quotes: ``|``/``;``/``&``/``<``/``>`` are inert in
# both, and shlex returns the wrapped chunk as one token. ``$`` and backtick are
# NOT inert inside double quotes -- bash still performs parameter
# expansion/command substitution there, so ``git log "$(rm x)"`` passes this
# token scan yet stays live in bash. That gap is pre-existing and is not sealed
# here, so the guard must not be advertised as "safe inside any quotes".
_SHELL_CONTROL_TOKENS: frozenset[str] = frozenset({"|", ";", "&", "`", "$", "<", ">"})

# Git options that write the command's output to a file. Scanned only after a
# matched ``git …`` allowlist family, so ``ls``/``cat`` never hit it.
_GIT_WRITE_OPTIONS: tuple[str, ...] = ("--output",)

# ``.*`` fires the guard for EVERY tool call so it can fail-closed (allow only
# the safe set); a deny-list of mutating tools was fail-open.
_READ_ONLY_HOOK_MATCHER = ".*"

# Catastrophic Bash commands denied in ALL phases (always-on guard, #177). These
# are the runaway-turn pathologies: full-filesystem-root scans that take hours,
# plus an unrecoverable wipe. Matched on the raw command via regex.
_DANGEROUS_COMMAND_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*find\s+/(\s|$)"),       # find / ...  (root-anchored scan)
    re.compile(r"^\s*grep\b.*\s/\s*$"),      # grep ... /  (root is the sole trailing path)
    # rm wiping filesystem root or its glob, with a recursive flag anywhere in
    # the option list (-rf, -fr, -R, --recursive) regardless of token order, so
    # ``rm --force --recursive /`` and ``rm -f -r /`` are caught too. Two
    # lookaheads: one for a recursive flag, one for ``/`` (or ``/*``) as a
    # standalone target. A subpath like ``/home`` is left alone — this is a
    # runaway/wipe backstop, not a security boundary (the read-only sandbox is).
    re.compile(r"^\s*rm\b(?=.*(?:^|\s)(?:-\w*[rR]\w*|--recursive)\b)(?=.*(?:^|\s)/\*?(?:\s|$)).*$"),
)

# Tools unconditionally permitted under the read-only profile (Bash handled
# separately via the command allowlist).
_READ_ONLY_ALLOWED_TOOLS: frozenset[str] = frozenset(
    {"Read", "Grep", "Glob", "StructuredOutput"}
)

# Ceiling for one foreground Bash call inside the CLI subprocess, in ms. The CLI
# clamps the tool's ``timeout`` to 600s by default, which is shorter than a
# real test suite under coverage -- and was exactly why a test-phase agent
# reached for ``run_in_background``. The host's own per-turn wall budget
# already bounds every turn (TEST_WALL_BUDGET_S is the largest one granted), so
# a shell call can never usefully outlive it; raising the CLI ceiling to match
# removes the incentive without loosening any host-side bound.
_BASH_TIMEOUT_MS = int(TEST_WALL_BUDGET_S * 1000)

# Environment for the CLI subprocess (merged over the inherited env by the SDK).
# daydream consumes a turn's final text as the phase result and stops reading
# the session when the turn ends; the CLI then tears down its background tasks.
# A backgrounded command therefore never reports, and the agent's "I'll wait
# for the notification" narration is what the host would parse as the verdict.
# Backgrounding is switched off at the source, and ``_background_bash_guard``
# below is the enforcement should a CLI build ignore the switch.
_CLI_ENV: dict[str, str] = {
    "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
    "BASH_DEFAULT_TIMEOUT_MS": str(_BASH_TIMEOUT_MS),
    "BASH_MAX_TIMEOUT_MS": str(_BASH_TIMEOUT_MS),
}

_AUDIT_TOOLS: tuple[str, ...] = ("Read", "Grep", "Glob", "StructuredOutput")
_AUDIT_GREP_OUTPUT_MODES = frozenset({"content", "files_with_matches", "count"})
_AUDIT_GREP_BOOL_FIELDS = frozenset({"-n", "-i", "multiline"})
_AUDIT_GREP_INT_FIELDS = frozenset({"head_limit", "offset"})


def _audit_cli_env(root: Path) -> dict[str, str]:
    """Return SDK environment overrides bound to one standalone audit repo."""
    git_dir = root / ".git"
    return {
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
        resolved = (candidate if candidate.is_absolute() else root / candidate).resolve(
            strict=True
        )
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


def _total_input_tokens(usage: dict[str, Any]) -> int | None:
    """Fold Anthropic's three input buckets into the true total input.

    Anthropic reports `input_tokens` as the *uncached remainder* only, with
    cache hits and writes split into `cache_read_input_tokens` and
    `cache_creation_input_tokens`. These two cache buckets are not mutually
    exclusive: a single response can both read one cache breakpoint and write
    another, so both may be non-zero at once. ATIF's `Metrics.prompt_tokens`
    is the total input, so sum `input_tokens`, `cache_read_input_tokens`, and
    `cache_creation_input_tokens` whenever present. Returns None when
    `input_tokens` is absent (preserves the no-token-count gate).
    """
    input_tokens = usage.get("input_tokens")
    if input_tokens is None:
        return None
    return (
        int(input_tokens)
        + int(usage.get("cache_read_input_tokens") or 0)
        + int(usage.get("cache_creation_input_tokens") or 0)
    )


class ClaudeAgentError(Exception):
    """Raised when the Claude agent run reports an error result.

    The SDK surfaces fatal run failures (invalid API key, execution errors,
    hitting max turns) as a ``ResultMessage`` with ``is_error=True`` rather
    than raising. Translating that flag into an exception here keeps an
    errored run from masquerading as a clean empty result downstream — e.g.
    a review exiting 0 with "no issues found" because the agent never ran.
    """


class MaxTurnsError(ClaudeAgentError):
    """Raised when the Claude agent run terminates by hitting the turn cap.

    A subtype of :class:`ClaudeAgentError` so existing ``except
    ClaudeAgentError`` handlers keep working, while callers that care about
    the max-turns case specifically can catch it and record/surface it
    distinctly. Carries the SDK ``subtype`` (``"error_max_turns"``) so the
    trajectory recorder can stamp it into the ATIF archive.
    """

    def __init__(self, message: str, *, subtype: str = "error_max_turns") -> None:
        super().__init__(message)
        self.subtype = subtype


def _denies_git_output_option(argv: list[str], start: int) -> bool:
    """True when an allowlisted ``git …`` argv writes output to a file.

    Scans ``argv[start:]`` (skipping the matched allowlist family words) for the
    ``--output`` option in both separated (``--output log.txt``) and equals
    (``--output=diff.patch``) forms. The scan stops at a standalone ``--`` path
    separator, so a literal path argument named ``--output`` after it stays
    allowed. Returns False (not denied) when no write option appears before that
    boundary.

    Args:
        argv: The posix-split argv tokens.
        start: Index of the first token after the matched allowlist family
            words (e.g. 2 for ``git status``).
    """
    for tok in argv[start:]:
        if tok == "--":
            return False
        if tok in _GIT_WRITE_OPTIONS or any(
            tok.startswith(opt + "=") for opt in _GIT_WRITE_OPTIONS
        ):
            return True
    return False


def _shlex_tokens(cmd: str, *, posix: bool, words: bool) -> list[str] | None:
    """Lex *cmd* via shlex; return the token list, or None on malformed quoting.

    Both the control-token scan and the argv reconstruction pass through this
    single helper so they share one explicit comment policy: ``commenters`` is
    cleared, making ``#`` always a literal character. Bash only treats a ``#``
    that begins a word as a comment; the default ``#`` commenter would instead
    strip everything after ANY ``#`` -- even mid-word -- hiding a trailing
    redirection/chaining token from the control-token scan. That is exactly the
    escape the deny set closes, so the comment semantics must be explicit and
    identical for both passes (a future edit cannot silently resurrect the
    blind spot by changing only one).

    ``words=True`` yields whole argv words; ``words=False`` yields per-char bare
    metacharacters so the control-token scan sees ``<``/``>``/``(``, etc. The
    ``words`` flag maps onto shlex ``whitespace_split``.
    """
    lexer = shlex.shlex(cmd, posix=posix)
    lexer.commenters = ""  # '#' is never a comment; keep every character.
    lexer.whitespace_split = words
    try:
        return list(lexer)
    except ValueError:
        return None


def _is_read_only_command(cmd: str) -> bool:
    """Return True only if *cmd* is a single allowlisted read-only command.

    Denies (returns False) on: an empty/blank command, any command containing a
    newline or carriage return, any command containing a shell-control
    metacharacter (``|``, ``;``, ``&``, backtick, ``$``, ``<``, ``>``) or a
    command-leading ``(``/``)`` subshell group, any command whose leading argv
    words do not match an allowlisted family word-for-word, and any allowlisted
    ``git …`` command that writes its output to a file via ``--output``.

    ``<``/``>``/``|``/``;``/``&``/``$``/backtick are shell operators wherever
    they appear unquoted, closing the redirection (``>``/``>>``/``<``) and
    command-substitution escapes that could otherwise create, truncate, or
    append files in the caller's working tree. ``(``/``)`` only execute as a
    subshell when they begin the command, so only that position is denied;
    parens mid-command or glued to a word (``--format=%C(red)%h``,
    ``cat foo(1).txt``) are bash syntax errors that never run and must stay
    allowed. Word-bounded argv matching uses posix ``shlex.split`` so a token
    merely *beginning* with an allowlisted word (``git logfoo``) is never an
    allowlist hit.

    Metacharacter detection uses ``shlex`` to avoid false positives from
    metacharacters that appear only inside quoted arguments (e.g.
    ``git log --grep='fix|bug'`` is safe and must be allowed).  Newlines and
    carriage returns are bash command separators but ``shlex`` treats them as
    whitespace and strips them, so they are rejected directly on the raw string.
    Malformed quoting makes ``shlex`` raise ``ValueError``; the shared lexing
    helper maps that to deny (fail-closed) and never propagates.
    """
    stripped = cmd.strip()
    if not stripped:
        return False
    if "\n" in cmd or "\r" in cmd:
        return False
    # Control-token pass: per-char bare tokens (``whitespace_split=False``), so
    # unquoted metacharacters (``&&`` -> ``&``) surface on their own. See
    # _SHELL_CONTROL_TOKENS.
    tokens = _shlex_tokens(stripped, posix=False, words=False)
    if tokens is None:
        return False  # Malformed quoting -- deny (fail-closed).
    for tok in tokens:
        if tok in _SHELL_CONTROL_TOKENS:
            return False
    if tokens[0] in ("(", ")"):
        # Command-leading ``(``/``)`` starts a subshell group that executes
        # (``( rm x )``, ``(rm x)``) -> deny. Anywhere else in a command bash
        # rejects unquoted parens (``--format=%C(red)%h``, ``foo(1).txt``,
        # ``ls -la ( x )``) as a syntax error, so nothing runs and the command
        # stays harmless; those previously-allowed forms must not be denied.
        return False
    # Argv pass: whole argv words (``whitespace_split=True``), matching the
    # allowlist families word-for-word (rejecting ``git logfoo``) and allowing
    # the ``git ... --output`` file-write scan.
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
    """Extract the Bash command from a PreToolUse payload.

    Returns ``None`` when the payload is not a Bash tool call (or is malformed),
    and ``""`` when it is Bash but the command is missing or not a string —
    matching the guards' fail-closed defaults.
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
            if any(
                field in tool_input and not _nonnegative_int(tool_input[field])
                for field in ("offset", "limit")
            ):
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
                not isinstance(output_mode, str)
                or output_mode not in _AUDIT_GREP_OUTPUT_MODES
            ):
                return _read_only_deny("audit isolation denied malformed Grep options")
            if any(
                field in tool_input and not isinstance(tool_input[field], bool)
                for field in _AUDIT_GREP_BOOL_FIELDS
            ) or any(
                field in tool_input and not _nonnegative_int(tool_input[field])
                for field in _AUDIT_GREP_INT_FIELDS
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


async def _read_only_guard(input_data: Any, tool_use_id: Any, context: Any) -> HookJSONOutput:
    """PreToolUse hook enforcing the read-only guard contract.

    Fires for ALL tools (matcher ``.*``). Explicitly allows only the safe set
    (Read, Grep, Glob, StructuredOutput, and allowlisted Bash commands) and
    denies everything else. Fails closed: malformed input → deny.
    Returns ``{}`` (allow) only for a permitted tool/command.
    """
    command = _bash_command(input_data)
    if command is not None:
        if _is_read_only_command(command):
            return {}
        return _read_only_deny(
            f"read-only guard: non-read-only Bash command blocked: {command!r}"
        )
    tool_name = input_data.get("tool_name") if isinstance(input_data, dict) else None
    if tool_name in _READ_ONLY_ALLOWED_TOOLS:
        return {}
    return _read_only_deny(
        f"read-only guard: tool {tool_name!r} is blocked (non-mutating contract)"
    )


def _is_dangerous_command(cmd: str) -> bool:
    """Return True if *cmd* is a catastrophic Bash command (always-on deny-list).

    Matches full-filesystem-root scans (``find /``, ``grep ... /``) and ``rm -rf /``
    — the runaway-turn pathologies. Conservative: a scoped path (``find core/...``)
    or a non-matching command (``ls``, ``rg foo src/``) returns False.
    """
    return any(pattern.search(cmd) for pattern in _DANGEROUS_COMMAND_PATTERNS)


async def _dangerous_command_guard(input_data: Any, tool_use_id: Any, context: Any) -> HookJSONOutput:
    """PreToolUse hook denying catastrophic Bash commands in ALL phases (#177).

    Registered unconditionally. Allows everything except a small deny-list of
    root-anchored scans and wipes (see ``_is_dangerous_command``). Codex has no
    equivalent PreToolUse seam (out of scope; its enforcement is ``--sandbox``).
    """
    command = _bash_command(input_data)
    if command is None:
        return {}
    if _is_dangerous_command(command):
        return _read_only_deny(f"dangerous command blocked (always-on guard): {command!r}")
    return {}


def _is_background_bash(input_data: Any) -> bool:
    """Return True if *input_data* is a Bash call asking to run in the background.

    Only a truthy ``run_in_background`` counts; a missing key, ``False``, or a
    non-Bash tool is a foreground call. Malformed payloads are not Bash calls.
    """
    if _bash_command(input_data) is None:
        return False
    return bool(_tool_input(input_data).get("run_in_background"))


async def _background_bash_guard(input_data: Any, tool_use_id: Any, context: Any) -> HookJSONOutput:
    """PreToolUse hook denying ``Bash(run_in_background=True)`` in ALL phases.

    Registered unconditionally. The deny reason tells the agent why and what to
    do instead, because the model has no other way to learn that a background
    command's result can never reach the host: the CLI kills background tasks
    when the turn ends, and the host reads the turn's final text as the result.
    Composes with ``CLAUDE_CODE_DISABLE_BACKGROUND_TASKS`` in ``_CLI_ENV``.
    """
    if not _is_background_bash(input_data):
        return {}
    return _read_only_deny(
        "background Bash blocked (always-on guard): daydream reads this turn's final text as the "
        "result and the CLI kills background tasks when the turn ends, so a backgrounded command "
        f"never reports. Run it in the foreground (timeout up to {_BASH_TIMEOUT_MS} ms) and wait "
        "for it to finish."
    )


_CLAUDE_EFFORT_LEVELS: frozenset[str] = frozenset(("low", "medium", "high", "xhigh", "max"))


def _claude_effort(value: str | None) -> EffortLevel | None:
    """Narrow a resolved reasoning effort to the SDK's ``EffortLevel``.

    Raises at construction rather than letting an unsupported level reach the
    CLI as ``--effort <junk>``, which fails mid-run with an opaque message.
    """
    if value is None:
        return None
    if value not in _CLAUDE_EFFORT_LEVELS:
        raise ValueError(
            f"Claude backend does not support reasoning effort {value!r}; "
            f"expected one of {sorted(_CLAUDE_EFFORT_LEVELS)}"
        )
    return cast(EffortLevel, value)


def _model_usage_totals(raw: dict[str, Any] | None) -> dict[str, ModelUsageTotals] | None:
    """Select public billing fields instead of exporting the SDK's raw payload."""
    if not raw:
        return None
    return {
        model: ModelUsageTotals(
            model_name=usage.get("canonicalModel") or model,
            provider_name=usage.get("provider"),
            input_tokens=_total_input_tokens({
                "input_tokens": usage.get("inputTokens"),
                "cache_read_input_tokens": usage.get("cacheReadInputTokens"),
                "cache_creation_input_tokens": usage.get("cacheCreationInputTokens"),
            }),
            output_tokens=usage.get("outputTokens"),
            cached_tokens=usage.get("cacheReadInputTokens"),
            cache_creation_tokens=usage.get("cacheCreationInputTokens"),
            cost_usd=usage.get("costUSD"),
        )
        for model, usage in raw.items()
    }


class ClaudeBackend:
    """Backend that wraps the Claude Agent SDK.

    Translates Claude SDK message types into the unified AgentEvent stream.
    """

    concise_fix_prompts = False

    def __init__(
        self,
        model: str,
        *,
        reasoning_effort: str | None = None,
        audit_root: Path | None = None,
        audit_outward_symlinks: frozenset[Path] = frozenset(),
    ):
        self.model = model
        self.reasoning_effort = _claude_effort(reasoning_effort)
        self.audit_root = audit_root.resolve(strict=True) if audit_root is not None else None
        self.audit_root_isolation = (
            AUDIT_ROOT_ISOLATION_V1 if self.audit_root is not None else None
        )
        lexical_links = frozenset(
            Path(os.path.abspath(path)) for path in audit_outward_symlinks
        )
        if self.audit_root is None and lexical_links:
            raise ValueError("audit_outward_symlinks requires audit_root")
        if self.audit_root is not None and any(
            not path.is_relative_to(self.audit_root) for path in lexical_links
        ):
            raise ValueError("audit_outward_symlinks must be inside audit_root")
        self.audit_outward_symlinks = lexical_links
        audit_symlinks = (
            lexical_links | _audit_symlink_inventory(self.audit_root)
            if self.audit_root is not None
            else frozenset()
        )
        self._audit_root_guard = (
            _build_audit_root_guard(self.audit_root, audit_symlinks)
            if self.audit_root is not None
            else None
        )
        self.fanout_concurrency = resolve_fanout_concurrency("DAYDREAM_FANOUT_CONCURRENCY", 8)
        self._active_clients: set[ClaudeSDKClient] = set()

    async def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: dict[str, Any] | None = None,
        continuation: ContinuationToken | None = None,
        agents: dict[str, AgentDefinition] | None = None,
        max_turns: int | None = None,
        read_only: bool = False,
        persist_session: bool = True,
    ) -> AsyncGenerator[AgentEvent, None]:
        """Execute a prompt and yield unified events.

        Args:
            continuation: When it carries ``backend == "claude"``, its
                ``session_id`` is passed to the SDK as ``options.resume`` so the
                CLI replays the prior turn's conversation. A token minted by any
                other backend is ignored (cold start), never an error. A retried
                attempt resumes from the same original token, since ``run_agent``
                passes the input token on every attempt.
            agents: Optional mapping of specialist name -> AgentDefinition for
                subagent support. Keys are the specialist names the lead agent
                dispatches by; they MUST be preserved verbatim.
            read_only: When True, register a ``PreToolUse`` guard hook that
                denies file-mutating tools (Write/Edit/...) and any Bash command
                not on ``READ_ONLY_BASH_ALLOWLIST``. The hook is the enforcement
                — under ``bypassPermissions`` ``allowed_tools`` does not restrict
                the toolset — so the tool list is left unchanged.
            persist_session: When True, the final ``ResultEvent`` mints a
                ``ContinuationToken`` carrying ``ResultMessage.session_id`` so a
                later call can resume this conversation. False suppresses the
                token (the turn is one-shot).

        Raises:
            ClaudeAgentError: If the agent run ends with an error result
                (``ResultMessage.is_error``), e.g. an invalid API key.
        """
        audit_root = self.audit_root
        audit_guard = self._audit_root_guard
        if audit_root is not None:
            try:
                resolved_cwd = cwd.resolve(strict=True)
            except (OSError, RuntimeError, ValueError) as exc:
                raise ClaudeAgentError(
                    "audit isolation requires an existing canonical cwd"
                ) from exc
            if resolved_cwd != audit_root:
                raise ClaudeAgentError("audit isolation requires the bound audit root cwd")
            if not read_only:
                raise ClaudeAgentError("audit isolation requires read_only=True")
            if continuation is not None:
                raise ClaudeAgentError("audit isolation does not allow continuation")
            if agents:
                raise ClaudeAgentError("audit isolation does not allow agents")
            persist_session = False

        output_format = (
            {"type": "json_schema", "schema": output_schema}
            if output_schema
            else None
        )

        # PreToolUse hooks — NOT allowed_tools — are the enforcement, since
        # bypassPermissions leaves the tool list unrestricted. The dangerous-command
        # and background-Bash guards are always-on (all phases); the read-only guard
        # composes on top when read_only=True.
        #
        # In ordinary mode (#887), the skill tool is intentionally left unguarded:
        # daydream no longer invokes a skill itself. Strict audit mode takes the
        # separate branch below and denies Skill along with every unhandled tool.
        if audit_guard is not None:
            assert audit_root is not None
            options = ClaudeAgentOptions(
                cwd=str(audit_root),
                permission_mode="bypassPermissions",
                tools=list(_AUDIT_TOOLS),
                allowed_tools=list(_AUDIT_TOOLS),
                mcp_servers={},
                strict_mcp_config=True,
                setting_sources=[],
                skills=[],
                plugins=[],
                agents=None,
                model=self.model,
                output_format=output_format,
                max_buffer_size=10 * 1024 * 1024,
                max_turns=max_turns,
                extra_args={"no-session-persistence": None},
                effort=self.reasoning_effort,
                env=_audit_cli_env(audit_root),
                hooks={
                    "PreToolUse": [
                        HookMatcher(matcher=_READ_ONLY_HOOK_MATCHER, hooks=[audit_guard])
                    ]
                },
            )
        else:
            pre_tool_use_hooks: list[HookCallback] = [
                _dangerous_command_guard,
                _background_bash_guard,
            ]
            if read_only:
                pre_tool_use_hooks.append(_read_only_guard)
            options = ClaudeAgentOptions(
                cwd=str(cwd),
                permission_mode="bypassPermissions",
                allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
                setting_sources=["user"],
                model=self.model,
                output_format=output_format,
                max_buffer_size=10 * 1024 * 1024,  # 10MB — handles large git diffs
                max_turns=max_turns,
                extra_args={"no-session-persistence": None} if not persist_session else {},
                # None leaves the CLI's ambient default; the SDK omits --effort.
                effort=self.reasoning_effort,
                env=dict(_CLI_ENV),
                hooks={
                    "PreToolUse": [
                        HookMatcher(
                            matcher=_READ_ONLY_HOOK_MATCHER,
                            hooks=pre_tool_use_hooks,
                        )
                    ]
                },
            )

        # Resume the prior conversation when the caller threaded a claude-minted
        # token through. Options stay otherwise byte-stable so prefix caching
        # survives.
        if continuation is not None and continuation.backend == "claude":
            resume_id = continuation.data.get("session_id")
            if resume_id:
                options.resume = resume_id

        if agents:
            options.agents = agents

        # P18 Task 1: closed typed effective-config admission, populated from
        # the exact options built above (never from config files/env).
        # - read_only/persist_session are actual passed controls (execute args).
        # - continuation_mode is "resume" only when a claude-minted token was
        #   applied; "fresh" otherwise. Fork is not a Claude surface.
        # - A nonempty agents mapping makes the request aggregate
        #   multi-model-capable without inspecting any AgentDefinition.
        resume_applied = (
            continuation is not None
            and continuation.backend == "claude"
            and bool(continuation.data.get("session_id"))
        )
        agents_nonempty = bool(agents)
        allowed_tools = options.allowed_tools
        audit_tools = options.allowed_tools if audit_guard is not None else None
        audit_tools_count = len(audit_tools) if audit_tools else None
        audit_tools_present = bool(audit_tools) if audit_tools is not None else None

        structured_result: Any = None
        # SDK session id from the terminal ResultMessage; minted into the
        # ContinuationToken so a later call can --resume this conversation.
        session_id: str | None = None
        # Latest AssistantMessage.model, stamped on the trailing CostEvent so the
        # recorder can upgrade the generic ``"claude"`` label to the real SDK id.
        last_assistant_model: str | None = None
        last_stop_reason: str | None = None
        # StructuredOutput ToolUseBlocks are skipped (result comes via
        # ResultMessage.structured_output); track their IDs so the matching
        # ToolResultBlocks aren't logged as unmatched_tool_results.
        skipped_tool_ids: set[str] = set()
        terminal_result: ResultEvent | None = None

        yield RequestEvent(
            prompt=prompt,
            model_name=self.model,
            session_id=options.resume,
            reasoning_effort=self.reasoning_effort,
            output_schema=output_schema,
            config=ClaudeRequestConfig(
                max_turns=max_turns,
                read_only=read_only,
                persist_session=persist_session,
                continuation_mode="resume" if resume_applied else "fresh",
                model_mode="multi_or_dynamic" if agents_nonempty else "single",
                permission_mode="bypassPermissions",
                allowed_tools_count=len(allowed_tools) if allowed_tools else None,
                allowed_tools_present=bool(allowed_tools),
                audit_tools_count=audit_tools_count,
                audit_tools_present=audit_tools_present,
                setting_sources_present=bool(options.setting_sources),
                native_output_format=output_format is not None,
                buffer_limit_bytes=10 * 1024 * 1024,
                hooks_enabled=True,
            ),
            model_source="configured",
            session_source="host_generated" if resume_applied else None,
        )

        async with ClaudeSDKClient(options=options) as client:
            self._active_clients.add(client)
            response = client.receive_response()
            response_terminated = False
            try:
                await client.query(prompt)
                async for msg in response:
                    if isinstance(msg, AssistantMessage):
                        msg_model = getattr(msg, "model", None)
                        if isinstance(msg_model, str) and msg_model:
                            last_assistant_model = msg_model
                        last_stop_reason = getattr(msg, "stop_reason", None) or last_stop_reason
                        session_id = getattr(msg, "session_id", None) or session_id
                        for block in msg.content:
                            if isinstance(block, TextBlock) and block.text:
                                yield TextEvent(text=block.text)
                            elif isinstance(block, ThinkingBlock) and block.thinking:
                                yield ThinkingEvent(text=block.thinking)
                            elif isinstance(block, ToolUseBlock):
                                if block.name == "StructuredOutput":
                                    # Drift guard: StructuredOutput must stay in the read-only
                                    # allow-set, else this passthrough becomes a mutation hole.
                                    assert "StructuredOutput" in _READ_ONLY_ALLOWED_TOOLS, (
                                        "StructuredOutput must remain in _READ_ONLY_ALLOWED_TOOLS "
                                        "to preserve the read_only non-mutation contract"
                                    )
                                    skipped_tool_ids.add(block.id)
                                    continue
                                yield ToolStartEvent(
                                    id=block.id,
                                    name=block.name,
                                    input=block.input or {},
                                )
                        # EVNT-06: MetricsEvent per AssistantMessage keyed by message_id.
                        # Rename SDK input/output_tokens → prompt/completion_tokens; cost_usd
                        # is None per-message (only on ResultMessage). Skip when either token
                        # count is missing (EVNT-02 types both as required int).
                        msg_usage = getattr(msg, "usage", None)
                        if (
                            msg_usage is not None
                            and msg_usage.get("input_tokens") is not None
                            and msg_usage.get("output_tokens") is not None
                        ):
                            total_input = _total_input_tokens(msg_usage)
                            assert total_input is not None  # guarded by input_tokens check above
                            yield MetricsEvent(
                                message_id=getattr(msg, "message_id", "") or "",
                                prompt_tokens=total_input,
                                completion_tokens=msg_usage["output_tokens"],
                                cached_tokens=msg_usage.get("cache_read_input_tokens"),
                                cost_usd=None,
                                model_name=last_assistant_model,
                                cache_creation_tokens=msg_usage.get("cache_creation_input_tokens"),
                                measurement_source="message_end",
                            )
                        yield TurnEndEvent(message_id=getattr(msg, "message_id", "") or "")

                    elif isinstance(msg, UserMessage):
                        for user_block in msg.content:
                            if isinstance(user_block, ToolResultBlock):
                                if user_block.tool_use_id in skipped_tool_ids:
                                    skipped_tool_ids.discard(user_block.tool_use_id)
                                    continue
                                content = user_block.content
                                content_str = content if isinstance(content, str) else (
                                    json.dumps(content, ensure_ascii=False) if content is not None else ""
                                )
                                yield ToolResultEvent(
                                    id=user_block.tool_use_id,
                                    output=content_str,
                                    is_error=user_block.is_error or False,
                                )

                    elif isinstance(msg, ResultMessage):
                        response_terminated = True
                        session_id = getattr(msg, "session_id", None) or session_id
                        model_usage = _model_usage_totals(getattr(msg, "model_usage", None))
                        if last_assistant_model is None and model_usage and len(model_usage) == 1:
                            last_assistant_model = next(iter(model_usage.values())).model_name
                        providers = {entry.provider_name for entry in (model_usage or {}).values()
                                     if entry.provider_name is not None}
                        provider = next(iter(providers)) if len(providers) == 1 else None
                        if msg.structured_output is not None:
                            structured_result = msg.structured_output
                        # EVNT-04/05: emit CostEvent when cost OR usage is available.
                        # Per-call semantics trusted for SDK 0.1.52 (D-14). Anthropic's raw
                        # `input_tokens` is the *uncached remainder* only; we fold in the
                        # cache-read and cache-creation buckets so the emitted value is the
                        # true total input, matching ATIF Metrics.prompt_tokens. cached_tokens
                        # stays the cache-read hit subset of that total.
                        result_usage = getattr(msg, "usage", None)
                        if msg.total_cost_usd is not None or result_usage is not None or model_usage:
                            usage = result_usage or {}
                            yield CostEvent(
                                cost_usd=msg.total_cost_usd,
                                input_tokens=_total_input_tokens(usage),
                                output_tokens=usage.get("output_tokens"),
                                cached_tokens=usage.get("cache_read_input_tokens"),
                                model_name=last_assistant_model,
                                provider_name=provider,
                                cache_creation_tokens=usage.get("cache_creation_input_tokens"),
                                model_usage=model_usage,
                                measurement_source="terminal",
                                cost_source="reported" if msg.total_cost_usd is not None else None,
                            )
                        terminal_result = ResultEvent(
                            structured_output=structured_result,
                            continuation=(
                                ContinuationToken(backend="claude", data={"session_id": session_id})
                                if persist_session and session_id and not msg.is_error else None
                            ),
                            model_name=last_assistant_model,
                            provider_name=provider,
                            session_id=session_id,
                            finish_reason=getattr(msg, "stop_reason", None) or last_stop_reason or msg.subtype,
                            duration_ms=getattr(msg, "duration_ms", None),
                            duration_api_ms=getattr(msg, "duration_api_ms", None),
                        )
                        if msg.is_error:
                            yield terminal_result
                            detail = msg.result or msg.subtype or "unknown error"
                            if msg.subtype == "error_max_turns":
                                raise MaxTurnsError(
                                    f"Claude agent run failed: {detail}", subtype="error_max_turns",
                                )
                            raise ClaudeAgentError(f"Claude agent run failed: {detail}")

                yield terminal_result or ResultEvent(
                    structured_output=structured_result,
                    model_name=last_assistant_model,
                    session_id=session_id,
                    finish_reason=last_stop_reason,
                    continuation=(
                        ContinuationToken(
                            backend="claude",
                            data={"session_id": session_id},
                        )
                        if persist_session and session_id
                        else None
                    ),
                )
            except GeneratorExit:
                if not response_terminated:
                    await client.interrupt()
                    async for _ in response:
                        pass
                raise
            finally:
                self._active_clients.discard(client)

    async def cancel(self) -> None:
        """Interrupt every active SDK client.

        Sends an interrupt to each in-flight agent client in turn; an error
        raised by any client's interrupt propagates to the caller.
        """
        for client in list(self._active_clients):
            await client.interrupt()
