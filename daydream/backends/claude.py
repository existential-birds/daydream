# daydream/backends/claude.py
"""Claude Agent SDK backend for daydream."""

from __future__ import annotations

import json
import os
import platform
import re
import shlex
import shutil
from collections.abc import AsyncGenerator, AsyncIterable
from contextlib import suppress
from dataclasses import asdict
from pathlib import Path, PureWindowsPath
from subprocess import PIPE
from typing import Any, cast

import anyio
from anyio.streams.text import TextReceiveStream, TextSendStream
from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, HookJSONOutput, HookMatcher
from claude_agent_sdk._errors import CLIConnectionError, CLINotFoundError
from claude_agent_sdk._internal._task_compat import spawn_detached
from claude_agent_sdk._internal.query import Query
from claude_agent_sdk._internal.transport import subprocess_cli as _sdk_subprocess
from claude_agent_sdk._version import __version__ as _sdk_version
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
    _hooks_to_internal_format,
)

from daydream.backends import (
    AUDIT_ROOT_ISOLATION,
    AgentEvent,
    BackendExecutionInput,
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

# Non-posix shlex splits controls into characters, including redirection.
# Leading parentheses are checked separately; mid-word parentheses produce
# Bash syntax errors. Operators already reject chained subshells.
#
# Quoted | ; & < > are inert, but $ and backticks inside double quotes still
# expand in Bash. For example, git log "$(rm x)" passes this token scan.
# Do not claim this guard makes arbitrary double-quoted arguments safe.
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
    # Match recursive rm flags in any order plus a standalone / or /* target.
    # Subpaths such as /home are outside this catastrophic-wipe backstop.
    re.compile(r"^\s*rm\b(?=.*(?:^|\s)(?:-\w*[rR]\w*|--recursive)\b)(?=.*(?:^|\s)/\*?(?:\s|$)).*$"),
)

# Tools unconditionally permitted under the read-only profile (Bash handled
# separately via the command allowlist).
_READ_ONLY_ALLOWED_TOOLS: frozenset[str] = frozenset(
    {"Read", "Grep", "Glob", "StructuredOutput"}
)

# Match the largest host wall budget so foreground test suites can exceed
# Claude's default 600s shell timeout without needing background execution.
_BASH_TIMEOUT_MS = int(TEST_WALL_BUDGET_S * 1000)

# Disable background tasks: the host consumes final turn text, then the CLI
# kills background work before it can report. _background_bash_guard enforces
# this when a CLI ignores the switch. SDK merges these over inherited env.
_CLI_ENV: dict[str, str] = {
    "CLAUDE_CODE_DISABLE_BACKGROUND_TASKS": "1",
    "BASH_DEFAULT_TIMEOUT_MS": str(_BASH_TIMEOUT_MS),
    "BASH_MAX_TIMEOUT_MS": str(_BASH_TIMEOUT_MS),
}

_AUDIT_TOOLS: tuple[str, ...] = ("Read", "Grep", "Glob", "StructuredOutput")
_AUDIT_GREP_OUTPUT_MODES = frozenset({"content", "files_with_matches", "count"})
_AUDIT_GREP_BOOL_FIELDS = frozenset({"-n", "-i", "multiline"})
_AUDIT_GREP_INT_FIELDS = frozenset({"head_limit", "offset"})


def _audit_cli_env(
    root: Path, *, base_environment: dict[str, str] | None = None
) -> dict[str, str]:
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


class _RunLocalSubprocessCLITransport(_sdk_subprocess.SubprocessCLITransport):
    """SDK subprocess transport that never merges the host process environment."""

    def __init__(
        self,
        options: ClaudeAgentOptions,
        *,
        environment: dict[str, str],
    ) -> None:
        self._run_environment = dict(environment)

        async def _empty_prompt() -> AsyncGenerator[dict[str, Any], None]:
            messages: tuple[dict[str, Any], ...] = ()
            for message in messages:
                yield message

        prompt: AsyncIterable[dict[str, Any]] = _empty_prompt()
        super().__init__(prompt=prompt, options=options)

    def _find_cli(self) -> str:
        bundled_cli = self._find_bundled_cli()
        if bundled_cli:
            return bundled_cli

        search_path = self._run_environment.get("PATH", "")
        which_hit = shutil.which("claude", path=search_path)
        if which_hit and (
            platform.system() != "Windows" or self._is_windows_native_exe(which_hit)
        ):
            return which_hit
        if platform.system() == "Windows":
            exe = shutil.which("claude.exe", path=search_path)
            if exe and self._is_windows_native_exe(exe):
                return exe

        home_raw = self._run_environment.get("HOME") or self._run_environment.get(
            "USERPROFILE"
        )
        if home_raw:
            home = Path(home_raw)
            locations = (
                [home / ".local/bin/claude.exe"]
                if platform.system() == "Windows"
                else [
                    home / ".npm-global/bin/claude",
                    home / ".local/bin/claude",
                    home / "node_modules/.bin/claude",
                    home / ".yarn/bin/claude",
                    home / ".claude/local/claude",
                ]
            )
            for path in locations:
                if path.exists() and path.is_file():
                    return str(path)
        if platform.system() != "Windows":
            for path in (Path("/usr/local/bin/claude"),):
                if path.exists() and path.is_file():
                    return str(path)
        if which_hit is not None:
            return which_hit
        raise CLINotFoundError(
            "Claude Code not found in the run-owned execution environment"
        )

    def _native_environment(self) -> dict[str, str]:
        process_env = {
            key: value
            for key, value in self._run_environment.items()
            if key != "CLAUDECODE"
        }
        process_env.setdefault("CLAUDE_CODE_ENTRYPOINT", "sdk-py")
        process_env["CLAUDE_AGENT_SDK_VERSION"] = _sdk_version

        try:
            from opentelemetry import propagate

            carrier: dict[str, str] = {}
            propagate.inject(carrier)
            if "traceparent" in carrier:
                for key in ("TRACEPARENT", "TRACESTATE"):
                    if key not in self._run_environment:
                        process_env.pop(key, None)
                for key, value in carrier.items():
                    upper = key.upper()
                    if upper not in self._run_environment:
                        process_env[upper] = value
        except Exception:  # noqa: BLE001 - tracing remains best effort
            _sdk_subprocess.logger.debug(
                "OTEL trace context injection failed", exc_info=True
            )

        if self._options.enable_file_checkpointing:
            process_env["CLAUDE_CODE_ENABLE_SDK_FILE_CHECKPOINTING"] = "true"
        if self._cwd:
            process_env["PWD"] = self._cwd
        return process_env

    async def connect(self) -> None:
        """Start the SDK CLI with only the run-owned complete environment."""
        if self._process:
            return
        if self._cli_path is None:
            self._cli_path = await anyio.to_thread.run_sync(self._find_cli)
        self._reject_windows_batch_cli(self._cli_path)
        if not self._run_environment.get("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK"):
            await self._check_claude_version()

        cmd = self._build_command()
        try:
            stderr_dest = PIPE if self._options.stderr is not None else None
            self._process = await anyio.open_process(
                cmd,
                stdin=PIPE,
                stdout=PIPE,
                stderr=stderr_dest,
                cwd=self._cwd,
                env=self._native_environment(),
                user=self._options.user,
            )
            _sdk_subprocess._ACTIVE_CHILDREN.add(self._process)
            if self._process.stdout:
                self._stdout_stream = TextReceiveStream(self._process.stdout)
            if stderr_dest is PIPE and self._process.stderr:
                self._stderr_stream = TextReceiveStream(self._process.stderr)
                self._stderr_task = spawn_detached(self._handle_stderr())
            if self._process.stdin:
                self._stdin_stream = TextSendStream(self._process.stdin)
            self._ready = True
        except FileNotFoundError as exc:
            if self._cwd and not Path(self._cwd).exists():
                error = CLIConnectionError(
                    f"Working directory does not exist: {self._cwd}"
                )
                self._exit_error = error
                raise error from exc
            error = CLINotFoundError(f"Claude Code not found at: {self._cli_path}")
            self._exit_error = error
            raise error from exc
        except Exception as exc:
            error = CLIConnectionError(f"Failed to start Claude Code: {exc}")
            self._exit_error = error
            raise error from exc

    async def _check_claude_version(self) -> None:
        """Run the SDK version probe with the same run-owned environment."""
        if self._cli_path is None:
            raise CLINotFoundError("CLI path not resolved. Call connect() first.")
        version_process = None
        try:
            with anyio.fail_after(2):
                version_process = await anyio.open_process(
                    [self._cli_path, "-v"],
                    stdout=PIPE,
                    stderr=PIPE,
                    env=self._native_environment(),
                )
                if version_process.stdout:
                    stdout_bytes = await version_process.stdout.receive()
                    version_output = stdout_bytes.decode().strip()
                    match = re.match(r"([0-9]+\.[0-9]+\.[0-9]+)", version_output)
                    if match:
                        version_parts = [int(value) for value in match.group(1).split(".")]
                        minimum_parts = [
                            int(value)
                            for value in _sdk_subprocess.MINIMUM_CLAUDE_CODE_VERSION.split(
                                "."
                            )
                        ]
                        if version_parts < minimum_parts:
                            _sdk_subprocess.logger.warning(
                                "Claude Code version %s at %s is unsupported; minimum is %s",
                                match.group(1),
                                self._cli_path,
                                _sdk_subprocess.MINIMUM_CLAUDE_CODE_VERSION,
                            )
        except Exception:
            pass
        finally:
            if version_process:
                with suppress(Exception):
                    version_process.terminate()
                with suppress(Exception):
                    await version_process.wait()


class _RunLocalClaudeSDKClient(ClaudeSDKClient):
    """Pinned-SDK startup using the run-owned initialization timeout.

    Override Query construction because the SDK otherwise reads the ambient
    CLAUDE_CODE_STREAM_CLOSE_TIMEOUT even with a custom transport. Retain the
    inherited error unwind, message API, disconnect, and context-manager lifecycle.
    """

    def __init__(
        self,
        *,
        options: ClaudeAgentOptions,
        transport: _RunLocalSubprocessCLITransport,
        initialize_timeout_s: float,
    ) -> None:
        super().__init__(options=options, transport=transport)
        self._run_initialize_timeout_s = initialize_timeout_s

    async def _connect_inner(
        self,
        prompt: str | AsyncIterable[dict[str, Any]] | None,
        actual_prompt: AsyncIterable[dict[str, Any]],
    ) -> None:
        assert self._custom_transport is not None
        self._transport = self._custom_transport
        await self._transport.connect()

        agents_dict = (
            {
                name: {
                    key: value
                    for key, value in asdict(agent_definition).items()
                    if value is not None
                }
                for name, agent_definition in self.options.agents.items()
            }
            if self.options.agents
            else None
        )
        exclude_dynamic_sections: bool | None = None
        system_prompt = self.options.system_prompt
        if isinstance(system_prompt, dict) and system_prompt.get("type") == "preset":
            candidate = system_prompt.get("exclude_dynamic_sections")
            if isinstance(candidate, bool):
                exclude_dynamic_sections = candidate

        self._query = Query(
            transport=self._transport,
            is_streaming_mode=True,
            can_use_tool=self.options.can_use_tool,
            hooks=(
                _hooks_to_internal_format(self.options.hooks)
                if self.options.hooks
                else None
            ),
            sdk_mcp_servers={},
            initialize_timeout=self._run_initialize_timeout_s,
            agents=agents_dict,
            exclude_dynamic_sections=exclude_dynamic_sections,
            skills=self.options.skills,
            forward_subagent_text=self.options.forward_subagent_text,
        )
        await self._query.start()
        await self._query.initialize()
        if isinstance(prompt, str):
            message = {
                "type": "user",
                "message": {"role": "user", "content": prompt},
                "parent_tool_use_id": None,
                "session_id": "default",
            }
            await self._transport.write(json.dumps(message) + "\n")
        elif prompt is not None and isinstance(prompt, AsyncIterable):
            self._query.spawn_task(self._query.stream_input(actual_prompt))


def _run_local_initialize_timeout_s(environment: dict[str, str]) -> float:
    """Parse the SDK initialization window from one run-owned environment."""
    raw = environment.get("CLAUDE_CODE_STREAM_CLOSE_TIMEOUT", "60000")
    try:
        timeout_ms = int(raw)
    except ValueError:
        timeout_ms = 60000
    return max(timeout_ms / 1000.0, 60.0)


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
    """Sum uncached input, cache reads, and cache writes for ATIF prompt_tokens.

    Both cache buckets may be nonzero in one response. An absent input_tokens
    keeps the total absent, preserving the no-token-count gate.
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
    """Translate an SDK is_error result into a failure instead of a clean empty review."""


class MaxTurnsError(ClaudeAgentError):
    """A ClaudeAgentError carrying the SDK turn-cap subtype for trajectory recording."""

    def __init__(self, message: str, *, subtype: str = "error_max_turns") -> None:
        super().__init__(message)
        self.subtype = subtype


def _denies_git_output_option(argv: list[str], start: int) -> bool:
    """Reject --output and --output= after the matched Git family in argv[start:].

    Stop at the standalone -- path separator, where --output is a literal path.
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
    """Lex command tokens, returning None for malformed quoting.

    Disable shlex comments in both passes: its default treats even a mid-word #
    as a comment, hiding later redirection or chaining. words=True preserves
    argv words; False exposes individual bare control characters.
    """
    lexer = shlex.shlex(cmd, posix=posix)
    lexer.commenters = ""  # '#' is never a comment; keep every character.
    lexer.whitespace_split = words
    try:
        return list(lexer)
    except ValueError:
        return None


def _is_read_only_command(cmd: str) -> bool:
    """Admit one allowlisted command with no shell controls or Git output writes.

    Reject raw newlines/carriage returns before shlex discards them as whitespace.
    Use a control-token pass followed by whole-word argv matching; malformed
    quoting fails closed and git logfoo cannot match git log. Quoted literal
    metacharacters remain allowed, subject to the double-quote limitation
    documented beside _SHELL_CONTROL_TOKENS. Leading parentheses are denied;
    mid-word parentheses are Bash syntax errors and remain allowed.
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
        # Deny leading subshell groups; mid-command unquoted parentheses are
        # Bash syntax errors and retain their previously allowed treatment.
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
    """Extract Bash command text; None means non-Bash/malformed payload.

    A Bash call with a missing or non-string command returns "" to fail closed.
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


async def _finalization_guard(
    input_data: Any, tool_use_id: Any, context: Any,
) -> HookJSONOutput:
    """Deny investigation while preserving the SDK schema serialization tool."""
    if isinstance(input_data, dict) and input_data.get("tool_name") == "StructuredOutput":
        return {}
    return _read_only_deny("finalization allows only structured output serialization")


async def _read_only_guard(input_data: Any, tool_use_id: Any, context: Any) -> HookJSONOutput:
    """Allow only inspection tools and allowlisted Bash; malformed/unknown tools deny.

    The .* matcher must run this guard for every tool under bypassPermissions.
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
    """Return True if *input_data* is a Bash call asking to run in the background.

    Only a truthy ``run_in_background`` counts; a missing key, ``False``, or a
    non-Bash tool is a foreground call. Malformed payloads are not Bash calls.
    """
    if _bash_command(input_data) is None:
        return False
    return bool(_tool_input(input_data).get("run_in_background"))


async def _background_bash_guard(input_data: Any, tool_use_id: Any, context: Any) -> HookJSONOutput:
    """Deny background Bash in every phase, reinforcing the CLI environment switch.

    The denial explains that the CLI kills background tasks at turn end and the
    host consumes final text, so background results never reach it.
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
    """Translate Claude SDK messages into the unified AgentEvent stream."""

    supports_finalization = True

    def __init__(
        self,
        model: str,
        *,
        reasoning_effort: str | None = None,
        audit_root: Path | None = None,
        audit_outward_symlinks: frozenset[Path] = frozenset(),
        execution_input: BackendExecutionInput | None = None,
    ):
        self.model = model
        self.reasoning_effort = _claude_effort(reasoning_effort)
        self._execution_input = execution_input
        self.audit_root = audit_root.resolve(strict=True) if audit_root is not None else None
        self.audit_root_isolation = (
            AUDIT_ROOT_ISOLATION if self.audit_root is not None else None
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
        if execution_input is not None:
            self.fanout_concurrency = execution_input.fanout_concurrency
            self.retry_policy = execution_input.retry_policy
        else:
            self.fanout_concurrency = resolve_fanout_concurrency(
                "DAYDREAM_FANOUT_CONCURRENCY", 8
            )
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
        finalization: bool = False,
    ) -> AsyncGenerator[AgentEvent, None]:
        """Yield normalized SDK events; error results raise ClaudeAgentError.

        Resume only Claude tokens, using the original input token on each retry;
        foreign tokens cold-start. Preserve specialist agent names verbatim.

        read_only is enforced by PreToolUse hooks: allowed_tools does not restrict
        tools under bypassPermissions. finalization removes investigative tools and
        uses low effort for this invocation, retaining guards and StructuredOutput.
        persist_session=False suppresses the terminal continuation token.
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

        effort = _claude_effort("low") if finalization else self.reasoning_effort
        output_format = (
            {"type": "json_schema", "schema": output_schema}
            if output_schema
            else None
        )

        # Permission preapproval does not restrict tools under bypassPermissions;
        # PreToolUse guards enforce mutation, background-task, and audit boundaries.
        options = ClaudeAgentOptions(
            cwd=str(audit_root if audit_guard is not None else cwd),
            permission_mode="bypassPermissions",
            model=self.model,
            output_format=output_format,
            max_buffer_size=10 * 1024 * 1024,
            max_turns=max_turns,
            extra_args={"no-session-persistence": None} if not persist_session else {},
            effort=effort,
        )
        pre_tool_use_hooks: list[HookCallback]
        if audit_guard is not None:
            assert audit_root is not None
            options.tools = list(_AUDIT_TOOLS)
            options.allowed_tools = list(_AUDIT_TOOLS)
            options.mcp_servers = {}
            options.strict_mcp_config = True
            options.setting_sources = []
            options.skills = []
            options.plugins = []
            options.agents = None
            options.env = _audit_cli_env(
                audit_root,
                base_environment=(
                    self._execution_input.child_environment()
                    if self._execution_input is not None
                    else None
                ),
            )
            pre_tool_use_hooks = [audit_guard]
        else:
            options.allowed_tools = ["Read", "Write", "Edit", "Bash", "Glob", "Grep"]
            options.setting_sources = ["user"]
            options.env = {
                **(self._execution_input.child_environment() if self._execution_input is not None else {}),
                **_CLI_ENV,
            }
            pre_tool_use_hooks = [_dangerous_command_guard, _background_bash_guard]
            if read_only:
                pre_tool_use_hooks.append(_read_only_guard)
        options.hooks = {
            "PreToolUse": [HookMatcher(matcher=_READ_ONLY_HOOK_MATCHER, hooks=pre_tool_use_hooks)]
        }

        if finalization:
            # `allowed_tools=[]` is a permission preapproval list, not tool
            # removal under bypassPermissions. SDK tools=[] emits --tools "";
            # the strict empty MCP config excludes externally configured tools.
            # Keep existing guards and allow only the native schema serializer.
            options.tools = []
            options.allowed_tools = []
            options.mcp_servers = {}
            options.strict_mcp_config = True
            assert options.hooks is not None
            options.hooks.setdefault("PreToolUse", []).append(
                HookMatcher(matcher=_READ_ONLY_HOOK_MATCHER, hooks=[_finalization_guard])
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

        # Record exact applied options, including resume only for an accepted
        # Claude token and multi-model capability for any nonempty agent mapping.
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
            reasoning_effort=effort,
            output_schema=output_schema,
            config=ClaudeRequestConfig(
                finalization=finalization,
                tools_count=len(options.tools) if isinstance(options.tools, list) else None,
                max_turns=max_turns,
                read_only=read_only,
                persist_session=persist_session,
                continuation_mode="resume" if resume_applied else "fresh",
                model_mode="multi_or_dynamic" if agents_nonempty else "single",
                permission_mode="bypassPermissions",
                allowed_tools_count=len(allowed_tools) if finalization or allowed_tools else None,
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

        if self._execution_input is not None:
            client: ClaudeSDKClient
            client = _RunLocalClaudeSDKClient(
                options=options,
                transport=_RunLocalSubprocessCLITransport(
                    options, environment=dict(options.env)
                ),
                initialize_timeout_s=_run_local_initialize_timeout_s(
                    dict(options.env)
                ),
            )
        else:
            client = ClaudeSDKClient(options=options)

        async with client:
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
                        # Emit cost when either cost or usage exists. Anthropic input excludes
                        # cache reads/writes; fold both in while retaining the cached subset.
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
        """Interrupt each active client in turn; propagate any interruption error."""
        for client in list(self._active_clients):
            await client.interrupt()
