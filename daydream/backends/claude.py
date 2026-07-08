# daydream/backends/claude.py
"""Claude Agent SDK backend for daydream."""

from __future__ import annotations

import re
import shlex
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, HookJSONOutput, HookMatcher
from claude_agent_sdk.types import (
    AgentDefinition,
    AssistantMessage,
    HookCallback,
    ResultMessage,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)

from daydream.backends import (
    AgentEvent,
    ContinuationToken,
    CostEvent,
    MetricsEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
    TurnEndEvent,
)

# Read-only Bash allowlist (failure summarizer): permitted only if the command
# begins with one of these prefixes AND has no shell-chaining metacharacter that
# could smuggle in a mutation. Mirrored in the summarizer prompt (phases.py).
READ_ONLY_BASH_ALLOWLIST: tuple[str, ...] = (
    "ls",
    "cat",
    "git status",
    "git log",
    "git show",
    "git blame",
    "git diff",
)

# Chaining metacharacters that can append a second, non-allowlisted command.
_CHAIN_METACHARS: tuple[str, ...] = (";", "&&", "||", "|", "`", "$(")

# Single-char danger tokens checked against shlex output: shlex (non-posix)
# splits ``&&``/``$(`` into single chars, so we check per-char. Safe inside
# quotes (shlex returns a quoted chunk as one token).
_DANGEROUS_TOKENS: frozenset[str] = frozenset({"|", ";", "&", "`", "$"})

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


def _total_input_tokens(usage: dict[str, Any]) -> int | None:
    """Fold Anthropic's three input buckets into the true total input.

    Anthropic reports `input_tokens` as the *uncached remainder* only, with
    cache hits and writes split into `cache_read_input_tokens` and
    `cache_creation_input_tokens` (mutually exclusive buckets). ATIF's
    `Metrics.prompt_tokens` is the total input, so fold all three. Returns
    None when `input_tokens` is absent (preserves the no-token-count gate).
    """
    input_tokens = usage.get("input_tokens")
    if input_tokens is None:
        return None
    return input_tokens + (usage.get("cache_read_input_tokens") or 0) + (usage.get("cache_creation_input_tokens") or 0)


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


def _is_read_only_command(cmd: str) -> bool:
    """Return True only if *cmd* is a single allowlisted read-only command.

    Denies (returns False) on: an empty/blank command, any command containing a
    newline or carriage return, any command whose first token is not an
    allowlisted prefix, and any command containing a shell chaining
    metacharacter (``;``, ``&&``, ``||``, ``|``, backtick, ``$(``).

    Metacharacter detection uses ``shlex`` to avoid false positives from
    metacharacters that appear only inside quoted arguments (e.g.
    ``git log --grep='fix|bug'`` is safe and must be allowed).  Newlines and
    carriage returns are bash command separators but ``shlex`` treats them as
    whitespace and elides them, so they are rejected directly on the raw string.
    """
    stripped = cmd.strip()
    if not stripped:
        return False
    if "\n" in cmd or "\r" in cmd:
        return False
    # Non-posix lex: quoted strings stay single tokens; unquoted metacharacters
    # appear as individual bare chars (``&&`` → ``&``, ``&``). See _DANGEROUS_TOKENS.
    try:
        tokens = list(shlex.shlex(stripped, posix=False))
    except ValueError:
        return False  # Malformed quoting — deny.
    for tok in tokens:
        if tok in _DANGEROUS_TOKENS:
            return False
    return any(
        stripped == prefix or stripped.startswith(prefix + " ")
        for prefix in READ_ONLY_BASH_ALLOWLIST
    )


def _read_only_deny(reason: str) -> HookJSONOutput:
    """Build a PreToolUse deny output (``permissionDecision="deny"``)."""
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


async def _read_only_guard(input_data: Any, tool_use_id: Any, context: Any) -> HookJSONOutput:
    """PreToolUse hook enforcing the read-only summarizer contract.

    Fires for ALL tools (matcher ``.*``). Explicitly allows only the safe set
    (Read, Grep, Glob, StructuredOutput, and allowlisted Bash commands) and
    denies everything else. Fails closed: malformed input → deny.
    Returns ``{}`` (allow) only for a permitted tool/command.
    """
    tool_name = input_data.get("tool_name") if isinstance(input_data, dict) else None
    if tool_name == "Bash":
        tool_input = input_data.get("tool_input") if isinstance(input_data, dict) else None
        command = ""
        if isinstance(tool_input, dict):
            raw = tool_input.get("command")
            command = raw if isinstance(raw, str) else ""
        if _is_read_only_command(command):
            return {}
        return _read_only_deny(
            f"read-only summarizer: non-read-only Bash command blocked: {command!r}"
        )
    if tool_name in _READ_ONLY_ALLOWED_TOOLS:
        return {}
    return _read_only_deny(
        f"read-only summarizer: tool {tool_name!r} is blocked (non-mutating contract)"
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
    tool_name = input_data.get("tool_name") if isinstance(input_data, dict) else None
    if tool_name != "Bash":
        return {}
    tool_input = input_data.get("tool_input") if isinstance(input_data, dict) else None
    command = ""
    if isinstance(tool_input, dict):
        raw = tool_input.get("command")
        command = raw if isinstance(raw, str) else ""
    if _is_dangerous_command(command):
        return _read_only_deny(f"dangerous command blocked (always-on guard): {command!r}")
    return {}


# A skill invocation daydream embeds in a prompt: a whitespace/line-anchored
# ``/{skill_key}`` token as emitted by ``format_skill_invocation`` (e.g.
# ``/beagle-python:review-python``). A filesystem path never matches — the key
# must end at whitespace/EOL, never at another ``/``.
_PROMPT_SKILL_PATTERN = re.compile(r"(?:^|(?<=\s))/([\w.-]+(?::[\w.-]+)*)(?=\s|$)")

# Beagle skills chain (e.g. a review skill loads review-verification-protocol),
# so any beagle-namespaced key stays invocable even when not in the prompt.
_CHAINABLE_SKILL_NAMESPACE_PREFIX = "beagle-"


def _prompt_skill_keys(prompt: str) -> frozenset[str]:
    """Skill keys explicitly invoked by *prompt* (see ``_PROMPT_SKILL_PATTERN``)."""
    return frozenset(_PROMPT_SKILL_PATTERN.findall(prompt))


def _make_skill_guard(allowed_skills: frozenset[str]) -> HookCallback:
    """Build the always-on PreToolUse hook that scopes the Skill tool.

    ``setting_sources=["user"]`` exposes every skill in the operator's Claude
    Code install — including the built-in ``/review``, which hunts for an open
    PR to review — and ``bypassPermissions`` leaves the Skill tool unguarded.
    The guard allows a Skill call only when the key was explicitly invoked by
    the prompt daydream sent, or is beagle-namespaced (skills chain within the
    plugin). Everything else — notably un-namespaced Claude Code built-ins —
    is denied with a redirect back to the prompt. Fails closed when the skill
    name is missing or malformed.
    """

    async def _skill_guard(input_data: Any, tool_use_id: Any, context: Any) -> HookJSONOutput:
        tool_name = input_data.get("tool_name") if isinstance(input_data, dict) else None
        if tool_name != "Skill":
            return {}
        tool_input = input_data.get("tool_input") if isinstance(input_data, dict) else None
        requested = tool_input.get("skill") if isinstance(tool_input, dict) else None
        if isinstance(requested, str) and (
            requested in allowed_skills
            or requested.split(":", 1)[0].startswith(_CHAINABLE_SKILL_NAMESPACE_PREFIX)
        ):
            return {}
        return _read_only_deny(
            f"skill {requested!r} was not requested by this task — do not invoke "
            "skills or slash commands on your own; follow the prompt instructions "
            "directly and reply with your analysis as plain text"
        )

    return _skill_guard


class ClaudeBackend:
    """Backend that wraps the Claude Agent SDK.

    Translates Claude SDK message types into the unified AgentEvent stream.
    """

    concise_fix_prompts = False

    def __init__(self, model: str):
        self.model = model
        self.fanout_concurrency = 4
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
    ) -> AsyncIterator[AgentEvent]:
        """Execute a prompt and yield unified events.

        Args:
            continuation: Ignored by Claude backend.
            agents: Optional mapping of specialist name -> AgentDefinition for
                subagent support. Keys are the specialist names the lead agent
                dispatches by; they MUST be preserved verbatim.
            read_only: When True, register a ``PreToolUse`` guard hook that
                denies file-mutating tools (Write/Edit/...) and any Bash command
                not on ``READ_ONLY_BASH_ALLOWLIST``. The hook is the enforcement
                — under ``bypassPermissions`` ``allowed_tools`` does not restrict
                the toolset — so the tool list is left unchanged.

        Raises:
            ClaudeAgentError: If the agent run ends with an error result
                (``ResultMessage.is_error``), e.g. an invalid API key.
        """
        output_format = (
            {"type": "json_schema", "schema": output_schema}
            if output_schema
            else None
        )

        # PreToolUse hooks — NOT allowed_tools — are the enforcement, since
        # bypassPermissions leaves the tool list unrestricted. The dangerous-command
        # and skill guards are always-on (all phases); the read-only guard composes
        # on top when read_only=True.
        pre_tool_use_hooks: list[HookCallback] = [
            _dangerous_command_guard,
            _make_skill_guard(_prompt_skill_keys(prompt)),
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
            hooks={
                "PreToolUse": [HookMatcher(matcher=_READ_ONLY_HOOK_MATCHER, hooks=pre_tool_use_hooks)]
            },
        )

        if agents:
            options.agents = agents

        structured_result: Any = None
        # Latest AssistantMessage.model, stamped on the trailing CostEvent so the
        # recorder can upgrade the generic ``"claude"`` label to the real SDK id.
        last_assistant_model: str | None = None
        # StructuredOutput ToolUseBlocks are skipped (result comes via
        # ResultMessage.structured_output); track their IDs so the matching
        # ToolResultBlocks aren't logged as unmatched_tool_results.
        skipped_tool_ids: set[str] = set()

        async with ClaudeSDKClient(options=options) as client:
            self._active_clients.add(client)
            try:
                await client.query(prompt)
                async for msg in client.receive_response():
                    if isinstance(msg, AssistantMessage):
                        msg_model = getattr(msg, "model", None)
                        if isinstance(msg_model, str) and msg_model:
                            last_assistant_model = msg_model
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
                            )
                        yield TurnEndEvent(message_id=getattr(msg, "message_id", "") or "")

                    elif isinstance(msg, UserMessage):
                        for user_block in msg.content:
                            if isinstance(user_block, ToolResultBlock):
                                if user_block.tool_use_id in skipped_tool_ids:
                                    skipped_tool_ids.discard(user_block.tool_use_id)
                                    continue
                                content_str = str(user_block.content) if user_block.content else ""
                                yield ToolResultEvent(
                                    id=user_block.tool_use_id,
                                    output=content_str,
                                    is_error=user_block.is_error or False,
                                )

                    elif isinstance(msg, ResultMessage):
                        if msg.is_error:
                            detail = msg.result or msg.subtype or "unknown error"
                            if msg.subtype == "error_max_turns":
                                raise MaxTurnsError(
                                    f"Claude agent run failed: {detail}",
                                    subtype="error_max_turns",
                                )
                            raise ClaudeAgentError(f"Claude agent run failed: {detail}")
                        if msg.structured_output is not None:
                            structured_result = msg.structured_output
                        # EVNT-04/05: emit CostEvent when cost OR usage is available.
                        # Per-call semantics trusted for SDK 0.1.52 (D-14). Anthropic's raw
                        # `input_tokens` is the *uncached remainder* only; we fold in the
                        # cache-read and cache-creation buckets so the emitted value is the
                        # true total input, matching ATIF Metrics.prompt_tokens. cached_tokens
                        # stays the cache-read hit subset of that total.
                        result_usage = getattr(msg, "usage", None)
                        if msg.total_cost_usd is not None or result_usage is not None:
                            usage = result_usage or {}
                            yield CostEvent(
                                cost_usd=msg.total_cost_usd,
                                input_tokens=_total_input_tokens(usage),
                                output_tokens=usage.get("output_tokens"),
                                cached_tokens=usage.get("cache_read_input_tokens"),
                                model_name=last_assistant_model,
                            )

                yield ResultEvent(
                    structured_output=structured_result,
                    continuation=None,
                )
            finally:
                self._active_clients.discard(client)

    async def cancel(self) -> None:
        """Interrupt every active SDK client.

        Sends an interrupt to each in-flight agent client in turn; an error
        raised by any client's interrupt propagates to the caller.
        """
        for client in list(self._active_clients):
            await client.interrupt()

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
        """Format a skill invocation for Claude.

        Claude uses /{namespace:skill} syntax.
        """
        result = f"/{skill_key}"
        if args:
            result = f"{result} {args}"
        return result
