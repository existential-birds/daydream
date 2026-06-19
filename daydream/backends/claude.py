# daydream/backends/claude.py
"""Claude Agent SDK backend for daydream."""

from __future__ import annotations

import shlex
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, HookJSONOutput, HookMatcher
from claude_agent_sdk.types import (
    AgentDefinition,
    AssistantMessage,
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

# Tools unconditionally permitted under the read-only profile (Bash handled
# separately via the command allowlist).
_READ_ONLY_ALLOWED_TOOLS: frozenset[str] = frozenset(
    {"Read", "Grep", "Glob", "StructuredOutput"}
)


class ClaudeAgentError(Exception):
    """Raised when the Claude agent run reports an error result.

    The SDK surfaces fatal run failures (invalid API key, execution errors,
    hitting max turns) as a ``ResultMessage`` with ``is_error=True`` rather
    than raising. Translating that flag into an exception here keeps an
    errored run from masquerading as a clean empty result downstream — e.g.
    a review exiting 0 with "no issues found" because the agent never ran.
    """


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


class ClaudeBackend:
    """Backend that wraps the Claude Agent SDK.

    Translates Claude SDK message types into the unified AgentEvent stream.
    """

    def __init__(self, model: str):
        self.model = model
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
            cwd: Working directory for the agent.
            prompt: The prompt to send.
            output_schema: Optional JSON schema for structured output.
            continuation: Ignored by Claude backend.
            agents: Optional mapping of specialist name -> AgentDefinition for
                subagent support. Keys are the specialist names the lead agent
                dispatches by; they MUST be preserved verbatim.
            read_only: When True, register a ``PreToolUse`` guard hook that
                denies file-mutating tools (Write/Edit/...) and any Bash command
                not on ``READ_ONLY_BASH_ALLOWLIST``. The hook is the enforcement
                — under ``bypassPermissions`` ``allowed_tools`` does not restrict
                the toolset — so the tool list is left unchanged.

        Yields:
            AgentEvent instances.

        Raises:
            ClaudeAgentError: If the agent run ends with an error result
                (``ResultMessage.is_error``), e.g. an invalid API key.

        """
        output_format = (
            {"type": "json_schema", "schema": output_schema}
            if output_schema
            else None
        )

        # Read-only profile: the PreToolUse hook — NOT allowed_tools — is the
        # enforcement, since bypassPermissions leaves the tool list unrestricted.
        options = ClaudeAgentOptions(
            cwd=str(cwd),
            permission_mode="bypassPermissions",
            allowed_tools=["Read", "Write", "Edit", "Bash", "Glob", "Grep"],
            setting_sources=["user"],
            model=self.model,
            output_format=output_format,
            max_buffer_size=10 * 1024 * 1024,  # 10MB — handles large git diffs
            max_turns=max_turns,
            hooks=(
                {"PreToolUse": [HookMatcher(matcher=_READ_ONLY_HOOK_MATCHER, hooks=[_read_only_guard])]}
                if read_only
                else None
            ),
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
                            yield MetricsEvent(
                                message_id=getattr(msg, "message_id", "") or "",
                                prompt_tokens=msg_usage["input_tokens"],
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
                            raise ClaudeAgentError(f"Claude agent run failed: {detail}")
                        if msg.structured_output is not None:
                            structured_result = msg.structured_output
                        # EVNT-04/05: emit CostEvent when cost OR usage is available.
                        # Per-call semantics trusted for SDK 0.1.52 (D-14). cached_tokens
                        # is a SUBSET of input_tokens, not additive (D-15).
                        result_usage = getattr(msg, "usage", None)
                        if msg.total_cost_usd is not None or result_usage is not None:
                            usage = result_usage or {}
                            yield CostEvent(
                                cost_usd=msg.total_cost_usd,
                                input_tokens=usage.get("input_tokens"),
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
        """Cancel all running agents."""
        for client in list(self._active_clients):
            await client.interrupt()

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
        """Format a skill invocation for Claude.

        Claude uses /{namespace:skill} syntax.

        Args:
            skill_key: Full skill key (e.g. "beagle-python:review-python").
            args: Optional arguments string.

        Returns:
            Formatted skill invocation string.

        """
        result = f"/{skill_key}"
        if args:
            result = f"{result} {args}"
        return result
