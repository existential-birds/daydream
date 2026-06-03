# daydream/backends/claude.py
"""Claude Agent SDK backend for daydream."""

from __future__ import annotations

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

# Read-only Bash allowlist for the enforced read-only profile (the failure
# summarizer). A command is permitted only if it begins with one of these
# prefixes AND contains no shell-chaining metacharacters that could smuggle in
# a mutating command. Mirrored in the summarizer prompt prose
# (``daydream/phases.py``). Conservative by design: any chain → deny.
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

# File-mutating tools that must be denied outright under read_only. Under
# ``permission_mode="bypassPermissions"`` the SDK's ``allowed_tools`` does NOT
# restrict the toolset (verified — a Write executed despite omission), so the
# PreToolUse hook below is the actual enforcement, not the tool list.
_READ_ONLY_HOOK_MATCHER = "Bash|Write|Edit|MultiEdit|NotebookEdit"


def _is_read_only_command(cmd: str) -> bool:
    """Return True only if *cmd* is a single allowlisted read-only command.

    Denies (returns False) on: an empty/blank command, any command whose first
    token is not an allowlisted prefix, and any command containing a shell
    chaining metacharacter (``;``, ``&&``, ``||``, ``|``, backtick, ``$(``).
    """
    stripped = cmd.strip()
    if not stripped:
        return False
    if any(meta in stripped for meta in _CHAIN_METACHARS):
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

    Fires (under ``bypassPermissions``) for Bash and the file-mutating tools.
    Denies every file-mutating tool outright and denies any Bash command that
    is not on the read-only allowlist. Fails closed: malformed input → deny.
    Returns ``{}`` (allow) only for an allowlisted read-only Bash command.
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
    # Any other matched (file-mutating) tool, or unparseable input → deny.
    return _read_only_deny(
        f"read-only summarizer: tool {tool_name!r} is blocked (non-mutating contract)"
    )


class ClaudeBackend:
    """Backend that wraps the Claude Agent SDK.

    Translates Claude SDK message types into the unified AgentEvent stream.
    """

    def __init__(self, model: str):
        self.model = model
        self._client: ClaudeSDKClient | None = None

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

        """
        output_format = (
            {"type": "json_schema", "schema": output_schema}
            if output_schema
            else None
        )

        # Enforced read-only profile (failure summarizer): register a PreToolUse
        # guard hook that denies file-mutating tools and non-allowlisted Bash.
        # The hook — NOT allowed_tools — is the enforcement: under
        # bypassPermissions the tool list does not restrict the toolset.
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
        # Track the most recently observed AssistantMessage.model so we can
        # stamp the real SDK model id (e.g. ``claude-opus-4-5-20250901``) on
        # the trailing CostEvent. The trajectory recorder uses this to
        # upgrade the generic ``"claude"`` backend label to the actual id.
        last_assistant_model: str | None = None
        # StructuredOutput ToolUseBlocks are intentionally skipped (the
        # structured result is captured via ResultMessage.structured_output
        # instead).  Track their IDs so the corresponding ToolResultBlocks
        # in the subsequent UserMessage are also skipped — otherwise the
        # trajectory recorder logs them as unmatched_tool_results.
        skipped_tool_ids: set[str] = set()

        async with ClaudeSDKClient(options=options) as client:
            self._client = client
            try:
                await client.query(prompt)
                async for msg in client.receive_response():
                    if isinstance(msg, AssistantMessage):
                        # Real SDK AssistantMessage has a ``model: str`` field
                        # (no ``usage``); see backends/__init__.py docstring.
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
                                    skipped_tool_ids.add(block.id)
                                    continue
                                yield ToolStartEvent(
                                    id=block.id,
                                    name=block.name,
                                    input=block.input or {},
                                )
                        # Phase 2 (EVNT-06): emit MetricsEvent per AssistantMessage
                        # keyed by message_id (D-04 maps MetricsEvent -> open Step).
                        # cost_usd is unavailable per-message (only on ResultMessage)
                        # — leave None per Claude's Discretion in CONTEXT.md.
                        # EVNT-02 field names: MetricsEvent uses `prompt_tokens` /
                        # `completion_tokens` (the ATIF/Metrics-side names). The SDK
                        # boundary keys are `input_tokens` / `output_tokens`; rename
                        # at this boundary. Skip emission when either is missing —
                        # EVNT-02 types both as int (not Optional) so a partial
                        # usage dict would yield a malformed event. `getattr` keeps
                        # us defensive against older test mocks that pre-date the
                        # SDK's `usage` field on AssistantMessage.
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
                        # TurnEndEvent closes the recorder's Step for THIS assistant turn.
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
                        if msg.structured_output is not None:
                            structured_result = msg.structured_output
                        # Phase 2 (EVNT-04, EVNT-05): emit CostEvent whenever cost OR
                        # usage data is available. Previously this branch dropped
                        # input_tokens / output_tokens / cached_tokens (always None).
                        # Trust per-call semantics for SDK 0.1.52 (D-14); if Phase 5
                        # TEST-06 finds the SDK reports cumulative, the fix lands
                        # there. cached_tokens is a SUBSET of input_tokens, NOT
                        # additive (D-15) — pass cache_read_input_tokens through
                        # directly. `getattr` keeps us defensive against older test
                        # mocks that pre-date the SDK's `usage` field on
                        # ResultMessage.
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
                self._client = None

    async def cancel(self) -> None:
        """Cancel the running agent."""
        if self._client is not None:
            await self._client.interrupt()

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
