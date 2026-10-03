# daydream/backends/claude.py
"""Claude Agent SDK backend for daydream."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, cast

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, HookMatcher
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
from daydream.backends._claude_guards import (
    _AUDIT_TOOLS as _AUDIT_TOOLS,
    _CLI_ENV as _CLI_ENV,
    _READ_ONLY_ALLOWED_TOOLS as _READ_ONLY_ALLOWED_TOOLS,
    _READ_ONLY_HOOK_MATCHER as _READ_ONLY_HOOK_MATCHER,
    READ_ONLY_BASH_ALLOWLIST as READ_ONLY_BASH_ALLOWLIST,
    _audit_cli_env as _audit_cli_env,
    _audit_symlink_inventory as _audit_symlink_inventory,
    _background_bash_guard as _background_bash_guard,
    _build_audit_root_guard as _build_audit_root_guard,
    _dangerous_command_guard as _dangerous_command_guard,
    _finalization_guard as _finalization_guard,
    _is_background_bash as _is_background_bash,
    _is_dangerous_command as _is_dangerous_command,
    _is_read_only_command as _is_read_only_command,
    _read_only_guard as _read_only_guard,
)
from daydream.backends._claude_transport import (
    _run_local_initialize_timeout_s as _run_local_initialize_timeout_s,
    _RunLocalClaudeSDKClient as _RunLocalClaudeSDKClient,
    _RunLocalSubprocessCLITransport as _RunLocalSubprocessCLITransport,
)


def _total_input_tokens(usage: dict[str, Any]) -> int | None:
    """Sum uncached input and both cache buckets; absent input_tokens keeps the total absent."""
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


_CLAUDE_EFFORT_LEVELS: frozenset[str] = frozenset(("low", "medium", "high", "xhigh", "max"))


def _claude_effort(value: str | None) -> EffortLevel | None:
    """Validate native SDK effort at construction, before an unsupported level reaches the CLI."""
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
        self.audit_root_isolation = AUDIT_ROOT_ISOLATION if self.audit_root is not None else None
        lexical_links = frozenset(Path(os.path.abspath(path)) for path in audit_outward_symlinks)
        if self.audit_root is None and lexical_links:
            raise ValueError("audit_outward_symlinks requires audit_root")
        if self.audit_root is not None and any(not path.is_relative_to(self.audit_root) for path in lexical_links):
            raise ValueError("audit_outward_symlinks must be inside audit_root")
        self.audit_outward_symlinks = lexical_links
        audit_symlinks = (
            lexical_links | _audit_symlink_inventory(self.audit_root) if self.audit_root is not None else frozenset()
        )
        self._audit_root_guard = (
            _build_audit_root_guard(self.audit_root, audit_symlinks) if self.audit_root is not None else None
        )
        if execution_input is not None:
            self.fanout_concurrency = execution_input.fanout_concurrency
            self.retry_policy = execution_input.retry_policy
        else:
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
        finalization: bool = False,
    ) -> AsyncGenerator[AgentEvent, None]:
        """Yield SDK events; error results raise ClaudeAgentError. Resume only Claude tokens, using the
        original token on retries and preserving specialist agent names. PreToolUse guards enforce
        read_only under bypassPermissions. Finalization removes investigation tools, uses low effort,
        and retains guards/StructuredOutput. persist_session=False suppresses continuation.
        """
        audit_root = self.audit_root
        audit_guard = self._audit_root_guard
        if audit_root is not None:
            try:
                resolved_cwd = cwd.resolve(strict=True)
            except (OSError, RuntimeError, ValueError) as exc:
                raise ClaudeAgentError("audit isolation requires an existing canonical cwd") from exc
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
        output_format = {"type": "json_schema", "schema": output_schema} if output_schema else None

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
                    self._execution_input.child_environment() if self._execution_input is not None else None
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
        options.hooks = {"PreToolUse": [HookMatcher(matcher=_READ_ONLY_HOOK_MATCHER, hooks=pre_tool_use_hooks)]}

        if finalization:
            # Permission preapproval does not remove tools: use SDK tools=[] and strict empty MCP config.
            # Retain guards and allow only native schema serialization.
            options.tools = []
            options.allowed_tools = []
            options.mcp_servers = {}
            options.strict_mcp_config = True
            assert options.hooks is not None
            options.hooks.setdefault("PreToolUse", []).append(
                HookMatcher(matcher=_READ_ONLY_HOOK_MATCHER, hooks=[_finalization_guard])
            )

        # Resume only Claude tokens; otherwise byte-stable options preserve prefix caching.
        if continuation is not None and continuation.backend == "claude":
            resume_id = continuation.data.get("session_id")
            if resume_id:
                options.resume = resume_id

        if agents:
            options.agents = agents

        # Record exact applied options, including resume only for an accepted
        # Claude token and multi-model capability for any nonempty agent mapping.
        resume_applied = (
            continuation is not None and continuation.backend == "claude" and bool(continuation.data.get("session_id"))
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
        # Track skipped StructuredOutput ids; output arrives on ResultMessage and matching results must also be skipped.
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
                transport=_RunLocalSubprocessCLITransport(options, environment=dict(options.env)),
                initialize_timeout_s=_run_local_initialize_timeout_s(dict(options.env)),
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
                        # Emit per-assistant metrics only with both token counts; rename SDK input/output fields.
                        # Per-message cost stays None because billing arrives on ResultMessage.
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
                                content_str = (
                                    content
                                    if isinstance(content, str)
                                    else (json.dumps(content, ensure_ascii=False) if content is not None else "")
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
                        providers = {
                            entry.provider_name
                            for entry in (model_usage or {}).values()
                            if entry.provider_name is not None
                        }
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
                                if persist_session and session_id and not msg.is_error
                                else None
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
                                    f"Claude agent run failed: {detail}",
                                    subtype="error_max_turns",
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
