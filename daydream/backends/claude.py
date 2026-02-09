# daydream/backends/claude.py
"""Claude Agent SDK backend for daydream."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from claude_agent_sdk.types import (
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
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
)


class ClaudeBackend:
    """Backend that wraps the Claude Agent SDK.

    Translates Claude SDK message types into the unified AgentEvent stream.
    """

    def __init__(self, model: str = "opus"):
        self.model = model
        self._client: ClaudeSDKClient | None = None

    async def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: dict[str, Any] | None = None,
        continuation: ContinuationToken | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Execute a prompt and yield unified events.

        Args:
            cwd: Working directory for the agent.
            prompt: The prompt to send.
            output_schema: Optional JSON schema for structured output.
            continuation: Ignored by Claude backend.

        Yields:
            AgentEvent instances.

        """
        output_format = (
            {"type": "json_schema", "schema": output_schema}
            if output_schema
            else None
        )

        options = ClaudeAgentOptions(
            cwd=str(cwd),
            permission_mode="bypassPermissions",
            setting_sources=["user", "project", "local"],
            model=self.model,
            output_format=output_format,
        )

        structured_result: Any = None

        async with ClaudeSDKClient(options=options) as client:
            self._client = client
            try:
                await client.query(prompt)
                async for msg in client.receive_response():
                    if isinstance(msg, AssistantMessage):
                        for block in msg.content:
                            if isinstance(block, TextBlock) and block.text:
                                yield TextEvent(text=block.text)
                            elif isinstance(block, ThinkingBlock) and block.thinking:
                                yield ThinkingEvent(text=block.thinking)
                            elif isinstance(block, ToolUseBlock):
                                if block.name == "StructuredOutput":
                                    continue
                                yield ToolStartEvent(
                                    id=block.id,
                                    name=block.name,
                                    input=block.input or {},
                                )

                    elif isinstance(msg, UserMessage):
                        for user_block in msg.content:
                            if isinstance(user_block, ToolResultBlock):
                                content_str = str(user_block.content) if user_block.content else ""
                                yield ToolResultEvent(
                                    id=user_block.tool_use_id,
                                    output=content_str,
                                    is_error=user_block.is_error or False,
                                )

                    elif isinstance(msg, ResultMessage):
                        if msg.structured_output is not None:
                            structured_result = msg.structured_output
                        if msg.total_cost_usd is not None:
                            yield CostEvent(
                                cost_usd=msg.total_cost_usd,
                                input_tokens=None,
                                output_tokens=None,
                            )

                yield ResultEvent(
                    structured_output=structured_result,
                    continuation=None,
                )
            finally:
                self._client = None

    async def cancel(self) -> None:
        """Cancel the running agent.

        The SDK client is cleaned up via the async context manager in execute().
        Cancellation is triggered by raising KeyboardInterrupt from a signal handler.
        """

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
