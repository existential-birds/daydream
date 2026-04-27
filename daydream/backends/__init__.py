# daydream/backends/__init__.py
"""Backend abstraction layer for daydream.

Defines the unified event stream, Backend protocol, and factory function.
Backends yield AgentEvent instances that the UI layer consumes without
knowing which backend produced them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from daydream.trajectory import now_iso

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from claude_agent_sdk.types import AgentDefinition


@dataclass
class TextEvent:
    """Agent text output.

    Attributes:
        text: The text emitted by the agent.
        timestamp: ISO 8601 UTC timestamp populated at backend yield time
            via ``now_iso()`` (Pitfall 2 single-source-of-truth).
    """

    text: str
    timestamp: str = field(default_factory=now_iso)


@dataclass
class ThinkingEvent:
    """Extended thinking / reasoning.

    Attributes:
        text: Reasoning content emitted by the agent.
        timestamp: ISO 8601 UTC timestamp populated at backend yield time.
    """

    text: str
    timestamp: str = field(default_factory=now_iso)


@dataclass
class ToolStartEvent:
    """Tool invocation started.

    Attributes:
        id: Tool call identifier (Claude block.id or Codex item.id /
            synthesized UUID).
        name: Tool function name.
        input: Tool arguments dict; may be empty but is never None.
        timestamp: ISO 8601 UTC timestamp populated at backend yield time.
    """

    id: str
    name: str
    input: dict[str, Any]
    timestamp: str = field(default_factory=now_iso)


@dataclass
class ToolResultEvent:
    """Tool invocation completed.

    Attributes:
        id: Tool call identifier matching the prior ToolStartEvent.id.
        output: Tool output as a string.
        is_error: True if the tool reported an error.
        timestamp: ISO 8601 UTC timestamp populated at backend yield time.
    """

    id: str
    output: str
    is_error: bool
    timestamp: str = field(default_factory=now_iso)


@dataclass
class CostEvent:
    """Cost and usage information (end-of-call signal feeding FinalMetrics).

    Attributes:
        cost_usd: Total cost in USD; None when unavailable (Codex always
            None per D-16).
        input_tokens: Prompt tokens (None when unavailable).
        output_tokens: Completion tokens (None when unavailable).
        cached_tokens: Cached portion of input_tokens (subset, NOT added
            to input_tokens per D-15). None when unavailable. Default
            ``None`` keeps existing 3-positional-arg call sites in
            ``backends/claude.py`` and ``backends/codex.py`` valid until
            Plans 03/04 update them.
        timestamp: ISO 8601 UTC timestamp populated at backend yield time.
    """

    cost_usd: float | None
    input_tokens: int | None
    output_tokens: int | None
    cached_tokens: int | None = None
    timestamp: str = field(default_factory=now_iso)


@dataclass
class MetricsEvent:
    """Per-step LLM token/cost usage.

    Emitted once per AssistantMessage by the Claude backend (keyed via
    ``AssistantMessage.message_id``), and once per ``turn.completed`` by
    the Codex backend (with empty ``message_id`` since Codex has no
    per-message id). The recorder uses ``message_id`` to attach Metrics
    to the correct agent Step (D-04, MAP-06).

    Attributes:
        message_id: Identifier matching the AssistantMessage that owns
            this metric. Empty string for Codex (D-16).
        prompt_tokens: Prompt tokens for this turn. REQUIRED per EVNT-02
            (int, not Optional) — every AssistantMessage / turn.completed
            carries it. Backends read the SDK key (Claude
            ``usage["input_tokens"]``, Codex ``usage["input_tokens"]``)
            and rename at the boundary.
        completion_tokens: Completion tokens for this turn. REQUIRED per
            EVNT-02 (int, not Optional). Backends read the SDK key
            (Claude ``usage["output_tokens"]``, Codex
            ``usage["output_tokens"]``) and rename at the boundary.
        cached_tokens: Subset of ``prompt_tokens`` served from cache
            (None when unavailable; Codex always None per D-16). NOT
            additive to ``prompt_tokens`` (D-15).
        cost_usd: Per-turn cost in USD (None when unavailable; Codex
            always None per D-16 — DO NOT synthesize from a token-price
            table).
        timestamp: ISO 8601 UTC timestamp populated at backend yield time.
    """

    message_id: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int | None
    cost_usd: float | None
    timestamp: str = field(default_factory=now_iso)


@dataclass
class ContinuationToken:
    """Opaque token for multi-turn interactions."""

    backend: str
    data: dict[str, Any]


@dataclass
class ResultEvent:
    """Final event in the stream. Carries structured output and continuation token.

    Attributes:
        structured_output: Schema-validated structured result, or None.
        continuation: Optional continuation token for multi-turn flows.
        timestamp: ISO 8601 UTC timestamp populated at backend yield time.
    """

    structured_output: Any | None
    continuation: ContinuationToken | None
    timestamp: str = field(default_factory=now_iso)


AgentEvent = (
    TextEvent
    | ThinkingEvent
    | ToolStartEvent
    | ToolResultEvent
    | CostEvent
    | MetricsEvent
    | ResultEvent
)


class Backend(Protocol):
    """Protocol for agent backends.

    Each backend yields a stream of AgentEvent instances from execute().
    """

    def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: dict[str, Any] | None = None,
        continuation: ContinuationToken | None = None,
        agents: dict[str, AgentDefinition] | None = None,
        max_turns: int | None = None,
    ) -> AsyncIterator[AgentEvent]: ...

    async def cancel(self) -> None: ...

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str: ...


def create_backend(name: str, model: str | None = None) -> Backend:
    """Create a backend by name.

    Args:
        name: Backend name ("claude" or "codex").
        model: Optional model override. Each backend has its own default.

    Returns:
        A Backend instance.

    Raises:
        ValueError: If the backend name is unknown.

    """
    if name == "claude":
        from daydream.backends.claude import ClaudeBackend
        return ClaudeBackend(model=model or "opus")
    if name == "codex":
        from daydream.backends.codex import CodexBackend
        return CodexBackend(model=model or "gpt-5.3-codex")
    raise ValueError(f"Unknown backend: {name!r}. Expected 'claude' or 'codex'.")


# Re-export ClaudeBackend at package level for convenience
from daydream.backends.claude import ClaudeBackend  # noqa: E402

__all__ = [
    "AgentEvent",
    "Backend",
    "ClaudeBackend",
    "ContinuationToken",
    "CostEvent",
    "MetricsEvent",
    "ResultEvent",
    "TextEvent",
    "ThinkingEvent",
    "ToolResultEvent",
    "ToolStartEvent",
    "create_backend",
]
