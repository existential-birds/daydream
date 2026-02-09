# daydream/backends/__init__.py
"""Backend abstraction layer for daydream.

Defines the unified event stream, Backend protocol, and factory function.
Backends yield AgentEvent instances that the UI layer consumes without
knowing which backend produced them.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@dataclass
class TextEvent:
    """Agent text output."""

    text: str


@dataclass
class ThinkingEvent:
    """Extended thinking / reasoning."""

    text: str


@dataclass
class ToolStartEvent:
    """Tool invocation started."""

    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolResultEvent:
    """Tool invocation completed."""

    id: str
    output: str
    is_error: bool


@dataclass
class CostEvent:
    """Cost and usage information."""

    cost_usd: float | None
    input_tokens: int | None
    output_tokens: int | None


@dataclass
class ContinuationToken:
    """Opaque token for multi-turn interactions."""

    backend: str
    data: dict[str, Any]


@dataclass
class ResultEvent:
    """Final event in the stream. Carries structured output and continuation token."""

    structured_output: Any | None
    continuation: ContinuationToken | None


AgentEvent = (
    TextEvent | ThinkingEvent | ToolStartEvent
    | ToolResultEvent | CostEvent | ResultEvent
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
    "ResultEvent",
    "TextEvent",
    "ThinkingEvent",
    "ToolResultEvent",
    "ToolStartEvent",
    "create_backend",
]
