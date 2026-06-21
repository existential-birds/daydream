# daydream/backends/__init__.py
"""Backend abstraction layer for daydream.

Defines the unified event stream, Backend protocol, and factory function.
Backends yield AgentEvent instances that the UI layer consumes without
knowing which backend produced them.

Event vocabulary (members of the ``AgentEvent`` TypeAlias union):

- ``TextEvent`` — agent text output.
- ``ThinkingEvent`` — extended reasoning / thinking content.
- ``ToolStartEvent`` — tool invocation started.
- ``ToolResultEvent`` — tool invocation completed.
- ``CostEvent`` — end-of-call cost/usage signal.
- ``MetricsEvent`` — per-turn LLM token/cost usage.
- ``TurnEndEvent`` — assistant-turn boundary; closes the recorder's open
  Step so multi-turn invocations are not collapsed into one Step.
- ``ResultEvent`` — final event in the stream; carries structured output
  and any continuation token.
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
        cost_usd: Total cost in USD; None when unavailable. Codex synthesizes
            via the #61 price table (#194 reverses D-16); None only when the
            model is unknown to the table.
        input_tokens: Prompt tokens (None when unavailable).
        output_tokens: Completion tokens (None when unavailable).
        cached_tokens: Cached portion of input_tokens (subset, NOT added
            to input_tokens per D-15). None when unavailable. Default
            ``None`` keeps existing 3-positional-arg call sites in
            ``backends/claude.py`` and ``backends/codex.py`` valid until
            Plans 03/04 update them.
        reasoning_tokens: Reasoning portion of output_tokens (subset, NOT
            additive — Codex's ``accounting.rs`` already counts these
            inside ``output_tokens``). Surfaces Codex's
            ``reasoning_output_tokens`` for cost attribution / perf
            observability (#192; openai/codex#26428 — count-only, no
            reasoning *content* is emitted). ``None`` on Claude (reasoning
            arrives via ThinkingEvent, a separate path) and when Codex
            omits the field.
        model_name: Real SDK model id observed during this call (e.g.
            ``claude-opus-4-5-20250901``). ``None`` when unavailable; the
            recorder uses it to upgrade a generic backend label
            (``"claude"``, ``"codex"``) to the actual model id.
        timestamp: ISO 8601 UTC timestamp populated at backend yield time.
    """

    cost_usd: float | None
    input_tokens: int | None
    output_tokens: int | None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None
    model_name: str | None = None
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
            (None when unavailable). NOT additive to ``prompt_tokens``
            (D-15).
        cost_usd: Per-turn cost in USD (None when unavailable). Codex
            synthesizes via the #61 price table (#194 reverses D-16); None
            only when the model is unknown to the table.
        reasoning_tokens: Reasoning portion of ``completion_tokens``
            (subset, NOT additive — Codex's ``accounting.rs`` already
            counts these inside ``output_tokens``). Surfaces Codex's
            ``reasoning_output_tokens`` for cost attribution / perf
            observability (#192; openai/codex#26428 — count-only, no
            reasoning *content* is emitted). ``None`` on Claude (reasoning
            arrives via ThinkingEvent, a separate path) and when Codex
            omits the field.
        model_name: Real SDK model id observed for this turn (e.g.
            ``claude-opus-4-5-20250901``). ``None`` when unavailable;
            recorder uses it to upgrade a generic backend label
            (``"claude"``, ``"codex"``) to the actual model id.
        timestamp: ISO 8601 UTC timestamp populated at backend yield time.
    """

    message_id: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int | None
    cost_usd: float | None
    reasoning_tokens: int | None = None
    model_name: str | None = None
    timestamp: str = field(default_factory=now_iso)


@dataclass
class TurnEndEvent:
    """Assistant-turn boundary signal.

    Emitted by each backend at the end of an assistant "turn" — for
    Claude, once per ``AssistantMessage``; for Codex, once per
    ``item.completed`` of type ``agent_message``. The trajectory
    recorder uses this to close its open Step so multi-turn invocations
    are recorded as one Step per turn (instead of collapsing into a
    single Step at invocation finish).

    Attributes:
        message_id: Correlator matching the message that ended this turn
            (e.g. Claude's ``AssistantMessage.message_id``). Empty string
            when the backend cannot supply one (Codex has no per-message
            id surface — D-04 correlator unused for Codex).
        timestamp: ISO 8601 UTC timestamp populated at backend yield time.
    """

    message_id: str = ""
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
    | TurnEndEvent
    | ResultEvent
)


class Backend(Protocol):
    """Protocol for agent backends.

    Each backend yields a stream of AgentEvent instances from execute().
    """

    model: str

    def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: dict[str, Any] | None = None,
        continuation: ContinuationToken | None = None,
        agents: dict[str, AgentDefinition] | None = None,
        max_turns: int | None = None,
        read_only: bool = False,
    ) -> AsyncIterator[AgentEvent]:
        """Yield AgentEvents for *prompt*.

        Args:
            read_only: When True, the backend enforces a non-mutating tool
                profile at the tool layer (Claude via a PreToolUse guard hook;
                Codex via ``--sandbox read-only``) so the agent can inspect
                history but cannot write/edit/delete or mutate the working
                tree. Wired True only for the failure-summarizer call.

                **Per-backend semantics diverge**: the Claude backend blocks
                git commits (the PreToolUse hook denies the Bash tool when the
                command is a mutating git operation), whereas the Codex backend
                permits git commits even under ``--sandbox read-only`` because
                Codex's sandbox only restricts filesystem writes, not git
                index/object-store operations.  Callers relying on a
                git-immutable guarantee must not assume ``read_only=True`` is
                sufficient across all backends.
        """
        ...

    async def cancel(self) -> None: ...

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str: ...


def create_backend(name: str, model: str | None = None) -> Backend:
    """Create a backend by name.

    Args:
        name: Backend name ("claude" or "codex").
        model: Optional CLI override. When ``None``, the per-backend default
            from :mod:`daydream.config` is used. This is the *only* place
            those defaults are read; downstream layers take ``model: str``
            as required.

    Returns:
        A Backend instance whose ``.model`` attribute is a non-empty string.

    Raises:
        ValueError: If the backend name is unknown.

    """
    from daydream.config import DEFAULT_CLAUDE_MODEL, DEFAULT_CODEX_MODEL

    if name == "claude":
        from daydream.backends.claude import ClaudeBackend
        return ClaudeBackend(model=model or DEFAULT_CLAUDE_MODEL)
    if name == "codex":
        from daydream.backends.codex import CodexBackend
        return CodexBackend(model=model or DEFAULT_CODEX_MODEL)
    raise ValueError(f"Unknown backend: {name!r}. Expected 'claude' or 'codex'.")


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
    "TurnEndEvent",
    "create_backend",
]
