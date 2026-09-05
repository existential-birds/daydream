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

import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from daydream.trajectory import now_iso

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
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
        exit_code: Exit code as reported by the backend; None when the
            backend has no structured exit code (Claude/Pi).
        status: Backend-native status string (e.g. Codex's
            "completed"/"declined"); None when unavailable.
        duration_ms: Wall-clock duration in milliseconds; None when
            unavailable.
        cancelled: True if the backend reported the tool call as cancelled.
        truncated: True if the backend marked the output as truncated.
        All five are optional; they default to None/False for backends
        without structured metadata (Claude/Pi).
    """

    id: str
    output: str
    is_error: bool
    timestamp: str = field(default_factory=now_iso)
    # Keep these after timestamp so existing positional ToolResultEvent
    # callers continue to interpret their arguments up to is_error/timestamp
    # as-is; the metadata fields are optional and defaulted.
    exit_code: int | None = None
    status: str | None = None
    duration_ms: float | None = None
    cancelled: bool = False
    truncated: bool = False


@dataclass
class CostEvent:
    """Cost and usage information (end-of-call signal feeding FinalMetrics).

    Attributes:
        cost_usd: Total cost in USD; None when unavailable. Codex synthesizes
            via the #61 price table (#194 reverses D-16); None only when the
            model is unknown to the table.
        input_tokens: Prompt tokens (None when unavailable).
        output_tokens: Completion tokens (None when unavailable).
        cached_tokens: Cache-read hit subset of input_tokens. input_tokens
            is the total input (backends fold cache read+creation into it);
            cached_tokens is the read subset, NOT added to input_tokens.
            None when unavailable. Default ``None`` keeps existing
            3-positional-arg call sites in ``backends/claude.py`` and
            ``backends/codex.py`` valid until Plans 03/04 update them.
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
            (``"claude"``, ``"codex"``, ``"osprey"``) to the actual model id.
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
        cached_tokens: Cache-read hit subset of ``prompt_tokens``
            (None when unavailable). ``prompt_tokens`` is the total input
            (backends fold cache read+creation into it); cached_tokens is
            the read subset, NOT additive to ``prompt_tokens``.
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
            (``"claude"``, ``"codex"``, ``"osprey"``) to the actual model id.
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
        structured_output: Structured result as emitted by the backend,
            schema-validated (or salvage-checked) at the run_agent return
            path when the caller opts in (``validate_structured_output``
            True), or None. Callers that pass
            ``validate_structured_output=False`` re-validate downstream.
        continuation: Optional continuation token for multi-turn flows.
        model_name: Real SDK model id observed for this invocation. Backends
            should populate this when the model is only available from a
            session-level terminal event rather than per-turn usage.
        timestamp: ISO 8601 UTC timestamp populated at backend yield time.
    """

    structured_output: Any | None
    continuation: ContinuationToken | None
    timestamp: str = field(default_factory=now_iso)
    # Keep this after timestamp so existing three-positional-argument
    # ResultEvent callers continue to interpret their third argument as the
    # timestamp.
    model_name: str | None = None


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


class AgentEventStream(AsyncIterator[AgentEvent], Protocol):
    """Closable event stream owned by one backend invocation.

    Closing the stream must release only the resources created by the matching
    :meth:`Backend.execute` call. It must not interrupt other streams returned
    by the same backend instance.
    """

    async def aclose(self) -> None:
        """Close this invocation and release its resources."""
        ...


class Backend(Protocol):
    """Protocol for agent backends.

    Each backend yields a stream of AgentEvent instances from execute().

    Optional extension: backends may expose ``fanout_concurrency: int`` as a
    scheduling hint for orchestrator-managed parallel calls. Callers combine
    the hint with their workflow ceiling via
    :func:`effective_fanout_concurrency`; absent hints fall back to four.

    Optional extension: backends may expose ``concise_fix_prompts: bool`` to
    request verbosity-suppressing fix-phase prompts (set True for pi/GLM, which
    produces verbose reasoning). When absent, the caller falls back to False via
    ``getattr(backend, "concise_fix_prompts", False)``.

    Optional extension: backends may expose ``read_only_disposable_clone: bool``
    to indicate the backend runs against a disposable read-only checkout (Codex).
    Such backends get over-budget diffs inlined truncated to the inline budget
    and exploration summaries inlined instead of file pointers, and their
    correction-loop rebuilds are framed with the untrusted-content boundary.
    When absent, the caller falls back to False via
    ``getattr(backend, "read_only_disposable_clone", False)``.

    Optional extension: backends may expose ``reasoning_effort``, the per-phase
    reasoning level resolved by ``daydream.runner._resolved_reasoning_effort``
    and applied through the driver's native knob (Claude
    ``ClaudeAgentOptions.effort``, Codex ``-c model_reasoning_effort=``, Pi
    ``--thinking``). All three shipped backends set it; it stays off the
    protocol because each narrows it to its own driver's literal vocabulary.
    Read it via ``getattr(backend, "reasoning_effort", None)``. It is set at
    construction rather than per ``execute`` call because a backend instance is
    already cached per resolved ``(kind, model, reasoning_effort)`` triple, so
    one instance serves exactly one effort level. ``None`` means no source
    supplied one and the driver applies its own ambient default.
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
        persist_session: bool = True,
    ) -> AgentEventStream:
        """Yield AgentEvents for *prompt*.

        Args:
            read_only: When True, the backend enforces a non-mutating tool
                profile at the tool layer (Claude via a PreToolUse guard hook)
                so the agent can inspect history but cannot write/edit/delete
                or mutate the working tree. The Codex backend combines its
                ``--sandbox read-only`` with a disposable standalone clone
                whenever *cwd* is a Git worktree root: the subprocess runs in
                a clone that mirrors HEAD, the staged index, and tracked /
                nonignored untracked files with no source remote, and the
                clone is deleted after the subprocess exits. Codex's sandbox
                restricts filesystem writes but not git index/object-store
                operations, so the clone is what makes a commit update only
                the disposable clone's refs and index — never the caller's
                HEAD, staged index, refs, or remotes, which are unreachable
                via any path the subprocess is given (its argv, stdin, env,
                and cwd); a model that independently discovers the source
                path could still write to its refs. Any other *cwd* — one
                outside a Git worktree, or inside a worktree but not at its
                root — uses the read-only sandbox in place. Callers select
                this flag explicitly per call site: the diagnostic subagents
                (setup-investigator, recommendation-verifier), the failure
                summarizer, and the exploration and repository
                reconnaissance specialists (pre_scan, repo_scan, improve
                recon) pass True, while mutating phases keep the False
                default.
            persist_session: When False, request an invocation that leaves no
                resumable backend session. Backends without persisted sessions
                accept and ignore this option.
        """
        ...
    async def cancel(self) -> None:
        """Cancel every active invocation on this backend.

        This backend-wide operation is reserved for process shutdown. Callers
        ending one invocation must close the corresponding AgentEventStream.
        """
        ...


def resolve_fanout_concurrency(env_var: str, default: int) -> int:
    """Read a backend's fan-out hint from *env_var*, falling back to *default*.

    The right value is a property of the endpoint serving the turns, not of the
    backend, which is why it is an environment override rather than a constant.
    A non-integer or non-positive value warns and falls back rather than failing
    the run: a malformed knob should not cost a review.
    """
    raw = os.environ.get(env_var)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s is not a valid integer; using default %d", env_var, default)
        return default
    if value <= 0:
        logger.warning("%s must be positive; using default %d", env_var, default)
        return default
    return value


def effective_fanout_concurrency(workflow_ceiling: int, backend: object) -> int:
    """Combine a positive workflow ceiling with a backend scheduling hint."""
    hint = getattr(backend, "fanout_concurrency", 4)
    if not isinstance(hint, int) or isinstance(hint, bool) or hint <= 0:
        hint = 4
    return min(workflow_ceiling, hint)


def create_backend(
    name: str,
    model: str | None = None,
    *,
    cwd: Path | None = None,
    reasoning_effort: str | None = None,
    osprey_binary: str | None = None,
) -> Backend:
    """Create a backend by name.

    Args:
        name: Backend name ("claude", "codex", "pi", or "osprey").
        model: Optional model override. Claude and Codex apply their built-in
            defaults here. Pi receives ``None`` unchanged so its own configured
            default can win before Pi's GLM fallback is selected.
        cwd: Target workspace used to resolve Pi's configured default model.
        reasoning_effort: Optional reasoning-effort override (one of
            ``daydream.config.REASONING_EFFORT_LEVELS``). Every backend applies
            it through its own native knob: Claude via
            ``ClaudeAgentOptions.effort``, Codex via
            ``-c model_reasoning_effort=...``, Pi via ``--thinking``.

    Returns:
        A Backend instance whose ``.model`` attribute is a non-empty string.

    Raises:
        ValueError: If the backend name is unknown.
    """
    from daydream.config import DEFAULT_CLAUDE_MODEL, DEFAULT_CODEX_MODEL

    if name == "claude":
        from daydream.backends.claude import ClaudeBackend
        return ClaudeBackend(model=model or DEFAULT_CLAUDE_MODEL, reasoning_effort=reasoning_effort)
    if name == "codex":
        from daydream.backends.codex import CodexBackend
        return CodexBackend(model=model or DEFAULT_CODEX_MODEL, reasoning_effort=reasoning_effort)
    if name == "pi":
        from daydream.backends.pi import PiBackend
        return PiBackend(model=model, cwd=cwd, reasoning_effort=reasoning_effort)
    if name == "osprey":
        from daydream.backends.osprey import OspreyBackend
        return OspreyBackend(
            model=model,
            cwd=cwd,
            reasoning_effort=reasoning_effort,
            osprey_binary=osprey_binary,
        )
    raise ValueError(f"Unknown backend: {name!r}. Expected 'claude', 'codex', 'pi', or 'osprey'.")


from daydream.backends.claude import ClaudeBackend, MaxTurnsError  # noqa: E402
from daydream.backends.osprey import OspreyBackend  # noqa: E402
from daydream.backends.pi import PiBackend  # noqa: E402

__all__ = [
    "AgentEvent",
    "AgentEventStream",
    "Backend",
    "ClaudeBackend",
    "ContinuationToken",
    "CostEvent",
    "MaxTurnsError",
    "MetricsEvent",
    "OspreyBackend",
    "PiBackend",
    "ResultEvent",
    "TextEvent",
    "ThinkingEvent",
    "ToolResultEvent",
    "ToolStartEvent",
    "TurnEndEvent",
    "create_backend",
    "effective_fanout_concurrency",
]
