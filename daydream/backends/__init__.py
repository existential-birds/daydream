"""Backend protocol and factory; events, admission, and execution settings have dedicated modules."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

from daydream.backends._events import (
    AgentEvent as AgentEvent,
    AssistantChoicePart as AssistantChoicePart,
    ContinuationToken as ContinuationToken,
    CostEvent as CostEvent,
    DiagnosticEvent as DiagnosticEvent,
    GenerationEndEvent as GenerationEndEvent,
    GenerationStartEvent as GenerationStartEvent,
    MetricsEvent as MetricsEvent,
    ModelUsageTotals as ModelUsageTotals,
    ReasoningChoicePart as ReasoningChoicePart,
    RequestEvent as RequestEvent,
    ResultEvent as ResultEvent,
    TextChoicePart as TextChoicePart,
    TextEvent as TextEvent,
    ThinkingEvent as ThinkingEvent,
    ToolCallChoicePart as ToolCallChoicePart,
    ToolResultEvent as ToolResultEvent,
    ToolStartEvent as ToolStartEvent,
    TurnEndEvent as TurnEndEvent,
    _new_generation_id as _new_generation_id,
)
from daydream.backends._evidence import (
    _DIAG_LIST_MEMBER_UNSAFE as _DIAG_LIST_MEMBER_UNSAFE,
    _DIAG_LIST_OVERFLOW as _DIAG_LIST_OVERFLOW,
    _MAX_MODEL_NAME_CHARS as _MAX_MODEL_NAME_CHARS,
    _MAX_ORDERED_IDENTITY_ENTRIES as _MAX_ORDERED_IDENTITY_ENTRIES,
    _MAX_PROVIDER_NAME_CHARS as _MAX_PROVIDER_NAME_CHARS,
    ClaudeRequestConfig as ClaudeRequestConfig,
    CodexRequestConfig as CodexRequestConfig,
    CostSource as CostSource,
    EffectiveRequestConfig as EffectiveRequestConfig,
    EvidenceDiagnostic as EvidenceDiagnostic,
    EvidenceSource as EvidenceSource,
    JsonValue as JsonValue,
    MeasurementSource as MeasurementSource,
    OspreyRequestConfig as OspreyRequestConfig,
    PiRequestConfig as PiRequestConfig,
    _admit_identity_label as _admit_identity_label,
    _admit_json_value as _admit_json_value,
    _admit_native_unix_ms as _admit_native_unix_ms,
    unix_ms_to_ns as unix_ms_to_ns,
)
from daydream.backends._execution import (
    BackendExecutionInput as BackendExecutionInput,
    RetryPolicy as RetryPolicy,
    _parsed_nonnegative_float as _parsed_nonnegative_float,
    _parsed_nonnegative_int as _parsed_nonnegative_int,
    _parsed_positive_int as _parsed_positive_int,
)

# host-side capability sentinel, not an external contract: Anthropic's hook
# token is spelled "PreToolUse" (see claude.py); do not version this literal.
AUDIT_ROOT_ISOLATION = "claude-pretooluse"
AuditIsolationReason = Literal[
    "unsupported_backend",
    "missing_capability",
    "wrong_capability",
    "wrong_root",
]


class AuditIsolationError(RuntimeError):
    """A backend cannot satisfy the requested improve audit boundary."""

    def __init__(
        self,
        backend_name: str,
        reason: AuditIsolationReason,
        *,
        phase: str | None = None,
    ) -> None:
        super().__init__(f"{backend_name}: {reason}")
        self.backend_name = backend_name
        self.reason = reason
        self.phase = phase


if TYPE_CHECKING:
    from claude_agent_sdk.types import AgentDefinition


class AgentEventStream(AsyncIterator[AgentEvent], Protocol):
    """Closable invocation-owned stream; closing it releases only that invocation, never sibling streams."""

    async def aclose(self) -> None:
        """Close this invocation and release its resources."""
        ...


class Backend(Protocol):
    """Yield normalized events from execute; keep invocation resources independent.

    Optional capabilities (read via getattr):
    - fanout_concurrency: scheduling hint, default 4; capped by the workflow.
    - concise_fix_prompts: suppress verbose fix reasoning, default False.
    - read_only_disposable_clone: inline bounded diffs and exploration context for
      disposable checkouts; correction prompts retain the untrusted-content boundary.
    - audit_root_isolation/audit_root: tool-layer confinement to an exact Improve
      snapshot. Callers require AUDIT_ROOT_ISOLATION and canonical root equality;
      this does not claim an OS sandbox.
    - supports_finalization: permits invocation-local finalization=True with reduced
      reasoning and native tool controls. Codex still requires a host zero-tool guard.
    - supports_tools_disabled: removes tools without lowering reasoning or changing
      the task. Distinct from finalization, read-only mode, and tool-call budgets.
    - reasoning_effort: native level fixed at construction; None defers to the driver.
      Backend instances are cached by kind, model, effort, and audit root.
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
        """Yield invocation-owned events; read_only requests non-mutating tools and persist_session=False
        requests no resume token. Stateless backends may ignore persistence. Codex additionally isolates
        Git worktree roots in disposable clones; see CodexBackend.execute for its path-hiding limitations.
        """
        ...

    async def cancel(self) -> None:
        """Cancel every active invocation at process shutdown; close individual streams for invocation-local cleanup."""
        ...


def resolve_fanout_concurrency(env_var: str, default: int) -> int:
    """Read a positive endpoint concurrency hint; warn and use the default if malformed."""
    return _parsed_positive_int(os.environ, env_var, default)


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
    audit_root: Path | None = None,
    audit_outward_symlinks: frozenset[Path] = frozenset(),
    execution_input: BackendExecutionInput | None = None,
) -> Backend:
    """Construct native model/effort settings; Claude/Codex use defaults and Pi resolves configuration.
    Only Claude supports an exact audit root with lexical outward-symlink restrictions.
    Unknown names raise ValueError.
    """
    from daydream.config import DEFAULT_CLAUDE_MODEL, DEFAULT_CODEX_MODEL

    if name == "claude":
        return ClaudeBackend(
            model=model or DEFAULT_CLAUDE_MODEL,
            reasoning_effort=reasoning_effort,
            audit_root=audit_root,
            audit_outward_symlinks=audit_outward_symlinks,
            execution_input=execution_input,
        )
    if audit_root is not None and name in {"codex", "pi", "osprey"}:
        raise AuditIsolationError(name, "unsupported_backend")
    if name == "codex":
        from daydream.backends.codex import CodexBackend

        return CodexBackend(
            model=model or DEFAULT_CODEX_MODEL,
            reasoning_effort=reasoning_effort,
            execution_input=execution_input,
        )
    if name == "pi":
        return PiBackend(
            model=model,
            cwd=cwd,
            reasoning_effort=reasoning_effort,
            execution_input=execution_input,
        )
    if name == "osprey":
        if execution_input is not None:
            raise ValueError("explicit BackendExecutionInput is not supported for osprey")
        return OspreyBackend(OspreyConfig(
            model=model,
            reasoning_effort=reasoning_effort,
            osprey_binary=osprey_binary or "",
        ))
    raise ValueError(f"Unknown backend: {name!r}. Expected 'claude', 'codex', 'pi', or 'osprey'.")


from daydream.backends.claude import ClaudeBackend, MaxTurnsError  # noqa: E402
from daydream.backends.osprey import OspreyBackend, OspreyConfig  # noqa: E402
from daydream.backends.pi import PiBackend  # noqa: E402

__all__ = [
    "AUDIT_ROOT_ISOLATION",
    "AgentEvent",
    "AgentEventStream",
    "AuditIsolationError",
    "Backend",
    "BackendExecutionInput",
    "ClaudeBackend",
    "ClaudeRequestConfig",
    "CodexRequestConfig",
    "ContinuationToken",
    "CostEvent",
    "DiagnosticEvent",
    "GenerationEndEvent",
    "GenerationStartEvent",
    "JsonValue",
    "MaxTurnsError",
    "MetricsEvent",
    "ModelUsageTotals",
    "OspreyBackend",
    "OspreyConfig",
    "OspreyRequestConfig",
    "PiBackend",
    "PiRequestConfig",
    "ReasoningChoicePart",
    "RequestEvent",
    "RetryPolicy",
    "ResultEvent",
    "TextChoicePart",
    "TextEvent",
    "ThinkingEvent",
    "ToolCallChoicePart",
    "ToolResultEvent",
    "ToolStartEvent",
    "TurnEndEvent",
    "create_backend",
    "effective_fanout_concurrency",
    "unix_ms_to_ns",
]
