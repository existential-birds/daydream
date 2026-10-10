"""Versioned extension types: flows, control signals, stack rules and errors.

Never import runner or phases here; FlowContext is type-only to avoid cycles."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from daydream.flows.engine import FlowContext
    from daydream.trajectory import DaydreamPhase

# Current/preferred extension contract version; doubles as the max accepted (range ceiling).
EXTENSION_API_VERSION: int = 8
# Oldest extension contract version still accepted (range floor).
MIN_SUPPORTED_EXTENSION_API_VERSION: int = 8


class ExtensionError(Exception):
    """Base error for the extension seam."""


class ExtensionVersionError(ExtensionError):
    """The extension module's ``DAYDREAM_EXT_API`` is absent or incompatible."""


class UnresolvedExtensionError(ExtensionError):
    """A registry lookup or flow entry names a piece that is not registered."""


@dataclass(frozen=True)
class ToolDecision:
    """Decision returned by a tool supervisor."""

    veto: bool
    reason: str = ""

    def __post_init__(self) -> None:
        if self.veto and not self.reason.strip():
            raise ValueError("veto decisions require a non-blank reason")


class ToolSupervisor(Protocol):
    """Synchronous callable that can veto a tool invocation."""

    def __call__(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        *,
        phase: DaydreamPhase,
    ) -> ToolDecision:
        ...


@dataclass(frozen=True)
class Stop:
    """Signal: end the whole flow immediately with ``exit_code``."""

    exit_code: int


@dataclass(frozen=True)
class BreakLoop:
    """Signal: end the enclosing loop group; the flow continues after it."""


@dataclass(frozen=True)
class FlowStep:
    """A named async step returning a control signal or None.

    config_phase selects its file-config key, defaulting to name; enabled is a
    per-run predicate that skips the step when false."""

    name: str
    run: Callable[[FlowContext], Awaitable[Stop | BreakLoop | None]]
    config_phase: str | None = None
    enabled: Callable[[FlowContext], bool] | None = None

    @property
    def phase_key(self) -> str:
        """Per-phase config key: ``config_phase`` when set, else ``name``."""
        return self.config_phase if self.config_phase is not None else self.name


@dataclass(frozen=True)
class LoopGroup:
    """An ordered group of step names repeated up to ``max_iterations(ctx)`` passes."""

    name: str
    steps: tuple[str, ...]
    max_iterations: Callable[[FlowContext], int]


@dataclass(frozen=True)
class CommentFinding:
    """Public ParsedIssue fields passed to a "finding" renderer."""

    path: str
    line: int | None
    title: str
    body: str
    is_cross_stack: bool
    severity: str | None
    confidence: str | None
    fingerprint: str | None


@dataclass(frozen=True)
class FindingRenderContext:
    """Finding placement: inline, file_level, or summary."""

    placement: str


@dataclass(frozen=True)
class SummaryFinding:
    """Summary entry whose host-rendered body_block already contains its finding marker."""

    finding: CommentFinding
    body_block: str


@dataclass(frozen=True)
class SummaryContext:
    """Summary-renderer input with optional host-grounded diagram blocks.
    The last, optional diagrams field preserves existing positional constructors."""

    findings: tuple[SummaryFinding, ...]
    agent_prompt: str
    review_info: str
    diagrams: str | None = None


@dataclass(frozen=True)
class StackRule:
    """A fork-registered stack: changed-file globs routed to a named stack."""

    stack_name: str
    patterns: tuple[str, ...]
