"""Flow engine: ordered, gated execution of registered flow steps.

``run_flow`` executes a named flow from the :class:`Registry` over a shared
:class:`FlowContext`. Steps communicate through ``ctx.data`` and signal
control flow by returning :class:`Stop` (end the flow with an exit code) or
:class:`BreakLoop` (end the enclosing :class:`LoopGroup`).

Step exceptions propagate unchanged — error handling lives inside step
bodies, exactly where the flow helpers' try/excepts sit today.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from daydream.extensions.api import (
    BreakLoop,
    FlowStep,
    LoopGroup,
    Stop,
    UnresolvedExtensionError,
)

if TYPE_CHECKING:
    from daydream.backends import Backend
    from daydream.extensions.registry import FlowEntry, Registry
    from daydream.runner import RunConfig
    from daydream.workspace import WorkContext


@dataclass
class FlowContext:
    """Shared state a flow's steps read and write.

    Attributes:
        config: The run configuration (backend/model precedence sources).
        work: The resolved workspace for the run.
        registry: The per-run extension registry the flow resolves against.
        data: Cross-step scratch state; steps replace locals with keys here.
        wall_budget_s: Per-agent wall-clock limit for this flow.
        tool_call_budget: Per-agent tool-call limit for this flow.
    """

    config: RunConfig
    work: WorkContext
    registry: Registry
    data: dict[str, Any] = field(default_factory=dict)
    wall_budget_s: float | None = None
    tool_call_budget: int | None = None
    _backend_cache: dict[tuple[str, str | None, str | None], Backend] = field(
        default_factory=dict, repr=False
    )

    def backend_for(self, phase: str) -> Backend:
        """Get or create the backend for ``phase``, reusing per-context instances.

        Instance-sharing semantics are identical to the flow helpers'
        ``backend_cache`` dicts today: backends are cached per resolved
        ``(backend_name, model, reasoning_effort)`` triple for the lifetime of
        this context.
        """
        from daydream.runner import _resolve_backend

        return _resolve_backend(
            self.config, phase, cache=self._backend_cache, cwd=self.work.repo
        )


def _resolve_steps(registry: Registry, flow_name: str, entries: list[FlowEntry]) -> dict[str, FlowStep]:
    """Pre-flight resolve pass: every entry (including loop-group bodies) must be a registered phase."""
    steps: dict[str, FlowStep] = {}
    for entry in entries:
        names = (entry,) if isinstance(entry, str) else entry.steps
        for name in names:
            try:
                steps[name] = registry.phase(name)
            except UnresolvedExtensionError:
                raise UnresolvedExtensionError(
                    f"flow '{flow_name}' references step '{name}', which is not a registered phase; "
                    "run 'daydream ext validate' to check the extension registry"
                ) from None
    return steps


async def _run_step(step: FlowStep, ctx: FlowContext) -> Stop | BreakLoop | None:
    """Run one step unless its ``enabled`` predicate gates it off."""
    if step.enabled is not None and not step.enabled(ctx):
        return None
    return await step.run(ctx)


async def _run_group(group: LoopGroup, steps: dict[str, FlowStep], ctx: FlowContext) -> Stop | None:
    """Run a loop group's body up to ``max_iterations(ctx)`` passes.

    Sets ``ctx.data["iteration"]`` (1-based) each pass. ``BreakLoop`` from a
    body step ends the group; ``Stop`` ends the whole flow.
    """
    for iteration in range(1, group.max_iterations(ctx) + 1):
        ctx.data["iteration"] = iteration
        for name in group.steps:
            signal = await _run_step(steps[name], ctx)
            if isinstance(signal, Stop):
                return signal
            if isinstance(signal, BreakLoop):
                return None
    return None


async def run_flow(registry: Registry, flow_name: str, ctx: FlowContext) -> int:
    """Execute the named flow's entries in order; return the flow's exit code.

    Resolves every entry via ``registry.phase()`` FIRST, so an unresolvable
    flow raises :class:`UnresolvedExtensionError` naming flow + step before
    any step executes. A ``Stop(code)`` signal returns ``code`` immediately;
    falling off the end returns 0. A ``BreakLoop`` outside a loop group is
    ignored.
    """
    entries = registry.flow(flow_name)
    steps = _resolve_steps(registry, flow_name, entries)
    for entry in entries:
        if isinstance(entry, str):
            signal = await _run_step(steps[entry], ctx)
        else:
            signal = await _run_group(entry, steps, ctx)
        if isinstance(signal, Stop):
            return signal.exit_code
    return 0
