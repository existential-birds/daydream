"""Ordered, gated registry flows over shared state.

Stop ends the flow; BreakLoop ends its enclosing group. Step exceptions propagate
unchanged, leaving error policy with each step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from daydream.extensions.api import (
    BreakLoop,
    FlowStep,
    LoopGroup as LoopGroup,
    Stop,
    UnresolvedExtensionError,
)
from daydream.github_app import GitHubExecutionInput
from daydream.observability.spans import step_scope
from daydream.run_context import bind_run_context, resolve_run_context

if TYPE_CHECKING:
    from daydream.artifact_visibility import ArtifactSession, PrivateWorkspaceOwner
    from daydream.backends import Backend, BackendExecutionInput
    from daydream.deep.state import DeepData
    from daydream.extensions.registry import FlowEntry, Registry
    from daydream.review_profile import Pipeline, ResolvedProfile
    from daydream.run_config import RunConfig
    from daydream.run_context import RunContext
    from daydream.workspace import AuditWorkspace, WorkContext


type BackendCache = dict[tuple[str, str | None, str | None, Path | None], Backend]


@dataclass
class FlowContext:
    """Run-owned dependencies and shared mutable step state.

    Unresolved or partial review profiles use packaged strategy/pipeline defaults.
    Direct callers may omit run_context to inherit the bound runtime or standalone
    policy. Backend instances share this context's cache and captured execution
    input; effort overrides retain the same workspace and audit boundary.

    Backend and GitHub execution are explicit private capabilities; standalone
    callers inherit ambient credentials. Never copy them into data or
    artifacts. Standalone artifact access requires explicit opt-in; runner contexts
    instead supply their owned artifact session.
    """

    config: RunConfig
    work: WorkContext
    registry: Registry
    data: dict[str, Any] = field(default_factory=dict)
    review_profile: ResolvedProfile | None = None
    audit_workspace: AuditWorkspace | None = None
    private_workspace_owner: PrivateWorkspaceOwner | None = None
    artifacts: ArtifactSession | None = None
    allow_standalone_artifacts: bool = field(default=False, kw_only=True)
    run_context: RunContext | None = field(default=None, kw_only=True)
    github_execution: GitHubExecutionInput = field(
        default_factory=GitHubExecutionInput, repr=False, compare=False, kw_only=True,
    )
    backend_execution: BackendExecutionInput | None = field(
        default=None, repr=False, compare=False, kw_only=True,
    )
    _backend_cache: BackendCache = field(
        default_factory=dict, repr=False
    )

    def deep_data(self) -> DeepData:
        """Admit the documented extension inputs and return this flow's exact live state."""
        from daydream.deep.state import validate_extension_inputs

        validate_extension_inputs(self.data)
        return cast("DeepData", self.data)

    def _backend(self, phase: str, *, effort: str | None = None) -> Backend:
        """Resolve ``phase``'s backend using this context's captured execution input."""
        from daydream.runner import _resolve_backend

        return _resolve_backend(
            self.config,
            phase,
            cache=self._backend_cache,
            cwd=self.work.repo,
            audit_workspace=self.audit_workspace,
            effort_override=effort,
            execution_input=self.backend_execution,
        )

    def backend_for(self, phase: str) -> Backend:
        """Cache instances by resolved backend, model, effort, and audit root for this context."""
        return self._backend(phase)

    def backend_for_effort(self, phase: str, effort: str) -> Backend:
        """Resolve an effort-specific backend, preserving any runner-bound factory seam."""
        return self._backend(phase, effort=effort)

    def strategy(self, stage: str) -> str:
        """Read a resolved profile strategy, falling back per stage to packaged defaults."""
        from daydream.review_profile import build_default_profile

        # Fall back per-stage on missing keys: ``parse_profile`` accepts partial
        # profiles, so a resolved profile may omit a given stage's strategy.
        # Rather than raising KeyError for a valid partial profile, use the
        # packaged default's strategy for that stage so steps stay operable.
        if self.review_profile is not None:
            strategies = self.review_profile.profile.strategies
            if stage in strategies:
                return strategies[stage].content
        return build_default_profile().strategies[stage].content

    def pipeline(self) -> Pipeline:
        """Read the resolved bounded pipeline or use the packaged default."""
        from daydream.review_profile import resolve_pipeline

        return resolve_pipeline(self.review_profile)


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
    with step_scope(
        step.name, phase=step.phase_key, iteration=ctx.data.get("iteration"), stack=ctx.config.stack,
    ) as observed:
        result = await step.run(ctx)
        observed.finish(result.exit_code if isinstance(result, Stop) else 0)
        return result


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
    """Resolve every entry before execution, then return the first Stop code or zero.

    Unknown entries name the flow and phase in UnresolvedExtensionError. BreakLoop
    outside a group is ignored; the resolved runtime stays bound across steps.
    """
    runtime = resolve_run_context(ctx.run_context)
    ctx.run_context = runtime
    with bind_run_context(runtime):
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
