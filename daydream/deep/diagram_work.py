"""Structured ownership of optional diagrams across the deep review flow.

Custom flows retain the complete serial ``diagram`` step. The packaged review
step can launch generation early only when its final join is still present.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import anyio

from daydream.agent_capacity import AgentCapacity, agent_capacity_scope
from daydream.backends import effective_fanout_concurrency
from daydream.deep.latency import (
    FAIL_SAFE_LATENCY_PROFILE,
    PROFILE_ROUTES,
    diff_signals,
    summarize_risk,
    wonder_decision,
)
from daydream.deep.settings import _resolve_opt_in, fresh_ttt
from daydream.deep.state import DeepState

if TYPE_CHECKING:
    from anyio.abc import TaskGroup

    from daydream.flows.engine import FlowContext


@dataclass
class DiagramWork:
    """One generation task; join before attachment and cancel before finalization."""

    group: TaskGroup
    capacity: AgentCapacity
    started: bool = False
    done: anyio.Event = field(default_factory=anyio.Event)
    error: Exception | None = None

    def start(self, ctx: FlowContext, *, primary_count: int) -> None:
        """Reserve every primary review slot, otherwise defer to the final step."""
        spare = min(2, self.capacity.capacity - primary_count)
        if self.started or spare <= 0:
            return
        self.started = True

        async def generate() -> None:
            from daydream.deep.diagram_steps import _generate_diagrams

            try:
                with agent_capacity_scope(self.capacity, optional=True):
                    await _generate_diagrams(ctx, concurrency=spare)
            except Exception as exc:  # deferred to the owning diagram step
                self.error = exc
            finally:
                self.done.set()

        self.group.start_soon(generate)

    async def join(self) -> None:
        """Observe completion and retain the original generation failure."""
        await self.done.wait()
        if self.error is not None:
            raise self.error


def diagram_work_for(ctx: FlowContext) -> DiagramWork | None:
    """Return ephemeral run ownership; never persist task handles in artifacts."""
    value = ctx.data.get("_diagram_work")
    return value if isinstance(value, DiagramWork) else None


@asynccontextmanager
async def diagram_work_scope(ctx: FlowContext) -> AsyncIterator[None]:
    """Own child lifetime only for compatible built-in review registrations."""
    from daydream.deep.merge_steps import _supervisor_mode
    from daydream.deep.orchestrator import STEPS, _diagram_enabled, _multi_stack_merge_enabled
    from daydream.deep.review_steps import _fold_default_alternatives

    entries = ctx.registry.flow("deep")
    prefix = list(STEPS[:next(index for index, step in enumerate(STEPS) if step.name == "diagram") + 1])
    compatible = (
        _diagram_enabled(ctx)
        and ctx.config.start_at not in ("merge", "fix")
        and entries[:len(prefix)] == [step.name for step in prefix]
        and all(ctx.registry.phase(step.name) == step for step in prefix)
    )
    if not compatible:
        yield
        return
    # Resolve only phases this review policy can use. Disabled phase overrides
    # must not require an installed/configurable provider merely to admit work.
    state = DeepState(ctx.data)
    phases = ["per_stack_review", "diagram"]
    if fresh_ttt(ctx.config) and not state.single_stack_mode:
        route = state.latency_route or PROFILE_ROUTES[FAIL_SAFE_LATENCY_PROFILE]
        summary = state.risk_summary or summarize_risk(diff_signals(diff="", changed_files=0, stack_count=0))
        if wonder_decision(route, summary, folded=_fold_default_alternatives(ctx), tier=state.tier).outcome == "run":
            phases.append("wonder")
    if _multi_stack_merge_enabled(ctx):
        phases.append("merge")
        if ctx.pipeline().arbitration.enabled:
            phases.append("arbiter")
            if ctx.pipeline().suppression.enabled or _resolve_opt_in(ctx.config, "precision_mode"):
                phases.append("suppression")
    if _supervisor_mode(ctx.config) == "llm":
        phases.append("supervise")
    # Different configured models may share a provider. Keep one conservative
    # pool until the final diagram step joins generation and releases admission.
    capacity = AgentCapacity(min(effective_fanout_concurrency(10, ctx.backend_for(phase)) for phase in phases))
    error: BaseException | None = None
    async with anyio.create_task_group() as group:
        ctx.data["_diagram_work"] = DiagramWork(group, capacity)
        try:
            with agent_capacity_scope(capacity):
                yield
        except BaseException as exc:
            error = exc
        finally:
            group.cancel_scope.cancel()
            ctx.data.pop("_diagram_work", None)
    if error is not None:
        raise error
