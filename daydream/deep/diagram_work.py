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
    from daydream.deep.orchestrator import STEPS, _diagram_enabled

    entries = ctx.registry.flow("deep")
    prefix = list(STEPS[:next(index for index, step in enumerate(STEPS) if step.name == "diagram") + 1])
    compatible = (
        _diagram_enabled(ctx)
        and entries[:len(prefix)] == [step.name for step in prefix]
        and all(ctx.registry.phase(step.name) == step for step in prefix)
    )
    if not compatible:
        yield
        return
    # Different configured models may still share one provider. Use the most
    # conservative hint and one admission pool throughout review and synthesis.
    capacity = AgentCapacity(min(effective_fanout_concurrency(10, ctx.backend_for(phase)) for phase in (
        "per_stack_review", "wonder", "diagram", "arbiter", "suppression", "merge", "supervise",
    )))
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
