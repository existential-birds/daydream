"""Bounded sibling tasks with trajectory forks and structured cancellation."""

from collections.abc import Awaitable, Callable, Iterable
from contextlib import nullcontext

import anyio

from daydream.trajectory import DispatchHandle, TrajectoryRecorder, maybe_fork


async def run_fanout[T, R](
    items: Iterable[T],
    operation: Callable[[T], Awaitable[R]],
    *,
    limiter: anyio.CapacityLimiter,
    recorder: TrajectoryRecorder | None = None,
    descriptor: Callable[[T], str] | None = None,
    dispatch: DispatchHandle | None = None,
    completed: Callable[[T, R], None] | None = None,
    failed: Callable[[T, Exception], None] | None = None,
) -> None:
    """Join siblings; uncaught failures cancel siblings and cancellation always propagates.

    Capacity precedes the optional fork, keeping queued children inactive. An
    explicit failure callback isolates ordinary operation/fork errors. Completion
    runs after the fork closes, allowing results to land while siblings continue.
    Callers own result ordering and dispatch completion.
    """
    async def invoke(item: T) -> None:
        try:
            async with limiter:
                fork = maybe_fork(recorder, descriptor(item), dispatch=dispatch) if descriptor else nullcontext()
                async with fork:
                    result = await operation(item)
        except Exception as exc:
            if failed is None:
                raise
            failed(item, exc)
            return
        if completed is not None:
            completed(item, result)

    async with anyio.create_task_group() as tasks:
        for item in items:
            tasks.start_soon(invoke, item)
