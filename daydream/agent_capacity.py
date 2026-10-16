"""Run-local admission when optional work overlaps primary model phases."""

from __future__ import annotations

from collections.abc import Callable, Coroutine, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from typing import Any, ParamSpec, TypeVar

import anyio

_P = ParamSpec("_P")
_T = TypeVar("_T")


class AgentCapacity:
    """Bound overlapping calls, admitting queued primary work before optional work.

    The caller additionally reserves the entire primary review fan-out before
    launching optional tasks. This admission remains active through synthesis so
    later phases and diagram repairs share the same provider capacity.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.enabled = True
        self._active = 0
        self._waiting_primary = 0
        self._condition = anyio.Condition()

    async def acquire(self, *, optional: bool) -> None:
        async with self._condition:
            if not optional:
                self._waiting_primary += 1
            try:
                while self._active >= self.capacity or optional and self._waiting_primary:
                    await self._condition.wait()
                self._active += 1
            finally:
                if not optional:
                    self._waiting_primary -= 1
                self._condition.notify_all()

    async def release(self) -> None:
        with anyio.CancelScope(shield=True):
            async with self._condition:
                self._active -= 1
                self._condition.notify_all()


_capacity: ContextVar[AgentCapacity | None] = ContextVar("agent_capacity", default=None)
_optional: ContextVar[bool] = ContextVar("optional_agent", default=False)


@contextmanager
def agent_capacity_scope(capacity: AgentCapacity, *, optional: bool = False) -> Iterator[None]:
    """Bind admission to this task and its children, restoring it on every exit."""
    capacity_token = _capacity.set(capacity)
    optional_token = _optional.set(optional)
    try:
        yield
    finally:
        _optional.reset(optional_token)
        _capacity.reset(capacity_token)


def admit_agent(function: Callable[_P, Coroutine[Any, Any, _T]]) -> Callable[_P, Coroutine[Any, Any, _T]]:
    """Admit an entire logical invocation, including its retries and finalization."""
    @wraps(function)
    async def admitted(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        capacity = _capacity.get()
        if capacity is None or not capacity.enabled:
            return await function(*args, **kwargs)
        await capacity.acquire(optional=_optional.get())
        try:
            return await function(*args, **kwargs)
        finally:
            await capacity.release()

    return admitted
