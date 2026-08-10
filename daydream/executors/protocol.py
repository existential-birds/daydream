"""The ``ReviewExecutor`` port (DAYDREAM_SERVICE_V1).

A Fortress adapter is anything implementing this protocol: it owns the compute
/ workspace lifecycle behind an opaque ``ExecutionRef`` and emits neutral
``ExecutionSnapshot`` / ``ArtifactEnvelope`` values. The controller and the
conformance suite depend only on this surface; vendor lifecycle is private to
the adapter.

This protocol intentionally mirrors the Backend seam's shape (async port
methods returning typed values) but is a separate concern: ``Backend`` drives
model-agent turns, ``ReviewExecutor`` owns execution/workspace lifecycle.
"""

from __future__ import annotations

from typing import Protocol

from daydream.executors.contract import (
    ArtifactEnvelope,
    ExecutionRef,
    ExecutionSnapshot,
    ExecutorCapability,
    ExecutorJob,
)


class ReviewExecutor(Protocol):
    """Neutral executor port: start/inspect/cancel/collect/release."""

    kind: str
    adapter_version: int
    capabilities: frozenset[ExecutorCapability]

    async def start(self, job: ExecutorJob) -> ExecutionRef:
        """Begin an execution for *job* and return its opaque reference.

        Must be idempotent for a repeat ``job`` on the same adapter: a second
        ``start`` with the same identity returns a reference bound to the same
        execution rather than launching a duplicate.
        """
        ...

    async def inspect(self, ref: ExecutionRef) -> ExecutionSnapshot:
        """Return the current lifecycle snapshot for *ref*."""
        ...

    async def cancel(self, ref: ExecutionRef) -> None:
        """Strongly cancel *ref*; subsequent inspect reports ``cancelled``."""
        ...

    async def collect(self, ref: ExecutionRef) -> ArtifactEnvelope:
        """Collect the bounded artifacts for *ref*.

        May be called only on a terminal execution (evaluated/cancelled/
        infra_error); a non-terminal collect is an implementation error.
        """
        ...

    async def release(self, ref: ExecutionRef, disposition: str) -> None:
        """Deterministically release *ref* with *disposition* (e.g. ``complete``).

        After release, ``inspect`` reports ``released`` (or raises
        ``UnknownExecutionError``); the execution's resources are gone.
        """
        ...


def is_review_executor(obj: object) -> bool:
    """Return True when *obj* looks like a conformant ReviewExecutor.

    Cheap structural check used by the registry seam; does not import or
    instantiate the adapter.
    """
    return all(hasattr(obj, name) for name in ("start", "inspect", "cancel", "collect", "release", "kind"))


class ExecutorAdmissionError(Exception):
    """A registered executor failed capability admission (wrapper)."""


# Re-exported for convenience so adapters and tests import one symbol set.
__all__ = [
    "ArtifactEnvelope",
    "ExecutorAdmissionError",
    "ExecutionRef",
    "ExecutionSnapshot",
    "ExecutorCapability",
    "ExecutorJob",
    "ReviewExecutor",
    "is_review_executor",
]
