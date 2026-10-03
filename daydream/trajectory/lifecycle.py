"""Identified phase scopes and causal fork dispatch with terminal evidence."""

from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager, nullcontext
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio

from daydream import timeutil, ui
from daydream.redaction import redact_value as redact_value
from daydream.trajectory.context import get_current_recorder
from daydream.trajectory.types import (
    DaydreamPhase as DaydreamPhase,
    LifecycleReasonCode as LifecycleReasonCode,
    LifecycleStatus as LifecycleStatus,
)

if TYPE_CHECKING:
    from daydream.trajectory.recorder import TrajectoryRecorder
_console = ui.create_console()


def maybe_fork(
    recorder: "TrajectoryRecorder | None",
    descriptor: str,
    *,
    dispatch: "DispatchHandle | None" = None,
) -> AbstractAsyncContextManager[Any]:
    """Return a fork CM if *recorder* is set, otherwise a no-op context manager."""
    if recorder is not None:
        return recorder.fork(descriptor, dispatch=dispatch)
    return nullcontext()


@dataclass
class PhaseEvent:
    """Identified phase start/terminal evidence, paired by session_id and scope_id with optional stage metadata."""

    phase: DaydreamPhase = field(metadata={"enum": True})
    event: str
    timestamp: str
    session_id: str | None = None
    scope_id: str | None = None
    status: LifecycleStatus | None = field(default=None, metadata={"enum": True})
    reason_code: LifecycleReasonCode | None = field(default=None, metadata={"enum": True})
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a fail-closed, JSON-serializable representation."""
        d: dict[str, Any] = {}
        for member in fields(PhaseEvent):
            if member.name == "metadata":
                continue
            value = getattr(self, member.name)
            if value is not None or member.default is not None:
                d[member.name] = value.value if member.metadata.get("enum") else value
        if self.metadata:
            try:
                metadata = redact_value(dict(self.metadata))
                if not isinstance(metadata, dict):
                    raise TypeError("phase metadata redaction returned a non-object")
                json.dumps(metadata, allow_nan=False)
                d["metadata"] = metadata
            except Exception:  # noqa: BLE001 - omit metadata rather than leak or mask
                pass
        return d


def _finish_terminal(
    handle: "PhaseScopeHandle | DispatchHandle",
    scope: str,
    status: LifecycleStatus,
    reason_code: LifecycleReasonCode | None,
) -> None:
    """Select one explicit terminal state before a scope closes."""
    if handle._closed:
        raise RuntimeError(f"{scope} scope is closed")
    if handle._decision_made:
        raise RuntimeError(f"{scope} scope terminal decision already made")
    if not isinstance(status, LifecycleStatus):
        raise TypeError(f"{scope} status must be LifecycleStatus")
    if reason_code is not None and not isinstance(reason_code, LifecycleReasonCode):
        raise TypeError(f"{scope} reason_code must be LifecycleReasonCode")
    handle.status = status
    handle.reason_code = reason_code
    handle._decision_made = True


@dataclass
class PhaseScopeHandle:
    """One identified phase occurrence with a single caller terminal decision."""

    scope_id: str
    status: LifecycleStatus = LifecycleStatus.SUCCEEDED
    reason_code: LifecycleReasonCode | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    _decision_made: bool = field(default=False, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def finish(
        self,
        status: LifecycleStatus,
        reason_code: LifecycleReasonCode | None = None,
    ) -> None:
        """Select one explicit terminal state before this scope closes."""
        _finish_terminal(self, "phase", status, reason_code)

    def _close(self) -> None:
        self._closed = True


def _lifecycle_exception_terminal(
    exc: BaseException,
) -> tuple[LifecycleStatus, LifecycleReasonCode]:
    """Classify an escaping body exception without retaining its text."""
    if isinstance(exc, anyio.get_cancelled_exc_class()):
        return LifecycleStatus.CANCELLED, LifecycleReasonCode.CANCELLED
    return LifecycleStatus.FAILED, LifecycleReasonCode.UNCAUGHT_EXCEPTION


def _host_lifecycle_terminal(
    stop_reason: str,
) -> tuple[LifecycleStatus, LifecycleReasonCode | None]:
    """Project one closed host stop reason onto lifecycle status evidence."""
    if stop_reason in {"completed", "passed", "no_ci"}:
        return LifecycleStatus.SUCCEEDED, None
    if stop_reason == "timed_out":
        return LifecycleStatus.TIMED_OUT, LifecycleReasonCode.TIMED_OUT
    if stop_reason in {"cancelled", "interrupted"}:
        return LifecycleStatus.CANCELLED, LifecycleReasonCode.CANCELLED
    return LifecycleStatus.FAILED, LifecycleReasonCode.DOMAIN_FAILURE


def _phase_scope_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Admit only the fixed lifecycle metadata surface used by phase scopes."""
    stage = metadata.get("stage")
    if not isinstance(stage, str) or re.fullmatch(r"[a-z0-9-]{1,64}", stage) is None:
        return {}
    return {"stage": stage}


def _emit_phase_start(
    recorder: "TrajectoryRecorder | None",
    phase: DaydreamPhase,
    scope_id: str,
    safe_metadata: dict[str, Any],
) -> None:
    if recorder is not None:
        recorder._emit_phase_event(
            phase,
            "phase_start",
            session_id=recorder.session_id,
            scope_id=scope_id,
            **safe_metadata,
        )


def _emit_phase_end(
    recorder: "TrajectoryRecorder | None",
    phase: DaydreamPhase,
    scope_id: str,
    safe_metadata: dict[str, Any],
    status: LifecycleStatus,
    reason_code: LifecycleReasonCode | None,
    **extra: Any,
) -> None:
    if recorder is None:
        return
    try:
        recorder._emit_phase_event(
            phase,
            "phase_end",
            session_id=recorder.session_id,
            scope_id=scope_id,
            status=status,
            reason_code=reason_code,
            **safe_metadata,
            **extra,
        )
    except Exception as exc:  # noqa: BLE001 - recording never masks the body
        ui.print_warning(_console, f"Trajectory phase recording failed: {type(exc).__name__}")


@asynccontextmanager
async def phase_scope(phase: DaydreamPhase, **metadata: Any) -> AsyncIterator[PhaseScopeHandle]:
    """Bracket a phase with identified timing events; no-op without an active recorder."""
    recorder = get_current_recorder()
    safe_metadata = _phase_scope_metadata(metadata)
    handle = PhaseScopeHandle(scope_id=recorder._next_phase_scope_id() if recorder is not None else "")
    _emit_phase_start(recorder, phase, handle.scope_id, safe_metadata)
    try:
        yield handle
    except BaseException as exc:
        handle.status, handle.reason_code = _lifecycle_exception_terminal(exc)
        raise
    finally:
        handle._close()
        _emit_phase_end(
            recorder, phase, handle.scope_id, safe_metadata, handle.status, handle.reason_code, **handle.extra
        )


@dataclass
class HostPhaseHandle:
    """Host stop reason; defaults to completed and may be replaced by the body."""

    stop_reason: str = "completed"


@asynccontextmanager
async def host_phase_scope(phase: DaydreamPhase, **metadata: Any) -> AsyncIterator[HostPhaseHandle]:
    """Bracket host work with duration and stop reason; escaping exceptions record failure."""
    handle = HostPhaseHandle()
    started = time.monotonic()
    async with phase_scope(phase, **metadata) as lifecycle:
        try:
            yield handle
        except BaseException:
            handle.stop_reason = "failed"
            raise
        finally:
            # phase_scope's own exception handler runs after this and wins the
            # terminal projection for an escaping body (FAILED/CANCELLED).
            if lifecycle.status is LifecycleStatus.SUCCEEDED:
                lifecycle.status, lifecycle.reason_code = _host_lifecycle_terminal(handle.stop_reason)
            lifecycle.extra = {
                "duration_ms": max(0, round((time.monotonic() - started) * 1000)),
                "stop_reason": handle.stop_reason,
            }


@dataclass(frozen=True)
class ForkIdentity:
    """Unique identity for one attempted fork within one dispatch."""

    fork_id: str
    descriptor: str
    ordinal: int


@dataclass(frozen=True)
class _CompletedFork:
    identity: ForkIdentity
    path: Path
    trajectory_id: str


@dataclass
class DispatchHandle:
    """Own fork attempts and terminal evidence for one causal fan-out."""

    dispatch_id: str
    phase: DaydreamPhase
    descriptors: tuple[str, ...]
    _recorder: "TrajectoryRecorder" = field(repr=False)
    _started_at: str = field(repr=False)
    status: LifecycleStatus = LifecycleStatus.SUCCEEDED
    reason_code: LifecycleReasonCode | None = None
    _decision_made: bool = field(default=False, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)
    _forks: list[ForkIdentity] = field(default_factory=list, init=False, repr=False)
    _primary_index: dict[str, int] = field(default_factory=dict, init=False, repr=False)
    _dynamic_count: int = field(default=0, init=False, repr=False)
    _completed: dict[str, _CompletedFork] = field(default_factory=dict, init=False, repr=False)
    _write_failures: int = field(default=0, init=False, repr=False)
    _completed_at: str = field(default="", init=False, repr=False)

    @property
    def started_at(self) -> str:
        return self._started_at

    @property
    def completed_at(self) -> str:
        return self._completed_at

    @property
    def planned_count(self) -> int:
        return len(self.descriptors) + self._dynamic_count

    @property
    def attempted_count(self) -> int:
        return len(self._forks)

    @property
    def completed_count(self) -> int:
        return len(self._completed)

    def register_fork(self, descriptor: str) -> ForkIdentity:
        """Register one actual attempt and return its unique fork identity."""
        if self._closed:
            raise RuntimeError("dispatch scope is closed")
        ordinal = len(self._forks) + 1
        identity = ForkIdentity(
            fork_id=f"{self.dispatch_id}:fork:{ordinal}",
            descriptor=descriptor,
            ordinal=ordinal,
        )
        assigned = set(self._primary_index.values())
        primary_index = next(
            (
                index
                for index, planned in enumerate(self.descriptors)
                if planned == descriptor and index not in assigned
            ),
            None,
        )
        if primary_index is None:
            self._dynamic_count += 1
        else:
            self._primary_index[identity.fork_id] = primary_index
        self._forks.append(identity)
        return identity

    def finish(
        self,
        status: LifecycleStatus,
        reason_code: LifecycleReasonCode | None = None,
    ) -> None:
        """Select one explicit terminal state before this dispatch closes."""
        _finish_terminal(self, "dispatch", status, reason_code)

    def _record_completed(self, completed: _CompletedFork) -> None:
        if completed.identity.fork_id not in {fork.fork_id for fork in self._forks}:
            raise RuntimeError("fork does not belong to this dispatch")
        self._completed[completed.identity.fork_id] = completed

    def _record_write_failure(self) -> None:
        self._write_failures += 1

    def _override_terminal(
        self,
        status: LifecycleStatus,
        reason_code: LifecycleReasonCode,
    ) -> None:
        """Make an escaping scope terminal authoritative over defaults."""
        self.status, self.reason_code = status, reason_code
        self._decision_made = True

    def _finalize_default(self) -> None:
        if self._decision_made or self._write_failures == 0:
            return
        self.status, self.reason_code = partial_or_failed_terminal(self.completed_count)

    def _ordered_completed(self) -> list[_CompletedFork]:
        def order(item: _CompletedFork) -> tuple[Any, ...]:
            primary = self._primary_index.get(item.identity.fork_id)
            if primary is not None:
                return (0, primary)
            return (1, item.identity.descriptor, item.identity.fork_id)

        return sorted(self._completed.values(), key=order)

    def _close(self, completed_at: str) -> None:
        self._completed_at = completed_at
        self._closed = True


def partial_or_failed_terminal(has_success: object) -> tuple[LifecycleStatus, LifecycleReasonCode]:
    """Derive the PARTIAL/SOME_CHILDREN_FAILED-or-FAILED/ALL_CHILDREN_FAILED terminal pair."""
    if has_success:
        return LifecycleStatus.PARTIAL, LifecycleReasonCode.SOME_CHILDREN_FAILED
    return LifecycleStatus.FAILED, LifecycleReasonCode.ALL_CHILDREN_FAILED


def finish_partial_or_failed(dispatch: DispatchHandle, has_results: object) -> None:
    """Close *dispatch* PARTIAL when some child succeeded, else FAILED."""
    dispatch.finish(*partial_or_failed_terminal(has_results))


@asynccontextmanager
async def dispatch_scope(
    recorder: "TrajectoryRecorder | None",
    *,
    phase: DaydreamPhase,
    descriptors: Sequence[str],
) -> AsyncIterator[DispatchHandle | None]:
    """Bracket one explicit fan-out and materialize its deterministic step."""
    planned = tuple(descriptors)
    if recorder is None or not planned:
        yield None
        return
    handle = DispatchHandle(
        dispatch_id=recorder._next_dispatch_id(),
        phase=phase,
        descriptors=planned,
        _recorder=recorder,
        _started_at=timeutil.now_iso(),
    )
    try:
        yield handle
    except BaseException as exc:
        handle._override_terminal(*_lifecycle_exception_terminal(exc))
        raise
    finally:
        handle._finalize_default()
        handle._close(timeutil.now_iso())
        try:
            recorder._create_dispatch_step(handle)
        except Exception as exc:  # noqa: BLE001 - recording never masks the body
            ui.print_warning(
                _console,
                f"Trajectory dispatch recording failed: {type(exc).__name__}",
            )
