"""Bounded host observations for one Pi attempt, independent of streamed output.

The runner owns the absolute deadline and freezes this collector synchronously
before recorder/span finalization. No native payload or identifier is accepted.
Timings are seconds since dispatch on Daydream's monotonic clock, not provider
timestamps. Observing activity never changes timeout or retry decisions.
"""

from __future__ import annotations

import math
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Literal

from daydream import clock

_MAX_COUNT = 2_147_483_647
LifecycleState = Literal["preparation", "prepared", "spawn_pending", "launched", "native_event", "generating"]
ProgressKind = Literal["text", "thinking", "toolcall", "tool"]


@dataclass(frozen=True)
class BackendLifecycleSnapshot:
    """Closed, content-free attempt evidence; absent observations remain None."""

    attempt: int
    backend: Literal["pi"] = "pi"
    timestamp_source: Literal["host_observed"] = "host_observed"
    clock: Literal["monotonic_elapsed"] = "monotonic_elapsed"
    state: LifecycleState = "preparation"
    response_idle_timeout_s: float | None = None
    tool_idle_timeout_s: float | None = None
    idle_detection_enabled: bool = False
    wall_allowance_s: float | None = None
    remaining_wall_s: float | None = None
    elapsed_s: float = 0.0
    request_prepared_s: float | None = None
    spawn_pending_s: float | None = None
    spawn_completed_s: float | None = None
    first_raw_line_s: float | None = None
    first_native_event_s: float | None = None
    generation_started_s: float | None = None
    last_progress_s: float | None = None
    last_progress_age_s: float | None = None
    raw_line_count: int = 0
    native_event_count: int = 0
    generation_count: int = 0
    text_delta_count: int = 0
    thinking_delta_count: int = 0
    toolcall_delta_count: int = 0
    tool_progress_count: int = 0
    active_tool_count: int = 0
    generation_open: bool = False

    def __post_init__(self) -> None:
        allowed_strings = {
            "backend": {"pi"}, "timestamp_source": {"host_observed"}, "clock": {"monotonic_elapsed"},
            "state": {"preparation", "prepared", "spawn_pending", "launched", "native_event", "generating"},
        }
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name in allowed_strings:
                valid = type(value) is str and value in allowed_strings[field.name]
            elif field.name in {"idle_detection_enabled", "generation_open"}:
                valid = type(value) is bool
            elif field.name == "attempt" or field.name.endswith("_count"):
                valid = type(value) is int and (1 if field.name == "attempt" else 0) <= value <= _MAX_COUNT
            else:
                valid = value is None or (type(value) in {int, float} and math.isfinite(value) and value >= 0)
            if not valid:
                raise ValueError("invalid backend lifecycle scalar")

    def to_dict(self) -> dict[str, str | bool | int | float | None]:
        """Serialize only this fixed schema, including missing observations."""
        return asdict(self)


class BackendLifecycleCollector:
    """Accumulate constant-space scalar evidence until an idempotent freeze."""

    def __init__(self, *, attempt: int, started_at: float, wall_allowance_s: float | None,
                 deadline: float | None) -> None:
        self._started_at = started_at
        self._deadline = deadline
        self._snapshot = BackendLifecycleSnapshot(attempt=attempt, wall_allowance_s=wall_allowance_s)
        self._active = False
        self._finalized = False

    def bind(self) -> Token[BackendLifecycleCollector | None]:
        """Bind this attempt only; concurrent invocations inherit separate collectors."""
        return _CURRENT.set(self)

    @staticmethod
    def reset(token: Token[BackendLifecycleCollector | None]) -> None:
        _CURRENT.reset(token)

    def _elapsed(self) -> float:
        return max(0.0, clock.monotonic() - self._started_at)

    def _update(self, **values: Any) -> None:
        # dataclasses.replace has heterogeneous keyword types; the snapshot
        # validates its closed scalar schema before accepting any update.
        if not self._finalized:
            self._snapshot = replace(self._snapshot, **values)

    def _increment(self, name: str) -> int:
        value: int = getattr(self._snapshot, name)
        return min(_MAX_COUNT, value + 1)

    def activate(self, *, response_idle_timeout_s: float | None, tool_idle_timeout_s: float | None) -> None:
        """Record effective settings; zero/None means the idle detector is disabled."""
        self._active = True
        self._update(response_idle_timeout_s=response_idle_timeout_s or 0.0,
                     tool_idle_timeout_s=tool_idle_timeout_s or 0.0,
                     idle_detection_enabled=bool(response_idle_timeout_s))

    def request_prepared(self) -> None:
        self._update(state="prepared", request_prepared_s=self._elapsed())

    def spawn_pending(self) -> None:
        self._update(state="spawn_pending", spawn_pending_s=self._elapsed())

    def spawn_completed(self) -> None:
        self._update(state="launched", spawn_completed_s=self._elapsed())

    def raw_line_received(self) -> None:
        self._update(raw_line_count=self._increment("raw_line_count"),
                     first_raw_line_s=self._snapshot.first_raw_line_s
                     if self._snapshot.first_raw_line_s is not None else self._elapsed())

    def native_event(self) -> None:
        self._update(native_event_count=self._increment("native_event_count"),
                     state="generating" if self._snapshot.generation_open else "native_event",
                     first_native_event_s=self._snapshot.first_native_event_s
                     if self._snapshot.first_native_event_s is not None else self._elapsed())

    def generation_started(self) -> None:
        self._update(state="generating", generation_open=True,
                     generation_count=self._increment("generation_count"),
                     generation_started_s=self._snapshot.generation_started_s
                     if self._snapshot.generation_started_s is not None else self._elapsed())

    def generation_ended(self) -> None:
        # A real assistant message_end is response progress, not a delta count.
        self._update(state="native_event", generation_open=False, last_progress_s=self._elapsed())

    def progress(self, kind: ProgressKind) -> None:
        field = {"text": "text_delta_count", "thinking": "thinking_delta_count",
                 "toolcall": "toolcall_delta_count", "tool": "tool_progress_count"}[kind]
        self._update(**{field: self._increment(field), "last_progress_s": self._elapsed()})

    def active_tools(self, count: int) -> None:
        bounded_count = min(_MAX_COUNT, max(0, count))
        if bounded_count != self._snapshot.active_tool_count:
            self._update(active_tool_count=bounded_count, last_progress_s=self._elapsed())

    def finalize(self) -> BackendLifecycleSnapshot | None:
        """Freeze without awaiting or yielding, including cancellation/GeneratorExit."""
        if not self._active:
            return None
        if not self._finalized:
            elapsed = self._elapsed()
            self._update(elapsed_s=elapsed,
                         remaining_wall_s=max(0.0, self._deadline - clock.monotonic())
                         if self._deadline is not None else None,
                         last_progress_age_s=max(0.0, elapsed - self._snapshot.last_progress_s)
                         if self._snapshot.last_progress_s is not None else None)
            self._finalized = True
        return self._snapshot


_CURRENT: ContextVar[BackendLifecycleCollector | None] = ContextVar("backend_lifecycle", default=None)


def current_backend_lifecycle() -> BackendLifecycleCollector | None:
    """Return the runner-owned collector, if executing inside an attempt."""
    return _CURRENT.get()
