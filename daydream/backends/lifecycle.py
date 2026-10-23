"""Bounded host observations for one CLI attempt, independent of streamed output.

The runner owns the absolute deadline and freezes this collector synchronously
before recorder/span finalization. No native payload or identifier is accepted.
Timings are seconds since dispatch on Daydream's monotonic clock, not provider
timestamps. Observing activity never changes timeout or retry decisions.
"""

from __future__ import annotations

import asyncio
import math
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Literal

import anyio

from daydream import clock
from daydream.diagnostics import sanitize_verbose_message

_MAX_COUNT = 2_147_483_647
LifecycleState = Literal["preparation", "prepared", "spawn_pending", "launched", "native_event", "generating"]
LifecycleStage = Literal["preterminal_stream", "terminal_drain", "process_reap", "cleanup"]
CodexPublicEventKind = Literal[
    "thread_started", "turn_started", "item_started", "item_updated", "item_completed",
    "turn_completed", "turn_failed", "error", "unknown",
]
ProgressKind = Literal["text", "thinking", "toolcall", "tool"]


@dataclass(frozen=True)
class BackendLifecycleFailure:
    """Closed host facts eligible for a public review warning; never provider prose."""

    stage: LifecycleStage
    allowance_s: float
    terminal_received: bool
    stdout_eof: bool
    process_returncode: int | None


def lifecycle_failure_diagnostic(failure: object) -> str | None:
    """Admit exact host records/scalars only; never dispatch a supplied formatter."""
    try:
        if type(failure) is not BackendLifecycleFailure:
            return None
        if (type(failure.stage) is not str or failure.stage not in {"terminal_drain", "process_reap", "cleanup"}
                or type(failure.allowance_s) not in {int, float}
                or not math.isfinite(failure.allowance_s) or failure.allowance_s <= 0
                or type(failure.terminal_received) is not bool or type(failure.stdout_eof) is not bool
                or (failure.process_returncode is not None
                    and (type(failure.process_returncode) is not int or abs(failure.process_returncode) > _MAX_COUNT))):
            return None
        milestones = ["turn.completed received" if failure.terminal_received else "turn.completed not observed",
                      "stdout EOF observed" if failure.stdout_eof else "stdout EOF not observed",
                      "child exit not observed" if failure.process_returncode is None
                      else f"child exit {failure.process_returncode} observed"]
        unmet = {"terminal_drain": "stdout EOF was not observed",
                 "process_reap": "actual child exit was not observed",
                 "cleanup": "process-group cleanup did not complete"}[failure.stage]
        text = (f"Codex {failure.stage} timeout after {failure.allowance_s:g}s: {unmet}; "
                f"{', '.join(milestones)}; review incomplete.")
        if failure.terminal_received:
            text += " Automatic retry disabled after completion."
        safe = sanitize_verbose_message(text)
        return safe[:1024] if type(safe) is str and safe else None
    except Exception:  # noqa: BLE001 - formatting must never expose a raw fallback or replace the primary error
        return None


@dataclass(frozen=True)
class BackendLifecycleSnapshot:
    """Closed, content-free attempt evidence; absent observations remain None."""

    attempt: int
    backend: Literal["pi", "codex"] = "pi"
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

    read_observation: Literal["pull_driven"] = "pull_driven"
    stream_idle_timeout_s: float | None = None
    terminal_drain_timeout_s: float | None = None
    process_reap_timeout_s: float | None = None
    cleanup_timeout_s: float | None = None
    last_raw_line_s: float | None = None
    last_native_event_s: float | None = None
    last_native_event_kind: CodexPublicEventKind | None = None
    terminal_received_s: float | None = None
    stdout_eof_s: float | None = None
    process_exit_s: float | None = None
    process_returncode: int | None = None
    teardown_started_s: float | None = None
    teardown_completed_s: float | None = None
    expiry_stage: LifecycleStage | None = None
    cleanup_expired: bool = False
    adapter_yield_count: int = 0
    last_adapter_yield_s: float | None = None
    consumer_request_count: int = 0
    consumer_event_count: int = 0
    last_consumer_request_s: float | None = None
    last_consumer_event_s: float | None = None
    recorder_event_count: int = 0
    last_recorder_event_s: float | None = None
    loop_sample_count: int = 0
    last_loop_sample_s: float | None = None
    loop_max_lag_s: float = 0.0

    def __post_init__(self) -> None:
        allowed_strings = {
            "backend": {"pi", "codex"}, "read_observation": {"pull_driven"},
            "timestamp_source": {"host_observed"}, "clock": {"monotonic_elapsed"},
            "state": {"preparation", "prepared", "spawn_pending", "launched", "native_event", "generating"},
        }
        optional_enums = {
            "last_native_event_kind": {"thread_started", "turn_started", "item_started", "item_updated",
                                       "item_completed", "turn_completed", "turn_failed", "error", "unknown"},
            "expiry_stage": {"preterminal_stream", "terminal_drain", "process_reap", "cleanup"},
        }
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name in allowed_strings:
                valid = type(value) is str and value in allowed_strings[field.name]
            elif field.name in optional_enums:
                valid = value is None or (type(value) is str and value in optional_enums[field.name])
            elif field.name == "process_returncode":
                valid = value is None or (type(value) is int and -_MAX_COUNT <= value <= _MAX_COUNT)
            elif field.name in {"idle_detection_enabled", "generation_open", "cleanup_expired"}:
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
        self._sampler: asyncio.Task[None] | None = None

    def bind(self) -> Token[BackendLifecycleCollector | None]:
        """Bind this attempt only; concurrent invocations inherit separate collectors."""
        return _CURRENT.set(self)

    @staticmethod
    def reset(token: Token[BackendLifecycleCollector | None]) -> None:
        _CURRENT.reset(token)

    def _elapsed(self) -> float:
        elapsed = clock.monotonic() - self._started_at
        return max(0.0, elapsed) if math.isfinite(elapsed) else self._snapshot.elapsed_s

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

    def activate_codex(self, *, stream_idle_timeout_s: float | None, terminal_drain_timeout_s: float,
                       process_reap_timeout_s: float, cleanup_timeout_s: float) -> None:
        """Observe effective runtime policy without owning or extending its deadlines."""
        self._active = True
        self._update(backend="codex", stream_idle_timeout_s=stream_idle_timeout_s,
                     idle_detection_enabled=bool(stream_idle_timeout_s),
                     terminal_drain_timeout_s=terminal_drain_timeout_s,
                     process_reap_timeout_s=process_reap_timeout_s, cleanup_timeout_s=cleanup_timeout_s)
        if self._sampler is None and not self._finalized:
            self._sampler = asyncio.create_task(self._sample_loop())

    async def _sample_loop(self) -> None:
        # Runs even when the stream consumer is suspended at a yield. A blocked
        # event loop is observable only after it resumes; this is not an OS probe.
        interval = 0.05
        loop = asyncio.get_running_loop()
        expected = loop.time() + interval
        while True:
            await asyncio.sleep(max(0.0, expected - loop.time()))
            lag = max(0.0, loop.time() - expected)
            self._update(loop_sample_count=self._increment("loop_sample_count"),
                         last_loop_sample_s=self._elapsed(),
                         loop_max_lag_s=max(self._snapshot.loop_max_lag_s, lag))
            expected = loop.time() + interval

    async def stop_sampling(self) -> None:
        """Join the attempt's sampler before synchronously freezing evidence."""
        if self._sampler is not None:
            task, self._sampler = self._sampler, None
            task.cancel()
            with anyio.CancelScope(shield=True):
                await asyncio.gather(task, return_exceptions=True)

    def _observe_boundary(self, count: str, timestamp: str) -> None:
        self._update(**{count: self._increment(count), timestamp: self._elapsed()})

    def adapter_yield(self) -> None:
        self._observe_boundary("adapter_yield_count", "last_adapter_yield_s")

    def consumer_requested(self) -> None:
        self._observe_boundary("consumer_request_count", "last_consumer_request_s")

    def consumer_event(self) -> None:
        self._observe_boundary("consumer_event_count", "last_consumer_event_s")

    def recorder_event(self) -> None:
        self._observe_boundary("recorder_event_count", "last_recorder_event_s")

    def terminal_received(self) -> None:
        if self._snapshot.terminal_received_s is None:
            self._update(terminal_received_s=self._elapsed())

    def stdout_eof(self) -> None:
        if self._snapshot.stdout_eof_s is None:
            self._update(stdout_eof_s=self._elapsed())

    def process_exited(self, returncode: int) -> None:
        if self._snapshot.process_exit_s is None:
            self._update(process_exit_s=self._elapsed(), process_returncode=returncode)

    def teardown_started(self) -> None:
        if self._snapshot.teardown_started_s is None:
            self._update(teardown_started_s=self._elapsed())

    def teardown_completed(self) -> None:
        if self._snapshot.teardown_completed_s is None:
            self._update(teardown_completed_s=self._elapsed())

    def expiry(self, stage: LifecycleStage) -> None:
        self._update(expiry_stage=self._snapshot.expiry_stage or stage,
                     cleanup_expired=self._snapshot.cleanup_expired or stage == "cleanup")

    def request_prepared(self) -> None:
        self._update(state="prepared", request_prepared_s=self._elapsed())

    def spawn_pending(self) -> None:
        self._update(state="spawn_pending", spawn_pending_s=self._elapsed())

    def spawn_completed(self) -> None:
        self._update(state="launched", spawn_completed_s=self._elapsed())

    def raw_line_received(self) -> None:
        self._update(raw_line_count=self._increment("raw_line_count"), last_raw_line_s=self._elapsed(),
                     first_raw_line_s=self._snapshot.first_raw_line_s
                     if self._snapshot.first_raw_line_s is not None else self._elapsed())

    def native_event(self, kind: CodexPublicEventKind | None = None) -> None:
        self._update(native_event_count=self._increment("native_event_count"),
                     last_native_event_s=self._elapsed(), last_native_event_kind=kind,
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

    def snapshot(self) -> BackendLifecycleSnapshot | None:
        """Observe immutable current facts without stopping sampling or freezing the attempt."""
        if not self._active:
            return None
        if self._finalized:
            return self._snapshot
        elapsed = self._elapsed()
        return replace(self._snapshot, elapsed_s=elapsed,
                       remaining_wall_s=max(0.0, self._deadline - clock.monotonic())
                       if self._deadline is not None else None,
                       last_progress_age_s=max(0.0, elapsed - self._snapshot.last_progress_s)
                       if self._snapshot.last_progress_s is not None else None)

    def finalize(self) -> BackendLifecycleSnapshot | None:
        """Freeze without awaiting or yielding, including cancellation/GeneratorExit."""
        snapshot = self.snapshot()
        if snapshot is not None:
            self._snapshot = snapshot
            self._finalized = True
        return snapshot


_CURRENT: ContextVar[BackendLifecycleCollector | None] = ContextVar("backend_lifecycle", default=None)


def current_backend_lifecycle() -> BackendLifecycleCollector | None:
    """Return the runner-owned collector, if executing inside an attempt."""
    return _CURRENT.get()
