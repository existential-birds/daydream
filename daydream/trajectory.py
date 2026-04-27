"""ATIF v1.6 trajectory recorder for daydream runs.

This module is the SOLE home for ATIF Pydantic model construction (D-19
module-bloat ban). Other modules (agent.py, phases.py, ui.py, runner.py,
backends/*) import only the public surface — never `daydream.atif.*`.

Lifecycle: ``runner.py`` opens ``async with TrajectoryRecorder(...) as
recorder`` once per run. ``agent.run_agent()`` opens an ``Invocation`` per
call against the recorder via ``get_current_recorder()``. Backends emit
``AgentEvent`` instances; the Invocation buffers them into ATIF Steps and
flushes to the parent Trajectory at scope exit. The Recorder writes the
Trajectory JSON on clean ``__aexit__``.

Phase 2 shipped the minimum surface: ONE ``ContextVar`` (``_RECORDER_VAR``),
and a no-op ``Redactor``. Phase 3 adds ``fork()`` for parallel task groups
(sibling trajectory files) reusing the single ``_RECORDER_VAR`` (D-04). Phase 4 fills
in the Redactor rule list.
"""

from __future__ import annotations

import json
import re
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

from daydream.atif import (
    Agent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    SubagentTrajectoryRef,
    ToolCall,
    Trajectory,
)
from daydream.ui import create_console, print_warning

if TYPE_CHECKING:
    from daydream.backends import AgentEvent

_console = create_console()
_INITIAL_TOTALS: dict[str, Any] = {"prompt": 0, "completion": 0, "cached": 0, "cost": 0.0, "any_cost_seen": False}  # noqa: E501 - module-level constant cloned via dict.copy() at recorder init


def _safe_descriptor(raw: str) -> str:
    """Slugify a descriptor to filesystem-safe characters (D-06).

    Raises:
        ValueError: If *raw* produces an empty slug after sanitization.
    """
    slug = re.sub(r"[^a-z0-9-]", "-", raw.lower())
    slug = re.sub(r"-{2,}", "-", slug)
    slug = slug.strip("-")
    if not slug:
        raise ValueError(f"Descriptor {raw!r} produces empty slug after sanitization")
    return slug


def now_iso() -> str:
    """Return current UTC time as ISO 8601 with trailing 'Z'.

    The single source of truth for timestamps in daydream's trajectory
    recording. Used by ``AgentEvent`` dataclass ``field(default_factory=...)``
    in ``daydream/backends/__init__.py`` (Plan 02), by recorder Step
    construction here, and by Phase 4 partial-write paths.

    Banned alternatives: the deprecated naive-utc helper from ``datetime``
    (Pitfall 2: lacks tzinfo, deprecated in 3.12+); ad-hoc
    ``datetime.now().isoformat()`` (no ``Z`` suffix — Pydantic timestamp
    validator requires ``Z`` or ``+00:00``).

    Returns:
        Timestamp string parseable by ``Step.validate_timestamp``.
    """
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


class DaydreamPhase(str, Enum):
    """Phase label for ``Step.extra['daydream_phase']`` (MAP-08).

    Values match ATIF ``extra`` field literals exactly. Required keyword-only
    arg on ``run_agent()`` (D-05); every call site in ``phases.py`` passes a
    literal member.
    """

    REVIEW = "review"
    PARSE = "parse"
    FIX = "fix"
    TEST = "test"
    INTENT = "intent"
    ALTERNATIVES = "alternatives"
    PLAN = "plan"
    PR_FEEDBACK = "pr_feedback"
    DEEP = "deep"
    EXPLORATION = "exploration"


class DaydreamRunFlow(str, Enum):
    """Run-flow label for ``Step.extra['daydream_run_flow']`` (MAP-09).

    Set once at recorder construction (D-07); recorder stamps every Step.
    """

    NORMAL = "normal"
    TTT = "ttt"
    PR = "pr"
    DEEP = "deep"


class Redactor:
    """No-op pass-through redactor (Phase 2 stub).

    Phase 2 ships the FINAL public API surface; Phase 4 (REDA-01..06) fills
    in regex pattern lists internally without changing the recorder call
    site. ``redact_step`` is invoked at per-Step flush time per D-13 so
    Phase 4's partial-write paths inherit the same redaction posture.
    """

    def redact_step(self, step: Step) -> Step:
        """Return the step unchanged (Phase 2). Phase 4 fills this in.

        Args:
            step: ATIF Step about to be appended to the Trajectory.

        Returns:
            The same Step instance — Phase 2 is a strict pass-through.
        """
        return step


# Module-level Singletons
# =======================
# This module uses a ContextVar (NOT a module-level dataclass instance per
# PROJECT.md Constraints "propagated via ContextVar (not AgentState)") for
# trajectory recorder propagation. Access via ``get_current_recorder()`` ONLY;
# never import ``_RECORDER_VAR`` directly. The setter is implicit via
# ``TrajectoryRecorder.__aenter__`` / ``__aexit__``. Test isolation goes
# through ``_reset_recorder_for_tests()`` from the autouse conftest fixture
# (CORE-10 / D-17, wired in Plan 07).

_RECORDER_VAR: ContextVar["TrajectoryRecorder | None"] = ContextVar(
    "_RECORDER_VAR", default=None,
)


def get_current_recorder() -> "TrajectoryRecorder | None":
    """Return the recorder for the current async context, or None if none active.

    The single public accessor for ``_RECORDER_VAR`` (D-10). ``agent.py`` reads
    this at the top of ``run_agent()`` and skips the entire Invocation lifecycle
    when None — direct test invocation of ``run_agent()`` without an active
    recorder is therefore a clean no-op (CORE-09).

    Returns:
        The active ``TrajectoryRecorder`` instance, or ``None`` if no
        ``async with TrajectoryRecorder(...)`` block is on the stack.
    """
    return _RECORDER_VAR.get()


def _reset_recorder_for_tests() -> None:
    """Test-only: clear the recorder ContextVar.

    Use exclusively from the autouse ``_reset_trajectory_recorder`` fixture
    in ``tests/conftest.py`` (CORE-10, D-17). Production code MUST go through
    ``TrajectoryRecorder.__aenter__`` / ``__aexit__``.
    """
    _RECORDER_VAR.set(None)


@dataclass
class Invocation:
    """Per-``run_agent()`` recording scope.

    Owns the Step buffer for one model conversation and the in-flight
    ``tool_call_id -> open-step`` map (CORE-06). ``parent`` linkage lives on
    ``TrajectoryRecorder`` (Phase 3, D-02), not on ``Invocation``.

    Attributes:
        recorder: Owning TrajectoryRecorder (shares step_id counter, Redactor).
        phase: DaydreamPhase label stamped on every Step (MAP-08, D-05).
        steps: Steps accumulated; flushed to ``recorder.steps`` at scope exit.
        _open_step_dict: In-progress agent-step state before flush.
        _in_flight_tools: tool_call_id -> open-step-state, so a ToolResultEvent
            always lands on the SAME step as its ToolStartEvent (Pitfall 3).
        _message_id_to_step: AssistantMessage.message_id -> open-step (D-04).
    """

    recorder: "TrajectoryRecorder"
    phase: DaydreamPhase
    steps: list[Step] = field(default_factory=list)
    _open_step_dict: dict[str, Any] | None = None
    _in_flight_tools: dict[str, dict[str, Any]] = field(default_factory=dict)
    _message_id_to_step: dict[str, dict[str, Any]] = field(default_factory=dict)
    _current_message_id: str = ""

    def observe_user_step(self, prompt: str) -> None:
        """Append a user Step at invocation start (MAP-01, Pitfall 4).

        Constructs a minimal user Step — only step_id / timestamp / source /
        message / extra. NO agent-only fields (model_name, tool_calls,
        metrics, reasoning_content) so Step.validate_agent_only_fields
        passes.
        """
        try:
            self._close_open_step()
            user_step = Step(
                step_id=self.recorder._next_step_id(),
                timestamp=now_iso(),
                source="user",
                message=prompt,
                extra={
                    "daydream_phase": self.phase.value,
                    "daydream_run_flow": self.recorder.run_flow.value,
                },
            )
            self.steps.append(self.recorder.redactor.redact_step(user_step))
        except Exception as exc:  # noqa: BLE001 - recording must never crash a run (Architecture Q7)
            print_warning(_console, f"Trajectory recording: {type(exc).__name__}: {exc}")

    def observe(self, event: "AgentEvent") -> None:
        """Dispatch an AgentEvent into the active Step buffer.

        Catch-and-degrade boundary (Architecture Q7): exceptions are caught
        here so trajectory recording NEVER crashes the user's review/fix
        run. The catch is local to this method — agent.py's event loop
        continues to surface its own errors.
        """
        try:
            self._dispatch(event)
        except Exception as exc:  # noqa: BLE001 - recording must never crash a run (Architecture Q7)
            print_warning(_console, f"Trajectory recording: {type(exc).__name__}: {exc}")

    def _dispatch(self, event: Any) -> None:
        # Function-local imports avoid load-order cycles with daydream.backends
        # AND defensively handle MetricsEvent's absence in Plan 02-01 (it lands
        # in Plan 02-02). Class-name fallback supports either path.
        from daydream.backends import (
            CostEvent,
            ResultEvent,
            TextEvent,
            ThinkingEvent,
            ToolResultEvent,
            ToolStartEvent,
        )
        try:
            from daydream.backends import MetricsEvent  # type: ignore[attr-defined]
        except ImportError:
            MetricsEvent = None  # type: ignore[assignment]

        if isinstance(event, TextEvent):
            self._ensure_open_step()
            assert self._open_step_dict is not None
            self._open_step_dict["_text_chunks"].append(event.text)
        elif isinstance(event, ThinkingEvent):
            self._ensure_open_step()
            assert self._open_step_dict is not None
            self._open_step_dict["_thinking_chunks"].append(event.text)
        elif isinstance(event, ToolStartEvent):
            self._ensure_open_step()
            assert self._open_step_dict is not None
            self._open_step_dict["_tool_calls"].append(
                ToolCall(tool_call_id=event.id, function_name=event.name, arguments=event.input or {})
            )
            # Map tool_call_id -> THIS open step so paired ToolResultEvent lands
            # on the SAME step (CORE-06, Pitfall 3).
            self._in_flight_tools[event.id] = self._open_step_dict
        elif isinstance(event, ToolResultEvent):
            host = self._in_flight_tools.pop(event.id, None)
            if host is None:
                # Dangling ToolResultEvent (Codex pending-id miss, Pitfall 3).
                # Mark via extra.unmatched_tool_results; do NOT emit a dangling
                # source_call_id reference (Trajectory validator hard-fail).
                self._ensure_open_step()
                assert self._open_step_dict is not None
                self._open_step_dict["_unmatched_tool_results"].append(event.id)
                return
            host["_observation_results"].append(
                ObservationResult(source_call_id=event.id, content=event.output)
            )
        elif (MetricsEvent is not None and isinstance(event, MetricsEvent)) or (
            type(event).__name__ == "MetricsEvent"
        ):
            # Class-name match supports the test stub before Plan 02-02's real
            # MetricsEvent lands. EVNT-02 attribute names verbatim (D-15:
            # cached_tokens is a SUBSET of prompt_tokens, not added).
            target = self._message_id_to_step.get(event.message_id, self._open_step_dict)
            if target is None:
                self._ensure_open_step()
                target = self._open_step_dict
            assert target is not None
            target["_metrics"] = Metrics(
                prompt_tokens=event.prompt_tokens,
                completion_tokens=event.completion_tokens,
                cached_tokens=event.cached_tokens,
                cost_usd=event.cost_usd,
            )
            # Aggregate into recorder-level totals for FinalMetrics (MAP-07).
            self.recorder._accumulate_metrics(
                prompt_tokens=event.prompt_tokens,
                completion_tokens=event.completion_tokens,
                cached_tokens=event.cached_tokens,
                cost_usd=event.cost_usd,
            )
        elif isinstance(event, CostEvent):
            # End-of-call signal. Phase 2 prefers MetricsEvent for per-step
            # Metrics (D-14); CostEvent path lights up in later phases.
            pass
        elif isinstance(event, ResultEvent):
            self._close_open_step()

    def _ensure_open_step(self) -> None:
        """Open a new agent step if none currently in flight."""
        if self._open_step_dict is not None:
            return
        self._open_step_dict = {
            "_text_chunks": [],
            "_thinking_chunks": [],
            "_tool_calls": [],
            "_observation_results": [],
            "_metrics": None,
            "_model_name": self.recorder.agent_model_name,
            "_unmatched_tool_results": [],
        }
        if self._current_message_id:
            self._message_id_to_step[self._current_message_id] = self._open_step_dict

    def _close_open_step(self) -> None:
        """Finalize the current open step into a Pydantic Step + redact + append."""
        if self._open_step_dict is None:
            return
        d = self._open_step_dict
        self._open_step_dict = None

        message_text = "".join(d["_text_chunks"])
        reasoning = "\n".join(d["_thinking_chunks"]) if d["_thinking_chunks"] else None
        tool_calls = list(d["_tool_calls"]) or None
        observation = (
            Observation(results=list(d["_observation_results"]))
            if d["_observation_results"]
            else None
        )
        extra: dict[str, Any] = {
            "daydream_phase": self.phase.value,
            "daydream_run_flow": self.recorder.run_flow.value,
        }
        if d["_unmatched_tool_results"]:
            extra["unmatched_tool_results"] = list(d["_unmatched_tool_results"])

        agent_step = Step(
            step_id=self.recorder._next_step_id(),
            timestamp=now_iso(),
            source="agent",
            message=message_text,
            model_name=d["_model_name"],
            reasoning_content=reasoning,
            tool_calls=tool_calls,
            observation=observation,
            metrics=d["_metrics"],
            extra=extra,
        )
        self.steps.append(self.recorder.redactor.redact_step(agent_step))

    def finish(self) -> None:
        """Close any open step and flush all steps to the parent recorder."""
        self._close_open_step()
        self.recorder._extend_steps(self.steps)


@dataclass
class TrajectoryRecorder:
    """Owns the per-run ATIF Trajectory and writes it to disk on clean exit.

    Phase 2 surface: ONE recorder per run, opened via ``async with`` from
    ``runner.py``. ``__aenter__`` sets ``_RECORDER_VAR``; ``__aexit__`` writes
    the trajectory and clears the ContextVar. Disk-write failure degrades to
    ``print_warning`` per D-11 (Phase 4 adds the explicit fail-loud branch).

    Attributes:
        path: Output JSON path; default ``<target>/.daydream/trajectory.json``.
        run_flow: Per-trajectory invariant (D-07) stamped on every Step.
        target_dir: Repo/target directory; recorded into Trajectory.extra.
        agent_model_name: Active model name; stamped into Agent and every
            agent Step's model_name.
        redactor: No-op in Phase 2 (D-12); Phase 4 fills in rule list.
        session_id: UUID4 generated at recorder init (CORE-07).
        steps: Sequential Steps from every Invocation, step_id 1..N.
        _step_id_counter: Monotonic; never decreases (Pitfall 1).
        _final_totals: Running tally for FinalMetrics aggregation (MAP-07).
        _previous_token: ContextVar reset token; used by __aexit__ to restore.
    """

    path: Path
    run_flow: DaydreamRunFlow
    target_dir: Path
    agent_model_name: str
    redactor: Redactor = field(default_factory=Redactor)
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    steps: list[Step] = field(default_factory=list)
    parent: TrajectoryRecorder | None = None
    descriptor: str = ""
    _step_id_counter: int = 0
    _final_totals: dict[str, Any] = field(default_factory=lambda: _INITIAL_TOTALS.copy())
    _previous_token: Any = None
    _registered_siblings: list[tuple[Path, str]] = field(default_factory=list)

    async def __aenter__(self) -> "TrajectoryRecorder":
        self._previous_token = _RECORDER_VAR.set(self)
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        try:
            self._write()
        except Exception as exc:  # noqa: BLE001 - implicit write degrade-with-warning per D-11
            print_warning(
                _console,
                f"Trajectory write failed: {type(exc).__name__}: {exc}",
            )
        finally:
            if self._previous_token is not None:
                _RECORDER_VAR.reset(self._previous_token)
                self._previous_token = None

    def invocation(self, *, phase: DaydreamPhase) -> "_InvocationCM":
        """Open an Invocation scope for one ``run_agent()`` call.

        Returns an async-context-manager that flushes its accumulated Steps
        to ``self.steps`` on exit. Phase 2 has no parent linkage — flat
        sequential append per D-08.
        """
        return _InvocationCM(self, phase)

    def _next_step_id(self) -> int:
        self._step_id_counter += 1
        return self._step_id_counter

    def _extend_steps(self, steps: list[Step]) -> None:
        self.steps.extend(steps)

    def _accumulate_metrics(
        self,
        *,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        cached_tokens: int | None,
        cost_usd: float | None,
    ) -> None:
        if prompt_tokens is not None:
            self._final_totals["prompt"] = (self._final_totals["prompt"] or 0) + prompt_tokens
        if completion_tokens is not None:
            self._final_totals["completion"] = (
                (self._final_totals["completion"] or 0) + completion_tokens
            )
        if cached_tokens is not None:
            self._final_totals["cached"] = (self._final_totals["cached"] or 0) + cached_tokens
        if cost_usd is not None:
            self._final_totals["cost"] = (self._final_totals["cost"] or 0.0) + cost_usd
            self._final_totals["any_cost_seen"] = True

    def _sibling_path_for(self, descriptor: str) -> Path:
        """Return the sibling trajectory file path for *descriptor*."""
        slug = _safe_descriptor(descriptor)
        return self.target_dir / ".daydream" / "trajectories" / f"{self.session_id[:8]}.{slug}.json"

    def fork(self, descriptor: str) -> "_ForkCM":
        """Create a child recorder for a parallel task group.

        Args:
            descriptor: Semantic label for the sibling (e.g. ``"fix-0"``).

        Returns:
            An async context manager yielding a child ``TrajectoryRecorder``.
        """
        return _ForkCM(parent=self, descriptor=descriptor)

    def _register_sibling(self, path: Path, descriptor: str) -> None:
        """Register a completed sibling trajectory (synchronous, no await)."""
        self._registered_siblings.append((path, descriptor))

    def create_dispatch_step(self, *, phase: DaydreamPhase) -> None:
        """Create an agent Step referencing all registered sibling trajectories.

        No-op when ``_registered_siblings`` is empty.
        """
        if not self._registered_siblings:
            return
        results: list[ObservationResult] = []
        for sibling_path, desc in self._registered_siblings:
            rel = str(sibling_path.relative_to(self.target_dir / ".daydream"))
            results.append(
                ObservationResult(
                    content=f"Dispatched to {desc}",
                    subagent_trajectory_ref=[
                        SubagentTrajectoryRef(session_id=self.session_id, trajectory_path=rel),
                    ],
                )
            )
        count = len(self._registered_siblings)
        step = Step(
            step_id=self._next_step_id(),
            timestamp=now_iso(),
            source="agent",
            model_name=self.agent_model_name,
            message=f"Dispatching {count} parallel {phase.value} tasks",
            observation=Observation(results=results),
            extra={
                "daydream_phase": phase.value,
                "daydream_run_flow": self.run_flow.value,
            },
        )
        self.steps.append(self.redactor.redact_step(step))
        self._registered_siblings.clear()

    def _build_trajectory(self) -> Trajectory:
        try:
            version = metadata.version("daydream")
        except metadata.PackageNotFoundError:
            version = "0.0.0"

        final_metrics = FinalMetrics(
            total_prompt_tokens=self._final_totals["prompt"] or None,
            total_completion_tokens=self._final_totals["completion"] or None,
            total_cached_tokens=self._final_totals["cached"] or None,
            total_cost_usd=(
                self._final_totals["cost"] if self._final_totals["any_cost_seen"] else None
            ),
            total_steps=len(self.steps),
        )
        return Trajectory(
            schema_version="ATIF-v1.6",
            session_id=self.session_id,
            agent=Agent(name="daydream", version=version, model_name=self.agent_model_name),
            steps=list(self.steps),
            final_metrics=final_metrics,
            extra={"target_dir": str(self.target_dir)},
        )

    def _write(self) -> None:
        # Empty trajectory: skip — Pydantic Trajectory.steps has min_length=1.
        # Phase 4 may revisit if empty runs need a stub file on disk.
        if not self.steps:
            return
        trajectory = self._build_trajectory()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(trajectory.to_json_dict(), indent=2), encoding="utf-8")


class _ForkCM:
    """Async context manager for forking a child recorder (D-01, D-02, D-03)."""

    def __init__(self, parent: TrajectoryRecorder, descriptor: str) -> None:
        self._parent = parent
        self._descriptor = descriptor
        self._child: TrajectoryRecorder | None = None

    async def __aenter__(self) -> TrajectoryRecorder:
        child = TrajectoryRecorder(
            path=self._parent._sibling_path_for(self._descriptor),
            run_flow=self._parent.run_flow,
            target_dir=self._parent.target_dir,
            agent_model_name=self._parent.agent_model_name,
            redactor=self._parent.redactor,
            session_id=self._parent.session_id,
        )
        child.parent = self._parent
        child.descriptor = self._descriptor
        child._previous_token = _RECORDER_VAR.set(child)
        self._child = child
        return child

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        child = self._child
        if child is None:
            return
        try:
            child._write()
        except Exception as exc:  # noqa: BLE001 - recording must never crash a run
            print_warning(_console, f"Sibling trajectory write failed: {type(exc).__name__}: {exc}")
        finally:
            if child._previous_token is not None:
                _RECORDER_VAR.reset(child._previous_token)
                child._previous_token = None
        if child.parent is not None and child.path.exists():
            child.parent._register_sibling(child.path, self._descriptor)


class _InvocationCM:
    """Async context manager wrapping an Invocation (internal helper)."""

    def __init__(self, recorder: TrajectoryRecorder, phase: DaydreamPhase) -> None:
        self._recorder = recorder
        self._phase = phase
        self._invocation: Invocation | None = None

    async def __aenter__(self) -> Invocation:
        self._invocation = Invocation(recorder=self._recorder, phase=self._phase)
        return self._invocation

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._invocation is not None:
            self._invocation.finish()
            self._invocation = None
