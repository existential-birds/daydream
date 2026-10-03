"""ATIF v1.7 persistence and ContextVar scope ownership. Invocation buffers events; lifecycle, timing,
and generation billing have separate owners.
"""

from __future__ import annotations

# external contract: "ATIF v1.7" names the Harbor trajectory-format version, not a project-owned name
import hashlib
import json
import re
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import daydream
from daydream import timeutil, ui
from daydream.atif import (
    Agent,
    FinalMetrics,
    Observation,
    ObservationResult,
    Step,
    SubagentTrajectoryRef,
    Trajectory,
)
from daydream.json_utils import atomic_write_json
from daydream.redaction import (
    redact_value,
)
from daydream.trajectory.context import _RECORDER_VAR
from daydream.trajectory.invocation import _GENERIC_MODEL_LABELS, Invocation
from daydream.trajectory.layout import (
    _DAYDREAM_DIRNAME,
    RUNS_DIRNAME,
    partial_document_path,
    run_directory,
    sibling_document_path,
)
from daydream.trajectory.lifecycle import DispatchHandle, ForkIdentity, PhaseEvent
from daydream.trajectory.redactor import Redactor
from daydream.trajectory.scopes import fork_scope, invocation_scope
from daydream.trajectory.types import (
    DaydreamPhase,
    DaydreamRunFlow,
    LifecycleReasonCode,
    LifecycleStatus,
    RunWriteSnapshot,
    TrajectoryDocumentSnapshot,
)
from daydream.ui import create_console, print_error

_console = create_console()


def _warn_partial_write(exc: Exception) -> None:
    """Warn that a partial trajectory snapshot write failed."""
    ui.print_warning(_console, f"Partial trajectory write failed: {type(exc).__name__}: {exc}")


_INITIAL_TOTALS: dict[str, Any] = {
    "prompt": 0,
    "completion": 0,
    "cached": 0,
    "cost": 0.0,
    "any_cost_seen": False,
}  # noqa: E501 - module-level constant cloned via dict.copy() at recorder init


def _safe_descriptor(raw: str) -> str:
    """Slugify a descriptor for filenames; reject an empty sanitized result."""
    slug = re.sub(r"[^a-z0-9-]", "-", raw.lower())
    slug = re.sub(r"-{2,}", "-", slug)
    slug = slug.strip("-")
    if not slug:
        raise ValueError(f"Descriptor {raw!r} produces empty slug after sanitization")
    return slug


TrajectoryWriteCallback = Callable[["TrajectoryRecorder", RunWriteSnapshot], None]
TrajectoryDocumentWriter = Callable[[TrajectoryDocumentSnapshot, Literal["complete", "partial"]], None]


class _SignalFlushRegistry:
    """Identity membership for every active recorder owned by one run."""

    def __init__(self) -> None:
        self._active: dict[int, TrajectoryRecorder] = {}
        self._completed: dict[str, TrajectoryDocumentSnapshot] = {}
        self._partial_state_digest = ""
        self._partial_cutoff_at = ""

    def register(self, recorder: TrajectoryRecorder) -> None:
        """Register *recorder* once by object identity."""
        self._active[id(recorder)] = recorder

    def unregister(self, recorder: TrajectoryRecorder) -> None:
        """Remove *recorder* when the same identity is still registered."""
        if self._active.get(id(recorder)) is recorder:
            self._active.pop(id(recorder), None)

    def retain(self, document: TrajectoryDocumentSnapshot) -> None:
        """Retain the exact bytes of a completed child until root archival."""
        self._completed[document.trajectory_id] = document

    def _partial_cutoff(self, recorders: Sequence[TrajectoryRecorder]) -> str:
        state = [(recorder.trajectory_id, recorder._partial_state_key()) for recorder in recorders]
        state.extend(
            (trajectory_id, hashlib.sha256(document.json_bytes).hexdigest())
            for trajectory_id, document in sorted(self._completed.items())
        )
        digest = hashlib.sha256(json.dumps(state, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if digest != self._partial_state_digest:
            self._partial_state_digest = digest
            self._partial_cutoff_at = timeutil.now_iso()
        return self._partial_cutoff_at

    def _root(self) -> TrajectoryRecorder | None:
        return next(
            (recorder for recorder in self._active.values() if recorder.parent is None),
            None,
        )

    def _write_snapshot(
        self,
        *,
        status: Literal["complete", "partial"],
        root: TrajectoryRecorder,
        prepared: Sequence[TrajectoryDocumentSnapshot],
        cutoff_at: str,
    ) -> RunWriteSnapshot:
        documents = dict(self._completed)
        documents.update((document.trajectory_id, document) for document in prepared)
        ordered = tuple(
            sorted(
                documents.values(),
                key=lambda document: (
                    document.trajectory_id != root.trajectory_id,
                    document.trajectory_id,
                ),
            )
        )
        snapshot = RunWriteSnapshot(
            status=status,
            cutoff_at=cutoff_at,
            root_trajectory_id=root.trajectory_id,
            documents=ordered,
        )
        for document in prepared:
            try:
                root._write_document(document, status)
            except Exception as exc:  # noqa: BLE001 - isolate every recorder write
                if status == "complete":
                    raise
                _warn_partial_write(exc)
        if root.on_write is not None:
            try:
                root.on_write(root, snapshot)
            except Exception:  # noqa: BLE001 - archive failure never affects recording
                pass
        return snapshot

    def flush_active(self) -> None:
        """Freeze every active recorder, write all, then call the root once."""
        recorders = tuple(self._active.values())
        root = self._root()
        if root is None:
            return
        cutoff = self._partial_cutoff(recorders)
        prepared: list[TrajectoryDocumentSnapshot] = []
        for recorder in recorders:
            if recorder is root:
                continue
            try:
                document = recorder._prepare_document(status="partial", cutoff_at=cutoff)
                if document is not None:
                    prepared.append(document)
            except Exception as exc:  # noqa: BLE001 - isolate each signal write
                _warn_partial_write(exc)
        child_evidence = bool(prepared or self._completed)
        try:
            root_document = root._prepare_document(
                status="partial",
                cutoff_at=cutoff,
                allow_empty_root=child_evidence,
            )
            if root_document is not None:
                prepared.insert(0, root_document)
        except Exception as exc:  # noqa: BLE001 - isolate each signal write
            _warn_partial_write(exc)
            return
        if child_evidence and root_document is None:
            return
        if prepared or self._completed:
            self._write_snapshot(
                status="partial",
                root=root,
                prepared=prepared,
                cutoff_at=cutoff,
            )

    def write_final(self, root: TrajectoryRecorder) -> None:
        """Write the final root document and publish one retained run snapshot."""
        cutoff_at = root._run_ended_at or timeutil.now_iso()
        document = root._prepare_document(
            status="complete",
            cutoff_at=cutoff_at,
            allow_empty_root=root.document_writer is not None,
        )
        if document is None:
            return
        self._write_snapshot(
            status="complete",
            root=root,
            prepared=(document,),
            cutoff_at=cutoff_at,
        )


# Nested roots own separate registries; sibling entry order never determines signal flush ownership.
_ACTIVE_SIGNAL_RUNS: list[_SignalFlushRegistry] = []


def flush_active_signal_recorders() -> None:
    """Synchronously flush every active recorder in the selected run."""
    if _ACTIVE_SIGNAL_RUNS:
        _ACTIVE_SIGNAL_RUNS[-1].flush_active()


@dataclass
class TrajectoryRecorder:
    """Own ATIF steps, child recorders, and redacted durable snapshots.

    The async scope restores its parent and writes on exit. Steps have ordered monotonic
    IDs; backend labels upgrade to native models. Callers supply identities, paths, and
    review/fix/test labels; omitted stages retain empty labels.
    """

    path: Path
    run_flow: DaydreamRunFlow
    target_dir: Path
    agent_model_name: str
    session_id: str
    artifact_run_dir: Path | None = None
    document_writer: TrajectoryDocumentWriter | None = None
    redactor: Redactor = field(default_factory=Redactor)
    steps: list[Step] = field(default_factory=list)
    parent: TrajectoryRecorder | None = None
    descriptor: str = ""
    explicit_path: bool = False
    pr_number: int | None = None
    pr_repo: str | None = None
    backend_name: str = ""
    review_backend_name: str = ""
    fix_backend_name: str = ""
    test_backend_name: str = ""
    _step_id_counter: int = 0
    _phase_scope_counter: int = 0
    _dispatch_counter: int = 0
    _invocation_counter: int = 0
    _final_totals: dict[str, Any] = field(default_factory=lambda: _INITIAL_TOTALS.copy())
    _folded_fork_totals: bool = False
    _previous_token: Any = None
    # Active buffers preserve work in partial snapshots during interrupted agent calls.
    _active_invocations: list[Invocation] = field(default_factory=list)
    # Identified phase boundaries serialize into extra.phase_events when present.
    _phase_events: list[PhaseEvent] = field(default_factory=list)
    # Completed invocation timing summaries registered at scope exit.
    # Serialized into Trajectory.extra["subtrajectories"] when non-empty.
    _subtrajectories: list[dict[str, Any]] = field(default_factory=list)
    # Runner supplies executed profile schema/name/source/digest; persist profile_* only when set.
    _profile: dict[str, Any] | None = None
    _aborted: bool = False
    on_write: TrajectoryWriteCallback | None = None
    _trajectory_id: str = ""
    _run_started_at: str = ""
    _run_ended_at: str = ""
    _signal_registry: _SignalFlushRegistry | None = field(default=None, init=False, repr=False, compare=False)

    async def __aenter__(self) -> "TrajectoryRecorder":
        self._run_started_at = timeutil.now_iso()
        self._run_ended_at = ""
        registry = _SignalFlushRegistry()
        self._signal_registry = registry
        registry.register(self)
        _ACTIVE_SIGNAL_RUNS.append(registry)
        self._previous_token = _RECORDER_VAR.set(self)
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, _exc_tb: Any) -> None:
        try:
            if exc_type is not None:
                self._aborted = True
            self._run_ended_at = timeutil.now_iso()
            self._write()
        except Exception as exc:  # noqa: BLE001 - branch on explicit_path per D-06
            if self.explicit_path:
                # D-06: user asked for it, deliver or fail loud
                print_error(
                    _console,
                    "Trajectory write failed",
                    f"{type(exc).__name__}: {exc}",
                )
                if exc_val is not None:
                    # An in-flight failure owns the exit; do not mask it with SystemExit.
                    exc_val.add_note(f"trajectory finalization also failed ({type(exc).__name__})")
                    return
                raise SystemExit(2) from exc
            # Implicit/default path — degrade with warning per CORE-09 / D-11
            ui.print_warning(
                _console,
                f"Trajectory write failed: {type(exc).__name__}: {exc}",
            )
        finally:
            registry = self._signal_registry
            if registry is not None:
                registry.unregister(self)
                self._signal_registry = None
                for index in range(len(_ACTIVE_SIGNAL_RUNS) - 1, -1, -1):
                    if _ACTIVE_SIGNAL_RUNS[index] is registry:
                        del _ACTIVE_SIGNAL_RUNS[index]
                        break
            if self._previous_token is not None:
                _RECORDER_VAR.reset(self._previous_token)
                self._previous_token = None

    def invocation(self, *, phase: DaydreamPhase) -> AbstractAsyncContextManager[Invocation]:
        """Open an invocation scope that flushes its steps on exit."""
        return invocation_scope(self, phase)

    @property
    def trajectory_id(self) -> str:
        """Return this recorder's stable, document-qualified identity."""
        if self._trajectory_id:
            return self._trajectory_id
        if self.descriptor:
            return f"{self.session_id}:{self.descriptor}"
        return self.session_id

    def _emit_phase_event(
        self,
        phase: DaydreamPhase,
        event: str,
        *,
        session_id: str | None = None,
        scope_id: str | None = None,
        status: LifecycleStatus | None = None,
        reason_code: LifecycleReasonCode | None = None,
        **metadata: Any,
    ) -> None:
        """Append a :class:`PhaseEvent` stamped with ``timeutil.now_iso()``."""
        self._phase_events.append(
            PhaseEvent(
                phase=phase,
                event=event,
                timestamp=timeutil.now_iso(),
                session_id=session_id,
                scope_id=scope_id,
                status=status,
                reason_code=reason_code,
                metadata=metadata,
            )
        )

    def record_profile(self, *, schema_version: int, name: str, source_kind: str, digest: str) -> None:
        """Persist the resolved policy name, source, and canonical digest for attribution."""
        self._profile = {
            "profile_schema_version": schema_version,
            "profile_name": name,
            "profile_source_kind": source_kind,
            "profile_digest": digest,
        }

    def emit_file_group_budget_exceeded(
        self,
        *,
        file: str,
        reason: str,
        items_processed: int,
        items_skipped: int,
        elapsed_s: float | None = None,
    ) -> None:
        """Record FIX budget counts; omit elapsed_s when absent to retain the wire shape."""
        metadata: dict[str, Any] = {
            "file": file,
            "reason": reason,
            "items_processed": items_processed,
            "items_skipped": items_skipped,
        }
        if elapsed_s is not None:
            metadata["elapsed_s"] = elapsed_s
        self._emit_phase_event(DaydreamPhase.FIX, "file_group_budget_exceeded", **metadata)

    def emit_agent_budget_stop(
        self,
        phase: DaydreamPhase,
        *,
        limit_expired: str,
        elapsed_s: float,
        backend_s: float,
        backoff_s: float,
        attempts: int,
        cleanup_elapsed_s: float | None = None,
        retry_stop_reason: str | None = None,
        circuit_state: str | None = None,
        retry_recovery_spent_s: float | None = None,
        partial_edit_handling: str | None = None,
    ) -> None:
        """Record one stop using durations, excluding initial useful work from retry spend.

        Retain interrupted-attempt edits; discard partial edits when stopping before dispatch.
        """
        self._emit_phase_event(
            phase,
            "agent_budget_stop",
            limit_expired=limit_expired,
            elapsed_s=elapsed_s,
            backend_s=backend_s,
            backoff_s=backoff_s,
            attempts=attempts,
            cleanup_elapsed_s=cleanup_elapsed_s,
            retry_stop_reason=retry_stop_reason,
            circuit_state=circuit_state,
            retry_recovery_spent_s=retry_recovery_spent_s,
            partial_edit_handling=partial_edit_handling,
        )

    def emit_supervisor_verdict(self, finding_id: int, action: str, reason: str) -> None:
        """Record a findings supervisor verdict in the deep phase."""
        self._emit_phase_event(
            DaydreamPhase.DEEP,
            "supervisor_verdict",
            finding_id=finding_id,
            action=action,
            reason=reason,
        )

    def emit_tool_veto(self, tool_name: str, reason: str, *, phase: DaydreamPhase = DaydreamPhase.FIX) -> None:
        """Record a tool-supervisor veto in the firing phase."""
        self._emit_phase_event(phase, "tool_veto", tool_name=tool_name, reason=reason)

    def emit_command_validation_summary(
        self,
        *,
        total_candidates: int,
        accepted: int,
        rejected: int,
        reasons: dict[str, int],
    ) -> None:
        """Record a redacted repository-command validation summary."""
        metadata: dict[str, Any] = {
            "counts": {
                "total_candidates": total_candidates,
                "accepted": accepted,
                "rejected": rejected,
            },
            "reasons": dict(sorted(reasons.items())),
        }
        self._emit_phase_event(
            DaydreamPhase.RECON,
            "command_validation",
            **metadata,
        )

    def _register_subtrajectory(self, inv: Invocation) -> None:
        """Register finalized invocation timing in trajectory.extra.subtrajectories."""
        self._subtrajectories.append(inv.summary())

    def _recursive_invocation_summaries(self) -> list[dict[str, Any]]:
        """Flatten document-qualified invocation evidence through nested forks."""
        summaries: list[dict[str, Any]] = []
        for summary in self._subtrajectories:
            if isinstance(summary.get("invocation_id"), str):
                summaries.append(dict(summary))
            nested = summary.get("invocations")
            if isinstance(nested, list):
                summaries.extend(dict(item) for item in nested if isinstance(item, dict))
        return summaries

    def _recursive_phase_event_summaries(self) -> list[dict[str, Any]]:
        """Flatten identified phase evidence while retaining document identity."""
        summaries = [
            {**event.to_dict(), "trajectory_id": self.trajectory_id}
            for event in self._phase_events
            if event.scope_id is not None
        ]
        for summary in self._subtrajectories:
            nested = summary.get("phase_events")
            if isinstance(nested, list):
                summaries.extend(dict(item) for item in nested if isinstance(item, dict))
        return summaries

    def _register_fork_subtrajectory(
        self,
        *,
        child: "TrajectoryRecorder",
        phase: str,
        identity: ForkIdentity | None,
        started_at: str,
        ended_at: str,
        sibling_trajectory_ref: str,
    ) -> None:
        """Register a per-fork timing summary on the parent trajectory."""
        summary: dict[str, Any] = {
            "trajectory_id": child.trajectory_id,
            "phase": phase,
            "descriptor": child.descriptor,
            "started_at": started_at,
            "ended_at": ended_at,
            "sibling_trajectory_ref": sibling_trajectory_ref,
            "phase_events": child._recursive_phase_event_summaries(),
            "invocations": child._recursive_invocation_summaries(),
        }
        if identity is not None:
            summary["fork_id"] = identity.fork_id
            summary["dispatch_id"] = identity.fork_id.rsplit(":fork:", 1)[0]
        self._subtrajectories.append(summary)

    def _next_step_id(self) -> int:
        self._step_id_counter += 1
        return self._step_id_counter

    def _next_phase_scope_id(self) -> str:
        self._phase_scope_counter += 1
        return f"{self.trajectory_id}:phase:{self._phase_scope_counter}"

    def _next_dispatch_id(self) -> str:
        self._dispatch_counter += 1
        return f"{self.trajectory_id}:dispatch:{self._dispatch_counter}"

    def _next_invocation_id(self) -> str:
        self._invocation_counter += 1
        return f"{self.trajectory_id}:invocation:{self._invocation_counter}"

    def _extend_steps(self, steps: list[Step]) -> None:
        """Merge completed invocation steps in ID order.

        Concurrent invocations may close out of order; ATIF requires sequential IDs."""
        self.steps.extend(steps)
        self.steps.sort(key=lambda s: s.step_id)

    def _upgrade_model_name(self, candidate: str) -> None:
        """Replace provisional backend labels with the first native model name.

        Upgrade closed agent steps too: a backend may report its model after TurnEnd."""
        if candidate and (self.agent_model_name or "") in _GENERIC_MODEL_LABELS:
            self.agent_model_name = candidate
            for index, step in enumerate(self.steps):
                if step.source == "agent" and (step.model_name or "") in _GENERIC_MODEL_LABELS:
                    self.steps[index] = self.redactor.redact_step(step.model_copy(update={"model_name": candidate}))

    def _accumulate_metrics(
        self,
        *,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        cached_tokens: int | None,
        cost_usd: float | None,
    ) -> None:
        if prompt_tokens is not None:
            self._final_totals["prompt"] += prompt_tokens
        if completion_tokens is not None:
            self._final_totals["completion"] += completion_tokens
        if cached_tokens is not None:
            self._final_totals["cached"] += cached_tokens
        if cost_usd is not None:
            self._final_totals["cost"] += cost_usd
            self._final_totals["any_cost_seen"] = True

    def _sibling_path_for(
        self,
        descriptor: str,
        identity: ForkIdentity | None = None,
    ) -> Path:
        """Keep siblings in the parent run; identified forks append a digest to prevent slug collisions."""
        slug = _safe_descriptor(descriptor)
        if identity is not None:
            identity_digest = hashlib.sha256(identity.fork_id.encode("utf-8")).hexdigest()
            slug = f"{slug[:80]}--{identity_digest}"
        run_dir = self.artifact_run_dir
        if run_dir is None:
            run_dir = run_directory(self.target_dir / _DAYDREAM_DIRNAME, self.session_id)
        return sibling_document_path(run_dir, f"{slug}.json")

    def _logical_child_trajectory_ref(self, child_path: Path) -> str:
        """Return a child path relative to the stable public ``.daydream`` root."""
        if self.artifact_run_dir is not None:
            relative = child_path.relative_to(self.artifact_run_dir)
            return (Path(RUNS_DIRNAME) / self.session_id / relative).as_posix()
        return child_path.relative_to(self.target_dir / _DAYDREAM_DIRNAME).as_posix()

    def fork(
        self,
        descriptor: str,
        *,
        dispatch: DispatchHandle | None = None,
    ) -> AbstractAsyncContextManager[TrajectoryRecorder]:
        """Create a parallel child recorder with a semantic descriptor such as fix-0."""
        return fork_scope(self, descriptor, dispatch)

    def _create_dispatch_step(self, dispatch: DispatchHandle) -> None:
        """Materialize one identified, start-stamped deterministic dispatch."""
        results: list[ObservationResult] = []
        for completed in dispatch._ordered_completed():
            results.append(
                ObservationResult(
                    content=f"Dispatched to {completed.identity.descriptor}",
                    subagent_trajectory_ref=[
                        SubagentTrajectoryRef(
                            trajectory_id=completed.trajectory_id,
                            session_id=self.session_id,
                            trajectory_path=self._logical_child_trajectory_ref(completed.path),
                        )
                    ],
                )
            )
        extra: dict[str, Any] = {
            "daydream_phase": dispatch.phase.value,
            "daydream_run_flow": self.run_flow.value,
            "dispatch_id": dispatch.dispatch_id,
            "dispatch_started_at": dispatch.started_at,
            "dispatch_completed_at": dispatch.completed_at,
            "planned_count": dispatch.planned_count,
            "attempted_count": dispatch.attempted_count,
            "completed_count": dispatch.completed_count,
            "dispatch_status": dispatch.status.value,
        }
        if dispatch.reason_code is not None:
            extra["reason_code"] = dispatch.reason_code.value
        redacted_extra = redact_value(extra)
        if not isinstance(redacted_extra, dict):
            raise TypeError("dispatch redaction returned a non-object")
        json.dumps(redacted_extra, allow_nan=False)
        step = Step(
            step_id=self._next_step_id(),
            timestamp=dispatch.started_at,
            source="agent",
            model_name=self.agent_model_name,
            message=f"Dispatching {dispatch.attempted_count} parallel {dispatch.phase.value} tasks",
            observation=Observation(results=results),
            llm_call_count=0,
            extra=redacted_extra,
        )
        self._extend_steps([self.redactor.redact_step(step)])

    def build_trajectory(
        self,
        steps: list[Step] | None = None,
        *,
        snapshot_at: str | None = None,
    ) -> Trajectory:
        if steps is None:
            steps = self.steps
        version = daydream.__version__
        final_metrics_extra: dict[str, Any] | None = None
        if self._folded_fork_totals:
            # Successful forks contribute usage, but steps stay document-local and cache tokens stay a prompt subset.
            final_metrics_extra = {
                "daydream_metric_scope": "whole_run_including_forks",
                "total_steps_scope": "local_trajectory",
            }

        final_metrics = FinalMetrics(
            total_prompt_tokens=self._final_totals["prompt"] or None,
            total_completion_tokens=self._final_totals["completion"] or None,
            total_cached_tokens=self._final_totals["cached"] or None,
            total_cost_usd=(self._final_totals["cost"] if self._final_totals["any_cost_seen"] else None),
            total_steps=len(steps),
            extra=final_metrics_extra,
        )
        extra: dict[str, Any] = {"target_dir": str(self.target_dir)}
        if self._run_started_at:
            extra["run_started_at"] = self._run_started_at
        if snapshot_at is not None:
            extra["snapshot_at"] = snapshot_at
        elif self._run_ended_at:
            extra["run_ended_at"] = self._run_ended_at
        if self.backend_name:
            # Persist representative and set per-phase backend labels like the manifest; omitted phases have no keys.
            extra["backend"] = self.backend_name
            if self.review_backend_name:
                extra["review_backend"] = self.review_backend_name
            if self.fix_backend_name:
                extra["fix_backend"] = self.fix_backend_name
            if self.test_backend_name:
                extra["test_backend"] = self.test_backend_name
        if self.pr_number is not None:
            extra["pr_number"] = self.pr_number
        if self.pr_repo is not None:
            extra["pr_repo"] = self.pr_repo
        if self._profile is not None:
            extra.update(self._profile)
        if self._phase_events:
            extra["phase_events"] = [e.to_dict() for e in self._phase_events]
        if self._subtrajectories:
            summaries = redact_value([dict(s) for s in self._subtrajectories])
            if isinstance(summaries, list):
                extra["subtrajectories"] = summaries
        # Document IDs qualify root session identity with fork descriptors, keeping sibling trajectories unique.
        return Trajectory(
            schema_version="ATIF-v1.7",
            session_id=self.session_id,
            trajectory_id=self.trajectory_id,
            agent=Agent(name="daydream", version=version, model_name=self.agent_model_name),
            steps=list(steps),
            final_metrics=final_metrics,
            extra=extra,
        )

    def _write(self) -> None:
        registry = self._signal_registry
        if self.parent is None and registry is not None:
            registry.write_final(self)
            return
        document = self._prepare_document(
            status="complete",
            cutoff_at=self._run_ended_at or timeutil.now_iso(),
        )
        if document is None:
            return
        self._write_document(document, "complete")
        if registry is not None:
            registry.retain(document)

    def _write_document(self, document: TrajectoryDocumentSnapshot, status: Literal["complete", "partial"]) -> None:
        """Write one frozen document through the host sink, or straight to disk."""
        if self.document_writer is not None:
            self.document_writer(document, status)
        elif status == "complete":
            atomic_write_json(document.path, json.loads(document.json_bytes))
        else:
            document.path.parent.mkdir(parents=True, exist_ok=True)
            document.path.write_text(document.json_bytes.decode("utf-8"), encoding="utf-8")

    def _partial_state_key(self) -> str:
        """Return a stable digest of this recorder's current partial state."""
        state = {
            "steps": [step.model_dump(mode="json") for step in self._snapshot_in_flight_steps()],
            "phase_events": [event.to_dict() for event in self._phase_events],
            "subtrajectories": self._subtrajectories,
            "active_invocations": [
                {
                    "invocation_id": invocation.invocation_id,
                    "phase": invocation.phase.value,
                    "started_at": invocation.started_at,
                }
                for invocation in self._active_invocations
            ],
            "aborted": self._aborted,
        }
        encoded = json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _prepare_document(
        self,
        *,
        status: Literal["complete", "partial"],
        cutoff_at: str,
        allow_empty_root: bool = False,
    ) -> TrajectoryDocumentSnapshot | None:
        """Freeze one canonical document without performing any filesystem write."""
        steps = self.steps if status == "complete" else self._snapshot_in_flight_steps()
        if not steps:
            if self.parent is not None or not allow_empty_root:
                return None
            # ATIF needs a step: early signals/host-only runs get an immutable system event, never a
            # fabricated invocation or live mutation.
            partial = status == "partial"
            steps = [
                Step(
                    step_id=1,
                    timestamp=cutoff_at,
                    source="system",
                    message="Daydream run snapshot" if partial else "Daydream host-only run snapshot",
                    extra={
                        "daydream_run_flow": self.run_flow.value,
                        "host_event": "partial_snapshot" if partial else "host_only_final_snapshot",
                    },
                )
            ]
        trajectory = self.build_trajectory(
            steps=list(steps),
            snapshot_at=cutoff_at if status == "partial" else None,
        )
        payload = trajectory.to_json_dict()
        if status == "partial" and self._active_invocations:
            extra = payload.setdefault("extra", {})
            summaries = list(extra.get("subtrajectories", []))
            summaries.extend(invocation.summary(partial=True) for invocation in self._active_invocations)
            extra["subtrajectories"] = summaries
        if self._aborted or status == "partial":
            payload.setdefault("extra", {})["partial"] = True
        path = self.path
        if status == "partial":
            path = partial_document_path(path)
        return TrajectoryDocumentSnapshot(
            trajectory_id=self.trajectory_id,
            path=path,
            json_bytes=json.dumps(payload, indent=2).encode("utf-8"),
        )

    def _snapshot_in_flight_steps(self) -> list[Step]:
        """Merge active/flushed steps by ID without mutation, preserving sequential ATIF IDs."""
        if not self._active_invocations:
            return list(self.steps)
        snapshot = list(self.steps)
        next_id = self._step_id_counter + 1
        for inv in self._active_invocations:
            snapshot.extend(inv.snapshot_steps(snapshot_step_id=next_id))
            if inv._open_step_dict is not None:
                next_id += 1
        snapshot.sort(key=lambda s: s.step_id)
        return snapshot

    def write_partial(self) -> None:
        """Flush each active recorder owned by the run once; never raise.

        The run registry deduplicates shared roots. Unentered recorders are ignored."""
        if self._signal_registry is not None:
            self._signal_registry.flush_active()
