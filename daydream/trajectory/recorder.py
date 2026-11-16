"""Run evidence recorder: scope ownership and durable local OpenTelemetry evidence.

runner binds one recorder through a ContextVar; agent opens an Invocation per
call. Each recorder owns one evidence document: an append-only OTLP/JSON log
journal (:mod:`daydream.trajectory.journal`) in the run's private directory.
Finished invocation steps, phase and dispatch lifecycle, invocation and fork
summaries, and profile identity are appended as they happen, so no whole-run
transcript stays in memory. Forks own child documents. Closing a document seals
it; a signal flush appends a snapshot and publishes a partial run snapshot of
prefix references. Usage totals live in each document's host-owned
:class:`~daydream.trajectory.billing.UsageLedger`.
"""

from __future__ import annotations

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
from daydream.redaction import _redact_unstructured_text, redact_value
from daydream.trajectory.billing import GENERIC_MODEL_LABELS, UsageLedger
from daydream.trajectory.context import _RECORDER_VAR
from daydream.trajectory.invocation import Invocation
from daydream.trajectory.journal import JournalWriter
from daydream.trajectory.layout import (
    _DAYDREAM_DIRNAME,
    DOCUMENT_SUFFIX,
    RUNS_DIRNAME,
    run_directory,
    sibling_document_path,
)
from daydream.trajectory.lifecycle import DispatchHandle, ForkIdentity, PhaseEvent
from daydream.trajectory.scopes import fork_scope, invocation_scope
from daydream.trajectory.types import (
    DaydreamPhase,
    DaydreamRunFlow,
    LifecycleReasonCode,
    LifecycleStatus,
    RunWriteSnapshot,
    TrajectoryDocumentReference,
    TrajectoryDocumentSnapshot,
)
from daydream.trajectory.view import (
    MODEL_EVENT,
    OPEN_EVENT,
    PHASE_EVENT,
    PROFILE_EVENT,
    SNAPSHOT_EVENT,
    STEP_EVENT,
    SUBTRAJECTORY_EVENT,
)
from daydream.ui import create_console, print_error

_console = create_console()


def _admitted_descriptor(raw: str) -> str:
    """Retain ordinary labels verbatim; admit private labels as distinct opaque identities."""
    if _redact_unstructured_text(raw) == raw:
        return raw
    return 'redacted-' + hashlib.sha256(raw.encode('utf-8')).hexdigest()


def _safe_descriptor(raw: str) -> str:
    """Slugify a descriptor for filenames; reject an empty sanitized result."""
    slug = re.sub(r"[^a-z0-9-]", "-", _admitted_descriptor(raw).lower())
    slug = re.sub(r"-{2,}", "-", slug)
    slug = slug.strip("-")
    if not slug:
        raise ValueError(f"Descriptor {raw!r} produces empty slug after sanitization")
    return slug


class _EvidenceIncomplete(OSError):
    """A document could not be sealed; its accepted prefix remains truthful partial evidence."""


TrajectoryWriteCallback = Callable[["EvidenceRecorder", RunWriteSnapshot], None]
TrajectoryDocumentWriter = Callable[
    [TrajectoryDocumentSnapshot, Literal["complete", "partial"]], TrajectoryDocumentReference
]


class _SignalFlushRegistry:
    """Identity membership for every active recorder owned by one run."""

    def __init__(self) -> None:
        self._active: dict[int, EvidenceRecorder] = {}
        self._completed: dict[str, TrajectoryDocumentReference] = {}
        self.failed_documents: list[str] = []
        # The last partial snapshot and the document state it covered; a repeated
        # flush with no new evidence republishes it with the same cutoff.
        self._last_partial: tuple[str, RunWriteSnapshot] | None = None

    def register(self, recorder: EvidenceRecorder) -> None:
        """Register *recorder* once by object identity."""
        self._active[id(recorder)] = recorder

    def unregister(self, recorder: EvidenceRecorder) -> None:
        """Remove *recorder* when the same identity is still registered."""
        if self._active.get(id(recorder)) is recorder:
            self._active.pop(id(recorder), None)

    def retain(self, reference: TrajectoryDocumentReference) -> None:
        """Retain a completed child's sealed reference until root archival."""
        self._completed[reference.trajectory_id] = reference

    def _root(self) -> EvidenceRecorder | None:
        return next((recorder for recorder in self._active.values() if recorder.parent is None), None)

    def _publish(self, root: EvidenceRecorder, status: Literal["complete", "partial"], cutoff_at: str,
                 live: Sequence[TrajectoryDocumentReference]) -> RunWriteSnapshot:
        documents: dict[str, TrajectoryDocumentReference] = dict(self._completed)
        documents.update((document.trajectory_id, document) for document in live)
        ordered = tuple(sorted(documents.values(), key=lambda document: (
            document.trajectory_id != root.trajectory_id, document.trajectory_id)))
        snapshot = RunWriteSnapshot(status=status, cutoff_at=cutoff_at,
                                    root_trajectory_id=root.trajectory_id, documents=ordered)
        if root.document_writer is not None:
            try:
                reference = next(document for document in ordered if document.trajectory_id == root.trajectory_id)
                root.document_writer(TrajectoryDocumentSnapshot(
                    root.trajectory_id, root.path, reference.read_bytes()), status)
            except Exception as exc:  # noqa: BLE001 - a partial explicit copy never interrupts recording
                if status == "complete":
                    raise
                ui.print_warning(_console, f"Partial evidence output failed: {type(exc).__name__}")
        if root.on_write is not None:
            try:
                root.on_write(root, snapshot)
            except Exception:  # noqa: BLE001 - archive failure never affects recording
                pass
        return snapshot

    def flush_active(self) -> None:
        """Append one common-cutoff snapshot to every live document, then publish their prefixes.

        A root that cannot take its snapshot publishes nothing, so no snapshot is ever
        disconnected from its root. A child that cannot take one keeps its last accepted
        prefix in the published snapshot. Repeating a flush with no new evidence or
        in-flight state republishes the previous snapshot with its cutoff."""
        root = self._root()
        if root is None:
            return
        bodies: list[tuple[EvidenceRecorder, dict[str, Any] | None]] = []
        for recorder in tuple(self._active.values()):
            body = recorder._snapshot_body()
            if body is None and recorder is root:
                return
            bodies.append((recorder, body))
        state = hashlib.sha256(json.dumps([
            *((recorder.trajectory_id, recorder._recorded, body) for recorder, body in bodies),
            *sorted((reference.trajectory_id, reference.size) for reference in self._completed.values()),
        ], sort_keys=True, default=str).encode()).hexdigest()
        if self._last_partial is not None and self._last_partial[0] == state:
            self._publish(root, "partial", self._last_partial[1].cutoff_at, self._last_partial[1].documents)
            return
        cutoff = timeutil.now_iso()
        live: list[TrajectoryDocumentReference] = []
        for recorder, body in bodies:
            appended = body is not None and recorder._journal.append(
                SNAPSHOT_EVENT, {"snapshot_at": cutoff, **body}, blocking=False)
            if not appended and recorder is root:
                return
            if appended:
                recorder._journal.sync()
            live.append(recorder._reference())
        self._last_partial = (state, self._publish(root, "partial", cutoff, live))


# Independently nested roots temporarily select their own run registry. The
# registries themselves hold sibling membership; sibling entry order never
# chooses which recorder a signal flushes.
_ACTIVE_SIGNAL_RUNS: list[_SignalFlushRegistry] = []


def flush_active_signal_recorders() -> None:
    """Synchronously flush every active recorder in the selected run."""
    if _ACTIVE_SIGNAL_RUNS:
        _ACTIVE_SIGNAL_RUNS[-1].flush_active()


@dataclass
class EvidenceRecorder:
    """Own one evidence document, its child recorders and its run's snapshots.

    An async context binds the recorder, seals its document on exit and restores
    the parent context. Steps receive monotonic IDs. Model names begin as backend
    labels and upgrade when native metadata arrives.

    The caller supplies the session identity, root document path and resolved
    backend labels. Non-empty review/fix/test labels are recorded as run identity;
    flows that omit those stages leave the corresponding labels empty. Sensitive
    values are redacted before they are written, with explicit failure markers."""

    path: Path
    run_flow: DaydreamRunFlow
    target_dir: Path
    agent_model_name: str
    session_id: str
    artifact_run_dir: Path | None = None
    document_writer: TrajectoryDocumentWriter | None = None
    parent: EvidenceRecorder | None = None
    descriptor: str = ""
    explicit_path: bool = False
    pr_number: int | None = None
    pr_repo: str | None = None
    immutable_input: dict[str, object] | None = None
    backend_name: str = ""
    review_backend_name: str = ""
    fix_backend_name: str = ""
    test_backend_name: str = ""
    usage: UsageLedger = field(default_factory=UsageLedger)
    _step_id_counter: int = 0
    _phase_scope_counter: int = 0
    _dispatch_counter: int = 0
    _invocation_counter: int = 0
    _step_count: int = 0
    # Evidence records accepted for this document, excluding signal snapshots.
    _recorded: int = 0
    # The phase of the lowest-numbered step, which labels this document's fork summary.
    _first_phase: tuple[int, str] | None = None
    _previous_token: Any = None
    # Active invocations whose in-flight steps have not been appended yet; a
    # signal snapshot materializes them so SIGINT mid-run_agent() keeps partial work.
    _active_invocations: list[Invocation] = field(default_factory=list)
    _active_dispatches: list[DispatchHandle] = field(default_factory=list, repr=False)
    # Compact identified phase events and invocation/fork summaries. They are also
    # appended as records; a fork's summary on its parent flattens them.
    _phase_events: list[PhaseEvent] = field(default_factory=list)
    _subtrajectories: list[dict[str, Any]] = field(default_factory=list)
    _aborted: bool = False
    on_write: TrajectoryWriteCallback | None = None
    _trajectory_id: str = ""
    _run_started_at: str = ""
    _run_ended_at: str = ""
    _writer: JournalWriter | None = field(default=None, init=False, repr=False, compare=False)
    _pending: list[tuple[str, Any, dict[str, Any] | None]] = field(default_factory=list, init=False, repr=False)
    _signal_registry: _SignalFlushRegistry | None = field(default=None, init=False, repr=False, compare=False)

    async def __aenter__(self) -> "EvidenceRecorder":
        self._run_started_at = timeutil.now_iso()
        self._run_ended_at = ""
        registry = _SignalFlushRegistry()
        self._signal_registry = registry
        registry.register(self)
        _ACTIVE_SIGNAL_RUNS.append(registry)
        self._previous_token = _RECORDER_VAR.set(self)
        self._open_document()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, _exc_tb: Any) -> None:
        try:
            if exc_type is not None:
                self._aborted = True
            self._run_ended_at = timeutil.now_iso()
            self._write_final()
        except Exception as exc:  # noqa: BLE001 - branch on explicit_path per D-06
            if isinstance(exc, _EvidenceIncomplete) and not self.explicit_path:
                return  # The journal already warned once; the run snapshot reports partial evidence.
            if self.explicit_path:
                # D-06: user asked for it, deliver or fail loud
                print_error(_console, "Evidence write failed", f"{type(exc).__name__}: {exc}")
                if exc_val is not None:
                    # An in-flight failure owns the exit; do not mask it with SystemExit.
                    exc_val.add_note(f"evidence finalization also failed ({type(exc).__name__})")
                    return
                raise SystemExit(2) from exc
            # Implicit/default path — degrade with warning per CORE-09 / D-11
            ui.print_warning(_console, f"Evidence write failed: {type(exc).__name__}: {exc}")
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
        """Open an invocation scope whose steps are appended on exit."""
        return invocation_scope(self, phase)

    @property
    def trajectory_id(self) -> str:
        """Return this recorder's stable, document-qualified identity."""
        if self._trajectory_id:
            return self._trajectory_id
        if self.descriptor:
            return f"{self.session_id}:{self.descriptor}"
        return self.session_id

    # -- evidence records -------------------------------------------------

    def _open_document(self) -> None:
        """Create this document's journal and append its identity record."""
        writer = self._writer = JournalWriter(self.path, version=daydream.__version__, resource={
            "service.name": "daydream", "service.version": daydream.__version__,
            "daydream.session.id": self.session_id, "daydream.trajectory.id": self.trajectory_id,
        })
        extra: dict[str, Any] = {"target_dir": str(self.target_dir), "run_started_at": self._run_started_at}
        if self.backend_name:
            # A representative backend plus per-phase keys, each only when the flow runs it.
            extra["backend"] = self.backend_name
            for key, value in (("review_backend", self.review_backend_name),
                               ("fix_backend", self.fix_backend_name), ("test_backend", self.test_backend_name)):
                if value:
                    extra[key] = value
        if self.pr_number is not None:
            extra["pr_number"] = self.pr_number
        if self.pr_repo is not None:
            extra["pr_repo"] = self.pr_repo
        if self.immutable_input is not None:
            extra["immutable_input"] = self.immutable_input
        writer.append(OPEN_EVENT, {
            "session_id": self.session_id, "trajectory_id": self.trajectory_id, "descriptor": self.descriptor,
            "parent_trajectory_id": None if self.parent is None else self.parent.trajectory_id,
            "run_flow": self.run_flow.value, "agent_version": daydream.__version__,
            "agent_model_name": self.agent_model_name, "extra": extra,
        }, {"daydream.flow": self.run_flow.value})
        pending, self._pending = self._pending, []
        for event_name, body, attributes in pending:
            writer.append(event_name, body, attributes)

    def _record(self, event_name: str, body: Any, attributes: dict[str, Any] | None = None) -> None:
        self._recorded += 1
        if self._writer is None:
            # Identity recorded before the run enters (its profile) follows the open record.
            self._pending.append((event_name, body, attributes))
            return
        self._writer.append(event_name, body, attributes)

    @property
    def _journal(self) -> JournalWriter:
        if self._writer is None:
            raise RuntimeError("evidence document is not open")
        return self._writer

    def _reference(self) -> TrajectoryDocumentReference:
        size, sha256 = self._journal.reference()
        return TrajectoryDocumentReference(self.trajectory_id, self.path, self.path, size, sha256)

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
        """Append one :class:`PhaseEvent` stamped with ``timeutil.now_iso()``."""
        phase_event = PhaseEvent(phase=phase, event=event, timestamp=timeutil.now_iso(), session_id=session_id,
                                 scope_id=scope_id, status=status, reason_code=reason_code, metadata=metadata)
        if scope_id is not None:
            self._phase_events.append(phase_event)
        self._record(PHASE_EVENT, phase_event.to_dict(), {"daydream.phase": phase.value, "daydream.event": event})

    def record_profile(self, *, schema_version: int, name: str, source_kind: str, digest: str) -> None:
        """Record the resolved policy name, source, and canonical digest for attribution."""
        self._record(PROFILE_EVENT, {
            "profile_schema_version": schema_version, "profile_name": name,
            "profile_source_kind": source_kind, "profile_digest": digest,
        })

    def emit_repair_outcome(
        self,
        *,
        execution_id: str,
        outcome: str,
        abort_reason: str | None,
        repair_reason_code: str | None,
        changed_paths: tuple[str, ...] = (),
    ) -> None:
        """Record one bounded repair turn's host outcome and its concise reason codes.

        Names and counts only: the scan needs to tell an interrupted repair from a
        completed one without carrying the turn's prose. The converged public
        reason travels as metadata, never as the lifecycle ``reason_code``.
        """
        self._emit_phase_event(
            DaydreamPhase.TEST, "repair_outcome", execution_id=execution_id, outcome=outcome,
            abort_reason=abort_reason, repair_reason_code=repair_reason_code, changed_paths=list(changed_paths),
        )

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
            "file": file, "reason": reason, "items_processed": items_processed, "items_skipped": items_skipped,
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
        partial_edit_handling: str | None = None,
    ) -> None:
        """Record an invocation stop once, using durations instead of monotonic deadlines.

        Retry stops carry retry-only attempts, backend time, backoff, reason and circuit
        state. Initial useful work is excluded. Partial edits are
        kept for an interrupted attempt and discarded when stopping before dispatch."""
        self._emit_phase_event(
            phase, "agent_budget_stop", limit_expired=limit_expired, elapsed_s=elapsed_s, backend_s=backend_s,
            backoff_s=backoff_s, attempts=attempts, cleanup_elapsed_s=cleanup_elapsed_s,
            retry_stop_reason=retry_stop_reason, circuit_state=circuit_state,
            partial_edit_handling=partial_edit_handling,
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
        self._emit_phase_event(DaydreamPhase.RECON, "command_validation", counts={
            "total_candidates": total_candidates, "accepted": accepted, "rejected": rejected,
        }, reasons=dict(sorted(reasons.items())))

    def _register_subtrajectory(self, inv: Invocation) -> None:
        """Append one finished invocation's identity, timing and generation billing summary."""
        self._append_summary(inv.summary())

    def _append_summary(self, summary: dict[str, Any]) -> None:
        safe = redact_value(summary)
        if not isinstance(safe, dict):
            raise ValueError("summary redaction returned a non-object")
        self._subtrajectories.append(safe)
        self._record(SUBTRAJECTORY_EVENT, safe, {"daydream.invocation.id": safe.get("invocation_id")})

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
        summaries = [{**event.to_dict(), "trajectory_id": self.trajectory_id} for event in self._phase_events]
        for summary in self._subtrajectories:
            nested = summary.get("phase_events")
            if isinstance(nested, list):
                summaries.extend(dict(item) for item in nested if isinstance(item, dict))
        return summaries

    def _register_fork_subtrajectory(
        self,
        *,
        child: "EvidenceRecorder",
        phase: str,
        identity: ForkIdentity | None,
        started_at: str,
        ended_at: str,
        sibling_trajectory_ref: str,
    ) -> None:
        """Append a per-fork timing summary to this (parent) document."""
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
        self._append_summary(summary)

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

    def _extend_steps(self, steps: list[dict[str, Any]]) -> None:
        """Append finished steps in ID order; readers order concurrent invocations by ID."""
        for step in sorted(steps, key=lambda item: item["step_id"]):
            attributes: dict[str, Any] = {"daydream.step.id": step["step_id"], "daydream.step.source": step["source"]}
            if step["source"] == "agent":
                attributes.update({"gen_ai.operation.name": "chat", "gen_ai.response.model": step.get("model_name")})
                metrics = step.get("metrics") or {}
                for key, name in (("prompt_tokens", "gen_ai.usage.input_tokens"),
                                  ("completion_tokens", "gen_ai.usage.output_tokens"),
                                  ("cached_tokens", "gen_ai.usage.cache_read.input_tokens"),
                                  ("cost_usd", "gen_ai.usage.cost")):
                    attributes[name] = metrics.get(key)
            self._record(STEP_EVENT, step, attributes)
            self._step_count += 1
            phase = (step.get("extra") or {}).get("daydream_phase")
            if isinstance(phase, str) and (self._first_phase is None or step["step_id"] < self._first_phase[0]):
                self._first_phase = (step["step_id"], phase)

    def _upgrade_model_name(self, candidate: str) -> None:
        """Replace provisional backend labels with the first native model name.

        Already appended agent steps upgrade too: a backend may report its model after TurnEnd."""
        if candidate and (self.agent_model_name or "") in GENERIC_MODEL_LABELS:
            self.agent_model_name = candidate
            self._record(MODEL_EVENT, {"model_name": candidate}, {"gen_ai.response.model": candidate})

    def _sibling_path_for(self, descriptor: str, identity: ForkIdentity | None = None) -> Path:
        """Keep sibling documents in the parent run directory.

        Legacy forks use <slug>; identified forks append a unique identity digest so
        repeated semantic descriptors never replace each other's documents."""
        slug = _safe_descriptor(descriptor)
        if identity is not None:
            identity_digest = hashlib.sha256(identity.fork_id.encode("utf-8")).hexdigest()
            slug = f"{slug[:80]}--{identity_digest}"
        run_dir = self.artifact_run_dir
        if run_dir is None:
            run_dir = run_directory(self.target_dir / _DAYDREAM_DIRNAME, self.session_id)
        return sibling_document_path(run_dir, f"{slug}{DOCUMENT_SUFFIX}")

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
    ) -> AbstractAsyncContextManager[EvidenceRecorder]:
        """Create a parallel child recorder with a semantic descriptor such as fix-0."""
        return fork_scope(self, descriptor, dispatch)

    def _create_dispatch_step(self, dispatch: DispatchHandle) -> None:
        """Append one identified, start-stamped deterministic dispatch step."""
        results = [{
            "content": f"Dispatched to {completed.identity.descriptor}",
            "subagent_trajectory_ref": [{
                "trajectory_id": completed.trajectory_id,
                "session_id": self.session_id,
                "trajectory_path": self._logical_child_trajectory_ref(completed.path),
            }],
        } for completed in dispatch._ordered_completed()]
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
        step: dict[str, Any] = {
            "step_id": self._next_step_id(),
            "timestamp": dispatch.started_at,
            "source": "agent",
            "model_name": self.agent_model_name,
            "message": f"Dispatching {dispatch.attempted_count} parallel {dispatch.phase.value} tasks",
            "observation": {"results": results},
            "llm_call_count": 0,
            "extra": redacted_extra,
        }
        self._extend_steps([step])

    # -- snapshots and sealing ------------------------------------------------

    def _partial_dispatch_summaries(self) -> list[dict[str, Any]]:
        """Retain planned and queued host work without a terminal dispatch decision."""
        summaries = [{
            "dispatch_id": dispatch.dispatch_id,
            "phase": dispatch.phase.value,
            "started_at": dispatch.started_at,
            "ended_at": None,
            "planned_descriptors": list(dispatch.descriptors),
            "planned_count": dispatch.planned_count,
            "attempted_count": dispatch.attempted_count,
            "completed_count": dispatch.completed_count,
            "queued_descriptors": [descriptor for index, descriptor in enumerate(dispatch.descriptors)
                                   if index not in dispatch._primary_index.values()],
            "forks": [{"fork_id": fork.fork_id, "descriptor": fork.descriptor, "ordinal": fork.ordinal}
                      for fork in dispatch._forks],
            "completed_fork_ids": sorted(dispatch._completed),
        } for dispatch in self._active_dispatches]
        safe = redact_value(summaries)
        if not isinstance(safe, list):
            raise ValueError("dispatch summary redaction returned a non-array")
        return safe

    def _snapshot_body(self) -> dict[str, Any] | None:
        """In-flight invocations, dispatches and child forks; ``None`` when they cannot be captured."""
        try:
            next_id = self._step_id_counter + 1
            active: list[dict[str, Any]] = []
            for invocation in self._active_invocations:
                active.append({"summary": invocation.summary(partial=True),
                               "steps": invocation.snapshot_steps(snapshot_step_id=next_id)})
                if invocation._open_step_dict is not None:
                    next_id += 1
            registry = self._signal_registry
            forks: list[dict[str, Any]] = [] if registry is None else [{
                "trajectory_id": child.trajectory_id, "descriptor": child.descriptor,
                "started_at": child._run_started_at, "ended_at": None,
                "sibling_trajectory_ref": self._logical_child_trajectory_ref(child.path), "invocations": [],
                **({"fork_id": child.trajectory_id, "dispatch_id": child.trajectory_id.rsplit(":fork:", 1)[0]}
                   if ":fork:" in child.trajectory_id else {}),
            } for child in registry._active.values() if child.parent is self]
            body = redact_value({
                "active_invocations": active, "active_dispatches": self._partial_dispatch_summaries(),
                "active_forks": forks, "final_metrics": self.usage.final_metrics(self._step_count),
            })
            if not isinstance(body, dict):
                raise TypeError("snapshot redaction returned a non-object")
            return body
        except Exception as exc:  # noqa: BLE001 - a signal snapshot is best effort, never a crash
            ui.print_warning(_console, f"Partial evidence snapshot failed: {type(exc).__name__}")
            return None

    def _seal(self, *, failed_documents: Sequence[str] = ()) -> bool:
        """Seal this document with its end time, totals and capture completeness."""
        return self._journal.seal({
            "run_ended_at": self._run_ended_at or timeutil.now_iso(),
            "aborted": self._aborted,
            "capture_complete": not failed_documents,
            "failed_documents": list(failed_documents),
            "final_metrics": self.usage.final_metrics(self._step_count),
        })

    def _write_final(self) -> None:
        registry = self._signal_registry
        if self.parent is not None or registry is None:
            if not self._seal():
                raise _EvidenceIncomplete(f"evidence document is incomplete ({self._journal.failure or 'not sealed'})")
            return
        failed = list(registry.failed_documents)
        if not self._seal(failed_documents=failed):
            # The root is unsealed: publish its accepted prefix as truthful partial evidence.
            registry._publish(self, "partial", self._run_ended_at, [self._reference()])
            raise _EvidenceIncomplete(f"evidence document is incomplete ({self._journal.failure or 'not sealed'})")
        registry._publish(self, "partial" if failed else "complete", self._run_ended_at, [self._reference()])

    def completed_child_documents(self) -> tuple[TrajectoryDocumentReference, ...]:
        """Return the run's completed child references in trajectory-id order."""
        registry = self._signal_registry
        if registry is None:
            raise RuntimeError("recorder is not active")
        return tuple(reference for _, reference in sorted(registry._completed.items()))

    def current_document(self) -> TrajectoryDocumentReference:
        """Reference this live document's accepted prefix."""
        return self._reference()

    def write_partial(self) -> None:
        """Flush each active recorder owned by the run once; never raise.

        The run registry deduplicates shared roots. Unentered recorders are ignored."""
        if self._signal_registry is not None:
            self._signal_registry.flush_active()
