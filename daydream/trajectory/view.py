"""Reduce one evidence journal to its document view.

The view is the compact, Daydream-owned reading of a document that archive,
evaluation, dataset and corpus consumers share::

    {"session_id", "trajectory_id", "parent_trajectory_id", "run_flow",
     "agent": {"name", "version", "model_name"}, "steps": [...], "final_metrics": {...}, "extra": {...}}

``extra`` carries the run identity and lifecycle evidence (``phase_events``,
``subtrajectories``, profile and backend identity, ``run_started_at`` and
``run_ended_at`` or ``snapshot_at``). A document without a verified seal, or whose
seal reports incomplete capture, reads as ``extra.partial``. The view never invents
evidence: an unsealed document without a snapshot has no end time and no totals.
"""

from __future__ import annotations

from typing import Any

from daydream.trajectory.billing import GENERIC_MODEL_LABELS
from daydream.trajectory.journal import JournalContents, read_journal

OPEN_EVENT = "daydream.evidence.open"
STEP_EVENT = "daydream.transcript.step"
MODEL_EVENT = "daydream.agent.model"
PHASE_EVENT = "daydream.phase_event"
SUBTRAJECTORY_EVENT = "daydream.subtrajectory"
PROFILE_EVENT = "daydream.profile"
SNAPSHOT_EVENT = "daydream.evidence.snapshot"


def _upgrade(step: dict[str, Any], model_name: str) -> dict[str, Any]:
    if step.get("source") == "agent" and (step.get("model_name") or "") in GENERIC_MODEL_LABELS:
        return {**step, "model_name": model_name}
    return step


def reduce_journal(contents: JournalContents) -> dict[str, Any]:
    """Reduce decoded, verified records to the document view."""
    records = contents.records
    if not records or records[0].event_name != OPEN_EVENT or not isinstance(records[0].body, dict):
        raise ValueError("evidence document has no open record")
    opened: dict[str, Any] = records[0].body
    model_name = opened.get("agent_model_name") or ""
    steps: dict[int, dict[str, Any]] = {}
    phase_events: list[dict[str, Any]] = []
    subtrajectories: list[dict[str, Any]] = []
    profile: dict[str, Any] | None = None
    snapshot: dict[str, Any] | None = None
    finished_after_snapshot: set[str] = set()
    for record in records[1:]:
        body = record.body
        name = record.event_name
        if name == STEP_EVENT:
            steps[body["step_id"]] = body
        elif name == MODEL_EVENT:
            model_name = body["model_name"]
            steps = {step_id: _upgrade(step, model_name) for step_id, step in steps.items()}
        elif name == PHASE_EVENT:
            phase_events.append(body)
        elif name == SUBTRAJECTORY_EVENT:
            subtrajectories.append(body)
            if snapshot is not None and isinstance(body.get("invocation_id"), str):
                finished_after_snapshot.add(body["invocation_id"])
        elif name == PROFILE_EVENT:
            profile = body
        elif name == SNAPSHOT_EVENT:
            snapshot, finished_after_snapshot = body, set()
    seal = contents.seal.body if contents.seal is not None else None

    extra: dict[str, Any] = dict(opened.get("extra") or {})
    partial = seal is None or not contents.complete or bool(seal.get("aborted")) \
        or seal.get("capture_complete") is False
    active: list[dict[str, Any]] = []
    if seal is not None:
        extra["run_ended_at"] = seal["run_ended_at"]
        if seal.get("capture_complete") is False:
            extra["snapshot_at"] = seal["run_ended_at"]
        final_metrics = dict(seal["final_metrics"])
    elif snapshot is not None:
        extra["snapshot_at"] = snapshot["snapshot_at"]
        for invocation in snapshot.get("active_invocations", []):
            if invocation["summary"].get("invocation_id") in finished_after_snapshot:
                continue
            for step in invocation["steps"]:
                steps.setdefault(step["step_id"], step)
            active.append(invocation["summary"])
        active.extend(snapshot.get("active_forks", []))
        final_metrics = {**snapshot["final_metrics"], "total_steps": len(steps)}
        if snapshot.get("active_dispatches"):
            extra["active_dispatches"] = snapshot["active_dispatches"]
    else:
        final_metrics = {"total_steps": len(steps)}
    if profile is not None:
        extra.update(profile)
    if phase_events:
        extra["phase_events"] = phase_events
    if subtrajectories or active:
        extra["subtrajectories"] = [*subtrajectories, *active]
    if partial:
        extra["partial"] = True
    return {
        "session_id": opened["session_id"],
        "trajectory_id": opened["trajectory_id"],
        "parent_trajectory_id": opened.get("parent_trajectory_id"),
        "run_flow": opened.get("run_flow"),
        "agent": {"name": "daydream", "version": opened["agent_version"], "model_name": model_name},
        "steps": [steps[step_id] for step_id in sorted(steps)],
        "final_metrics": final_metrics,
        "extra": extra,
    }


def read_document_view(raw: bytes) -> dict[str, Any]:
    """Decode, verify and reduce one evidence document's bytes."""
    return reduce_journal(read_journal(raw))
