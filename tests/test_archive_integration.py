"""Recorder callback, strict archive round-trip, and CLI archive/evaluation integration."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from daydream.archive import finalize_archive_run
from daydream.archive.manifest import archive_recorder_provenance_from_snapshot
from daydream.artifact_visibility import (
    ArtifactEvidenceProvenance,
    ArtifactTreeSnapshot,
)
from daydream.artifacts.filesystem import manifest_tree
from daydream.run_snapshot import (
    ArchiveRunSnapshot,
    ManifestRunIdentity,
    RunPhaseCapabilities,
)
from daydream.trajectory import (
    RUN_DOCUMENT_NAME,
    DaydreamPhase,
    DaydreamRunFlow,
    EvidenceRecorder,
    RunWriteSnapshot,
    TrajectoryDocumentReference,
    now_iso,
)
from daydream.trajectory.journal import JournalWriter
from daydream.trajectory.view import OPEN_EVENT, PHASE_EVENT, SNAPSHOT_EVENT, STEP_EVENT, SUBTRAJECTORY_EVENT
from tests.harness.trajectory import make_recorder, read_trajectory


def _add_user_step(recorder: EvidenceRecorder) -> None:
    """Append one minimal user step to the recorder's evidence document."""
    recorder._extend_steps([{"step_id": recorder._next_step_id(), "timestamp": now_iso(), "source": "user",
        "message": "test prompt",
        "extra": {"daydream_phase": DaydreamPhase.REVIEW.value, "daydream_run_flow": DaydreamRunFlow.NORMAL.value},
    }])


def evidence_reference(payload: Mapping[str, Any], private_path: Path) -> TrajectoryDocumentReference:
    """Write a document view's content as evidence records at *private_path* and reference the whole file.

    Steps without a ``step_id`` take their 1-based position.
    ``extra.snapshot_at`` leaves the document unsealed behind a snapshot record; otherwise it is sealed
    at ``extra.run_ended_at`` (default ``2026-01-01T00:00:01Z``) with the payload's final metrics."""
    extra = dict(payload.get("extra") or {})
    ended = extra.pop("run_ended_at", "2026-01-01T00:00:01Z")
    snapshot_at = extra.pop("snapshot_at", None)
    phase_events = extra.pop("phase_events", [])
    subtrajectories = extra.pop("subtrajectories", [])
    aborted = bool(extra.pop("partial", False))
    metrics = dict(payload.get("final_metrics") or {})
    private_path.parent.mkdir(parents=True, exist_ok=True)
    writer = JournalWriter(private_path, resource={}, version="test")
    writer.append(OPEN_EVENT, {
        "session_id": payload["session_id"], "trajectory_id": payload["trajectory_id"], "descriptor": "",
        "parent_trajectory_id": None, "run_flow": "normal", "agent_version": "test",
        "agent_model_name": (payload.get("agent") or {}).get("model_name", ""), "extra": extra,
    })
    for index, step in enumerate(payload.get("steps") or [], start=1):
        writer.append(STEP_EVENT, {"step_id": index, **step})
    for event in phase_events:
        writer.append(PHASE_EVENT, event)
    for summary in subtrajectories:
        writer.append(SUBTRAJECTORY_EVENT, summary)
    if snapshot_at is None:
        writer.seal({"run_ended_at": ended, "aborted": aborted, "capture_complete": True, "failed_documents": [],
                     "final_metrics": metrics})
    else:
        writer.append(SNAPSHOT_EVENT, {"snapshot_at": snapshot_at, "final_metrics": metrics})
    writer.close()
    size, sha256 = writer.reference()
    return TrajectoryDocumentReference(payload["trajectory_id"], private_path, private_path, size, sha256)


def _finalize_strict_archive(recorder: EvidenceRecorder, snapshot: RunWriteSnapshot, config: Any, target: Path, *,
    destinations: tuple[Any, ...] = (), dump_path: Path | None = None,
) -> None:
    """Run strict finalization; the re-attested target tree must exclude the archive directory."""
    session_id = recorder.session_id
    finalize_archive_run(run=ArchiveRunSnapshot(recorder_provenance=archive_recorder_provenance_from_snapshot(
                write_snapshot=snapshot, run_flow=recorder.run_flow,
            ),
            identity=ManifestRunIdentity(model=None, backend="claude", review_backend=None, review_model=None,
                fix_backend="claude" if recorder.run_flow is DaydreamRunFlow.NORMAL else None,
                test_backend="claude" if recorder.run_flow is DaydreamRunFlow.NORMAL else None,
                review_only=False,
                deep=recorder.run_flow is DaydreamRunFlow.NORMAL, profile=None,
                phases=RunPhaseCapabilities(merge=recorder.run_flow is DaydreamRunFlow.NORMAL,
                    fix=recorder.run_flow is DaydreamRunFlow.NORMAL, test=recorder.run_flow is DaydreamRunFlow.NORMAL,
                    push=False, remote_ci=False,
                ),
            ), trajectories=snapshot,
        ),
        artifacts=ArtifactTreeSnapshot(
            session_id=session_id, workspace_key="workspace", root=target, manifest=manifest_tree(target),
            destinations=destinations,
        ),
        artifact_provenance=ArtifactEvidenceProvenance(
            workspace_key="workspace", session_id=session_id, public_source=target,
            # Live-root-relative routes retain their paths inside the frozen copy.
            live_root=target,
        ), config=config, work=None, dump_path=dump_path,
    )


async def test_on_write_failure_does_not_raise(tmp_path: Path) -> None:
    def on_write_boom(recorder: EvidenceRecorder, snapshot: RunWriteSnapshot,) -> None:
        raise RuntimeError("archive exploded")
    recorder = make_recorder(tmp_path, on_write=on_write_boom)
    async with recorder:
        _add_user_step(recorder)
    traj_path = tmp_path / ".daydream" / RUN_DOCUMENT_NAME
    assert traj_path.exists()
    data = read_trajectory(traj_path)
    assert data["session_id"] == recorder.session_id
    assert len(data["steps"]) == 1
