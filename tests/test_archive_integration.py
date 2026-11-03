"""Recorder callback, strict archive round-trip, and CLI archive/evaluation integration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from daydream.archive import finalize_archive_run
from daydream.archive.manifest import archive_recorder_provenance_from_snapshot
from daydream.artifact_visibility import (
    ArtifactEvidenceProvenance,
    ArtifactTreeSnapshot,
)
from daydream.artifacts.filesystem import manifest_tree
from daydream.atif import Step
from daydream.run_snapshot import (
    ArchiveRunSnapshot,
    ManifestRunIdentity,
    RunPhaseCapabilities,
)
from daydream.trajectory import (
    DaydreamPhase,
    DaydreamRunFlow,
    RunWriteSnapshot,
    TrajectoryRecorder,
    now_iso,
)
from tests.harness.trajectory import make_recorder


def _add_user_step(recorder: TrajectoryRecorder) -> None:
    """Append a minimal user Step so the recorder has at least one step and won't skip _write."""
    step = Step(step_id=recorder._next_step_id(), timestamp=now_iso(), source="user", message="test prompt",
        extra={"daydream_phase": DaydreamPhase.REVIEW.value, "daydream_run_flow": DaydreamRunFlow.NORMAL.value,},
    )
    recorder.steps.append(step)


def _finalize_strict_archive(recorder: TrajectoryRecorder, snapshot: RunWriteSnapshot, config: Any, target: Path, *,
    destinations: tuple[Any, ...] = (), dump_path: Path | None = None,
) -> None:
    """Run strict finalization; the re-attested target tree must exclude the archive directory."""
    session_id = recorder.session_id
    finalize_archive_run(run=ArchiveRunSnapshot(recorder_provenance=archive_recorder_provenance_from_snapshot(
                write_snapshot=snapshot, run_flow=recorder.run_flow,
            ),
            identity=ManifestRunIdentity(skill="python", model=None, backend="claude", review_backend=None,
                fix_backend="claude" if recorder.run_flow is DaydreamRunFlow.NORMAL else None,
                test_backend="claude" if recorder.run_flow is DaydreamRunFlow.NORMAL else None,
                per_stack_review_backend=None, per_stack_review_model=None, review_only=False,
                deep=recorder.run_flow is DaydreamRunFlow.NORMAL, profile=None,
                phases=RunPhaseCapabilities(per_stack_review=False, merge=recorder.run_flow is DaydreamRunFlow.NORMAL,
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
    def on_write_boom(recorder: TrajectoryRecorder, snapshot: RunWriteSnapshot,) -> None:
        raise RuntimeError("archive exploded")
    recorder = make_recorder(tmp_path, on_write=on_write_boom)
    async with recorder:
        _add_user_step(recorder)
    traj_path = tmp_path / ".daydream" / "trajectory.json"
    assert traj_path.exists()
    data = json.loads(traj_path.read_text(encoding="utf-8"))
    assert data["session_id"] == recorder.session_id
    assert len(data["steps"]) == 1
