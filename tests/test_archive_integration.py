"""Recorder callback, strict archive round-trip, and CLI archive/evaluation integration."""

from __future__ import annotations

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
)
from tests.harness.trajectory import make_recorder, read_trajectory


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
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as invocation:
            invocation.observe_user_step("test prompt")
    traj_path = tmp_path / ".daydream" / RUN_DOCUMENT_NAME
    assert traj_path.exists()
    data = read_trajectory(traj_path)
    assert data["session_id"] == recorder.session_id
    assert len(data["steps"]) == 1
