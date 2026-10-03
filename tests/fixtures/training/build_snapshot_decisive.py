"""Extend the fake-Hub snapshot with accepted, rejected, and unanswered sessions; the label step
resolves the last. Preserve IDs, manifests, curation, and Hub wiring, recomputing evidence digests
under the shared serializer and retaining native profile/stack fields.
"""

from __future__ import annotations

from tests.fixtures.training.build_hub_snapshot import (
    REPO_ID,
    SNAPSHOT_REVISION,
    _snapshot_files,
    _snapshot_trajectory,
)
from tests.harness.hub import FakeHub

__all__ = ["REPO_ID", "SNAPSHOT_REVISION", "build_snapshot_decisive"]

_DECISIVE_DISPOSITIONS = {"sess-a": "accepted", "sess-b": "rejected", "sess-c": "unanswered"}


def _snapshot_trajectory_decisive(session_id: str) -> dict[str, object]:
    """The base snapshot trajectory with the session's disposition applied."""
    trajectory = _snapshot_trajectory(session_id)
    resolutions = trajectory["resolutions"]
    assert isinstance(resolutions, list) and len(resolutions) == 1
    resolution = dict(resolutions[0])
    resolution["disposition"] = _DECISIVE_DISPOSITIONS[session_id]
    trajectory["resolutions"] = [resolution]
    return trajectory


def _add_license_evidence(data: dict[str, object]) -> None:
    # MIT evidence admits each batch under the projector's pinned license policy.
    data["license_evidence"] = {"spdx_id": "MIT", "source": "github-api"}


def build_snapshot_decisive(*, hostile: bool = False) -> FakeHub:
    """Materialize the pinned three-session snapshot as an in-memory FakeHub."""
    files = _snapshot_files(
        trajectory_fn=_snapshot_trajectory_decisive, manifest_hook=_add_license_evidence, hostile=hostile,
    )
    hub = FakeHub(repo_id=REPO_ID, private=True, files=files)
    hub.commit_revision(SNAPSHOT_REVISION)
    return hub
