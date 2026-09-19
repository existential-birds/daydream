"""Decisive-finding fake-Hub snapshot for the end-to-end annotation pipeline.

Extends :mod:`tests.fixtures.training.build_hub_snapshot` with exactly one
delta: each session's per-finding resolution carries a disposition mix that
exercises every materialization class — sess-a automatic ``accepted``,
sess-b automatic ``rejected``, sess-c ``unanswered`` (human-resolved by the
label step). Ids, manifests, curation derivation, and FakeHub wiring are
unchanged; the evidence digest is recomputed per the shared serializer
contract, and the profile/stack fields mirror ``_snapshot_trajectory``.
"""

from __future__ import annotations

from daydream.archive.hydrate_client import FakeHub
from tests.fixtures.training.build_hub_snapshot import (
    REPO_ID,
    SNAPSHOT_REVISION,
    _snapshot_files,
)

__all__ = ["REPO_ID", "SNAPSHOT_REVISION", "build_snapshot_decisive"]

# One decisive class per session: automatic accepted, automatic rejected,
# human-resolved (label step) — the §9 alias ids from the base snapshot.
_DECISIVE_DISPOSITIONS = {"sess-a": "accepted", "sess-b": "rejected", "sess-c": "unanswered"}


def _snapshot_trajectory_decisive(session_id: str) -> dict[str, object]:
    """The base snapshot trajectory with the session's disposition applied."""
    from tests.fixtures.training.build_hub_snapshot import _snapshot_trajectory

    trajectory = _snapshot_trajectory(session_id)
    resolutions = trajectory["resolutions"]
    assert isinstance(resolutions, list) and len(resolutions) == 1
    resolution = dict(resolutions[0])
    resolution["disposition"] = _DECISIVE_DISPOSITIONS[session_id]
    trajectory["resolutions"] = [resolution]
    return trajectory


def _add_license_evidence(data: dict[str, object]) -> None:
    # Required by the projection admission gate and bundle loader for every
    # admitted batch (MIT, accepted by the policy the projector run pins).
    data["license_evidence"] = {"spdx_id": "MIT", "source": "github-api"}


def build_snapshot_decisive(*, hostile: bool = False) -> FakeHub:
    """Materialize the pinned three-session snapshot as an in-memory FakeHub."""
    files = _snapshot_files(
        trajectory_fn=_snapshot_trajectory_decisive,
        manifest_hook=_add_license_evidence,
        hostile=hostile,
    )
    hub = FakeHub(repo_id=REPO_ID, private=True, files=files)
    hub.commit_revision(SNAPSHOT_REVISION)
    return hub
