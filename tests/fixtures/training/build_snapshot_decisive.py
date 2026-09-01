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

import json

from daydream.archive.hydrate_client import FakeHub
from daydream.archive.hydrate_rules import (
    ADMISSION_POLICY_VERSION,
    HYDRATION_INDEX_SCHEMA_VERSION,
    SANITIZER_VERSION,
    derive_curation_id,
)
from tests.fixtures.training.build_archive import FIXTURE_SESSIONS
from tests.fixtures.training.build_hub_snapshot import (
    REPO_ID,
    SNAPSHOT_REVISION,
    _snapshot_manifest,
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


def build_snapshot_decisive(*, hostile: bool = False) -> FakeHub:
    """Materialize the pinned three-session snapshot as an in-memory FakeHub."""
    files: dict[str, bytes] = {}
    for session_id, session in zip(
        ("sess-a", "sess-b", "sess-c"), FIXTURE_SESSIONS, strict=False
    ):
        manifest = _snapshot_manifest(
            session_id, session.repo_slug, session.skill, session.outcome_labels
        )
        files[f"{session_id}/manifest.json"] = json.dumps(
            manifest.to_dict(), indent=2
        ).encode()
        files[f"{session_id}/trajectory.json"] = json.dumps(
            _snapshot_trajectory_decisive(session_id), indent=2
        ).encode()
    # Non-run metadata and derived outputs: hydration must ignore them.
    files["README.md"] = b"production trajectory archive\n"
    files["dataset_info.json"] = b'{"dataset": "daydream-trajectories"}\n'
    files["curated/cur-old/batches/old/manifest.json"] = b'{"derived": true}\n'
    files["annotations/latest/sessions.jsonl"] = b'{"derived": true}\n'
    files["bronze/manifest.json"] = b'{"bronze": true}\n'
    curation_id = derive_curation_id(
        SNAPSHOT_REVISION,
        SANITIZER_VERSION,
        HYDRATION_INDEX_SCHEMA_VERSION,
        ADMISSION_POLICY_VERSION,
    )
    files[f"curated/{curation_id}/resume/ledger.jsonl"] = b""

    if hostile:
        files["../../escape.txt"] = b"pwned"
        files["/etc/daydream-escape"] = b"pwned"

    hub = FakeHub(repo_id=REPO_ID, private=True, files=files)
    hub.commit_revision(SNAPSHOT_REVISION)
    return hub
