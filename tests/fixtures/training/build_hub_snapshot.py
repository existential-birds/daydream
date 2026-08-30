"""Builds the offline fake-Hub snapshot for the hydrate integration suite (M22).

Serializes three session bundles from ``build_archive.FIXTURE_SESSIONS`` (the
§9 fixture matrix is reused, not re-invented) into a
:class:`~daydream.archive.hydrate_client.FakeHub` file tree: ``manifest.json`` +
``trajectory.json`` per session under ``bundles/<session_id>/``, a bronze
companion file (to assert M10 immutability), an empty remote resume ledger, and
everything pinned under a deterministic 40-hex ``SNAPSHOT_REVISION``.

``hostile=True`` injects traversal-style relpaths (``../../escape.txt`` and an
absolute ``/etc/...`` path) so the trust boundary can be exercised end-to-end.

No network, no ``huggingface_hub`` import, no absolute VM-local paths: the
builder is pure in-memory construction.
"""

from __future__ import annotations

import hashlib
import json

from daydream.archive.hydrate_client import FakeHub
from daydream.archive.hydrate_rules import (
    ADMISSION_POLICY_VERSION,
    HYDRATION_INDEX_SCHEMA_VERSION,
    SANITIZER_VERSION,
    derive_curation_id,
)
from daydream.archive.manifest import Manifest
from tests.fixtures.training.build_archive import _MINIMAL_TRAJECTORY, FIXTURE_SESSIONS

REPO_ID = "org/private-ds"
SNAPSHOT_REVISION = hashlib.sha256(b"fixture-hub-snapshot-v1").hexdigest()[:40]

# Three §9 sessions, aliased to the stable ids the integration scenarios assert on.
_SNAPSHOT_SESSION_IDS = ("sess-a", "sess-b", "sess-c")


def _snapshot_manifest(session_id: str, repo_slug: str, skill: str, outcome_labels: tuple[str, ...]) -> Manifest:
    """Build a Manifest from §9 session data with staging-safe path fields.

    ``archive_path``/``source_path`` are snapshot-internal placeholders (never
    pytest/tmp paths, which would trip the fixture-exclusion registry); the
    orchestrator rewrites both to staging-local values at index time.
    """
    return Manifest(
        session_id=session_id,
        archived_at="2026-05-17T00:00:00+00:00",
        status="complete",
        pipeline_status="succeeded",
        skill=skill,
        repo_slug=repo_slug,
        branch="feat/x",
        base_branch="main",
        head_sha="abc123",
        grounding_rate=0.9,
        outcome_labels=json.dumps(list(outcome_labels)),
        archive_path=f"/archive/runs/{session_id}",
        remote_url=f"https://github.com/{repo_slug}",
    )


def build_snapshot(*, hostile: bool = False) -> FakeHub:
    """Materialize the pinned three-session snapshot as an in-memory FakeHub."""
    files: dict[str, bytes] = {}
    for session_id, session in zip(_SNAPSHOT_SESSION_IDS, FIXTURE_SESSIONS, strict=False):
        manifest = _snapshot_manifest(
            session_id, session.repo_slug, session.skill, session.outcome_labels
        )
        files[f"bundles/{session_id}/manifest.json"] = json.dumps(
            manifest.to_dict(), indent=2
        ).encode()
        files[f"bundles/{session_id}/trajectory.json"] = json.dumps(
            _MINIMAL_TRAJECTORY, indent=2
        ).encode()
    # Bronze companion content: hydration must never touch it (M10).
    files["bronze/manifest.json"] = b'{"bronze": true}\n'
    # Remote resume ledger, seeded empty: the Hub is the canonical resume state.
    curation_id = derive_curation_id(
        SNAPSHOT_REVISION, SANITIZER_VERSION, HYDRATION_INDEX_SCHEMA_VERSION,
        ADMISSION_POLICY_VERSION,
    )
    files[f"curated/{curation_id}/resume/ledger.jsonl"] = b""

    if hostile:
        files["../../escape.txt"] = b"pwned"
        files["/etc/daydream-escape"] = b"pwned"

    hub = FakeHub(repo_id=REPO_ID, private=True, files=files)
    hub.commit_revision(SNAPSHOT_REVISION)
    return hub
