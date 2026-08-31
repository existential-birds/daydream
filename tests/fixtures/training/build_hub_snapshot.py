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


def _snapshot_trajectory(session_id: str) -> dict[str, object]:
    """The minimal trajectory plus one unanswered per-finding resolution.

    Additive keys only: the #981/hydrate consumers keep parsing the same
    fields, while the #1055 annotation pipeline gets the adjudication-shaped
    finding it needs to build a non-empty queue. The evidence digest is
    recomputed from the evidence list so the fixture satisfies the shared
    serializer's digest contract by construction.
    """
    evidence = [
        {
            "reply_id": "r1",
            "body_sha256": hashlib.sha256(f"reply-1-{session_id}".encode()).hexdigest(),
        }
    ]
    trajectory: dict[str, object] = dict(_MINIMAL_TRAJECTORY)
    trajectory["session_id"] = session_id
    trajectory["trajectory_id"] = f"{session_id}:root"
    trajectory["resolutions"] = [
        {
            "fingerprint": f"fp-{session_id}",
            "disposition": "unanswered",
            "evidence": evidence,
            "evidence_digest": hashlib.sha256(
                json.dumps(evidence, sort_keys=True).encode()
            ).hexdigest(),
            # Native review-profile fields (issue #885, R12) — the shared
            # serializer nests these under ``profile`` in the canonical
            # record, so the projection must surface them at the two-bundle
            # boundary rather than dropping them.
            "profile_schema_version": 2,
            "profile_name": "pr_review",
            "profile_source_kind": "builtin",
            "profile_digest": "d" * 64,
            "profile": "pr_review",
            "stack": "python",
        }
    ]
    return trajectory


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
        manifest = _snapshot_manifest(session_id, session.repo_slug, session.skill, session.outcome_labels)
        files[f"bundles/{session_id}/manifest.json"] = json.dumps(manifest.to_dict(), indent=2).encode()
        files[f"bundles/{session_id}/trajectory.json"] = json.dumps(
            _snapshot_trajectory(session_id), indent=2
        ).encode()
    # Bronze companion content: hydration must never touch it (M10).
    files["bronze/manifest.json"] = b'{"bronze": true}\n'
    # Remote resume ledger, seeded empty: the Hub is the canonical resume state.
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


class AnnotationsHub(FakeHub):
    """Annotations-capable fake Hub: public ``files`` dict + ``mutate_bundle`` seam.

    Extends :class:`FakeHub` for the per-finding annotation-snapshot pipeline
    (#1055): the ``annotations/<curation-id>/<snapshot-id>/`` prefix is
    pre-seeded and :meth:`mutate_annotation_file` lets tests tamper with (or
    add to) the published bundle in place. Private by default, like the real
    target repo. (``mutate_bundle`` keeps FakeHub's bundle-collision signature;
    annotation mutations go through the renamed seam.)
    """

    def __init__(
        self,
        *,
        curation_id: str,
        snapshot_id: str,
        private: bool = True,
        files: dict[str, bytes] | None = None,
    ) -> None:
        self.prefix = f"annotations/{curation_id}/{snapshot_id}/"
        seeded = {f"{self.prefix}preview-manifest.json": b"{}\n"}
        seeded.update(files or {})
        super().__init__(repo_id=REPO_ID, private=private, files=seeded)

    def mutate_annotation_file(self, relpath: str, data: bytes) -> None:
        """Overwrite (or add) one file under the annotations prefix."""
        self.files[f"{self.prefix}{relpath}"] = data


def build_annotations_hub(curation_id: str, snapshot_id: str, *, private: bool = True) -> AnnotationsHub:
    """Materialize an empty annotations bundle hub for one snapshot pin."""
    return AnnotationsHub(curation_id=curation_id, snapshot_id=snapshot_id, private=private)
