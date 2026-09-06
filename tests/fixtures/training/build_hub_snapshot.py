"""Builds the offline fake-Hub snapshot for the hydrate integration suite (M22).

Serializes three session bundles from ``build_archive.FIXTURE_SESSIONS`` (the
§9 fixture matrix is reused, not re-invented) into a
:class:`~daydream.archive.hydrate_client.FakeHub` file tree: ``manifest.json`` +
``trajectory.json`` per session under the producer's canonical
``<session_id>/`` layout, non-run metadata, derived ``curated/**`` and
``annotations/**`` outputs, and everything pinned under a deterministic
40-hex ``SNAPSHOT_REVISION``.

``hostile=True`` injects traversal-style relpaths (``../../escape.txt`` and an
absolute ``/etc/...`` path) so the trust boundary can be exercised end-to-end.

No network, no ``huggingface_hub`` import, no absolute VM-local paths: the
builder is pure in-memory construction.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from daydream.archive.hydrate import RepoInfo
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

# Pinned archive fixture (issue #1094 Task 10): a second digest-pinned revision
# whose five sessions exercise the full enrich -> gate -> v2-identity path.
PINNED_REVISION = hashlib.sha256(b"fixture-hub-snapshot-pinned-v1").hexdigest()[:40]

# (session_id, repo_slug, declared license_evidence | None):
#   pin-declared  — well-formed declared MIT evidence (enrichment never re-derives)
#   pin-enrich    — legacy record, repo identity only (enrichment fills MIT)
#   pin-gpl       — enrichment resolves GPL-3.0-only -> c8_copyleft_unopted
#   pin-unknown   — enrichment cannot resolve -> repo_commit_unresolved
#   pin-c5        — C5-listed repo (getsentry/sentry) -> c5_excluded_repo
_PINNED_SESSIONS: tuple[tuple[str, str, dict[str, str] | None], ...] = (
    ("pin-declared", "acme/widget", {"spdx_id": "MIT", "source": "producer"}),
    ("pin-enrich", "acme/widget", None),
    ("pin-gpl", "acme/copyleft", None),
    ("pin-unknown", "ghost/nope", None),
    ("pin-c5", "getsentry/sentry", None),
)

# The pinned policy: the same content as the checked-in production SPDX policy
# (daydream/training/schema/license-policy-production.json), committed as a
# fixture so the policy digest in every pinned-fixture run is stable.
PINNED_POLICY_FIXTURE = Path(__file__).parent / "license-policy-pinned-fixture.json"

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
        manifest = _snapshot_manifest(
            session_id, session.repo_slug, session.skill, session.outcome_labels
        )
        files[f"{session_id}/manifest.json"] = json.dumps(
            manifest.to_dict(), indent=2
        ).encode()
        files[f"{session_id}/trajectory.json"] = json.dumps(
            _snapshot_trajectory(session_id), indent=2
        ).encode()
    # Non-run metadata and derived outputs: hydration must ignore them.
    files["README.md"] = b"production trajectory archive\n"
    files["dataset_info.json"] = b'{"dataset": "daydream-trajectories"}\n'
    files["curated/cur-old/batches/old/manifest.json"] = b'{"derived": true}\n'
    files["annotations/latest/sessions.jsonl"] = b'{"derived": true}\n'
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


def build_pinned_snapshot() -> FakeHub:
    """Materialize the pinned five-session archive fixture as an in-memory FakeHub.

    Same builder pattern as :func:`build_snapshot` (shared manifest/trajectory
    constructors, canonical ``<session_id>/`` layout, pinned 40-hex revision);
    the sessions carry the mixed declared/enriched/C5/copyleft/unknown license
    matrix the Task 10 integration test asserts on.
    """
    files: dict[str, bytes] = {}
    for session_id, repo_slug, evidence in _PINNED_SESSIONS:
        manifest = _snapshot_manifest(session_id, repo_slug, "pr_review", ("merged",))
        data = manifest.to_dict()
        if evidence is not None:
            data["license_evidence"] = evidence
        files[f"{session_id}/manifest.json"] = json.dumps(data, indent=2).encode()
        files[f"{session_id}/trajectory.json"] = json.dumps(
            _snapshot_trajectory(session_id), indent=2
        ).encode()
    hub = FakeHub(repo_id=REPO_ID, private=True, files=files)
    hub.commit_revision(PINNED_REVISION)
    return hub


class AnnotationsHub(FakeHub):
    """Revisioned external annotation store with deterministic CAS races.

    Initial and derived heads are content-addressed. Pinned reads access copied
    revision trees, never the mutable ``files`` compatibility view. Atomic tree
    rejection does not model the real Hub's separately possible LFS pre-upload.
    Explicit rival commits let tests introduce concurrency at batch/data/success
    boundaries without replacing any production publication behavior.
    """

    def __init__(
        self,
        *,
        curation_id: str = "",
        snapshot_id: str = "",
        private: bool = True,
        files: dict[str, bytes] | None = None,
        repo_id: str = REPO_ID,
    ) -> None:
        if bool(curation_id) != bool(snapshot_id):
            raise ValueError("both legacy annotation identity components are required")
        self.prefix = f"annotations/{curation_id}/{snapshot_id}/" if curation_id else ""
        seeded = {f"{self.prefix}preview-manifest.json": b"{}\n"} if self.prefix else {}
        seeded.update(files or {})
        identity = {path: hashlib.sha256(data).hexdigest() for path, data in sorted(seeded.items())}
        initial = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:40]
        super().__init__(repo_id=repo_id, private=private, files=seeded, head_sha=initial)
        self.commit_revision(initial, ref="main")
        self.info_revision_log: list[str | None] = []
        self.listed_revision_log: list[str | None] = []
        self.downloaded_revision_log: list[tuple[str, str | None]] = []
        self.atomic_attempt_log: list[dict[str, Any]] = []
        self._queued_rivals: list[tuple[str, dict[str, bytes], str]] = []

    def repo_info(self, revision: str | None = None) -> RepoInfo:
        self.info_revision_log.append(revision)
        return super().repo_info(revision)

    def list_repo_files(self, revision: str | None = None) -> list[str]:
        self.listed_revision_log.append(revision)
        return super().list_repo_files(revision)

    def download_file(self, path_in_repo: str, revision: str | None = None) -> bytes:
        self.downloaded_revision_log.append((path_in_repo, revision))
        return super().download_file(path_in_repo, revision)

    def revision_files(self, revision: str) -> dict[str, bytes]:
        """Return a detached view, so assertions cannot mutate pinned content."""
        return self._revision_tree(self._resolve_revision(revision))

    def queue_concurrent_commit(
        self, stage: str, files: dict[str, bytes], message: str = "rival publication",
    ) -> None:
        """Schedule one genuine competing tree commit before a matching attempt."""
        if stage not in {"batch", "data", "success"}:
            raise ValueError("unknown annotation publication stage")
        self._queued_rivals.append((stage, dict(files), message))

    def seed_remote_files(self, files: dict[str, bytes], message: str = "fixture seed") -> str:
        """Create an immutable external revision, without retaining local files."""
        with tempfile.TemporaryDirectory(prefix="daydream-annotation-fixture-") as staging:
            mapping: dict[str | Path, Path] = {}
            for index, (path, data) in enumerate(sorted(files.items())):
                local = Path(staging) / str(index)
                local.write_bytes(data)
                mapping[path] = local
            return super().commit_files_atomic(
                mapping, message, parent_commit=self._head, branch="main",
            )

    def commit_files_atomic(
        self, mapping: dict[str | Path, Path], commit_message: str, *,
        parent_commit: str, branch: str,
    ) -> str:
        paths = sorted(map(str, mapping))
        stage = (
            "success" if any(path.endswith("/_SUCCESS") for path in paths)
            else "batch" if any(path.endswith("/batch-latest.json") for path in paths)
            else "data"
        )
        self.atomic_attempt_log.append({
            "contains": paths, "parent_commit": parent_commit, "branch": branch, "stage": stage,
        })
        for index, (rival_stage, payloads, message) in enumerate(self._queued_rivals):
            if rival_stage == stage:
                self._queued_rivals.pop(index)
                self.seed_remote_files(payloads, message)
                break
        return super().commit_files_atomic(
            mapping, commit_message, parent_commit=parent_commit, branch=branch,
        )

    def mutate_annotation_file(self, relpath: str, data: bytes) -> None:
        """Seed changed bytes at a new revision without corrupting old pins."""
        if not self.prefix:
            raise ValueError("legacy annotation prefix is required")
        self.seed_remote_files({f"{self.prefix}{relpath}": data})


def build_annotations_hub(curation_id: str, snapshot_id: str, *, private: bool = True) -> AnnotationsHub:
    """Materialize an empty annotations bundle hub for one snapshot pin."""
    return AnnotationsHub(curation_id=curation_id, snapshot_id=snapshot_id, private=private)


@dataclass(frozen=True)
class PublicationHubs:
    """The sole durable byte stores and public pins shared across fake VMs."""

    source: FakeHub
    annotations: AnnotationsHub
    source_revision: str
    policy_path: Path


def build_publication_hubs() -> PublicationHubs:
    """Build separate offline source/annotation repositories, no VM-local inputs."""
    from tests.fixtures.training.build_snapshot_decisive import build_snapshot_decisive

    return PublicationHubs(
        source=build_snapshot_decisive(),
        annotations=AnnotationsHub(repo_id="org/private-annotations"),
        source_revision=SNAPSHOT_REVISION,
        policy_path=PINNED_POLICY_FIXTURE,
    )
