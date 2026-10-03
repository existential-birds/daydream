"""Hydration contracts shared by discovery, staging, publication, and Hub adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


class HydrationError(Exception):
    """Base class for every hydrate failure mode; the orchestrator decides."""


class HubConcurrentUpdateError(HydrationError):
    """An atomic Hub commit was rejected because its parent is no longer head."""


class HubUnavailableError(HydrationError):
    """The ``huggingface_hub`` extra or its ``HF_TOKEN`` prerequisite is missing."""


class HubDownloadError(HydrationError):
    """A requested path/revision does not exist in the Hub repo (fail-closed)."""


class StageError(HydrationError):
    """Staging the pinned snapshot failed (download, digest, or path violation)."""


class NoSessionCandidatesError(HydrationError):
    """The snapshot looks like a run archive but contains no complete sessions."""


class MovingBranchError(HydrationError):
    """A symbolic ref (moving branch/tag) was requested without ``exploratory=True``."""


class PublicDestinationError(HydrationError):
    """The target Hub repo is public — hydration publishes only to private repos (M17)."""


class VerificationError(HydrationError):
    """The clean-room verification cycle failed — success is never reported (M20)."""


@dataclass(frozen=True)
class RepoInfo:
    """Minimal repo metadata the hydration flow needs."""

    sha: str
    private: bool


@runtime_checkable
class HubClient(Protocol):
    """Narrow Hub surface hydration depends on (list/download/commit/repo-info)."""

    def repo_info(self, revision: str | None = None) -> RepoInfo: ...

    def list_repo_files(self, revision: str | None = None) -> list[str]: ...

    def download_file(self, path_in_repo: str, revision: str | None = None) -> bytes: ...

    def upload_files(
        self, mapping: dict[str | Path, Path], commit_message: str
    ) -> None: ...

    @property
    def repo_private(self) -> bool: ...


@dataclass(frozen=True)
class DownloadResult:
    """Outcome of one :func:`download_snapshot` pass over the pinned revision."""

    downloaded: int = 0
    skipped: int = 0
    digests: dict[str, str] = field(default_factory=dict)  # relpath -> sha256
    discovered: int = 0
    run_shaped_manifests: int = 0
    incomplete_manifests: tuple[str, ...] = ()


@dataclass(frozen=True)
class IngestResult:
    """Outcome of the ingest gate for one staged session bundle (issue #982 M4/M6)."""

    session_id: str
    status: str  # "admitted" | "quarantined"
    reason_code: str | None = None


@dataclass
class DedupeResult:
    """Outcome of one :func:`dedupe_admitted` pass over ``stage/runs/`` (M7/M8/M9)."""

    admitted: int = 0
    skipped: int = 0
    collisions: int = 0
    collision_ids: list[str] = field(default_factory=list)
    excluded: list[tuple[str, str]] = field(default_factory=list)  # (session_id, reason_code)


@dataclass(frozen=True)
class ResumeState:
    """Resume checkpoint derived from the remote Hub ledger (never VM-local state)."""

    completed_sessions: set[str] = field(default_factory=set)
    redownloaded: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class HydrateHubConfig:
    """Operator configuration for ``run_hydrate_hub`` (harvest ``RunConfig`` discipline)."""

    source_repo: str
    source_revision: str
    destination_repo: str
    stage_dir: Path
    exploratory: bool = False
    # License admission gate (issue #1080): the policy file is the digest-pinned
    # versioned artifact; allow_copyleft is the exact-slug C8 opt-in set.
    license_policy_path: str | None = None
    allow_copyleft: frozenset[str] = frozenset()


@dataclass
class HydrateSummary:
    """Outcome of one ``run_hydrate_hub`` invocation; ``verified`` gates success."""

    source_commit: str
    curation_id: str
    output_commit_sha: str | None = None
    dry_run_discovered: int = 0
    dry_run_admitted: int = 0
    dry_run_rejected: int = 0
    dry_run_incomplete_manifests: tuple[str, ...] = ()
    verify_admitted: int = 0
    verified: bool = False
    # Four admission buckets from the import ledger; empty without a license policy.
    license_admission: dict[str, int] = field(default_factory=dict)
