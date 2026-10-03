"""Artifact identities, destination routes, and durable state vocabulary."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import ConfigDict, StrictInt, StrictStr

_SCHEMA_VERSION = 1
_DAYDREAM = ".daydream"
_REVIEW_OUTPUT = ".review-output.md"
#: Recognized public directory anchors. ``review-cache`` must survive public
#: .daydream publication and reopen without a per-run destination route.
_PUBLIC_DIRECTORY_ANCHORS = frozenset(
    ("runs", "deep", "exploration", "partial-fixes", "improve", "intents", "review-cache")
)
_PUBLIC_FILE_ANCHORS = frozenset(("diff.patch", "hunk-index.json", "recommended.patch", ".DS_Store"))
_PUBLIC_ANCHORS = _PUBLIC_DIRECTORY_ANCHORS | _PUBLIC_FILE_ANCHORS
_OPERATIONAL_NAMES = frozenset(("worktrees", "audit"))


class ArtifactVisibilityError(RuntimeError):
    """A live artifact namespace could not be opened or used safely."""


class _SessionState(str, Enum):
    """Lifecycle of an :class:`ArtifactSession`.

    Exactly the state strings the session always cycled through; ``str``-based
    so serialized diagnostics and comparisons keep their prior textual form.
    """

    ACTIVE = "active"
    FROZEN = "frozen"
    PUBLISHING = "publishing"
    PUBLISHED = "published"
    CLOSED = "closed"


class _Transition(str, Enum):
    """Journalled in-flight transaction state; ``PUBLISH_*`` members publish."""

    DETACH_STAGED = "DETACH_STAGED"
    DETACH_CANONICAL = "DETACH_CANONICAL"
    DETACH_REMOVING = "DETACH_REMOVING"
    DETACHED = "DETACHED"
    PUBLISH_STAGED = "PUBLISH_STAGED"
    PUBLISH_BACKED_UP = "PUBLISH_BACKED_UP"
    PUBLISH_INSTALLED = "PUBLISH_INSTALLED"
    PUBLISH_VERIFIED = "PUBLISH_VERIFIED"

    @property
    def is_publish(self) -> bool:
        return self.name.startswith("PUBLISH_")


class _TerminalState(str, Enum):
    """Journalled outcome a retired transaction's cleanup ticket records."""

    DETACHED_RECONCILED = "DETACHED_RECONCILED"
    PUBLISH_RECONCILED = "PUBLISH_RECONCILED"
    PUBLISH_VERIFIED = "PUBLISH_VERIFIED"


class OutputLabel(str, Enum):
    """Closed classes of output registered before model execution."""

    PUBLIC_DAYDREAM = "public_daydream"
    PUBLIC_REVIEW_OUTPUT = "public_review_output"
    EXPLICIT_TRAJECTORY = "explicit_trajectory"
    EXPLICIT_TRAJECTORY_PARTIAL = "explicit_trajectory_partial"
    FINDINGS_OUTPUT = "findings_output"
    DUMP_DIRECTORY = "dump_directory"


class DestinationDelivery(str, Enum):
    """Closed host delivery policy for an operator-requested output."""

    DEFERRED = "deferred"
    LIVE_EXTERNAL = "live_external"
    FINALIZATION_MERGE = "finalization_merge"


class ArtifactDisposition(str, Enum):
    """Host outcome applied after immutable evidence finalization."""

    COMPLETE = "complete"
    PARTIAL_EVIDENCE = "partial_evidence"
    ROLLBACK = "rollback"


class _ExternalEntryPurpose(str, Enum):
    PROBE_EXCHANGE_A = "probe_exchange_a"
    PROBE_EXCHANGE_B = "probe_exchange_b"
    PROBE_LINK_TARGET = "probe_link_target"
    PUBLICATION_STAGE = "publication_stage"
    PUBLICATION_LINK_TARGET = "publication_link_target"
    MISSING_PARENT = "missing_parent"


class _ExternalEntryLifecycle(str, Enum):
    CREATION_INTENT = "creation_intent"
    ATTESTED = "attested"
    OPERATION_PREPARED = "operation_prepared"
    INSTALLED = "installed"
    REVERSAL_ATTEMPTED = "reversal_attempted"
    CLEANUP_PREPARED = "cleanup_prepared"
    RETIRED = "retired"
    CONFLICT = "conflict"


_EXTERNAL_PURPOSE_VALUES = frozenset(value.value for value in _ExternalEntryPurpose)
_EXTERNAL_LIFECYCLE_VALUES = frozenset(value.value for value in _ExternalEntryLifecycle)


@dataclass(frozen=True)
class ArtifactManifestEntry:
    """One no-follow filesystem entry in an immutable artifact tree."""

    path: str
    kind: Literal["directory", "file"]
    size: int
    mode: int
    sha256: str | None


@dataclass(frozen=True)
class ArtifactLayout:
    """One run's private namespace; every derived path is a property."""

    repo: Path
    source: Path
    git_common_dir: Path
    source_git_dir: Path
    repo_git_dir: Path
    operational_workspaces_root: Path
    session_id: str
    state_root: Path

    @property
    def artifact_runtime_root(self) -> Path:
        return self.state_root.parent

    @property
    def workspace_key(self) -> str:
        return self.state_root.name

    @property
    def live_root(self) -> Path:
        from daydream.trajectory import run_directory

        return run_directory(self.state_root, self.session_id) / "live"

    @property
    def daydream_dir(self) -> Path:
        return self.live_root / _DAYDREAM

    @property
    def review_output(self) -> Path:
        return self.live_root / _REVIEW_OUTPUT

    @property
    def public_daydream_dir(self) -> Path:
        return self.source / _DAYDREAM

    @property
    def public_review_output(self) -> Path:
        return self.source / _REVIEW_OUTPUT


@dataclass(frozen=True)
class ArtifactWorkspaceIdentity:
    """Canonical source ownership and private host namespace."""

    repo: Path
    source: Path
    git_common_dir: Path
    source_git_dir: Path
    repo_git_dir: Path
    operational_state_root: Path
    state_root: Path
    workspace_key: str


@dataclass(frozen=True)
class PrivateRootLocations:
    """Sibling private roots selected once by runner composition."""

    artifact_runtime: Path
    operational_workspaces: Path


@dataclass(frozen=True)
class PrivateWorkspaceOwner:
    """Validated source/Git ownership shared by artifacts and worktrees."""

    source: Path
    git_common_dir: Path
    workspace_key: str
    artifact_state_root: Path
    operational_state_root: Path


@dataclass(frozen=True)
class RoutedDestination:
    label: OutputLabel
    requested: Path
    write_path: Path | None
    frozen_path: Path | None
    delivery: DestinationDelivery


@dataclass(frozen=True)
class TrajectoryOutputRoute:
    run_dir: Path
    full: RoutedDestination
    partial: RoutedDestination


@dataclass(frozen=True)
class _DestinationRecord:
    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")

    record_id: StrictStr
    requested: StrictStr
    base: StrictStr
    relative: StrictStr
    label: OutputLabel
    delivery: DestinationDelivery
    expected_kind: Literal["file", "directory"]
    baseline_state: Literal["absent", "file", "directory"]
    baseline: tuple[ArtifactManifestEntry, ...]
    missing_parents: tuple[StrictStr, ...]
    expected_dev: StrictInt | None = None
    expected_ino: StrictInt | None = None
    prepared_sha256: StrictStr | None = None
    installed_sha256: StrictStr | None = None
    published: tuple[ArtifactManifestEntry, ...] = ()


@dataclass(frozen=True)
class ArtifactTreeSnapshot:
    """Frozen run tree consumed by later archive/evaluation/publication."""

    session_id: str
    workspace_key: str
    root: Path
    manifest: tuple[ArtifactManifestEntry, ...]
    destinations: tuple[RoutedDestination, ...]


@dataclass
class _RoutedRecord:
    """One registered route paired with the destination ledger row it owns."""

    route: RoutedDestination
    record: _DestinationRecord
    late: Path | None = None


@dataclass(frozen=True)
class ArtifactEvidenceProvenance:
    """Where one run's evidence lived, as paths the consumer must not rebuild."""

    workspace_key: str
    session_id: str
    public_source: Path
    live_root: Path

    @property
    def public_daydream_dir(self) -> Path:
        return self.public_source / _DAYDREAM

    @property
    def public_review_output(self) -> Path:
        return self.public_source / _REVIEW_OUTPUT


_PUBLIC_LABELS = (OutputLabel.PUBLIC_DAYDREAM, OutputLabel.PUBLIC_REVIEW_OUTPUT)
_TRAJECTORY_LABELS = (OutputLabel.EXPLICIT_TRAJECTORY, OutputLabel.EXPLICIT_TRAJECTORY_PARTIAL)
