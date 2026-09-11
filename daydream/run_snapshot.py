"""Public immutable identity and trajectory inputs for run archiving."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from daydream.trajectory import DaydreamRunFlow

if TYPE_CHECKING:
    from daydream.trajectory import RunWriteSnapshot


@dataclass(frozen=True)
class ArchiveRecorderProvenance:
    """Immutable recorder identity recovered from one frozen root document."""

    session_id: str
    run_flow: DaydreamRunFlow
    pr_number: int | None
    pr_repo: str | None


@dataclass(frozen=True)
class RunPhaseCapabilities:
    """Phases the resolved run can execute."""

    per_stack_review: bool
    merge: bool
    fix: bool
    test: bool
    push: bool
    remote_ci: bool


@dataclass(frozen=True)
class RunProfileIdentity:
    """Resolved review-profile identity."""

    schema_version: int
    name: str
    source_kind: str
    digest: str


@dataclass(frozen=True)
class ManifestRunIdentity:
    """Effective non-recorder identity serialized in a run manifest."""

    flow_name: str | None
    skill: str | None
    model: str | None
    backend: str
    review_backend: str | None
    fix_backend: str | None
    test_backend: str | None
    per_stack_review_backend: str | None
    per_stack_review_model: str | None
    review_only: bool
    deep: bool
    profile: RunProfileIdentity | None
    phases: RunPhaseCapabilities


@dataclass(frozen=True)
class ArchiveRunSnapshot:
    """Joined immutable inputs for one archive finalization."""

    recorder_provenance: ArchiveRecorderProvenance
    identity: ManifestRunIdentity
    trajectories: RunWriteSnapshot


__all__ = [
    "ArchiveRecorderProvenance",
    "ArchiveRunSnapshot",
    "ManifestRunIdentity",
    "RunPhaseCapabilities",
    "RunProfileIdentity",
]
