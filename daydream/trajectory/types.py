"""Trajectory identity, lifecycle vocabulary, immutable write snapshots, and checked document reads."""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, cast


class DaydreamPhase(str, Enum):
    """Phase label for ``Step.extra['daydream_phase']`` (MAP-08).

    Values match ATIF ``extra`` field literals exactly. Required keyword-only
    arg on ``run_agent()`` (D-05); every call site in ``phases.py`` passes a
    literal member.
    """

    REVIEW = "review"
    PARSE = "parse"
    FIX = "fix"
    TEST = "test"
    INTENT = "intent"
    ALTERNATIVES = "alternatives"
    DEEP = "deep"
    EXPLORATION = "exploration"
    VERIFY = "verify"
    RECON = "recon"
    AUDIT = "audit"
    VET = "vet"
    PLAN_WRITE = "plan_write"
    DIAGRAM = "diagram"
    MERGE = "merge"
    # Host-side (non-agent) operations (issue #726): each is bracketed by
    # phase events carrying ``duration_ms`` and ``stop_reason`` so the
    # trajectory can tell test execution, hook runs, commits, pushes, and
    # exact-SHA remote CI verification
    # apart without inferring from step timestamps.
    TEST_EXECUTION = "test-execution"
    HOOK_RUN = "hook-run"
    COMMIT = "commit"
    PUSH = "push"
    REMOTE_CI = "remote-ci"


class DaydreamRunFlow(str, Enum):
    """Run-flow label for ``Step.extra['daydream_run_flow']`` (MAP-09).

    Set once at recorder construction (D-07); recorder stamps every Step.
    """

    NORMAL = "normal"
    TTT = "ttt"
    PR = "pr"
    DEEP = "deep"
    CUSTOM = "custom"
    IMPROVE = "improve"
    DIAGRAM = "diagram"


class LifecycleStatus(str, Enum):
    """Closed terminal states for phase and dispatch lifecycle evidence."""

    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    SKIPPED = "skipped"


class LifecycleReasonCode(str, Enum):
    """Low-cardinality reasons safe to persist across trajectory surfaces."""

    NO_ELIGIBLE_WORK = "no_eligible_work"
    SOME_CHILDREN_FAILED = "some_children_failed"
    ALL_CHILDREN_FAILED = "all_children_failed"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    DOMAIN_FAILURE = "domain_failure"
    UNCAUGHT_EXCEPTION = "uncaught_exception"


class TrajectoryDocumentIntegrityError(ValueError):
    """A referenced trajectory file is missing or no longer holds its written bytes."""


class _TrajectoryDocument:
    """One read interface over prepared bytes and checked on-disk references."""

    trajectory_id: str
    path: Path

    def read_bytes(self) -> bytes:
        raise NotImplementedError

    def validated(self, session_id: str) -> tuple[bytes, dict[str, Any]]:
        """Return the exact document bytes and their decoded, identity-checked payload."""
        raw = self.read_bytes()
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("JSON is malformed")
        if payload.get("trajectory_id") != self.trajectory_id or payload.get("session_id") != session_id:
            raise ValueError("identity is malformed")
        return raw, cast(dict[str, Any], payload)

    def validated_payload(self, session_id: str) -> dict[str, Any]:
        """Decode the exact document bytes and verify their document/run identity."""
        return self.validated(session_id)[1]


@dataclass(frozen=True)
class TrajectoryDocumentSnapshot(_TrajectoryDocument):
    """One canonical prepared trajectory payload for a single write cutoff."""

    trajectory_id: str
    path: Path
    json_bytes: bytes

    def read_bytes(self) -> bytes:
        if type(self.trajectory_id) is not str or not self.trajectory_id or type(self.json_bytes) is not bytes:
            raise ValueError("document is malformed")
        return self.json_bytes

    def validate(self, session_id: str) -> None:
        """Prepared bytes are in memory, so retention validates them in full."""
        self.validated_payload(session_id)


@dataclass(frozen=True)
class TrajectoryDocumentReference(_TrajectoryDocument):
    """A written trajectory document kept on disk: identity, logical and private paths, size, digest.

    ``path`` is the recorder's logical document path (its file name names archive
    and analyzer entries); ``private_path`` is the private file actually written
    or its frozen copy."""

    trajectory_id: str
    path: Path
    private_path: Path
    size: int
    sha256: str

    @classmethod
    def written(cls, trajectory_id: str, path: Path, private_path: Path, content: bytes) -> TrajectoryDocumentReference:
        """Describe the exact *content* just written at *private_path*."""
        return cls(trajectory_id, path, private_path, len(content), hashlib.sha256(content).hexdigest())

    def validate(self, session_id: str) -> None:
        """Check reference metadata without filesystem I/O."""
        if (
            type(self.trajectory_id) is not str or not self.trajectory_id or self.trajectory_id == session_id
            or not isinstance(self.path, Path) or not isinstance(self.private_path, Path)
            or type(self.size) is not int or self.size < 0
            or type(self.sha256) is not str or not re.fullmatch(r"[0-9a-f]{64}", self.sha256)
        ):
            raise ValueError("document reference is malformed")

    def read_bytes(self) -> bytes:
        """Re-read the referenced regular file and verify its size and SHA-256."""
        try:
            fd = os.open(self.private_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(fd, "rb") as stream:
                metadata = os.fstat(stream.fileno())
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_size != self.size:
                    raise TrajectoryDocumentIntegrityError("referenced document changed")
                raw = stream.read(self.size + 1)
        except OSError as exc:
            raise TrajectoryDocumentIntegrityError("referenced document is unavailable") from exc
        if len(raw) != self.size or hashlib.sha256(raw).hexdigest() != self.sha256:
            raise TrajectoryDocumentIntegrityError("referenced document changed")
        return raw


TrajectoryDocument = TrajectoryDocumentSnapshot | TrajectoryDocumentReference


@dataclass(frozen=True)
class RunWriteSnapshot:
    """Immutable run-wide write input consumed by the Task 5 callback boundary."""

    status: Literal["complete", "partial"]
    cutoff_at: str
    root_trajectory_id: str
    documents: tuple[TrajectoryDocument, ...]

    def validate(self, session_id: str) -> None:
        """Validate one run-wide immutable cutoff before retaining or projecting it."""
        if self.root_trajectory_id != session_id:
            raise ValueError("root identity does not match the run")
        if self.status not in ("complete", "partial") or type(self.cutoff_at) is not str or not self.cutoff_at:
            raise ValueError("metadata is malformed")
        identities: set[str] = set()
        for document in self.documents:
            document.validate(session_id)
            if document.trajectory_id in identities:
                raise ValueError("contains duplicate document identity")
            identities.add(document.trajectory_id)
        if session_id not in identities:
            raise ValueError("is missing its root document")
