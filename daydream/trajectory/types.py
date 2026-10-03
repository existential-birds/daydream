"""Trajectory identity, lifecycle vocabulary, and immutable write snapshots."""
from __future__ import annotations

import json
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


@dataclass(frozen=True)
class TrajectoryDocumentSnapshot:
    """One canonical prepared trajectory payload for a single write cutoff."""

    trajectory_id: str
    path: Path
    json_bytes: bytes

    def validated_payload(self, session_id: str) -> dict[str, Any]:
        """Decode the exact immutable bytes and verify their document/run identity."""
        if type(self.trajectory_id) is not str or not self.trajectory_id or type(self.json_bytes) is not bytes:
            raise ValueError("document is malformed")
        payload = json.loads(self.json_bytes)
        if not isinstance(payload, dict):
            raise ValueError("JSON is malformed")
        if payload.get("trajectory_id") != self.trajectory_id or payload.get("session_id") != session_id:
            raise ValueError("identity is malformed")
        return cast(dict[str, Any], payload)


@dataclass(frozen=True)
class RunWriteSnapshot:
    """Immutable run-wide write input consumed by the Task 5 callback boundary."""

    status: Literal["complete", "partial"]
    cutoff_at: str
    root_trajectory_id: str
    documents: tuple[TrajectoryDocumentSnapshot, ...]

    def validate(self, session_id: str) -> None:
        """Validate one run-wide immutable cutoff before retaining or projecting it."""
        if self.root_trajectory_id != session_id:
            raise ValueError("root identity does not match the run")
        if self.status not in ("complete", "partial") or type(self.cutoff_at) is not str or not self.cutoff_at:
            raise ValueError("metadata is malformed")
        identities: set[str] = set()
        for document in self.documents:
            document.validated_payload(session_id)
            if document.trajectory_id in identities:
                raise ValueError("contains duplicate document identity")
            identities.add(document.trajectory_id)
        if session_id not in identities:
            raise ValueError("is missing its root document")
