"""Trajectory identity, lifecycle vocabulary, and immutable write snapshots."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal


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


@dataclass(frozen=True)
class RunWriteSnapshot:
    """Immutable run-wide write input consumed by the Task 5 callback boundary."""

    status: Literal["complete", "partial"]
    cutoff_at: str
    root_trajectory_id: str
    documents: tuple[TrajectoryDocumentSnapshot, ...]
