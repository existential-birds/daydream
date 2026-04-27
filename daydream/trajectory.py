"""ATIF v1.6 trajectory recorder for daydream runs.

This module is the SOLE home for ATIF Pydantic model construction (D-19
module-bloat ban). Other modules (agent.py, phases.py, ui.py, runner.py,
backends/*) import only the public surface — never `daydream.atif.*`.

Lifecycle: ``runner.py`` opens ``async with TrajectoryRecorder(...) as
recorder`` once per run. ``agent.run_agent()`` opens an ``Invocation`` per
call against the recorder via ``get_current_recorder()``. Backends emit
``AgentEvent`` instances; the Invocation buffers them into ATIF Steps and
flushes to the parent Trajectory at scope exit. The Recorder writes the
Trajectory JSON on clean ``__aexit__``.

Phase 2 ships the minimum surface: ONE ``ContextVar`` (``_RECORDER_VAR``),
no parent linkage on ``Invocation``, and a no-op ``Redactor``. Phase 3 adds
the second ContextVar and parent linkage; Phase 4 fills in the Redactor
rule list.
"""

from __future__ import annotations

from contextvars import ContextVar
from datetime import datetime, timezone
from enum import Enum

from daydream.atif import (
    Step,
)
from daydream.ui import create_console

_console = create_console()


def now_iso() -> str:
    """Return current UTC time as ISO 8601 with trailing 'Z'.

    The single source of truth for timestamps in daydream's trajectory
    recording. Used by ``AgentEvent`` dataclass ``field(default_factory=...)``
    in ``daydream/backends/__init__.py`` (Plan 02), by recorder Step
    construction here, and by Phase 4 partial-write paths.

    Banned alternatives: the deprecated naive-utc helper from ``datetime``
    (Pitfall 2: lacks tzinfo, deprecated in 3.12+); ad-hoc
    ``datetime.now().isoformat()`` (no ``Z`` suffix — Pydantic timestamp
    validator requires ``Z`` or ``+00:00``).

    Returns:
        Timestamp string parseable by ``Step.validate_timestamp``.
    """
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


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
    PLAN = "plan"
    PR_FEEDBACK = "pr_feedback"
    DEEP = "deep"
    EXPLORATION = "exploration"


class DaydreamRunFlow(str, Enum):
    """Run-flow label for ``Step.extra['daydream_run_flow']`` (MAP-09).

    Set once at recorder construction time from ``runner.run()`` /
    ``run_pr_feedback()`` / ``run_trust()`` / ``run_deep()`` (D-07). The
    recorder stamps every Step with this value automatically.
    """

    NORMAL = "normal"
    TTT = "ttt"
    PR = "pr"
    DEEP = "deep"


class Redactor:
    """No-op pass-through redactor (Phase 2 stub).

    Phase 2 ships the FINAL public API surface; Phase 4 (REDA-01..06) fills
    in regex pattern lists internally without changing the recorder call
    site. ``redact_step`` is invoked at per-Step flush time per D-13 so
    Phase 4's partial-write paths inherit the same redaction posture.
    """

    def redact_step(self, step: Step) -> Step:
        """Return the step unchanged (Phase 2). Phase 4 fills this in.

        Args:
            step: ATIF Step about to be appended to the Trajectory.

        Returns:
            The same Step instance — Phase 2 is a strict pass-through.
        """
        return step


# Module-level Singletons
# =======================
# This module uses a ContextVar (NOT a module-level dataclass instance per
# PROJECT.md Constraints "propagated via ContextVar (not AgentState)") for
# trajectory recorder propagation. Access via ``get_current_recorder()`` ONLY;
# never import ``_RECORDER_VAR`` directly. The setter is implicit via
# ``TrajectoryRecorder.__aenter__`` / ``__aexit__``. Test isolation goes
# through ``_reset_recorder_for_tests()`` from the autouse conftest fixture
# (CORE-10 / D-17, wired in Plan 07).

_RECORDER_VAR: ContextVar["TrajectoryRecorder | None"] = ContextVar(  # type: ignore[name-defined]  # noqa: F821 - TrajectoryRecorder defined below; forward reference resolved at runtime via PEP 563 string annotations
    "_RECORDER_VAR", default=None,
)


def get_current_recorder() -> "TrajectoryRecorder | None":  # type: ignore[name-defined]  # noqa: F821 - TrajectoryRecorder defined below; forward reference
    """Return the recorder for the current async context, or None if none active.

    The single public accessor for ``_RECORDER_VAR`` (D-10). ``agent.py`` reads
    this at the top of ``run_agent()`` and skips the entire Invocation lifecycle
    when None — direct test invocation of ``run_agent()`` without an active
    recorder is therefore a clean no-op (CORE-09).

    Returns:
        The active ``TrajectoryRecorder`` instance, or ``None`` if no
        ``async with TrajectoryRecorder(...)`` block is on the stack.
    """
    return _RECORDER_VAR.get()


def _reset_recorder_for_tests() -> None:
    """Test-only: clear the recorder ContextVar.

    Use exclusively from the autouse ``_reset_trajectory_recorder`` fixture
    in ``tests/conftest.py`` (CORE-10, D-17). Production code MUST go through
    ``TrajectoryRecorder.__aenter__`` / ``__aexit__``.
    """
    _RECORDER_VAR.set(None)
