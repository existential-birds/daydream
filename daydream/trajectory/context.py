"""Context-local access to the active trajectory recorder."""

from __future__ import annotations

from contextvars import ContextVar
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from daydream.trajectory.recorder import TrajectoryRecorder


# Recorder propagation uses a ContextVar (not a module-level dataclass, per
# PROJECT.md "propagated via ContextVar (not AgentState)"). Access via
# get_current_recorder() ONLY. Test isolation resets _RECORDER_VAR and
# _ACTIVE_SIGNAL_RUNS directly (CORE-10 / D-17).
_RECORDER_VAR: ContextVar["TrajectoryRecorder | None"] = ContextVar(
    "_RECORDER_VAR",
    default=None,
)


def get_current_recorder() -> "TrajectoryRecorder | None":
    """Return the active async-context recorder, or None outside a recorded run."""
    return _RECORDER_VAR.get()


def current_session_id() -> str | None:
    """Session id of the active recorder, or ``None`` when no recorder is active."""
    recorder = get_current_recorder()
    return recorder.session_id if recorder is not None else None
