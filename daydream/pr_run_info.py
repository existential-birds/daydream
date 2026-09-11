"""Live, session-owned trajectory acquisition for PR run-info markdown."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from daydream.atif import Trajectory
from daydream.pr_comment_renderer import render_run_info
from daydream.pricing import load_user_prices, resolve_prices

if TYPE_CHECKING:
    from daydream.artifact_visibility import ArtifactSession
    from daydream.trajectory import TrajectoryRecorder


class RunInfoStatus(StrEnum):
    """Whether live run details rendered or degraded to the safe fallback."""

    RENDERED = "rendered"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class LiveRunInfoSource:
    """Explicit live owners needed to acquire the parent and sibling values."""

    recorder: TrajectoryRecorder | None
    artifacts: ArtifactSession | None


@dataclass(frozen=True)
class RunInfoResult:
    """Rendered markdown plus an operator-safe acquisition disposition."""

    markdown: str
    status: RunInfoStatus
    diagnostic: str | None = None


_FALLBACK_MARKDOWN = render_run_info(())


def _unavailable(diagnostic: str) -> RunInfoResult:
    return RunInfoResult(
        markdown=_FALLBACK_MARKDOWN,
        status=RunInfoStatus.UNAVAILABLE,
        diagnostic=diagnostic,
    )


def render_live_run_info(source: LiveRunInfoSource) -> RunInfoResult:
    """Acquire validated live trajectories and render them without trajectory-file reads.

    Diagnostics are fixed low-cardinality categories. Exception messages,
    session paths, and trajectory contents never cross this boundary.
    """
    recorder = source.recorder
    if recorder is None:
        return _unavailable("run info: recorder unavailable")
    artifacts = source.artifacts
    if artifacts is None:
        return _unavailable("run info: artifact session unavailable")

    session_id = recorder.session_id
    root_id = recorder.trajectory_id
    if not session_id or root_id != session_id:
        return _unavailable("run info: trajectory identity invalid")

    try:
        sibling_snapshots = artifacts.snapshot_completed_sibling_trajectories(
            session_id=session_id
        )
    except Exception:  # noqa: BLE001 - live review must still post
        return _unavailable("run info: trajectory snapshot unavailable")

    try:
        parent = recorder.build_trajectory()
    except Exception:  # noqa: BLE001 - live review must still post
        return _unavailable("run info: parent trajectory unavailable")
    if parent.session_id != session_id or parent.trajectory_id != root_id:
        return _unavailable("run info: trajectory identity invalid")

    trajectories = [parent]
    seen_ids = {root_id}
    for snapshot in sibling_snapshots:
        try:
            sibling = Trajectory.model_validate_json(snapshot.json_bytes)
        except Exception:  # noqa: BLE001 - one corrupt retained value invalidates the view
            return _unavailable("run info: trajectory document invalid")
        sibling_id = sibling.trajectory_id
        if (
            not sibling_id
            or sibling.session_id != session_id
            or sibling_id != snapshot.trajectory_id
            or sibling_id in seen_ids
        ):
            return _unavailable("run info: trajectory identity invalid")
        seen_ids.add(sibling_id)
        trajectories.append(sibling)

    try:
        prices = resolve_prices(load_user_prices())
        markdown = render_run_info(trajectories, prices=prices)
    except Exception:  # noqa: BLE001 - pricing/rendering cannot block review posting
        return _unavailable("run info: rendering unavailable")
    if markdown == _FALLBACK_MARKDOWN:
        return _unavailable("run info: rendering unavailable")
    return RunInfoResult(
        markdown=markdown,
        status=RunInfoStatus.RENDERED,
    )


__all__ = [
    "LiveRunInfoSource",
    "RunInfoResult",
    "RunInfoStatus",
    "render_live_run_info",
]
