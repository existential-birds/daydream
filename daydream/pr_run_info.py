"""Live, session-owned trajectory acquisition for PR run-info markdown."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from daydream.atif import Trajectory
from daydream.pr_comment_renderer import render_run_info
from daydream.pricing import load_user_prices, resolve_prices
from daydream.trajectory import TrajectoryDocumentIntegrityError

if TYPE_CHECKING:
    from daydream.artifact_visibility import ArtifactSession
    from daydream.trajectory import TrajectoryRecorder


@dataclass(frozen=True)
class LiveRunInfoSource:
    """Explicit live owners needed to acquire the parent and sibling values."""

    recorder: TrajectoryRecorder | None
    artifacts: ArtifactSession | None


@dataclass(frozen=True)
class RunInfoResult:
    """Rendered markdown plus an operator-safe acquisition disposition."""

    markdown: str
    diagnostic: str | None = None


_FALLBACK_MARKDOWN = render_run_info(())


def _unavailable(diagnostic: str) -> RunInfoResult:
    return RunInfoResult(
        markdown=_FALLBACK_MARKDOWN,
        diagnostic=diagnostic,
    )


def _print_omitted_child_warning(exc: TrajectoryDocumentIntegrityError) -> None:
    from daydream.agent import console
    from daydream.ui import print_warning

    print_warning(console, f"run info: omitted a missing or changed child trajectory ({type(exc).__name__})")


def render_live_run_info(source: LiveRunInfoSource) -> RunInfoResult:
    """Acquire validated live trajectories and render them while the session is active.

    The parent comes from the recorder's memory. Completed children come from
    the recorder's file references and are read through the checked
    (no-follow, size- and SHA-256-verified) document read; a child whose file is
    missing or changed is omitted with a warning rather than blocking the post.
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
        artifacts.require_active()
        references = recorder.completed_child_documents()
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
    for reference in references:
        try:
            sibling = Trajectory.model_validate_json(reference.read_bytes())
        except TrajectoryDocumentIntegrityError as exc:
            _print_omitted_child_warning(exc)
            continue
        except Exception:  # noqa: BLE001 - one corrupt retained value invalidates the view
            return _unavailable("run info: trajectory document invalid")
        sibling_id = sibling.trajectory_id
        if (
            not sibling_id
            or sibling.session_id != session_id
            or sibling_id != reference.trajectory_id
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
    )


__all__ = [
    "LiveRunInfoSource",
    "render_live_run_info",
]
