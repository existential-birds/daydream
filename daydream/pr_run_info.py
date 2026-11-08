"""Live, session-owned trajectory acquisition for PR run-info markdown."""

from __future__ import annotations

from collections.abc import Iterator
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


class _Unavailable(Exception):
    """A retained trajectory invalidated the live view while it was being folded."""

    def __init__(self, diagnostic: str) -> None:
        super().__init__(diagnostic)
        self.diagnostic = diagnostic


def _print_omitted_child_warning(exc: TrajectoryDocumentIntegrityError) -> None:
    from daydream.agent import console
    from daydream.ui import print_warning

    print_warning(console, f"run info: omitted a missing or changed child trajectory ({type(exc).__name__})")


def render_live_run_info(source: LiveRunInfoSource) -> RunInfoResult:
    """Acquire validated live trajectories and render them while the recorder is active.

    The parent comes from recorder memory; completed children are checked reads
    of the recorder registry's file references, decoded and folded one at a time,
    and a missing or changed child is omitted with a warning rather than blocking the post.
    Prices resolve before the fold, so a pricing failure reports rendering unavailable
    even when a retained document is also invalid.
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
        references = recorder.completed_child_documents()
    except Exception:  # noqa: BLE001 - live review must still post
        return _unavailable("run info: trajectory snapshot unavailable")

    try:
        parent = recorder.build_trajectory()
    except Exception:  # noqa: BLE001 - live review must still post
        return _unavailable("run info: parent trajectory unavailable")
    if parent.session_id != session_id or parent.trajectory_id != root_id:
        return _unavailable("run info: trajectory identity invalid")

    def trajectories() -> Iterator[Trajectory]:
        yield parent
        seen_ids = {root_id}
        for reference in references:
            try:
                sibling = Trajectory.model_validate_json(reference.read_bytes())
            except TrajectoryDocumentIntegrityError as exc:
                _print_omitted_child_warning(exc)
                continue
            except Exception:  # noqa: BLE001 - one corrupt retained value invalidates the view
                raise _Unavailable("run info: trajectory document invalid") from None
            sibling_id = sibling.trajectory_id
            if (
                not sibling_id
                or sibling.session_id != session_id
                or sibling_id != reference.trajectory_id
                or sibling_id in seen_ids
            ):
                raise _Unavailable("run info: trajectory identity invalid")
            seen_ids.add(sibling_id)
            yield sibling
            del sibling  # release this child before the next one is read

    try:
        prices = resolve_prices(load_user_prices())
        markdown = render_run_info(trajectories(), prices=prices)
    except _Unavailable as exc:
        return _unavailable(exc.diagnostic)
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
