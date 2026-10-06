"""Segment sibling trajectories in recorder fork-registration order; reject duplicate ids."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Segment:
    """One per-agent segment of a session's trajectory tree."""

    segment_id: str
    trajectory_id: str
    session_id: str


def _descriptor(trajectory_id: str) -> str:
    """Derive the agent descriptor from a ``<session>:<descriptor>`` id."""
    _, _, desc = trajectory_id.partition(":")
    return desc or trajectory_id


def segment(trajectory: dict[str, Any]) -> list[Segment]:
    """Enumerate siblings as seg-0..n-1 in registration order; use the root only without siblings.

    Duplicate trajectory_id values raise ValueError naming the id.
    """
    summaries = (trajectory.get("extra") or {}).get("subtrajectories") or []
    refs = [summary for summary in summaries if "invocation_id" not in summary
            and summary.get("trajectory_id") != trajectory.get("trajectory_id")]
    if not refs:
        refs = trajectory.get("subagent_trajectory_ref") or []
    if not refs:
        root_id = str(trajectory.get("trajectory_id", ""))
        return [
            Segment(
                segment_id="seg-0",
                trajectory_id=root_id,
                session_id=str(trajectory.get("session_id", "")),
            )
        ]

    seen: set[str] = set()
    segments: list[Segment] = []
    for order_index, ref in enumerate(refs):
        trajectory_id = str(ref.get("trajectory_id", ""))
        if trajectory_id in seen:
            raise ValueError(
                f"duplicate segmentation key (descriptor={_descriptor(trajectory_id)!r}): "
                f"{trajectory_id!r} — segmentation must be a total order"
            )
        seen.add(trajectory_id)
        segments.append(
            Segment(
                segment_id=f"seg-{order_index}",
                trajectory_id=trajectory_id,
                session_id=str(ref.get("session_id", trajectory.get("session_id", ""))),
            )
        )
    return segments


__all__ = ["Segment", "segment"]
