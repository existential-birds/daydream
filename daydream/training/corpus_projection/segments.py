"""Deterministic per-agent segmentation of ATIF trajectories (projection).

Pinned rule (spike 0B): sibling registration order in
``TrajectoryRecorder.fork()`` is append-ordered, so enumeration of the
trajectory's ``subagent_trajectory_ref`` list is the fork registration
order. Segmentation must never be a coin-flip, so duplicate sibling keys
raise.
"""

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
    """Segment a trajectory dict into per-agent ``Segment`` records.

    Ordering follows the trajectory's ``subagent_trajectory_ref`` list order
    (fork registration order, per the Task 0B pinned rule). The root trajectory
    is ``seg-0`` only when no siblings exist; otherwise siblings are ``seg-0..n-1``.

    Raises:
        ValueError: when two sibling refs share the same
            ``(descriptor, trajectory_id)`` key — the message names both.
    """
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

    seen: dict[tuple[str, str], str] = {}
    segments: list[Segment] = []
    for order_index, ref in enumerate(refs):
        trajectory_id = str(ref.get("trajectory_id", ""))
        descriptor = _descriptor(trajectory_id)
        key = (descriptor, trajectory_id)
        if key in seen:
            raise ValueError(
                f"duplicate segmentation key (descriptor={descriptor!r}): "
                f"{seen[key]!r} and {trajectory_id!r} — segmentation must be a total order"
            )
        seen[key] = trajectory_id
        segments.append(
            Segment(
                segment_id=f"seg-{order_index}",
                trajectory_id=trajectory_id,
                session_id=str(ref.get("session_id", trajectory.get("session_id", ""))),
            )
        )
    return segments


__all__ = ["Segment", "segment"]
