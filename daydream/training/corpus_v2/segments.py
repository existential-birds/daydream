"""Deterministic per-agent segmentation of ATIF trajectories (corpus v2).

Pinned rule (spike 0B): sibling registration order in
``TrajectoryRecorder.fork()`` is append-ordered, so enumeration of the
trajectory's ``subagent_trajectory_ref`` list is the fork registration
order; ``(order_index, descriptor)`` is a total order across concurrent
forks. Segmentation must never be a coin-flip, so duplicate sibling keys
raise.
"""

from dataclasses import dataclass, field
from typing import Any

from daydream.training.corpus import _build_spans


@dataclass(frozen=True)
class Segment:
    """One per-agent segment of a session's trajectory tree."""

    segment_id: str
    trajectory_id: str
    session_id: str
    order_index: int
    descriptor: str
    spans: list[dict[str, Any]] = field(default_factory=list)


def _descriptor(trajectory_id: str) -> str:
    """Derive the agent descriptor from a ``<session>:<descriptor>`` id."""
    _, _, desc = trajectory_id.partition(":")
    return desc or trajectory_id


def segment(trajectory: dict[str, Any]) -> list[Segment]:
    """Segment a trajectory dict into per-agent ``Segment`` records.

    Ordering follows the trajectory's ``subagent_trajectory_ref`` list order
    (fork registration order, per the Task 0B pinned rule) with a stable
    ``(order_index, descriptor)`` tie-break. The root trajectory is ``seg-0``
    only when no siblings exist; otherwise siblings are ``seg-0..n-1``.

    Spans are computed per sibling document with the v1 ``_build_spans``
    helper (Pattern Q) when the ref inlines a document (``steps`` key);
    external refs (``trajectory_path`` only) carry empty spans.

    Raises:
        ValueError: when two sibling refs share the same
            ``(order_index, descriptor)`` key — the message names both.
    """
    refs = trajectory.get("subagent_trajectory_ref") or []
    if not refs:
        root_id = str(trajectory.get("trajectory_id", ""))
        return [
            Segment(
                segment_id="seg-0",
                trajectory_id=root_id,
                session_id=str(trajectory.get("session_id", "")),
                order_index=0,
                descriptor=_descriptor(root_id),
                spans=_build_spans(trajectory),
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
        spans = _build_spans(ref if "steps" in ref else {})
        segments.append(
            Segment(
                segment_id=f"seg-{order_index}",
                trajectory_id=trajectory_id,
                session_id=str(ref.get("session_id", trajectory.get("session_id", ""))),
                order_index=order_index,
                descriptor=descriptor,
                spans=spans if "steps" in ref else [],
            )
        )
    return segments


# Plan-facing alias: the segmentation entry point is also exported under the
# name used in the implementation plan.
segment_agents = segment

__all__ = ["Segment", "segment", "segment_agents"]
