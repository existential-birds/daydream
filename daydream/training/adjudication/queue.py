"""Per-finding adjudication queue generation (issue #984, Task 1).

Builds a deterministic, ``record_id``-keyed adjudication queue from segmented
sessions by consuming the projector's ``return_adjudication=True`` second
return value — the projector is the single enumeration authority for the
non-decisive set, so the queue builder adds no second filtering pass.
"""

from typing import Mapping, Sequence

from daydream.training.corpus_v2.identity import record_id
from daydream.training.corpus_v2.projector import project_findings
from daydream.training.labeler_versions import ADJUDICATION_LABELER_VERSION

__all__ = ["build_queue"]

_NON_DECISIVE_DISPOSITIONS = frozenset({"ambiguous", "unanswered", "missing"})

_ITEM_KEYS = (
    "record_id",
    "fingerprint",
    "disposition",
    "evidence",
    "evidence_digest",
    "session_id",
    "trajectory_id",
    "segment_id",
    "profile",
    "stack",
    "status",
    "rubric_version",
)


def build_queue(
    sessions: Sequence[Mapping[str, object]], *, rubric_version: str = ADJUDICATION_LABELER_VERSION
) -> list[dict[str, object]]:
    """Build the adjudication queue items for the given segmented sessions.

    Each non-decisive (task-only) adjudication entry becomes one queue item
    keyed by the recomputed ``record_id``. Decisive/gold resolutions never
    enter the queue: only the projector's adjudication set is consumed, and
    every entry's disposition is additionally asserted to be non-decisive —
    a violation raises ``ValueError`` naming the fingerprint (fail-closed).

    Ordering is by ``record_id`` (hex digest), so identical inputs produce
    byte-identical queues regardless of session order.
    """
    items: list[dict[str, object]] = []
    for session in sessions:
        _, adjudication = project_findings(session, return_adjudication=True)
        session_id = str(session.get("session_id"))
        trajectory_id = str(session.get("trajectory_id"))
        segment_id = str(session.get("segment_id"))
        by_fingerprint: dict[str, Mapping[str, object]] = {}
        resolutions = session.get("resolutions")
        for raw in resolutions if isinstance(resolutions, list) else []:
            if isinstance(raw, Mapping) and raw.get("fingerprint"):
                by_fingerprint[str(raw["fingerprint"])] = raw
        for entry in adjudication:
            fingerprint = str(entry["fingerprint"])
            disposition = entry["disposition"]
            if disposition not in _NON_DECISIVE_DISPOSITIONS:
                raise ValueError(
                    f"build_queue: adjudication entry for fingerprint {fingerprint!r} has "
                    f"non-queue disposition {disposition!r}; expected one of "
                    f"{sorted(_NON_DECISIVE_DISPOSITIONS)}"
                )
            resolution = by_fingerprint.get(fingerprint)
            if resolution is None:
                raise ValueError(
                    f"build_queue: adjudication entry fingerprint {fingerprint!r} not found "
                    f"in session {session_id!r} resolutions"
                )
            item: dict[str, object] = {
                "record_id": record_id(session_id, trajectory_id, segment_id, fingerprint),
                "fingerprint": fingerprint,
                "disposition": disposition,
                "evidence": entry["evidence"],
                "evidence_digest": resolution.get("evidence_digest"),
                "session_id": session_id,
                "trajectory_id": trajectory_id,
                "segment_id": segment_id,
                "profile": resolution.get("profile"),
                "stack": resolution.get("stack"),
                "status": "open",
                "rubric_version": rubric_version,
            }
            assert set(item) == set(_ITEM_KEYS), "queue item key drift"
            items.append(item)
    items.sort(key=lambda item: str(item["record_id"]))
    return items
