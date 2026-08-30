"""Per-finding projection for corpus v2 (Req 6, Req 18, D8).

Each ``PerFindingResolution`` becomes its own record with a per-record
``outcome_label`` — a mixed session (some findings accepted, some rejected)
never collapses into a run-level aggregate like v1's ``contested``
``outcome_label``. Non-decisive dispositions route to an adjudication
report (report output, not a pipeline stage); the projector stays pure and
deterministic.

``run_build_corpus_v2()`` wiring lands in a later task; this module owns
only the record-assembly core.
"""

from typing import Literal, Mapping, overload

from daydream.training.corpus_v2.identity import record_id
from daydream.training.corpus_v2.tiers import classify_tier

__all__ = ["project_findings"]

Record = dict[str, object]


@overload
def project_findings(session: Mapping[str, object], *, return_adjudication: Literal[False] = False) -> list[Record]: ...


@overload
def project_findings(
    session: Mapping[str, object], *, return_adjudication: Literal[True]
) -> tuple[list[Record], list[Record]]: ...


def project_findings(
    session: Mapping[str, object], *, return_adjudication: bool = False
) -> list[Record] | tuple[list[Record], list[Record]]:
    """Project one segmented session's per-finding resolutions into records.

    Returns the record list, or ``(records, adjudication_entries)`` when
    ``return_adjudication`` is true. Raises ``ValueError`` naming the
    session and the offending key on a malformed resolution.
    """
    session_id = session.get("session_id")
    trajectory_id = session.get("trajectory_id")
    segment_id = session.get("segment_id")
    resolutions = session.get("resolutions")
    for name, value in (
        ("session_id", session_id),
        ("trajectory_id", trajectory_id),
        ("segment_id", segment_id),
        ("resolutions", resolutions),
    ):
        if not value:
            raise ValueError(f"project_findings: session missing required key {name!r}")
    if not isinstance(resolutions, list):
        raise ValueError(
            f"project_findings: session {session_id!r} key 'resolutions' "
            f"must be a list, got {type(resolutions).__name__}"
        )

    records: list[Record] = []
    adjudication: list[Record] = []
    for index, resolution in enumerate(resolutions):
        if not isinstance(resolution, Mapping):
            raise ValueError(
                f"project_findings: session {session_id!r} resolutions[{index}] "
                f"is not a mapping (got {type(resolution).__name__})"
            )
        fingerprint = resolution.get("fingerprint")
        if not fingerprint:
            raise ValueError(
                f"project_findings: session {session_id!r} resolutions[{index}] "
                "missing required key 'fingerprint'"
            )
        tier = classify_tier(resolution)
        disposition = resolution.get("disposition")
        evidence = list(resolution.get("evidence") or [])
        record = {
            "record_id": record_id(
                str(session_id), str(trajectory_id), str(segment_id), str(fingerprint)
            ),
            "record_type": "outcome-finding",
            "session_id": session_id,
            "trajectory_id": trajectory_id,
            "segment_id": segment_id,
            "finding_fingerprint": fingerprint,
            "tier": tier,
            "disposition": disposition,
            "outcome_label": disposition if tier == "gold" else None,
            "evidence": evidence,
        }
        records.append(record)
        if tier == "task-only":
            adjudication.append(
                {
                    "fingerprint": fingerprint,
                    "disposition": disposition,
                    "evidence": evidence,
                    "reason": f"non-decisive disposition {disposition!r}",
                }
            )

    if return_adjudication:
        return records, adjudication
    return records
