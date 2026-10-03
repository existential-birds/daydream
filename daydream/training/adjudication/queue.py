"""Build deterministic record_id-keyed queues from validated source findings.

Projection and adjudication share finding validation and disposition policy.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from daydream.training.adjudication.precedence import HUMAN_ROLES, reopen_on_digest_change
from daydream.training.corpus_projection.identity import record_id
from daydream.training.corpus_projection.projector import finding_resolutions
from daydream.training.corpus_projection.provenance import extract_provenance
from daydream.training.dispositions import (
    NON_DECISIVE_DISPOSITIONS as _NON_DECISIVE_DISPOSITIONS,
)
from daydream.training.labeler_versions import ADJUDICATION_LABELER_VERSION

__all__ = ["build_queue", "_NON_DECISIVE_DISPOSITIONS"]


def _profile_label(
    resolution: Mapping[str, object], provenance: Mapping[str, Any]
) -> str | None:
    """Read canonical profile_name or legacy flat/nested profile through extract_provenance.

    Absent labels remain absent; never stringify None or a profile dictionary.
    """
    profile_name = provenance["profile"].get("profile_name")
    if profile_name is not None:
        return str(profile_name)
    flat = resolution.get("profile")
    if isinstance(flat, Mapping):
        return None
    return str(flat) if flat is not None else None

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
    "prior_disposition",
    "review_required",
)


def build_queue(
    sessions: Sequence[Mapping[str, object]],
    *,
    rubric_version: str = ADJUDICATION_LABELER_VERSION,
    prior_observations: Mapping[str, Mapping[str, Any]] | None = None,
    include_decisive: bool = False,
) -> list[dict[str, object]]:
    """Rebuild a record_id-sorted queue with evidence-drift reopening.

    Select non-decisive source findings. Findings with prior observations also remain
    available for human review, even when automatically decisive. include_decisive
    adds the complete record set for drift checks using the same item shape,
    status, and observation logic.

    A completed human judgment with an old evidence digest reopens with its prior
    disposition retained. Other items start open. Missing fresh digests or invalid
    dispositions raise ValueError naming the fingerprint. Identical inputs produce
    identical queues regardless of session order.
    """
    items: list[dict[str, object]] = []
    for session in sessions:
        session_id = str(session.get("session_id"))
        trajectory_id = str(session.get("trajectory_id"))
        segment_id = str(session.get("segment_id"))
        for resolution, _tier in finding_resolutions(session):
            fingerprint = str(resolution["fingerprint"])
            disposition = resolution["disposition"]
            finding_id = record_id(session_id, trajectory_id, segment_id, fingerprint)
            if (
                not include_decisive and disposition not in _NON_DECISIVE_DISPOSITIONS
                and finding_id not in (prior_observations or {})
            ):
                continue
            provenance = extract_provenance(resolution)
            fresh_digest = resolution.get("evidence_digest")
            if not isinstance(fresh_digest, str) or not fresh_digest:
                raise ValueError(
                    f"build_queue: fresh evidence for fingerprint {fingerprint!r} in session "
                    f"{session_id!r} is missing required field 'evidence_digest'"
                )
            item: dict[str, object] = {
                "record_id": finding_id,
                "fingerprint": fingerprint,
                "disposition": disposition,
                "evidence": list(resolution.get("evidence") or []),
                "evidence_digest": fresh_digest,
                "session_id": session_id,
                "trajectory_id": trajectory_id,
                "segment_id": segment_id,
                "profile": _profile_label(resolution, provenance),
                "stack": provenance["stack"],
                "status": "open",
                "rubric_version": rubric_version,
                "prior_disposition": None,
                "review_required": False,
            }
            prior = (prior_observations or {}).get(str(item["record_id"]))
            if prior is not None:
                # Propagate a stored review-required flag (e.g. model-suggested
                # labels) so `show` can render the item as needing review.
                item["review_required"] = bool(prior.get("review_required", False))
            if (
                prior is not None
                and prior.get("role") in HUMAN_ROLES
                and reopen_on_digest_change(prior, current_digest=fresh_digest)
            ):
                item["status"] = "reopened"
                item["prior_disposition"] = str(prior["disposition"])
            assert set(item) == set(_ITEM_KEYS), "queue item key drift"
            items.append(item)
    items.sort(key=lambda item: str(item["record_id"]))
    return items
