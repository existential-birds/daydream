"""Build deterministic record_id-keyed queues using the projector's finding enumeration.

Its adjudication output owns the non-decisive set; do not add a second filter.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from daydream.training.adjudication.precedence import HUMAN_ROLES, reopen_on_digest_change
from daydream.training.corpus_projection.projector import project_findings
from daydream.training.corpus_projection.provenance import extract_provenance
from daydream.training.dispositions import (
    NON_DECISIVE_DISPOSITIONS as _NON_DECISIVE_DISPOSITIONS,
    is_decisive,
)
from daydream.training.labeler_versions import ADJUDICATION_LABELER_VERSION
from daydream.training.record_identity import record_finding_id

__all__ = ["build_queue", "_NON_DECISIVE_DISPOSITIONS"]


def _profile_label(resolution: Mapping[str, object], provenance: Mapping[str, Any]) -> str | None:
    """Read canonical profile_name or flat/nested profile through extract_provenance.

    Absent labels remain absent; never stringify None or a profile dictionary.
    """
    profile_name = provenance["profile"].get("profile_name")
    if profile_name is not None:
        return str(profile_name)
    flat = resolution.get("profile")
    if isinstance(flat, Mapping):
        nested_name = flat.get("profile_name")
        if nested_name is not None:
            return str(nested_name)
        return None
    return str(flat) if flat is not None else None


_ITEM_KEYS = (
    "record_id",
    "fingerprint",
    "disposition",
    "evidence",
    "evidence_digest",
    "evidence_digest_scheme",
    "reply_captures",
    "correction",
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

    Normally consume the projector's non-decisive adjudication entries, asserting
    that disposition contract. Findings with prior observations also remain
    available for human review, even when automatically decisive. include_decisive
    adds the complete record set for drift checks using the same item shape,
    status, and observation logic.

    A completed human judgment with an old evidence digest reopens with its prior
    disposition retained. Other items start open. Missing fresh digests or invalid
    dispositions raise ValueError naming the fingerprint. Identical inputs produce
    identical queues regardless of session order.
    """
    expanded: list[Mapping[str, object]] = []
    for session in sessions:
        resolutions = session.get("resolutions")
        if (
            isinstance(resolutions, list)
            and resolutions
            and all(isinstance(r, Mapping) and r.get("item_uid") for r in resolutions)
        ):
            expanded.extend({**session, "resolutions": [r]} for r in resolutions)
        else:
            expanded.append(session)
    items: list[dict[str, object]] = []
    for session in expanded:
        records, adjudication = project_findings(session, return_adjudication=True)
        entries = adjudication + [r for r in records if is_decisive(str(r.get("disposition")))]
        for entry in adjudication:
            if entry["disposition"] not in _NON_DECISIVE_DISPOSITIONS:
                raise ValueError(
                    f"build_queue: adjudication entry for fingerprint "
                    f"{entry.get('fingerprint') or entry.get('finding_fingerprint')!r} "
                    f"has non-queue disposition {entry['disposition']!r}"
                )
        session_id = str(session.get("session_id"))
        trajectory_id = str(session.get("trajectory_id"))
        segment_id = str(session.get("segment_id"))
        by_fingerprint: dict[str, Mapping[str, object]] = {}
        resolutions = session.get("resolutions")
        for raw in resolutions if isinstance(resolutions, list) else []:
            if isinstance(raw, Mapping) and raw.get("fingerprint"):
                by_fingerprint[str(raw["fingerprint"])] = raw
        for entry in entries:
            fingerprint = str(entry.get("fingerprint") or entry.get("finding_fingerprint"))
            disposition = entry["disposition"]
            finding_id = record_finding_id(session_id, trajectory_id, segment_id, str(entry["item_uid"]))
            if (
                not include_decisive
                and disposition not in _NON_DECISIVE_DISPOSITIONS
                and finding_id not in (prior_observations or {})
            ):
                continue
            resolution = by_fingerprint.get(fingerprint)
            if resolution is None:
                raise ValueError(
                    f"build_queue: adjudication entry fingerprint {fingerprint!r} not found "
                    f"in session {session_id!r} resolutions"
                )
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
                "evidence": entry["evidence"],
                "evidence_digest": fresh_digest,
                "evidence_digest_scheme": resolution["evidence_digest_scheme"],
                "reply_captures": resolution.get("reply_captures", []),
                "correction": resolution.get("correction"),
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
