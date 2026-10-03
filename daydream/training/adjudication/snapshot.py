"""Pure canonical finding serializer and evidence digests shared by preview and harvest."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field as dataclass_field, replace
from datetime import datetime
from typing import TYPE_CHECKING, Any

from daydream.json_utils import canonical_json
from daydream.training._immutable_json import freeze_json, thaw_json
from daydream.training.adjudication.precedence import HUMAN_ROLES, effective_adjudication, reopen_on_digest_change
from daydream.training.dispositions import DECISIVE_DISPOSITIONS
from daydream.training.labeler_versions import (
    ADJUDICATION_LABELER_VERSION,
    ANNOTATION_SNAPSHOT_SCHEMA_VERSION,
    HUMAN_LABELER_VERSION,
    REPLY_CLASSIFIER_VERSION,
    reply_evidence_digest,
)

if TYPE_CHECKING:
    from daydream.training.corpus_projection.tiers import Tier
    from daydream.training.labeler_signals import PerFindingResolution


__all__ = [
    "ANNOTATION_SNAPSHOT_SCHEMA_VERSION",
    "FindingRecord",
    "record_evidence_digest",
    "snapshot_id",
]

# The K2 preview-pin components: the snapshot id is a content-addressed digest
# over exactly these fields, so idempotence and drift detection are by
# construction — any pin change produces a new snapshot id (AC 8).
_PIN_FIELDS = (
    "curation_id",
    "sanitized_hub_commit",
    "source_hub_commit",
    "archive_index_digest",
    "evidence_observed_at",
    "as_of",
    "labeler_version",
    "rubric_version",
    "classifier_version",
)


def record_evidence_digest(
    per_finding_evidence_lists: Sequence[Sequence[Mapping[str, Any]]],
) -> str | None:
    """Use the shared reply-evidence digest over flattened session evidence.

    No replies yields None, retaining the distinct digest-less dedup identity.
    """
    evidence = [thaw_json(entry) for per_finding in per_finding_evidence_lists for entry in per_finding]
    return reply_evidence_digest(evidence) if evidence else None



@dataclass(frozen=True)
class FindingRecord:
    """One owned source finding, identity and decision; wire views derive from it.

    Source resolutions may omit a digest for projection. Queue/snapshot admission
    requires one. Canonical metadata retains the historical nested wire view,
    but decisions never mutate it: archive and annotation serializers apply the
    effective disposition together to the top-level and nested views.
    """

    session_id: Any
    trajectory_id: Any
    segment_id: Any
    resolution: Mapping[str, Any]
    metadata: Mapping[str, Any] = dataclass_field(default_factory=dict)
    synchronize_nested: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "resolution", freeze_json(self.resolution))
        object.__setattr__(self, "metadata", freeze_json(self.metadata))

    @property
    def fingerprint(self) -> Any:
        return thaw_json(self.resolution["fingerprint"])

    @property
    def record_id(self) -> str:
        from daydream.training.corpus_projection.identity import record_id

        return record_id(*(str(value) for value in (
            self.session_id, self.trajectory_id, self.segment_id, self.fingerprint,
        )))

    @property
    def provenance(self) -> dict[str, Any]:
        from daydream.training.corpus_projection.provenance import extract_provenance

        return extract_provenance(thaw_json(self.resolution))

    @property
    def evidence(self) -> list[Any]:
        return [thaw_json(entry) for entry in self.resolution.get("evidence") or []]

    @property
    def tier(self) -> Tier:
        from daydream.training.corpus_projection.tiers import classify_tier

        return classify_tier(thaw_json(self.resolution))

    @classmethod
    def from_session(cls, session: Mapping[str, Any]) -> Iterator[FindingRecord]:
        for name in ("session_id", "trajectory_id", "segment_id", "resolutions"):
            if not session.get(name):
                raise ValueError(f"project_findings: session missing required key {name!r}")
        resolutions = session["resolutions"]
        if not isinstance(resolutions, list):
            raise ValueError(
                f"project_findings: session {session['session_id']!r} key 'resolutions' "
                f"must be a list, got {type(resolutions).__name__}"
            )
        for index, resolution in enumerate(resolutions):
            if not isinstance(resolution, Mapping):
                raise ValueError(
                    f"project_findings: session {session['session_id']!r} resolutions[{index}] "
                    f"is not a mapping (got {type(resolution).__name__})"
                )
            if not resolution.get("fingerprint"):
                raise ValueError(
                    f"project_findings: session {session['session_id']!r} resolutions[{index}] "
                    "missing required key 'fingerprint'"
                )
            from daydream.training.corpus_projection.tiers import classify_tier

            classify_tier(resolution)
            finding = cls(session["session_id"], session["trajectory_id"], session["segment_id"], resolution)
            yield finding

    def project(self, *, record_type: str = "outcome-finding") -> dict[str, Any]:
        from daydream.training.corpus_projection.identity import record_id
        from daydream.training.corpus_projection.tiers import classify_tier

        tier = classify_tier(thaw_json(self.resolution), record_type=record_type)
        provenance = self.provenance
        identity = self.record_id if record_type == "outcome-finding" else record_id(
            str(self.session_id), str(self.trajectory_id), str(self.segment_id),
            f"{record_type}:{self.fingerprint}",
        )
        return {
            "record_id": identity,
            "record_type": record_type,
            "session_id": self.session_id,
            "trajectory_id": self.trajectory_id,
            "task_segment": self.segment_id,
            "finding_fingerprint": self.fingerprint,
            "tier": tier,
            "disposition": self.resolution.get("disposition"),
            "outcome_label": (
                self.resolution.get("disposition")
                if record_type == "outcome-finding" and tier == "gold" else None
            ),
            "evidence": self.evidence,
            "profile": provenance["profile"],
            "stack": provenance["stack"],
        }

    def adjudication(self) -> dict[str, Any]:
        disposition = self.resolution.get("disposition")
        return {
            "fingerprint": self.fingerprint,
            "disposition": disposition,
            "evidence": self.evidence,
            "exclusion_reason": (
                f"non-decisive disposition {disposition!r} — missing decisive "
                "human verdict (evidence carried for the adjudication pass)"
            ),
        }

    def queue(self, rubric_version: str, prior: Mapping[str, Any] | None = None) -> dict[str, Any]:
        digest = self.resolution.get("evidence_digest")
        if not isinstance(digest, str) or not digest:
            raise ValueError(
                f"build_queue: fresh evidence for fingerprint {str(self.fingerprint)!r} in session "
                f"{str(self.session_id)!r} is missing required field 'evidence_digest'"
            )
        provenance = self.provenance
        profile = provenance["profile"].get("profile_name")
        if profile is None:
            profile = self.resolution.get("profile")
            if isinstance(profile, Mapping):
                profile = None
        reopened = prior is not None and prior.get("role") in HUMAN_ROLES and reopen_on_digest_change(prior, digest)
        return {
            "record_id": self.record_id,
            "fingerprint": str(self.fingerprint),
            "disposition": self.resolution["disposition"],
            "evidence": self.evidence,
            "evidence_digest": digest,
            "session_id": str(self.session_id),
            "trajectory_id": str(self.trajectory_id),
            "segment_id": str(self.segment_id),
            "profile": str(profile) if profile is not None else None,
            "stack": provenance["stack"],
            "status": "reopened" if reopened else "open",
            "rubric_version": rubric_version,
            "prior_disposition": str(prior["disposition"]) if reopened and prior is not None else None,
            "review_required": bool(prior.get("review_required", False)) if prior is not None else False,
        }

    @classmethod
    def from_annotation(cls, row: Mapping[str, Any]) -> FindingRecord:
        keys = ("fingerprint", "disposition", "evidence", "evidence_digest", "profile", "stack")
        return cls(
            row.get("session_id"),
            row.get("trajectory_id"),
            row.get("segment_id"),
            {key: row[key] for key in keys if key in row},
            {key: value for key, value in row.items() if key not in keys},
        )

    def canonical(self, *, project_conflict: bool = False) -> dict[str, Any]:
        """Project conflicts to ambiguous in both annotation views to prevent gold.

        The tier gate reads disposition/evidence, so a conflict flag alone cannot
        exclude gold. Archive rubric views retain the fresh source disposition
        for provenance; source snapshots retain absent fields and nested extras.
        """
        resolution = thaw_json(self.resolution)
        if project_conflict and self.metadata.get("conflicting"):
            resolution["disposition"] = "ambiguous"
        row = {**thaw_json(self.metadata), **resolution}
        if self.synchronize_nested or (project_conflict and self.metadata.get("conflicting")):
            row["resolutions"] = [
                {**nested, "disposition": resolution["disposition"]}
                for nested in row.get("resolutions") or []
            ]
        return row

    def adjudicate(
        self, observations: Sequence[Mapping[str, Any]], fresh: Mapping[str, Any] | None,
        *, conflicting: bool, as_of: str | None,
    ) -> tuple[FindingRecord, bool]:
        metadata = thaw_json(self.metadata)
        resolution = thaw_json(self.resolution)
        if conflicting:
            metadata["conflicting"] = True
        human = False
        if observations:
            resolved = effective_adjudication(observations)
            if resolved["evidence_digest"] == str(resolution["evidence_digest"]):
                if resolved["conflict"]:
                    metadata["conflicting"] = True
                if resolved["review_required"] and resolved["role"] in HUMAN_ROLES:
                    metadata["review_required"] = True
                    resolution["disposition"] = "ambiguous"
                if (
                    resolved["role"] in HUMAN_ROLES
                    and resolved["disposition"] in DECISIVE_DISPOSITIONS
                    and not resolved["conflict"]
                    and not resolved["review_required"]
                ):
                    resolution["disposition"] = resolved["disposition"]
                    metadata["human_labeler"], metadata["human_role"] = resolved["labeler"], resolved["role"]
                    if metadata.get("conflicting"):
                        metadata["conflicting"] = False
                    human = True
        if metadata.get("conflicting") and fresh is not None:
            resolution["disposition"] = str(fresh["disposition"])
        metadata["evidence_after_as_of"] = evidence_after_as_of(resolution, as_of)
        return replace(self, resolution=resolution, metadata=metadata), human


    @classmethod
    def snapshot(
        cls, session: Mapping[str, Any], resolution: PerFindingResolution, *, evidence_observed_at: str,
        as_of: str | None = None, conflicting: bool = False,
    ) -> FindingRecord:
        """Pin one source finding; preserve its original nested provenance view."""
        from daydream.training.corpus_projection.identity import record_id
        from daydream.training.corpus_projection.provenance import extract_provenance

        fingerprint = resolution.fingerprint
        evidence_digest = resolution.evidence_digest
        if not isinstance(evidence_digest, str) or not evidence_digest:
            raise ValueError(
                f"build_canonical_record: resolution for fingerprint {fingerprint!r} is missing "
                "required field 'evidence_digest'"
            )

        session_id = session["session_id"]
        trajectory_id = session["trajectory_id"]
        segment_id = session["segment_id"]

        # Provenance comes from the session's resolution row joined by fingerprint.
        rows = [row for row in session.get("resolutions") or [] if row.get("fingerprint") == fingerprint]
        if len(rows) != 1:
            raise ValueError(
                f"build_canonical_record: session {session_id!r} has {len(rows)} resolution rows "
                f"for fingerprint {fingerprint!r}; expected exactly 1"
            )
        provenance = extract_provenance(rows[0])

        source = {
            "fingerprint": fingerprint,
            "disposition": resolution.disposition,
            "evidence": [thaw_json(entry) for entry in resolution.evidence],
            "evidence_digest": evidence_digest,
            "profile": provenance["profile"],
            "stack": provenance["stack"],
        }
        metadata: dict[str, Any] = {
            "record_id": record_id(session_id, trajectory_id, segment_id, fingerprint),
            "session_id": session_id,
            "trajectory_id": trajectory_id,
            "segment_id": segment_id,
            "rubric_version": ADJUDICATION_LABELER_VERSION,
            "classifier_version": REPLY_CLASSIFIER_VERSION,
            "labeler_version": HUMAN_LABELER_VERSION,
            "schema_version": f"annotation-snapshot/{ANNOTATION_SNAPSHOT_SCHEMA_VERSION}",
            "evidence_observed_at": evidence_observed_at,
            # Self-contained session view: consumers that rebuild the queue or the
            # projection from canonical records (``FindingRecord.from_session``/``build_queue``)
            # consume the session shape (``resolutions`` list), and the materialized
            # per-finding record must be directly consumable without a second shape.
            "resolutions": [dict(rows[0])],
        }
        if as_of is not None:
            metadata["as_of"] = as_of
        if conflicting:
            metadata["conflicting"] = True
        return cls(session_id, trajectory_id, segment_id, source, metadata, synchronize_nested=False)


def evidence_after_as_of(record: Mapping[str, Any], as_of: str | None) -> bool:
    """Created-after-pin evidence remains visible but cannot be gold."""
    if not as_of:
        return False
    pin = datetime.fromisoformat(as_of)
    return any(
        isinstance(entry, Mapping) and entry.get("created_at")
        and datetime.fromisoformat(str(entry["created_at"])) > pin
        for entry in record.get("evidence") or []
    )

def snapshot_id(pin: Mapping[str, str]) -> str:
    """Hash canonical sorted-key JSON of exactly _PIN_FIELDS.

    Require nonempty components except as_of: missing/empty as_of has one canonical
    unpinned identity distinct from every pinned one. Invalid fields raise by name.
    """
    components = {}
    for field in _PIN_FIELDS:
        value = pin.get(field)
        if field == "as_of" and value in (None, ""):
            components[field] = ""
            continue
        if not isinstance(value, str) or not value:
            raise ValueError(f"snapshot_id: pin is missing required component {field!r}")
        components[field] = value
    canonical = canonical_json(components)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
