"""Host-stamped ledger of record targeting, bound verdicts, survival, and revisions.

Only the verdict-application seam writes adjudication-provenance.json. Write
errors propagate. Unusable reads yield no provenance so selection re-verifies;
malformed records are dropped individually.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from daydream.deep.artifacts import DeepArtifact
from daydream.json_utils import atomic_write_json, read_json_object
from daydream.supervision import REVISABLE_FINDING_FIELDS

#: On-disk schema tag. A ledger carrying any other value is treated as absent.
PROVENANCE_FORMAT = "adjudication-provenance/1"


@dataclass(frozen=True)
class RecordProvenance:
    """What the adjudication passes did to one canonical record ``uid``."""

    uid: str
    passes: tuple[str, ...]
    verdict_bound: bool
    kept: bool
    revised_fields: tuple[str, ...]

    @property
    def targeted(self) -> bool:
        """True when at least one adjudication pass saw this record."""
        return bool(self.passes)

    @property
    def confirmed(self) -> bool:
        """True only when a bound verdict kept the record."""
        return self.verdict_bound and self.kept

    @property
    def materially_revised(self) -> bool:
        """True when a verdict rewrote at least one revisable field."""
        return bool(self.revised_fields)

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-safe record body written under the uid key."""
        return {
            "uid": self.uid,
            "passes": list(self.passes),
            "verdict_bound": self.verdict_bound,
            "kept": self.kept,
            "revised_fields": list(self.revised_fields),
        }

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], *, uid: str | None = None
    ) -> RecordProvenance | None:
        """Parse known fields, ignoring extras; the enclosing key can supply a missing UID."""
        resolved_uid = data.get("uid", uid)
        if not isinstance(resolved_uid, str) or not resolved_uid:
            return None
        passes = data.get("passes", ())
        if not isinstance(passes, (list, tuple)) or not all(isinstance(p, str) for p in passes):
            return None
        revised = data.get("revised_fields", ())
        if not isinstance(revised, (list, tuple)) or not all(isinstance(r, str) for r in revised):
            return None
        verdict_bound = data.get("verdict_bound", False)
        # A missing ``kept`` defaults to unkept, the fail-open direction: an
        # absent input selects, never skips (the verify-selection read path
        # treats an unkept record as needing verification). The writer always
        # emits an explicit ``kept``, so this default only fires for a
        # malformed or foreign record body.
        kept = data.get("kept", False)
        if not isinstance(verdict_bound, bool) or not isinstance(kept, bool):
            return None
        return cls(resolved_uid, tuple(passes), verdict_bound, kept, tuple(revised))


def find_revision_delta(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> tuple[str, ...]:
    """Return the revisable fields whose value changed, in declaration order.

    A field absent from ``after`` is not a change, and a missing key on either
    side is treated as ``None`` rather than an error.
    """
    changed: list[str] = []
    for field in REVISABLE_FINDING_FIELDS:
        if field not in after:
            continue
        if before.get(field) != after.get(field):
            changed.append(field)
    return tuple(changed)


def _merge_outcome(
    existing: RecordProvenance | None,
    outcome: RecordProvenance,
    pass_name: str,
) -> RecordProvenance:
    """Fold one outcome into a uid's accumulated provenance, additively."""
    passes: list[str] = list(existing.passes) if existing else []
    for name in (*outcome.passes, pass_name):
        if name and name not in passes:
            passes.append(name)
    verdict_bound = (existing.verdict_bound if existing else False) or outcome.verdict_bound
    kept = (existing.kept if existing else True) and outcome.kept
    combined = set(existing.revised_fields if existing else ()) | set(outcome.revised_fields)
    revised = [field for field in REVISABLE_FINDING_FIELDS if field in combined]
    revised.extend(field for field in sorted(combined) if field not in revised)
    return RecordProvenance(outcome.uid, tuple(passes), verdict_bound, kept, tuple(revised))


def record_provenance(
    deep_dir: Path,
    *,
    pass_name: str,
    outcomes: Iterable[RecordProvenance],
) -> Path:
    """Atomically merge outcomes by UID and write records in sorted UID order.

    Preserve first-seen pass order, OR verdict_bound, AND kept, and union revisions
    in declaration order. Write errors propagate.
    """
    merged = dict(load_provenance(deep_dir))
    for outcome in outcomes:
        merged[outcome.uid] = _merge_outcome(merged.get(outcome.uid), outcome, pass_name)
    records = {uid: merged[uid].as_dict() for uid in sorted(merged)}
    payload = {"format": PROVENANCE_FORMAT, "records": records}
    path = DeepArtifact.ADJUDICATION_PROVENANCE.at(deep_dir)
    atomic_write_json(path, payload, indent=2, trailing_newline=True)
    return path


def load_provenance(deep_dir: Path) -> dict[str, RecordProvenance]:
    """Read the typed ledger or {}; discard malformed records individually."""
    raw = read_json_object(DeepArtifact.ADJUDICATION_PROVENANCE.at(deep_dir))
    if raw.get("format") != PROVENANCE_FORMAT:
        return {}
    records = raw.get("records")
    if not isinstance(records, dict):
        return {}
    loaded: dict[str, RecordProvenance] = {}
    for uid, value in records.items():
        if not isinstance(uid, str) or not isinstance(value, dict):
            continue
        provenance = RecordProvenance.from_dict(value, uid=uid)
        if provenance is not None:
            loaded[uid] = provenance
    return loaded


__all__ = [
    "PROVENANCE_FORMAT",
    "RecordProvenance",
    "find_revision_delta",
    "load_provenance",
    "record_provenance",
]
