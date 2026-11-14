"""Host-stamped ledger of record targeting, bound verdicts, survival, and revisions.

Only the verdict-application seam writes adjudication-provenance.json. Write
errors propagate. Unusable reads yield no provenance so selection re-verifies;
malformed records are dropped individually.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from daydream.deep.artifacts import DeepArtifact
from daydream.json_utils import read_json_object

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
    "load_provenance",
]
