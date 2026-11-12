"""Grounding verdicts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ElementCheck:
    """One element's evidence verdict, keyed by a stable within-kind ref.

    Grounded elements can still be pruned, flattened or capped. strength records
    whether a definition or only a token proved the symbol; defined_at cites the
    definition. snapped_line records citation repair within three lines and is
    also applied to spec_final. in_changed_hunk uses the repaired citation.
    """

    element: str
    ref: str
    grounded: bool
    reason: str | None = None
    strength: str | None = None
    snapped_line: int | None = None
    in_changed_hunk: bool = False
    defined_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe repair-turn input."""
        return asdict(self)


@dataclass
class GroundingReport:
    """One diagram kind's check/prune/cap/floor result.

    Elements retain proposal order within each type. Render only when omit_reasons
    is empty. Rejected flowchart roots retain their schema-shaped proposal (or
    None), empty nodes/edges, and TOO_FEW_NODES so floor-only consumers also omit.
    """

    elements: list[ElementCheck]
    spec_final: dict[str, Any]
    omit_reasons: list[str]
    rejected: str | None

    def ungrounded(self) -> list[ElementCheck]:
        """Return the failing checks, in element order (the repair-turn input)."""
        return [check for check in self.elements if not check.grounded]
