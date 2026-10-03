"""Grounding verdicts and the shared reason-code vocabulary."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from daydream.deep.diagram_types import FlowchartSpec, SequenceSpec

# --- Vocabulary --------------------------------------------------------------

# Reason codes shared by both kinds. Every one names a fact about the cited
# evidence itself, so the same check runs for a sequence message, a block
# branch and a flowchart node.
_SHARED_REASON_CODES = frozenset(
    {
        "PATH_ESCAPES_REPO",
        "FILE_MISSING",
        "LINE_OUT_OF_RANGE",
        "SYMBOL_NOT_ON_LINE",
        "NOT_A_BRANCH_STATEMENT",
        # Invalid shape has no evidence to adjudicate.
        "MALFORMED_ELEMENT",
    }
)
_SEQUENCE_REASON_CODES = frozenset(
    {
        "EVIDENCE_NOT_IN_SOURCE_PARTICIPANT",
        "CALLEE_NOT_DEFINED_IN_TARGET",
        "NOT_A_REPLY_STATEMENT",
        "PARTICIPANT_NO_FILES",
        "PARTICIPANT_FILE_MISSING",
        "REPLY_NOT_IN_ENCLOSING_FUNCTION",
        "REPLY_NOT_PRECEDED_BY_CALL",
        "EXTERNAL_MISUSED",
    }
)
_FLOWCHART_REASON_CODES = frozenset(
    {
        "ROOT_NOT_CANDIDATE",
        "NODE_OUTSIDE_ROOT",
        "NOT_AN_EXECUTABLE_STATEMENT",
        "NOT_A_TERMINAL_STATEMENT",
        "SUBROUTINE_NOT_DEFINED",
        "SUBROUTINE_NOT_CALLED_HERE",
        "DECISION_EDGES_INVALID",
        "EDGE_ENDPOINT_UNGROUNDED",
        "MULTIPLE_START",
    }
)

#: Every per-element reason code either grounder can emit. Exposed so the
#: repair-turn prompt and the tests enumerate one list rather than three.
REASON_CODES: frozenset[str] = (
    _SHARED_REASON_CODES | _SEQUENCE_REASON_CODES | _FLOWCHART_REASON_CODES
)

#: Every whole-kind omission reason. ``NO_END`` is a floor reason, not an
#: element reason: a flowchart with no terminal node is structurally fine, it
#: is simply not worth drawing.
OMIT_REASONS: frozenset[str] = frozenset(
    {
        "TOO_FEW_MESSAGES",
        "NO_CHANGED_INTERACTION",
        "TOO_FEW_PARTICIPANTS",
        "TOO_FEW_NODES",
        "NO_DECISION",
        "NO_END",
    }
)

# --- Report types ------------------------------------------------------------


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
class GroundingReport[Spec: SequenceSpec | FlowchartSpec]:
    """One diagram kind's check/prune/cap/floor result.

    Elements retain proposal order within each type. Render only when omit_reasons
    is empty. Rejected flowchart roots retain their schema-shaped proposal (or
    None), empty nodes/edges, and TOO_FEW_NODES so floor-only consumers also omit.
    """

    elements: list[ElementCheck]
    spec_final: Spec
    omit_reasons: list[str]
    rejected: str | None

    def ungrounded(self) -> list[ElementCheck]:
        """Return the failing checks, in element order (the repair-turn input)."""
        return [check for check in self.elements if not check.grounded]
