"""Pure recommendation-verifier selection.

Selection returns one decision per canonical item, in order, with the reason it
was selected. Only items of a verdict-exempt lens stay out of the verifier.

Changed-text evidence comes from added lines in the cited hunk, without
reapplying location snapping.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from daydream.deep.records import item_uid
from daydream.deep.risk_categories import categories_in
from daydream.hunk_index import parse_hunks
from daydream.json_utils import canonical_json

#: On-disk selection-rule version, stored in the verdicts artifact.
SELECTION_RULE_VERSION: int = 1

#: The verifier-relevant components of one canonical item, and only those.
_DIGEST_FIELDS: tuple[str, ...] = (
    "item_uid",
    "lens",
    "file",
    "line",
    "severity",
    "confidence",
    "description",
    "rationale",
    "evidence",
)

#: Lenses the verifier never adjudicates, with the exemption reason each records.
_EXEMPT_LENSES: dict[str, str] = {
    "structural": "exempt:structural",
    "wonder": "exempt:wonder",
}

def changed_text_at(diff_text: str, file: str, line: object) -> str:
    """Return all added text in the cited line's hunk, or "" without a signal.

    The citation itself must be an added line. Missing files, malformed or empty
    diffs, and non-positive/non-integer lines yield no signal, never a skip.
    """
    if not isinstance(line, int) or isinstance(line, bool) or line <= 0:
        return ""
    if not file:
        return ""
    info = parse_hunks(diff_text).get(file)
    if info is None:
        return ""
    added_text: dict[int, tuple[int, str]] = info["added_text"]
    hit = added_text.get(line)
    if hit is None:
        return ""
    wanted_hunk = hit[0]
    return "\n".join(
        text
        for _line, (hunk, text) in sorted(added_text.items())
        if hunk == wanted_hunk
    )


@dataclass(frozen=True)
class SelectionConfig:
    """Resolved policy: verify_all selects every non-exempt item."""

    verify_all: bool = True


@dataclass(frozen=True)
class SelectionDecision:
    """One canonical item's selection verdict, with the evidence that produced it."""

    item_uid: str
    item_id: int | None
    selected: bool
    reason_code: str
    reason: str
    provenance: dict[str, Any]
    content_digest: str
    verdict_reused: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-safe artifact body for this decision (Pattern A)."""
        return {
            "item_uid": self.item_uid,
            "item_id": self.item_id,
            "selected": self.selected,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "provenance": self.provenance,
            "content_digest": self.content_digest,
            "verdict_reused": self.verdict_reused,
        }


def item_content_digest(item: Mapping[str, Any]) -> str:
    """Hash verifier-relevant fields, excluding positional IDs and attached verdicts; missing fields hash as None."""
    payload = {field: item.get(field) for field in _DIGEST_FIELDS}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def select_items(
    items: Sequence[Mapping[str, Any]],
    *,
    diff_text: str,
    config: SelectionConfig,
) -> list[SelectionDecision]:
    """Classify every canonical item into select/skip, in input order (MH2).

    Returns exactly one decision per input item, including exempt structural and
    wonder-lens items. The classifier grounds on the diff text, never on index
    ranges (see :func:`changed_text_at`).
    """
    return [
        _decide(
            dict(item) if isinstance(item, Mapping) else {},
            diff_text=diff_text,
            config=config,
        )
        for item in items
    ]


def _decide(
    data: Mapping[str, Any],
    *,
    diff_text: str,
    config: SelectionConfig,
) -> SelectionDecision:
    """Apply the fixed branch order to one item; first match wins."""
    uid = item_uid(dict(data))
    item_id = data.get("id")
    lens = str(data.get("lens", ""))

    def decision(selected: bool, reason_code: str, reason: str) -> SelectionDecision:
        return SelectionDecision(
            item_uid=uid,
            item_id=item_id if isinstance(item_id, int) and not isinstance(item_id, bool) else None,
            selected=selected,
            reason_code=reason_code,
            reason=reason,
            provenance={},
            content_digest=item_content_digest(data),
        )

    exempt = _EXEMPT_LENSES.get(lens)
    if exempt:
        return decision(False, exempt, f"lens {lens!r} is verdict-exempt")
    if config.verify_all:
        return decision(True, "verify_all", "verify_all is enabled")
    if lens == "cross-stack":
        return decision(True, "cross_stack", "cross-stack finding")
    category = _risk_category(data, diff_text)
    if category:
        return decision(
            True, f"risk_category:{category}", f"changed text matches mandatory category {category!r}"
        )
    weak = _weak_evidence_reason(data)
    if weak:
        return decision(True, weak, _weak_reason_text(weak))
    # No reviewer records an adjudication ledger, so no finding is certified as routine.
    return decision(True, "unadjudicated:no_provenance", "no adjudication provenance for the item's records")


def _risk_category(data: Mapping[str, Any], diff_text: str) -> str | None:
    """Return the first mandatory category whose triggers occur, or ``None``.

    Reads only the item's own text, its cited file, and the added text of the
    covered hunk -- never the whole diff (Decision 4). ``categories_in`` already
    enumerates in declaration order, so its first hit *is* that first category.
    """
    file = data.get("file")
    file_text = file if isinstance(file, str) else ""
    text = "\n".join(str(data.get(key, "")) for key in ("description", "rationale", "evidence"))
    text = f"{text}\n{file_text}\n{changed_text_at(diff_text, file_text, data.get('line'))}"
    matched = categories_in(text)
    return matched[0] if matched else None


def _weak_evidence_reason(data: Mapping[str, Any]) -> str | None:
    """Check confidence, concrete evidence, then location trust; severity is irrelevant."""
    if str(data.get("confidence", "")).upper() != "HIGH":
        return "weak_evidence:confidence"
    evidence = str(data.get("evidence", "")).strip()
    if not evidence or evidence.lower() in _placeholder_evidence():
        return "weak_evidence:placeholder_evidence"
    if data.get("location_distrust") or "location_cited_line" in data:
        return "weak_evidence:located_beyond_tolerance"
    return None


def _placeholder_evidence() -> frozenset[str]:
    """Read the approval gate's vocabulary lazily to avoid the phase import cycle."""
    from daydream.phases.findings import (
        _PLACEHOLDER_EVIDENCE,
    )

    return _PLACEHOLDER_EVIDENCE


def _weak_reason_text(reason_code: str) -> str:
    suffix = reason_code.partition(":")[2]
    return {
        "confidence": "confidence is not HIGH",
        "placeholder_evidence": "evidence is blank or placeholder text",
        "located_beyond_tolerance": "location validation relocated or distrusted the citation",
    }.get(suffix, "weak evidence")


__all__ = [
    "SELECTION_RULE_VERSION",
    "SelectionConfig",
    "SelectionDecision",
    "changed_text_at",
    "item_content_digest",
    "select_items",
]
