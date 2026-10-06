"""Pure recommendation-verifier selection and prior-verdict reuse.

Selection returns one decision per canonical item, in order, including exempt
structural and wonder findings. Missing or unreadable evidence selects an item;
only a confirmed, unrevised, strongly evidenced routine finding may be skipped.
Configuration can widen selection; unknown risk categories raise.

Changed-text evidence comes from added lines in the cited hunk, without
reapplying location snapping. Reuse requires the same rule version, durable item
identity, verifier-relevant digest, and a resolved prior verdict.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from daydream.deep.adjudication_provenance import RecordProvenance
from daydream.deep.records import item_source_uids, item_uid
from daydream.deep.risk_categories import (
    UnknownRiskCategoryError,
    categories_in,
    validate_extra_categories,
)
from daydream.hunk_index import parse_hunks
from daydream.json_utils import canonical_json

#: On-disk selection-rule version. Bump whenever the predicate's semantics
#: change; a prior verify artifact carrying another value is never reused (SH3).
SELECTION_RULE_VERSION: int = 1

#: The verifier-relevant components of one canonical item, and only those.
#: Renumbering ``id`` or attaching a verdict must never invalidate a reuse.
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

#: The single skip reason: reached only for a strongly evidenced, confirmed,
#: unrevised, non-cross-stack, non-risk item.
SKIP_REASON_CODE = "strongly_evidenced_adjudicated_routine"

#: Verdict values that always force re-verification on resume (MH9).
_UNRESOLVED_VERDICTS: frozenset[str] = frozenset({"contradicts", "uncertain"})


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
    """Resolved policy: verify_all selects every non-exempt item; extras are validated, never applied.

    The mandatory vocabulary itself is what selection reads. ``extra_categories``
    survives only as provenance -- it is echoed into the selection artifact --
    after :func:`validate_extra_categories` has rejected unknown names.
    """

    verify_all: bool = True
    extra_categories: tuple[str, ...] = ()


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


def resolve_selection_config(
    *, verify_all: bool | None, extra_categories: Iterable[str] | None
) -> SelectionConfig:
    """Default absent verify_all to True and reject unknown categories before dispatch."""
    categories = tuple(extra_categories or ())
    validate_extra_categories(categories)
    return SelectionConfig(
        verify_all=True if verify_all is None else bool(verify_all),
        extra_categories=categories,
    )


def plan_reuse(
    prior_payload: Mapping[str, Any] | None,
    decisions: Sequence[SelectionDecision],
) -> tuple[dict[str, dict[str, Any]], list[SelectionDecision]]:
    """Reuse resolved verdicts with matching rule version, item UID, and content digest.

    Only selected decisions with a matching prior verdict qualify. Contradicts and
    uncertain verdicts always re-verify. Missing or malformed input produces misses.
    Return reused verdicts keyed by UID and remapped to current item IDs, plus the
    remaining decisions in input order.
    """
    if not isinstance(prior_payload, Mapping):
        return {}, list(decisions)
    if prior_payload.get("rule_version") != SELECTION_RULE_VERSION:
        return {}, list(decisions)
    prior_decisions = prior_payload.get("decisions")
    if not isinstance(prior_decisions, list):
        return {}, list(decisions)
    verdicts = prior_payload.get("verdicts")
    verdict_by_id: dict[int, Mapping[str, Any]] = {}
    if isinstance(verdicts, list):
        for verdict in verdicts:
            if not isinstance(verdict, Mapping):
                continue
            issue_id = verdict.get("issue_id")
            if isinstance(issue_id, int) and not isinstance(issue_id, bool):
                verdict_by_id[issue_id] = verdict
    prior_by_uid: dict[str, Mapping[str, Any]] = {}
    for prior in prior_decisions:
        if isinstance(prior, Mapping) and isinstance(prior.get("item_uid"), str):
            prior_by_uid[prior["item_uid"]] = prior

    reused: dict[str, dict[str, Any]] = {}
    to_verify: list[SelectionDecision] = []
    for decision in decisions:
        prior = prior_by_uid.get(decision.item_uid)
        if (
            not decision.selected
            or prior is None
            or prior.get("content_digest") != decision.content_digest
        ):
            to_verify.append(decision)
            continue
        prior_id = prior.get("item_id")
        issue_id = (
            prior_id
            if isinstance(prior_id, int) and not isinstance(prior_id, bool)
            else decision.item_id
        )
        verdict = verdict_by_id.get(issue_id) if issue_id is not None else None
        if verdict is None or verdict.get("verdict") in _UNRESOLVED_VERDICTS:
            to_verify.append(decision)
            continue
        body = dict(verdict)
        # Re-key the reused body to the current decision's id: the verdict was
        # matched on durable uid plus the id-invariant content digest (MH14), so
        # its prior ``issue_id`` is stale whenever a resume renumbers merged
        # item ids. ``_attach_verdicts`` binds verdicts by the current
        # ``id`` -> ``issue_id`` join; a stale id would misattribute the verdict
        # to whatever item now holds the old number (or leave it unmatched).
        body["issue_id"] = decision.item_id
        reused[decision.item_uid] = body
    return reused, to_verify


def item_content_digest(item: Mapping[str, Any]) -> str:
    """Hash verifier-relevant fields, excluding positional IDs and attached verdicts.

    Missing fields hash as None; renumbering alone does not invalidate reuse.
    """
    payload = {field: item.get(field) for field in _DIGEST_FIELDS}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def select_items(
    items: Sequence[Mapping[str, Any]],
    *,
    provenance: Mapping[str, RecordProvenance] | None,
    diff_text: str,
    config: SelectionConfig,
) -> list[SelectionDecision]:
    """Classify every canonical item into select/skip, in input order (MH2).

    Returns exactly one decision per input item, including exempt structural and
    wonder-lens items. The classifier grounds on the diff text, never on index
    ranges (see :func:`changed_text_at`).
    """
    ledger = provenance if isinstance(provenance, Mapping) else None
    return [
        _decide(
            dict(item) if isinstance(item, Mapping) else {},
            ledger=ledger,
            diff_text=diff_text,
            config=config,
        )
        for item in items
    ]


def _decide(
    data: Mapping[str, Any],
    *,
    ledger: Mapping[str, RecordProvenance] | None,
    diff_text: str,
    config: SelectionConfig,
) -> SelectionDecision:
    """Apply the fixed branch order to one item; first match wins."""
    uid = item_uid(dict(data))
    item_id = data.get("id")
    lens = str(data.get("lens", ""))
    evidence, present, unadjudicated = _item_provenance(_attributed_uids(data), ledger)

    def decision(selected: bool, reason_code: str, reason: str) -> SelectionDecision:
        return SelectionDecision(
            item_uid=uid,
            item_id=item_id if isinstance(item_id, int) and not isinstance(item_id, bool) else None,
            selected=selected,
            reason_code=reason_code,
            reason=reason,
            provenance=evidence,
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
    if any(record.materially_revised for record in present):
        return decision(True, "materially_revised", "adjudication materially rewrote the finding")
    if unadjudicated:
        return decision(True, unadjudicated, _unadjudicated_reason_text(unadjudicated))
    prior = _prior_verdict(data)
    if prior in _UNRESOLVED_VERDICTS:
        return decision(True, f"prior_verdict:{prior}", f"prior verifier verdict {prior!r}")
    if not diff_text:
        # An unreadable or absent diff.patch is an absent input (``_read_diff_text``
        # degrades to ""): the changed lines a finding cites cannot be read, so a
        # mandatory risk category grounded on changed text cannot be ruled out.
        # Absent inputs select, never skip (uniform failure direction).
        return decision(True, "unreadable_diff", "diff.patch is missing or unreadable; cannot certify routine strength")
    return decision(False, SKIP_REASON_CODE, "strongly evidenced, confirmed, unrevised routine finding")


def _attributed_uids(item: Mapping[str, Any]) -> list[str]:
    """Use source_uids, falling back to item_uid; neither means unadjudicated."""
    data = dict(item)
    attributed = item_source_uids(data)
    if attributed:
        return attributed
    own = item_uid(data)
    return [own] if own else []


def _item_provenance(
    uids: Sequence[str], ledger: Mapping[str, RecordProvenance] | None
) -> tuple[dict[str, Any], list[RecordProvenance], str | None]:
    """Return ledger evidence, parsed records, and the first unadjudicated reason.

    Every attributed UID must be present, targeted, bound, and kept to clear the
    check. Missing evidence selects the item.
    """
    evidence: dict[str, Any] = {}
    present: list[RecordProvenance] = []
    if not uids or not ledger:
        return evidence, present, "unadjudicated:no_provenance"
    for uid in uids:
        record = ledger.get(uid)
        if record is None:
            continue
        evidence[uid] = record.as_dict()
    if len(evidence) != len(uids):
        return evidence, present, "unadjudicated:missing_provenance"
    present = [ledger[uid] for uid in uids]
    if any(not record.targeted for record in present):
        return evidence, present, "unadjudicated:not_targeted"
    if any(not record.verdict_bound for record in present):
        return evidence, present, "unadjudicated:verdict_unbound"
    if any(not record.kept for record in present):
        return evidence, present, "unadjudicated:not_kept"
    return evidence, present, None


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


def _prior_verdict(data: Mapping[str, Any]) -> str:
    """Return a prior verifier verdict attached to *data*, or ``""``."""
    value = data.get("verifier_verdict")
    return value if isinstance(value, str) else ""


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


def _unadjudicated_reason_text(reason_code: str) -> str:
    suffix = reason_code.partition(":")[2]
    return {
        "no_provenance": "no adjudication provenance for the item's records",
        "missing_provenance": "an attributed record is missing from the provenance ledger",
        "not_targeted": "no adjudication pass targeted the item's records",
        "verdict_unbound": "an adjudication verdict could not be bound to the record",
        "not_kept": "adjudication did not keep the record",
    }.get(suffix, "no confirmed independent adjudication")


__all__ = [
    "SELECTION_RULE_VERSION",
    "SKIP_REASON_CODE",
    "SelectionConfig",
    "SelectionDecision",
    "UnknownRiskCategoryError",
    "changed_text_at",
    "item_content_digest",
    "plan_reuse",
    "resolve_selection_config",
    "select_items",
]
