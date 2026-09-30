"""Host-side, pure inputs for the recommendation-verifier selection decision.

Two responsibilities live here, both deterministic, total, and free of I/O,
randomness, and time:

* :func:`changed_text_at` is the reader. It answers "what text did the diff add
  at (or around) the line this finding cites?" from the run's own ``diff.patch``
  text. It walks the unified diff once, tracks the current post-state file (the
  ``+++ b/`` header, unquoted exactly as ``daydream/hunk_index.py`` unquotes it)
  and the new-side line counter, and returns the newline-joined text of the
  ``+`` content lines in the hunk covering ``line`` for ``file``. A cited line
  that is not itself an added line yields the empty string, as does a missing
  file, a malformed or empty diff, or a non-positive line number -- the
  classifier reads ``""`` as "no changed-line signal", never as a skip signal
  (Pattern B, fail-open). The reader is deliberately range-free: a cited line
  the hunk index snapped is already reflected in ``item["line"]`` before this
  runs, so importing the index for ranges would double-apply the snap.
* :func:`select_items` is the predicate. It classifies every canonical merged
  item into select/skip from four persisted inputs -- the item's own text, the
  changed lines it cites, the host-stamped adjudication provenance ledger, and
  (on resume) the prior verify artifact -- with the first matching branch in a
  fixed order winning. It returns exactly one :class:`SelectionDecision` per
  input item, in input order, including exempt items, so the decisions
  reconcile 1:1 with the canonical item list (MH10). Structural and wonder-lens
  items are marked exempt rather than silently omitted; ``verify_all``
  reproduces today's "verify every non-exempt item" behaviour and configuration
  can only widen selection (MH13).

Failure direction is uniform: every unreadable or absent input selects, never
skips. The only exception is :class:`UnknownRiskCategoryError`, which is a
configuration error surfaced to the operator, not a data error. The predicate
never raises for a malformed item, a missing ledger, a missing diff, or a
missing hunk index.
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
    resolve_mandatory_categories,
)
from daydream.hunk_index import _HUNK_HEADER, _header_path
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
    """Return the added text of the hunk a cited ``file``/``line`` falls in.

    Walks ``diff_text`` once, resolving the current post-state path from each
    ``+++`` header and the new-side line counter from each hunk header. When
    ``line`` is one of the added (``+``) lines of ``file``, returns the
    newline-joined text of every added line in that same hunk (a hunk's added
    text is the unit the classifier reads); otherwise returns ``""``.

    Args:
        diff_text: Unified-diff text, as written to ``diff.patch``.
        file: Repo-relative path to look up.
        line: New-side line number the finding cites.

    Returns:
        The added lines' text, or ``""`` when there is no changed-line signal.
    """
    if not isinstance(line, int) or isinstance(line, bool) or line <= 0:
        return ""
    if not file:
        return ""

    # new_line -> (hunk ordinal, added text), collected only for the queried file.
    added_by_line: dict[int, tuple[int, str]] = {}
    current_file: str | None = None
    current_hunk: int | None = None
    hunk_count = 0
    new_line = 0
    prev_old_header = False
    for raw in diff_text.splitlines():
        if raw.startswith(("--- ", '--- "')):
            prev_old_header = True
            continue
        if raw.startswith("+++ ") and prev_old_header:
            prev_old_header = False
            current_file = _header_path(raw)
            current_hunk = None
            new_line = 0
            continue
        prev_old_header = False
        header = _HUNK_HEADER.match(raw)
        if raw.startswith("@@") and header:
            new_start = int(header.group(3))
            new_count = int(header.group(4)) if header.group(4) else 1
            new_line = new_start
            if new_count == 0:
                # Empty new-side range (pure deletion): no added lines to read.
                current_hunk = None
                continue
            current_hunk = hunk_count
            hunk_count += 1
        elif current_file == file and raw.startswith("+"):
            if current_hunk is not None:
                added_by_line[new_line] = (current_hunk, raw[1:])
            new_line += 1
        elif raw.startswith("-"):
            continue
        elif raw.startswith(" "):
            new_line += 1

    hit = added_by_line.get(line)
    if hit is None:
        return ""
    wanted_hunk = hit[0]
    return "\n".join(
        text
        for added_line, (hunk, text) in sorted(added_by_line.items())
        if hunk == wanted_hunk
    )


@dataclass(frozen=True)
class SelectionConfig:
    """Resolved verify-selection policy.

    ``verify_all`` is the conservative toggle: ``True`` reproduces today's
    "verify every non-exempt item" behaviour. ``extra_categories`` names
    additive risk categories and is always validated against the declared
    vocabulary by :meth:`categories`; it can never remove a built-in category.
    """

    verify_all: bool = True
    extra_categories: tuple[str, ...] = ()

    @property
    def categories(self) -> tuple[str, ...]:
        """The built-in mandatory categories followed by the validated extras."""
        return resolve_mandatory_categories(self.extra_categories)


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
    """Resolve the config-file knobs into a frozen :class:`SelectionConfig`.

    An absent ``verify_all`` degrades to ``True`` here, the fail-safe for a
    direct caller that has not resolved the run-level ``DEFAULT_VERIFY_ALL``; a
    non-bool is treated as absent. Every extra category is validated here so an
    unrecognised name fails loudly (:class:`UnknownRiskCategoryError`) before
    any backend call rather than silently widening or narrowing selection.
    """
    categories = tuple(extra_categories or ())
    resolve_mandatory_categories(categories)
    return SelectionConfig(
        verify_all=True if verify_all is None else bool(verify_all),
        extra_categories=categories,
    )


def plan_reuse(
    prior_payload: Mapping[str, Any] | None,
    decisions: Sequence[SelectionDecision],
) -> tuple[dict[str, dict[str, Any]], list[SelectionDecision]]:
    """Split *decisions* into reusable prior verdicts and items to re-verify.

    A verdict is reused only when every one of these holds (SH3/MH14):

    * the prior artifact's ``rule_version`` equals
      :data:`SELECTION_RULE_VERSION` -- a semantics move invalidates all reuse;
    * the decision is selected (an exempt/skipped item has no verdict to reuse);
    * the prior artifact carries a decision for the same ``item_uid`` whose
      ``content_digest`` equals this decision's -- durable identity plus
      verifier-relevant content, never the positional ``id``;
    * a verdict was recorded for that item, keyed by its canonical ``issue_id``.

    MH9 is re-checked here rather than trusted from the prior artifact: a prior
    ``contradicts`` or ``uncertain`` verdict always lands in ``to_verify``,
    whatever the digest says. Any absent, unreadable, or malformed prior
    payload is a total miss -- every decision re-verifies, never a partial
    reuse and never a raise (Pattern B).

    Returns ``(reused, to_verify)`` where ``reused`` maps ``item_uid`` to the
    verbatim prior verdict body and ``to_verify`` preserves input order.
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
        reused[decision.item_uid] = dict(verdict)
    return reused, to_verify


def item_content_digest(item: Mapping[str, Any]) -> str:
    """Return a sha256 over the verifier-relevant components of *item*.

    Only :data:`_DIGEST_FIELDS` participate, so renumbering ``id`` or attaching
    a verdict cannot invalidate a reuse while a text change can (MH14). Missing
    fields are digested as ``None`` so absence is a stable value, not an error.
    """
    payload = {field: item.get(field) for field in _DIGEST_FIELDS}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def select_items(
    items: Sequence[Mapping[str, Any]],
    *,
    provenance: Mapping[str, RecordProvenance] | None,
    hunk_index: Mapping[str, Any],
    diff_text: str,
    config: SelectionConfig,
) -> list[SelectionDecision]:
    """Classify every canonical item into select/skip, in input order (MH2).

    Returns exactly one decision per input item, including exempt structural and
    wonder-lens items. ``hunk_index`` is accepted for symmetry with the run's
    persisted inputs but is intentionally unread: the classifier grounds on the
    diff text, never on index ranges (see :func:`changed_text_at`).
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
    category = _risk_category(data, diff_text, config)
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
    return decision(False, SKIP_REASON_CODE, "strongly evidenced, confirmed, unrevised routine finding")


def _attributed_uids(item: Mapping[str, Any]) -> list[str]:
    """Return the record uids whose provenance adjudicates *item*.

    The merge agent's ``source_uids`` lead; an item that never went through the
    merge agent falls back to its own ``item_uid``. An item with neither has no
    usable attribution and is unadjudicated (MH7).
    """
    data = dict(item)
    attributed = item_source_uids(data)
    if attributed:
        return attributed
    own = item_uid(data)
    return [own] if own else []


def _item_provenance(
    uids: Sequence[str], ledger: Mapping[str, RecordProvenance] | None
) -> tuple[dict[str, Any], list[RecordProvenance], str | None]:
    """Return ``(evidence, present, unadjudicated_reason)`` for one item.

    ``evidence`` is the JSON-safe per-uid ledger body; ``present`` the parsed
    records, for the revision test; ``unadjudicated_reason`` is ``None`` only
    when every attributed uid is present and confirmed. Any missing, untargeted,
    unbound, or unkept record selects the item (the fail direction).
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


def _risk_category(
    data: Mapping[str, Any], diff_text: str, config: SelectionConfig
) -> str | None:
    """Return the first mandatory category whose triggers occur, or ``None``.

    Reads only the item's own text, its cited file, and the added text of the
    covered hunk -- never the whole diff (Decision 4).
    """
    file = data.get("file")
    file_text = file if isinstance(file, str) else ""
    text = "\n".join(str(data.get(key, "")) for key in ("description", "rationale", "evidence"))
    text = f"{text}\n{file_text}\n{changed_text_at(diff_text, file_text, data.get('line'))}"
    matched = set(categories_in(text))
    for category in config.categories:
        if category in matched:
            return category
    return None


def _weak_evidence_reason(data: Mapping[str, Any]) -> str | None:
    """Return the weak-evidence reason code for *data*, or ``None``.

    Exactly three conditions, checked in order; severity is never consulted
    (MH5).
    """
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
    """The placeholder-evidence vocabulary, shared with the approval gate.

    Imported lazily: ``daydream.phases`` imports this module, so a top-level
    import would be a cycle.
    """
    from daydream.phases import _PLACEHOLDER_EVIDENCE

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
