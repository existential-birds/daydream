"""Bind canonical verifier decisions and select confined fix-round targets."""

from __future__ import annotations

from typing import Any

from daydream.deep.fix_state import FixCycleState
from daydream.deep.records import item_uid
from daydream.flows.engine import FlowContext
from daydream.phases import FIX_VERIFY_ACTIONABLE_VERDICTS, FIX_VERIFY_RETARGETABLE_VERDICTS

ACTIONABLE_VERDICTS = frozenset(FIX_VERIFY_ACTIONABLE_VERDICTS)
RETARGETABLE_VERDICTS = frozenset(FIX_VERIFY_RETARGETABLE_VERDICTS)


def _attach_verdicts(items: list[dict[str, Any]], payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Attach advisory verifier fields in place by canonical id/issue_id.

    Normalization must make item ids unique across all lenses. Unmatched items
    remain unchanged; return the same list.
    """
    payload = payload if isinstance(payload, dict) else {"verdicts": []}
    verdict_lookup: dict[int, dict[str, Any]] = {}
    for entry in payload.get("verdicts", []) or []:
        if not isinstance(entry, dict):
            continue
        issue_id = entry.get("issue_id")
        if not isinstance(issue_id, int):
            continue
        assumptions = entry.get("unverified_assumptions")
        verdict_lookup[issue_id] = {
            "verdict": entry.get("verdict", ""),
            "evidence": entry.get("evidence", ""),
            "unverified_assumptions": assumptions if isinstance(assumptions, list) else [],
        }
    for item in items:
        item_id = item.get("id")
        if not isinstance(item_id, int):
            continue
        match = verdict_lookup.get(item_id)
        if match is not None:
            item["verifier_verdict"] = match["verdict"]
            item["evidence"] = match["evidence"]
            item["unverified_assumptions"] = match["unverified_assumptions"]
    return items


def _verdict_buckets(
    items: list[dict[str, Any]], payload: object
) -> tuple[list[int | None], list[int | None], list[int | None], list[int | None], list[int | None]]:
    """Partition every canonical item into matched, unmatched, skipped, structural or other.

    Checked persisted selection decisions own the skip/lens split. Structural and
    wonder exemptions both use the structural bucket, never operator skips.
    """
    decisions_by_uid = _selection_decisions(payload)
    matched: list[int | None] = []
    unmatched: list[int | None] = []
    skipped: list[int | None] = []
    structural: list[int | None] = []
    other: list[int | None] = []
    for item in items:
        item_id = item.get("id")
        verdict = item.get("verifier_verdict")
        uid = item_uid(item)
        if uid not in decisions_by_uid:
            raise ValueError(f"Verifier selection omits canonical item {uid!r}")
        decision = decisions_by_uid[uid]
        exempt = decision["reason_code"] in ("exempt:structural", "exempt:wonder")
        if exempt:
            structural.append(item_id)
        elif decision["selected"] is False:
            skipped.append(item_id)
        elif verdict is not None:
            matched.append(item_id)
        elif isinstance(item_id, int) and not isinstance(item_id, bool):
            unmatched.append(item_id)
        else:
            other.append(item_id)
    return matched, unmatched, skipped, structural, other


def _selection_decisions(payload: object) -> dict[str, dict[str, Any]]:
    """Require current verifier selection decisions keyed by unique durable UID."""
    block = payload.get("selection") if isinstance(payload, dict) else None
    decisions = block.get("decisions") if isinstance(block, dict) else None
    if not isinstance(decisions, list):
        raise ValueError("Verifier artifact requires selection decisions")
    out: dict[str, dict[str, Any]] = {}
    for decision in decisions:
        if (
            not isinstance(decision, dict)
            or not isinstance(decision.get("item_uid"), str)
            or not decision["item_uid"]
            or decision["item_uid"] in out
            or type(decision.get("selected")) is not bool
            or not isinstance(decision.get("reason_code"), str)
        ):
            raise ValueError("Verifier artifact contains malformed or duplicate selection decisions")
        out[decision["item_uid"]] = decision
    return out


def _round_dispatch_items(ctx: FlowContext, canonical: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy the canonical items selected for this fix round.

    The first round dispatches all findings. Later rounds retain only actionable
    verdicts and carry their reasons into prompts. Retargeting is confined to the
    authorized edit set and cannot widen it; canonical items remain unchanged.
    """
    deep_data = ctx.deep_data()
    iteration = deep_data.get("iteration")
    state = FixCycleState.require(ctx)
    outcomes = {} if state.candidate is None else state.candidate.outcomes
    if iteration in (None, 1) or not outcomes:
        initial_dispatch = [dict(i) for i in canonical]
        for item in initial_dispatch:
            uid = item.get("item_uid")
            target = item.get("file")
            if isinstance(uid, str) and isinstance(target, str):
                state.last_fix_target_by_uid[uid] = target
        return initial_dispatch
    round_number = iteration if isinstance(iteration, int) else 1
    dispatched: list[dict[str, Any]] = []
    for item in canonical:
        uid = item.get("item_uid")
        outcome = outcomes.get(uid) if isinstance(uid, str) else None
        if not outcome or outcome.get("verdict") not in ACTIONABLE_VERDICTS:
            continue
        copy = dict(item)
        if outcome.get("verdict") in RETARGETABLE_VERDICTS:
            accepted = state.footprint.accept_retarget(
                ctx.work.repo,
                str(uid),
                outcome.get("path"),
                phase="fix",
                round_number=round_number,
            )
            if accepted is not None:
                copy["file"] = accepted
                copy["fix_verify_path"] = accepted
        copy["fix_verify_verdict"] = outcome.get("verdict")
        copy["fix_verify_reason"] = outcome.get("reason") or ""
        target = copy.get("file")
        if isinstance(uid, str) and isinstance(target, str):
            state.last_fix_target_by_uid[uid] = target
        dispatched.append(copy)
    return dispatched
