"""Verify for review and fix phases."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import anyio

from daydream import agent, config as phase_config, review_profile as _rp
from daydream.backends import (
    Backend,
)
from daydream.check_claims import substantiate_check_claims
from daydream.deep.adjudication_provenance import load_provenance
from daydream.deep.artifacts import DeepArtifact
from daydream.deep.verify_selection import (
    SELECTION_RULE_VERSION,
    SKIP_REASON_CODE,
    SelectionConfig,
    SelectionDecision,
    plan_reuse,
    select_items,
)
from daydream.extensions import get_registry
from daydream.json_utils import read_json_object
from daydream.phases.inputs import _recipe_for_work, append_extended_facts
from daydream.phases.schemas import (
    FIX_VERIFY_RETARGETABLE_VERDICTS,
    FIX_VERIFY_VERDICTS,
    FIX_VERIFY_VERDICTS_SCHEMA,
    RECOMMENDATION_VERDICTS_SCHEMA,
)
from daydream.run_context import RunContext, bind_resolved_run_context, resolve_run_context
from daydream.test_execution import (
    load_test_recipe,
)
from daydream.trajectory import (
    DaydreamPhase,
)
from daydream.workspace import WorkContext


def _coerce_verdicts_payload(value: Any) -> dict[str, Any]:
    """Return a verdicts list of dictionaries, dropping malformed entries individually."""
    if not isinstance(value, dict):
        return {"verdicts": []}
    raw = value.get("verdicts")
    if not isinstance(raw, list):
        return {"verdicts": []}
    return {"verdicts": [entry for entry in raw if isinstance(entry, dict)]}


def _json_or_none(value: Any) -> Any:
    """Parse *value* as JSON when it is a string, else pass it through."""
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return None


async def _run_verifier(
    backend: Backend,
    work: WorkContext,
    prompt: str,
    schema: dict[str, Any],
    run_context: RunContext,
) -> Any:
    """Run one read-only VERIFY turn and return its JSON-decoded result."""
    result, _, _ = await agent.run_agent(
        backend,
        work.repo,
        prompt,
        output_schema=schema,
        tool_call_budget=phase_config.DEFAULT_TOOL_CALL_BUDGET,
        wall_budget_s=phase_config.DEFAULT_WALL_BUDGET_S,
        phase=DaydreamPhase.VERIFY,
        read_only=True,
        run_context=run_context,
    )
    return _json_or_none(result)


@bind_resolved_run_context
async def phase_verify_recommendations(
    backend: Backend,
    work: WorkContext,
    *,
    merged_items_path: Path,
    deep_dir: Path,
    strategy: str | None = None,
    selection: SelectionConfig | None = None,
    run_context: RunContext | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Classify every canonical item and persist its selection decision beside verdicts.

    Only selected items reach the prompt; an empty selection still writes an artifact
    without a backend call. Reuse requires stable UID/content/rule version and a
    resolved prior verdict; contradicts and uncertain always re-verify. Reused
    selections carry verdict_reused. No selection config means verify all non-exempt
    items.
    """
    run_context = resolve_run_context(run_context)
    output_path = DeepArtifact.VERDICTS.at(deep_dir)

    items: list[dict[str, Any]] = json.loads(merged_items_path.read_text()).get("items", [])
    config = selection if selection is not None else SelectionConfig(verify_all=True)
    decisions = select_items(
        items,
        provenance=load_provenance(deep_dir),
        diff_text=_read_diff_text(deep_dir),
        config=config,
    )
    # Read prior evidence before overwrite; missing/malformed artifacts force full
    # re-verification rather than partial reuse or failure.
    reused, to_verify = plan_reuse(_prior_selection_payload(read_json_object(output_path)), decisions)
    decisions = [
        replace(decision, verdict_reused=True) if decision.item_uid in reused else decision
        for decision in decisions
    ]
    selection_block: dict[str, Any] = {
        "rule_version": SELECTION_RULE_VERSION,
        "mode": "verify_all" if config.verify_all else "selective",
        "extra_categories": list(config.extra_categories),
        "decisions": [decision.as_dict() for decision in decisions],
        "selected": sum(1 for decision in decisions if decision.selected),
        # Count selected skips, excluding lens exemptions; verify_all reports zero.
        "skipped": sum(1 for decision in decisions if decision.reason_code == SKIP_REASON_CODE),
        "reused": len(reused),
    }
    # Exempt decisions have no reusable verdict but must still stay out of prompts.
    pending_uids = {decision.item_uid for decision in to_verify if decision.selected}
    pending_items = [
        item
        for item, decision in zip(items, decisions, strict=True)
        if decision.selected and decision.item_uid in pending_uids
    ]
    reused_verdicts = [reused[decision.item_uid] for decision in decisions if decision.item_uid in reused]

    if not pending_items:
        # Nothing new to verify: the reused verdicts (possibly empty) are the
        # whole artifact, still schema-valid, still written (MH11/MH14).
        payload: dict[str, Any] = {"verdicts": reused_verdicts, "selection": selection_block}
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2))
        return output_path, payload

    prompt = get_registry().prompt("verify")(
        strategy=strategy if strategy is not None else _rp.build_default_profile().strategies["verification"].content,
        items=pending_items,
        cwd=work.repo,
        output_path=output_path,
    )
    prompt = append_extended_facts(prompt, load_test_recipe(deep_dir))

    candidate = await _run_verifier(
        backend, work, prompt, RECOMMENDATION_VERDICTS_SCHEMA, run_context,
    )
    payload = _coerce_verdicts_payload(candidate)
    payload["verdicts"] = _merge_reused_verdicts(reused_verdicts, payload["verdicts"], decisions, reused)
    payload["selection"] = selection_block

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    return output_path, payload


def _prior_selection_payload(artifact: dict[str, Any]) -> dict[str, Any] | None:
    """Combine prior selection metadata and verdicts for reuse; absent/foreign blocks miss."""
    selection = artifact.get("selection")
    if not isinstance(selection, dict):
        return None
    verdicts = artifact.get("verdicts")
    return {**selection, "verdicts": verdicts if isinstance(verdicts, list) else []}


def _merge_reused_verdicts(
    reused_verdicts: list[dict[str, Any]],
    fresh_verdicts: list[dict[str, Any]],
    decisions: list[SelectionDecision],
    reused: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join prior and fresh verdicts, preserving reused slots against overreaching model output."""
    reused_ids = {
        decision.item_id
        for decision in decisions
        if decision.item_uid in reused and decision.item_id is not None
    }
    kept_fresh = [
        verdict for verdict in fresh_verdicts if verdict.get("issue_id") not in reused_ids
    ]
    return [*reused_verdicts, *kept_fresh]


def _read_diff_text(deep_dir: Path) -> str:
    """Read diff.patch beside deep artifacts first, then at the artifact root; return empty text if absent."""
    for candidate in (deep_dir / "diff.patch", deep_dir.parent / "diff.patch"):
        try:
            return candidate.read_text()
        except OSError:
            continue
    return ""


@bind_resolved_run_context
async def phase_fix_verify(
    backend: Backend,
    work: WorkContext,
    items: list[dict[str, Any]],
    changed_hunks: str,
    *,
    console_lock: anyio.Lock | None = None,
    round_number: int = 1,
    run_context: RunContext | None = None,
) -> list[dict[str, Any]]:
    """Read-only audit of the supplied retained patch, with one outcome per canonical finding.

    Return resolved, unresolved, wrong_target, or regressed outcomes keyed by the
    supplied item IDs, in order. Missing verdicts become unresolved; path-bearing
    outcomes are cleaned/defaulted here because the schema cannot require paths
    conditionally. Use the registered fix-verify prompt; console_lock serializes UI.

    This phase never reverts changes. Orchestration may continue exhausted unresolved
    or wrong-target findings through tests, but blocks regressions and new actionable
    outcomes during post-test stabilization. Empty changed_hunks remains a valid
    input; the prompt limits inspection to the supplied hunks.
    """
    del console_lock  # interactive progress is not printed per finding here
    run_context = resolve_run_context(run_context)
    if not items:
        return []

    prompt = get_registry().prompt("fix-verify")(
        items=items,
        changed_hunks=changed_hunks,
        cwd=work.repo,
        round_number=round_number,
    )
    prompt = append_extended_facts(prompt, _recipe_for_work(work))
    prompt += (
        "\nIf regressed relies on a deterministic lint/test/type/build/check failure, "
        "set check_command to the repository-declared command (or null if unknown). "
        "The host validates it; do not invent diagnostic failures. "
        "Set check_only true only when the entire regressed verdict relies solely on that check failure; "
        "set it false for any semantic or additional defect.\n"
    )

    candidate = await _run_verifier(
        backend, work, prompt, FIX_VERIFY_VERDICTS_SCHEMA, run_context,
    )
    payload = _coerce_verdicts_payload(candidate)
    by_id: dict[int, dict[str, Any]] = {}
    for entry in payload["verdicts"]:
        issue_id = entry.get("issue_id")
        if not isinstance(issue_id, int):
            continue
        verdict = entry.get("verdict")
        if verdict not in FIX_VERIFY_VERDICTS:
            continue
        cleaned: dict[str, Any] = {
            "issue_id": issue_id,
            "verdict": verdict,
            "reason": entry.get("reason") or "",
        }
        if "check_command" in entry:
            cleaned["check_command"] = entry["check_command"]
        if "check_only" in entry:
            cleaned["check_only"] = entry["check_only"]
        path = entry.get("path")
        if verdict in FIX_VERIFY_RETARGETABLE_VERDICTS and isinstance(path, str) and path.strip():
            cleaned["path"] = path.strip()
        by_id[issue_id] = cleaned

    # Invariant: dispatched-count == outcome-count. Findings the agent omitted
    # get the honest non-fixed terminal, never a silent drop.
    verdicts: list[dict[str, Any]] = []
    for item in items:
        issue_id = item.get("id")
        entry = by_id.get(issue_id) if isinstance(issue_id, int) else None
        if entry is None:
            entry = {
                "issue_id": issue_id,
                "verdict": "unresolved",
                "reason": "no verifier verdict",
            }
        verdicts.append(entry)
    return await substantiate_check_claims(work.repo, verdicts, recipe=_recipe_for_work(work))
