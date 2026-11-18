"""Verify for review and fix phases."""

import json
from pathlib import Path
from typing import Any

from daydream import agent, config as phase_config, review_profile as _rp
from daydream.backends import (
    Backend,
)
from daydream.check_claims import substantiate_check_claims
from daydream.deep.artifacts import DeepArtifact
from daydream.extensions import get_registry
from daydream.phases.inputs import _recipe_for_work, append_extended_facts
from daydream.phases.schemas import (
    FIX_VERIFY_RETARGETABLE_VERDICTS,
    FIX_VERIFY_VERDICTS,
    FIX_VERIFY_VERDICTS_SCHEMA,
    RECOMMENDATION_VERDICTS_SCHEMA,
)
from daydream.review_budget import BudgetLedger
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
        budget=BudgetLedger("verify", wall_s=phase_config.DEFAULT_WALL_BUDGET_S),
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
    run_context: RunContext | None = None,
) -> dict[str, Any]:
    """Classify every canonical item and persist the verdicts the verifier returns for them."""
    from daydream.deep.review_terminal import load_canonical_items  # review_terminal imports phases

    run_context = resolve_run_context(run_context)
    output_path = DeepArtifact.VERDICTS.at(deep_dir)

    # Validate again: an extension step may have rewritten the items since the fix gate.
    items = load_canonical_items(work.repo, merged_items_path)
    prompt = get_registry().prompt("verify")(
        strategy=strategy if strategy is not None else _rp.build_default_profile().strategies["verification"].content,
        items=items,
        cwd=work.repo,
        output_path=output_path,
    )
    prompt = append_extended_facts(prompt, load_test_recipe(deep_dir))

    candidate = await _run_verifier(
        backend, work, prompt, RECOMMENDATION_VERDICTS_SCHEMA, run_context,
    )
    payload = _coerce_verdicts_payload(candidate)
    payload["verdicts"] = _asked_verdicts(payload["verdicts"], items)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    return payload


def _asked_verdicts(fresh_verdicts: list[dict[str, Any]], items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only the verdicts for the findings this turn was asked to verify."""
    asked = {item["id"] for item in items}
    return [verdict for verdict in fresh_verdicts if verdict.get("issue_id") in asked]


@bind_resolved_run_context
async def phase_fix_verify(
    backend: Backend,
    work: WorkContext,
    items: list[dict[str, Any]],
    changed_hunks: str,
    *,
    round_number: int = 1,
    run_context: RunContext | None = None,
) -> list[dict[str, Any]]:
    """Read-only audit of the supplied retained patch, with one outcome per canonical finding.

    Return resolved, unresolved, wrong_target, or regressed outcomes keyed by the
    supplied item IDs, in order. Missing verdicts become unresolved; path-bearing
    outcomes are cleaned/defaulted here because the schema cannot require paths
    conditionally. Use the registered fix-verify prompt.

    This phase never reverts changes. Orchestration may continue exhausted unresolved
    or wrong-target findings through tests, but blocks regressions and new actionable
    outcomes during post-test stabilization. Empty changed_hunks remains a valid
    input; the prompt limits inspection to the supplied hunks.
    """
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
        entry = by_id.get(item["id"])
        if entry is None:
            entry = {
                "issue_id": item["id"],
                "verdict": "unresolved",
                "reason": "no verifier verdict",
            }
        verdicts.append(entry)
    return await substantiate_check_claims(work.repo, verdicts, recipe=_recipe_for_work(work))
