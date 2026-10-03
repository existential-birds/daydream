"""Batched supervision, arbitration, and suppression over already discovered findings."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Required, TypedDict, Unpack

from daydream import agent, config as phase_config, review_profile, ui
from daydream.artifact_visibility import ArtifactSession
from daydream.backends import Backend, ContinuationToken
from daydream.deep.artifacts import deep_dir
from daydream.deep.records import record_uid
from daydream.extensions import get_registry
from daydream.json_utils import validates_schema
from daydream.phases.inputs import _prepare_existing_phase_inputs, append_extended_facts
from daydream.phases.schemas import ARBITER_SCHEMA, SUPERVISE_SCHEMA, SUPPRESSION_SCHEMA
from daydream.prompts.authorial_intent import AUTHORITATIVE_INTENT_BLOCK
from daydream.review_budget import ReviewLimits
from daydream.review_evidence import FinalizationContext
from daydream.run_context import RunContext, bind_resolved_run_context, resolve_run_context
from daydream.test_execution import load_test_recipe
from daydream.trajectory import DaydreamPhase
from daydream.workspace import WorkContext


class IncompleteVerdicts(dict[int, dict[str, Any]]):
    """Compatible empty verdict mapping preserving a terminal budget witness."""

    def __init__(self, reason: str, verdicts: dict[int, dict[str, Any]] | None = None) -> None:
        super().__init__(verdicts or {})
        self.budget_reason = reason


class _IncompleteAdjudication(list[dict[str, Any]]):
    def __init__(self, reason: str) -> None:
        super().__init__()
        self.budget_reason = reason


class AdjudicationInputs(TypedDict, total=False):
    """Grounding and runtime controls shared by all adjudication phases."""

    diff_path: Required[Path]
    intent_path: Required[Path]
    alternatives_path: Required[Path]
    exploration_dir: Path | None
    strategy: str | None
    artifact_session: ArtifactSession | None
    allow_standalone: bool
    run_context: RunContext | None


@dataclass(frozen=True)
class _Adjudication:
    name: str
    strategy: str
    hero: str
    label: str
    progress: str
    schema: dict[str, Any]
    result_key: str
    tool_limit: int
    task: str
    semantics: str


_SUPERVISOR = _Adjudication(
    "supervise", "supervision", "SUPERVISE", "Supervisor", "Supervising {count} merged finding(s)",
    SUPERVISE_SCHEMA, "verdicts", 12,
    "Finalize supervision of supplied canonical findings",
    "Return verdicts only for supplied canonical integer ids. "
    "Do not invent findings or claim unresolved adjudication complete.",
)
_ARBITER = _Adjudication(
    "arbiter", "arbitration", "ARBITRATE", "Arbiter", "Arbitrating {count} high-severity/contested finding(s)",
    ARBITER_SCHEMA, "findings", 16,
    "Finalize arbitration of supplied findings",
    "Echo arb_id for each resolved input finding, preserving its identity. "
    "Do not discover new findings. Unresolved inputs remain unadjudicated.",
)
_SUPPRESSION = _Adjudication(
    "suppression", "suppression", "SUPPRESS", "Suppression", "Suppression-reviewing {count} borderline finding(s)",
    SUPPRESSION_SCHEMA, "findings", 12,
    "Finalize suppression decisions for supplied findings",
    "Echo sup_id for resolved findings only. "
    "Do not invent new findings or drop an unresolved finding as if disproved.",
)


async def _adjudicate(
    backend: Backend,
    work: WorkContext,
    mode: _Adjudication,
    records: list[dict[str, Any]],
    *,
    input_path: Path | None = None,
    intent_authoritative: bool = False,
    **inputs: Unpack[AdjudicationInputs],
) -> tuple[list[dict[str, Any]] | None, ContinuationToken | None]:
    """Persist targets, run a bounded adjudication, and account for incomplete coverage."""
    from daydream.deep.prompts import build_supervise_prompt

    ui.print_phase_hero(agent.console, mode.hero, ui.phase_subtitle(mode.hero))
    ui.print_dim(agent.console, f"Model: {backend.model}")
    ui.print_info(agent.console, mode.progress.format(count=len(records)))
    dd = deep_dir(
        work.repo, session=inputs.get("artifact_session"), allow_standalone=inputs.get("allow_standalone", False),
    )
    path = input_path if input_path is not None else dd / f"{mode.name}-input.json"
    path.write_text(json.dumps(records, indent=2))
    exploration_dir = inputs.get("exploration_dir")
    strategy = inputs.get("strategy")
    default_strategy = review_profile.build_default_profile().strategies[mode.strategy].content
    resolved_strategy = strategy if strategy is not None else default_strategy
    builder = get_registry().prompt(mode.name)
    # The built-in supervisor only adjudicates supplied canonical items. With
    # none, its result is known; context artifacts cannot introduce targets.
    # A custom policy or builder still runs because its contract may differ.
    if (
        mode is _SUPERVISOR and not records and builder is build_supervise_prompt
        and resolved_strategy == default_strategy
    ):
        return [], None
    prompt_args: dict[str, Any] = {
        "strategy": resolved_strategy,
        f"{mode.name}_input_path": path,
        "diff_path": inputs["diff_path"],
        "intent_path": inputs["intent_path"],
        "alternatives_path": inputs["alternatives_path"],
        "cwd": work.repo,
        "exploration_dir": exploration_dir,
    }
    target_label = (
        "input findings (adjudication targets, not source evidence)"
        if mode is _SUPERVISOR else "adjudication targets (not source evidence)"
    )
    supplied_context = [(target_label, json.dumps(records))]
    if mode is _ARBITER:
        prompt_args["intent_authoritative"] = intent_authoritative
        supplied_context.append((
            "intent authority", AUTHORITATIVE_INTENT_BLOCK if intent_authoritative else "Intent is advisory context.",
        ))
    prompt = append_extended_facts(builder(**prompt_args), load_test_recipe(dd))
    input_label = f"{mode.name}-input"
    sanctioned_inputs = _prepare_existing_phase_inputs(
        backend, work,
        {input_label: path, "diff": inputs["diff_path"], "intent": inputs["intent_path"],
         "alternatives": inputs["alternatives_path"]},
        capture_without_session=True,
        exploration_dir=exploration_dir,
    )
    result, continuation, budget_reason = await agent.run_agent(
        backend, work.repo, prompt,
        output_schema=mode.schema,
        require_full_schema=True,
        phase=DaydreamPhase.DEEP,
        review_limits=ReviewLimits(120, 60, mode.tool_limit, discovery=False),
        finalization_context=FinalizationContext(
            task=mode.task,
            input_priority=(input_label, "diff", "intent"),
            assigned_files=tuple(sorted({str(item["file"]) for item in records if "file" in item})),
            output_semantics=mode.semantics,
            supplied_context=tuple(supplied_context),
        ),
        tool_call_budget=phase_config.DEFAULT_TOOL_CALL_BUDGET,
        wall_budget_s=phase_config.DEFAULT_WALL_BUDGET_S,
        sanctioned_inputs=sanctioned_inputs,
        run_context=resolve_run_context(inputs.get("run_context")),
    )
    if budget_reason:
        ui.print_warning(agent.console, f"{mode.label} budget exhausted; continuing with incomplete adjudication.")
        return _IncompleteAdjudication(budget_reason), None
    if not isinstance(result, dict) or not validates_schema(result, mode.schema):
        from daydream.phases.review import ReviewOutputError
        raise ReviewOutputError(result)
    return result[mode.result_key], continuation


def _index_records(records: list[dict[str, Any]], id_key: str) -> list[dict[str, Any]]:
    """Index records by stable host identity for arbiter and merge use."""
    fields = ("file", "line", "severity", "confidence", "description", "rationale", "evidence")
    return [{id_key: n, "uid": record_uid(record), **{key: record.get(key) for key in fields}}
            for n, record in enumerate(records, 1)]


def _rekey_verdicts(findings: list[dict[str, Any]], id_key: str, label: str) -> dict[int, dict[str, Any]]:
    """Re-key verdicts by their integer host id and report the kept/dropped tally."""
    if isinstance(findings, _IncompleteAdjudication):
        return IncompleteVerdicts(findings.budget_reason)
    verdicts = {finding[id_key]: finding for finding in findings if isinstance(finding.get(id_key), int)}
    kept = sum(1 for verdict in verdicts.values() if verdict.get("keep"))
    ui.print_info(agent.console, f"{label}: kept {kept}, dropped {len(verdicts) - kept}")
    return verdicts


@bind_resolved_run_context
async def phase_supervise_review(
    backend: Backend, work: WorkContext, *, items: list[dict[str, Any]], **inputs: Unpack[AdjudicationInputs],
) -> dict[int, dict[str, Any]]:
    """Adjudicate canonical items, retaining only verdicts for their integer ids.

    Input is deliberately verbatim, including host fields, for an auditable
    correspondence between supplied canonical items and echoed ids.
    """
    findings, _ = await _adjudicate(backend, work, _SUPERVISOR, items, **inputs)
    if isinstance(findings, _IncompleteAdjudication):
        return IncompleteVerdicts(findings.budget_reason)
    item_ids = {item.get("id") for item in items if isinstance(item.get("id"), int)}
    verdicts: dict[int, dict[str, Any]] = {}
    for verdict in findings or []:
        item_id = verdict.get("id")
        if isinstance(item_id, int) and item_id in item_ids:
            cleaned = dict(verdict)
            for field in ("severity", "confidence", "description", "rationale", "evidence"):
                if cleaned.get(field) is None:
                    cleaned.pop(field, None)
            verdicts[item_id] = cleaned
    if set(verdicts) != item_ids or len(findings or []) != len(item_ids):
        return IncompleteVerdicts("evidence_incomplete", verdicts)
    return verdicts


@bind_resolved_run_context
async def phase_arbiter_review(
    backend: Backend, work: WorkContext, *, selected_records: list[dict[str, Any]],
    intent_authoritative: bool = False, input_path: Path | None = None, **inputs: Unpack[AdjudicationInputs],
) -> tuple[dict[int, dict[str, Any]], ContinuationToken | None]:
    """Arbitrate a scoped selection; a shard can supply its own input artifact path."""
    findings, continuation = await _adjudicate(
        backend, work, _ARBITER, _index_records(selected_records, "arb_id"),
        input_path=input_path, intent_authoritative=intent_authoritative, **inputs,
    )
    return ({} if findings is None else _rekey_verdicts(findings, "arb_id", "Arbiter")), continuation


@bind_resolved_run_context
async def phase_suppression_review(
    backend: Backend, work: WorkContext, *, selected_records: list[dict[str, Any]],
    **inputs: Unpack[AdjudicationInputs],
) -> dict[int, dict[str, Any]]:
    """Suppress unsupported borderline findings after scoped skeptical review."""
    findings, _ = await _adjudicate(backend, work, _SUPPRESSION, _index_records(selected_records, "sup_id"), **inputs)
    return {} if findings is None else _rekey_verdicts(findings, "sup_id", "Suppression")
