"""Merge for review and fix phases."""

import json
from pathlib import Path
from typing import Any

from daydream import agent, config as phase_config, review_profile as _rp, ui
from daydream.artifact_visibility import (
    ArtifactSession,
    review_output_path_for,
)
from daydream.backends import (
    Backend,
    ContinuationToken,
)
from daydream.deep.artifacts import (
    DeepArtifact,
    deep_dir,
)
from daydream.deep.records import (
    record_issues,
    stack_name_from_records_source,
)
from daydream.extensions import get_registry
from daydream.phases.findings import (
    CrossStackMergeError,
    _append_structural_and_write_merged,
    _reset_merged_outputs,
    _validate_agent_source_uids,
)
from daydream.phases.inputs import _prepare_existing_phase_inputs, append_extended_facts
from daydream.phases.schemas import MERGED_ITEMS_SCHEMA
from daydream.prompts.authorial_intent import (
    AUTHORITATIVE_INTENT_BLOCK,
)
from daydream.review_budget import (
    ReviewLimits,
)
from daydream.review_evidence import FinalizationContext
from daydream.run_context import RunContext, bind_resolved_run_context, resolve_run_context
from daydream.test_execution import (
    load_test_recipe,
)
from daydream.trajectory import (
    DaydreamPhase,
)
from daydream.workspace import WorkContext


def _empty_merge_inputs(per_stack_records_paths: list[Path], alternatives_path: Path) -> bool:
    """Prove there are no synthesis targets from readable, completed artifacts.

    Missing or malformed records cannot establish clean coverage. Alternatives
    are independent merge inputs, so an empty reviewer pool alone is insufficient.
    """
    if not per_stack_records_paths:
        return False
    try:
        for path in per_stack_records_paths:
            if record_issues(json.loads(path.read_text())) != []:
                return False
        # Unlike records, the persisted alternatives contract is a bare list;
        # an issues envelope does not establish completed alternative coverage.
        if json.loads(alternatives_path.read_text()) != []:
            return False
    except (OSError, ValueError):
        return False
    return True


@bind_resolved_run_context
async def phase_cross_stack_merge(
    backend: Backend,
    work: WorkContext,
    *,
    per_stack_records_paths: list[Path],
    intent_path: Path,
    alternatives_path: Path,
    dedup_candidates_path: Path,
    exploration_dir: Path | None = None,
    failed_stacks: dict[str, str] | None = None,
    structural_records_path: Path | None = None,
    intent_authoritative: bool = False,
    continuation: ContinuationToken | None = None,
    strategy: str | None = None,
    artifact_session: ArtifactSession | None = None,
    allow_standalone: bool = False,
    run_context: RunContext | None = None,
) -> Path:
    """Merge language findings, append host-tagged structural records, and publish both formats.

    per_stack_records_paths excludes the structural meta-stack; structural_records_path
    is appended separately, preserving severity and defaulting unlabeled records to
    high. The normalized list is written as merged-items.json, rendered in deep/,
    and copied to the repository report path, which is returned.

    failed_stacks names uncovered scopes. Authoritative intent requires fresh,
    head-matched PR evidence; an absent strategy uses the packaged merge strategy.
    Unparseable output raises CrossStackMergeError for salvage; invalid items raise
    ValueError. Agent fan-out stays host-owned through run_agent.
    """
    run_context = resolve_run_context(run_context)
    dd = deep_dir(work.repo, session=artifact_session, allow_standalone=allow_standalone)
    canonical_path = review_output_path_for(
        work.repo,
        session=artifact_session,
        allow_standalone=allow_standalone,
    )
    report_path = DeepArtifact.MERGED_REPORT.at(dd)
    items_path = DeepArtifact.MERGED_ITEMS.at(dd)

    # Clear stale outputs so a failed merge agent can't leave behind
    # outdated content that downstream stages would silently consume.
    _reset_merged_outputs(canonical_path, report_path, items_path)

    # Only the packaged contract promises to synthesize existing inputs. A
    # custom builder or distinct policy may perform additional work even when
    # those inputs are empty. Import lazily to avoid the phases/prompts cycle.
    from daydream.deep.prompts import build_merge_prompt

    builder = get_registry().prompt("merge")
    default_strategy = _rp.build_default_profile().strategies["merge"].content
    resolved_strategy = strategy if strategy is not None else default_strategy
    if (builder is build_merge_prompt and resolved_strategy == default_strategy
            and _empty_merge_inputs(per_stack_records_paths, alternatives_path)):
        if failed_stacks:
            ui.print_warning(
                agent.console,
                "Cross-stack merge completed with uncovered stacks: "
                + "; ".join(f"{name}: {reason}" for name, reason in sorted(failed_stacks.items())),
            )
        _append_structural_and_write_merged(
            [], structural_records_path, items_path, report_path, canonical_path,
        )
        return canonical_path

    prompt = builder(
        strategy=resolved_strategy,
        per_stack_records_paths=per_stack_records_paths,
        intent_path=intent_path,
        alternatives_path=alternatives_path,
        dedup_candidates_path=dedup_candidates_path,
        exploration_dir=exploration_dir,
        failed_stacks=failed_stacks,
        intent_authoritative=intent_authoritative,
        resumed_from_arbiter=continuation is not None,
    )
    prompt = append_extended_facts(prompt, load_test_recipe(dd))
    merge_inputs: dict[str, Path | None] = {
        "intent": intent_path,
        "alternatives": alternatives_path,
        "dedup-candidates": dedup_candidates_path,
        **{
            f"stack-records-{index:03d}": path
            for index, path in enumerate(sorted(per_stack_records_paths))
        },
    }
    sanctioned_inputs = _prepare_existing_phase_inputs(
        backend, work, merge_inputs, exploration_dir=exploration_dir, capture_without_session=True,
    )
    ui.print_phase_hero(agent.console, "MERGE", ui.phase_subtitle("MERGE"))
    ui.print_dim(agent.console, f"Model: {backend.model}")
    result, _, budget_reason = await agent.run_agent(
        backend,
        work.repo,
        prompt,
        output_schema=MERGED_ITEMS_SCHEMA,
        require_full_schema=True,
        phase=DaydreamPhase.MERGE,
        continuation=continuation,
        review_limits=ReviewLimits(180, 60, 16, discovery=False),
        finalization_context=FinalizationContext(
            task="Merge completed review records",
            input_priority=tuple(label for label in merge_inputs if label.startswith("stack-records-"))
            + ("intent", "dedup-candidates"),
            output_semantics="Return merged items in the required schema, preserving source finding identities and "
            "grounded defects. Empty items is valid only if the supplied completed records establish no findings. "
            "Do not infer clean coverage from absent, incomplete, or omitted records.",
            supplied_context=(("failed stacks", json.dumps(failed_stacks or {})),
                              ("intent authority", AUTHORITATIVE_INTENT_BLOCK
                               if intent_authoritative else "Intent is advisory context.")),
        ),
        tool_call_budget=phase_config.DEFAULT_TOOL_CALL_BUDGET,
        wall_budget_s=phase_config.REVIEW_WALL_BUDGET_S,
        sanctioned_inputs=sanctioned_inputs,
        run_context=run_context,
    )

    # Invalid current envelopes trigger host salvage of completed reviewer records.
    item_list = result.get("items") if isinstance(result, dict) else None
    if item_list is None or budget_reason is not None:
        # Use the UID owner's canonical source-name parser for error context.
        stack_context = [
            stack_name_from_records_source(p.name) for p in per_stack_records_paths
        ]
        raise CrossStackMergeError(
            type(result).__name__,
            stack_context,
            message=(f"Cross-stack merge budget exhausted: {budget_reason}" if budget_reason
                     else f"Cross-stack merge returned no item list (got {type(result).__name__})"),
            budget_reason=budget_reason,
        )
    agent_items: list[dict[str, Any]] = item_list

    # Validate agent provenance before structural folding unions source UIDs;
    # otherwise an invented UID could contaminate another item's attribution.
    # Validation removes bad claims without failing the merge.
    _validate_agent_source_uids(agent_items, per_stack_records_paths, structural_records_path)

    # Share append/render with the bypass to keep lenses and verifier counts aligned.
    _append_structural_and_write_merged(
        agent_items, structural_records_path, items_path, report_path, canonical_path
    )
    return canonical_path
