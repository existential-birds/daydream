"""Finalize one review result inside the recorder, before artifact publication."""
from __future__ import annotations

import copy
import json
from functools import partial
from pathlib import Path
from typing import Any

import jsonschema

from daydream import git_ops
from daydream.agent import console
from daydream.artifact_visibility import ArtifactSourceUnavailableError
from daydream.deep.artifacts import DeepArtifact
from daydream.deep.state import DeepState
from daydream.flows.engine import FlowContext
from daydream.json_utils import atomic_write_bytes, atomic_write_json
from daydream.phases.schemas import MERGED_ITEMS_SCHEMA
from daydream.pr_review import PRInfo, parsed_issues_from_items, resolve_review_renderers
from daydream.pr_run_info import LiveRunInfoSource, render_live_run_info
from daydream.review_budget import render_review_warnings, review_warnings
from daydream.review_result import ReasonCode, deliver_finalized_review
from daydream.trajectory import get_current_recorder
from daydream.ui import print_warning


def refresh_review_report(ctx: FlowContext, *, no_diff: bool = False) -> dict[str, Any]:
    """Refresh presentation after authorized work without refreezing or republishing review authority."""
    from daydream.artifact_visibility import review_output_path_for
    from daydream.deep.render import insert_review_context, render_held_section, render_report
    from daydream.deep.report_context import REPORT_CONTEXT, collect_report_context

    state = DeepState(ctx.data)
    result = ctx.data['terminal_review_result']
    items = ctx.data['terminal_review_items']
    context = collect_report_context(state.dd, result,
                                     unavailable_scopes=ctx.data.get('unavailable_report_scopes'),
                                     author_unavailable=ctx.data.get('author_report_context_unavailable', False))
    atomic_write_json(state.dd / REPORT_CONTEXT, context)
    deep_report = DeepArtifact.MERGED_REPORT.at(state.dd)
    # The current deep report is the only source of report sections. The routed
    # canonical path can still contain a previous run's detached public report.
    if not no_diff and result['projection_valid'] and deep_report.is_file():
        existing = deep_report.read_text(encoding='utf-8')
    else:
        existing = render_report(items)
        held = ctx.data.get('terminal_review_held', []) if not no_diff else []
        if held:
            existing = existing.rstrip() + '\n\n' + render_held_section(held) + '\n'
    # Supervision rendered before recording its own completion, so its generated
    # prefix may be stale. Replace only that host prefix with frozen evidence.
    if existing.startswith('⚠️ **Review incomplete.** '):
        _, anchor, body = existing.partition('# Review\n')
        if anchor:
            existing = anchor + body
    warning = render_review_warnings(review_warnings(state.dd))
    if warning:
        existing = warning + '\n\n' + existing
    rendered = insert_review_context(existing, context, result, findings_count=len(items)).encode('utf-8')
    output = review_output_path_for(ctx.work.repo, session=ctx.artifacts,
                                   allow_standalone=ctx.allow_standalone_artifacts)
    for report in {deep_report, output}:
        atomic_write_bytes(report, rendered)
    return context


def finalize_review(ctx: FlowContext, pipeline_state: str, *, no_diff: bool = False) -> int:
    """Freeze validated coverage and findings once, without dispatching a model."""
    from daydream.runner import _write_findings_for_parsed

    state = DeepState(ctx.data)
    coverage = state.review_coverage
    if ctx.work.captured_source is not None:
        try:
            ctx.work.captured_source.validate()
        except (git_ops.GitError, OSError):
            coverage.require_phase('snapshot')
            coverage.record_phase('snapshot', 'failed', reasons=[ReasonCode.EVIDENCE_INCOMPLETE])
            print_warning(console, 'Captured source integrity failed; rerun the review.')
    items: list[dict[str, Any]] = []
    held: list[dict[str, Any]] = []
    projection_valid = no_diff
    recovery_failed = False
    path = ctx.data.get("items_file", DeepArtifact.MERGED_ITEMS.at(state.dd))
    if not no_diff and isinstance(path, Path) and not path.is_file() and "record_pool" in ctx.data:
        from daydream.phases.findings import _write_single_stack_merged_items
        try:
            _write_single_stack_merged_items(
                ctx.work.repo, state.dd, state.record_pool.language, state.record_pool.structural_path,
                failed_stacks=state.unfinished_scopes or None, artifact_session=ctx.artifacts,
                allow_standalone=ctx.allow_standalone_artifacts,
            )
        except ValueError as exc:
            recovery_failed = True
            print_warning(console, f"Cannot recover terminal findings: {type(exc).__name__}")
    if not no_diff and isinstance(path, Path) and path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            # Host canonical items add UIDs/placement provenance and optional held items.
            # Validate authored fields before projecting; never normalize malformed records away.
            schema = copy.deepcopy(MERGED_ITEMS_SCHEMA)
            schema["additionalProperties"] = True
            item_schema = schema["properties"]["items"]["items"]
            item_schema["additionalProperties"] = True
            item_schema["required"].remove("related_files")
            item_schema["properties"]["confidence"] = {"enum": ["HIGH", "MEDIUM", "LOW"]}
            jsonschema.validate(payload, schema)
            items = payload["items"]
            raw_held = payload.get('held')
            if isinstance(raw_held, list) and all(isinstance(item, dict)
                    and {'id', 'file', 'line', 'description'} <= item.keys() for item in raw_held):
                held = raw_held
            projection_valid = True
            if "findings" in coverage.required_phases:
                coverage.record_phase("findings", "complete")
        except (OSError, ValueError, jsonschema.ValidationError) as exc:
            coverage.require_phase("findings")
            coverage.record_phase("findings", "failed", reasons=[ReasonCode.MALFORMED_ARTIFACT])
            print_warning(console, f"Cannot validate terminal findings: {type(exc).__name__}")
    elif not no_diff:
        coverage.require_phase("findings")
        coverage.record_phase("findings", "failed", reasons=[
            ReasonCode.MALFORMED_ARTIFACT if recovery_failed else ReasonCode.MISSING_ARTIFACT,
        ])
    if pipeline_state == "completed" and projection_valid and "pipeline" in coverage.required_phases:
        coverage.record_phase("pipeline", "complete")
    result = coverage.finalize(pipeline_state, projection_valid=projection_valid)
    atomic_write_json(DeepArtifact.REVIEW_COVERAGE.at(state.dd), coverage.to_dict())
    ctx.data["terminal_review_result"] = result
    ctx.data['terminal_review_items'] = items
    ctx.data['terminal_review_held'] = held
    # Embedded evaluation consumes the same validated result as publication,
    # before optional rendering, without inventing a live GitHub PR target.
    recorder = get_current_recorder()
    deliver_finalized_review(result, items, usage={} if recorder is None else recorder.review_usage())
    try:
        context = refresh_review_report(ctx, no_diff=no_diff)
    except (OSError, ValueError, TypeError, RecursionError, ArtifactSourceUnavailableError) as exc:
        diagnostic = f'Review report unavailable: {type(exc).__name__}'
        ctx.data['review_report_diagnostic'] = diagnostic
        print_warning(console, diagnostic)
        context = None
    ctx.data['terminal_review_context'] = context
    if ctx.config.findings_out is None:
        return 0
    pr = ctx.data.get("analyzed_pr")
    if not isinstance(pr, PRInfo):
        return 1
    info = render_live_run_info(LiveRunInfoSource(get_current_recorder(), ctx.artifacts))
    if info.diagnostic is not None:
        print_warning(console, info.diagnostic)
    write_findings = partial(
        _write_findings_for_parsed,
        ctx.work.repo, ctx.config, parsed_issues_from_items(items), run_info=info.markdown,
        review_warnings=review_warnings(state.dd), renderers=resolve_review_renderers(ctx.registry),
        diagrams=(state.diagrams or {}).get("payload"), auth=ctx.github_execution.auth,
        captured_pr=pr, terminal_result=result, snapshot_diff=ctx.data["snapshot_diff"],
        review_context=context,
    )
    if ctx.artifacts is None:
        return write_findings(frozen_only=False)
    try:
        return ctx.artifacts.code_workspace.run(partial(write_findings, frozen_only=False))
    except ArtifactSourceUnavailableError:
        return write_findings(frozen_only=True)
