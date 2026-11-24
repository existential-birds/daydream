"""Finalize one review result inside the recorder, before artifact publication."""
from __future__ import annotations

import copy
import json
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

import jsonschema

from daydream import git_ops
from daydream.agent import console
from daydream.artifact_visibility import ArtifactSourceUnavailableError
from daydream.deep.artifacts import DeepArtifact
from daydream.deep.state import DeepState
from daydream.extensions.api import ExtensionError
from daydream.flows.engine import FlowContext
from daydream.json_utils import atomic_write_bytes, atomic_write_json, validate_json_schema
from daydream.phases.schemas import MERGED_ITEMS_SCHEMA
from daydream.pr_review import PRInfo, parsed_issues_from_items, resolve_review_renderers
from daydream.pr_run_info import LiveRunInfoSource, render_live_run_info
from daydream.repository_paths import git_observed_path_is_confined
from daydream.review_budget import ReasonCode, StopRecord, render_review_warnings, review_warnings
from daydream.review_result import deliver_finalized_review
from daydream.trajectory import DaydreamPhase, get_current_recorder, phase_scope, run_usage
from daydream.ui import print_info, print_warning

# MERGED_ITEMS_SCHEMA relaxed for canonical records that extension steps may rewrite.
_CANONICAL_ITEMS_SCHEMA = copy.deepcopy(MERGED_ITEMS_SCHEMA)
_CANONICAL_ITEMS_SCHEMA["additionalProperties"] = True
_CANONICAL_ITEMS_SCHEMA["properties"]["items"]["items"].update(
    additionalProperties=True, required=["id", "description", "file", "line", "rationale"])
_CANONICAL_ITEMS_SCHEMA["properties"]["items"]["items"]["properties"].update(
    line={"type": ["integer", "null"]}, confidence={"enum": ["HIGH", "MEDIUM", "LOW"]})
# What load_canonical_items raises for unreadable, undecodable, too deeply nested, unconfined or
# schema-invalid items.
CANONICAL_ITEMS_ERRORS = (OSError, ValueError, RecursionError, jsonschema.ValidationError)


def load_canonical_items(repo: Path, path: Path) -> list[dict[str, Any]]:
    """Read merged items that extension steps may have rewritten; raise on any malformed record.

    Canonical items carry host identities and an integer display ``id``; confidence and lens are
    optional because the reviewer asserts none, and severity because an extension step may drop
    it, which withholds approval. Never normalize malformed records away.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_json_schema(payload, _CANONICAL_ITEMS_SCHEMA)
    if not all(git_observed_path_is_confined(repo, item["file"], allow_leaf_symlink=True) for item in payload["items"]):
        raise ValueError("a finding path is not a confined repository path")
    items: list[dict[str, Any]] = payload["items"]
    return items


def refresh_review_report(ctx: FlowContext, *, no_diff: bool = False) -> dict[str, Any]:
    """Refresh presentation after authorized work without refreezing or republishing review authority."""
    from daydream.artifact_visibility import review_output_path_for
    from daydream.deep.render import (
        insert_diagrams_section,
        insert_review_context,
        render_report,
    )
    from daydream.deep.report_context import REPORT_CONTEXT, collect_report_context

    state = DeepState(ctx.data)
    result = ctx.data['terminal_review_result']
    items = ctx.data['terminal_review_items']
    context = collect_report_context(state.dd, result,
                                     author_unavailable=ctx.data.get('author_report_context_unavailable', False))
    atomic_write_json(state.dd / REPORT_CONTEXT, context)
    deep_report = DeepArtifact.MERGED_REPORT.at(state.dd)
    # The current deep report is the only source of report sections. The routed
    # canonical path can still contain a previous run's detached public report.
    if not no_diff and result['projection_valid'] and deep_report.is_file():
        existing = deep_report.read_text(encoding='utf-8')
    else:
        existing = render_report(items)
    # Supervision rendered before recording its own completion, so its generated
    # prefix may be stale. Replace only that host prefix with frozen evidence.
    if existing.startswith('⚠️ **Review incomplete.** '):
        _, anchor, body = existing.partition('# Review\n')
        if anchor:
            existing = anchor + body
    # Diagrams are generated before the report exists, so the report takes them here.
    blocks = (state.diagrams or {}).get('blocks')
    if blocks:
        existing = insert_diagrams_section(existing, blocks)
    warning = render_review_warnings(review_warnings(result))
    if warning:
        existing = warning + '\n\n' + existing
    rendered = insert_review_context(existing, context, result, findings_count=len(items)).encode('utf-8')
    output = review_output_path_for(ctx.work.repo, session=ctx.artifacts,
                                   allow_standalone=ctx.allow_standalone_artifacts)
    for report in {deep_report, output}:
        atomic_write_bytes(report, rendered)
    return context


def _author_description(ctx: FlowContext) -> str | None:
    """The author's description of the analyzed head: embedder-frozen text, else the open PR's own body."""
    from daydream.committed_review import bounded_author_description, fit_author_context
    from daydream.deep.report_context import retain_author_context

    config, work = ctx.config, ctx.work
    frozen = bounded_author_description(config)
    description: str | None = None
    head = work.head_sha
    if frozen is not None:
        description = frozen.for_head(head)
    elif config.pr_number is not None:
        try:
            view = git_ops.gh_pr_view(work.source if work.captured_source is not None else work.repo,
                                      config.pr_number, auth=ctx.github_execution.auth)
        except git_ops.GitError as exc:
            print_warning(console, f"Could not load PR #{config.pr_number} description ({exc}); "
                                   "continuing without PR description context")
            view = None
        if view is not None:
            state, head = view.get("state", ""), view.get("headRefOid") or None
            if state and state.upper() != "OPEN" or head and work.head_sha and head != work.head_sha:
                print_warning(console, f"PR #{config.pr_number} is not open at the analyzed head; "
                                       "skipping its description")
            else:
                description = view.get("body") or None
                if description is not None and (fitted := fit_author_context(description)) != description:
                    print_warning(console, f"PR #{config.pr_number} description exceeds the reviewer's context "
                                           "bound; the reviewer receives a truncated copy")
                    description = fitted
    ctx.data['author_report_context_unavailable'] = True
    try:
        retain_author_context(DeepState(ctx.data).dd, description, head)
    except (OSError, ValueError, TypeError, RecursionError) as exc:
        print_warning(console, f'Author report context unavailable: {type(exc).__name__}')
    else:
        ctx.data['author_report_context_unavailable'] = False
    return description if description and description.strip() else None


async def plan_committed_review(
    ctx: FlowContext, *, run_id: str, revision: dict[str, Any], index_path: Path, files: list[str], started: float,
) -> None:
    """Plan the run's only reviewer on the flow's own workspace, session and backends, before any model work."""
    from daydream import committed_review
    from daydream.run_config import resolved_backend_name

    settings = committed_review.review_settings(ctx.config)
    description = _author_description(ctx)
    backends = {}
    # A bound review context stays on its adapter instance, so the reviewer never shares one with a later phase.
    reviewers = replace(ctx, _backend_cache={})
    for kind, phase in committed_review.REVIEW_PHASES:
        backends[kind] = reviewers.backend_for(phase)
        committed_review.bind_native_context(
            settings, kind, phase, resolved_backend_name(ctx.config, phase), backends[kind], ctx.work.source)
    await committed_review.bind_optional_fast_backends(
        ctx.config, settings, backends, ctx.work.source,
    )
    state = DeepState(ctx.data)
    program = await committed_review.plan_review(
        ctx.config, ctx.work, backends, settings, run_id=run_id, revision=revision, diff_path=state.diff_path,
        index_path=index_path, files=files, started=started,
        # Captured untrusted author context: never authoritative over a claim.
        historical_context=None if description is None else {"context": description},
        import_graph=state.import_graph,
    )
    ctx.data["committed_review"] = program
    ctx.data["review_coverage"] = program.review.coverage


async def _step_committed_review(ctx: FlowContext) -> None:
    """Dispatch the planned review; its confirmed findings become the canonical items, even when it ends early."""
    from daydream.findings import committed_finding_item

    program = ctx.data.get("committed_review")
    if program is None:
        raise ExtensionError("the review phase runs only in the deep flow, which plans the review before its "
                             "steps; insert steps into 'deep' with set_flow instead of borrowing 'review'")

    def publish_items() -> None:
        # Every consumer reads the line as a head line, so a location only the old side has carries none.
        atomic_write_json(DeepArtifact.MERGED_ITEMS.at(DeepState(ctx.data).dd), {"items": [
            {**committed_finding_item(finding), "line": program.review.head_line(finding)}
            for finding in program.review.findings()]})

    try:
        await program.dispatch(get_current_recorder())
    except BaseException:
        await program.quiesce()
        publish_items()
        raise
    await program.quiesce()
    # The host projection is this flow's merge: it alone produces the canonical items.
    async with phase_scope(DaydreamPhase.MERGE, stage="committed-findings"):
        publish_items()


def finalize_review(ctx: FlowContext, pipeline_state: str, *, stop: StopRecord | None = None) -> int:
    """Seal validated coverage and findings once, without dispatching a model.

    A failed pipeline is retained as run-wide unfinished evidence. Extension steps may
    have rewritten the canonical items; an unusable projection fails the analysis.
    """
    from daydream.committed_review_output import allowance_sentence, write_committed_review_output
    from daydream.committed_review_summary import build_review_summary
    from daydream.runner import _write_findings_for_parsed

    state = DeepState(ctx.data)
    coverage = state.review_coverage
    program = ctx.data["committed_review"]
    no_diff = not program.review.assignments
    if pipeline_state == "failed":
        program.fail(stop or StopRecord(ReasonCode.UNEXPECTED_ANALYSIS_FAILURE, "pipeline"))
    items: list[dict[str, Any]] = []
    projection_valid = no_diff
    path = DeepArtifact.MERGED_ITEMS.at(state.dd)
    if not no_diff:
        try:
            items = load_canonical_items(ctx.work.repo, path)
            projection_valid = True
        except CANONICAL_ITEMS_ERRORS as exc:
            print_warning(console, f"Cannot validate terminal findings: {type(exc).__name__}")
    sealed = program.seal(projection_valid=projection_valid)
    result = sealed["terminal_result"]
    if "source_integrity" in {entry["assignment"] for entry in sealed["execution"]}:
        print_warning(console, 'Captured source integrity failed; rerun the review.')
    atomic_write_json(DeepArtifact.REVIEW_COVERAGE.at(state.dd), coverage.to_dict())
    delivered, confirmed = len(items), len(sealed["findings"])
    outcome = (f"Review {result['analysis_state']}: {delivered or 'no'} confirmed "
               f"finding{'' if delivered == 1 else 's'}"
               + ("." if delivered == confirmed else f" delivered (the reviewer confirmed {confirmed})."))
    if result["analysis_state"] == "complete":
        print_info(console, outcome)
    else:
        unfinished = "; ".join(review_warnings(result)) or ", ".join(result["reason_codes"])
        allowance = allowance_sentence(result["reason_codes"], str)
        print_warning(console, f"{outcome} Not finished: {unfinished}." + (f" {allowance}" if allowance else ""))
    # A cancelled run hands nothing over, to its embedder or to a poster: the published files hold the sealed result.
    handed_over = pipeline_state != "cancelled"
    summary = build_review_summary(sealed, config=ctx.config, recorder=get_current_recorder(),
                                   delivered=delivered if handed_over else 0)
    # Under ``deep/``, a registered public directory: a top-level file would be an unknown public
    # name that refuses the next run, on this build and on an earlier one, once its private proof is gone.
    write_committed_review_output(ctx.artifacts, sealed, report_path=DeepArtifact.MERGED_REPORT.at(state.dd),
                                  result_path=state.dd / "committed-review.json",
                                  summary=summary, summary_path=state.dd / "review-summary.json")
    ctx.data["terminal_review_result"] = result
    ctx.data['terminal_review_items'] = items
    # Embedded evaluation consumes the same validated result as publication,
    # before optional rendering, without inventing a live GitHub PR target.
    if handed_over:
        deliver_finalized_review(result, items, usage=run_usage(), summary=summary)
    try:
        context = refresh_review_report(ctx, no_diff=no_diff)
    except (OSError, ValueError, TypeError, RecursionError, ArtifactSourceUnavailableError) as exc:
        diagnostic = f'Review report unavailable: {type(exc).__name__}'
        ctx.data['review_report_diagnostic'] = diagnostic
        print_warning(console, diagnostic)
        context = None
    ctx.data['terminal_review_context'] = context
    failed = int(result["analysis_state"] == "failed")
    if ctx.config.findings_out is None or not handed_over:
        return failed
    pr = ctx.data.get("analyzed_pr")
    if not isinstance(pr, PRInfo):
        return 1
    info = render_live_run_info(LiveRunInfoSource(get_current_recorder(), ctx.artifacts))
    if info.diagnostic is not None:
        print_warning(console, info.diagnostic)
    write_findings = partial(
        _write_findings_for_parsed,
        ctx.work.repo, ctx.config, parsed_issues_from_items(items), run_info=info.markdown,
        review_warnings=review_warnings(result), renderers=resolve_review_renderers(ctx.registry),
        diagrams=(state.diagrams or {}).get("payload"), auth=ctx.github_execution.auth,
        captured_pr=pr, terminal_result=result, snapshot_diff=ctx.data["snapshot_diff"],
        review_context=context,
    )
    if ctx.artifacts is None:
        return write_findings(frozen_only=False) or failed
    try:
        return ctx.artifacts.code_workspace.run(partial(write_findings, frozen_only=False)) or failed
    except ArtifactSourceUnavailableError:
        return write_findings(frozen_only=True) or failed
