"""Deep adjudication, merge, supervision, and review publication stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from daydream.agent import console
from daydream.artifact_visibility import review_output_path_for
from daydream.deep.artifacts import (
    DeepArtifact,
    persist_review_coverage,
    review_stage,
)
from daydream.deep.dedup import (
    build_dedup_candidates,
    build_record_dedup_candidates,
)
from daydream.deep.records import (
    record_uid,
    stack_name_from_uid,
)
from daydream.deep.render import _PIPELINE_STAGE_NAMES, render_held_section, render_report
from daydream.deep.reuse_key import (
    merge_key_payload,
    phase_identity_for,
)
from daydream.deep.reuse_store import (
    reuse_cache_for,
)
from daydream.deep.review_reuse import ReviewReuseUnit, _loop_grounding, _records_bytes_by_basename
from daydream.deep.settings import _resolve_opt_in
from daydream.deep.state import DeepData
from daydream.extensions.api import Stop
from daydream.flows.engine import FlowContext
from daydream.json_utils import dataclass_payload
from daydream.phases import (
    CrossStackMergeError,
    phase_cross_stack_merge,
    phase_supervise_review,
)
from daydream.phases.findings import (
    _write_single_stack_merged_items,
)
from daydream.phases.merge import merge_is_host_noop
from daydream.review_budget import (
    render_review_warnings,
    review_warnings,
)
from daydream.review_result import ReasonCode, reason_for_budget
from daydream.supervision import RuleBasedSupervisor, apply_findings_verdicts
from daydream.trajectory import (
    DaydreamPhase,
    LifecycleReasonCode,
    LifecycleStatus,
    get_current_recorder,
    phase_scope,
)
from daydream.ui import print_error, print_info, print_stage_progress, print_warning

if TYPE_CHECKING:
    from daydream.run_config import RunConfig


def _supervisor_mode(config: RunConfig) -> str:
    """Resolve the file-config-only findings supervisor mode."""
    file_config = config.file_config
    mode = file_config.supervisor if file_config is not None else None
    return mode if mode in {"off", "rules", "llm"} else "off"


def _merge_contributing_records(deep_data: DeepData) -> dict[str, bytes | None]:
    """Every records file the merge reads, keyed by basename.

    The primary-scope stacks (including the structural reviewer's records) plus the
    structural meta-stack. An unreadable file becomes a named miss (see
    :func:`_records_bytes_by_basename`).
    """
    paths = list(deep_data["record_pool"].language_paths)
    structural = deep_data["record_pool"].structural_path
    if structural is not None:
        paths.append(structural)
    return _records_bytes_by_basename(paths)


def _merge_store_payload(dd: Path) -> dict[str, bytes] | None:
    """The merge's owned artifacts as a payload map, or ``None`` when incomplete.

    ``merged-items.json`` and ``dedup-candidates.json`` are mandatory outputs of
    a completed cross-stack merge; the rendered ``review-output.md`` is the
    render-only report and its absence degrades only the copy, never the store
    (the JSON artifacts are the claim).
    """
    mandatory = (DeepArtifact.MERGED_ITEMS.at(dd).name, DeepArtifact.DEDUP_CANDIDATES.at(dd).name)
    payload: dict[str, bytes] = {}
    for name in mandatory:
        path = dd / name
        if not path.is_file():
            return None
        payload[name] = path.read_bytes()
    report = DeepArtifact.MERGED_REPORT.at(dd)
    if report.is_file():
        payload[report.name] = report.read_bytes()
    return payload


def _drop_cross_stack_duplicates(dd: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate host-written salvage by dropping each pair's record_b_uid.

    Keep the deterministic a-side. Reviewer ids and locations are not globally
    unique; only UID membership identifies the intended b-side across stacks.
    """
    dedup_p = DeepArtifact.DEDUP_CANDIDATES.at(dd)
    if not dedup_p.is_file():
        return records
    dedup = json.loads(dedup_p.read_text())
    pairs = dedup.get("record_duplicate_pairs") if isinstance(dedup, dict) else None
    if not isinstance(pairs, list):
        raise ValueError("Dedup artifact requires a record_duplicate_pairs list")
    dropped_uids: set[str] = set()
    for pair in pairs:
        if not isinstance(pair, dict) or not isinstance(pair.get("record_b_uid"), str) or not pair["record_b_uid"]:
            raise ValueError("Dedup artifact requires a nonempty record_b_uid for every pair")
        dropped_uids.add(pair["record_b_uid"])
    if not dropped_uids:
        return records
    # ``record_uid`` is ``""`` for an item with no pre-merge identity and ``""``
    # is never in ``dropped_uids``, so such an item is always kept.
    kept = [r for r in records if record_uid(r) not in dropped_uids]
    if len(kept) != len(records):
        print_info(
            console,
            f"Cross-stack merge salvage: dropped {len(records) - len(kept)} "
            "duplicate per-stack record(s) via the D-27 dedup pre-filter",
        )
    return kept


async def _step_cross_stack_merge(ctx: FlowContext) -> Stop | None:
    """Build dedup candidates and merge stack records, salvaging unparseable responses.

    A malformed response persists partial items/report/failure and stops resumably.
    Budget exhaustion persists the same salvage but publishes it successfully
    with incomplete-coverage diagnostics.
    """
    deep_data = ctx.deep_data()
    with review_stage(deep_data, "merge", persist=True, reasons=(ReasonCode.SYNTHESIS_FAILURE,)):
        dd = deep_data["dd"]
        alts_p: Path = deep_data["alts_path"]
        all_records: list[dict[str, Any]] = deep_data["record_pool"].language
        failed_stacks: dict[str, str] = deep_data["review_coverage"].unfinished_scopes
        coverage = deep_data["review_coverage"]
        host_noop = merge_is_host_noop(
            deep_data["record_pool"], deep_data["alts_path"],
            builder=ctx.registry.prompt("merge"), strategy=ctx.strategy("merge"),
        )

        async with phase_scope(
            DaydreamPhase.MERGE, stage="cross-stack-agent"
        ) as phase:
            # Dedup pre-filter (D-27).
            alt_issues_for_dedup: list[dict[str, Any]] = (
                json.loads(alts_p.read_text()) if alts_p.exists() else []
            )
            pairs = build_dedup_candidates(all_records, alt_issues_for_dedup)
            record_pairs = build_record_dedup_candidates(
                all_records, sources=[deep_data["record_pool"].paths[stack_name_from_uid(record_uid(r))].name
                                      for r in all_records]
            )
            dedup_p = DeepArtifact.DEDUP_CANDIDATES.at(dd)
            dedup_p.write_text(
                json.dumps(
                    {
                        "record_alt_pairs": [dataclass_payload(p) for p in pairs],
                        "record_duplicate_pairs": [
                            dataclass_payload(p) for p in record_pairs
                        ],
                    },
                    indent=2,
                )
            )

            # Issue #733 — the cross-stack merge is one content-addressed unit over
            # every contributing records file (per-stack and structural),
            # the failed-stack set, the structural presence flag, and its
            # schema/profile/model/effort contract. Intent, alternatives and the
            # pre-scan are recorded grounding only, so a moved pre-scan can never
            # move the key (MH2/MH16). The key is computed before dispatch, then a
            # hit restores the JSON artifacts and rendered report and skips the
            # model call; a miss runs and stores under that same key.
            # A resume (``--start-at ttt|per-stack|merge``) is an explicit request
            # to re-run the merge, so reuse never short-circuits it (A13: reuse does
            # not change ``--start-at`` semantics). Only the default ``"review"``
            # fresh run looks up or writes the store; every resume takes the
            # unchanged path.
            reuse = (
                reuse_cache_for(ctx) if ctx.config.start_at == "review" else None
            )
            merge_unit: ReviewReuseUnit | None = None
            if reuse is not None:
                merge_identity = phase_identity_for(ctx, "merge")
                merge_payload = merge_key_payload(
                    contributing_records=_merge_contributing_records(deep_data),
                    structural_records_present=deep_data["record_pool"].structural_path is not None,
                    failed_stacks=sorted(failed_stacks),
                    identity=merge_identity,
                    grounding=_loop_grounding(deep_data),
                )
                merge_unit = ReviewReuseUnit(reuse, "merge", merge_identity, merge_payload, coverage=coverage)
                if merge_unit.restore(deep_data["dd"]):
                    coverage.record_phase("merge", "complete", noop=host_noop)
                    return None

            # Cross-stack merge (D-23..D-26).
            try:
                await phase_cross_stack_merge(
                    ctx.backend_for("merge"),
                    ctx.work,
                    record_pool=deep_data["record_pool"],
                    intent_path=deep_data["intent_path"],
                    alternatives_path=alts_p,
                    dedup_candidates_path=dedup_p,
                    exploration_dir=deep_data["exploration_dir"],
                    failed_stacks=failed_stacks or None,
                    intent_authoritative=(deep_data.get("intent_authoritative") or False),
                    continuation=deep_data.get("arbiter_continuation"),
                    strategy=ctx.strategy("merge"),
                    run_context=ctx.run_context,
                    artifact_session=ctx.artifacts,
                    allow_standalone=ctx.allow_standalone_artifacts,
                )
            except CrossStackMergeError as exc:
                coverage.record_phase(
                    "merge", "incomplete" if exc.budget_reason else "failed",
                    reasons=(ReasonCode.SYNTHESIS_FAILURE, reason_for_budget(exc.budget_reason))
                    if exc.budget_reason else (ReasonCode.SYNTHESIS_FAILURE,), diagnostic=str(exc),
                )
                phase.finish(
                    LifecycleStatus.PARTIAL if exc.budget_reason else LifecycleStatus.FAILED,
                    LifecycleReasonCode.DOMAIN_FAILURE,
                )
                _salvage_merge_failure(ctx, exc)
                return None if exc.budget_reason else Stop(1)
            coverage.record_phase("merge", "complete", noop=host_noop)
            # Issue #733 — store only a completed merge, once the same artifacts a
            # fresh run leaves are final on disk. A failed or budget-exhausted
            # merge returns above and never reaches here.
            if merge_unit is not None:
                merge_unit.store(lambda: _merge_store_payload(dd))
        return None


def _salvage_merge_failure(ctx: FlowContext, exc: CrossStackMergeError) -> None:
    """Consolidate surviving language and structural findings after synthesis failure."""
    deep_data = ctx.deep_data()
    dd = deep_data["dd"]
    message = f"{exc}; consolidating surviving per-stack records into a partial report."
    if exc.budget_reason:
        print_warning(console, message + " Continuing to review publication.")
    else:
        print_error(console, "Cross-stack merge failed", message + " Relaunch with --start-at fix to resume.")

    # Build the partial canonical report from the surviving records via the
    # single-stack write helper. Coverage retains the synthesis failure. Apply the D-27
    # dedup pre-filter; these host-written items carry no merge-agent provenance.
    records = _drop_cross_stack_duplicates(dd, deep_data["record_pool"].language)
    _write_single_stack_merged_items(
        ctx.work.repo,
        dd,
        deep_data["record_pool"],
        records=records,
        failed_stacks=deep_data["review_coverage"].unfinished_scopes or None,
        artifact_session=ctx.artifacts,
        allow_standalone=ctx.allow_standalone_artifacts,
    )

    print_info(console, f"Wrote partial merged items to {dd}")


async def _step_single_stack_merge(ctx: FlowContext) -> None:
    """Tiny-diff single-stack bypass (#172): host-side merged-items write."""
    deep_data = ctx.deep_data()
    failed_stacks: dict[str, str] = deep_data["review_coverage"].unfinished_scopes

    # Issue #172 — tiny-diff single-stack bypass. A ≤2-file diff
    # has nothing to cross-stack-merge and nothing contested to
    # arbitrate, so the host writes ``merged-items.json`` directly
    # via ``normalize_items`` + the exact structural-tagging logic
    # from ``phase_cross_stack_merge``. No arbiter, no dedup, no
    # merge agent. Downstream consumers (fix gate, verifier, PR
    # posting) read the canonical JSON unchanged (AC6).
    async with phase_scope(DaydreamPhase.MERGE, stage="single-stack-host"):
        _write_single_stack_merged_items(
            ctx.work.repo,
            deep_data["dd"],
            deep_data["record_pool"],
            failed_stacks=failed_stacks or None,
            artifact_session=ctx.artifacts,
            allow_standalone=ctx.allow_standalone_artifacts,
        )
    deep_data["review_coverage"].record_phase("merge", "complete", noop=True)
    persist_review_coverage(deep_data["dd"], deep_data["review_coverage"])


async def _step_load_items(ctx: FlowContext) -> Stop | None:
    """Host-side merged-items guard + render-only markdown recovery."""
    deep_data = ctx.deep_data()
    target_dir = ctx.work.repo
    dd = deep_data["dd"]

    print_stage_progress(console, 5, 5, _PIPELINE_STAGE_NAMES[4])
    merged_report = review_output_path_for(
        target_dir,
        session=ctx.artifacts,
        allow_standalone=ctx.allow_standalone_artifacts,
    )

    # merged-items.json is the canonical source of truth; review-output.md is
    # render-only. The missing-input guard keys on the JSON so a --start-at fix
    # resume with surviving JSON but absent markdown proceeds rather than bailing.
    items_file = DeepArtifact.MERGED_ITEMS.at(dd)
    if not items_file.is_file():
        print_error(
            console,
            "Missing Merged Items",
            f"Expected canonical merged items at {items_file}",
        )
        return Stop(1)

    # Best-effort recover the render-only markdown from the deep-dir copy for
    # the exit message when the canonical file is absent (e.g. a --start-at fix
    # resume where the copy to the canonical path never ran). Non-fatal.
    if not merged_report.exists():
        deep_copy = DeepArtifact.MERGED_REPORT.at(dd)
        if deep_copy.exists():
            merged_report.write_text(deep_copy.read_text())



    warning = render_review_warnings(review_warnings(dd))
    if warning:
        for report in (merged_report, DeepArtifact.MERGED_REPORT.at(dd)):
            if report.exists() and warning not in report.read_text():
                report.write_text(warning + "\n\n" + report.read_text())

    deep_data["merged_report"] = merged_report
    deep_data["items_file"] = items_file
    return None


async def _step_findings_out(ctx: FlowContext) -> Stop:
    """Stop at the review boundary; recorder-scoped finalization owns the export."""
    ctx.data["findings_projection_ready"] = True
    return Stop(0)


async def _step_supervise(ctx: FlowContext) -> None:
    """Record the required supervision stage at its artifact completion boundary."""
    deep_data = ctx.deep_data()
    coverage = deep_data["review_coverage"]
    from daydream.deep.prompts import build_supervise_prompt
    from daydream.review_profile import build_default_profile

    with review_stage(deep_data, "supervision", persist=True):
        strategy = ctx.strategy("supervision")
        default_strategy = build_default_profile().strategies["supervision"].content
        input_items = json.loads(deep_data["items_file"].read_text())["items"]
        noop = not input_items and (_supervisor_mode(ctx.config) == "rules" or (
            (strategy is None or strategy == default_strategy)
            and ctx.registry.prompt("supervise") is build_supervise_prompt))
        budget_reason = await _supervise_items(ctx)
        if budget_reason:
            coverage.record_phase("supervision", "incomplete", reasons=(reason_for_budget(budget_reason),),
                                  diagnostic=budget_reason)
        else:
            coverage.record_phase("supervision", "complete", noop=noop)


async def _supervise_items(ctx: FlowContext) -> str | None:
    """Apply the configured findings supervisor to canonical merged items."""
    deep_data = ctx.deep_data()
    mode = _supervisor_mode(ctx.config)
    file_config = ctx.config.file_config
    items_file: Path = deep_data["items_file"]
    items = json.loads(items_file.read_text())["items"]
    if mode == "rules":
        assert file_config is not None, "rules mode requires file_config (guaranteed by _supervisor_mode)"
        deny_globs = file_config.supervisor_deny_globs
        verdicts = RuleBasedSupervisor(deny_globs=deny_globs).review_findings(items)
    else:
        async with phase_scope(DaydreamPhase.DEEP, stage="supervise"):
            verdicts = await phase_supervise_review(
                ctx.backend_for("supervise"),
                ctx.work,
                items=items,
                diff_path=deep_data["diff_path"],
                intent_path=deep_data["intent_path"],
                alternatives_path=deep_data["alts_path"],
                exploration_dir=deep_data["exploration_dir"],
                strategy=ctx.strategy("supervision"),
                run_context=ctx.run_context,
                artifact_session=ctx.artifacts,
                allow_standalone=ctx.allow_standalone_artifacts,
            )
    kept, held, events = apply_findings_verdicts(items, verdicts)
    items_file.write_text(json.dumps({"items": kept, "held": held}, indent=2))

    report = render_report(kept)
    warning = render_review_warnings(review_warnings(deep_data["dd"]))
    if warning:
        report = warning + "\n\n" + report
    held_section = render_held_section(held)
    if held_section:
        report = report.rstrip() + "\n\n" + held_section + "\n"
    deep_report = DeepArtifact.MERGED_REPORT.at(deep_data["dd"])
    deep_report.write_text(report)
    deep_data["merged_report"].write_text(report)

    recorder = get_current_recorder()
    if recorder is not None:
        for finding_id, action, reason in events:
            recorder.emit_supervisor_verdict(finding_id, action, reason)
    from daydream.phases.adjudication import IncompleteVerdicts

    return verdicts.budget_reason if isinstance(verdicts, IncompleteVerdicts) else None


async def _step_post_review(ctx: FlowContext) -> Stop | None:
    """Offer to post in loop/shallow modes; ``--comment`` auto-posts.

    In comment mode posting is the run's deliverable, so a missing PR or a
    failed GitHub submission ends the run with exit code 1 instead of the
    warn-and-continue the default deep flow gets (#8). Report-only review mode
    never resolves a PR or enters the posting helper.
    """
    deep_data = ctx.deep_data()
    if ctx.config.findings_out is None:
        from daydream.deep.review_terminal import finalize_review

        if not deep_data["review_coverage"].is_finalized:
            finalize_review(ctx, "completed")
    if str(deep_data.get("mode", "loop")) == "review":
        return None

    from daydream.pr_review import PostStatus, post_review_to_pr_from_report, resolve_review_renderers
    from daydream.pr_run_info import LiveRunInfoSource, render_live_run_info

    recorder = get_current_recorder()
    run_info = render_live_run_info(LiveRunInfoSource(recorder, ctx.artifacts))
    if run_info.diagnostic is not None:
        print_warning(console, run_info.diagnostic)

    items_file: Path = deep_data["items_file"]
    pr_kwargs: dict[str, Any] = (
        {"pr_number": ctx.config.pr_number}
        if ctx.config.pr_number is not None
        else {}
    )
    warnings = review_warnings(deep_data["dd"])
    if warnings:
        pr_kwargs["review_warnings"] = warnings
    outcome = await post_review_to_pr_from_report(
        ctx.work.repo,
        items_file,
        run_info=run_info.markdown,
        renderers=resolve_review_renderers(ctx.registry),
        console=console,
        post=str(deep_data.get("mode", "loop")) == "comment",
        approve_on_clean=_resolve_opt_in(ctx.config, "approve_on_clean"),
        diagram_blocks=(deep_data.get("diagrams") or {}).get("blocks"),
        run_context=ctx.run_context,
        auth=ctx.github_execution.auth,
        **pr_kwargs,
    )
    if str(deep_data.get("mode", "loop")) == "comment" and outcome in (PostStatus.NO_PR, PostStatus.FAILED):
        return Stop(1)
    return None
