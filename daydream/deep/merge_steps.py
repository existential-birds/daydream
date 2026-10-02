"""Deep adjudication, merge, supervision, and review publication stages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from daydream.agent import console
from daydream.artifact_visibility import review_output_path_for
from daydream.deep.artifacts import (
    MERGE_FAILURE_KEY,
    _load_failures,
    dedup_candidates_path,
    merged_items_path,
    merged_report_path,
    per_stack_failures_path,
    persist_review_coverage,
)
from daydream.deep.dedup import (
    build_dedup_candidates,
    build_record_dedup_candidates,
)
from daydream.deep.records import (
    record_uid,
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
from daydream.deep.state import DeepState
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
from daydream.redaction import redact_text
from daydream.review_budget import (
    clear_review_budget_stop,
    record_review_budget_stop,
    render_review_warnings,
    review_warnings,
)
from daydream.review_result import ReasonCode, reason_for_budget, reason_for_exception
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


def _merge_contributing_records(deep_state: DeepState) -> dict[str, bytes | None]:
    """Every records file the merge reads, keyed by basename.

    The primary-scope stacks (including the structural reviewer's records) plus the
    structural meta-stack. An unreadable file becomes a named miss (see
    :func:`_records_bytes_by_basename`).
    """
    paths = list(deep_state.records_paths)
    structural = deep_state.structural_records_path_or_none
    if structural is not None:
        paths.append(structural)
    return _records_bytes_by_basename(paths)


def _merge_is_host_noop(ctx: FlowContext, deep_state: DeepState) -> bool:
    """Mirror the packaged phase's proof of an eligible empty synthesis."""
    from daydream.deep.prompts import build_merge_prompt
    from daydream.phases.merge import _empty_merge_inputs
    from daydream.review_profile import build_default_profile

    strategy = ctx.strategy("merge")
    default = build_default_profile().strategies["merge"].content
    return (ctx.registry.prompt("merge") is build_merge_prompt
            and (strategy is None or strategy == default)
            and _empty_merge_inputs(deep_state.records_paths, deep_state.alts_path))


def _merge_store_payload(dd: Path) -> dict[str, bytes] | None:
    """The merge's owned artifacts as a payload map, or ``None`` when incomplete.

    ``merged-items.json`` and ``dedup-candidates.json`` are mandatory outputs of
    a completed cross-stack merge; the rendered ``review-output.md`` is the
    render-only report and its absence degrades only the copy, never the store
    (the JSON artifacts are the claim).
    """
    mandatory = (merged_items_path(dd).name, dedup_candidates_path(dd).name)
    payload: dict[str, bytes] = {}
    for name in mandatory:
        path = dd / name
        if not path.is_file():
            return None
        payload[name] = path.read_bytes()
    report = merged_report_path(dd)
    if report.is_file():
        payload[report.name] = report.read_bytes()
    return payload


def _clear_merge_failure(dd: Path) -> None:
    """Clear a stale ``__merge__`` salvage record after a successful re-merge.

    ``_salvage_merge_failure`` is the only writer of ``MERGE_FAILURE_KEY``; a
    later successful cross-stack merge must supersede it so a subsequent resume
    doesn't emit a misleading 'merged results are PARTIAL' warning for a merge
    that actually succeeded.
    """
    failures_p = per_stack_failures_path(dd)
    loaded = _load_failures(failures_p)
    if MERGE_FAILURE_KEY not in loaded:
        return
    loaded.pop(MERGE_FAILURE_KEY, None)
    if loaded:
        failures_p.write_text(json.dumps(loaded, indent=2, sort_keys=True))
    elif failures_p.exists():
        failures_p.unlink()


#: Cap on how many unidentifiable dedup pairs ``_drop_cross_stack_duplicates``
#: names individually in its aggregate warning below, mirroring
#: ``phases._MAX_REPORTED_UNKNOWN_UIDS``: an artifact written before
#: ``record_b_uid`` existed can carry many such pairs at once, and naming the
#: first few and counting the rest keeps the message actionable instead of
#: turning it into the per-pair flood the aggregation exists to avoid.
_MAX_REPORTED_UNIDENTIFIABLE_PAIRS = 10


def _drop_cross_stack_duplicates(dd: Path, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate host-written salvage by dropping each pair's record_b_uid.

    Keep the deterministic a-side. Reviewer ids and locations are not globally
    unique; only UID membership identifies the intended b-side across stacks.
    """
    dedup_p = dedup_candidates_path(dd)
    if not dedup_p.is_file():
        return records
    try:
        dedup = json.loads(dedup_p.read_text())
    except json.JSONDecodeError:
        return records
    dropped_uids: set[str] = set()
    unidentifiable_pairs: list[str] = []
    for pair in dedup.get("record_duplicate_pairs", []) or []:
        if not isinstance(pair, dict):
            continue
        b_uid = pair.get("record_b_uid")
        if isinstance(b_uid, str) and b_uid:
            dropped_uids.add(b_uid)
            continue
        # Unreachable within one run; the guard is for a resume reading an
        # older deep dir predating the field. Skip rather than fall back to the
        # buggy ``(id, file)`` key -- a duplicate is a smaller error than a loss.
        unidentifiable_pairs.append(
            f"{pair.get('record_a_id')!r}/{pair.get('record_b_id')!r}"
        )
    if unidentifiable_pairs:
        # ONE warning for the whole salvage, not one per pair (mirrors
        # ``phases._validate_agent_source_uids``): an artifact written before
        # ``record_b_uid`` existed can carry many such pairs at once, and
        # per-pair reporting would bury the run's real output under identical
        # lines.
        shown = ", ".join(unidentifiable_pairs[:_MAX_REPORTED_UNIDENTIFIABLE_PAIRS])
        if len(unidentifiable_pairs) > _MAX_REPORTED_UNIDENTIFIABLE_PAIRS:
            shown += f", +{len(unidentifiable_pairs) - _MAX_REPORTED_UNIDENTIFIABLE_PAIRS} more"
        print_warning(
            console,
            "Cross-stack merge salvage: dedup pair(s) carries no record_b_uid, so "
            "neither side can be identified; keeping both records (issue #1111). "
            f"({len(unidentifiable_pairs)} pair(s): {shown})",
        )
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
    """Record failures across merge preparation and execution without losing their cause."""
    try:
        return await _cross_stack_merge(ctx)
    except Exception as exc:
        state = DeepState(ctx.data)
        if state.review_coverage is not None:
            state.review_coverage.record_phase(
                "merge", "failed", reasons=(ReasonCode.SYNTHESIS_FAILURE, reason_for_exception(exc)),
            )
            _persist_failure_coverage(state, exc)
        raise


def _persist_failure_coverage(state: DeepState, original_error: Exception) -> None:
    """Keep a failed evidence write secondary to the phase's original exception."""
    assert state.review_coverage is not None
    try:
        persist_review_coverage(state.dd, state.review_coverage)
    except Exception as persistence_error:
        original_error.add_note(f"Review coverage persistence failed: {type(persistence_error).__name__}")


async def _cross_stack_merge(ctx: FlowContext) -> Stop | None:
    """Build dedup candidates and merge stack records, salvaging unparseable responses.

    A malformed response persists partial items/report/failure and stops resumably.
    Budget exhaustion persists the same salvage but publishes it successfully
    with incomplete-coverage diagnostics.
    """
    deep_state = DeepState(ctx.data)
    dd = deep_state.dd
    alts_p: Path = deep_state.alts_path
    all_records: list[dict[str, Any]] = deep_state.records
    failed_stacks: dict[str, str] = deep_state.failed_stacks
    coverage = deep_state.review_coverage
    host_noop = _merge_is_host_noop(ctx, deep_state)

    async with phase_scope(
        DaydreamPhase.MERGE, stage="cross-stack-agent"
    ) as phase:
        # Dedup pre-filter (D-27).
        alt_issues_for_dedup: list[dict[str, Any]] = (
            json.loads(alts_p.read_text()) if alts_p.exists() else []
        )
        pairs = build_dedup_candidates(all_records, alt_issues_for_dedup)
        record_pairs = build_record_dedup_candidates(
            all_records, sources=deep_state.record_sources
        )
        dedup_p = dedup_candidates_path(dd)
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
                contributing_records=_merge_contributing_records(deep_state),
                structural_records_present=deep_state.structural_records_path is not None,
                failed_stacks=sorted(failed_stacks),
                identity=merge_identity,
                grounding=_loop_grounding(deep_state),
            )
            merge_unit = ReviewReuseUnit(reuse, "merge", merge_identity, merge_payload, coverage=coverage)
            if merge_unit.restore(deep_state.dd):
                if coverage is not None:
                    coverage.record_phase("merge", "complete", noop=host_noop)
                    persist_review_coverage(dd, coverage)
                _clear_merge_failure(dd)
                clear_review_budget_stop(dd, "Cross-stack merge")
                return None

        # Cross-stack merge (D-23..D-26).
        try:
            await phase_cross_stack_merge(
                ctx.backend_for("merge"),
                ctx.work,
                per_stack_records_paths=deep_state.records_paths,
                intent_path=deep_state.intent_path,
                alternatives_path=alts_p,
                dedup_candidates_path=dedup_p,
                exploration_dir=deep_state.exploration_dir,
                failed_stacks=failed_stacks or None,
                structural_records_path=deep_state.structural_records_path,
                intent_authoritative=deep_state.intent_authoritative,
                continuation=deep_state.arbiter_continuation,
                strategy=ctx.strategy("merge"),
                run_context=ctx.run_context,
                artifact_session=ctx.artifacts,
                allow_standalone=ctx.allow_standalone_artifacts,
            )
        except CrossStackMergeError as exc:
            if coverage is not None:
                coverage.record_phase(
                    "merge", "incomplete" if exc.budget_reason else "failed",
                    reasons=(ReasonCode.SYNTHESIS_FAILURE, reason_for_budget(exc.budget_reason))
                    if exc.budget_reason else (ReasonCode.SYNTHESIS_FAILURE,),
                )
                persist_review_coverage(dd, coverage)
            phase.finish(
                LifecycleStatus.PARTIAL if exc.budget_reason else LifecycleStatus.FAILED,
                LifecycleReasonCode.DOMAIN_FAILURE,
            )
            if exc.budget_reason:
                record_review_budget_stop(dd, "Cross-stack merge", exc.budget_reason)
            _salvage_merge_failure(ctx, exc)
            return None if exc.budget_reason else Stop(1)
        # Issue #361: a successful re-merge supersedes any stale salvage record, so
        # the structured ``MERGE_FAILURE_KEY`` entry is cleared here -- otherwise a
        # later ``--start-at merge``/``fix`` resume still warns 'merged results are
        # PARTIAL' even though the cross-stack merge has since succeeded.
        _clear_merge_failure(dd)
        clear_review_budget_stop(dd, "Cross-stack merge")
        if coverage is not None:
            coverage.record_phase("merge", "complete", noop=host_noop)
            persist_review_coverage(dd, coverage)
        # Issue #733 — store only a completed merge, once the same artifacts a
        # fresh run leaves are final on disk. A failed or budget-exhausted
        # merge returns above and never reaches here.
        if merge_unit is not None:
            merge_unit.store(lambda: _merge_store_payload(dd))
    return None


def _salvage_merge_failure(ctx: FlowContext, exc: CrossStackMergeError) -> None:
    """Persist partial surviving findings and a structured MERGE_FAILURE_KEY entry.

    Deduplicate language records, retain structural findings and existing stack
    failures. Missing/malformed prior failure data means no prior failures;
    errors writing either salvage artifact propagate as the project error type.
    """
    deep_state = DeepState(ctx.data)
    dd = deep_state.dd
    message = f"{exc}; consolidating surviving per-stack records into a partial report."
    if exc.budget_reason:
        print_warning(console, message + " Continuing to review publication.")
    else:
        print_error(console, "Cross-stack merge failed", message + " Relaunch with --start-at fix to resume.")

    # Build the partial canonical report from the surviving records via the
    # single-stack write helper (recoverability comes from the ``__merge__``
    # failure record + resumable stop, not a root flag, #361). Apply the D-27
    # dedup pre-filter; these host-written items carry no merge-agent provenance.
    records = _drop_cross_stack_duplicates(dd, deep_state.records)
    _write_single_stack_merged_items(
        ctx.work.repo,
        dd,
        records,
        deep_state.structural_records_path,
        failed_stacks=deep_state.failed_stacks_or_none or None,
        artifact_session=ctx.artifacts,
        allow_standalone=ctx.allow_standalone_artifacts,
    )

    # Record the failure for resume. Never drop existing per-stack entries.
    failures_p = per_stack_failures_path(dd)
    failures = _load_failures(failures_p)
    failures[MERGE_FAILURE_KEY] = {
        "response_shape": exc.response_shape,
        "stack_context": exc.stack_context,
        "message": redact_text(str(exc)),
    }
    failures_p.write_text(json.dumps(failures, indent=2, sort_keys=True))
    print_info(console, f"Wrote partial merged items and merge-failure record to {dd}")


async def _step_single_stack_merge(ctx: FlowContext) -> None:
    """Tiny-diff single-stack bypass (#172): host-side merged-items write."""
    deep_state = DeepState(ctx.data)
    failed_stacks: dict[str, str] = deep_state.failed_stacks

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
            deep_state.dd,
            deep_state.records,
            deep_state.structural_records_path,
            failed_stacks=failed_stacks or None,
            artifact_session=ctx.artifacts,
            allow_standalone=ctx.allow_standalone_artifacts,
        )
    if deep_state.review_coverage is not None:
        deep_state.review_coverage.record_phase("merge", "complete", noop=True)
        persist_review_coverage(deep_state.dd, deep_state.review_coverage)


async def _step_load_items(ctx: FlowContext) -> Stop | None:
    """Host-side merged-items guard + render-only markdown recovery."""
    deep_state = DeepState(ctx.data)
    target_dir = ctx.work.repo
    dd = deep_state.dd

    print_stage_progress(console, 5, 5, _PIPELINE_STAGE_NAMES[4])
    merged_report = review_output_path_for(
        target_dir,
        session=ctx.artifacts,
        allow_standalone=ctx.allow_standalone_artifacts,
    )

    # merged-items.json is the canonical source of truth; review-output.md is
    # render-only. The missing-input guard keys on the JSON so a --start-at fix
    # resume with surviving JSON but absent markdown proceeds rather than bailing.
    items_file = merged_items_path(dd)
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
        deep_copy = merged_report_path(dd)
        if deep_copy.exists():
            merged_report.write_text(deep_copy.read_text())



    warning = render_review_warnings(review_warnings(dd))
    if warning:
        for report in (merged_report, merged_report_path(dd)):
            if report.exists() and warning not in report.read_text():
                report.write_text(warning + "\n\n" + report.read_text())

    deep_state.merged_report = merged_report
    deep_state.items_file = items_file
    return None


async def _step_findings_out(ctx: FlowContext) -> Stop:
    """Stop at the review boundary; recorder-scoped finalization owns the export."""
    ctx.data["findings_projection_ready"] = True
    return Stop(0)


async def _step_supervise(ctx: FlowContext) -> None:
    """Record the required supervision stage at its artifact completion boundary."""
    deep_state = DeepState(ctx.data)
    coverage = deep_state.review_coverage
    from daydream.deep.prompts import build_supervise_prompt
    from daydream.review_profile import build_default_profile

    try:
        strategy = ctx.strategy("supervision")
        default_strategy = build_default_profile().strategies["supervision"].content
        input_items = json.loads(deep_state.items_file.read_text())["items"]
        noop = not input_items and (_supervisor_mode(ctx.config) == "rules" or (
            (strategy is None or strategy == default_strategy)
            and ctx.registry.prompt("supervise") is build_supervise_prompt))
        budget_reason = await _supervise_items(ctx)
    except Exception as exc:
        if coverage is not None:
            coverage.record_phase("supervision", "failed", reasons=(reason_for_exception(exc),))
            _persist_failure_coverage(deep_state, exc)
        raise
    if coverage is not None:
        if budget_reason:
            coverage.record_phase("supervision", "incomplete", reasons=(reason_for_budget(budget_reason),))
        else:
            coverage.record_phase("supervision", "complete", noop=noop)
        persist_review_coverage(deep_state.dd, coverage)


async def _supervise_items(ctx: FlowContext) -> str | None:
    """Apply the configured findings supervisor to canonical merged items."""
    deep_state = DeepState(ctx.data)
    mode = _supervisor_mode(ctx.config)
    file_config = ctx.config.file_config
    items_file: Path = deep_state.items_file
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
                diff_path=deep_state.diff_path,
                intent_path=deep_state.intent_path,
                alternatives_path=deep_state.alts_path,
                exploration_dir=deep_state.exploration_dir,
                strategy=ctx.strategy("supervision"),
                run_context=ctx.run_context,
                artifact_session=ctx.artifacts,
                allow_standalone=ctx.allow_standalone_artifacts,
            )
    kept, held, events = apply_findings_verdicts(items, verdicts)
    items_file.write_text(json.dumps({"items": kept, "held": held}, indent=2))

    report = render_report(kept)
    warning = render_review_warnings(review_warnings(deep_state.dd))
    if warning:
        report = warning + "\n\n" + report
    held_section = render_held_section(held)
    if held_section:
        report = report.rstrip() + "\n\n" + held_section + "\n"
    deep_report = merged_report_path(deep_state.dd)
    deep_report.write_text(report)
    deep_state.merged_report.write_text(report)

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
    deep_state = DeepState(ctx.data)
    if ctx.config.findings_out is None and deep_state.review_coverage is not None:
        from daydream.deep.review_terminal import finalize_review

        if not deep_state.review_coverage.is_finalized:
            finalize_review(ctx, "completed")
    if deep_state.mode == "review":
        return None

    from daydream.pr_review import PostStatus, post_review_to_pr_from_report, resolve_review_renderers
    from daydream.pr_run_info import LiveRunInfoSource, render_live_run_info

    recorder = get_current_recorder()
    run_info = render_live_run_info(LiveRunInfoSource(recorder, ctx.artifacts))
    if run_info.diagnostic is not None:
        print_warning(console, run_info.diagnostic)

    items_file: Path = deep_state.items_file
    pr_kwargs: dict[str, Any] = (
        {"pr_number": ctx.config.pr_number}
        if ctx.config.pr_number is not None
        else {}
    )
    warnings = review_warnings(deep_state.dd)
    if warnings:
        pr_kwargs["review_warnings"] = warnings
    outcome = await post_review_to_pr_from_report(
        ctx.work.repo,
        items_file,
        run_info=run_info.markdown,
        renderers=resolve_review_renderers(ctx.registry),
        console=console,
        post=deep_state.mode == "comment",
        approve_on_clean=_resolve_opt_in(ctx.config, "approve_on_clean"),
        diagram_blocks=(deep_state.diagrams or {}).get("blocks"),
        run_context=ctx.run_context,
        auth=ctx.github_execution.auth,
        **pr_kwargs,
    )
    if deep_state.mode == "comment" and outcome in (PostStatus.NO_PR, PostStatus.FAILED):
        return Stop(1)
    return None
