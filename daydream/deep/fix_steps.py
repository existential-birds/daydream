"""Authorized fix cycles, retained-tree evidence, test, commit, CI, and cleanup."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from daydream import git_ops
from daydream.agent import console
from daydream.artifact_visibility import artifact_dir_for, review_output_path_for
from daydream.config import (
    DEFAULT_GROUP_MAX_SERIAL_ITEMS,
    DEFAULT_GROUP_MAX_WALL_S,
    DEFAULT_QUALITY_GATE_ENABLED,
    DEFAULT_VERIFY_ALL,
    REVIEW_OUTPUT_FILE,
)
from daydream.deep import fix_state
from daydream.deep.artifacts import DeepArtifact
from daydream.deep.fix_selection import (
    ACTIONABLE_VERDICTS,
    _attach_verdicts as _attach_verdicts,
    _round_dispatch_items as _round_dispatch_items,
    _verdict_buckets as _verdict_buckets,
)
from daydream.deep.fix_state import EvidenceKey, FixCycleState
from daydream.deep.quality_gate import QualityGateThresholds, _evaluate_quality_gate, capture_quality
from daydream.deep.records import stamp_item_uids
from daydream.deep.scope_issues import (
    _resolve_changed_files,
)
from daydream.deep.settings import _resolve_config_value, _resolve_opt_in
from daydream.deep.verify_selection import SelectionConfig, resolve_selection_config
from daydream.extensions.api import BreakLoop, Stop
from daydream.fix_footprint import AuthorizedFixFootprint
from daydream.flows.engine import FlowContext
from daydream.json_utils import atomic_write_json
from daydream.phases import (
    PushAttemptError,
    PushReceipt,
    TestAndHealResult,
    TestAttemptEvidence,
    phase_commit_push,
    phase_fix_parallel,
    phase_test_and_heal,
    phase_test_once,
    phase_verify_recommendations,
    require_empty_staged_index,
    severity_sorted,
)
from daydream.repository_paths import strip_dot_slash
from daydream.run_context import resolve_run_context
from daydream.trajectory import (
    DaydreamPhase,
    current_session_id,
    now_iso,
    phase_scope,
    redact_structured_text,
)
from daydream.ui import (
    format_verdict_join,
    print_error,
    print_fix_complete,
    print_success,
    print_verification_summary,
    print_warning,
)

if TYPE_CHECKING:
    from daydream.run_config import RunConfig


def _record_fix_preflight_rejection(dd: Path, items: list[dict[str, Any]]) -> None:
    """Record an admitted cycle's blocked items without reflecting unsafe paths."""
    failures = {
        item["item_uid"]: "fix_preflight_rejected: fix cycle did not start"
        for item in items
    }
    try:
        atomic_write_json(DeepArtifact.FIX_FAILURES.at(dd), failures, sort_keys=True)
    except OSError as exc:
        print_error(console, "Fix preflight failure audit failed", str(exc))


def _artifact_dir(ctx: FlowContext) -> Path:
    """Resolve the active private artifact directory for this run."""
    return artifact_dir_for(
        ctx.work.repo,
        session=ctx.artifacts,
        allow_standalone=ctx.allow_standalone_artifacts,
    )


async def _step_fix_gate(ctx: FlowContext) -> Stop | None:
    """Fix-apply gate; on accept, load and severity-sort the canonical items."""
    deep_data = ctx.deep_data()
    # Fix-apply gate across the two interaction axes. ``--yes`` auto-applies;
    # an unattended run with no assumption declines (safe_default=False) so a
    # piped/CI run never mutates without intent; otherwise prompt.
    decision = resolve_run_context(ctx.run_context).confirm(
        safe_default=False,
        question="Apply fixes now? [y/N]",
        default="n",
        console=console,
    )
    if not decision:
        print_success(console, f"Report written to {deep_data['merged_report']}. Exiting.")
        return Stop(0)

    # A rejected index preflight must preserve prior artifacts as well as
    # source bytes. Only an admitted run may start a new evidence session.
    try:
        initial_index = require_empty_staged_index(ctx.work)
    except (OSError, git_ops.GitError) as exc:
        print_error(console, "Fix preflight failed", str(exc))
        return Stop(1)

    # An accepted gate starts a new evidence session. No prior run's success,
    # patch, or policy audit may be inherited if this run later stops early.
    dd: Path = deep_data["dd"]
    stale_paths = (
        DeepArtifact.FIX_FOOTPRINT.at(dd),
        DeepArtifact.FIX_OUTCOMES.at(dd),
        DeepArtifact.TEST_VERDICT.at(dd),
        DeepArtifact.RECOMMENDED_CAPTURE.at(dd),
        DeepArtifact.GENERATED_FILE_VIOLATIONS.at(dd),
        DeepArtifact.FIX_FAILURES.at(dd),
        DeepArtifact.FIX_LEFTOVER_UNTRACKED.at(dd),
        DeepArtifact.STABILIZATION_FAILED.at(dd),
        _artifact_dir(ctx) / "recommended.patch",
    )
    try:
        for stale in stale_paths:
            stale.unlink(missing_ok=True)
    except OSError as exc:
        print_error(console, "Fix preflight failed", str(exc))
        return Stop(1)

    # Read canonical merged items directly (validated above). Replaces an LLM
    # re-parse of the markdown, which silently dropped structural findings; here
    # they are ordinary tagged items that reach phase_fix like any other.
    items_file: Path = deep_data["items_file"]
    items: list[dict[str, Any]] = json.loads(items_file.read_text())["items"]
    # Normalize legal ./ prefixes once to match Git-derived footprint paths.
    for _item in items:
        _file = _item.get("file")
        if isinstance(_file, str):
            _item["file"] = strip_dot_slash(_file)
    stamp_item_uids(items)
    if not items:
        print_success(console, "No actionable items -- done.")
        return Stop(0)

    # Authorize canonical primary/related paths even outside the diff, without
    # widening each fixer to every reviewed file.
    changed_files = _resolve_changed_files(ctx)

    try:
        stable_head = git_ops.head_sha(ctx.work.repo)
        stable_ref = git_ops.stash_create(ctx.work.repo) or stable_head
        preexisting_untracked = git_ops.snapshot_untracked_paths(
            ctx.work.repo, include_runtime_artifacts=False
        )
        preexisting_gitlinks = git_ops.snapshot_worktree_gitlinks(ctx.work.repo)
        footprint = AuthorizedFixFootprint.build(
            ctx.work.repo, set(changed_files or []), items
        )
    except (OSError, ValueError, git_ops.GitError) as exc:
        _record_fix_preflight_rejection(dd, items)
        print_error(console, "Fix preflight failed", str(exc))
        return Stop(1)

    session_id = current_session_id() or ctx.work.run_id
    state = FixCycleState(
        session_id=session_id,
        stable_ref=stable_ref,
        stable_head=stable_head,
        initial_index=initial_index,
        preexisting_untracked=preexisting_untracked,
        preexisting_gitlinks=preexisting_gitlinks,
        footprint=footprint,
    )
    deep_data["fix_cycle_state"] = state
    try:
        initial_key = EvidenceKey(fix_state._capture_full_delta_key(ctx.work, state), footprint.policy_revision)
        fix_state._write_footprint_audit(ctx, state, initial_key)
    except (OSError, git_ops.GitError) as exc:
        _record_fix_preflight_rejection(dd, items)
        print_error(console, "Fix preflight failed", str(exc))
        return Stop(1)

    # Severity-ordered (high before medium before low), stable within a
    # tier so equal-severity items keep their canonical merge order.
    deep_data["items"] = severity_sorted(items)
    return None


async def _step_verify(ctx: FlowContext) -> None:
    """Recommendation verification (#83) + verdict join rendering."""
    deep_data = ctx.deep_data()
    dd = deep_data["dd"]
    items: list[dict[str, Any]] = deep_data["items"]

    # Accepted fresh and resumed fix gates verify recommendations; declined gates
    # produce neither a verifier call nor its artifact.
    selection = _resolve_verify_selection(ctx.config)
    async with phase_scope(DaydreamPhase.VERIFY):
        verdicts_file, verdicts_payload = await phase_verify_recommendations(
            ctx.backend_for("verify"),
            ctx.work,
            merged_items_path=deep_data["items_file"],
            deep_dir=dd,
            strategy=ctx.strategy("verification"),
            selection=selection,
            run_context=ctx.run_context,
        )
    print_verification_summary(console, verdicts_file)

    # Attach verifier verdicts to items by `id` (advisory; phase_fix reads them).
    items = _attach_verdicts(items, verdicts_payload)
    deep_data["items"] = items
    matched_ids, unmatched_ids, skipped_ids, structural_ids, other_ids = _verdict_buckets(
        items, verdicts_payload
    )
    # Use persisted selection decisions to distinguish operator skips from exempt
    # lenses, so coverage accounting includes every fix-loop item.
    console.print(
        format_verdict_join(
            matched=matched_ids,
            unmatched=unmatched_ids,
            skipped=skipped_ids,
            structural=structural_ids,
            other=other_ids,
            total=len(items),
        )
    )


def _resolve_verify_selection(config: RunConfig) -> SelectionConfig:
    """Resolve CLI-over-file verification selection, then built-in defaults.

    Only real booleans control verify_all. Extra risk categories are additive;
    unknown names raise before verification instead of changing selection silently.
    """
    file_config = config.file_config
    verify_all = config.verify_all
    if verify_all is None and file_config is not None:
        verify_all = file_config.verify_all
    extra = config.verify_extra_risk_categories
    if extra is None and file_config is not None:
        extra = file_config.extra_risk_categories
    return resolve_selection_config(
        verify_all=verify_all if verify_all is not None else DEFAULT_VERIFY_ALL,
        extra_categories=extra,
    )


MAX_POST_TEST_STABILIZATION_PASSES = 2


def _confinement_stop(
    ctx: FlowContext,
    state: FixCycleState,
    phase: str,
    round_number: int | None,
    message: str,
    error: str | None = None,
    *,
    warning: bool = False,
) -> Stop:
    """Restore protected state, report the terminal failure, and stop the flow."""
    confinement_error = fix_state._enforce_terminal_confinement(
        ctx,
        state,
        phase=phase,
        round_number=round_number,
    )
    if warning:
        print_warning(console, message)
    elif error is not None:
        print_error(console, message, error)
    if confinement_error is not None:
        label = (
            "Test failure confinement failed"
            if phase == "test_failure"
            else "Fix failure confinement failed"
        )
        print_error(console, label, confinement_error)
    return Stop(1)


def _stabilization_stop(
    ctx: FlowContext,
    state: FixCycleState,
    reason: str,
    *,
    round_number: int | None,
) -> Stop:
    """Fail closed after finalization while still restoring and auditing scope."""
    confinement_error = fix_state._enforce_terminal_confinement(
        ctx,
        state,
        phase="post_test_failure",
        round_number=round_number,
    )
    if confinement_error is not None:
        reason = f"{reason}; confinement failed: {confinement_error}"
    try:
        fix_state._persist_stabilization_failure(ctx, state, reason)
    except OSError as exc:
        print_error(console, "Stabilization failure audit failed", str(exc))
    return Stop(1)


async def _step_fix(ctx: FlowContext) -> Stop | None:
    """Run one policy-bound fix round using its own complete rollback point."""
    state = FixCycleState.require(ctx)
    deep_data = ctx.deep_data()
    items = _round_dispatch_items(ctx, deep_data["items"])
    if not items:
        return None
    try:
        round_snapshot = fix_state._round_rollback_snapshot(state, ctx.work)
    except Exception as exc:
        print_error(console, "Fix snapshot failed", str(exc))
        return Stop(1)
    config = ctx.config
    quality_enabled = _resolve_config_value(
        config, "quality_gate_enabled", DEFAULT_QUALITY_GATE_ENABLED
    )
    reviewed_python = {
        path for path in (_resolve_changed_files(ctx) or []) if path.endswith(".py")
    }
    if quality_enabled:
        quality_before, quality_unavailable = await capture_quality(
            _artifact_dir(ctx),
            ctx.work.repo,
            reviewed_python,
        )
    else:
        quality_before, quality_unavailable = None, None
    exploration_dir = deep_data.get("exploration_dir")
    exploration_dir = exploration_dir if isinstance(exploration_dir, Path) else None
    test_map_path = exploration_dir / "test-map.json" if exploration_dir else None
    intent_p: Path = deep_data["intent_path"]
    grounded = config.start_at not in ("per-stack", "merge", "fix")
    async with phase_scope(DaydreamPhase.FIX):
        try:
            failures = await phase_fix_parallel(
                ctx.backend_for("fix"),
                ctx.work,
                items,
                intent_path=intent_p if grounded and intent_p.exists() else None,
                group_max_wall_s=_resolve_config_value(
                    config, "group_max_wall_s", DEFAULT_GROUP_MAX_WALL_S
                ),
                group_max_serial_items=_resolve_config_value(
                    config, "group_max_serial_items", DEFAULT_GROUP_MAX_SERIAL_ITEMS
                ),
                # One cumulative retry-overhead allowance per file group, read
                # directly (not via ``_resolve_config_value``) because it is
                # tri-state: unset must stay None so ``run_agent`` applies its
                # default instead of reading it as an explicit disable.
                retry_recovery_allowance_s=(
                    config.file_config.retry_recovery_allowance_s
                    if config.file_config is not None
                    else None
                ),
                exploration_dir=exploration_dir,
                test_map_path=test_map_path,
                footprint=state.footprint,
                round_snapshot=round_snapshot,
                file_scope_issues=_resolve_opt_in(config, "scope_issue_filing"),
                auth=ctx.github_execution.auth,
                run_context=ctx.run_context,
            )
        except Exception as exc:
            return _confinement_stop(ctx, state, "fix_failure", deep_data.get("iteration"), "Fix failed", str(exc))
    budget_prefix = "file_group_budget_exceeded:"
    exception_failures = {
        path: reason
        for path, reason in failures.items()
        if not reason.startswith(budget_prefix)
    }
    failures_artifact = DeepArtifact.FIX_FAILURES.at(deep_data["dd"])
    try:
        if failures:
            atomic_write_json(failures_artifact, failures, sort_keys=True)
        else:
            failures_artifact.unlink(missing_ok=True)
    except Exception as exc:
        return _confinement_stop(
            ctx, state, "fix_failure", deep_data.get("iteration"), "Fix failure audit failed", str(exc)
        )
    if exception_failures:
        confinement_error = fix_state._enforce_terminal_confinement(
            ctx,
            state,
            phase="fix_failure",
            round_number=deep_data.get("iteration"),
        )
        artifact_errors: list[str] = []
        try:
            leftover = sorted(
                set(
                    git_ops.snapshot_untracked_paths(
                        ctx.work.repo, include_runtime_artifacts=False
                    )
                )
                - set(state.preexisting_untracked)
            )
            if leftover:
                atomic_write_json(DeepArtifact.FIX_LEFTOVER_UNTRACKED.at(deep_data["dd"]), leftover)
        except Exception as exc:
            artifact_errors.append(str(exc))
        print_warning(
            console,
            "Failed fix groups were restored; this run will not commit successful "
            "sibling-group edits: " + ", ".join(sorted(exception_failures)),
        )
        if confinement_error is not None:
            print_error(console, "Fix failure confinement failed", confinement_error)
        if artifact_errors:
            print_error(console, "Fix failure audit failed", "; ".join(artifact_errors))
        return Stop(1)
    try:
        fix_state._strict_scope_and_scrub(
            ctx,
            state,
            phase="fix",
            round_number=deep_data.get("iteration"),
        )
        snapshot = fix_state.capture_retained_tree(ctx.work, state)
    except Exception as exc:
        return _confinement_stop(
            ctx, state, "fix_failure", deep_data.get("iteration"), "Fix scope enforcement failed", str(exc)
        )
    deep_data["fix_round_snapshot"] = snapshot
    try:
        await _evaluate_quality_gate(
            enabled=quality_enabled,
            thresholds=QualityGateThresholds.from_config(config),
            daydream_dir=_artifact_dir(ctx),
            code_workspace=ctx.work.repo,
            dd=deep_data["dd"],
            candidates={path for path in snapshot.paths if path.endswith(".py")}
            | {
                str(item["file"])
                for item in deep_data["items"]
                if isinstance(item.get("file"), str) and str(item["file"]).endswith(".py")
            },
            before=quality_before,
            before_unavailable_reason=quality_unavailable,
            iteration=deep_data.get("iteration"),
        )
    except Exception as exc:
        return _confinement_stop(
            ctx, state, "fix_failure", deep_data.get("iteration"), "Fix quality evaluation failed", str(exc)
        )
    return None


async def verify_retained_tree(
    ctx: FlowContext,
    snapshot: fix_state.RetainedTreeSnapshot,
    items: list[dict[str, Any]],
    *,
    pass_number: int,
) -> dict[str, dict[str, Any]]:
    """Verify all canonical findings and join numeric wire ids to durable uids."""
    from daydream.phases import phase_fix_verify

    async with phase_scope(DaydreamPhase.VERIFY):
        verdicts = await phase_fix_verify(
            ctx.backend_for("verify"),
            ctx.work,
            items,
            snapshot.verifier_patch,
            round_number=pass_number,
            run_context=ctx.run_context,
        )
    by_id = {
        item.get("id"): item
        for item in items
        if isinstance(item.get("id"), int) and isinstance(item.get("item_uid"), str)
    }
    state = FixCycleState.require(ctx)
    outcomes: dict[str, dict[str, Any]] = {}
    for verdict in verdicts:
        item = by_id.get(verdict.get("issue_id"))
        if item is None:
            continue
        stored = dict(verdict)
        stored["issue_id"] = item["id"]
        uid = item["item_uid"]
        if stored.get("verdict") == "resolved":
            target = state.last_fix_target_by_uid.get(uid)
            if target is not None:
                stored["path"] = target
        outcomes[uid] = stored
    return outcomes


def _persist_fix_outcomes_current(
    ctx: FlowContext,
    state: FixCycleState,
    key: EvidenceKey,
    outcomes: dict[str, dict[str, Any]],
) -> None:
    deep_data = ctx.deep_data()
    atomic_write_json(
        DeepArtifact.FIX_OUTCOMES.at(deep_data["dd"]),
        {
            "session_id": state.session_id,
            "evidence_key": fix_state._evidence_payload(key),
            "outcomes": outcomes,
        },
        sort_keys=True,
    )


async def _step_fix_verify(ctx: FlowContext) -> BreakLoop | Stop | None:
    state = FixCycleState.require(ctx)
    deep_data = ctx.deep_data()
    snapshot = deep_data.get("fix_round_snapshot")
    if not isinstance(snapshot, fix_state.RetainedTreeSnapshot):
        try:
            snapshot = fix_state.capture_retained_tree(ctx.work, state)
        except Exception as exc:
            return _confinement_stop(
                ctx, state, "fix_verify_failure", deep_data.get("iteration"),
                "Fix verification capture failed", str(exc)
            )
    iteration = deep_data.get("iteration")
    round_number = iteration if isinstance(iteration, int) else 1
    try:
        outcomes = await verify_retained_tree(
            ctx, snapshot, deep_data["items"], pass_number=round_number
        )
        key = EvidenceKey(snapshot.tree_key, state.footprint.policy_revision)
        state.latest_retained = snapshot
        state.verifier_key = key
        deep_data["fix_outcomes"] = outcomes
        _persist_fix_outcomes_current(ctx, state, key, outcomes)
        fix_state._write_footprint_audit(ctx, state, key)
    except Exception as exc:
        return _confinement_stop(ctx, state, "fix_verify_failure", round_number, "Fix verification failed", str(exc))
    actionable = _actionable_verdicts(outcomes)
    if actionable and iteration not in (None, 3):
        return None
    _render_fix_outcome_summary(deep_data["items"], outcomes)
    if "regressed" in actionable:
        print_error(
            console, "Fix verification failed",
            "The retained changes introduce a regression; commit and push blocked.",
        )
        return Stop(1)
    if actionable:
        print_warning(
            console,
            f"Fix attempts exhausted with {len(actionable)} finding(s) still unresolved; "
            "continuing to validate the retained changes before commit and push.",
        )
    return BreakLoop()


def _actionable_verdicts(outcomes: dict[str, dict[str, Any]]) -> list[str]:
    """Verdicts that schedule re-dispatch in a later round (issue #744)."""
    return [
        v["verdict"]
        for v in outcomes.values()
        if v.get("verdict") in ACTIONABLE_VERDICTS
    ]


def _render_fix_outcome_summary(
    items: list[dict[str, Any]],
    outcomes: dict[str, dict[str, Any]],
) -> None:
    """Render final verdicts with the last dispatch round's numbering.

    Previously resolved findings absent from that round remain in the durable
    fix-outcomes artifact but are omitted from these terminal lines.
    """
    if not outcomes:
        return
    numbering_by_uid = {
        item["item_uid"]: num
        for num, item in enumerate(items, start=1)
        if isinstance(item.get("item_uid"), str)
    }
    total = len(items)
    for outcome_key, verdict in outcomes.items():
        num = numbering_by_uid.get(outcome_key)
        if num is None:
            continue
        print_fix_complete(console, num, total, outcome=verdict.get("verdict"))


def _test_attempt_payload(attempt: TestAttemptEvidence) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "session_id": attempt.session_id,
        "kind": attempt.kind,
        "command": list(attempt.command) if attempt.command is not None else "agent-fallback",
        "passed": attempt.passed,
        "input_tree_key": attempt.input_tree_key,
        "output_tree_key": attempt.output_tree_key,
    }
    if attempt.identity is not None:
        payload["identity"] = attempt.identity.payload()
    return payload


def _persist_test_verdict(
    ctx: FlowContext,
    state: FixCycleState,
    *,
    passed: bool,
    ignored: bool,
    attempts: list[TestAttemptEvidence],
) -> None:
    deep_data = ctx.deep_data()
    from daydream.remote_ci import local_host_facts

    atomic_write_json(
        DeepArtifact.TEST_VERDICT.at(deep_data["dd"]),
        {
            "session_id": state.session_id,
            "passed": passed,
            "ignored": ignored,
            "retries": max(0, len(attempts) - 1),
            "attempts": [_test_attempt_payload(attempt) for attempt in attempts],
            "local_host": local_host_facts(),
        },
        sort_keys=True,
    )


def _authorize_final_red_override(ctx: FlowContext) -> bool:
    """Require a fresh interactive decision for a changed-tree red retest."""
    run_context = resolve_run_context(ctx.run_context)
    policy = run_context.policy
    if policy.assume is not None or not policy.interactive:
        return False
    return run_context.confirm(
        safe_default=False,
        question="Final no-heal validation is still red. Ignore and continue? [y/N]",
        default="n",
        console=console,
    )


async def finalize_retained_tree_after_test(
    ctx: FlowContext, result: TestAndHealResult
) -> Stop | None:
    """Strictly stabilize post-heal state in at most two guard passes."""
    deep_data = ctx.deep_data()
    state = FixCycleState.require(ctx)
    attempts = list(result.attempts)
    if not attempts:
        return _stabilization_stop(
            ctx, state, "test produced no evidence", round_number=None
        )
    evidence = attempts[-1]
    ignored = result.ignored

    def _stop(reason: str) -> Stop:
        return _stabilization_stop(ctx, state, reason, round_number=pass_number)

    for pass_number in range(1, MAX_POST_TEST_STABILIZATION_PASSES + 1):
        try:
            mutated = fix_state._strict_scope_and_scrub(
                ctx,
                state,
                phase="post_test",
                round_number=pass_number,
            )
            snapshot = fix_state.capture_retained_tree(ctx.work, state)
            key = EvidenceKey(snapshot.tree_key, state.footprint.policy_revision)
            fix_state._write_footprint_audit(ctx, state, key)
        except Exception as exc:
            return _stop(f"guard/capture/audit failed: {exc}")

        if state.verifier_key != key:
            prior_outcomes = deep_data.get("fix_outcomes", {}) or {}
            try:
                outcomes = await verify_retained_tree(
                    ctx, snapshot, deep_data["items"], pass_number=pass_number
                )
                _persist_fix_outcomes_current(ctx, state, key, outcomes)
            except Exception as exc:
                return _stop(f"final verifier failed: {exc}")
            state.verifier_key = key
            deep_data["fix_outcomes"] = outcomes
            if any(
                outcome.get("verdict") == "regressed"
                or (
                    outcome.get("verdict") in ACTIONABLE_VERDICTS
                    and prior_outcomes.get(uid, {}).get("verdict")
                    not in {"unresolved", "wrong_target"}
                )
                for uid, outcome in outcomes.items()
            ):
                return _stop("final verifier remains actionable")

        matching_test = (
            evidence.session_id == state.session_id
            and evidence.input_tree_key == snapshot.tree_key
            and evidence.output_tree_key == snapshot.tree_key
        )
        ran_test = False
        if not matching_test and pass_number == 1:
            try:
                evidence, _, _ = await phase_test_once(
                    ctx.backend_for("test"),
                    ctx.work,
                    config=ctx.config,
                    session_id=state.session_id,
                    capture_tree_key=lambda: fix_state._capture_full_delta_key(ctx.work, state),
                    recipe=deep_data.get("test_recipe"),
                    run_context=ctx.run_context,
                )
            except Exception as exc:
                return _stop(f"final test failed to run: {exc}")
            attempts.append(evidence)
            ignored = False if evidence.passed else _authorize_final_red_override(ctx)
            ran_test = True
            _persist_test_verdict(
                ctx,
                state,
                passed=evidence.passed,
                ignored=ignored,
                attempts=attempts,
            )

        if pass_number == 1 and (mutated or ran_test):
            continue
        stable_test = (
            evidence.input_tree_key == snapshot.tree_key
            and evidence.output_tree_key == snapshot.tree_key
            and (evidence.passed or ignored)
        )
        if mutated or state.verifier_key != key or not stable_test:
            return _stop("post-test tree did not stabilize")

        try:
            patch_path = _artifact_dir(ctx) / "recommended.patch"
            patch_path.parent.mkdir(parents=True, exist_ok=True)
            patch_path.write_bytes(snapshot.recommended_patch)
            atomic_write_json(
                DeepArtifact.RECOMMENDED_CAPTURE.at(deep_data["dd"]),
                {
                    "session_id": state.session_id,
                    "capture_point": "post_test",
                    "tree_key": snapshot.tree_key,
                    "evidence_key": fix_state._evidence_payload(key),
                },
                sort_keys=True,
            )
        except OSError as exc:
            return _stop(f"recommended capture failed: {exc}")
        state.latest_retained = snapshot
        # The finalized tree and the evidence that validated it are one
        # fact: only the attempt whose before/after tree key equals the
        # retained tree key may become the reuse offer (issue #1408).
        if (
            evidence.input_tree_key == snapshot.tree_key
            and evidence.output_tree_key == snapshot.tree_key
        ):
            state.latest_test_evidence = evidence
        return None

    return _stabilization_stop(
        ctx,
        state,
        "post-test pass bound exhausted",
        round_number=MAX_POST_TEST_STABILIZATION_PASSES,
    )


async def _step_test(ctx: FlowContext) -> Stop | None:
    """Run typed, identity-bound tests and strictly finalize the retained tree."""
    deep_data = ctx.deep_data()
    state = FixCycleState.require(ctx)
    async with phase_scope(DaydreamPhase.TEST):
        try:
            result = await phase_test_and_heal(
                ctx.backend_for("test"),
                ctx.work,
                feedback_items=deep_data["items"],
                config=ctx.config,
                session_id=state.session_id,
                capture_tree_key=lambda: fix_state._capture_full_delta_key(ctx.work, state),
                footprint=state.footprint,
                run_context=ctx.run_context,
                artifact_session=ctx.artifacts,
                allow_standalone=ctx.allow_standalone_artifacts,
                recipe=deep_data.get("test_recipe"),
            )
            if not isinstance(result, TestAndHealResult):
                raise TypeError("phase_test_and_heal returned an invalid evidence result")
            _persist_test_verdict(
                ctx,
                state,
                passed=result.passed,
                ignored=result.ignored,
                attempts=list(result.attempts),
            )
        except Exception as exc:
            return _confinement_stop(ctx, state, "test_failure", None, "Test evidence failed", str(exc))
    if not result.proceed:
        return _confinement_stop(ctx, state, "test_failure", None, "Tests failed after fix attempt.", warning=True)
    return await finalize_retained_tree_after_test(ctx, result)


def _persist_push_verdict(
    ctx: FlowContext,
    receipt: PushReceipt,
    *,
    status: Literal["succeeded", "failed"],
    started_at: str,
    diagnostic: str | None = None,
) -> None:
    """Replace the current session's exact push-attempt outcome atomically."""
    deep_data = ctx.deep_data()
    state = FixCycleState.require(ctx)
    payload: dict[str, object] = {
        "schema_version": 1,
        "session_id": state.session_id,
        "status": status,
        "remote": receipt.remote,
        "branch": receipt.branch,
        "pushed_sha": receipt.sha,
        "pushed_repository": receipt.pushed_repository,
        "started_at": started_at,
        "updated_at": now_iso(),
    }
    if diagnostic is not None:
        payload["diagnostic"] = redact_structured_text(diagnostic)[:2_000]
    atomic_write_json(
        DeepArtifact.PUSH_VERDICT.at(deep_data["dd"]),
        payload,
        indent=2,
        sort_keys=True,
        trailing_newline=True,
    )


async def _step_commit(ctx: FlowContext) -> Stop | None:
    """Stage the finalized retained paths once, then commit and push them."""
    deep_data = ctx.deep_data()
    state = FixCycleState.require(ctx)
    snapshot = state.latest_retained
    if snapshot is None:
        print_error(console, "Commit/Push Failed", "retained tree was not finalized")
        return Stop(1)
    key = EvidenceKey(snapshot.tree_key, state.footprint.policy_revision)
    try:
        for path in sorted(snapshot.paths):
            state.footprint.record_git_event(
                action="stage",
                path=path,
                origin="staging",
                phase="commit",
                round_number=None,
                reason="final retained path selected for the validated index",
            )
        fix_state._write_footprint_audit(ctx, state, key)
    except Exception as exc:
        print_error(console, "Commit/Push Failed", str(exc))
        return Stop(1)
    started_at = now_iso()
    try:
        receipt = await phase_commit_push(
            ctx.backend_for("fix"),
            ctx.work,
            config=ctx.config,
            items=[
                item for item in deep_data.get("items", []) or []
                if (deep_data.get("fix_outcomes", {}) or {}).get(item.get("item_uid", ""), {}).get("verdict")
                == "resolved"
            ],
            retained_paths=snapshot.paths,
            retained_states=snapshot.states,
            initial_index=state.initial_index,
            recipe=deep_data.get("test_recipe"),
            evidence=state.latest_test_evidence,
            retained_tree_key=snapshot.tree_key,
            run_context=ctx.run_context,
        )
    except PushAttemptError as exc:
        try:
            _persist_push_verdict(
                ctx,
                exc.receipt,
                status="failed",
                started_at=started_at,
                diagnostic=str(exc),
            )
        except Exception as artifact_exc:
            print_error(console, "Push verdict persistence failed", str(artifact_exc))
        print_error(console, "Commit/Push Failed", str(exc))
        return Stop(1)
    except Exception as exc:
        print_error(console, "Commit/Push Failed", str(exc))
        return Stop(1)
    if receipt is not None:
        try:
            _persist_push_verdict(
                ctx,
                receipt,
                status="succeeded",
                started_at=started_at,
            )
        except Exception as exc:
            print_error(console, "Push verdict persistence failed", str(exc))
            return Stop(1)
        deep_data["push_receipt"] = receipt
    return None


async def _perform_cleanup(ctx: FlowContext) -> None:
    """Remove review output on successful exits when cleanup is enabled.

    Explicit flags win; otherwise --yes accepts, unattended runs retain it, and
    interactive runs prompt. Early Stop(0) honors cleanup; failures keep evidence.
    """
    config = ctx.config
    target_dir = ctx.work.repo

    if config.cleanup is True:
        enabled = True
    elif config.cleanup is False:
        enabled = False
    else:
        enabled = resolve_run_context(ctx.run_context).confirm(
            safe_default=False,
            question="Cleanup review output after completion? [y/N]",
            default="n",
            console=console,
        )

    if not enabled:
        return
    review_output_path = review_output_path_for(
        target_dir,
        session=ctx.artifacts,
        allow_standalone=ctx.allow_standalone_artifacts,
    )
    if review_output_path.exists():
        review_output_path.unlink()
        print_success(console, f"Cleaned up {REVIEW_OUTPUT_FILE}")
