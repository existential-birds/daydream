
"""Fix fanout for review and fix phases."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio

import daydream.phases.fix as fixing
from daydream import agent, config as phase_config, git_ops, ui
from daydream.backends import (
    Backend,
    effective_fanout_concurrency,
)
from daydream.fanout import run_fanout
from daydream.file_group_budget import FileGroupBudget
from daydream.fix_footprint import AuthorizedFixFootprint
from daydream.fix_isolation import FixIsolationRound
from daydream.phases.fix import (
    _parse_test_map,
    _preflight_finding_file_refs,
    group_items_by_footprint,
)
from daydream.run_context import RunContext, bind_resolved_run_context, resolve_run_context
from daydream.trajectory import (
    DaydreamPhase,
    dispatch_scope,
    finish_partial_or_failed,
    get_current_recorder,
)
from daydream.workspace import WorkContext


@asynccontextmanager
async def _isolated_fix_fanout(
    work: WorkContext,
    footprint: AuthorizedFixFootprint,
    *,
    file_scope_issues: bool,
    auth: git_ops.GitHubAuth,
) -> AsyncIterator[FixIsolationRound]:
    """Join isolated writers, restore direct parent writes, then import fixes."""
    isolation = FixIsolationRound(work, footprint, capture_discarded_edits=file_scope_issues)
    try:
        try:
            yield isolation
        except BaseException as fanout_error:
            try:
                isolation.restore_parent()
                isolation.publish()
            except BaseException as restore_error:
                raise BaseExceptionGroup(
                    "fix fan-out failed and parent restoration also failed",
                    [fanout_error, restore_error],
                ) from None
            raise
        else:
            isolation.restore_parent()
            isolation.publish()
            if file_scope_issues:
                from daydream.deep.scope_issues import _file_reverted_edit_issue

                for path, patch in isolation.discarded_edits:
                    _file_reverted_edit_issue(work.repo, path, patch, auth=auth)
    finally:
        isolation.close()


@bind_resolved_run_context
async def phase_fix_parallel(
    backend: Backend,
    work: WorkContext,
    items: list[dict[str, Any]],
    *,
    footprint: AuthorizedFixFootprint | None = None,
    file_scope_issues: bool = False,
    auth: git_ops.GitHubAuth = git_ops.INHERIT_GITHUB_AUTH,
    round_snapshot: git_ops.WorktreeRollbackSnapshot | None = None,
    limiter_size: int = 10,
    intent_path: Path | None = None,
    group_max_wall_s: float = phase_config.DEFAULT_GROUP_MAX_WALL_S,
    group_max_serial_items: int = phase_config.DEFAULT_GROUP_MAX_SERIAL_ITEMS,
    retry_recovery_allowance_s: float | None = None,
    exploration_dir: Path | None = None,
    test_map_path: Path | None = None,
    run_context: RunContext | None = None,
) -> dict[str, str]:
    """Fix authorized footprint groups concurrently, preserving caller severity order.

    Each group tries one batch turn, then individual findings on batch failure.
    Prompts receive group edit scope and run context. Backend concurrency, group
    wall time, serial-call count, and retry recovery allowance bound dispatch;
    intent, exploration, and parsed test-map hints are forwarded to each turn.

    Return file-to-reason failures. Exceptions and interrupted turns restore the
    group snapshot; budget stops before dispatch preserve completed edits. All
    budget failures use file_group_budget_exceeded; callers must not revert again.
    """
    run_context = resolve_run_context(run_context)
    if footprint is None or round_snapshot is None:
        raise TypeError("phase_fix_parallel requires footprint and round_snapshot")
    # Validate every reference before grouping, progress, prompts, or recovery.
    # UnconfinedFindingError aborts before an unsafe path becomes a key, fork,
    # failure entry, or checkout argument. Per-fix calls re-resolve their refs.
    _preflight_finding_file_refs(work.repo, items)

    raw_groups = group_items_by_footprint(items, footprint)
    # Pair stable display counters with items instead of relying on object IDs.
    counter = 0
    groups_numbered: list[tuple[str, list[tuple[dict[str, Any], int]]]] = []
    for file_key, group_items in raw_groups:
        numbered: list[tuple[dict[str, Any], int]] = []
        for item in group_items:
            counter += 1
            numbered.append((item, counter))
        groups_numbered.append((file_key, numbered))

    recorder = get_current_recorder()
    failures: dict[str, str] = {}
    successful_groups: set[str] = set()
    _failures_lock = anyio.Lock()
    limiter = anyio.CapacityLimiter(
        effective_fanout_concurrency(limiter_size, backend)
    )
    _console_lock = anyio.Lock()
    total = len(items)
    # Normalize the test map once for every group to share.
    test_map = _parse_test_map(test_map_path, work.repo)

    async def _record_budget_stop(
        fkey: str,
        reason: str,
        grp_len: int,
        budget: FileGroupBudget,
    ) -> None:
        """Record a group budget stop: trajectory event (with group elapsed),
        failures entry, and warning."""
        processed = budget.items_processed
        skipped = grp_len - processed
        if recorder is not None:
            recorder.emit_file_group_budget_exceeded(
                file=fkey, reason=reason,
                items_processed=processed, items_skipped=skipped,
                elapsed_s=budget.elapsed_s(),
            )
        async with _failures_lock:
            failures[fkey] = f"file_group_budget_exceeded: {reason}"
        async with _console_lock:
            ui.print_warning(
                agent.console,
                f"File group '{fkey}' budget exceeded ({reason}) after {processed} "
                f"item(s); skipping remaining {skipped}.",
            )

    def _restore_group_or_raise(
        context: str, paths: frozenset[str], group_work: WorkContext,
        group_snapshot: git_ops.WorktreeRollbackSnapshot,
    ) -> None:
        """Restore the complete group from the round snapshot, or fail the group."""
        try:
            isolation.audit_group(group_work.repo, group_snapshot)
            git_ops.restore_group_worktree_from_snapshot(group_work.repo, group_snapshot, paths)
        except Exception as restore_err:  # noqa: BLE001
            raise RuntimeError(
                f"failed to restore the complete fix group {context}"
            ) from restore_err

    async def _fix_group_serially(
        fkey: str,
        grp: list[tuple[dict[str, Any], int]],
        budget: FileGroupBudget,
        edit_scope: frozenset[str],
        group_work: WorkContext,
        group_snapshot: git_ops.WorktreeRollbackSnapshot,
    ) -> None:
        """Apply findings serially within the remaining group deadline and call count.

        Check before dispatch and bound each turn mid-call. A mid-call wall stop
        restores the entire group snapshot; a pre-dispatch stop preserves completed
        edits. Completed calls consume one slot. A turn's own wall limit is absorbed
        when the larger group budget still has time.
        """
        for item, item_num in grp:
            budget_reason = budget.check()
            if budget_reason is not None:
                await _record_budget_stop(fkey, budget_reason, len(grp), budget)
                return
            turn_reason = await fixing.phase_fix(
                backend, group_work, item, item_num, total,
                edit_scope=edit_scope,
                read_scope=footprint.run_allowed_paths,
                console_lock=_console_lock,
                intent_path=intent_path,
                exploration_dir=exploration_dir,
                test_map=test_map,
                run_context=run_context,
                deadline=budget.deadline,
                retry_recovery_allowance_s=retry_recovery_allowance_s,
            )
            if turn_reason == "wall_budget_exceeded":
                _restore_group_or_raise("after its turn wall budget cut", edit_scope, group_work, group_snapshot)
                if group_max_wall_s > phase_config.DEFAULT_WALL_BUDGET_S and budget.remaining() > 0:
                    # The shorter per-turn limit fired while the group still has time.
                    # Charge the attempted slot and continue to its remaining findings.
                    budget.record_item()
                    successful_groups.add(fkey)
                    continue
                # The group's own wall ceiling ended the turn mid-call. Do NOT
                # record the item as processed: the turn did not complete.
                await _record_budget_stop(fkey, "group_wall_budget_exceeded", len(grp), budget)
                return
            # Supervisor vetoes are policy outcomes already recorded in the trajectory.
            # Charge their slots without starving remaining findings.
            budget.record_item()
            successful_groups.add(fkey)

    dispatch_descriptors = tuple(
        "fix-" + file_key.replace("/", "-").replace("\\", "-")
        for file_key, _ in groups_numbered
    )
    async with dispatch_scope(
        recorder,
        phase=DaydreamPhase.FIX,
        descriptors=dispatch_descriptors,
    ) as dispatch:
        async with _isolated_fix_fanout(
            work, footprint, file_scope_issues=file_scope_issues, auth=auth,
        ) as isolation:
            # Complete host checkout preparation before starting any backend.
            # Synchronous cloning must not block a live sibling or consume its
            # cumulative model-call wall budget.
            group_workspaces = {
                file_key: isolation.create_group(footprint.group_paths([item for item, _ in numbered]))
                for file_key, numbered in groups_numbered
            }
            async def fix_group(group: tuple[str, list[tuple[dict[str, Any], int]]]) -> None:
                fkey, grp = group
                group_work, group_snapshot = group_workspaces[fkey]
                budget = FileGroupBudget(
                    max_wall_seconds=group_max_wall_s,
                    max_serial_items=group_max_serial_items,
                )
                try:
                    grp_items = [item for item, _ in grp]
                    grp_nums = [num for _, num in grp]
                    edit_scope = footprint.group_paths(grp_items)
                    is_real_batch = len(grp_items) > 1 and fkey != "<no-file>"
                    if not is_real_batch:
                        # Single/no-file groups dispatch directly without a batched fallback.
                        await _fix_group_serially(fkey, grp, budget, edit_scope, group_work, group_snapshot)
                    else:
                        # Check before batch dispatch so zero/tiny budgets skip the call.
                        pre_batch_reason = budget.check()
                        if pre_batch_reason is not None:
                            await _record_budget_stop(fkey, pre_batch_reason, len(grp), budget)
                            return
                        try:
                            await fixing.phase_fix_batched(
                                backend, group_work, grp_items, grp_nums, total,
                                edit_scope=edit_scope,
                                read_scope=footprint.run_allowed_paths,
                                console_lock=_console_lock,
                                intent_path=intent_path,
                                exploration_dir=exploration_dir,
                                test_map=test_map,
                                run_context=run_context,
                                deadline=budget.deadline,
                                retry_recovery_allowance_s=retry_recovery_allowance_s,
                            )
                            budget.record_item()
                            successful_groups.add(fkey)
                        except Exception:  # noqa: BLE001 -- batched failure falls back to per-finding fixes
                            # Restore the group snapshot so fallback cannot reapply partial batch edits.
                            _restore_group_or_raise(
                                "before fallback", edit_scope, group_work, group_snapshot,
                            )
                            await _fix_group_serially(
                                fkey, grp, budget, edit_scope, group_work, group_snapshot,
                            )
                    if fkey in successful_groups:
                        isolation.retain_group(group_work.repo, edit_scope, group_snapshot)
                except Exception as e:  # noqa: BLE001 -- intentionally broad for parallel isolation
                    # Recovery restores the complete group, so earlier
                    # serial progress cannot justify a partial status.
                    successful_groups.discard(fkey)
                    failure: BaseException = e
                    try:
                        grp_items = [item for item, _ in grp]
                        isolation.audit_group(group_work.repo, group_snapshot)
                        git_ops.restore_group_worktree_from_snapshot(
                            group_work.repo,
                            group_snapshot,
                            footprint.group_paths(grp_items),
                        )
                    except Exception as restore_err:  # noqa: BLE001
                        failure = RuntimeError(
                            "failed to restore the complete failed fix group"
                        )
                        failure.__cause__ = restore_err
                    reason = f"{type(failure).__name__}: {failure}"
                    async with _failures_lock:
                        failures[fkey] = reason
                    async with _console_lock:
                        ui.print_warning(
                            agent.console,
                            f"Fixes for '{fkey}' failed ({reason}); its complete group was "
                            "restored and successful sibling-group edits were preserved.",
                        )


            await run_fanout(
                groups_numbered, fix_group, limiter=limiter, recorder=recorder, dispatch=dispatch,
                descriptor=lambda group: "fix-" + group[0].replace("/", "-").replace("\\", "-"),
            )
        if dispatch is not None and failures:
            finish_partial_or_failed(dispatch, successful_groups)

    return failures
