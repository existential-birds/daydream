"""Arbitrate and suppress findings with stable identities and resumable group evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anyio

from daydream.agent import console
from daydream.backends import Backend, effective_fanout_concurrency
from daydream.deep.adjudication_provenance import (
    RecordProvenance,
    find_revision_delta,
    record_provenance,
)
from daydream.deep.arbiter import (
    ArbiterGroup,
    contested_indices,
    partition_arbiter_targets,
    select_arbiter_targets,
    select_suppression_targets,
)
from daydream.deep.artifacts import (
    DeepArtifact,
    arbiter_group_complete_path,
    arbiter_group_input_path,
    arbiter_group_verdicts_path,
    per_stack_records_path,
    review_stage,
)
from daydream.deep.latency import (
    FAIL_SAFE_LATENCY_PROFILE,
    PROFILE_ROUTES,
    ArbiterPlan,
    PlannedGroup,
    arbiter_plan,
)
from daydream.deep.records import (
    partition_record_sources,
    record_uid,
    stack_name_from_records_source,
    stack_name_from_uid,
)
from daydream.deep.reuse_key import (
    arbiter_key_payload,
    phase_identity_for,
)
from daydream.deep.reuse_store import (
    reuse_cache_for,
)
from daydream.deep.review_reuse import ReviewReuseUnit, _loop_grounding, _records_bytes_by_basename
from daydream.deep.routing_record import write_routing_record
from daydream.deep.settings import _resolve_opt_in
from daydream.deep.state import DeepState
from daydream.flows.engine import FlowContext
from daydream.json_utils import read_json_object
from daydream.phases import (
    phase_arbiter_review,
    phase_suppression_review,
)
from daydream.phases.adjudication import IncompleteVerdicts
from daydream.phases.review import valid_record_artifact
from daydream.review_result import ReasonCode, ReviewCoverage, reason_for_budget, reason_for_exception
from daydream.supervision import revise_finding_fields
from daydream.trajectory import (
    DaydreamPhase,
    dispatch_scope,
    get_current_recorder,
    maybe_fork,
    phase_scope,
)
from daydream.ui import print_warning


def _apply_adjudication_verdicts(
    records: list[dict[str, Any]],
    sources: list[str],
    targets: list[int],
    verdicts: dict[int, dict[str, Any]],
    *,
    pass_name: str,
    id_field: str,
    fail_closed: bool,
) -> tuple[list[dict[str, Any]], list[str], list[RecordProvenance]]:
    """Apply positional agent verdicts using target UIDs snapshotted before mutation.

    ``targets[k]`` corresponds to agent id k+1 in ``id_field``. Revise bound
    findings without changing file/line; explicit keep:false drops them.
    Missing UIDs/verdicts or mismatched ids warn: arbiter retains the original
    (fail_closed=False), suppression drops it (True). UID-less drops use their
    original positions. Unselected records pass through unchanged.

    Return aligned records/sources plus per-target provenance recording the
    original UID, binding, retention and materially revised fields.
    """
    import warnings

    outcomes: list[RecordProvenance] = []

    def _record_outcome(
        uid: str,
        *,
        verdict_bound: bool,
        kept: bool = True,
        revised_fields: tuple[str, ...] = (),
    ) -> None:
        outcomes.append(
            RecordProvenance(uid, (pass_name,), verdict_bound, kept, revised_fields)
        )

    polarity_action = (
        "dropping the unconfirmed record"
        if fail_closed
        else "retaining the original record unchanged"
    )
    # Snapshot ``offset -> uid`` BEFORE anything mutates or drops a record
    # (issue #1111). This loop is where ``targets``' indices stop being
    # load-bearing: every host-side decision after it keys on the uid.
    target_uids: dict[int, str] = {
        offset: record_uid(records[record_index]) for offset, record_index in enumerate(targets)
    }
    # Apply revisions by durable UID; selection indices may shift after compaction.
    by_uid: dict[str, dict[str, Any]] = {}
    for record in records:
        record_key = record_uid(record)
        if record_key:
            by_uid[record_key] = record
    dropped: set[str] = set()
    # Unidentified targeted records still honor fail polarity through positional tracking.
    dropped_positions: set[int] = set()

    def _warn_unconfirmable(message: str, *, record_index: int, uid: str) -> None:
        """Warn and apply the configured polarity, using position only when the UID is absent."""
        warnings.warn(message, stacklevel=3)
        _record_outcome(uid, verdict_bound=False, kept=not fail_closed)
        if fail_closed:
            if uid:
                dropped.add(uid)
            else:
                dropped_positions.add(record_index)

    for offset, record_index in enumerate(targets):
        verdict_id = offset + 1
        uid = target_uids[offset]
        if not uid:
            # Broken invariant, reported loudly and failed per the caller's
            # polarity like every other unconfirmable branch here.
            _warn_unconfirmable(
                f"{pass_name.capitalize()} target {id_field}={verdict_id} "
                f"(record_index={record_index}) carries no uid, so its verdict cannot be bound "
                f"to a record identity; {polarity_action} (issue #1111).",
                record_index=record_index,
                uid="",
            )
            continue
        verdict = verdicts.get(verdict_id)
        if verdict is None:
            # No verdict returned for this id -- fail per the caller's polarity.
            _warn_unconfirmable(
                f"{pass_name.capitalize()} returned no verdict for {id_field}={verdict_id} "
                f"(record_index={record_index}, uid={uid}); {polarity_action}.",
                record_index=record_index,
                uid=uid,
            )
            continue
        if verdict.get(id_field) != verdict_id:
            # Secondary key guard: the id field in the verdict must match the key
            # we looked it up by. A mismatch would silently bind the verdict to the
            # wrong record -- fail per the caller's polarity rather than mis-apply.
            _warn_unconfirmable(
                f"{pass_name.capitalize()} verdict {id_field} mismatch: "
                f"expected {id_field}={verdict_id} "
                f"but verdict contains {id_field}={verdict.get(id_field)!r} "
                f"(record_index={record_index}, uid={uid}); {polarity_action}.",
                record_index=record_index,
                uid=uid,
            )
            continue
        if not verdict.get("keep", False):
            _record_outcome(uid, verdict_bound=True, kept=False)
            dropped.add(uid)
            continue
        # Revise IN PLACE rather than rebuilding the dict: the caller holds this
        # same list and ``_rewrite_stack_records`` persists these very dicts, so
        # a fresh dict would have to be threaded back into both.
        before = dict(by_uid[uid])
        revise_finding_fields(by_uid[uid], verdict)
        _record_outcome(
            uid,
            verdict_bound=True,
            revised_fields=find_revision_delta(before, by_uid[uid]),
        )

    new_records: list[dict[str, Any]] = []
    new_sources: list[str] = []
    for i, (record, source) in enumerate(zip(records, sources, strict=True)):
        # Dropped by uid; the positional set covers only the unidentifiable
        # records warned about above.
        if record_uid(record) in dropped or i in dropped_positions:
            continue
        new_records.append(record)
        new_sources.append(source)
    return new_records, new_sources, outcomes


def _rewrite_stack_records(
    deep_dir_path: Path,
    stack_record_paths: list[Path],
    records: list[dict[str, Any]],
    sources: list[str],
) -> None:
    """Persist revised records to every stack file, including empty issue lists.

    Route by required durable UID stack. Warn on records outside the supplied
    paths; merge later reads these files.
    """
    by_stack: dict[Path, list[dict[str, Any]]] = {path: [] for path in stack_record_paths}
    for record, source in zip(records, sources, strict=True):
        uid = record_uid(record)
        if not uid:
            raise ValueError("Adjudicated records require a durable UID")
        dest = per_stack_records_path(deep_dir_path, stack_name_from_uid(uid))
        if dest in by_stack:
            by_stack[dest].append(record)
        else:
            # A record routing outside ``stack_record_paths`` would be silently
            # ERASED (the file is rewritten wholesale), so warn loudly (issue
            # #1111). Warn rather than raise: every other verdict is already
            # computed and belongs on disk.
            print_warning(
                console,
                f"Adjudicated record uid={record_uid(record) or '<none>'} (source {source}) "
                f"routes to {dest.name}, which is not among the records files being rewritten "
                f"({', '.join(sorted(path.name for path in stack_record_paths))}); "
                "its adjudication will not reach disk (issue #1111).",
            )
    for dest_path, stack_records in by_stack.items():
        previous = read_json_object(dest_path)
        incomplete = previous.get("incomplete") is True
        keys = ("scope_id", "analyzed_revision", "originating_run_id")
        if any(key not in previous for key in keys):
            raise ValueError("Adjudicated records require persisted scope/revision/run identity")
        binding = {key: previous[key] for key in keys}
        dest_path.write_text(
            json.dumps({"issues": stack_records, **binding,
                        **({"incomplete": True} if incomplete else {})}, indent=2)
        )


def _rejoin_structural_records(
    all_records: list[dict[str, Any]],
    record_sources: list[str],
    structural_records: list[dict[str, Any]],
    structural_sources: list[str],
    records_paths: list[Path],
    structural_path: Path | None,
) -> tuple[list[dict[str, Any]], list[str], set[str], list[Path], range]:
    """Combine structural and language records for contested-location adjudication.

    Return aligned records/sources, structural UIDs, all rewrite paths and the
    structural index range. The range includes UID-less structural records and
    must be used for contested_only before reordering. The UID set survives
    rebuilds and drives the later split; the structural path ensures revisions
    also reach the file merge reads.
    """
    # An unidentified structural record cannot be re-partitioned reliably.
    # Warn about degraded routing while preserving adjudication results.
    structural_ids = {uid for rec in structural_records if (uid := record_uid(rec))}
    unidentified = len(structural_records) - len(structural_ids)
    if unidentified:
        print_warning(
            console,
            f"{unidentified} structural record(s) carry no uid; they will be adjudicated but "
            "rejoin the language-stack pool after arbitration instead of the structural pool "
            "(issue #1111).",
        )
    adjudicated = all_records + structural_records
    adjudicated_sources = record_sources + structural_sources
    rewrite_paths = list(records_paths)
    if structural_path is not None:
        rewrite_paths.append(structural_path)
    structural_range = range(len(all_records), len(adjudicated))
    return adjudicated, adjudicated_sources, structural_ids, rewrite_paths, structural_range




def _load_group_verdicts(dd: Path, group: PlannedGroup) -> dict[int, dict[str, Any]] | None:
    """Load a completed group's persisted verdicts, or ``None`` when it must rerun.

    A group is reusable only when its completion marker *and* its verdicts file
    exist and the file's persisted ``target_uids`` still match the planned
    group. A membership mismatch means the diff/selection moved under the saved
    group, so reusing it would bind verdicts to the wrong records -- rerun.
    """
    marker = arbiter_group_complete_path(dd, group.group_id)
    verdicts_path = arbiter_group_verdicts_path(dd, group.group_id)
    if not marker.is_file() or not verdicts_path.is_file():
        return None
    try:
        payload = json.loads(verdicts_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    if list(payload.get("target_uids", [])) != list(group.target_uids):
        return None
    raw = payload.get("verdicts")
    if not isinstance(raw, dict):
        return None
    loaded: dict[int, dict[str, Any]] = {}
    for key, verdict in raw.items():
        if not isinstance(verdict, dict):
            return None
        try:
            loaded[int(key)] = verdict
        except (TypeError, ValueError):
            return None
    return loaded


def _persist_group_verdicts(
    dd: Path, group: PlannedGroup, verdicts: dict[int, dict[str, Any]]
) -> None:
    """Persist one group's verdicts, then its completion marker.

    The marker is written last so a crash between the two leaves a group that a
    resume will rerun rather than reuse half-written verdicts.
    """
    payload = {
        "group_id": group.group_id,
        "target_uids": list(group.target_uids),
        "verdicts": {str(key): verdict for key, verdict in sorted(verdicts.items())},
    }
    arbiter_group_verdicts_path(dd, group.group_id).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    arbiter_group_complete_path(dd, group.group_id).write_text("")


def _merge_group_verdicts(
    plan: ArbiterPlan,
    arbiter_targets: list[int],
    group_verdicts: dict[str, dict[int, dict[str, Any]]],
    targets_by_group: dict[str, list[int]],
) -> dict[int, dict[str, Any]]:
    """Translate group-local arb_id values into run-wide selection ordinals.

    Use each group's original target indices so completion order and later
    record compaction cannot change the binding.
    """
    positions = {index: offset + 1 for offset, index in enumerate(arbiter_targets)}
    merged: dict[int, dict[str, Any]] = {}
    for group in plan.groups:
        target_indices = targets_by_group.get(group.group_id, ())
        for local_id, verdict in group_verdicts.get(group.group_id, {}).items():
            if local_id < 1 or local_id > len(target_indices):
                continue
            global_index = target_indices[local_id - 1]
            position = positions.get(global_index)
            if position is None:
                continue
            revised = dict(verdict)
            revised["arb_id"] = position
            merged[position] = revised
    return merged


def _arbiter_groups_record(
    plan: ArbiterPlan,
    *,
    effort_pin: str | None,
    reused: dict[str, bool],
) -> list[dict[str, Any]]:
    """Serialize the plan's groups for the routing record.

    An explicit effort pin (A4) replaces every group's planned effort and says
    so in the reason, so the record states when a conscious user knob overrode
    the route rather than silently reporting a route the run did not take.
    """
    records: list[dict[str, Any]] = []
    for group in plan.groups:
        if effort_pin is not None:
            effort = effort_pin
            reason = f"explicit {effort_pin} effort pin overrode the route (A4)"
        else:
            effort = group.effort
            reason = group.reason
        records.append(
            {
                "group_id": group.group_id,
                "target_uids": list(group.target_uids),
                "effort": effort,
                "reason": reason,
                "reused": reused.get(group.group_id, False),
            }
        )
    return records


def _review_context_kwargs(ctx: FlowContext, deep_state: DeepState, *, strategy: str) -> dict[str, Any]:
    """Structured-review context shared by the arbiter and suppression calls."""
    return {
        "diff_path": deep_state.diff_path,
        "intent_path": deep_state.intent_path,
        "alternatives_path": deep_state.alts_path,
        "exploration_dir": deep_state.exploration_dir,
        "strategy": strategy,
        "run_context": ctx.run_context,
        "artifact_session": ctx.artifacts,
        "allow_standalone": ctx.allow_standalone_artifacts,
    }


def _record_verdict_coverage(coverage: ReviewCoverage, phase: str, target_count: int,
                             verdicts: dict[int, dict[str, Any]], *, complete: bool = True) -> bool:
    """All adjudication stages require one validated verdict for every target."""
    reason = (reason_for_budget(verdicts.budget_reason) if isinstance(verdicts, IncompleteVerdicts)
              else ReasonCode.EVIDENCE_INCOMPLETE
              if not complete or set(verdicts) != set(range(1, target_count + 1)) else None)
    coverage.record_phase(phase, "incomplete" if reason else "complete",
                          reasons=(reason,) if reason else (), noop=target_count == 0 and reason is None)
    return reason is None


async def _run_sharded_arbiter(
    ctx: FlowContext,
    deep_state: DeepState,
    plan: ArbiterPlan,
    arbiter_targets: list[int],
    adjudicated: list[dict[str, Any]],
    *,
    effort_pin: str | None,
    targets_by_group: dict[str, list[int]],
) -> tuple[dict[int, dict[str, Any]], dict[str, bool], list[str]]:
    """Fan one arbiter call per incomplete group out under the run's ceiling.

    Returns the merged verdict mapping (keyed by run-wide target ordinal), the
    per-group ``reused`` flags, and the ids of groups whose call raised. A
    group's exception is caught here rather than propagated: the run continues,
    the group contributes no verdicts, and its targets retain their original
    records (``_apply_adjudication_verdicts(..., fail_closed=False)``).
    """
    dd = deep_state.dd
    group_verdicts: dict[str, dict[int, dict[str, Any]]] = {}
    reused: dict[str, bool] = {}
    failed_groups: list[str] = []
    pending: list[PlannedGroup] = []
    for group in plan.groups:
        deep_state.review_coverage.require_phase(group.group_id)
        loaded = _load_group_verdicts(dd, group)
        if loaded is None:
            pending.append(group)
        else:
            group_verdicts[group.group_id] = loaded
            reused[group.group_id] = True
            prior = deep_state.review_coverage.phases.get(group.group_id)
            if prior is None or prior["status"] != "complete":
                pending.append(group)
                group_verdicts.pop(group.group_id, None)
                reused.pop(group.group_id, None)

    if pending:
        recorder = get_current_recorder()
        arbiter_backend = ctx.backend_for("arbiter")
        limiter = anyio.CapacityLimiter(effective_fanout_concurrency(10, arbiter_backend))
        descriptors = tuple(f"deep-{group.group_id}" for group in pending)
        async with phase_scope(DaydreamPhase.DEEP, stage="arbiter"):
            async with dispatch_scope(
                recorder, phase=DaydreamPhase.DEEP, descriptors=descriptors
            ) as dispatch:
                async with anyio.create_task_group() as tg:
                    for group in pending:

                        async def _arbitrate_one(
                            planned: PlannedGroup = group,
                        ) -> None:
                            async with limiter:
                                try:
                                    async with maybe_fork(
                                        recorder,
                                        f"deep-{planned.group_id}",
                                        dispatch=dispatch,
                                    ):
                                        call_backend = (
                                            ctx.backend_for("arbiter")
                                            if effort_pin is not None
                                            else ctx.backend_for_effort("arbiter", planned.effort)
                                        )
                                        group_verdicts_call, _continuation = await phase_arbiter_review(
                                            call_backend,
                                            ctx.work,
                                            selected_records=[
                                                adjudicated[i]
                                                for i in targets_by_group.get(planned.group_id, ())
                                            ],
                                            input_path=arbiter_group_input_path(
                                                dd, planned.group_id
                                            ),
                                            **_review_context_kwargs(
                                                ctx, deep_state, strategy=ctx.strategy("arbitration")
                                            ),
                                            intent_authoritative=deep_state.intent_authoritative,
                                        )
                                except Exception as exc:  # noqa: BLE001 -- per-group isolation; fail-open
                                    failed_groups.append(planned.group_id)
                                    deep_state.review_coverage.record_phase(planned.group_id, "failed",
                                                                           reasons=(reason_for_exception(exc),))
                                    reused[planned.group_id] = False
                                    print_warning(
                                        console,
                                        f"Arbiter group {planned.group_id} failed "
                                        f"({type(exc).__name__}: {exc}); its findings "
                                        "remain unadjudicated.",
                                    )
                                    return
                                if not _record_verdict_coverage(deep_state.review_coverage, planned.group_id,
                                        len(targets_by_group[planned.group_id]), group_verdicts_call):
                                    failed_groups.append(planned.group_id)
                                    if not isinstance(group_verdicts_call, IncompleteVerdicts):
                                        group_verdicts[planned.group_id] = group_verdicts_call
                                    return
                                try:
                                    _persist_group_verdicts(dd, planned, group_verdicts_call)
                                except (OSError, ValueError) as exc:
                                    failed_groups.append(planned.group_id)
                                    deep_state.review_coverage.record_phase(planned.group_id, "failed",
                                                                           reasons=(ReasonCode.MALFORMED_ARTIFACT,))
                                    print_warning(console,
                                                  f"Arbiter group {planned.group_id} could not persist its verdicts "
                                                  f"({type(exc).__name__}); its findings remain unadjudicated.")
                                    return
                                group_verdicts[planned.group_id] = group_verdicts_call
                                reused[planned.group_id] = False

                        tg.start_soon(_arbitrate_one)

    return (
        _merge_group_verdicts(plan, arbiter_targets, group_verdicts, targets_by_group),
        reused,
        failed_groups,
    )


def _unsharded_arbiter_backend(ctx: FlowContext, *, effort_pin: str | None) -> Backend:
    """Use an explicit effort pin, otherwise xhigh, for every unsharded selection.

    Sharded route effort does not apply here. Backend resolution retains ambient
    defaults for backends excluded from the deep effort table.
    """
    if effort_pin is not None:
        return ctx.backend_for("arbiter")
    return ctx.backend_for_effort("arbiter", "xhigh")


def _arbiter_plan_component(plan: ArbiterPlan) -> dict[str, Any]:
    """The keying view of the arbiter plan: sharded flag + each group's target uids."""
    return {
        "sharded": plan.sharded,
        "groups": [list(group.target_uids) for group in plan.groups],
    }


def _arbiter_store_payload(
    dd: Path, rewrite_paths: list[Path], plan: ArbiterPlan
) -> dict[str, bytes] | None:
    """Collect a completed arbiter's on-disk outputs, or ``None`` if any is missing.

    The unit is storable only as a whole: every rewritten records file, each
    planned group's verdicts + marker when the plan sharded (the unsharded path
    persists no group files), and the whole-block ``arbiter-complete`` marker.
    A missing piece leaves no entry rather than a partial one.
    """
    paths = list(rewrite_paths)
    if plan.sharded:
        for group in plan.groups:
            paths.extend((
                arbiter_group_verdicts_path(dd, group.group_id),
                arbiter_group_complete_path(dd, group.group_id),
            ))
    paths.append(DeepArtifact.ADJUDICATION_COMPLETE.at(dd))
    try:
        return {path.name: path.read_bytes() for path in paths}
    except OSError:
        return None


def _reload_adjudicated_records(deep_state: DeepState) -> bool:
    """Refresh ``ctx.data`` from the restored post-arbitration records files.

    A whole-unit hit must present exactly what a real adjudication left behind,
    not the pre-arbitration in-memory copy, so dedup and merge see the revised
    records. Returns ``False`` (leaving the caller's state untouched) when any
    file cannot be parsed, which makes the hit a miss instead of a split brain.
    """
    structural_path = deep_state.structural_records_path_or_none
    groups: list[tuple[list[dict[str, Any]], list[str]]] = []
    for paths in (deep_state.records_paths, [structural_path] if structural_path is not None else []):
        records: list[dict[str, Any]] = []
        sources: list[str] = []
        for path in paths:
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return False
            if not valid_record_artifact(loaded, scope_id=stack_name_from_records_source(path.name),
                                         analyzed_revision=deep_state.review_coverage.revision.to_dict()):
                return False
            rows = loaded["issues"]
            records.extend(rows)
            sources.extend(path.name for _ in rows)
        groups.append((records, sources))
    # Publish only after every file loaded, retaining the old state on a miss.
    deep_state.records, deep_state.record_sources = groups[0]
    deep_state.structural_records, deep_state.structural_record_sources = groups[1]
    return True


def _try_reuse_arbiter(
    unit: ReviewReuseUnit,
    deep_state: DeepState,
    plan: ArbiterPlan,
) -> bool:
    """Try to restore a completed whole-arbiter unit; ``True`` when it hit.

    A hit restores every rewritten records file, the per-group verdicts and
    markers (a sharded payload), and the whole-block marker, then reloads
    ``ctx.data`` from those files and records the hit with its grounding delta.
    Any failure is recorded as a miss and the caller runs the real adjudication.
    """
    hit = unit.lookup(deep_state.dd)
    if hit is None:
        return False
    if not _reload_adjudicated_records(deep_state):
        unit.cache.record("arbiter", outcome="miss", reason="restored records unreadable", key=unit.key)
        return False
    unit.record_hit(hit)
    write_routing_record(
        deep_state.dd,
        {
            "arbiter": {
                "sharded": plan.sharded,
                "reason": f"reused whole unit ({plan.reason})",
                "groups": _arbiter_groups_record(
                    plan,
                    effort_pin=None,
                    reused={group.group_id: True for group in plan.groups},
                ),
                "verdicts_applied": 0,
                "failed_groups": [],
            }
        },
    )
    return True


async def _step_arbiter(ctx: FlowContext) -> None:
    """Scoped arbiter over high-severity/contested findings (#168).

    Two shapes: the unsharded path (forensic, or a selection that fits one
    co-located group) is today's single serial call at ``xhigh``; the sharded
    path fans one call per planned group out under the run's fan-out ceiling,
    applies one merged verdict mapping, and persists per-group artifacts so a
    resume reruns only the incomplete groups.
    """
    deep_state = DeepState(ctx.data)
    if ctx.pipeline().arbitration.enabled:
        deep_state.review_coverage.require_phase("arbiter")
    with review_stage(deep_state, lambda: ("suppression" if "suppression" in deep_state.review_coverage.phases
        and deep_state.review_coverage.phases["suppression"]["status"] == "uncovered" else "arbiter"), persist=True):
        config = ctx.config
        dd = deep_state.dd
        all_records: list[dict[str, Any]] = deep_state.records
        record_sources: list[str] = deep_state.record_sources
        # Adjudicate language and structural findings together, then restore their
        # partition for dedup and merge.
        structural_records: list[dict[str, Any]] = deep_state.structural_records
        structural_sources: list[str] = deep_state.structural_record_sources

        # A merge resume skips adjudication only with a whole-pass completion marker.
        adjudication_marker = DeepArtifact.ADJUDICATION_COMPLETE.at(dd)
        if (
            ctx.pipeline().arbitration.enabled
            and (config.start_at != "merge" or not adjudication_marker.is_file())
        ):
            structural_path: Path | None = deep_state.structural_records_path_or_none
            adjudicated, adjudicated_sources, structural_ids, rewrite_paths, structural_range = (
                _rejoin_structural_records(
                    all_records, record_sources, structural_records, structural_sources,
                    deep_state.records_paths, structural_path,
                )
            )

            arbiter_targets = select_arbiter_targets(
                adjudicated, adjudicated_sources,
                min_severity=ctx.pipeline().arbitration.min_severity,
                contested_location=ctx.pipeline().arbitration.contested_location,
                contested_only=structural_range,
            )
            # Suppression exclusions use durable UIDs; indices shift and locations can collide.
            # Unidentified records are handled explicitly at the exclusion site.
            arbitrated_ids = {uid for i in arbiter_targets if (uid := record_uid(adjudicated[i]))}
            arbiter_slice: dict[str, Any] = {
                "sharded": False,
                "reason": "no arbiter targets selected",
                "groups": [],
                "verdicts_applied": 0,
                "failed_groups": [],
            }
            adjudication_complete = True
            # The resolved suppression opt-in is part of the arbiter unit's contract
            # (MH8): a payload stored with it off must not be served to a run with it
            # on, so it is keyed, not merely recorded.
            precision_mode = bool(
                ctx.pipeline().suppression.enabled or _resolve_opt_in(config, "precision_mode")
            )
            if precision_mode:
                deep_state.review_coverage.require_phase("suppression")
            # The reuse handles are populated only when there are arbiter targets;
            # the whole-unit store at the end of the block reads them back.
            arbiter_unit: ReviewReuseUnit | None = None
            plan: ArbiterPlan | None = None
            if arbiter_targets:
                route = deep_state.latency_route or PROFILE_ROUTES[FAIL_SAFE_LATENCY_PROFILE]
                # Only sharded routes partition; an unsharded route may have a zero group bound.
                groups = (
                    partition_arbiter_targets(
                        adjudicated,
                        arbiter_targets,
                        edges=deep_state.import_graph,
                        max_targets=route.group_max_targets,
                    )
                    if route.arbiter_sharded
                    else [
                        ArbiterGroup(
                            "arbiter-group-0",
                            tuple(arbiter_targets),
                            tuple(record_uid(adjudicated[i]) for i in arbiter_targets),
                        )
                    ]
                )
                contested = (
                    contested_indices(
                        adjudicated,
                        adjudicated_sources,
                        contested_only=structural_range,
                    )
                    if ctx.pipeline().arbitration.contested_location
                    else frozenset()
                )
                plan = arbiter_plan(route, groups, records=adjudicated, contested=contested)
                deep_state.arbiter_plan = plan
                targets_by_group = {
                    group.group_id: list(group.target_indices) for group in groups
                }
                from daydream.run_config import _explicit_reasoning_effort_pin

                effort_pin = _explicit_reasoning_effort_pin(config, "arbiter")
                # Capture the key before adjudication mutates records, then reuse or store under it.
                reuse = reuse_cache_for(ctx)
                arbiter_identity = phase_identity_for(ctx, "arbiter")
                if reuse is not None:
                    contributing = _records_bytes_by_basename(rewrite_paths)
                    arbiter_payload = arbiter_key_payload(
                        contributing_records=contributing,
                        structural_records=(
                            contributing.get(structural_path.name)
                            if structural_path is not None
                            else None
                        ),
                        plan=_arbiter_plan_component(plan),
                        precision_mode=precision_mode,
                        identity=arbiter_identity,
                        grounding=_loop_grounding(deep_state),
                    )
                    arbiter_unit = ReviewReuseUnit(reuse, "arbiter", arbiter_identity, arbiter_payload,
                                                   coverage=deep_state.review_coverage)
                    if _try_reuse_arbiter(arbiter_unit, deep_state, plan):
                        deep_state.review_coverage.record_phase("arbiter", "complete")
                        if precision_mode:
                            deep_state.review_coverage.record_phase("suppression", "complete")
                        return
                if plan.sharded:
                    verdicts, reused, failed_groups = await _run_sharded_arbiter(
                        ctx,
                        deep_state,
                        plan,
                        arbiter_targets,
                        adjudicated,
                        effort_pin=effort_pin,
                        targets_by_group=targets_by_group,
                    )
                    adjudication_complete = not failed_groups
                else:
                    failed_groups = []
                    reused = {plan.groups[0].group_id: False}
                    async with phase_scope(DaydreamPhase.DEEP, stage="arbiter"):
                        arbiter_backend = _unsharded_arbiter_backend(ctx, effort_pin=effort_pin)
                        verdicts, arbiter_continuation = await phase_arbiter_review(
                            arbiter_backend,
                            ctx.work,
                            selected_records=[adjudicated[i] for i in arbiter_targets],
                            **_review_context_kwargs(ctx, deep_state, strategy=ctx.strategy("arbitration")),
                            intent_authoritative=deep_state.intent_authoritative,
                        )
                        # Identity gate: only resume when merge runs on the very same
                        # backend instance. A per-phase override that resolves a
                        # different backend gets the cold path.
                        if arbiter_continuation is not None and arbiter_backend is ctx.backend_for("merge"):
                            deep_state.arbiter_continuation = arbiter_continuation
                adjudicated, adjudicated_sources, arbiter_outcomes = _apply_adjudication_verdicts(
                    adjudicated, adjudicated_sources, arbiter_targets, verdicts,
                    pass_name="arbiter",
                    id_field="arb_id",
                    fail_closed=False,
                )
                _rewrite_stack_records(
                    dd, rewrite_paths, adjudicated, adjudicated_sources
                )
                record_provenance(dd, pass_name="arbiter", outcomes=arbiter_outcomes)
                arbiter_slice = {
                    "sharded": plan.sharded,
                    "reason": plan.reason,
                    "groups": _arbiter_groups_record(plan, effort_pin=effort_pin, reused=reused),
                    "verdicts_applied": len(verdicts),
                    "failed_groups": list(failed_groups),
                }
            adjudication_complete = _record_verdict_coverage(
                deep_state.review_coverage, "arbiter", len(arbiter_targets),
                verdicts if arbiter_targets else {}, complete=adjudication_complete,
            )
            write_routing_record(dd, {"arbiter": arbiter_slice})

            # Precision-mode suppression pass (#232), OPT-IN: a skeptical second
            # opinion on borderline (LOW-confidence / low-severity uncontested)
            # findings, dropping any it cannot confirm (fail-CLOSED). Excludes the
            # arbiter's targets; one batched call via the cheaper `suppression` key.
            if precision_mode:
                # Exclude structural records (high-conviction by construction,
                # #1103) and any record with no uid: suppression is fail-CLOSED, so
                # unidentifiable records must be kept rather than droppable.
                suppression_exclude = [
                    i
                    for i, r in enumerate(adjudicated)
                    if not (uid := record_uid(r)) or uid in arbitrated_ids or uid in structural_ids
                ]
                suppression_targets = select_suppression_targets(
                    adjudicated,
                    suppression_exclude,
                    severity_classes=ctx.pipeline().suppression.severity_classes,
                    confidence_classes=ctx.pipeline().suppression.confidence_classes,
                )
                if suppression_targets:
                    async with phase_scope(DaydreamPhase.DEEP, stage="suppression"):
                        sup_verdicts = await phase_suppression_review(
                            ctx.backend_for("suppression"),
                            ctx.work,
                            selected_records=[adjudicated[i] for i in suppression_targets],
                            **_review_context_kwargs(ctx, deep_state, strategy=ctx.strategy("suppression")),
                        )
                    adjudicated, adjudicated_sources, suppression_outcomes = _apply_adjudication_verdicts(
                        adjudicated, adjudicated_sources, suppression_targets, sup_verdicts,
                        pass_name="suppression",
                        id_field="sup_id",
                        fail_closed=True,
                    )
                    _rewrite_stack_records(
                        dd, rewrite_paths, adjudicated, adjudicated_sources
                    )
                    record_provenance(dd, pass_name="suppression", outcomes=suppression_outcomes)
                adjudication_complete = _record_verdict_coverage(
                    deep_state.review_coverage, "suppression", len(suppression_targets),
                    sup_verdicts if suppression_targets else {},
                ) and adjudication_complete
            # Whole-block marker: written only when every planned group completed,
            # so an interrupted sharded fan-out forces the block to re-enter and
            # reruns only its incomplete groups (and the opt-in suppression pass).
            if adjudication_complete:
                adjudication_marker.write_text("")
                # Store only a completed whole-unit adjudication (every planned
                # group's files present) under the pre-dispatch key; a partial entry
                # must never be served as this unit's output.
                if arbiter_unit is not None and plan is not None:
                    arbiter_unit.store(lambda: _arbiter_store_payload(dd, rewrite_paths, plan))
            all_records, record_sources, structural_records, structural_sources = (
                partition_record_sources(adjudicated, adjudicated_sources, structural_ids)
            )
        deep_state.records = all_records
        deep_state.record_sources = record_sources
        deep_state.structural_records = structural_records
        deep_state.structural_record_sources = structural_sources
