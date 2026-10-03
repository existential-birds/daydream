"""Arbitrate and suppress findings with stable identities and resumable group evidence."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import anyio

from daydream.agent import _validates_schema, console
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
    record_uid,
)
from daydream.deep.reuse_key import (
    arbiter_key_payload,
    digest_or_absent,
    phase_identity_for,
)
from daydream.deep.reuse_store import (
    reuse_cache_for,
)
from daydream.deep.review_reuse import ReviewReuseUnit, _loop_grounding, _records_bytes_by_basename
from daydream.deep.routing_record import write_routing_record
from daydream.deep.settings import _resolve_opt_in
from daydream.deep.state import DeepState
from daydream.fanout import run_fanout
from daydream.flows.engine import FlowContext
from daydream.output_schema import record_array_schema, strict_object
from daydream.phases import (
    phase_arbiter_review,
    phase_suppression_review,
)
from daydream.phases.adjudication import IncompleteVerdicts
from daydream.phases.schemas import ARBITER_SCHEMA, SUPPRESSION_SCHEMA
from daydream.review_result import ReasonCode, ReviewCoverage, reason_for_budget, reason_for_exception
from daydream.supervision import revise_finding_fields
from daydream.trajectory import (
    DaydreamPhase,
    dispatch_scope,
    get_current_recorder,
    phase_scope,
)
from daydream.ui import print_warning


def _apply_adjudication_verdicts(
    records: list[dict[str, Any]],
    targets: list[int],
    verdicts: dict[int, dict[str, Any]],
    *,
    pass_name: str,
    id_field: str,
    fail_closed: bool,
) -> tuple[list[dict[str, Any]], list[RecordProvenance]]:
    """Apply positional agent verdicts using target UIDs snapshotted before mutation.

    ``targets[k]`` corresponds to agent id k+1 in ``id_field``. Revise bound
    findings without changing file/line; explicit keep:false drops them.
    Missing UIDs/verdicts or mismatched ids warn: arbiter retains the original
    (fail_closed=False), suppression drops it (True). UID-less drops use their
    original positions. Unselected records pass through unchanged.

    Return retained records plus per-target provenance recording the
    original UID, binding, retention and materially revised fields.
    """
    import warnings

    outcomes: list[RecordProvenance] = []
    dropped: set[str] = set()
    dropped_positions: set[int] = set()
    action = "dropping the unconfirmed record" if fail_closed else "retaining the original record unchanged"
    # Keep the input list intact until every verdict is applied; revisions never alter UIDs.
    for verdict_id, record_index in enumerate(targets, 1):
        record = records[record_index]
        uid = record_uid(record)
        verdict = verdicts.get(verdict_id)
        error = None
        if not uid:
            error = (f"target {id_field}={verdict_id} (record_index={record_index}) carries no uid, "
                     "so its verdict cannot be bound to a record identity")
        elif verdict is None:
            error = f"returned no verdict for {id_field}={verdict_id} (record_index={record_index}, uid={uid})"
        elif verdict.get(id_field) != verdict_id:
            error = (f"verdict {id_field} mismatch: expected {id_field}={verdict_id} "
                     f"but verdict contains {id_field}={verdict.get(id_field)!r} "
                     f"(record_index={record_index}, uid={uid})")
        bound = error is None
        kept = bool(verdict and verdict.get("keep", False)) if bound else not fail_closed
        revised: tuple[str, ...] = ()
        if error is not None:
            warnings.warn(f"{pass_name.capitalize()} {error}; {action}.", stacklevel=2)
        elif kept:
            assert verdict is not None
            before = dict(record)
            revise_finding_fields(record, verdict)
            revised = find_revision_delta(before, record)
        outcomes.append(RecordProvenance(uid, (pass_name,), bound, kept, revised))
        if not kept:
            if uid:
                dropped.add(uid)
            else:
                dropped_positions.add(record_index)
    return [record for i, record in enumerate(records)
            if record_uid(record) not in dropped and i not in dropped_positions], outcomes


def _load_group_verdicts(
    dd: Path, group: PlannedGroup, *, contract: str, records_digest: str,
) -> dict[int, dict[str, Any]] | None:
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
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or set(payload) != {
        'group_id', 'target_uids', 'verdicts', 'contract', 'input_digest', 'settled_digest',
    }:
        return None
    if payload['group_id'] != group.group_id or payload['target_uids'] != list(group.target_uids):
        return None
    if payload['contract'] != contract or records_digest not in (payload['input_digest'], payload['settled_digest']):
        return None
    raw = payload['verdicts']
    ids = range(1, len(group.target_uids) + 1)
    if not isinstance(raw, dict) or set(raw) != {str(i) for i in ids}:
        return None
    if not _validates_schema({'findings': list(raw.values())}, ARBITER_SCHEMA):
        return None
    if any(raw[str(i)]['arb_id'] != i for i in ids):
        return None
    return {i: raw[str(i)] for i in ids}


def _persist_group_verdicts(
    dd: Path, group: PlannedGroup, verdicts: dict[int, dict[str, Any]], *, contract: str, records_digest: str
) -> None:
    """Persist one group's verdicts, then its completion marker.

    The marker is written last so a crash between the two leaves a group that a
    resume will rerun rather than reuse half-written verdicts.
    """
    payload = {
        "group_id": group.group_id,
        "target_uids": list(group.target_uids),
        "contract": contract, "input_digest": records_digest, "settled_digest": records_digest,
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
                          reasons=(reason,) if reason else (), noop=target_count == 0 and reason is None,
                          diagnostic=verdicts.budget_reason if isinstance(verdicts, IncompleteVerdicts) else None)
    return reason is None


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _arbiter_backend(ctx: FlowContext, group: PlannedGroup, *, sharded: bool, effort_pin: str | None) -> Backend:
    return (ctx.backend_for("arbiter") if effort_pin is not None else
            ctx.backend_for_effort("arbiter", group.effort if sharded else "xhigh"))


def _adjudication_contract(ctx: FlowContext, state: DeepState, phase: str, backend: Backend,
                           effort: str | None, schema: dict[str, Any], strategy: str) -> str:
    """Bind the actual execution and the source context seen by adjudication."""
    return _digest({'backend': type(backend).__qualname__, 'model': backend.model,
        'effort': getattr(backend, 'reasoning_effort', None), 'requested_effort': effort, 'schema': schema,
        'profile': phase_identity_for(ctx, phase).profile_digest,
        'strategy': ctx.strategy(strategy), 'intent_authoritative': state.intent_authoritative,
        'revision': state.review_coverage.revision.to_dict(),
        'grounding': {**_loop_grounding(state), 'intent': digest_or_absent(state.intent_path.read_text())}})


def _arbiter_plan_component(ctx: FlowContext, state: DeepState, plan: ArbiterPlan,
                            effort_pin: str | None) -> dict[str, Any]:
    """Bind each actual execution contract, independent of informational routing reasons."""
    return {'sharded': plan.sharded, 'groups': [
        {'group_id': group.group_id, 'target_uids': list(group.target_uids), 'effort': group.effort,
         'contract': _adjudication_contract(ctx, state, 'arbiter',
            _arbiter_backend(ctx, group, sharded=plan.sharded, effort_pin=effort_pin),
            effort_pin or (group.effort if plan.sharded else 'xhigh'), ARBITER_SCHEMA, 'arbitration')}
        for group in plan.groups]}


_PLAN_PROOF_SCHEMA = strict_object(
    {
        "sharded": {"type": "boolean"},
        "groups": record_array_schema({
                    "group_id": {"type": "string", "pattern": r"^arbiter-group-\d+$"},
                    "target_uids": {
                        "type": "array",
                        "minItems": 1,
                        "uniqueItems": True,
                        "items": {"type": "string", "pattern": r"^.+:[1-9][0-9]*$"},
                    },
                    "effort": {"enum": ["medium", "high", "xhigh"]},
                    "contract": {"type": "string", "pattern": r"^[0-9a-f]{64}$"},
                }, minItems=1, maxItems=10000),
    }
)


def _completed_adjudication(ctx: FlowContext, state: DeepState, contract: dict[str, Any]) -> bool:
    """Validate saved execution against current policy and its host-settled evidence.

    Selection can change after a confirmed revision or dismissal. Reconstruct
    the completed plan to keep those findings out of a second suppression pass.
    """
    phases = ['arbiter', *(['suppression'] if contract['precision_mode'] else [])]
    if any(state.review_coverage.phases.get(phase, {}).get('status') != 'complete' for phase in phases):
        return False
    try:
        saved = json.loads(DeepArtifact.ADJUDICATION_COMPLETE.at(state.dd).read_text())
    except (OSError, ValueError):
        return False
    if not isinstance(saved, dict):
        return False
    plan = saved.get('plan')
    if (plan is None) != state.review_coverage.phases['arbiter']['noop']:
        return False
    if plan is not None:
        if not _validates_schema(plan, _PLAN_PROOF_SCHEMA):
            return False
        from daydream.run_config import _explicit_reasoning_effort_pin

        group_ids = [g['group_id'] for g in plan['groups']]
        uids = [uid for g in plan['groups'] for uid in g['target_uids']]
        if (len(set(group_ids)) != len(group_ids) or len(set(uids)) != len(uids)
                or (not plan['sharded'] and len(group_ids) != 1)):
            return False
        if plan['sharded'] and any(state.review_coverage.phases.get(g, {}).get('status') != 'complete'
                                   for g in group_ids):
            return False
        groups = tuple(PlannedGroup(g['group_id'], tuple(g['target_uids']), g['effort'], 'completed')
                       for g in plan['groups'])
        contract = {**contract, 'plan': _arbiter_plan_component(ctx, state,
            ArbiterPlan(plan['sharded'], groups, 'completed'), _explicit_reasoning_effort_pin(ctx.config, 'arbiter'))}
    return bool(saved == {**contract, 'records': _digest(state.record_pool.records)})


async def _run_arbiter(
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
    contracts = {group["group_id"]: group["contract"]
                 for group in _arbiter_plan_component(ctx, deep_state, plan, effort_pin)["groups"]}
    records_digests = {group.group_id: _digest([adjudicated[i] for i in targets_by_group[group.group_id]])
                       for group in plan.groups} if plan.sharded else {}

    async def invoke(group: PlannedGroup) -> dict[int, dict[str, Any]]:
        backend = _arbiter_backend(ctx, group, sharded=plan.sharded, effort_pin=effort_pin)
        verdicts, token = await phase_arbiter_review(
            backend, ctx.work,
            selected_records=[adjudicated[i] for i in
                              (targets_by_group[group.group_id] if plan.sharded else arbiter_targets)],
            input_path=arbiter_group_input_path(deep_state.dd, group.group_id) if plan.sharded else None,
            **_review_context_kwargs(ctx, deep_state, strategy=ctx.strategy("arbitration")),
            intent_authoritative=deep_state.intent_authoritative,
        )
        if not plan.sharded and token is not None and backend is ctx.backend_for("merge"):
            deep_state.arbiter_continuation = token
        return verdicts

    if not plan.sharded:
        async with phase_scope(DaydreamPhase.DEEP, stage="arbiter"):
            return await invoke(plan.groups[0]), {plan.groups[0].group_id: False}, []

    dd = deep_state.dd
    group_verdicts: dict[str, dict[int, dict[str, Any]]] = {}
    reused: dict[str, bool] = {}
    failed_groups: list[str] = []
    pending: list[PlannedGroup] = []
    for group in plan.groups:
        deep_state.review_coverage.require_phase(group.group_id)
        loaded = _load_group_verdicts(dd, group, contract=contracts[group.group_id],
                                      records_digest=records_digests[group.group_id])
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
                def record_failure(planned: PlannedGroup, exc: Exception) -> None:
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

                def save_verdicts(planned: PlannedGroup, group_verdicts_call: dict[int, dict[str, Any]]) -> None:
                    if not _record_verdict_coverage(deep_state.review_coverage, planned.group_id,
                            len(targets_by_group[planned.group_id]), group_verdicts_call):
                        failed_groups.append(planned.group_id)
                        if not isinstance(group_verdicts_call, IncompleteVerdicts):
                            group_verdicts[planned.group_id] = group_verdicts_call
                        return
                    try:
                        _persist_group_verdicts(dd, planned, group_verdicts_call,
                            contract=contracts[planned.group_id],
                            records_digest=records_digests[planned.group_id])
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


                await run_fanout(
                    pending, invoke, limiter=limiter, recorder=recorder, dispatch=dispatch,
                    descriptor=lambda group: f"deep-{group.group_id}",
                    completed=save_verdicts, failed=record_failure,
                )

    return (
        _merge_group_verdicts(plan, arbiter_targets, group_verdicts, targets_by_group),
        reused,
        failed_groups,
    )


def _arbiter_store_payload(
    dd: Path, rewrite_paths: list[Path], plan: ArbiterPlan
) -> dict[str, bytes] | None:
    """Collect a completed arbiter's on-disk outputs, or ``None`` if any is missing.

    The unit is storable only as a whole: every rewritten records file, each
    planned group's verdicts + marker when the plan sharded (the unsharded path
    persists no group files), and the whole-block completion proof.
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
    try:
        restored = deep_state.record_pool.reload(deep_state.review_coverage.revision.to_dict())
    except (OSError, ValueError):
        return False
    deep_state.record_pool = restored
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
        pool = deep_state.record_pool

        # A merge resume skips adjudication only with a whole-pass completion marker.
        adjudication_marker = DeepArtifact.ADJUDICATION_COMPLETE.at(dd)
        if ctx.pipeline().arbitration.enabled:
            structural_path = pool.structural_path
            adjudicated = pool.records
            structural_ids = {record_uid(record) for record in pool.structural}
            rewrite_paths = list(pool.paths.values())
            structural_range = range(len(pool.language), len(adjudicated))

            precision_mode = bool(ctx.pipeline().suppression.enabled or _resolve_opt_in(config, "precision_mode"))
            if precision_mode:
                deep_state.review_coverage.require_phase("suppression")
            from daydream.run_config import _resolved_reasoning_effort

            completion_contract: dict[str, Any] = {'plan': None, 'precision_mode': precision_mode,
                'profile': phase_identity_for(ctx, 'arbiter').profile_digest,
                'route': asdict(deep_state.latency_route) if deep_state.latency_route is not None else None,
                'suppression': _adjudication_contract(ctx, deep_state, 'suppression',
                    ctx.backend_for('suppression'), _resolved_reasoning_effort(config, 'suppression'),
                    SUPPRESSION_SCHEMA, 'suppression') if precision_mode else None}
            if config.start_at == 'merge' and _completed_adjudication(ctx, deep_state, completion_contract):
                return

            arbiter_targets = select_arbiter_targets(
                adjudicated,
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
            # Capture the subject before host reconciliation changes any records.
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
                plan_contract = _arbiter_plan_component(ctx, deep_state, plan, effort_pin)
                completion_contract['plan'] = plan_contract
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
                        plan=completion_contract,
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
                verdicts, reused, failed_groups = await _run_arbiter(
                    ctx, deep_state, plan, arbiter_targets, adjudicated,
                    effort_pin=effort_pin, targets_by_group=targets_by_group,
                )
                adjudication_complete = not failed_groups
                adjudicated, arbiter_outcomes = _apply_adjudication_verdicts(
                    adjudicated, arbiter_targets, verdicts,
                    pass_name="arbiter",
                    id_field="arb_id",
                    fail_closed=False,
                )
                pool.replace(adjudicated)
                pool.save()
                if plan.sharded:
                    settled = {record_uid(record): record for record in adjudicated}
                    for group in plan.groups:
                        if group.group_id in failed_groups:
                            continue
                        path = arbiter_group_verdicts_path(dd, group.group_id)
                        saved = json.loads(path.read_text())
                        saved['settled_digest'] = _digest([settled[uid] for uid in group.target_uids if uid in settled])
                        path.write_text(json.dumps(saved))
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
                    adjudicated, suppression_outcomes = _apply_adjudication_verdicts(
                        adjudicated, suppression_targets, sup_verdicts,
                        pass_name="suppression",
                        id_field="sup_id",
                        fail_closed=True,
                    )
                    pool.replace(adjudicated)
                    pool.save()
                    record_provenance(dd, pass_name="suppression", outcomes=suppression_outcomes)
                adjudication_complete = _record_verdict_coverage(
                    deep_state.review_coverage, "suppression", len(suppression_targets),
                    sup_verdicts if suppression_targets else {},
                ) and adjudication_complete
            # Whole-block marker: written only when every planned group completed,
            # so an interrupted sharded fan-out forces the block to re-enter and
            # reruns only its incomplete groups (and the opt-in suppression pass).
            if adjudication_complete:
                adjudication_marker.write_text(json.dumps({
                    **completion_contract,
                    "records": _digest(adjudicated)}))
                # Store only a completed whole-unit adjudication (every planned
                # group's files present) under the pre-dispatch key; a partial entry
                # must never be served as this unit's output.
                if arbiter_unit is not None and plan is not None:
                    arbiter_unit.store(lambda: _arbiter_store_payload(dd, rewrite_paths, plan))
            pool.replace(adjudicated)
