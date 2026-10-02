"""Bounded category audits and fail-closed vetting of grounded findings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio

from daydream import trajectory
from daydream.agent import console, run_agent
from daydream.backends import effective_fanout_concurrency
from daydream.config import (
    AUDIT_CATEGORIES,
    VET_BATCH_MAX_FINDINGS,
    EffortTier,
)
from daydream.extensions.api import Stop
from daydream.improve import artifacts
from daydream.improve.context import _audit_repo
from daydream.improve.partition import (
    Partition,
    PartitionGroup,
    PartitionStackOmission,
)
from daydream.improve.plan_index import (
    load_rejections,
    record_rejections,
)
from daydream.improve.prioritize import (
    _map_axis_severity,
    aggregate_cross_service,
    order_by_leverage,
)
from daydream.improve.prompts import (
    AUDIT_FINDINGS_SCHEMA,
    VET_SCHEMA,
)
from daydream.improve.redaction import redact_model_value
from daydream.services import (
    Service,
)
from daydream.ui import print_error

if TYPE_CHECKING:
    from daydream.flows.engine import FlowContext


from daydream.improve import audit_scope
from daydream.improve.reporting import (
    _findings_table,
)

_PROVENANCE_VALUES = {"introduced", "inherited"}


@dataclass(frozen=True)
class _AuditAssignment:
    category: str
    group: PartitionGroup

    @property
    def key(self) -> str:
        return f"{self.category}:{self.group.name}"


def _audit_assignments(
    categories: tuple[str, ...],
    groups: list[PartitionGroup],
) -> list[_AuditAssignment]:
    return [_AuditAssignment(category=category, group=group) for category in categories for group in groups]


def resolve_categories(
    tier: EffortTier,
    focus: str | None,
) -> tuple[str, ...]:
    """Resolve the audit categories for an effort tier and optional focus."""
    if focus in {"security", "performance", "tests"}:
        return (focus,)
    if focus == "branch":
        return AUDIT_CATEGORIES
    return tier.categories or AUDIT_CATEGORIES


def _schema_with_provenance(
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Return a structured-output schema extended with branch provenance."""
    extended = json.loads(json.dumps(schema))
    if not isinstance(extended, dict):
        raise RuntimeError("computed provenance schema is not an object")
    items = extended["properties"][
        "findings" if "findings" in extended["properties"] else "verdicts"
    ]["items"]
    items["properties"]["provenance"] = {
        "type": "string",
        "enum": sorted(_PROVENANCE_VALUES),
    }
    items["required"].append("provenance")
    return extended


async def _run_audit_assignments(
    ctx: FlowContext,
    *,
    assignments: list[_AuditAssignment],
    branch_focus: bool,
    backend: Any,
    recorder: trajectory.TrajectoryRecorder | None,
    limiter: anyio.CapacityLimiter,
    dispatch: trajectory.DispatchHandle | None,
) -> tuple[
    dict[str, tuple[_AuditAssignment, list[dict[str, Any]]]],
    dict[str, str],
]:
    """Run audit siblings while binding every attempted fork to one dispatch."""
    tier: EffortTier = ctx.data["effort_tier"]
    results: dict[str, tuple[_AuditAssignment, list[dict[str, Any]]]] = {}
    failures: dict[str, str] = {}
    async with anyio.create_task_group() as task_group:
        for assignment in assignments:
            scope_note = (
                f"Audit the {assignment.group.stack} stack in this group."
                if assignment.group.stack
                else "Audit this group's surface."
            )
            if ctx.config.improve_scope:
                scope_note += (
                    f"\nService scope slice: `{ctx.config.improve_scope}`. "
                    "The slice bounds where the audit searches. Slicing bounds "
                    "where you search, never what you may read; cross-service "
                    "boundary findings (traffic and data flow between services) "
                    "remain in scope."
                )
            if branch_focus:
                scope_note += (
                    "\nThis is a branch-focused audit. Limit findings to the "
                    "changed-file scope above. Tag every finding with "
                    '`provenance: "introduced"` when the supplied diff is '
                    "evidence that the branch introduced it; otherwise tag it "
                    '`provenance: "inherited"`.\n'
                    "Merge-base diff:\n```diff\n"
                    f"{ctx.data['branch_diff']}\n```"
                )
            prompt = ctx.registry.prompt("audit")(
                category=assignment.category,
                strategy=ctx.strategy(f"improve.audit.{assignment.category}"),
                group=audit_scope._group_dict(assignment.group),
                scope_note=scope_note,
                recon_summary=json.dumps(ctx.data["recon"], sort_keys=True),
                cwd=_audit_repo(ctx),
                tier=tier,
            )
            if branch_focus:
                prompt += (
                    "\nFor this branch-focused audit, the structured-output "
                    "schema additionally requires each finding to include "
                    '`provenance` as either `"introduced"` or `"inherited"`.'
                )

            async def _task(
                current: _AuditAssignment = assignment,
                task_prompt: str = prompt,
            ) -> None:
                descriptor = f"audit-{current.category}-{current.group.name}"
                async with limiter:
                    async with trajectory.maybe_fork(
                        recorder, descriptor, dispatch=dispatch
                    ):
                        try:
                            output, _, _ = await run_agent(
                                backend,
                                _audit_repo(ctx),
                                task_prompt,
                                phase=trajectory.DaydreamPhase.AUDIT,
                                output_schema=(
                                    _schema_with_provenance(AUDIT_FINDINGS_SCHEMA)
                                    if branch_focus
                                    else AUDIT_FINDINGS_SCHEMA
                                ),
                                read_only=True,
                                persist_session=False,
                                run_context=ctx.run_context,
                            )
                            raw_findings = (
                                output.get("findings", [])
                                if isinstance(output, dict)
                                else []
                            )
                            findings = [
                                redact_model_value(finding)
                                for finding in raw_findings
                                if isinstance(finding, dict)
                            ]
                            results[current.key] = (current, findings)
                        except Exception as exc:  # noqa: BLE001
                            failures[current.key] = trajectory.redact_text(
                                f"{type(exc).__name__}: {exc}"
                            )

            task_group.start_soon(_task)
    return results, failures


def _finish_fanout(
    phase: trajectory.PhaseScopeHandle,
    dispatch: trajectory.DispatchHandle | None,
    *,
    failed: int,
    total: int,
) -> None:
    """Record one terminal fan-out decision from its failed-child count."""
    status, reason = trajectory.partial_or_failed_terminal(failed < total)
    if dispatch is not None:
        dispatch.finish(status, reason)
    phase.finish(status, reason)


async def _step_audit(ctx: FlowContext) -> Stop | None:
    """Run tier-driven category audits and persist grounded findings."""
    directory: Path = ctx.data["improve_dir"]
    tier: EffortTier = ctx.data["effort_tier"]
    services: list[Service] = ctx.data["services"]
    partitions: list[Partition] = ctx.data["partitions"]
    groups: list[PartitionGroup] = ctx.data["partition_groups"]
    categories = resolve_categories(tier, ctx.config.improve_focus)
    branch_focus = ctx.config.improve_focus == "branch"
    assignments = _audit_assignments(categories, groups)
    backend = ctx.backend_for("audit")
    recorder = trajectory.get_current_recorder()
    limiter = anyio.CapacityLimiter(
        effective_fanout_concurrency(tier.max_concurrency, backend)
    )
    descriptors = tuple(
        f"audit-{assignment.category}-{assignment.group.name}"
        for assignment in assignments
    )
    async with trajectory.phase_scope(trajectory.DaydreamPhase.AUDIT) as phase:
        async with trajectory.dispatch_scope(
            recorder, phase=trajectory.DaydreamPhase.AUDIT, descriptors=descriptors
        ) as dispatch:
            results, failures = await _run_audit_assignments(
                ctx,
                assignments=assignments,
                branch_focus=branch_focus,
                backend=backend,
                recorder=recorder,
                limiter=limiter,
                dispatch=dispatch,
            )
            if failures:
                _finish_fanout(
                    phase, dispatch, failed=len(failures), total=len(assignments)
                )
            elif not assignments:
                phase.finish(
                    trajectory.LifecycleStatus.SKIPPED,
                    trajectory.LifecycleReasonCode.NO_ELIGIBLE_WORK,
                )

    if assignments and len(failures) == len(assignments):
        print_error(
            console,
            "Improve audit failed",
            "every audit assignment failed",
        )
        return Stop(1)

    per_group: dict[str, list[dict[str, Any]]] = {
        group.name: [] for group in groups
    }
    discarded_no_evidence = 0
    dropped_low_confidence = 0
    # A partition whose files span stacks is bundled into one group per stack, so
    # the same code is audited more than once and returns byte-identical findings
    # (same fingerprint). Collapse them here — the first pass keeps the finding;
    # later ones would otherwise inflate every count and mint a duplicate plan.
    seen_fingerprints: set[str] = set()
    for assignment in assignments:
        result = results.get(assignment.key)
        if result is None:
            continue
        _, raw_findings = result
        assignment_findings: list[dict[str, Any]] = []
        for finding in raw_findings:
            stamped = audit_scope._stamp_finding(
                finding,
                assignment.category,
                services,
                partitions,
                repo=_audit_repo(ctx),
            )
            if stamped is None:
                discarded_no_evidence += 1
                continue
            if tier.high_confidence_only and stamped.get("confidence") != "HIGH":
                dropped_low_confidence += 1
                continue
            fingerprint = str(stamped.get("fingerprint") or "")
            if fingerprint in seen_fingerprints:
                continue
            seen_fingerprints.add(fingerprint)
            assignment_findings.append(stamped)
        per_group[assignment.group.name].extend(assignment_findings)

    # Cap per group first so one noisy group cannot consume a tier's whole
    # finding budget, then apply the tier cap to the merged set.
    dropped_by_cap = 0
    grounded: list[dict[str, Any]] = []
    for group in groups:
        group_findings = order_by_leverage(per_group[group.name])
        if tier.max_findings is not None and len(group_findings) > tier.max_findings:
            dropped_by_cap += len(group_findings) - tier.max_findings
            group_findings = group_findings[: tier.max_findings]
        grounded.extend(group_findings)

    ordered = order_by_leverage(grounded)
    if tier.max_findings is not None and len(ordered) > tier.max_findings:
        dropped_by_cap += len(ordered) - tier.max_findings
        ordered = ordered[: tier.max_findings]

    combined = artifacts.write_artifact(
        directory / "audit-findings.json",
        {
            "categories_run": list(categories),
            "failed": dict(sorted(failures.items())),
            "findings": ordered,
        },
        phase=trajectory.DaydreamPhase.AUDIT,
    )
    _record_audit_coverage(
        directory,
        partitions,
        groups,
        ctx.data["partition_omissions"],
        failures=failures,
        assignments=assignments,
    )
    ctx.data["audit"] = combined
    ctx.data["audit_discarded_no_evidence"] = discarded_no_evidence
    ctx.data["audit_dropped_low_confidence"] = dropped_low_confidence
    ctx.data["audit_dropped_by_cap"] = dropped_by_cap
    return None


def _record_audit_coverage(
    directory: Path,
    partitions: list[Partition],
    groups: list[PartitionGroup],
    omissions: list[PartitionStackOmission],
    *,
    failures: dict[str, str],
    assignments: list[_AuditAssignment],
) -> None:
    """Rewrite the coverage ledger with what the audit actually reached."""
    failed_groups = {
        assignment.group.name
        for assignment in assignments
        if assignment.key in failures
    }
    ledger = audit_scope._coverage_ledger(
        partitions, groups, omissions, failed_groups=failed_groups
    )
    for entry in ledger["groups"]:
        entry["status"] = "failed" if entry["name"] in failed_groups else "audited"
    ledger["failed_assignments"] = dict(sorted(failures.items()))
    artifacts.write_artifact((directory / artifacts.COVERAGE_FILENAME), ledger, phase=trajectory.DaydreamPhase.AUDIT)


def _apply_vet_verdicts(
    findings: list[dict[str, Any]],
    verdicts: list[Any],
    *,
    rejected_at_sha: str | None,
    repo: Path | None = None,
    default_provenance: str | None = None,
    services: list[Service] | None = None,
    partitions: list[Partition] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Match 1-based vet IDs in any order; absent or malformed verdicts drop the finding."""
    by_vet_id: dict[int, dict[str, Any]] = {}
    for verdict in verdicts:
        if isinstance(verdict, dict) and isinstance(verdict.get("vet_id"), int):
            by_vet_id[verdict["vet_id"]] = verdict
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    corrected_fields = (
        "severity",
        "impact",
        "effort",
        "risk",
        "confidence",
        "maintenance_signals",
        "change_shape",
        "path",
        "line",
        "provenance",
    )
    for offset, finding in enumerate(findings):
        vet_id = offset + 1
        verdict = by_vet_id.get(vet_id)
        if verdict is None:
            continue
        if not verdict.get("keep", False):
            rejected.append(
                {
                    "fingerprint": finding["fingerprint"],
                    "title": finding.get("title", ""),
                    "path": finding.get("path", ""),
                    "reason": verdict.get("reason") or "vet rejected finding",
                    "rejected_at_sha": rejected_at_sha,
                }
            )
            continue
        corrected = dict(finding)
        for field in corrected_fields:
            if verdict.get(field) is not None:
                corrected[field] = verdict[field]
        # Omit unknown severity rather than promoting it through a fallback.
        severity = corrected.get("severity")
        mapped_severity = _map_axis_severity(severity)
        if mapped_severity is not None:
            corrected["severity"] = mapped_severity
        elif "severity" in corrected:
            del corrected["severity"]
        # Unlike the other corrected fields, ``None`` is a meaningful vet
        # correction here: it explicitly retracts an audit-time reuse target.
        if "reuse_target" in verdict:
            corrected["reuse_target"] = verdict["reuse_target"]
        location_corrected = verdict.get("path") is not None or verdict.get("line") is not None
        if location_corrected:
            audit_scope._correct_primary_evidence_location(
                corrected,
                path=str(corrected.get("path", "")),
                line=corrected.get("line"),
            )
        if default_provenance is not None and corrected.get("provenance") not in _PROVENANCE_VALUES:
            corrected["provenance"] = default_provenance
        if location_corrected:
            restamped = audit_scope._stamp_finding(
                corrected,
                str(corrected.get("category", "")),
                services or [],
                partitions or [],
                repo=repo or Path.cwd(),
            )
            if restamped is not None:
                corrected = restamped
            else:
                continue
        kept.append(corrected)
    return kept, rejected


async def _step_vet(ctx: FlowContext) -> None:
    """Re-verify audit findings and persist model-confirmed rejections."""
    directory: Path = ctx.data["improve_dir"]
    plans_dir = ctx.work.repo / "daydream_plans"
    previous = load_rejections(plans_dir)
    branch_focus = ctx.config.improve_focus == "branch"
    audit_findings = ctx.data["audit"].get("findings", [])
    candidates = [
        finding
        for finding in audit_findings
        if isinstance(finding, dict)
        and finding.get("fingerprint") not in previous
    ]
    previously_rejected = len(audit_findings) - len(candidates)

    by_category: dict[str, list[dict[str, Any]]] = {}
    for finding in candidates:
        category = str(finding.get("category", "unknown"))
        by_category.setdefault(category, []).append(finding)

    # One prompt inlines its whole batch as JSON, so batches are bounded and
    # fanned out rather than run as one serial prompt per category.
    batches = [
        (category, category_findings[offset : offset + VET_BATCH_MAX_FINDINGS])
        for category, category_findings in by_category.items()
        for offset in range(0, len(category_findings), VET_BATCH_MAX_FINDINGS)
    ]
    backend = ctx.backend_for("vet")
    tier: EffortTier = ctx.data["effort_tier"]
    recorder = trajectory.get_current_recorder()
    limiter = anyio.CapacityLimiter(
        effective_fanout_concurrency(tier.max_concurrency, backend)
    )
    results: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = [
        ([], []) for _ in batches
    ]
    failed_slots: set[int] = set()

    descriptors = tuple(
        f"vet-{category}-{index:02d}"
        for index, (category, _batch) in enumerate(batches)
    )
    async with trajectory.phase_scope(trajectory.DaydreamPhase.VET) as phase:
        async with trajectory.dispatch_scope(
            recorder, phase=trajectory.DaydreamPhase.VET, descriptors=descriptors
        ) as dispatch:
            async with anyio.create_task_group() as task_group:
                for index, (category, batch) in enumerate(batches):
                    indexed = [
                        {**finding, "vet_id": vet_id}
                        for vet_id, finding in enumerate(batch, start=1)
                    ]
                    prompt = ctx.registry.prompt("vet")(
                        strategy=ctx.strategy("improve.vetting"),
                        findings=indexed,
                        cwd=_audit_repo(ctx),
                    )
                    if branch_focus:
                        prompt += (
                            "\nConfirm each candidate's branch provenance against this "
                            "merge-base diff. Return `provenance` as `introduced` only "
                            "when the diff supports that conclusion; otherwise return "
                            "`inherited`.\n```diff\n"
                            f"{ctx.data['branch_diff']}\n```"
                        )

                    async def _task(
                        slot: int = index,
                        descriptor: str = f"vet-{category}-{index:02d}",
                        batch_findings: list[dict[str, Any]] = batch,
                        task_prompt: str = prompt,
                    ) -> None:
                        async with limiter:
                            async with trajectory.maybe_fork(
                                recorder, descriptor, dispatch=dispatch
                            ):
                                try:
                                    output, _, _ = await run_agent(
                                        backend,
                                        _audit_repo(ctx),
                                        task_prompt,
                                        phase=trajectory.DaydreamPhase.VET,
                                        output_schema=(
                                            _schema_with_provenance(VET_SCHEMA)
                                            if branch_focus
                                            else VET_SCHEMA
                                        ),
                                        read_only=True,
                                        persist_session=False,
                                        run_context=ctx.run_context,
                                    )
                                except Exception:  # noqa: BLE001 - no verdict fails closed
                                    output = {}
                                    failed_slots.add(slot)
                                safe_output = redact_model_value(output)
                                verdicts = (
                                    safe_output.get("verdicts", [])
                                    if isinstance(safe_output, dict)
                                    and isinstance(safe_output.get("verdicts"), list)
                                    else []
                                )
                                verdict_ids = {
                                    verdict.get("vet_id")
                                    for verdict in verdicts
                                    if isinstance(verdict, dict)
                                    and type(verdict.get("vet_id")) is int
                                }
                                if verdict_ids != set(
                                    range(1, len(batch_findings) + 1)
                                ):
                                    failed_slots.add(slot)
                                results[slot] = _apply_vet_verdicts(
                                    batch_findings,
                                    verdicts,
                                    rejected_at_sha=ctx.work.head_sha,
                                    repo=_audit_repo(ctx),
                                    default_provenance=(
                                        "inherited" if branch_focus else None
                                    ),
                                    services=ctx.data["services"],
                                    partitions=ctx.data["partitions"],
                                )

                    task_group.start_soon(_task)
            if failed_slots:
                _finish_fanout(
                    phase, dispatch, failed=len(failed_slots), total=len(batches)
                )
            elif not batches:
                phase.finish(
                    trajectory.LifecycleStatus.SKIPPED,
                    trajectory.LifecycleReasonCode.NO_ELIGIBLE_WORK,
                )

    kept = [finding for batch_kept, _ in results for finding in batch_kept]
    rejected = [
        finding for _, batch_rejected in results for finding in batch_rejected
    ]

    record_rejections(plans_dir, rejected)
    findings = aggregate_cross_service(order_by_leverage(kept))
    ordered_defects = order_by_leverage(findings)
    vetted = artifacts.write_artifact(
        (directory / artifacts.VETTED_FINDINGS_FILENAME),
        {"findings": ordered_defects},
        phase=trajectory.DaydreamPhase.VET,
    )
    ctx.data["vetted"] = vetted
    ctx.data["previously_rejected"] = previously_rejected
    ctx.data["vet_rejected"] = len(rejected)
    ctx.data["defects"] = ordered_defects
    if branch_focus:
        introduced = [finding for finding in ordered_defects if finding.get("provenance") == "introduced"]
        inherited = [finding for finding in ordered_defects if finding.get("provenance") != "introduced"]
        ctx.data["findings_table"] = (
            "### Introduced by this branch\n\n"
            f"{_findings_table(introduced)}\n\n"
            "### Inherited from the base\n\n"
            f"{_findings_table(inherited, start=len(introduced) + 1)}"
        )
    else:
        ctx.data["findings_table"] = _findings_table(ordered_defects)
