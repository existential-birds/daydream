"""Registered flow steps for repository-wide improve advising."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio

import daydream.agent as agent
from daydream import git_ops
from daydream.agent import console, get_non_interactive, run_agent
from daydream.backends import effective_fanout_concurrency
from daydream.config import (
    AUDIT_CATEGORIES,
    PLAN_WRITE_MAX_CONCURRENCY,
    VET_BATCH_MAX_FINDINGS,
    EffortTier,
)
from daydream.config_file import DaydreamFileConfig
from daydream.deep.detection import StackAssignment, detect_stacks
from daydream.deep.orchestrator import _diff_changed_files, get_installed_skills
from daydream.exploration_runner import repo_scan
from daydream.extensions.api import FlowStep, Stop
from daydream.improve.artifacts import (
    coverage_path,
    plan_write_diagnostics_path,
    recon_path,
    report_path,
    vetted_findings_path,
)
from daydream.improve.assemble import (
    AssemblyIssue,
    assemble_plan,
    render_issue,
)
from daydream.improve.command_contract import (
    RECON_COMMAND_SCHEMA,
    validate_recon_commands,
)
from daydream.improve.partition import (
    PARTITION_MAX_FILES,
    Partition,
    PartitionGroup,
    build_partitions,
    group_partitions,
    stack_by_path,
)
from daydream.improve.plans import (
    PlanWriteSession,
    _attempt_diagnostic,
    _markdown_cell,
    load_rejections,
    record_plan_write_diagnostics,
    record_rejections,
)
from daydream.improve.prioritize import (
    aggregate_cross_service,
    leverage_score,
    order_by_leverage,
    partition_direction,
)
from daydream.improve.prompts import (
    AUDIT_FINDINGS_SCHEMA,
    PLAN_AUTHOR_SCHEMA,
    RECON_COMMAND_CONTRACT_BULLET,
    VET_SCHEMA,
    build_plan_writer_repair_prompt,
)
from daydream.improve.services import Service, enumerate_services, filter_scope
from daydream.pr_review import compute_fingerprint
from daydream.trajectory import (
    DaydreamPhase,
    get_current_recorder,
    maybe_fork,
    phase_scope,
    redact_text,
)
from daydream.ui import print_error, print_info, print_success, print_warning

if TYPE_CHECKING:
    from daydream.flows.engine import FlowContext


RECON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "languages",
        "commands",
        "conventions",
        "intent_docs",
    ],
    "properties": {
        "languages": {"type": "array", "items": {"type": "string"}},
        "commands": {
            "type": "array",
            "items": RECON_COMMAND_SCHEMA,
        },
        "conventions": {"type": "array", "items": {"type": "string"}},
        "intent_docs": {"type": "array", "items": {"type": "string"}},
    },
}

_EVIDENCE_LOCATION = re.compile(
    r"^`?(.+?):(\d+)(?::\d+)?(?:`|\b)"
)
_PLAN_SLUG = re.compile(r"^[a-z0-9-]{1,60}$")
_PROVENANCE_VALUES = {"introduced", "inherited"}


def _redact_model_value(value: Any) -> Any:
    """Redact nested model-authored strings before host use or persistence."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [_redact_model_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_model_value(item) for item in value)
    if isinstance(value, dict):
        return {
            key: _redact_model_value(item)
            for key, item in value.items()
        }
    return value


def _artifact_provenance(*, phase: DaydreamPhase) -> dict[str, str]:
    """Return host-authored identity tying an improve artifact to this run."""
    recorder = get_current_recorder()
    if recorder is None:
        return {"session_id": "unrecorded", "phase": phase.value}
    try:
        trajectory_path = recorder.path.relative_to(recorder.target_dir).as_posix()
    except ValueError:
        trajectory_path = str(recorder.path)
    return {
        "session_id": recorder.session_id,
        "phase": phase.value,
        "trajectory_path": trajectory_path,
    }


def _run_session_id() -> str | None:
    recorder = get_current_recorder()
    return recorder.session_id if recorder is not None else None


def _report_with_provenance(content: str) -> str:
    session_id = _run_session_id()
    if session_id is None:
        return content
    heading, separator, remainder = content.partition("\n")
    return (
        f"{heading}\n\nDaydream run: `{session_id}`\n"
        f"{separator}{remainder.lstrip()}"
    )


def _with_artifact_provenance(
    payload: dict[str, Any],
    *,
    phase: DaydreamPhase,
) -> dict[str, Any]:
    return {
        "artifact_provenance": _artifact_provenance(phase=phase),
        **payload,
    }


@dataclass(frozen=True)
class _AuditAssignment:
    category: str
    group: PartitionGroup
    skill: str | None

    @property
    def key(self) -> str:
        return f"{self.category}:{self.group.name}"


def _build_recon_prompt(
    repo: Path,
    services: list[Service],
    groups: list[PartitionGroup],
    exploration_summary: str,
) -> str:
    service_lines = "\n".join(
        f"- {service.name}: {service.root.as_posix()}" for service in services
    )
    audited_roots = sorted(
        {root for group in groups for root in group.roots if root != "."}
    )
    root_list = ", ".join(f"`{root}`" for root in audited_roots)
    return f"""IMPROVE_RECON

Read the repository at {repo} without modifying it. Return structured
reconnaissance facts only:

- languages and frameworks in active use;
- {RECON_COMMAND_CONTRACT_BULLET}
- conventions that implementation plans must preserve;
- intent documents such as README, roadmap, ADR, and architecture files.

Services:
{service_lines or "- repository root"}

Audited subtrees ({len(audited_roots)}): {root_list or "the repository root"}.
Return the per-subtree build, test, and lint commands for these too, not only
the repository-wide ones: set each command's `working_directory` to the
directory it actually runs in, and set `applicability.scope` to
`in-scope-paths` naming the subtrees it governs whenever it does not genuinely
govern the whole repository.

Existing repository scan:
{exploration_summary or "No additional conventions detected."}
"""


async def _step_recon(ctx: FlowContext) -> Stop | None:
    """Enumerate services, inspect repository conventions, and detect stacks."""
    target = ctx.work.repo
    directory: Path = ctx.data["improve_dir"]
    description_mode = ctx.config.improve_plan_description is not None
    branch_focus = ctx.config.improve_focus == "branch"
    if (
        branch_focus
        and ctx.work.head_branch is not None
        and ctx.work.head_branch == ctx.work.base_branch
    ):
        print_error(
            console,
            "Branch Focus Requires a Feature Branch",
            f"cwd is on the base branch {ctx.work.base_branch!r} -- "
            "there are no branch changes to audit.\n"
            "Check out a feature branch and re-run, or run a full improve "
            "audit without --focus branch.",
        )
        return Stop(1)

    branch_diff = (
        git_ops.diff(target, ctx.work.base_branch) if branch_focus else ""
    )
    branch_files = _diff_changed_files(branch_diff) if branch_focus else []
    if branch_focus:
        # Branch focus needs every category over one small diff, run serially —
        # but the requested --effort tier still owns confidence filtering,
        # finding caps, and audit depth, or the run silently contradicts the
        # tier the report claims it used.
        requested: EffortTier = ctx.data["effort_tier"]
        ctx.data["effort_tier"] = replace(
            requested, categories=None, max_concurrency=1
        )
    ctx.data["branch_diff"] = branch_diff
    ctx.data["branch_files"] = branch_files

    all_services = (
        []
        if description_mode
        else enumerate_services(
            target,
            ctx.config.file_config or DaydreamFileConfig(),
        )
    )
    services = all_services
    if ctx.config.improve_scope and not description_mode:
        try:
            services = filter_scope(
                services,
                ctx.config.improve_scope,
                (ctx.config.file_config or DaydreamFileConfig()).improve_service_groups,
            )
        except ValueError as exc:
            print_error(console, "Invalid Improve Scope", str(exc))
            return Stop(1)
    if branch_focus:
        services = _services_for_files(services, tuple(branch_files))

    stacks: list[StackAssignment] = []
    partitions: list[Partition] = []
    groups: list[PartitionGroup] = []
    skipped: list[Partition] = []
    if not description_mode:
        installed = get_installed_skills()
        availability = (
            installed if installed is not None else ctx.registry.stack_keys()
        )
        tracked = branch_files if branch_focus else git_ops.ls_files(target)
        stacks = detect_stacks(
            tracked,
            skill_availability=availability,
            registry=ctx.registry,
        )
        if ctx.config.improve_scope:
            stacks = _stacks_for_services(stacks, services)
            tracked = sorted({path for stack in stacks for path in stack.files})
        partitions, groups, skipped = _partition_repository(
            ctx,
            tracked,
            services,
            stacks,
            branch_focus=branch_focus,
        )
        coverage_path(directory).write_text(
            json.dumps(
                _with_artifact_provenance(
                    _coverage_ledger(partitions, groups, skipped),
                    phase=DaydreamPhase.RECON,
                ),
                indent=2,
            )
            + "\n"
        )

    backend = ctx.backend_for("recon")
    async with phase_scope(DaydreamPhase.RECON):
        exploration = await repo_scan(backend, target)
        recon, _, _ = await run_agent(
            backend,
            target,
            _build_recon_prompt(
                target, services, groups, exploration.to_prompt_section()
            ),
            phase=DaydreamPhase.RECON,
            output_schema=RECON_SCHEMA,
            read_only=True,
            persist_session=False,
        )

    recon_data: dict[str, Any] = {}
    total_candidates = 0
    valid_commands: list[dict[str, Any]] = []
    command_errors: list[str] = []
    safe_recon = _redact_model_value(recon)
    if isinstance(safe_recon, dict):
        raw_commands = safe_recon.get("commands")
        total_candidates = len(raw_commands) if isinstance(raw_commands, list) else 0
        candidate_commands, command_errors = validate_recon_commands(
            safe_recon,
            repo=target,
        )
        valid_commands = candidate_commands
        recon_data = {
            "artifact_type": "daydream.improve-recon",
            "artifact_provenance": _artifact_provenance(
                phase=DaydreamPhase.RECON
            ),
            **{
                field: value
                if isinstance((value := safe_recon.get(field)), list)
                and all(isinstance(item, str) for item in value)
                else []
                for field in ("languages", "conventions", "intent_docs")
            },
            "commands": valid_commands,
            "command_rejections": [
                {
                    "code": error.partition("@")[0],
                    "pointer": error.partition("@")[2] or "/",
                }
                for error in command_errors
            ],
        }
    else:
        recon_data = {
            "artifact_type": "daydream.improve-recon",
            "artifact_provenance": _artifact_provenance(
                phase=DaydreamPhase.RECON
            ),
            "commands": [],
            "command_rejections": [
                {"code": "RECON_CONTAINER_INVALID", "pointer": "/"}
            ],
        }
    recon_path(directory).write_text(json.dumps(recon_data, indent=2) + "\n")
    reasons = Counter(error.partition("@")[0] for error in command_errors)
    recorder = get_current_recorder()
    if recorder is not None:
        recorder.emit_command_validation_summary(
            total_candidates=total_candidates,
            accepted=len(valid_commands),
            rejected=total_candidates - len(valid_commands),
            reasons=dict(reasons),
        )
    if not recon_data.get("commands"):
        reason_summary = ", ".join(
            f"{code}: {count}"
            for code, count in sorted(reasons.items())
        ) or "the model returned no usable command container"
        candidate_summary = (
            f"{total_candidates} repository command candidates were found but "
            "rejected. "
            if total_candidates
            else (
                "The repository command container was rejected before "
                "candidates could be enumerated. "
            )
        )
        print_warning(
            console,
            "Repository command candidates rejected. "
            + candidate_summary
            + f"Reasons: {reason_summary}. Audit and planning will continue "
            "without executable verification commands. "
            "Rejection codes are recorded in "
            ".daydream/improve/recon.json under `command_rejections`.",
        )

    ctx.data["all_services"] = all_services
    ctx.data["services"] = services
    ctx.data["recon"] = recon_data
    ctx.data["stacks"] = stacks
    ctx.data["partitions"] = partitions
    ctx.data["partition_groups"] = groups
    ctx.data["partitions_not_audited"] = skipped
    return None


def _partition_repository(
    ctx: FlowContext,
    tracked: list[str],
    services: list[Service],
    stacks: list[StackAssignment],
    *,
    branch_focus: bool,
) -> tuple[list[Partition], list[PartitionGroup], list[Partition]]:
    """Cover the audited surface with partitions and pack them into groups.

    Branch focus and the ``quick`` tier bypass partitioning: both audit one
    synthetic whole-surface group, so their fan-out stays exactly one agent per
    category.
    """
    stack_of = stack_by_path(stacks)
    file_config = ctx.config.file_config or DaydreamFileConfig()
    max_files = file_config.improve_partition_max_files or PARTITION_MAX_FILES
    tier: EffortTier = ctx.data["effort_tier"]
    max_groups = (
        file_config.improve_max_partition_groups or tier.max_partition_groups
    )

    if branch_focus or ctx.config.improve_effort == "quick":
        whole = Partition(
            name="branch" if branch_focus else "repository",
            root=".",
            source="branch" if branch_focus else "quick",
            service=None,
            files=tuple(tracked),
        )
        return [whole], [_whole_surface_group(whole, stack_of)], []

    partitions = build_partitions(tracked, services, max_files=max_files)
    groups, skipped = group_partitions(
        partitions,
        stack_of,
        max_files=max_files,
        max_groups=max_groups,
    )
    return partitions, groups, skipped


def _whole_surface_group(
    partition: Partition, stack_of: dict[str, str]
) -> PartitionGroup:
    counts = Counter(stack_of.get(path, "generic") for path in partition.files)
    dominant = (
        min(sorted(counts), key=lambda stack: (-counts[stack], stack))
        if counts
        else "generic"
    )
    return PartitionGroup(
        name="group-01", stack=dominant, partitions=(partition,)
    )


def _partition_dict(partition: Partition) -> dict[str, Any]:
    return {
        "name": partition.name,
        "root": partition.root,
        "file_count": len(partition.files),
        "service": partition.service,
    }


def _group_dict(group: PartitionGroup) -> dict[str, Any]:
    return {
        "name": group.name,
        "stack": group.stack,
        "file_count": group.file_count,
        "partitions": [
            _partition_dict(partition) for partition in group.partitions
        ],
    }


def _coverage_ledger(
    partitions: list[Partition],
    groups: list[PartitionGroup],
    skipped: list[Partition],
) -> dict[str, Any]:
    """Build the coverage ledger recording what the audit did and did not cover."""
    return {
        "artifact_type": "daydream.improve-coverage",
        "partitions": [
            {**_partition_dict(partition), "source": partition.source}
            for partition in partitions
        ],
        "groups": [
            {
                "name": group.name,
                "stack": group.stack,
                "file_count": group.file_count,
                "partitions": [
                    partition.name for partition in group.partitions
                ],
            }
            for group in groups
        ],
        "not_audited": [
            {
                "partition": partition.name,
                "root": partition.root,
                "file_count": len(partition.files),
                "reason": "group-ceiling",
            }
            for partition in skipped
        ],
    }


def _audit_assignments(
    ctx: FlowContext,
    categories: tuple[str, ...],
    groups: list[PartitionGroup],
) -> list[_AuditAssignment]:
    assignments: list[_AuditAssignment] = []
    for category in categories:
        for group in groups:
            skill = (
                ctx.registry.skill_if_registered(
                    f"audit:{category}:{group.stack}"
                )
                if group.stack
                else None
            )
            if skill is None:
                skill = ctx.registry.skill_if_registered(f"audit:{category}")
            assignments.append(
                _AuditAssignment(category=category, group=group, skill=skill)
            )
    return assignments


def _services_for_files(
    services: list[Service], files: tuple[str, ...]
) -> list[Service]:
    if not files:
        return services
    return [
        service
        for service in services
        if any(
            path == service.root.as_posix()
            or path.startswith(f"{service.root.as_posix()}/")
            for path in files
        )
    ]


def _stacks_for_services(
    stacks: list[StackAssignment],
    services: list[Service],
) -> list[StackAssignment]:
    roots = tuple(service.root.as_posix() for service in services)
    scoped: list[StackAssignment] = []
    for stack in stacks:
        files = [
            path
            for path in stack.files
            if any(path == root or path.startswith(f"{root}/") for root in roots)
        ]
        if files:
            scoped.append(
                StackAssignment(
                    stack_name=stack.stack_name,
                    skill_invocation=stack.skill_invocation,
                    files=files,
                    is_docs_only=stack.is_docs_only,
                )
            )
    return scoped


def _evidence_paths(finding: dict[str, Any]) -> list[str]:
    evidence = finding.get("evidence")
    if not isinstance(evidence, list):
        return []
    paths: list[str] = []
    for entry in evidence:
        if not isinstance(entry, str):
            continue
        match = _EVIDENCE_LOCATION.match(entry.strip())
        if match is not None:
            paths.append(match.group(1).strip("`"))
    return paths


def _owning_partition(
    partitions: list[Partition], evidence_paths: list[str]
) -> str | None:
    if not evidence_paths:
        return None
    path = evidence_paths[0]
    for partition in sorted(
        partitions, key=lambda item: -len(item.root)
    ):
        if partition.root == "." or path.startswith(f"{partition.root}/"):
            return partition.name
    return None


def _stamp_finding(
    finding: dict[str, Any],
    category: str,
    services: list[Service],
    partitions: list[Partition],
) -> dict[str, Any] | None:
    evidence_paths = _evidence_paths(finding)
    if not evidence_paths:
        return None
    stamped = dict(finding)
    stamped["category"] = category
    stamped["partition"] = _owning_partition(partitions, evidence_paths)
    stamped["services"] = [
        service.name
        for service in services
        if any(
            path == service.root.as_posix()
            or path.startswith(f"{service.root.as_posix()}/")
            for path in evidence_paths
        )
    ]
    stamped["fingerprint"] = compute_fingerprint(
        str(stamped.get("path", "")),
        str(stamped.get("title", "")),
        str(stamped.get("body", "")),
    )
    return stamped


def resolve_categories(
    tier: EffortTier,
    focus: str | None,
) -> tuple[str, ...]:
    """Resolve the audit categories for an effort tier and optional focus."""
    if focus in {"security", "performance", "tests"}:
        return (focus,)
    if focus == "next":
        return ("direction",)
    if focus == "branch":
        return AUDIT_CATEGORIES
    return tier.categories or AUDIT_CATEGORIES


def _schema_with_provenance(
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Return a structured-output schema extended with branch provenance."""
    extended = json.loads(json.dumps(schema))
    items = extended["properties"][
        "findings" if "findings" in extended["properties"] else "verdicts"
    ]["items"]
    items["properties"]["provenance"] = {
        "type": "string",
        "enum": sorted(_PROVENANCE_VALUES),
    }
    items["required"].append("provenance")
    return extended


async def _step_audit(ctx: FlowContext) -> Stop | None:
    """Run tier-driven category audits and persist grounded findings."""
    directory: Path = ctx.data["improve_dir"]
    tier: EffortTier = ctx.data["effort_tier"]
    services: list[Service] = ctx.data["services"]
    partitions: list[Partition] = ctx.data["partitions"]
    groups: list[PartitionGroup] = ctx.data["partition_groups"]
    categories = resolve_categories(tier, ctx.config.improve_focus)
    branch_focus = ctx.config.improve_focus == "branch"
    assignments = _audit_assignments(ctx, categories, groups)
    backend = ctx.backend_for("audit")
    recorder = get_current_recorder()
    limiter = anyio.CapacityLimiter(
        effective_fanout_concurrency(tier.max_concurrency, backend)
    )
    results: dict[str, tuple[_AuditAssignment, list[dict[str, Any]]]] = {}
    failures: dict[str, str] = {}

    async with anyio.create_task_group() as task_group:
        for assignment in assignments:
            invocation = (
                backend.format_skill_invocation(assignment.skill)
                if assignment.skill is not None
                else None
            )
            scope_note = (
                f"Audit the {assignment.group.stack} stack in this group."
                if assignment.group.stack
                else "Audit this group's surface."
            )
            if ctx.config.improve_focus == "next":
                scope_note += (
                    "\nReturn 4–6 grounded suggestions with honest tradeoffs "
                    "and design/spike-sized next steps."
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
                skill_invocation=invocation,
                group=_group_dict(assignment.group),
                scope_note=scope_note,
                recon_summary=json.dumps(ctx.data["recon"], sort_keys=True),
                cwd=ctx.work.repo,
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
                    async with maybe_fork(recorder, descriptor):
                        try:
                            output, _, _ = await run_agent(
                                backend,
                                ctx.work.repo,
                                task_prompt,
                                phase=DaydreamPhase.AUDIT,
                                output_schema=(
                                    _schema_with_provenance(
                                        AUDIT_FINDINGS_SCHEMA,
                                    )
                                    if branch_focus
                                    else AUDIT_FINDINGS_SCHEMA
                                ),
                                read_only=True,
                                persist_session=False,
                            )
                            raw_findings = (
                                output.get("findings", [])
                                if isinstance(output, dict)
                                else []
                            )
                            findings = [
                                _redact_model_value(finding)
                                for finding in raw_findings
                                if isinstance(finding, dict)
                            ]
                            results[current.key] = (current, findings)
                        except Exception as exc:  # noqa: BLE001
                            failures[current.key] = redact_text(
                                f"{type(exc).__name__}: {exc}"
                            )

            task_group.start_soon(_task)

    if recorder is not None:
        recorder.create_dispatch_step(phase=DaydreamPhase.AUDIT)

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
    for assignment in assignments:
        result = results.get(assignment.key)
        if result is None:
            continue
        _, raw_findings = result
        assignment_findings: list[dict[str, Any]] = []
        for finding in raw_findings:
            stamped = _stamp_finding(
                finding,
                assignment.category,
                services,
                partitions,
            )
            if stamped is None:
                discarded_no_evidence += 1
                continue
            if tier.high_confidence_only and stamped.get("confidence") != "HIGH":
                dropped_low_confidence += 1
                continue
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

    combined = _with_artifact_provenance(
        {
            "categories_run": list(categories),
            "failed": dict(sorted(failures.items())),
            "findings": ordered,
        },
        phase=DaydreamPhase.AUDIT,
    )
    (directory / "audit-findings.json").write_text(
        json.dumps(combined, indent=2) + "\n"
    )
    _record_audit_coverage(
        directory,
        partitions,
        groups,
        ctx.data["partitions_not_audited"],
        failures=failures,
        assignments=assignments,
    )
    ctx.data["audit"] = combined
    ctx.data["audit_discarded_no_evidence"] = discarded_no_evidence
    ctx.data["audit_dropped_low_confidence"] = dropped_low_confidence
    ctx.data["audit_dropped_by_cap"] = dropped_by_cap
    return None


def _group_roots_cell(group: PartitionGroup, *, limit: int = 4) -> str:
    """Render a group's roots for a report line, truncating a long tail."""
    roots = group.roots
    shown = ", ".join(f"`{root}/`" for root in roots[:limit])
    remainder = len(roots) - limit
    return f"{shown} +{remainder} more" if remainder > 0 else shown


def _record_audit_coverage(
    directory: Path,
    partitions: list[Partition],
    groups: list[PartitionGroup],
    skipped: list[Partition],
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
    ledger = _coverage_ledger(partitions, groups, skipped)
    for entry in ledger["groups"]:
        entry["status"] = "failed" if entry["name"] in failed_groups else "audited"
    ledger["failed_assignments"] = dict(sorted(failures.items()))
    coverage_path(directory).write_text(
        json.dumps(
            _with_artifact_provenance(ledger, phase=DaydreamPhase.AUDIT),
            indent=2,
        )
        + "\n"
    )


def _apply_vet_verdicts(
    findings: list[dict[str, Any]],
    verdicts: list[Any],
    *,
    rejected_at_sha: str,
    default_provenance: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply 1-based, vet_id-keyed vet verdicts with fail-closed polarity.

    Verdicts are matched by ``vet_id`` rather than array position, so the
    model may return them in any order. A finding with no matching verdict
    (missing, non-dict, or unmatched id) is dropped, preserving the
    fail-closed polarity.
    """
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
        if (
            default_provenance is not None
            and corrected.get("provenance") not in _PROVENANCE_VALUES
        ):
            corrected["provenance"] = default_provenance
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
    recorder = get_current_recorder()
    limiter = anyio.CapacityLimiter(
        effective_fanout_concurrency(tier.max_concurrency, backend)
    )
    results: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = [
        ([], []) for _ in batches
    ]

    async with anyio.create_task_group() as task_group:
        for index, (category, batch) in enumerate(batches):
            indexed = [
                {**finding, "vet_id": vet_id}
                for vet_id, finding in enumerate(batch, start=1)
            ]
            prompt = ctx.registry.prompt("vet")(
                findings=indexed,
                cwd=ctx.work.repo,
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
                    async with maybe_fork(recorder, descriptor):
                        try:
                            output, _, _ = await run_agent(
                                backend,
                                ctx.work.repo,
                                task_prompt,
                                phase=DaydreamPhase.VET,
                                output_schema=(
                                    _schema_with_provenance(VET_SCHEMA)
                                    if branch_focus
                                    else VET_SCHEMA
                                ),
                                read_only=True,
                                persist_session=False,
                            )
                        except Exception:  # noqa: BLE001 - no verdict fails closed
                            output = {}
                        safe_output = _redact_model_value(output)
                        verdicts = (
                            safe_output.get("verdicts", [])
                            if isinstance(safe_output, dict)
                            and isinstance(safe_output.get("verdicts"), list)
                            else []
                        )
                        results[slot] = _apply_vet_verdicts(
                            batch_findings,
                            verdicts,
                            rejected_at_sha=ctx.work.head_sha,
                            default_provenance=(
                                "inherited" if branch_focus else None
                            ),
                        )

            task_group.start_soon(_task)

    if recorder is not None:
        recorder.create_dispatch_step(phase=DaydreamPhase.VET)

    kept = [finding for batch_kept, _ in results for finding in batch_kept]
    rejected = [
        finding for _, batch_rejected in results for finding in batch_rejected
    ]

    record_rejections(plans_dir, rejected)
    findings = aggregate_cross_service(order_by_leverage(kept))
    defects, direction = partition_direction(findings)
    ordered_defects = order_by_leverage(defects)
    ordered_direction = order_by_leverage(direction)
    vetted = _with_artifact_provenance(
        {
            "findings": [*ordered_defects, *ordered_direction],
            "defects": ordered_defects,
            "direction": ordered_direction,
        },
        phase=DaydreamPhase.VET,
    )
    vetted_findings_path(directory).write_text(
        json.dumps(vetted, indent=2) + "\n"
    )
    ctx.data["vetted"] = vetted
    ctx.data["previously_rejected"] = previously_rejected
    ctx.data["vet_rejected"] = len(rejected)
    ctx.data["defects"] = ordered_defects
    ctx.data["direction"] = ordered_direction
    if branch_focus:
        introduced = [
            finding
            for finding in ordered_defects
            if finding.get("provenance") == "introduced"
        ]
        inherited = [
            finding
            for finding in ordered_defects
            if finding.get("provenance") != "introduced"
        ]
        ctx.data["findings_table"] = (
            "### Introduced by this branch\n\n"
            f"{_findings_table(introduced)}\n\n"
            "### Inherited from the base\n\n"
            f"{_findings_table(inherited, start=len(introduced) + 1)}"
        )
    else:
        ctx.data["findings_table"] = _findings_table(ordered_defects)
    ctx.data["direction_section"] = _direction_section(
        ordered_direction,
        start=len(ordered_defects) + 1,
    )


def _evidence_cell(finding: dict[str, Any]) -> str:
    evidence = finding.get("evidence", [])
    if not isinstance(evidence, list):
        return "—"
    return "<br>".join(_markdown_cell(entry) for entry in evidence) or "—"


def _findings_table(
    findings: list[dict[str, Any]],
    *,
    start: int = 1,
) -> str:
    lines = [
        "| # | Finding | Category | Impact | Effort | Risk | Confidence | Evidence |",
        "|---:|---|---|---|---|---|---|---|",
    ]
    for number, finding in enumerate(findings, start=start):
        lines.append(
            "| "
            + " | ".join(
                (
                    str(number),
                    _markdown_cell(finding.get("title")),
                    _markdown_cell(finding.get("category")),
                    _markdown_cell(finding.get("impact")),
                    _markdown_cell(finding.get("effort")),
                    _markdown_cell(finding.get("risk")),
                    _markdown_cell(finding.get("confidence")),
                    _evidence_cell(finding),
                )
            )
            + " |"
        )
    if not findings:
        lines.append("| — | No vetted defect findings. | — | — | — | — | — | — |")
    return "\n".join(lines)


def _direction_section(
    findings: list[dict[str, Any]],
    *,
    start: int,
    limit: int = 4,
) -> str:
    if not findings:
        return "## Direction\n\nNo grounded direction findings."
    entries: list[str] = []
    for number, finding in enumerate(findings[:limit], start=start):
        entries.append(
            f"### {number}. {_markdown_cell(finding.get('title'))}\n\n"
            f"{_markdown_cell(finding.get('body'))} "
            f"Impact: {_markdown_cell(finding.get('impact'))}; "
            f"effort: {_markdown_cell(finding.get('effort'))}; "
            f"fix risk: {_markdown_cell(finding.get('risk'))}. "
            f"Evidence: {_evidence_cell(finding)}."
        )
    return "## Direction\n\n" + "\n\n".join(entries)


def _parse_selection(raw: str, *, total: int) -> list[int] | None:
    """Parse comma-separated numbers and inclusive ranges."""
    if not raw.strip():
        return []
    selected: list[int] = []
    try:
        for part in raw.split(","):
            token = part.strip()
            if not token:
                return None
            numbers: range | tuple[int, ...]
            if "-" in token:
                bounds = [piece.strip() for piece in token.split("-", 1)]
                start, end = (int(piece) for piece in bounds)
                if start > end:
                    return None
                numbers = range(start, end + 1)
            else:
                numbers = (int(token),)
            for number in numbers:
                if number < 1 or number > total:
                    return None
                if number not in selected:
                    selected.append(number)
    except ValueError:
        return None
    return selected


def _default_selection(defects: list[dict[str, Any]]) -> list[int]:
    return list(range(1, min(5, len(defects)) + 1))


def _selection_prompt(
    defects: list[dict[str, Any]],
    direction: list[dict[str, Any]],
) -> str:
    sections = [
        "Choose findings to turn into plans (comma-separated numbers or ranges).",
        _findings_table(defects),
    ]
    if direction:
        sections.append(
            _direction_section(
                direction,
                start=len(defects) + 1,
                limit=len(direction),
            )
        )
    return "\n\n".join(sections)


async def _step_select(ctx: FlowContext) -> Stop | None:
    """Persist the user's plan selection or the silent unattended default."""
    defects: list[dict[str, Any]] = ctx.data["defects"]
    direction: list[dict[str, Any]] = ctx.data["direction"]
    default_findings = (
        direction if ctx.config.improve_focus == "next" else defects
    )
    default_numbers = _default_selection(default_findings)
    mode = "non-interactive-default" if get_non_interactive() else "interactive"
    selected_numbers = default_numbers

    if not default_findings:
        ctx.data["selected_findings"] = []
        ctx.data["selection_mode"] = mode
        (ctx.data["improve_dir"] / "selected.json").write_text(
            json.dumps(
                _with_artifact_provenance(
                    {"mode": mode, "selected": []},
                    phase=DaydreamPhase.PLAN_WRITE,
                ),
                indent=2,
            )
            + "\n"
        )
        print_success(console, "No vetted defect findings -- done.")
        return None

    if not get_non_interactive():
        default_text = (
            f"1-{len(default_numbers)}" if len(default_numbers) > 1 else "1"
        )
        prompt = _selection_prompt(defects, direction)
        raw = agent.prompt_user(console, prompt, default=default_text)
        parsed = _parse_selection(
            raw,
            total=len(defects) + len(direction),
        )
        if parsed is None:
            raw = agent.prompt_user(
                console,
                "Invalid selection; try once more",
                default=default_text,
            )
            parsed = _parse_selection(
                raw,
                total=len(defects) + len(direction),
            )
        selected_numbers = parsed if parsed is not None else default_numbers

    selectable = [*defects, *direction]
    selected = [
        selectable[number - 1]["fingerprint"] for number in selected_numbers
    ]
    ctx.data["selected_findings"] = [
        selectable[number - 1] for number in selected_numbers
    ]
    ctx.data["selection_mode"] = mode
    (ctx.data["improve_dir"] / "selected.json").write_text(
        json.dumps(
            _with_artifact_provenance(
                {"mode": mode, "selected": selected},
                phase=DaydreamPhase.PLAN_WRITE,
            ),
            indent=2,
        )
        + "\n"
    )
    return None


def _plan_slug(value: Any, title: Any) -> str:
    candidate = str(value or "")
    if _PLAN_SLUG.fullmatch(candidate):
        return candidate
    derived = re.sub(r"[^a-z0-9]+", "-", str(title or "").lower()).strip("-")
    return derived[:60].rstrip("-") or "plan"


def _verification_commands(recon: dict[str, Any]) -> list[dict[str, Any]]:
    raw_commands = recon.get("commands")
    if not isinstance(raw_commands, list):
        return []
    return [
        command
        for command in raw_commands
        if isinstance(command, dict)
    ]


def _legacy_verification_commands(recon: dict[str, Any]) -> list[str]:
    """Return the documented prompt-override compatibility view."""
    return [
        command
        for record in _verification_commands(recon)
        if isinstance((command := record.get("command")), str)
    ]


def _description_finding(description: str) -> dict[str, Any]:
    """Represent a user-requested change as one plan-writer input."""
    return {
        "title": description,
        "category": "requested",
        "path": "",
        "line": None,
        "body": (
            "Investigate the repository and write a single implementation "
            f"plan for this requested change: {description}"
        ),
        "impact": "MED",
        "effort": "M",
        "risk": "MED",
        "confidence": "HIGH",
        "evidence": [],
        "fingerprint": compute_fingerprint(
            "",
            description,
            "User-requested improve plan",
        ),
    }


async def _step_write_plans(ctx: FlowContext) -> None:
    """Write selected findings as host-stamped, reconciling handoff plans."""
    description = ctx.config.improve_plan_description
    if description is not None:
        selected = [_description_finding(description)]
        ctx.data["selected_findings"] = selected
        ctx.data["selection_mode"] = "description"
    else:
        selected = ctx.data["selected_findings"]
    backend = ctx.backend_for("plan_write")
    recorder = get_current_recorder()
    limiter = anyio.CapacityLimiter(
        effective_fanout_concurrency(PLAN_WRITE_MAX_CONCURRENCY, backend)
    )
    authoring_diagnostics: list[tuple[int, dict[str, Any]]] = []
    plans_dir = ctx.work.repo / "daydream_plans"
    try:
        planned_at = git_ops.head_sha(ctx.work.repo)
    except git_ops.GitError:
        planned_at = ctx.work.head_sha

    session = PlanWriteSession(
        plans_dir,
        planned_at=planned_at,
        non_interactive_default=(
            ctx.data["selection_mode"] == "non-interactive-default"
        ),
        run_session_id=_run_session_id(),
    )
    # Numbers are claimed here, in selection order, before any writer runs, so
    # a plan's number never depends on which writer finishes first.
    reservations = session.reserve(selected)
    pending = {reservation.index for reservation in reservations}
    total = sum(
        1 for reservation in reservations if reservation.number is not None
    )
    landed = 0
    announced: set[int] = set()

    def _land(index: int, record: dict[str, Any]) -> None:
        """Persist one writer's result and report the movement."""
        nonlocal landed
        pending.discard(index)
        outcome = session.commit(reservations[index], record)
        if reservations[index].number is None:
            return
        landed += 1
        if outcome.status == "written" and outcome.number is not None:
            announced.add(outcome.number)
            print_success(
                console,
                f"Plan {outcome.number:03d} written to "
                f"daydream_plans/{outcome.path} ({landed}/{total}).",
            )
        elif outcome.status == "deferred":
            print_info(
                console,
                f"Plan {outcome.number:03d} for {outcome.title} is held until "
                f"its dependencies land ({landed}/{total}).",
            )
        else:
            print_info(
                console,
                f"No plan file written for {outcome.title} "
                f"({landed}/{total}).",
            )

    async with anyio.create_task_group() as task_group:
        for selection_index, finding in enumerate(selected):
            if reservations[selection_index].number is None:
                _land(selection_index, {"finding": finding})
                continue
            descriptor = (
                f"plan-{_plan_slug('', finding.get('title'))}-"
                f"{selection_index + 1:03d}"
            )
            attempt = {
                "descriptor": descriptor,
                "backend": type(backend).__name__,
                "model": getattr(backend, "model", "unknown-model"),
            }
            try:
                prompt = ctx.registry.prompt("plan-writer")(
                    finding=finding,
                    recon_summary=json.dumps(
                        ctx.data["recon"],
                        sort_keys=True,
                    ),
                    verification_commands=_legacy_verification_commands(
                        ctx.data["recon"]
                    ),
                    cwd=ctx.work.repo,
                )
            except Exception:  # noqa: BLE001 - isolate each plan safely
                _land(
                    selection_index,
                    {
                        "finding": finding,
                        "_attempt": {
                            **attempt,
                            "received_result": None,
                            "errors": ("PROMPT_CONSTRUCTION_FAILED",),
                        },
                        "error": True,
                    },
                )
                continue

            async def _task(
                current: dict[str, Any] = finding,
                current_index: int = selection_index,
                task_prompt: str = prompt,
                task_descriptor: str = descriptor,
                task_attempt: dict[str, Any] = attempt,
            ) -> None:
                async def _call(
                    generation_prompt: str,
                ) -> tuple[Any, str | None]:
                    async with phase_scope(DaydreamPhase.PLAN_WRITE):
                        output, _, aborted_reason = await run_agent(
                            backend,
                            ctx.work.repo,
                            generation_prompt,
                            phase=DaydreamPhase.PLAN_WRITE,
                            output_schema=PLAN_AUTHOR_SCHEMA,
                            read_only=True,
                            persist_session=False,
                        )
                    return output, aborted_reason

                async def _call_once_retried(
                    generation_prompt: str,
                ) -> tuple[Any, str | None]:
                    """Absorb one transport crash; a second one is terminal.

                    The retry replaces only the crashed generation, so a crash
                    on the repair never restarts generation 0, and the
                    two-generation authoring-repair budget is unchanged.
                    """
                    try:
                        return await _call(generation_prompt)
                    except Exception:
                        return await _call(generation_prompt)

                async def _generate() -> dict[str, Any]:
                    current_prompt = task_prompt
                    for generation_index in range(2):
                        output, aborted_reason = await _call_once_retried(
                            current_prompt
                        )
                        output = _redact_model_value(output)
                        if aborted_reason is not None:
                            abort_code = {
                                "tool_call_budget_exceeded": (
                                    "TOOL_CALL_BUDGET_EXCEEDED"
                                ),
                                "wall_budget_exceeded": (
                                    "WALL_BUDGET_EXCEEDED"
                                ),
                            }.get(
                                aborted_reason,
                                (
                                    "TOOL_VETOED"
                                    if aborted_reason.startswith(
                                        "tool_vetoed:"
                                    )
                                    else "AGENT_ABORTED"
                                ),
                            )
                            return {
                                "finding": current,
                                "_attempt": {
                                    **task_attempt,
                                    "received_result": output,
                                    "errors": (abort_code,),
                                },
                                "error": True,
                            }
                        if isinstance(output, dict):
                            assembled, issues = assemble_plan(
                                output,
                                repo=ctx.work.repo,
                                recon_commands=_verification_commands(
                                    ctx.data["recon"]
                                ),
                            )
                        else:
                            assembled = None
                            issues = (
                                AssemblyIssue(
                                    code="NO_STRUCTURED_OBJECT",
                                    pointer="/",
                                ),
                            )
                        if assembled is not None and not issues:
                            return {
                                "finding": current,
                                "_attempt": task_attempt,
                                **assembled,
                            }
                        rendered_issues = tuple(
                            render_issue(issue) for issue in issues
                        )
                        stage = (
                            "authoring"
                            if isinstance(output, dict)
                            else "transport"
                        )
                        if generation_index == 0:
                            authoring_diagnostics.append(
                                (
                                    current_index,
                                    _attempt_diagnostic(
                                        finding=current,
                                        attempt=task_attempt,
                                        received=output,
                                        disposition="retried",
                                        stage=stage,
                                        errors=rendered_issues,
                                    ),
                                )
                            )
                            current_prompt = (
                                build_plan_writer_repair_prompt(
                                    task_prompt,
                                    issues,
                                )
                            )
                            continue
                        if not isinstance(output, dict):
                            return {
                                "finding": current,
                                "_attempt": {
                                    **task_attempt,
                                    "received_result": output,
                                    "errors": ("NO_STRUCTURED_OBJECT",),
                                },
                                "error": True,
                            }
                        return {
                            "finding": current,
                            "_attempt": {
                                **task_attempt,
                                "received_result": output,
                                "errors": rendered_issues,
                                "validation": True,
                            },
                            "error": True,
                        }
                    return {"finding": current}

                async with limiter:
                    async with maybe_fork(recorder, task_descriptor):
                        try:
                            record = await _generate()
                        except Exception as exc:  # noqa: BLE001 - isolate each plan safely
                            category = getattr(exc, "category", "UNKNOWN")
                            stable_category = (
                                category
                                if category
                                in {
                                    "RATE_LIMIT",
                                    "TIMEOUT",
                                    "STREAM_DROP",
                                    "PROCESS_EXIT",
                                    "AUTH_CONFIG",
                                    "UNKNOWN",
                                }
                                else "UNKNOWN"
                            )
                            record = {
                                "finding": current,
                                "_attempt": {
                                    **task_attempt,
                                    "received_result": None,
                                    "errors": (stable_category,),
                                },
                                "error": True,
                            }
                # Each writer's plan reaches disk here, while its slower
                # siblings are still running.
                _land(current_index, record)

            task_group.start_soon(_task)

    for index in sorted(pending):
        _land(index, {"finding": selected[index]})
    result = session.finish()
    for entry in result["written"]:
        if entry["number"] not in announced:
            print_success(
                console,
                f"Plan {entry['number']:03d} written to "
                f"daydream_plans/{entry['path']}.",
            )
    record_plan_write_diagnostics(
        plan_write_diagnostics_path(ctx.data["improve_dir"]),
        [
            *(
                entry
                for _, entry in sorted(
                    authoring_diagnostics,
                    key=lambda item: item[0],
                )
            ),
            *result["diagnostics"],
        ],
        artifact_provenance=_artifact_provenance(
            phase=DaydreamPhase.PLAN_WRITE
        ),
    )
    ctx.data["plan_write"] = result
    ctx.data["plan_exit_code"] = (
        1 if result["failed"] and not result["written"] else 0
    )
    if result["skipped"]:
        print_warning(
            console,
            f"Skipped {len(result['skipped'])} already planned or rejected finding(s).",
        )
    if result["failed"]:
        for diagnostic in result["diagnostics"]:
            if diagnostic["disposition"] != "blocked":
                continue
            reasons = ", ".join(
                f"{error['code']} at {error['pointer']}"
                + (f" ({error['detail']})" if error.get("detail") else "")
                for error in diagnostic["errors"]
            )
            print_warning(
                console,
                "Plan blocked for "
                f"{diagnostic['finding']['title']}: {reasons}.",
            )
        print_warning(
            console,
            f"Plan writing failed for {len(result['failed'])} finding(s).",
        )


def _render_report(
    services: list[Service],
    all_services: list[Service],
    stacks: list[StackAssignment],
    audit: dict[str, Any],
    findings: list[dict[str, Any]],
    discarded_no_evidence: int,
    dropped_low_confidence: int,
    dropped_by_cap: int,
    previously_rejected: int,
    vet_rejected: int,
    findings_table: str,
    direction_section: str,
    effort: str,
    scope: str | None,
    plan_write: dict[str, list[dict[str, Any]]],
    partitions: list[Partition],
    partitions_not_audited: list[Partition],
    groups: list[PartitionGroup],
) -> str:
    service_lines = (
        "\n".join(
            f"- **{service.name}** — `{service.root.as_posix()}`"
            for service in services
        )
        or "- No service roots detected."
    )
    top_offender_lines = _top_offender_lines(findings)
    stack_lines = (
        "\n".join(f"- **{stack.stack_name}**" for stack in stacks)
        or "- No stacks detected."
    )
    roots_by_group = {group.name: _group_roots_cell(group) for group in groups}
    failures = audit.get("failed", {})
    failed_assignment_lines = (
        "\n".join(
            f"- **{assignment.replace(':', ' / ')}** "
            f"({roots_by_group.get(assignment.partition(':')[2], 'unknown group')})"
            f" — {reason}"
            for assignment, reason in failures.items()
        )
        or "- None."
    )
    not_audited_lines = (
        (
            "- Partitions not audited (reason: group-ceiling; raise "
            "`max-partition-groups` to include them):\n"
            + "\n".join(
                f"  - **{partition.name}** — `{partition.root}/` "
                f"({len(partition.files)} files)"
                for partition in partitions_not_audited
            )
        )
        if partitions_not_audited
        else f"- All {len(partitions)} partitions were audited."
    )
    tier_bound = {
        "quick": (
            "Recon hotspots only; categories outside correctness, security, "
            "and tests were not audited."
        ),
        "standard": (
            "Coverage was hotspot-weighted across key packages; the partition "
            "ledger below is authoritative for what was reached."
        ),
        "deep": (
            "Every partitioned package was in scope; untracked files are never "
            "audited, and the partition ledger below is authoritative for what "
            "was reached."
        ),
    }[effort]
    if scope:
        audited_roots = {service.root for service in services}
        unaudited = [
            service for service in all_services if service.root not in audited_roots
        ]
        unaudited_lines = (
            "\n".join(
                f"  - **{service.name}** — `{service.root.as_posix()}`"
                for service in unaudited
            )
            or "  - No other detected service directories."
        )
        scope_statement = (
            f"Service scope slicing was limited to `{scope}`. The following "
            "detected services/directories were not audited:\n"
            f"{unaudited_lines}"
        )
    else:
        scope_statement = "No explicit service scope slicing was requested."
    plan_lines = (
        f"- Plans written: {len(plan_write['written'])}\n"
        f"- Findings skipped as already planned or rejected: "
        f"{len(plan_write['skipped'])}\n"
        f"- Plans blocked by plan-writing failure: {len(plan_write['failed'])}\n"
        f"{_blocked_plan_attempt_lines(plan_write)}"
    )
    return (
        "# Improve Report\n\n"
        "## Findings\n\n"
        f"{findings_table}\n\n"
        f"{direction_section}\n\n"
        "## Services\n\n"
        f"{service_lines}\n\n"
        "## Top offenders\n\n"
        f"{top_offender_lines}\n\n"
        "## Stacks\n\n"
        f"{stack_lines}\n\n"
        "## What ran\n\n"
        "- Read-only repository reconnaissance\n"
        f"- Read-only audits across {len(audit.get('categories_run', []))} categories\n\n"
        "## What was not audited\n\n"
        f"- {tier_bound}\n"
        f"- {scope_statement}\n\n"
        f"{not_audited_lines}\n\n"
        "### Failed audit assignments\n\n"
        f"{failed_assignment_lines}\n\n"
        "## Audit filtering\n\n"
        f"- Findings without `path:line` evidence discarded: {discarded_no_evidence}\n"
        f"- Findings rejected during vetting: {vet_rejected}\n"
        f"- Non-HIGH-confidence findings dropped by tier: {dropped_low_confidence}\n"
        f"- Lowest-leverage findings dropped by tier cap: {dropped_by_cap}\n"
        f"- Previously rejected findings suppressed: {previously_rejected}\n"
        "\n## Plan writing\n\n"
        f"{plan_lines}"
    )


def _blocked_plan_attempt_lines(
    plan_write: dict[str, list[dict[str, Any]]],
) -> str:
    blocked = [
        diagnostic
        for diagnostic in plan_write.get("diagnostics", [])
        if diagnostic.get("disposition") == "blocked"
    ]
    if not blocked:
        return ""
    lines = ["- Blocked attempt details:"]
    for diagnostic in blocked:
        finding = diagnostic["finding"]
        errors = ", ".join(
            f"`{error['code']}` at `{error['pointer']}`"
            + (f" ({error['detail']})" if error.get("detail") else "")
            for error in diagnostic["errors"]
        )
        lines.append(
            f"  - **{_markdown_cell(finding['title'])}** "
            f"(`{finding['fingerprint'][:12]}`) — "
            f"{diagnostic['stage']}: {errors}"
        )
    lines.append(
        "  - See `.daydream/improve/plan-write-diagnostics.json` for "
        "sanitized attempt metadata."
    )
    return "\n".join(lines) + "\n"


def _top_offender_lines(findings: list[dict[str, Any]]) -> str:
    totals: dict[str, float] = {}
    for finding in findings:
        # A finding outside every detected service is still located: its
        # partition names the tree it came from.
        raw_owners: list[Any] = []
        for key in ("services", "partitions"):
            value = finding.get(key)
            if isinstance(value, list):
                raw_owners.extend(value)
        raw_owners.append(finding.get("partition"))
        owners = [item for item in raw_owners if isinstance(item, str) and item]
        for owner in dict.fromkeys(owners):
            totals[owner] = totals.get(owner, 0.0) + leverage_score(finding)
    if not totals:
        return "- No vetted findings were assigned to a detected service."
    return "\n".join(
        f"- **{service}** — summed leverage {total:.2f}"
        for service, total in sorted(
            totals.items(),
            key=lambda item: (-item[1], item[0]),
        )
    )


async def _step_report(ctx: FlowContext) -> Stop | None:
    """Render the improve report for reconnaissance and audit coverage."""
    if ctx.config.improve_plan_description is not None:
        plan_write = ctx.data["plan_write"]
        report_path(ctx.data["improve_dir"]).write_text(
            _report_with_provenance(
                "# Improve Report\n\n"
                "## What ran\n\n"
                "- Read-only repository reconnaissance\n"
                "- Targeted investigation and plan writing from the supplied description\n\n"
                "## Outcome\n\n"
                f"- Plans written: {len(plan_write['written'])}\n"
                f"- Requests skipped as already planned: {len(plan_write['skipped'])}\n"
                f"- Plan-writing failures: {len(plan_write['failed'])}\n"
                f"{_blocked_plan_attempt_lines(plan_write)}"
            ),
            encoding="utf-8",
        )
        if ctx.data["plan_exit_code"]:
            return Stop(ctx.data["plan_exit_code"])
        print_success(console, "Description plan complete.")
        return None

    report_path(ctx.data["improve_dir"]).write_text(
        _report_with_provenance(
            _render_report(
                ctx.data["services"],
                ctx.data["all_services"],
                ctx.data["stacks"],
                ctx.data["audit"],
                ctx.data["vetted"]["findings"],
                ctx.data["audit_discarded_no_evidence"],
                ctx.data["audit_dropped_low_confidence"],
                ctx.data["audit_dropped_by_cap"],
                ctx.data["previously_rejected"],
                ctx.data["vet_rejected"],
                ctx.data["findings_table"],
                ctx.data["direction_section"],
                ctx.config.improve_effort,
                ctx.config.improve_scope,
                ctx.data["plan_write"],
                ctx.data["partitions"],
                ctx.data["partitions_not_audited"],
                ctx.data["partition_groups"],
            )
        )
    )
    if ctx.data["plan_exit_code"]:
        print_error(
            console,
            "Improve planning failed",
            f"{len(ctx.data['plan_write']['failed'])} selected plan(s) failed.",
        )
        return Stop(ctx.data["plan_exit_code"])
    print_success(
        console,
        "Improve audit complete: "
        f"{len(ctx.data['services'])} services, "
        f"{len(ctx.data['stacks'])} stacks, "
        f"{len(ctx.data['vetted']['findings'])} vetted findings.",
    )
    return None


def _is_audit_run(ctx: FlowContext) -> bool:
    return ctx.config.improve_plan_description is None


STEPS: tuple[FlowStep, ...] = (
    FlowStep(name="recon", run=_step_recon),
    FlowStep(name="audit", run=_step_audit, enabled=_is_audit_run),
    FlowStep(name="vet", run=_step_vet, enabled=_is_audit_run),
    FlowStep(name="select-plans", run=_step_select, enabled=_is_audit_run),
    FlowStep(
        name="write-plans",
        run=_step_write_plans,
        config_phase="plan_write",
    ),
    FlowStep(
        name="improve-report",
        run=_step_report,
        config_phase="recon",
    ),
)
