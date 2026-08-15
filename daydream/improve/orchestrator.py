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
from daydream.deep.orchestrator import _diff_changed_files
from daydream.deep.prompts import _DIFF_BLOCK_SPLIT, _diff_block_path
from daydream.exploration_runner import repo_scan
from daydream.extensions.api import FlowStep, Stop
from daydream.improve.artifacts import (
    coverage_path,
    plan_write_diagnostics_path,
    published_issues_path,
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
    path_is_confined,
    valid_repository_file_path,
    validate_host_commands,
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
    load_rejections,
    plan_slug,
    prune_stale_reanchor_worktrees,
    reanchored_plan_rows,
    record_plan_write_diagnostics,
    record_rejections,
)
from daydream.improve.prioritize import (
    aggregate_cross_service,
    leverage_score,
    order_by_leverage,
)
from daydream.improve.prompts import (
    AUDIT_FINDINGS_SCHEMA,
    CHANGE_SHAPES,
    MAINTENANCE_SIGNALS,
    PLAN_AUTHOR_SCHEMA,
    RECON_COMMAND_CONTRACT_BULLET,
    VET_SCHEMA,
    build_plan_writer_repair_prompt,
)
from daydream.improve.publish import ImprovePublishError, IssuePublisher
from daydream.improve.render import markdown_cell
from daydream.improve.repo_commands import enumerate_repository_commands
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
from daydream.workspace import prune_stale_audit_worktrees

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

_EVIDENCE_LOCATION = re.compile(r"^`?(.+?):(\d+)(?::(\d+))?(?:`|\b)")


def _audit_repo(ctx: FlowContext) -> Path:
    """Return the detached audit worktree used as the model cwd for improve turns.

    The runner opens one audit worktree per improve run (see
    :func:`daydream.workspace.open_audit_workspace`) and stores its path on
    ``ctx.data["audit_repo"]``. Every advisory model turn (recon/audit/vet/
    plan-write) runs with this path as its ``cwd``; the target worktree is never
    a model cwd, so a model commit can never reach the target's HEAD, named
    refs, or staged index.

    Raises:
        KeyError: If the runner did not open an audit workspace (a wiring bug —
            fail loud, never fall back to ``ctx.work.repo``).
    """
    return Path(ctx.data["audit_repo"])

_PROVENANCE_VALUES = {"introduced", "inherited"}
_MAINTENANCE_SIGNALS = set(MAINTENANCE_SIGNALS)
_CHANGE_SHAPES = set(CHANGE_SHAPES)
_REUSE_TARGET = re.compile(r"^(?:repo:[^#\s]+#[^\s]+|stdlib:[^\s]+|dep:[^:\s]+:[^\s]+)$")


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
        f"- {service.name}: {(repo / service.root).as_posix()}" for service in services
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


def _command_enumeration_directories(
    repo: Path,
    services: list[Service],
    groups: list[PartitionGroup],
) -> list[str]:
    """Return the repository-relative roots to enumerate commands under."""
    directories = ["."]
    seen = {".", ""}
    candidates = {
        *(service.root.as_posix() for service in services),
        *(root for group in groups for root in group.roots),
    }
    for candidate in sorted(candidates):
        normalized = candidate.removeprefix("./").rstrip("/")
        if normalized in seen:
            continue
        seen.add(normalized)
        if (repo / normalized).is_dir():
            directories.append(normalized)
    return directories


def _host_enumerated_commands(
    repo: Path,
    services: list[Service],
    groups: list[PartitionGroup],
    *,
    model_commands: list[dict[str, Any]],
) -> tuple[int, list[dict[str, Any]], list[str]]:
    """Derive the Make/manifest commands the recon model is told to skip.

    Returns ``(candidates, validated, errors)``. Enumeration is read-only and
    never fatal: a failure leaves the run on the model's commands alone and is
    surfaced as a warning plus a recorded rejection code.
    """
    try:
        enumerated = enumerate_repository_commands(
            repo,
            directories=_command_enumeration_directories(repo, services, groups),
            reserved_ids=[command["id"] for command in model_commands],
        )
    except Exception as exc:
        print_warning(
            console,
            "Host command enumeration failed; continuing with the recon "
            "model's commands only. "
            f"{type(exc).__name__}: {exc}",
        )
        return 0, [], ["HOST_COMMAND_ENUMERATION_FAILED@/host_commands"]
    already_cited = {
        (command["command"], command["working_directory"])
        for command in model_commands
    }
    candidates = [
        command
        for command in enumerated
        if (command["command"], command["working_directory"]) not in already_cited
    ]
    validated, errors = validate_host_commands(candidates, repo=repo)
    return len(candidates), validated, errors


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
            branch_diff, branch_files = _restrict_diff_to_services(branch_diff, services)
    if branch_focus:
        services = _services_for_files(services, tuple(branch_files))

    ctx.data["branch_diff"] = branch_diff
    ctx.data["branch_files"] = branch_files

    stacks: list[StackAssignment] = []
    partitions: list[Partition] = []
    groups: list[PartitionGroup] = []
    skipped: list[Partition] = []
    if not description_mode:
        # Availability is resolved once in runner.run and threaded via config;
        # None flows through to detect_stacks' optimistic default.
        tracked = branch_files if branch_focus else git_ops.ls_files(target)
        stacks = detect_stacks(
            tracked,
            skill_availability=ctx.config.skill_availability,
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
    audit_repo = _audit_repo(ctx)
    async with phase_scope(DaydreamPhase.RECON):
        exploration = await repo_scan(backend, audit_repo)
        recon, _, _ = await run_agent(
            backend,
            audit_repo,
            _build_recon_prompt(
                audit_repo, services, groups, exploration.to_prompt_section()
            ),
            phase=DaydreamPhase.RECON,
            output_schema=RECON_SCHEMA,
            read_only=True,
            persist_session=False,
        )

    total_candidates = 0
    valid_commands: list[dict[str, Any]] = []
    command_errors: list[str] = []
    model_fields: dict[str, Any] = {}
    safe_recon = _redact_model_value(recon)
    if isinstance(safe_recon, dict):
        raw_commands = safe_recon.get("commands")
        total_candidates = len(raw_commands) if isinstance(raw_commands, list) else 0
        valid_commands, command_errors = validate_recon_commands(
            safe_recon,
            repo=target,
        )
        model_fields = {
            field: value
            if isinstance((value := safe_recon.get(field)), list)
            and all(isinstance(item, str) for item in value)
            else []
            for field in ("languages", "conventions", "intent_docs")
        }
    else:
        command_errors = ["RECON_CONTAINER_INVALID@/"]

    # Make targets and manifest scripts are host-enumerated, not model-cited:
    # the recon prompt forbids reporting them, so without this the only
    # verification command a Makefile-driven repository has never lands.
    host_candidates, host_commands, host_errors = _host_enumerated_commands(
        target,
        services,
        groups,
        model_commands=valid_commands,
    )
    total_candidates += host_candidates
    valid_commands = [*valid_commands, *host_commands]
    command_errors = [*command_errors, *host_errors]

    recon_data: dict[str, Any] = {
        "artifact_type": "daydream.improve-recon",
        "artifact_provenance": _artifact_provenance(phase=DaydreamPhase.RECON),
        **model_fields,
        "commands": valid_commands,
        "command_rejections": [
            {
                "code": error.partition("@")[0],
                "pointer": error.partition("@")[2] or "/",
            }
            for error in command_errors
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


def _restrict_diff_to_services(
    diff: str, services: list[Service]
) -> tuple[str, list[str]]:
    """Keep only the diff blocks whose file lives under a scoped service root.

    Under ``--scope`` the branch diff must not leak changes from out-of-scope
    services into audit and vetting prompts, so both the diff text and the
    derived changed-file list are narrowed to the scoped roots.
    """
    roots = tuple(service.root.as_posix() for service in services)
    selected: list[str] = []
    files: list[str] = []
    for block in _DIFF_BLOCK_SPLIT.split(diff):
        path = _diff_block_path(block)
        if path is None:
            continue
        if any(path == root or path.startswith(f"{root}/") for root in roots):
            selected.append(block)
            if path not in files:
                files.append(path)
    return "".join(selected), files


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


def _evidence_paths(
    finding: dict[str, Any], *, repo: Path
) -> list[str] | None:
    evidence = finding.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        return None
    paths: list[str] = []
    for entry in evidence:
        if not isinstance(entry, str):
            return None
        match = _EVIDENCE_LOCATION.match(entry.strip())
        if match is None:
            return None
        path = match.group(1).strip("`")
        if (
            not valid_repository_file_path(path)
            or not path_is_confined(repo, path)
        ):
            return None
        candidate = Path(path)
        if candidate.is_absolute():
            return None
        try:
            resolved = (repo / candidate).resolve()
            resolved.relative_to(repo.resolve())
        except (OSError, ValueError):
            return None
        if not resolved.is_file():
            return None
        try:
            line_count = len(resolved.read_text(errors="replace").splitlines())
        except OSError:
            return None
        start_line = int(match.group(2))
        end_line = int(match.group(3) or start_line)
        if start_line < 1 or end_line < start_line or end_line > line_count:
            return None
        paths.append(path)
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
    *,
    repo: Path,
) -> dict[str, Any] | None:
    evidence_paths = _evidence_paths(finding, repo=repo)
    if evidence_paths is None:
        return None
    stamped = dict(finding)
    raw_signals = stamped.get("maintenance_signals")
    stamped["maintenance_signals"] = (
        list(dict.fromkeys(signal for signal in raw_signals if signal in _MAINTENANCE_SIGNALS))
        if isinstance(raw_signals, list)
        else []
    )
    if stamped.get("change_shape") not in _CHANGE_SHAPES:
        stamped["change_shape"] = "unknown"
    reuse_target = stamped.get("reuse_target")
    if not isinstance(reuse_target, str) or _REUSE_TARGET.fullmatch(reuse_target) is None:
        stamped["reuse_target"] = None
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


def _correct_primary_evidence_location(
    finding: dict[str, Any],
    *,
    path: str,
    line: int | None,
) -> None:
    """Align the primary evidence citation with a vetted path correction."""
    evidence = finding.get("evidence")
    if not isinstance(evidence, list):
        return
    evidence = list(evidence)
    finding["evidence"] = evidence
    for index, entry in enumerate(evidence):
        if not isinstance(entry, str):
            continue
        match = _EVIDENCE_LOCATION.match(entry.strip())
        if match is None:
            continue
        evidence_line = line if line is not None else int(match.group(2))
        location = f"{path}:{evidence_line}"
        if entry.lstrip().startswith("`"):
            location = f"`{location}`"
        evidence[index] = location + entry.strip()[match.end() :]
        return


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
                    async with maybe_fork(recorder, descriptor):
                        try:
                            output, _, _ = await run_agent(
                                backend,
                                _audit_repo(ctx),
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
            stamped = _stamp_finding(
                finding,
                assignment.category,
                services,
                partitions,
                repo=ctx.work.repo,
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
    repo: Path | None = None,
    default_provenance: str | None = None,
    services: list[Service] | None = None,
    partitions: list[Partition] | None = None,
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
        # The vet contract uses human-readable severity names while the
        # prioritizer's axes use the audit vocabulary. Normalize at the host
        # boundary so aggregation cannot silently promote every vetted item to
        # its conservative fallback.
        severity = corrected.get("severity")
        if isinstance(severity, str):
            corrected["severity"] = {
                "high": "HIGH",
                "medium": "MED",
                "low": "LOW",
            }.get(severity.lower(), severity)
        # Unlike the other corrected fields, ``None`` is a meaningful vet
        # correction here: it explicitly retracts an audit-time reuse target.
        if "reuse_target" in verdict:
            corrected["reuse_target"] = verdict["reuse_target"]
        location_corrected = verdict.get("path") is not None or verdict.get("line") is not None
        if location_corrected:
            _correct_primary_evidence_location(
                corrected,
                path=str(corrected.get("path", "")),
                line=corrected.get("line"),
            )
        if default_provenance is not None and corrected.get("provenance") not in _PROVENANCE_VALUES:
            corrected["provenance"] = default_provenance
        if location_corrected:
            restamped = _stamp_finding(
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
                    async with maybe_fork(recorder, descriptor):
                        try:
                            output, _, _ = await run_agent(
                                backend,
                                _audit_repo(ctx),
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
                            repo=ctx.work.repo,
                            default_provenance=(
                                "inherited" if branch_focus else None
                            ),
                            services=ctx.data["services"],
                            partitions=ctx.data["partitions"],
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
    ordered_defects = order_by_leverage(findings)
    vetted = _with_artifact_provenance(
        {
            "findings": ordered_defects,
            "defects": ordered_defects,
        },
        phase=DaydreamPhase.VET,
    )
    vetted_findings_path(directory).write_text(json.dumps(vetted, indent=2) + "\n")
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


def _evidence_cell(finding: dict[str, Any]) -> str:
    evidence = finding.get("evidence", [])
    if not isinstance(evidence, list):
        return "—"
    return "<br>".join(markdown_cell(entry) for entry in evidence) or "—"


def _findings_table(
    findings: list[dict[str, Any]],
    *,
    start: int = 1,
) -> str:
    lines = [
        "| # | Finding | Category | Change | Impact | Effort | Risk | Confidence | Evidence |",
        "|---:|---|---|---|---|---|---|---|---|",
    ]
    for number, finding in enumerate(findings, start=start):
        lines.append(
            "| "
            + " | ".join(
                (
                    str(number),
                    markdown_cell(finding.get("title")),
                    markdown_cell(finding.get("category")),
                    markdown_cell(finding.get("change_shape", "unknown")),
                    markdown_cell(finding.get("impact")),
                    markdown_cell(finding.get("effort")),
                    markdown_cell(finding.get("risk")),
                    markdown_cell(finding.get("confidence")),
                    _evidence_cell(finding),
                )
            )
            + " |"
        )
    if not findings:
        lines.append("| — | No vetted findings. | — | — | — | — | — | — | — |")
    return "\n".join(lines)


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


def _automatic_issue_publishing(ctx: FlowContext) -> bool:
    """Return whether this Improve run is configured to publish plans."""
    file_config = ctx.config.file_config or DaydreamFileConfig()
    return bool(getattr(file_config, "improve_github_publish_issues", False))


def _selection_prompt(
    findings: list[dict[str, Any]],
) -> str:
    return "\n\n".join(
        [
            "Choose findings to turn into plans (comma-separated numbers or ranges).",
            _findings_table(findings),
        ]
    )


async def _step_select(ctx: FlowContext) -> Stop | None:
    """Persist the user's plan selection or the silent unattended default."""
    default_findings: list[dict[str, Any]] = ctx.data["defects"]
    publish_all = get_non_interactive() and _automatic_issue_publishing(ctx)
    default_numbers = list(range(1, len(default_findings) + 1)) if publish_all else _default_selection(default_findings)
    mode = (
        "automatic-publishing"
        if publish_all
        else ("non-interactive-default" if get_non_interactive() else "interactive")
    )
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
        default_text = f"1-{len(default_numbers)}" if len(default_numbers) > 1 else "1"
        prompt = _selection_prompt(default_findings)
        raw = agent.prompt_user(console, prompt, default=default_text)
        parsed = _parse_selection(
            raw,
            total=len(default_findings),
        )
        if parsed is None:
            raw = agent.prompt_user(
                console,
                "Invalid selection; try once more",
                default=default_text,
            )
            parsed = _parse_selection(
                raw,
                total=len(default_findings),
            )
        selected_numbers = parsed if parsed is not None else default_numbers

    selectable = default_findings
    selected = [selectable[number - 1]["fingerprint"] for number in selected_numbers]
    ctx.data["selected_findings"] = [selectable[number - 1] for number in selected_numbers]
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
        "maintenance_signals": [],
        "change_shape": "unknown",
        "reuse_target": None,
        "fingerprint": compute_fingerprint(
            "",
            description,
            "User-requested improve plan",
        ),
    }


def _expected_plan_fingerprints(finding: dict[str, Any]) -> list[str]:
    members = finding.get("member_fingerprints")
    if isinstance(members, list):
        valid = [item for item in members if isinstance(item, str) and item]
        if valid:
            return valid
    fingerprint = finding.get("fingerprint")
    return [fingerprint] if isinstance(fingerprint, str) and fingerprint else []


async def _step_write_plans(ctx: FlowContext) -> None:
    """Write selected findings as host-stamped, reconciling handoff plans."""
    prune_stale_reanchor_worktrees(ctx.work.repo)
    prune_stale_audit_worktrees(ctx.work.repo)
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
        non_interactive_default=(ctx.data["selection_mode"] in {"non-interactive-default", "automatic-publishing"}),
        run_session_id=_run_session_id(),
    )
    # Numbers are claimed here, in selection order, before any writer runs, so
    # a plan's number never depends on which writer finishes first.
    reservations = session.reserve(selected)
    pending = {reservation.index for reservation in reservations}
    total = sum(1 for reservation in reservations if reservation.number is not None)
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
            path = outcome.path or ""
            landing = (
                path if Path(path).is_absolute() else f"daydream_plans/{path}"
            )
            print_success(
                console,
                f"Plan {outcome.number:03d} written to {landing} "
                f"({landed}/{total}).",
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
                f"plan-{plan_slug(finding.get('title'))}-"
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
                    cwd=_audit_repo(ctx),
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
                            _audit_repo(ctx),
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
                    """Absorb one non-retryable crash; a second one is terminal.

                    ``run_agent`` already spent its per-backend attempt budget on
                    retryable backend errors, so re-calling for one of those would
                    just duplicate an exhausted budget — it propagates. This is
                    the backstop for a crash ``run_agent`` never retried (e.g. a
                    process exit). A deliberate tool-supervisor veto is a policy
                    stop, not an API failure, so it propagates unretried. The
                    retry replaces only the crashed generation, so it never
                    restarts generation 0, and the two-generation
                    authoring-repair budget is unchanged.
                    """
                    try:
                        return await _call(generation_prompt)
                    except agent._ToolSupervisorFailure:
                        raise
                    except Exception as exc:
                        if getattr(exc, "retryable", False):
                            raise
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
                                recon_commands=_verification_commands(ctx.data["recon"]),
                                expected_fingerprints=(_expected_plan_fingerprints(current)),
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
                            current_prompt = build_plan_writer_repair_prompt(
                                    task_prompt,
                                    issues,
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
            path = entry["path"] or ""
            landing = (
                path if Path(path).is_absolute() else f"daydream_plans/{path}"
            )
            print_success(
                console,
                f"Plan {entry['number']:03d} written to {landing}.",
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


def _local_plan_path(repo: Path, entry: dict[str, Any]) -> Path | None:
    raw_path = entry.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = Path(raw_path)
    return path if path.is_absolute() else repo / "daydream_plans" / path


def _publication_package_id(
    finding: dict[str, Any],
    entry: dict[str, Any] | None = None,
) -> str:
    """Return the plan's stored package identity before the current finding's."""
    return str(
        (entry or {}).get("package_fingerprint")
        or finding.get("package_fingerprint")
        or finding.get("fingerprint")
        or ""
    )


def _publication_members(
    entry: dict[str, Any],
    finding: dict[str, Any],
    field: str,
) -> tuple[str, ...]:
    """Read stored plan membership first, with current-finding compatibility."""
    raw = entry[field] if field in entry else finding.get(field)
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(value for value in raw if isinstance(value, str) and value)


def _publication_identity(
    entry: dict[str, Any],
    finding: dict[str, Any],
    plan_path: Path | None,
) -> dict[str, Any]:
    """Extract the per-package identity shared by every publication record."""
    return {
        "package_id": _publication_package_id(finding, entry),
        "title": str(entry.get("title") or finding.get("title") or "Improve repository"),
        "plan_path": plan_path.name if plan_path is not None else None,
        "member_fingerprints": list(_publication_members(entry, finding, "member_fingerprints")),
        "member_aliases": list(_publication_members(entry, finding, "member_aliases")),
    }


def _publication_failure(
    identity: dict[str, Any],
    *,
    stage: str,
    error: str,
) -> dict[str, Any]:
    """Build a publication failure record from a shared package identity."""
    return {**identity, "stage": stage, "error": error}


def _write_publication_artifact(
    ctx: FlowContext,
    publication: dict[str, Any],
) -> None:
    published_issues_path(ctx.data["improve_dir"]).write_text(
        json.dumps(
            _with_artifact_provenance(
                publication,
                phase=DaydreamPhase.PLAN_WRITE,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


async def _step_publish_issues(ctx: FlowContext) -> None:
    """Copy each validated local plan into one idempotent GitHub issue."""
    enabled = _automatic_issue_publishing(ctx)
    publication: dict[str, Any] = {
        "enabled": enabled,
        "status": "running" if enabled else "disabled",
        "repository": ctx.config.pr_repo,
        "published": [],
        "failed": [],
    }
    ctx.data["issue_publication"] = publication
    if not enabled:
        _write_publication_artifact(ctx, publication)
        return

    candidates: list[tuple[dict[str, Any], dict[str, Any], Path]] = []
    represented: set[str] = set()
    plan_write = ctx.data["plan_write"]
    for disposition in ("written", "skipped", "failed"):
        for entry in plan_write.get(disposition, []):
            if not isinstance(entry, dict):
                continue
            nested_finding = entry.get("finding")
            finding = nested_finding if isinstance(nested_finding, dict) else entry
            package_id = _publication_package_id(finding, entry)
            if package_id:
                represented.add(package_id)
            current_package_id = _publication_package_id(finding)
            if current_package_id:
                represented.add(current_package_id)
            plan_path = _local_plan_path(ctx.work.repo, entry)
            identity = _publication_identity(entry, finding, plan_path)
            if disposition == "failed":
                identity["plan_path"] = None
                publication["failed"].append(
                    _publication_failure(
                        identity,
                        stage="plan-write",
                        error="Plan writing did not produce a validated local plan.",
                    )
                )
            elif plan_path is None:
                publication["failed"].append(
                    _publication_failure(
                        identity,
                        stage="plan-reconciliation",
                        error="The selected package was skipped without an unambiguous validated local plan.",
                    )
                )
            elif not plan_path.is_file():
                publication["failed"].append(
                    _publication_failure(
                        identity,
                        stage="local-plan",
                        error="The validated local plan file is unavailable.",
                    )
                )
            else:
                candidates.append((entry, finding, plan_path))

    for finding in ctx.data.get("selected_findings", []):
        if not isinstance(finding, dict):
            continue
        package_id = _publication_package_id(finding)
        if package_id in represented:
            continue
        publication["failed"].append(
            {
                "package_id": package_id,
                "title": str(finding.get("title") or "Improve repository"),
                "plan_path": None,
                "stage": "plan-accounting",
                "error": ("Plan writing produced no outcome for the selected package."),
            }
        )

    _write_publication_artifact(ctx, publication)
    if not candidates:
        _finish_publication(ctx, publication)
        _write_publication_artifact(ctx, publication)
        return

    try:
        publisher = IssuePublisher.connect(
            ctx.work.repo,
            repo_slug=ctx.config.pr_repo,
        )
    except ImprovePublishError as exc:
        safe_error = redact_text(str(exc))
        for entry, finding, plan_path in candidates:
            publication["failed"].append(
                _publication_failure(
                    _publication_identity(entry, finding, plan_path),
                    stage="preflight",
                    error=safe_error,
                )
            )
        _finish_publication(ctx, publication)
        _write_publication_artifact(ctx, publication)
        print_error(console, "Improve issue publishing failed", safe_error)
        return

    publication["repository"] = publisher.repo_slug
    for entry, finding, plan_path in sorted(
        candidates,
        key=lambda item: int(item[0].get("number") or 0),
    ):
        identity = _publication_identity(entry, finding, plan_path)
        try:
            result = publisher.publish(
                package_id=identity["package_id"],
                title=identity["title"],
                plan_path=plan_path,
                member_fingerprints=identity["member_fingerprints"],
                member_aliases=identity["member_aliases"],
            )
        except (ImprovePublishError, ValueError) as exc:
            publication["failed"].append(
                _publication_failure(
                    identity,
                    stage="issue-create",
                    error=redact_text(str(exc)),
                )
            )
            print_warning(
                console,
                f"Issue publishing failed for {identity['title']}: {redact_text(str(exc))}",
            )
        else:
            publication["published"].append(
                {
                    **identity,
                    "disposition": result.disposition,
                    "issue_url": result.issue_url,
                }
            )
            print_success(
                console,
                f"Issue {result.disposition} for {identity['title']}: {result.issue_url}",
            )
        _set_publication_status(publication)
        _write_publication_artifact(ctx, publication)

    _finish_publication(ctx, publication)
    _write_publication_artifact(ctx, publication)


def _set_publication_status(publication: dict[str, Any]) -> None:
    """Reflect the current run's issue-publication outcome in its artifact."""
    if not publication.get("enabled"):
        publication["status"] = "disabled"
    elif publication.get("failed") and publication.get("published"):
        publication["status"] = "partial"
    elif publication.get("failed"):
        publication["status"] = "failed"
    else:
        publication["status"] = "complete"


def _finish_publication(
    ctx: FlowContext,
    publication: dict[str, Any],
) -> None:
    """Finalize publication status and preserve complete-vs-partial exits."""
    _set_publication_status(publication)
    if publication.get("failed"):
        publication_exit = 2 if publication.get("published") else 1
        ctx.data["plan_exit_code"] = max(
            ctx.data["plan_exit_code"],
            publication_exit,
        )


def _reanchored_report_section(plans_dir: Path) -> str:
    """Render the durable ``Re-anchored plans`` report section.

    Reads the durable plan index (never the session result), so prior-run
    re-anchors surface even when the current run re-anchored none. Returns a
    self-contained markdown section including its own header and a trailing
    blank line.
    """
    rows = reanchored_plan_rows(plans_dir)
    if not rows:
        return "## Re-anchored plans\n\n- No re-anchored plans.\n\n"
    table = (
        "| Plan | Title | Status | Landing path |\n"
        "| --- | --- | --- | --- |\n"
    )
    for entry in rows:
        table += (
            f"| {entry.number:03d} | {markdown_cell(entry.title)} "
            f"| `{entry.status}` | `{entry.landing_path or '(unavailable)'}` |\n"
        )
    return f"## Re-anchored plans\n\n{table}\n"


def _render_report(ctx: FlowContext) -> str:
    services = ctx.data["services"]
    all_services = ctx.data["all_services"]
    stacks = ctx.data["stacks"]
    audit = ctx.data["audit"]
    findings = ctx.data["vetted"]["findings"]
    discarded_no_evidence = ctx.data["audit_discarded_no_evidence"]
    dropped_low_confidence = ctx.data["audit_dropped_low_confidence"]
    dropped_by_cap = ctx.data["audit_dropped_by_cap"]
    previously_rejected = ctx.data["previously_rejected"]
    vet_rejected = ctx.data["vet_rejected"]
    findings_table = ctx.data["findings_table"]
    effort = ctx.config.improve_effort
    scope = ctx.config.improve_scope
    plan_write = ctx.data["plan_write"]
    issue_publication = ctx.data.get("issue_publication")
    partitions = ctx.data["partitions"]
    partitions_not_audited = ctx.data["partitions_not_audited"]
    groups = ctx.data["partition_groups"]
    service_lines = (
        "\n".join(f"- **{service.name}** — `{service.root.as_posix()}`" for service in services)
        or "- No service roots detected."
    )
    top_offender_lines = _top_offender_lines(findings)
    cleanup_pressure_lines = _cleanup_pressure_lines(findings)
    stack_lines = "\n".join(f"- **{stack.stack_name}**" for stack in stacks) or "- No stacks detected."
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
            "Recon hotspots only; categories outside correctness, security, tests, and tech debt were not audited."
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
    publication_lines = _publication_report_lines(issue_publication)
    return (
        "# Improve Report\n\n"
        "## Findings\n\n"
        f"{findings_table}\n\n"
        "## Cleanup pressure\n\n"
        f"{cleanup_pressure_lines}\n\n"
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
        f"{plan_lines}\n\n"
        f"{_reanchored_report_section(ctx.work.repo / 'daydream_plans')}"
        "## GitHub issues\n\n"
        f"{publication_lines}"
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
            f"  - **{markdown_cell(finding['title'])}** "
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


def _cleanup_pressure_lines(findings: list[dict[str, Any]]) -> str:
    """Summarize expected portfolio direction without inventing LOC counts."""
    counts = Counter(str(finding.get("change_shape", "unknown")) for finding in findings)
    subtractive = sum(counts[shape] for shape in ("delete", "reuse", "consolidate"))
    return (
        f"- Subtractive packages (delete/reuse/consolidate): {subtractive}\n"
        f"- Neutral or unknown packages: "
        f"{counts['neutral'] + counts['unknown']}\n"
        f"- Additive packages: {counts['additive']}\n"
        "- This is a prioritization preference, not a gate; exact LOC changes "
        "are measured only after implementation."
    )


def _publication_report_lines(publication: dict[str, Any] | None) -> str:
    if publication is None or not publication.get("enabled"):
        return "- Automatic issue publishing was disabled."
    published = publication.get("published", [])
    failed = publication.get("failed", [])
    dispositions = Counter(str(entry.get("disposition")) for entry in published if isinstance(entry, dict))
    unavailable = sum(
        1
        for entry in failed
        if isinstance(entry, dict)
        and entry.get("stage")
        in {
            "plan-write",
            "plan-reconciliation",
            "local-plan",
            "plan-accounting",
        }
    )
    github_failures = len(failed) - unavailable if isinstance(failed, list) else 0
    return (
        f"- Issues created: {dispositions['created']}\n"
        f"- Existing issues reused: {dispositions['existing']}\n"
        f"- Ambiguous creates reconciled: {dispositions['reconciled']}\n"
        f"- Packages without a validated local plan: {unavailable}\n"
        f"- GitHub publication failures: {github_failures}"
    )


def _improve_failure_message(ctx: FlowContext) -> tuple[str, str]:
    """Describe planning and publication failures without conflating them."""
    plan_failures = len(ctx.data["plan_write"]["failed"])
    publication = ctx.data.get("issue_publication")
    failed = publication.get("failed", []) if isinstance(publication, dict) and publication.get("enabled") else []
    local_plan_failures = sum(
        1
        for entry in failed
        if isinstance(entry, dict) and entry.get("stage") in {"plan-reconciliation", "local-plan", "plan-accounting"}
    )
    github_failures = sum(
        1 for entry in failed if isinstance(entry, dict) and entry.get("stage") in {"preflight", "issue-create"}
    )
    issue_failures = local_plan_failures + github_failures
    if plan_failures and issue_failures:
        heading = "Improve planning and issue publishing failed"
    elif issue_failures:
        heading = "Improve issue publishing failed"
    else:
        heading = "Improve planning failed"
    details = []
    if plan_failures:
        details.append(f"{plan_failures} selected plan(s) failed")
    if local_plan_failures:
        details.append(f"{local_plan_failures} selected package(s) lacked a local plan")
    if github_failures:
        details.append(f"{github_failures} GitHub operation(s) failed")
    return heading, "; ".join(details) + "."


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
                f"{_blocked_plan_attempt_lines(plan_write)}\n\n"
                "## GitHub issues\n\n"
                f"{_publication_report_lines(ctx.data.get('issue_publication'))}"
            ),
            encoding="utf-8",
        )
        if ctx.data["plan_exit_code"]:
            heading, detail = _improve_failure_message(ctx)
            print_error(console, heading, detail)
            return Stop(ctx.data["plan_exit_code"])
        print_success(console, "Description plan complete.")
        return None

    report_path(ctx.data["improve_dir"]).write_text(
        _report_with_provenance(_render_report(ctx))
    )
    if ctx.data["plan_exit_code"]:
        heading, detail = _improve_failure_message(ctx)
        print_error(console, heading, detail)
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
        name="publish-improve-issues",
        run=_step_publish_issues,
        config_phase="recon",
    ),
    FlowStep(
        name="improve-report",
        run=_step_report,
        config_phase="recon",
    ),
)
