"""Read-only reconnaissance and host-validated repository verification commands."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from daydream import git_ops, trajectory
from daydream.agent import console, run_agent
from daydream.config import (
    DEFAULT_TOOL_CALL_BUDGET,
    DEFAULT_WALL_BUDGET_S,
    EffortTier,
)
from daydream.config_file import DaydreamFileConfig
from daydream.deep.detection import StackAssignment, detect_stacks
from daydream.deep.diff import _diff_changed_files
from daydream.exploration import EXPLORATION_SECTION_PREFIX, ExplorationContext
from daydream.exploration_runner import EXPLORATION_MAX_TURNS, _sample_paths, repo_scan
from daydream.extensions.api import Stop
from daydream.improve import artifacts
from daydream.improve.command_contract import (
    RECON_COMMAND_SCHEMA,
    validate_host_commands,
    validate_recon_commands,
)
from daydream.improve.context import _audit_repo
from daydream.improve.partition import (
    Partition,
    PartitionGroup,
    PartitionStackOmission,
)
from daydream.improve.prompts import (
    RECON_COMMAND_CONTRACT_BULLET,
)
from daydream.improve.repo_commands import enumerate_repository_commands
from daydream.prompts.grounding import (
    REPOSITORY_SEARCH_GUIDANCE,
    UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY,
)
from daydream.redaction import redact_value
from daydream.repository_paths import canonicalize_working_directory
from daydream.review_profile import build_default_profile
from daydream.services import (
    Service,
    enumerate_services,
    filter_scope,
)
from daydream.ui import print_error, print_warning

if TYPE_CHECKING:
    from daydream.flows.engine import FlowContext


from daydream.improve import audit_scope

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


# Keep the standalone recon contract for custom survey compatibility. The default
# investigation also returns guidelines instead of relying on a second model to
# relay them from an exploration summary.
DISCOVERY_SCHEMA: dict[str, Any] = {
    **RECON_SCHEMA,
    "required": [*RECON_SCHEMA["required"], "guidelines"],
    "properties": {
        **RECON_SCHEMA["properties"],
        "guidelines": {"type": "array", "items": {"type": "string"}},
    },
}


def _build_discovery_prompt(
    repo: Path, services: list[Service], groups: list[PartitionGroup], tracked: list[str],
) -> str:
    sample = _sample_paths(tracked, 500)
    sample_block = "\n".join(f"- {path}" for path in sample) or "- (no tracked files)"
    return _build_recon_prompt(repo, services, groups, "") + f"""
Combined repository discovery (survey and reconnaissance in one investigation):
Survey the repository's steady-state conventions as well as reconnaissance facts.
Read root and applicable nested AGENTS.md / CLAUDE.md and house-style configuration.
Where configuration is silent, inspect representative source to infer conventions.
Return conventions with their descriptions and source paths in `conventions`;
return applicable project instructions in `guidelines`. Include active frameworks
in `languages` and intent-document paths in `intent_docs`. These facts go directly
to audit and plan writers; no later survey will fill gaps.
Do not execute any discovered command candidate. Reuse each file read for all facts.

Host inventory sample: {len(sample)} of {len(tracked)} tracked files.
This is a starting point, not complete investigation coverage. Inspect real source
directories beyond the sample when necessary using read-only file tools. The host
has already enumerated the tracked inventory; no second Git enumeration is needed.
<tracked_file_sample>
{sample_block}
</tracked_file_sample>

{REPOSITORY_SEARCH_GUIDANCE}

This Improve snapshot permits only Read/Glob/Grep inspection tools; Bash and
command execution are unavailable. The shell examples above describe search scope
only: reuse the host inventory, use scoped Glob and Grep on explicit regular files,
and do not execute helper, Git, or discovered commands.
"""


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
    # `exploration_summary` (ExplorationContext.to_prompt_section()) opens with
    # the same boundary emitted below (exploration.EXPLORATION_SECTION_PREFIX);
    # strip it so the untrusted-content warning appears exactly once in the
    # recon prompt.
    summary_prefix = EXPLORATION_SECTION_PREFIX
    if exploration_summary.startswith(summary_prefix):
        exploration_summary = exploration_summary.removeprefix(summary_prefix)
    scan_section = f"Existing repository scan:\n{exploration_summary}" if exploration_summary else ""
    return f"""IMPROVE_RECON

{UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY}

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

{scan_section}
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
    """Return candidate count, validated commands, and rejection codes.

    Enumeration failure warns and falls back to model commands; it never aborts
    reconnaissance.
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
    def _dedup_key(
        command: dict[str, Any],
    ) -> tuple[str, str]:
        """Match absolute and relative working-directory spellings against the same host command."""
        return (
            command["command"],
            canonicalize_working_directory(repo, command["working_directory"]),
        )

    already_cited = {_dedup_key(command) for command in model_commands}
    candidates = [
        command for command in enumerated if _dedup_key(command) not in already_cited
    ]
    validated, errors = validate_host_commands(candidates, repo=repo)
    return len(candidates), validated, errors


async def _step_recon(ctx: FlowContext) -> Stop | None:
    """Enumerate services, inspect repository conventions, and detect stacks."""
    target = _audit_repo(ctx)
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

    branch_diff = ""
    if branch_focus:
        audit = ctx.audit_workspace
        if audit is None or audit.branch_base_sha is None or ctx.work.head_sha is None:
            raise RuntimeError("branch-focus improve has no pinned audit diff base")
        branch_diff = git_ops.diff(
            target,
            audit.branch_base_sha,
            head=ctx.work.head_sha,
        )
    branch_files = _diff_changed_files(branch_diff) if branch_focus else []
    if branch_focus:
        # Branch focus needs every category over one small diff. The requested
        # tier still owns concurrency, confidence filtering, finding caps, and
        # audit depth; audit/vet apply their backend limit when dispatching.
        requested: EffortTier = ctx.data["effort_tier"]
        ctx.data["effort_tier"] = replace(requested, categories=None)

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
            branch_diff, branch_files = audit_scope._restrict_diff_to_services(branch_diff, services)
    if branch_focus:
        services = audit_scope._services_for_files(services, tuple(branch_files))

    ctx.data["branch_diff"] = branch_diff
    ctx.data["branch_files"] = branch_files

    # Survey remains repository-wide even for branch/scoped/description runs.
    # Reuse this inventory for partitioning instead of having repo_scan query it.
    try:
        repository_files = git_ops.ls_files(target)
    except Exception:
        # In branch/description modes the old optional survey alone owned this
        # query and degraded to an empty sample. Full audits require inventory.
        if not (branch_focus or description_mode):
            raise
        repository_files = []
    stacks: list[StackAssignment] = []
    partitions: list[Partition] = []
    groups: list[PartitionGroup] = []
    omissions: list[PartitionStackOmission] = []
    if not description_mode:
        # Availability is resolved once in runner.run and threaded via config;
        # None flows through to detect_stacks' optimistic default.
        tracked = branch_files if branch_focus else repository_files
        stacks = detect_stacks(
            tracked,
            registry=ctx.registry,
            target_dir=target,
        )
        if ctx.config.improve_scope:
            stacks = audit_scope._stacks_for_services(stacks, services)
            tracked = sorted({path for stack in stacks for path in stack.files})
        partitions, groups, omissions = audit_scope._partition_repository(
            ctx,
            tracked,
            services,
            stacks,
            branch_focus=branch_focus,
        )
        artifacts.write_artifact(
            directory / artifacts.COVERAGE_FILENAME,
            audit_scope._coverage_ledger(partitions, groups, omissions),
            phase=trajectory.DaydreamPhase.RECON,
        )

    backend = ctx.backend_for("recon")
    survey_key = "exploration.repository_survey"
    survey_strategy = ctx.strategy(survey_key)
    combined = survey_strategy == build_default_profile().strategies[survey_key].content
    exploration = ExplorationContext()
    mode = "combined" if combined else "survey-recon"
    async with trajectory.phase_scope(trajectory.DaydreamPhase.RECON, stage=mode):
        if not combined:
            async with trajectory.phase_scope(
                trajectory.DaydreamPhase.EXPLORATION, stage="repo-survey"
            ):
                exploration = await repo_scan(
                    backend, target, strategies={survey_key: survey_strategy}, run_context=ctx.run_context,
                )
        recon, _, _ = await run_agent(
            backend,
            target,
            _build_discovery_prompt(target, services, groups, repository_files)
            if combined else _build_recon_prompt(target, services, groups, exploration.to_prompt_section()),
            phase=trajectory.DaydreamPhase.RECON,
            output_schema=DISCOVERY_SCHEMA if combined else RECON_SCHEMA,
            read_only=True,
            persist_session=False,
            validate_structured_output=False,
            max_turns=EXPLORATION_MAX_TURNS if combined else None,
            wall_budget_s=DEFAULT_WALL_BUDGET_S if combined else None,
            tool_call_budget=DEFAULT_TOOL_CALL_BUDGET,
            run_context=ctx.run_context,
        )

    total_candidates = 0
    valid_commands: list[dict[str, Any]] = []
    command_errors: list[str] = []
    model_fields: dict[str, Any] = {field: [] for field in ("languages", "conventions", "intent_docs", "guidelines")}
    safe_recon = redact_value(recon)
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
            for field in ("languages", "conventions", "intent_docs", "guidelines")
        }
    else:
        command_errors = ["RECON_CONTAINER_INVALID@/"]

    # A custom survey is best-effort, but its admitted guidance must not depend
    # on the recon model remembering to repeat it. Redact before persistence.
    model_fields["conventions"].extend(redact_value([
        f"{item.name}: {item.description}" + (f" (source: {item.source})" if item.source else "")
        for item in exploration.conventions
    ]))
    model_fields["guidelines"].extend(redact_value(exploration.guidelines))

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
        "discovery_mode": mode,
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
    recon_data = artifacts.write_artifact(
        directory / artifacts.RECON_FILENAME, recon_data, phase=trajectory.DaydreamPhase.RECON,
    )
    reasons = Counter(error.partition("@")[0] for error in command_errors)
    recorder = trajectory.get_current_recorder()
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
    ctx.data["partition_omissions"] = omissions
    return None
