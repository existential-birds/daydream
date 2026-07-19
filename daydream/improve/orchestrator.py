"""Registered flow steps for repository-wide improve advising."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio

import daydream.agent as agent
from daydream import git_ops
from daydream.agent import console, get_non_interactive, run_agent
from daydream.config import AUDIT_CATEGORIES, EffortTier
from daydream.config_file import DaydreamFileConfig
from daydream.deep.detection import StackAssignment, detect_stacks
from daydream.deep.orchestrator import get_installed_skills
from daydream.exploration_runner import repo_scan
from daydream.extensions.api import FlowStep, Stop
from daydream.improve.artifacts import (
    audit_findings_path,
    recon_path,
    report_path,
    services_path,
    vetted_findings_path,
)
from daydream.improve.plans import load_rejections, record_rejections
from daydream.improve.prioritize import order_by_leverage, partition_direction
from daydream.improve.prompts import AUDIT_FINDINGS_SCHEMA, VET_SCHEMA
from daydream.improve.services import Service, enumerate_services, filter_scope
from daydream.pr_review import compute_fingerprint
from daydream.trajectory import (
    DaydreamPhase,
    get_current_recorder,
    maybe_fork,
    phase_scope,
)
from daydream.ui import print_error, print_success

if TYPE_CHECKING:
    from daydream.flows.engine import FlowContext


_RECON_SCHEMA: dict[str, Any] = {
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
            "type": "object",
            "additionalProperties": False,
            "required": ["build", "test", "lint"],
            "properties": {
                key: {"type": "array", "items": {"type": "string"}}
                for key in ("build", "test", "lint")
            },
        },
        "conventions": {"type": "array", "items": {"type": "string"}},
        "intent_docs": {"type": "array", "items": {"type": "string"}},
    },
}

_EVIDENCE_LOCATION = re.compile(
    r"^`?(.+?):(\d+)(?::\d+)?(?:`|\b)"
)


@dataclass(frozen=True)
class _AuditAssignment:
    category: str
    stack: str | None
    skill: str | None
    files: tuple[str, ...]

    @property
    def key(self) -> str:
        return (
            f"{self.category}:{self.stack}"
            if self.stack is not None
            else self.category
        )


def _service_dict(service: Service) -> dict[str, str]:
    return {
        "name": service.name,
        "root": service.root.as_posix(),
        "source": service.source,
    }


def _tracked_files(repo: Path) -> list[str]:
    proc = git_ops._run_git(  # noqa: SLF001 - git_ops owns the subprocess boundary
        repo,
        ["ls-files", "-z"],
        capture_bytes=True,
    )
    if proc.returncode != 0:
        return []
    stdout = proc.stdout if isinstance(proc.stdout, bytes) else proc.stdout.encode()
    return [
        path.decode("utf-8", errors="surrogateescape")
        for path in stdout.split(b"\0")
        if path
    ]


def _build_recon_prompt(
    repo: Path, services: list[Service], exploration_summary: str
) -> str:
    service_lines = "\n".join(
        f"- {service.name}: {service.root.as_posix()}" for service in services
    )
    return f"""IMPROVE_RECON

Read the repository at {repo} without modifying it. Return structured
reconnaissance facts only:

- languages and frameworks in active use;
- build, test, and lint commands supported by repository files;
- conventions that implementation plans must preserve;
- intent documents such as README, roadmap, ADR, and architecture files.

Services:
{service_lines or "- repository root"}

Existing repository scan:
{exploration_summary or "No additional conventions detected."}
"""


async def _step_recon(ctx: FlowContext) -> Stop | None:
    """Enumerate services, inspect repository conventions, and detect stacks."""
    target = ctx.work.repo
    directory: Path = ctx.data["improve_dir"]
    services = enumerate_services(
        target,
        ctx.config.file_config or DaydreamFileConfig(),
    )
    if ctx.config.improve_scope:
        try:
            services = filter_scope(services, ctx.config.improve_scope)
        except ValueError as exc:
            print_error(console, "Invalid Improve Scope", str(exc))
            return Stop(1)

    services_path(directory).write_text(
        json.dumps(
            {"services": [_service_dict(service) for service in services]},
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
            _build_recon_prompt(target, services, exploration.to_prompt_section()),
            phase=DaydreamPhase.RECON,
            output_schema=_RECON_SCHEMA,
            read_only=True,
        )

    recon_data = recon if isinstance(recon, dict) else {}
    recon_path(directory).write_text(json.dumps(recon_data, indent=2) + "\n")

    installed = get_installed_skills()
    availability = (
        installed if installed is not None else ctx.registry.stack_keys()
    )
    stacks = detect_stacks(
        _tracked_files(target),
        skill_availability=availability,
        registry=ctx.registry,
    )
    ctx.data["services"] = services
    ctx.data["recon"] = recon_data
    ctx.data["stacks"] = stacks
    return None


def _audit_assignments(
    ctx: FlowContext,
    categories: tuple[str, ...],
    stacks: list[StackAssignment],
) -> list[_AuditAssignment]:
    assignments: list[_AuditAssignment] = []
    for category in categories:
        remaining_files: set[str] = set()
        for stack in stacks:
            skill = ctx.registry.skill_if_registered(
                f"audit:{category}:{stack.stack_name}"
            )
            if skill is None:
                remaining_files.update(stack.files)
                continue
            assignments.append(
                _AuditAssignment(
                    category=category,
                    stack=stack.stack_name,
                    skill=skill,
                    files=tuple(stack.files),
                )
            )
        if remaining_files or not stacks:
            assignments.append(
                _AuditAssignment(
                    category=category,
                    stack=None,
                    skill=ctx.registry.skill_if_registered(f"audit:{category}"),
                    files=tuple(sorted(remaining_files)),
                )
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


def _stamp_finding(
    finding: dict[str, Any],
    category: str,
    services: list[Service],
) -> dict[str, Any] | None:
    evidence_paths = _evidence_paths(finding)
    if not evidence_paths:
        return None
    stamped = dict(finding)
    stamped["category"] = category
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


async def _step_audit(ctx: FlowContext) -> None:
    """Run tier-driven category audits and persist grounded findings."""
    directory: Path = ctx.data["improve_dir"]
    tier: EffortTier = ctx.data["effort_tier"]
    services: list[Service] = ctx.data["services"]
    stacks: list[StackAssignment] = ctx.data["stacks"]
    categories = tier.categories or AUDIT_CATEGORIES
    assignments = _audit_assignments(ctx, categories, stacks)
    backend = ctx.backend_for("audit")
    recorder = get_current_recorder()
    limiter = anyio.CapacityLimiter(tier.max_concurrency)
    results: dict[str, tuple[_AuditAssignment, list[dict[str, Any]]]] = {}
    failures: dict[str, str] = {}

    async with anyio.create_task_group() as task_group:
        for assignment in assignments:
            invocation = (
                backend.format_skill_invocation(assignment.skill)
                if assignment.skill is not None
                else None
            )
            scoped_services = _services_for_files(services, assignment.files)
            scope_note = (
                f"Audit the {assignment.stack} stack. Relevant tracked files: "
                + ", ".join(assignment.files)
                if assignment.stack is not None
                else "Cover the remaining repository surface. Relevant tracked files: "
                + (", ".join(assignment.files) or "(all tracked files)")
            )
            prompt = ctx.registry.prompt("audit")(
                category=assignment.category,
                skill_invocation=invocation,
                services=scoped_services,
                scope_note=scope_note,
                recon_summary=json.dumps(ctx.data["recon"], sort_keys=True),
                cwd=ctx.work.repo,
                tier=tier,
            )

            async def _task(
                current: _AuditAssignment = assignment,
                task_prompt: str = prompt,
            ) -> None:
                descriptor = f"audit-{current.category}"
                if current.stack is not None:
                    descriptor += f"-{current.stack}"
                async with maybe_fork(recorder, descriptor):
                    try:
                        async with limiter:
                            output, _, _ = await run_agent(
                                backend,
                                ctx.work.repo,
                                task_prompt,
                                phase=DaydreamPhase.AUDIT,
                                output_schema=AUDIT_FINDINGS_SCHEMA,
                                read_only=True,
                            )
                        raw_findings = (
                            output.get("findings", [])
                            if isinstance(output, dict)
                            else []
                        )
                        findings = [
                            finding
                            for finding in raw_findings
                            if isinstance(finding, dict)
                        ]
                        results[current.key] = (current, findings)
                    except Exception as exc:  # noqa: BLE001
                        failures[current.key] = (
                            f"{type(exc).__name__}: {exc}"
                        )

            task_group.start_soon(_task)

    if recorder is not None:
        recorder.create_dispatch_step(phase=DaydreamPhase.AUDIT)

    grounded: list[dict[str, Any]] = []
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
            )
            if stamped is None:
                discarded_no_evidence += 1
                continue
            if tier.high_confidence_only and stamped.get("confidence") != "HIGH":
                dropped_low_confidence += 1
                continue
            assignment_findings.append(stamped)
        audit_findings_path(
            directory,
            assignment.category,
            assignment.stack,
        ).write_text(
            json.dumps({"findings": assignment_findings}, indent=2) + "\n"
        )
        grounded.extend(assignment_findings)

    ordered = order_by_leverage(grounded)
    dropped_by_cap = 0
    if tier.max_findings is not None and len(ordered) > tier.max_findings:
        dropped_by_cap = len(ordered) - tier.max_findings
        ordered = ordered[: tier.max_findings]

    combined = {
        "categories_run": list(categories),
        "failed": dict(sorted(failures.items())),
        "findings": ordered,
    }
    (directory / "audit-findings.json").write_text(
        json.dumps(combined, indent=2) + "\n"
    )
    ctx.data["audit"] = combined
    ctx.data["audit_discarded_no_evidence"] = discarded_no_evidence
    ctx.data["audit_dropped_low_confidence"] = dropped_low_confidence
    ctx.data["audit_dropped_by_cap"] = dropped_by_cap


def _apply_vet_verdicts(
    findings: list[dict[str, Any]],
    verdicts: list[Any],
    *,
    rejected_at_sha: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply positional, 1-based vet verdicts with fail-closed polarity."""
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
    )
    for offset, finding in enumerate(findings):
        vet_id = offset + 1
        verdict = verdicts[offset] if offset < len(verdicts) else None
        if not isinstance(verdict, dict) or verdict.get("vet_id") != vet_id:
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
        kept.append(corrected)
    return kept, rejected


async def _step_vet(ctx: FlowContext) -> None:
    """Re-verify audit findings and persist model-confirmed rejections."""
    directory: Path = ctx.data["improve_dir"]
    plans_dir = ctx.work.repo / "daydream_plans"
    previous = load_rejections(plans_dir)
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

    backend = ctx.backend_for("vet")
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for category_findings in by_category.values():
        indexed = [
            {**finding, "vet_id": vet_id}
            for vet_id, finding in enumerate(category_findings, start=1)
        ]
        prompt = ctx.registry.prompt("vet")(
            findings=indexed,
            cwd=ctx.work.repo,
        )
        try:
            async with phase_scope(DaydreamPhase.VET):
                output, _, _ = await run_agent(
                    backend,
                    ctx.work.repo,
                    prompt,
                    phase=DaydreamPhase.VET,
                    output_schema=VET_SCHEMA,
                    read_only=True,
                )
        except Exception:  # noqa: BLE001 - no verdict fails closed
            output = {}
        verdicts = (
            output.get("verdicts", [])
            if isinstance(output, dict)
            and isinstance(output.get("verdicts"), list)
            else []
        )
        category_kept, category_rejected = _apply_vet_verdicts(
            category_findings,
            verdicts,
            rejected_at_sha=ctx.work.head_sha,
        )
        kept.extend(category_kept)
        rejected.extend(category_rejected)

    record_rejections(plans_dir, rejected)
    vetted = {"findings": order_by_leverage(kept)}
    vetted_findings_path(directory).write_text(
        json.dumps(vetted, indent=2) + "\n"
    )
    ctx.data["vetted"] = vetted
    ctx.data["previously_rejected"] = previously_rejected
    ctx.data["vet_rejected"] = len(rejected)


def _markdown_cell(value: Any) -> str:
    """Render a value safely inside a Markdown table cell."""
    return str(value or "—").replace("|", "\\|").replace("\n", " ")


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


async def _step_prioritize(ctx: FlowContext) -> None:
    """Partition and leverage-order vetted findings for reporting and selection."""
    directory: Path = ctx.data["improve_dir"]
    payload = json.loads(vetted_findings_path(directory).read_text())
    raw_findings = payload.get("findings", [])
    findings = [
        finding for finding in raw_findings if isinstance(finding, dict)
    ]
    defects, direction = partition_direction(findings)
    ordered_defects = order_by_leverage(defects)
    ordered_direction = order_by_leverage(direction)
    vetted = {
        "findings": [*ordered_defects, *ordered_direction],
        "defects": ordered_defects,
        "direction": ordered_direction,
    }
    vetted_findings_path(directory).write_text(
        json.dumps(vetted, indent=2) + "\n"
    )
    ctx.data["vetted"] = vetted
    ctx.data["defects"] = ordered_defects
    ctx.data["direction"] = ordered_direction
    ctx.data["findings_table"] = _findings_table(ordered_defects)
    ctx.data["direction_section"] = _direction_section(
        ordered_direction,
        start=len(ordered_defects) + 1,
    )


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
    default_numbers = _default_selection(defects)
    mode = "non-interactive-default" if get_non_interactive() else "interactive"
    selected_numbers = default_numbers

    if not defects:
        (ctx.data["improve_dir"] / "selected.json").write_text(
            json.dumps({"mode": mode, "selected": []}, indent=2) + "\n"
        )
        print_success(console, "No vetted defect findings -- done.")
        return Stop(0)

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
    (ctx.data["improve_dir"] / "selected.json").write_text(
        json.dumps({"mode": mode, "selected": selected}, indent=2) + "\n"
    )
    return None


def _render_report(
    services: list[Service],
    stacks: list[StackAssignment],
    audit: dict[str, Any],
    discarded_no_evidence: int,
    dropped_low_confidence: int,
    dropped_by_cap: int,
    previously_rejected: int,
    vet_rejected: int,
    findings_table: str,
    direction_section: str,
    effort: str,
    scope: str | None,
) -> str:
    service_lines = (
        "\n".join(
            f"- **{service.name}** — `{service.root.as_posix()}`"
            for service in services
        )
        or "- No service roots detected."
    )
    stack_lines = (
        "\n".join(f"- **{stack.stack_name}**" for stack in stacks)
        or "- No stacks detected."
    )
    failures = audit.get("failed", {})
    failed_assignment_lines = (
        "\n".join(
            f"- **{assignment}** — {reason}"
            for assignment, reason in failures.items()
        )
        or "- None."
    )
    tier_bound = {
        "quick": (
            "Recon hotspots only; categories outside correctness, security, "
            "and tests were not audited."
        ),
        "standard": (
            "Coverage was hotspot-weighted across key packages; exhaustive "
            "whole-repository coverage was not attempted."
        ),
        "deep": (
            "Coverage included every detected package; untracked files and "
            "surfaces outside detected services were not audited."
        ),
    }[effort]
    scope_statement = (
        f"Service scope slicing was limited to `{scope}`; other services were "
        "not audited."
        if scope
        else "No explicit service scope slicing was requested."
    )
    return (
        "# Improve Report\n\n"
        "## Findings\n\n"
        f"{findings_table}\n\n"
        f"{direction_section}\n\n"
        "## Services\n\n"
        f"{service_lines}\n\n"
        "## Stacks\n\n"
        f"{stack_lines}\n\n"
        "## What ran\n\n"
        "- Read-only repository reconnaissance\n"
        f"- Read-only audits across {len(audit.get('categories_run', []))} categories\n\n"
        "## What was not audited\n\n"
        f"- {tier_bound}\n"
        f"- {scope_statement}\n\n"
        "### Failed audit assignments\n\n"
        f"{failed_assignment_lines}\n\n"
        "## Audit filtering\n\n"
        f"- Findings without `path:line` evidence discarded: {discarded_no_evidence}\n"
        f"- Findings rejected during vetting: {vet_rejected}\n"
        f"- Non-HIGH-confidence findings dropped by tier: {dropped_low_confidence}\n"
        f"- Lowest-leverage findings dropped by tier cap: {dropped_by_cap}\n"
        f"- Previously rejected findings suppressed: {previously_rejected}\n"
    )


async def _step_report(ctx: FlowContext) -> None:
    """Render the improve report for reconnaissance and audit coverage."""
    report_path(ctx.data["improve_dir"]).write_text(
        _render_report(
            ctx.data["services"],
            ctx.data["stacks"],
            ctx.data["audit"],
            ctx.data["audit_discarded_no_evidence"],
            ctx.data["audit_dropped_low_confidence"],
            ctx.data["audit_dropped_by_cap"],
            ctx.data["previously_rejected"],
            ctx.data["vet_rejected"],
            ctx.data["findings_table"],
            ctx.data["direction_section"],
            ctx.config.improve_effort,
            ctx.config.improve_scope,
        )
    )
    print_success(
        console,
        "Improve audit complete: "
        f"{len(ctx.data['services'])} services, "
        f"{len(ctx.data['stacks'])} stacks, "
        f"{len(ctx.data['vetted']['findings'])} vetted findings.",
    )


STEPS: tuple[FlowStep, ...] = (
    FlowStep(name="recon", run=_step_recon),
    FlowStep(name="audit", run=_step_audit),
    FlowStep(name="vet", run=_step_vet),
    FlowStep(name="prioritize", run=_step_prioritize),
    FlowStep(
        name="improve-report",
        run=_step_report,
        config_phase="recon",
    ),
    FlowStep(name="select-plans", run=_step_select),
)
