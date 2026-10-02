"""Improve findings, coverage, plan outcomes, and operator-facing reports."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from daydream import trajectory
from daydream.agent import console
from daydream.extensions.api import Stop
from daydream.improve import artifacts
from daydream.improve.partition import (
    PartitionGroup,
)
from daydream.improve.plan_index import (
    reanchored_plan_rows,
)
from daydream.improve.prioritize import (
    leverage_score,
)
from daydream.improve.render import markdown_cell
from daydream.ui import print_error, print_success

if TYPE_CHECKING:
    from daydream.flows.engine import FlowContext


from daydream.improve.audit_scope import (
    _coverage_ledger,
)


def _report_with_provenance(content: str) -> str:
    session_id = trajectory.current_session_id()
    if session_id is None:
        return content
    heading, separator, remainder = content.partition("\n")
    return (
        f"{heading}\n\nDaydream run: `{session_id}`\n"
        f"{separator}{remainder.lstrip()}"
    )


def _group_roots_cell(group: PartitionGroup, *, limit: int = 4) -> str:
    """Render a group's roots for a report line, truncating a long tail."""
    roots = group.roots
    shown = ", ".join(f"`{root}/`" for root in roots[:limit])
    remainder = len(roots) - limit
    return f"{shown} +{remainder} more" if remainder > 0 else shown


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


def _reanchored_report_section(plans_dir: Path) -> str:
    """Render current and prior re-anchors from the durable index as a complete Markdown section."""
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


def _coverage_bullet(entry: dict[str, Any]) -> str:
    """Render one coverage ledger entry as a report bullet."""
    prefix = (
        f"  - **{entry['partition']}** — `{entry['root']}/` "
        f"({entry['file_count']} files) "
    )
    if "audited_stacks" in entry:
        return (
            prefix
            + f"— partially audited (audited: {', '.join(entry['audited_stacks'])}; "
            f"omitted: {', '.join(entry['omitted_stacks'])})"
        )
    return (
        prefix
        + f"— not audited (omitted: {', '.join(entry['omitted_stacks'])})"
    )


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
    partition_omissions = ctx.data["partition_omissions"]
    groups = ctx.data["partition_groups"]
    failures = audit.get("failed", {})
    failed_groups = {key.partition(":")[2] for key in failures}
    ledger = _coverage_ledger(
        partitions, groups, partition_omissions, failed_groups=failed_groups
    )
    not_audited = ledger["not_audited"]
    partially_audited = ledger["partially_audited"]
    service_lines = (
        "\n".join(f"- **{service.name}** — `{service.root.as_posix()}`" for service in services)
        or "- No service roots detected."
    )
    top_offender_lines = _top_offender_lines(findings)
    cleanup_pressure_lines = _cleanup_pressure_lines(findings)
    stack_lines = "\n".join(f"- **{stack.stack_name}**" for stack in stacks) or "- No stacks detected."
    roots_by_group = {group.name: _group_roots_cell(group) for group in groups}
    failed_assignment_lines = (
        "\n".join(
            f"- **{assignment.replace(':', ' / ')}** "
            f"({roots_by_group.get(assignment.partition(':')[2], 'unknown group')})"
            f" — {reason}"
            for assignment, reason in failures.items()
        )
        or "- None."
    )
    coverage_bullets = [
        _coverage_bullet(entry) for entry in not_audited + partially_audited
    ]
    coverage_lines = (
        (
            "- Partitions not fully audited (reason: group-ceiling; raise "
            "`max-partition-groups` to include them):\n"
            + "\n".join(coverage_bullets)
        )
        if coverage_bullets
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
        f"{coverage_lines}\n\n"
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
        (ctx.data["improve_dir"] / artifacts.REPORT_FILENAME).write_text(
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
        success_message = "Description plan complete."
    else:
        (ctx.data["improve_dir"] / artifacts.REPORT_FILENAME).write_text(
            _report_with_provenance(_render_report(ctx))
        )
        success_message = (
            "Improve audit complete: "
            f"{len(ctx.data['services'])} services, "
            f"{len(ctx.data['stacks'])} stacks, "
            f"{len(ctx.data['vetted']['findings'])} vetted findings."
        )
    if ctx.data["plan_exit_code"]:
        heading, detail = _improve_failure_message(ctx)
        print_error(console, heading, detail)
        return Stop(ctx.data["plan_exit_code"])
    print_success(console, success_message)
    return None
