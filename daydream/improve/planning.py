"""Operator selection and incremental, deterministic plan-writer fan-out."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio

from daydream import git_ops, trajectory
from daydream.agent import console
from daydream.backends import effective_fanout_concurrency
from daydream.config import (
    PLAN_WRITE_MAX_CONCURRENCY,
)
from daydream.extensions.api import Stop
from daydream.fanout import run_fanout
from daydream.improve import artifacts
from daydream.improve.authoring import _verification_commands, author_plan, failed_plan_result
from daydream.improve.context import _audit_repo
from daydream.improve.plan_diagnostics import (
    _attempt_diagnostic,
    record_plan_write_diagnostics,
)
from daydream.improve.plans import (
    PlanWriteSession,
)
from daydream.improve.reanchor import (
    prune_stale_reanchor_worktrees,
)
from daydream.improve.render import plan_slug
from daydream.pr_review import compute_fingerprint
from daydream.run_context import resolve_run_context
from daydream.ui import print_info, print_success, print_warning

if TYPE_CHECKING:
    from daydream.flows.engine import FlowContext


from daydream.improve.issue_publication import (
    _automatic_issue_publishing,
)
from daydream.improve.reporting import (
    _findings_table,
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
    run_context = resolve_run_context(ctx.run_context)
    interactive = run_context.policy.interactive
    publish_all = not interactive and _automatic_issue_publishing(ctx)
    default_numbers = list(range(1, len(default_findings) + 1)) if publish_all else _default_selection(default_findings)
    mode = (
        "automatic-publishing"
        if publish_all
        else ("non-interactive-default" if not interactive else "interactive")
    )
    selected_numbers = default_numbers

    if not default_findings:
        print_success(console, "No vetted defect findings -- done.")
    elif interactive:
        default_text = f"1-{len(default_numbers)}" if len(default_numbers) > 1 else "1"
        for prompt in (_selection_prompt(default_findings), "Invalid selection; try once more"):
            raw = run_context.choice(
                prompt, default=default_text, safe_default=default_text, console=console,
            )
            parsed = _parse_selection(raw, total=len(default_findings))
            if parsed is not None:
                selected_numbers = parsed
                break

    selected_findings = [default_findings[number - 1] for number in selected_numbers]
    selected = [finding["fingerprint"] for finding in selected_findings]
    ctx.data["selected_findings"] = selected_findings
    ctx.data["selection_mode"] = mode
    artifacts.write_artifact(
        ctx.data["improve_dir"] / "selected.json",
        {"mode": mode, "selected": selected},
        phase=trajectory.DaydreamPhase.PLAN_WRITE,
    )
    return None


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


async def _step_write_plans(ctx: FlowContext) -> None:
    """Write selected findings as host-stamped, reconciling handoff plans."""
    description = ctx.config.improve_plan_description
    if description is not None:
        selected = [_description_finding(description)]
        ctx.data["selected_findings"] = selected
        ctx.data["selection_mode"] = "description"
    else:
        selected = ctx.data["selected_findings"]
    if ctx.work.is_unborn:
        diagnostics = [
            _attempt_diagnostic(
                finding=finding,
                attempt=None,
                received=None,
                disposition="blocked",
                stage="plan-accounting",
                errors=("UNBORN_PLAN_ANCHOR_UNAVAILABLE@/planned_at",),
            )
            for finding in selected
        ]
        result = {
            "written": [],
            "skipped": [],
            "failed": [
                {
                    **finding,
                    "errors": ["UNBORN_PLAN_ANCHOR_UNAVAILABLE"],
                }
                for finding in selected
            ],
            "diagnostics": diagnostics,
        }
        record_plan_write_diagnostics(
            (ctx.data["improve_dir"] / artifacts.PLAN_WRITE_DIAGNOSTICS_FILENAME),
            diagnostics,
            artifact_provenance=artifacts._artifact_provenance(
                phase=trajectory.DaydreamPhase.PLAN_WRITE
            ),
        )
        ctx.data["plan_write"] = result
        ctx.data["plan_exit_code"] = 1 if selected else 0
        return
    assert ctx.work.head_sha is not None
    prune_stale_reanchor_worktrees(ctx.work.repo, private_workspace_owner=ctx.private_workspace_owner)
    backend = ctx.backend_for("plan_write")
    recorder = trajectory.get_current_recorder()
    limiter = anyio.CapacityLimiter(
        effective_fanout_concurrency(PLAN_WRITE_MAX_CONCURRENCY, backend)
    )
    authoring_diagnostics: list[tuple[int, dict[str, Any]]] = []
    plans_dir = ctx.work.repo / "daydream_plans"
    planned_at: str
    try:
        planned_at = git_ops.head_sha(ctx.work.repo)
    except git_ops.GitError:
        planned_at = ctx.work.head_sha

    session = PlanWriteSession(
        plans_dir,
        planned_at=planned_at,
        non_interactive_default=(ctx.data["selection_mode"] in {"non-interactive-default", "automatic-publishing"}),
        run_session_id=trajectory.current_session_id(),
        private_workspace_owner=ctx.private_workspace_owner,
    )
    # Numbers are claimed here, in selection order, before any writer runs, so
    # a plan's number never depends on which writer finishes first.
    reservations = session.reserve(selected)
    total = sum(1 for reservation in reservations if reservation.number is not None)
    landed = 0

    def _land(index: int, record: dict[str, Any]) -> None:
        """Persist one writer's result and report the movement."""
        nonlocal landed
        outcome = session.commit(reservations[index], record)
        if reservations[index].number is None:
            return
        landed += 1
        if outcome.status == "written" and outcome.number is not None:
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

    jobs: list[tuple[int, dict[str, Any], str, dict[str, Any]]] = []
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
            _land(selection_index, failed_plan_result(finding, attempt, ("PROMPT_CONSTRUCTION_FAILED",)))
            continue

        jobs.append((selection_index, finding, prompt, attempt))

    async def write_plan(job: tuple[int, dict[str, Any], str, dict[str, Any]]) -> dict[str, Any]:
        current_index, current, task_prompt, task_attempt = job
        try:
            record = await author_plan(
                ctx, backend, finding=current, prompt=task_prompt, attempt=task_attempt,
                record_retry=lambda diagnostic: authoring_diagnostics.append(
                    (current_index, diagnostic)
                ),
            )
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
            record = failed_plan_result(current, task_attempt, (stable_category,))
        return record

    await run_fanout(
        jobs, write_plan, limiter=limiter, recorder=recorder,
        descriptor=lambda job: str(job[3]["descriptor"]),
        completed=lambda job, record: _land(job[0], record),
    )

    result = session.finish()
    record_plan_write_diagnostics(
        (ctx.data["improve_dir"] / artifacts.PLAN_WRITE_DIAGNOSTICS_FILENAME),
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
        artifact_provenance=artifacts._artifact_provenance(
            phase=trajectory.DaydreamPhase.PLAN_WRITE
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
