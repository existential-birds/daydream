"""Plan generation, one transport-crash retry, and one authoring repair.

Each generation owns a phase interval. Model output reaches the assembler even
when schema-invalid so repair feedback names only validated diagnostic codes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from daydream.agent import _ToolSupervisorFailure, run_agent
from daydream.backends import Backend
from daydream.flows.engine import FlowContext
from daydream.improve.assemble import assemble_plan
from daydream.improve.context import _audit_repo
from daydream.improve.plan_contract import AssemblyIssue, render_issue
from daydream.improve.plan_diagnostics import (
    _attempt_diagnostic,
)
from daydream.improve.prompts import build_plan_writer_repair_prompt
from daydream.improve.redaction import redact_model_value
from daydream.improve.schemas import PLAN_AUTHOR_SCHEMA
from daydream.trajectory import DaydreamPhase, phase_scope


def _verification_commands(recon: dict[str, Any]) -> list[dict[str, Any]]:
    raw_commands = recon.get("commands")
    if not isinstance(raw_commands, list):
        return []
    return [
        command
        for command in raw_commands
        if isinstance(command, dict)
    ]


def _expected_plan_fingerprints(finding: dict[str, Any]) -> list[str]:
    members = finding.get("member_fingerprints")
    if isinstance(members, list):
        valid = [item for item in members if isinstance(item, str) and item]
        if valid:
            return valid
    fingerprint = finding.get("fingerprint")
    return [fingerprint] if isinstance(fingerprint, str) and fingerprint else []


def failed_plan_result(
    finding: dict[str, Any], attempt: dict[str, Any], errors: tuple[str, ...],
    *, received: Any = None, validation: bool = False,
) -> dict[str, Any]:
    """Keep every blocked plan on the same accounting and diagnostic contract."""
    details = {**attempt, "received_result": received, "errors": errors}
    if validation:
        details["validation"] = True
    return {"finding": finding, "_attempt": details, "error": True}


async def _generate_once(
    ctx: FlowContext, backend: Backend, prompt: str,
) -> tuple[Any, str | None]:
    async with phase_scope(DaydreamPhase.PLAN_WRITE):
        # The assembler, rather than run_agent, owns fail-closed schema validation
        # and exposes invalid structured output to the single repair pass.
        output, _, aborted_reason = await run_agent(
            backend, _audit_repo(ctx), prompt, phase=DaydreamPhase.PLAN_WRITE,
            output_schema=PLAN_AUTHOR_SCHEMA, read_only=True, persist_session=False,
            validate_structured_output=False, run_context=ctx.run_context,
        )
    return output, aborted_reason


async def _generate_with_crash_retry(
    ctx: FlowContext, backend: Backend, prompt: str,
) -> tuple[Any, str | None]:
    """Retry one crash without resetting an exhausted backend retry budget.

    Tool-supervisor vetoes always propagate. This replaces only the crashed
    generation; it never spends another authoring-repair generation.
    """
    try:
        return await _generate_once(ctx, backend, prompt)
    except _ToolSupervisorFailure:
        raise
    except Exception as exc:
        if getattr(exc, "retryable", False):
            raise
        return await _generate_once(ctx, backend, prompt)


async def author_plan(
    ctx: FlowContext, backend: Backend, *, finding: dict[str, Any],
    prompt: str, attempt: dict[str, Any], record_retry: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    """Return a complete plan or one blocked result; record retries immediately.

    The callback retains the first diagnostic even if the second generation
    raises. Exceptions remain the caller's responsibility so sibling writers
    can land independently while the failed writer receives a stable category.
    """
    generation_prompt = prompt
    for generation in range(2):
        output, aborted = await _generate_with_crash_retry(ctx, backend, generation_prompt)
        output = redact_model_value(output)
        if aborted is not None:
            code = {
                "tool_call_budget_exceeded": "TOOL_CALL_BUDGET_EXCEEDED",
                "wall_budget_exceeded": "WALL_BUDGET_EXCEEDED",
            }.get(aborted, "TOOL_VETOED" if aborted.startswith("tool_vetoed:") else "AGENT_ABORTED")
            return failed_plan_result(finding, attempt, (code,), received=output)
        if isinstance(output, dict):
            assembled, issues = assemble_plan(
                output, repo=_audit_repo(ctx), recon_commands=_verification_commands(ctx.data["recon"]),
                expected_fingerprints=_expected_plan_fingerprints(finding),
            )
        else:
            assembled, issues = None, (AssemblyIssue(code="NO_STRUCTURED_OBJECT", pointer="/"),)
        if assembled is not None and not issues:
            return {"finding": finding, "_attempt": attempt, "plan": assembled}
        errors = tuple(render_issue(issue) for issue in issues)
        if generation == 1:
            return failed_plan_result(
                finding, attempt, errors if isinstance(output, dict) else ("NO_STRUCTURED_OBJECT",),
                received=output, validation=isinstance(output, dict),
            )
        record_retry(_attempt_diagnostic(
            finding=finding, attempt=attempt, received=output, disposition="retried",
            stage="authoring" if isinstance(output, dict) else "transport", errors=errors,
        ))
        generation_prompt = build_plan_writer_repair_prompt(prompt, issues)
    raise AssertionError("authoring must finish after its repair generation")
