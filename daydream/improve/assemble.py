"""Expand validated model-authored plans into host-owned execution handoffs."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from daydream.improve import plan_contract as contract, plan_normalization as normalization
from daydream.improve.plan_contract import AssemblyIssue, render_issue
from daydream.improve.render import plan_slug, redact_secret_values

GIT_PUSH_POLICY = "never-without-operator-instruction"
GIT_PULL_REQUEST_POLICY = "never-without-operator-instruction"
STOP_REQUIRED_ACTION = "STOP_AND_REPORT"


# Sha-free by design: the Status section renders the planned-at commit, so
# assembly stays git-free.
GIT_BRANCH_BASIS = (
    "Branch from the operator's current checkout. HEAD is expected to have "
    "moved past the planned-at commit; see Before you start."
)


def _expand_command_ref(
    ref: dict[str, Any],
    *,
    recon_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    base = recon_by_id[ref["recon_command_id"]]
    appended = ref["appended_args"]
    return {
        "purpose": base["purpose"],
        "command": (
            base["command"]
            if appended is None
            else f"{base['command']} {appended}"
        ),
        "working_directory": base["working_directory"],
        "expected_success": deepcopy(base["expected_success"]),
        "note": ref["note"],
    }


def _expand_commands(
    normalized: dict[str, Any],
    *,
    recon_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expand validated references in place and return the first-use table."""
    table: list[dict[str, Any]] = []
    seen: set[tuple[str, str | None]] = set()
    for _, ref, _ in contract._iter_command_refs(normalized):
        key = (ref["recon_command_id"], ref["appended_args"])
        expanded = _expand_command_ref(ref, recon_by_id=recon_by_id)
        if key not in seen:
            seen.add(key)
            table.append(deepcopy(expanded))
        ref.clear()
        ref.update(expanded)
    return table


def _resolve_excerpt(repo: Path, path: str, start: int, end: int) -> str:
    # Repository bytes are spliced in after _redact_strings has already run over
    # the authored content, so they must be redacted here.
    lines = (repo / path).read_text(encoding="utf-8").splitlines()
    return redact_secret_values("\n".join(lines[start - 1 : end]))


def _boilerplate_stop_conditions(
    normalized: dict[str, Any],
    step_count: int,
) -> list[dict[str, Any]]:
    scope = normalized["scope"]
    existing_paths = contract._entry_paths(scope["existing_paths"])
    in_scope_paths = [*existing_paths, *contract._entry_paths(scope["new_paths"])]
    specifications = [
        (
            "drift",
            "Before editing a file, read the exact line range quoted for "
            "it in the Current state section and compare it to the quoted "
            "text. It does not match character for character.",
            "Report the mismatched file, the quoted excerpt, and the "
            "current repository content.",
            existing_paths,
        ),
        (
            "repeated-verification-failure",
            "A verification in this plan fails, you make exactly one "
            "correction, and it fails again — two failures total for the "
            "same verification. Do not attempt a third time.",
            "Report both failing command outputs and the correction that "
            "was attempted.",
            [],
        ),
        (
            "out-of-scope-change",
            "Completing a step requires editing a path that is not "
            "declared in this plan's scope.",
            "Report the required path and why the declared scope "
            "boundary is insufficient.",
            in_scope_paths,
        ),
    ]
    conditions = [
        {
            "kind": kind,
            "condition": condition,
            "required_action": STOP_REQUIRED_ACTION,
            "evidence_to_report": evidence,
            "related_paths": paths,
            "related_step_ids": [],
        }
        for kind, condition, evidence, paths in specifications
    ]

    def mapped(kind: str, condition: dict[str, Any]) -> dict[str, Any]:
        return {
            "kind": kind,
            "condition": condition["condition"],
            "required_action": STOP_REQUIRED_ACTION,
            "evidence_to_report": condition["evidence_to_report"],
            "related_paths": list(condition["related_paths"]),
            "related_step_ids": [
                f"step-{number}"
                for number in condition["related_step_numbers"]
                if number <= step_count
            ],
        }

    conditions.append(
        mapped("false-assumption", normalized["false_assumption"])
    )
    return conditions


def _injected_done_criteria(normalized: dict[str, Any]) -> list[dict[str, Any]]:
    criteria = list(normalized["done_criteria"])
    kinds = {criterion["kind"] for criterion in criteria}
    def criterion(kind: str, description: str) -> dict[str, Any]:
        return {"kind": kind, "description": description[:500], "verification": None}

    if "behavior" not in kinds:
        # ``why_this_matters.intended_outcome`` is schema-required at >=30
        # characters, so the derived criterion is never empty or a stub.
        outcome = normalized["why_this_matters"]["intended_outcome"]
        description = f"The plan's intended outcome holds: {outcome}"
        criteria.insert(
            0,
            criterion("behavior", description),
        )
    test_plan = normalized["test_plan"]
    test_mode = test_plan["mode"]
    if "test-gate" not in kinds and test_mode != "not-applicable":
        if test_mode == "existing-coverage":
            symbols = ", ".join(coverage["symbol"] for coverage in test_plan["existing_coverage"])
            description = f"The cited existing coverage passes: {symbols}."
        else:
            symbols = ", ".join(case["test_symbol"] for case in test_plan["cases"])
            description = f"Every named test-plan case passes: {symbols}."
        criteria.append(
            criterion("test-gate", description)
        )
    for step in normalized["steps"]:
        for change in step["changes"]:
            if change["operation"] != "delete":
                continue
            path = change["path"]
            symbol = change["symbol"]
            if any(
                criterion["kind"] == "static-invariant" and change["target_state"] in criterion["description"]
                for criterion in criteria
            ):
                continue
            if symbol == path:
                description = f"The deleted path `{path}` is absent. Target state: {change['target_state']}"
            else:
                description = (
                    f"The deleted target `{symbol}` is absent from `{path}`. Target state: {change['target_state']}"
                )
            criteria.append(
                criterion("static-invariant", description)
            )
    if "scope-integrity" not in kinds:
        scope = normalized["scope"]
        paths = ", ".join(
            [
                *contract._entry_paths(scope["existing_paths"]),
                *contract._entry_paths(scope["new_paths"]),
            ]
        )
        description = f"Only the declared in-scope paths change: {paths}."
        criteria.append(
            criterion("scope-integrity", description)
        )
    return [
        {"id": f"done-{index}", **criterion}
        for index, criterion in enumerate(criteria, start=1)
    ]


def assemble_plan(
    authored: Any,
    *,
    repo: Path,
    recon_commands: Sequence[dict[str, Any]],
    expected_fingerprints: Sequence[str] | None = None,
) -> tuple[dict[str, Any] | None, tuple[AssemblyIssue, ...]]:
    """Return a host-assembled plan, or all remaining authoring issues.

    Normalization and validation only read repository files. Successful output
    is ready for rendering and landing without another model-content gate.
    """
    normalized = normalization._normalize_authored(authored, repo=repo)
    if normalized is None:
        return None, (AssemblyIssue("NO_STRUCTURED_OBJECT", "/"),)
    normalization._declare_stop_condition_paths(normalized, repo=repo)
    recon_by_id = {
        command["id"]: command
        for command in recon_commands
        if isinstance(command, dict) and isinstance(command.get("id"), str)
    }
    normalization._repair_command_scope(normalized, recon_by_id=recon_by_id)
    issues = contract._collect_issues(
        normalized,
        repo=repo,
        recon_by_id=recon_by_id,
        expected_fingerprints=expected_fingerprints,
    )
    if issues:
        return None, tuple(issues)

    commands = _expand_commands(normalized, recon_by_id=recon_by_id)
    normalized["current_state_excerpts"] = [
        {
            "path": entry["path"],
            "line_anchor": {
                "start_line": entry["start_line"],
                "end_line": entry["end_line"],
            },
            "file_role": entry["file_role"],
            "verbatim_excerpt": _resolve_excerpt(
                repo, entry["path"], entry["start_line"], entry["end_line"]
            ),
        }
        for entry in normalized["context_excerpts"]
    ]
    # Normalization already owns a detached copy. Enrich that one plan rather
    # than projecting its validated fields through another parallel shape.
    normalized["commands_you_will_need"] = commands
    normalized["git_workflow"].update(
        branch_name=f"improve/{plan_slug(normalized['title'])}",
        branch_basis=GIT_BRANCH_BASIS,
        push_policy=GIT_PUSH_POLICY,
        pull_request_policy=GIT_PULL_REQUEST_POLICY,
    )
    for index, step in enumerate(normalized["steps"], start=1):
        step.update(id=f"step-{index}", order=index)
    normalized["done_criteria"] = _injected_done_criteria(normalized)
    normalized["stop_conditions"] = _boilerplate_stop_conditions(normalized, len(normalized["steps"]))
    for authored_only in ("context_excerpts", "false_assumption", "additional_command_refs"):
        del normalized[authored_only]
    return normalized, ()


__all__ = [
    "GIT_BRANCH_BASIS",
    "GIT_PULL_REQUEST_POLICY",
    "GIT_PUSH_POLICY",
    "STOP_REQUIRED_ACTION",
    "AssemblyIssue",
    "assemble_plan",
    "render_issue",
]
