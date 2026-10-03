"""Expand validated model-authored plans into host-owned execution handoffs."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from daydream.improve import plan_contract as contract, plan_normalization as normalization
from daydream.improve.plan_contract import AssemblyIssue, render_issue
from daydream.improve.redaction import redact_model_value
from daydream.improve.render import redact_secret_values

GIT_PUSH_POLICY = "never-without-operator-instruction"
GIT_PULL_REQUEST_POLICY = "never-without-operator-instruction"
STOP_REQUIRED_ACTION = "STOP_AND_REPORT"


# Sha-free by design: the Status section renders the planned-at commit, so
# assembly stays git-free.
GIT_BRANCH_BASIS = (
    "Branch from the operator's current checkout. HEAD is expected to have "
    "moved past the planned-at commit; see Before you start."
)


def _resolve_excerpt(repo: Path, path: str, start: int, end: int) -> str:
    # Repository bytes are spliced in after _redact_strings has already run over
    # the authored content, so they must be redacted here.
    lines = (repo / path).read_text(encoding="utf-8").splitlines()
    return redact_secret_values("\n".join(lines[start - 1 : end]))


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
    return criteria


@dataclass(frozen=True)
class AdmittedPlan:
    """One admitted authoring object with captured repository and command facts."""

    authored: dict[str, Any]
    recon_by_id: dict[str, dict[str, Any]]
    excerpts: tuple[tuple[dict[str, Any], str], ...]

    def command(self, ref: dict[str, Any] | None) -> dict[str, Any] | None:
        if ref is None:
            return None
        base = self.recon_by_id[ref["recon_command_id"]]
        appended = ref["appended_args"]
        result: dict[str, Any] = redact_model_value({
            "purpose": base["purpose"],
            "command": base["command"] if appended is None else f"{base['command']} {appended}",
            "working_directory": base["working_directory"],
            "expected_success": base["expected_success"],
            "note": ref["note"],
        })
        return result

    def commands(self) -> list[dict[str, Any]]:
        table: dict[tuple[str, str | None], dict[str, Any]] = {}
        for _, ref, _ in contract._iter_command_refs(self.authored):
            key = (ref["recon_command_id"], ref["appended_args"])
            if key not in table:
                command = self.command(ref)
                assert command is not None
                table[key] = command
        return list(table.values())

    def done_criteria(self) -> list[dict[str, Any]]:
        criteria: list[dict[str, Any]] = redact_model_value(_injected_done_criteria(self.authored))
        return criteria


def assemble_plan(
    authored: Any,
    *,
    repo: Path,
    recon_commands: Sequence[dict[str, Any]],
    expected_fingerprints: Sequence[str] | None = None,
) -> tuple[AdmittedPlan | None, tuple[AssemblyIssue, ...]]:
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

    # Retain the admitted authoring object. Capture external facts now; rendering
    # derives host numbering and policy without manufacturing a second plan shape.
    captured_commands = {
        ref["recon_command_id"]: {
            name: deepcopy(recon_by_id[ref["recon_command_id"]][name])
            for name in ("purpose", "command", "working_directory", "expected_success")
        }
        for _, ref, _ in contract._iter_command_refs(normalized)
    }
    excerpts = tuple(
        (entry, _resolve_excerpt(repo, entry["path"], entry["start_line"], entry["end_line"]))
        for entry in normalized["context_excerpts"]
    )
    return AdmittedPlan(normalized, captured_commands, excerpts), ()


__all__ = [
    "GIT_BRANCH_BASIS",
    "GIT_PULL_REQUEST_POLICY",
    "GIT_PUSH_POLICY",
    "STOP_REQUIRED_ACTION",
    "AssemblyIssue",
    "assemble_plan",
    "AdmittedPlan",
    "render_issue",
]
