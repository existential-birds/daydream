import json
import re
import subprocess
from datetime import date
from pathlib import Path
from typing import Any

import anyio
import pytest

from daydream.backends import ResultEvent, TextEvent, ToolResultEvent, ToolStartEvent
from daydream.config import AUDIT_CATEGORIES
from daydream.config_file import DaydreamFileConfig
from daydream.exploration_runner import repo_scan
from daydream.extensions.loader import build_registry
from daydream.git_ops import head_sha
from daydream.improve.assemble import assemble_plan
from daydream.improve.orchestrator import (
    _apply_vet_verdicts,
)
from daydream.improve.plans import render_plan
from daydream.improve.prompts import PLAN_AUTHOR_SCHEMA
from daydream.runner import RunConfig, run


def _plan_ref(
    recon_command_id: str,
    *,
    appended_args: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    return {
        "recon_command_id": recon_command_id,
        "appended_args": appended_args,
        "note": note,
    }


def _stub_recon_commands(*, all_invalid: bool = False) -> list[dict[str, Any]]:
    scope_kind = "unsupported" if all_invalid else "whole-repository"
    return [
        {
            "id": "test-suite",
            "purpose": "Run the repository Python test suite",
            "command": "uv run pytest",
            "working_directory": ".",
            "expected_success": {
                "exit_code": 0,
                "observable_result": (
                    "exit 0 and the selected pytest tests pass"
                ),
            },
            "applicability": {
                "scope": {"kind": scope_kind},
                "preconditions": [],
                "rationale": (
                    "The root configuration declares the test command."
                ),
            },
            "evidence": {
                "kind": "literal-command",
                "source_path": "pyproject.toml",
                "line_anchor": {"start_line": 5, "end_line": 5},
                "verbatim_excerpt": 'test-command = "uv run pytest"',
            },
        },
        {
            "id": "git-diff",
            "purpose": "Check that unrelated paths remain unchanged",
            "command": "git diff --exit-code",
            "working_directory": ".",
            "expected_success": {
                "exit_code": 0,
                "observable_result": (
                    "exit 0 and no unexpected diff is reported"
                ),
            },
            "applicability": {
                "scope": {"kind": scope_kind},
                "preconditions": [],
                "rationale": (
                    "The root configuration declares the scope command."
                ),
            },
            "evidence": {
                "kind": "literal-command",
                "source_path": "pyproject.toml",
                "line_anchor": {"start_line": 6, "end_line": 6},
                "verbatim_excerpt": 'scope-command = "git diff --exit-code"',
            },
        },
    ]


def _authored_plan_result(finding: dict[str, Any]) -> dict[str, Any]:
    requested_context = (
        f" for the requested change: {finding['title']}"
        if finding.get("category") == "requested"
        else ""
    )
    raw_slug = re.sub(
        r"[^a-z0-9]+", "-", finding["title"].lower()
    ).strip("-")
    slug = raw_slug if len(raw_slug) >= 3 else f"requested-change-{raw_slug}"
    title = (
        f"Spike {finding['title']}"
        if finding.get("category") == "direction"
        else (
            finding["title"]
            if len(finding["title"]) >= 12
            else f"Implement requested change {finding['title']}"
        )
    )
    step_gate = _plan_ref(
        "test-suite",
        appended_args="apps/billing/test_api.py -q",
        note="The focused billing suite proves the changed behavior.",
    )
    test_gate = _plan_ref(
        "test-suite",
        appended_args="apps/billing/test_api.py -q",
    )
    scope_gate = _plan_ref(
        "git-diff",
        appended_args="-- README.md",
        note="Documentation must stay untouched by the billing change.",
    )
    return {
        "slug": slug,
        "title": title,
        "priority": "P1",
        "dependencies": [],
        "why_this_matters": {
            "problem": (
                f"{finding['title']} affects the billing service contract today."
            ),
            "concrete_cost": (
                "Leaving the billing behavior unchanged creates avoidable "
                "maintenance and verification cost."
            ),
            "intended_outcome": (
                "The billing service has an explicit implementation and named "
                "regression coverage."
            ),
        },
        "scope": {
            "existing_paths": [
                {
                    "path": "apps/billing/api.py",
                    "role": (
                        "Implement the selected billing behavior"
                        f"{requested_context}."
                    ),
                    "excerpts": [{"start_line": 1, "end_line": 2}],
                }
            ],
            "new_paths": [
                {
                    "path": "apps/billing/test_api.py",
                    "role": (
                        "Add named regression coverage for billing"
                        f"{requested_context}."
                    ),
                }
            ],
            "out_of_scope_paths": [
                {
                    "path": "README.md",
                    "reason": "The billing implementation does not alter documentation.",
                }
            ],
            "out_of_scope_behaviors": [
                {
                    "behavior": "The public service_name return type remains a string.",
                    "reason": "Existing service consumers depend on the return type.",
                }
            ],
        },
        "context_excerpts": [],
        "git_workflow": {
            "commit_boundaries": "Commit the billing behavior and test as one logical unit.",
            "commit_message_example": "fix: preserve billing service contract",
        },
        "steps": [
            {
                "title": "Implement the billing service behavior",
                "changes": [
                    {
                        "path": "apps/billing/api.py",
                        "symbol": "service_name",
                        "operation": "modify",
                        "instruction": (
                            "Implement the requested behavior at the service_name "
                            f"boundary{requested_context}."
                        ),
                        "target_state": (
                            "service_name preserves its string contract with "
                            f"explicit behavior{requested_context}."
                        ),
                    }
                ],
                "verification": step_gate,
            },
            {
                "title": "Add the named billing regression",
                "changes": [
                    {
                        "path": "apps/billing/test_api.py",
                        "symbol": "test_service_name_preserves_contract",
                        "operation": "create",
                        "instruction": (
                            "Create a focused regression for the service_name "
                            f"contract{requested_context}."
                        ),
                        "target_state": (
                            "test_service_name_preserves_contract checks the "
                            f"requested behavior{requested_context}."
                        ),
                    }
                ],
                "verification": test_gate,
            },
        ],
        "test_plan": {
            "exemplars": [
                {
                    "path": "apps/catalog/api.py",
                    "symbol": "service_name",
                    "pattern_to_copy": (
                        "Use the neighboring service's direct service_name function style."
                    ),
                }
            ],
            "cases": [
                {
                    "name": (
                        "Billing service preserves the requested contract"
                        f"{requested_context}"
                    ),
                    "test_file": "apps/billing/test_api.py",
                    "test_symbol": "test_service_name_preserves_contract",
                    "kind": "unit",
                    "setup": (
                        "Import service_name from the billing API module"
                        f"{requested_context}."
                    ),
                    "action": (
                        "Call service_name once without additional arguments"
                        f"{requested_context}."
                    ),
                    "assertions": [
                        "The returned value is the expected billing service "
                        f"string{requested_context}."
                    ],
                    "verification": test_gate,
                }
            ],
        },
        "done_criteria": [
            {
                "kind": "behavior",
                "description": (
                    "service_name implements the requested billing behavior"
                    f"{requested_context}."
                ),
                "verification": step_gate,
            },
            {
                "kind": "test-gate",
                "description": "The named billing service regression test passes.",
                "verification": test_gate,
            },
            {
                "kind": "scope-integrity",
                "description": "No file outside the declared billing scope changes.",
                "verification": scope_gate,
            },
        ],
        "false_assumption": {
            "condition": (
                "The service_name public return type is not a string "
                f"contract{requested_context}."
            ),
            "evidence_to_report": (
                "Report the actual callers and observed return contract"
                f"{requested_context}."
            ),
            "related_paths": ["apps/billing/api.py"],
            "related_step_numbers": [1],
        },
        "additional_stop_conditions": [],
        "additional_command_refs": [],
        "maintenance_notes": {
            "future_interactions": [
                {
                    "area": "Billing service discovery",
                    "note": "Revisit the contract if service discovery becomes dynamic.",
                }
            ],
            "review_risks": [
                {
                    "risk": "The billing service return contract could change accidentally.",
                    "review_check": "Confirm the named regression exercises the public boundary.",
                }
            ],
            "deferred_items": [],
        },
    }


def _without_verification_commands(plan: dict[str, Any]) -> dict[str, Any]:
    """Represent a useful plan when recon found no host-verified commands."""
    plan["additional_command_refs"] = []
    for step in plan["steps"]:
        step["verification"] = None
    for case in plan["test_plan"]["cases"]:
        case["verification"] = None
    for criterion in plan["done_criteria"]:
        criterion["verification"] = None
    return plan


def _finding_from_prompt(prompt: str) -> dict[str, Any]:
    for block in re.findall(r"```json\n(.*?)\n```", prompt, flags=re.DOTALL):
        try:
            candidate = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "fingerprint" in candidate:
            return candidate
    raise AssertionError("no finding block in plan-writer prompt")


class _ImproveStubBackend:
    """Backend stub for improve-flow tests."""

    model = "mock-model"

    def __init__(
        self,
        target: Path,
        *,
        n_findings: int | None = None,
        attempt_write: bool = False,
        fanout_concurrency: int = 4,
        audit_delay: float = 0,
    ) -> None:
        self._target = target
        self._n_findings = n_findings
        self._attempt_write = attempt_write
        self._write_attempted = False
        self.reasoning_effort: str | None = None
        self.fanout_concurrency = fanout_concurrency
        self.audit_delay = audit_delay
        self.audit_active = 0
        self.audit_peak = 0
        self.calls: list[dict[str, Any]] = []
        self.fail_categories: set[str] = set()
        self.vet_reject_titles: set[str] = set()
        self.return_legacy_plan = False
        self.return_secret_invalid_priority = False
        self.return_secret_invalid_priority_once = False
        self.plan_writer_calls = 0
        self.inject_credential = False
        self.all_recon_commands_invalid = False
        self.recon_commands_override: list[dict[str, Any]] | None = None
        self.recon_commands_extra: list[dict[str, Any]] = []
        self.recon_languages_override: Any = None
        self.recon_output_override: Any = None
        self.abort_plan_on_tool_budget = False
        self.plan_tool_calls_before_result = 0
        self.plan_dependency_slug: str | None = None
        self.plan_file_role_override: str | None = None
        self.plan_problem_override: str | None = None
        self.plan_instruction_override: str | None = None
        self.plan_ungate_steps = False
        self.plan_sloppy = False
        self.plan_bad_recon_id_attempts = 0
        self.plan_missing_path_attempts = 0
        self.plan_review_result = "typed"

    async def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: Any = None,
        continuation: Any = None,
        agents: Any = None,
        max_turns: Any = None,
        read_only: bool = False,
        persist_session: bool = True,
    ):
        marker = "other"
        category = None
        if "you are the **pattern-scanner** specialist" in prompt.lower():
            marker = "repo-scan"
        elif "IMPROVE_RECON" in prompt:
            marker = "recon"
        elif "read-only improve audit specialist" in prompt:
            marker = "audit"
            headings = {
                "correctness": "## Correctness / Bugs",
                "security": "## Security",
                "performance": "## Performance",
                "tests": "## Test Coverage",
                "tech-debt": "## Tech Debt & Architecture",
                "dependencies": "## Dependencies & Migrations",
                "dx": "## DX & Tooling",
                "docs": "## Docs",
                "direction": "## Direction",
            }
            category = next(
                name for name, heading in headings.items() if heading in prompt
            )
        elif "You are the improve vet." in prompt:
            marker = "vet"
        elif "You are reviewing an existing daydream implementation plan" in prompt:
            marker = "plan-reviewer"
        elif "You are writing a self-contained implementation plan" in prompt or (
            isinstance(output_schema, dict)
            and "false_assumption" in output_schema.get("properties", {})
        ):
            marker = "plan-writer"
        self.calls.append(
            {
                "cwd": cwd,
                "prompt": prompt,
                "output_schema": output_schema,
                "agents": agents,
                "max_turns": max_turns,
                "read_only": read_only,
                "persist_session": persist_session,
                "marker": marker,
                # Observed at the execute seam: what this turn actually ran on.
                "model": self.model,
                "reasoning_effort": self.reasoning_effort,
            }
        )
        if self._attempt_write and not self._write_attempted:
            self._write_attempted = True
            write_path = self._target / "agent-write-attempt.txt"
            yield ToolStartEvent(
                id="improve-write-attempt",
                name="Write",
                input={
                    "file_path": str(write_path),
                    "content": "must not be written",
                },
            )
            if read_only:
                yield ToolResultEvent(
                    id="improve-write-attempt",
                    output="denied by read-only backend profile",
                    is_error=True,
                )
            else:
                write_path.write_text("must not be written")
                yield ToolResultEvent(
                    id="improve-write-attempt",
                    output="written",
                    is_error=False,
                )
        if category in self.fail_categories:
            raise RuntimeError(f"{category} audit failed")
        if category is not None and self.audit_delay:
            self.audit_active += 1
            self.audit_peak = max(self.audit_peak, self.audit_active)
            try:
                await anyio.sleep(self.audit_delay)
            finally:
                self.audit_active -= 1
        if "you are the **pattern-scanner** specialist" in prompt.lower():
            yield ResultEvent(
                structured_output={
                    "conventions": [
                        {
                            "name": "OpenAPI First",
                            "description": "openapi.yaml is the HTTP contract",
                            "source": "CLAUDE.md",
                        }
                    ],
                    "guidelines": [],
                },
                continuation=None,
            )
            return
        if "IMPROVE_RECON" in prompt:
            if self.recon_output_override is not None:
                yield ResultEvent(
                    structured_output=self.recon_output_override,
                    continuation=None,
                )
                return
            commands = self.recon_commands_override or _stub_recon_commands(
                all_invalid=self.all_recon_commands_invalid
            )
            commands = [*commands, *self.recon_commands_extra]
            yield ResultEvent(
                structured_output={
                    "languages": (
                        self.recon_languages_override
                        if self.recon_languages_override is not None
                        else ["python", "typescript"]
                    ),
                    "commands": commands,
                    "conventions": ["OpenAPI First"],
                    "intent_docs": ["README.md"],
                },
                continuation=None,
            )
            return
        if category is not None:
            category_index = AUDIT_CATEGORIES.index(category)
            if (
                self._n_findings is not None
                and category_index >= self._n_findings
            ):
                yield ResultEvent(
                    structured_output={"findings": []},
                    continuation=None,
                )
                return
            title = f"{category.title()} finding"
            impact = "HIGH"
            effort = "S"
            if category == "correctness":
                title = "high-leverage-title"
            elif category == "docs":
                title = "low-leverage-title"
                impact = "LOW"
                effort = "L"
            findings = [
                {
                    "title": (
                        f"{title} OPENAI_API_KEY=sk-secret123456"
                        if self.inject_credential
                        else title
                    ),
                    "category": "wrong-agent-category",
                    "path": "apps/billing/api.py",
                    "line": 1,
                    "body": f"Concrete {category} impact and fix.",
                    "impact": impact,
                    "effort": effort,
                    "risk": "LOW",
                    "confidence": "HIGH",
                    "evidence": ["apps/billing/api.py:1"],
                }
            ]
            if category == "performance":
                findings.append(
                    {
                        "title": "Phantom N+1",
                        "category": "wrong-agent-category",
                        "path": "apps/catalog/api.py",
                        "line": 1,
                        "body": "Claims a query loop that is not present.",
                        "impact": "HIGH",
                        "effort": "S",
                        "risk": "LOW",
                        "confidence": "HIGH",
                        "evidence": ["apps/catalog/api.py:1"],
                    }
                )
            yield ResultEvent(
                structured_output={"findings": findings},
                continuation=None,
            )
            return
        if marker == "vet":
            match = re.search(
                r"Candidates .*?:\n```json\n(.*?)\n```",
                prompt,
                flags=re.DOTALL,
            )
            assert match is not None
            candidates = json.loads(match.group(1))
            yield ResultEvent(
                structured_output={
                    "verdicts": [
                        {
                            "vet_id": candidate["vet_id"],
                            "keep": candidate["title"]
                            not in self.vet_reject_titles,
                            "reason": (
                                "No query loop exists at the cited location."
                                if candidate["title"] in self.vet_reject_titles
                                else "Confirmed from cited evidence."
                            ),
                            "severity": None,
                            "impact": candidate["impact"],
                            "effort": candidate["effort"],
                            "risk": candidate["risk"],
                            "confidence": candidate["confidence"],
                            "path": candidate["path"],
                            "line": candidate["line"],
                        }
                        for candidate in candidates
                    ]
                },
                continuation=None,
            )
            return
        if marker == "plan-writer":
            self.plan_writer_calls += 1
            if self.inject_credential:
                yield TextEvent(text="OPENAI_API_KEY=sk-secret123456")
            if self.abort_plan_on_tool_budget:
                for index in range(3):
                    yield ToolStartEvent(
                        id=f"plan-budget-{index}",
                        name="Read",
                        input={"file_path": "apps/billing/api.py"},
                    )
                return
            for index in range(self.plan_tool_calls_before_result):
                yield ToolStartEvent(
                    id=f"plan-read-{index}",
                    name="Read",
                    input={"file_path": "apps/billing/api.py"},
                )
            finding = _finding_from_prompt(prompt)
            if self.return_legacy_plan:
                yield ResultEvent(
                    structured_output={
                        "slug": "legacy-plan",
                        "title": "Legacy Markdown implementation plan",
                        "priority": "P1",
                        "depends_on": [],
                        "markdown": "## Steps\n\nMake the change.",
                    },
                    continuation=None,
                )
                return
            plan = _authored_plan_result(finding)
            if self.plan_file_role_override is not None:
                plan["scope"]["existing_paths"][0]["role"] = (
                    self.plan_file_role_override
                )
            if self.plan_problem_override is not None:
                plan["why_this_matters"]["problem"] = self.plan_problem_override
            if self.plan_instruction_override is not None:
                plan["steps"][0]["changes"][0]["instruction"] = (
                    self.plan_instruction_override
                )
            if self.plan_ungate_steps:
                # The shape a plan takes when recon verified no commands: the
                # contract tells the writer to use null verification everywhere.
                for step in plan["steps"]:
                    step["verification"] = None
                for case in plan["test_plan"]["cases"]:
                    case["verification"] = None
                for criterion in plan["done_criteria"]:
                    criterion["verification"] = None
                plan["additional_command_refs"] = []
            if self.plan_sloppy:
                plan["debug_notes"] = (
                    "Planner scratch notes that must never reach the artifact."
                )
                plan["markdown"] = "## Steps\n\nMake the change."
                plan["scope"]["out_of_scope_paths"].append(
                    {
                        "path": "apps/billing/api.py",
                        "reason": (
                            "The billing implementation file must not change further."
                        ),
                    }
                )
                plan["scope"]["existing_paths"][0]["role"] = (
                    "Billing role " + "x" * 293
                )
                plan["why_this_matters"]["problem"] = (
                    "Callers keep sending secret: hunter2realvalue in "
                    "production traffic today."
                )
            if self.plan_writer_calls <= self.plan_bad_recon_id_attempts:
                plan["steps"][0]["verification"] = _plan_ref("make-tests")
            if self.plan_writer_calls <= self.plan_missing_path_attempts:
                plan["scope"]["existing_paths"].append(
                    {
                        "path": "apps/billing/legacy_api.py",
                        "role": "Reference the retired billing module for parity.",
                        "excerpts": [{"start_line": 1, "end_line": 2}],
                    }
                )
            if self.all_recon_commands_invalid:
                plan = _without_verification_commands(plan)
            if self.return_secret_invalid_priority or (
                self.return_secret_invalid_priority_once
                and self.plan_writer_calls == 1
            ):
                plan["priority"] = "TOKEN=PRIVATE_SCHEMA_SECRET"
                plan["title"] = "PRIVATE_SCHEMA_SECRET rejected title"
                plan["dependencies"] = [
                    {
                        "slug": "private-schema-secret",
                        "reason": "PRIVATE_SCHEMA_SECRET rejected dependency",
                    }
                ]
            if self.inject_credential:
                plan["priority"] = "OPENAI_API_KEY=sk-secret123456"
            if self.plan_dependency_slug:
                plan["dependencies"] = [
                    {
                        "slug": self.plan_dependency_slug,
                        "reason": (
                            "The billing implementation must follow the established "
                            "billing foundation contract."
                        ),
                    }
                ]
            yield ResultEvent(
                structured_output=plan,
                continuation=None,
            )
            return
        if marker == "plan-reviewer":
            if self.plan_review_result == "markdown":
                yield ResultEvent(
                    structured_output={
                        "critique": "The original plan lacked executable detail.",
                        "markdown": (
                            "# Plan 001: Fix\n\n"
                            "## Status\n\n- **Priority**: P1\n\n"
                            "## Why this matters\n\nConcrete impact.\n\n"
                            "## Current state\n\n`apps/billing/api.py:1`.\n\n"
                            "## Commands you will need\n\n"
                            "| Purpose | Command | Expected on success |\n"
                            "|---|---|---|\n"
                            "| Test | `uv run pytest` | exit 0 |\n\n"
                            "## Scope\n\nOnly `apps/billing/api.py`.\n\n"
                            "## Steps\n\n### Step 1\n\nMake the change.\n\n"
                            "## Test plan\n\nRun `uv run pytest`.\n\n"
                            "## Done criteria\n\n- [ ] Tests pass.\n\n"
                            "## STOP conditions\n\nStop on drift.\n"
                        ),
                    },
                    continuation=None,
                )
                return
            finding = {
                "title": "Preserve the billing service contract",
                "category": "correctness",
            }
            plan = _authored_plan_result(finding)
            plan["title"] = "Reviewer attempted renamed billing plan"
            plan["slug"] = "reviewer-renamed-billing-plan"
            plan["priority"] = "P3"
            plan["maintenance_notes"]["future_interactions"][0]["note"] = (
                "Recheck the billing contract when service discovery becomes dynamic."
            )
            yield ResultEvent(
                structured_output={
                    "critique": "The original plan lacked executable detail.",
                    "plan": plan,
                },
                continuation=None,
            )
            return
        raise AssertionError(f"unexpected improve prompt: {prompt[:120]}")

    async def cancel(self) -> None:
        pass

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
        return f"/{skill_key}"


class _ProductionPathPlannerError(RuntimeError):
    category = "PROCESS_EXIT"
    retryable = False


class _ProductionPathRateLimitError(RuntimeError):
    category = "RATE_LIMIT"
    retryable = True


class _ProductionPathBackend(_ImproveStubBackend):
    """Delayed Pi-shaped backend for full improve production-path regressions."""

    recon_secret = "OPENAI_API_KEY=sk-task10-recon-secret123456"
    planner_secret = "ZAI_API_KEY=task10-planner-secret123456"

    def __init__(self, target: Path, *, failed_title: str | None = None) -> None:
        super().__init__(
            target,
            n_findings=len(AUDIT_CATEGORIES),
            fanout_concurrency=10,
        )
        self.failed_title = failed_title
        self.audit_findings_issued = 0
        self.plan_active = 0
        self.peak_active = 0

    def _recon_commands(self) -> list[dict[str, Any]]:
        commands: list[dict[str, Any]] = []
        for index in range(42):
            test_command = index % 2 == 0
            commands.append(
                {
                    "id": (
                        "test-suite"
                        if index == 0
                        else "git-diff"
                        if index == 1
                        else f"verified-command-{index:02d}"
                    ),
                    "purpose": f"Run verified repository command {index + 1}",
                    "command": (
                        "uv run pytest"
                        if test_command
                        else "git diff --exit-code"
                    ),
                    "working_directory": ".",
                    "expected_success": {
                        "exit_code": 0,
                        "observable_result": (
                            "exit 0 and the repository command succeeds"
                        ),
                    },
                    "applicability": {
                        "scope": {"kind": "whole-repository"},
                        "preconditions": [],
                        "rationale": (
                            "The root configuration declares this command."
                        ),
                    },
                    "evidence": {
                        "kind": "literal-command",
                        "source_path": "pyproject.toml",
                        "line_anchor": {
                            "start_line": 5 if test_command else 6,
                            "end_line": 5 if test_command else 6,
                        },
                        "verbatim_excerpt": (
                            'test-command = "uv run pytest"'
                            if test_command
                            else 'scope-command = "git diff --exit-code"'
                        ),
                    },
                }
            )
        commands.append(
            {
                **commands[0],
                "id": "malformed-secret-bearing-command",
                "purpose": f"Rejected planner content {self.recon_secret}",
                "command": "pytest ...",
            }
        )
        return commands

    async def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: Any = None,
        continuation: Any = None,
        agents: Any = None,
        max_turns: Any = None,
        read_only: bool = False,
        persist_session: bool = True,
    ):
        if "IMPROVE_RECON" in prompt:
            self.calls.append(
                {
                    "cwd": cwd,
                    "prompt": prompt,
                    "output_schema": output_schema,
                    "agents": agents,
                    "max_turns": max_turns,
                    "read_only": read_only,
                    "persist_session": persist_session,
                    "marker": "recon",
                }
            )
            yield ResultEvent(
                structured_output={
                    "languages": ["python", "typescript"],
                    "commands": self._recon_commands(),
                    "conventions": ["OpenAPI First"],
                    "intent_docs": ["README.md"],
                },
                continuation=None,
            )
            return

        if "read-only improve audit specialist" in prompt:
            headings = {
                "correctness": "## Correctness / Bugs",
                "security": "## Security",
                "performance": "## Performance",
                "tests": "## Test Coverage",
                "tech-debt": "## Tech Debt & Architecture",
                "dependencies": "## Dependencies & Migrations",
                "dx": "## DX & Tooling",
                "docs": "## Docs",
                "direction": "## Direction",
            }
            category = next(
                name for name, heading in headings.items() if heading in prompt
            )
            self.calls.append(
                {
                    "cwd": cwd,
                    "prompt": prompt,
                    "output_schema": output_schema,
                    "agents": agents,
                    "max_turns": max_turns,
                    "read_only": read_only,
                    "persist_session": persist_session,
                    "marker": "audit",
                }
            )
            if self.audit_findings_issued >= 10:
                findings: list[dict[str, Any]] = []
            else:
                self.audit_findings_issued += 1
                finding_number = self.audit_findings_issued
                findings = [
                    {
                        "title": f"Production finding {finding_number:02d}",
                        "category": category,
                        "path": "apps/billing/api.py",
                        "line": 1,
                        "body": (
                            "The billing service contract needs explicit "
                            f"production-path coverage {finding_number:02d}."
                        ),
                        "impact": "HIGH",
                        "effort": "S",
                        "risk": "LOW",
                        "confidence": "HIGH",
                        "evidence": ["apps/billing/api.py:1"],
                    }
                ]
            yield ResultEvent(
                structured_output={"findings": findings},
                continuation=None,
            )
            return

        if "You are writing a self-contained implementation plan" in prompt or (
            isinstance(output_schema, dict)
            and "false_assumption" in output_schema.get("properties", {})
        ):
            finding = _finding_from_prompt(prompt)
            if self.plan_active >= 2:
                raise _ProductionPathRateLimitError(
                    "provider plan concurrency limit exceeded"
                )
            self.plan_active += 1
            self.peak_active = max(self.peak_active, self.plan_active)
            try:
                await anyio.sleep(0.05)
                if finding["title"] == self.failed_title:
                    raise _ProductionPathPlannerError(f"planner process metadata {self.planner_secret}")
            finally:
                self.plan_active -= 1

        async for event in super().execute(
            cwd,
            prompt,
            output_schema=output_schema,
            continuation=continuation,
            agents=agents,
            max_turns=max_turns,
            read_only=read_only,
            persist_session=persist_session,
        ):
            yield event


@pytest.fixture
def tmp_git_repo(improve_monorepo_target: Path) -> Path:
    return improve_monorepo_target


def _install_improve_stub(
    monkeypatch: pytest.MonkeyPatch,
    target: Path,
    *,
    n_findings: int | None = None,
    attempt_write: bool = False,
    fanout_concurrency: int = 4,
    audit_delay: float = 0,
) -> _ImproveStubBackend:
    stub = _ImproveStubBackend(
        target,
        n_findings=n_findings,
        attempt_write=attempt_write,
        fanout_concurrency=fanout_concurrency,
        audit_delay=audit_delay,
    )
    monkeypatch.setattr("daydream.runner.create_backend", lambda *args, **kwargs: stub)
    return stub


def _install_per_phase_improve_stubs(
    monkeypatch: pytest.MonkeyPatch,
    target: Path,
) -> list[dict[str, Any]]:
    """Install a factory that mints one stub per resolved backend triple.

    ``_resolve_backend`` caches on ``(name, model, reasoning_effort)``, so one
    stub per triple is exactly one stub per distinct phase tier. Every stub
    shares a single ``calls`` list, and each recorded call carries the
    ``model``/``reasoning_effort`` the turn actually ran on — the observable at
    the ``Backend.execute`` seam.
    """
    shared_calls: list[dict[str, Any]] = []

    def _factory(
        name: str,
        model: str | None = None,
        *,
        cwd: Path | None = None,
        reasoning_effort: str | None = None,
    ) -> _ImproveStubBackend:
        stub = _ImproveStubBackend(target)
        stub.calls = shared_calls
        stub.model = model or "mock-model"
        stub.reasoning_effort = reasoning_effort
        return stub

    monkeypatch.setattr("daydream.runner.create_backend", _factory)
    return shared_calls


def _tiers_by_marker(calls: list[dict[str, Any]]) -> dict[str, set[tuple[str, str | None]]]:
    tiers: dict[str, set[tuple[str, str | None]]] = {}
    for call in calls:
        tiers.setdefault(str(call["marker"]), set()).add(
            (str(call["model"]), call["reasoning_effort"])
        )
    return tiers


def _git_status_porcelain(repo: Path) -> str:
    return subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _untracked(repo: Path) -> list[str]:
    return subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()


def _scan_trajectory_extra(run_root: Path, traj: Path, key: str) -> list[str]:
    values: list[str] = []
    for trajectory_file in list(run_root.rglob("*.json")) + (
        [traj] if traj.exists() else []
    ):
        try:
            payload = json.loads(trajectory_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict):
            continue
        for step in payload.get("steps", []):
            value = (step.get("extra") or {}).get(key)
            if value:
                values.append(value)
    return values


def _dd(repo: Path, name: str) -> Path:
    return repo / ".daydream" / "improve" / name


def _improve_observable_texts(repo: Path) -> list[str]:
    return [
        path.read_text(encoding="utf-8")
        for root in (
            repo / ".daydream" / "improve",
            repo / ".daydream" / "runs",
            repo / "daydream_plans",
        )
        if root.exists()
        for path in root.rglob("*")
        if path.is_file()
    ]


def _forbidden_input(*_args: Any, **_kwargs: Any) -> str:
    raise AssertionError(
        "input() was called in non-interactive mode -- stdin must not be touched"
    )


def _force_interactive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("daydream.runner._stdin_isatty", lambda: True)
    monkeypatch.delenv("CI", raising=False)


@pytest.mark.anyio
async def test_repo_scan_seeds_specialists_from_tracked_files(tmp_git_repo: Path) -> None:
    stub = _ImproveStubBackend(tmp_git_repo)
    ctx = await repo_scan(stub, tmp_git_repo, max_files=500)
    assert any(c.name == "OpenAPI First" for c in ctx.conventions)
    prompt = stub.calls[0]["prompt"]
    assert stub.calls[0]["marker"] == "repo-scan"
    assert "api.py" in prompt
    assert stub.calls[0]["read_only"] is True


def test_registry_seeds_audit_slots_and_improve_prompts() -> None:
    r = build_registry()
    assert r.skill("audit:correctness:python") == "beagle-python:review-python"
    assert r.skill("audit:security:elixir") == "beagle-elixir:elixir-security-review"
    assert r.skill_if_registered("audit:dx") is None
    for name in ("audit", "vet", "plan-writer"):
        assert callable(r.prompt(name))


@pytest.mark.anyio
async def test_credentials_never_reach_improve_observables(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    secret = "OPENAI_API_KEY=sk-secret123456"
    stub = _install_improve_stub(
        monkeypatch,
        improve_monorepo_target,
        n_findings=1,
    )
    stub.inject_credential = True

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        )
    )

    assert code == 1
    console_output = capsys.readouterr().out
    artifact_texts = [
        path.read_text(encoding="utf-8")
        for root in (
            improve_monorepo_target / ".daydream" / "improve",
            improve_monorepo_target / ".daydream" / "runs",
            improve_monorepo_target / "daydream_plans",
        )
        if root.exists()
        for path in root.rglob("*")
        if path.is_file()
    ]
    observables = [console_output, *artifact_texts]
    assert all(secret not in observable for observable in observables)
    assert "[REDACTED" in "\n".join(observables)


@pytest.mark.anyio
async def test_improve_recon_writes_artifacts_and_never_mutates_source(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = _install_improve_stub(monkeypatch, improve_monorepo_target)
    before = _git_status_porcelain(improve_monorepo_target)
    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        )
    )
    assert code == 0
    dd = improve_monorepo_target / ".daydream" / "improve"
    services = json.loads((dd / "services.json").read_text())
    assert {service["name"] for service in services["services"]} == {
        "billing",
        "catalog",
    }
    assert (dd / "report.md").is_file()
    trajectories = list(
        (improve_monorepo_target / ".daydream" / "runs").glob(
            "*/trajectory.json"
        )
    )
    assert len(trajectories) == 1
    trajectory = json.loads(trajectories[0].read_text())
    assert trajectory["steps"]
    assert all(
        step["extra"]["daydream_run_flow"] == "improve"
        for step in trajectory["steps"]
    )
    assert any(
        step["extra"]["daydream_phase"] == "recon"
        for step in trajectory["steps"]
    )
    run_marker = f"Daydream run: `{trajectory['session_id']}`"
    assert run_marker in (dd / "report.md").read_text()
    plans_dir = improve_monorepo_target / "daydream_plans"
    assert run_marker in (plans_dir / "README.md").read_text()
    assert all(
        run_marker in plan.read_text()
        for plan in plans_dir.glob("[0-9][0-9][0-9]-*.md")
    )
    plan_diagnostics = json.loads(
        (dd / "plan-write-diagnostics.json").read_text()
    )
    assert (
        plan_diagnostics["artifact_provenance"]["session_id"]
        == trajectory["session_id"]
    )
    assert all(call["read_only"] for call in stub.calls)
    assert _git_status_porcelain(improve_monorepo_target) == before


@pytest.mark.anyio
async def test_run_with_no_findings_writes_report_and_empty_plan_diagnostics(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _install_improve_stub(
        monkeypatch,
        improve_monorepo_target,
        n_findings=0,
    )

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        )
    )

    assert code == 0
    assert [call for call in stub.calls if call["marker"] == "audit"]
    assert not [call for call in stub.calls if call["marker"] == "plan-writer"]
    diagnostics = json.loads(
        _dd(
            improve_monorepo_target,
            "plan-write-diagnostics.json",
        ).read_text(encoding="utf-8")
    )
    assert diagnostics["attempts"] == []
    assert "Plans written: 0" in _dd(
        improve_monorepo_target,
        "report.md",
    ).read_text(encoding="utf-8")


@pytest.mark.anyio
async def test_improve_continues_audit_and_planning_when_recon_has_no_valid_commands(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _install_improve_stub(monkeypatch, improve_monorepo_target)
    stub.all_recon_commands_invalid = True

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        )
    )

    assert [call for call in stub.calls if call["marker"] == "audit"]
    assert [call for call in stub.calls if call["marker"] == "plan-writer"]
    recon = json.loads(_dd(improve_monorepo_target, "recon.json").read_text())
    assert recon["commands"] == []
    diagnostics_path = _dd(
        improve_monorepo_target,
        "command-validation-diagnostics.json",
    )
    diagnostics = json.loads(diagnostics_path.read_text())
    assert diagnostics["counts"] == {
        "total_candidates": 2,
        "accepted": 0,
        "rejected": 2,
    }
    assert [item["candidate_id"] for item in diagnostics["rejections"]] == [
        "test-suite",
        "git-diff",
    ]
    assert all(
        item["evidence"] == {
            "source_path": "pyproject.toml",
            "line_anchor": {"start_line": 5 + index, "end_line": 5 + index},
        }
        and item["validation_stage"] == "schema"
        and item["errors"] == [
            {
                "code": "RECON_APPLICABILITY_INVALID",
                "pointer": f"/commands/{index}/applicability/scope/kind",
            }
        ]
        for index, item in enumerate(diagnostics["rejections"])
    )
    diagnostics_text = diagnostics_path.read_text()
    assert "verbatim_excerpt" not in diagnostics_text
    assert "uv run pytest" not in diagnostics_text
    report = _dd(improve_monorepo_target, "report.md")
    plans = sorted(
        (improve_monorepo_target / "daydream_plans").glob(
            "[0-9][0-9][0-9]-*.md"
        )
    )
    plan_diagnostics = json.loads(
        _dd(
            improve_monorepo_target,
            "plan-write-diagnostics.json",
        ).read_text(encoding="utf-8")
    )
    assert code == 0
    assert report.is_file()
    assert plans
    assert all("**Command**" not in plan.read_text() for plan in plans)
    assert any(
        attempt["disposition"] == "success"
        for attempt in plan_diagnostics["attempts"]
    )
    assert all(call["read_only"] for call in stub.calls)
    trajectories = list(
        (improve_monorepo_target / ".daydream" / "runs").glob(
            "*/trajectory.json"
        )
    )
    trajectory = json.loads(trajectories[0].read_text())
    validation_events = [
        event
        for event in trajectory["extra"]["phase_events"]
        if event["event"] == "command_validation"
    ]
    assert validation_events[0]["metadata"]["counts"] == diagnostics["counts"]
    assert recon["artifact_provenance"]["session_id"] == trajectory["session_id"]
    assert diagnostics["artifact_provenance"]["session_id"] == trajectory["session_id"]


@pytest.mark.anyio
async def test_unrelated_recon_container_error_preserves_valid_commands(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _install_improve_stub(monkeypatch, improve_monorepo_target)
    stub.recon_languages_override = {"secret-model-prose": "must not persist"}

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        )
    )

    assert code == 0
    assert [call for call in stub.calls if call["marker"] == "audit"]
    diagnostics_path = _dd(
        improve_monorepo_target,
        "command-validation-diagnostics.json",
    )
    diagnostics = json.loads(diagnostics_path.read_text())
    expected_error = {
        "code": "RECON_CONTAINER_INVALID",
        "pointer": "/languages",
    }
    assert diagnostics["counts"] == {
        "total_candidates": 2,
        "accepted": 2,
        "rejected": 0,
    }
    assert diagnostics["container_errors"] == [expected_error]
    assert diagnostics["rejections"] == []
    diagnostics_text = diagnostics_path.read_text()
    recon_text = _dd(improve_monorepo_target, "recon.json").read_text()
    assert "secret-model-prose" not in diagnostics_text + recon_text
    assert "verbatim_excerpt" not in diagnostics_text
    assert "uv run pytest" not in diagnostics_text
    recon = json.loads(recon_text)
    assert len(recon["commands"]) == 2
    assert all(
        command["evidence"]["verbatim_excerpt"]
        in {
            'test-command = "uv run pytest"',
            'scope-command = "git diff --exit-code"',
        }
        for command in recon["commands"]
    )
    assert recon["command_rejections"] == [expected_error]


@pytest.mark.anyio
async def test_non_array_commands_preserve_diagnostics_and_continue_audit(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stub = _install_improve_stub(monkeypatch, improve_monorepo_target)
    secret = "OPENAI_API_KEY=sk-secret123456"
    model_prose = "private arbitrary model explanation"
    rejected_command = "uv run pytest --private-selection"
    stub.recon_output_override = {
        "languages": [model_prose],
        "commands": {
            "command": rejected_command,
            "verbatim_excerpt": secret,
        },
        "conventions": [model_prose],
        "intent_docs": [model_prose],
    }

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        )
    )

    assert code == 1
    assert [call for call in stub.calls if call["marker"] == "audit"]
    assert [call for call in stub.calls if call["marker"] == "plan-writer"]
    assert _dd(improve_monorepo_target, "report.md").is_file()
    diagnostics_path = _dd(
        improve_monorepo_target,
        "command-validation-diagnostics.json",
    )
    diagnostics_text = diagnostics_path.read_text()
    diagnostics = json.loads(diagnostics_text)
    error = {"code": "RECON_COMMANDS_INVALID", "pointer": "/commands"}
    assert diagnostics["counts"] == {
        "total_candidates": 0,
        "accepted": 0,
        "rejected": 0,
    }
    assert diagnostics["container_errors"] == [error]
    assert diagnostics["rejections"] == [
        {
            "candidate_id": "commands-container",
            "evidence": {"source_path": None, "line_anchor": None},
            "validation_stage": "container",
            "errors": [error],
        }
    ]
    for private_value in (secret, model_prose, rejected_command, "verbatim_excerpt"):
        assert private_value not in diagnostics_text

    console_output = capsys.readouterr().out
    for private_value in (secret, model_prose, rejected_command):
        assert private_value not in console_output

@pytest.mark.anyio
async def test_standard_effort_fans_out_all_nine_categories_read_only(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = _install_improve_stub(monkeypatch, improve_monorepo_target)
    await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        )
    )
    audited = json.loads(
        _dd(improve_monorepo_target, "audit-findings.json").read_text()
    )
    assert set(audited["categories_run"]) == set(AUDIT_CATEGORIES)
    audit_calls = [call for call in stub.calls if call["marker"] == "audit"]
    assert audit_calls and all(call["read_only"] for call in audit_calls)


@pytest.mark.anyio
async def test_pi_improve_retains_valid_commands_and_avoids_provider_overload(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _force_interactive(monkeypatch)
    monkeypatch.setattr(
        "daydream.agent.prompt_user",
        lambda *args, **kwargs: "1-5",
    )
    backend = _ProductionPathBackend(improve_monorepo_target)
    monkeypatch.setattr(
        "daydream.runner.create_backend",
        lambda *args, **kwargs: backend,
    )

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=False,
            archive=False,
        )
    )

    recon = json.loads(
        _dd(improve_monorepo_target, "recon.json").read_text(encoding="utf-8")
    )
    plan_files = sorted(
        (improve_monorepo_target / "daydream_plans").glob(
            "[0-9][0-9][0-9]-*.md"
        )
    )
    console_output = capsys.readouterr().out
    observables = [
        console_output,
        *_improve_observable_texts(improve_monorepo_target),
    ]

    assert code == 0
    assert recon["commands"]
    assert recon["command_rejections"] == [
        {
            "code": "RECON_MALFORMED_COMMAND",
            "pointer": "/commands/42/command",
        }
    ]
    assert plan_files
    assert all(backend.recon_secret not in text for text in observables)
    assert all(backend.planner_secret not in text for text in observables)


@pytest.mark.anyio
async def test_pi_improve_partial_failure_is_successful_and_safe(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _force_interactive(monkeypatch)
    monkeypatch.setattr(
        "daydream.agent.prompt_user",
        lambda *args, **kwargs: "1-5",
    )
    backend = _ProductionPathBackend(
        improve_monorepo_target,
        failed_title="Production finding 03",
    )
    monkeypatch.setattr(
        "daydream.runner.create_backend",
        lambda *args, **kwargs: backend,
    )

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=False,
            archive=False,
        )
    )

    plan_files = sorted(
        (improve_monorepo_target / "daydream_plans").glob(
            "[0-9][0-9][0-9]-*.md"
        )
    )
    diagnostics_text = _dd(
        improve_monorepo_target,
        "plan-write-diagnostics.json",
    ).read_text(encoding="utf-8")
    diagnostics = json.loads(diagnostics_text)
    failed = [
        attempt
        for attempt in diagnostics["attempts"]
        if attempt["disposition"] == "blocked"
    ]
    console_output = capsys.readouterr().out
    observables = [
        console_output,
        *_improve_observable_texts(improve_monorepo_target),
    ]

    assert code == 0
    assert len(plan_files) == 4
    assert len(failed) == 1
    assert failed[0]["finding"]["title"] == "Production finding 03"
    assert failed[0]["errors"] == [{"code": "PROCESS_EXIT", "pointer": "/"}]
    assert "Plan blocked for Production finding 03: PROCESS_EXIT at /." in console_output
    report = _dd(improve_monorepo_target, "report.md").read_text(encoding="utf-8")
    assert "Plans written: 4" in report
    assert "Plans blocked by plan-writing failure: 1" in report
    assert all(backend.recon_secret not in text for text in observables)
    assert all(backend.planner_secret not in text for text in observables)


@pytest.mark.anyio
async def test_quick_effort_restricts_categories(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_improve_stub(monkeypatch, improve_monorepo_target)
    await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            improve_effort="quick",
            non_interactive=True,
            archive=False,
        )
    )
    audited = json.loads(
        _dd(improve_monorepo_target, "audit-findings.json").read_text()
    )
    assert set(audited["categories_run"]) == {
        "correctness",
        "security",
        "tests",
    }


@pytest.mark.anyio
async def test_focus_security_audits_single_category(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_improve_stub(monkeypatch, improve_monorepo_target)
    await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            improve_focus="security",
            non_interactive=True,
            archive=False,
        )
    )
    audited = json.loads(
        _dd(improve_monorepo_target, "audit-findings.json").read_text()
    )
    assert audited["categories_run"] == ["security"]


@pytest.mark.anyio
async def test_focus_next_is_direction_only_and_plans_are_spikes(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_improve_stub(monkeypatch, improve_monorepo_target)
    await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            improve_focus="next",
            non_interactive=True,
            archive=False,
        )
    )
    audited = json.loads(
        _dd(improve_monorepo_target, "audit-findings.json").read_text()
    )
    assert audited["categories_run"] == ["direction"]
    plan = next(
        (improve_monorepo_target / "daydream_plans").glob("0*.md")
    ).read_text()
    assert "spike" in plan.lower()


@pytest.mark.anyio
async def test_branch_focus_scopes_audit_to_merge_base_diff_and_tags_provenance(
    improve_branch_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = _install_improve_stub(monkeypatch, improve_branch_target)
    await run(
        RunConfig(
            target=str(improve_branch_target),
            flow_name="improve",
            improve_focus="branch",
            non_interactive=True,
            archive=False,
        )
    )
    audit_calls = [call for call in stub.calls if call["marker"] == "audit"]
    assert all(
        "apps/billing/api.py" in call["prompt"] for call in audit_calls
    )
    vetted = json.loads(
        _dd(improve_branch_target, "vetted-findings.json").read_text()
    )
    assert {finding["provenance"] for finding in vetted["findings"]} <= {
        "introduced",
        "inherited",
    }


@pytest.mark.anyio
async def test_branch_focus_on_base_branch_reports_and_exits_cleanly(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_improve_stub(monkeypatch, improve_monorepo_target)
    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            improve_focus="branch",
            non_interactive=True,
            archive=False,
        )
    )
    assert code == 1


@pytest.mark.anyio
async def test_failed_category_is_reported_not_silently_dropped(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = _install_improve_stub(monkeypatch, improve_monorepo_target)
    stub.fail_categories = {"performance"}
    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        )
    )
    assert code == 0
    report = _dd(improve_monorepo_target, "report.md").read_text()
    assert "performance" in report.lower()
    assert "not audited" in report.lower()


@pytest.mark.anyio
async def test_vet_rejects_unconfirmed_finding_with_reason_and_persists(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = _install_improve_stub(monkeypatch, improve_monorepo_target)
    stub.vet_reject_titles = {"Phantom N+1"}
    await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        )
    )

    vetted = json.loads(
        _dd(improve_monorepo_target, "vetted-findings.json").read_text()
    )
    assert all(
        finding["title"] != "Phantom N+1" for finding in vetted["findings"]
    )
    rejected = json.loads(
        (
            improve_monorepo_target / "daydream_plans" / "rejected.json"
        ).read_text()
    )
    assert rejected["rejected"][0]["title"] == "Phantom N+1"
    assert rejected["rejected"][0]["reason"]


@pytest.mark.anyio
async def test_previously_rejected_finding_is_not_revetted_or_rereported(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = _install_improve_stub(monkeypatch, improve_monorepo_target)
    stub.vet_reject_titles = {"Phantom N+1"}
    config = RunConfig(
        target=str(improve_monorepo_target),
        flow_name="improve",
        non_interactive=True,
        archive=False,
    )
    await run(config)

    stub.calls.clear()
    await run(config)

    vet_calls = [call for call in stub.calls if call["marker"] == "vet"]
    assert all("Phantom N+1" not in call["prompt"] for call in vet_calls)
    report = _dd(improve_monorepo_target, "report.md").read_text()
    assert "previously rejected" in report.lower()


@pytest.mark.anyio
async def test_non_interactive_selects_top_findings_never_touching_stdin(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("builtins.input", _forbidden_input)
    _install_improve_stub(
        monkeypatch,
        improve_monorepo_target,
        n_findings=8,
    )
    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        )
    )
    assert code == 0
    selected = json.loads(
        _dd(improve_monorepo_target, "selected.json").read_text()
    )
    assert len(selected["selected"]) == 5
    assert selected["mode"] == "non-interactive-default"


@pytest.mark.anyio
async def test_non_interactive_run_writes_plans_and_index(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_improve_stub(
        monkeypatch,
        improve_monorepo_target,
        n_findings=8,
    )
    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        )
    )
    plans_dir = improve_monorepo_target / "daydream_plans"
    plan_files = sorted(plans_dir.glob("[0-9][0-9][0-9]-*.md"))
    assert code == 0
    assert 1 <= len(plan_files) <= 5
    index = (plans_dir / "README.md").read_text()
    assert "non-interactive default" in index.lower()
    assert head_sha(improve_monorepo_target)[:7] in plan_files[0].read_text()


@pytest.mark.anyio
async def test_all_legacy_plan_results_block_and_return_failure(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = _install_improve_stub(
        monkeypatch,
        improve_monorepo_target,
        n_findings=1,
    )
    stub.return_legacy_plan = True

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        )
    )

    plans_dir = improve_monorepo_target / "daydream_plans"
    assert code == 1
    assert stub.plan_writer_calls == 2
    assert not list(plans_dir.glob("[0-9][0-9][0-9]-*.md"))
    assert "BLOCKED (PLAN_VALIDATION_FAILED: " in (
        plans_dir / "README.md"
    ).read_text()
    report = _dd(improve_monorepo_target, "report.md").read_text()
    assert "Plans written: 0" in report
    diagnostics_text = _dd(
        improve_monorepo_target,
        "plan-write-diagnostics.json",
    ).read_text(encoding="utf-8")
    assert "AUTHOR_SCHEMA_INVALID" in diagnostics_text
    assert "Make the change." not in diagnostics_text


@pytest.mark.anyio
async def test_failed_planning_reports_budget_failure_and_nonzero_exit(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stub = _install_improve_stub(
        monkeypatch,
        improve_monorepo_target,
        n_findings=1,
    )
    stub.abort_plan_on_tool_budget = True
    monkeypatch.setattr(
        "daydream.runner.IMPROVE_PHASE_BUDGETS",
        {"plan_write": (3600.0, 1)},
        raising=False,
    )

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        )
    )

    diagnostics_text = _dd(
        improve_monorepo_target,
        "plan-write-diagnostics.json",
    ).read_text(encoding="utf-8")
    diagnostics = json.loads(diagnostics_text)
    console_output = capsys.readouterr().out
    assert code == 1
    assert diagnostics["attempts"][0]["errors"] == [
        {"code": "TOOL_CALL_BUDGET_EXCEEDED", "pointer": "/"}
    ]
    assert "NO_STRUCTURED_OBJECT" not in diagnostics_text
    assert "Improve audit complete" not in console_output
    assert "Plan writing failed" in console_output


@pytest.mark.anyio
async def test_real_improve_flow_plans_from_live_dirty_source_without_running_candidates(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_path = improve_monorepo_target / "apps/billing/api.py"
    user_edit = (
        "def service_name():\n"
        '    return "billing-from-live-working-tree"\n'
        "# verify manually: touch candidate-command-ran\n"
    )
    source_path.write_text(user_edit, encoding="utf-8")
    dirty_status = _git_status_porcelain(improve_monorepo_target)
    plans_dir = improve_monorepo_target / "daydream_plans"
    plans_dir.mkdir()
    (plans_dir / "001-billing-foundation.md").write_text(
        "# Plan 001: Establish billing foundation\n",
        encoding="utf-8",
    )
    (plans_dir / "README.md").write_text(
        "# Implementation Plans\n\n"
        "## Execution order & status\n\n"
        "| Plan | Title | Priority | Effort | Depends on | Status |\n"
        "|------|-------|----------|--------|------------|--------|\n"
        "| [001](001-billing-foundation.md) "
        "<!-- fingerprint:trusted-existing-foundation --> | "
        "Establish billing foundation | P1 | S | — | TODO |\n",
        encoding="utf-8",
    )
    stub = _install_improve_stub(
        monkeypatch,
        improve_monorepo_target,
        n_findings=1,
    )
    stub.plan_dependency_slug = "billing-foundation"
    stub.recon_commands_extra = [
        {
            "id": "manual-sentinel",
            "purpose": "Manually verify candidate commands are never auto-executed",
            "command": "touch candidate-command-ran",
            "working_directory": ".",
            "expected_success": {
                "exit_code": 0,
                "observable_result": "exit 0 and the manual sentinel exists",
            },
            "applicability": {
                "scope": {"kind": "whole-repository"},
                "preconditions": [],
                "rationale": "The live source records this manual-only command.",
            },
            "evidence": {
                "kind": "literal-command",
                "source_path": "apps/billing/api.py",
                "line_anchor": {"start_line": 3, "end_line": 3},
                "verbatim_excerpt": "touch candidate-command-ran",
            },
        }
    ]

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        )
    )

    assert code == 0
    generated = sorted(plans_dir.glob("[0-9][0-9][0-9]-high-leverage-title.md"))
    assert len(generated) == 1
    plan = generated[0].read_text(encoding="utf-8")
    assert (
        'def service_name():\n    return "billing-from-live-working-tree"'
        in plan
    )
    assert "uv run pytest apps/billing/test_api.py -q" in plan
    assert "billing-foundation" in plan
    assert source_path.read_text(encoding="utf-8") == user_edit
    assert _git_status_porcelain(improve_monorepo_target) == dirty_status
    assert not (improve_monorepo_target / "candidate-command-ran").exists()

    report = _dd(improve_monorepo_target, "report.md").read_text(
        encoding="utf-8"
    )
    assert "high-leverage-title" in report
    assert _dd(
        improve_monorepo_target,
        "command-validation-diagnostics.json",
    ).is_file()
    recon = json.loads(
        _dd(improve_monorepo_target, "recon.json").read_text(encoding="utf-8")
    )
    sentinel = next(
        command
        for command in recon["commands"]
        if command["id"] == "manual-sentinel"
    )
    assert sentinel["evidence"]["verbatim_excerpt"] == (
        "# verify manually: touch candidate-command-ran"
    )

    index = (plans_dir / "README.md").read_text(encoding="utf-8")
    assert "| high-leverage-title | P1 | S | billing-foundation | TODO |" in index

    diagnostics = json.loads(
        _dd(
            improve_monorepo_target,
            "plan-write-diagnostics.json",
        ).read_text(encoding="utf-8")
    )
    assert diagnostics["artifact_type"] == "daydream.plan-write-diagnostics"
    assert any(
        attempt["disposition"] == "success"
        and attempt["artifact"]["path"] == generated[0].name
        for attempt in diagnostics["attempts"]
    )


@pytest.mark.anyio
async def test_schema_invalid_planner_metadata_never_reaches_observables(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stub = _install_improve_stub(
        monkeypatch,
        improve_monorepo_target,
        n_findings=1,
    )
    stub.return_secret_invalid_priority = True

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        )
    )

    assert code == 1
    index = (
        improve_monorepo_target / "daydream_plans/README.md"
    ).read_text(encoding="utf-8")
    report = _dd(improve_monorepo_target, "report.md").read_text(encoding="utf-8")
    diagnostics = _dd(
        improve_monorepo_target,
        "plan-write-diagnostics.json",
    ).read_text(encoding="utf-8")
    console_output = capsys.readouterr().out
    for observable in (index, report, console_output, diagnostics):
        assert "PRIVATE_SCHEMA_SECRET" not in observable
        assert "TOKEN=" not in observable
        assert "SCHEMA_INVALID" in observable


@pytest.mark.anyio
async def test_interactive_selection_honors_user_choice(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_interactive(monkeypatch)
    monkeypatch.setattr("daydream.agent.prompt_user", lambda *a, **kw: "2")
    _install_improve_stub(
        monkeypatch,
        improve_monorepo_target,
        n_findings=8,
    )
    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=False,
            archive=False,
        )
    )
    assert code == 0
    selected = json.loads(
        _dd(improve_monorepo_target, "selected.json").read_text()
    )
    assert len(selected["selected"]) == 1
    assert selected["mode"] == "interactive"


@pytest.mark.anyio
async def test_report_orders_by_leverage_and_separates_direction(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_improve_stub(
        monkeypatch,
        improve_monorepo_target,
        n_findings=9,
    )
    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        )
    )
    assert code == 0
    report = _dd(improve_monorepo_target, "report.md").read_text()
    assert report.index("high-leverage-title") < report.index(
        "low-leverage-title"
    )
    assert "## Direction" in report
    assert "not audited" in report.lower()


@pytest.mark.anyio
async def test_scope_slices_search_but_report_names_the_unaudited_rest(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = _install_improve_stub(monkeypatch, improve_monorepo_target)
    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            improve_scope="apps/billing",
            non_interactive=True,
            archive=False,
        )
    )

    assert code == 0
    audit_calls = [call for call in stub.calls if call["marker"] == "audit"]
    assert all("apps/billing" in call["prompt"] for call in audit_calls)
    report = _dd(improve_monorepo_target, "report.md").read_text()
    assert "catalog" in report and "not audited" in report.lower()


@pytest.mark.anyio
async def test_group_scope_expands_named_service_group_to_all_members(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = _install_improve_stub(monkeypatch, improve_monorepo_target)
    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            improve_scope="core",
            file_config=DaydreamFileConfig(
                improve_service_groups={"core": ["apps/billing", "apps/catalog"]}
            ),
            non_interactive=True,
            archive=False,
        )
    )

    assert code == 0
    audit_calls = [call for call in stub.calls if call["marker"] == "audit"]
    audited_paths = {call["prompt"] for call in audit_calls}
    assert any("apps/billing" in p for p in audited_paths)
    assert any("apps/catalog" in p for p in audited_paths)
    report = _dd(improve_monorepo_target, "report.md").read_text()
    # the group covered every detected service, so the unaudited list is empty
    assert "No other detected service directories." in report


@pytest.mark.anyio
async def test_plan_subverb_skips_audit_and_writes_single_plan(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_improve_stub(monkeypatch, improve_monorepo_target)
    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            improve_plan_description="add rate limiting",
            non_interactive=True,
            archive=False,
        )
    )

    assert code == 0
    assert not _dd(improve_monorepo_target, "audit-findings.json").exists()
    plans = list(
        (improve_monorepo_target / "daydream_plans").glob(
            "[0-9][0-9][0-9]-*.md"
        )
    )
    assert len(plans) == 1
    assert "rate limiting" in plans[0].read_text().lower()


@pytest.mark.anyio
async def test_plan_subverb_repairs_schema_invalid_plan_once(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _install_improve_stub(monkeypatch, improve_monorepo_target)
    stub.return_secret_invalid_priority_once = True

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            improve_plan_description="add rate limiting",
            non_interactive=True,
            archive=False,
        )
    )

    plan_calls = [
        call for call in stub.calls if call["marker"] == "plan-writer"
    ]
    assert code == 0
    assert len(plan_calls) == 2
    repair_prompt = plan_calls[1]["prompt"]
    assert "PRIVATE_SCHEMA_SECRET" not in repair_prompt
    assert "TOKEN=" not in repair_prompt
    assert all(
        call["output_schema"] == PLAN_AUTHOR_SCHEMA
        and call["read_only"] is True
        and call["persist_session"] is False
        for call in plan_calls
    )
    plans = list(
        (improve_monorepo_target / "daydream_plans").glob(
            "[0-9][0-9][0-9]-*.md"
        )
    )
    assert len(plans) == 1
    persisted = "\n".join(
        path.read_text(encoding="utf-8")
        for root in (
            improve_monorepo_target / ".daydream",
            improve_monorepo_target / "daydream_plans",
        )
        if root.exists()
        for path in root.rglob("*")
        if path.is_file()
    )
    assert "PRIVATE_SCHEMA_SECRET" not in persisted
    assert "TOKEN=" not in persisted


@pytest.mark.anyio
async def test_plan_subverb_blocks_after_one_unsuccessful_repair(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _install_improve_stub(monkeypatch, improve_monorepo_target)
    stub.return_secret_invalid_priority = True

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            improve_plan_description="add rate limiting",
            non_interactive=True,
            archive=False,
        )
    )

    plan_calls = [
        call for call in stub.calls if call["marker"] == "plan-writer"
    ]
    diagnostics = json.loads(
        _dd(
            improve_monorepo_target,
            "plan-write-diagnostics.json",
        ).read_text(encoding="utf-8")
    )
    assert code == 1
    assert len(plan_calls) == 2
    dispositions = [
        attempt["disposition"] for attempt in diagnostics["attempts"]
    ]
    assert dispositions == ["retried", "blocked"]
    for attempt in diagnostics["attempts"]:
        assert attempt["stage"] == "authoring"
        assert any(
            error["code"] == "AUTHOR_SCHEMA_INVALID"
            and error["pointer"] == "/priority"
            for error in attempt["errors"]
        )
    assert not list(
        (improve_monorepo_target / "daydream_plans").glob(
            "[0-9][0-9][0-9]-*.md"
        )
    )


@pytest.mark.anyio
async def test_plan_subverb_clamps_over_length_prose_without_repair(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _install_improve_stub(monkeypatch, improve_monorepo_target)
    over_length_role = "Billing role " + "x" * 293
    assert len(over_length_role) == 306
    stub.plan_file_role_override = over_length_role

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            improve_plan_description="add rate limiting",
            non_interactive=True,
            archive=False,
        )
    )

    plan_calls = [
        call for call in stub.calls if call["marker"] == "plan-writer"
    ]
    assert code == 0
    assert len(plan_calls) == 1
    plans = list(
        (improve_monorepo_target / "daydream_plans").glob(
            "[0-9][0-9][0-9]-*.md"
        )
    )
    assert len(plans) == 1
    plan_text = plans[0].read_text(encoding="utf-8")
    assert over_length_role[:299] + "…" in plan_text
    assert over_length_role not in plan_text
    diagnostics = json.loads(
        _dd(
            improve_monorepo_target,
            "plan-write-diagnostics.json",
        ).read_text(encoding="utf-8")
    )
    assert [
        attempt["disposition"] for attempt in diagnostics["attempts"]
    ] == ["success"]


@pytest.mark.anyio
async def test_plan_subverb_accepts_placeholder_secret_syntax(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _install_improve_stub(monkeypatch, improve_monorepo_target)
    stub.plan_problem_override = (
        "Callers must send X-Internal-Service-Secret: <internalSecret> in "
        "production and X-Internal-Service-Secret: test-secret in tests."
    )

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            improve_plan_description="add rate limiting",
            non_interactive=True,
            archive=False,
        )
    )

    plan_calls = [
        call for call in stub.calls if call["marker"] == "plan-writer"
    ]
    assert code == 0
    assert len(plan_calls) == 1
    plans = list(
        (improve_monorepo_target / "daydream_plans").glob(
            "[0-9][0-9][0-9]-*.md"
        )
    )
    assert len(plans) == 1
    plan_text = plans[0].read_text(encoding="utf-8")
    assert "X-Internal-Service-Secret: <internalSecret>" in plan_text
    assert "X-Internal-Service-Secret: test-secret" in plan_text


@pytest.mark.anyio
async def test_secret_value_never_reaches_any_artifact(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _install_improve_stub(
        monkeypatch,
        improve_monorepo_target,
        n_findings=1,
    )
    stub.plan_problem_override = (
        "Document the rotation runbook including secret: hunter2realvalue "
        "before the credential expires."
    )
    stub.plan_bad_recon_id_attempts = 1

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        )
    )

    plan_calls = [
        call for call in stub.calls if call["marker"] == "plan-writer"
    ]
    plans = list(
        (improve_monorepo_target / "daydream_plans").glob(
            "[0-9][0-9][0-9]-*.md"
        )
    )
    assert code == 0
    assert len(plan_calls) == 2
    assert len(plans) == 1
    plan_text = plans[0].read_text(encoding="utf-8")
    assert "<redacted>" in plan_text
    assert "hunter2realvalue" not in plan_text
    assert all(
        "hunter2realvalue" not in call["prompt"] for call in stub.calls
    )
    diagnostics_text = _dd(
        improve_monorepo_target, "plan-write-diagnostics.json"
    ).read_text(encoding="utf-8")
    assert "hunter2realvalue" not in diagnostics_text
    diagnostics = json.loads(diagnostics_text)
    assert [
        attempt["disposition"] for attempt in diagnostics["attempts"]
    ] == ["retried", "success"]
    assert all(
        "hunter2realvalue" not in observable
        for observable in _improve_observable_texts(improve_monorepo_target)
    )


@pytest.mark.anyio
async def test_sloppy_but_salvageable_output_is_normalized_and_written(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _install_improve_stub(monkeypatch, improve_monorepo_target)
    stub.plan_sloppy = True

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            improve_plan_description="add rate limiting",
            non_interactive=True,
            archive=False,
        )
    )

    plan_calls = [
        call for call in stub.calls if call["marker"] == "plan-writer"
    ]
    plans = list(
        (improve_monorepo_target / "daydream_plans").glob(
            "[0-9][0-9][0-9]-*.md"
        )
    )
    assert code == 0
    assert len(plan_calls) == 1
    assert len(plans) == 1
    plan_text = plans[0].read_text(encoding="utf-8")
    sloppy_role = "Billing role " + "x" * 293
    assert sloppy_role[:299] + "…" in plan_text
    assert sloppy_role not in plan_text
    assert "<redacted>" in plan_text
    assert "hunter2realvalue" not in plan_text
    assert "Make the change." not in plan_text
    assert "Planner scratch notes" not in plan_text
    assert "The billing implementation file must not change further." not in plan_text
    assert "The billing implementation does not alter documentation." in plan_text
    assert all(
        "hunter2realvalue" not in observable
        and "Planner scratch notes" not in observable
        for observable in _improve_observable_texts(improve_monorepo_target)
    )
    diagnostics = json.loads(
        _dd(
            improve_monorepo_target,
            "plan-write-diagnostics.json",
        ).read_text(encoding="utf-8")
    )
    assert [
        attempt["disposition"] for attempt in diagnostics["attempts"]
    ] == ["success"]


@pytest.mark.anyio
async def test_n_selected_findings_produce_n_plans_first_attempt(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _install_improve_stub(
        monkeypatch,
        improve_monorepo_target,
        n_findings=3,
    )
    stub.vet_reject_titles = {"Phantom N+1"}

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        )
    )

    plans_dir = improve_monorepo_target / "daydream_plans"
    plans = sorted(plans_dir.glob("[0-9][0-9][0-9]-*.md"))
    assert code == 0
    assert stub.plan_writer_calls == 3
    assert len(plans) == 3
    for plan_path in plans:
        plan_text = plan_path.read_text(encoding="utf-8")
        assert "`uv run pytest apps/billing/test_api.py -q`" in plan_text
        assert "`git diff --exit-code -- README.md`" in plan_text
        assert "exit 0 and the selected pytest tests pass" in plan_text
        assert "recon_command_id" not in plan_text
    index = (plans_dir / "README.md").read_text(encoding="utf-8")
    assert index.count("| TODO |") == 3
    assert "BLOCKED (PLAN_" not in index


@pytest.mark.anyio
async def test_bad_recon_id_gets_named_feedback_and_retry_succeeds(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _install_improve_stub(monkeypatch, improve_monorepo_target)
    stub.plan_bad_recon_id_attempts = 1
    stub.plan_missing_path_attempts = 1

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            improve_plan_description="add rate limiting",
            non_interactive=True,
            archive=False,
        )
    )

    plan_calls = [
        call for call in stub.calls if call["marker"] == "plan-writer"
    ]
    assert code == 0
    assert len(plan_calls) == 2
    repair_prompt = plan_calls[1]["prompt"]
    assert "RECON_COMMAND_UNKNOWN" in repair_prompt
    assert "/steps/0/verification" in repair_prompt
    assert "valid recon command ids: test-suite, git-diff" in repair_prompt
    assert "make-tests" not in repair_prompt
    assert "EXISTING_PATH_MISSING" in repair_prompt
    assert "/scope/existing_paths/1/path" in repair_prompt
    plans = list(
        (improve_monorepo_target / "daydream_plans").glob(
            "[0-9][0-9][0-9]-*.md"
        )
    )
    assert len(plans) == 1
    plan_text = plans[0].read_text(encoding="utf-8")
    assert "uv run pytest apps/billing/test_api.py -q" in plan_text
    assert "apps/billing/legacy_api.py" not in plan_text
    diagnostics = json.loads(
        _dd(
            improve_monorepo_target,
            "plan-write-diagnostics.json",
        ).read_text(encoding="utf-8")
    )
    assert [
        attempt["disposition"] for attempt in diagnostics["attempts"]
    ] == ["retried", "success"]
    first_attempt_codes = {
        error["code"] for error in diagnostics["attempts"][0]["errors"]
    }
    assert {"RECON_COMMAND_UNKNOWN", "EXISTING_PATH_MISSING"} <= first_attempt_codes


@pytest.mark.anyio
async def test_persistent_authoring_failure_blocks_with_full_code_list(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _install_improve_stub(monkeypatch, improve_monorepo_target)
    stub.plan_bad_recon_id_attempts = 99

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            improve_plan_description="add rate limiting",
            non_interactive=True,
            archive=False,
        )
    )

    plan_calls = [
        call for call in stub.calls if call["marker"] == "plan-writer"
    ]
    plans_dir = improve_monorepo_target / "daydream_plans"
    assert code == 1
    assert len(plan_calls) == 2
    assert not list(plans_dir.glob("[0-9][0-9][0-9]-*.md"))
    index = (plans_dir / "README.md").read_text(encoding="utf-8")
    blocked_rows = [
        line
        for line in index.splitlines()
        if "BLOCKED (PLAN_VALIDATION_FAILED: " in line
    ]
    assert len(blocked_rows) == 1
    assert re.search(r"\| 001 <!-- fingerprint:", blocked_rows[0])
    diagnostics = json.loads(
        _dd(
            improve_monorepo_target,
            "plan-write-diagnostics.json",
        ).read_text(encoding="utf-8")
    )
    dispositions = [
        attempt["disposition"] for attempt in diagnostics["attempts"]
    ]
    assert dispositions == ["retried", "blocked"]
    for attempt in diagnostics["attempts"]:
        assert attempt["stage"] == "authoring"
        assert any(
            error["code"] == "RECON_COMMAND_UNKNOWN"
            for error in attempt["errors"]
        )


@pytest.mark.anyio
async def test_review_plan_rejects_files_outside_daydream_plans(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_improve_stub(monkeypatch, improve_monorepo_target)
    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            improve_review_plan="README.md",
            non_interactive=True,
            archive=False,
        )
    )

    assert code == 1


def _write_review_plan_fixture(
    target: Path,
) -> tuple[Path, Path, str, dict[str, Any]]:
    planned_at = head_sha(target)
    finding: dict[str, Any] = {
        "title": "Preserve the billing service contract",
        "category": "correctness",
        "path": "apps/billing/api.py",
        "line": 1,
        "body": (
            "The billing service contract needs explicit implementation and "
            "named regression coverage."
        ),
        "impact": "MED",
        "effort": "M",
        "risk": "LOW",
        "confidence": "HIGH",
        "evidence": ["apps/billing/api.py:1"],
        "fingerprint": "f" * 64,
    }
    authored = _authored_plan_result(finding)
    authored["slug"] = "billing-contract"
    authored["title"] = finding["title"]
    assembled, issues = assemble_plan(
        authored,
        repo=target,
        recon_commands=_stub_recon_commands(),
    )
    assert not issues and assembled is not None
    typed = {
        **assembled,
        "slug": "billing-contract",
        "title": finding["title"],
    }
    plans_dir = target / "daydream_plans"
    plans_dir.mkdir()
    plan = plans_dir / "007-billing-contract.md"
    plan.write_text(
        render_plan(
            finding,
            plan=typed,
            planned_at=planned_at,
            number=7,
            planned_on=date(2024, 1, 2),
        ),
        encoding="utf-8",
    )
    index = plans_dir / "README.md"
    index.write_text(
        "# Implementation Plans\n\n"
        "## Execution order & status\n\n"
        "| Plan | Title | Priority | Effort | Depends on | Status |\n"
        "|------|-------|----------|--------|------------|--------|\n"
        "| [007](007-billing-contract.md) "
        f"<!-- fingerprint:{finding['fingerprint']} --> | "
        "Preserve the billing service contract | P1 | M | — | TODO |\n",
        encoding="utf-8",
    )
    return plan, index, planned_at, typed


@pytest.mark.anyio
async def test_review_plan_rejects_heading_only_markdown_and_keeps_original(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, index, _, _ = _write_review_plan_fixture(
        improve_monorepo_target
    )
    original = plan.read_bytes()
    original_index = index.read_bytes()
    stub = _install_improve_stub(monkeypatch, improve_monorepo_target)
    stub.plan_review_result = "markdown"

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            improve_review_plan=str(plan),
            non_interactive=True,
            archive=False,
        )
    )

    assert code == 1
    assert plan.read_bytes() == original
    assert index.read_bytes() == original_index
    review_call = next(
        call for call in stub.calls if call["marker"] == "plan-reviewer"
    )
    assert set(review_call["output_schema"]["required"]) == {
        "critique",
        "plan",
    }


@pytest.mark.anyio
async def test_review_plan_typed_rewrite_preserves_identity_and_index(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, index, planned_at, _ = _write_review_plan_fixture(
        improve_monorepo_target
    )
    original = plan.read_bytes()
    original_index = index.read_bytes()
    sibling = plan.parent / "008-leave-alone.md"
    sibling.write_bytes(b"sibling plan must remain byte-for-byte unchanged\n")
    sibling_original = sibling.read_bytes()
    _install_improve_stub(monkeypatch, improve_monorepo_target)

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            improve_review_plan=str(plan),
            non_interactive=True,
            archive=False,
        )
    )

    assert code == 0
    rewritten = plan.read_text(encoding="utf-8")
    assert plan.read_bytes() != original
    assert rewritten.startswith(
        "# Plan 007: Preserve the billing service contract\n"
    )
    assert 'def service_name():\n    return "billing"' in rewritten
    assert "`uv run pytest apps/billing/test_api.py -q`" in rewritten
    assert "## Scope" in rewritten
    assert "## Steps" in rewritten
    assert "## Test plan" in rewritten
    assert "## Done criteria" in rewritten
    assert "## STOP conditions" in rewritten
    assert "## Maintenance notes" in rewritten
    assert "- **Priority**: P1" in rewritten
    assert "- **Effort**: M" in rewritten
    assert "- **Risk**: LOW" in rewritten
    assert "- **Category**: correctness" in rewritten
    assert f"- **Planned at**: commit `{planned_at[:7]}`, 2024-01-02" in rewritten
    assert index.read_bytes() == original_index
    assert sibling.read_bytes() == sibling_original


@pytest.mark.anyio
async def test_full_run_leaves_tracked_tree_and_untracked_set_untouched(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_improve_stub(
        monkeypatch,
        improve_monorepo_target,
        attempt_write=True,
    )
    before_status = _git_status_porcelain(improve_monorepo_target)

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        )
    )

    assert code == 0
    assert _git_status_porcelain(improve_monorepo_target) == before_status
    new_untracked = _untracked(improve_monorepo_target)
    assert all(
        path.startswith(("daydream_plans/", ".daydream/"))
        for path in new_untracked
    )


@pytest.mark.anyio
async def test_every_agent_call_in_every_mode_is_read_only(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configs = (
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        ),
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            improve_focus="next",
            non_interactive=True,
            archive=False,
        ),
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            improve_plan_description="x",
            non_interactive=True,
            archive=False,
        ),
    )
    for config in configs:
        stub = _install_improve_stub(monkeypatch, improve_monorepo_target)
        code = await run(config)
        assert code == 0
        assert stub.calls and all(call["read_only"] for call in stub.calls)


@pytest.mark.anyio
async def test_trajectory_records_improve_flow_and_phases(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_improve_stub(monkeypatch, improve_monorepo_target)

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        )
    )

    assert code == 0
    trajectories = list(
        (improve_monorepo_target / ".daydream" / "runs").glob(
            "*/trajectory.json"
        )
    )
    assert len(trajectories) == 1
    trajectory = trajectories[0]
    run_root = trajectory.parent
    flows = _scan_trajectory_extra(
        run_root,
        trajectory,
        "daydream_run_flow",
    )
    phases = _scan_trajectory_extra(
        run_root,
        trajectory,
        "daydream_phase",
    )
    assert flows and set(flows) == {"improve"}
    assert {"recon", "audit", "vet", "plan_write"} <= set(phases)


def test_apply_vet_verdicts_matches_by_vet_id_ignoring_order() -> None:
    """Reordered verdicts must not silently drop findings.

    Regression for the positional/order-sensitive matcher: the model may
    return verdicts in any order, and matching must be by ``vet_id``.
    """
    findings = [
        {"fingerprint": "a", "title": "A", "path": "a.py", "line": 1},
        {"fingerprint": "b", "title": "B", "path": "b.py", "line": 2},
        {"fingerprint": "c", "title": "C", "path": "c.py", "line": 3},
    ]
    # Returned in reverse order with 1-based vet_ids.
    verdicts = [
        {"vet_id": 3, "keep": True, "reason": "ok"},
        {"vet_id": 1, "keep": False, "reason": "rejected"},
        {"vet_id": 2, "keep": True, "reason": "ok"},
    ]
    kept, rejected = _apply_vet_verdicts(
        findings, verdicts, rejected_at_sha="sha"
    )
    kept_fps = {f["fingerprint"] for f in kept}
    rejected_fps = {f["fingerprint"] for f in rejected}
    assert kept_fps == {"b", "c"}
    assert rejected_fps == {"a"}


def test_apply_vet_verdicts_drops_finding_when_verdict_missing() -> None:
    """A finding whose vet_id has no matching verdict is dropped (fail-closed)."""
    findings = [
        {"fingerprint": "a", "title": "A", "path": "a.py", "line": 1},
        {"fingerprint": "b", "title": "B", "path": "b.py", "line": 2},
    ]
    # Only vet_id=2 is provided; vet_id=1 (model obeying the old zero-based
    # prose would emit vet_id=0) must be dropped, not kept.
    verdicts = [{"vet_id": 2, "keep": True, "reason": "ok"}]
    kept, rejected = _apply_vet_verdicts(
        findings, verdicts, rejected_at_sha="sha"
    )
    assert {f["fingerprint"] for f in kept} == {"b"}
    assert rejected == []


@pytest.mark.anyio
async def test_improve_pi_calls_are_ephemeral(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _install_improve_stub(
        monkeypatch,
        improve_monorepo_target,
        n_findings=1,
    )

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        )
    )

    assert code == 0
    improve_calls = [
        call
        for call in stub.calls
        if call["marker"] in {"recon", "audit", "vet", "plan-writer"}
    ]
    assert {call["marker"] for call in improve_calls} == {
        "recon",
        "audit",
        "vet",
        "plan-writer",
    }
    assert all(call["persist_session"] is False for call in improve_calls)


@pytest.mark.anyio
async def test_improve_phases_resolve_their_own_model_and_reasoning_tier(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plan authoring runs on the top model tier at max reasoning; recon does not.

    Observed at the ``Backend.execute`` seam: each recorded turn carries the
    model and reasoning effort the backend serving it was constructed with.
    """
    calls = _install_per_phase_improve_stubs(monkeypatch, improve_monorepo_target)

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        )
    )

    assert code == 0
    tiers = _tiers_by_marker(calls)
    assert tiers["plan-writer"] == {("claude-opus-4-8", "max")}
    assert tiers["vet"] == {("claude-opus-4-8", "xhigh")}
    assert tiers["audit"] == {("claude-sonnet-5", "high")}
    assert tiers["recon"] == {("claude-sonnet-5", "low")}


@pytest.mark.anyio
async def test_review_plan_runs_at_top_model_tier_and_max_reasoning(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _, _, _ = _write_review_plan_fixture(improve_monorepo_target)
    calls = _install_per_phase_improve_stubs(monkeypatch, improve_monorepo_target)

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            improve_review_plan=str(plan),
            non_interactive=True,
            archive=False,
        )
    )

    assert code == 0
    review_calls = [call for call in calls if call["marker"] == "plan-reviewer"]
    assert review_calls
    assert all(
        (call["model"], call["reasoning_effort"]) == ("claude-opus-4-8", "max")
        for call in review_calls
    )


@pytest.mark.anyio
async def test_config_file_phase_override_outranks_plan_write_defaults(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``[tool.daydream.phases.plan_write]`` table still wins over the new defaults."""
    calls = _install_per_phase_improve_stubs(monkeypatch, improve_monorepo_target)

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            file_config=DaydreamFileConfig(
                phases={
                    "plan_write": {
                        "model": "claude-sonnet-5",
                        "reasoning_effort": "medium",
                    }
                }
            ),
            non_interactive=True,
            archive=False,
        )
    )

    assert code == 0
    tiers = _tiers_by_marker(calls)
    assert tiers["plan-writer"] == {("claude-sonnet-5", "medium")}
    # Unrelated phases keep their own defaults.
    assert tiers["recon"] == {("claude-sonnet-5", "low")}
    assert tiers["vet"] == {("claude-opus-4-8", "xhigh")}


@pytest.mark.anyio
async def test_improve_runs_unbudgeted_so_a_long_turn_is_never_truncated(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plan turn spending 200 tool calls completes and writes its plan.

    The flow used to cap every phase at 50 calls / 1800 s; ten of 49 archived
    audit turns recorded a real ``tool_call_budget_exceeded`` abort under it,
    and a budget abort returns partial output the flow reads as complete.
    """
    stub = _install_improve_stub(
        monkeypatch, improve_monorepo_target, n_findings=1
    )
    stub.plan_tool_calls_before_result = 200

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        )
    )

    assert code == 0
    plans = list(
        (improve_monorepo_target / "daydream_plans").glob("[0-9][0-9][0-9]-*.md")
    )
    assert len(plans) == 1
    diagnostics = json.loads(
        _dd(improve_monorepo_target, "plan-write-diagnostics.json").read_text()
    )
    assert not any(
        error["code"] in ("TOOL_CALL_BUDGET_EXCEEDED", "WALL_BUDGET_EXCEEDED")
        for attempt in diagnostics["attempts"]
        for error in attempt["errors"]
    )


@pytest.mark.anyio
async def test_long_step_instruction_reaches_the_plan_whole(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 2000-char step instruction renders in full, ending on its last word.

    The old 1500-char prose clamp cut real plan instructions off mid-sentence,
    handing the executor an order that stopped in the middle of a requirement.
    """
    stub = _install_improve_stub(monkeypatch, improve_monorepo_target)
    instruction = (
        "In apps/billing/api.py, replace the body of service_name. "
        + "Keep every existing caller working. " * 45
        + "Do NOT modify any other file in this step."
    )
    assert 1500 < len(instruction) <= 4000
    stub.plan_instruction_override = instruction

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            improve_plan_description="add rate limiting",
            non_interactive=True,
            archive=False,
        )
    )

    assert code == 0
    plan_text = next(
        (improve_monorepo_target / "daydream_plans").glob("[0-9][0-9][0-9]-*.md")
    ).read_text(encoding="utf-8")
    assert instruction in plan_text
    assert "…" not in plan_text


@pytest.mark.anyio
async def test_over_length_instruction_is_repaired_not_silently_truncated(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Past the schema ceiling the host asks for a rewrite instead of cutting."""
    stub = _install_improve_stub(monkeypatch, improve_monorepo_target)
    stub.plan_instruction_override = "Replace service_name. " + "x" * 4000

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            improve_plan_description="add rate limiting",
            non_interactive=True,
            archive=False,
        )
    )

    assert code == 1
    diagnostics = json.loads(
        _dd(improve_monorepo_target, "plan-write-diagnostics.json").read_text()
    )
    errors = [
        error
        for attempt in diagnostics["attempts"]
        for error in attempt["errors"]
    ]
    assert any(
        error["pointer"] == "/steps/0/changes/0/instruction" for error in errors
    ), errors
    # The plan writer was asked again rather than a mangled plan being written.
    assert stub.plan_writer_calls == 2
    assert not list(
        (improve_monorepo_target / "daydream_plans").glob("[0-9][0-9][0-9]-*.md")
    )


@pytest.mark.anyio
async def test_empty_secret_named_assignments_do_not_eat_the_next_line(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Redaction must not delete plan content it mistakes for a secret value.

    An instruction naming empty ``.env`` placeholders lost two of its five
    lines: each empty ``*_SECRET=``/``*_TOKEN=`` consumed the following line as
    its "value" and the replacement dropped the newline with it.
    """
    stub = _install_improve_stub(monkeypatch, improve_monorepo_target)
    stub.plan_instruction_override = (
        "Create .env.dev.example at the repository root with exactly these "
        "five empty assignment lines, in this order and with no value after "
        "any equals sign:\n"
        "CLERK_SECRET_KEY=\n"
        "CLOUDFLARE_ACCOUNT_ID=\n"
        "CLOUDFLARE_API_TOKEN=\n"
        "CLOUDFLARE_ACCOUNT_HASH=\n"
        "INTERNAL_SERVICE_SECRET="
    )

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            improve_plan_description="repoint dev env secrets",
            non_interactive=True,
            archive=False,
        )
    )

    assert code == 0
    plan_text = next(
        (improve_monorepo_target / "daydream_plans").glob("[0-9][0-9][0-9]-*.md")
    ).read_text(encoding="utf-8")
    for key in (
        "CLERK_SECRET_KEY=",
        "CLOUDFLARE_ACCOUNT_ID=",
        "CLOUDFLARE_API_TOKEN=",
        "CLOUDFLARE_ACCOUNT_HASH=",
        "INTERNAL_SERVICE_SECRET=",
    ):
        assert key in plan_text, key
    assert "[REDACTED_ENV_VAR]" not in plan_text


@pytest.mark.anyio
async def test_rendered_plan_gives_a_literal_executor_no_room_to_guess(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Walk the rendered artifact for the points a zero-context agent stalls on."""
    _install_improve_stub(monkeypatch, improve_monorepo_target, n_findings=1)

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        )
    )

    assert code == 0
    plan_path = next(
        (improve_monorepo_target / "daydream_plans").glob("[0-9][0-9][0-9]-*.md")
    )
    text = plan_path.read_text(encoding="utf-8")
    head_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=improve_monorepo_target,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    # Preconditions: the executor is told where it must be standing, with the
    # full commit id and an exact expected result per command.
    assert "## Before you start" in text
    assert f"`git cat-file -e {head_sha}^{{commit}}`" in text
    assert "`git status --porcelain` — expected: no output at all." in text
    assert head_sha in text  # full sha, not only the 7-char Status stamp
    # A moved HEAD is expected, not a stop: drift is scoped to this plan's own
    # files, because plans are executed days or weeks after they are written.
    assert "You are expected to be running it later, from a HEAD" in text
    assert "that has moved on — that is normal and is not by itself a reason to" in text
    assert f"`git diff --name-only {head_sha} HEAD --" in text
    assert "Files outside this list do not matter." in text
    # The branch comes off the executor's current HEAD, not the planned-at sha.
    assert "`git switch --create improve/" in text
    assert f"`git switch --create improve/batch-billing-contract {head_sha}`" not in text
    assert "branches from your current HEAD, which is what you want." in text

    # Why-this-matters is labelled, so the intended outcome cannot be misread
    # as a statement about the code as it stands.
    assert "- **Problem**:" in text
    assert "- **Cost of leaving it**:" in text
    assert "- **Intended outcome (does not describe the code today)**:" in text

    # No judgement calls in host-owned wording.
    assert "unless a reviewer maintains the index" not in text
    assert "Do not skip a\n> step, reorder steps, or substitute your own judgement" in text

    # Every command says where to run it.
    assert "| Purpose | Run from | Command | Expected on success |" in text
    assert "**Run from**: the repository root" in text
    assert "Run this now, before starting the next step." in text

    # Ordering and section relationships are stated, not implied.
    assert "Do these in the order they are numbered." in text
    assert "write it once, not twice." in text
    assert "Explicitly out of scope for this plan. Do not implement them." in text

    # Finishing is literal, and never `git add -A`.
    assert "## Finishing" in text
    assert "never `git add -A`" in text
    assert "git add apps/billing/api.py apps/billing/test_api.py" in text
    assert "4. Do not push and do not open a pull request." in text
    assert "from\n`TODO` to `DONE`" in text or "`TODO` to `DONE`" in text

    # The two previously unactionable STOP conditions now name the check.
    assert (
        "Before editing a file, read the exact line range quoted for it in "
        "the Current state section" in text
    )
    assert "two failures total for the same verification" in text


@pytest.mark.anyio
async def test_ungated_step_and_scope_criterion_still_get_a_real_check(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A step the model left ungated must not render a dead end.

    Five of six steps in a real replayed plan carried no command and rendered
    only "No host-verified command is attached to this step."
    """
    stub = _install_improve_stub(
        monkeypatch, improve_monorepo_target, n_findings=1
    )
    stub.plan_ungate_steps = True

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        )
    )

    assert code == 0
    text = next(
        (improve_monorepo_target / "daydream_plans").glob("[0-9][0-9][0-9]-*.md")
    ).read_text(encoding="utf-8")

    assert "No host-verified command is attached to this step." not in text
    assert "No repository command was verified during planning for this step." in text
    assert (
        "confirm every **Target state** sentence above is now literally true"
        in text
    )
    assert "From the repository root run `git status --porcelain`." in text
    # The host-injected scope-integrity criterion is always ungated by the
    # model, and is exactly the one the host can always check itself.
    assert "(scope-integrity)" in text
    assert (
        "**Check**: from the repository root run `git status --porcelain`."
        in text
    )
    assert "No host-verified command is attached." not in text
    # An ungated test case names the symbol to run and forbids guessing a runner.
    assert (
        "run only `test_service_name_preserves_contract` in "
        "`apps/billing/test_api.py` using this repository's own test runner"
        in text
    )
    assert "stop and report that — do not guess a command." in text


@pytest.mark.anyio
async def test_plan_writer_is_told_to_leave_the_executor_no_decisions(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The anti-ambiguity contract and per-field guidance reach the writer.

    Observed at the ``Backend.execute`` seam: the prompt text and the schema
    the plan-writer call actually received.
    """
    stub = _install_improve_stub(
        monkeypatch, improve_monorepo_target, n_findings=1
    )

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        )
    )

    assert code == 0
    call = next(c for c in stub.calls if c["marker"] == "plan-writer")
    prompt = call["prompt"]
    assert "cannot infer and will not look" in prompt
    assert "has never seen this repository" in prompt
    for banned in ("the relevant handler", "as appropriate", "update accordingly"):
        assert banned in prompt, banned
    assert "Length is never a reason to compress." in prompt

    changes = call["output_schema"]["properties"]["steps"]["items"][
        "properties"
    ]["changes"]["items"]["properties"]
    assert "Banned:" in changes["instruction"]["description"]
    assert changes["instruction"]["maxLength"] == 4000
    assert "re-read the file" in changes["target_state"]["description"]
    assert "verbatim from the file" in changes["symbol"]["description"]
    done = call["output_schema"]["properties"]["done_criteria"]["items"][
        "properties"
    ]["description"]["description"]
    assert "without judgement" in done
