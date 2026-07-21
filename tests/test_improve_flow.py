import json
import re
import subprocess
from datetime import date
from pathlib import Path
from typing import Any

import anyio
import pytest

from daydream.backends import ResultEvent, TextEvent, ToolResultEvent, ToolStartEvent
from daydream.config import AUDIT_CATEGORIES, EFFORT_TIERS
from daydream.config_file import DaydreamFileConfig
from daydream.exploration_runner import repo_scan
from daydream.extensions.loader import build_registry
from daydream.git_ops import head_sha
from daydream.improve.orchestrator import (
    _RECON_SCHEMA,
    _apply_vet_verdicts,
    _build_recon_prompt,
)
from daydream.improve.plans import render_plan
from daydream.improve.prompts import PLAN_WRITER_SCHEMA
from daydream.runner import RunConfig, run


def _plan_command(
    purpose: str,
    observable: str,
    *,
    paths: list[str] | None = None,
) -> dict[str, Any]:
    scoped_paths = paths or []
    return {
        "purpose": purpose,
        "command": "uv run pytest apps/billing/test_api.py -q",
        "working_directory": ".",
        "expected_success": {
            "exit_code": 0,
            "observable_result": observable,
        },
        "applicability": {
            "scope": (
                {"kind": "in-scope-paths", "paths": scoped_paths}
                if scoped_paths
                else {"kind": "whole-repository"}
            ),
            "preconditions": [],
            "rationale": "The focused command verifies the requested billing behavior.",
        },
        "provenance": {
            "kind": "planner-derived",
            "recon_command_id": "test-suite",
            "source_path": "apps/billing/api.py",
        },
    }


def _typed_plan_result(
    finding: dict[str, Any],
    target: Path,
) -> dict[str, Any]:
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
    step_gate = _plan_command(
        "Verify the billing service implementation",
        "exit 0 and the named billing regression test passes",
        paths=["apps/billing/api.py", "apps/billing/test_api.py"],
    )
    test_gate = _plan_command(
        "Run the named billing regression",
        "exit 0 and test_service_name_preserves_contract passes",
    )
    scope_gate = {
        **_plan_command(
            "Confirm unrelated documentation is unchanged",
            "exit 0 and README.md has no changes",
        ),
        "command": "git diff --exit-code -- README.md",
        "provenance": {
            "kind": "planner-derived",
            "recon_command_id": "git-diff",
            "source_path": "apps/billing/api.py",
        },
    }
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
        "current_state_excerpts": [
            {
                "path": "apps/billing/api.py",
                "line_anchor": {"start_line": 1, "end_line": 2},
                "file_role": "Billing service name implementation under change.",
                "verbatim_excerpt": (
                    target / "apps/billing/api.py"
                ).read_text(encoding="utf-8").rstrip("\n"),
            }
        ],
        "commands_you_will_need": [
            _plan_command(
                "Run focused billing tests",
                "exit 0 and all focused billing tests pass",
            )
        ],
        "scope": {
            "existing_paths": [
                {
                    "path": "apps/billing/api.py",
                    "role": (
                        "Implement the selected billing behavior"
                        f"{requested_context}."
                    ),
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
        "git_workflow": {
            "branch_name": "use the operator's current branch",
            "branch_basis": "Recon found no repository branch naming convention.",
            "commit_boundaries": "Commit the billing behavior and test as one logical unit.",
            "commit_message_example": "fix: preserve billing service contract",
            "push_policy": "never-without-operator-instruction",
            "pull_request_policy": "never-without-operator-instruction",
        },
        "steps": [
            {
                "id": "step-1",
                "order": 1,
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
                "id": "step-2",
                "order": 2,
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
                "id": "done-1",
                "kind": "behavior",
                "description": (
                    "service_name implements the requested billing behavior"
                    f"{requested_context}."
                ),
                "verification": step_gate,
            },
            {
                "id": "done-2",
                "kind": "test-gate",
                "description": "The named billing service regression test passes.",
                "verification": test_gate,
            },
            {
                "id": "done-3",
                "kind": "scope-integrity",
                "description": "No file outside the declared billing scope changes.",
                "verification": scope_gate,
            },
        ],
        "stop_conditions": [
            {
                "kind": "drift",
                "condition": (
                    "apps/billing/api.py no longer matches the quoted service_name excerpt."
                ),
                "required_action": "STOP_AND_REPORT",
                "evidence_to_report": (
                    "Report the live service_name definition and planned-at commit."
                ),
                "related_paths": ["apps/billing/api.py"],
                "related_step_ids": ["step-1"],
            },
            {
                "kind": "repeated-verification-failure",
                "condition": "The focused billing verification command fails twice.",
                "required_action": "STOP_AND_REPORT",
                "evidence_to_report": "Report both command outputs and attempted correction.",
                "related_paths": ["apps/billing/test_api.py"],
                "related_step_ids": ["step-2"],
            },
            {
                "kind": "out-of-scope-change",
                "condition": "The billing change requires editing README.md or catalog files.",
                "required_action": "STOP_AND_REPORT",
                "evidence_to_report": "Report the required path and why it is necessary.",
                "related_paths": ["README.md"],
                "related_step_ids": ["step-1"],
            },
            {
                "kind": "false-assumption",
                "condition": (
                    "The service_name public return type is not a string "
                    f"contract{requested_context}."
                ),
                "required_action": "STOP_AND_REPORT",
                "evidence_to_report": (
                    "Report the actual callers and observed return contract"
                    f"{requested_context}."
                ),
                "related_paths": ["apps/billing/api.py"],
                "related_step_ids": ["step-1"],
            },
        ],
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
        self.fanout_concurrency = fanout_concurrency
        self.audit_delay = audit_delay
        self.audit_active = 0
        self.audit_peak = 0
        self.calls: list[dict[str, Any]] = []
        self.fail_categories: set[str] = set()
        self.vet_reject_titles: set[str] = set()
        self.return_legacy_plan = False
        self.return_degenerate_plan = False
        self.return_secret_invalid_priority = False
        self.inject_credential = False
        self.all_recon_commands_invalid = False
        self.recon_commands_override: list[dict[str, Any]] | None = None
        self.recon_languages_override: Any = None
        self.recon_output_override: Any = None
        self.abort_plan_on_tool_budget = False
        self.plan_dependency_slug: str | None = None
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
        elif "You are writing a self-contained implementation plan" in prompt:
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
            commands = self.recon_commands_override or [
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
                        "scope": {
                            "kind": (
                                "unsupported"
                                if self.all_recon_commands_invalid
                                else "whole-repository"
                            ),
                        },
                        "preconditions": [],
                        "rationale": (
                            "The root configuration declares the test command."
                        ),
                    },
                    "evidence": {
                        "kind": "literal-command",
                        "source_path": "pyproject.toml",
                        "line_anchor": {
                            "start_line": 5,
                            "end_line": 5,
                        },
                        "verbatim_excerpt": (
                            'test-command = "uv run pytest"'
                        ),
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
                        "scope": {
                            "kind": (
                                "unsupported"
                                if self.all_recon_commands_invalid
                                else "whole-repository"
                            ),
                        },
                        "preconditions": [],
                        "rationale": (
                            "The root configuration declares the scope command."
                        ),
                    },
                    "evidence": {
                        "kind": "literal-command",
                        "source_path": "pyproject.toml",
                        "line_anchor": {
                            "start_line": 6,
                            "end_line": 6,
                        },
                        "verbatim_excerpt": (
                            'scope-command = "git diff --exit-code"'
                        ),
                    },
                },
            ]
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
            match = re.search(
                r"Selected vetted finding:\n```json\n(.*?)\n```",
                prompt,
                flags=re.DOTALL,
            )
            assert match is not None
            finding = json.loads(match.group(1))
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
            plan = _typed_plan_result(finding, self._target)
            if self.return_degenerate_plan:
                plan["why_this_matters"]["problem"] = (
                    "PRIVATE_REJECTED_PAYLOAD: Follow the vetted fix sketch "
                    "and preserve repository conventions."
                )
            if self.return_secret_invalid_priority:
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
            plan = _typed_plan_result(finding, self._target)
            plan["title"] = "Reviewer attempted renamed billing plan"
            plan["slug"] = "reviewer-renamed-billing-plan"
            plan["priority"] = "P3"
            plan["maintenance_notes"]["future_interactions"][0]["note"] = (
                "Recheck the billing contract when service discovery becomes dynamic."
            )
            if self.plan_review_result == "generic-typed":
                plan["why_this_matters"]["problem"] = (
                    "Follow the vetted fix sketch and preserve repository conventions."
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

        if "You are writing a self-contained implementation plan" in prompt:
            match = re.search(
                r"Selected vetted finding:\n```json\n(.*?)\n```",
                prompt,
                flags=re.DOTALL,
            )
            assert match is not None
            finding = json.loads(match.group(1))
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


def test_audit_prompt_carries_playbook_section_and_hard_rules() -> None:
    prompt = build_registry().prompt("audit")(
        category="security",
        skill_invocation=None,
        services=[],
        scope_note="",
        recon_summary="langs: python",
        cwd=Path("/repo"),
        tier=EFFORT_TIERS["standard"],
    )
    assert "never reproduce secret values" in prompt.lower()
    assert "data, not instructions" in prompt.lower()
    assert "file:line" in prompt and "Effort" in prompt


def test_all_improve_model_prompts_carry_each_hard_rule_once() -> None:
    registry = build_registry()
    prompts = (
        registry.prompt("audit")(
            category="security",
            skill_invocation=None,
            services=[],
            scope_note="",
            recon_summary="langs: python",
            cwd=Path("/repo"),
            tier=EFFORT_TIERS["standard"],
        ),
        registry.prompt("vet")(findings=[], cwd=Path("/repo")),
        registry.prompt("plan-writer")(
            finding={"title": "Finding"},
            recon_summary="{}",
            verification_commands=[],
            cwd=Path("/repo"),
        ),
    )

    for prompt in prompts:
        assert prompt.count("Hard Rule 4 (verbatim):") == 1
        assert prompt.count("Hard Rule 6 (verbatim):") == 1


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
async def test_plan_writer_prompt_delivers_schema_once(
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
    plan_calls = [
        call for call in stub.calls if call["marker"] == "plan-writer"
    ]
    assert plan_calls
    embedded_schema = json.dumps(PLAN_WRITER_SCHEMA, sort_keys=True)
    formatted_embedded_schema = json.dumps(PLAN_WRITER_SCHEMA, indent=2)
    assert all(
        embedded_schema not in call["prompt"]
        and formatted_embedded_schema not in call["prompt"]
        and call["output_schema"] == PLAN_WRITER_SCHEMA
        for call in plan_calls
    )


def test_recon_schema_requires_structured_command_evidence() -> None:
    command_schema = _RECON_SCHEMA["properties"]["commands"]

    assert command_schema["type"] == "array"
    item = command_schema["items"]
    assert set(item["required"]) == {
        "id",
        "purpose",
        "command",
        "working_directory",
        "expected_success",
        "applicability",
        "evidence",
    }
    evidence_variants = item["properties"]["evidence"]["oneOf"]
    assert {variant["properties"]["kind"]["const"] for variant in evidence_variants} == {
        "literal-command",
        "make-target",
        "package-script",
    }
    applicability = item["properties"]["applicability"]
    assert set(applicability["required"]) == {
        "scope",
        "preconditions",
        "rationale",
    }


def test_recon_prompt_explains_canonical_applicability_and_evidence() -> None:
    prompt = _build_recon_prompt(Path("/repo"), [], "No conventions found.")

    assert "scope" in prompt
    assert "whole-repository" in prompt
    assert "in-scope-paths" in prompt
    assert "preconditions" in prompt
    assert "literal-command" in prompt
    assert "make-target" in prompt
    assert "package-script" in prompt
    assert "independent" in prompt


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
async def test_shelfspace_shaped_recon_retains_six_commands_and_starts_audit(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    makefile_lines = [
        "build:",
        "test:",
        "test-frontend:",
        "lint:",
        "typecheck:",
    ]
    (improve_monorepo_target / "Makefile").write_text(
        "\n".join(makefile_lines) + "\n",
        encoding="utf-8",
    )
    admin = improve_monorepo_target / "admin-dashboard"
    admin.mkdir()
    package_lines = [
        "{",
        '  "packageManager": "pnpm@10.0.0",',
        '  "scripts": {',
        '    "test": "vitest run"',
        "  }",
        "}",
    ]
    (admin / "package.json").write_text(
        "\n".join(package_lines) + "\n",
        encoding="utf-8",
    )
    stub = _install_improve_stub(
        monkeypatch,
        improve_monorepo_target,
        n_findings=0,
    )

    def make_command(
        target: str,
        line: int,
        *,
        preconditions: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "id": f"make-{target}",
            "purpose": f"Run the repository {target} automation target",
            "command": f"make {target}",
            "working_directory": ".",
            "expected_success": {
                "exit_code": 0,
                "observable_result": f"exit 0 and the {target} target succeeds",
            },
            "applicability": {
                "scope": {"kind": "whole-repository"},
                "preconditions": preconditions or [],
                "rationale": (
                    "The root Makefile declares the exact repository automation target."
                ),
            },
            "evidence": {
                "kind": "make-target",
                "source_path": "Makefile",
                "line_anchor": {"start_line": line, "end_line": line},
                "verbatim_excerpt": f"{target}:",
                "target": target,
            },
        }

    stub.recon_commands_override = [
        make_command(
            "build",
            1,
            preconditions=["Docker daemon is available when container images are built"],
        ),
        make_command("test", 2),
        make_command("test-frontend", 3),
        {
            "id": "admin-dashboard-test",
            "purpose": "Run the admin dashboard unit test suite",
            "command": "pnpm test",
            "working_directory": "admin-dashboard",
            "expected_success": {
                "exit_code": 0,
                "observable_result": "exit 0 and the Vitest test suite passes",
            },
            "applicability": {
                "scope": {
                    "kind": "in-scope-paths",
                    "paths": ["admin-dashboard/"],
                },
                "preconditions": ["pnpm dependencies are installed"],
                "rationale": (
                    "The admin dashboard package manifest declares this test script."
                ),
            },
            "evidence": {
                "kind": "package-script",
                "source_path": "admin-dashboard/package.json",
                "line_anchor": {"start_line": 4, "end_line": 4},
                "verbatim_excerpt": '    "test": "vitest run"',
                "package_manager": "pnpm",
                "script": "test",
                "working_directory": "admin-dashboard",
            },
        },
        make_command("lint", 4),
        make_command("typecheck", 5),
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
    recon = json.loads(_dd(improve_monorepo_target, "recon.json").read_text())
    assert [command["command"] for command in recon["commands"]] == [
        "make build",
        "make test",
        "make test-frontend",
        "pnpm test",
        "make lint",
        "make typecheck",
    ]
    assert recon["commands"][0]["applicability"]["preconditions"]
    assert recon["commands"][3]["applicability"]["scope"]["paths"] == [
        "admin-dashboard"
    ]
    assert [call for call in stub.calls if call["marker"] == "audit"]


@pytest.mark.anyio
async def test_improve_stops_before_fanout_when_recon_has_no_valid_commands(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
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

    assert code == 1
    assert not [call for call in stub.calls if call["marker"] == "audit"]
    assert not [call for call in stub.calls if call["marker"] == "plan-writer"]
    assert not list(
        (improve_monorepo_target / "daydream_plans").glob(
            "[0-9][0-9][0-9]-*.md"
        )
    )
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
    console_output = capsys.readouterr().out
    assert "2 repository command candidates were found but rejected" in console_output
    assert "RECON_APPLICABILITY_INVALID: 2" in console_output
    assert "command-validation-diagnostics.json" in console_output
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
async def test_invalid_recon_container_rejects_every_candidate_with_diagnostics(
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

    assert code == 1
    assert not [call for call in stub.calls if call["marker"] == "audit"]
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
        "accepted": 0,
        "rejected": 2,
    }
    assert diagnostics["container_errors"] == [expected_error]
    assert [item["candidate_id"] for item in diagnostics["rejections"]] == [
        "test-suite",
        "git-diff",
    ]
    assert all(
        item["validation_stage"] == "container"
        and item["errors"] == [expected_error]
        for item in diagnostics["rejections"]
    )
    diagnostics_text = diagnostics_path.read_text()
    recon_text = _dd(improve_monorepo_target, "recon.json").read_text()
    assert "secret-model-prose" not in diagnostics_text + recon_text
    assert "verbatim_excerpt" not in diagnostics_text
    assert "uv run pytest" not in diagnostics_text
    recon = json.loads(recon_text)
    assert recon["commands"] == []
    assert recon["command_rejections"] == [expected_error]


@pytest.mark.anyio
async def test_non_array_commands_persist_container_diagnostics_before_stop(
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
    assert not [call for call in stub.calls if call["marker"] == "audit"]
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
    normalized_console = " ".join(console_output.lower().split())
    assert "command container was rejected" in normalized_console
    assert "before candidates" in normalized_console
    assert "RECON_COMMANDS_INVALID: 1" in console_output
    for private_value in (secret, model_prose, rejected_command):
        assert private_value not in console_output

    trajectory_path = next(
        (improve_monorepo_target / ".daydream" / "runs").glob(
            "*/trajectory.json"
        )
    )
    trajectory = json.loads(trajectory_path.read_text())
    event = next(
        event
        for event in trajectory["extra"]["phase_events"]
        if event["event"] == "command_validation"
    )
    assert event["metadata"]["counts"] == diagnostics["counts"]
    assert event["metadata"]["reasons"] == {"RECON_COMMANDS_INVALID": 1}
    assert event["metadata"]["container_errors"] == [error]


@pytest.mark.anyio
async def test_missing_recon_fields_have_distinct_container_error_pointers(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _install_improve_stub(monkeypatch, improve_monorepo_target)
    stub.recon_output_override = {"commands": []}

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        )
    )

    assert code == 1
    diagnostics = json.loads(
        _dd(
            improve_monorepo_target,
            "command-validation-diagnostics.json",
        ).read_text()
    )
    expected = [
        {"code": "RECON_CONTAINER_INVALID", "pointer": "/languages"},
        {"code": "RECON_CONTAINER_INVALID", "pointer": "/conventions"},
        {"code": "RECON_CONTAINER_INVALID", "pointer": "/intent_docs"},
    ]
    assert diagnostics["container_errors"] == expected
    assert len({error["pointer"] for error in diagnostics["container_errors"]}) == 3
    assert diagnostics["rejections"] == []

    trajectory_path = next(
        (improve_monorepo_target / ".daydream" / "runs").glob(
            "*/trajectory.json"
        )
    )
    trajectory = json.loads(trajectory_path.read_text())
    event = next(
        event
        for event in trajectory["extra"]["phase_events"]
        if event["event"] == "command_validation"
    )
    assert event["metadata"]["container_errors"] == expected


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
    assert any(
        "beagle-python:review-python" in call["prompt"]
        for call in audit_calls
    )


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("effort", "backend_hint", "expected_peak"),
    [
        ("standard", 10, 10),
        ("standard", 4, 4),
        ("quick", 10, 1),
    ],
    ids=["pi-ten", "claude-four", "quick-pi-one"],
)
async def test_improve_effective_backend_concurrency(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    effort: str,
    backend_hint: int,
    expected_peak: int,
) -> None:
    stub = _install_improve_stub(
        monkeypatch,
        improve_monorepo_target,
        fanout_concurrency=backend_hint,
        audit_delay=0.01,
    )

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            improve_effort=effort,
            non_interactive=True,
            archive=False,
        )
    )

    assert code == 0
    assert stub.audit_peak == expected_peak


@pytest.mark.anyio
async def test_pi_improve_retains_42_commands_and_writes_ten_plans(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _force_interactive(monkeypatch)
    monkeypatch.setattr(
        "daydream.agent.prompt_user",
        lambda *args, **kwargs: "1-10",
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
    assert len(recon["commands"]) == 42
    assert recon["command_rejections"] == [
        {
            "code": "RECON_MALFORMED_COMMAND",
            "pointer": "/commands/42/command",
        }
    ]
    assert len(plan_files) == 10
    assert backend.peak_active == 10
    assert all(backend.recon_secret not in text for text in observables)
    assert all(backend.planner_secret not in text for text in observables)


@pytest.mark.anyio
async def test_pi_improve_partial_failure_is_nonzero_and_safe(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _force_interactive(monkeypatch)
    monkeypatch.setattr(
        "daydream.agent.prompt_user",
        lambda *args, **kwargs: "1-10",
    )
    backend = _ProductionPathBackend(
        improve_monorepo_target,
        failed_title="Production finding 06",
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

    assert code == 1
    assert len(plan_files) == 9
    assert len(failed) == 1
    assert failed[0]["finding"]["title"] == "Production finding 06"
    assert failed[0]["errors"] == [{"code": "PROCESS_EXIT", "pointer": "/"}]
    assert "Plan blocked for Production finding 06: PROCESS_EXIT at /." in console_output
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
    stub = _install_improve_stub(monkeypatch, improve_monorepo_target)
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
    audit_prompts = [
        call["prompt"] for call in stub.calls if call["marker"] == "audit"
    ]
    assert audit_prompts and all(
        "4–6 grounded suggestions" in prompt for prompt in audit_prompts
    )
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
    assert not list(plans_dir.glob("[0-9][0-9][0-9]-*.md"))
    assert "BLOCKED (PLAN_VALIDATION_FAILED: LEGACY_MARKDOWN_OUTPUT" in (
        plans_dir / "README.md"
    ).read_text()
    plan_calls = [
        call for call in stub.calls if call["marker"] == "plan-writer"
    ]
    report = _dd(improve_monorepo_target, "report.md").read_text()
    assert (
        f"Plans blocked by plan-writing failure: {len(plan_calls)}" in report
    )
    assert len(plan_calls) == 3


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
        "daydream.runner.DEFAULT_TOOL_CALL_BUDGET",
        1,
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
async def test_real_improve_flow_writes_complete_typed_plan_and_attempt_diagnostics(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        )
    )

    assert code == 0
    plan_calls = [
        call for call in stub.calls if call["marker"] == "plan-writer"
    ]
    assert len(plan_calls) >= 2
    assert all(call["read_only"] for call in plan_calls)
    assert len({id(call) for call in plan_calls}) == len(plan_calls)
    selected = json.loads(
        _dd(improve_monorepo_target, "selected.json").read_text(encoding="utf-8")
    )
    assert len(selected["selected"]) == len(plan_calls)

    generated = sorted(plans_dir.glob("[0-9][0-9][0-9]-high-leverage-title.md"))
    assert len(generated) == 1
    plan = generated[0].read_text(encoding="utf-8")
    planned_at = head_sha(improve_monorepo_target)
    assert (
        f"git diff --stat {planned_at}..HEAD -- "
        "apps/billing/api.py apps/billing/test_api.py"
    ) in plan
    assert (
        "`apps/billing/api.py:1-2` — Billing service name implementation "
        "under change."
    ) in plan
    assert 'def service_name():\n    return "billing"' in plan
    assert (
        "| Run focused billing tests | "
        "`uv run pytest apps/billing/test_api.py -q` | "
        "exit 0; exit 0 and all focused billing tests pass |"
    ) in plan
    assert (
        "`apps/billing/api.py` (existing) — Implement the selected billing behavior."
    ) in plan
    assert (
        "`apps/billing/test_api.py` (create) — Add named regression coverage for billing."
    ) in plan
    assert (
        "`README.md` — The billing implementation does not alter documentation."
    ) in plan
    assert "The public service_name return type remains a string." in plan
    assert "### Step 1: Implement the billing service behavior" in plan
    assert "`apps/billing/api.py` — `service_name` (modify)" in plan
    assert "### Step 2: Add the named billing regression" in plan
    assert (
        "`apps/billing/test_api.py::test_service_name_preserves_contract`"
    ) in plan
    assert "The returned value is the expected billing service string." in plan
    assert "**done-1 (behavior)**" in plan
    assert "**done-2 (test-gate)**" in plan
    assert "**done-3 (scope-integrity)**" in plan
    assert (
        "**false-assumption** — The service_name public return type is not "
        "a string contract."
    ) in plan
    assert "## Maintenance notes" in plan
    assert "**Billing service discovery**" in plan
    assert planned_at[:7] in plan

    index = (plans_dir / "README.md").read_text(encoding="utf-8")
    generated_number = int(generated[0].name.split("-", 1)[0])
    assert (
        f"| [{generated_number:03d}]({generated[0].name}) "
        "<!-- fingerprint:"
    ) in index
    assert "| high-leverage-title | P1 | S | billing-foundation | TODO |" in index
    assert (
        f"{generated_number:03d} depends on billing-foundation because "
        "The billing implementation must follow the established billing "
        "foundation contract."
    ) in index

    diagnostics = json.loads(
        _dd(
            improve_monorepo_target,
            "plan-write-diagnostics.json",
        ).read_text(encoding="utf-8")
    )
    assert diagnostics["schema_version"] == 1
    assert diagnostics["artifact_type"] == "daydream.plan-write-diagnostics"
    assert len(diagnostics["attempts"]) == len(plan_calls)
    assert any(
        attempt["disposition"] == "success"
        and attempt["artifact"]["path"] == generated[0].name
        for attempt in diagnostics["attempts"]
    )
    assert all(
        attempt["planner"]["backend"] == "_ImproveStubBackend"
        and attempt["planner"]["model"] == "mock-model"
        and attempt["planner"]["descriptor"].startswith("plan-")
        for attempt in diagnostics["attempts"]
    )
    assert len(
        {
            attempt["planner"]["descriptor"]
            for attempt in diagnostics["attempts"]
        }
    ) == len(plan_calls)
    assert all(attempt["finding"]["fingerprint"] for attempt in diagnostics["attempts"])


@pytest.mark.anyio
async def test_real_improve_flow_blocks_semantically_degenerate_plans_actionably(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    stub = _install_improve_stub(
        monkeypatch,
        improve_monorepo_target,
        n_findings=1,
    )
    stub.return_degenerate_plan = True

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        )
    )

    assert code == 1
    plans_dir = improve_monorepo_target / "daydream_plans"
    assert not list(plans_dir.glob("[0-9][0-9][0-9]-*.md"))
    plan_calls = [
        call for call in stub.calls if call["marker"] == "plan-writer"
    ]
    assert len(plan_calls) >= 2
    index = (plans_dir / "README.md").read_text(encoding="utf-8")
    report = _dd(improve_monorepo_target, "report.md").read_text(encoding="utf-8")
    console_output = capsys.readouterr().out
    assert index.count(
        "BLOCKED (PLAN_VALIDATION_FAILED: DEGENERATE_CONTENT)"
    ) == 1
    assert "DEGENERATE_CONTENT" in report
    assert report.count("- **high-leverage-title** (") == len(plan_calls)
    assert "DEGENERATE_CONTENT" in console_output

    diagnostic_path = _dd(
        improve_monorepo_target,
        "plan-write-diagnostics.json",
    )
    diagnostics_text = diagnostic_path.read_text(encoding="utf-8")
    diagnostics = json.loads(diagnostics_text)
    assert len(diagnostics["attempts"]) == len(plan_calls)
    assert {
        (attempt["stage"], attempt["disposition"])
        for attempt in diagnostics["attempts"]
    } == {("semantic", "blocked")}
    assert all(
        attempt["errors"] == [
            {
                "code": "DEGENERATE_CONTENT",
                "pointer": "/",
            }
        ]
        for attempt in diagnostics["attempts"]
    )
    assert all(attempt["received"]["type"] == "object" for attempt in diagnostics["attempts"])
    assert all("schema_version" not in attempt["received"] for attempt in diagnostics["attempts"])
    assert all(attempt["received"]["sha256"] for attempt in diagnostics["attempts"])
    assert all(attempt["received"]["serialized_length"] > 100 for attempt in diagnostics["attempts"])
    for observable in (index, report, console_output, diagnostics_text):
        assert "PRIVATE_REJECTED_PAYLOAD" not in observable
        assert "Follow the vetted fix sketch" not in observable


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

    assert code == 1
    audit_calls = [call for call in stub.calls if call["marker"] == "audit"]
    assert all("apps/billing" in call["prompt"] for call in audit_calls)
    assert any(
        "bounds where the audit searches" in call["prompt"].lower()
        for call in audit_calls
    )
    report = _dd(improve_monorepo_target, "report.md").read_text()
    assert "catalog" in report and "not audited" in report.lower()


@pytest.mark.anyio
async def test_group_scope_expands_named_service_group_to_all_members(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ``--scope <group>`` must expand the named group from improve_service_groups
    # to its members (previously this raised ValueError and stopped with code 1).
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

    assert code == 1
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
    typed = _typed_plan_result(finding, target)
    typed["slug"] = "billing-contract"
    typed["title"] = finding["title"]
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
async def test_review_plan_rejects_semantically_generic_typed_plan_unchanged(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, index, _, _ = _write_review_plan_fixture(
        improve_monorepo_target
    )
    original = plan.read_bytes()
    original_index = index.read_bytes()
    stub = _install_improve_stub(monkeypatch, improve_monorepo_target)
    stub.plan_review_result = "generic-typed"

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
    assert f"git diff --stat {planned_at}..HEAD -- " in rewritten
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
