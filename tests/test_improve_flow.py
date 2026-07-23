import json
import re
import subprocess
from pathlib import Path
from typing import Any

import anyio
import pytest

from daydream.backends import ResultEvent, TextEvent, ToolResultEvent, ToolStartEvent
from daydream.config import AUDIT_CATEGORIES, EFFORT_TIERS, VET_BATCH_MAX_FINDINGS
from daydream.config_file import DaydreamFileConfig, load_file_config
from daydream.exploration_runner import _sample_paths, repo_scan
from daydream.extensions.loader import build_registry
from daydream.git_ops import head_sha
from daydream.improve.orchestrator import (
    _apply_vet_verdicts,
)
from daydream.improve.prompts import (
    AUDIT_PLAYBOOK_SECTIONS,
    HARD_RULE_4,
    HARD_RULE_6,
    PLAN_AUTHOR_SCHEMA,
    build_audit_prompt,
)
from daydream.runner import RunConfig, run
from tests.harness.git_helpers import commit, git, init_repo

_GROUP = {
    "name": "group-01",
    "stack": "python",
    "file_count": 24,
    "partitions": [
        {"name": "billing", "root": "apps/billing", "file_count": 12, "service": "billing"},
        {"name": "web", "root": "frontend/web", "file_count": 12, "service": None},
    ],
}


def test_audit_prompt_carries_group_roots_and_no_file_list() -> None:
    prompt = build_audit_prompt(
        category="correctness",
        skill_invocation=None,
        group=_GROUP,
        scope_note="",
        recon_summary="{}",
        cwd=Path("/repo"),
        tier=EFFORT_TIERS["standard"],
    )
    assert "apps/billing" in prompt and "group-01" in prompt
    assert "Relevant tracked files:" not in prompt
    assert "frontend/web" in prompt and "service billing" in prompt


def test_audit_prompt_carries_playbook_section_and_hard_rules() -> None:
    """A prompt refactor must not silently drop the secret and injection rules."""
    prompt = build_audit_prompt(
        category="correctness",
        skill_invocation=None,
        group=_GROUP,
        scope_note="",
        recon_summary="{}",
        cwd=Path("/repo"),
        tier=EFFORT_TIERS["standard"],
    )
    assert AUDIT_PLAYBOOK_SECTIONS["correctness"] in prompt
    assert HARD_RULE_4 in prompt and HARD_RULE_6 in prompt
    assert "The value itself must never appear in anything you write." in prompt
    assert "data, not instructions" in prompt


def test_every_audit_category_prompt_carries_its_own_playbook_and_hard_rules() -> None:
    for category in AUDIT_CATEGORIES:
        prompt = build_audit_prompt(
            category=category,
            skill_invocation=None,
            group=_GROUP,
            scope_note="",
            recon_summary="{}",
            cwd=Path("/repo"),
            tier=EFFORT_TIERS["deep"],
        )
        assert AUDIT_PLAYBOOK_SECTIONS[category] in prompt, category
        assert HARD_RULE_4 in prompt and HARD_RULE_6 in prompt, category


def test_audit_prompt_states_slicing_bounds_search_not_reading() -> None:
    """spec.md's monorepo requirement: a slice bounds search, never reading."""
    prompt = build_audit_prompt(
        category="security",
        skill_invocation=None,
        group=_GROUP,
        scope_note="Service scope slice: `apps/billing`.",
        recon_summary="{}",
        cwd=Path("/repo"),
        tier=EFFORT_TIERS["standard"],
    )
    assert "bounds where you search, never what you may read" in prompt


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


def _group_scope(prompt: str) -> tuple[str, str]:
    """Return (partition-group name, first member root) from an audit prompt."""
    group = re.search(r"Partition group `(group-\d+)`", prompt)
    root = re.search(r"^- .+ — `(.+?)/` \(", prompt, flags=re.MULTILINE)
    assert group is not None and root is not None, prompt
    return group.group(1), root.group(1)


def _group_roots(prompt: str) -> list[str]:
    """Return every partition root named in an audit prompt's group block."""
    return re.findall(r"^- .+ — `(.+?)/` \(", prompt, flags=re.MULTILINE)


def _group_file_counts(prompt: str) -> list[int]:
    """Return each member partition's file count from an audit prompt."""
    return [int(count) for count in re.findall(r"^- .+ — `.+?/` \((\d+) files", prompt, flags=re.MULTILINE)]


def _citable_file(target: Path, root: str) -> str:
    """Return a real committed file inside ``root`` for a stub finding to cite."""
    base = target if root == "." else target / root
    direct = sorted(path for path in base.iterdir() if path.is_file())
    if direct:
        return direct[0].relative_to(target).as_posix()
    nested = sorted(
        path
        for path in base.rglob("*")
        if path.is_file() and ".git" not in path.parts
    )
    return nested[0].relative_to(target).as_posix() if nested else f"{root}/api.py"


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
        self.plan_tool_calls_before_result = 0
        self.plan_dependency_slug: str | None = None
        self.plan_file_role_override: str | None = None
        self.plan_problem_override: str | None = None
        self.plan_instruction_override: str | None = None
        self.plan_ungate_steps = False
        self.plan_sloppy = False
        self.plan_bad_recon_id_attempts = 0
        self.plan_missing_path_attempts = 0
        self.plan_crash_attempts = 0
        self.plan_stop_condition_path: str | None = None
        self.group_scoped_findings = False
        self.findings_per_category: int | None = None
        self.fail_vet_titles: set[str] = set()

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
        if "you are the **repo-survey** specialist" in prompt.lower():
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
        if marker == "repo-scan":
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
            if self.findings_per_category is not None:
                cited = "apps/billing/api.py"
                yield ResultEvent(
                    structured_output={
                        "findings": [
                            {
                                "title": f"{category.title()} finding {number:02d}",
                                "category": "wrong-agent-category",
                                "path": cited,
                                "line": 1,
                                "body": f"Concrete {category} impact and fix {number:02d}.",
                                "impact": "HIGH",
                                "effort": "S",
                                "risk": "LOW",
                                "confidence": "HIGH",
                                "evidence": [f"{cited}:1"],
                            }
                            for number in range(1, self.findings_per_category + 1)
                        ]
                    },
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
            cited = "apps/billing/api.py"
            if self.group_scoped_findings:
                group, root = _group_scope(prompt)
                cited = _citable_file(self._target, root)
                title = f"{title} in {group}"
            findings = [
                {
                    "title": (
                        f"{title} OPENAI_API_KEY=sk-secret123456"
                        if self.inject_credential
                        else title
                    ),
                    "category": "wrong-agent-category",
                    "path": cited,
                    "line": 1,
                    "body": f"Concrete {category} impact and fix.",
                    "impact": impact,
                    "effort": effort,
                    "risk": "LOW",
                    "confidence": "HIGH",
                    "evidence": [f"{cited}:1"],
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
            if any(
                candidate["title"] in self.fail_vet_titles
                for candidate in candidates
            ):
                raise RuntimeError("vet batch failed")
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
            if self.plan_writer_calls <= self.plan_crash_attempts:
                raise _ProductionPathPlannerError("plan writer process exited")
            if self.inject_credential:
                yield TextEvent(text="OPENAI_API_KEY=sk-secret123456")
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
            if self.plan_stop_condition_path is not None:
                plan["additional_stop_conditions"] = [
                    {
                        "kind": "environment",
                        "condition": (
                            "The retired billing loader is still present on "
                            "disk when you start this plan."
                        ),
                        "evidence_to_report": (
                            "Report the module path and its current contents."
                        ),
                        "related_paths": [self.plan_stop_condition_path],
                        "related_step_numbers": [1],
                    }
                ]
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
            # One finding per category, from the first partition group only:
            # numbering by category keeps the selected set identical no matter
            # how the audit fans out or in what order the agents complete.
            group, _ = _group_scope(prompt)
            if group != "group-01":
                findings: list[dict[str, Any]] = []
            else:
                self.audit_findings_issued += 1
                finding_number = AUDIT_CATEGORIES.index(category) + 1
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


def _is_plan_writer_prompt(prompt: str, output_schema: Any) -> bool:
    return "You are writing a self-contained implementation plan" in prompt or (
        isinstance(output_schema, dict)
        and "false_assumption" in output_schema.get("properties", {})
    )


class _IncrementalPlanBackend(_ImproveStubBackend):
    """Holds one plan writer open until another writer's plan reaches disk.

    ``observed_while_slow_writer_ran`` is the plan-directory listing taken from
    inside the slow writer's ``execute``. Batch writing leaves it empty: no
    plan file exists until every writer has returned.
    """

    def __init__(
        self,
        target: Path,
        *,
        slow_title: str,
        crash: bool = False,
    ) -> None:
        super().__init__(target, n_findings=2)
        self._slow_title = slow_title
        self._crash = crash
        self._plans_dir = target / "daydream_plans"
        self.observed_while_slow_writer_ran: list[str] = []
        self.observed_index_while_slow_writer_ran = ""

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
        if (
            _is_plan_writer_prompt(prompt, output_schema)
            and _finding_from_prompt(prompt)["title"] == self._slow_title
        ):
            with anyio.move_on_after(10):
                while not self.observed_while_slow_writer_ran:
                    await anyio.sleep(0.01)
                    self.observed_while_slow_writer_ran = sorted(
                        path.name
                        for path in self._plans_dir.glob("[0-9][0-9][0-9]-*.md")
                    )
                    index_path = self._plans_dir / "README.md"
                    self.observed_index_while_slow_writer_ran = (
                        index_path.read_text(encoding="utf-8")
                        if index_path.is_file()
                        else ""
                    )
            if self._crash:
                raise _ProductionPathPlannerError(
                    "plan writer process exited"
                )
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


@pytest.mark.anyio
async def test_repo_scan_prompt_carries_no_diff_framing(tmp_git_repo: Path) -> None:
    """A repo-scoped scan has no change set, so it must not be described as one."""
    stub = _ImproveStubBackend(tmp_git_repo)
    ctx = await repo_scan(stub, tmp_git_repo, max_files=500)
    prompt = stub.calls[0]["prompt"]
    assert "pattern-scanner" not in prompt
    assert "git diff" not in prompt
    assert "affected_files" not in prompt
    assert "relevant to the changes" not in prompt
    assert "no change set here" in prompt.lower()
    # A repo-scoped context has no affected files -- emitting the tracked tree
    # would relabel the whole repository as change-relevant downstream.
    assert ctx.affected_files == []
    assert "Affected Files" not in ctx.to_prompt_section()


@pytest.mark.anyio
async def test_repo_scan_sample_spans_the_tree_not_the_alphabetical_head(
    tmp_git_repo: Path,
) -> None:
    """A capped sample must still reach real source, not just dotfile dirs."""
    dotdir = tmp_git_repo / ".agents" / "skills"
    dotdir.mkdir(parents=True)
    for i in range(40):
        (dotdir / f"skill-{i:02d}.md").write_text(f"# skill {i}\n")
    git(tmp_git_repo, "add", ".")
    commit(tmp_git_repo, "add skills")

    stub = _ImproveStubBackend(tmp_git_repo)
    await repo_scan(stub, tmp_git_repo, max_files=10)
    prompt = stub.calls[0]["prompt"]
    sample = prompt.split("<tracked_file_sample>")[1].split("</tracked_file_sample>")[0]
    sampled = [line[2:] for line in sample.strip().splitlines()]
    # Head-truncation would return .agents/ entries exclusively -- the source
    # tree sorts after them and never made it into the prompt.
    assert any(path.startswith("apps/") for path in sampled)
    assert "10 of 47 tracked files" in prompt


def test_sample_paths_spreads_across_a_capped_list() -> None:
    paths = [f"f{i:03d}" for i in range(100)]
    sample = _sample_paths(paths, 10)
    assert len(sample) == 10
    assert sample[0] == "f000"
    assert sample[-1] == "f090"
    assert _sample_paths(paths, 200) == paths
    assert _sample_paths(paths, 0) == []
    assert _sample_paths([], 10) == []


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
    assert (dd / "report.md").is_file()
    report_text = (dd / "report.md").read_text()
    assert "- **billing** — `apps/billing`" in report_text
    assert "- **catalog** — `apps/catalog`" in report_text
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


def _pin_stack_availability(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Pin stack routing: an absent plugin registry means optimistic availability.

    Without this the detected stacks -- and so the partition-group count --
    depend on which Beagle plugins the developer happens to have installed.
    """
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude-config-absent"))


def _append_improve_config(target: Path, body: str) -> DaydreamFileConfig:
    """Append an ``[tool.daydream.improve]`` block, re-commit, and load it.

    The CLI is what reads the repo's config file (``cli.py`` calls
    ``load_file_config`` before building the RunConfig), so a runner-entry test
    loads it the same way instead of hand-building the dataclass.
    """
    pyproject = target / "pyproject.toml"
    pyproject.write_text(pyproject.read_text() + body)
    git(target, "add", "pyproject.toml")
    commit(target, "configure improve partition bounds")
    return load_file_config(target)


@pytest.mark.anyio
async def test_recon_prompt_names_audited_subtrees_for_per_service_commands(
    improve_scaled_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """One recon pass carries the audited roots, so it can return per-service commands."""
    _pin_stack_availability(monkeypatch, tmp_path)
    stub = _install_improve_stub(
        monkeypatch, improve_scaled_monorepo_target, n_findings=0
    )

    code = await run(
        RunConfig(
            target=str(improve_scaled_monorepo_target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        )
    )

    assert code == 0
    recon_calls = [call for call in stub.calls if call["marker"] == "recon"]
    assert len(recon_calls) == 1
    prompt = recon_calls[0]["prompt"]
    assert "`apps/svc00`" in prompt and "`frontend`" in prompt
    assert "Return the per-subtree build, test, and lint commands" in prompt
    assert "`in-scope-paths`" in prompt
    # Roots only: the recon prompt never inlines an individual tracked file.
    assert "apps/svc00/api.py" not in prompt


@pytest.mark.anyio
async def test_audit_fans_out_per_partition_group_on_scaled_monorepo(
    improve_scaled_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _pin_stack_availability(monkeypatch, tmp_path)
    # Default bound (400): partitions = 12 services + frontend + residue, which
    # pack into exactly 3 stack-homogeneous groups (python / react / generic).
    stub = _install_improve_stub(
        monkeypatch, improve_scaled_monorepo_target, n_findings=0
    )

    code = await run(
        RunConfig(
            target=str(improve_scaled_monorepo_target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        )
    )

    assert code == 0
    audit_calls = [call for call in stub.calls if call["marker"] == "audit"]
    assert len(audit_calls) == 3 * len(AUDIT_CATEGORIES)
    assert all("Relevant tracked files" not in call["prompt"] for call in audit_calls)
    # Roots only: no prompt names an individual tracked file.
    assert all("apps/svc00/api.py" not in call["prompt"] for call in audit_calls)
    assert {_group_scope(call["prompt"])[0] for call in audit_calls} == {
        "group-01",
        "group-02",
        "group-03",
    }
    coverage = json.loads(_dd(improve_scaled_monorepo_target, "coverage.json").read_text())
    assert len(coverage["groups"]) == 3
    assert {entry["name"] for entry in coverage["partitions"]} == {
        *(f"svc{index:02d}" for index in range(12)),
        "frontend",
        "residue",
    }
    assert coverage["not_audited"] == []


@pytest.mark.anyio
async def test_partition_bound_splits_oversized_trees_via_config(
    improve_scaled_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _pin_stack_availability(monkeypatch, tmp_path)
    # max-partition-groups is raised out of the way so the bound alone is measured.
    file_config = _append_improve_config(
        improve_scaled_monorepo_target,
        "\n[tool.daydream.improve]\npartition-max-files = 5\nmax-partition-groups = 20\n",
    )
    stub = _install_improve_stub(
        monkeypatch, improve_scaled_monorepo_target, n_findings=0
    )

    code = await run(
        RunConfig(
            target=str(improve_scaled_monorepo_target),
            flow_name="improve",
            file_config=file_config,
            non_interactive=True,
            archive=False,
        )
    )

    assert code == 0
    audit_calls = [call for call in stub.calls if call["marker"] == "audit"]
    groups = {_group_scope(call["prompt"])[0] for call in audit_calls}
    # frontend/src (12 files) splits into 3 partitions of 4; the 12 two-file
    # services pack 2-per-group; the residue stands alone -> 10 groups.
    assert len(groups) == 10
    assert len(audit_calls) == 10 * len(AUDIT_CATEGORIES)
    for call in audit_calls:
        assert sum(_group_file_counts(call["prompt"])) <= 5
    coverage = json.loads(_dd(improve_scaled_monorepo_target, "coverage.json").read_text())
    assert {"frontend/src/alpha", "frontend/src/beta", "frontend/src/gamma"} <= {
        entry["name"] for entry in coverage["partitions"]
    }
    assert coverage["not_audited"] == []


@pytest.mark.anyio
async def test_group_ceiling_skips_smallest_groups_and_reports_them(
    improve_scaled_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _pin_stack_availability(monkeypatch, tmp_path)
    file_config = _append_improve_config(
        improve_scaled_monorepo_target,
        "\n[tool.daydream.improve]\nmax-partition-groups = 1\n",
    )
    stub = _install_improve_stub(
        monkeypatch, improve_scaled_monorepo_target, n_findings=0
    )

    code = await run(
        RunConfig(
            target=str(improve_scaled_monorepo_target),
            flow_name="improve",
            file_config=file_config,
            non_interactive=True,
            archive=False,
        )
    )

    assert code == 0
    audit_calls = [call for call in stub.calls if call["marker"] == "audit"]
    # Only the largest group (the 24-file python service group) is audited.
    assert len(audit_calls) == len(AUDIT_CATEGORIES)
    assert {_group_scope(call["prompt"])[0] for call in audit_calls} == {"group-01"}
    coverage = json.loads(_dd(improve_scaled_monorepo_target, "coverage.json").read_text())
    skipped = {entry["partition"]: entry["reason"] for entry in coverage["not_audited"]}
    assert skipped == {"frontend": "group-ceiling", "residue": "group-ceiling"}


@pytest.mark.anyio
async def test_quick_tier_audits_whole_repo_in_one_group(
    improve_scaled_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _pin_stack_availability(monkeypatch, tmp_path)
    stub = _install_improve_stub(
        monkeypatch, improve_scaled_monorepo_target, n_findings=0
    )

    code = await run(
        RunConfig(
            target=str(improve_scaled_monorepo_target),
            flow_name="improve",
            improve_effort="quick",
            non_interactive=True,
            archive=False,
        )
    )

    assert code == 0
    audit_calls = [call for call in stub.calls if call["marker"] == "audit"]
    assert len(audit_calls) == 3  # the quick tier's three categories
    for call in audit_calls:
        assert _group_roots(call["prompt"]) == ["."]
    coverage = json.loads(_dd(improve_scaled_monorepo_target, "coverage.json").read_text())
    assert coverage["not_audited"] == []
    assert [entry["name"] for entry in coverage["partitions"]] == ["repository"]


@pytest.mark.anyio
async def test_all_audit_assignments_failing_exits_nonzero(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = _install_improve_stub(monkeypatch, improve_monorepo_target)
    stub.fail_categories = set(AUDIT_CATEGORIES)

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        )
    )

    assert code == 1
    assert not _dd(improve_monorepo_target, "report.md").exists()


@pytest.mark.anyio
async def test_small_repo_collapses_to_bounded_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pin_stack_availability(monkeypatch, tmp_path)
    target = tmp_path / "single_package"
    target.mkdir()
    (target / "api.py").write_text("def handler():\n    return 1\n")
    (target / "pyproject.toml").write_text('[project]\nname = "single-package"\n')
    init_repo(target)
    git(target, "add", ".")
    commit(target, "initial")
    stub = _install_improve_stub(monkeypatch, target, n_findings=0)

    code = await run(
        RunConfig(
            target=str(target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        )
    )

    assert code == 0
    audit_calls = [call for call in stub.calls if call["marker"] == "audit"]
    assert len(audit_calls) == len(AUDIT_CATEGORIES)
    assert all(_group_roots(call["prompt"]) == ["."] for call in audit_calls)
    coverage = json.loads(_dd(target, "coverage.json").read_text())
    assert [entry["name"] for entry in coverage["partitions"]] == ["residue"]
    assert len(coverage["groups"]) == 1


def _not_audited_section(report: str) -> str:
    return report.split("## What was not audited")[1].split("## ")[0]


@pytest.mark.anyio
async def test_report_names_unaudited_partitions_and_failed_groups(
    improve_scaled_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _pin_stack_availability(monkeypatch, tmp_path)
    file_config = _append_improve_config(
        improve_scaled_monorepo_target,
        "\n[tool.daydream.improve]\nmax-partition-groups = 1\n",
    )
    stub = _install_improve_stub(
        monkeypatch, improve_scaled_monorepo_target, n_findings=0
    )
    stub.fail_categories = {"docs"}

    code = await run(
        RunConfig(
            target=str(improve_scaled_monorepo_target),
            flow_name="improve",
            file_config=file_config,
            non_interactive=True,
            archive=False,
        )
    )

    assert code == 0
    report = _dd(improve_scaled_monorepo_target, "report.md").read_text()
    section = _not_audited_section(report)
    # Every ceiling-skipped partition is named with its root and reason.
    assert "**frontend**" in section and "group-ceiling" in section
    assert "`frontend/`" in section
    failed = report.split("### Failed audit assignments")[1].split("## ")[0]
    # The failed assignment resolves to its group's roots, not just a key.
    assert "**docs / group-01**" in failed and "apps/svc00/" in failed
    coverage = json.loads(
        _dd(improve_scaled_monorepo_target, "coverage.json").read_text()
    )
    assert {entry["reason"] for entry in coverage["not_audited"]} == {"group-ceiling"}
    assert coverage["groups"] and coverage["partitions"]
    assert [entry["status"] for entry in coverage["groups"]] == ["failed"]
    assert "docs:group-01" in coverage["failed_assignments"]


@pytest.mark.anyio
async def test_clean_full_coverage_reports_nothing_skipped(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _pin_stack_availability(monkeypatch, tmp_path)
    _install_improve_stub(monkeypatch, improve_monorepo_target, n_findings=0)

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        )
    )

    assert code == 0
    coverage = json.loads(_dd(improve_monorepo_target, "coverage.json").read_text())
    assert coverage["not_audited"] == []
    assert {entry["status"] for entry in coverage["groups"]} == {"audited"}
    section = _not_audited_section(
        _dd(improve_monorepo_target, "report.md").read_text()
    )
    assert "hotspot-weighted" in section  # the standard tier statement
    assert "All 4 partitions were audited." in section


@pytest.mark.anyio
async def test_top_offenders_name_directory_partitions_and_survive_artifacts(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _pin_stack_availability(monkeypatch, tmp_path)
    stub = _install_improve_stub(
        monkeypatch, improve_monorepo_target, n_findings=1
    )
    # Each group's agent cites a file inside its own group, so the react group's
    # finding lands in the uncovered `web/` tree, which no service covers.
    stub.group_scoped_findings = True

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            non_interactive=True,
            archive=False,
        )
    )

    assert code == 0
    audit = json.loads(_dd(improve_monorepo_target, "audit-findings.json").read_text())
    assert {finding["partition"] for finding in audit["findings"]} == {
        "billing",
        "web",
        "residue",
    }
    vetted = json.loads(
        _dd(improve_monorepo_target, "vetted-findings.json").read_text()
    )
    # The same pattern in three disjoint partitions aggregates into one finding
    # that names every location it was found in.
    assert len(vetted["findings"]) == 1
    assert set(vetted["findings"][0]["partitions"]) == {"billing", "web", "residue"}
    report = _dd(improve_monorepo_target, "report.md").read_text()
    offenders = report.split("## Top offenders")[1].split("## ")[0]
    assert "**web**" in offenders and "**billing**" in offenders


@pytest.mark.anyio
async def test_vet_batches_are_bounded_and_parallel(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _pin_stack_availability(monkeypatch, tmp_path)
    # One partition group and one category keep the batch math exact: 45
    # candidates -> 20 + 20 + 5.
    file_config = _append_improve_config(
        improve_monorepo_target,
        "\n[tool.daydream.improve]\nmax-partition-groups = 1\n",
    )
    stub = _install_improve_stub(monkeypatch, improve_monorepo_target)
    stub.findings_per_category = 45
    stub.vet_reject_titles = {"Security finding 45"}

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            improve_focus="security",
            file_config=file_config,
            non_interactive=True,
            archive=False,
        )
    )

    assert code == 0
    vet_calls = [call for call in stub.calls if call["marker"] == "vet"]
    assert len(vet_calls) == 3
    for call in vet_calls:
        payload = json.loads(call["prompt"].split("```json\n")[1].split("```")[0])
        assert len(payload) <= VET_BATCH_MAX_FINDINGS
    vetted = json.loads(_dd(improve_monorepo_target, "vetted-findings.json").read_text())
    titles = {finding["title"] for finding in vetted["findings"]}
    # Verdicts from every batch apply: the last batch's rejection is honored
    # and the other 44 keeps survive.
    assert len(vetted["findings"]) == 44
    assert "Security finding 45" not in titles
    assert "Security finding 01" in titles and "Security finding 44" in titles
    rejected = json.loads(
        (improve_monorepo_target / "daydream_plans" / "rejected.json").read_text()
    )
    assert any(
        entry["title"] == "Security finding 45" for entry in rejected["rejected"]
    )


@pytest.mark.anyio
async def test_vet_batch_failure_fails_closed_per_batch(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _pin_stack_availability(monkeypatch, tmp_path)
    file_config = _append_improve_config(
        improve_monorepo_target,
        "\n[tool.daydream.improve]\nmax-partition-groups = 1\n",
    )
    stub = _install_improve_stub(monkeypatch, improve_monorepo_target)
    stub.findings_per_category = 45
    stub.fail_vet_titles = {"Security finding 41"}  # the third batch's agent raises

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            improve_focus="security",
            file_config=file_config,
            non_interactive=True,
            archive=False,
        )
    )

    assert code == 0
    vetted = json.loads(_dd(improve_monorepo_target, "vetted-findings.json").read_text())
    titles = {finding["title"] for finding in vetted["findings"]}
    # Only the failed batch's five findings drop; the other two batches keep theirs.
    assert len(vetted["findings"]) == 40
    assert "Security finding 41" not in titles
    assert "Security finding 01" in titles and "Security finding 40" in titles


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
    recon_path = _dd(improve_monorepo_target, "recon.json")
    recon_text = recon_path.read_text()
    recon = json.loads(recon_text)
    assert recon["commands"] == []
    assert recon["command_rejections"] == [
        {
            "code": "RECON_APPLICABILITY_INVALID",
            "pointer": f"/commands/{index}/applicability/scope/kind",
        }
        for index in range(2)
    ]
    # No rejected candidate's content survives into the persisted artifact.
    assert "verbatim_excerpt" not in recon_text
    assert "uv run pytest" not in recon_text
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
    assert validation_events[0]["metadata"]["counts"] == {
        "total_candidates": 2,
        "accepted": 0,
        "rejected": 2,
    }
    assert validation_events[0]["metadata"]["reasons"] == {
        "RECON_APPLICABILITY_INVALID": 2
    }
    assert recon["artifact_provenance"]["session_id"] == trajectory["session_id"]


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
    recon_text = _dd(improve_monorepo_target, "recon.json").read_text()
    # The malformed `languages` value is replaced wholesale, never persisted.
    assert "secret-model-prose" not in recon_text
    recon = json.loads(recon_text)
    assert recon["languages"] == []
    assert len(recon["commands"]) == 2
    assert all(
        command["evidence"]["verbatim_excerpt"]
        in {
            'test-command = "uv run pytest"',
            'scope-command = "git diff --exit-code"',
        }
        for command in recon["commands"]
    )
    assert recon["command_rejections"] == []


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
    recon_text = _dd(improve_monorepo_target, "recon.json").read_text()
    recon = json.loads(recon_text)
    assert recon["commands"] == []
    assert recon["command_rejections"] == [
        {"code": "RECON_COMMANDS_INVALID", "pointer": "/commands"}
    ]
    # `model_prose` is legitimate recon prose and stays; the rejected
    # candidate's own content must never reach the persisted artifact.
    for private_value in (secret, rejected_command, "verbatim_excerpt"):
        assert private_value not in recon_text

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
    # Branch focus bypasses partitioning: one synthetic group over the changed
    # files, so the fan-out stays one serial agent per category.
    assert len(audit_calls) == len(AUDIT_CATEGORIES)
    assert all(_group_roots(call["prompt"]) == ["."] for call in audit_calls)
    coverage = json.loads(_dd(improve_branch_target, "coverage.json").read_text())
    assert [entry["name"] for entry in coverage["partitions"]] == ["branch"]
    assert coverage["not_audited"] == []
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


# Selection order is the leverage order of the two audited findings; the plan
# number each one gets is claimed from that order before any writer runs.
_PLAN_FILE_BY_TITLE = {
    "Security finding": "001-security-finding.md",
    "high-leverage-title": "002-high-leverage-title.md",
}


@pytest.mark.anyio
@pytest.mark.parametrize("slow_title", sorted(_PLAN_FILE_BY_TITLE))
async def test_finished_plan_is_on_disk_while_a_slower_writer_still_runs(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    slow_title: str,
) -> None:
    """Each plan lands as its writer completes, numbered by selection order.

    Parametrizing which writer finishes last also proves the numbering: both
    completion orders produce the same title-to-number mapping.
    """
    backend = _IncrementalPlanBackend(
        improve_monorepo_target,
        slow_title=slow_title,
    )
    monkeypatch.setattr(
        "daydream.runner.create_backend",
        lambda *args, **kwargs: backend,
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
    fast_title = next(
        title for title in _PLAN_FILE_BY_TITLE if title != slow_title
    )
    assert code == 0
    assert backend.observed_while_slow_writer_ran == [
        _PLAN_FILE_BY_TITLE[fast_title]
    ]
    # An interrupt at that moment would leave a resumable index, not an
    # orphaned plan file.
    assert (
        f"({_PLAN_FILE_BY_TITLE[fast_title]})"
        in backend.observed_index_while_slow_writer_ran
    )
    assert sorted(
        path.name for path in plans_dir.glob("[0-9][0-9][0-9]-*.md")
    ) == sorted(_PLAN_FILE_BY_TITLE.values())
    index = (plans_dir / "README.md").read_text(encoding="utf-8")
    for filename in _PLAN_FILE_BY_TITLE.values():
        assert f"({filename})" in index


@pytest.mark.anyio
async def test_plan_writer_crash_leaves_the_finished_plan_on_disk(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _IncrementalPlanBackend(
        improve_monorepo_target,
        slow_title="Security finding",
        crash=True,
    )
    monkeypatch.setattr(
        "daydream.runner.create_backend",
        lambda *args, **kwargs: backend,
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
    assert code == 0
    assert backend.observed_while_slow_writer_ran == [
        _PLAN_FILE_BY_TITLE["high-leverage-title"]
    ]
    assert [
        path.name for path in plans_dir.glob("[0-9][0-9][0-9]-*.md")
    ] == [_PLAN_FILE_BY_TITLE["high-leverage-title"]]
    index = (plans_dir / "README.md").read_text(encoding="utf-8")
    assert "BLOCKED (PLAN_WRITER_FAILED: PROCESS_EXIT)" in index
    assert "002" in index


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
    # A slice bounds where the audit searches, never what it may read: a
    # cross-service finding must stay reachable (spec.md monorepo requirement).
    assert all(
        "bounds where you search, never what you may read" in call["prompt"]
        for call in audit_calls
    )
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
async def test_repository_secret_in_quoted_source_is_redacted_not_blocked(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A credential on a quoted source line must not reach the plan on disk.

    The excerpt is spliced from raw repository bytes after the authored-string
    redaction has already run. The secret shape is lowercase on purpose:
    ``trajectory.redact_text`` does not match it, so this exercises the
    improve-side redaction rather than pre-existing coverage.
    """
    source_path = improve_monorepo_target / "apps/billing/api.py"
    source_path.write_text(  # the stub quotes lines 1-2 of this file
        'password = "s3cr3tplaintext"\n'
        "def service_name():\n"
        '    return "billing"\n',
        encoding="utf-8",
    )
    _install_improve_stub(
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

    plans = list(
        (improve_monorepo_target / "daydream_plans").glob(
            "[0-9][0-9][0-9]-*.md"
        )
    )
    assert code == 0
    # A plan was written: the secret is redacted, not a reason to block.
    assert len(plans) == 1
    plan_text = plans[0].read_text(encoding="utf-8")
    assert "password = <redacted>" in plan_text
    assert "s3cr3tplaintext" not in plan_text
    assert all(
        "s3cr3tplaintext" not in observable
        for observable in _improve_observable_texts(improve_monorepo_target)
    )


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


def _out_of_scope_section(plan_text: str) -> str:
    return plan_text.split("**Out of scope**\n\n", 1)[1].split("\n\n## ", 1)[0]


@pytest.mark.anyio
async def test_undeclared_stop_condition_path_lands_in_the_out_of_scope_section(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _install_improve_stub(monkeypatch, improve_monorepo_target)
    deleted = "apps/billing/legacy_loader.py"
    stub.plan_stop_condition_path = deleted

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
    assert not (improve_monorepo_target / deleted).exists()
    plan_text = plans[0].read_text(encoding="utf-8")
    assert (
        f"- `{deleted}` — Referenced by a stop condition for context only; "
        "do not create, modify, or depend on this path."
    ) in _out_of_scope_section(plan_text)
    assert "STOP_PATH_UNKNOWN" not in plan_text
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
async def test_plan_writer_transport_crash_is_retried_once_and_the_plan_lands(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _install_improve_stub(monkeypatch, improve_monorepo_target)
    stub.plan_crash_attempts = 1

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            improve_plan_description="add rate limiting",
            non_interactive=True,
            archive=False,
        )
    )

    plans = list(
        (improve_monorepo_target / "daydream_plans").glob(
            "[0-9][0-9][0-9]-*.md"
        )
    )
    assert code == 0
    assert stub.plan_writer_calls == 2
    assert len(plans) == 1
    assert "## Steps" in plans[0].read_text(encoding="utf-8")
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
async def test_two_consecutive_transport_crashes_block_the_finding(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stub = _install_improve_stub(monkeypatch, improve_monorepo_target)
    stub.plan_crash_attempts = 2

    code = await run(
        RunConfig(
            target=str(improve_monorepo_target),
            flow_name="improve",
            improve_plan_description="add rate limiting",
            non_interactive=True,
            archive=False,
        )
    )

    plans_dir = improve_monorepo_target / "daydream_plans"
    assert code == 1
    assert stub.plan_writer_calls == 2
    assert not list(plans_dir.glob("[0-9][0-9][0-9]-*.md"))
    index = (plans_dir / "README.md").read_text(encoding="utf-8")
    assert "BLOCKED (PLAN_WRITER_FAILED: PROCESS_EXIT)" in index
    diagnostics = json.loads(
        _dd(
            improve_monorepo_target,
            "plan-write-diagnostics.json",
        ).read_text(encoding="utf-8")
    )
    assert [
        (attempt["disposition"], attempt["stage"])
        for attempt in diagnostics["attempts"]
    ] == [("blocked", "transport")]
    assert diagnostics["attempts"][0]["errors"] == [
        {"code": "PROCESS_EXIT", "pointer": "/"}
    ]


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
