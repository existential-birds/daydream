"""Shared improve-flow fake ``Backend`` family and plan/recon payload builders.

Extracted from ``tests/test_improve_flow.py`` so a second test module stops
reaching into a first test module's underscore-prefixed names — the exact
condition this harness package exists to end (see
``tests/harness/phase_backend.py``). ``tests/test_extension_prompts_integration.py``
imported ``_dd`` and ``_ImproveStubBackend`` straight out of the flow test file;
both now live here under public names.

:class:`ImproveStubBackend` is a *dispatch* fake: it classifies each improve
turn off prompt substrings (repo-survey / ``IMPROVE_RECON`` / audit-specialist /
vet / plan-writer), records every call, and serves a synthetic response per
marker. Its many public attributes are per-test switches — set them on the
instance after construction to shape one turn's payload (a rejected vet
verdict, a schema-invalid enum, a crashing plan writer, an injected credential).

Three subclasses bend one axis each and are otherwise the same fake:

* :class:`ProductionPathBackend` — Pi-shaped delays, a 42-command recon menu and
  a provider concurrency ceiling, for the full production-path regressions.
* :class:`IncrementalPlanBackend` — holds one plan writer open and observes the
  plan directory from inside it (proves plans land as each writer finishes).
* :class:`OutOfOrderPlanBackend` — makes the first-selected writer finish last.

This is a verbatim move: behaviour is unchanged from the pre-extraction file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import anyio
import pytest

from daydream.backends import ResultEvent, TextEvent, ToolResultEvent, ToolStartEvent
from daydream.backends._subprocess import StreamStalledError
from daydream.config import AUDIT_CATEGORIES


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
        "title": title,
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
        "context_excerpts": [
            {
                "path": "apps/billing/api.py",
                "start_line": 1,
                "end_line": 2,
                "file_role": (
                    f"Quote the billing behavior being changed{requested_context}."
                ),
            }
        ],
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
        "additional_command_refs": [],
    }


_MENU_ID = re.compile(r"^- id `([a-z0-9-]+)`", re.MULTILINE)


def _menu_ids(prompt: str) -> list[str]:
    """Return the recon command ids offered to the plan writer, in order."""
    return _MENU_ID.findall(prompt)


def _gate_plan_on(plan: dict[str, Any], command_id: str) -> dict[str, Any]:
    """Point every verification slot at one recon id, run verbatim."""
    plan["additional_command_refs"] = []
    for step in plan["steps"]:
        step["verification"] = _plan_ref(command_id)
    for case in plan["test_plan"]["cases"]:
        case["verification"] = _plan_ref(command_id)
    for criterion in plan["done_criteria"]:
        criterion["verification"] = _plan_ref(command_id)
    return plan


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


def group_scope(prompt: str) -> tuple[str, str]:
    """Return (partition-group name, first member root) from an audit prompt."""
    group = re.search(r"Partition group `(group-\d+)`", prompt)
    root = re.search(r"^- .+ — `(.+?)/` \(", prompt, flags=re.MULTILINE)
    assert group is not None and root is not None, prompt
    return group.group(1), root.group(1)


def group_roots(prompt: str) -> list[str]:
    """Return every partition root named in an audit prompt's group block."""
    return re.findall(r"^- .+ — `(.+?)/` \(", prompt, flags=re.MULTILINE)


def group_file_counts(prompt: str) -> list[int]:
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


class ImproveStubBackend:
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
        # Instant retries in-test: run_agent reads these off the backend, so a
        # retryable failure (a stall, a rate-limit) re-arms with no backoff sleep.
        self.retry_attempts = 20
        self.retry_base_delay_s = 0.0
        self.retry_max_delay_s = 0.0
        self.audit_delay = audit_delay
        self.audit_active = 0
        self.audit_peak = 0
        self.calls: list[dict[str, Any]] = []
        self.fail_categories: set[str] = set()
        self.vet_reject_titles: set[str] = set()
        self.return_legacy_plan = False
        self.return_secret_invalid_enum = False
        self.return_secret_invalid_enum_once = False
        self.plan_writer_calls = 0
        self.inject_credential = False
        self.all_recon_commands_invalid = False
        self.recon_commands_override: list[dict[str, Any]] | None = None
        self.recon_commands_extra: list[dict[str, Any]] = []
        self.recon_languages_override: Any = None
        self.recon_output_override: Any = None
        self.plan_tool_calls_before_result = 0
        self.plan_file_role_override: str | None = None
        self.plan_problem_override: str | None = None
        self.plan_instruction_override: str | None = None
        self.plan_ungate_steps = False
        self.plan_gate_on_first_menu_id = False
        self.plan_sloppy = False
        self.plan_bad_recon_id_attempts = 0
        self.plan_missing_path_attempts = 0
        self.plan_unquoted_path_attempts = 0
        self.plan_crash_attempts = 0
        self.plan_stall_attempts = 0
        self.plan_rate_limit_always = False
        self.plan_stop_condition_path: str | None = None
        self.plan_no_test_exemplars = False
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
                group, root = group_scope(prompt)
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
            if self.plan_rate_limit_always:
                raise _ProductionPathRateLimitError("provider rate limit")
            if self.plan_writer_calls <= self.plan_crash_attempts:
                raise _ProductionPathPlannerError("plan writer process exited")
            if self.plan_writer_calls <= self.plan_stall_attempts:
                raise StreamStalledError(cli="pi", timeout_s=1.0)
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
            if self.plan_gate_on_first_menu_id:
                # A writer can only gate on what the menu actually offers.
                offered = _menu_ids(prompt)
                plan = _gate_plan_on(plan, offered[0]) if offered else plan
            if self.plan_file_role_override is not None:
                plan["scope"]["existing_paths"][0]["role"] = (
                    self.plan_file_role_override
                )
                plan["context_excerpts"][0]["file_role"] = (
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
            if self.plan_no_test_exemplars:
                # The only honest answer in a repository with no test files.
                plan["test_plan"]["exemplars"] = []
            if self.plan_stop_condition_path is not None:
                plan["false_assumption"]["related_paths"].append(
                    self.plan_stop_condition_path
                )
            if self.plan_writer_calls <= self.plan_bad_recon_id_attempts:
                plan["steps"][0]["verification"] = _plan_ref("make-tests")
            if self.plan_writer_calls <= self.plan_unquoted_path_attempts:
                plan["context_excerpts"] = []
            if self.plan_writer_calls <= self.plan_missing_path_attempts:
                plan["scope"]["existing_paths"].append(
                    {
                        "path": "apps/billing/legacy_api.py",
                        "role": "Reference the retired billing module for parity.",
                    }
                )
            if self.all_recon_commands_invalid:
                plan = _without_verification_commands(plan)
            if self.return_secret_invalid_enum or (
                self.return_secret_invalid_enum_once
                and self.plan_writer_calls == 1
            ):
                # A schema-invalid enum carrying a credential: the value must
                # fail validation AND never surface in any host observable.
                plan["steps"][0]["changes"][0]["operation"] = (
                    "TOKEN=PRIVATE_SCHEMA_SECRET"
                )
                plan["title"] = "PRIVATE_SCHEMA_SECRET rejected title"
            if self.inject_credential:
                plan["steps"][0]["changes"][0]["operation"] = (
                    "OPENAI_API_KEY=sk-secret123456"
                )
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


class ProductionPathBackend(ImproveStubBackend):
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
            group, _ = group_scope(prompt)
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


class IncrementalPlanBackend(ImproveStubBackend):
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


class OutOfOrderPlanBackend(ImproveStubBackend):
    """Make the first-selected plan writer be the last one to return.

    The hold is on the other writers' completion count, not on a sleep, so the
    completion sequence is deterministic: every writer but the first-selected
    one returns immediately, and the first-selected one only returns once the
    rest are done. ``completion_order`` is the selection rank of each writer in
    the order they actually finished.
    """

    def __init__(self, target: Path, *, n_findings: int) -> None:
        super().__init__(target, n_findings=n_findings)
        self._selected_path = (
            target / ".daydream" / "improve" / "selected.json"
        )
        self.completion_order: list[int] = []

    def _selection(self) -> list[str]:
        return list(
            json.loads(self._selected_path.read_text(encoding="utf-8"))[
                "selected"
            ]
        )

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
        rank: int | None = None
        if _is_plan_writer_prompt(prompt, output_schema):
            selection = self._selection()
            rank = selection.index(_finding_from_prompt(prompt)["fingerprint"])
            if rank == 0:
                with anyio.move_on_after(10):
                    while len(self.completion_order) < len(selection) - 1:
                        await anyio.sleep(0.01)
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
        if rank is not None:
            self.completion_order.append(rank)


def install_improve_stub(
    monkeypatch: pytest.MonkeyPatch,
    target: Path,
    *,
    n_findings: int | None = None,
    attempt_write: bool = False,
    fanout_concurrency: int = 4,
    audit_delay: float = 0,
) -> ImproveStubBackend:
    stub = ImproveStubBackend(
        target,
        n_findings=n_findings,
        attempt_write=attempt_write,
        fanout_concurrency=fanout_concurrency,
        audit_delay=audit_delay,
    )
    monkeypatch.setattr("daydream.runner.create_backend", lambda *args, **kwargs: stub)
    return stub


def install_per_phase_improve_stubs(
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
    ) -> ImproveStubBackend:
        stub = ImproveStubBackend(target)
        stub.calls = shared_calls
        stub.model = model or "mock-model"
        stub.reasoning_effort = reasoning_effort
        return stub

    monkeypatch.setattr("daydream.runner.create_backend", _factory)
    return shared_calls


def improve_artifact(repo: Path, name: str) -> Path:
    """Path to a host-written improve artifact under ``.daydream/improve/``."""
    return repo / ".daydream" / "improve" / name
