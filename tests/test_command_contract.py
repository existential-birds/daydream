"""Lookaround-free command-contract schema patterns (issue #304).

OpenAI rejects regex lookaround in JSON Schema ``pattern`` values with
``invalid_json_schema``. The exported path patterns must contain none, while
preserving the accept/reject grammar.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from daydream.improve.command_contract import (
    DIRECTORY_SCOPE_SCHEMA,
    REPOSITORY_FILE_PATH_SCHEMA,
    WORKING_DIRECTORY_SCHEMA,
    canonicalize_directory_scope,
    path_is_confined,
    valid_directory_scope_lexical,
    valid_repository_file_path,
    validate_applicability,
)
from daydream.repository_paths import canonicalize_working_directory
from daydream.runner import RunConfig, run
from tests.harness.improve_backend import improve_artifact, install_improve_stub

MakeConfig = Callable[..., RunConfig]

ALL_SCHEMA_PATTERNS = [
    REPOSITORY_FILE_PATH_SCHEMA["pattern"],
    DIRECTORY_SCOPE_SCHEMA["pattern"],
    WORKING_DIRECTORY_SCHEMA["pattern"],
]


@pytest.mark.parametrize("pattern", ALL_SCHEMA_PATTERNS)
def test_schema_patterns_are_lookaround_free(pattern: str) -> None:
    """Regression pin: no lookaround may reach an exported schema pattern.

    OpenAI rejects ``(?=``, ``(?!``, ``(?<=``, ``(?<!`` with
    ``invalid_json_schema``. Plain ``(?:`` groups are valid and expected.
    """
    assert re.search(r"\(\?[=!<]", pattern) is None


PATH_ACCEPTS = [
    "src/main.rs",
    "a/b/c",
    ".github/workflows/ci.yml",
    "foo.bar",
    "foo..bar",
    # issues #572/#573: legal filenames the over-tight grammar rejected
    "foo bar.py",
    "Café.md",
    "file#1.py",
    "a%file.txt",
    "(x).py",
    "a&b.py",
    "~/.bashrc",
    "./foo.py",
    "space name.py",
]

LEGAL_FILENAMES = [
    "foo bar.py",
    "Café.md",
    "file#1.py",
    "a%file.txt",
    "(x).py",
    "a&b.py",
    "~/.bashrc",
    "space name.py",
]

MULTI_DOT_PATHS = [
    "..cache",
    "...",
    "src/...generated",
]

PATH_REJECTS = [
    "a/b/./c",
    "..",
    "a/../b",
    "/abs/path",
    "a//b",
    "$()",
    "${x}",
    "a$b",     # $ excluded entirely (shell-expansion risk)
    "x`y",     # backtick excluded (shell metachar)
    ".",       # bare current dir is not a valid file path
    "./",      # bare ./ is not a valid file path
]


@pytest.mark.parametrize("value", PATH_ACCEPTS)
def test_file_path_schema_accepts(value: str) -> None:
    assert Draft202012Validator(REPOSITORY_FILE_PATH_SCHEMA).is_valid(value)
    assert re.compile(REPOSITORY_FILE_PATH_SCHEMA["pattern"]).match(value)


@pytest.mark.parametrize("value", PATH_REJECTS)
def test_file_path_schema_rejects(value: str) -> None:
    assert not Draft202012Validator(REPOSITORY_FILE_PATH_SCHEMA).is_valid(value)
    assert not re.compile(REPOSITORY_FILE_PATH_SCHEMA["pattern"]).match(value)


@pytest.mark.parametrize("value", PATH_ACCEPTS)
def test_directory_scope_schema_accepts(value: str) -> None:
    assert Draft202012Validator(DIRECTORY_SCOPE_SCHEMA).is_valid(value)
    assert re.compile(DIRECTORY_SCOPE_SCHEMA["pattern"]).match(value)


@pytest.mark.parametrize("value", PATH_REJECTS)
def test_directory_scope_schema_rejects(value: str) -> None:
    assert not Draft202012Validator(DIRECTORY_SCOPE_SCHEMA).is_valid(value)
    assert not re.compile(DIRECTORY_SCOPE_SCHEMA["pattern"]).match(value)


def test_file_path_empty_and_length_bounds() -> None:
    # empty always rejected
    assert not Draft202012Validator(REPOSITORY_FILE_PATH_SCHEMA).is_valid("")
    # 4096 (PATH_MAX) is the cap: accepted up to and including, rejected beyond
    assert Draft202012Validator(REPOSITORY_FILE_PATH_SCHEMA).is_valid("a" * 4096)
    assert not Draft202012Validator(REPOSITORY_FILE_PATH_SCHEMA).is_valid("a" * 4097)
    # the lexical gate must agree with the schema at the same cap
    assert valid_repository_file_path("a" * 4096)
    assert not valid_repository_file_path("a" * 4097)
    # the directory-scope pair must agree at the same 4096 (PATH_MAX) cap
    assert Draft202012Validator(DIRECTORY_SCOPE_SCHEMA).is_valid("a" * 4096)
    assert not Draft202012Validator(DIRECTORY_SCOPE_SCHEMA).is_valid("a" * 4097)
    assert valid_directory_scope_lexical("a" * 4096)
    assert not valid_directory_scope_lexical("a" * 4097)
    # PATH_MAX is a byte budget (POSIX), so the lexical gates measure UTF-8
    # bytes, not code points: 2048 × 2-byte é == 4096 bytes == the cap.
    assert valid_repository_file_path("é" * 2048)
    assert not valid_repository_file_path("é" * 2049)
    assert valid_directory_scope_lexical("é" * 2048)
    assert not valid_directory_scope_lexical("é" * 2049)


def test_working_directory_accepts_cwd() -> None:
    assert Draft202012Validator(WORKING_DIRECTORY_SCHEMA).is_valid(".")
    regex = re.compile(WORKING_DIRECTORY_SCHEMA["pattern"])
    assert regex.match(".")
    assert regex.match("src/main.rs")
    # A well-formed absolute path is lexically acceptable (confinement is
    # enforced by path_is_confined + the host check, not this pattern).
    assert regex.match("/abs/path")
    assert Draft202012Validator(WORKING_DIRECTORY_SCHEMA).is_valid("/abs/path")
    # Absolute form still requires a non-empty segment after the slash.
    assert not regex.match("/")
    assert not regex.match("//double-slash")
    # The ./ prefix is relative-only: /./foo must not be schema-legal (the
    # runtime confinement gate rejects it, so the gates must agree).
    assert not regex.match("/./foo")
    assert not Draft202012Validator(WORKING_DIRECTORY_SCHEMA).is_valid("/./foo")
    # issues #572/#573: inherited grammar now accepts legal names; cap is 4096
    assert Draft202012Validator(WORKING_DIRECTORY_SCHEMA).is_valid("Café.md")
    assert Draft202012Validator(WORKING_DIRECTORY_SCHEMA).is_valid("./foo")
    assert Draft202012Validator(WORKING_DIRECTORY_SCHEMA).is_valid("a" * 4096)
    assert not Draft202012Validator(WORKING_DIRECTORY_SCHEMA).is_valid("a" * 4097)


def test_lexical_gates_reject_trailing_newline() -> None:
    """Trailing newline is rejected by the fullmatch gates, not the schema.

    The exported patterns anchor with ^/$ because the A/Z anchors are not
    valid ECMA-262 (Codex/OpenAI strict mode rejects them), and Python
    re.search lets $ match before a trailing newline, so the schema alone
    cannot express this rejection. The runtime gates are the enforcement point.
    """
    for value in ("foo\n", "src/main.rs\n"):
        assert not valid_repository_file_path(value)
        assert not valid_directory_scope_lexical(value)


@pytest.mark.parametrize("value", MULTI_DOT_PATHS)
def test_multi_dot_paths_are_accepted_by_schemas_and_validators(
    value: str, tmp_path: Path
) -> None:
    assert Draft202012Validator(REPOSITORY_FILE_PATH_SCHEMA).is_valid(value)
    assert Draft202012Validator(DIRECTORY_SCOPE_SCHEMA).is_valid(value)
    assert Draft202012Validator(WORKING_DIRECTORY_SCHEMA).is_valid(value)
    assert valid_repository_file_path(value)
    assert valid_directory_scope_lexical(value)
    assert path_is_confined(tmp_path, value)
    assert path_is_confined(tmp_path, value, directory_scope=True)


CONSISTENCY_ACCEPT = PATH_ACCEPTS + MULTI_DOT_PATHS
CONSISTENCY_REJECT = PATH_REJECTS


def test_schema_and_lexical_gate_agree_on_corpus() -> None:
    """Schema-valid must equal runtime-accepted for every corpus path."""
    file_validator = Draft202012Validator(REPOSITORY_FILE_PATH_SCHEMA)
    for value in CONSISTENCY_ACCEPT:
        assert file_validator.is_valid(value), f"schema should accept {value!r}"
        assert valid_repository_file_path(value), f"lexical gate should accept {value!r}"
    for value in CONSISTENCY_REJECT:
        assert not file_validator.is_valid(value), f"schema should reject {value!r}"
        assert not valid_repository_file_path(value), f"lexical gate should reject {value!r}"


def test_validate_applicability_collapses_dot_slash_scope_spellings(
    tmp_path: Path,
) -> None:
    """``./foo`` and ``foo`` are the same directory: string dedup must collapse them."""
    repo = tmp_path / "repo"
    (repo / "frontend").mkdir(parents=True)

    def applicability(paths: list[str]) -> dict[str, Any]:
        return {
            "scope": {"kind": "in-scope-paths", "paths": paths},
            "preconditions": [],
            "rationale": "same directory spelled twice",
        }

    normalized, rejection = validate_applicability(
        applicability(["frontend/", "./frontend", "frontend"]), repo=repo
    )
    assert rejection is not None
    assert (rejection.code, rejection.pointer) == (
        "RECON_APPLICABILITY_INVALID",
        "/scope/paths",
    )
    assert normalized is None
    # a lone ./-prefixed scope is legal and canonicalizes to the bare spelling
    normalized, rejection = validate_applicability(
        applicability(["./frontend"]), repo=repo
    )
    assert rejection is None
    assert normalized is not None
    assert normalized["scope"]["paths"] == ["frontend"]
    assert canonicalize_directory_scope("./frontend") == "frontend"
    assert canonicalize_directory_scope("./frontend/") == "frontend"


def test_legal_filenames_are_confined(tmp_path: Path) -> None:
    """Legal filenames pass the confinement walk for real files; traversal never does."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "~").mkdir()  # corpus spelling ~/.bashrc nests through a ~ dir
    for name in LEGAL_FILENAMES:
        (repo / name).write_text("x")
    for name in LEGAL_FILENAMES + ["./foo bar.py"]:
        assert path_is_confined(repo, name), f"{name!r} must be confined"
    # security carve-outs still rejected by the confinement gate too
    assert not path_is_confined(repo, "../x")
    assert not path_is_confined(repo, "/abs/path")
    assert not path_is_confined(repo, "a/../b")


def test_path_is_confined_allow_absolute(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    (repo / "improve").mkdir(parents=True)
    abs_under = (repo / "improve").as_posix()
    abs_outside = (repo.parent / "outside").as_posix()
    # allow_absolute=True: confined absolute accepted, outside rejected.
    assert path_is_confined(repo, abs_under, allow_absolute=True)
    assert not path_is_confined(repo, abs_outside, allow_absolute=True)
    # default (allow_absolute=False) must still reject absolute — this is what
    # keeps evidence source_path / applicability scope paths unchanged.
    assert not path_is_confined(repo, abs_under)


def test_path_is_confined_allow_absolute_symlinked_ancestor(
    tmp_path: Path,
) -> None:
    """A symlinked ancestor of the repo must not reject the absolute spelling.

    Regression pin for the macOS ``/tmp`` -> ``/private/tmp`` case: with
    ``allow_absolute=True`` the ancestor-skip strips the repo-root prefix from
    the walk, so an absolute value spelled through a symlinked ancestor of the
    repo is accepted exactly like the identical relative spelling.
    """
    real = tmp_path / "real"
    (real / "repo" / "improve").mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    repo = alias / "repo"
    abs_under = (repo / "improve").as_posix()
    assert path_is_confined(repo, abs_under, allow_absolute=True)
    # Baseline: the identical relative spelling is confined either way.
    assert path_is_confined(repo, "improve")
    # The resolved spelling of the same path is confined too.
    assert path_is_confined(repo, (real / "repo" / "improve").as_posix(), allow_absolute=True)


def test_canonicalize_working_directory_relative_and_dot(tmp_path: Path) -> None:
    """Must-have #2: relative / ./ / trailing-slash / root spellings collapse."""
    assert canonicalize_working_directory(tmp_path, ".") == "."
    assert canonicalize_working_directory(tmp_path, "sub") == "sub"
    assert canonicalize_working_directory(tmp_path, "./sub") == "sub"
    assert canonicalize_working_directory(tmp_path, "sub/") == "sub"


def test_canonicalize_working_directory_absolute_collapses_to_relative(tmp_path: Path) -> None:
    """Must-have #1: an absolute in-repo spelling maps to its repo-relative form."""
    assert canonicalize_working_directory(tmp_path, f"{tmp_path}/sub") == "sub"
    assert canonicalize_working_directory(tmp_path, f"{tmp_path}") == "."


def test_canonicalize_working_directory_all_spellings_share_one_key(tmp_path: Path) -> None:
    """Discriminating: the SAME directory in every accepted spelling yields ONE key."""
    spelled = {
        canonicalize_working_directory(tmp_path, wd)
        for wd in ("sub", "./sub", "sub/", f"{tmp_path}/sub")
    }
    assert spelled == {"sub"}


def test_canonicalize_working_directory_resolved_repo_spelling(tmp_path: Path) -> None:
    """The repo.resolve() fallback base: a symlinked repo strips a value
    spelled with its RESOLVED path, which the literal prefix cannot match."""
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    # Literal spelling strips on the first (unresolved) base.
    assert canonicalize_working_directory(link, f"{link}/sub") == "sub"
    # Resolved spelling matches only the repo.resolve() base.
    assert canonicalize_working_directory(link, f"{link.resolve()}/sub") == "sub"


def test_canonicalize_working_directory_collapses_parent_traversal(tmp_path: Path) -> None:
    """A ``..``-containing absolute spelling of an in-repo directory yields the
    same key as its plain spelling (the issue #649 dedup miss)."""
    assert canonicalize_working_directory(tmp_path, f"{tmp_path}/sub/../sub") == "sub"


@pytest.mark.anyio
async def test_host_enumeration_dedups_absolute_model_wd(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    """Must-have #1 at the entrypoint: an absolute-spelled model
    working_directory collapses against the relative host record, so the
    host command is NOT re-admitted into recon.json.

    Fixture contract: ``improve_monorepo_target`` must remain a monorepo
    that (a) declares the test command ``uv run pytest`` in its root
    ``pyproject.toml`` and (b) contains at least one service directory
    under ``apps/``. Both facts are derived from the fixture's actual
    content below rather than hardcoded, so any layout/names that retain
    that semantic content keep this test driving the dedup path; a future
    editor changing the fixture must preserve those two properties or this
    test fails with the attributable errors below."""
    # Discover a real service directory from the fixture instead of
    # hardcoding a name: any directory under apps/ works, because dedup
    # collapses the absolute model wd against the relative host wd.
    apps_dir = improve_monorepo_target / "apps"
    service_entries = (
        sorted(p for p in apps_dir.iterdir() if p.is_dir()) if apps_dir.is_dir() else []
    )
    if not service_entries:
        raise AssertionError(
            "improve_monorepo_target fixture must contain at least one "
            "service directory under apps/"
        )
    service = service_entries[0].name
    rel = f"apps/{service}"
    abs_wd = f"{improve_monorepo_target}/apps/{service}"
    monkeypatch.setattr(
        "daydream.improve.orchestrator.enumerate_repository_commands",
        lambda repo, *, directories=(".",), reserved_ids=(): [
            {
                "id": "make-check",
                "purpose": "Run the repository test suite",
                "command": "uv run pytest",
                "working_directory": rel,
                "expected_success": {
                    "exit_code": 0,
                    "observable_result": "exit 0 and the tests pass",
                },
                "applicability": {
                    "scope": {"kind": "in-scope-paths", "paths": [rel]},
                    "preconditions": [],
                    "rationale": f"The {service} service declares the test command.",
                },
                "evidence": {
                    "kind": "host-derived",
                    "source_path": f"{rel}/pyproject.toml",
                    "line_anchor": {"start_line": 1, "end_line": 1},
                    "verbatim_excerpt": "[project]",
                },
            }
        ],
    )
    stub = install_improve_stub(monkeypatch, improve_monorepo_target)

    # Derive the evidence anchor from the fixture's actual content instead of
    # hardcoding a line number: the root pyproject.toml must declare the test
    # subject command, and the validated line is wherever that declaration
    # lives (any layout that keeps the declaration works).
    root_pyproject = (
        improve_monorepo_target / "pyproject.toml"
    ).read_text(encoding="utf-8").splitlines()
    anchor_line = next(
        (i for i, line in enumerate(root_pyproject, 1) if "uv run pytest" in line),
        None,
    )
    if anchor_line is None:
        raise AssertionError(
            "improve_monorepo_target fixture must declare test command "
            "'uv run pytest' in its root pyproject.toml"
        )

    stub.recon_output_override = {
        "languages": ["python"],
        "commands": [
            {
                "id": "model-check",
                "purpose": "Run the repository test suite",
                "command": "uv run pytest",
                # Absolute spelling of the SAME directory the host enumerates
                # relative; dedup collapses the two spellings.
                "working_directory": abs_wd,
                "expected_success": {
                    "exit_code": 0,
                    "observable_result": "exit 0 and the tests pass",
                },
                "applicability": {
                    "scope": {"kind": "whole-repository"},
                    "preconditions": [],
                    "rationale": "The root configuration declares the test command.",
                },
                "evidence": {
                    "kind": "literal-command",
                    "source_path": "pyproject.toml",
                    "line_anchor": {"start_line": anchor_line, "end_line": anchor_line},
                    "verbatim_excerpt": None,
                },
            }
        ],
        "conventions": ["OpenAPI First"],
        "intent_docs": ["README.md"],
    }
    stub.plan_gate_on_first_menu_id = True

    code = await run(make_config(improve_monorepo_target, flow_name="improve"))

    assert code == 0
    recon = json.loads(
        improve_artifact(improve_monorepo_target, "recon.json").read_text(encoding="utf-8")
    )
    # The host command was deduped against the absolute model wd: no
    # re-admitted make-check record, no rejection noise.
    assert [command["id"] for command in recon["commands"]] == ["model-check"]
    assert recon["command_rejections"] == []


def test_host_enumeration_does_not_dedup_different_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Discriminating: a genuinely different directory is NOT collapsed —
    normalization must not widen dedup to 'same command anywhere'."""
    from daydream.improve import orchestrator as orch

    host_record = {
        "id": "make-check",
        "command": "make check",
        "working_directory": "sub",
    }
    monkeypatch.setattr(
        orch, "enumerate_repository_commands", lambda *a, **k: [host_record]
    )
    model = [{
        "id": "m1",
        "command": "make check",
        "working_directory": f"{tmp_path}/other",
    }]
    count, _, _ = orch._host_enumerated_commands(
        tmp_path, [], [], model_commands=model
    )
    assert count == 1  # different directory stays a separate candidate


def test_path_is_confined_allow_absolute_cannot_escape_via_ancestor_skip(
    tmp_path: Path,
) -> None:
    """The ancestor-skip must not enable a symlink escape from containment.

    Regression pin for the claim that skipping symlink tests on components
    above the repo root cannot escape containment: a symlink at or below the
    repo root that points outside the repo is still rejected even with
    ``allow_absolute=True``.
    """
    repo = tmp_path / "repo"
    (repo / "improve").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (repo / "improve" / "escape").symlink_to(outside, target_is_directory=True)
    abs_escape = (repo / "improve" / "escape").as_posix()
    assert not path_is_confined(repo, abs_escape, allow_absolute=True)
    # A direct absolute path to the outside directory is rejected as well.
    assert not path_is_confined(repo, outside.as_posix(), allow_absolute=True)


def _iter_schema_patterns(schema: Any) -> Iterator[str]:
    """Yield the string value of every ``pattern`` key in a schema tree."""
    if isinstance(schema, dict):
        for key, value in schema.items():
            if key == "pattern" and isinstance(value, str):
                yield value
            else:
                yield from _iter_schema_patterns(value)
    elif isinstance(schema, list):
        for item in schema:
            yield from _iter_schema_patterns(item)


@pytest.mark.anyio
async def test_improve_recon_schema_sent_to_backend_is_lookaround_free(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    make_config: MakeConfig,
) -> None:
    stub = install_improve_stub(monkeypatch, improve_monorepo_target)
    code = await run(make_config(improve_monorepo_target, flow_name="improve"))
    assert code == 0
    schemas = [
        call["output_schema"]
        for call in stub.calls
        if call.get("output_schema")
    ]
    assert schemas, "expected at least one structured-output turn"
    for schema in schemas:
        for pattern in _iter_schema_patterns(schema):
            assert re.search(r"\(\?[=!<]", pattern) is None
