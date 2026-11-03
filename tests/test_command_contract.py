"""Lookaround-free command-contract schema patterns (issue #304).

OpenAI rejects regex lookaround in JSON Schema ``pattern`` values with
``invalid_json_schema``. The exported path patterns must contain none, while
preserving the accept/reject grammar.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from daydream.improve import recon as orch
from daydream.improve.command_contract import (
    WORKING_DIRECTORY_SCHEMA,
    canonicalize_directory_scope,
    validate_applicability,
    validate_recon_commands,
)


def test_validate_applicability_collapses_dot_slash_scope_spellings(tmp_path: Path,) -> None:
    """``./foo`` and ``foo`` are the same directory: string dedup must collapse them."""
    repo = tmp_path / "repo"
    (repo / "frontend").mkdir(parents=True)

    def applicability(paths: list[str]) -> dict[str, Any]:
        return {"scope": {"kind": "in-scope-paths", "paths": paths}, "preconditions": [],
            "rationale": "same directory spelled twice",
        }

    normalized, rejection = validate_applicability(applicability(["frontend/", "./frontend", "frontend"]), repo=repo)
    assert rejection is not None
    assert (rejection.code, rejection.pointer) == ("RECON_APPLICABILITY_INVALID", "/scope/paths",)
    assert normalized is None
    # a lone ./-prefixed scope is legal and canonicalizes to the bare spelling
    normalized, rejection = validate_applicability(applicability(["./frontend"]), repo=repo)
    assert rejection is None
    assert normalized is not None
    assert normalized["scope"]["paths"] == ["frontend"]
    assert canonicalize_directory_scope("./frontend") == "frontend"
    assert canonicalize_directory_scope("./frontend/") == "frontend"
    # The schema accepts "." and confinable absolute paths; "/./foo" is rejected
    # by both the schema and the runtime confinement gate, so the gates agree.
    working_directory = Draft202012Validator(WORKING_DIRECTORY_SCHEMA)
    assert [working_directory.is_valid(value) for value in (".", "src/main.rs", "/abs/path", "/", "//x", "/./foo")] == [
        True, True, True, False, False, False]


def test_host_enumeration_dedups_only_the_same_absolute_or_relative_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    """An absolute model working directory collapses against the relative host record.

    A genuinely different directory is not collapsed, so normalization never widens
    dedup to 'same command anywhere'. The absolute model record itself still
    validates against its confined directory and cited declaration line.
    """
    (tmp_path / "sub").mkdir()
    (tmp_path / "other").mkdir()
    (tmp_path / "pyproject.toml").write_text("[tool.x]\ntest = 'make check'\n")
    host_record = {"id": "make-check", "command": "make check", "working_directory": "sub"}
    monkeypatch.setattr(orch, "enumerate_repository_commands", lambda *a, **k: [host_record])
    for model_wd, expected in ((f"{tmp_path}/sub", 0), (f"{tmp_path}/other", 1)):
        model = [{"id": "m1", "command": "make check", "working_directory": model_wd}]
        count, _, _ = orch._host_enumerated_commands(tmp_path, [], [], model_commands=model)
        assert count == expected, model_wd
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (tmp_path / "escape").symlink_to(outside, target_is_directory=True)

    def record(command_id: str, working_directory: str) -> dict[str, Any]:
        return {
            "id": command_id, "purpose": "Run the repository test suite", "command": "make check",
            "working_directory": working_directory,
            "expected_success": {"exit_code": 0, "observable_result": "exit 0 and the tests pass"},
            "applicability": {"scope": {"kind": "whole-repository"}, "preconditions": [],
                              "rationale": "The root configuration declares the test command."},
            "evidence": {"kind": "literal-command", "source_path": "pyproject.toml",
                         "line_anchor": {"start_line": 2, "end_line": 2}, "verbatim_excerpt": None},
        }

    validated, errors = validate_recon_commands({"commands": [
        record("model-check", f"{tmp_path}/sub"), record("escape-check", f"{tmp_path}/escape"),
        record("outside-check", outside.as_posix()),
    ]}, repo=tmp_path)
    assert [command["id"] for command in validated] == ["model-check"]
    assert validated[0]["evidence"]["verbatim_excerpt"] == "test = 'make check'"
    assert errors == [f"RECON_WORKING_DIRECTORY_INVALID@/commands/{index}/working_directory" for index in (1, 2)]
