"""Lookaround-free command-contract schema patterns (issue #304).

OpenAI rejects regex lookaround in JSON Schema ``pattern`` values with
``invalid_json_schema``. The exported path patterns must contain none, while
preserving the accept/reject grammar.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from daydream.improve.command_contract import (
    DIRECTORY_SCOPE_SCHEMA,
    REPOSITORY_FILE_PATH_SCHEMA,
    WORKING_DIRECTORY_SCHEMA,
    path_is_confined,
    valid_directory_scope_lexical,
    valid_repository_file_path,
)

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
    assert not Draft202012Validator(REPOSITORY_FILE_PATH_SCHEMA).is_valid("")
    assert not Draft202012Validator(REPOSITORY_FILE_PATH_SCHEMA).is_valid(
        "a" * 513
    )


def test_working_directory_accepts_cwd() -> None:
    assert Draft202012Validator(WORKING_DIRECTORY_SCHEMA).is_valid(".")
    regex = re.compile(WORKING_DIRECTORY_SCHEMA["pattern"])
    assert regex.match(".")
    assert regex.match("src/main.rs")
    assert not regex.match("/abs/path")


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
