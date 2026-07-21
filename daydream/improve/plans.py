"""Persistent plan-directory state for the improve advisor flow."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from daydream.improve.command_contract import (
    command_argv as _command_argv,
)
from daydream.improve.command_contract import (
    has_shell_composition as _has_shell_composition,
)
from daydream.improve.command_contract import (
    literal_command_error as _literal_command_error,
)
from daydream.improve.command_contract import (
    path_is_confined as _path_is_confined,
)
from daydream.improve.command_contract import (
    valid_directory_scope_lexical as _valid_directory_scope,
)
from daydream.improve.command_contract import (
    valid_repository_file_path as _valid_repository_file_path,
)
from daydream.improve.prompts import PLAN_WRITER_SCHEMA
from daydream.trajectory import redact_text

REJECTIONS_SCHEMA_VERSION = 1
PLAN_WRITE_DIAGNOSTICS_SCHEMA_VERSION = 1
_FINGERPRINT_MARKER = re.compile(
    r"<!--\s*fingerprint:([^\s>]+)\s*-->"
)
_NUMBERED_PLAN = re.compile(r"^(\d{3})-[a-z0-9-]+\.md$")
_HOST_BLOCKED_STATUS = re.compile(
    r"^BLOCKED \(PLAN_(?:WRITER|VALIDATION)_FAILED: [^()\r\n]+\)$"
)
_SECTION = re.compile(r"^## (.+?)\s*$", re.MULTILINE)


def _redact_model_value(value: Any) -> Any:
    """Redact nested model-authored strings before durable host rendering."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [_redact_model_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_model_value(item) for item in value)
    if isinstance(value, dict):
        return {
            key: _redact_model_value(item)
            for key, item in value.items()
        }
    return value


def load_rejections(plans_dir: Path) -> dict[str, dict[str, Any]]:
    """Load durable rejections keyed by fingerprint.

    An absent, unreadable, malformed, or structurally invalid file is treated
    as empty so stale user-authored state cannot prevent a fresh audit.
    """
    path = plans_dir / "rejected.json"
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != REJECTIONS_SCHEMA_VERSION
        or not isinstance(payload.get("rejected"), list)
    ):
        return {}

    rejections: dict[str, dict[str, Any]] = {}
    for entry in payload["rejected"]:
        if not isinstance(entry, dict):
            continue
        fingerprint = entry.get("fingerprint")
        if isinstance(fingerprint, str) and fingerprint:
            rejections[fingerprint] = entry
    return rejections


def record_rejections(
    plans_dir: Path, entries: Sequence[dict[str, Any]]
) -> None:
    """Append rejection entries to the versioned durable envelope."""
    if not entries:
        return
    rejected = [
        _redact_model_value(entry)
        for entry in load_rejections(plans_dir).values()
    ]
    rejected.extend(
        _redact_model_value(dict(entry))
        for entry in entries
    )
    plans_dir.mkdir(parents=True, exist_ok=True)
    (plans_dir / "rejected.json").write_text(
        json.dumps(
            {
                "schema_version": REJECTIONS_SCHEMA_VERSION,
                "rejected": rejected,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _markdown_cell(value: Any) -> str:
    """Render a value safely inside a Markdown table cell."""
    return redact_text(str(value or "—")).replace("|", "\\|").replace("\n", " ")


def _section_content(markdown: str) -> dict[str, str]:
    matches = list(_SECTION.finditer(markdown))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else None
        sections[match.group(1)] = markdown[match.end() : end].strip()
    return sections


def resolve_review_plan_path(repo: Path, requested: str) -> Path:
    """Resolve a review target confined to ``repo/daydream_plans``."""
    plans_dir = (repo / "daydream_plans").resolve()
    candidate = Path(requested)
    if not candidate.is_absolute():
        candidate = repo / candidate
    candidate = candidate.resolve()
    if not candidate.is_relative_to(plans_dir):
        raise ValueError(
            "review-plan only accepts files under "
            f"{plans_dir}; received {requested!r}"
        )
    if not candidate.is_file():
        raise ValueError(f"review-plan file does not exist: {candidate}")
    return candidate


_SECRET_VALUE = re.compile(
    r"(?i)\b(?:token|password|secret|api[_-]?key)\b\s*[:=]\s*[^\s]+"
)
_SAFE_METADATA_LABEL = re.compile(r"^[A-Za-z0-9._:/-]{1,160}$")


def _all_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [
            text
            for child in value.values()
            for text in _all_strings(child)
        ]
    if isinstance(value, list):
        return [text for child in value for text in _all_strings(child)]
    return []


def _safe_metadata_label(value: Any, *, fallback: str) -> str:
    text = redact_text(str(value or "").strip())
    if not _SAFE_METADATA_LABEL.fullmatch(text):
        return fallback
    return text


def _received_metadata(value: Any) -> dict[str, Any]:
    received_type = (
        "null"
        if value is None
        else "object"
        if isinstance(value, dict)
        else "array"
        if isinstance(value, list)
        else type(value).__name__
    )
    metadata: dict[str, Any] = {
        "type": received_type,
        "object_count": 0,
        "array_count": 0,
        "string_count": 0,
        "string_length": 0,
        "top_level_count": (
            len(value) if isinstance(value, (dict, list)) else None
        ),
    }

    def count_shape(item: Any) -> None:
        if isinstance(item, dict):
            metadata["object_count"] += 1
            for child in item.values():
                count_shape(child)
        elif isinstance(item, list):
            metadata["array_count"] += 1
            for child in item:
                count_shape(child)
        elif isinstance(item, str):
            metadata["string_count"] += 1
            metadata["string_length"] += len(item)

    count_shape(value)
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError):
        metadata["sha256"] = None
        metadata["serialized_length"] = None
    else:
        metadata["sha256"] = hashlib.sha256(serialized).hexdigest()
        metadata["serialized_length"] = len(serialized)
    return metadata


def _validation_error(code_with_pointer: str) -> dict[str, str]:
    code, separator, embedded_pointer = code_with_pointer.partition("@")
    if separator and embedded_pointer.startswith("/"):
        pointer = embedded_pointer
    elif code.startswith("DEPENDENCY_"):
        pointer = "/dependencies"
    elif code.startswith(("EXCERPT_", "EXISTING_PATH_EXCERPT")):
        pointer = "/current_state_excerpts"
    elif code.startswith(("SCOPE_", "EMPTY_SCOPE", "NEW_PATH_", "EXISTING_PATH_")):
        pointer = "/scope"
    elif code.startswith(("STEP_", "CREATE_PATH_", "CHANGE_PATH_", "TARGET_STATE_")):
        pointer = "/steps"
    elif code.startswith(("TEST_",)):
        pointer = "/test_plan"
    elif code.startswith("DONE_"):
        pointer = "/done_criteria"
    elif code.startswith("STOP_"):
        pointer = "/stop_conditions"
    elif code in {
        "MALFORMED_COMMAND",
        "RECON_COMMAND_MISMATCH",
        "PLANNER_COMMAND_PREFIX_MISMATCH",
        "PLANNER_COMMAND_SHELL_COMPOSITION",
        "PATH_OUT_OF_SCOPE",
    }:
        pointer = "/commands_you_will_need"
    elif code == "MALFORMED_PATH":
        pointer = "/scope"
    else:
        pointer = "/"
    return {"code": code, "pointer": pointer}


def _validation_stage(errors: Sequence[str]) -> str:
    codes = [error.partition("@")[0] for error in errors]
    if any(code.startswith("DEPENDENCY_") for code in codes):
        return "dependency"
    if any(code in {"SCHEMA_INVALID", "LEGACY_MARKDOWN_OUTPUT"} for code in codes):
        return "schema"
    if any(code == "RENDER_FAILED" for code in codes):
        return "render"
    return "semantic"


def _attempt_diagnostic(
    *,
    finding: dict[str, Any],
    attempt: dict[str, Any] | None,
    received: Any,
    disposition: str,
    stage: str,
    errors: Sequence[str] = (),
    artifact: dict[str, Any] | None = None,
) -> dict[str, Any]:
    attempt = attempt or {}
    return {
        "recorded_at": datetime.now(UTC).isoformat(),
        "finding": {
            "fingerprint": str(finding.get("fingerprint") or ""),
            "title": redact_text(
                str(finding.get("title") or "Selected finding")
            ),
        },
        "planner": {
            "descriptor": _safe_metadata_label(
                attempt.get("descriptor"),
                fallback="plan-writer",
            ),
            "backend": _safe_metadata_label(
                attempt.get("backend"),
                fallback="unknown-backend",
            ),
            "model": _safe_metadata_label(
                attempt.get("model"),
                fallback="unknown-model",
            ),
        },
        "disposition": disposition,
        "stage": stage,
        "errors": [_validation_error(error) for error in errors],
        "validation_errors": [_validation_error(error) for error in errors],
        "received": _received_metadata(received),
        "artifact": artifact,
    }


def record_plan_write_diagnostics(
    path: Path,
    attempts: Sequence[dict[str, Any]],
    *,
    artifact_provenance: dict[str, str] | None = None,
) -> None:
    """Append sanitized plan-attempt metadata without retaining model content."""
    existing_attempts: list[dict[str, Any]] = []
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            existing = None
        if (
            isinstance(existing, dict)
            and existing.get("schema_version")
            == PLAN_WRITE_DIAGNOSTICS_SCHEMA_VERSION
            and isinstance(existing.get("attempts"), list)
            and (
                artifact_provenance is None
                or existing.get("artifact_provenance")
                == artifact_provenance
            )
        ):
            existing_attempts = [
                _redact_model_value(item)
                for item in existing["attempts"]
                if isinstance(item, dict)
            ]
    payload = {
        "schema_version": PLAN_WRITE_DIAGNOSTICS_SCHEMA_VERSION,
        "artifact_type": "daydream.plan-write-diagnostics",
        **(
            {"artifact_provenance": dict(artifact_provenance)}
            if artifact_provenance is not None
            else {}
        ),
        "attempts": [
            _redact_model_value(item)
            for item in [*existing_attempts, *attempts]
        ],
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )


def _read_host_file(repo: Path, path: str) -> str | None:
    try:
        return (repo / path).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return None


def _head_matches(repo: Path, planned_at: str) -> bool:
    planned = _git(repo, "rev-parse", "--verify", f"{planned_at}^{{commit}}")
    head = _git(repo, "rev-parse", "--verify", "HEAD")
    return (
        planned.returncode == 0
        and head.returncode == 0
        and planned.stdout.strip() == head.stdout.strip()
    )


def valid_plan_path(value: str, *, repo: Path) -> bool:
    """Return whether a repository file path is safe and confined to ``repo``."""
    return _valid_repository_file_path(value) and _path_is_confined(repo, value)


def valid_directory_scope(value: str, *, repo: Path) -> bool:
    """Return whether an authored directory prefix is safe and repo-confined."""
    return _valid_directory_scope(value) and _path_is_confined(
        repo,
        value,
        directory_scope=True,
    )


def _command_candidates(result: Any) -> list[Any]:
    """Find authored command fields for semantic validation."""
    if isinstance(result, dict):
        return [
            *([result["command"]] if "command" in result else []),
            *[
                command
                for value in result.values()
                for command in _command_candidates(value)
            ],
        ]
    if isinstance(result, list):
        return [
            command
            for value in result
            for command in _command_candidates(value)
        ]
    return []


def _authored_path_candidates(result: Any) -> tuple[list[Any], list[Any]]:
    """Find authored repository-file and directory-prefix fields."""
    if isinstance(result, dict):
        file_paths: list[Any] = []
        directory_scopes: list[Any] = []
        for key, value in result.items():
            if key == "out_of_scope_paths" and isinstance(value, list):
                for entry in value:
                    if isinstance(entry, dict):
                        directory_scopes.append(entry.get("path"))
                        for entry_key, entry_value in entry.items():
                            if entry_key != "path":
                                nested_files, nested_scopes = (
                                    _authored_path_candidates(entry_value)
                                )
                                file_paths.extend(nested_files)
                                directory_scopes.extend(nested_scopes)
                continue
            if key in {"path", "source_path", "test_file"}:
                file_paths.append(value)
            elif key == "paths" and isinstance(value, list):
                directory_scopes.extend(value)
            elif key == "related_paths" and isinstance(value, list):
                file_paths.extend(value)
            nested_files, nested_scopes = _authored_path_candidates(value)
            file_paths.extend(nested_files)
            directory_scopes.extend(nested_scopes)
        return file_paths, directory_scopes
    if isinstance(result, list):
        file_paths = []
        directory_scopes = []
        for value in result:
            nested_files, nested_scopes = _authored_path_candidates(value)
            file_paths.extend(nested_files)
            directory_scopes.extend(nested_scopes)
        return file_paths, directory_scopes
    return [], []


def _working_directory_candidates(result: Any) -> list[Any]:
    """Find authored working directories for semantic validation."""
    if isinstance(result, dict):
        candidates = (
            [result["working_directory"]]
            if "working_directory" in result
            else []
        )
        return [
            *candidates,
            *[
                working_directory
                for value in result.values()
                for working_directory in _working_directory_candidates(value)
            ],
        ]
    if isinstance(result, list):
        return [
            working_directory
            for value in result
            for working_directory in _working_directory_candidates(value)
        ]
    return []


def _command_error(
    command: dict[str, Any],
    in_scope: set[str],
    *,
    repo: Path,
    recon_by_id: dict[str, dict[str, Any]] | None = None,
) -> str | None:
    literal = command["command"]
    if _literal_command_error(literal, allow_shell_composition=True):
        return "MALFORMED_COMMAND"
    working_directory = command["working_directory"]
    if (
        working_directory != "."
        and not _valid_repository_file_path(working_directory)
    ):
        return "COMMAND_WORKING_DIRECTORY_INVALID"
    if (
        not _path_is_confined(repo, working_directory)
        or not (repo / working_directory).is_dir()
    ):
        return "PATH_OUTSIDE_REPOSITORY"
    applicability = command["applicability"]
    scope = applicability["scope"]
    paths = scope.get("paths", [])
    if any(not _valid_directory_scope(path) for path in paths):
        return "MALFORMED_PATH"
    if any(
        not _path_is_confined(repo, path, directory_scope=True)
        for path in paths
    ):
        return "PATH_OUTSIDE_REPOSITORY"
    if any(
        not any(
            in_scope_path == path.rstrip("/")
            or in_scope_path.startswith(f"{path.rstrip('/')}/")
            for in_scope_path in in_scope
        )
        for path in paths
    ):
        return "COMMAND_SCOPE_MISMATCH"
    provenance = command["provenance"]
    recon_id = provenance["recon_command_id"]
    source_path = provenance["source_path"]
    if provenance["kind"] == "planner-derived" and source_path not in in_scope:
        return "COMMAND_PATH_OUT_OF_SCOPE"
    if (
        provenance["kind"] == "planner-derived"
        and _has_shell_composition(literal)
    ):
        return "PLANNER_COMMAND_SHELL_COMPOSITION"
    if recon_by_id is not None:
        base = recon_by_id.get(recon_id)
        if base is None:
            return "RECON_COMMAND_UNKNOWN"
        if provenance["kind"] == "recon":
            if (
                source_path != base["evidence"]["source_path"]
                or any(
                    command[key] != base[key]
                    for key in (
                        "purpose",
                        "command",
                        "working_directory",
                        "expected_success",
                        "applicability",
                    )
                )
            ):
                return "RECON_COMMAND_MISMATCH"
        else:
            base_argv = _command_argv(base["command"])
            plan_argv = _command_argv(literal)
            if (
                base_argv is None
                or plan_argv is None
                or plan_argv[: len(base_argv)] != base_argv
                or command["working_directory"] != base["working_directory"]
            ):
                return "PLANNER_COMMAND_PREFIX_MISMATCH"
    return None


def validate_plan_result(
    result: Any,
    *,
    repo: Path,
    planned_at: str,
    finding: dict[str, Any] | None = None,
    recon_commands: Sequence[dict[str, Any]] | None = None,
) -> tuple[str, ...]:
    """Return stable fail-closed errors for a planner result.

    Backend schema enforcement is advisory. This host validation is the
    authoritative write boundary and intentionally returns codes, never rejected
    values, so callers can safely persist its result.
    """
    if isinstance(result, dict) and "markdown" in result:
        return ("LEGACY_MARKDOWN_OUTPUT",)
    if not isinstance(result, dict):
        return ("SCHEMA_INVALID@/",)
    schema_result = result
    raw_excerpts = result.get("current_state_excerpts")
    if isinstance(raw_excerpts, list):
        schema_result = {
            **result,
            "current_state_excerpts": [
                (
                    {
                        **excerpt,
                        "verbatim_excerpt": None,
                    }
                    if isinstance(excerpt, dict)
                    and not isinstance(excerpt.get("verbatim_excerpt"), str)
                    else excerpt
                )
                for excerpt in raw_excerpts
            ],
        }
    schema_errors = sorted(
        Draft202012Validator(PLAN_WRITER_SCHEMA).iter_errors(schema_result),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if schema_errors:
        error = schema_errors[0]
        pointer_parts = [str(part) for part in error.absolute_path]
        if error.validator == "required":
            missing = sorted(set(error.validator_value) - set(error.instance))
            if missing:
                pointer_parts.append(missing[0])
        pointer = "/" + "/".join(pointer_parts) if pointer_parts else "/"
        return (f"SCHEMA_INVALID@{pointer}",)

    if any(
        _literal_command_error(
            command,
            allow_shell_composition=True,
        )
        for command in _command_candidates(result)
    ):
        return ("MALFORMED_COMMAND",)
    repository_file_paths, directory_scopes = _authored_path_candidates(result)
    if any(
        not isinstance(path, str) or not _valid_repository_file_path(path)
        for path in repository_file_paths
    ) or any(
        not isinstance(path, str) or not _valid_directory_scope(path)
        for path in directory_scopes
    ):
        return ("MALFORMED_PATH",)
    if any(
        not _path_is_confined(repo, path)
        for path in repository_file_paths
        if isinstance(path, str)
    ) or any(
        not _path_is_confined(repo, path, directory_scope=True)
        for path in directory_scopes
        if isinstance(path, str)
    ) or any(
        not isinstance(working_directory, str)
        or (
            working_directory != "."
            and not _valid_repository_file_path(working_directory)
        )
        or not _path_is_confined(repo, working_directory)
        for working_directory in _working_directory_candidates(result)
    ):
        return ("PATH_OUTSIDE_REPOSITORY",)

    strings = _all_strings(result)
    if any(_SECRET_VALUE.search(value) for value in strings):
        return ("SECRET_CONTENT_REDACTED",)
    commit = _git(repo, "cat-file", "-e", f"{planned_at}^{{commit}}")
    if commit.returncode != 0:
        return ("PLANNED_AT_INVALID",)
    ancestor = _git(repo, "merge-base", "--is-ancestor", planned_at, "HEAD")
    if ancestor.returncode != 0:
        return ("PLANNED_AT_NOT_ANCESTOR",)

    scope = result["scope"]
    existing = [item["path"] for item in scope["existing_paths"]]
    new = [item["path"] for item in scope["new_paths"]]
    excluded = [item["path"] for item in scope["out_of_scope_paths"]]
    all_paths = [*existing, *new, *excluded]
    if not existing and not new:
        return ("EMPTY_SCOPE",)
    if any(
        not _valid_repository_file_path(path)
        for path in [*existing, *new]
    ) or any(not _valid_directory_scope(path) for path in excluded):
        return ("MALFORMED_PATH",)
    if len(set(all_paths)) != len(all_paths):
        return ("SCOPE_PATH_CONFLICT",)
    if any(_read_host_file(repo, path) is None for path in existing):
        return ("EXISTING_PATH_MISSING",)
    if any((repo / path).exists() for path in new):
        return ("NEW_PATH_ALREADY_EXISTS",)
    in_scope = set(existing + new)
    recon_by_id = (
        {command["id"]: command for command in recon_commands}
        if recon_commands is not None
        else None
    )

    excerpt_paths: set[str] = set()
    for excerpt in result["current_state_excerpts"]:
        path = excerpt["path"]
        if not _valid_repository_file_path(path):
            return ("MALFORMED_PATH",)
        source = _read_host_file(repo, path)
        if source is None:
            return ("EXCERPT_PATH_MISSING",)
        start = excerpt["line_anchor"]["start_line"]
        end = excerpt["line_anchor"]["end_line"]
        lines = source.splitlines()
        if end < start or end > len(lines):
            return ("EXCERPT_ANCHOR_INVALID",)
        actual = "\n".join(lines[start - 1 : end])
        excerpt["verbatim_excerpt"] = actual
        excerpt_paths.add(path)
    if not set(existing) <= excerpt_paths:
        return ("EXISTING_PATH_EXCERPT_MISSING",)

    steps = result["steps"]
    if [step["order"] for step in steps] != list(range(1, len(steps) + 1)):
        return ("STEP_ORDER_INVALID",)
    step_ids = [step["id"] for step in steps]
    if len(set(step_ids)) != len(step_ids):
        return ("STEP_ID_DUPLICATE",)
    for step in steps:
        changed_paths: set[str] = set()
        for change in step["changes"]:
            path = change["path"]
            changed_paths.add(path)
            if path not in in_scope:
                return ("STEP_PATH_OUT_OF_SCOPE",)
            if change["operation"] == "create" and path not in new:
                return ("CREATE_PATH_NOT_NEW",)
            if change["operation"] != "create" and path not in existing:
                return ("CHANGE_PATH_NOT_EXISTING",)
        verification = step["verification"]
        if verification is not None:
            command_error = _command_error(
                verification,
                in_scope,
                repo=repo,
                recon_by_id=recon_by_id,
            )
            if command_error:
                return (command_error,)
            scope = verification["applicability"]["scope"]
            scope_paths = scope.get("paths", [])
            if (
                scope["kind"] != "whole-repository"
                and any(
                    not any(
                        changed_path == path.rstrip("/")
                        or changed_path.startswith(f"{path.rstrip('/')}/")
                        for path in scope_paths
                    )
                    for changed_path in changed_paths
                )
            ):
                return ("STEP_GATE_SCOPE_MISMATCH",)

    for command in result["commands_you_will_need"]:
        if command_error := _command_error(
            command,
            in_scope,
            repo=repo,
            recon_by_id=recon_by_id,
        ):
            return (command_error,)
    test_symbols: set[str] = set()
    for exemplar in result["test_plan"]["exemplars"]:
        source = _read_host_file(repo, exemplar["path"])
        if source is None or exemplar["symbol"] not in source:
            return ("TEST_EXEMPLAR_INVALID",)
    for case in result["test_plan"]["cases"]:
        if case["test_file"] not in in_scope:
            return ("TEST_PATH_OUT_OF_SCOPE",)
        if case["test_symbol"] in test_symbols:
            return ("TEST_SYMBOL_DUPLICATE",)
        test_symbols.add(case["test_symbol"])
        if case["verification"] is not None:
            if command_error := _command_error(
                case["verification"],
                in_scope,
                repo=repo,
                recon_by_id=recon_by_id,
            ):
                return (command_error,)

    done_kinds = {criterion["kind"] for criterion in result["done_criteria"]}
    if not {"behavior", "test-gate", "scope-integrity"} <= done_kinds:
        return ("DONE_CRITERIA_INCOMPLETE",)
    for criterion in result["done_criteria"]:
        if criterion["verification"] is not None:
            if command_error := _command_error(
                criterion["verification"],
                in_scope,
                repo=repo,
                recon_by_id=recon_by_id,
            ):
                return (command_error,)
    stop_kinds = {condition["kind"] for condition in result["stop_conditions"]}
    if not {
        "drift",
        "repeated-verification-failure",
        "out-of-scope-change",
        "false-assumption",
    } <= stop_kinds:
        return ("STOP_CONDITIONS_INCOMPLETE",)
    if any(
        path not in in_scope and path not in set(excluded)
        for condition in result["stop_conditions"]
        for path in condition["related_paths"]
    ):
        return ("STOP_PATH_UNKNOWN",)
    if any(
        step_id not in set(step_ids)
        for condition in result["stop_conditions"]
        for step_id in condition["related_step_ids"]
    ):
        return ("STOP_STEP_UNKNOWN",)

    return ()


def _dependency_cycle_slugs(
    selections: Sequence[dict[str, Any]],
) -> set[str]:
    graph: dict[str, set[str]] = {}
    for selection in selections:
        slug = selection.get("slug")
        dependencies = selection.get("dependencies")
        if not isinstance(slug, str) or not isinstance(dependencies, list):
            continue
        graph[slug] = {
            dependency["slug"]
            for dependency in dependencies
            if isinstance(dependency, dict)
            and isinstance(dependency.get("slug"), str)
        }

    cyclic: set[str] = set()
    visiting: list[str] = []
    visited: set[str] = set()

    def visit(slug: str) -> None:
        if slug in visiting:
            cyclic.update(visiting[visiting.index(slug) :])
            return
        if slug in visited:
            return
        visiting.append(slug)
        for dependency in graph.get(slug, set()):
            if dependency in graph:
                visit(dependency)
        visiting.pop()
        visited.add(slug)

    for slug in graph:
        visit(slug)
    return cyclic


def _dependency_order(
    selections: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return a stable dependency-first order; retain input order for cycles."""
    remaining = list(selections)
    candidate_slugs = {
        selection.get("slug")
        for selection in remaining
        if isinstance(selection.get("slug"), str)
    }
    emitted: set[str] = set()
    ordered: list[dict[str, Any]] = []
    while remaining:
        ready: list[dict[str, Any]] = []
        for selection in remaining:
            dependencies = selection.get("dependencies")
            internal = {
                dependency.get("slug")
                for dependency in dependencies
                if isinstance(dependency, dict)
            } if isinstance(dependencies, list) else set()
            if not (internal & candidate_slugs) - emitted:
                ready.append(selection)
        if not ready:
            ordered.extend(remaining)
            break
        for selection in ready:
            remaining.remove(selection)
            ordered.append(selection)
            slug = selection.get("slug")
            if isinstance(slug, str):
                emitted.add(slug)
    return ordered


def _commands_table(commands: Sequence[dict[str, Any]]) -> str:
    if not commands:
        return "No host-verified repository commands were available during planning."
    lines = [
        "| Purpose | Command | Expected on success |",
        "|---------|---------|---------------------|",
    ]
    lines.extend(
        f"| {_markdown_cell(command['purpose'])} | `{command['command']}` | "
        f"exit {command['expected_success']['exit_code']}; "
        f"{_markdown_cell(command['expected_success']['observable_result'])} |"
        for command in commands
    )
    return "\n".join(lines)


def _render_verification(command: dict[str, Any] | None) -> str:
    if command is None:
        return "No host-verified command is attached to this step."
    expected = command["expected_success"]
    exit_prefix = f"exit {expected['exit_code']} and "
    observable = expected["observable_result"]
    expected_text = (
        observable
        if observable.casefold().startswith(exit_prefix.casefold())
        else f"exit {expected['exit_code']}; {observable}"
    )
    return (
        f"**Purpose**: {command['purpose']}\n\n"
        f"**Command**: `{command['command']}`\n\n"
        f"**Expected**: {expected_text}"
    )


def _render_verification_list(
    command: dict[str, Any] | None,
    *,
    indent: str,
) -> str:
    if command is None:
        return f"{indent}- No host-verified command is attached."
    expected = command["expected_success"]
    return "\n".join(
        (
            f"{indent}- **Purpose**: {command['purpose']}",
            f"{indent}- **Command**: `{command['command']}`",
            (
                f"{indent}- **Expected**: exit {expected['exit_code']}; "
                f"{expected['observable_result']}"
            ),
        )
    )


def render_plan(
    finding: dict[str, Any],
    *,
    plan: dict[str, Any],
    planned_at: str,
    number: int,
    planned_on: date | None = None,
    run_session_id: str | None = None,
) -> str:
    """Render validated PlanWriterResult data without authored Markdown."""
    finding = _redact_model_value(finding)
    plan = _redact_model_value(plan)
    planned_date = planned_on or date.today()
    scope = plan["scope"]
    dependencies = ", ".join(
        dependency["slug"] for dependency in plan["dependencies"]
    ) or "none"

    why = plan["why_this_matters"]
    current_state: list[str] = []
    for excerpt in plan["current_state_excerpts"]:
        anchor = excerpt["line_anchor"]
        current_state.extend(
            [
                (
                    f"- `{excerpt['path']}:{anchor['start_line']}-"
                    f"{anchor['end_line']}` — {excerpt['file_role']}"
                ),
                "",
                "```text",
                excerpt["verbatim_excerpt"],
                "```",
            ]
        )

    in_scope_lines = [
        *[
            f"- `{entry['path']}` (existing) — {entry['role']}"
            for entry in scope["existing_paths"]
        ],
        *[
            f"- `{entry['path']}` (create) — {entry['role']}"
            for entry in scope["new_paths"]
        ],
    ]
    out_scope_lines = [
        *[
            f"- `{entry['path']}` — {entry['reason']}"
            for entry in scope["out_of_scope_paths"]
        ],
        *[
            f"- {entry['behavior']} — {entry['reason']}"
            for entry in scope["out_of_scope_behaviors"]
        ],
    ]
    workflow = plan["git_workflow"]
    step_sections: list[str] = []
    for step in sorted(plan["steps"], key=lambda item: item["order"]):
        changes = "\n".join(
            (
                f"- `{change['path']}` — `{change['symbol']}` "
                f"({change['operation']}): {change['instruction']} "
                f"Target state: {change['target_state']}"
            )
            for change in step["changes"]
        )
        step_sections.append(
            f"### Step {step['order']}: {step['title']}\n\n"
            f"{changes}\n\n"
            f"**Verify**\n\n{_render_verification(step['verification'])}"
        )
    test_plan = plan["test_plan"]
    exemplar_lines = "\n".join(
        f"- `{item['path']}` — `{item['symbol']}`: {item['pattern_to_copy']}"
        for item in test_plan["exemplars"]
    )
    case_lines = "\n\n".join(
        (
            f"- **{case['name']}** — `{case['test_file']}::"
            f"{case['test_symbol']}` ({case['kind']})\n"
            f"  - Setup: {case['setup']}\n"
            f"  - Action: {case['action']}\n"
            + "\n".join(
                f"  - Assert: {assertion}" for assertion in case["assertions"]
            )
            + "\n  - Verification:\n"
            + _render_verification_list(case["verification"], indent="    ")
        )
        for case in test_plan["cases"]
    )
    done_lines = "\n".join(
        f"- [ ] **{criterion['id']} ({criterion['kind']})**: "
        f"{criterion['description']}\n"
        f"{_render_verification_list(criterion['verification'], indent='  ')}"
        for criterion in plan["done_criteria"]
    )
    stop_lines = "\n".join(
        f"- **{condition['kind']}** — {condition['condition']} "
        f"STOP and report: {condition['evidence_to_report']}"
        for condition in plan["stop_conditions"]
    )
    notes = plan["maintenance_notes"]
    future_lines = "\n".join(
        f"- **{item['area']}**: {item['note']}"
        for item in notes["future_interactions"]
    )
    risk_lines = "\n".join(
        f"- {item['risk']} Review check: {item['review_check']}"
        for item in notes["review_risks"]
    )
    deferred_lines = "\n".join(
        f"- {item['item']} Reason: {item['reason']} "
        f"Revisit when: {item['revisit_trigger']}"
        for item in notes["deferred_items"]
    ) or "- None."

    return (
        f"# Plan {number:03d}: {plan['title']}\n\n"
        "> **Executor instructions**: Follow this plan step by step. Use each provided\n"
        "> verification command and confirm the expected result before moving to the\n"
        "> next step. If anything in the \"STOP conditions\" section occurs, stop and\n"
        "> report — do not improvise. When done, update the status row for this plan\n"
        "> in `daydream_plans/README.md` unless a reviewer maintains the index.\n"
        "\n"
        "## Status\n\n"
        f"- **Priority**: {plan['priority']}\n"
        f"- **Effort**: {finding.get('effort', '—')}\n"
        f"- **Risk**: {finding.get('risk', '—')}\n"
        f"- **Depends on**: {dependencies}\n"
        f"- **Category**: {finding.get('category', '—')}\n"
        f"- **Planned at**: commit `{planned_at[:7]}`, {planned_date.isoformat()}\n\n"
        + (
            f"Daydream run: `{run_session_id}`\n\n"
            if run_session_id is not None
            else ""
        )
        +
        "## Why this matters\n\n"
        f"{why['problem']} {why['concrete_cost']} {why['intended_outcome']}\n\n"
        "## Current state\n\n"
        + "\n".join(current_state)
        + "\n\n## Commands you will need\n\n"
        + _commands_table(plan["commands_you_will_need"])
        + "\n\n## Scope\n\n**In scope**\n\n"
        + "\n".join(in_scope_lines)
        + "\n\n**Out of scope**\n\n"
        + "\n".join(out_scope_lines)
        + "\n\n## Git workflow\n\n"
        f"- **Branch**: {workflow['branch_name']} ({workflow['branch_basis']})\n"
        f"- **Commit boundaries**: {workflow['commit_boundaries']}\n"
        f"- **Commit example**: `{workflow['commit_message_example']}`\n"
        "- **Push**: never without operator instruction\n"
        "- **Pull request**: never without operator instruction\n\n"
        "## Steps\n\n"
        + "\n\n".join(step_sections)
        + "\n\n## Test plan\n\n### Exemplars\n\n"
        + exemplar_lines
        + "\n\n### Named cases\n\n"
        + case_lines
        + "\n\n## Done criteria\n\n"
        + done_lines
        + "\n\n## STOP conditions\n\n"
        + stop_lines
        + "\n\n## Maintenance notes\n\n### Future interactions\n\n"
        + future_lines
        + "\n\n### Review risks\n\n"
        + risk_lines
        + "\n\n### Deferred items\n\n"
        + deferred_lines
        + "\n"
    )


def _existing_index_rows(index_text: str) -> list[str]:
    rows: list[str] = []
    for line in index_text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 6 or cells[0] in {"Plan", "------"}:
            continue
        if set(cells[0]) == {"-"}:
            continue
        rows.append(line)
    return rows


def _row_cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip("|").split("|")]


def _row_number(row: str) -> int | None:
    cells = _row_cells(row)
    if not cells:
        return None
    match = re.search(r"\b(\d{3})\b", cells[0])
    return int(match.group(1)) if match is not None else None


def _row_plan_path(row: str) -> str | None:
    cells = _row_cells(row)
    if not cells:
        return None
    match = re.search(r"\[\d{3}\]\(([^)]+)\)", cells[0])
    if match is None:
        return None
    filename = match.group(1)
    if Path(filename).name != filename or _NUMBERED_PLAN.fullmatch(filename) is None:
        return None
    return filename


def _row_has_plan_artifact(row: str, plans_dir: Path) -> bool:
    filename = _row_plan_path(row)
    if filename is not None and (plans_dir / filename).is_file():
        return True
    number = _row_number(row)
    return number is not None and any(
        plans_dir.glob(f"{number:03d}-*.md")
    )


def _retryable_host_blocked_row(row: str, plans_dir: Path) -> bool:
    cells = _row_cells(row)
    if len(cells) != 6 or _row_has_plan_artifact(row, plans_dir):
        return False
    return _HOST_BLOCKED_STATUS.fullmatch(cells[-1]) is not None


def planned_fingerprints(plans_dir: Path) -> set[str]:
    """Return fingerprints with durable executable/non-transient status."""
    index_path = plans_dir / "README.md"
    if not index_path.is_file():
        return set()
    try:
        index_text = index_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return set()
    rows = _existing_index_rows(index_text)
    return {
        match.group(1)
        for row in rows
        if not _retryable_host_blocked_row(row, plans_dir)
        if (match := _FINGERPRINT_MARKER.search(row)) is not None
    }


def _highest_plan_number(plans_dir: Path, rows: Sequence[str]) -> int:
    numbers = [
        int(match.group(1))
        for path in plans_dir.glob("[0-9][0-9][0-9]-*.md")
        if (match := _NUMBERED_PLAN.match(path.name)) is not None
    ]
    for row in rows:
        match = re.search(r"\b(\d{3})\b", row)
        if match is not None:
            numbers.append(int(match.group(1)))
    return max(numbers, default=0)


_DEPENDENCY_NOTE = re.compile(
    r"^(\d{3}) depends on ([a-z0-9-]+) because (.+)$"
)


def _existing_dependency_notes(
    index_text: str,
) -> tuple[dict[tuple[int, str], str], list[str]]:
    notes = _section_content(index_text).get("Dependency notes", "")
    by_edge: dict[tuple[int, str], str] = {}
    unstructured: list[str] = []
    for line in notes.splitlines():
        stripped = line.strip()
        if not stripped or stripped == "- None recorded.":
            continue
        if not stripped.startswith("- "):
            continue
        content = stripped[2:]
        match = _DEPENDENCY_NOTE.fullmatch(content)
        if match is None:
            if content not in unstructured:
                unstructured.append(content)
            continue
        key = (int(match.group(1)), match.group(2))
        by_edge.setdefault(key, content)
    return by_edge, unstructured


def _dependency_edges(rows: Sequence[str]) -> list[tuple[int, str]]:
    edges: list[tuple[int, str]] = []
    for row in rows:
        cells = _row_cells(row)
        number = _row_number(row)
        if len(cells) != 6 or number is None or cells[4] == "—":
            continue
        edges.extend(
            (number, dependency.strip())
            for dependency in cells[4].split(",")
            if re.fullmatch(r"[a-z0-9-]+", dependency.strip())
        )
    return edges


def _merged_dependency_notes(
    rows: Sequence[str],
    *,
    existing: dict[tuple[int, str], str],
    new: dict[tuple[int, str], str],
    unstructured: Sequence[str],
) -> list[str]:
    notes = [
        new[edge] if edge in new else existing[edge]
        for edge in _dependency_edges(rows)
        if edge in existing or edge in new
    ]
    notes.extend(note for note in unstructured if note not in notes)
    return notes


def _render_index(
    rows: Sequence[str],
    *,
    plans_dir: Path,
    non_interactive_default: bool,
    dependency_notes: Sequence[str],
    run_session_id: str | None,
) -> str:
    rejections = load_rejections(plans_dir)
    default_note = (
        "\nThe non-interactive default selected the top-N vetted defect "
        "findings by leverage.\n"
        if non_interactive_default
        else ""
    )
    rejected_lines = [
        f"- {_markdown_cell(entry.get('title'))}: "
        f"{_markdown_cell(entry.get('reason') or 'rejected during vetting')} "
        f"<!-- fingerprint:{fingerprint} -->"
        for fingerprint, entry in rejections.items()
    ]
    return (
        "# Implementation Plans\n\n"
        f"Generated by daydream improve on {date.today().isoformat()}. Execute "
        "in the order below unless dependencies say otherwise. Read each plan "
        "fully, honor its STOP conditions, and update its row when done.\n"
        + (
            f"\nDaydream run: `{run_session_id}`\n"
            if run_session_id is not None
            else ""
        )
        +
        f"{default_note}\n"
        "## Execution order & status\n\n"
        "| Plan | Title | Priority | Effort | Depends on | Status |\n"
        "|------|-------|----------|--------|------------|--------|\n"
        + ("\n".join(rows) if rows else "| — | No plans written. | — | — | — | — |")
        + "\n\nStatus values: TODO | IN PROGRESS | DONE | BLOCKED "
        "(with one-line reason) | REJECTED (with one-line rationale)\n\n"
        "## Dependency notes\n\n"
        + (
            "\n".join(f"- {note}" for note in dependency_notes)
            if dependency_notes
            else "- None recorded."
        )
        + "\n\n## Findings considered and rejected\n\n"
        + ("\n".join(rejected_lines) if rejected_lines else "- None.")
        + "\n"
    )


def _blocked_index_row(
    *,
    number: int,
    marker: str,
    finding: dict[str, Any],
    status: str,
) -> str:
    """Render a blocked row without consulting rejected planner metadata."""
    trusted_title = str(finding.get("title") or "Selected finding")
    return (
        f"| {number:03d} {marker} | {_markdown_cell(trusted_title)} | P2 | "
        f"{_markdown_cell(finding.get('effort'))} | — | {status} |"
    )


def write_plans(
    plans_dir: Path,
    selections: Sequence[dict[str, Any]],
    *,
    planned_at: str,
    commands: Sequence[dict[str, Any]] | None = None,
    non_interactive_default: bool = False,
    run_session_id: str | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Reconcile selected plan-writer results into files and the durable index."""
    plans_dir.mkdir(parents=True, exist_ok=True)
    index_path = plans_dir / "README.md"
    index_text = (
        index_path.read_text(encoding="utf-8") if index_path.is_file() else ""
    )
    rows = _existing_index_rows(index_text)
    existing_dependency_notes, unstructured_dependency_notes = (
        _existing_dependency_notes(index_text)
    )
    fingerprints = planned_fingerprints(plans_dir)
    rejected = load_rejections(plans_dir)
    next_number = _highest_plan_number(plans_dir, rows) + 1
    result: dict[str, list[dict[str, Any]]] = {
        "written": [],
        "skipped": [],
        "failed": [],
        "diagnostics": [],
    }
    new_dependency_notes: dict[tuple[int, str], str] = {}
    safe_selections = [
        safe
        for selection in selections
        if isinstance(
            (safe := _redact_model_value(selection)),
            dict,
        )
    ]
    candidate_slugs = {
        selection.get("slug")
        for selection in safe_selections
        if isinstance(selection.get("slug"), str)
    }
    cycle_slugs = _dependency_cycle_slugs(safe_selections)
    ordered_selections = _dependency_order(safe_selections)
    available_slugs = {
        match.group(1).split("-", 1)[1]
        for path in plans_dir.glob("[0-9][0-9][0-9]-*.md")
        if (match := re.match(r"^(\d{3}-.+)\.md$", path.name))
    }

    for selection in ordered_selections:
        finding = selection.get("finding")
        if not isinstance(finding, dict):
            continue
        fingerprint = str(finding.get("fingerprint") or "")
        attempt = (
            selection.get("_attempt")
            if isinstance(selection.get("_attempt"), dict)
            else None
        )
        if fingerprint in fingerprints or fingerprint in rejected:
            result["skipped"].append(finding)
            if attempt is not None:
                received = {
                    key: value
                    for key, value in selection.items()
                    if key not in {"finding", "error"} and not key.startswith("_")
                }
                result["diagnostics"].append(
                    _attempt_diagnostic(
                        finding=finding,
                        attempt=attempt,
                        received=received,
                        disposition="skipped",
                        stage="reconciliation",
                        errors=("ALREADY_PLANNED_OR_REJECTED",),
                    )
                )
            continue
        retry_rows = [
            row
            for row in rows
            if (
                (match := _FINGERPRINT_MARKER.search(row)) is not None
                and match.group(1) == fingerprint
                and _retryable_host_blocked_row(row, plans_dir)
            )
        ]
        reserved_numbers = [
            number
            for row in retry_rows
            if (number := _row_number(row)) is not None
        ]
        rows = [row for row in rows if row not in retry_rows]
        if reserved_numbers:
            number = min(reserved_numbers)
        else:
            number = next_number
            next_number += 1
        slug = str(selection.get("slug") or "plan")
        trusted_title = str(finding.get("title") or "Selected finding")
        title = str(selection.get("title") or trusted_title)
        priority = str(selection.get("priority") or "P2")
        raw_dependencies = selection.get("dependencies")
        dependencies: list[Any] = (
            raw_dependencies if isinstance(raw_dependencies, list) else []
        )
        depends_on = [
            str(dependency.get("slug"))
            for dependency in dependencies
            if isinstance(dependency, dict)
        ]
        marker = f"<!-- fingerprint:{fingerprint} -->"
        if selection.get("error"):
            raw_errors = attempt.get("errors") if attempt is not None else None
            if not isinstance(raw_errors, (list, tuple)) and attempt is not None:
                legacy_code = attempt.get("transport_error_code")
                raw_errors = (legacy_code,) if isinstance(legacy_code, str) else ()
            error_codes = tuple(
                code
                for code in (
                    raw_errors if isinstance(raw_errors, (list, tuple)) else ()
                )
                if isinstance(code, str)
                and re.fullmatch(r"[A-Z][A-Z0-9_]{1,63}", code)
            )
            if not error_codes:
                error_codes = ("UNKNOWN",)
            transport_code = error_codes[0]
            rows.append(
                _blocked_index_row(
                    number=number,
                    marker=marker,
                    finding=finding,
                    status=f"BLOCKED (PLAN_WRITER_FAILED: {transport_code})",
                )
            )
            result["failed"].append(finding)
            result["diagnostics"].append(
                _attempt_diagnostic(
                    finding=finding,
                    attempt=attempt,
                    received=(
                        attempt.get("received_result")
                        if attempt is not None
                        else None
                    ),
                    disposition="blocked",
                    stage="transport",
                    errors=error_codes,
                )
            )
            continue

        plan_result = {
            key: value
            for key, value in selection.items()
            if key not in {"finding", "error"} and not key.startswith("_")
        }
        errors = validate_plan_result(
            plan_result,
            repo=plans_dir.parent,
            planned_at=planned_at,
            finding=finding,
            recon_commands=commands,
        )
        if not errors:
            if slug in depends_on:
                errors = ("DEPENDENCY_SELF_REFERENCE",)
            elif slug in cycle_slugs:
                errors = ("DEPENDENCY_CYCLE",)
            elif any(
                dependency not in candidate_slugs
                and dependency not in available_slugs
                for dependency in depends_on
            ):
                errors = ("DEPENDENCY_UNKNOWN",)
            elif any(dependency not in available_slugs for dependency in depends_on):
                errors = ("DEPENDENCY_UNAVAILABLE",)
        if not errors and not _head_matches(plans_dir.parent, planned_at):
            errors = ("PLAN_HEAD_CHANGED",)
        if errors:
            code_list = ",".join(errors)
            rows.append(
                _blocked_index_row(
                    number=number,
                    marker=marker,
                    finding=finding,
                    status=(
                        "BLOCKED (PLAN_VALIDATION_FAILED: "
                        f"{code_list})"
                    ),
                )
            )
            result["failed"].append(finding)
            result["diagnostics"].append(
                _attempt_diagnostic(
                    finding=finding,
                    attempt=attempt,
                    received=plan_result,
                    disposition="blocked",
                    stage=_validation_stage(errors),
                    errors=errors,
                )
            )
            continue

        filename = f"{number:03d}-{slug}.md"
        try:
            text = render_plan(
                finding,
                plan=plan_result,
                planned_at=planned_at,
                number=number,
                run_session_id=run_session_id,
            )
        except Exception:  # noqa: BLE001 - persist a safe render disposition
            rows.append(
                _blocked_index_row(
                    number=number,
                    marker=marker,
                    finding=finding,
                    status=(
                        "BLOCKED (PLAN_VALIDATION_FAILED: RENDER_FAILED)"
                    ),
                )
            )
            result["failed"].append(finding)
            result["diagnostics"].append(
                _attempt_diagnostic(
                    finding=finding,
                    attempt=attempt,
                    received=plan_result,
                    disposition="blocked",
                    stage="render",
                    errors=("RENDER_FAILED",),
                )
            )
            continue
        if not _head_matches(plans_dir.parent, planned_at):
            errors = ("PLAN_HEAD_CHANGED",)
            rows.append(
                _blocked_index_row(
                    number=number,
                    marker=marker,
                    finding=finding,
                    status=(
                        "BLOCKED (PLAN_VALIDATION_FAILED: "
                        f"{errors[0]})"
                    ),
                )
            )
            result["failed"].append(finding)
            result["diagnostics"].append(
                _attempt_diagnostic(
                    finding=finding,
                    attempt=attempt,
                    received=plan_result,
                    disposition="blocked",
                    stage=_validation_stage(errors),
                    errors=errors,
                )
            )
            continue
        (plans_dir / filename).write_text(text, encoding="utf-8")
        rows.append(
            f"| [{number:03d}]({filename}) {marker} | "
            f"{_markdown_cell(title)} | {_markdown_cell(priority)} | "
            f"{_markdown_cell(finding.get('effort'))} | "
            f"{_markdown_cell(', '.join(depends_on))} | TODO |"
        )
        if depends_on:
            new_dependency_notes.update(
                {
                    (number, dependency["slug"]): (
                        f"{number:03d} depends on {dependency['slug']} "
                        f"because {_markdown_cell(dependency['reason'])}"
                    )
                    for dependency in dependencies
                }
            )
        result["written"].append({**selection, "number": number, "path": filename})
        result["diagnostics"].append(
            _attempt_diagnostic(
                finding=finding,
                attempt=attempt,
                received=plan_result,
                disposition="success",
                stage="success",
                artifact={"path": filename, "status": "TODO"},
            )
        )
        fingerprints.add(fingerprint)
        available_slugs.add(slug)

    rows.sort(
        key=lambda row: (
            _row_number(row) is None,
            _row_number(row) or 0,
        )
    )
    dependency_notes = _merged_dependency_notes(
        rows,
        existing=existing_dependency_notes,
        new=new_dependency_notes,
        unstructured=unstructured_dependency_notes,
    )
    index_path.write_text(
        _render_index(
            rows,
            plans_dir=plans_dir,
            non_interactive_default=(
                non_interactive_default
                or "non-interactive default" in index_text.lower()
            ),
            dependency_notes=dependency_notes,
            run_session_id=run_session_id,
        ),
        encoding="utf-8",
    )
    return result
