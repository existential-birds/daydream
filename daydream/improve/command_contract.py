"""Canonical structured repository-command contract and host validation."""

from __future__ import annotations

import re
import shlex
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from jsonschema import Draft202012Validator

REPOSITORY_FILE_PATH_PATTERN = (
    r"^(?!/)(?!.*(?:^|/)\.{1,2}(?:/|$))(?!.*\$\(|.*\$\{)"
    r"[A-Za-z0-9._+@$-]+(?:/[A-Za-z0-9._+@$-]+)*$"
)
DIRECTORY_SCOPE_PATTERN = (
    r"^(?!/)(?!.*(?:^|/)\.{1,2}(?:/|$))(?!.*\$\(|.*\$\{)"
    r"[A-Za-z0-9._+@$-]+(?:/[A-Za-z0-9._+@$-]+)*/?$"
)

REPOSITORY_FILE_PATH_SCHEMA: dict[str, Any] = {
    "type": "string",
    "minLength": 1,
    "maxLength": 512,
    "pattern": REPOSITORY_FILE_PATH_PATTERN,
}
DIRECTORY_SCOPE_SCHEMA: dict[str, Any] = {
    "type": "string",
    "minLength": 1,
    "maxLength": 512,
    "pattern": DIRECTORY_SCOPE_PATTERN,
}
WORKING_DIRECTORY_SCHEMA: dict[str, Any] = {
    "type": "string",
    "minLength": 1,
    "maxLength": 512,
    "pattern": rf"^(?:\.|{REPOSITORY_FILE_PATH_PATTERN[1:-1]})$",
}
LINE_ANCHOR_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["start_line", "end_line"],
    "properties": {
        "start_line": {"type": "integer", "minimum": 1},
        "end_line": {"type": "integer", "minimum": 1},
    },
}
EXPECTED_SUCCESS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["exit_code", "observable_result"],
    "properties": {
        "exit_code": {"type": "integer", "enum": [0]},
        "observable_result": {
            "type": "string",
            "minLength": 15,
            "maxLength": 500,
        },
    },
}

WHOLE_REPOSITORY_SCOPE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["kind"],
    "properties": {
        "kind": {"type": "string", "enum": ["whole-repository"]},
    },
}
IN_SCOPE_PATHS_SCOPE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["kind", "paths"],
    "properties": {
        "kind": {"type": "string", "enum": ["in-scope-paths"]},
        "paths": {
            "type": "array",
            "minItems": 1,
            "items": DIRECTORY_SCOPE_SCHEMA,
        },
    },
}
SCOPE_SCHEMA: dict[str, Any] = {
    "anyOf": [
        WHOLE_REPOSITORY_SCOPE_SCHEMA,
        IN_SCOPE_PATHS_SCOPE_SCHEMA,
    ],
}
APPLICABILITY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["scope", "preconditions", "rationale"],
    "properties": {
        "scope": SCOPE_SCHEMA,
        "preconditions": {
            "type": "array",
            "items": {
                "type": "string",
                "minLength": 5,
                "maxLength": 300,
            },
        },
        "rationale": {
            "type": "string",
            "minLength": 20,
            "maxLength": 500,
        },
    },
}

_EVIDENCE_PROPERTIES: dict[str, Any] = {
    "kind": {"type": "string", "enum": ["literal-command"]},
    "source_path": REPOSITORY_FILE_PATH_SCHEMA,
    "line_anchor": LINE_ANCHOR_SCHEMA,
    # Discovery excerpts are hints only. The host replaces this value with the
    # canonical source slice after resolving and bounds-checking the locator.
    "verbatim_excerpt": {"type": ["string", "null"]},
}
EVIDENCE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": list(_EVIDENCE_PROPERTIES),
    "properties": _EVIDENCE_PROPERTIES,
}

HOST_EVIDENCE_KIND = "host-derived"
# Make targets and manifest scripts are enumerated by the host, so their
# evidence is a declaration site, not a verbatim copy of the invocation. This
# schema is deliberately NOT reachable from the model-facing recon output
# schema: a model may only ever claim `literal-command`, which still has to
# survive the "command appears in the cited slice" check.
HOST_EVIDENCE_SCHEMA: dict[str, Any] = {
    **EVIDENCE_SCHEMA,
    "properties": {
        **_EVIDENCE_PROPERTIES,
        "kind": {"type": "string", "enum": [HOST_EVIDENCE_KIND]},
        "verbatim_excerpt": {"type": "string"},
    },
}

_COMMAND_PROPERTIES: dict[str, Any] = {
    "purpose": {"type": "string", "minLength": 10, "maxLength": 200},
    "command": {
        "type": "string",
        "minLength": 3,
        "maxLength": 1000,
        "pattern": r"^[^\x00-\x1f\x7f]+$",
    },
    "working_directory": WORKING_DIRECTORY_SCHEMA,
    "expected_success": EXPECTED_SUCCESS_SCHEMA,
    "applicability": APPLICABILITY_SCHEMA,
}
COMMAND_REF_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["recon_command_id", "appended_args", "note"],
    "properties": {
        "recon_command_id": {"type": "string", "minLength": 3, "maxLength": 80},
        # Focused argv suffix; null runs the recon command verbatim.
        "appended_args": {"type": ["string", "null"], "maxLength": 400},
        "note": {"type": ["string", "null"], "maxLength": 300},
    },
}
_OPTIONAL_COMMAND_REF_SCHEMA: dict[str, Any] = {
    **COMMAND_REF_SCHEMA,
    "type": ["object", "null"],
}
RECON_COMMAND_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["id", *_COMMAND_PROPERTIES, "evidence"],
    "properties": {
        "id": {
            "type": "string",
            "minLength": 3,
            "maxLength": 80,
            "pattern": r"^[a-z0-9]+(?:-[a-z0-9]+)*$",
        },
        **_COMMAND_PROPERTIES,
        "evidence": EVIDENCE_SCHEMA,
    },
}
HOST_RECON_COMMAND_SCHEMA: dict[str, Any] = {
    **RECON_COMMAND_SCHEMA,
    "properties": {
        **RECON_COMMAND_SCHEMA["properties"],
        "evidence": HOST_EVIDENCE_SCHEMA,
    },
}

_REPOSITORY_FILE_PATH = re.compile(REPOSITORY_FILE_PATH_PATTERN)
_DIRECTORY_SCOPE = re.compile(DIRECTORY_SCOPE_PATTERN)
_COMMAND_ARROW = re.compile(
    r"[\u2190-\u21ff\u27f0-\u27ff\u2900-\u297f]|->|=>"
)
_COMMAND_NARRATIVE_PREFIX = re.compile(
    r"(?i)^(?:run|execute|use|command|build|test|lint|ci|"
    r"continuous integration|pre-?push hook)\b[^;&|]*:\s+\S"
)
_SHELL_CONTROL_TOKEN = re.compile(r"^[|&;<>]+$")
_SHELL_SUBSTITUTION = re.compile(r"\$\(|`|[<>]\(")


@dataclass(frozen=True)
class ContractRejection:
    """Stable host-validation failure without rejected model prose."""

    code: str
    pointer: str

    def render(self, prefix: str) -> str:
        return f"{self.code}@{prefix}{self.pointer}"


def valid_repository_file_path(value: str) -> bool:
    """Return whether ``value`` has the safe repository-file grammar."""
    return bool(_REPOSITORY_FILE_PATH.fullmatch(value))


def valid_directory_scope_lexical(value: str) -> bool:
    """Return whether ``value`` is a safe file-or-directory scope."""
    return bool(_DIRECTORY_SCOPE.fullmatch(value))


def path_is_confined(
    repo: Path,
    value: str,
    *,
    directory_scope: bool = False,
) -> bool:
    """Return whether a lexical repository path crosses no symlink/root edge."""
    validator = (
        valid_directory_scope_lexical
        if directory_scope
        else valid_repository_file_path
    )
    if value != "." and not validator(value):
        return False
    root = repo.resolve()
    candidate = repo
    if value != ".":
        for part in PurePosixPath(value.rstrip("/")).parts:
            candidate /= part
            try:
                if candidate.is_symlink():
                    return False
                if not candidate.exists():
                    break
            except OSError:
                return False
    try:
        return candidate.resolve(strict=False).is_relative_to(root)
    except OSError:
        return False


def canonicalize_directory_scope(value: str) -> str:
    """Canonicalize the sole lossless scope spelling difference."""
    return value.rstrip("/")


def _json_pointer(parts: list[object]) -> str:
    if not parts:
        return ""
    return "".join(
        f"/{str(part).replace('~', '~0').replace('/', '~1')}"
        for part in parts
    )


def _schema_rejection(
    value: Any,
    schema: dict[str, Any],
    *,
    code: str,
) -> ContractRejection | None:
    errors = sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda error: repr(list(error.absolute_path)),
    )
    if not errors:
        return None
    error = _most_specific_schema_error(errors[0])
    path = list(error.absolute_path)
    if error.validator == "required" and isinstance(error.instance, dict):
        missing = sorted(set(error.validator_value) - set(error.instance))
        if missing:
            path.append(missing[0])
    return ContractRejection(code, _json_pointer(path))


def _most_specific_schema_error(error: Any) -> Any:
    """Select the deepest actionable leaf beneath combinator failures."""
    leaves = [
        _most_specific_schema_error(child)
        for child in error.context
    ]
    if not leaves:
        return error
    return max(
        leaves,
        key=lambda child: (
            len(list(child.absolute_path)),
            child.validator == "pattern",
            child.validator == "const",
        ),
    )


def validate_applicability(
    applicability: Any,
    *,
    repo: Path,
) -> tuple[dict[str, Any] | None, ContractRejection | None]:
    """Validate and lexically canonicalize command applicability."""
    schema_rejection = _schema_rejection(
        applicability,
        APPLICABILITY_SCHEMA,
        code="RECON_APPLICABILITY_INVALID",
    )
    if schema_rejection is not None:
        return None, schema_rejection
    assert isinstance(applicability, dict)
    scope = applicability["scope"]
    if scope["kind"] == "whole-repository":
        return dict(applicability), None

    paths = scope["paths"]
    for index, path in enumerate(paths):
        if (
            not valid_directory_scope_lexical(path)
            or not path_is_confined(repo, path, directory_scope=True)
            or not (repo / path.rstrip("/")).exists()
        ):
            return None, ContractRejection(
                "RECON_APPLICABILITY_INVALID",
                f"/scope/paths/{index}",
            )
    normalized = {
        **applicability,
        "scope": {
            **scope,
            "paths": [
                canonicalize_directory_scope(path)
                for path in paths
            ],
        },
    }
    if len(set(normalized["scope"]["paths"])) != len(paths):
        return None, ContractRejection(
            "RECON_APPLICABILITY_INVALID",
            "/scope/paths",
        )
    return normalized, None


def command_argv(literal: str) -> list[str] | None:
    try:
        argv = shlex.split(literal)
    except ValueError:
        return None
    return argv or None


def _command_syntax_tokens(literal: str) -> list[str] | None:
    try:
        lexer = shlex.shlex(
            literal,
            posix=True,
            punctuation_chars="()|&;<>",
        )
        lexer.whitespace_split = True
        lexer.commenters = ""
        return list(lexer)
    except ValueError:
        return None


def has_shell_composition(literal: str) -> bool:
    if _SHELL_SUBSTITUTION.search(literal):
        return True
    tokens = _command_syntax_tokens(literal)
    return tokens is None or any(
        _SHELL_CONTROL_TOKEN.fullmatch(token)
        for token in tokens
    )


def literal_command_error(
    literal: Any,
    *,
    allow_shell_composition: bool = False,
) -> str | None:
    """Return a stable error for non-literal or malformed command text."""
    if not isinstance(literal, str):
        return "MALFORMED_COMMAND"
    argv = command_argv(literal)
    if (
        not literal.strip()
        or literal != literal.strip()
        or any(ord(char) < 32 or ord(char) == 127 for char in literal)
        or _COMMAND_ARROW.search(literal)
        or _COMMAND_NARRATIVE_PREFIX.search(literal)
        or argv is None
    ):
        return "MALFORMED_COMMAND"
    syntax_tokens = _command_syntax_tokens(literal)
    has_unquoted_parentheses = (
        syntax_tokens is not None
        and any(token in {"(", ")"} for token in syntax_tokens)
    )
    if (
        syntax_tokens is None
        or (
            has_unquoted_parentheses
            and not (
                allow_shell_composition
                and _SHELL_SUBSTITUTION.search(literal)
            )
        )
        or any(
            token.casefold() in {"...", "todo", "tbd", "${todo}"}
            for token in argv
        )
        or (
            not allow_shell_composition
            and has_shell_composition(literal)
        )
    ):
        return "MALFORMED_COMMAND"
    first = argv[0]
    if (
        first.startswith(("-", "$", "#"))
        or first.endswith(":")
        or ":" in first
    ):
        return "MALFORMED_COMMAND"
    return None


def _source_slice(
    repo: Path,
    evidence: dict[str, Any],
) -> tuple[str | None, ContractRejection | None]:
    source_path = evidence["source_path"]
    if (
        not valid_repository_file_path(source_path)
        or not path_is_confined(repo, source_path)
    ):
        return None, ContractRejection("RECON_EVIDENCE_INVALID", "/evidence")
    try:
        source = (repo / source_path).read_text(encoding="utf-8")
        start = evidence["line_anchor"]["start_line"]
        end = evidence["line_anchor"]["end_line"]
    except (OSError, UnicodeError, KeyError, TypeError):
        return None, ContractRejection("RECON_EVIDENCE_INVALID", "/evidence")
    lines = source.splitlines()
    if end < start or end > len(lines):
        return None, ContractRejection("RECON_EVIDENCE_MISMATCH", "/evidence")
    return "\n".join(lines[start - 1 : end]), None


def _validate_evidence(
    command: dict[str, Any],
    *,
    repo: Path,
    require_command_in_excerpt: bool,
) -> tuple[dict[str, Any] | None, ContractRejection | None]:
    evidence = command["evidence"]
    excerpt, rejection = _source_slice(repo, evidence)
    if rejection is not None:
        return None, rejection
    assert excerpt is not None
    if require_command_in_excerpt and command["command"] not in excerpt:
        return None, ContractRejection(
            "RECON_EVIDENCE_MISMATCH",
            "/evidence",
        )
    return {**evidence, "verbatim_excerpt": excerpt}, None


def validate_recon_commands(
    recon: dict[str, Any],
    *,
    repo: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate recon records independently while preserving valid siblings."""
    commands = recon.get("commands")
    if not isinstance(commands, list):
        return [], ["RECON_COMMANDS_INVALID@/commands"]
    return _validate_command_records(
        commands,
        repo=repo,
        schema=RECON_COMMAND_SCHEMA,
        pointer_prefix="/commands",
        require_command_in_excerpt=True,
    )


def validate_host_commands(
    commands: Sequence[Any],
    *,
    repo: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate host-enumerated records against the same safety gates.

    Host records are constructed, not cited, so the invocation is not required
    to appear in the declaration slice. Everything else — the id grammar, the
    literal-command gate that rejects shell composition, working-directory and
    evidence-path confinement, and the excerpt re-read — is identical.
    """
    return _validate_command_records(
        list(commands),
        repo=repo,
        schema=HOST_RECON_COMMAND_SCHEMA,
        pointer_prefix="/host_commands",
        require_command_in_excerpt=False,
    )


def _validate_command_records(
    commands: list[Any],
    *,
    repo: Path,
    schema: dict[str, Any],
    pointer_prefix: str,
    require_command_in_excerpt: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    validated: list[dict[str, Any]] = []
    errors: list[str] = []
    ids = Counter(
        item.get("id")
        for item in commands
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    )
    validator = Draft202012Validator(schema)
    schema_codes = {
        "id": "RECON_COMMAND_ID_INVALID",
        "command": "RECON_MALFORMED_COMMAND",
        "working_directory": "RECON_WORKING_DIRECTORY_INVALID",
        "applicability": "RECON_APPLICABILITY_INVALID",
        "evidence": "RECON_EVIDENCE_INVALID",
    }

    def reject(index: int, rejection: ContractRejection) -> None:
        errors.append(rejection.render(f"{pointer_prefix}/{index}"))

    for index, command in enumerate(commands):
        candidate = command
        if isinstance(command, dict):
            evidence = command.get("evidence")
            if (
                isinstance(evidence, dict)
                and not isinstance(evidence.get("verbatim_excerpt"), str)
            ):
                candidate = {
                    **command,
                    "evidence": {
                        **evidence,
                        "verbatim_excerpt": None,
                    },
                }
        record_errors = sorted(
            validator.iter_errors(candidate),
            key=lambda error: repr(list(error.absolute_path)),
        )
        if record_errors:
            error = _most_specific_schema_error(record_errors[0])
            path = list(error.absolute_path)
            code = schema_codes.get(
                path[0] if path else "",
                "RECON_COMMANDS_INVALID",
            )
            reject(index, ContractRejection(code, _json_pointer(path)))
            continue
        assert isinstance(candidate, dict)
        command = candidate
        if literal_command_error(command["command"]):
            reject(
                index,
                ContractRejection(
                    "RECON_MALFORMED_COMMAND",
                    "/command",
                ),
            )
            continue
        if ids[command["id"]] > 1:
            reject(
                index,
                ContractRejection("RECON_COMMAND_ID_INVALID", "/id"),
            )
            continue
        working_directory = command["working_directory"]
        if (
            (
                working_directory != "."
                and not valid_repository_file_path(working_directory)
            )
            or not path_is_confined(repo, working_directory)
            or not (repo / working_directory).is_dir()
        ):
            reject(
                index,
                ContractRejection(
                    "RECON_WORKING_DIRECTORY_INVALID",
                    "/working_directory",
                ),
            )
            continue
        applicability, applicability_rejection = validate_applicability(
            command["applicability"],
            repo=repo,
        )
        if applicability_rejection is not None:
            reject(
                index,
                ContractRejection(
                    applicability_rejection.code,
                    f"/applicability{applicability_rejection.pointer}",
                ),
            )
            continue
        canonical_evidence, evidence_rejection = _validate_evidence(
            command,
            repo=repo,
            require_command_in_excerpt=require_command_in_excerpt,
        )
        if evidence_rejection is not None:
            reject(index, evidence_rejection)
            continue
        assert applicability is not None and canonical_evidence is not None
        validated.append(
            {
                **command,
                "applicability": applicability,
                "evidence": canonical_evidence,
            }
        )
    return validated, errors


__all__ = [
    "APPLICABILITY_SCHEMA",
    "COMMAND_REF_SCHEMA",
    "DIRECTORY_SCOPE_PATTERN",
    "DIRECTORY_SCOPE_SCHEMA",
    "EVIDENCE_SCHEMA",
    "EXPECTED_SUCCESS_SCHEMA",
    "HOST_EVIDENCE_KIND",
    "HOST_EVIDENCE_SCHEMA",
    "HOST_RECON_COMMAND_SCHEMA",
    "RECON_COMMAND_SCHEMA",
    "REPOSITORY_FILE_PATH_PATTERN",
    "REPOSITORY_FILE_PATH_SCHEMA",
    "SCOPE_SCHEMA",
    "canonicalize_directory_scope",
    "command_argv",
    "has_shell_composition",
    "literal_command_error",
    "path_is_confined",
    "valid_directory_scope_lexical",
    "valid_repository_file_path",
    "validate_applicability",
    "validate_host_commands",
    "validate_recon_commands",
]
