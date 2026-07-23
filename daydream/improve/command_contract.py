"""Canonical structured repository-command contract and host validation."""

from __future__ import annotations

import json
import re
import shlex
from collections import Counter
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

_EVIDENCE_COMMON_PROPERTIES: dict[str, Any] = {
    "source_path": REPOSITORY_FILE_PATH_SCHEMA,
    "line_anchor": LINE_ANCHOR_SCHEMA,
    # Discovery excerpts are hints only. The host replaces this value with the
    # canonical source slice after resolving and bounds-checking the locator.
    "verbatim_excerpt": {"type": ["string", "null"]},
}


def _evidence_variant(
    kind: str,
    specific: dict[str, Any],
) -> dict[str, Any]:
    properties = {
        "kind": {"type": "string", "enum": [kind]},
        **_EVIDENCE_COMMON_PROPERTIES,
        **specific,
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
    }


LITERAL_COMMAND_EVIDENCE_SCHEMA = _evidence_variant(
    "literal-command",
    {},
)
MAKE_TARGET_EVIDENCE_SCHEMA = _evidence_variant(
    "make-target",
    {
        "target": {
            "type": "string",
            "minLength": 1,
            "maxLength": 160,
            "pattern": r"^[A-Za-z0-9][A-Za-z0-9_.-]*$",
        },
    },
)
PACKAGE_SCRIPT_EVIDENCE_SCHEMA = _evidence_variant(
    "package-script",
    {
        "package_manager": {
            "type": "string",
            "enum": ["npm", "pnpm", "yarn", "bun"],
        },
        "script": {
            "type": "string",
            "minLength": 1,
            "maxLength": 160,
            "pattern": r"^[A-Za-z0-9][A-Za-z0-9:._-]*$",
        },
        "working_directory": WORKING_DIRECTORY_SCHEMA,
    },
)
EVIDENCE_SCHEMA: dict[str, Any] = {
    "anyOf": [
        LITERAL_COMMAND_EVIDENCE_SCHEMA,
        MAKE_TARGET_EVIDENCE_SCHEMA,
        PACKAGE_SCRIPT_EVIDENCE_SCHEMA,
    ],
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
_MAKE_DECLARATION = re.compile(
    r"^(?P<targets>[A-Za-z0-9][A-Za-z0-9_.-]*"
    r"(?:\s+[A-Za-z0-9][A-Za-z0-9_.-]*)*)\s*:"
)


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
) -> tuple[str | None, str | None, ContractRejection | None]:
    source_path = evidence["source_path"]
    if (
        not valid_repository_file_path(source_path)
        or not path_is_confined(repo, source_path)
    ):
        return None, None, ContractRejection(
            "RECON_EVIDENCE_INVALID", "/evidence"
        )
    try:
        source = (repo / source_path).read_text(encoding="utf-8")
        start = evidence["line_anchor"]["start_line"]
        end = evidence["line_anchor"]["end_line"]
    except (OSError, UnicodeError, KeyError, TypeError):
        return None, None, ContractRejection(
            "RECON_EVIDENCE_INVALID", "/evidence"
        )
    lines = source.splitlines()
    if end < start or end > len(lines):
        return None, None, ContractRejection(
            "RECON_EVIDENCE_MISMATCH", "/evidence"
        )
    return source, "\n".join(lines[start - 1 : end]), None


def _package_invocation(package_manager: str, script: str) -> str:
    if package_manager == "npm":
        return f"npm run {script}"
    if package_manager == "bun":
        return f"bun run {script}"
    return f"{package_manager} {script}"


def _json_key_spans(source: str) -> dict[tuple[str, ...], list[tuple[int, int]]]:
    """Return exact source spans for object keys at each parsed JSON path."""
    decoder = json.JSONDecoder()
    spans: dict[tuple[str, ...], list[tuple[int, int]]] = {}

    def skip_whitespace(position: int) -> int:
        while position < len(source) and source[position].isspace():
            position += 1
        return position

    def parse_value(position: int, path: tuple[str, ...]) -> int:
        position = skip_whitespace(position)
        if position >= len(source):
            raise ValueError
        if source[position] == "{":
            position = skip_whitespace(position + 1)
            if position < len(source) and source[position] == "}":
                return position + 1
            while True:
                key_start = position
                key, key_end = decoder.raw_decode(source, position)
                if not isinstance(key, str) or source[position] != '"':
                    raise ValueError
                key_path = (*path, key)
                spans.setdefault(key_path, []).append((key_start, key_end))
                position = skip_whitespace(key_end)
                if position >= len(source) or source[position] != ":":
                    raise ValueError
                position = parse_value(position + 1, key_path)
                position = skip_whitespace(position)
                if position < len(source) and source[position] == "}":
                    return position + 1
                if position >= len(source) or source[position] != ",":
                    raise ValueError
                position = skip_whitespace(position + 1)
        if source[position] == "[":
            position = skip_whitespace(position + 1)
            if position < len(source) and source[position] == "]":
                return position + 1
            while True:
                position = parse_value(position, path)
                position = skip_whitespace(position)
                if position < len(source) and source[position] == "]":
                    return position + 1
                if position >= len(source) or source[position] != ",":
                    raise ValueError
                position = skip_whitespace(position + 1)
        _, end = decoder.raw_decode(source, position)
        return end

    end = skip_whitespace(parse_value(0, ()))
    if end != len(source):
        raise ValueError
    return spans


def _anchor_source_span(
    source: str,
    evidence: dict[str, Any],
) -> tuple[int, int]:
    lines = source.splitlines(keepends=True)
    start_line = evidence["line_anchor"]["start_line"]
    end_line = evidence["line_anchor"]["end_line"]
    return (
        sum(len(line) for line in lines[: start_line - 1]),
        sum(len(line) for line in lines[:end_line]),
    )


def _validate_evidence(
    command: dict[str, Any],
    *,
    repo: Path,
) -> tuple[dict[str, Any] | None, ContractRejection | None]:
    evidence = command["evidence"]
    source, excerpt, rejection = _source_slice(repo, evidence)
    if rejection is not None:
        return None, rejection
    assert source is not None and excerpt is not None
    canonical_evidence = {
        **evidence,
        "verbatim_excerpt": excerpt,
    }
    kind = evidence["kind"]
    if kind == "literal-command":
        if command["command"] not in excerpt:
            return None, ContractRejection(
                "RECON_EVIDENCE_MISMATCH",
                "/evidence",
            )
        return canonical_evidence, None

    if kind == "make-target":
        target = evidence["target"]
        expected_source = (
            "Makefile"
            if command["working_directory"] == "."
            else f"{command['working_directory']}/Makefile"
        )
        declarations = [
            match.group("targets").split()
            for line in excerpt.splitlines()
            if (match := _MAKE_DECLARATION.match(line))
        ]
        if (
            evidence["source_path"] != expected_source
            or target not in {item for group in declarations for item in group}
            or command["command"] != f"make {target}"
        ):
            return None, ContractRejection(
                "RECON_EVIDENCE_MISMATCH",
                "/evidence",
            )
        return canonical_evidence, None

    package_manager = evidence["package_manager"]
    script = evidence["script"]
    working_directory = evidence["working_directory"]
    expected_source = (
        "package.json"
        if working_directory == "."
        else f"{working_directory}/package.json"
    )
    try:
        manifest = json.loads(source)
        key_spans = _json_key_spans(source)
    except (json.JSONDecodeError, ValueError):
        return None, ContractRejection(
            "RECON_EVIDENCE_MISMATCH", "/evidence"
        )
    package_manager_declaration = manifest.get("packageManager")
    scripts = manifest.get("scripts")
    script_key_spans = key_spans.get(("scripts", script), [])
    anchor_start, anchor_end = _anchor_source_span(source, evidence)
    if (
        command["working_directory"] != working_directory
        or evidence["source_path"] != expected_source
        or not isinstance(package_manager_declaration, str)
        or not package_manager_declaration.startswith(f"{package_manager}@")
        or not isinstance(scripts, dict)
        or not isinstance(scripts.get(script), str)
        or len(key_spans.get(("scripts",), [])) != 1
        or len(script_key_spans) != 1
        or not (
            anchor_start <= script_key_spans[0][0]
            and script_key_spans[0][1] <= anchor_end
        )
        or command["command"] != _package_invocation(package_manager, script)
    ):
        return None, ContractRejection(
            "RECON_EVIDENCE_MISMATCH", "/evidence"
        )
    return canonical_evidence, None


def validate_recon_commands(
    recon: dict[str, Any],
    *,
    repo: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate recon records independently while preserving valid siblings."""
    commands = recon.get("commands")
    if not isinstance(commands, list):
        return [], ["RECON_COMMANDS_INVALID@/commands"]
    validated: list[dict[str, Any]] = []
    errors: list[str] = []
    ids = Counter(
        item.get("id")
        for item in commands
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    )
    validator = Draft202012Validator(RECON_COMMAND_SCHEMA)
    schema_codes = {
        "id": "RECON_COMMAND_ID_INVALID",
        "command": "RECON_MALFORMED_COMMAND",
        "working_directory": "RECON_WORKING_DIRECTORY_INVALID",
        "applicability": "RECON_APPLICABILITY_INVALID",
        "evidence": "RECON_EVIDENCE_INVALID",
    }

    def reject(index: int, rejection: ContractRejection) -> None:
        errors.append(rejection.render(f"/commands/{index}"))

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
    "validate_recon_commands",
]
