"""Deterministic host assembly of model-authored improve plans.

The model authors judgment content only (``PLAN_AUTHOR_SCHEMA``); this module
normalizes it, collects every authoring issue at once, and expands the result
into the assembled plan shape (``PLAN_WRITER_SCHEMA``) that ``render_plan`` and
``write_plans`` already consume. Assembly is pure with respect to the model
output: filesystem reads only, no randomness, no wall-clock reads.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from copy import deepcopy
from dataclasses import dataclass
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
    path_is_confined as _path_is_confined,
)
from daydream.improve.command_contract import (
    valid_directory_scope_lexical as _valid_directory_scope,
)
from daydream.improve.command_contract import (
    valid_repository_file_path as _valid_repository_file_path,
)
from daydream.improve.plans import _clamp_node, redact_secret_values
from daydream.improve.prompts import PLAN_AUTHOR_SCHEMA

GIT_PUSH_POLICY = "never-without-operator-instruction"
GIT_PULL_REQUEST_POLICY = "never-without-operator-instruction"
STOP_REQUIRED_ACTION = "STOP_AND_REPORT"
# Sha-free by design: the Status section renders the planned-at commit, so
# assembly stays git-free.
GIT_BRANCH_BASIS = (
    "Branch from the operator's current checkout. HEAD is expected to have "
    "moved past the planned-at commit; see Before you start."
)

_PLACEHOLDER_ARG_TOKENS = {"...", "todo", "tbd", "${todo}"}


@dataclass(frozen=True)
class AssemblyIssue:
    """One authoring defect, fully addressed and actionable."""

    code: str
    pointer: str
    detail: str | None = None
    hint: str | None = None


def render_issue(issue: AssemblyIssue) -> str:
    """Render ``CODE@/pointer#detail`` for the existing diagnostics plumbing."""
    rendered = f"{issue.code}@{issue.pointer}"
    if issue.detail:
        rendered += f"#{issue.detail}"
    return rendered


def _branch_name(slug: str) -> str:
    return f"improve/{slug}"


_AUTHOR_PROSE_FIELD_PATTERNS: tuple[tuple[str, ...], ...] = (
    ("dependencies", "*", "reason"),
    ("why_this_matters", "problem"),
    ("why_this_matters", "concrete_cost"),
    ("why_this_matters", "intended_outcome"),
    ("scope", "existing_paths", "*", "role"),
    ("scope", "new_paths", "*", "role"),
    ("scope", "out_of_scope_paths", "*", "reason"),
    ("scope", "out_of_scope_behaviors", "*", "behavior"),
    ("scope", "out_of_scope_behaviors", "*", "reason"),
    ("context_excerpts", "*", "file_role"),
    ("git_workflow", "commit_boundaries"),
    ("git_workflow", "commit_message_example"),
    ("steps", "*", "title"),
    # ``instruction`` and ``target_state`` are deliberately absent: they are the
    # executable payload, not render-only prose. Clamping them cut real plans
    # off mid-sentence ("...currently saying it …"), handing the executor an
    # unfinished order. An over-length one is now an authoring issue the model
    # repairs by splitting the change, never a silent truncation.
    ("steps", "*", "verification", "note"),
    ("test_plan", "exemplars", "*", "pattern_to_copy"),
    ("test_plan", "cases", "*", "name"),
    ("test_plan", "cases", "*", "setup"),
    ("test_plan", "cases", "*", "action"),
    ("test_plan", "cases", "*", "assertions", "*"),
    ("test_plan", "cases", "*", "verification", "note"),
    ("done_criteria", "*", "description"),
    ("done_criteria", "*", "verification", "note"),
    ("false_assumption", "condition"),
    ("false_assumption", "evidence_to_report"),
    ("additional_stop_conditions", "*", "condition"),
    ("additional_stop_conditions", "*", "evidence_to_report"),
    ("additional_command_refs", "*", "note"),
    ("maintenance_notes", "future_interactions", "*", "area"),
    ("maintenance_notes", "future_interactions", "*", "note"),
    ("maintenance_notes", "review_risks", "*", "risk"),
    ("maintenance_notes", "review_risks", "*", "review_check"),
    ("maintenance_notes", "deferred_items", "*", "item"),
    ("maintenance_notes", "deferred_items", "*", "reason"),
    ("maintenance_notes", "deferred_items", "*", "revisit_trigger"),
)


def _author_schema_max_length(pattern: tuple[str, ...]) -> int:
    node: dict[str, Any] = PLAN_AUTHOR_SCHEMA
    for segment in pattern:
        node = node["items"] if segment == "*" else node["properties"][segment]
    return int(node["maxLength"])


_AUTHOR_PROSE_CLAMP_LIMITS: tuple[tuple[tuple[str, ...], int], ...] = tuple(
    (pattern, _author_schema_max_length(pattern))
    for pattern in _AUTHOR_PROSE_FIELD_PATTERNS
)


def _strip_unknown(value: Any, schema: dict[str, Any]) -> Any:
    if isinstance(value, dict) and "properties" in schema:
        return {
            key: _strip_unknown(value[key], sub_schema)
            for key, sub_schema in schema["properties"].items()
            if key in value
        }
    if isinstance(value, list) and "items" in schema:
        return [_strip_unknown(item, schema["items"]) for item in value]
    return value


def _redact_strings(value: Any) -> Any:
    if isinstance(value, str):
        return redact_secret_values(value)
    if isinstance(value, list):
        return [_redact_strings(item) for item in value]
    if isinstance(value, dict):
        return {key: _redact_strings(item) for key, item in value.items()}
    return value


def _exists_on_disk(repo: Path, path: str) -> bool:
    try:
        return (repo / path).is_file()
    except (OSError, ValueError):
        return False


def _read_repo_file(repo: Path, path: str) -> str | None:
    try:
        return (repo / path).read_text(encoding="utf-8")
    except (OSError, UnicodeError, ValueError):
        return None


def _entry_paths(entries: Any) -> list[str]:
    if not isinstance(entries, list):
        return []
    return [
        entry["path"]
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    ]


def _dedup_scope(normalized: dict[str, Any], *, repo: Path) -> None:
    scope = normalized.get("scope")
    if not isinstance(scope, dict):
        return
    for list_name in ("existing_paths", "new_paths", "out_of_scope_paths"):
        entries = scope.get(list_name)
        if not isinstance(entries, list):
            continue
        seen: set[str] = set()
        kept: list[Any] = []
        for entry in entries:
            path = entry.get("path") if isinstance(entry, dict) else None
            if isinstance(path, str):
                if path in seen:
                    continue
                seen.add(path)
            kept.append(entry)
        scope[list_name] = kept
    existing = scope.get("existing_paths")
    new = scope.get("new_paths")
    if isinstance(existing, list) and isinstance(new, list):
        conflicts = set(_entry_paths(existing)) & set(_entry_paths(new))
        for path in conflicts:
            list_name = "new_paths" if _exists_on_disk(repo, path) else "existing_paths"
            scope[list_name] = [
                entry
                for entry in scope[list_name]
                if not (isinstance(entry, dict) and entry.get("path") == path)
            ]
    in_scope = set(_entry_paths(scope.get("existing_paths"))) | set(
        _entry_paths(scope.get("new_paths"))
    )
    out_entries = scope.get("out_of_scope_paths")
    if isinstance(out_entries, list):
        scope["out_of_scope_paths"] = [
            entry
            for entry in out_entries
            if not (
                isinstance(entry, dict)
                and isinstance(entry.get("path"), str)
                and entry["path"].rstrip("/") in in_scope
            )
        ]


def _drop_self_dependencies(normalized: dict[str, Any]) -> None:
    dependencies = normalized.get("dependencies")
    if not isinstance(dependencies, list):
        return
    slug = normalized.get("slug")
    normalized["dependencies"] = [
        dependency
        for dependency in dependencies
        if not (isinstance(dependency, dict) and dependency.get("slug") == slug)
    ]


def _clamp_anchor(anchor: Any, line_count: int) -> None:
    if not isinstance(anchor, dict):
        return
    start = anchor.get("start_line")
    end = anchor.get("end_line")
    if (
        isinstance(start, int)
        and isinstance(end, int)
        and 1 <= start <= line_count
        and end > line_count
    ):
        anchor["end_line"] = line_count


def _clamp_excerpt_end_lines(normalized: dict[str, Any], *, repo: Path) -> None:
    line_counts: dict[str, int | None] = {}

    def count_for(path: Any) -> int | None:
        if not isinstance(path, str):
            return None
        if path not in line_counts:
            source = _read_repo_file(repo, path)
            line_counts[path] = (
                len(source.splitlines()) if source is not None else None
            )
        return line_counts[path]

    scope = normalized.get("scope")
    entries = scope.get("existing_paths") if isinstance(scope, dict) else None
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        count = count_for(entry.get("path"))
        if count is None:
            continue
        excerpts = entry.get("excerpts")
        for anchor in excerpts if isinstance(excerpts, list) else []:
            _clamp_anchor(anchor, count)
    context = normalized.get("context_excerpts")
    for entry in context if isinstance(context, list) else []:
        if not isinstance(entry, dict):
            continue
        count = count_for(entry.get("path"))
        if count is not None:
            _clamp_anchor(entry, count)


def _normalize_authored(
    authored: Any,
    *,
    repo: Path,
) -> dict[str, Any] | None:
    """Apply the deterministic category-c repairs; None if not an object."""
    if not isinstance(authored, dict):
        return None
    normalized = _strip_unknown(authored, PLAN_AUTHOR_SCHEMA)
    normalized = _redact_strings(normalized)
    for pattern, limit in _AUTHOR_PROSE_CLAMP_LIMITS:
        _clamp_node(normalized, pattern, limit)
    _dedup_scope(normalized, repo=repo)
    _drop_self_dependencies(normalized)
    _clamp_excerpt_end_lines(normalized, repo=repo)
    return normalized


def _json_pointer(parts: Sequence[str]) -> str:
    if not parts:
        return "/"
    return "".join(
        f"/{part.replace('~', '~0').replace('/', '~1')}" for part in parts
    )


_LENGTH_PHRASINGS = {
    "maxLength": "at most {limit} characters (it has {actual})",
    "minLength": "at least {limit} characters (it has {actual})",
    "maxItems": "at most {limit} items (it has {actual})",
    "minItems": "at least {limit} items (it has {actual})",
}


def _length_hint(validator: str, limit: Any, actual: int) -> str:
    return (
        "Rewrite the value at this pointer to "
        + _LENGTH_PHRASINGS[validator].format(limit=limit, actual=actual)
        + "; keep every other field unchanged in meaning."
    )


_AddIssue = Callable[..., None]


def _schema_issues(normalized: dict[str, Any], add: _AddIssue) -> None:
    errors = sorted(
        Draft202012Validator(PLAN_AUTHOR_SCHEMA).iter_errors(normalized),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    for error in errors:
        parts = [str(part) for part in error.absolute_path]
        if error.validator == "required" and isinstance(error.instance, dict):
            missing = sorted(set(error.validator_value) - set(error.instance))
            for key in missing:
                add("AUTHOR_SCHEMA_INVALID", _json_pointer([*parts, key]))
            continue
        detail = None
        hint = None
        if error.validator in _LENGTH_PHRASINGS and isinstance(
            error.instance, (str, list)
        ):
            actual = len(error.instance)
            detail = f"{error.validator}={error.validator_value};actual={actual}"
            hint = _length_hint(error.validator, error.validator_value, actual)
        elif error.validator == "enum":
            hint = "valid values: " + ", ".join(
                str(value) for value in error.validator_value
            )
        add("AUTHOR_SCHEMA_INVALID", _json_pointer(parts), detail, hint)


def _covers(prefix: str, target: str) -> bool:
    stripped = prefix.rstrip("/")
    return target == stripped or target.startswith(f"{stripped}/")


def _scope_paths_of(command: dict[str, Any]) -> list[str] | None:
    """Return the command's scope paths, or None for whole-repository."""
    applicability = command.get("applicability")
    scope = applicability.get("scope") if isinstance(applicability, dict) else None
    if not isinstance(scope, dict) or scope.get("kind") != "in-scope-paths":
        return None
    return [path for path in scope.get("paths", []) if isinstance(path, str)]


def _command_covers_all(command: dict[str, Any], targets: Sequence[str]) -> bool:
    scope_paths = _scope_paths_of(command)
    if scope_paths is None:
        return True
    return all(
        any(_covers(path, target) for path in scope_paths) for target in targets
    )


def _command_scope_fits_plan(
    command: dict[str, Any], in_scope: Sequence[str]
) -> bool:
    """Mirror the self-check: every scope path covers >=1 in-scope plan path."""
    scope_paths = _scope_paths_of(command)
    if scope_paths is None:
        return True
    return all(
        any(_covers(path, target) for target in in_scope)
        for path in scope_paths
    )


def _appended_args_invalid(appended: str) -> bool:
    if not appended or appended != appended.strip():
        return True
    if any(ord(char) < 32 or ord(char) == 127 for char in appended):
        return True
    if "${" in appended or _has_shell_composition(appended):
        return True
    argv = _command_argv(appended)
    if argv is None:
        return True
    return any(token.casefold() in _PLACEHOLDER_ARG_TOKENS for token in argv)


def _iter_command_refs(
    normalized: dict[str, Any],
) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield (pointer, ref) pairs in first-use document order."""
    steps = normalized.get("steps")
    for index, step in enumerate(steps if isinstance(steps, list) else []):
        if isinstance(step, dict) and isinstance(step.get("verification"), dict):
            yield f"/steps/{index}/verification", step["verification"]
    test_plan = normalized.get("test_plan")
    cases = test_plan.get("cases") if isinstance(test_plan, dict) else None
    for index, case in enumerate(cases if isinstance(cases, list) else []):
        if isinstance(case, dict) and isinstance(case.get("verification"), dict):
            yield f"/test_plan/cases/{index}/verification", case["verification"]
    criteria = normalized.get("done_criteria")
    for index, criterion in enumerate(
        criteria if isinstance(criteria, list) else []
    ):
        if isinstance(criterion, dict) and isinstance(
            criterion.get("verification"), dict
        ):
            yield f"/done_criteria/{index}/verification", criterion["verification"]
    extra = normalized.get("additional_command_refs")
    for index, ref in enumerate(extra if isinstance(extra, list) else []):
        if isinstance(ref, dict):
            yield f"/additional_command_refs/{index}", ref


def _collect_issues(
    normalized: dict[str, Any],
    *,
    repo: Path,
    recon_by_id: dict[str, dict[str, Any]],
) -> list[AssemblyIssue]:
    issues: list[AssemblyIssue] = []
    seen: set[tuple[str, str, str | None]] = set()

    def add(
        code: str,
        pointer: str,
        detail: str | None = None,
        hint: str | None = None,
    ) -> None:
        key = (code, pointer, detail)
        if key in seen:
            return
        seen.add(key)
        issues.append(AssemblyIssue(code, pointer, detail, hint))

    def check_path(pointer: str, value: Any, *, directory: bool = False) -> bool:
        if not isinstance(value, str):
            return False
        validator = (
            _valid_directory_scope if directory else _valid_repository_file_path
        )
        if not validator(value):
            add("MALFORMED_PATH", pointer)
            return False
        if not _path_is_confined(repo, value, directory_scope=directory):
            add("PATH_OUTSIDE_REPOSITORY", pointer)
            return False
        return True

    _schema_issues(normalized, add)

    scope = normalized.get("scope")
    scope = scope if isinstance(scope, dict) else {}
    existing_entries = scope.get("existing_paths")
    existing_entries = existing_entries if isinstance(existing_entries, list) else []
    new_entries = scope.get("new_paths")
    new_entries = new_entries if isinstance(new_entries, list) else []
    out_entries = scope.get("out_of_scope_paths")
    out_entries = out_entries if isinstance(out_entries, list) else []
    existing_paths = _entry_paths(existing_entries)
    new_paths = _entry_paths(new_entries)
    excluded_paths = _entry_paths(out_entries)
    in_scope = set(existing_paths) | set(new_paths)
    lexical_in_scope = [
        path
        for path in [*existing_paths, *new_paths]
        if _valid_repository_file_path(path)
    ]
    in_scope_hint = (
        "declared in-scope paths: " + ", ".join(lexical_in_scope)
        if lexical_in_scope
        else "declare the path in scope.existing_paths or scope.new_paths first"
    )

    if not existing_entries and not new_entries:
        add("EMPTY_SCOPE", "/scope")

    for index, entry in enumerate(existing_entries):
        if not isinstance(entry, dict):
            continue
        pointer = f"/scope/existing_paths/{index}/path"
        path = entry.get("path")
        if not isinstance(path, str) or not check_path(pointer, path):
            continue
        source = _read_repo_file(repo, path)
        if source is None:
            add("EXISTING_PATH_MISSING", pointer)
            continue
        line_count = len(source.splitlines())
        excerpts = entry.get("excerpts")
        for anchor_index, anchor in enumerate(
            excerpts if isinstance(excerpts, list) else []
        ):
            if not isinstance(anchor, dict):
                continue
            start = anchor.get("start_line")
            end = anchor.get("end_line")
            if not isinstance(start, int) or not isinstance(end, int):
                continue
            if start > line_count or end < start:
                add(
                    "EXCERPT_ANCHOR_INVALID",
                    f"/scope/existing_paths/{index}/excerpts/{anchor_index}",
                    f"lines={line_count}",
                )
    for index, entry in enumerate(new_entries):
        if not isinstance(entry, dict):
            continue
        pointer = f"/scope/new_paths/{index}/path"
        path = entry.get("path")
        if not isinstance(path, str) or not check_path(pointer, path):
            continue
        try:
            path_exists = (repo / path).exists()
        except (OSError, ValueError):
            path_exists = False
        if path_exists:
            add(
                "NEW_PATH_ALREADY_EXISTS",
                pointer,
                hint="declare it under existing_paths instead",
            )
    for index, entry in enumerate(out_entries):
        if isinstance(entry, dict):
            check_path(
                f"/scope/out_of_scope_paths/{index}/path",
                entry.get("path"),
                directory=True,
            )

    context = normalized.get("context_excerpts")
    for index, entry in enumerate(context if isinstance(context, list) else []):
        if not isinstance(entry, dict):
            continue
        pointer = f"/context_excerpts/{index}"
        path = entry.get("path")
        if not isinstance(path, str) or not check_path(f"{pointer}/path", path):
            continue
        source = _read_repo_file(repo, path)
        if source is None:
            add("EXCERPT_PATH_MISSING", f"{pointer}/path")
            continue
        line_count = len(source.splitlines())
        start = entry.get("start_line")
        end = entry.get("end_line")
        if (
            isinstance(start, int)
            and isinstance(end, int)
            and (start > line_count or end < start)
        ):
            add("EXCERPT_ANCHOR_INVALID", pointer, f"lines={line_count}")

    recon_ids_hint = (
        "valid recon command ids: " + ", ".join(recon_by_id)
        if recon_by_id
        else "no verified recon commands exist; use null verification"
    )
    fitting_ids = [
        recon_id
        for recon_id, command in recon_by_id.items()
        if _command_scope_fits_plan(command, sorted(in_scope))
    ]

    def check_ref(pointer: str, ref: Any) -> dict[str, Any] | None:
        if not isinstance(ref, dict):
            return None
        recon_id = ref.get("recon_command_id")
        base = recon_by_id.get(recon_id) if isinstance(recon_id, str) else None
        if isinstance(recon_id, str) and base is None:
            add(
                "RECON_COMMAND_UNKNOWN",
                f"{pointer}/recon_command_id",
                hint=recon_ids_hint,
            )
        appended = ref.get("appended_args")
        if isinstance(appended, str) and _appended_args_invalid(appended):
            add("MALFORMED_APPENDED_ARGS", f"{pointer}/appended_args")
        if base is not None and not _command_scope_fits_plan(
            base, sorted(in_scope)
        ):
            add(
                "COMMAND_SCOPE_MISMATCH",
                pointer,
                hint=(
                    "recon commands whose scope fits this plan: "
                    + (", ".join(fitting_ids) or "none")
                ),
            )
        return base

    steps = normalized.get("steps")
    steps = steps if isinstance(steps, list) else []
    for step_index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
        changed_paths: list[str] = []
        changes = step.get("changes")
        for change_index, change in enumerate(
            changes if isinstance(changes, list) else []
        ):
            if not isinstance(change, dict):
                continue
            pointer = f"/steps/{step_index}/changes/{change_index}/path"
            path = change.get("path")
            if not isinstance(path, str) or not check_path(pointer, path):
                continue
            changed_paths.append(path)
            if path not in in_scope:
                add("STEP_PATH_OUT_OF_SCOPE", pointer, hint=in_scope_hint)
            elif change.get("operation") == "create" and path not in new_paths:
                add("CREATE_PATH_NOT_NEW", pointer)
            elif (
                change.get("operation") != "create"
                and path not in existing_paths
            ):
                add("CHANGE_PATH_NOT_EXISTING", pointer)
        verification = step.get("verification")
        if isinstance(verification, dict):
            gate_pointer = f"/steps/{step_index}/verification"
            base = check_ref(gate_pointer, verification)
            if base is not None and not _command_covers_all(base, changed_paths):
                covering = [
                    recon_id
                    for recon_id, command in recon_by_id.items()
                    if _command_covers_all(command, changed_paths)
                ]
                add(
                    "STEP_GATE_SCOPE_MISMATCH",
                    gate_pointer,
                    hint=(
                        "recon commands covering this step's paths: "
                        + (", ".join(covering) or "none")
                    ),
                )

    test_plan = normalized.get("test_plan")
    test_plan = test_plan if isinstance(test_plan, dict) else {}
    exemplars = test_plan.get("exemplars")
    for index, exemplar in enumerate(
        exemplars if isinstance(exemplars, list) else []
    ):
        if not isinstance(exemplar, dict):
            continue
        pointer = f"/test_plan/exemplars/{index}"
        path = exemplar.get("path")
        symbol = exemplar.get("symbol")
        if not isinstance(path, str) or not check_path(f"{pointer}/path", path):
            continue
        source = _read_repo_file(repo, path)
        if (
            source is None
            or not isinstance(symbol, str)
            or symbol not in source
        ):
            add("TEST_EXEMPLAR_INVALID", pointer)
    cases = test_plan.get("cases")
    test_symbols: set[str] = set()
    for index, case in enumerate(cases if isinstance(cases, list) else []):
        if not isinstance(case, dict):
            continue
        test_file = case.get("test_file")
        file_pointer = f"/test_plan/cases/{index}/test_file"
        if check_path(file_pointer, test_file) and test_file not in in_scope:
            add("TEST_PATH_OUT_OF_SCOPE", file_pointer, hint=in_scope_hint)
        symbol = case.get("test_symbol")
        if isinstance(symbol, str):
            if symbol in test_symbols:
                add(
                    "TEST_SYMBOL_DUPLICATE",
                    f"/test_plan/cases/{index}/test_symbol",
                )
            test_symbols.add(symbol)

    criteria = normalized.get("done_criteria")
    criteria = criteria if isinstance(criteria, list) else []
    if not any(
        isinstance(criterion, dict) and criterion.get("kind") == "behavior"
        for criterion in criteria
    ):
        add(
            "DONE_CRITERIA_INCOMPLETE",
            "/done_criteria",
            hint="author at least one done criterion with kind behavior",
        )

    for pointer, ref in _iter_command_refs(normalized):
        if pointer.startswith("/steps/"):
            continue
        check_ref(pointer, ref)

    step_count = len(steps)
    known_paths = in_scope | set(excluded_paths)
    lexical_known = [
        path
        for path in [*lexical_in_scope, *excluded_paths]
        if _valid_repository_file_path(path)
    ]
    known_hint = (
        "declared paths: " + ", ".join(lexical_known)
        if lexical_known
        else "declare the path in scope first"
    )

    def check_stop_references(pointer: str, condition: Any) -> None:
        if not isinstance(condition, dict):
            return
        related_paths = condition.get("related_paths")
        for index, path in enumerate(
            related_paths if isinstance(related_paths, list) else []
        ):
            path_pointer = f"{pointer}/related_paths/{index}"
            if check_path(path_pointer, path) and path not in known_paths:
                add("STOP_PATH_UNKNOWN", path_pointer, hint=known_hint)
        numbers = condition.get("related_step_numbers")
        for index, number in enumerate(
            numbers if isinstance(numbers, list) else []
        ):
            if isinstance(number, int) and number > step_count:
                add(
                    "STOP_STEP_UNKNOWN",
                    f"{pointer}/related_step_numbers/{index}",
                    f"steps={step_count}",
                )

    check_stop_references("/false_assumption", normalized.get("false_assumption"))
    extra_conditions = normalized.get("additional_stop_conditions")
    for index, condition in enumerate(
        extra_conditions if isinstance(extra_conditions, list) else []
    ):
        check_stop_references(f"/additional_stop_conditions/{index}", condition)

    return issues


def _expand_command_ref(
    ref: dict[str, Any],
    *,
    recon_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    base = recon_by_id[ref["recon_command_id"]]
    appended = ref["appended_args"]
    return {
        "purpose": base["purpose"],
        "command": (
            base["command"]
            if appended is None
            else f"{base['command']} {appended}"
        ),
        "working_directory": base["working_directory"],
        "expected_success": deepcopy(base["expected_success"]),
        "applicability": deepcopy(base["applicability"]),
        "provenance": {
            "kind": "recon" if appended is None else "planner-derived",
            "recon_command_id": base["id"],
            "source_path": base["evidence"]["source_path"],
        },
        "note": ref["note"],
    }


def _expand_optional_ref(
    ref: dict[str, Any] | None,
    *,
    recon_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    if ref is None:
        return None
    return _expand_command_ref(ref, recon_by_id=recon_by_id)


def _derived_commands_table(
    normalized: dict[str, Any],
    *,
    recon_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    seen: set[tuple[str, str | None]] = set()
    for _, ref in _iter_command_refs(normalized):
        key = (ref["recon_command_id"], ref["appended_args"])
        if key in seen:
            continue
        seen.add(key)
        table.append(_expand_command_ref(ref, recon_by_id=recon_by_id))
    return table


def _resolve_excerpt(repo: Path, path: str, start: int, end: int) -> str:
    lines = (repo / path).read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[start - 1 : end])


def _boilerplate_stop_conditions(
    normalized: dict[str, Any],
    step_count: int,
) -> list[dict[str, Any]]:
    scope = normalized["scope"]
    existing_paths = _entry_paths(scope["existing_paths"])
    in_scope_paths = [*existing_paths, *_entry_paths(scope["new_paths"])]
    conditions = [
        {
            "kind": "drift",
            "condition": (
                "Before editing a file, read the exact line range quoted for "
                "it in the Current state section and compare it to the quoted "
                "text. It does not match character for character."
            ),
            "required_action": STOP_REQUIRED_ACTION,
            "evidence_to_report": (
                "Report the mismatched file, the quoted excerpt, and the "
                "current repository content."
            ),
            "related_paths": existing_paths,
            "related_step_ids": [],
        },
        {
            "kind": "repeated-verification-failure",
            "condition": (
                "A verification in this plan fails, you make exactly one "
                "correction, and it fails again — two failures total for the "
                "same verification. Do not attempt a third time."
            ),
            "required_action": STOP_REQUIRED_ACTION,
            "evidence_to_report": (
                "Report both failing command outputs and the correction that "
                "was attempted."
            ),
            "related_paths": [],
            "related_step_ids": [],
        },
        {
            "kind": "out-of-scope-change",
            "condition": (
                "Completing a step requires editing a path that is not "
                "declared in this plan's scope."
            ),
            "required_action": STOP_REQUIRED_ACTION,
            "evidence_to_report": (
                "Report the required path and why the declared scope "
                "boundary is insufficient."
            ),
            "related_paths": in_scope_paths,
            "related_step_ids": [],
        },
    ]

    def mapped(kind: str, condition: dict[str, Any]) -> dict[str, Any]:
        return {
            "kind": kind,
            "condition": condition["condition"],
            "required_action": STOP_REQUIRED_ACTION,
            "evidence_to_report": condition["evidence_to_report"],
            "related_paths": list(condition["related_paths"]),
            "related_step_ids": [
                f"step-{number}"
                for number in condition["related_step_numbers"]
                if number <= step_count
            ],
        }

    conditions.append(
        mapped("false-assumption", normalized["false_assumption"])
    )
    conditions.extend(
        mapped(condition["kind"], condition)
        for condition in normalized["additional_stop_conditions"]
    )
    return conditions


def _injected_done_criteria(
    normalized: dict[str, Any],
    *,
    recon_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    criteria = [
        {
            "kind": criterion["kind"],
            "description": criterion["description"],
            "verification": _expand_optional_ref(
                criterion["verification"], recon_by_id=recon_by_id
            ),
        }
        for criterion in normalized["done_criteria"]
    ]
    kinds = {criterion["kind"] for criterion in criteria}
    description_limit = 500
    if "test-gate" not in kinds:
        symbols = ", ".join(
            case["test_symbol"] for case in normalized["test_plan"]["cases"]
        )
        description = f"Every named test-plan case passes: {symbols}."
        criteria.append(
            {
                "kind": "test-gate",
                "description": description[:description_limit],
                "verification": None,
            }
        )
    if "scope-integrity" not in kinds:
        scope = normalized["scope"]
        paths = ", ".join(
            [
                *_entry_paths(scope["existing_paths"]),
                *_entry_paths(scope["new_paths"]),
            ]
        )
        description = f"Only the declared in-scope paths change: {paths}."
        criteria.append(
            {
                "kind": "scope-integrity",
                "description": description[:description_limit],
                "verification": None,
            }
        )
    return [
        {"id": f"done-{index}", **criterion}
        for index, criterion in enumerate(criteria, start=1)
    ]


def assemble_plan(
    authored: Any,
    *,
    repo: Path,
    recon_commands: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any] | None, tuple[AssemblyIssue, ...]]:
    """Normalize, collect ALL authoring issues, then expand.

    Returns ``(assembled, ())`` on success or ``(None, issues)`` when
    authoring defects remain after normalization. The assembled dict has the
    ``PLAN_WRITER_SCHEMA`` shape so ``render_plan``/``write_plans`` and review
    parsing are unchanged. Never raises on model content.
    """
    normalized = _normalize_authored(authored, repo=repo)
    if normalized is None:
        return None, (AssemblyIssue("NO_STRUCTURED_OBJECT", "/"),)
    recon_by_id = {
        command["id"]: command
        for command in recon_commands
        if isinstance(command, dict) and isinstance(command.get("id"), str)
    }
    issues = _collect_issues(normalized, repo=repo, recon_by_id=recon_by_id)
    if issues:
        return None, tuple(issues)

    scope = normalized["scope"]
    current_state_excerpts = [
        {
            "path": entry["path"],
            "line_anchor": {
                "start_line": anchor["start_line"],
                "end_line": anchor["end_line"],
            },
            "file_role": entry["role"],
            "verbatim_excerpt": _resolve_excerpt(
                repo, entry["path"], anchor["start_line"], anchor["end_line"]
            ),
        }
        for entry in scope["existing_paths"]
        for anchor in entry["excerpts"]
    ]
    current_state_excerpts.extend(
        {
            "path": entry["path"],
            "line_anchor": {
                "start_line": entry["start_line"],
                "end_line": entry["end_line"],
            },
            "file_role": entry["file_role"],
            "verbatim_excerpt": _resolve_excerpt(
                repo, entry["path"], entry["start_line"], entry["end_line"]
            ),
        }
        for entry in normalized["context_excerpts"]
    )
    assembled = {
        "slug": normalized["slug"],
        "title": normalized["title"],
        "priority": normalized["priority"],
        "dependencies": deepcopy(normalized["dependencies"]),
        "why_this_matters": dict(normalized["why_this_matters"]),
        "current_state_excerpts": current_state_excerpts,
        "commands_you_will_need": _derived_commands_table(
            normalized, recon_by_id=recon_by_id
        ),
        "scope": {
            "existing_paths": [
                {"path": entry["path"], "role": entry["role"]}
                for entry in scope["existing_paths"]
            ],
            "new_paths": deepcopy(scope["new_paths"]),
            "out_of_scope_paths": deepcopy(scope["out_of_scope_paths"]),
            "out_of_scope_behaviors": deepcopy(scope["out_of_scope_behaviors"]),
        },
        "git_workflow": {
            "branch_name": _branch_name(normalized["slug"]),
            "branch_basis": GIT_BRANCH_BASIS,
            "commit_boundaries": normalized["git_workflow"]["commit_boundaries"],
            "commit_message_example": (
                normalized["git_workflow"]["commit_message_example"]
            ),
            "push_policy": GIT_PUSH_POLICY,
            "pull_request_policy": GIT_PULL_REQUEST_POLICY,
        },
        "steps": [
            {
                "id": f"step-{index}",
                "order": index,
                "title": step["title"],
                "changes": deepcopy(step["changes"]),
                "verification": _expand_optional_ref(
                    step["verification"], recon_by_id=recon_by_id
                ),
            }
            for index, step in enumerate(normalized["steps"], start=1)
        ],
        "test_plan": {
            "exemplars": deepcopy(normalized["test_plan"]["exemplars"]),
            "cases": [
                {
                    **{key: case[key] for key in case if key != "verification"},
                    "verification": _expand_optional_ref(
                        case["verification"], recon_by_id=recon_by_id
                    ),
                }
                for case in normalized["test_plan"]["cases"]
            ],
        },
        "done_criteria": _injected_done_criteria(
            normalized, recon_by_id=recon_by_id
        ),
        "stop_conditions": _boilerplate_stop_conditions(
            normalized, len(normalized["steps"])
        ),
        "maintenance_notes": deepcopy(normalized["maintenance_notes"]),
    }
    return assembled, ()


__all__ = [
    "GIT_BRANCH_BASIS",
    "GIT_PULL_REQUEST_POLICY",
    "GIT_PUSH_POLICY",
    "STOP_REQUIRED_ACTION",
    "AssemblyIssue",
    "assemble_plan",
    "render_issue",
]
