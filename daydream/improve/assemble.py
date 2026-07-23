"""Deterministic host assembly of model-authored improve plans.

The model authors judgment content only (``PLAN_AUTHOR_SCHEMA``); this module
normalizes it, collects every authoring issue at once, and expands the result
into the assembled plan shape that ``render_plan`` and ``write_plans`` already
consume. Assembly is the single validation boundary for model-authored plan
content: nothing downstream re-checks it. Assembly is pure with respect to the
model output: filesystem reads only, no randomness, no wall-clock reads.
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
from daydream.improve.plans import plan_slug, redact_secret_values
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


def _branch_name(title: str) -> str:
    return f"improve/{plan_slug(title)}"


_AUTHOR_PROSE_FIELD_PATTERNS: tuple[tuple[str, ...], ...] = (
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
    ("additional_command_refs", "*", "note"),
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


def _clamp_string(value: Any, limit: int) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[: limit - 1] + "…"
    return value


def _clamp_node(node: Any, pattern: tuple[str, ...], limit: int) -> None:
    head, rest = pattern[0], pattern[1:]
    if head == "*":
        if not isinstance(node, list):
            return
        for index, child in enumerate(node):
            if rest:
                _clamp_node(child, rest, limit)
            else:
                node[index] = _clamp_string(child, limit)
        return
    if not isinstance(node, dict):
        return
    if rest:
        _clamp_node(node.get(head), rest, limit)
    elif head in node:
        node[head] = _clamp_string(node[head], limit)


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


def _context_excerpts(normalized: dict[str, Any]) -> list[Any]:
    """Return the plan's excerpt list, creating it when a repair must append."""
    context = normalized.get("context_excerpts")
    if not isinstance(context, list):
        context = []
        normalized["context_excerpts"] = context
    return context


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


_STEP_PATH_ROLE = (
    "Named by a plan step but left out of the authored scope lists; the host "
    "declared it in scope so the executor is allowed to change it."
)
_TEST_PATH_ROLE = (
    "Named by a test-plan case but left out of the authored scope lists; the "
    "host declared it in scope so the executor is allowed to write it."
)


def _referenced_paths(normalized: dict[str, Any]) -> Iterator[tuple[str, str]]:
    """Yield every (path, host role) a step change or test case names."""
    steps = normalized.get("steps")
    for step in steps if isinstance(steps, list) else []:
        changes = step.get("changes") if isinstance(step, dict) else None
        for change in changes if isinstance(changes, list) else []:
            path = change.get("path") if isinstance(change, dict) else None
            if isinstance(path, str):
                yield path, _STEP_PATH_ROLE
    test_plan = normalized.get("test_plan")
    cases = test_plan.get("cases") if isinstance(test_plan, dict) else None
    for case in cases if isinstance(cases, list) else []:
        path = case.get("test_file") if isinstance(case, dict) else None
        if isinstance(path, str):
            yield path, _TEST_PATH_ROLE


def _declare_referenced_paths(
    normalized: dict[str, Any],
    *,
    repo: Path,
) -> None:
    """Declare every well-formed step/test path the plan left out of scope.

    Which list a path belongs in is disk truth, not judgment, so the host
    settles it the same way ``_dedup_scope`` and ``_relocate_existing_new_paths``
    already do rather than spending a repair generation on the bookkeeping gap.
    Every path is appended to ``new_paths``; ``_relocate_existing_new_paths``
    runs later in normalization and moves the ones that exist on disk into
    ``existing_paths``, giving them the head-of-file ``context_excerpts`` anchor
    every existing path needs and the drift stop condition quotes, and turning
    any ``create`` of them into a ``modify``. Running before ``_dedup_scope``
    also lets that pass drop an out-of-scope entry the new declaration
    contradicts.

    Malformed and unconfined paths are deliberately skipped so they still fail
    as ``MALFORMED_PATH`` / ``PATH_OUTSIDE_REPOSITORY`` at their own step or
    test-case pointer: this repair must never launder an escaping path into
    scope. A path that exists on disk but is empty has no line to anchor, so it
    stays in ``new_paths`` and a step that means to modify it still fails as
    ``CHANGE_PATH_NOT_EXISTING``.
    """
    scope = normalized.get("scope")
    if not isinstance(scope, dict):
        return
    existing_entries = scope.get("existing_paths")
    new_entries = scope.get("new_paths")
    if not isinstance(existing_entries, list) or not isinstance(
        new_entries, list
    ):
        return
    declared = {*_entry_paths(existing_entries), *_entry_paths(new_entries)}
    for path, role in _referenced_paths(normalized):
        if (
            path in declared
            or not _valid_repository_file_path(path)
            or not _path_is_confined(repo, path)
        ):
            continue
        declared.add(path)
        new_entries.append({"path": path, "role": role})


_RELOCATED_EXCERPT_MAX_LINES = 40


def _quote_head_of_file(
    normalized: dict[str, Any],
    *,
    path: str,
    role: str,
    line_count: int,
) -> None:
    """Anchor the head of a relocated path unless the plan already quotes it.

    The path's own role is reused verbatim as the excerpt's ``file_role``: it is
    the sentence describing this file the plan already carries, and both fields
    hold the same kind of prose under the same length bounds.
    """
    context = _context_excerpts(normalized)
    if any(
        isinstance(entry, dict) and entry.get("path") == path
        for entry in context
    ):
        return
    context.append(
        {
            "path": path,
            "start_line": 1,
            "end_line": min(line_count, _RELOCATED_EXCERPT_MAX_LINES),
            "file_role": role,
        }
    )


def _relocate_existing_new_paths(
    normalized: dict[str, Any],
    *,
    repo: Path,
) -> None:
    """Move a declared-new path that already exists on disk into scope.existing_paths.

    ``_dedup_scope`` already settles this exact defect from disk when a path is
    declared in both lists; a path declared only under ``new_paths`` gets the
    same answer here rather than costing a repair generation. Every existing
    path must be quoted in ``context_excerpts`` and the drift stop condition
    compares against that quote, so the host anchors the head of the real file
    when the path is not quoted already, and any step that meant to ``create``
    the path is switched to ``modify`` so the plan stays coherent.

    Three cases are deliberately left alone. Malformed and unconfined paths stay
    in ``new_paths`` so they still fail as ``MALFORMED_PATH`` /
    ``PATH_OUTSIDE_REPOSITORY``. A path occupied by something that is not a
    regular file still fails as ``NEW_PATH_ALREADY_EXISTS`` — the host cannot
    turn a directory into a file. An existing but empty file has no line to
    anchor, and writing a new file's content into it is what ``create`` already
    means, so it stays a create.
    """
    scope = normalized.get("scope")
    if not isinstance(scope, dict):
        return
    new_entries = scope.get("new_paths")
    existing_entries = scope.get("existing_paths")
    if not isinstance(new_entries, list) or not isinstance(
        existing_entries, list
    ):
        return
    kept: list[Any] = []
    relocated: set[str] = set()
    for entry in new_entries:
        path = entry.get("path") if isinstance(entry, dict) else None
        role = entry.get("role") if isinstance(entry, dict) else None
        source = (
            _read_repo_file(repo, path)
            if isinstance(path, str)
            and isinstance(role, str)
            and _valid_repository_file_path(path)
            and _path_is_confined(repo, path)
            and _exists_on_disk(repo, path)
            else None
        )
        line_count = len(source.splitlines()) if source is not None else 0
        if line_count < 1:
            kept.append(entry)
            continue
        existing_entries.append({"path": path, "role": role})
        _quote_head_of_file(
            normalized,
            path=str(path),
            role=str(role),
            line_count=line_count,
        )
        relocated.add(str(path))
    scope["new_paths"] = kept
    if not relocated:
        return
    steps = normalized.get("steps")
    for step in steps if isinstance(steps, list) else []:
        changes = step.get("changes") if isinstance(step, dict) else None
        for change in changes if isinstance(changes, list) else []:
            if (
                isinstance(change, dict)
                and change.get("path") in relocated
                and change.get("operation") == "create"
            ):
                change["operation"] = "modify"


_TEST_SYMBOL_MAX_LENGTH = _author_schema_max_length(
    ("test_plan", "cases", "*", "test_symbol")
)


def _disambiguate_test_symbols(normalized: dict[str, Any]) -> None:
    """Number a repeated test symbol instead of blocking on the collision.

    Two cases naming the same test is a naming slip, not a defect in the work
    the plan describes. The first occurrence keeps the authored name and each
    repeat gets the lowest unused ``_<n>`` suffix, so every case still names one
    distinct test the executor can write and run.
    """
    test_plan = normalized.get("test_plan")
    cases = test_plan.get("cases") if isinstance(test_plan, dict) else None
    seen: set[str] = set()
    for case in cases if isinstance(cases, list) else []:
        symbol = case.get("test_symbol") if isinstance(case, dict) else None
        if not isinstance(symbol, str):
            continue
        candidate = symbol
        suffix = 2
        while candidate in seen:
            tail = f"_{suffix}"
            candidate = symbol[: _TEST_SYMBOL_MAX_LENGTH - len(tail)] + tail
            suffix += 1
        seen.add(candidate)
        case["test_symbol"] = candidate


def _clamp_excerpt_end_lines(normalized: dict[str, Any], *, repo: Path) -> None:
    line_counts: dict[str, int | None] = {}
    context = normalized.get("context_excerpts")
    for anchor in context if isinstance(context, list) else []:
        if not isinstance(anchor, dict):
            continue
        path = anchor.get("path")
        if not isinstance(path, str):
            continue
        if path not in line_counts:
            source = _read_repo_file(repo, path)
            line_counts[path] = (
                len(source.splitlines()) if source is not None else None
            )
        line_count = line_counts[path]
        start = anchor.get("start_line")
        end = anchor.get("end_line")
        if (
            line_count is not None
            and isinstance(start, int)
            and isinstance(end, int)
            and 1 <= start <= line_count
            and end > line_count
        ):
            anchor["end_line"] = line_count


_STOP_PATH_OUT_OF_SCOPE_REASON = (
    "Referenced by a stop condition for context only; do not create, modify, "
    "or depend on this path."
)


def _declare_stop_condition_paths(
    normalized: dict[str, Any],
    *,
    repo: Path,
) -> None:
    """Declare every well-formed stop-condition path the plan left undeclared.

    A stop condition legitimately names paths the plan never touches — a file
    it expects to already be deleted, most of all. Blocking the plan over that
    bookkeeping gap wastes a repair generation, so the host closes it instead:
    declaring a path out-of-scope only ever restricts the executor further, and
    the injected entry renders in the plan's own out-of-scope section.
    Malformed and unconfined paths are left alone so they still fail as
    ``MALFORMED_PATH`` / ``PATH_OUTSIDE_REPOSITORY``.
    """
    scope = normalized.get("scope")
    if not isinstance(scope, dict):
        return
    out_entries = scope.get("out_of_scope_paths")
    if not isinstance(out_entries, list):
        return
    declared = {
        *_entry_paths(scope.get("existing_paths")),
        *_entry_paths(scope.get("new_paths")),
        *_entry_paths(out_entries),
    }
    condition = normalized.get("false_assumption")
    if not isinstance(condition, dict):
        return
    related = condition.get("related_paths")
    for path in related if isinstance(related, list) else []:
        if (
            not isinstance(path, str)
            or path in declared
            or not _valid_repository_file_path(path)
            or not _path_is_confined(repo, path)
        ):
            continue
        declared.add(path)
        out_entries.append(
            {"path": path, "reason": _STOP_PATH_OUT_OF_SCOPE_REASON}
        )


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
    _declare_referenced_paths(normalized, repo=repo)
    _dedup_scope(normalized, repo=repo)
    _relocate_existing_new_paths(normalized, repo=repo)
    _disambiguate_test_symbols(normalized)
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


_COMMAND_NOTE_MAX_LENGTH = _author_schema_max_length(
    ("additional_command_refs", "*", "note")
)
_RETARGETED_COMMAND_NOTE = (
    "Retargeted by the host: the command this plan named is verified only for "
    "paths this plan does not change, so this repository-wide command runs "
    "instead."
)
_COMMAND_SCOPE_CAVEAT = (
    "Scope caveat from the host: this command's verified applicability does "
    "not cover every path this plan changes, and no verified command that "
    "covers them is available here. Run it as written and report what it "
    "reports; do not substitute a command of your own."
)


def _annotate_ref(ref: dict[str, Any], addition: str) -> None:
    """Append host text to a ref note, clamping the model's half, never ours."""
    note = ref.get("note")
    room = _COMMAND_NOTE_MAX_LENGTH - len(addition) - 1
    keep = (
        f"{_clamp_string(note, room)} "
        if isinstance(note, str) and note and room > 1
        else ""
    )
    ref["note"] = keep + addition


def _repair_command_scope(
    normalized: dict[str, Any],
    *,
    recon_by_id: dict[str, dict[str, Any]],
) -> None:
    """Retarget or annotate a verified command whose scope misses the plan.

    A command whose applicability does not line up with the plan's changed
    paths runs a slightly too narrow or too broad check; it does not make the
    plan wrong, so it is repaired rather than thrown away. The host prefers a
    verified repository-wide command, which by definition covers every path,
    and otherwise keeps the model's choice and states the caveat in the note
    the plan renders — the executor is told the truth either way.

    Two things are deliberately left alone. A ``recon_command_id`` naming no
    verified command at all is a hallucination, not a scope imperfection, and
    still fails as ``RECON_COMMAND_UNKNOWN``. A ref carrying ``appended_args``
    is never retargeted either: the suffix was authored for the command it was
    attached to, and pasting it onto a different one composes a command nobody
    verified.
    """
    scope = normalized.get("scope")
    scope = scope if isinstance(scope, dict) else {}
    in_scope = sorted(
        {
            *_entry_paths(scope.get("existing_paths")),
            *_entry_paths(scope.get("new_paths")),
        }
    )
    repository_wide = next(
        (
            recon_id
            for recon_id, command in recon_by_id.items()
            if _scope_paths_of(command) is None
        ),
        None,
    )
    steps = normalized.get("steps")
    step_targets: dict[str, list[str]] = {}
    for index, step in enumerate(steps if isinstance(steps, list) else []):
        changes = step.get("changes") if isinstance(step, dict) else None
        step_targets[f"/steps/{index}/verification"] = [
            change["path"]
            for change in (changes if isinstance(changes, list) else [])
            if isinstance(change, dict) and isinstance(change.get("path"), str)
        ]
    for pointer, ref in _iter_command_refs(normalized):
        recon_id = ref.get("recon_command_id")
        base = recon_by_id.get(recon_id) if isinstance(recon_id, str) else None
        if base is None:
            continue
        if _command_scope_fits_plan(base, in_scope) and _command_covers_all(
            base, step_targets.get(pointer, ())
        ):
            continue
        if repository_wide is not None and ref.get("appended_args") is None:
            ref["recon_command_id"] = repository_wide
            _annotate_ref(ref, _RETARGETED_COMMAND_NOTE)
            continue
        _annotate_ref(ref, _COMMAND_SCOPE_CAVEAT)


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

    context = normalized.get("context_excerpts")
    context = context if isinstance(context, list) else []
    quoted_paths = set(_entry_paths(context))

    if not existing_entries and not new_entries:
        add("EMPTY_SCOPE", "/scope")

    for index, entry in enumerate(existing_entries):
        if not isinstance(entry, dict):
            continue
        pointer = f"/scope/existing_paths/{index}/path"
        path = entry.get("path")
        if not isinstance(path, str) or not check_path(pointer, path):
            continue
        if _read_repo_file(repo, path) is None:
            add("EXISTING_PATH_MISSING", pointer)
            continue
        # The drift stop condition tells the executor to compare each file it
        # is about to edit against the text quoted for it, so a path this plan
        # changes without quoting leaves that condition nothing to compare.
        if path not in quoted_paths:
            add(
                "EXISTING_PATH_NOT_QUOTED",
                pointer,
                hint=(
                    "add a context_excerpts entry anchoring the lines of this "
                    "file the plan changes; every scope.existing_paths path "
                    "must be quoted there"
                ),
            )
    for index, entry in enumerate(new_entries):
        if not isinstance(entry, dict):
            continue
        pointer = f"/scope/new_paths/{index}/path"
        path = entry.get("path")
        if not isinstance(path, str) or not check_path(pointer, path):
            continue
        # An existing regular file was already relocated into existing_paths by
        # normalization; what survives here is a path occupied by a directory or
        # another non-file, which no host repair can turn into a new file.
        try:
            occupied = (repo / path).exists() and not (repo / path).is_file()
        except (OSError, ValueError):
            occupied = False
        if occupied:
            add(
                "NEW_PATH_ALREADY_EXISTS",
                pointer,
                hint="a directory already occupies this path; name a file path",
            )
    for index, entry in enumerate(out_entries):
        if isinstance(entry, dict):
            check_path(
                f"/scope/out_of_scope_paths/{index}/path",
                entry.get("path"),
                directory=True,
            )

    for index, entry in enumerate(context):
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

    def check_ref(pointer: str, ref: Any) -> None:
        # Imperfect command scope was already retargeted or annotated by
        # _repair_command_scope; a command that does not exist at all is a
        # hallucination no host repair can settle, so it still blocks.
        if not isinstance(ref, dict):
            return
        recon_id = ref.get("recon_command_id")
        if isinstance(recon_id, str) and recon_id not in recon_by_id:
            add(
                "RECON_COMMAND_UNKNOWN",
                f"{pointer}/recon_command_id",
                hint=recon_ids_hint,
            )
        appended = ref.get("appended_args")
        if isinstance(appended, str) and _appended_args_invalid(appended):
            add("MALFORMED_APPENDED_ARGS", f"{pointer}/appended_args")

    steps = normalized.get("steps")
    steps = steps if isinstance(steps, list) else []
    for step_index, step in enumerate(steps):
        if not isinstance(step, dict):
            continue
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
            # Every well-formed path is in scope by now: normalization declares
            # the ones the plan left out, so what remains to check is only
            # whether the operation agrees with which list the path landed in.
            if change.get("operation") == "create" and path not in new_paths:
                add("CREATE_PATH_NOT_NEW", pointer)
            elif (
                change.get("operation") != "create"
                and path not in existing_paths
            ):
                add("CHANGE_PATH_NOT_EXISTING", pointer)
        verification = step.get("verification")
        if isinstance(verification, dict):
            check_ref(f"/steps/{step_index}/verification", verification)

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
    for index, case in enumerate(cases if isinstance(cases, list) else []):
        if not isinstance(case, dict):
            continue
        # Normalization already declared a well-formed test_file in scope and
        # disambiguated any repeated test_symbol; the check that stays is the
        # path grammar and confinement one.
        check_path(f"/test_plan/cases/{index}/test_file", case.get("test_file"))

    for pointer, ref in _iter_command_refs(normalized):
        if pointer.startswith("/steps/"):
            continue
        check_ref(pointer, ref)

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

    check_stop_references("/false_assumption", normalized.get("false_assumption"))

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
    # Repository bytes are spliced in after _redact_strings has already run over
    # the authored content, so they must be redacted here.
    lines = (repo / path).read_text(encoding="utf-8").splitlines()
    return redact_secret_values("\n".join(lines[start - 1 : end]))


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
    if "behavior" not in kinds:
        # ``why_this_matters.intended_outcome`` is schema-required at >=30
        # characters, so the derived criterion is never empty or a stub.
        outcome = normalized["why_this_matters"]["intended_outcome"]
        description = f"The plan's intended outcome holds: {outcome}"
        criteria.insert(
            0,
            {
                "kind": "behavior",
                "description": description[:description_limit],
                "verification": None,
            },
        )
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
    shape ``render_plan``/``write_plans`` and review parsing consume. Never
    raises on model content.
    """
    normalized = _normalize_authored(authored, repo=repo)
    if normalized is None:
        return None, (AssemblyIssue("NO_STRUCTURED_OBJECT", "/"),)
    _declare_stop_condition_paths(normalized, repo=repo)
    recon_by_id = {
        command["id"]: command
        for command in recon_commands
        if isinstance(command, dict) and isinstance(command.get("id"), str)
    }
    _repair_command_scope(normalized, recon_by_id=recon_by_id)
    issues = _collect_issues(normalized, repo=repo, recon_by_id=recon_by_id)
    if issues:
        return None, tuple(issues)

    scope = normalized["scope"]
    current_state_excerpts = [
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
    ]
    assembled = {
        "title": normalized["title"],
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
            "branch_name": _branch_name(normalized["title"]),
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
