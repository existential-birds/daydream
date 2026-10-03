"""Deterministic repairs for authored plan scope, excerpts, prose, and command selection."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from daydream.improve import plan_contract as contract
from daydream.improve.command_contract import (
    path_is_confined as _path_is_confined,
    valid_repository_file_path as _valid_repository_file_path,
)
from daydream.improve.render import redact_secret_values
from daydream.improve.schemas import PLAN_AUTHOR_SCHEMA

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
    ("test_plan", "rationale"),
    ("test_plan", "existing_coverage", "*", "behavior"),
    ("test_plan", "existing_coverage", "*", "verification", "note"),
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


_AUTHOR_PROSE_CLAMP_LIMITS = {
    pattern: _author_schema_max_length(pattern) for pattern in _AUTHOR_PROSE_FIELD_PATTERNS
}


def _clamp_string(value: Any, limit: int) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[: limit - 1] + "…"
    return value


def _normalize_value(
    value: Any, schema: dict[str, Any], pattern: tuple[str, ...] = (),
) -> Any:
    """Copy schema fields, redact strings, and clamp only display prose.

    Malformed values retain their shape for complete diagnostics. Instruction
    and target-state strings are never truncated.
    """
    if isinstance(value, dict):
        properties = schema.get("properties")
        fields = properties if properties is not None else dict.fromkeys(value, {})
        return {
            key: _normalize_value(value[key], subschema, (*pattern, key))
            for key, subschema in fields.items()
            if key in value
        }
    if isinstance(value, list):
        return [_normalize_value(item, schema.get("items", {}), (*pattern, "*")) for item in value]
    if isinstance(value, str):
        redacted = redact_secret_values(value)
        limit = _AUTHOR_PROSE_CLAMP_LIMITS.get(pattern)
        return _clamp_string(redacted, limit) if limit is not None else redacted
    return value


def _exists_on_disk(repo: Path, path: str) -> bool:
    if not _valid_repository_file_path(path) or not _path_is_confined(repo, path):
        return False
    try:
        return (repo / path).is_file()
    except (OSError, ValueError):
        return False


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
        conflicts = set(contract._entry_paths(existing)) & set(contract._entry_paths(new))
        for path in conflicts:
            list_name = "new_paths" if _exists_on_disk(repo, path) else "existing_paths"
            scope[list_name] = [
                entry
                for entry in scope[list_name]
                if not (isinstance(entry, dict) and entry.get("path") == path)
            ]
    in_scope = set(contract._entry_paths(scope.get("existing_paths"))) | set(
        contract._entry_paths(scope.get("new_paths"))
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
    for _, change in contract._changes(normalized):
        path = change.get("path")
        if isinstance(path, str):
            yield path, _STEP_PATH_ROLE
    test_plan = normalized.get("test_plan")
    cases = test_plan.get("cases") if isinstance(test_plan, dict) else None
    for _, case in contract._object_entries(cases):
        path = case.get("test_file")
        if isinstance(path, str):
            yield path, _TEST_PATH_ROLE


def _declare_referenced_paths(
    normalized: dict[str, Any],
    *,
    repo: Path,
) -> None:
    """Declare confined step/test paths before scope deduplication.

    Append as new paths; relocation then anchors nonempty existing files and
    changes create operations to modify. Malformed or escaping paths stay at
    their authored pointers for validation. Empty files remain new because they
    have no excerpt to anchor.
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
    declared = {*contract._entry_paths(existing_entries), *contract._entry_paths(new_entries)}
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
    """Anchor a relocated file once, reusing its authored role."""
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
    """Move nonempty existing files from new to existing scope and quote them.

    Convert their create operations to modify. Malformed, escaping, non-file,
    and empty-file paths remain unchanged for validation; an empty file can
    still be created but cannot supply a drift excerpt.
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
            contract._read_repo_file(repo, path)
            if isinstance(path, str) and isinstance(role, str)
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
    for _, change in contract._changes(normalized):
        if change.get("path") in relocated and change.get("operation") == "create":
            change["operation"] = "modify"


def _clamp_excerpt_end_lines(normalized: dict[str, Any], *, repo: Path) -> None:
    line_counts: dict[str, int | None] = {}
    context = normalized.get("context_excerpts")
    for _, anchor in contract._object_entries(context):
        path = anchor.get("path")
        if not isinstance(path, str):
            continue
        if path not in line_counts:
            source = contract._read_repo_file(repo, path)
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
    """Declare confined stop-condition paths as read-only context.

    Missing paths may describe expected absences. Malformed or escaping paths
    remain unchanged for validation at their authored pointers.
    """
    scope = normalized.get("scope")
    if not isinstance(scope, dict):
        return
    out_entries = scope.get("out_of_scope_paths")
    if not isinstance(out_entries, list):
        return
    declared = {
        *contract._entry_paths(scope.get("existing_paths")),
        *contract._entry_paths(scope.get("new_paths")),
        *contract._entry_paths(out_entries),
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
    """Copy and normalize author content; return None for non-objects."""
    if not isinstance(authored, dict):
        return None
    normalized: dict[str, Any] = _normalize_value(authored, PLAN_AUTHOR_SCHEMA)
    _declare_referenced_paths(normalized, repo=repo)
    _dedup_scope(normalized, repo=repo)
    _relocate_existing_new_paths(normalized, repo=repo)
    _clamp_excerpt_end_lines(normalized, repo=repo)
    return normalized


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


_COMMAND_NOTE_MAX_LENGTH = _author_schema_max_length(("additional_command_refs", "*", "note"))
_RETARGETED_COMMAND_NOTE = (
    "Retargeted by the host: the command this plan named does not cover this "
    "verification's targets, so this repository-wide command runs instead."
)
_COMMAND_SCOPE_CAVEAT = (
    "Scope caveat from the host: this command's verified applicability does "
    "not cover every target of this verification, and no verified command "
    "that covers them is available here. Run it as written and report what "
    "it reports; do not substitute a command of your own."
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
    """Prefer a verified repository-wide gate when a command misses its targets.

    Otherwise retain the command with a scope caveat. Never retarget a reference
    with appended arguments: those belong to its original command. Unknown IDs
    remain validation failures. Existing coverage targets its read-only test;
    step and test-case gates target writable paths.
    """
    scope = normalized.get("scope")
    scope = scope if isinstance(scope, dict) else {}
    in_scope = sorted(
        {
            *contract._entry_paths(scope.get("existing_paths")),
            *contract._entry_paths(scope.get("new_paths")),
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
    for _, ref, targets in contract._iter_command_refs(normalized):
        recon_id = ref.get("recon_command_id")
        base = recon_by_id.get(recon_id) if isinstance(recon_id, str) else None
        if base is None:
            continue
        targets = in_scope if targets is None else targets
        if _command_scope_fits_plan(base, targets) and _command_covers_all(base, targets):
            continue
        if repository_wide is not None and ref.get("appended_args") is None:
            ref["recon_command_id"] = repository_wide
            _annotate_ref(ref, _RETARGETED_COMMAND_NOTE)
            continue
        _annotate_ref(ref, _COMMAND_SCOPE_CAVEAT)
