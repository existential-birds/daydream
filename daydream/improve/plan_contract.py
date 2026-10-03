"""Plan authoring validation: schema, confined paths, exact test declarations, and commands."""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from daydream.improve.command_contract import (
    command_argv as _command_argv,
    has_shell_composition as _has_shell_composition,
    json_pointer,
    path_is_confined as _path_is_confined,
    valid_directory_scope_lexical as _valid_directory_scope,
    valid_repository_file_path as _valid_repository_file_path,
)
from daydream.improve.schemas import PLAN_AUTHOR_SCHEMA
from daydream.repository_paths import is_test_path

_PLACEHOLDER_ARG_TOKENS = {"...", "todo", "tbd", "${todo}"}
_DOCUMENTATION_SUFFIXES = {".adoc", ".md", ".mdx", ".rst", ".txt"}
_DOCUMENTATION_NAMES = {
    "authors",
    "changelog",
    "code_of_conduct",
    "contributing",
    "license",
    "maintainers",
    "readme",
    "security",
}
_COMMENT_KIND = re.compile(r"\b(?:comment|docstring)\b", re.IGNORECASE)
_ABSENCE_WORD = re.compile(r"\b(?:absent|delete[ds]?|no longer|remove[ds]?)\b", re.IGNORECASE)
_COMMENT_REWRITE = re.compile(r"\b(?:condense|rewrite|shorten|simplif(?:y|ies)|trim)\b", re.IGNORECASE)
_UNCHANGED_EXECUTION = re.compile(
    r"(?:\b(?:code|executable behavior|function body|runtime behavior)\b.{0,100}"
    r"\b(?:identical|unchanged|unmodified)\b|"
    r"\b(?:identical|unchanged|unmodified)\b.{0,100}"
    r"\b(?:code|executable behavior|function body|runtime behavior)\b)",
    re.IGNORECASE,
)
_IDENTIFIER = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")


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


def _read_repo_file(repo: Path, path: str) -> str | None:
    if not _valid_repository_file_path(path) or not _path_is_confined(repo, path):
        return None
    try:
        candidate = repo / path
        if not candidate.is_file():
            return None
        return candidate.read_text(encoding="utf-8")
    except (OSError, UnicodeError, ValueError):
        return None


def _is_documentation_path(path: str) -> bool:
    candidate = Path(path)
    if any(part.casefold() in {"doc", "docs", "documentation"} for part in candidate.parts):
        return True
    stem = candidate.stem.casefold()
    return candidate.suffix.casefold() in _DOCUMENTATION_SUFFIXES or stem in _DOCUMENTATION_NAMES


def _is_comment_only_change(change: dict[str, Any]) -> bool:
    if change.get("operation") not in {"modify", "delete"}:
        return False
    symbol = str(change.get("symbol") or "")
    instruction = str(change.get("instruction") or "")
    target_state = str(change.get("target_state") or "")
    authored_change = f"{symbol} {instruction}"
    if not (_COMMENT_KIND.search(authored_change) and _COMMENT_KIND.search(target_state)):
        return False
    return bool(
        _ABSENCE_WORD.search(target_state)
        or (_COMMENT_REWRITE.search(instruction) and _UNCHANGED_EXECUTION.search(target_state))
    )


def _not_applicable_change_allowed(change: dict[str, Any]) -> bool:
    """Permit omitted tests only for docs, exact comment cleanup, or test deletion.

    Production deletion still needs named coverage: model prose cannot prove
    code is dead. Non-test plans separately require a static or step gate.
    """

    path = change.get("path")
    if not isinstance(path, str):
        return False
    if _is_documentation_path(path) or _is_comment_only_change(change):
        return True
    return change.get("operation") == "delete" and is_test_path(path)


def _has_test_declaration(path: str, source: str, symbol: str) -> bool:
    """Match an exact test declaration, never a substring or prose mention."""

    if not symbol or "\n" in symbol or "\r" in symbol:
        return False
    test_path = is_test_path(path)
    if Path(path).suffix.casefold() == ".py" and _IDENTIFIER.fullmatch(symbol):
        if not test_path or not symbol.casefold().startswith("test"):
            return False
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return False
        return any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == symbol for node in ast.walk(tree)
        )

    escaped = re.escape(symbol)
    if _IDENTIFIER.fullmatch(symbol):
        path_based_declaration_patterns = (
            # Python/Ruby, Rust, Go/Swift, and JavaScript named functions.
            rf"^[ \t]*(?:(?:export|internal|private|protected|public|"
            rf"pub(?:\([^)]*\))?|async|static)\s+)*(?:def|fn|function)\s+"
            rf"{escaped}(?![A-Za-z0-9_$])\s*(?:\(|:)",
            rf"^[ \t]*(?:(?:internal|private|protected|public|static)\s+)*"
            rf"func(?:\s+\([^)]*\))?\s+{escaped}(?![A-Za-z0-9_$])\s*\(",
            # JavaScript/TypeScript arrow-function tests.
            rf"^[ \t]*(?:(?:export|async)\s+)*(?:const|let|var)\s+"
            rf"{escaped}(?![A-Za-z0-9_$])\s*=",
            # C#, JVM, and similar method declarations. Requiring at least one
            # modifier prevents an ordinary call expression from qualifying.
            rf"^[ \t]*(?:(?:async|final|internal|override|private|protected|"
            rf"public|static|virtual)\s+)+(?:fun|void|[A-Za-z_$]"
            rf"[A-Za-z0-9_$<>?,.\[\]]*)\s+{escaped}"
            rf"(?![A-Za-z0-9_$])\s*\(",
        )
        if test_path and any(re.search(pattern, source, re.MULTILINE) for pattern in path_based_declaration_patterns):
            return True
        annotated_declaration_patterns = (
            # C/C++ test macros are test declarations wherever they live.
            rf"^[ \t]*TEST(?:_[FP])?\s*\([^,\n]+,\s*"
            rf"{escaped}(?![A-Za-z0-9_$])\s*\)",
            # Attribute/annotation-driven Rust, JVM, and .NET test methods.
            rf"(?ms)^[ \t]*(?:#\[[^\]]*test[^\]]*\]|@Test|"
            rf"\[(?:Fact|Test|TestMethod|Theory)\])\s*\n"
            rf"[ \t]*(?:(?:public|private|protected|static|final|async|pub)\s+)*"
            rf"(?:fun|fn|void|[A-Za-z_$][A-Za-z0-9_$<>?,.\[\]]*)\s+"
            rf"{escaped}(?![A-Za-z0-9_$])\s*\(",
        )
        if any(re.search(pattern, source, re.MULTILINE) for pattern in annotated_declaration_patterns):
            return True

    # Frameworks such as Jest, RSpec, and ExUnit use an exact quoted title as
    # the test's durable identity rather than a language-level function name.
    for quote in ('"', "'"):
        quoted = re.escape(f"{quote}{symbol}{quote}")
        if re.search(
            rf"^[ \t]*(?:it|scenario|specify|test)\s*(?:\(\s*)?{quoted}",
            source,
            re.MULTILINE,
        ):
            return True
    return False


def _entry_paths(entries: Any) -> list[str]:
    return [entry["path"] for _, entry in _object_entries(entries) if isinstance(entry.get("path"), str)]


def _object_entries(entries: Any) -> Iterator[tuple[int, dict[str, Any]]]:
    """Skip malformed entries while retaining their positions for diagnostics."""
    if isinstance(entries, list):
        for index, entry in enumerate(entries):
            if isinstance(entry, dict):
                yield index, entry


def _changes(normalized: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    for step_index, step in _object_entries(normalized.get("steps")):
        for change_index, change in _object_entries(step.get("changes")):
            yield f"/steps/{step_index}/changes/{change_index}", change


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
                add("AUTHOR_SCHEMA_INVALID", json_pointer([*parts, key], root="/"))
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
        add("AUTHOR_SCHEMA_INVALID", json_pointer(parts, root="/"), detail, hint)


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
) -> Iterator[tuple[str, dict[str, Any], list[str] | None]]:
    """Yield each reference with its pointer and targets, in document order.

    ``None`` targets means the complete writable scope. Existing coverage
    targets its read-only test file; step gates target their changed paths.
    """
    def verification_refs(
        entries: Any, pointer: str, target_field: str | None = None,
    ) -> Iterator[tuple[str, dict[str, Any], list[str] | None]]:
        for index, entry in _object_entries(entries):
            if isinstance(entry.get("verification"), dict):
                targets = None
                if pointer == "/steps":
                    targets = _entry_paths(entry.get("changes"))
                elif target_field is not None and isinstance(entry.get(target_field), str):
                    targets = [entry[target_field]]
                yield f"{pointer}/{index}/verification", entry["verification"], targets

    yield from verification_refs(normalized.get("steps"), "/steps")
    test_plan = normalized.get("test_plan")
    existing_coverage = test_plan.get("existing_coverage") if isinstance(test_plan, dict) else None
    yield from verification_refs(existing_coverage, "/test_plan/existing_coverage", "path")
    cases = test_plan.get("cases") if isinstance(test_plan, dict) else None
    yield from verification_refs(cases, "/test_plan/cases", "test_file")
    yield from verification_refs(normalized.get("done_criteria"), "/done_criteria")
    extra = normalized.get("additional_command_refs")
    for index, ref in _object_entries(extra):
        yield f"/additional_command_refs/{index}", ref, None


def _collect_issues(
    normalized: dict[str, Any],
    *,
    repo: Path,
    recon_by_id: dict[str, dict[str, Any]],
    expected_fingerprints: Sequence[str] | None = None,
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

    def check_symbol_entries(entries: Any, base: str, code: str) -> None:
        """Flag each entry whose named test symbol is absent from its file."""
        for index, entry in _object_entries(entries):
            pointer = f"{base}/{index}"
            path = entry.get("path")
            symbol = entry.get("symbol")
            if not isinstance(path, str) or not check_path(f"{pointer}/path", path):
                continue
            source = _read_repo_file(repo, path)
            if source is None or not isinstance(symbol, str) or not _has_test_declaration(path, source, symbol):
                add(code, pointer)

    _schema_issues(normalized, add)

    covered = normalized.get("covered_fingerprints")
    if isinstance(covered, list) and all(isinstance(item, str) for item in covered):
        if len(covered) != len(set(covered)):
            add(
                "COVERED_FINGERPRINT_DUPLICATE",
                "/covered_fingerprints",
                hint="List each finding fingerprint exactly once.",
            )
        if expected_fingerprints is not None:
            expected = set(expected_fingerprints)
            actual = set(covered)
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            if missing or extra:
                add(
                    "COVERED_FINGERPRINTS_MISMATCH",
                    "/covered_fingerprints",
                    f"missing={len(missing)};extra={len(extra)}",
                    hint=("replace with exactly: " + ", ".join(str(item) for item in expected_fingerprints)),
                )

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

    for index, entry in _object_entries(existing_entries):
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
    for index, entry in _object_entries(new_entries):
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
    for index, entry in _object_entries(out_entries):
        check_path(
            f"/scope/out_of_scope_paths/{index}/path",
            entry.get("path"),
            directory=True,
        )

    for index, entry in _object_entries(context):
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
    for step_index, step in _object_entries(steps):
        for change_index, change in _object_entries(step.get("changes")):
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
    mode = test_plan.get("mode")
    existing_coverage = test_plan.get("existing_coverage")
    existing_coverage = existing_coverage if isinstance(existing_coverage, list) else []
    exemplars = test_plan.get("exemplars")
    exemplars = exemplars if isinstance(exemplars, list) else []
    cases = test_plan.get("cases")
    cases = cases if isinstance(cases, list) else []

    if mode == "new-or-updated-tests":
        if not cases:
            # Preserve the long-standing diagnostic while moving this rule out
            # of JSON Schema and into the mode-aware host boundary.
            add(
                "AUTHOR_SCHEMA_INVALID",
                "/test_plan/cases",
                "minItems=1;actual=0",
                "new-or-updated-tests mode requires at least one named case",
            )
        if existing_coverage:
            add(
                "TEST_PLAN_MODE_CONFLICT",
                "/test_plan/existing_coverage",
                hint=("new-or-updated-tests mode uses named cases; leave existing_coverage empty"),
            )
    elif mode == "existing-coverage":
        if not existing_coverage:
            add(
                "EXISTING_COVERAGE_REQUIRED",
                "/test_plan/existing_coverage",
                hint=("cite at least one existing test path and symbol, or choose another test-plan mode"),
            )
        if cases:
            add(
                "TEST_PLAN_MODE_CONFLICT",
                "/test_plan/cases",
                hint="existing-coverage mode must not author new test cases",
            )
        if exemplars:
            add(
                "TEST_PLAN_MODE_CONFLICT",
                "/test_plan/exemplars",
                hint="existing-coverage mode does not need test-writing exemplars",
            )
    elif mode == "not-applicable":
        for field, entries in (
            ("existing_coverage", existing_coverage),
            ("exemplars", exemplars),
            ("cases", cases),
        ):
            if entries:
                add(
                    "TEST_PLAN_MODE_CONFLICT",
                    f"/test_plan/{field}",
                    hint=f"not-applicable mode requires an empty {field} array",
                )
        criteria = normalized.get("done_criteria")
        criteria = criteria if isinstance(criteria, list) else []
        if any(isinstance(criterion, dict) and criterion.get("kind") == "test-gate" for criterion in criteria):
            add(
                "TEST_PLAN_MODE_CONFLICT",
                "/done_criteria",
                hint=("not-applicable mode must not claim a test gate; use a step gate or static invariant"),
            )
        if not any(
            isinstance(criterion, dict) and criterion.get("kind") in {"static-invariant", "step-gate"}
            for criterion in criteria
        ):
            add(
                "NON_TEST_VERIFICATION_REQUIRED",
                "/done_criteria",
                hint=(
                    "not-applicable mode requires a static-invariant or "
                    "step-gate criterion stating the exact non-test check"
                ),
            )
        for pointer, change in _changes(normalized):
            if not _not_applicable_change_allowed(change):
                add(
                    "TEST_PLAN_NOT_APPLICABLE_UNSAFE",
                    f"{pointer}/operation",
                    hint=(
                        "use existing-coverage or new-or-updated-tests for "
                        "production behavior; not-applicable is limited to "
                        "documentation, exact comment/docstring cleanup, or "
                        "deleting a redundant test"
                    ),
                )

    check_symbol_entries(existing_coverage, "/test_plan/existing_coverage", "EXISTING_COVERAGE_INVALID")
    check_symbol_entries(exemplars, "/test_plan/exemplars", "TEST_EXEMPLAR_INVALID")
    seen_test_symbols: set[tuple[str, str]] = set()
    for index, case in _object_entries(cases):
        test_file = case.get("test_file")
        test_symbol = case.get("test_symbol")
        check_path(f"/test_plan/cases/{index}/test_file", test_file)
        if isinstance(test_file, str) and isinstance(test_symbol, str):
            identity = (test_file, test_symbol)
            if identity in seen_test_symbols:
                add(
                    "DUPLICATE_TEST_SYMBOL",
                    f"/test_plan/cases/{index}/test_symbol",
                    hint=(
                        "combine cases with the same behavior into one "
                        "parameterized test, or use meaningful distinct symbols "
                        "for genuinely different behaviors"
                    ),
                )
            seen_test_symbols.add(identity)

    for pointer, ref, _ in _iter_command_refs(normalized):
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
