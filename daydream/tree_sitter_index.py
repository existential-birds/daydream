"""Tree-sitter-backed static import resolution for the exploration phase.

Provides ``detect_affected_files()``: a pure, synchronous function that turns a
git diff into the impact surface (changed files + 1-hop imports/importers) for
Python, TypeScript/TSX/JavaScript, Go, and Rust. No async, no Backend, no UI.

Adding a new language requires only:
    1. A lazy factory function returning a tree_sitter ``Language``.
    2. One entry in the ``LANGUAGES`` dict.
    3. One query constant + a branch in ``_query_for_language``.
    4. A resolver branch in ``_resolve_import``.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from tree_sitter import Language, Parser, Query, QueryCursor

from daydream.exploration import FileInfo

if TYPE_CHECKING:
    pass


# --- Lazy language factories -------------------------------------------------


def _python_lang() -> Language:
    import tree_sitter_python

    return Language(tree_sitter_python.language())


def _typescript_lang() -> Language:
    import tree_sitter_typescript

    return Language(tree_sitter_typescript.language_typescript())


def _tsx_lang() -> Language:
    import tree_sitter_typescript

    return Language(tree_sitter_typescript.language_tsx())


def _go_lang() -> Language:
    import tree_sitter_go

    return Language(tree_sitter_go.language())


def _rust_lang() -> Language:
    import tree_sitter_rust

    return Language(tree_sitter_rust.language())


# --- Registry ----------------------------------------------------------------

LANGUAGES: dict[str, tuple[str, Callable[[], Language]]] = {
    ".py": ("python", _python_lang),
    ".ts": ("typescript", _typescript_lang),
    ".tsx": ("tsx", _tsx_lang),
    ".js": ("javascript", _typescript_lang),
    ".jsx": ("tsx", _tsx_lang),
    ".go": ("go", _go_lang),
    ".rs": ("rust", _rust_lang),
}

_PARSER_CACHE: dict[str, Parser] = {}


def get_parser(language_id: str) -> Parser | None:
    """Return a cached ``Parser`` for the given language id, or None."""
    if language_id in _PARSER_CACHE:
        return _PARSER_CACHE[language_id]
    factory: Callable[[], Language] | None = None
    for _, (lid, fac) in LANGUAGES.items():
        if lid == language_id:
            factory = fac
            break
    if factory is None:
        return None
    try:
        parser = Parser(factory())
    except Exception:
        return None
    _PARSER_CACHE[language_id] = parser
    return parser


# --- Query strings -----------------------------------------------------------

PYTHON_IMPORT_QUERY = """
(import_statement name: (dotted_name) @import)
(import_from_statement module_name: (dotted_name) @import)
(import_from_statement module_name: (relative_import) @import)
"""

TYPESCRIPT_IMPORT_QUERY = """
(import_statement source: (string) @import)
(call_expression
  function: (identifier) @fn
  arguments: (arguments (string) @import)
  (#eq? @fn "require"))
"""

GO_IMPORT_QUERY = """
(import_spec path: (interpreted_string_literal) @import)
"""

RUST_IMPORT_QUERY = """
(use_declaration argument: (_) @import)
"""


def _query_for_language(language_id: str) -> str | None:
    """Return the import query string for the given language id, or None."""
    if language_id == "python":
        return PYTHON_IMPORT_QUERY
    if language_id in ("typescript", "tsx", "javascript"):
        return TYPESCRIPT_IMPORT_QUERY
    if language_id == "go":
        return GO_IMPORT_QUERY
    if language_id == "rust":
        return RUST_IMPORT_QUERY
    return None


def extract_imports(parser: Parser, source: bytes, query_string: str) -> list[str]:
    """Parse ``source`` and return decoded captured import strings.

    Args:
        parser: A tree-sitter ``Parser`` already configured for the language.
        source: Raw file bytes to parse.
        query_string: A tree-sitter Query S-expression with ``@import`` captures.

    Returns:
        Decoded import strings, with surrounding quotes stripped. Returns an
        empty list on any parse/query failure (graceful degradation per D-06).
    """
    try:
        tree = parser.parse(source)
        language = parser.language
        if language is None:
            return []
        query = Query(language, query_string)
        cursor = QueryCursor(query)
        captures = cursor.captures(tree.root_node)
        results: list[str] = []
        nodes = captures.get("import", [])
        for node in nodes:
            text = node.text
            if text is None:
                continue
            decoded = text.decode("utf-8", errors="replace").strip().strip("\"'")
            if decoded:
                results.append(decoded)
        return results
    except Exception:
        return []


# --- Diff parsing ------------------------------------------------------------


@dataclass
class _DiffEntry:
    status: str  # "A", "M", "D", "R"
    path: str


def _parse_diff_name_status(diff_text: str) -> list[_DiffEntry]:
    """Extract (status, path) pairs from a unified git diff."""
    entries: list[_DiffEntry] = []
    lines = diff_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("diff --git "):
            # Parse `diff --git a/PATH b/PATH`
            parts = line.split(" b/", 1)
            path: str | None = None
            if len(parts) == 2:
                path = parts[1].strip()
            status = "M"
            # Look ahead for status hints in the next few lines.
            j = i + 1
            while j < len(lines) and not lines[j].startswith("diff --git "):
                hint = lines[j]
                if hint.startswith("new file mode"):
                    status = "A"
                elif hint.startswith("deleted file mode"):
                    status = "D"
                elif hint.startswith("rename to "):
                    status = "R"
                    path = hint[len("rename to ") :].strip()
                if hint.startswith("@@"):
                    break
                j += 1
            if path:
                entries.append(_DiffEntry(status=status, path=path))
        i += 1
    return entries


# --- Import resolution -------------------------------------------------------


def _resolve_python_import(import_str: str, repo_root: Path, importer: Path) -> list[Path]:
    candidates: list[Path] = []
    if import_str.startswith("."):
        # Relative import; resolve against importer's package directory.
        base = importer.parent
        cleaned = import_str.lstrip(".")
        parts = cleaned.split(".") if cleaned else []
        target = base
        for part in parts:
            target = target / part
        candidates.extend([target.with_suffix(".py"), target / "__init__.py"])
    else:
        parts = import_str.split(".")
        target = repo_root
        for part in parts:
            target = target / part
        candidates.extend([target.with_suffix(".py"), target / "__init__.py"])
        # Also try resolving the parent (e.g. `from foo.bar import baz`).
        if len(parts) >= 2:
            parent = repo_root
            for part in parts[:-1]:
                parent = parent / part
            candidates.extend([parent.with_suffix(".py"), parent / "__init__.py"])
    return [c for c in candidates if c.exists() and c.is_file()]


def _resolve_ts_import(import_str: str, repo_root: Path, importer: Path) -> list[Path]:
    if not import_str.startswith("."):
        return []
    base = importer.parent / import_str
    suffixes = [".ts", ".tsx", ".d.ts", ".js", ".jsx"]
    candidates: list[Path] = [base.with_suffix(s) for s in suffixes]
    candidates.extend([base / f"index{s}" for s in suffixes])
    return [c for c in candidates if c.exists() and c.is_file()]


def _resolve_go_import(import_str: str, repo_root: Path, importer: Path) -> list[Path]:
    # Best-effort: walk repo for a directory whose suffix matches the import path.
    if not import_str:
        return []
    parts = import_str.strip("/").split("/")
    suffix = parts[-1]
    matches: list[Path] = []
    try:
        for candidate in repo_root.rglob(suffix):
            if candidate.is_dir():
                # Confirm at least one .go file lives in it.
                if any(candidate.glob("*.go")):
                    matches.extend(candidate.glob("*.go"))
                    break
    except OSError:
        return []
    return matches


def _resolve_rust_import(import_str: str, repo_root: Path, importer: Path) -> list[Path]:
    if import_str.startswith("std::") or "::" not in import_str:
        return []
    if import_str.startswith("crate::"):
        rest = import_str[len("crate::") :]
    else:
        rest = import_str
    parts = rest.split("::")
    # Drop the trailing item name (often a type/fn) and try the module path.
    module_parts = parts[:-1] if len(parts) > 1 else parts
    if not module_parts:
        return []
    src = repo_root / "src"
    target = src
    for part in module_parts:
        target = target / part
    candidates = [target.with_suffix(".rs"), target / "mod.rs"]
    return [c for c in candidates if c.exists() and c.is_file()]


def _resolve_import(language_id: str, import_str: str, repo_root: Path, importer: Path) -> list[Path]:
    if language_id == "python":
        return _resolve_python_import(import_str, repo_root, importer)
    if language_id in ("typescript", "tsx", "javascript"):
        return _resolve_ts_import(import_str, repo_root, importer)
    if language_id == "go":
        return _resolve_go_import(import_str, repo_root, importer)
    if language_id == "rust":
        return _resolve_rust_import(import_str, repo_root, importer)
    return []


# --- Reverse edges (importers) ----------------------------------------------


def _find_importers(repo_root: Path, modified_path: str) -> list[str]:
    """Best-effort `git grep` for files that mention the modified file's stem."""
    stem = Path(modified_path).stem
    if not stem or stem in {"__init__", "mod", "index"}:
        return []
    try:
        # `git -C REPO grep -l -- STEM` -- args fully controlled, not user input.
        result = subprocess.run(  # noqa: S603 # hardcoded args, no shell, repo_root from caller
            ["git", "-C", str(repo_root), "grep", "-l", "--", stem],
            capture_output=True,
            text=True,
            timeout=10,
            shell=False,
            check=False,
        )
    except (subprocess.SubprocessError, OSError):
        return []
    if result.returncode not in (0, 1):
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip() and line.strip() != modified_path]


# --- Public API --------------------------------------------------------------


def detect_affected_files(
    diff_text: str,
    repo_root: Path,
    depth: int = 1,
) -> list[FileInfo]:
    """Return changed files plus their 1-hop import dependencies.

    Args:
        diff_text: Raw output of ``git diff`` (unified format).
        repo_root: Repository root used for resolving import paths on disk.
        depth: Reserved for future multi-hop tracing. Only ``depth=1`` is
            supported. Passing any other value raises ``NotImplementedError``.

    Returns:
        A list of ``FileInfo`` entries containing the modified files (always)
        plus their direct imports (``role="imports"``) and direct importers
        (``role="imported_by"``). Deduplicated by ``(path, role)``.

    Raises:
        NotImplementedError: If ``depth != 1``.
    """
    if depth != 1:
        raise NotImplementedError("depth > 1 reserved for future use")

    results: list[FileInfo] = []
    seen: set[tuple[str, str]] = set()

    def _add(path: str, role: str) -> None:
        key = (path, role)
        if key in seen:
            return
        seen.add(key)
        results.append(FileInfo(path=path, role=role))

    entries = _parse_diff_name_status(diff_text)

    for entry in entries:
        _add(entry.path, "modified")

        if entry.status == "D":
            continue

        suffix = Path(entry.path).suffix
        lang_entry = LANGUAGES.get(suffix)
        if lang_entry is None:
            continue
        language_id, _factory = lang_entry

        abs_path = repo_root / entry.path
        try:
            source = abs_path.read_bytes()
        except (FileNotFoundError, OSError):
            continue

        parser = get_parser(language_id)
        query_string = _query_for_language(language_id)
        if parser is None or query_string is None:
            continue

        imports = extract_imports(parser, source, query_string)
        for imp in imports:
            resolved_paths = _resolve_import(language_id, imp, repo_root, abs_path)
            for resolved in resolved_paths:
                try:
                    rel = resolved.resolve().relative_to(repo_root.resolve())
                except (ValueError, OSError):
                    continue
                _add(str(rel), "imports")

        for importer in _find_importers(repo_root, entry.path):
            _add(importer, "imported_by")

    return results


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    samples = {
        "python": (b"import os\nfrom collections import abc\n", PYTHON_IMPORT_QUERY),
        "typescript": (b'import { x } from "./y";\n', TYPESCRIPT_IMPORT_QUERY),
        "go": (b'package main\nimport "example.com/m"\n', GO_IMPORT_QUERY),
        "rust": (b"use crate::models::User;\n", RUST_IMPORT_QUERY),
    }
    for lid, (src, q) in samples.items():
        parser = get_parser(lid)
        if parser is None:
            print(f"{lid}: no parser")
            continue
        print(f"{lid}: {extract_imports(parser, src, q)}")
