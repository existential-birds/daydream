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

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from tree_sitter import Language, Parser, Query, QueryCursor

from daydream import git_ops
from daydream.exploration import FileInfo
from daydream.git_ops import GitError

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
(import_from_statement module_name: (relative_import)) @import
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

# Definition queries (symbol index). Mirror the import-query style: capture the
# whole definition node as ``@def`` so we can read its field-named ``name`` node
# and its 1-based start/end lines.
PYTHON_DEF_QUERY = """
(function_definition) @def
(class_definition) @def
"""

RUST_DEF_QUERY = """
(function_item) @def
(struct_item) @def
(enum_item) @def
(trait_item) @def
(impl_item) @def
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


def _def_query_for_language(language_id: str) -> str | None:
    """Return the definition query string for a language, or None."""
    if language_id == "python":
        return PYTHON_DEF_QUERY
    if language_id == "rust":
        return RUST_DEF_QUERY
    return None


def _definition_kind(node_type: str) -> str:
    """Map a tree-sitter definition node type to ``function``/``class``."""
    if node_type in ("function_definition", "function_item"):
        return "function"
    return "class"


def extract_definitions(
    parser: Parser, source: bytes, query_string: str
) -> list[dict[str, object]]:
    """Parse ``source`` and return captured definition records.

    Returns a list of ``{name, line, end_line, kind}`` dicts where ``line``/``end_line``
    are 1-based (``start_point[0] + 1`` / ``end_point[0] + 1``). ``kind`` is
    ``"function"`` or ``"class"``. Returns an empty list on any parse/query
    failure (graceful degradation per D-06, matching ``extract_imports``).
    """
    try:
        tree = parser.parse(source)
        language = parser.language
        if language is None:
            return []
        query = Query(language, query_string)
        cursor = QueryCursor(query)
        captures = cursor.captures(tree.root_node)
        result: list[dict[str, object]] = []
        for node in captures.get("def", []):
            name_node = node.child_by_field_name("name")
            if name_node is None or name_node.text is None:
                continue
            result.append(
                {
                    "name": name_node.text.decode("utf-8", errors="replace"),
                    "line": node.start_point[0] + 1,
                    "end_line": node.end_point[0] + 1,
                    "kind": _definition_kind(node.type),
                }
            )
        return result
    except Exception:
        return []


def extract_imports(parser: Parser, source: bytes, query_string: str) -> list[str]:
    """Parse ``source`` and return decoded captured import strings.

    Args:
        parser: A tree-sitter ``Parser`` already configured for the language.
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


def _module_candidates(base: Path, dotted: str) -> list[Path]:
    """Resolve a dotted module name under `base` to .py and package candidates."""
    target = base
    for part in dotted.split("."):
        target = target / part
    return [target.with_suffix(".py"), target / "__init__.py"]


def _resolve_python_import(import_str: str, repo_root: Path, importer: Path) -> list[Path]:
    candidates: list[Path] = []
    if import_str.startswith("from "):
        # Relative component of a `from` statement; parse it to read the
        # ImportFrom level (ascent) and retained aliases (bare-relative names).
        # Best-effort: malformed/unsupported captures degrade to no candidates.
        try:
            body = ast.parse(import_str).body
        except SyntaxError:
            return []
        if len(body) != 1 or not isinstance(body[0], ast.ImportFrom):
            return []
        node = body[0]
        if node.level < 1:
            return []
        base = importer.parent
        for _ in range(node.level - 1):
            base = base.parent
        if node.module is not None:
            candidates.extend(_module_candidates(base, node.module))
        else:
            candidates.append(base / "__init__.py")
            for alias in node.names:
                if alias.name == "*":
                    continue
                candidates.extend(_module_candidates(base, alias.name))
    else:
        parts = import_str.split(".")
        candidates.extend(_module_candidates(repo_root, import_str))
        # Also try resolving the parent (e.g. `from foo.bar import baz`).
        if len(parts) >= 2:
            candidates.extend(_module_candidates(repo_root, ".".join(parts[:-1])))
    return [c for c in candidates if c.exists() and c.is_file()]


def _resolve_ts_import(import_str: str, repo_root: Path, importer: Path) -> list[Path]:
    if not import_str.startswith("."):
        return []
    base = importer.parent / import_str
    suffixes = [".ts", ".tsx", ".d.ts", ".js", ".jsx"]
    candidates: list[Path] = [base.with_suffix(s) for s in suffixes]
    candidates.extend([base / f"index{s}" for s in suffixes])
    return [c for c in candidates if c.exists() and c.is_file()]


def _build_go_package_index(repo_root: Path) -> dict[str, tuple[Path, ...]]:
    """Index Go files by their parent directory's terminal name.

    Traverses ``repo_root.rglob("*.go")`` exactly once and groups each file
    under the terminal name of its parent directory. The first directory to
    claim a terminal name wins: files under a later directory with the same
    terminal name are ignored. Best-effort: an ``OSError`` during traversal
    degrades to an empty index.
    """
    index: dict[str, list[Path]] = {}
    owners: dict[str, Path] = {}
    try:
        for candidate in repo_root.rglob("*.go"):
            if not candidate.is_file():
                continue
            parent = candidate.parent
            name = parent.name
            if not name:
                continue
            owner = owners.get(name)
            if owner is None:
                owners[name] = parent
                index[name] = [candidate]
            elif owner == parent:
                index[name].append(candidate)
    except OSError:
        return {}
    return {name: tuple(files) for name, files in index.items()}


def _resolve_go_import(import_str: str, package_index: dict[str, tuple[Path, ...]]) -> list[Path]:
    # Best-effort: look up the import's terminal component in the indexed tree.
    if not import_str:
        return []
    suffix = import_str.strip("/").split("/")[-1]
    return list(package_index.get(suffix, ()))


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


def _resolve_import(
    language_id: str,
    import_str: str,
    repo_root: Path,
    importer: Path,
    go_package_index: dict[str, tuple[Path, ...]] | None = None,
) -> list[Path]:
    if language_id == "python":
        return _resolve_python_import(import_str, repo_root, importer)
    if language_id in ("typescript", "tsx", "javascript"):
        return _resolve_ts_import(import_str, repo_root, importer)
    if language_id == "go":
        return _resolve_go_import(import_str, go_package_index or {})
    if language_id == "rust":
        return _resolve_rust_import(import_str, repo_root, importer)
    return []


# --- Reverse edges (importers) ----------------------------------------------


# Stems whose bare name is too common for a reverse-import grep to be
# meaningful: an entrypoint or ubiquitous module name matches thousands of
# files (e.g. ``app`` matches every file mentioning "app"). The static
# ``imports`` edges still capture forward dependencies precisely, and the
# dependency-tracer agent greps call sites itself, so skipping these loses
# nothing but noise -- UNLESS the file actually defines a symbol, in which case
# a reverse lookup is warranted (``_eligible_for_reverse_grep`` rescues
# generic stems whose path appears in the symbol index; issue #745).
_GENERIC_STEMS = frozenset(
    {
        "__init__", "mod", "index", "main", "app", "api", "base", "common",
        "utils", "util", "helpers", "constants", "config", "settings",
        "models", "model", "types", "type", "schema", "schemas", "client",
        "server", "service", "services", "handler", "handlers", "views",
        "urls", "test", "tests", "conftest", "setup",
    }
)

# Reverse-edge grep is a best-effort seed, not an exhaustive index. A
# legitimately widely-imported module can match hundreds of files; cap the
# seed so no single module can blow the downstream prompt's context window.
def _eligible_for_reverse_grep(path: str, defining_paths: set[str]) -> bool:
    """Whether a modified path should be part of the reverse-import grep.

    Invalid/empty stems are always skipped (they cannot be grepped). A generic
    stem (in :data:`_GENERIC_STEMS`) is skipped UNLESS the file actually
    defines a symbol per the symbol index (``defining_paths``) -- a ``config.py``
    that defines ``load_config`` deserves a reverse lookup; one that defines
    nothing stays skipped. Non-generic stems are always eligible.
    """
    stem = Path(path).stem
    if not stem or "\x00" in stem or "\r" in stem or "\n" in stem:
        return False
    if stem in _GENERIC_STEMS and path not in defining_paths:
        return False
    return True


_MAX_IMPORTERS = 40

# Restrict the reverse-edge grep to source files. A doc, plan, or config file
# cannot import a code module, so matches in them are always false positives.
_CODE_PATHSPECS: tuple[str, ...] = tuple(f"*{suffix}" for suffix in LANGUAGES)


def _build_importer_lookup(
    repo_root: Path,
    modified_paths: list[str],
    defining_paths: set[str] | None = None,
) -> dict[str, list[str]]:
    """Return one batched reverse-import lookup for all *modified_paths*.

    Feeds every changed module stem to a single :func:`git_ops.grep_fixed_matches`
    call, then groups the ``(path, pattern)`` pairs per modified path. Each
    modified path's list excludes that exact path and is capped at
    :data:`_MAX_IMPORTERS`, so paths sharing a stem keep separate, per-path
    caps. ``defining_paths`` names the modified paths that define at least one
    symbol per the symbol index; a generic-stem path in this set is rescued
    from the generic-stem skip (issue #745). Best-effort: a ``GitError`` (or no
    usable stems) degrades to empty lists with zero git calls.
    """
    defining_paths = defining_paths or set()
    lookup: dict[str, list[str]] = {path: [] for path in modified_paths}
    unique_stems: list[str] = []
    seen_stems: set[str] = set()
    for path in modified_paths:
        if not _eligible_for_reverse_grep(path, defining_paths):
            continue
        stem = Path(path).stem
        if stem in seen_stems:
            continue
        seen_stems.add(stem)
        unique_stems.append(stem)
    if not unique_stems:
        return lookup

    try:
        pairs = git_ops.grep_fixed_matches(
            repo_root, unique_stems, word=True, pathspecs=_CODE_PATHSPECS
        )
    except GitError:
        return lookup

    by_stem: dict[str, list[str]] = {}
    for path, matched in pairs:
        if matched not in seen_stems:
            continue
        bucket = by_stem.setdefault(matched, [])
        if path not in bucket:
            bucket.append(path)
    for path in modified_paths:
        if not _eligible_for_reverse_grep(path, defining_paths):
            continue
        stem = Path(path).stem
        lookup[path] = [p for p in by_stem.get(stem, ()) if p != path][:_MAX_IMPORTERS]
    return lookup


# --- Public API --------------------------------------------------------------


def build_symbol_index(repo_root: Path, paths: list[str]) -> dict[str, list[dict[str, object]]]:
    """Build a symbol index of function/class definitions for ``paths``.

    Only Python and Rust sources are indexed (the issue's symbol scope). Each
    path is resolved relative to ``repo_root``; a file that parses or queries
    with no definitions simply contributes nothing (graceful degradation per
    D-06).

    Returns ``{name: [{"path", "line", "end_line", "kind"}]}`` keyed by
    definition name (a name can be defined in multiple files).
    """
    index: dict[str, list[dict[str, object]]] = {}
    for path in paths:
        lang_entry = LANGUAGES.get(Path(path).suffix)
        if lang_entry is None:
            continue
        language_id, _factory = lang_entry
        query_string = _def_query_for_language(language_id)
        if query_string is None:
            continue
        abs_path = repo_root / path
        try:
            source = abs_path.read_bytes()
        except (FileNotFoundError, OSError):
            continue
        parser = get_parser(language_id)
        if parser is None:
            continue
        for definition in extract_definitions(parser, source, query_string):
            index.setdefault(str(definition["name"]), []).append(
                {
                    "path": path,
                    "line": definition["line"],
                    "end_line": definition["end_line"],
                    "kind": definition["kind"],
                }
            )
    return index


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
    reverse_paths = [
        entry.path
        for entry in entries
        if entry.status != "D" and Path(entry.path).suffix in LANGUAGES
    ]
    # A generic-stem file (e.g. ``config``) is rescued from the reverse-import
    # skip when it actually defines a symbol (issue #745). The defining set is
    # derived from the definition queries run in this single forward pass, so
    # the changed files are read and parsed only once -- no separate
    # ``build_symbol_index`` pass re-reading and re-parsing the same file set.
    defining_paths: set[str] = set()
    go_package_index: dict[str, tuple[Path, ...]] | None = None

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
        if parser is None:
            continue

        # A generic-stem path that actually defines a symbol is rescued from the
        # generic-stem reverse-import skip below (issue #745).
        def_query = _def_query_for_language(language_id)
        if def_query is not None and extract_definitions(parser, source, def_query):
            defining_paths.add(entry.path)

        query_string = _query_for_language(language_id)
        if query_string is None:
            continue

        imports = extract_imports(parser, source, query_string)
        if language_id == "go" and imports and go_package_index is None:
            go_package_index = _build_go_package_index(repo_root)
        for imp in imports:
            resolved_paths = _resolve_import(language_id, imp, repo_root, abs_path, go_package_index)
            for resolved in resolved_paths:
                try:
                    rel = resolved.resolve().relative_to(repo_root.resolve())
                except (ValueError, OSError):
                    continue
                _add(str(rel), "imports")

    # Reverse edges need the complete defining set, so the batched grep runs
    # after the forward pass (whose reads already covered every file once).
    importers_by_path = _build_importer_lookup(repo_root, reverse_paths, defining_paths)
    for path, importers in importers_by_path.items():
        for importer in importers:
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
