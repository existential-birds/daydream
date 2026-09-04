"""Tree-sitter-backed static import resolution for the exploration phase.

Provides ``detect_affected_files()``: a pure, synchronous function that turns a
git diff into the impact surface (changed files + 1-hop imports/importers) for
Python, TypeScript/TSX/JavaScript, Go, and Rust. No async, no Backend, no UI.

Adding a new language requires only:
    1. A lazy factory function returning a tree_sitter ``Language``.
    2. One entry in the ``LANGUAGES`` dict.
    3. One query constant + a branch in ``_query_for_language``.
    4. A resolver branch in ``_resolve_import``.

Also provides the grounding primitives the diagram phase needs
(:func:`language_for_path`, :func:`branch_statement_lines`,
:func:`is_branch_line`, :func:`is_terminal_line`, :func:`definitions_in_file`
plus the :data:`BRANCH_NODE_TYPES`/:data:`TERMINAL_NODE_TYPES` tables). All of
them fail open -- an unknown language, a missing file or a parse failure yields
an empty/false answer -- except :exc:`TreeSitterBadVersionError`, which
:func:`get_parser` raises unwrapped so a known-bad install can never parse
natively; callers that must survive that wrap the call.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from tree_sitter import Language, Node, Parser, Query, QueryCursor

from daydream import git_ops
from daydream._tree_sitter_safety import assert_tree_sitter_safe
from daydream.exploration import FileInfo
from daydream.git_ops import GitError

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
    # Single shared choke point for every native-analysis entry point: a
    # known-bad installed tree-sitter raises before any ``Language``/``Parser``
    # construction, so no native parsing happens on a bad install. No-op on
    # good installs; errors propagate to the caller unwrapped.
    assert_tree_sitter_safe()
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
# NOTE: ``(impl_item) @def`` above can never yield a record -- ``impl_item`` has no
# ``name`` field, so ``extract_definitions`` drops it. Left as-is deliberately:
# changing the shared query would change reverse-import-edge policy (issue #1113).

# --- Diagram definition queries (issue #1113) --------------------------------
#
# A SEPARATE query family from ``_def_query_for_language`` above. The shared
# query feeds ``detect_affected_files``' ``defining_paths`` gate, which decides
# whether a generic-stem path earns reverse-import edges; widening it to
# TypeScript/Go would silently add those edges for every ``index.ts``/``main.go``
# in every repo. Diagram grounding needs the wide surface, so it gets its own
# query plus its own kind mapper (``_diagram_definition_kind``) rather than
# inheriting ``_definition_kind``'s ``else: return "class"`` fallback.
#
# Every pattern below was confirmed to compile against the installed grammars and
# to place ``@def`` on a node whose ``name`` field resolves (otherwise
# ``extract_definitions`` would silently drop it).

# The same string is valid for the ``typescript``, ``tsx`` and ``javascript``
# ids -- all three are served by the tree-sitter-typescript grammars.
TYPESCRIPT_DIAGRAM_DEF_QUERY = """
(function_declaration name: (identifier)) @def
(generator_function_declaration name: (identifier)) @def
(function_signature name: (identifier)) @def
(method_definition name: (property_identifier)) @def
(method_signature name: (property_identifier)) @def
(class_declaration name: (type_identifier)) @def
(abstract_class_declaration name: (type_identifier)) @def
(interface_declaration name: (type_identifier)) @def
(type_alias_declaration name: (type_identifier)) @def
(enum_declaration name: (identifier)) @def
(variable_declarator name: (identifier) value: [(arrow_function) (function_expression)]) @def
"""

GO_DIAGRAM_DEF_QUERY = """
(function_declaration name: (identifier)) @def
(method_declaration name: (field_identifier)) @def
(type_spec name: (type_identifier)) @def
(type_alias name: (type_identifier)) @def
"""

RUST_DIAGRAM_DEF_QUERY = """
(function_item) @def
(function_signature_item) @def
(struct_item) @def
(enum_item) @def
(trait_item) @def
(union_item) @def
(type_item) @def
(mod_item) @def
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


def _diagram_def_query_for_language(language_id: str) -> str | None:
    """Return the *diagram* definition query for a language, or None.

    Deliberately separate from :func:`_def_query_for_language`: see the comment
    above :data:`TYPESCRIPT_DIAGRAM_DEF_QUERY`.
    """
    if language_id == "python":
        return PYTHON_DEF_QUERY
    if language_id in ("typescript", "tsx", "javascript"):
        return TYPESCRIPT_DIAGRAM_DEF_QUERY
    if language_id == "go":
        return GO_DIAGRAM_DEF_QUERY
    if language_id == "rust":
        return RUST_DIAGRAM_DEF_QUERY
    return None


def _definition_kind(node_type: str) -> str:
    """Map a tree-sitter definition node type to ``function``/``class``."""
    if node_type in ("function_definition", "function_item"):
        return "function"
    return "class"


# Kind buckets for the diagram definition queries. Kept separate from
# :func:`_definition_kind`, whose ``else: return "class"`` fallback would stamp
# ``method_definition``/``variable_declarator``/``type_spec`` as ``"class"``.
_DIAGRAM_FUNCTION_NODES = frozenset(
    {
        "function_definition",
        "function_declaration",
        "generator_function_declaration",
        "function_signature",
        "method_definition",
        "method_signature",
        "method_declaration",
        "variable_declarator",
        "function_item",
        "function_signature_item",
    }
)
_DIAGRAM_CLASS_NODES = frozenset(
    {
        "class_definition",
        "class_declaration",
        "abstract_class_declaration",
        "interface_declaration",
        "enum_declaration",
        "struct_item",
        "enum_item",
        "trait_item",
        "union_item",
    }
)
_DIAGRAM_TYPE_NODES = frozenset(
    {"type_alias_declaration", "type_spec", "type_alias", "type_item"}
)


def _diagram_definition_kind(node_type: str) -> str:
    """Map a diagram definition node type to a kind label.

    Returns ``"function"``, ``"class"``, ``"type"``, ``"module"``, or
    ``"other"`` for an unrecognized node type (fail-open: an unmapped capture
    still yields a usable definition record, just an unclassified one).
    """
    if node_type in _DIAGRAM_FUNCTION_NODES:
        return "function"
    if node_type in _DIAGRAM_CLASS_NODES:
        return "class"
    if node_type in _DIAGRAM_TYPE_NODES:
        return "type"
    if node_type == "mod_item":
        return "module"
    return "other"


def extract_definitions(
    parser: Parser,
    source: bytes,
    query_string: str,
    *,
    kind_for: Callable[[str], str] = _definition_kind,
) -> list[dict[str, object]]:
    """Parse ``source`` and return captured definition records.

    Returns a list of ``{name, line, end_line, kind}`` dicts where ``line``/``end_line``
    are 1-based (``start_point[0] + 1`` / ``end_point[0] + 1``). ``kind`` comes from
    ``kind_for`` (default :func:`_definition_kind`, i.e. ``"function"``/``"class"``;
    :func:`definitions_in_file` passes :func:`_diagram_definition_kind`). Returns an
    empty list on any parse/query failure (graceful degradation per D-06, matching
    ``extract_imports``).
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
                    "kind": kind_for(node.type),
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


# --- Control-flow tables (issue #1113) ---------------------------------------
#
# Every node type below was confirmed by parsing purpose-built snippets with the
# installed grammars and by compiling ``(<type>) @x`` as a real ``Query`` against
# the language. Nothing here may be extended by analogy: a node kind that does
# not exist makes the whole query fail to compile.
#
# ``.js`` is parsed by the TypeScript grammar and ``.jsx`` by the TSX grammar
# (there is no ``tree_sitter_javascript`` installed), so the three JS/TS ids share
# one table.

_TYPESCRIPT_BRANCH_NODES = frozenset(
    {
        "if_statement",
        "switch_statement",
        "switch_case",
        "switch_default",
        "try_statement",
        "catch_clause",
        "finally_clause",
        "for_statement",
        "for_in_statement",
        "while_statement",
        "do_statement",
        "ternary_expression",
        "optional_chain",
    }
)

# Node types that open a control-flow branch. Head nodes only: ``else_clause`` is
# deliberately absent (an ``else`` adds no new condition, matching
# ``eval/analyzer._CC_DECISION_TYPES``), which is also what keeps an
# if/else-if/else chain at N decisions instead of 2N -- see
# :func:`branch_statement_lines`.
BRANCH_NODE_TYPES: dict[str, frozenset[str]] = {
    "python": frozenset(
        {
            "if_statement",
            "elif_clause",
            "for_statement",
            "while_statement",
            "try_statement",
            "except_clause",
            "finally_clause",
            "with_statement",
            "match_statement",
            "case_clause",
            "conditional_expression",
            "boolean_operator",
            "assert_statement",
        }
    ),
    "typescript": _TYPESCRIPT_BRANCH_NODES,
    "tsx": _TYPESCRIPT_BRANCH_NODES,
    "javascript": _TYPESCRIPT_BRANCH_NODES,
    "go": frozenset(
        {
            "if_statement",
            "for_statement",
            "expression_switch_statement",
            "type_switch_statement",
            "select_statement",
            "expression_case",
            "default_case",
            "type_case",
            "communication_case",
        }
    ),
    "rust": frozenset(
        {
            "if_expression",
            "match_expression",
            "match_arm",
            "while_expression",
            "for_expression",
            "loop_expression",
            "let_condition",
            "try_expression",
        }
    ),
}

# Statements that end a control-flow path. ``break``/``continue``/``goto``/
# ``fallthrough`` are non-local *jumps*, not terminals, and are excluded.
TERMINAL_NODE_TYPES: dict[str, frozenset[str]] = {
    "python": frozenset({"return_statement", "raise_statement"}),
    "typescript": frozenset({"return_statement", "throw_statement"}),
    "tsx": frozenset({"return_statement", "throw_statement"}),
    "javascript": frozenset({"return_statement", "throw_statement"}),
    "go": frozenset({"return_statement"}),
    "rust": frozenset({"return_expression"}),
}

# Exit-like calls have no dedicated node kind, so they are matched on the text of
# the call's ``function:`` child (``attribute`` for python, ``member_expression``
# for TS, ``selector_expression`` for go, ``scoped_identifier`` for rust).
TERMINAL_CALL_NAMES: dict[str, frozenset[str]] = {
    "python": frozenset({"sys.exit", "os._exit", "exit", "quit"}),
    "typescript": frozenset({"process.exit"}),
    "tsx": frozenset({"process.exit"}),
    "javascript": frozenset({"process.exit"}),
    "go": frozenset(
        {"os.Exit", "panic", "log.Fatal", "log.Fatalf", "log.Panic", "t.Fatal", "t.Fatalf"}
    ),
    "rust": frozenset(
        {"std::process::exit", "process::exit", "std::process::abort", "process::abort"}
    ),
}

# Rust's panic family is a ``macro_invocation``, which on its own is far too broad
# to be a terminal (``assert!``, ``println!`` and ``vec!`` are all
# ``macro_invocation``), so the ``macro:`` name check below is mandatory.
TERMINAL_MACRO_NAMES: frozenset[str] = frozenset(
    {"panic", "unreachable", "todo", "unimplemented"}
)

_CALL_NODE_TYPES: dict[str, frozenset[str]] = {
    "python": frozenset({"call"}),
    "typescript": frozenset({"call_expression"}),
    "tsx": frozenset({"call_expression"}),
    "javascript": frozenset({"call_expression"}),
    "go": frozenset({"call_expression"}),
    "rust": frozenset({"call_expression"}),
}

# Switch-like containers and the case nodes they own. Counting both levels would
# report one decision twice, so :func:`branch_statement_lines` reports the case
# level and drops the container whenever the container actually has cases. Go has
# four container kinds and shares ``default_case`` across three of them, which is
# why the relation is expressed as two flat per-language sets rather than pairs.
_SWITCH_CONTAINER_TYPES: dict[str, frozenset[str]] = {
    "python": frozenset({"match_statement"}),
    "typescript": frozenset({"switch_statement"}),
    "tsx": frozenset({"switch_statement"}),
    "javascript": frozenset({"switch_statement"}),
    "go": frozenset(
        {"expression_switch_statement", "type_switch_statement", "select_statement"}
    ),
    "rust": frozenset({"match_expression"}),
}
_SWITCH_CASE_TYPES: dict[str, frozenset[str]] = {
    "python": frozenset({"case_clause"}),
    "typescript": frozenset({"switch_case", "switch_default"}),
    "tsx": frozenset({"switch_case", "switch_default"}),
    "javascript": frozenset({"switch_case", "switch_default"}),
    "go": frozenset({"expression_case", "default_case", "type_case", "communication_case"}),
    "rust": frozenset({"match_arm"}),
}
# Body nodes that sit between a switch-like container and its cases: ``block``
# (python ``match``), ``switch_body`` (TS), ``match_block`` (rust). Go's cases are
# direct children of their container.
_SWITCH_BODY_TYPES: frozenset[str] = frozenset({"block", "switch_body", "match_block"})

# Fallback line classifiers for a file whose language has no grammar here. Pure
# text, no parse -- deliberately crude, and only ever reached when
# ``language_id`` is None/unknown or the parser is unavailable.
_BRANCH_KEYWORD_RE = re.compile(
    r"^\s*(if|elif|else|for|while|match|case|try|except|switch|catch|loop)\b"
)
_TERMINAL_KEYWORD_RE = re.compile(
    r"^\s*(return|raise|throw|panic|os\.Exit|std::process::exit|sys\.exit)\b"
)


def language_for_path(path: str) -> str | None:
    """Return the tree-sitter language id for ``path``, or None if unsupported.

    Suffix matching is case-sensitive (``LANGUAGES`` is keyed by exact suffix).
    Note ``.js`` maps to the id ``"javascript"`` but is parsed by the TypeScript
    grammar, and ``.jsx`` maps to ``"tsx"``.
    """
    entry = LANGUAGES.get(Path(path).suffix)
    if entry is None:
        return None
    return entry[0]


def _walk(node: Node) -> Iterator[Node]:
    """Yield ``node`` and every descendant (order unspecified)."""
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(current.children)


def _container_owns_cases(node: Node, case_types: frozenset[str]) -> bool:
    """Whether a switch-like ``node`` has at least one case of its own.

    Looks at direct children and through one body wrapper
    (:data:`_SWITCH_BODY_TYPES`) -- the only two shapes the grammars produce. A
    nested switch's cases live under that switch's own body, so they are never
    mistaken for the outer container's.
    """
    for child in node.children:
        if child.type in case_types:
            return True
        if child.type in _SWITCH_BODY_TYPES:
            for grandchild in child.children:
                if grandchild.type in case_types:
                    return True
    return False


def branch_statement_lines(language_id: str, source: bytes) -> list[int]:
    """Return the 1-based lines of ``source`` that open a control-flow branch.

    Sorted and deduplicated. Returns ``[]`` for an unknown language or on any
    parse/query failure (fail-open).

    Two counting rules are applied so one decision point is never reported
    twice:

    * ``else``/``else if`` chains contribute only their condition-bearing head
      nodes. ``else_clause`` is absent from :data:`BRANCH_NODE_TYPES` entirely.
      In python the chain is flat (``elif_clause`` is a *sibling* of the
      ``if_statement``) while in TS/TSX/JS and rust it nests (the second
      ``if_statement``/``if_expression`` lives inside the first
      ``else_clause``) -- both shapes yield one line per condition because the
      nested head starts on the same physical line as its ``else``. Go has no
      ``else`` node at all; its ``alternative:`` field holds a ``block`` or a
      nested ``if_statement`` directly.
    * A switch-like container (``match``/``switch``/``select``) is reported only
      when it owns no cases; otherwise its cases are reported instead. This
      holds for nested switches too, each level being resolved independently.

    Raises:
        TreeSitterBadVersionError: propagated from :func:`get_parser` when the
            installed tree-sitter is in the known-bad set. Callers that must not
            fail on a bad install wrap this call.
    """
    branch_types = BRANCH_NODE_TYPES.get(language_id)
    if branch_types is None:
        return []
    parser = get_parser(language_id)
    if parser is None:
        return []
    containers = _SWITCH_CONTAINER_TYPES.get(language_id, frozenset())
    case_types = _SWITCH_CASE_TYPES.get(language_id, frozenset())
    lines: set[int] = set()
    try:
        tree = parser.parse(source)
        for node in _walk(tree.root_node):
            if node.type not in branch_types:
                continue
            if node.type in containers and _container_owns_cases(node, case_types):
                continue
            lines.add(node.start_point[0] + 1)
    except Exception:
        return []
    return sorted(lines)


def _source_line(source: bytes, line: int) -> str:
    """Return the 1-based ``line`` of ``source`` as text, or ``""`` if absent."""
    if line < 1:
        return ""
    rows = source.split(b"\n")
    if line > len(rows):
        return ""
    return rows[line - 1].decode("utf-8", errors="replace")


def _terminal_call_name(node: Node) -> str | None:
    """Return the callee text of a call ``node``, or None when unreadable."""
    callee = node.child_by_field_name("function")
    if callee is None or callee.text is None:
        return None
    return callee.text.decode("utf-8", errors="replace").strip()


def is_branch_line(language_id: str | None, source: bytes, line: int) -> bool:
    """Whether the 1-based ``line`` of ``source`` opens a control-flow branch.

    Unlike :func:`branch_statement_lines` this is a membership question, not a
    count, so no de-duplication rule applies: a ``switch (x) {`` line is a branch
    line even though :func:`branch_statement_lines` reports its cases instead.

    Falls back to a keyword regex over the raw line when ``language_id`` is
    None/unknown, when no parser is available, or when parsing fails (fail-open).

    Raises:
        TreeSitterBadVersionError: propagated from :func:`get_parser`; callers
            that must not fail on a bad install wrap this call.
    """
    branch_types = BRANCH_NODE_TYPES.get(language_id or "")
    parser = get_parser(language_id) if language_id else None
    if branch_types is None or parser is None:
        return bool(_BRANCH_KEYWORD_RE.match(_source_line(source, line)))
    try:
        tree = parser.parse(source)
        for node in _walk(tree.root_node):
            if node.type in branch_types and node.start_point[0] + 1 == line:
                return True
    except Exception:
        return bool(_BRANCH_KEYWORD_RE.match(_source_line(source, line)))
    return False


def is_terminal_line(language_id: str | None, source: bytes, line: int) -> bool:
    """Whether the 1-based ``line`` of ``source`` ends a control-flow path.

    True for a ``return``/``raise``/``throw`` statement, for an exit-like call in
    :data:`TERMINAL_CALL_NAMES` (matched on the callee text -- those calls have no
    dedicated node kind), and for a rust panic-family ``macro_invocation`` named
    in :data:`TERMINAL_MACRO_NAMES`.

    Falls back to a keyword regex over the raw line when ``language_id`` is
    None/unknown, when no parser is available, or when parsing fails (fail-open).

    Raises:
        TreeSitterBadVersionError: propagated from :func:`get_parser`; callers
            that must not fail on a bad install wrap this call.
    """
    terminal_types = TERMINAL_NODE_TYPES.get(language_id or "")
    parser = get_parser(language_id) if language_id else None
    if terminal_types is None or parser is None:
        return bool(_TERMINAL_KEYWORD_RE.match(_source_line(source, line)))
    call_types = _CALL_NODE_TYPES.get(language_id or "", frozenset())
    call_names = TERMINAL_CALL_NAMES.get(language_id or "", frozenset())
    try:
        tree = parser.parse(source)
        for node in _walk(tree.root_node):
            if node.start_point[0] + 1 != line:
                continue
            if node.type in terminal_types:
                return True
            if node.type in call_types and _terminal_call_name(node) in call_names:
                return True
            if node.type == "macro_invocation":
                macro = node.child_by_field_name("macro")
                if macro is not None and macro.text is not None:
                    name = macro.text.decode("utf-8", errors="replace").strip()
                    if name in TERMINAL_MACRO_NAMES:
                        return True
    except Exception:
        return bool(_TERMINAL_KEYWORD_RE.match(_source_line(source, line)))
    return False


def definitions_in_file(repo_root: Path, path: str) -> list[dict[str, object]]:
    """Return the definitions in ``repo_root / path`` for diagram grounding.

    Each record is ``{"name", "line", "end_line", "kind"}`` with 1-based lines and
    a kind from :func:`_diagram_definition_kind`. Returns ``[]`` for a missing or
    unreadable file, for an unsupported language, and on any parse/query failure
    (fail-open).

    Note: a definition node's range spans its **body**, not its header (a
    decorator sits outside it), and two definitions can share one physical line
    (``{ meth() {}, get g() {} }``) -- so a line number alone is not a unique key.

    Raises:
        TreeSitterBadVersionError: propagated from :func:`get_parser`; callers
            that must not fail on a bad install wrap this call.
    """
    language_id = language_for_path(path)
    if language_id is None:
        return []
    query_string = _diagram_def_query_for_language(language_id)
    if query_string is None:
        return []
    try:
        source = (repo_root / path).read_bytes()
    except OSError:
        return []
    parser = get_parser(language_id)
    if parser is None:
        return []
    return extract_definitions(
        parser, source, query_string, kind_for=_diagram_definition_kind
    )


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
        TreeSitterBadVersionError: If the installed tree-sitter version is in
            the known-bad set (issue #1087): the index consumer refuses to
            construct a parser that could SIGSEGV the process.
    """
    # Single shared choke point for every native-analysis entry point: a
    # known-bad installed tree-sitter raises before any ``Language``/``Parser``
    # construction, so no native parsing happens on a bad install. No-op on
    # good installs; errors propagate to the caller unwrapped.
    assert_tree_sitter_safe()

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
