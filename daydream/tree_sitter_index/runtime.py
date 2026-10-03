"""Safe native parser construction, language queries, and definition capture."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from tree_sitter import Language, Node, Parser, Query, QueryCursor

from daydream._tree_sitter_safety import assert_tree_sitter_safe


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
    # Keep unsafe-version errors outside the fail-open native parsing boundary.
    assert_tree_sitter_safe()
    try:
        parser = Parser(factory())
    except Exception:
        return None
    _PARSER_CACHE[language_id] = parser
    return parser



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

# Symbol captures also control generic-stem reverse-import eligibility.
PYTHON_DEF_QUERY = """
(function_definition) @def @function
(class_definition) @def @class
"""

RUST_DEF_QUERY = """
(function_item) @def @function
(struct_item) @def @class
(enum_item) @def @class
(trait_item) @def @class
(impl_item) @def
"""
# impl_item has no name and is skipped. Keep the symbol query stable: widening
# it to TS/Go would grant generic index.ts/main.go files new reverse-import edges.
# Diagram queries deliberately cover more definitions and use their own kinds.
# TS, TSX, and JS share the TypeScript grammars.
TYPESCRIPT_DIAGRAM_DEF_QUERY = """
(function_declaration name: (identifier)) @def @function
(generator_function_declaration name: (identifier)) @def @function
(function_signature name: (identifier)) @def @function
(method_definition name: (property_identifier)) @def @function
(method_signature name: (property_identifier)) @def @function
(class_declaration name: (type_identifier)) @def @class
(abstract_class_declaration name: (type_identifier)) @def @class
(interface_declaration name: (type_identifier)) @def @class
(type_alias_declaration name: (type_identifier)) @def @type
(enum_declaration name: (identifier)) @def @class
(variable_declarator name: (identifier) value: [(arrow_function) (function_expression)]) @def @function
"""

GO_DIAGRAM_DEF_QUERY = """
(function_declaration name: (identifier)) @def @function
(method_declaration name: (field_identifier)) @def @function
(type_spec name: (type_identifier)) @def @type
(type_alias name: (type_identifier)) @def @type
"""

RUST_DIAGRAM_DEF_QUERY = """
(function_item) @def @function
(function_signature_item) @def @function
(struct_item) @def @class
(enum_item) @def @class
(trait_item) @def @class
(union_item) @def @class
(type_item) @def @type
(mod_item) @def @module
"""


_IMPORT_QUERIES = {
    "python": PYTHON_IMPORT_QUERY,
    "typescript": TYPESCRIPT_IMPORT_QUERY,
    "tsx": TYPESCRIPT_IMPORT_QUERY,
    "javascript": TYPESCRIPT_IMPORT_QUERY,
    "go": GO_IMPORT_QUERY,
    "rust": RUST_IMPORT_QUERY,
}
_SYMBOL_QUERIES = {"python": PYTHON_DEF_QUERY, "rust": RUST_DEF_QUERY}
_DIAGRAM_QUERIES = {
    "python": PYTHON_DEF_QUERY,
    "typescript": TYPESCRIPT_DIAGRAM_DEF_QUERY,
    "tsx": TYPESCRIPT_DIAGRAM_DEF_QUERY,
    "javascript": TYPESCRIPT_DIAGRAM_DEF_QUERY,
    "go": GO_DIAGRAM_DEF_QUERY,
    "rust": RUST_DIAGRAM_DEF_QUERY,
}


def _query_for_language(language_id: str) -> str | None:
    return _IMPORT_QUERIES.get(language_id)


def _def_query_for_language(language_id: str) -> str | None:
    return _SYMBOL_QUERIES.get(language_id)


def _diagram_def_query_for_language(language_id: str) -> str | None:
    return _DIAGRAM_QUERIES.get(language_id)


def _definition_kind(node_type: str) -> str:
    """Map a tree-sitter definition node type to ``function``/``class``."""
    if node_type in ("function_definition", "function_item"):
        return "function"
    return "class"


def _captures(parser: Parser, source: bytes, query_string: str) -> dict[str, list[Node]]:
    tree = parser.parse(source)
    if parser.language is None:
        return {}
    return QueryCursor(Query(parser.language, query_string)).captures(tree.root_node)


def _captured_definitions(
    parser: Parser,
    source: bytes,
    query_string: str,
    *,
    kind_for: Callable[[str], str] | None = None,
) -> list[dict[str, object]]:
    """Return captured {name, line, end_line, kind} records with 1-based lines.

    Native captures own declared kinds; an explicit callback takes precedence.
    Unnamed captures are skipped. Parse, query, or record failures return []."""
    try:
        captures = _captures(parser, source, query_string)
        kinds = {
            node.id: kind
            for kind in ("function", "class", "type", "module")
            for node in captures.get(kind, [])
        }
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
                    "kind": (
                        kind_for(node.type)
                        if kind_for is not None
                        else kinds[node.id]
                    ),
                }
            )
        return result
    except Exception:
        return []


def extract_definitions(
    parser: Parser,
    source: bytes,
    query_string: str,
    *,
    kind_for: Callable[[str], str] = _definition_kind,
) -> list[dict[str, object]]:
    """Return named captures using the caller's node-type kind policy."""
    if kind_for is None:
        return []
    return _captured_definitions(parser, source, query_string, kind_for=kind_for)


def extract_imports(parser: Parser, source: bytes, query_string: str) -> list[str]:
    """Return decoded, unquoted imports; parse, query, or decoding failures yield []."""
    try:
        captures = _captures(parser, source, query_string)
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


def language_for_path(path: str) -> str | None:
    """Return the language for an exact, case-sensitive suffix, or None."""
    entry = LANGUAGES.get(Path(path).suffix)
    if entry is None:
        return None
    return entry[0]


def definitions_in_file(repo_root: Path, path: str) -> list[dict[str, object]]:
    """Return diagram definitions with 1-based ranges, or [] when unavailable.

    Ranges include bodies but exclude decorators; distinct definitions can share
    a line. Unsafe parser versions still raise."""
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
    return _captured_definitions(parser, source, query_string)
