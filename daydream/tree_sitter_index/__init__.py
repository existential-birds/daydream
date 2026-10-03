"""Static import impact and diagram grounding for Python, TS/JS, Go, and Rust.

Unknown languages, unreadable files, and parse failures yield empty results.
Known-bad tree-sitter versions raise before native parser construction; callers
that must survive an unsafe installation handle TreeSitterBadVersionError."""

from .imports import (
    detect_affected_files as detect_affected_files,
)
from .runtime import (
    GO_DIAGRAM_DEF_QUERY as GO_DIAGRAM_DEF_QUERY,
    GO_IMPORT_QUERY as GO_IMPORT_QUERY,
    LANGUAGES as LANGUAGES,
    PYTHON_DEF_QUERY as PYTHON_DEF_QUERY,
    PYTHON_IMPORT_QUERY as PYTHON_IMPORT_QUERY,
    RUST_DEF_QUERY as RUST_DEF_QUERY,
    RUST_DIAGRAM_DEF_QUERY as RUST_DIAGRAM_DEF_QUERY,
    RUST_IMPORT_QUERY as RUST_IMPORT_QUERY,
    TYPESCRIPT_DIAGRAM_DEF_QUERY as TYPESCRIPT_DIAGRAM_DEF_QUERY,
    TYPESCRIPT_IMPORT_QUERY as TYPESCRIPT_IMPORT_QUERY,
    definitions_in_file as definitions_in_file,
    extract_definitions as extract_definitions,
    extract_imports as extract_imports,
    get_parser as get_parser,
    language_for_path as language_for_path,
)
from .statements import (
    BRANCH_NODE_TYPES as BRANCH_NODE_TYPES,
    EXECUTABLE_STATEMENT_NODE_TYPES as EXECUTABLE_STATEMENT_NODE_TYPES,
    TERMINAL_CALL_NAMES as TERMINAL_CALL_NAMES,
    TERMINAL_MACRO_NAMES as TERMINAL_MACRO_NAMES,
    TERMINAL_NODE_TYPES as TERMINAL_NODE_TYPES,
    StatementLines as StatementLines,
    branch_statement_lines as branch_statement_lines,
    statement_lines as statement_lines,
)
