"""Changed-file import graphs.

Parsing/grammar failures and timeouts leave a file without edges. Callers also
guard unsafe tree-sitter installations.
"""

from __future__ import annotations

import time
from pathlib import Path, PurePosixPath

from daydream.tree_sitter_index import (
    LANGUAGES,
    extract_imports,
    get_parser,
)
from daydream.tree_sitter_index.runtime import _query_for_language

# Extension -> tree-sitter language id for the import graph, derived from the
# canonical registry so the two can never drift. ``.js`` is excluded because
# ``_resolve_import`` does not resolve JavaScript specifiers; a file with an
# unknown extension contributes no edges.
_LANG_BY_EXT: dict[str, str] = {
    ext: lang for ext, (lang, _) in LANGUAGES.items() if ext != ".js"
}

_GRAPH_BUILD_WALL_BUDGET_S = 5.0


def _resolve_import(import_str: str, file: str) -> list[str]:
    """Map imports to candidate repository paths for Python, TypeScript, Go, and Rust.

    Python relative dots resolve from the importer; dotted names include package
    __init__.py candidates. TypeScript includes sibling files/index modules; Rust
    uses its first non-prefix module. Unresolvable imports return no candidates.
    """
    suffix = PurePosixPath(file).suffix.lower()
    text = import_str.strip()
    # The python relative-import query captures the whole ``from .x import y``
    # statement; keep only the module name that follows ``from``.
    if text.startswith("from ") and " import " in text:
        text = text[len("from ") : text.index(" import ")].strip()
    rel_dots = 0
    while text.startswith("."):
        rel_dots += 1
        text = text[1:]
    text = text.lstrip("/")
    if suffix == ".py":
        parts = [p for p in text.split(".") if p]
    else:
        parts = [p for p in text.replace("::", "/").split("/") if p]
    if not parts:
        # ``from . import x``-style: no module path to resolve.
        return []
    if rel_dots == 0:
        base_rel = ""
    else:
        dir_parts = [p for p in str(PurePosixPath(file).parent).split("/") if p and p != "."]
        up = max(0, rel_dots - 1)
        if up:
            dir_parts = dir_parts[:-up]
        base_rel = "/".join(dir_parts)
    suffixes = {
        ".py": (".py", "/__init__.py"),
        ".ts": (".ts", ".tsx", ".jsx", "/index.ts", "/index.tsx", "/index.jsx"),
        ".go": (".go",),
        ".rs": (".rs", "/mod.rs"),
    }.get(".ts" if suffix in (".tsx", ".jsx") else suffix)
    if suffixes is None:
        return []
    if suffix == ".rs":
        while parts and parts[0] in ("crate", "self", "super"):
            parts.pop(0)
        if not parts:
            return []
        parts = parts[:1]
    stem = "/".join(parts)
    candidates = [stem + sfx for sfx in suffixes]
    return [(base_rel + "/" + c) if base_rel else c for c in candidates]


def build_import_graph_with_status(
    changed_files: list[str], repo_root: Path
) -> tuple[dict[str, set[str]], bool]:
    """Return ``(graph, complete)`` for changed files within a five-second wall budget.

    ``complete`` is ``True`` only when every file in *changed_files* had a supported
    grammar, an available parser and query, succeeded reading its bytes, and finished
    within the wall budget. Empty *changed_files* returns ``({}, True)``.
    """
    changed = set(changed_files)
    graph: dict[str, set[str]] = {}
    complete = True
    deadline = time.monotonic() + _GRAPH_BUILD_WALL_BUDGET_S
    for idx, file in enumerate(changed_files):
        graph[file] = set()
        if time.monotonic() > deadline:
            complete = False
            break
        lang_id = _LANG_BY_EXT.get(PurePosixPath(file).suffix.lower())
        if lang_id is None:
            complete = False
            continue
        parser = get_parser(lang_id)
        if parser is None:
            complete = False
            continue
        query = _query_for_language(lang_id)
        if query is None:
            complete = False
            continue
        try:
            source = (repo_root / file).read_bytes()
        except Exception:
            complete = False
            continue
        imports = extract_imports(parser, source, query)
        for imp in imports:
            for candidate in _resolve_import(imp, file):
                if candidate in changed:
                    graph[file].add(candidate)
                    break
        if time.monotonic() > deadline:
            if idx + 1 < len(changed_files):
                complete = False
            break
    return graph, complete


def build_import_graph(
    changed_files: list[str], repo_root: Path
) -> dict[str, set[str]]:
    """Return directed imports between changed files within a five-second wall budget.

    Unknown grammars and unreadable files contribute no edges. Timeout returns the
    partial graph.
    """
    graph, _ = build_import_graph_with_status(changed_files, repo_root)
    return graph
