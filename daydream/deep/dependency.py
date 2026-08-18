"""Lightweight tree-sitter import graph for dependency-aware sharding (#731).

Turns a language stack's changed files into an undirected import graph so the
sharder can co-locate files that reference each other and surface a bounded
cross-shard frontier. Everything here is fail-open: a parse error, unknown
grammar, or timeout must never raise -- the caller falls back to deterministic
sorted-singleton packing, so no file is ever dropped.
"""

from __future__ import annotations

import time
from pathlib import Path, PurePosixPath

from daydream.tree_sitter_index import (
    _query_for_language,
    extract_imports,
    get_parser,
)

# Extension -> tree-sitter language id (subset of the tree_sitter_index
# registry relevant to sharding's dependency graph). Unknown extensions are
# fail-open: the file becomes a singleton, never dropped.
_LANG_BY_EXT: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "typescript",
    ".go": "go",
    ".rs": "rust",
}

_GRAPH_BUILD_WALL_BUDGET_S = 5.0


def _resolve_import(import_str: str, file: str) -> list[str]:
    """Best-effort map an import string to candidate repo-relative changed paths.

    Python dotted names (``import a.b`` -> ``a/b.py`` plus the package
    ``a/b/__init__.py``); relative ``.``/``..`` prefixes resolve against the
    importing file's own directory (``from .x import y`` in ``pkg/a.py`` ->
    ``pkg/x.py``); TypeScript ``./x``/``../x`` specifiers resolve to sibling
    ``.ts``/``.tsx``/``.jsx`` files; Go ``import "pkg/x"`` and Rust ``use x::y``
    resolve to ``.go``/``.rs`` files. Returns an empty list when nothing
    resolvable (fail-open).
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
    if suffix == ".py":
        stems = ["/".join(parts)]
        suffixes: tuple[str, ...] = (".py", "/__init__.py")
    elif suffix in (".ts", ".tsx", ".jsx"):
        stems = ["/".join(parts)]
        suffixes = (".ts", ".tsx", ".jsx", "/index.ts", "/index.tsx", "/index.jsx")
    elif suffix == ".go":
        stems = ["/".join(parts)]
        suffixes = (".go",)
    elif suffix == ".rs":
        while parts and parts[0] in ("crate", "self", "super"):
            parts.pop(0)
        if not parts:
            return []
        # The first path segment names the module file (``b.rs``/``b/mod.rs``).
        stems = ["/".join(parts[:1])]
        suffixes = (".rs", "/mod.rs")
    else:
        return []
    candidates = [stem + sfx for stem in stems for sfx in suffixes]
    return [(base_rel + "/" + c) if base_rel else c for c in candidates]


def build_import_graph(
    changed_files: list[str], repo_root: Path
) -> dict[str, set[str]]:
    """Parse each changed file and return ``{file: set_of_changed_files_it_imports}``.

    Uses the installed tree-sitter grammars (already pinned as deps). A file
    that fails to parse, has no grammar, or whose imports don't resolve to a
    changed file simply gets an empty edge set -- never raises. The whole build
    is wall-time-bounded (~5s) and on timeout returns the partial edges built so
    far (the caller falls back to sorted-singleton packing for the rest).
    """
    changed = set(changed_files)
    graph: dict[str, set[str]] = {}
    deadline = time.monotonic() + _GRAPH_BUILD_WALL_BUDGET_S
    for file in changed_files:
        graph[file] = set()
        if time.monotonic() > deadline:
            break
        lang_id = _LANG_BY_EXT.get(PurePosixPath(file).suffix.lower())
        if lang_id is None:
            continue
        parser = get_parser(lang_id)
        if parser is None:
            continue
        query = _query_for_language(lang_id)
        if query is None:
            continue
        try:
            source = (repo_root / file).read_bytes()
        except Exception:
            continue
        imports = extract_imports(parser, source, query)
        for imp in imports:
            for candidate in _resolve_import(imp, file):
                if candidate in changed:
                    graph[file].add(candidate)
                    break
        if time.monotonic() > deadline:
            break
    return graph


def co_locate_groups(files: list[str], edges: dict[str, set[str]]) -> list[list[str]]:
    """Connected components of the undirected graph restricted to ``files``.

    ``edges`` maps a file to the set of files it imports. The undirected closure
    is computed over ``files`` only. Each component is returned sorted; the
    components are returned in the sorted order of their smallest file. A file
    with no edges is a singleton. Raises on nothing -- pure over already-computed
    edges.
    """
    wanted = set(files)
    adjacency: dict[str, set[str]] = {f: set() for f in files}
    for src, deps in edges.items():
        if src not in wanted:
            continue
        for dep in deps:
            if dep in wanted:
                adjacency[src].add(dep)
                adjacency[dep].add(src)

    seen: set[str] = set()
    groups: list[list[str]] = []
    for start in sorted(files):
        if start in seen:
            continue
        component: list[str] = []
        stack = [start]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            component.append(node)
            stack.extend(adjacency[node])
        groups.append(sorted(component))
    groups.sort(key=lambda comp: comp[0])
    return groups
