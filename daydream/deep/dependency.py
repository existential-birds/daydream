"""Lightweight tree-sitter import graph for dependency-aware sharding (#731).

Turns a language stack's changed files into an undirected import graph so the
sharder can co-locate files that reference each other and surface a bounded
cross-shard frontier. Everything here is fail-open: a parse error, unknown
grammar, or timeout must never raise -- the caller falls back to deterministic
sorted-singleton packing, so no file is ever dropped.
"""

from __future__ import annotations


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
