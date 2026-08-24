"""Stack detection and file routing for deep-review mode.

Pure-logic classifier that maps a list of changed files to StackAssignment records
per D-11..D-16 (see .planning/phases/05-deep-review-mode/05-CONTEXT.md).

The routing order is significant (Pitfall 6 in 05-RESEARCH.md):
    0. Fork StackRule globs       -> fork stack, first match wins (per-file)
    1. .md pinning (D-14)         -> generic unconditionally
    2. Extension lookup (D-11)    -> stack by _EXT_TO_STACK
    3. Config promotion (D-13)    -> promote only on co-change
    4. Ambiguous nearest-ancestor (D-12)
    5. Equal-depth fallthrough (D-12c) -> generic

The detection is registry-independent for built-in stacks: the same changed
files produce the same ordered stack scopes whether a plugin registry is
present or not. Stack identity is routing metadata, and fork-registered stacks
contribute only their names and changed-file patterns.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from daydream.config import STRUCTURE_STACK_NAME
from daydream.extensions import Registry, StackRule, get_registry

# Extension -> stack-key (lowercase, matches the supported built-in stacks).
# This table is about review routing, not syntactic parsing
# (tree_sitter_index.LANGUAGES serves a different purpose).
_EXT_TO_STACK: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".ts": "react",
    ".tsx": "react",
    ".js": "react",
    ".jsx": "react",
    ".ex": "elixir",
    ".exs": "elixir",
    ".go": "go",
    ".rs": "rust",
    ".swift": "ios",
}

# Config files promoted only when a co-changed stack file signals ownership.
# filename -> stack-key
_CONFIG_OWNERSHIP_SIGNALS: dict[str, str] = {
    "pyproject.toml": "python",
    "setup.py": "python",
    "setup.cfg": "python",
    "package.json": "react",
    "tsconfig.json": "react",
    "mix.exs": "elixir",
    "go.mod": "go",
    "go.sum": "go",
    "Cargo.toml": "rust",
    "Cargo.lock": "rust",
    "Package.swift": "ios",
}

# Generic-fallback stack key. Not a built-in language stack — it is a synthetic
# bucket signalling "run the native generic review agent".
GENERIC_STACK = "generic"


@dataclass
class StackAssignment:
    """Routing result for one detected stack.

    Attributes:
        stack_name: Lower-case stack key, e.g. "python" or "generic".
        files: Files routed to this stack. Never empty for entries in the returned list.
        is_docs_only: True when this assignment represents a docs-only diff (triggers D-20
            notice). Only set on the ``generic`` bucket, and only when no non-generic stacks
            were detected in the whole diff. Non-generic buckets never have a docs-only mix.
    """

    stack_name: str
    files: list[str] = field(default_factory=list)
    is_docs_only: bool = False
    # Issue #731: cross-shard frontier. For a shard with a synthetic ``#``
    # ``stack_name``, the bounded set of files in *other* shards of the same
    # language that share a tree-sitter import edge with this shard's files.
    # Never added to ``files`` (union of primary sets stays the changed set).
    # Empty for unsplit stacks and the structural meta-stack.
    frontier_files: list[str] = field(default_factory=list)


def _ext(path: str) -> str:
    """Return lowercase suffix of ``path`` (empty string if none)."""
    return PurePosixPath(path).suffix.lower()


def _basename(path: str) -> str:
    """Return the final path component of ``path``."""
    return PurePosixPath(path).name


def _is_config_generic_default(path: str) -> bool:
    """Config / infra files that route to generic unless promoted (D-13)."""
    suffix = _ext(path)
    if suffix in {".yaml", ".yml", ".toml"}:
        return True
    base = _basename(path)
    if base == "Dockerfile":
        return True
    if path.startswith(".github/workflows/") and suffix in {".yml", ".yaml"}:
        return True
    return False


def _nearest_ancestor_stack(path: str, assigned: dict[str, str]) -> str | None:
    """Walk up ``path``'s ancestors; return the non-generic stack of the deepest ancestor
    that contains an already-assigned unambiguous sibling (D-12).

    Returns None on equal-depth ambiguity (D-12c) or when no ancestor match exists.
    """
    p = PurePosixPath(path)
    # Walk deepest-first so "nearest" = first match.
    for parent in p.parents:
        ancestor = str(parent)
        prefix = ancestor + "/" if ancestor and ancestor != "." else ""
        stacks_here = {
            stack
            for file_path, stack in assigned.items()
            if file_path != path
            and stack != GENERIC_STACK
            and file_path.startswith(prefix)
        }
        if len(stacks_here) == 1:
            return next(iter(stacks_here))
        if len(stacks_here) > 1:
            return None  # equal-depth ambiguity -> fallthrough (D-12c)
    return None


def _match_stack_rule(path: str, rules: tuple[StackRule, ...]) -> StackRule | None:
    """Return the first fork rule whose glob matches ``path`` (registration order)."""
    for rule in rules:
        if any(fnmatch.fnmatchcase(path, pattern) for pattern in rule.patterns):
            return rule
    return None


def detect_stacks(
    changed_files: list[str],
    *,
    registry: Registry | None = None,
) -> list[StackAssignment]:
    """Route changed files to stacks per D-11..D-16.

    Detection is registry-independent for built-in stacks: the same changed
    files produce the same ordered scopes whether a plugin registry is present
    or not. The registry is consulted only for fork stack rules.

    Args:
        changed_files: Paths (POSIX-style, repo-relative) of files that changed in the diff.
        registry: Extension registry for fork stack rules only. Defaults to the
            current context's registry (``get_registry()``).

    Returns:
        One StackAssignment per distinct stack that received at least one file,
        plus a synthetic ``structure`` meta-stack appended last whenever the diff
        contains at least one file and is not docs-only. The structure stack
        carries the full set of changed files (union across languages).
        Ordering: non-generic language stacks alphabetical, then generic,
        then structure last. Ordering is informational only -- the orchestrator
        iterates the full list in parallel.
    """
    if registry is None:
        registry = get_registry()
    rules = registry.stack_rules()

    assigned: dict[str, str] = {}  # path -> stack_name
    ambiguous: list[str] = []

    # Unambiguous routing (fork rules + extension + .md pinning + config default).
    for path in changed_files:
        # Fork StackRule globs win per-file, before any built-in routing.
        rule = _match_stack_rule(path, rules)
        if rule is not None:
            assigned[path] = rule.stack_name
            continue

        base = _basename(path)
        suffix = _ext(path)

        # D-14: .md pinned unconditionally.
        if suffix == ".md":
            assigned[path] = GENERIC_STACK
            continue

        # D-11: extension lookup.
        if suffix in _EXT_TO_STACK:
            assigned[path] = _EXT_TO_STACK[suffix]
            continue

        # D-13: config/infra default-generic (may be promoted in pass 2).
        if _is_config_generic_default(path) or base in _CONFIG_OWNERSHIP_SIGNALS:
            assigned[path] = GENERIC_STACK
            continue

        # Otherwise ambiguous — resolved in pass 3.
        ambiguous.append(path)

    # Promote config files whose owner stack is present in the diff (D-13).
    present_stacks = {s for s in assigned.values() if s != GENERIC_STACK}
    for path in list(assigned.keys()):
        base = _basename(path)
        owner = _CONFIG_OWNERSHIP_SIGNALS.get(base)
        if owner and owner in present_stacks:
            assigned[path] = owner

    # Refresh present stacks after promotion.
    present_stacks = {s for s in assigned.values() if s != GENERIC_STACK}

    # Ambiguous files (D-12).
    for path in ambiguous:
        if len(present_stacks) == 1:
            # D-12 single-stack shortcut: unconditional join.
            assigned[path] = next(iter(present_stacks))
            continue
        if len(present_stacks) == 0:
            assigned[path] = GENERIC_STACK
            continue
        nearest = _nearest_ancestor_stack(path, assigned)
        assigned[path] = nearest if nearest is not None else GENERIC_STACK

    # D-16 is removed: a built-in stack never degrades to generic merely
    # because a plugin registry is absent. Unknown/unassigned files route to the
    # native generic fallback, never a detected built-in stack.

    groups: dict[str, list[str]] = {}
    for path, stack in assigned.items():
        groups.setdefault(stack, []).append(path)

    # is_docs_only means "this whole diff is docs-only" (triggers D-20 notice). A mixed
    # diff (docs + code) must not flag the generic bucket as docs-only even though that
    # bucket only contains .md files.
    non_generic_stacks = [k for k in groups if k != GENERIC_STACK]
    diff_is_docs_only = (
        not non_generic_stacks
        and GENERIC_STACK in groups
        and all(_ext(f) == ".md" for f in groups[GENERIC_STACK])
    )

    results: list[StackAssignment] = []
    for stack_name in sorted(non_generic_stacks):
        files = sorted(groups[stack_name])
        results.append(
            StackAssignment(
                stack_name=stack_name,
                files=files,
                is_docs_only=all(_ext(f) == ".md" for f in files),
            )
        )
    if GENERIC_STACK in groups:
        files = sorted(groups[GENERIC_STACK])
        results.append(
            StackAssignment(
                stack_name=GENERIC_STACK,
                files=files,
                is_docs_only=diff_is_docs_only,
            )
        )

    # Structural meta-stack: appended unconditionally for any non-docs-only diff
    # with at least one changed file (caller gates on ctx.pipeline().structural_enabled).
    # Carries the union of all changed files so the structural reviewer judges the
    # whole change across language boundaries.
    if changed_files and not diff_is_docs_only:
        results.append(
            StackAssignment(
                stack_name=STRUCTURE_STACK_NAME,
                files=sorted(changed_files),
                is_docs_only=False,
            )
        )
    return results
