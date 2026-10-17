"""Deterministic stack routing for changed files.

Order: first matching fork rule, bounded local JS/TS evidence, Markdown pinning,
extension lookup, config promotion, nearest ancestor, then generic on ambiguity.
Built-in scopes are registry-independent; extensions contribute names and globs.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from daydream.config import STRUCTURE_STACK_NAME
from daydream.deep.framework_detection import JS_SOURCE_SUFFIXES, FrameworkInspection
from daydream.extensions import Registry, StackRule, get_registry

# Extension -> stack-key (lowercase, matches the supported built-in stacks).
# This table is about review routing, not syntactic parsing
# (tree_sitter_index.LANGUAGES serves a different purpose).
_EXT_TO_STACK: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
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


def base_stack_name(stack_name: str) -> str:
    """Select stack policy without replacing a public shard's identity."""
    base, separator, shard = stack_name.partition("#")
    return base if separator and shard.isdecimal() else stack_name


@dataclass
class StackAssignment:
    """Nonempty file scope; only generic may mark an entirely docs-only diff."""

    stack_name: str
    files: list[str] = field(default_factory=list)
    is_docs_only: bool = False
    # Issue #731: cross-shard frontier. For a shard with a synthetic ``#``
    # ``stack_name``, the bounded set of files in *other* shards of the same
    # language that share a tree-sitter import edge with this shard's files.
    # Never added to ``files`` (union of primary sets stays the changed set).
    # Empty for unsplit stacks and the structural meta-stack.
    frontier_files: list[str] = field(default_factory=list)
    # Automatic tiny-diff collapse must retain explicit extension routing.
    has_explicit_rule: bool = False


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
    # Repository infrastructure has no ownership relationship to a co-changed
    # frontend file. Keep it out of the ambiguous single-stack shortcut while
    # preserving that inference for configs/helpers nested inside an app.
    # JS project configs use bounded local evidence instead of co-change promotion.
    if PurePosixPath(path).parent == PurePosixPath(".") and (base == "Makefile" or suffix == ".json"):
        return True
    if path.startswith("scripts/") and suffix in {".mjs", ".cjs"}:
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
    target_dir: Path | None = None,
) -> list[StackAssignment]:
    """Route changed POSIX paths, consulting the registry only for fork rules.

    Return nonempty language scopes alphabetically, then generic, then structure.
    The structure meta-stack contains all changed files unless the diff is docs-only.
    An omitted registry resolves from the current context.
    """
    if registry is None:
        registry = get_registry()
    rules = registry.stack_rules()
    inspection = FrameworkInspection(target_dir)
    explicit: set[str] = set()

    assigned: dict[str, str] = {}  # path -> stack_name
    ambiguous: list[str] = []

    # Unambiguous routing (fork rules + extension + .md pinning + config default).
    for path in sorted(set(changed_files)):
        # Fork StackRule globs win per-file, before any built-in routing.
        rule = _match_stack_rule(path, rules)
        if rule is not None:
            assigned[path] = rule.stack_name
            explicit.add(path)
            continue

        base = _basename(path)
        suffix = _ext(path)
        if suffix in JS_SOURCE_SUFFIXES:
            assigned[path] = inspection.source_stack(path)
            continue
        if base in {"package.json", "tsconfig.json"}:
            assigned[path] = inspection.config_stack(path)
            continue

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
        if path not in explicit and owner and owner in present_stacks:
            assigned[path] = owner

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
                has_explicit_rule=any(f in explicit for f in files),
            )
        )
    if GENERIC_STACK in groups:
        files = sorted(groups[GENERIC_STACK])
        results.append(
            StackAssignment(
                stack_name=GENERIC_STACK,
                files=files,
                is_docs_only=diff_is_docs_only,
                has_explicit_rule=any(f in explicit for f in files),
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
                files=sorted(set(changed_files)),
                is_docs_only=False,
            )
        )
    return results
