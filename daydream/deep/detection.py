"""Stack detection and file routing for deep-review mode.

Pure-logic classifier that maps a list of changed files to StackAssignment records
per D-11..D-16 (see .planning/phases/05-deep-review-mode/05-CONTEXT.md).

The routing order is significant (Pitfall 6 in 05-RESEARCH.md):
    1. .md pinning (D-14)         -> generic unconditionally
    2. Extension lookup (D-11)    -> stack by _EXT_TO_STACK
    3. Config promotion (D-13)    -> promote only on co-change
    4. Ambiguous nearest-ancestor (D-12)
    5. Equal-depth fallthrough (D-12c) -> generic
    6. Missing-skill fallthrough (D-16) -> generic
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import PurePosixPath

from daydream.config import SKILL_MAP

# Extension -> stack-key (lowercase, matches SKILL_MAP keys). Keep in sync with
# SKILL_MAP; this table is about review-skill routing, not syntactic parsing
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

# Generic-fallback stack key. Not present in SKILL_MAP by design — it's a synthetic
# bucket signalling "run the generic review agent".
GENERIC_STACK = "generic"


@dataclass
class StackAssignment:
    """Routing result for one detected stack.

    Attributes:
        stack_name: Lower-case stack key, e.g. "python" or "generic".
        skill_invocation: Beagle skill key from SKILL_MAP (e.g. "beagle-python:review-python"),
            or None for the generic fallback.
        files: Files routed to this stack. Never empty for entries in the returned list.
        is_docs_only: True when this assignment represents a docs-only diff (triggers D-20
            notice). Only set on the ``generic`` bucket, and only when no non-generic stacks
            were detected in the whole diff. Non-generic buckets never have a docs-only mix.
    """

    stack_name: str
    skill_invocation: str | None
    files: list[str] = field(default_factory=list)
    is_docs_only: bool = False


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


def detect_stacks(
    changed_files: list[str],
    skill_availability: set[str] | None = None,
) -> list[StackAssignment]:
    """Route changed files to stacks per D-11..D-16.

    Args:
        changed_files: Paths (POSIX-style, repo-relative) of files that changed in the diff.
        skill_availability: Lower-case stack keys for which a Beagle skill is installed.
            Defaults to all keys in SKILL_MAP (optimistic availability — runtime
            MissingSkillError would be handled separately).

    Returns:
        One StackAssignment per distinct stack that received at least one file.
        Ordering: non-generic stacks alphabetical, then generic last.
    """
    if skill_availability is None:
        skill_availability = set(SKILL_MAP.keys())

    assigned: dict[str, str] = {}  # path -> stack_name
    ambiguous: list[str] = []

    # Pass 1: unambiguous routing (extension + .md pinning + config default).
    for path in changed_files:
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

    # Pass 2: promote config files whose owner stack is present in the diff (D-13).
    present_stacks = {s for s in assigned.values() if s != GENERIC_STACK}
    for path in list(assigned.keys()):
        base = _basename(path)
        owner = _CONFIG_OWNERSHIP_SIGNALS.get(base)
        if owner and owner in present_stacks:
            assigned[path] = owner

    # Refresh present stacks after promotion.
    present_stacks = {s for s in assigned.values() if s != GENERIC_STACK}

    # Pass 3: ambiguous files (D-12).
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

    # Pass 4: missing-skill fallthrough (D-16) — move files of any stack without an
    # installed skill into generic.
    for path, stack in list(assigned.items()):
        if stack != GENERIC_STACK and stack not in skill_availability:
            assigned[path] = GENERIC_STACK

    # Group -> StackAssignment.
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
                skill_invocation=SKILL_MAP.get(stack_name),
                files=files,
                is_docs_only=all(_ext(f) == ".md" for f in files),
            )
        )
    if GENERIC_STACK in groups:
        files = sorted(groups[GENERIC_STACK])
        results.append(
            StackAssignment(
                stack_name=GENERIC_STACK,
                skill_invocation=None,
                files=files,
                is_docs_only=diff_is_docs_only,
            )
        )
    return results
