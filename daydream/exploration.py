"""Exploration result types for review prompt injection.

Holds the typed data model that exploration subagents populate and review
agents consume via to_prompt_section(). Includes graceful degradation for
exploration failures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


@dataclass
class FileInfo:
    """Information about a file relevant to the review.

    Attributes:
        path: Relative file path.
        role: Relationship to the diff -- "modified", "imported_by", "imports", "test".
        summary: Brief description of the file's purpose.
    """

    path: str
    role: str
    summary: str = ""


@dataclass
class Convention:
    """A codebase convention or pattern detected during exploration.

    Attributes:
        name: Short convention name (e.g. "snake_case functions").
        description: What the convention entails.
        source: Where it was found -- "CLAUDE.md", "inferred from code", etc.
    """

    name: str
    description: str
    source: str = ""


@dataclass
class Dependency:
    """A dependency relationship between files.

    Attributes:
        source: File that depends on target.
        target: File being depended upon.
        relationship: Type -- "imports", "calls", "extends", "tests".
    """

    source: str
    target: str
    relationship: str


@dataclass
class ExplorationContext:
    """Aggregated exploration results for review prompt injection.

    Populated by exploration subagents in Phase 2, consumed by review
    agents via to_prompt_section() in Phase 3.

    Attributes:
        affected_files: Files relevant to the review.
        conventions: Detected codebase conventions.
        dependencies: Dependency relationships between files.
        guidelines: Project guideline snippets (from CLAUDE.md etc).
        raw_notes: Unstructured exploration notes.
    """

    affected_files: list[FileInfo] = field(default_factory=list)
    conventions: list[Convention] = field(default_factory=list)
    dependencies: list[Dependency] = field(default_factory=list)
    guidelines: list[str] = field(default_factory=list)
    raw_notes: str = ""

    def to_prompt_section(self) -> str:
        """Render exploration context as text for prompt injection.

        Returns empty string when all fields are empty/default, so it adds
        nothing to the review prompt for unexplored contexts.

        Returns:
            Markdown-formatted exploration context, or empty string.
        """
        sections: list[str] = []

        if self.affected_files:
            lines = ["## Affected Files"]
            for f in self.affected_files:
                line = f"- `{f.path}` ({f.role})"
                if f.summary:
                    line += f" — {f.summary}"
                lines.append(line)
            sections.append("\n".join(lines))

        if self.conventions:
            lines = ["## Codebase Conventions"]
            for c in self.conventions:
                line = f"- **{c.name}**: {c.description}"
                if c.source:
                    line += f" (source: {c.source})"
                lines.append(line)
            sections.append("\n".join(lines))

        if self.dependencies:
            lines = ["## Dependencies"]
            for d in self.dependencies:
                lines.append(f"- `{d.source}` {d.relationship} `{d.target}`")
            sections.append("\n".join(lines))

        if self.guidelines:
            lines = ["## Project Guidelines"]
            for g in self.guidelines:
                lines.append(f"- {g}")
            sections.append("\n".join(lines))

        if self.raw_notes:
            sections.append(f"## Additional Notes\n{self.raw_notes}")

        if not sections:
            return ""

        return "# Exploration Context\n\n" + "\n\n".join(sections) + "\n"


async def safe_explore(
    explore_fn: Callable[..., Awaitable[ExplorationContext]],
    *args: Any,
    **kwargs: Any,
) -> ExplorationContext:
    """Run exploration with graceful degradation.

    Catches any exception from explore_fn and returns an empty
    ExplorationContext instead. Displays a warning banner via Rich UI.

    Args:
        explore_fn: Async callable that performs exploration.
        *args: Positional args forwarded to explore_fn.
        **kwargs: Keyword args forwarded to explore_fn.

    Returns:
        ExplorationContext from explore_fn, or empty ExplorationContext on failure.
    """
    try:
        return await explore_fn(*args, **kwargs)
    except Exception:
        from daydream.ui import create_console, print_warning

        console = create_console()
        print_warning(console, "Exploration failed -- proceeding with review only")
        return ExplorationContext()


def merge_contexts(*contexts: ExplorationContext) -> ExplorationContext:
    """Fold multiple partial ExplorationContext instances into one.

    De-duplication rules:
    - FileInfo: keyed on (path, role); the entry with the longer summary wins.
    - Convention: keyed on name; first occurrence wins.
    - Dependency: keyed on (source, target, relationship).
    - guidelines: keyed on string identity.
    - raw_notes: non-empty values joined with a double newline.

    Args:
        *contexts: Any number of ExplorationContext instances.

    Returns:
        A new merged ExplorationContext (always a fresh instance with fresh
        list fields, even when called with a single argument).
    """
    files_by_key: dict[tuple[str, str], FileInfo] = {}
    for ctx in contexts:
        for f in ctx.affected_files:
            key = (f.path, f.role)
            existing = files_by_key.get(key)
            if existing is None or len(f.summary) > len(existing.summary):
                files_by_key[key] = f

    seen_conv: set[str] = set()
    conventions: list[Convention] = []
    for ctx in contexts:
        for c in ctx.conventions:
            if c.name in seen_conv:
                continue
            seen_conv.add(c.name)
            conventions.append(c)

    seen_deps: set[tuple[str, str, str]] = set()
    dependencies: list[Dependency] = []
    for ctx in contexts:
        for d in ctx.dependencies:
            key_d = (d.source, d.target, d.relationship)
            if key_d in seen_deps:
                continue
            seen_deps.add(key_d)
            dependencies.append(d)

    seen_guidelines: set[str] = set()
    guidelines: list[str] = []
    for ctx in contexts:
        for g in ctx.guidelines:
            if g in seen_guidelines:
                continue
            seen_guidelines.add(g)
            guidelines.append(g)

    raw_notes = "\n\n".join(ctx.raw_notes for ctx in contexts if ctx.raw_notes)

    return ExplorationContext(
        affected_files=list(files_by_key.values()),
        conventions=conventions,
        dependencies=dependencies,
        guidelines=guidelines,
        raw_notes=raw_notes,
    )
