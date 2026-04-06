"""Exploration result types for review prompt injection.

Holds the typed data model that exploration subagents populate and review
agents consume via to_prompt_section(). Includes graceful degradation for
exploration failures.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
