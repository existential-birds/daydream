"""Exploration evidence shared by review prompts, persisted artifacts and cache keys."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from daydream.prompts.grounding import UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

# Single rendering of the untrusted-content warning shared by the inline prompt
# section and every persisted artifact, so the boundary text lives in one place.
_BOUNDARY_BLOCKQUOTE = f"> {UNTRUSTED_REPOSITORY_CONTENT_BOUNDARY}"

# Opening of to_prompt_section()'s canonical header: the section heading plus the
# blockquote boundary. Exposed so consumers that strip the embedded boundary
# (e.g. improve's recon prompt) share this rendering instead of re-deriving it.
EXPLORATION_SECTION_PREFIX = f"# Exploration Context\n\n{_BOUNDARY_BLOCKQUOTE}\n\n"


def _no_data_artifact(title: str) -> str:
    """Markdown body for a persisted artifact when nothing was collected."""
    return f"# {title}\n{_BOUNDARY_BLOCKQUOTE}\n\nNo data collected.\n"


@dataclass
class FileInfo:
    """Relevant file, its role and static/LLM provenance.

    source_file links test rows to their covered source when known.
    """

    path: str
    role: str
    summary: str = ""
    provenance: str = "static"
    source_file: str = ""


@dataclass
class Convention:
    """Named repository convention with its evidence source."""

    name: str
    description: str
    source: str = ""


@dataclass
class Dependency:
    """Directed file relationship: source imports/calls/extends/tests target."""

    source: str
    target: str
    relationship: str


@dataclass
class ExplorationContext:
    """Collected files, conventions, dependencies, guidelines and notes for review."""

    affected_files: list[FileInfo] = field(default_factory=list)
    conventions: list[Convention] = field(default_factory=list)
    dependencies: list[Dependency] = field(default_factory=list)
    guidelines: list[str] = field(default_factory=list)
    raw_notes: str = ""
    completed: bool = True

    def to_prompt_section(self) -> str:
        """Render exploration context as text for prompt injection.

        Returns empty string when all fields are empty/default, so it adds
        nothing to the review prompt for unexplored contexts.
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

        return (
            EXPLORATION_SECTION_PREFIX
            + "\n\n".join(sections)
            + "\n"
        )

    def write_to_dir(self, exploration_dir: Path) -> Path:
        """Write exploration results as markdown files and structured JSON."""
        exploration_dir.mkdir(parents=True, exist_ok=True)

        affected_rows = [
            {"path": f.path, "role": f.role, "summary": f.summary, "provenance": f.provenance}
            for f in self.affected_files
        ]
        exploration_data = {
            "affected_files": affected_rows,
            "conventions": [
                {"name": c.name, "description": c.description, "source": c.source}
                for c in self.conventions
            ],
            "dependencies": [
                {"source": d.source, "target": d.target, "relationship": d.relationship}
                for d in self.dependencies
            ],
        }
        (exploration_dir / "exploration.json").write_text(json.dumps(exploration_data, indent=2) + "\n")
        test_mapping = [
            {"test_file": f.path, "source_file": f.source_file}
            for f in self.affected_files
            if f.role == "test" and f.source_file
        ]
        (exploration_dir / "test-map.json").write_text(
            json.dumps({"test_mapping": test_mapping}, indent=2) + "\n"
        )

        if self.affected_files:
            lines = ["# Affected Files", _BOUNDARY_BLOCKQUOTE, "",
                     "Files relevant to the current review, discovered by exploration.",
                     "| File | Role | Provenance | Summary |", "|------|------|------------|---------|"]
            for f in self.affected_files:
                lines.append(f"| `{f.path}` | {f.role} | {f.provenance} | {f.summary} |")
            (exploration_dir / "affected_files.md").write_text("\n".join(lines) + "\n")
        else:
            (exploration_dir / "affected_files.md").write_text(_no_data_artifact("Affected Files"))

        if self.conventions or self.guidelines:
            lines = ["# Codebase Conventions", _BOUNDARY_BLOCKQUOTE, "",
                     "Conventions detected during pre-scan exploration."]
            if self.conventions:
                lines.append("## Conventions")
                for c in self.conventions:
                    line = f"- **{c.name}**: {c.description}"
                    if c.source:
                        line += f" (source: {c.source})"
                    lines.append(line)
                lines.append("")
            if self.guidelines:
                lines.append("## Project Guidelines")
                for g in self.guidelines:
                    lines.append(f"- {g}")
                lines.append("")
            (exploration_dir / "conventions.md").write_text("\n".join(lines))
        else:
            (exploration_dir / "conventions.md").write_text(_no_data_artifact("Codebase Conventions"))

        if self.dependencies:
            lines = ["# Dependencies", _BOUNDARY_BLOCKQUOTE, "",
                     "Import and call relationships between files.",
                     "| Source | Relationship | Target |", "|--------|-------------|--------|"]
            for d in self.dependencies:
                lines.append(f"| `{d.source}` | {d.relationship} | `{d.target}` |")
            (exploration_dir / "dependencies.md").write_text("\n".join(lines) + "\n")
        else:
            (exploration_dir / "dependencies.md").write_text(_no_data_artifact("Dependencies"))

        summary_lines = ["# Exploration Summary", _BOUNDARY_BLOCKQUOTE, "",
                         "Pre-scan exploration results for the current review.",
                         "| File | Contents |", "|------|----------|"]
        if self.affected_files:
            role_counts: dict[str, int] = {}
            for f in self.affected_files:
                role_counts[f.role] = role_counts.get(f.role, 0) + 1
            role_str = ", ".join(f"{v} {k}" for k, v in role_counts.items())
            summary_lines.append(f"| `affected_files.md` | {len(self.affected_files)} files ({role_str}) |")
        else:
            summary_lines.append("| `affected_files.md` | No data collected |")

        conv_count = len(self.conventions)
        guide_count = len(self.guidelines)
        if conv_count or guide_count:
            parts: list[str] = []
            if conv_count:
                parts.append(f"{conv_count} convention{'s' if conv_count != 1 else ''}")
            if guide_count:
                parts.append(f"{guide_count} guideline{'s' if guide_count != 1 else ''}")
            summary_lines.append(f"| `conventions.md` | {', '.join(parts)} |")
        else:
            summary_lines.append("| `conventions.md` | No data collected |")

        if self.dependencies:
            dep_count = len(self.dependencies)
            summary_lines.append(
                f"| `dependencies.md` | {dep_count} dependency edge{'s' if dep_count != 1 else ''} |"
            )
        else:
            summary_lines.append("| `dependencies.md` | No data collected |")

        summary_text = "\n".join(summary_lines) + "\n"
        if self.raw_notes:
            summary_text += f"\n## Additional Notes\n{self.raw_notes}\n"

        (exploration_dir / "summary.md").write_text(summary_text)

        return exploration_dir


async def safe_explore(
    explore_fn: Callable[..., Awaitable[ExplorationContext]],
    *args: Any,
    **kwargs: Any,
) -> ExplorationContext:
    """Warn and return an incomplete empty context when exploration raises."""
    try:
        return await explore_fn(*args, **kwargs)
    except Exception:
        from daydream.ui import create_console, print_warning

        console = create_console()
        print_warning(console, "Exploration failed -- proceeding with review only")
        return ExplorationContext(completed=False)


def merge_contexts(*contexts: ExplorationContext) -> ExplorationContext:
    """Merge into fresh lists, preserving first-seen ordering.

    Files key by (path, role): longest summary wins, static provenance dominates.
    Conventions key by name; dependencies by their triple; guidelines by text.
    Nonempty notes join with a blank line. Input objects remain unchanged.
    """
    files_by_key: dict[tuple[str, str], FileInfo] = {}
    static_keys: set[tuple[str, str]] = set()
    source_by_key: dict[tuple[str, str], str] = {}
    for ctx in contexts:
        for f in ctx.affected_files:
            key = (f.path, f.role)
            if f.provenance == "static":
                static_keys.add(key)
            if f.source_file:
                source_by_key.setdefault(key, f.source_file)
            existing = files_by_key.get(key)
            if existing is None or len(f.summary) > len(existing.summary):
                files_by_key[key] = f
    for key in static_keys:
        winner = files_by_key[key]
        source = winner.source_file
        if winner.provenance != "static" or (not source and key in source_by_key):
            files_by_key[key] = replace(
                winner, provenance="static", source_file=source or source_by_key.get(key, source),
            )

    conventions: dict[str, Convention] = {}
    for ctx in contexts:
        for c in ctx.conventions:
            conventions.setdefault(c.name, c)

    dependencies: dict[tuple[str, str, str], Dependency] = {}
    for ctx in contexts:
        for d in ctx.dependencies:
            dependencies.setdefault((d.source, d.target, d.relationship), d)

    guidelines = list(dict.fromkeys(g for ctx in contexts for g in ctx.guidelines))

    raw_notes = "\n\n".join(ctx.raw_notes for ctx in contexts if ctx.raw_notes)

    return ExplorationContext(
        affected_files=list(files_by_key.values()),
        conventions=list(conventions.values()),
        dependencies=list(dependencies.values()),
        guidelines=guidelines,
        raw_notes=raw_notes,
    )


CACHE_KEY_FILENAME = "cache-key"

# Bump when the artifact generator changes (e.g. a new boundary rendering) so
# upgrades force regeneration instead of serving pre-upgrade artifacts on a key match.
_CACHE_VERSION = 5


def exploration_cache_key(
    head_sha: str, diff: str, tier: str, *, strategies: dict[str, str] | None = None,
) -> str:
    """Hash format version, HEAD, diff, tier and effective strategies for exact reuse.

    Uncommitted edits are intentionally excluded. Bump _CACHE_VERSION when artifact
    rendering changes so an upgrade cannot reuse older output bytes.
    """
    payload = json.dumps([_CACHE_VERSION, head_sha, diff, tier, strategies or {}], sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_key_path(exploration_dir: Path) -> Path:
    """Sibling file holding the key the directory's contents were produced from."""
    return exploration_dir / CACHE_KEY_FILENAME


def read_cache_key(exploration_dir: Path) -> str | None:
    """The stored key, or None when absent/unreadable."""
    try:
        return cache_key_path(exploration_dir).read_text(encoding="utf-8").strip()
    except OSError:
        return None
