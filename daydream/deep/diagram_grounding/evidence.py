"""Repository-confined source and symbol evidence for diagram grounding."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from daydream.deep.diagram_types import (
    as_int as _norm_line,
)
from daydream.git_ops import GitError, grep_fixed_matches
from daydream.repository_paths import path_is_confined, strip_dot_slash, valid_repository_file_path
from daydream.tree_sitter_index import (
    StatementLines,
    definitions_in_file,
    language_for_path,
    statement_lines,
)

# Snap order for ``SYMBOL_NOT_ON_LINE``: nearest line first, the line above
# before the line below at equal distance. Fixed so a snapped citation is
# reproducible from the same spec and tree.
_SNAP_OFFSETS: tuple[int, ...] = (-1, 1, -2, 2, -3, 3)

# Upper bound on the files a repo-wide subroutine lookup will parse. ``git
# grep`` narrows the search to files that contain the token at all; parsing the
# first few of those in sorted order is enough to find a definition without
# turning one node check into a whole-repository index build.
_MAX_DEFINITION_PROBE_FILES = 20


def token_hits(
    repo_root: Path, symbol: str, files: list[str] | None = None
) -> list[str]:
    """Return sorted token-bearing paths; Git/read failures provide no evidence."""
    if not symbol:
        return []
    pathspecs = [strip_dot_slash(path) for path in files] if files else None
    try:
        matches = grep_fixed_matches(
            repo_root, [symbol], word=True, pathspecs=pathspecs
        )
    except (GitError, OSError):
        return []
    return sorted({path for path, _ in matches})


class RepoSymbols:
    """Lazy per-file definition index shared across diagram kinds and repair turns.

    A native parser failure disables further extraction for this instance;
    unindexable files fall back to token evidence, recorded as strength="token".
    """

    def __init__(self, repo_root: Path) -> None:
        self._repo_root = repo_root
        self._by_file: dict[str, list[dict[str, object]]] = {}
        self._parsing_available = True

    def _indexable(self, path: str) -> bool:
        """Whether definitions can be extracted from ``path`` at all."""
        return self._parsing_available and language_for_path(strip_dot_slash(path)) is not None

    def _file_definitions(self, path: str) -> list[dict[str, object]]:
        """Return the memoized definition records for one repo-relative path."""
        key = strip_dot_slash(path)
        cached = self._by_file.get(key)
        if cached is not None:
            return cached
        records: list[dict[str, object]] = []
        if self._indexable(key):
            try:
                records = definitions_in_file(self._repo_root, key)
            except Exception:
                self._parsing_available = False
                records = []
        for record in records:
            record["file"] = key
        self._by_file[key] = records
        return records

    def definitions(
        self, symbol: str, files: list[str] | None = None
    ) -> list[dict[str, object]]:
        """Find symbol definitions in the given files or cached paths, sorted by (file, line)."""
        if not symbol:
            return []
        paths = (
            [strip_dot_slash(path) for path in files]
            if files is not None
            else sorted(self._by_file)
        )
        found: list[dict[str, object]] = []
        for path in paths:
            found.extend(
                record
                for record in self._file_definitions(path)
                if record.get("name") == symbol
            )
        return sorted(found, key=lambda r: (str(r.get("file", "")), int(str(r.get("line", 0)))))


def definition_location(record: dict[str, object]) -> str:
    """Return the ``path:line`` citation for one definition record."""
    return f"{record.get('file', '')}:{record.get('line', 0)}"


def resolve_in_files(
    repo_root: Path, symbols: RepoSymbols, symbol: str, files: list[str]
) -> tuple[str | None, str | None]:
    """Return (strength, defined_at); absent evidence yields (None, None).

    Token fallback applies only to unindexable files: a call-site mention in a
    parseable file must never substitute for a definition.
    """
    records = symbols.definitions(symbol, files)
    if records:
        return "definition", definition_location(records[0])
    unindexable = [path for path in files if not symbols._indexable(path)]
    if unindexable and token_hits(repo_root, symbol, unindexable):
        return "token", None
    return None, None


def resolve_anywhere(
    repo_root: Path, symbols: RepoSymbols, symbol: str
) -> tuple[str | None, str | None]:
    """Resolve a definition in bounded token-bearing files, with unindexable-file fallback."""
    hits = token_hits(repo_root, symbol)[:_MAX_DEFINITION_PROBE_FILES]
    if not hits:
        return None, None
    indexable = [path for path in hits if symbols._indexable(path)]
    records = symbols.definitions(symbol, indexable) if indexable else []
    if records:
        return "definition", definition_location(records[0])
    if len(indexable) < len(hits):
        return "token", None
    return None, None


# --- Evidence primitives -----------------------------------------------------


class SourceCache:
    """Memoized head-tree file reader for one grounding pass."""

    def __init__(self, repo_root: Path) -> None:
        self._repo_root = repo_root
        self._bytes: dict[str, bytes | None] = {}
        self._lines: dict[str, list[str]] = {}
        self._statements: dict[str, StatementLines] = {}

    def read(self, path: str) -> bytes | None:
        """Return the file's bytes, or None when it is missing or unreadable."""
        if path not in self._bytes:
            try:
                self._bytes[path] = (self._repo_root / path).read_bytes()
            except OSError:
                self._bytes[path] = None
        return self._bytes[path]

    def lines(self, path: str) -> list[str]:
        """Return the file's decoded lines (empty for a missing file)."""
        if path not in self._lines:
            data = self.read(path)
            self._lines[path] = (
                data.decode("utf-8", errors="replace").splitlines()
                if data is not None
                else []
            )
        return self._lines[path]

    def line_count(self, path: str) -> int:
        """Return the file's line count."""
        return len(self.lines(path))

    def line_text(self, path: str, line: int) -> str:
        """Return the 1-based ``line``, or ``""`` when out of range."""
        rows = self.lines(path)
        return rows[line - 1] if 1 <= line <= len(rows) else ""


    def statements(self, path: str) -> StatementLines:
        """Capture source memberships, retaining keyword fallback on a bad install."""
        if path not in self._statements:
            source = self.read(path) or b""
            try:
                captured = statement_lines(language_for_path(path), source)
            except Exception:
                captured = statement_lines(None, source)
            self._statements[path] = captured
        return self._statements[path]


def norm_str(value: Any) -> str:
    """Return ``value`` when it is a string, else ``""``."""
    return value if isinstance(value, str) else ""


def token_on_line(text: str, symbol: str) -> bool:
    """Whether ``symbol`` occurs in ``text`` at word boundaries.

    Lookarounds rather than ``\\b`` so a symbol that begins or ends with a
    non-word character (``panic!``, ``*ptr``) is still bounded correctly.
    """
    if not symbol:
        return False
    return (
        re.search(rf"(?<!\w){re.escape(symbol)}(?!\w)", text) is not None
    )


def in_ranges(ranges: list[tuple[int, int]], line: int) -> bool:
    """Whether ``line`` falls inside any inclusive ``(start, end)`` range."""
    return any(start <= line <= end for start, end in ranges)


def check_path(repo_root: Path, file: Any) -> tuple[str, str | None]:
    """Return ``(normalized_path, reason)`` for a cited path."""
    if not isinstance(file, str) or not file:
        return "", "FILE_MISSING"
    normalized = strip_dot_slash(file)
    if not valid_repository_file_path(normalized) or not path_is_confined(
        repo_root, normalized
    ):
        return normalized, "PATH_ESCAPES_REPO"
    return normalized, None


def check_location(
    repo_root: Path,
    sources: SourceCache,
    file: Any,
    line: Any,
) -> tuple[str, int, str | None]:
    """Verify a ``file:line`` citation exists at head.

    Returns ``(normalized_path, line, reason)``. Check order is fixed: path
    grammar and confinement, then existence, then line range.
    """
    normalized, reason = check_path(repo_root, file)
    if reason is not None:
        return normalized, _norm_line(line), reason
    if sources.read(normalized) is None:
        return normalized, _norm_line(line), "FILE_MISSING"
    cited = _norm_line(line)
    if cited < 1 or cited > sources.line_count(normalized):
        return normalized, cited, "LINE_OUT_OF_RANGE"
    return normalized, cited, None


def snap_symbol(
    sources: SourceCache,
    file: str,
    line: int,
    symbol: str,
    *,
    within: tuple[int, int] | None = None,
) -> int | None:
    """Find symbol evidence within three lines and optional bounds, preferring above on ties."""
    for offset in _SNAP_OFFSETS:
        candidate = line + offset
        if candidate < 1 or candidate > sources.line_count(file):
            continue
        if within is not None and not (within[0] <= candidate <= within[1]):
            continue
        if token_on_line(sources.line_text(file, candidate), symbol):
            return candidate
    return None


def branch_line(sources: SourceCache, file: str, line: int) -> bool:
    """Whether ``file:line`` opens a control-flow branch (fail-open on a bad install)."""
    return line in sources.statements(file).branches


def terminal_line(sources: SourceCache, file: str, line: int) -> bool:
    """Whether ``file:line`` ends a control-flow path (fail-open on a bad install)."""
    return line in sources.statements(file).terminals


def reply_line(sources: SourceCache, file: str, line: int) -> bool:
    """Whether ``file:line`` is a return statement suitable for a reply."""
    return re.match(r"^\s*return\b", sources.line_text(file, line)) is not None


def reply_definition(
    symbols: RepoSymbols, symbol: str, file: str, line: int
) -> dict[str, object] | None:
    """Return the enclosing function named by ``symbol`` at ``file:line``."""
    for definition in symbols.definitions(symbol, [file]):
        start = int(str(definition.get("line", 0)))
        end = int(str(definition.get("end_line", 0)))
        if definition.get("kind") == "function" and start <= line <= end:
            return definition
    return None


def executable_line(sources: SourceCache, file: str, line: int) -> bool:
    """Whether ``file:line`` starts an executable statement."""
    return line in sources.statements(file).executable
