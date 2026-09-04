"""Deterministic grounding for the proposed diagram specs (issue #1113).

The model never writes mermaid. For each diagram kind it proposes a structured
JSON spec whose every element carries ``file:line`` (and, where relevant,
``symbol``) evidence, and this module is the sole authority on whether that
evidence is real. Nothing that cannot be confirmed here is ever rendered.

:func:`ground_sequence` and :func:`ground_flowchart` each perform four passes
in one call and return a single :class:`GroundingReport`:

1. **check** -- every element is verified against the head tree (path
   confinement, file existence, line range, symbol tokens, tree-sitter node
   kinds, definition lookup) and against the phase's trajectory read receipts.
   One reason code per failing element.
2. **prune** -- ungrounded elements are removed, together with the structure
   that depended on them (a block whose branch condition failed is flattened,
   an edge whose node was dropped is dropped, a decision left underspecified is
   demoted, nodes unreachable from ``start`` are removed).
3. **cap** -- the render caps from :mod:`daydream.config` are applied *here*,
   before the floor, in a fixed deterministic order. Enforcing them in the
   renderer instead would let a cap-induced drop leave a rendered diagram below
   its own omission floor, and would let ``spec_final`` disagree with what was
   drawn.
4. **floor** -- a kind too thin to be worth drawing is reported through
   :attr:`GroundingReport.omit_reasons`; the caller renders only when that list
   is empty.

``spec_final`` is rebuilt key-by-key rather than copied, so it carries exactly
the keys of the sequence/flowchart schema and no annotations: the privileged
Phase B poster re-validates it against an ``additionalProperties: false``
schema before re-rendering, so a single bookkeeping key smuggled into it would
fail the post. Per-element bookkeeping the renderer needs -- which
``spec_final`` slot an element ended up in, where its callee is defined, the
snapped line -- travels on the report's :class:`ElementCheck` list instead.

Pure: no LLM call, no network, no writes. The only I/O is reading files under
the repository root and one ``git grep`` per symbol fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from daydream.config import (
    DIAGRAM_MAX_BLOCKS,
    DIAGRAM_MAX_EDGES,
    DIAGRAM_MAX_MESSAGES,
    DIAGRAM_MAX_NODES,
    DIAGRAM_MAX_PARTICIPANTS,
)
from daydream.deep.coverage import _path_component_matches, _strip_dot_slash
from daydream.deep.diagram_types import CandidateRoot
from daydream.git_ops import GitError, grep_fixed_matches
from daydream.repository_paths import path_is_confined, valid_repository_file_path
from daydream.tree_sitter_index import (
    definitions_in_file,
    is_branch_line,
    is_executable_statement_line,
    is_terminal_line,
    language_for_path,
)

# --- Vocabulary --------------------------------------------------------------

# Reason codes shared by both kinds. Every one names a fact about the cited
# evidence itself, so the same check runs for a sequence message, a block
# branch and a flowchart node.
_SHARED_REASON_CODES = frozenset(
    {
        "PATH_ESCAPES_REPO",
        "FILE_MISSING",
        "LINE_OUT_OF_RANGE",
        "SYMBOL_NOT_ON_LINE",
        "NOT_A_BRANCH_STATEMENT",
        "FILE_NOT_READ_BY_MODEL",
        # Not in the spec's table: the one code for an element whose *shape* is
        # wrong (non-object entry, missing/blank/duplicate identifier, kind
        # outside its enum). Those cases have no evidence to adjudicate, and
        # borrowing an evidence code for them would put a false explanation in
        # front of the repair turn.
        "MALFORMED_ELEMENT",
    }
)
_SEQUENCE_REASON_CODES = frozenset(
    {
        "EVIDENCE_NOT_IN_SOURCE_PARTICIPANT",
        "CALLEE_NOT_DEFINED_IN_TARGET",
        "PARTICIPANT_NO_FILES",
        "PARTICIPANT_FILE_MISSING",
        "EXTERNAL_MISUSED",
    }
)
_FLOWCHART_REASON_CODES = frozenset(
    {
        "ROOT_NOT_CANDIDATE",
        "NODE_OUTSIDE_ROOT",
        "NOT_AN_EXECUTABLE_STATEMENT",
        "NOT_A_TERMINAL_STATEMENT",
        "SUBROUTINE_NOT_DEFINED",
        "SUBROUTINE_NOT_CALLED_HERE",
        "DECISION_EDGES_INVALID",
        "EDGE_ENDPOINT_UNGROUNDED",
        "MULTIPLE_START",
    }
)

#: Every per-element reason code either grounder can emit. Exposed so the
#: repair-turn prompt and the tests enumerate one list rather than three.
REASON_CODES: frozenset[str] = (
    _SHARED_REASON_CODES | _SEQUENCE_REASON_CODES | _FLOWCHART_REASON_CODES
)

#: Every whole-kind omission reason. ``NO_END`` is a floor reason, not an
#: element reason: a flowchart with no terminal node is structurally fine, it
#: is simply not worth drawing.
OMIT_REASONS: frozenset[str] = frozenset(
    {
        "TOO_FEW_MESSAGES",
        "NO_CHANGED_INTERACTION",
        "TOO_FEW_PARTICIPANTS",
        "TOO_FEW_NODES",
        "NO_DECISION",
        "NO_END",
    }
)

_PARTICIPANT_KINDS = frozenset({"internal", "external"})
_MESSAGE_KINDS = frozenset({"call", "reply", "self"})
_BLOCK_KINDS = frozenset({"alt", "opt", "loop"})
_NODE_KINDS = frozenset({"start", "end", "process", "decision", "subroutine", "io"})

# Snap order for ``SYMBOL_NOT_ON_LINE``: nearest line first, the line above
# before the line below at equal distance. Fixed so a snapped citation is
# reproducible from the same spec and tree.
_SNAP_OFFSETS: tuple[int, ...] = (-1, 1, -2, 2, -3, 3)

# Upper bound on the files a repo-wide subroutine lookup will parse. ``git
# grep`` narrows the search to files that contain the token at all; parsing the
# first few of those in sorted order is enough to find a definition without
# turning one node check into a whole-repository index build.
_MAX_DEFINITION_PROBE_FILES = 20

# Sequence and flowchart floors (spec section 4.3).
_MIN_MESSAGES = 3
_MIN_PARTICIPANTS = 2
_MIN_NODES = 4


# --- Report types ------------------------------------------------------------


@dataclass
class ElementCheck:
    """The verdict on one proposed diagram element.

    Attributes:
        element: ``"participant"``, ``"message"``, ``"block"``, ``"branch"``,
            ``"root"``, ``"node"`` or ``"edge"``.
        ref: Stable identifier within its kind -- participant name, proposed
            message index, ``"b0"`` / ``"b0.1"`` for a block and its branch,
            root function name, node id, or ``"from->to"`` for an edge. Unique
            across the elements of one kind, which is what lets the renderer
            correlate a table row back to its check.
        grounded: Whether every check passed. ``False`` means the element was
            pruned as unproven; a grounded element may still be absent from
            ``spec_final`` (structurally flattened, unused, or cap-trimmed),
            which is what ``final_index is None`` records.
        reason: The reason code from :data:`REASON_CODES`, or None when
            grounded.
        strength: ``"definition"`` when a symbol was resolved to a tree-sitter
            definition, ``"token"`` when only a word-boundary token match was
            available (a language without a definition query, or a symbol
            proven only at its call site), None when no symbol was resolved.
        snapped_line: The line the citation was moved to when a +/-3 snap
            rescued a ``SYMBOL_NOT_ON_LINE`` failure, else None. The snapped
            value is also written into ``spec_final``'s evidence.
        in_changed_hunk: Whether the (post-snap) evidence line falls in a
            head-side changed hunk. Input to the sequence floor; never
            rendered.
        defined_at: ``"path:line"`` of the callee/subroutine definition when one
            was found, else None.
        final_index: 0-based index of the element in its ``spec_final``
            collection (for a branch, its index within its own block's
            ``branches``; for the flowchart root, 0 when accepted), or None when
            the element is not in ``spec_final``.
    """

    element: str
    ref: str
    grounded: bool
    reason: str | None = None
    strength: str | None = None
    snapped_line: int | None = None
    in_changed_hunk: bool = False
    defined_at: str | None = None
    final_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-safe form written into ``diagram.json``."""
        return {
            "element": self.element,
            "ref": self.ref,
            "grounded": self.grounded,
            "reason": self.reason,
            "strength": self.strength,
            "snapped_line": self.snapped_line,
            "in_changed_hunk": self.in_changed_hunk,
            "defined_at": self.defined_at,
            "final_index": self.final_index,
        }


@dataclass
class GroundingReport:
    """The outcome of one kind's check/prune/cap/floor pass.

    Attributes:
        elements: One :class:`ElementCheck` per proposed element, in proposal
            order within each element type.
        spec_final: The pruned and capped spec, carrying exactly the schema's
            keys so the Phase B poster can re-validate and re-render it.
        summary: ``{"proposed", "grounded", "pruned"}`` element counts.
            ``pruned`` counts elements dropped **as ungrounded** only; drops
            made by a render cap are in :attr:`capped`.
        capped: Per-collection count of elements dropped by a render cap;
            ``{}`` when no cap bound.
        root_range: Flowchart only -- the accepted root's ``(line, end_line)``
            taken from its :class:`~daydream.deep.diagram_types.CandidateRoot`,
            not from the model. None for a sequence diagram and for a rejected
            root.
        omit_reasons: Non-empty when the capped spec is below its floor, in
            which case the caller must not render this kind.
        rejected: ``"ROOT_NOT_CANDIDATE"`` when the whole flowchart spec was
            rejected, else None. A rejection also lists ``TOO_FEW_NODES`` in
            :attr:`omit_reasons` -- it is literally true of the empty result,
            and it means a caller that only consults the floor can never read a
            rejected spec as renderable. ``spec_final["root"]`` then holds the
            model's own (schema-shaped) root so the repair turn can see what was
            refused, or None when it was not even well-formed.
    """

    elements: list[ElementCheck]
    spec_final: dict[str, Any]
    summary: dict[str, int]
    capped: dict[str, int]
    root_range: tuple[int, int] | None
    omit_reasons: list[str]
    rejected: str | None

    def ungrounded(self) -> list[ElementCheck]:
        """Return the failing checks, in element order (the repair-turn input)."""
        return [check for check in self.elements if not check.grounded]


# --- Symbol resolution -------------------------------------------------------


def _token_hits(
    repo_root: Path, symbol: str, files: list[str] | None = None
) -> list[str]:
    """Return sorted repo-relative paths whose text contains ``symbol`` as a word.

    ``git grep`` is the only search that does not require an index build, and it
    is deliberately fail-closed here: a repository with no commits, no git
    history at all, or a git invocation that errors yields "not found" rather
    than an exception, because an unprovable symbol must be pruned, never
    rendered on the strength of a failed lookup.
    """
    if not symbol:
        return []
    pathspecs = [_strip_dot_slash(path) for path in files] if files else None
    try:
        matches = grep_fixed_matches(
            repo_root, [symbol], word=True, pathspecs=pathspecs
        )
    except (GitError, OSError):
        return []
    return sorted({path for path, _ in matches})


class RepoSymbols:
    """Lazy per-file tree-sitter definition index for one repository.

    Definitions are extracted only for the files a check actually asks about,
    because a diagram cites a handful of files out of a repository that may hold
    thousands. Results are memoized per path for the lifetime of the instance,
    so one instance should be shared across both grounders and both turns of a
    run.

    A native failure -- most importantly the known-bad-install
    ``TreeSitterBadVersionError`` that :func:`~daydream.tree_sitter_index.get_parser`
    raises -- disables definition extraction for the rest of the instance's life
    instead of propagating. Grounding then falls back to word-boundary token
    evidence and records ``strength="token"``, which is the same degradation a
    language without a definition query gets.
    """

    def __init__(self, repo_root: Path) -> None:
        self._repo_root = repo_root
        self._by_file: dict[str, list[dict[str, object]]] = {}
        self._parsing_available = True

    def _indexable(self, path: str) -> bool:
        """Whether definitions can be extracted from ``path`` at all."""
        return self._parsing_available and language_for_path(_strip_dot_slash(path)) is not None

    def _file_definitions(self, path: str) -> list[dict[str, object]]:
        """Return the memoized definition records for one repo-relative path."""
        key = _strip_dot_slash(path)
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
        """Return ``{file, name, line, end_line, kind}`` records defining ``symbol``.

        Searches ``files`` when given, else every path indexed so far. Sorted by
        ``(file, line)`` so the first record -- the one whose location is
        reported as ``defined_at`` -- is reproducible.
        """
        if not symbol:
            return []
        paths = (
            [_strip_dot_slash(path) for path in files]
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

    def token_defined_anywhere(self, symbol: str) -> bool:
        """Whether ``symbol`` appears as a whole word anywhere in the repository."""
        return bool(_token_hits(self._repo_root, symbol))


def _definition_location(record: dict[str, object]) -> str:
    """Return the ``path:line`` citation for one definition record."""
    return f"{record.get('file', '')}:{record.get('line', 0)}"


def _resolve_in_files(
    repo_root: Path, symbols: RepoSymbols, symbol: str, files: list[str]
) -> tuple[str | None, str | None]:
    """Resolve ``symbol`` to a definition inside ``files``.

    Returns ``(strength, defined_at)``, or ``(None, None)`` when the symbol has
    no definition there. The token fallback is reached only when at least one of
    ``files`` is in a language with no definition query (or definition
    extraction is disabled): otherwise a token match would let any mention of
    the name -- including the call site being checked -- pass as a definition.
    """
    records = symbols.definitions(symbol, files)
    if records:
        return "definition", _definition_location(records[0])
    unindexable = [path for path in files if not symbols._indexable(path)]
    if unindexable and _token_hits(repo_root, symbol, unindexable):
        return "token", None
    return None, None


def _resolve_anywhere(
    repo_root: Path, symbols: RepoSymbols, symbol: str
) -> tuple[str | None, str | None]:
    """Resolve ``symbol`` to a definition anywhere in the repository.

    ``git grep`` narrows the candidate files first, then each candidate in a
    language with a definition query is parsed. Same fallback rule as
    :func:`_resolve_in_files`: token evidence counts only when some file holding
    the token cannot be parsed for definitions.
    """
    hits = _token_hits(repo_root, symbol)[:_MAX_DEFINITION_PROBE_FILES]
    if not hits:
        return None, None
    indexable = [path for path in hits if symbols._indexable(path)]
    records = symbols.definitions(symbol, indexable) if indexable else []
    if records:
        return "definition", _definition_location(records[0])
    if len(indexable) < len(hits):
        return "token", None
    return None, None


# --- Evidence primitives -----------------------------------------------------


class _SourceCache:
    """Memoized head-tree file reader for one grounding pass."""

    def __init__(self, repo_root: Path) -> None:
        self._repo_root = repo_root
        self._bytes: dict[str, bytes | None] = {}
        self._lines: dict[str, list[str]] = {}

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


def _as_list(value: Any) -> list[Any]:
    """Return ``value`` when it is a list, else the empty list."""
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> dict[str, Any]:
    """Return ``value`` when it is a dict, else the empty dict."""
    return value if isinstance(value, dict) else {}


def _norm_str(value: Any) -> str:
    """Return ``value`` when it is a string, else ``""``."""
    return value if isinstance(value, str) else ""


def _norm_optional_str(value: Any) -> str | None:
    """Return ``value`` when it is a non-empty string, else None."""
    return value if isinstance(value, str) and value else None


def _norm_line(value: Any) -> int:
    """Return ``value`` when it is a real int (not a bool), else 0."""
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _token_on_line(text: str, symbol: str) -> bool:
    """Whether ``symbol`` occurs in ``text`` at word boundaries.

    Lookarounds rather than ``\\b`` so a symbol that begins or ends with a
    non-word character (``panic!``, ``*ptr``) is still bounded correctly.
    """
    if not symbol:
        return False
    return (
        re.search(rf"(?<!\w){re.escape(symbol)}(?!\w)", text) is not None
    )


def _was_read(read_paths: set[str], relative: str) -> bool:
    """Whether any recorded read receipt names ``relative``.

    ``read_paths`` are raw tool-call paths, usually absolute, so matching is by
    path component (``/repo/pkg/api.py`` covers ``pkg/api.py`` but
    ``/repo/notapi.py`` does not cover ``api.py``).
    """
    return any(
        _path_component_matches(_strip_dot_slash(path), relative)
        for path in read_paths
    )


def _in_ranges(ranges: list[tuple[int, int]], line: int) -> bool:
    """Whether ``line`` falls inside any inclusive ``(start, end)`` range."""
    return any(start <= line <= end for start, end in ranges)


def _check_path(repo_root: Path, file: Any) -> tuple[str, str | None]:
    """Return ``(normalized_path, reason)`` for a cited path."""
    if not isinstance(file, str) or not file:
        return "", "FILE_MISSING"
    normalized = _strip_dot_slash(file)
    if not valid_repository_file_path(normalized) or not path_is_confined(
        repo_root, normalized
    ):
        return normalized, "PATH_ESCAPES_REPO"
    return normalized, None


def _check_location(
    repo_root: Path,
    sources: _SourceCache,
    read_paths: set[str],
    file: Any,
    line: Any,
) -> tuple[str, int, str | None]:
    """Verify a ``file:line`` citation exists at head and was read.

    Returns ``(normalized_path, line, reason)``. Check order is fixed: path
    grammar and confinement, then existence, then line range, then the
    trajectory read receipt -- each later check would be meaningless if an
    earlier one failed.
    """
    normalized, reason = _check_path(repo_root, file)
    if reason is not None:
        return normalized, _norm_line(line), reason
    if sources.read(normalized) is None:
        return normalized, _norm_line(line), "FILE_MISSING"
    cited = _norm_line(line)
    if cited < 1 or cited > sources.line_count(normalized):
        return normalized, cited, "LINE_OUT_OF_RANGE"
    if not _was_read(read_paths, normalized):
        return normalized, cited, "FILE_NOT_READ_BY_MODEL"
    return normalized, cited, None


def _snap_symbol(
    sources: _SourceCache,
    file: str,
    line: int,
    symbol: str,
    *,
    within: tuple[int, int] | None = None,
) -> int | None:
    """Return a line within +/-3 of ``line`` carrying ``symbol``, else None.

    Mirrors the realignment bookkeeping of
    ``deep.location_validator._align_evidence`` -- rewrite the citation, record
    what it was moved to -- without calling it: that helper realigns a
    free-text evidence string on a review record, while a diagram citation is a
    structured ``{file, line}`` pair.
    """
    for offset in _SNAP_OFFSETS:
        candidate = line + offset
        if candidate < 1 or candidate > sources.line_count(file):
            continue
        if within is not None and not (within[0] <= candidate <= within[1]):
            continue
        if _token_on_line(sources.line_text(file, candidate), symbol):
            return candidate
    return None


def _branch_line(sources: _SourceCache, file: str, line: int) -> bool:
    """Whether ``file:line`` opens a control-flow branch (fail-open on a bad install)."""
    source = sources.read(file) or b""
    try:
        return is_branch_line(language_for_path(file), source, line)
    except Exception:
        return is_branch_line(None, source, line)


def _terminal_line(sources: _SourceCache, file: str, line: int) -> bool:
    """Whether ``file:line`` ends a control-flow path (fail-open on a bad install)."""
    source = sources.read(file) or b""
    try:
        return is_terminal_line(language_for_path(file), source, line)
    except Exception:
        return is_terminal_line(None, source, line)


def _executable_line(sources: _SourceCache, file: str, line: int) -> bool:
    """Whether ``file:line`` starts an executable statement."""
    source = sources.read(file) or b""
    try:
        return is_executable_statement_line(language_for_path(file), source, line)
    except Exception:
        return False


def _summary(elements: list[ElementCheck]) -> dict[str, int]:
    """Return the ``{proposed, grounded, pruned}`` counts for ``elements``."""
    grounded = sum(1 for check in elements if check.grounded)
    return {
        "proposed": len(elements),
        "grounded": grounded,
        "pruned": len(elements) - grounded,
    }


def _nonzero(counts: dict[str, int]) -> dict[str, int]:
    """Drop zero entries so ``capped`` is ``{}`` when no cap bound."""
    return {key: value for key, value in counts.items() if value}


# --- Sequence ----------------------------------------------------------------


def _require(record: dict[str, Any] | None) -> dict[str, Any]:
    """Return ``record``, asserting it is present (grounded messages always are)."""
    assert record is not None
    return record


def _find(checks: list[ElementCheck], ref: str) -> ElementCheck:
    """Return the check with ``ref`` (refs are unique within an element kind)."""
    for check in checks:
        if check.ref == ref:
            return check
    raise KeyError(ref)


def _normalize_participant(raw: dict[str, Any]) -> dict[str, Any]:
    """Return the schema-shaped participant for ``spec_final``."""
    files = [file for file in _as_list(raw.get("files")) if isinstance(file, str)]
    return {
        "name": _norm_str(raw.get("name")),
        "kind": _norm_str(raw.get("kind")),
        "files": [_strip_dot_slash(file) for file in files],
        "service": _norm_optional_str(raw.get("service")),
    }


def _normalize_message(raw: dict[str, Any]) -> dict[str, Any]:
    """Return the schema-shaped message for ``spec_final``."""
    evidence = _as_dict(raw.get("evidence"))
    return {
        "from": _norm_str(raw.get("from")),
        "to": _norm_str(raw.get("to")),
        "label": _norm_str(raw.get("label")),
        "kind": _norm_str(raw.get("kind")),
        "changed": bool(raw.get("changed")),
        "evidence": {
            "file": _strip_dot_slash(_norm_str(evidence.get("file"))),
            "line": _norm_line(evidence.get("line")),
            "symbol": _norm_str(evidence.get("symbol")),
        },
    }


def _ground_participants(
    repo_root: Path,
    sources: _SourceCache,
    participants: list[dict[str, Any]],
    source_indices: dict[str, list[int]],
) -> tuple[list[ElementCheck], dict[str, dict[str, Any]]]:
    """Check every participant and return its checks plus the accepted map.

    An ``external`` participant is misused when it declares files (it has none
    in the repository by definition) or when it sources any message other than
    the entrypoint: the spec allows an external actor to open the interaction or
    to receive a call/reply, never to originate a later step, because there
    would be no in-repo line to cite for it.
    """
    checks: list[ElementCheck] = []
    accepted: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for raw in participants:
        record = _normalize_participant(raw)
        name = record["name"]
        ref = name or f"<unnamed:{len(checks)}>"
        if not name or record["kind"] not in _PARTICIPANT_KINDS or name in seen:
            checks.append(
                ElementCheck("participant", ref, False, "MALFORMED_ELEMENT")
            )
            continue
        seen.add(name)
        reason: str | None = None
        if record["kind"] == "external":
            later = [i for i in source_indices.get(name, []) if i != 0]
            if record["files"] or later:
                reason = "EXTERNAL_MISUSED"
        elif not record["files"]:
            reason = "PARTICIPANT_NO_FILES"
        else:
            for file in record["files"]:
                normalized, path_reason = _check_path(repo_root, file)
                if path_reason is not None:
                    reason = path_reason
                    break
                if sources.read(normalized) is None:
                    reason = "PARTICIPANT_FILE_MISSING"
                    break
        checks.append(ElementCheck("participant", name, reason is None, reason))
        if reason is None:
            accepted[name] = record
    return checks, accepted


def _message_source_files(
    index: int,
    record: dict[str, Any],
    accepted: dict[str, dict[str, Any]],
) -> list[str]:
    """Return the participant files the message's evidence must live in.

    Normally the ``from`` participant's files -- a call site or a return
    statement is in the caller. The one exception is the entrypoint: when
    message 0 comes from an external actor there is no in-repo file on the
    sending side, and the spec pins its evidence to the handler definition line
    in the receiving participant instead.
    """
    sender = accepted.get(record["from"], {})
    if index == 0 and sender.get("kind") == "external":
        return list(accepted.get(record["to"], {}).get("files", []))
    return list(sender.get("files", []))


def _ground_message(
    repo_root: Path,
    sources: _SourceCache,
    symbols: RepoSymbols,
    read_paths: set[str],
    hunk_ranges: dict[str, list[tuple[int, int]]],
    index: int,
    record: dict[str, Any],
    accepted: dict[str, dict[str, Any]],
) -> ElementCheck:
    """Check one message and rewrite its evidence line on a successful snap."""
    check = ElementCheck("message", str(index), True)
    if record["kind"] not in _MESSAGE_KINDS:
        check.grounded, check.reason = False, "MALFORMED_ELEMENT"
        return check
    # An unknown or ungrounded endpoint is not a separate reason code: the
    # message fails on the endpoint that cannot hold its evidence (source) or
    # cannot own its callee (target), which is what the repair turn must fix.
    if record["from"] not in accepted:
        check.grounded, check.reason = False, "EVIDENCE_NOT_IN_SOURCE_PARTICIPANT"
        return check
    if record["to"] not in accepted:
        check.grounded, check.reason = False, "CALLEE_NOT_DEFINED_IN_TARGET"
        return check

    evidence = record["evidence"]
    file, line, reason = _check_location(
        repo_root, sources, read_paths, evidence["file"], evidence["line"]
    )
    evidence["file"], evidence["line"] = file, line
    if reason is not None:
        check.grounded, check.reason = False, reason
        return check

    source_files = _message_source_files(index, record, accepted)
    if file not in source_files:
        check.grounded, check.reason = False, "EVIDENCE_NOT_IN_SOURCE_PARTICIPANT"
        return check

    symbol = evidence["symbol"]
    if not symbol or not _token_on_line(sources.line_text(file, line), symbol):
        snapped = _snap_symbol(sources, file, line, symbol)
        if snapped is None:
            check.grounded, check.reason = False, "SYMBOL_NOT_ON_LINE"
            check.in_changed_hunk = _in_ranges(hunk_ranges.get(file, []), line)
            return check
        evidence["line"] = line = snapped
        check.snapped_line = snapped
    check.in_changed_hunk = _in_ranges(hunk_ranges.get(file, []), line)

    target = accepted[record["to"]]
    if record["kind"] in ("call", "self") and target["kind"] == "internal":
        strength, defined_at = _resolve_in_files(
            repo_root, symbols, symbol, target["files"]
        )
        if strength is None:
            check.grounded, check.reason = False, "CALLEE_NOT_DEFINED_IN_TARGET"
            return check
        check.strength, check.defined_at = strength, defined_at
    else:
        # A reply names its enclosing function and an outbound call names a
        # client method; neither is a callee this repository defines, so the
        # word-boundary match on the cited line is the whole proof.
        check.strength = "token"
    return check


def _ground_branch(
    repo_root: Path,
    sources: _SourceCache,
    read_paths: set[str],
    hunk_ranges: dict[str, list[tuple[int, int]]],
    ref: str,
    raw: Any,
    message_count: int,
) -> tuple[ElementCheck, dict[str, Any]]:
    """Check one block branch, returning its check and its ``spec_final`` payload."""
    record = _as_dict(raw)
    evidence = _as_dict(record.get("evidence"))
    condition = _norm_str(record.get("condition")).strip()
    indices = [
        index
        for index in _as_list(record.get("messages"))
        if isinstance(index, int)
        and not isinstance(index, bool)
        and 0 <= index < message_count
    ]
    file, line, reason = _check_location(
        repo_root, sources, read_paths, evidence.get("file"), evidence.get("line")
    )
    if reason is None and not condition:
        # A branch with no condition text names no branch: there is nothing for
        # the ``alt``/``opt``/``loop`` header to say and nothing to verify.
        reason = "NOT_A_BRANCH_STATEMENT"
    if reason is None and not _branch_line(sources, file, line):
        reason = "NOT_A_BRANCH_STATEMENT"
    check = ElementCheck("branch", ref, reason is None, reason)
    check.in_changed_hunk = _in_ranges(hunk_ranges.get(file, []), line)
    payload = {
        "condition": condition,
        "evidence": {"file": file, "line": line},
        "messages": indices,
    }
    return check, payload


@dataclass
class _KeptBlock:
    """One surviving block: its proposal index, kind, and surviving branches."""

    index: int
    kind: str
    branches: list[tuple[int, dict[str, Any]]]


def _assemble_blocks(
    kept: list[_KeptBlock], final_positions: dict[int, int]
) -> list[_KeptBlock]:
    """Remap branch message indices and drop what is left empty.

    A branch keeps only the messages that survived, remapped to their positions
    in the final ``messages`` list. A branch with no surviving message, and a
    block with no surviving branch, is dropped -- its messages stay in
    ``messages`` and simply render flat, which is exactly the spec's rule for a
    block whose condition could not be grounded.

    Arity is normalized here too: ``alt`` needs at least two branches to mean
    anything, ``opt``/``loop`` take exactly one.
    """
    result: list[_KeptBlock] = []
    for block in kept:
        branches: list[tuple[int, dict[str, Any]]] = []
        for branch_index, payload in block.branches:
            remapped = [
                final_positions[index]
                for index in payload["messages"]
                if index in final_positions
            ]
            if not remapped:
                continue
            branches.append(
                (branch_index, {**payload, "messages": remapped})
            )
        if block.kind == "alt":
            if len(branches) < 2:
                continue
        else:
            branches = branches[:1]
        if not branches:
            continue
        result.append(_KeptBlock(block.index, block.kind, branches))
    return result


def ground_sequence(
    spec: dict[str, Any],
    *,
    repo_root: Path,
    hunk_ranges: dict[str, list[tuple[int, int]]],
    read_paths: set[str],
    symbols: RepoSymbols,
) -> GroundingReport:
    """Check, prune, cap and floor-test a proposed sequence-diagram spec.

    Args:
        spec: The model's proposed spec, shaped per ``SEQUENCE_SPEC_SCHEMA``.
            Malformed entries are reported, never trusted.
        repo_root: Head-tree root every cited path is resolved against.
        hunk_ranges: Head-side changed line ranges per repo-relative path, used
            for ``in_changed_hunk`` and the "at least one changed interaction"
            floor.
        read_paths: Raw completed-read tool-call paths from the diagram phase's
            trajectory. An empty set means every citation fails
            ``FILE_NOT_READ_BY_MODEL``, which is the intended fail-closed
            behavior when the fork's trajectory is missing.
        symbols: Shared definition index for callee resolution.

    Returns:
        The :class:`GroundingReport`. ``spec_final`` always carries exactly the
        schema's keys; the caller renders only when ``omit_reasons`` is empty.
    """
    sources = _SourceCache(repo_root)
    raw_participants = _as_list(spec.get("participants"))
    raw_messages = _as_list(spec.get("messages"))
    raw_blocks = _as_list(spec.get("blocks"))

    # Participant checks need to know which messages each name sources, so the
    # external-actor rule is decidable before any message is adjudicated.
    normalized_messages: list[dict[str, Any] | None] = [
        _normalize_message(raw) if isinstance(raw, dict) else None
        for raw in raw_messages
    ]
    source_indices: dict[str, list[int]] = {}
    for index, record in enumerate(normalized_messages):
        if record is not None:
            source_indices.setdefault(record["from"], []).append(index)

    participant_checks, accepted = _ground_participants(
        repo_root,
        sources,
        [raw for raw in raw_participants if isinstance(raw, dict)],
        source_indices,
    )

    message_checks: list[ElementCheck] = []
    for index, record in enumerate(normalized_messages):
        if record is None:
            message_checks.append(
                ElementCheck("message", str(index), False, "MALFORMED_ELEMENT")
            )
            continue
        message_checks.append(
            _ground_message(
                repo_root,
                sources,
                symbols,
                read_paths,
                hunk_ranges,
                index,
                record,
                accepted,
            )
        )

    block_checks: list[ElementCheck] = []
    branch_checks: list[ElementCheck] = []
    kept_blocks: list[_KeptBlock] = []
    for block_index, raw_block in enumerate(raw_blocks):
        block_ref = f"b{block_index}"
        record = _as_dict(raw_block)
        kind = _norm_str(record.get("kind"))
        raw_branches = _as_list(record.get("branches"))
        if not isinstance(raw_block, dict) or kind not in _BLOCK_KINDS:
            block_checks.append(
                ElementCheck("block", block_ref, False, "MALFORMED_ELEMENT")
            )
            for branch_index in range(len(raw_branches)):
                branch_checks.append(
                    ElementCheck(
                        "branch",
                        f"{block_ref}.{branch_index}",
                        False,
                        "MALFORMED_ELEMENT",
                    )
                )
            continue
        surviving: list[tuple[int, dict[str, Any]]] = []
        for branch_index, raw_branch in enumerate(raw_branches):
            check, payload = _ground_branch(
                repo_root,
                sources,
                read_paths,
                hunk_ranges,
                f"{block_ref}.{branch_index}",
                raw_branch,
                len(normalized_messages),
            )
            branch_checks.append(check)
            if check.grounded:
                surviving.append((branch_index, payload))
        block_checks.append(
            ElementCheck(
                "block",
                block_ref,
                bool(surviving),
                None if surviving else "NOT_A_BRANCH_STATEMENT",
            )
        )
        if surviving:
            kept_blocks.append(_KeptBlock(block_index, kind, surviving))

    # --- prune ---------------------------------------------------------------
    kept_messages = [
        index for index, check in enumerate(message_checks) if check.grounded
    ]
    used = {
        name
        for index in kept_messages
        for name in (
            _require(normalized_messages[index])["from"],
            _require(normalized_messages[index])["to"],
        )
    }
    kept_participants = [name for name in accepted if name in used]
    pruned_block_count = len(
        _assemble_blocks(
            kept_blocks, {index: pos for pos, index in enumerate(kept_messages)}
        )
    )

    # --- cap -----------------------------------------------------------------
    capped: dict[str, int] = {"participants": 0, "messages": 0, "blocks": 0}
    if len(kept_participants) > DIAGRAM_MAX_PARTICIPANTS:
        capped["participants"] = len(kept_participants) - DIAGRAM_MAX_PARTICIPANTS
        kept_participants = kept_participants[:DIAGRAM_MAX_PARTICIPANTS]
    participant_set = set(kept_participants)
    orphaned = [
        index
        for index in kept_messages
        if _require(normalized_messages[index])["from"] not in participant_set
        or _require(normalized_messages[index])["to"] not in participant_set
    ]
    if orphaned:
        dropped = set(orphaned)
        capped["messages"] += len(orphaned)
        kept_messages = [index for index in kept_messages if index not in dropped]
    if len(kept_messages) > DIAGRAM_MAX_MESSAGES:
        capped["messages"] += len(kept_messages) - DIAGRAM_MAX_MESSAGES
        kept_messages = kept_messages[:DIAGRAM_MAX_MESSAGES]
    final_positions = {index: pos for pos, index in enumerate(kept_messages)}
    final_blocks = _assemble_blocks(kept_blocks[:DIAGRAM_MAX_BLOCKS], final_positions)
    capped["blocks"] = max(0, pruned_block_count - len(final_blocks))

    # --- assemble ------------------------------------------------------------
    spec_final: dict[str, Any] = {
        "participants": [accepted[name] for name in kept_participants],
        "messages": [_require(normalized_messages[index]) for index in kept_messages],
        "blocks": [
            {
                "kind": block.kind,
                "branches": [payload for _, payload in block.branches],
            }
            for block in final_blocks
        ],
    }
    for position, name in enumerate(kept_participants):
        _find(participant_checks, name).final_index = position
    for index, position in final_positions.items():
        message_checks[index].final_index = position
    for position, block in enumerate(final_blocks):
        _find(block_checks, f"b{block.index}").final_index = position
        for branch_position, (branch_index, _) in enumerate(block.branches):
            _find(
                branch_checks, f"b{block.index}.{branch_index}"
            ).final_index = branch_position

    # --- floor ---------------------------------------------------------------
    omit_reasons: list[str] = []
    if len(spec_final["messages"]) < _MIN_MESSAGES:
        omit_reasons.append("TOO_FEW_MESSAGES")
    if len(spec_final["participants"]) < _MIN_PARTICIPANTS:
        omit_reasons.append("TOO_FEW_PARTICIPANTS")
    if not any(
        message_checks[index].in_changed_hunk for index in kept_messages
    ):
        omit_reasons.append("NO_CHANGED_INTERACTION")

    elements = participant_checks + message_checks + block_checks + branch_checks
    return GroundingReport(
        elements=elements,
        spec_final=spec_final,
        summary=_summary(elements),
        capped=_nonzero(capped),
        root_range=None,
        omit_reasons=omit_reasons,
        rejected=None,
    )


# --- Flowchart ---------------------------------------------------------------


def _normalize_node(raw: dict[str, Any]) -> dict[str, Any]:
    """Return the schema-shaped flowchart node for ``spec_final``."""
    evidence = _as_dict(raw.get("evidence"))
    return {
        "id": _norm_str(raw.get("id")),
        "kind": _norm_str(raw.get("kind")),
        "label": _norm_str(raw.get("label")),
        "evidence": {
            "file": _strip_dot_slash(_norm_str(evidence.get("file"))),
            "line": _norm_line(evidence.get("line")),
            "symbol": _norm_optional_str(evidence.get("symbol")),
        },
    }


def _normalize_edge(raw: dict[str, Any]) -> dict[str, Any]:
    """Return the schema-shaped flowchart edge for ``spec_final``."""
    return {
        "from": _norm_str(raw.get("from")),
        "to": _norm_str(raw.get("to")),
        "label": _norm_optional_str(raw.get("label")),
    }


def _match_candidate(
    root: dict[str, Any], candidate_roots: list[CandidateRoot]
) -> CandidateRoot | None:
    """Return the candidate the proposed root echoes exactly, else None.

    Exact ``(file, name, line)`` match: the candidate list is handed to the
    model verbatim, so a root it did not copy from that list is a root whose
    range this pass never computed -- and without a range ``NODE_OUTSIDE_ROOT``
    is undecidable, which is the whole point of constraining the choice.
    """
    file = _strip_dot_slash(_norm_str(root.get("file")))
    name = _norm_str(root.get("name"))
    line = _norm_line(root.get("line"))
    for candidate in candidate_roots:
        if (
            _strip_dot_slash(candidate.file) == file
            and candidate.name == name
            and candidate.line == line
        ):
            return candidate
    return None


def _ground_node(
    repo_root: Path,
    sources: _SourceCache,
    symbols: RepoSymbols,
    read_paths: set[str],
    hunk_ranges: dict[str, list[tuple[int, int]]],
    record: dict[str, Any],
    root_file: str,
    root_range: tuple[int, int],
) -> ElementCheck:
    """Check one flowchart node and rewrite its evidence line on a snap."""
    check = ElementCheck("node", record["id"], True)
    if record["kind"] not in _NODE_KINDS:
        check.grounded, check.reason = False, "MALFORMED_ELEMENT"
        return check
    evidence = record["evidence"]
    file, line, reason = _check_location(
        repo_root, sources, read_paths, evidence["file"], evidence["line"]
    )
    evidence["file"], evidence["line"] = file, line
    if reason is not None:
        check.grounded, check.reason = False, reason
        return check
    if file != root_file or not (root_range[0] <= line <= root_range[1]):
        check.grounded, check.reason = False, "NODE_OUTSIDE_ROOT"
        return check

    symbol = evidence["symbol"]
    if record["kind"] == "subroutine":
        if not symbol:
            check.grounded, check.reason = False, "MALFORMED_ELEMENT"
            return check
        if not _token_on_line(sources.line_text(file, line), symbol):
            snapped = _snap_symbol(sources, file, line, symbol, within=root_range)
            if snapped is None:
                check.grounded, check.reason = False, "SUBROUTINE_NOT_CALLED_HERE"
                return check
            evidence["line"] = line = snapped
            check.snapped_line = snapped
        strength, defined_at = _resolve_anywhere(repo_root, symbols, symbol)
        if strength is None:
            check.grounded, check.reason = False, "SUBROUTINE_NOT_DEFINED"
            check.in_changed_hunk = _in_ranges(hunk_ranges.get(file, []), line)
            return check
        check.strength, check.defined_at = strength, defined_at
    elif symbol and not _token_on_line(sources.line_text(file, line), symbol):
        snapped = _snap_symbol(sources, file, line, symbol, within=root_range)
        if snapped is None:
            check.grounded, check.reason = False, "SYMBOL_NOT_ON_LINE"
            check.in_changed_hunk = _in_ranges(hunk_ranges.get(file, []), line)
            return check
        evidence["line"] = line = snapped
        check.snapped_line = snapped
        check.strength = "token"
    elif symbol:
        check.strength = "token"

    if record["kind"] == "end" and not _terminal_line(sources, file, line):
        check.grounded, check.reason = False, "NOT_A_TERMINAL_STATEMENT"
    elif record["kind"] == "decision" and not _branch_line(sources, file, line):
        check.grounded, check.reason = False, "NOT_A_BRANCH_STATEMENT"
    elif record["kind"] not in {"end", "decision"} and not _executable_line(
        sources, file, line
    ):
        check.grounded, check.reason = False, "NOT_AN_EXECUTABLE_STATEMENT"
    check.in_changed_hunk = _in_ranges(hunk_ranges.get(file, []), line)
    return check


def _decision_edge_reason(
    edge: dict[str, Any], nodes: dict[str, dict[str, Any]], seen_labels: set[str]
) -> str | None:
    """Return ``DECISION_EDGES_INVALID`` when an edge out of a decision is unusable.

    Every edge leaving a decision must carry a label, and two edges out of one
    decision may not carry the same label -- an unlabeled or duplicated branch
    renders as an unreadable fork and proves nothing about which way control
    goes. The node itself is not un-grounded by this; it is demoted to
    ``process`` by :func:`_structural_pass` once its usable fan-out is known.
    """
    if nodes.get(edge["from"], {}).get("kind") != "decision":
        return None
    label = edge["label"]
    if not label or label in seen_labels:
        return "DECISION_EDGES_INVALID"
    seen_labels.add(label)
    return None


def _structural_pass(
    node_ids: list[str],
    nodes: dict[str, dict[str, Any]],
    edges: list[tuple[str, dict[str, Any]]],
    start_id: str | None,
) -> tuple[list[str], list[tuple[str, dict[str, Any]]], set[str]]:
    """Demote thin decisions and drop everything unreachable from ``start``.

    Runs to a fixed point because the two rules feed each other: dropping an
    unreachable node drops its incoming edge, which can leave a decision with
    one branch, whose demotion is then visible in the next round. Both rules
    only ever remove, so the loop terminates.

    Returns the surviving node ids in order, the surviving edges, and the ids of
    the decisions demoted to ``process``.
    """
    kept_ids = list(node_ids)
    kept_edges = list(edges)
    demoted: set[str] = set()
    for _ in range(len(node_ids) + 2):
        changed = False
        alive = set(kept_ids)
        kept_edges = [
            item
            for item in kept_edges
            if item[1]["from"] in alive and item[1]["to"] in alive
        ]
        for node_id in kept_ids:
            if node_id in demoted or nodes[node_id]["kind"] != "decision":
                continue
            labels = {
                item[1]["label"]
                for item in kept_edges
                if item[1]["from"] == node_id and item[1]["label"]
            }
            if len(labels) < 2:
                demoted.add(node_id)
                changed = True
        reachable: set[str] = set()
        if start_id is not None and start_id in alive:
            frontier = [start_id]
            reachable.add(start_id)
            while frontier:
                current = frontier.pop()
                for item in kept_edges:
                    if item[1]["from"] == current and item[1]["to"] not in reachable:
                        reachable.add(item[1]["to"])
                        frontier.append(item[1]["to"])
        if set(kept_ids) != reachable:
            kept_ids = [node_id for node_id in kept_ids if node_id in reachable]
            changed = True
        if not changed:
            break
    return kept_ids, kept_edges, demoted


def _rejected_root(root: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return the schema-shaped root, or None when it is not even well-formed."""
    if root is None:
        return None
    file = _strip_dot_slash(_norm_str(root.get("file")))
    name = _norm_str(root.get("name"))
    line = _norm_line(root.get("line"))
    if not file or not name or line < 1:
        return None
    return {"file": file, "name": name, "line": line}


def ground_flowchart(
    spec: dict[str, Any],
    *,
    repo_root: Path,
    hunk_ranges: dict[str, list[tuple[int, int]]],
    read_paths: set[str],
    candidate_roots: list[CandidateRoot],
    symbols: RepoSymbols,
) -> GroundingReport:
    """Check, prune, cap and floor-test a proposed flowchart spec.

    Args:
        spec: The model's proposed spec, shaped per ``FLOWCHART_SPEC_SCHEMA``.
        repo_root: Head-tree root every cited path is resolved against.
        hunk_ranges: Head-side changed line ranges per repo-relative path. The
            root's range must still overlap one of them, re-checked here so a
            repair turn cannot re-root the diagram onto unchanged code.
        read_paths: Raw completed-read tool-call paths from the diagram phase's
            trajectory; an empty set fails every citation closed.
        candidate_roots: The run's eligible roots. A root outside this list is
            rejected outright -- it has no verified range, so no node inside it
            could be checked.
        symbols: Shared definition index for subroutine resolution.

    Returns:
        The :class:`GroundingReport`. ``rejected`` is set when the root itself
        failed, in which case ``spec_final`` carries no nodes or edges.
    """
    sources = _SourceCache(repo_root)
    raw_root = spec.get("root")
    root: dict[str, Any] | None = _as_dict(raw_root) if isinstance(raw_root, dict) else None
    candidate = _match_candidate(root, candidate_roots) if root is not None else None
    root_name = _norm_str(root.get("name")) if root is not None else ""
    root_ref = root_name or "<no-root>"
    if candidate is None or not _overlaps_hunks(candidate, hunk_ranges):
        check = ElementCheck("root", root_ref, False, "ROOT_NOT_CANDIDATE")
        return GroundingReport(
            elements=[check],
            spec_final={"root": _rejected_root(root), "nodes": [], "edges": []},
            summary=_summary([check]),
            capped={},
            root_range=None,
            omit_reasons=["TOO_FEW_NODES"],
            rejected="ROOT_NOT_CANDIDATE",
        )
    root_range = (candidate.line, candidate.end_line)
    root_file = _strip_dot_slash(candidate.file)
    root_check = ElementCheck(
        "root", root_ref, True, in_changed_hunk=True, final_index=0
    )
    root_final = {"file": root_file, "name": candidate.name, "line": candidate.line}

    # --- node checks ---------------------------------------------------------
    node_checks: list[ElementCheck] = []
    nodes: dict[str, dict[str, Any]] = {}
    node_order: list[str] = []
    seen_ids: set[str] = set()
    start_seen = False
    for position, raw_node in enumerate(_as_list(spec.get("nodes"))):
        if not isinstance(raw_node, dict):
            node_checks.append(
                ElementCheck("node", f"<malformed:{position}>", False, "MALFORMED_ELEMENT")
            )
            continue
        record = _normalize_node(raw_node)
        node_id = record["id"]
        if not node_id or node_id in seen_ids:
            node_checks.append(
                ElementCheck(
                    "node", node_id or f"<unnamed:{position}>", False, "MALFORMED_ELEMENT"
                )
            )
            continue
        seen_ids.add(node_id)
        check = _ground_node(
            repo_root,
            sources,
            symbols,
            read_paths,
            hunk_ranges,
            record,
            root_file,
            root_range,
        )
        if check.grounded and record["kind"] == "start":
            if start_seen:
                check.grounded, check.reason = False, "MULTIPLE_START"
            else:
                start_seen = True
        node_checks.append(check)
        if check.grounded:
            nodes[node_id] = record
            node_order.append(node_id)

    # --- edge checks ---------------------------------------------------------
    edge_checks: list[ElementCheck] = []
    edges: list[tuple[str, dict[str, Any]]] = []
    seen_refs: set[str] = set()
    decision_labels: dict[str, set[str]] = {}
    for position, raw_edge in enumerate(_as_list(spec.get("edges"))):
        record = _normalize_edge(_as_dict(raw_edge))
        ref = f"{record['from']}->{record['to']}"
        if (
            not isinstance(raw_edge, dict)
            or not record["from"]
            or not record["to"]
            or ref in seen_refs
        ):
            edge_checks.append(
                ElementCheck(
                    "edge", ref if record["from"] or record["to"] else f"<malformed:{position}>",
                    False,
                    "MALFORMED_ELEMENT",
                )
            )
            continue
        seen_refs.add(ref)
        if record["from"] not in nodes or record["to"] not in nodes:
            edge_checks.append(
                ElementCheck("edge", ref, False, "EDGE_ENDPOINT_UNGROUNDED")
            )
            continue
        reason = _decision_edge_reason(
            record, nodes, decision_labels.setdefault(record["from"], set())
        )
        edge_checks.append(ElementCheck("edge", ref, reason is None, reason))
        if reason is None:
            edges.append((ref, record))

    start_id = next(
        (node_id for node_id in node_order if nodes[node_id]["kind"] == "start"), None
    )

    # --- prune ---------------------------------------------------------------
    kept_ids, kept_edges, demoted = _structural_pass(node_order, nodes, edges, start_id)
    pruned_nodes, pruned_edges = len(kept_ids), len(kept_edges)
    # An edge the *prune* pass dropped is ungrounded per the spec's own wording
    # ("an edge whose node was pruned"), whether its endpoint failed its own
    # check or fell out as unreachable. Edges the cap drops below are a
    # different story and stay grounded: counting them here as well as in
    # ``capped`` would report one drop twice, and would tell the repair turn
    # that a perfectly good edge was unproven.
    survived_prune = {ref for ref, _ in kept_edges}
    for check in edge_checks:
        if check.grounded and check.ref not in survived_prune:
            check.grounded, check.reason = False, "EDGE_ENDPOINT_UNGROUNDED"

    # --- cap -----------------------------------------------------------------
    capped: dict[str, int] = {"nodes": 0, "edges": 0}
    if len(kept_ids) > DIAGRAM_MAX_NODES:
        head = kept_ids[:DIAGRAM_MAX_NODES]
        if start_id is not None and start_id not in head:
            # The start node is the only node the diagram cannot lose: without
            # it nothing is reachable and the whole kind collapses. Trade the
            # last node in spec order for it rather than exceed the cap.
            head = [start_id, *head[:-1]]
        kept_ids = head
    alive = set(kept_ids)
    kept_edges = [
        item
        for item in kept_edges
        if item[1]["from"] in alive and item[1]["to"] in alive
    ]
    kept_edges = kept_edges[:DIAGRAM_MAX_EDGES]
    kept_ids, kept_edges, demoted = _structural_pass(
        kept_ids, nodes, kept_edges, start_id
    )
    capped["nodes"] = max(0, pruned_nodes - len(kept_ids))
    capped["edges"] = max(0, pruned_edges - len(kept_edges))

    # --- assemble ------------------------------------------------------------
    spec_final: dict[str, Any] = {
        "root": root_final,
        "nodes": [
            {
                **nodes[node_id],
                "kind": "process" if node_id in demoted else nodes[node_id]["kind"],
            }
            for node_id in kept_ids
        ],
        "edges": [dict(record) for _, record in kept_edges],
    }
    for position, node_id in enumerate(kept_ids):
        _find(node_checks, node_id).final_index = position
    for position, (ref, _) in enumerate(kept_edges):
        _find(edge_checks, ref).final_index = position

    # --- floor ---------------------------------------------------------------
    kinds = [node["kind"] for node in spec_final["nodes"]]
    omit_reasons: list[str] = []
    if len(kinds) < _MIN_NODES or "start" not in kinds:
        omit_reasons.append("TOO_FEW_NODES")
    if "end" not in kinds:
        omit_reasons.append("NO_END")
    if "decision" not in kinds:
        omit_reasons.append("NO_DECISION")

    elements = [root_check, *node_checks, *edge_checks]
    return GroundingReport(
        elements=elements,
        spec_final=spec_final,
        summary=_summary(elements),
        capped=_nonzero(capped),
        root_range=root_range,
        omit_reasons=omit_reasons,
        rejected=None,
    )


def _overlaps_hunks(
    candidate: CandidateRoot, hunk_ranges: dict[str, list[tuple[int, int]]]
) -> bool:
    """Whether the candidate's range overlaps a head-side changed hunk."""
    ranges = hunk_ranges.get(_strip_dot_slash(candidate.file), [])
    return any(
        start <= candidate.end_line and candidate.line <= end for start, end in ranges
    )


__all__ = [
    "OMIT_REASONS",
    "REASON_CODES",
    "ElementCheck",
    "GroundingReport",
    "RepoSymbols",
    "ground_flowchart",
    "ground_sequence",
]
