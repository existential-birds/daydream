"""Resumable keyboard-driven terminal curation client.

Pure-client UI over :mod:`daydream.benchmark.curation`: every mutating action
``[a/e/n/x/c/r/d/z/i/q]`` maps one-to-one onto a service operation and never
mutates the case YAML/model directly. Rendering is plain-string builders for
deterministic tests; Rich stays available for live styling.
"""

from pathlib import Path
from typing import Any, Callable

from daydream.benchmark import curation as cu


def parse_indices(spec: str, n: int) -> list[int]:
    """Parse a comma-separated 1-based selector into sorted unique 0-based indices.

    Accepts single numbers and a single ``a-b`` range (a reversed ``b-a`` range
    spans ``a..b`` inclusive). Raises :class:`ValueError` for any index or range
    endpoint outside ``1..n`` (including ``0``), a repeated index or an
    overlapping range, non-numeric or empty tokens, a single-point range, and
    multiple range tokens.
    """
    if not spec or not spec.strip():
        raise ValueError("empty index selector")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    tokens = spec.split(",")
    if sum(1 for t in tokens if "-" in t) > 1:
        raise ValueError(f"multiple ranges not allowed in {spec!r}")
    selected: set[int] = set()
    for token in tokens:
        if not token:
            raise ValueError(f"empty segment in {spec!r}")
        if "-" in token:
            parts = token.split("-")
            if len(parts) != 2:
                raise ValueError(f"malformed range {token!r}")
            if not parts[0].isdigit() or not parts[1].isdigit():
                raise ValueError(f"non-numeric range endpoint in {token!r}")
            start, end = int(parts[0]), int(parts[1])
            if start == end:
                raise ValueError(f"range {token!r} is a single point")
            low, high = min(start, end), max(start, end)
            if low < 1 or high > n:
                raise ValueError(f"range endpoint out of range in {token!r}")
            span = set(range(low, high + 1))
            if span & selected:
                raise ValueError(f"range {token!r} overlaps the selection")
            selected |= span
        else:
            if not token.isdigit():
                raise ValueError(f"non-numeric index {token!r}")
            index = int(token)
            if index < 1 or index > n:
                raise ValueError(f"index out of range {token!r}")
            if index in selected:
                raise ValueError(f"repeated index {token!r}")
            selected.add(index)
    return sorted(i - 1 for i in selected)


_INDEX_COLUMNS = (
    "case_id",
    "pr_number",
    "head_prefix",
    "changed_files",
    "changed_lines",
    "evidence_count",
    "state",
    "gold_mode",
    "gold_count",
)


def render_index_table(cases: list[dict[str, Any]]) -> str:
    """Render a plain-text index header + one row per case (every value ``str``)."""
    lines = [" | ".join(_INDEX_COLUMNS)]
    for case in cases:
        lines.append(" | ".join(str(case.get(k, "-")) for k in _INDEX_COLUMNS))
    return "\n".join(lines)


def _prompt(read_line: Callable[[str], str], message: str) -> str:
    """Print *message* to stdout, then read one input line with *read_line*."""
    print(message, end="", flush=True)
    return read_line(message)


def _run_case(root: Path, case_id: str, read_line: Callable[[str], str]) -> str:
    """Run one case session; returns ``"quit"`` or ``"done"`` (Task 5 expands)."""
    return "quit"


def run_curate_tui(
    root: Path,
    case_id: str | None = None,
    *,
    read_line: Callable[[str], str] | None = None,
) -> int:
    """Drive the resumable curation terminal client.

    Queue mode (``case_id is None``) lists cases, renders the index, and prompts
    for a case (id, or 1-based row number) or ``q`` to quit. Single-case mode
    opens *case_id* once and returns its outcome.
    """
    read = read_line or input
    if case_id is not None:
        _run_case(root, case_id, read)
        return 0
    while True:
        cases = cu.list_cases(root)
        print(render_index_table(cases))
        text = _prompt(read, "case (id or number), or q: ")
        if text in ("a", "A"):
            continue
        stripped = text.strip()
        if stripped in ("q", "Q"):
            return 0
        selected: str = stripped
        if stripped.isdigit():
            index = int(stripped) - 1
            if not (0 <= index < len(cases)):
                print(f"no case at row {stripped}; try again")
                continue
            selected = cases[index]["case_id"]
        if _run_case(root, selected, read) == "quit":
            return 0

