"""Pre-report finding-location validator (issue #745).

Every finding ``file:line`` must land on a real changed line before it reaches
the report. This module validates a finding against the persisted hunk index
(the run-time authority for changed file/line ranges) and returns a five-field
check, then snaps in-tolerance findings to the nearest hunk boundary and
demotes-with-annotation beyond-tolerance ones. It owns authority; the posting
time ``resolve_line``/``snap_to_hunk`` backstop in ``pr_review`` is knowingly
left as a no-op-on-valid fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from daydream.pr_review import HUNK_TOLERANCE


@dataclass
class LocationCheck:
    """Five-field boolean/int check for one finding location."""

    file_exists: bool
    line_exists: bool
    in_hunk: bool
    nearest_hunk: tuple[int, int] | None
    distance: int | None


def validate_finding(
    index: dict[str, Any],
    file: str,
    line: int,
    tolerance: int = HUNK_TOLERANCE,
) -> LocationCheck:
    """Validate one finding ``(file, line)`` against the hunk ``index``.

    ``index`` is the persisted hunk index shape returned by
    ``daydream.hunk_index.parse_hunks`` / ``load_hunk_index``: ``{path:
    {"hunks": [{"new_start","new_end",...}], ...}}``.

    Returns:
        A :class:`LocationCheck` with:
          - ``file_exists``: ``file`` is a key in the index.
          - ``in_hunk``: ``line`` falls within any ``[new_start, new_end]``.
          - ``nearest_hunk``: the ``(new_start, new_end)`` range minimizing
            distance to ``line`` (distance to a range is 0 inside it, else the
            min distance to either boundary).
          - ``distance``: ``0`` when in-hunk, else the min boundary distance.
          - ``line_exists``: ``line >= 1`` and ``line <= max(new_end)`` across
            the file's hunks when ``file_exists`` (a hunk-derived approximation
            of existence -- the index holds no full file length); ``False``
            when the file is absent.

        A missing file key yields an all-``False``/``None`` check, never raises.
    """
    info = index.get(file)
    if info is None:
        return LocationCheck(False, False, False, None, None)

    hunks = info.get("hunks") or []
    in_hunk = False
    best: tuple[int, int] | None = None
    best_dist: int | None = None
    max_new_end = 0
    for start, end in _ranges(hunks):
        max_new_end = max(max_new_end, end)
        dist = _distance(line, start, end)
        if dist == 0:
            in_hunk = True
            best = (start, end)
            best_dist = 0
            continue
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best = (start, end)
    line_exists = line >= 1 and line <= max_new_end
    return LocationCheck(
        file_exists=True,
        line_exists=line_exists,
        in_hunk=in_hunk,
        nearest_hunk=best,
        distance=0 if in_hunk else best_dist,
    )


def _ranges(hunks: list[dict[str, Any]]) -> list[tuple[int, int]]:
    return [(h["new_start"], h["new_end"]) for h in hunks]


def _distance(line: int, start: int, end: int) -> int:
    if start <= line <= end:
        return 0
    if line < start:
        return start - line
    return line - end


def validate_records(
    index: dict[str, Any],
    records: list[dict[str, Any]],
    tolerance: int = HUNK_TOLERANCE,
) -> list[dict[str, Any]]:
    """Validate every finding record and snap/demote as needed.

    For each record with a ``file`` + integer ``line``:
      - ``distance <= tolerance`` and not in-hunk: snap ``record["line"]`` to
        the nearest hunk boundary (the record's ``nearest_hunk`` start or end)
        and align the ``file:line`` citation in ``record["evidence"]`` to the
        snapped line, so the evidence and snapped line never diverge.
      - ``distance > tolerance``: demote the record in place -- lower
        ``record["severity"]`` to ``"low"``, ``record["confidence"]`` to
        ``"LOW"`` and set ``record["location_note"]`` to a demotion annotation
        naming the file, cited line, nearest hunk and distance -- so the
        unverified citation no longer reaches the report at full severity.

    The mutation surface is the snap (line + evidence) and the demote
    (severity/confidence/location_note); an in-hunk record is untouched.
    Records without a file/line (or with a non-int line) pass through unchanged.
    Never raises. Returns the (possibly mutated) record list.
    """
    for record in records:
        file = record.get("file")
        line = record.get("line")
        if file is None or not isinstance(line, int):
            continue
        check = validate_finding(index, file, line, tolerance=tolerance)
        if check.distance is None or check.nearest_hunk is None:
            continue
        if check.in_hunk or check.distance == 0:
            continue
        if check.distance <= tolerance:
            start, end = check.nearest_hunk
            snapped = start if line < start else end
            record["line"] = snapped
            _align_evidence(record, file, line, snapped)
        else:
            start, end = check.nearest_hunk
            record["severity"] = "low"
            record["confidence"] = "LOW"
            record["location_note"] = (
                f"cited line {line} in {file} is {check.distance} lines from the "
                f"nearest hunk {start}..{end} (tolerance {tolerance}); demoted to "
                f"informational (unverified citation)."
            )
    return records


def _align_evidence(
    record: dict[str, Any], file: str, old_line: int, new_line: int
) -> None:
    """Realign the ``file:old_line`` citation in ``record["evidence"]`` to ``new_line``.

    Snapping ``record["line"]`` to a hunk boundary otherwise leaves a stale original
    ``file:old_line`` citation in ``record["evidence"]``, so the merged item would
    carry a snapped boundary line next to a mismatched evidence citation. Only the
    citation for this record's own ``file`` and the pre-snap ``old_line`` is
    rewritten; unrelated evidence text is preserved verbatim. Missing or
    non-string evidence is left untouched.
    """
    evidence = record.get("evidence")
    if not isinstance(evidence, str):
        return
    old_citation = f"{file}:{old_line}"
    new_citation = f"{file}:{new_line}"
    if old_citation in evidence:
        record["evidence"] = evidence.replace(old_citation, new_citation)
