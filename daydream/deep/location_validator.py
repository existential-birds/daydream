"""Validate report citations against the persisted changed-line index.

Snap nearby citations and demote distant ones without losing their originals.
Posting separately rechecks placement against the live branch diff.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from daydream.hunk_index import range_distance
from daydream.pr_review import HUNK_TOLERANCE


@dataclass
class LocationCheck:
    """Three-field boolean/int check for one finding location."""

    in_hunk: bool
    nearest_hunk: tuple[int, int] | None
    distance: int | None


def validate_finding(
    index: dict[str, Any],
    file: str,
    line: int,
) -> LocationCheck:
    """Find the nearest persisted new-side hunk and boundary distance for a citation.

    Distance is zero inside a hunk. Missing files return no hunk or distance.
    """
    info = index.get(file)
    if info is None:
        return LocationCheck(False, None, None)

    hunks = info.get("hunks") or []
    best: tuple[int, int] | None = None
    best_dist: int | None = None
    for start, end in ((h["new_start"], h["new_end"]) for h in hunks):
        dist = range_distance(line, start, end)
        if dist == 0:
            best = (start, end)
            best_dist = 0
            continue
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best = (start, end)
    return LocationCheck(
        in_hunk=best_dist == 0,
        nearest_hunk=best,
        distance=best_dist,
    )


def validate_records(
    index: dict[str, Any],
    records: list[dict[str, Any]],
    tolerance: int = HUNK_TOLERANCE,
) -> list[dict[str, Any]]:
    """Snap nearby citations and demote distant ones in place, returning the same list.

    For relocated/distrusted records, preserve the original location_cited_line.
    Snapping aligns both line and its evidence citation. Demotion preserves
    severity_before_demotion, sets location_distrust, lowers severity/confidence,
    and adds a location_note. In-hunk records and records without usable file/line
    pass through unchanged.

    This is a single-pass operation: reapplying can overwrite preserved originals.
    """
    for record in records:
        file = record.get("file")
        line = record.get("line")
        if file is None or not isinstance(line, int):
            continue
        # Structural whole-file findings (``lens="structural"``) carry
        # ``line: 0`` -- a whole-file citation, not a real changed line.
        # ``_is_evidenced`` exempts the structural lens from the grounded-citation
        # requirement for exactly this reason, so the snap/demote must not treat
        # ``line: 0`` as a citation just because the file is in the hunk index
        # and snap it to a boundary or demote a whole-file finding (issue #745).
        if record.get("lens") == "structural" and line == 0:
            continue
        check = validate_finding(index, file, line)
        if check.distance is None or check.nearest_hunk is None:
            continue
        if check.in_hunk:
            continue
        # Non-destructive location record (issue #1106): the snap below
        # overwrites ``line``, so preserve what the reviewer actually cited or
        # the citation's accuracy becomes unmeasurable after the fact. Mirrors
        # the ``severity_before_demotion`` precedent in the demote branch.
        record["location_cited_line"] = line
        if check.distance <= tolerance:
            start, end = check.nearest_hunk
            snapped = start if line < start else end
            record["line"] = snapped
            _align_evidence(record, file, line, snapped)
        else:
            start, end = check.nearest_hunk
            # Non-destructive demotion (issue #972 R2): keep the originally
            # adjudicated severity recoverable and mark the record so the
            # approval gate and the report renderer can surface the demotion
            # instead of silently reading the lowered value as the verdict.
            record["severity_before_demotion"] = record.get("severity")
            record["location_distrust"] = True
            record["severity"] = "low"
            record["confidence"] = "LOW"
            record["location_note"] = (
                f"cited line {line} in {file} is {check.distance} lines from the "
                f"nearest hunk {start}..{end} (tolerance {tolerance}); demoted to "
                f"low (unverified citation)."
            )
    return records


def _align_evidence(
    record: dict[str, Any], file: str, old_line: int, new_line: int
) -> None:
    """Rewrite only this record's file:old_line citation; preserve all unrelated text.

    Missing or non-string evidence is unchanged.
    """
    evidence = record.get("evidence")
    if not isinstance(evidence, str):
        return
    old_citation = f"{file}:{old_line}"
    new_citation = f"{file}:{new_line}"
    if old_citation in evidence:
        record["evidence"] = evidence.replace(old_citation, new_citation)
