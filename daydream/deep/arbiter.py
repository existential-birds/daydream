"""Pure selection and deterministic grouping for arbiter and suppression passes.

Arbitration selects sufficiently severe or contested findings. Isolated findings
ranked below the severity threshold receive no arbiter review, an intentional
cost trade-off. Group identity and membership are stable resume inputs.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from daydream.deep.dependency import co_locate_groups
from daydream.deep.records import record_uid, stack_name_from_uid
from daydream.severity import CANONICAL_LEVELS, SEVERITY_RANK

# Canonical confidence vocabulary (uppercase HIGH|MEDIUM|LOW schema enum,
# mirroring ``review_profile._CONFIDENCE_LEVELS``).
_CONFIDENCE_LEVELS: frozenset[str] = frozenset(("HIGH", "MEDIUM", "LOW"))


def _severity(record: dict[str, Any]) -> str:
    """Lowercase severity; absent/non-string values cannot qualify by severity."""
    value = record.get("severity")
    return value.lower() if isinstance(value, str) else ""


def _confidence(record: dict[str, Any]) -> str:
    """Normalize a record's confidence to an uppercase string ("" when absent)."""
    value = record.get("confidence")
    return value.upper() if isinstance(value, str) else ""


def _at_or_above(min_severity: str) -> frozenset[str]:
    """Return canonical levels at least as severe as the validated threshold."""
    if min_severity not in SEVERITY_RANK:
        raise ValueError(
            f"min_severity must be one of {', '.join(CANONICAL_LEVELS)}; got {min_severity!r}"
        )
    rank = SEVERITY_RANK[min_severity]
    return frozenset(level for level, level_rank in SEVERITY_RANK.items() if level_rank <= rank)


def contested_indices(
    records: list[dict[str, Any]],
    *,
    contested_only: Iterable[int] = (),
) -> frozenset[int]:
    """Select locations where distinct stacks report different nonempty severities.

    Ordinary records group by (file, line). A contested-only line-0 record also
    joins a file's sole ordinary location, when exactly one exists; widening to
    several locations would conflate unrelated defects. Whole-file records retain
    their own line-0 group. Each record carries its host-minted scope UID.
    """
    severity_exempt = set(contested_only)
    contested: set[int] = set()
    by_location: dict[tuple[Any, Any], list[int]] = defaultdict(list)
    whole_file: dict[Any, list[int]] = defaultdict(list)
    for i, record in enumerate(records):
        if record.get("line") == 0 and i in severity_exempt:
            whole_file[record.get("file")].append(i)
        else:
            by_location[(record.get("file"), record.get("line"))].append(i)
    lines_per_file: dict[Any, set[Any]] = defaultdict(set)
    for file, line in by_location:
        lines_per_file[file].add(line)
    for (file, _line), indices in by_location.items():
        if len(lines_per_file[file]) == 1:
            indices.extend(whole_file.get(file, ()))
    # Preserve an existing ordinary line-0 group, already widened above.
    for file, indices in whole_file.items():
        by_location.setdefault((file, 0), list(indices))

    for indices in by_location.values():
        stacks = {stack_name_from_uid(record_uid(records[i])) for i in indices}
        severities = {severity for i in indices if (severity := _severity(records[i]))}
        if len(stacks) >= 2 and len(severities) >= 2:
            contested.update(indices)
    return frozenset(contested)


def select_arbiter_targets(
    records: list[dict[str, Any]],
    min_severity: str = "high",
    contested_location: bool = True,
    contested_only: Iterable[int] = (),
) -> list[int]:
    """Return sorted unique indices qualifying by severity or contested location.

    Records in contested_only skip severity selection but remain eligible through
    contested_indices. Missing severity cannot qualify by severity. A noncanonical minimum raises ValueError.
    """
    if any(not stack_name_from_uid(record_uid(record)) for record in records):
        raise ValueError("Arbiter records require scope UIDs")
    severity_exempt = set(contested_only)
    eligible = _at_or_above(min_severity)
    selected = {
        i for i, record in enumerate(records)
        if i not in severity_exempt and _severity(record) in eligible
    }
    if contested_location:
        selected.update(contested_indices(records, contested_only=severity_exempt))

    return sorted(selected)


@dataclass(frozen=True)
class ArbiterGroup:
    """A deterministic, location-atomic chunk with a positional resume key.

    Persisted target_uids mirror target_indices to detect changed membership.
    """

    group_id: str
    target_indices: tuple[int, ...]
    target_uids: tuple[str, ...]


def _line_value(record: dict[str, Any]) -> int:
    """Return a record's line as a sortable int (missing/non-int -> -1)."""
    line = record.get("line")
    return line if isinstance(line, int) else -1


def _chunk_location_atomic(
    members: list[int], records: list[dict[str, Any]], max_targets: int
) -> list[list[int]]:
    """Keep each location intact; close at the next location after exceeding the cap.

    The strict > threshold and whole-location runs make max_targets a soft bound.
    """
    chunks: list[list[int]] = []
    current: list[int] = []
    for index in members:
        location = (records[index].get("file"), _line_value(records[index]))
        if current:
            previous = (records[current[-1]].get("file"), _line_value(records[current[-1]]))
            if len(current) > max_targets and location != previous:
                chunks.append(current)
                current = []
        current.append(index)
    if current:
        chunks.append(current)
    return chunks


def _group_severity_rank(group: ArbiterGroup, records: list[dict[str, Any]]) -> int:
    """Most-severe rank in *group* (unknown/absent severity ranks least severe)."""
    return min(
        SEVERITY_RANK.get(_severity(records[i]), len(SEVERITY_RANK))
        for i in group.target_indices
    )


def partition_arbiter_targets(
    records: list[dict[str, Any]],
    target_indices: Iterable[int],
    *,
    edges: dict[str, set[str]],
    max_targets: int,
) -> list[ArbiterGroup]:
    """Group selected records by file dependency component and location.

    Members sort by (file, line, uid, index), then form location-atomic chunks.
    Groups sort by severity and UID tuple before receiving positional IDs.
    Input order does not affect output; repeated indices remain repeated.
    Invalid indices or max_targets below 1 raise ValueError.
    """
    indices = list(target_indices)
    for index in indices:
        if not isinstance(index, int) or index < 0 or index >= len(records):
            raise ValueError(f"arbiter target index out of range: {index!r}")
    if max_targets < 1:
        raise ValueError(f"max_targets must be >= 1; got {max_targets}")

    def file_of(index: int) -> str:
        return str(records[index].get("file"))

    components = co_locate_groups(sorted({file_of(i) for i in indices}), edges)
    unpacked: list[ArbiterGroup] = []
    for component in components:
        members = [i for i in indices if file_of(i) in component]
        members.sort(key=lambda i: (file_of(i), _line_value(records[i]), record_uid(records[i]), i))
        for chunk in _chunk_location_atomic(members, records, max_targets):
            unpacked.append(
                ArbiterGroup(
                    "",
                    tuple(chunk),
                    tuple(record_uid(records[i]) for i in chunk),
                )
            )
    unpacked.sort(key=lambda group: (_group_severity_rank(group, records), group.target_uids))
    return [
        ArbiterGroup(f"arbiter-group-{i}", group.target_indices, group.target_uids)
        for i, group in enumerate(unpacked)
    ]


def select_suppression_targets(
    records: list[dict[str, Any]],
    exclude: Iterable[int] = (),
    severity_classes: tuple[str, ...] = ("low",),
    confidence_classes: tuple[str, ...] = ("LOW",),
) -> list[int]:
    """Select records matching a severity or confidence class, except excluded indices.

    Callers exclude arbiter targets to keep high-severity/contested findings out.
    Classes are case-normalized and validated; results retain record order.
    """
    classes = frozenset(cls.lower() for cls in severity_classes)
    unknown = classes - frozenset(CANONICAL_LEVELS)
    if unknown:
        raise ValueError(
            f"severity_classes must be a subset of {', '.join(sorted(CANONICAL_LEVELS))}; "
            f"got unknown value(s): {', '.join(sorted(unknown))}"
        )
    confidence = frozenset(cnf.upper() for cnf in confidence_classes)
    unknown_conf = confidence - _CONFIDENCE_LEVELS
    if unknown_conf:
        raise ValueError(
            f"confidence_classes must be a subset of {', '.join(sorted(_CONFIDENCE_LEVELS))}; "
            f"got unknown value(s): {', '.join(sorted(unknown_conf))}"
        )
    excluded = set(exclude)
    return [
        i for i, record in enumerate(records)
        if i not in excluded and (
            _confidence(record) in confidence or _severity(record) in classes
        )
    ]
