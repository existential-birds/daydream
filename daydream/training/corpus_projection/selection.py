"""Deterministic output-share caps over projected records."""

import math
from collections import Counter
from collections.abc import Callable, Mapping
from typing import Any, NoReturn, cast

Record = dict[str, object]


def retain_group_limits(
    records: list[Record], key: Callable[[Record], Any], limits: Mapping[Any, int],
) -> tuple[list[Record], dict[Any, int]]:
    """Keep the first records within each configured group limit, counting drops."""
    kept: list[Record] = []
    taken: Counter[Any] = Counter()
    excluded: Counter[Any] = Counter()
    for record in records:
        group = key(record)
        limit = limits.get(group)
        if limit is not None and taken[group] >= limit:
            excluded[group] += 1
        else:
            kept.append(record)
            taken[group] += 1
    return kept, dict(excluded)


_SHARE_DIMENSIONS: tuple[tuple[str, Callable[[Record], Any]], ...] = (
    # One dimension table consumed by both the share-cap selection stage and
    # the report builder, so the vocabulary uses (limit keys, configured keys,
    # applied keys, exclusion-key prefixes, ``max_<name>_share`` flags) can
    # never drift apart.
    ("repo", lambda r: cast(dict[str, Any], r.get("lineage") or {}).get("repo_slug")),
    ("profile", lambda r: cast(dict[str, Any], r.get("profile") or {}).get("profile_name")),
)


def _max_keep_count(total: int, limit: float) -> int:
    """Largest ``allowed`` with ``allowed / total <= limit`` (float-safety
    guards around the floor; the same arithmetic runs on every pass, so the
    result is deterministic and order-invariant)."""
    allowed = math.floor(limit * total)
    while (allowed + 1) / total <= limit:
        allowed += 1
    while allowed > 0 and allowed / total > limit:
        allowed -= 1
    return allowed


def _raise_share_caps_fail_closed(
    *,
    limit: float,
    dimension: str,
    population: int | None = None,
    value: Any | None = None,
    values: list[Any] | None = None,
) -> NoReturn:
    """Refuse a cap that would empty a nonempty population, naming the dimension."""
    flag = f"max_{dimension}_share"
    if values is not None:
        raise ValueError(
            f"{flag}={limit} would reduce the total population to zero across "
            f"{dimension} values {sorted(str(v) for v in values)} — fail-closed"
        )
    raise ValueError(
        f"{flag}={limit} for {dimension}={value!r} would reduce the total population "
        f"to zero (population {population}) — fail-closed"
    )


def _apply_share_caps(
    records: list[Record], limits: Mapping[str, float | None],
) -> tuple[list[Record], dict[str, int]]:
    """Keep lowest record ids until all configured output shares satisfy their caps.

    Apply repo, then profile, repeating the sequence to a fixed point:
    correlated dimensions can make an earlier cap exceed its limit again.
    None values form their own (none) bucket. Count each exclusion by dimension
    and value. An initially empty corpus stays empty, but trimming a nonempty
    population to zero raises ValueError. Input order never affects selection."""
    dimensions: list[tuple[str, Callable[[Record], Any], float]] = []
    for name, getter in _SHARE_DIMENSIONS:
        limit = limits.get(name)
        if limit is not None:
            dimensions.append((name, getter, limit))
    exclusions: dict[str, int] = {}
    population = sorted(records, key=lambda r: str(r["record_id"]))
    if not population:
        # Empty population (no decisive findings): configured caps exclude
        # nothing — a previously completing zero-record build stays completing.
        return population, exclusions
    entry_total = len(population)
    for dimension, getter, limit in dimensions:
        # Entry check for later dimensions: earlier trims would otherwise make a different dimension refuse,
        # or report a smaller population.
        entry_values = set(map(getter, population))
        if len(entry_values) == 1 and _max_keep_count(entry_total, limit) == 0:
            _raise_share_caps_fail_closed(
                limit=limit,
                dimension=dimension,
                population=entry_total,
                value=next(iter(entry_values)),
            )
    previous_size = -1
    while len(population) != previous_size:
        previous_size = len(population)
        for dimension, getter, limit in dimensions:
            while True:
                total = len(population)
                counts = Counter(map(getter, population))
                over: dict[Any, int] = {}
                for value, count in counts.items():
                    if count / total > limit:
                        # Largest keep-count whose share of the current
                        # population is still <= limit.
                        allowed = _max_keep_count(total, limit)
                        if allowed == 0 and count == total:
                            # Sole remaining value: trimming it would empty the
                            # population, so fail closed (never emit a lone
                            # value above its cap).
                            _raise_share_caps_fail_closed(
                                limit=limit,
                                dimension=dimension,
                                population=total,
                                value=value,
                            )
                        over[value] = allowed
                if not over:
                    break
                kept, dropped = retain_group_limits(population, getter, over)
                for value, count in dropped.items():
                    key = f"{dimension}:{str(value) if value is not None else '(none)'}"
                    exclusions[key] = exclusions.get(key, 0) + count
                if not kept:
                    _raise_share_caps_fail_closed(
                        limit=limit,
                        dimension=dimension,
                        values=list(over),
                    )
                population = kept
    return population, exclusions


def _share_caps_report(
    records: list[Record],
    exclusions: dict[str, int],
    limits: Mapping[str, float | None],
) -> dict[str, Any]:
    """Report configured limits, final group counts/shares, and exclusions.

    Summary and lineage share this block and the selection dimension vocabulary."""
    configured = {
        name: limits.get(name)
        for name, _getter in _SHARE_DIMENSIONS
        if limits.get(name) is not None
    }
    applied: dict[str, dict[str, dict[str, float]]] = {}
    total = len(records)
    for name, getter in _SHARE_DIMENSIONS:
        counts = Counter(map(getter, records))
        applied[name] = {
            str(value) if value is not None else "(none)": {
                "count": count,
                "share": (count / total) if total else 0.0,
            }
            for value, count in sorted(
                counts.items(), key=lambda kv: (kv[0] is None, str(kv[0]))
            )
        }
    return {
        "version": 2,
        "configured": configured,
        "applied": applied,
        "exclusions": dict(sorted(exclusions.items())),
    }
