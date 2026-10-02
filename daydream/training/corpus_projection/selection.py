"""Deterministic tier and output-share caps over projected records."""

import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from typing import Any, NoReturn, cast

Record = dict[str, object]


def count_by(records: Iterable[Any], key: Callable[[Any], str]) -> dict[str, int]:
    return dict(sorted(Counter(map(key, records)).items()))


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


_SHARE_DIMENSIONS: tuple[tuple[str, str, Callable[[Record], Any]], ...] = (
    # One dimension table consumed by both the share-cap selection stage and
    # the report builder, so the three vocabulary uses (configured keys,
    # applied keys, exclusion-key prefixes) can never drift apart: every
    # consumer spells the repository dimension ``repo``, matching the
    # ``max_repo_share`` flag.
    ("stack", "max_stack_share", lambda r: r.get("stack")),
    (
        "repo",
        "max_repo_share",
        lambda r: cast(dict[str, Any], r.get("lineage") or {}).get("repo_slug"),
    ),
    (
        "profile",
        "max_profile_share",
        lambda r: cast(dict[str, Any], r.get("profile") or {}).get("profile_name"),
    ),
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
    flag: str,
    limit: float,
    dimension: str,
    population: int | None = None,
    value: Any | None = None,
    values: list[Any] | None = None,
) -> NoReturn:
    """Refuse a cap that would empty a nonempty population, naming the dimension."""
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
    records: list[Record],
    *,
    max_stack_share: float | None,
    max_repo_share: float | None,
    max_profile_share: float | None,
) -> tuple[list[Record], dict[str, int]]:
    """Keep lowest record ids until all configured output shares satisfy their caps.

    Apply stack, repo, then profile, repeating the sequence to a fixed point:
    correlated dimensions can make an earlier cap exceed its limit again.
    None values form their own (none) bucket. Count each exclusion by dimension
    and value. An initially empty corpus stays empty, but trimming a nonempty
    population to zero raises ValueError. Input order never affects selection."""
    limits = {
        "stack": max_stack_share,
        "repo": max_repo_share,
        "profile": max_profile_share,
    }
    dimensions: list[tuple[str, str, Callable[[Record], Any], float]] = []
    for name, flag, getter in _SHARE_DIMENSIONS:
        limit = limits[name]
        if limit is not None:
            dimensions.append((name, flag, getter, limit))
    exclusions: dict[str, int] = {}
    population = sorted(records, key=lambda r: str(r["record_id"]))
    if not population:
        # Empty population (no decisive findings, or tier caps trimmed
        # everything): configured caps exclude nothing — a previously
        # completing zero-record build stays completing.
        return population, exclusions
    # Fail-closed degeneracy pre-check against the original input population:
    # a cap that cannot keep a single record of an over-share group covering
    # the whole input would collapse it to zero — refuse before doing any work.
    # (Populations reduced by earlier dimensions or by a fixed-point re-pass
    # are handled by the fail-closed sole-value check in the trim loop below,
    # so this check is invariant and runs once per configured dimension on the
    # entry population.)
    entry_total = len(population)
    for dimension, flag, getter, limit in dimensions:
        entry_counts = Counter(map(getter, population))
        for value, count in entry_counts.items():
            if count / entry_total > limit and count == entry_total:
                if _max_keep_count(entry_total, limit) == 0:
                    _raise_share_caps_fail_closed(
                        flag=flag,
                        limit=limit,
                        dimension=dimension,
                        population=entry_total,
                        value=value,
                    )
    previous_size = -1
    while len(population) != previous_size:
        previous_size = len(population)
        for dimension, flag, getter, limit in dimensions:
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
                            # Sole remaining value after earlier reductions or
                            # exclusions: trimming it further would empty the
                            # population entirely — the same terminal state as
                            # the entry degeneracy pre-check above, so fail
                            # closed (never emit a lone value above its cap).
                            _raise_share_caps_fail_closed(
                                flag=flag,
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
                        flag=flag,
                        limit=limit,
                        dimension=dimension,
                        values=list(over),
                    )
                population = kept
    return population, exclusions


def _share_caps_report(
    records: list[Record],
    exclusions: dict[str, int],
    *,
    max_stack_share: float | None,
    max_repo_share: float | None,
    max_profile_share: float | None,
) -> dict[str, Any]:
    """Report configured limits, final group counts/shares, and exclusions.

    Summary and lineage share this block and the selection dimension vocabulary."""
    limits = {
        "stack": max_stack_share,
        "repo": max_repo_share,
        "profile": max_profile_share,
    }
    configured = {
        name: limits[name]
        for name, _flag, _getter in _SHARE_DIMENSIONS
        if limits[name] is not None
    }
    applied: dict[str, dict[str, dict[str, float]]] = {}
    total = len(records)
    for name, _flag, getter in _SHARE_DIMENSIONS:
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
        "version": 1,
        "configured": configured,
        "applied": applied,
        "exclusions": dict(sorted(exclusions.items())),
    }
