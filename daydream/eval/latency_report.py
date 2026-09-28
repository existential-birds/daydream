"""Per-profile latency/recall comparison and lens-to-shipped attribution (issue #732).

The MH13 report is a *comparison*, not a scoreboard: it reads a small, fixed,
hand-built corpus (:mod:`tests.fixtures.latency_profiles`) and states what each
latency profile cost and found on exactly those inputs. It makes no statistical
claim beyond the corpus, and the command it documents
(``uv run python -m daydream.eval.latency_report --corpus .../manifest.json``)
is the sole consumer of the corpus.

Two attribution rules are deliberately kept apart from
``analyze_findings.per_lens``, which stays *raw pre-merge* attribution:

* ``shipped_by_lens`` counts the ``lens`` field the merge schema *requires* on
  every shipped item (A7), so a wonder-routing change is evaluated against what
  wonder actually contributed to the posted review.
* ``citations`` counts the corroborating ``(Sources: ...)`` prose separately and
  reports its coverage rate; a shipped item with no parseable citation is still
  attributed by ``lens`` and is never silently folded into another lens.

Every percentage is reported over the runs that exist. A profile with no runs
reports zeroed metrics rather than extrapolating from another profile.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from daydream.deep.latency import (
    _CONCURRENCY_TRIGGERS,
    _INTERFACE_TRIGGERS,
    _MIGRATION_TRIGGERS,
    _PERSISTENCE_TRIGGERS,
    _SECURITY_TRIGGERS,
    LATENCY_PROFILES,
)
from daydream.severity import normalize_severity
from daydream.trajectory import RUNS_DIRNAME

_REPO_ROOT = Path(__file__).resolve().parents[2]
"""Repository root, used to resolve a manifest's ``corpus_dir`` from anywhere."""

_LENSES: tuple[str, ...] = ("per-stack", "cross-stack", "structural", "wonder")
"""The ``lens`` vocabulary the merge schema guarantees on shipped items."""

_SOURCES_RE = re.compile(r"\(Sources:[^)]*\)")
"""One compiled matcher for the ``(Sources: ...)`` citation contract."""

_PHASE_TIMING_KEYS: dict[str, tuple[str, ...]] = {
    "wonder": ("alternatives",),
    "arbiter": ("deep", "arbiter"),
}
"""Report phase -> the ``timing.phase_timings`` keys that measure it, in order.

``compute_timing_summary`` keys ``phase_timings`` by the ``DaydreamPhase``
value only, and every arbiter call runs inside ``phase_scope(DaydreamPhase.DEEP,
stage="arbiter")`` -- there is no ``ARBITER`` phase member, so real runs carry
the arbiter wall-clock under ``deep``, never ``arbiter``. The ``deep`` bucket
aggregates all deep-phase brackets (arbiter, suppression, supervision, review,
uncovered sweep), so the report's ``arbiter`` latency is that shared aggregate.
The legacy ``arbiter`` key is kept only as a fallback for hand-authored corpora
that predate the pipeline keying, so the committed fixtures still report their
arbiter bucket instead of silently collapsing to an empty sample.
"""

_HIGH_SEVERITIES = frozenset({"high", "critical"})
"""Severities that count toward high-severity recall (``critical`` is defensive)."""


def attribute_shipped_lens(items: Iterable[Any]) -> dict[str, Any]:
    """Attribute shipped items to their ``lens`` and report citation coverage.

    ``by_lens`` counts the schema-required ``lens`` field; an unknown or missing
    lens counts under no lens key and is reported as ``unattributed`` -- it is
    never silently folded into ``per-stack``. ``citations_present`` counts items
    whose ``rationale`` carries a parseable ``(Sources: ...)`` citation, and
    ``citation_coverage`` is that fraction over every item considered.
    """
    by_lens = dict.fromkeys(_LENSES, 0)
    citations_present = 0
    unattributed = 0
    total = 0
    for item in items:
        if not isinstance(item, Mapping):
            continue
        total += 1
        lens = item.get("lens")
        if isinstance(lens, str) and lens in by_lens:
            by_lens[lens] += 1
        else:
            unattributed += 1
        rationale = item.get("rationale")
        if isinstance(rationale, str) and _SOURCES_RE.search(rationale):
            citations_present += 1
    return {
        "by_lens": by_lens,
        "citations_present": citations_present,
        "citation_coverage": round(citations_present / total, 4) if total else 0.0,
        "unattributed": unattributed,
        "total": total,
    }


def _percentile(samples: Sequence[float], quantile: float) -> float:
    """Nearest-rank percentile (no interpolation) over a sample list.

    The rank is ``ceil(q * n)`` in 1-based terms, so the result is always an
    observed sample: deterministic and independent of interpolation choices.
    """
    if not samples:
        return 0.0
    ordered = sorted(samples)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return round(float(ordered[index]), 4)


def _resolve_corpus_dir(manifest: Mapping[str, Any], corpus_dir: Path | None) -> Path:
    """Resolve the corpus root from an explicit path or the manifest's declaration."""
    if corpus_dir is not None:
        return Path(corpus_dir)
    declared = manifest.get("corpus_dir")
    if isinstance(declared, str):
        return _REPO_ROOT / declared
    return _REPO_ROOT


def _load_object(path: Path) -> dict[str, Any]:
    """Load a JSON object, returning ``{}`` for absent/malformed/non-object files."""
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _load_items(path: Path) -> list[Any]:
    """Load the ``items`` list from a merged-items file, or ``[]`` when unusable."""
    loaded = _load_object(path)
    items = loaded.get("items")
    return items if isinstance(items, list) else []


def _golden_pairs(case: Mapping[str, Any]) -> set[tuple[Any, Any]]:
    """Normalize a case's golden high-severity ``(file, line)`` pairs."""
    pairs: set[tuple[Any, Any]] = set()
    declared = case.get("golden_high_severity")
    if not isinstance(declared, list):
        return pairs
    for entry in declared:
        if (
            isinstance(entry, (list, tuple))
            and len(entry) == 2
            and isinstance(entry[0], str)
            and isinstance(entry[1], int)
            and not isinstance(entry[1], bool)
        ):
            pairs.add((entry[0], entry[1]))
    return pairs


def _is_high_severity(item: Mapping[str, Any]) -> bool:
    normalized = normalize_severity(item.get("severity"))
    return normalized in _HIGH_SEVERITIES


def _line_of(item: Mapping[str, Any]) -> int | None:
    line = item.get("line")
    if isinstance(line, bool) or not isinstance(line, int):
        return None
    return line


def _phase_timings(evaluation: Mapping[str, Any]) -> dict[str, float | None]:
    """Extract the wonder/arbiter wall-clock seconds from an evaluation file."""
    timing = evaluation.get("timing")
    phase_timings = timing.get("phase_timings") if isinstance(timing, Mapping) else None
    if not isinstance(phase_timings, Mapping):
        return {"wonder": None, "arbiter": None}
    result: dict[str, float | None] = {"wonder": None, "arbiter": None}
    for phase, keys in _PHASE_TIMING_KEYS.items():
        for key in keys:
            bucket = phase_timings.get(key)
            seconds = bucket.get("wall_clock_seconds") if isinstance(bucket, Mapping) else None
            if isinstance(seconds, (int, float)) and not isinstance(seconds, bool):
                result[phase] = float(seconds)
                break
    return result


def _contested_outcomes(routing: Mapping[str, Any], items: Sequence[Any]) -> tuple[int, int]:
    """Kept/dropped counts for groups the routing record names as contested.

    A group's ``reason`` names its forcing signal; a group whose reason contains
    ``contested`` carries contested targets. A contested target uid counts as
    ``kept`` when a shipped item names it in ``source_uids`` and as ``dropped``
    when none does. "Contested" is read from the record, never re-derived.
    """
    arbiter = routing.get("arbiter")
    groups = arbiter.get("groups") if isinstance(arbiter, Mapping) else None
    if not isinstance(groups, list):
        return 0, 0
    shipped_uids: set[str] = set()
    for item in items:
        if not isinstance(item, Mapping):
            continue
        source_uids = item.get("source_uids")
        if isinstance(source_uids, list):
            shipped_uids.update(uid for uid in source_uids if isinstance(uid, str))
    kept = 0
    dropped = 0
    for group in groups:
        if not isinstance(group, Mapping):
            continue
        reason = group.get("reason")
        if not isinstance(reason, str) or "contested" not in reason.lower():
            continue
        target_uids = group.get("target_uids")
        if not isinstance(target_uids, list):
            continue
        for uid in target_uids:
            if not isinstance(uid, str):
                continue
            if uid in shipped_uids:
                kept += 1
            else:
                dropped += 1
    return kept, dropped


def _decision(routing: Mapping[str, Any]) -> dict[str, Any]:
    """Summarise one run's routing record for the report's ``decisions`` section."""
    wonder = routing.get("wonder")
    arbiter = routing.get("arbiter")
    groups = arbiter.get("groups") if isinstance(arbiter, Mapping) else None
    return {
        "wonder": dict(wonder) if isinstance(wonder, Mapping) else {},
        "arbiter": {
            "sharded": bool(arbiter.get("sharded")) if isinstance(arbiter, Mapping) else False,
            "reason": arbiter.get("reason") if isinstance(arbiter, Mapping) else None,
            "groups": [
                {
                    "group_id": group.get("group_id"),
                    "effort": group.get("effort"),
                    "reason": group.get("reason"),
                }
                for group in (groups if isinstance(groups, list) else [])
                if isinstance(group, Mapping)
            ],
        },
    }


def build_report(
    manifest: Mapping[str, Any], *, corpus_dir: Path | None = None
) -> dict[str, Any]:
    """Aggregate the committed corpus into the per-profile comparison report.

    ``corpus_dir`` overrides the manifest's ``corpus_dir``; this is how
    :func:`main` resolves run directories relative to the manifest it loaded
    rather than relative to the repository root.
    """
    if not isinstance(manifest, Mapping):
        raise ValueError("corpus manifest must be a JSON object")
    root = _resolve_corpus_dir(manifest, corpus_dir)
    raw_cases = manifest.get("cases")
    cases: list[Mapping[str, Any]] = (
        [c for c in raw_cases if isinstance(c, Mapping)] if isinstance(raw_cases, list) else []
    )

    samples: dict[str, dict[str, list[float]]] = {
        profile: {phase: [] for phase in _PHASE_TIMING_KEYS} for profile in LATENCY_PROFILES
    }
    golden_total: dict[str, int] = dict.fromkeys(LATENCY_PROFILES, 0)
    golden_found: dict[str, int] = dict.fromkeys(LATENCY_PROFILES, 0)
    shipped_total: dict[str, int] = dict.fromkeys(LATENCY_PROFILES, 0)
    false_positives: dict[str, int] = dict.fromkeys(LATENCY_PROFILES, 0)
    contested: dict[str, list[int]] = {profile: [0, 0] for profile in LATENCY_PROFILES}
    shipped_by_lens: dict[str, dict[str, int]] = {
        profile: dict.fromkeys(_LENSES, 0) for profile in LATENCY_PROFILES
    }
    unattributed: dict[str, int] = dict.fromkeys(LATENCY_PROFILES, 0)
    decisions: dict[str, list[dict[str, Any]]] = {profile: [] for profile in LATENCY_PROFILES}
    citations_present = 0
    citations_total = 0
    case_names: list[str] = []

    for case in cases:
        name = case.get("name")
        if not isinstance(name, str):
            continue
        case_names.append(name)
        golden = _golden_pairs(case)
        declared = case.get("profiles")
        profiles: list[Any] = declared if isinstance(declared, list) else list(LATENCY_PROFILES)
        for profile in profiles:
            if profile not in shipped_by_lens:
                continue
            run_dir = root / RUNS_DIRNAME / name / str(profile)
            if not run_dir.is_dir():
                continue
            items = _load_items(run_dir / "merged-items.json")
            attribution = attribute_shipped_lens(items)
            timings = _phase_timings(_load_object(run_dir / "evaluation.json"))
            routing = _load_object(run_dir / "latency-routing.json")
            for phase, seconds in timings.items():
                if seconds is not None:
                    samples[str(profile)][phase].append(seconds)
            matched_here: set[tuple[Any, Any]] = set()
            for item in items:
                if not isinstance(item, Mapping):
                    continue
                key = (item.get("file"), _line_of(item))
                if key not in golden:
                    false_positives[str(profile)] += 1
                elif _is_high_severity(item):
                    matched_here.add(key)
            golden_total[str(profile)] += len(golden)
            golden_found[str(profile)] += len(matched_here)
            shipped_total[str(profile)] += attribution["total"]
            kept, dropped = _contested_outcomes(routing, items)
            contested[str(profile)][0] += kept
            contested[str(profile)][1] += dropped
            for lens, count in attribution["by_lens"].items():
                shipped_by_lens[str(profile)][lens] += count
            unattributed[str(profile)] += attribution["unattributed"]
            citations_present += attribution["citations_present"]
            citations_total += attribution["total"]
            decisions[str(profile)].append(
                {"case": name, **_decision(routing)}
            )

    profiles_report: dict[str, Any] = {}
    for profile in LATENCY_PROFILES:
        runs = len(decisions[profile])
        phase_latency = {
            phase: {"p50": _percentile(samples[profile][phase], 0.5), "p90": _percentile(samples[profile][phase], 0.9)}
            for phase in _PHASE_TIMING_KEYS
        }
        recall = (
            round(golden_found[profile] / golden_total[profile], 4)
            if golden_total[profile]
            else 0.0
        )
        false_positive_rate = (
            round(false_positives[profile] / shipped_total[profile], 4)
            if shipped_total[profile]
            else 0.0
        )
        profiles_report[profile] = {
            "runs": runs,
            "phase_latency_seconds": phase_latency,
            "high_severity_recall": recall,
            "false_positive_rate": false_positive_rate,
            "contested": {"kept": contested[profile][0], "dropped": contested[profile][1]},
            "shipped_by_lens": shipped_by_lens[profile],
            "unattributed_shipped": unattributed[profile],
            "decisions": decisions[profile],
        }

    return {
        "corpus": manifest.get("corpus", "latency-profiles"),
        "profiles": profiles_report,
        "cases": case_names,
        "citations": {
            "present": citations_present,
            "total": citations_total,
            "coverage": round(citations_present / citations_total, 4) if citations_total else 0.0,
        },
        "calibration": {
            "surface_signals": {
                "security": list(_SECURITY_TRIGGERS),
                "concurrency": list(_CONCURRENCY_TRIGGERS),
                "persistence": list(_PERSISTENCE_TRIGGERS),
                "interface": list(_INTERFACE_TRIGGERS),
                "migration": list(_MIGRATION_TRIGGERS),
            },
            "note": (
                "These are the committed default trigger lists from "
                "daydream.deep.latency; they are the calibration surface this "
                "report exists to revisit, not a claim of statistical support."
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Print the per-profile report for ``--corpus`` as JSON; non-zero on error."""
    parser = argparse.ArgumentParser(
        prog="daydream.eval.latency_report",
        description="Emit the per-profile latency/recall/lens report for a committed corpus.",
    )
    parser.add_argument("--corpus", required=True, help="Path to the corpus manifest.json.")
    args = parser.parse_args(argv)
    corpus_path = Path(args.corpus)
    try:
        manifest = json.loads(corpus_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"latency_report: cannot read corpus manifest {corpus_path}: {exc}", file=sys.stderr)
        return 2
    if not isinstance(manifest, dict):
        print(f"latency_report: corpus manifest {corpus_path} is not a JSON object", file=sys.stderr)
        return 2
    report = build_report(manifest, corpus_dir=corpus_path.parent)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
