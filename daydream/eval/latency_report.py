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

A case may additionally declare ``sample_group`` (``cold`` or ``warm``); the
report then emits a ``review_runtime`` block that keeps a cold fix and a warm
reuse loop apart, using the same nearest-rank percentiles (MH14). A case with no
``sample_group`` reports as ``ungrouped`` and is otherwise rendered as before.

A manifest may also declare ``selection_cases``; the report then emits a
``verify_selection`` block comparing today's conservative verifier against the
optimised selective mode over exactly those cases. The block's ``proposed`` arm
carries a *projected* latency derived from the selected-item ratio, never a
second measured arm, so the single-arm corpus is labelled rather than passed off
as a paired measurement. The report still makes no statistical claim beyond the
corpus: it states the observed subset and its coverage.
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

from daydream.deep.adjudication_provenance import load_provenance
from daydream.deep.latency import LATENCY_PROFILES
from daydream.deep.risk_categories import CATEGORY_TRIGGERS
from daydream.deep.verify_selection import SelectionConfig, select_items
from daydream.hunk_index import load_hunk_index
from daydream.json_utils import read_json_object
from daydream.severity import is_high_severity
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
    "verify": ("verify",),
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

The ``verify`` entry reads the archived recommendation-verifier wall-clock for
the ``verify_selection`` comparison; it is not part of the per-profile block.
"""

_PROFILE_PHASE_TIMING_KEYS: tuple[str, ...] = ("wonder", "arbiter")
"""The phases the per-profile ``phase_latency_seconds`` block reports.

Kept apart from :data:`_PHASE_TIMING_KEYS` so adding a verify timing for the
selection block leaves the existing per-profile output byte-identical.
"""


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


def _load_items(path: Path) -> list[Any]:
    """Load the ``items`` list from a merged-items file, or ``[]`` when unusable."""
    loaded = read_json_object(path)
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
        return dict.fromkeys(_PHASE_TIMING_KEYS)
    result: dict[str, float | None] = dict.fromkeys(_PHASE_TIMING_KEYS)
    for phase, keys in _PHASE_TIMING_KEYS.items():
        for key in keys:
            bucket = phase_timings.get(key)
            seconds = bucket.get("wall_clock_seconds") if isinstance(bucket, Mapping) else None
            if isinstance(seconds, (int, float)) and not isinstance(seconds, bool):
                result[phase] = float(seconds)
                break
    return result

_UNGROUPED = "ungrouped"
"""Sample-group key for a case that declares no ``sample_group``."""

_RUNTIME_TARGET = "Target: 5-15 min for a small follow-up fix"
"""MH14's stated target, carried verbatim beside the cold/warm measurements."""


def _seconds(value: Any) -> float | None:
    """Coerce a corpus sample to float seconds, rejecting bools and non-numbers."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _seconds_text(value: float) -> str:
    """Render a measured second count without a spurious trailing ``.0``."""
    return f"{value:g}"


def _case_sample_series(
    case: Mapping[str, Any], root: Path
) -> tuple[dict[str, list[float]], list[dict[str, str]]]:
    """Named elapsed-second series for one corpus case, plus any skipped files.

    A case may declare ``profiles`` as an inline mapping of series name to the
    seconds observed for that series, or as the list of latency profiles whose
    ``evaluation.json`` run directories hold the measurements. Inline samples are
    the unit-test path; run directories are the committed-corpus path. A run file
    that cannot be read is reported with its path and reason instead of being
    silently dropped.
    """
    declared = case.get("profiles")
    series: dict[str, list[float]] = {}
    skipped: list[dict[str, str]] = []
    if isinstance(declared, Mapping):
        for name, values in declared.items():
            if not isinstance(values, list):
                continue
            collected = [seconds for value in values if (seconds := _seconds(value)) is not None]
            if collected:
                series[str(name)] = collected
        return series, skipped
    name = case.get("name")
    if not isinstance(name, str) or not isinstance(declared, list):
        return series, skipped
    for profile in declared:
        evaluation_path = root / RUNS_DIRNAME / name / str(profile) / "evaluation.json"
        if not evaluation_path.is_file():
            skipped.append({"case": name, "path": str(evaluation_path), "error": "evaluation.json is missing"})
            continue
        timings = _phase_timings(read_json_object(evaluation_path))
        for phase in _PROFILE_PHASE_TIMING_KEYS:
            seconds = timings.get(phase)
            if seconds is not None:
                series.setdefault(f"{profile}.{phase}", []).append(seconds)
    return series, skipped


def _runtime_report(
    manifest: Mapping[str, Any], cases: Sequence[Mapping[str, Any]], root: Path
) -> dict[str, Any]:
    """Group the corpus's measured samples by ``sample_group`` (MH14).

    A case without ``sample_group`` is ungrouped; a group is only reported when
    at least one of its cases carried samples. Percentiles reuse the module's
    nearest-rank helper, so the cold/warm report and the per-profile report agree
    by construction. The header names the corpus, the observed sample size per
    group, and the stated target, so the report stands without the docs.
    """
    groups: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    skipped: list[dict[str, str]] = []
    for case in cases:
        group = case.get("sample_group")
        group_name = group if isinstance(group, str) and group else _UNGROUPED
        series, case_skipped = _case_sample_series(case, root)
        skipped.extend(case_skipped)
        if not series:
            continue
        if group_name not in groups:
            groups[group_name] = {"n": 0, "profiles": {}}
            order.append(group_name)
        bucket = groups[group_name]
        bucket["n"] += 1
        for name, values in series.items():
            bucket["profiles"].setdefault(name, []).extend(values)

    corpus = manifest.get("corpus", "latency-profiles")
    rendered: dict[str, Any] = {}
    lines = [f"Review runtime corpus: {corpus}", _RUNTIME_TARGET]
    for group_name in order:
        bucket = groups[group_name]
        profiles = {
            name: {"p50": _percentile(values, 0.5), "p90": _percentile(values, 0.9)}
            for name, values in bucket["profiles"].items()
        }
        rendered[group_name] = {"n": bucket["n"], "profiles": profiles}
        parts = [
            f"{name} p50={_seconds_text(stats['p50'])} p90={_seconds_text(stats['p90'])}"
            for name, stats in profiles.items()
        ]
        lines.append(f"{group_name}: n={bucket['n']}" + ("; " + "; ".join(parts) if parts else ""))
    counted = sum(bucket["n"] for bucket in groups.values())
    if skipped:
        lines.append(f"skipped: {len(skipped)} unreadable case file(s)")
    return {
        "corpus": corpus,
        "target": _RUNTIME_TARGET,
        "groups": rendered,
        "considered": counted + len(skipped),
        "skipped": skipped,
        "header": "\n".join(lines),
    }


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


_VERIFY_CONTRADICTIONS: frozenset[str] = frozenset({"contradicts", "uncertain"})
"""Archived verifier verdicts that mean the verifier would have contradicted the fix."""

_FAILED_FIX_VERDICTS: frozenset[str] = frozenset({"unresolved", "wrong_target", "regressed"})
"""``fix-outcomes.json`` verdicts that count as a failed fix for the report."""

_PROJECTION_NOTE = (
    "Single-arm corpus: the proposed mode's latency scales the archived measured verify "
    "wall-clock samples by the selected-item ratio (proposed items / current items); it is a "
    "labelled projection, not a second measurement."
)
"""The one-line methodology carried beside every projected latency figure (Assumption 6)."""


def _read_text(path: Path) -> str:
    """Read a corpus text file, or ``""`` when it is missing or unreadable (Pattern B)."""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _archived_verdicts(run_dir: Path) -> dict[int, str]:
    """Map archived verifier ``issue_id`` -> verdict string, ignoring malformed entries."""
    payload = read_json_object(run_dir / "recommendation-verdicts.json")
    verdicts = payload.get("verdicts")
    loaded: dict[int, str] = {}
    if isinstance(verdicts, list):
        for verdict in verdicts:
            if not isinstance(verdict, Mapping):
                continue
            issue_id = verdict.get("issue_id")
            value = verdict.get("verdict")
            if isinstance(issue_id, int) and not isinstance(issue_id, bool) and isinstance(value, str):
                loaded[issue_id] = value
    return loaded


def _fix_failure_count(run_dir: Path) -> int:
    """Count the case's reverted or failed fix outcomes from ``fix-outcomes.json``."""
    payload = read_json_object(run_dir / "fix-outcomes.json")
    outcomes = payload.get("outcomes")
    if not isinstance(outcomes, Mapping):
        return 0
    count = 0
    for outcome in outcomes.values():
        if not isinstance(outcome, Mapping):
            continue
        if outcome.get("reverted") is True or outcome.get("verdict") in _FAILED_FIX_VERDICTS:
            count += 1
    return count


def _selection_case_run_dir(root: Path, case: Mapping[str, Any]) -> Path | None:
    """Resolve one ``selection_cases`` entry's run directory, or ``None`` when incomplete."""
    name = case.get("name")
    profile = case.get("profile")
    if not isinstance(name, str) or not isinstance(profile, str):
        return None
    return root / RUNS_DIRNAME / name / profile


def _empty_mode_counts() -> dict[str, int]:
    return {
        "items": 0,
        "skipped": 0,
        "selected_anchors": 0,
        "anchor_total": 0,
        "contradictory_fixes": 0,
        "reverted_and_failed_fixes": 0,
    }


def _selection_mode_report(
    counts: Mapping[str, int], samples: Sequence[float], *, projected: bool
) -> dict[str, Any]:
    """Render one selection mode's counters plus the nearest-rank latency figures."""
    anchor_total = counts["anchor_total"]
    recall = round(counts["selected_anchors"] / anchor_total, 4) if anchor_total else 0.0
    items = counts["items"]
    report: dict[str, Any] = {
        "items": items,
        "calls": 1 if items > 0 else 0,
        "skipped": counts["skipped"],
        "high_severity_recall": recall,
        "contradictory_fixes": counts["contradictory_fixes"],
        "reverted_and_failed_fixes": counts["reverted_and_failed_fixes"],
        "latency_seconds": {
            "p50": _percentile(samples, 0.5),
            "p90": _percentile(samples, 0.9),
        },
    }
    report["latency_seconds_projected"] = projected
    if projected:
        report["latency_seconds_methodology"] = _PROJECTION_NOTE
    return report


def _selection_block(cases: Sequence[Mapping[str, Any]], root: Path) -> dict[str, Any]:
    """Build the MH16 ``verify_selection`` comparison over the declared cases.

    Two modes run the same predicate over the same archived cases: ``current``
    is today's conservative ``verify_all`` behaviour, ``proposed`` the selective
    mode. ``current`` reports the archived measured verify latency; ``proposed``
    scales those same measured samples by the selected-item ratio and labels the
    result projected, because the corpus holds one arm only. ``flip_allowed`` is
    gated on the contradiction counter and the recall anchor alone -- latency is
    reported, never gated.
    """
    current = _empty_mode_counts()
    proposed = _empty_mode_counts()
    case_names: list[str] = []
    samples: list[float] = []
    skipped: list[dict[str, str]] = []

    for case in cases:
        run_dir = _selection_case_run_dir(root, case)
        name = case.get("name")
        if run_dir is None:
            skipped.append({"case": str(name), "path": str(root), "error": "case names no profile"})
            continue
        case_names.append(str(name))
        merged_path = run_dir / "merged-items.json"
        diff_path = run_dir / "diff.patch"
        verdicts_path = run_dir / "recommendation-verdicts.json"
        evaluation_path = run_dir / "evaluation.json"
        required = (merged_path, diff_path, verdicts_path, evaluation_path)
        if not all(path.is_file() for path in required):
            missing = ", ".join(path.name for path in required if not path.is_file())
            skipped.append(
                {"case": str(name), "path": str(run_dir), "error": f"unreadable case artifact(s): {missing}"}
            )
            continue
        items = _load_items(merged_path)
        provenance = load_provenance(run_dir)
        hunk_index = load_hunk_index(run_dir)
        diff_text = _read_text(diff_path)
        verdicts = _archived_verdicts(run_dir)
        golden = _golden_pairs(case)
        for counts, config in (
            (current, SelectionConfig(verify_all=True)),
            (proposed, SelectionConfig(verify_all=False)),
        ):
            decisions = select_items(
                items,
                provenance=provenance,
                hunk_index=hunk_index,
                diff_text=diff_text,
                config=config,
            )
            selected = [decision for decision in decisions if decision.selected]
            rejected = [decision for decision in decisions if not decision.selected]
            counts["items"] += len(selected)
            counts["skipped"] += len(rejected)
            counts["contradictory_fixes"] += sum(
                1
                for decision in rejected
                if decision.item_id is not None and verdicts.get(decision.item_id) in _VERIFY_CONTRADICTIONS
            )
            selected_keys = {
                (item.get("file"), _line_of(item))
                for decision, item in zip(decisions, items)
                if isinstance(item, Mapping) and decision.selected and is_high_severity(item.get("severity"))
            }
            counts["anchor_total"] += len(golden)
            counts["selected_anchors"] += sum(1 for pair in golden if pair in selected_keys)
        failures = _fix_failure_count(run_dir)
        current["reverted_and_failed_fixes"] += failures
        proposed["reverted_and_failed_fixes"] += failures
        seconds = _phase_timings(read_json_object(evaluation_path)).get("verify")
        if isinstance(seconds, (int, float)) and not isinstance(seconds, bool):
            samples.append(float(seconds))

    ratio = proposed["items"] / current["items"] if current["items"] else 0.0
    current_report = _selection_mode_report(current, samples, projected=False)
    proposed_report = _selection_mode_report(proposed, [sample * ratio for sample in samples], projected=True)
    return {
        "cases": case_names,
        "modes": {"current": current_report, "proposed": proposed_report},
        "projection": _PROJECTION_NOTE,
        "flip_allowed": (
            proposed_report["contradictory_fixes"] == 0
            and proposed_report["high_severity_recall"] >= current_report["high_severity_recall"]
        ),
        "skipped": skipped,
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
        profile: {phase: [] for phase in _PROFILE_PHASE_TIMING_KEYS} for profile in LATENCY_PROFILES
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
            timings = _phase_timings(read_json_object(run_dir / "evaluation.json"))
            routing = read_json_object(run_dir / "latency-routing.json")
            for phase in _PROFILE_PHASE_TIMING_KEYS:
                seconds = timings.get(phase)
                if seconds is not None:
                    samples[str(profile)][phase].append(seconds)
            matched_here: set[tuple[Any, Any]] = set()
            for item in items:
                if not isinstance(item, Mapping):
                    continue
                key = (item.get("file"), _line_of(item))
                if key not in golden:
                    false_positives[str(profile)] += 1
                elif is_high_severity(item.get("severity")):
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
            for phase in _PROFILE_PHASE_TIMING_KEYS
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

    report = {
        "corpus": manifest.get("corpus", "latency-profiles"),
        "profiles": profiles_report,
        "review_runtime": _runtime_report(manifest, cases, root),
        "cases": case_names,
        "citations": {
            "present": citations_present,
            "total": citations_total,
            "coverage": round(citations_present / citations_total, 4) if citations_total else 0.0,
        },
        "calibration": {
            "surface_signals": {
                "security": list(CATEGORY_TRIGGERS["security"]),
                "concurrency": list(CATEGORY_TRIGGERS["concurrency"]),
                "persistence": list(CATEGORY_TRIGGERS["persistence"]),
                "interface": list(CATEGORY_TRIGGERS["public-interface"]),
                "migration": list(CATEGORY_TRIGGERS["migration"]),
            },
            "note": (
                "These are the committed default trigger lists from "
                "daydream.deep.risk_categories; they are the calibration surface "
                "this report exists to revisit, not a claim of statistical support."
            ),
        },
    }
    raw_selection = manifest.get("selection_cases")
    if isinstance(raw_selection, list):
        selection_cases = [case for case in raw_selection if isinstance(case, Mapping)]
        report["verify_selection"] = _selection_block(selection_cases, root)
    return report


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
    manifest.setdefault("corpus", corpus_path.name)
    report = build_report(manifest, corpus_dir=corpus_path.parent)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
