"""Deterministic reward-calibration tooling (issue #999, M2 validation matrix).

``run_calibration`` validates a corpus-v2 bundle fail-closed before any
statistics are computed: every gate raises :class:`CalibrationError` naming the
offending record/field, and no artifact is ever written unless all gates pass.

Split membership is re-derived read-only from ``lineage.json`` (salt +
holdout/val rates) via a deterministic hash of the record id, mirroring the
corpus-v2 projector contract; the derived train/val/holdout sets must stay
pairwise disjoint (a duplicated record id means the corpus was hand-edited).

Reward weights, thresholds, and grid ranges come exclusively from
``CalibrationConfig.candidates`` — this module never imports
``reward.DEFAULT_WEIGHTS`` nor mutates any reward default (M10).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from daydream.training.exclusion import load_exclusion_list
from daydream.training.labeler_versions import (
    LABELER_POLICY_VERSION,
    REPLY_CLASSIFIER_VERSION,
    RUBRIC_SCHEMA_VERSION,
)
from daydream.training.reward import REWARD_VERSION

__all__ = ["CalibrationConfig", "CalibrationError", "assign_split", "run_calibration"]

#: Artifact schema version for calibration output (emitted by a later milestone).
ARTIFACT_SCHEMA_VERSION = "calibration-artifact-v1"

_RECORD_SCHEMA_VERSION = "2"
_LINEAGE_SCHEMA_VERSION = "corpus-v2"
_ALLOWED_LICENSE_DECISIONS = frozenset({"allow", "deny-recorded"})

_REQUIRED_RECORD_FIELDS = ("schema_version", "record_id", "session_id", "repo_slug", "reward_version", "lineage")
_REQUIRED_LINEAGE_FIELDS = ("split", "as_of", "valid_at", "license_decision")
# Version stamps (labeler_policy_version, reply_classifier_version,
# rubric_schema_version, reward_version) are validated by the stamp gate so an
# absent stamp fails as "unrecognized label version stamp", not a missing field.
_REQUIRED_BUNDLE_FIELDS = ("schema_version", "salt", "holdout_rate", "val_rate", "as_of", "valid_at")

# Test seam only: corruption variants injected by tests/test_calibration.py.
_CORRUPTION_FLAGS = frozenset(
    {
        "schema_version",
        "posterior",
        "label-version",
        "c5-repo",
        "license",
        "digest",
        "split-overlap",
        "drop-session",
    }
)


class CalibrationError(ValueError):
    """A calibration input failed a fail-closed gate. Message names the culprit."""


@dataclass(frozen=True)
class CalibrationConfig:
    """Inputs and knobs for one calibration run.

    ``corruptions`` is a **test seam** (never CLI surface): named mutations
    applied while loading the corpus so the gate matrix can be exercised
    against otherwise-identical fixtures.
    """

    corpus_dir: Path
    gold_labels: Path
    breakdowns: Path
    out_dir: Path
    run_id: str
    seed: int
    candidates: dict[str, Any]
    stage0_scores: Path | None = None
    model_digest: str | None = None
    grid_points: int = 9
    bootstrap_resamples: int = 1000
    corruptions: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        unknown = self.corruptions - _CORRUPTION_FLAGS
        if unknown:
            raise CalibrationError(f"unknown corruption flags: {sorted(unknown)}")


def assign_split(record_id: str, *, holdout_rate: float, val_rate: float, salt: str) -> str:
    """Deterministically re-derive a record's split from its id (read-only).

    Mirrors the corpus-v2 split contract: a seeded hash of ``salt:record_id``
    selects holdout / val / train in fixed rate order.
    """
    digest = hashlib.sha256(f"{salt}:{record_id}".encode("utf-8")).digest()
    u = int.from_bytes(digest[:8], "big") / float(1 << 64)
    if u < holdout_rate:
        return "holdout"
    if u < holdout_rate + val_rate:
        return "val"
    return "train"


def _gate(condition: bool, message: str) -> None:
    """Fail-closed gate: raise ``CalibrationError`` with a stable message."""
    if not condition:
        raise CalibrationError(message)


def _parse_iso(value: str, field_name: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise CalibrationError(f"{field_name} {value!r} is not ISO-8601 parseable") from exc


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise CalibrationError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
        _gate(isinstance(record, dict), f"{path}:{lineno}: record is not a JSON object")
        records.append(record)
    return records


def _apply_corruptions(records: list[dict[str, Any]], flags: frozenset[str]) -> None:
    """Test seam: mutate loaded records per corruption flag."""
    if not flags:
        return
    first = records[0]
    if "schema_version" in flags:
        first["schema_version"] = "1"
    if "posterior" in flags:
        first["lineage"]["valid_at"] = "2026-06-01T00:00:00+00:00"
    if "label-version" in flags:
        del first["lineage"]["labeler_policy_version"]
    if "c5-repo" in flags:
        first["repo_slug"] = "getsentry/sentry"
    if "license" in flags:
        first["lineage"]["license_decision"] = "unknown"
    if "split-overlap" in flags:
        records.append(json.loads(json.dumps(records[-1])))
    if "drop-session" in flags:
        del first["session_id"]


def _check_digests(corpus_dir: Path) -> None:
    sums_path = corpus_dir / "SHA256SUMS"
    _gate(sums_path.exists(), f"missing SHA256SUMS manifest at {sums_path}")
    for entry in sums_path.read_text().splitlines():
        if not entry.strip():
            continue
        expected, _, name = entry.partition("  ")
        target = corpus_dir / name.strip()
        _gate(target.exists(), f"SHA256SUMS names missing file {target}")
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        _gate(actual == expected.strip(), f"digest mismatch for {target}")


def _check_record_schema(record: dict[str, Any], index: int) -> str:
    where = f"record {record.get('record_id', f'#{index}')}"
    missing = [f for f in _REQUIRED_RECORD_FIELDS if f not in record]
    _gate(not missing, f"{where}: missing required field(s) {missing}")
    rid = str(record["record_id"])
    lineage = record["lineage"]
    _gate(isinstance(lineage, dict), f"record {rid}: lineage is not an object")
    missing_lin = [f for f in _REQUIRED_LINEAGE_FIELDS if f not in lineage]
    _gate(not missing_lin, f"record {rid}: lineage missing required field(s) {missing_lin}")
    _gate(
        record["schema_version"] == _RECORD_SCHEMA_VERSION,
        f'record {rid}: schema_version {record["schema_version"]!r} is not "{_RECORD_SCHEMA_VERSION}"',
    )
    _parse_iso(str(lineage["as_of"]), f"record {rid}: lineage.as_of")
    valid_at = _parse_iso(str(lineage["valid_at"]), f"record {rid}: lineage.valid_at")
    as_of = _parse_iso(str(lineage["as_of"]), f"record {rid}: lineage.as_of")
    _gate(
        valid_at <= as_of,
        f"record {rid}: valid_at {lineage['valid_at']} is posterior to as_of {lineage['as_of']}",
    )
    return rid


def _check_version_stamps(record: dict[str, Any], rid: str) -> None:
    lineage = record["lineage"]
    stamps = {
        "labeler_policy_version": LABELER_POLICY_VERSION,
        "reply_classifier_version": REPLY_CLASSIFIER_VERSION,
        "rubric_schema_version": RUBRIC_SCHEMA_VERSION,
    }
    for field_name, recognized in stamps.items():
        _gate(
            lineage.get(field_name) == recognized,
            f"record {rid}: unrecognized label version stamp {field_name}={lineage.get(field_name)!r}"
            f" (expected {recognized!r})",
        )
    _gate(
        record["reward_version"] == REWARD_VERSION,
        f"record {rid}: unrecognized label version stamp reward_version={record['reward_version']!r}"
        f" (expected {REWARD_VERSION!r})",
    )


def _check_splits(
    records: list[dict[str, Any]], salt: str, holdout_rate: float, val_rate: float
) -> dict[str, set[str]]:
    derived: dict[str, set[str]] = {"train": set(), "val": set(), "holdout": set()}
    seen: dict[str, str] = {}
    for record in records:
        rid = str(record["record_id"])
        split = assign_split(rid, holdout_rate=holdout_rate, val_rate=val_rate, salt=salt)
        derived[split].add(rid)
        prior = seen.get(rid)
        _gate(
            prior is None,
            f"split membership overlaps: record_id {rid} appears in multiple records"
            + (f" (first in {prior}, then in {split})" if prior else ""),
        )
        seen[rid] = split
    return derived


# ---------------------------------------------------------------------------
# Deterministic statistics core (M3) — stdlib only, every float rounded to 4dp.
# ---------------------------------------------------------------------------


def _round4(value: Any) -> Any:
    """Recursively round every float to 4 decimal places for byte-stable output."""
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {k: _round4(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_round4(v) for v in value]
    return value


def _load_inputs(
    config: CalibrationConfig, record_ids: list[str]
) -> tuple[dict[str, bool], dict[str, dict[str, float]]]:
    """Join gold labels and intrinsic breakdowns onto the corpus by record_id."""
    gold_raw = json.loads(config.gold_labels.read_text())
    _gate(isinstance(gold_raw, dict), f"{config.gold_labels}: gold labels must be an object keyed by record_id")
    gold: dict[str, bool] = {}
    for rid, value in gold_raw.items():
        _gate(rid in set(record_ids), f"{config.gold_labels}: gold label for unknown record_id {rid!r}")
        _gate(
            isinstance(value, dict) and isinstance(value.get("accepted"), bool),
            f'{config.gold_labels}: gold label for {rid!r} must be {{"accepted": bool}}',
        )
        gold[rid] = value["accepted"]
    breakdown_raw = json.loads(config.breakdowns.read_text())
    _gate(isinstance(breakdown_raw, dict), f"{config.breakdowns}: breakdowns must be an object keyed by record_id")
    breakdowns: dict[str, dict[str, float]] = {}
    for rid, axes in breakdown_raw.items():
        _gate(rid in set(record_ids), f"{config.breakdowns}: breakdown for unknown record_id {rid!r}")
        _gate(isinstance(axes, dict), f"{config.breakdowns}: breakdown for {rid!r} must be an object")
        for axis, value in axes.items():
            _gate(isinstance(value, (int, float)) and not isinstance(value, bool),
                  f"{config.breakdowns}: axis {axis!r} for {rid!r} must be numeric")
        breakdowns[rid] = {k: float(v) for k, v in axes.items()}
    for rid in record_ids:
        _gate(rid in gold, f"{config.gold_labels}: missing gold label for corpus record {rid!r}")
        _gate(rid in breakdowns, f"{config.breakdowns}: missing breakdown for corpus record {rid!r}")
    return gold, breakdowns


def _midrank_auc(scores: list[float], labels: list[bool]) -> float:
    """AUC via midrank Mann-Whitney U; 0.5 when one class is absent."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return 0.5
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        mid = (i + j) / 2 + 1  # 1-based midrank
        for k in range(i, j + 1):
            ranks[order[k]] = mid
        i = j + 1
    rank_sum = sum(r for r, lab in zip(ranks, labels) if lab)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _bootstrap_auc_ci(
    scores: list[float], labels: list[bool], seed: int, resamples: int
) -> tuple[float, float]:
    """Seeded percentile bootstrap CI over paired (score, label) resamples."""
    rng = random.Random(seed)
    n = len(scores)
    aucs = []
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        aucs.append(_midrank_auc([scores[i] for i in idx], [labels[i] for i in idx]))
    aucs.sort()
    lo = aucs[min(int(0.025 * resamples), resamples - 1)]
    hi = aucs[min(int(0.975 * resamples), resamples - 1)]
    return lo, hi


def _point_biserial(values: list[float], labels: list[bool]) -> float:
    """Pearson correlation between a numeric axis and the binary label; 0.0 if degenerate."""
    binary = [1.0 if lab else 0.0 for lab in labels]
    if len(set(values)) < 2 or len(set(binary)) < 2:
        return 0.0
    mx = statistics.fmean(values)
    mb = statistics.fmean(binary)
    num = sum((v - mx) * (b - mb) for v, b in zip(values, binary))
    den = math.sqrt(sum((v - mx) ** 2 for v in values) * sum((b - mb) ** 2 for b in binary))
    if den == 0:
        return 0.0
    return num / den


def _grid(axis: str, values: list[float] | None, grid_points: int) -> list[float]:
    """Resolve a candidate axis's configured range to ``grid_points`` grid values.

    A candidate axis with no supplied range is an error naming the axis — there
    are no built-in reward defaults to fall back on.
    """
    if not values:
        raise CalibrationError(
            f"candidate axis {axis!r} has no supplied grid range; calibration never invents reward defaults"
        )
    if len(values) == 1 or grid_points <= 1:
        return sorted(round(float(v), 4) for v in values)
    lo, hi = min(values), max(values)
    step = (hi - lo) / (grid_points - 1)
    return sorted({round(lo + i * step, 4) for i in range(grid_points)})


def _compute_metrics(
    records: list[dict[str, Any]],
    gold: dict[str, bool],
    breakdowns: dict[str, dict[str, float]],
    config: CalibrationConfig,
) -> dict[str, Any]:
    record_ids = [str(r["record_id"]) for r in records]
    labels = [gold[rid] for rid in record_ids]

    lengths = [len(json.dumps(r, sort_keys=True)) for r in records]
    if len(lengths) >= 4:
        q1, q3 = statistics.quantiles(lengths, n=4, method="inclusive")[::2]
    else:
        q1, q3 = min(lengths), max(lengths)
    length_distribution = {
        "median": statistics.median(lengths),
        "mean": statistics.fmean(lengths),
        "iqr": q3 - q1,
        "min": float(min(lengths)),
        "max": float(max(lengths)),
    }

    class_balance = {
        "accepted": sum(1 for lab in labels if lab),
        "rejected": sum(1 for lab in labels if not lab),
    }

    axes = sorted({axis for axes_map in breakdowns.values() for axis in axes_map})
    per_axis_correlations = {
        f"{axis}_axis": _point_biserial([breakdowns[rid][axis] for rid in record_ids], labels)
        for axis in axes
    }

    auc_bootstrap_ci: dict[str, list[float]] = {}
    for axis, values in config.candidates.items():
        _gate(
            all(axis in breakdowns[rid] for rid in record_ids),
            f"candidate axis {axis!r} is absent from breakdowns for at least one corpus record",
        )
        scores = [breakdowns[rid][axis] for rid in record_ids]
        for point in _grid(axis, values, config.grid_points):
            lo, hi = _bootstrap_auc_ci(scores, labels, config.seed, config.bootstrap_resamples)
            auc_bootstrap_ci[f"{axis}={point}"] = [lo, hi]

    return {
        "length_distribution": length_distribution,
        "class_balance": class_balance,
        "per_axis_correlations": per_axis_correlations,
        "auc_bootstrap_ci": auc_bootstrap_ci,
    }


def _write_artifact(config: CalibrationConfig, record_count: int, metrics: dict[str, Any]) -> None:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "run_id": config.run_id,
        "seed": config.seed,
        "grid_points": config.grid_points,
        "bootstrap_resamples": config.bootstrap_resamples,
        "record_count": record_count,
        "metrics": metrics,
    }
    payload = json.dumps(_round4(artifact), sort_keys=True, indent=2) + "\n"
    (config.out_dir / "calibration.json").write_text(payload)


def run_calibration(config: CalibrationConfig) -> dict[str, Any]:
    """Validate the corpus bundle fail-closed; return a summary on success.

    Gates run in order: schema version, SHA256SUMS digests, posterior evidence,
    version stamps, C5 exclusion, license decision, split re-derivation. No
    artifact or partial file is written on any failure.
    """
    corpus_dir = config.corpus_dir
    corpus_path = corpus_dir / "corpus.jsonl"
    lineage_path = corpus_dir / "lineage.json"
    _gate(corpus_path.exists(), f"missing corpus at {corpus_path}")
    _gate(lineage_path.exists(), f"missing lineage at {lineage_path}")

    if "digest" in config.corruptions:
        # Test seam: tamper with the corpus so the pristine manifest mismatches.
        tampered = json.loads(corpus_path.read_text().splitlines()[0])
        tampered["session_id"] = "sess-tampered"
        corpus_path.write_text(json.dumps(tampered, sort_keys=True) + "\n")

    _check_digests(corpus_dir)

    bundle = json.loads(lineage_path.read_text())
    _gate(isinstance(bundle, dict), f"{lineage_path}: lineage is not an object")
    missing_bundle = [f for f in _REQUIRED_BUNDLE_FIELDS if f not in bundle]
    _gate(not missing_bundle, f"{lineage_path}: lineage missing required field(s) {missing_bundle}")
    _gate(
        bundle["schema_version"] == _LINEAGE_SCHEMA_VERSION,
        f"{lineage_path}: schema_version {bundle['schema_version']!r} is not {_LINEAGE_SCHEMA_VERSION!r}",
    )

    records = _load_jsonl(corpus_path)
    _apply_corruptions(records, config.corruptions)
    for index, record in enumerate(records):
        rid = _check_record_schema(record, index)
        _check_version_stamps(record, rid)

    excluded = load_exclusion_list()
    for record in records:
        rid = str(record["record_id"])
        slug = str(record["repo_slug"])
        _gate(slug not in excluded, f"record {rid}: repository {slug} is in the excluded repository list (C5)")
        decision = str(record["lineage"]["license_decision"])
        _gate(
            decision in _ALLOWED_LICENSE_DECISIONS,
            f"record {rid}: license decision {decision!r} is not one of {sorted(_ALLOWED_LICENSE_DECISIONS)}",
        )

    holdout_rate = float(bundle["holdout_rate"])
    val_rate = float(bundle["val_rate"])
    _check_splits(records, str(bundle["salt"]), holdout_rate, val_rate)

    record_ids = [str(r["record_id"]) for r in records]
    gold, breakdowns = _load_inputs(config, record_ids)
    metrics = _compute_metrics(records, gold, breakdowns, config)
    _write_artifact(config, len(records), metrics)

    return {
        "run_id": config.run_id,
        "record_count": len(records),
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "seed": config.seed,
        "metrics": metrics,
    }
