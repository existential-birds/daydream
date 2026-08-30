"""Deterministic reward-calibration tooling (issue #999, M2 validation matrix).

``run_calibration`` validates a calibration bundle (wire format documented in
``docs/calibration.md``) fail-closed before any statistics are computed: every
gate raises :class:`CalibrationError` naming the offending record/field, and no
artifact is ever written unless all gates pass.

Split membership is re-derived read-only from ``lineage.json`` (salt +
holdout/val rates) via a deterministic hash of the record id, mirroring the
corpus-v2 projector contract; each record's stored ``lineage['split']`` must
match its re-derived split, and the derived train/val/holdout sets must stay
pairwise disjoint (a duplicated record id means the corpus was hand-edited).

Reward weights, thresholds, and grid ranges come exclusively from
``CalibrationConfig.candidates`` — this module never imports any reward
weights default nor mutates any reward default (M10).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from importlib import metadata
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

#: Artifact schema version for calibration output.
ARTIFACT_SCHEMA_VERSION = "calibration-artifact-v1"


def _tool_version() -> str:
    """Package version of the running daydream distribution (deterministic per install)."""
    try:
        return metadata.version("daydream")
    except metadata.PackageNotFoundError:  # pragma: no cover - dev checkouts
        return "0.0.0"


TOOL_VERSION = _tool_version()

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
        if self.stage0_scores is not None and self.model_digest is None:
            raise CalibrationError("--model-digest is required when --stage0-scores is given")


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
        expected, sep, name = entry.partition("  ")
        _gate(
            bool(sep and expected.strip() and name.strip()),
            f"SHA256SUMS line {entry!r} is malformed (expected '<sha256>  <name>')",
        )
        target = corpus_dir / name.strip()
        _gate(target.is_file(), f"SHA256SUMS names missing file {target}")
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
    as_of = _parse_iso(str(lineage["as_of"]), f"record {rid}: lineage.as_of")
    valid_at = _parse_iso(str(lineage["valid_at"]), f"record {rid}: lineage.valid_at")
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
        stored = str(record["lineage"]["split"])
        _gate(
            stored == split,
            f"record {rid}: stored split {stored!r} does not match the split "
            f"re-derived from lineage.json ({split!r}) via calibration.assign_split",
        )
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
    known_ids = set(record_ids)
    gold: dict[str, bool] = {}
    for rid, value in gold_raw.items():
        _gate(rid in known_ids, f"{config.gold_labels}: gold label for unknown record_id {rid!r}")
        _gate(
            isinstance(value, dict) and isinstance(value.get("accepted"), bool),
            f'{config.gold_labels}: gold label for {rid!r} must be {{"accepted": bool}}',
        )
        gold[rid] = value["accepted"]
    breakdown_raw = json.loads(config.breakdowns.read_text())
    _gate(isinstance(breakdown_raw, dict), f"{config.breakdowns}: breakdowns must be an object keyed by record_id")
    breakdowns: dict[str, dict[str, float]] = {}
    for rid, axes in breakdown_raw.items():
        _gate(rid in known_ids, f"{config.breakdowns}: breakdown for unknown record_id {rid!r}")
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


def _threshold_decision_metrics(
    scores: list[float], labels: list[bool], threshold: float
) -> dict[str, float]:
    """Precision / recall / specificity of ``accepted iff axis >= threshold``.

    These are the per-candidate statistics: unlike AUC, they vary with the
    cut-off, so the ``--candidate``/``--grid-points`` resolution genuinely
    differentiates the ``axis=point`` rows.
    """
    tp = fp = fn = tn = 0
    for score, label in zip(scores, labels):
        predicted = score >= threshold
        if predicted and label:
            tp += 1
        elif predicted and not label:
            fp += 1
        elif not predicted and label:
            fn += 1
        else:
            tn += 1
    return {
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
    }


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

    # AUC is rank-invariant under any monotone transform, so the bootstrap CI
    # is per-axis; the candidate grid is resolved into per-point decision
    # metrics (precision/recall/specificity at each cut-off) below.
    auc_bootstrap_ci: dict[str, list[float]] = {}
    threshold_metrics: dict[str, dict[str, float]] = {}
    for axis, values in config.candidates.items():
        _gate(
            all(axis in breakdowns[rid] for rid in record_ids),
            f"candidate axis {axis!r} is absent from breakdowns for at least one corpus record",
        )
        scores = [breakdowns[rid][axis] for rid in record_ids]
        lo, hi = _bootstrap_auc_ci(scores, labels, config.seed, config.bootstrap_resamples)
        auc_bootstrap_ci[axis] = [lo, hi]
        for point in _grid(axis, values, config.grid_points):
            threshold_metrics[f"{axis}={point}"] = _threshold_decision_metrics(scores, labels, point)

    return {
        "length_distribution": length_distribution,
        "class_balance": class_balance,
        "per_axis_correlations": per_axis_correlations,
        "auc_bootstrap_ci": auc_bootstrap_ci,
        "threshold_metrics": threshold_metrics,
    }


def _axis_residuals(axis_values: list[float], preference: list[float]) -> list[float]:
    """Least-squares residual of the axis after removing the preference term."""
    mean_a = statistics.fmean(axis_values)
    mean_p = statistics.fmean(preference)
    var_p = sum((p - mean_p) ** 2 for p in preference)
    if var_p == 0:
        return list(axis_values)
    cov = sum((a - mean_a) * (p - mean_p) for a, p in zip(axis_values, preference))
    beta = cov / var_p
    intercept = mean_a - beta * mean_p
    return [a - (intercept + beta * p) for a, p in zip(axis_values, preference)]


def _load_stage0_scores(
    config: CalibrationConfig, record_ids: list[str]
) -> dict[str, float]:
    """Load and fail-closed join stage-0 scores onto the corpus by record_id.

    The score file is a JSON object keyed by record_id, each value
    ``{"score": float, "model_digest": str}``. Every score must match a corpus
    record and every record must have a score — no partial join. When
    ``config.model_digest`` is set, every score must carry the same digest.
    """
    raw = json.loads(config.stage0_scores.read_text())  # type: ignore[union-attr]
    _gate(
        isinstance(raw, dict),
        f"{config.stage0_scores}: stage-0 scores must be an object keyed by record_id",
    )
    known = set(record_ids)
    scores: dict[str, float] = {}
    for rid, value in raw.items():
        _gate(
            rid in known,
            f"{config.stage0_scores}: stage-0 score for record_id {rid!r} has no matching record in the corpus",
        )
        _gate(
            isinstance(value, dict) and isinstance(value.get("score"), (int, float))
            and not isinstance(value.get("score"), bool),
            f'{config.stage0_scores}: stage-0 score for {rid!r} must be {{"score": float, "model_digest": str}}',
        )
        if config.model_digest is not None:
            _gate(
                value.get("model_digest") == config.model_digest,
                f"{config.stage0_scores}: stage-0 score for {rid!r} carries model_digest"
                f" {value.get('model_digest')!r}, expected {config.model_digest!r}",
            )
        scores[rid] = float(value["score"])
    for rid in record_ids:
        _gate(
            rid in scores,
            f"{config.stage0_scores}: missing stage-0 score for corpus record {rid!r}",
        )
    return scores


def _stage0_analysis(
    config: CalibrationConfig,
    record_ids: list[str],
    splits: dict[str, set[str]],
    gold: dict[str, bool],
    breakdowns: dict[str, dict[str, float]],
) -> dict[str, Any]:
    """M4: marginal held-out value of each intrinsic axis beyond the stage-0
    preference score. Absent scores are reported as explicit ``unavailable`` —
    never fabricated from defaults."""
    if config.stage0_scores is None:
        return {"status": "unavailable"}
    scores = _load_stage0_scores(config, record_ids)
    preference = [scores[rid] for rid in record_ids]
    # Evaluate on the holdout split; fall back to the full corpus when the
    # deterministic split leaves holdout empty (tiny synthetic corpora).
    heldout_ids = sorted(splits["holdout"]) or sorted(record_ids)
    heldout_index = [record_ids.index(rid) for rid in heldout_ids]
    axes = sorted({axis for axes_map in breakdowns.values() for axis in axes_map})
    marginal: dict[str, float] = {}
    for axis in axes:
        values = [breakdowns[rid][axis] for rid in record_ids]
        residuals = _axis_residuals(values, preference)
        labels = [gold[record_ids[i]] for i in heldout_index]
        marginal[axis] = _point_biserial([residuals[i] for i in heldout_index], labels)
    return {"status": "ok", "marginal_value_per_axis": marginal}


def _sha256_of(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _split_digest(splits: dict[str, set[str]]) -> str:
    """S1: digest pinning the re-derived split membership over record ids."""
    payload = json.dumps(
        {name: sorted(ids) for name, ids in sorted(splits.items())}, sort_keys=True
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _candidate_grid(config: CalibrationConfig) -> dict[str, list[float]]:
    return {
        axis: _grid(axis, values, config.grid_points)
        for axis, values in sorted(config.candidates.items())
    }


def _render_report(
    artifact: dict[str, Any], record_count: int, class_balance: dict[str, int]
) -> str:
    """Human summary derived from the same in-memory numbers as the artifact."""
    metrics = artifact["metrics"]
    lines = [
        "# Reward calibration report",
        "",
        f"- Run: `{artifact['run_id']}`",
        f"- Artifact schema: `{artifact['schema_version']}` (tool {artifact['tool_version']})",
        f"- Resampling seed: {artifact['resampling_seed']}",
        f"- Records: {record_count} (accepted {class_balance['accepted']}, rejected {class_balance['rejected']})",
        f"- Corpus digest: `{artifact['corpus_digest']}`",
        f"- Split digest: `{artifact['split_digest']}`",
        "",
        "## Candidate settings",
        "",
    ]
    for axis, grid in artifact["candidate_settings"].items():
        lines.append(f"- `{axis}` grid: {json.dumps(grid)}")
    lines += ["", "## Per-axis correlations", ""]
    for name, value in sorted(metrics["per_axis_correlations"].items()):
        lines.append(f"- `{name}`: {value}")
    lines += ["", "## AUC bootstrap CIs (per axis)", ""]
    for name, ci in sorted(metrics["auc_bootstrap_ci"].items()):
        lines.append(f"- `{name}`: [{ci[0]}, {ci[1]}]")
    lines += ["", "## Candidate decision metrics (accepted iff axis >= point)", ""]
    for name, dm in sorted(metrics["threshold_metrics"].items()):
        lines.append(
            f"- `{name}`: precision={dm['precision']}, recall={dm['recall']}, "
            f"specificity={dm['specificity']}"
        )
    lines += ["", "## Stage-0 analysis", ""]
    stage0 = artifact["stage0_analysis"]
    if stage0["status"] == "ok":
        for axis, value in sorted(stage0["marginal_value_per_axis"].items()):
            lines.append(f"- `{axis}` marginal value: {value}")
    else:
        lines.append("- unavailable (no stage-0 score file supplied)")
    if artifact["warnings"]:
        lines += ["", "## Warnings", ""]
        for warning in artifact["warnings"]:
            lines.append(f"- {warning}")
    return "\n".join(lines) + "\n"


def _write_outputs(
    config: CalibrationConfig,
    record_count: int,
    splits: dict[str, set[str]],
    metrics: dict[str, Any],
    stage0_analysis: dict[str, Any],
    input_digests: dict[str, str],
    corpus_digest: str,
    bundle: dict[str, Any],
    version_stamps: dict[str, str],
) -> None:
    """Build the artifact + report from already-computed numbers and write both
    with ``calibration.json`` replaced last: a failure between the two replaces
    leaves the previously recorded artifact — and its run identity — in place,
    so a same-run re-run overwrites both and self-heals the directory. Temp
    files are cleaned up on every path."""
    warnings: list[str] = []
    if stage0_analysis["status"] != "ok":
        warnings.append("stage-0 score file not supplied; marginal analysis unavailable")
    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "run_id": config.run_id,
        "resampling_seed": config.seed,
        "grid_points": config.grid_points,
        "bootstrap_resamples": config.bootstrap_resamples,
        "record_count": record_count,
        "input_digests": input_digests,
        "corpus_digest": corpus_digest,
        "split_digest": _split_digest(splits),
        "lineage": {
            "schema_version": bundle["schema_version"],
            "content_digests": bundle.get("content_digests"),
            "as_of": bundle["as_of"],
            "valid_at": bundle["valid_at"],
            "salt": bundle["salt"],
            "holdout_rate": float(bundle["holdout_rate"]),
            "val_rate": float(bundle["val_rate"]),
        },
        "version_stamps": version_stamps,
        "candidate_settings": _candidate_grid(config),
        "metrics": metrics,
        "stage0_analysis": stage0_analysis,
        "warnings": warnings,
    }
    rounded = _round4(artifact)
    artifact_payload = json.dumps(rounded, sort_keys=True, indent=2) + "\n"
    report_payload = _render_report(rounded, record_count, metrics["class_balance"])

    config.out_dir.mkdir(parents=True, exist_ok=True)
    artifact_tmp = config.out_dir / ".calibration.json.tmp"
    report_tmp = config.out_dir / ".report.md.tmp"
    artifact_tmp.write_text(artifact_payload)
    report_tmp.write_text(report_payload)
    try:
        os.replace(report_tmp, config.out_dir / "report.md")
        os.replace(artifact_tmp, config.out_dir / "calibration.json")
    finally:
        artifact_tmp.unlink(missing_ok=True)
        report_tmp.unlink(missing_ok=True)


def _check_out_dir_collision(config: CalibrationConfig) -> None:
    """M7: refuse to overwrite an artifact from a different run identity.

    Mirrors the fail-loud resume guard in ``lineage.validate_resume`` but
    compares a single recorded ``run_id``. A collision is detected before any
    computation so the prior artifact is never touched; re-running with the
    same run id is a permitted resume/overwrite.
    """
    prior = config.out_dir / "calibration.json"
    if not prior.exists():
        return
    try:
        prior_run_id = json.loads(prior.read_text()).get("run_id")
    except (json.JSONDecodeError, OSError) as exc:
        raise CalibrationError(
            f"output directory {config.out_dir} holds an unreadable calibration.json: {exc}"
        ) from exc
    _gate(
        prior_run_id == config.run_id,
        f"output directory {config.out_dir} holds run identity {prior_run_id!r},"
        f" refusing to write run {config.run_id!r} over it",
    )


def run_calibration(config: CalibrationConfig) -> dict[str, Any]:
    """Validate the corpus bundle fail-closed; return a summary on success.

    Gates run in order: schema version, SHA256SUMS digests, posterior evidence,
    version stamps, C5 exclusion, license decision, split re-derivation. No
    artifact or partial file is written on any failure.
    """
    _check_out_dir_collision(config)

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
    _gate(bool(records), f"{corpus_path}: corpus contains no records")
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
    splits = _check_splits(records, str(bundle["salt"]), holdout_rate, val_rate)

    input_digests = {
        "corpus.jsonl": _sha256_of(corpus_path),
        "lineage.json": _sha256_of(lineage_path),
        config.gold_labels.name: _sha256_of(config.gold_labels),
        config.breakdowns.name: _sha256_of(config.breakdowns),
    }
    if config.stage0_scores is not None:
        input_digests[config.stage0_scores.name] = _sha256_of(config.stage0_scores)
    corpus_digest = input_digests["corpus.jsonl"]
    version_stamps = {
        "labeler_policy_version": LABELER_POLICY_VERSION,
        "reply_classifier_version": REPLY_CLASSIFIER_VERSION,
        "rubric_schema_version": RUBRIC_SCHEMA_VERSION,
        "reward_version": REWARD_VERSION,
    }

    record_ids = [str(r["record_id"]) for r in records]
    gold, breakdowns = _load_inputs(config, record_ids)
    metrics = _compute_metrics(records, gold, breakdowns, config)
    stage0_analysis = _stage0_analysis(config, record_ids, splits, gold, breakdowns)
    _write_outputs(
        config,
        len(records),
        splits,
        metrics,
        stage0_analysis,
        input_digests,
        corpus_digest,
        bundle,
        version_stamps,
    )

    return {
        "run_id": config.run_id,
        "record_count": len(records),
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "tool_version": TOOL_VERSION,
        "resampling_seed": config.seed,
        "corpus_digest": corpus_digest,
        "split_digest": _split_digest(splits),
        "metrics": metrics,
        "stage0_analysis": stage0_analysis,
    }
