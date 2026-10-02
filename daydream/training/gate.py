"""Validate the learned outcome model against a frozen held-out split.

The digest binds sorted held-out row ids and seed before training; its sidecar
feeds resume and manifest evidence. Gate thresholds must be in (0, 1), and
missing, empty, or single-class evidence refuses evaluation. Ratios are
measured at evaluation time; every verdict carries its evidence digest.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from daydream.json_utils import atomic_write_json
from daydream.training.reward_model import _read_admitted_rows, score_comment

_LABELS = {"accepted": 1.0, "rejected": 0.0}


@dataclass(frozen=True)
class GateConfig:
    """Required separation and calibration thresholds, each strictly between 0 and 1.

    Separation is mean accepted minus mean rejected score. Calibration is
    1 - mean(abs(score - label)); thresholds are supplied by calibration policy.
    """

    min_separation: float = 0.1
    min_calibration: float = 0.5

    def __post_init__(self) -> None:
        for name in ("min_separation", "min_calibration"):
            value = getattr(self, name)
            if not (0.0 < value < 1.0):
                raise ValueError(
                    f"GateConfig.{name} must be in (0, 1) exclusive (got {value!r}); "
                    "degenerate thresholds would silently pass or fail every gate"
                )

    def thresholds(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class FrozenSplit:
    """Admitted train/held-out rows frozen before model training.

    The digest binds held-out ids and seed; fingerprint is its first eight
    characters. digest_path is a sidecar filename relative to the labels directory.
    held_out_fraction records the rate that determined the partition size.
    """

    digest: str
    fingerprint: str
    digest_path: str
    train_rows: list[dict[str, Any]] = field(default_factory=list)
    held_out_rows: list[dict[str, Any]] = field(default_factory=list)
    seed: int = 0
    held_out_fraction: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "fingerprint": self.fingerprint,
            "digest_path": self.digest_path,
            "train_rows": len(self.train_rows),
            "held_out_rows": len(self.held_out_rows),
            "seed": self.seed,
            "held_out_fraction": self.held_out_fraction,
        }


@dataclass(frozen=True)
class GateReport:
    """Gate verdict and measurements bound to split, model, thresholds, and row count.

    accepted_ratio is measured from held-out labels; evidence_digest binds the
    complete evidence payload for reproducibility.
    """

    passed: bool
    separation: float
    calibration: float
    accepted_ratio: float
    evidence_digest: str
    thresholds: dict[str, float]
    held_out_rows: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _split_digest(held_out_ids: list[str], seed: int) -> str:
    """SHA-256 of the sorted held-out row ids plus the seed (content address)."""
    payload = json.dumps({"held_out_ids": sorted(held_out_ids), "seed": seed}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _build_frozen_split(
    labels_path: str | Path,
    *,
    train_rows: list[dict[str, Any]],
    held_out_rows: list[dict[str, Any]],
    seed: int,
    held_out_fraction: float,
) -> FrozenSplit:
    """Write the canonical split sidecar shared by label and projection producers."""
    held_out_ids = [str(r["comment_id"]) for r in held_out_rows]
    digest = _split_digest(held_out_ids, seed)
    sidecar_path = Path(labels_path).parent / (Path(labels_path).name + ".gate-split.json")
    atomic_write_json(
        sidecar_path,
        {
            "digest": digest,
            "seed": seed,
            "held_out_fraction": held_out_fraction,
            "held_out_ids": sorted(held_out_ids),
            "train_ids": sorted(str(r["comment_id"]) for r in train_rows),
        },
        sort_keys=True,
    )
    return FrozenSplit(
        digest=digest,
        fingerprint=digest[:8],
        digest_path=sidecar_path.name,
        train_rows=train_rows,
        held_out_rows=held_out_rows,
        seed=seed,
        held_out_fraction=held_out_fraction,
    )


def freeze_split(
    labels_path: str | Path, *, held_out_fraction: float, seed: int
) -> FrozenSplit:
    """Admit, sort, then deterministically shuffle labels before training.

    The held-out fraction must be in (0, 1). Empty admitted input or an empty
    held-out side raises ValueError; success writes the digest sidecar.
    """
    if not (0.0 < held_out_fraction < 1.0):
        raise ValueError(
            f"held_out_fraction must be in (0, 1) exclusive (got {held_out_fraction!r})"
        )
    labels_path = Path(labels_path)
    rows = _read_admitted_rows(labels_path)
    if not rows:
        raise ValueError(f"labels file {labels_path} contains no admissible gold outcome rows")

    ordered = sorted(rows, key=lambda r: str(r["comment_id"]))
    shuffled = list(ordered)
    random.Random(seed).shuffle(shuffled)
    n_train = max(1, round((1.0 - held_out_fraction) * len(shuffled)))
    train_rows = shuffled[:n_train]
    held_out_rows = shuffled[n_train:]
    if not held_out_rows:
        raise ValueError(
            f"split leaves no held-out rows ({len(shuffled)} admitted rows with "
            f"held_out_fraction={held_out_fraction}); add rows or lower the fraction"
        )

    return _build_frozen_split(
        labels_path,
        train_rows=train_rows,
        held_out_rows=held_out_rows,
        seed=seed,
        held_out_fraction=held_out_fraction,
    )


def _evidence_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def evaluate_gate(model: Any, split: FrozenSplit | None, config: GateConfig) -> GateReport:
    """Measure separation, calibration, and label balance on held-out rows only.

    Missing split/model evidence, empty holdout, invalid labels, or a single class
    raises RuntimeError. Both configured thresholds must pass.
    """
    if split is None:
        raise RuntimeError(
            "gate evidence missing: no frozen split was supplied; freeze the split "
            "with freeze_split() before training and pass it to evaluate_gate() — "
            "the gate refuses closed rather than evaluating without a held-out split"
        )
    if not split.held_out_rows:
        raise RuntimeError(
            f"gate evidence missing: frozen split {split.fingerprint} has an empty "
            "held-out side; nothing to evaluate against"
        )
    score = getattr(model, "state_dict", None)
    if not callable(score):
        raise RuntimeError(
            "gate evidence missing: model does not expose a scorable state (expected an "
            "OutcomeModel); refusing to evaluate against an unknown model"
        )

    labels: list[float] = []
    scores: list[float] = []
    for row in split.held_out_rows:
        label = str(row.get("label"))
        if label not in _LABELS:
            raise RuntimeError(
                f"gate evidence missing: held-out row {row.get('comment_id')!r} has "
                f"non-outcome label {label!r}; the gate requires accepted/rejected labels"
            )
        labels.append(_LABELS[label])
        scores.append(score_comment(model, str(row["text"])))

    accepted = [s for s, y in zip(scores, labels) if y == 1.0]
    rejected = [s for s, y in zip(scores, labels) if y == 0.0]
    if not accepted or not rejected:
        missing = "accepted" if not accepted else "rejected"
        raise RuntimeError(
            f"gate evidence missing: held-out split {split.fingerprint} contains no "
            f"{missing!r} rows; separation and calibration are undefined on a "
            "single-class held-out side — refusing closed"
        )

    separation = sum(accepted) / len(accepted) - sum(rejected) / len(rejected)
    calibration = 1.0 - (sum(abs(s - y) for s, y in zip(scores, labels)) / len(labels))
    accepted_ratio = len(accepted) / len(labels)

    thresholds = config.thresholds()
    passed = separation >= thresholds["min_separation"] and calibration >= thresholds["min_calibration"]
    evidence = _evidence_digest(
        {
            "split_digest": split.digest,
            "model_fingerprint": getattr(model, "model_fingerprint", ""),
            "thresholds": thresholds,
            "held_out_rows": len(labels),
            "separation": separation,
            "calibration": calibration,
            "accepted_ratio": accepted_ratio,
        }
    )
    return GateReport(
        passed=passed,
        separation=separation,
        calibration=calibration,
        accepted_ratio=accepted_ratio,
        evidence_digest=evidence,
        thresholds=thresholds,
        held_out_rows=len(labels),
    )


__all__ = ["FrozenSplit", "GateConfig", "GateReport", "evaluate_gate", "freeze_split"]
