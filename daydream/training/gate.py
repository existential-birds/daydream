"""Validate the learned outcome model against a frozen held-out split.

The digest binds sorted held-out row ids and seed before training; its sidecar
feeds resume and manifest evidence. Gate thresholds must be in (0, 1), and
missing, empty, or single-class evidence refuses evaluation. Ratios are
measured at evaluation time; every verdict carries its evidence digest.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from daydream.json_utils import atomic_write_json
from daydream.training.corpus import _is_admitted_outcome_gold
from daydream.training.reward_model import score_comment

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
    """Admit gold outcomes once, retaining immutable canonical train/held-out rows.

    The digest binds held-out ids and seed; fingerprint is its first eight
    characters. digest_path is a sidecar filename relative to the labels directory.
    held_out_fraction records the rate that determined the partition size.
    """

    digest_path: str
    train_rows: tuple[Mapping[str, Any], ...]
    held_out_rows: tuple[Mapping[str, Any], ...]
    digest: str = field(init=False)
    seed: int = 0
    held_out_fraction: float = 0.0

    def __post_init__(self) -> None:
        def admit(rows: tuple[Mapping[str, Any], ...]) -> tuple[Mapping[str, Any], ...]:
            admitted = []
            for row in rows:
                if not isinstance(row, Mapping):
                    raise ValueError("outcome row must be an object")
                row_id = row.get("comment_id") or row.get("session_id")
                if not isinstance(row_id, str) or not row_id:
                    raise ValueError("outcome row is missing a string 'comment_id'/'session_id'")
                text = row.get("text")
                if not isinstance(text, str):
                    text = row.get("review_output")
                if not isinstance(text, str):
                    raise ValueError(f"row {row_id} is missing a string 'text'/'review_output'")
                label = row.get("label") or row.get("outcome_label")
                if not isinstance(label, str) or label not in _LABELS:
                    raise ValueError(f"row {row_id} has non-gold label {label!r}")
                if not _is_admitted_outcome_gold(
                    label, bool(row.get("has_posterior", True)), row.get("labeler_policy_version"),
                    bool(row.get("decisive_mix", False)), bool(row.get("decisive_only", True)),
                    allowed_labels=frozenset(_LABELS),
                ):
                    raise ValueError(f"row {row_id} is refused by the gold-outcome gate")
                admitted.append(MappingProxyType({"comment_id": row_id, "text": text, "label": label}))
            return tuple(admitted)

        object.__setattr__(self, "train_rows", admit(self.train_rows))
        object.__setattr__(self, "held_out_rows", admit(self.held_out_rows))
        if not self.held_out_rows:
            raise ValueError("frozen split leaves no held-out rows")
        if {row["label"] for row in (*self.train_rows, *self.held_out_rows)} != set(_LABELS):
            raise ValueError("outcome population must contain both classes: accepted and rejected")
        object.__setattr__(self, "digest", _split_digest([row["comment_id"] for row in self.held_out_rows], self.seed))

    @property
    def fingerprint(self) -> str:
        return self.digest[:8]

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
    """Admit the pinned population before writing its canonical split sidecar."""
    sidecar_path = Path(labels_path).parent / (Path(labels_path).name + ".gate-split.json")
    split = FrozenSplit(
        digest_path=sidecar_path.name,
        train_rows=tuple(train_rows),
        held_out_rows=tuple(held_out_rows),
        seed=seed,
        held_out_fraction=held_out_fraction,
    )
    atomic_write_json(
        sidecar_path,
        {
            "digest": split.digest,
            "seed": seed,
            "held_out_fraction": held_out_fraction,
            "held_out_ids": sorted(row["comment_id"] for row in split.held_out_rows),
            "train_ids": sorted(row["comment_id"] for row in split.train_rows),
        },
        sort_keys=True,
    )
    return split


def _evidence_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def evaluate_gate(model: Any, split: FrozenSplit | None, config: GateConfig) -> GateReport:
    """Measure separation, calibration, and label balance on held-out rows only.

    Missing split/model evidence or a single held-out class raises RuntimeError.
    The split owns row admission and nonempty holdout; both thresholds must pass.
    """
    if split is None:
        raise RuntimeError(
            "gate evidence missing: no frozen split was supplied; freeze the split "
            "from the admitted projection before training and pass it to evaluate_gate() — "
            "the gate refuses closed rather than evaluating without a held-out split"
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
        labels.append(_LABELS[row["label"]])
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


__all__ = ["FrozenSplit", "GateConfig", "GateReport", "evaluate_gate"]
