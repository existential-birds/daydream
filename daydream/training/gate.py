"""Stage-0 offline gate: frozen split + separation/calibration validation (M3, S2).

This is the *offline validation harness* for the learned outcome reward model
(:mod:`daydream.training.reward_model`), not a scoring function. It mirrors the
version/digest stamping discipline of :mod:`daydream.training.reward`
(``reward.py`` fingerprint discipline): every gate verdict carries an
``evidence_digest`` so the decision is reproducible and auditable.

Contract points:

- **Frozen split (M3/M18)**: the train/held-out partition is frozen *before*
  the reward model trains. It is content-addressed — the digest is the
  SHA-256 of the sorted held-out row ids plus the seed — and the digest is
  written beside the labels file for the resume guard (M18's split digest)
  and the stage manifest (M16). The same labels + seed always reproduce the
  same digest; a different seed yields a different digest.
- **Gate refusal (M4's underlying rule)**: missing split/label evidence raises
  :class:`RuntimeError` naming what is missing. The gate never fails open.
- **Thresholds are documented, never silent**: ``GateConfig`` thresholds are
  supplied by the calibration run (Open Question 1); construction rejects
  values outside ``(0, 1)`` so no degenerate threshold can silently pass.
- **S2**: the accepted ratio is measured from the held-out labels at gate
  time — never read from a stored/stale figure.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from daydream.training.reward_model import _read_admitted_rows, score_comment

_LABELS = {"accepted": 1.0, "rejected": 0.0}


@dataclass(frozen=True)
class GateConfig:
    """Documented gate thresholds (values pinned by the calibration run,
    Open Question 1 — never hard-coded defaults that silently pass).

    Attributes:
        min_separation: Minimum required gap between the mean composite score
            of the accepted population and the rejected population on the
            held-out split.
        min_calibration: Minimum required calibration score on the held-out
            split, where calibration ``= 1 - mean(|score - label|)`` (1.0 is
            perfectly calibrated, 0.0 is maximally miscalibrated).

    Raises:
        ValueError: When any threshold is ≤ 0 or ≥ 1.
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
        return {"min_separation": self.min_separation, "min_calibration": self.min_calibration}


@dataclass(frozen=True)
class FrozenSplit:
    """A frozen train/held-out partition of the gold labels, frozen before
    the reward model trains.

    Attributes:
        digest: Content-addressed SHA-256 of the sorted held-out row ids plus
            the seed (M18's split digest; feeds the stage manifest, M16).
        fingerprint: Short (8-char) form of the digest for display/stamping.
        digest_path: Path (relative to the labels file's directory) of the
            digest sidecar written beside the split for the resume guard.
        train_rows / held_out_rows: The admitted rows in each partition.
        seed: The seed that froze the shuffle.
        held_out_fraction: The fraction that determined the partition size.
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
    """JSON-serializable verdict of the Stage-0 offline gate.

    Attributes:
        passed: Whether both documented thresholds were met on the held-out
            split.
        separation: Mean accepted composite score minus mean rejected
            composite score on the held-out split.
        calibration: ``1 - mean(|score - label|)`` on the held-out split.
        accepted_ratio: Fraction of accepted rows in the held-out split,
            measured at gate time (S2) — never a stored figure.
        evidence_digest: SHA-256 over the full evidence payload (split digest,
            model fingerprint, thresholds, counts, and measurements) so the
            verdict is reproducible and auditable.
        thresholds: The documented thresholds the verdict was judged against.
        held_out_rows: Number of held-out rows evaluated.
    """

    passed: bool
    separation: float
    calibration: float
    accepted_ratio: float
    evidence_digest: str
    thresholds: dict[str, float]
    held_out_rows: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "separation": self.separation,
            "calibration": self.calibration,
            "accepted_ratio": self.accepted_ratio,
            "evidence_digest": self.evidence_digest,
            "thresholds": dict(self.thresholds),
            "held_out_rows": self.held_out_rows,
        }


def _split_digest(held_out_ids: list[str], seed: int) -> str:
    """SHA-256 of the sorted held-out row ids plus the seed (content address)."""
    payload = json.dumps({"held_out_ids": sorted(held_out_ids), "seed": seed}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def freeze_split(
    labels_path: str | Path, *, held_out_fraction: float, seed: int
) -> FrozenSplit:
    """Freeze the train/held-out split of gold accepted/rejected rows.

    Must be called **before** the reward model trains. Rows are admitted via
    the same gold-outcome gate as the reward model (fail closed on refusal),
    deterministically shuffled with ``seed``, and partitioned so the held-out
    side holds ``held_out_fraction`` of the admitted rows (at least one).

    A digest sidecar (``<labels>.gate-split.json``) is written beside the
    labels file recording the split digest for the resume guard (M18) and the
    stage manifest (M16).

    Args:
        labels_path: JSONL labels file (``comment_id``/``text``/``label`` rows).
        held_out_fraction: Fraction of admitted rows reserved for the gate;
            must be in ``(0, 1)`` exclusive.
        seed: Seed freezing the shuffle; part of the digest.

    Returns:
        The frozen :class:`FrozenSplit`.

    Raises:
        ValueError: On an unreadable/refused row or a degenerate fraction.
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

    held_out_ids = [str(r["comment_id"]) for r in held_out_rows]
    digest = _split_digest(held_out_ids, seed)
    digest_path = labels_path.name + ".gate-split.json"
    sidecar = {
        "digest": digest,
        "seed": seed,
        "held_out_fraction": held_out_fraction,
        "held_out_ids": sorted(held_out_ids),
        "train_ids": sorted(str(r["comment_id"]) for r in train_rows),
    }
    sidecar_path = labels_path.parent / digest_path
    tmp = sidecar_path.with_name(sidecar_path.name + ".tmp")
    tmp.write_text(json.dumps(sidecar, indent=2, sort_keys=True))
    os.replace(tmp, sidecar_path)

    return FrozenSplit(
        digest=digest,
        fingerprint=digest[:8],
        digest_path=digest_path,
        train_rows=train_rows,
        held_out_rows=held_out_rows,
        seed=seed,
        held_out_fraction=held_out_fraction,
    )


def _evidence_digest(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def evaluate_gate(model: Any, split: FrozenSplit | None, config: GateConfig) -> GateReport:
    """Evaluate the Stage-0 gate on the held-out split only.

    Computes the separation of composite scores between the accepted and
    rejected held-out populations, a calibration measure, and the accepted
    ratio measured at gate time (S2). ``passed`` is determined by the
    documented :class:`GateConfig` thresholds.

    Args:
        model: A trained :class:`~daydream.training.reward_model.OutcomeModel`.
        split: The frozen split. ``None`` — or a split missing evidence —
            refuses the gate.
        config: Documented thresholds.

    Returns:
        The :class:`GateReport` verdict.

    Raises:
        RuntimeError: When the split or its label evidence is missing, the
            held-out side is empty or single-class (separation/calibration are
            undefined), or the model carries no score function. The gate
            refuses closed — never fails open.
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
