"""Stage-0 two-class learned outcome reward model (M1, C9, S2).

Trains a small two-class classifier on **gold accepted/rejected** evidence and
scores a finished comment to a ``[0, 1]`` outcome term for the Stage-0 rubric
(:mod:`daydream.training.rubric_v2`). This is the *learned* outcome term —
a sibling of the intrinsic composite in :mod:`daydream.training.reward`, never
a rewrite of it.

Contract points:

- **C9**: training input must contain **both** classes; a labels file with
  only one admitted class raises :class:`ValueError` naming the missing
  class. A positive-only model cannot rank.
- **Gold admission**: input admission reuses the gold-outcome gate
  (``daydream.training.corpus._is_admitted_outcome_gold`` semantics) via the
  corpus loading path: a row is admitted only when its label is an accepted-
  or rejected-class gold label, it carries posterior evidence, the
  reply-classifier policy version is known, and its rubric is decisive-only.
  Legacy rows (no ``labeler_policy_version``) are refused — the guard
  working, never a silent fallback.
- **S2**: the actual accepted/rejected ratio at training time is computed
  from the admitted rows and reported on the model
  (:attr:`OutcomeModel.label_ratio_reported`). No stored/stale figure is
  ever consulted.
- **Determinism**: the frozen split (fractions + seed) and the training pass
  are fully deterministic for a given labels file and seed.
- **Failure propagation**: an unreadable row raises :class:`ValueError`
  naming the row id. There is no skip-and-warn and no default label.

The checkpoint on disk is written by the coordinator (a later task); this
module returns the in-memory model plus a serializable state dict.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from daydream.training.corpus import _is_admitted_outcome_gold

if TYPE_CHECKING:
    from daydream.training.gate import FrozenSplit

# Gold labels usable for outcome training. ``_OUTCOME_GOLD_LABELS`` in the
# corpus module covers the accepted class only (the corpus itself is
# accepted-only per C9); the reward model additionally needs the rejected
# class to learn the negative side of the ranking.
_REJECTED_GOLD_LABELS = frozenset({"rejected"})
_GOLD_LABELS = frozenset({"accepted"}) | _REJECTED_GOLD_LABELS

_FLOOR = 0.0
_CEILING = 1.0
"""Score range for :func:`score_comment`."""

_CHAR_NGRAM_WEIGHT = 0.3
"""Relative weight of a char-trigram occurrence versus a word token."""


@dataclass(frozen=True)
class OutcomeModel:
    """A trained two-class outcome model (frozen; mirror of the
    ``RewardWeights`` fingerprint discipline in ``reward.py``).

    Attributes:
        weights: Token → weight mapping learned on the train split.
        bias: Scalar bias term.
        split_digest: SHA-256 digest of the frozen split (seed + sorted
            held-out row ids) — the same payload the Stage-0 gate hashes, so
            a score's provenance reconciles with the manifest's
            ``run_identity.split_digest``.
        label_ratio_reported: The **actual** accepted fraction among admitted
            training rows, computed at training time (S2).
        train_rows / held_out_rows: Admitted row counts per split.
        held_out_accuracy: Accuracy of the trained model on the held-out
            split (diagnostic evidence for the Stage-0 gate).
        model_fingerprint: Stable 8-char digest of the model state, stamped
            on scores (analogous to ``_weights_fingerprint``).
    """

    weights: dict[str, float]
    bias: float
    split_digest: str
    label_ratio_reported: float
    train_rows: int
    held_out_rows: int
    held_out_accuracy: float
    model_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.model_fingerprint:
            object.__setattr__(self, "model_fingerprint", _model_fingerprint(self.weights, self.bias))

    def state_dict(self) -> dict[str, Any]:
        """Serializable state dict for the coordinator's checkpoint writer."""
        return {
            "weights": dict(self.weights),
            "bias": self.bias,
            "split_digest": self.split_digest,
            "label_ratio_reported": self.label_ratio_reported,
            "train_rows": self.train_rows,
            "held_out_rows": self.held_out_rows,
            "held_out_accuracy": self.held_out_accuracy,
            "model_fingerprint": self.model_fingerprint,
        }


def _model_fingerprint(weights: dict[str, float], bias: float) -> str:
    """Stable 8-char SHA-256 fingerprint of the model state. Pure; no I/O."""
    payload = {"bias": bias, "weights": weights}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:8]


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens. Deliberately dependency-free for determinism."""
    return "".join(ch if ch.isalnum() else " " for ch in text.lower()).split()


def _features(text: str) -> dict[str, float]:
    """L2-normalized bag-of-words + char-trigram features for one comment.

    Word tokens carry full weight; char trigrams (prefixed ``c:``) carry a
    reduced weight so morphological variants of seen words ("grounded" vs
    "grounding") still contribute signal without dominating exact words.
    """
    tokens = _tokenize(text)
    counts: dict[str, float] = {}
    for tok in tokens:
        counts[tok] = counts.get(tok, 0.0) + 1.0
        for i in range(len(tok) - 2):
            gram = "c:" + tok[i : i + 3]
            counts[gram] = counts.get(gram, 0.0) + _CHAR_NGRAM_WEIGHT
    norm = math.sqrt(sum(v * v for v in counts.values()))
    if norm > 0:
        return {tok: v / norm for tok, v in counts.items()}
    return {}


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _read_admitted_rows(labels_path: str | Path) -> list[dict[str, Any]]:
    """Read and admit gold outcome rows, failing closed.

    Admission reuses ``_is_admitted_outcome_gold`` semantics: an
    accepted/rejected gold label backed by posterior evidence, a known
    reply-classifier policy version, and a decisive-only rubric. Rows carry
    these as fields (``has_posterior``, ``labeler_policy_version``,
    ``decisive_mix``, ``decisive_only``). Labels may be written as
    ``label``/``text``/``comment_id`` (the coordinator's Stage-0 labels
    emission) or ``outcome_label``/``review_output``/``session_id`` (the
    production ``run_build_corpus`` export shape). An absent
    ``labeler_policy_version`` refuses the row — a legacy row is refused,
    never silently given a fallback version; the guard is working, never a
    silent admission. Any *explicit* failing value is honored and refuses.

    Raises:
        ValueError: On a row that is not a JSON object, missing
            ``comment_id``/``session_id``, ``text``/``review_output``, or
            ``label``/``outcome_label``, an unknown label, or a row refused
            by the gold-outcome gate. The row id is always named.
    """
    rows: list[dict[str, Any]] = []
    with Path(labels_path).open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"row {lineno} in {labels_path} is not valid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"row {lineno} in {labels_path} is not a JSON object")
            row_id = row.get("comment_id") or row.get("session_id")
            if not isinstance(row_id, str) or not row_id:
                raise ValueError(
                    f"row {lineno} in {labels_path} is missing a string 'comment_id'/'session_id'"
                )
            text = row.get("text")
            if not isinstance(text, str):
                text = row.get("review_output")
                if not isinstance(text, str):
                    raise ValueError(
                        f"row {row_id} in {labels_path} is missing a string 'text'/'review_output'"
                    )
            label = row.get("label") or row.get("outcome_label")
            if label not in _GOLD_LABELS:
                raise ValueError(
                    f"row {row_id} in {labels_path} has non-gold label {label!r}; "
                    f"expected one of {sorted(_GOLD_LABELS)}"
                )
            has_posterior = bool(row.get("has_posterior", True))
            policy_version = row.get("labeler_policy_version")
            decisive_mix = bool(row.get("decisive_mix", False))
            decisive_only = bool(row.get("decisive_only", True))
            if label == "accepted":
                admitted = _is_admitted_outcome_gold(
                    label, has_posterior, policy_version, decisive_mix, decisive_only
                )
            else:  # "rejected": mirror of the gold-outcome guard for the negative class
                admitted = has_posterior and policy_version is not None and not decisive_mix and decisive_only
            if not admitted:
                raise ValueError(
                    f"row {row_id} in {labels_path} is refused by the gold-outcome gate "
                    f"(label={label!r}, has_posterior={has_posterior}, "
                    f"labeler_policy_version={policy_version!r}, decisive_mix={decisive_mix}, "
                    f"decisive_only={decisive_only}); refusing rather than silently admitting"
                )
            # Normalize a production-shape row (outcome_label/review_output/
            # session_id) onto the canonical label/text/comment_id keys the
            # training and split code reads, without mutating the caller's copy
            # of the record.
            row["comment_id"] = str(row_id)
            row["text"] = text
            row["label"] = label
            rows.append(row)
    return rows


def _train_logistic(
    examples: list[tuple[dict[str, float], float]],
    *,
    epochs: int,
    lr: float,
    l2: float,
    seed: int,
) -> tuple[dict[str, float], float]:
    """Deterministic logistic regression via averaged SGD with a seeded shuffle."""
    rng = random.Random(seed)
    weights: dict[str, float] = {}
    bias = 0.0
    order = list(range(len(examples)))
    for _ in range(epochs):
        rng.shuffle(order)
        for idx in order:
            feats, y = examples[idx]
            z = bias + sum(weights.get(tok, 0.0) * v for tok, v in feats.items())
            err = _sigmoid(z) - y
            for tok, v in feats.items():
                g = err * v + l2 * weights.get(tok, 0.0)
                weights[tok] = weights.get(tok, 0.0) - lr * g
            bias -= lr * err
    return weights, bias


def train_outcome_model(
    labels_path: str | Path,
    *,
    split: FrozenSplit,
    seed: int,
    epochs: int = 20,
    lr: float = 0.5,
    l2: float = 1e-4,
) -> OutcomeModel:
    """Train the two-class outcome model on gold accepted/rejected evidence.

    Args:
        labels_path: JSONL labels file (one JSON object per line with
            ``comment_id``, ``text``, ``label``; optional gold-gate fields).
        split: The frozen train/held-out partition produced by
            :func:`daydream.training.gate.freeze_split` — the single author
            of the split and its digest.
        seed: Seed for the SGD training pass (the split shuffle's own seed is
            already frozen in ``split``).
        epochs / lr / l2: Training hyperparameters (deterministic given seed).

    Returns:
        The frozen :class:`OutcomeModel`.

    Raises:
        ValueError: When the file yields fewer than two classes (C9 — the
            missing class is named), or a row is unreadable or refused by the
            gold-outcome gate.
    """
    rows = _read_admitted_rows(labels_path)
    if not rows:
        raise ValueError(f"labels file {labels_path} contains no admissible gold outcome rows")

    n_accepted = sum(1 for r in rows if r["label"] == "accepted")
    n_rejected = len(rows) - n_accepted
    missing: list[str] = []
    if n_accepted == 0:
        missing.append("accepted")
    if n_rejected == 0:
        missing.append("rejected")
    if missing:
        raise ValueError(
            f"cannot train a two-class outcome model on a single class: labels file "
            f"{labels_path} has no {missing[0]!r} rows after gold admission "
            f"(accepted={n_accepted}, rejected={n_rejected}). C9: a positive-only "
            "model cannot rank — training data must contain both classes."
        )

    train_rows = split.train_rows
    held_out_rows = split.held_out_rows
    split_digest = split.digest

    label_to_y = {"accepted": 1.0, "rejected": 0.0}
    train_examples = [(_features(str(r["text"])), label_to_y[str(r["label"])]) for r in train_rows]
    weights, bias = _train_logistic(train_examples, epochs=epochs, lr=lr, l2=l2, seed=seed)

    eval_model = OutcomeModel(
        weights=weights, bias=bias, split_digest=split_digest,
        label_ratio_reported=0.0, train_rows=0, held_out_rows=0, held_out_accuracy=0.0,
    )
    correct = 0
    for row in held_out_rows:
        y = label_to_y[str(row["label"])]
        pred = 1.0 if score_comment(eval_model, str(row["text"])) >= 0.5 else 0.0
        if pred == y:
            correct += 1
    held_out_accuracy = correct / len(held_out_rows)

    label_ratio_reported = n_accepted / len(rows)
    return OutcomeModel(
        weights=weights,
        bias=bias,
        split_digest=split_digest,
        label_ratio_reported=label_ratio_reported,
        train_rows=len(train_rows),
        held_out_rows=len(held_out_rows),
        held_out_accuracy=held_out_accuracy,
    )


def score_comment(model: OutcomeModel, text: str) -> float:
    """Score one finished comment to a ``[0, 1]`` outcome term.

    Deterministic for a fixed model: the same text always yields the same
    score (pure function of the frozen weights and the text features).
    """
    feats = _features(text)
    z = model.bias + sum(model.weights.get(tok, 0.0) * v for tok, v in feats.items())
    return max(_FLOOR, min(_CEILING, _sigmoid(z)))
