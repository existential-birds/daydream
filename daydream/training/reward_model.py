"""Learn a deterministic two-class outcome model from gold labels.

Admission requires accepted/rejected labels, posterior evidence, a policy
version, and decisive-only data. Both classes are mandatory, and reported
ratios are measured from admitted rows. Malformed or refused rows raise with
their identity. This learned outcome term remains separate from intrinsic reward.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from daydream.training.corpus import _is_admitted_outcome_gold

if TYPE_CHECKING:
    from daydream.training.gate import FrozenSplit

_GOLD_LABELS = frozenset({"accepted", "rejected"})

_FLOOR = 0.0
_CEILING = 1.0
"""Score range for :func:`score_comment`."""

_CHAR_NGRAM_WEIGHT = 0.3
"""Relative weight of a char-trigram occurrence versus a word token."""


@dataclass(frozen=True)
class OutcomeModel:
    """Frozen classifier state and measured train/holdout evidence.

    split_digest matches the gate's frozen partition. label_ratio_reported is the
    accepted fraction of admitted training input; held_out_accuracy measures the
    held-out partition. model_fingerprint is the first eight digest characters.
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
        return asdict(self)


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
    """Read and admit gold accepted/rejected JSONL rows; refuse malformed or legacy rows.

    Accept label/text/comment_id and outcome_label/review_output/session_id spellings.
    Missing policy versions never receive a fallback. Explicit admission failures
    remain failures; normalize admitted rows to canonical keys for training.
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
            admitted = _is_admitted_outcome_gold(
                label, has_posterior, policy_version, decisive_mix, decisive_only,
                allowed_labels=_GOLD_LABELS,
            )
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
    """Train on the frozen partition, measuring class balance and held-out accuracy.

    Both classes must exist after admission or ValueError names the missing one.
    The split is supplied by the gate; seed controls only the deterministic SGD
    shuffle. Unreadable or refused input propagates before training.
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

    model = OutcomeModel(
        weights=weights, bias=bias, split_digest=split_digest,
        label_ratio_reported=n_accepted / len(rows), train_rows=len(train_rows),
        held_out_rows=len(held_out_rows), held_out_accuracy=0.0,
    )
    correct = sum(
        (1.0 if score_comment(model, str(row["text"])) >= 0.5 else 0.0) == label_to_y[str(row["label"])]
        for row in held_out_rows
    )
    return replace(model, held_out_accuracy=correct / len(held_out_rows))


def score_comment(model: OutcomeModel, text: str) -> float:
    """Score one finished comment to a ``[0, 1]`` outcome term.

    Deterministic for a fixed model: the same text always yields the same
    score (pure function of the frozen weights and the text features).
    """
    feats = _features(text)
    z = model.bias + sum(model.weights.get(tok, 0.0) * v for tok, v in feats.items())
    return max(_FLOOR, min(_CEILING, _sigmoid(z)))
