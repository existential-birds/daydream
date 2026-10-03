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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from daydream.training.gate import FrozenSplit

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
    split: FrozenSplit,
    *,
    seed: int,
    epochs: int = 20,
    lr: float = 0.5,
    l2: float = 1e-4,
) -> OutcomeModel:
    """Train on the frozen partition, measuring class balance and held-out accuracy.

    The split owns immutable gold admission and both-class refusal; counts and
    fitting use that same population. Seed controls only the deterministic SGD
    shuffle, never the projection's frozen membership or row order.
    """
    rows = (*split.train_rows, *split.held_out_rows)
    n_accepted = sum(row["label"] == "accepted" for row in rows)

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
