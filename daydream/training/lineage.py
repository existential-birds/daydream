"""Run identity, LOCKED_FIELDS resume guard, and per-stage lineage digests.

Implements M16 (per-stage split/lineage digests) and M18 (resume guard) from
issue #91. A resumed training run must abort loudly — never warn — when any
locked run-identity field differs from the run being resumed, because a silent
drift in e.g. the learning rate or corpus digest would make the resulting
adapter's lineage unauditable and the comparison across stages meaningless.

Digests follow the content-addressing style of
:func:`daydream.training.corpus._trajectory_set_hash`: canonical JSON of the
record lineage fields, sha256-hex. Digests are deterministic functions of
content, never of ordering.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, fields
from typing import Any

__all__ = ["LOCKED_FIELDS", "ResumeAborted", "RunIdentity", "stage_digests", "validate_resume"]


@dataclass(frozen=True)
class RunIdentity:
    """Immutable run-identity snapshot stamped into the stage manifest.

    Every field is locked (see :data:`LOCKED_FIELDS`): a resumed run whose
    identity differs in any field is aborted by :func:`validate_resume`.
    """

    base_model: str
    tokenizer_renderer: str
    max_seq_len: int
    lora_rank: int
    lora_targets: tuple[str, ...]
    optimizer: str
    learning_rate: float
    corpus_digest: str
    split_digest: str
    profile_policy: str
    reward_version: str
    reward_weights: dict[str, float]
    stack_pins: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["lora_targets"] = list(self.lora_targets)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunIdentity:
        kwargs = {f.name: data[f.name] for f in fields(cls)}
        kwargs["lora_targets"] = tuple(kwargs["lora_targets"])
        return cls(**kwargs)


LOCKED_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(RunIdentity))
"""Frozen tuple of every locked run-identity field.

Covers exactly the AC5 list — base model, tokenizer/renderer, max sequence
length, LoRA rank + target modules, optimizer + learning rate, corpus digest,
split digest, profile policy, reward version + weights, and exact stack pins
(verifiers + prime-rl versions). The AC5 list is the floor, not the ceiling:
adding a field to :class:`RunIdentity` automatically locks it.
"""

assert frozenset(LOCKED_FIELDS) >= {
    "base_model", "tokenizer_renderer", "max_seq_len", "lora_rank", "lora_targets",
    "optimizer", "learning_rate", "corpus_digest", "split_digest", "profile_policy",
    "reward_version", "reward_weights", "stack_pins",
}


class ResumeAborted(ValueError):
    """Raised when a resumed run's identity differs from the prior run's.

    A ``ValueError`` subclass so callers that treat any identity mismatch as a
    hard configuration error can catch the broader type.
    """


def validate_resume(prior: RunIdentity, changed: RunIdentity) -> None:
    """Compare every locked field; abort loudly on any difference.

    Raises :class:`ResumeAborted` listing **every** differing field name — a
    loud abort, never a warning. Identical identities pass silently.
    """
    prior_d, changed_d = prior.to_dict(), changed.to_dict()
    differing = sorted(f for f in LOCKED_FIELDS if prior_d[f] != changed_d[f])
    if differing:
        details = "; ".join(f"{f}: {prior_d[f]!r} -> {changed_d[f]!r}" for f in differing)
        raise ResumeAborted(
            f"resume aborted: locked run-identity fields differ from the prior run: {details}"
        )


_RECORD_LINEAGE_FIELDS: tuple[str, ...] = (
    "session_id",
    "evidence_tier",
    "base_sha",
    "head_sha",
    "diff_identity",
    "daydream_version",
    "profile_digest",
    "detected_stack",
    "label_source",
    "label_version",
    "reward_version",
    "split",
)
"""Corpus-record lineage fields carried through (validated present), not re-derived (M16)."""


def _content_hash(payload: Any) -> str:
    """sha256 of canonical JSON — order-independent, content-addressed."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=sorted).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _split_digest(records: list[dict[str, Any]]) -> str:
    """Digest of the record set's split identity (session ids), corpus-style."""
    session_ids = [r.get("session_id", "") for r in records]
    return hashlib.sha256("\n".join(sorted(session_ids)).encode("utf-8")).hexdigest()


def _lineage_digest(records: list[dict[str, Any]]) -> str:
    """Digest over the record lineage fields each record carries through (M16).

    Only fields actually present on a record contribute, so the digest changes
    whenever a carried-through lineage field (evidence tier, split, reward
    version, …) differs — the fields are validated upstream and carried
    through, never re-derived here.
    """
    payloads = sorted(
        json.dumps({f: r.get(f) for f in _RECORD_LINEAGE_FIELDS if f in r}, sort_keys=True, default=str)
        for r in records
    )
    return _content_hash(payloads)


def stage_digests(stage_outputs: dict[str, dict[str, Any]]) -> dict[str, dict[str, str]]:
    """Emit ``split_digest`` and ``lineage_digest`` per stage (M16).

    ``stage_outputs`` maps stage name → dict with a ``records`` list of corpus
    records. Digests are content-addressed and order-independent.
    """
    digests: dict[str, dict[str, str]] = {}
    for stage, outputs in stage_outputs.items():
        records = list(outputs.get("records", []))
        digests[stage] = {
            "split_digest": _split_digest(records),
            "lineage_digest": _lineage_digest(records),
        }
    return digests
