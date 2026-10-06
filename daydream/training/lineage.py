"""Locked run identity for auditable adapter lineage.

Resume must fail on identity drift. The locked ``corpus_digest`` is a directory
digest over the projection tree, so any change to a carried lineage field also
changes it and aborts :func:`validate_resume`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

__all__ = ["LOCKED_FIELDS", "ResumeAborted", "RunIdentity", "validate_resume"]


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
adding a field to :class:`RunIdentity` automatically locks it. That floor is
asserted in ``tests/test_training_lineage_guard.py``, so it is enforced once,
in CI, instead of as a second executable copy of the list here.
"""


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
