"""Frozen split assignment from record id and pinned salt, without RNG or state."""

import hashlib
from typing import Literal

__all__ = ["assign_split", "SPLIT_FILENAMES"]

Split = Literal["train", "validation", "holdout"]

SPLIT_FILENAMES: dict[Split, str] = {
    "train": "train.jsonl",
    "validation": "validation.jsonl",
    "holdout": "holdout.jsonl",
}


def assign_split(record_id: str, *, holdout_rate: float, val_rate: float, salt: str) -> Split:
    """Map sha256(salt <US> record_id) uniformly to [0, 1).

    Partition at holdout_rate and holdout_rate + val_rate into holdout, validation,
    and train. Rates must be nonnegative and their sum at most one.
    """
    if holdout_rate < 0.0 or val_rate < 0.0 or holdout_rate + val_rate > 1.0:
        raise ValueError(
            f"assign_split: invalid rates holdout_rate={holdout_rate!r} "
            f"val_rate={val_rate!r} (require non-negative and holdout+val <= 1)"
        )
    digest = hashlib.sha256(f"{salt}\x1f{record_id}".encode("utf-8")).digest()
    u = int.from_bytes(digest[:32], "big") / 2**256
    if u < holdout_rate:
        return "holdout"
    if u < holdout_rate + val_rate:
        return "validation"
    return "train"
