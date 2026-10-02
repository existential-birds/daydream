"""Classify projection tiers solely from record type, disposition, and human/developer evidence.
Intrinsic rewards and LLM self-scores cannot promote records to gold.
"""

from typing import Literal, Mapping

from daydream.training.dispositions import (
    DECISIVE_DISPOSITIONS as _DECISIVE_DISPOSITIONS,
    NON_DECISIVE_DISPOSITIONS as _NON_DECISIVE_DISPOSITIONS,
)

__all__ = ["GoldGateError", "classify_tier", "_NON_DECISIVE_DISPOSITIONS"]

Tier = Literal["gold", "silver", "task-only"]

class GoldGateError(ValueError):
    """A decisive disposition without human/developer evidence is an error, never a silent demotion.

    """


def classify_tier(resolution: Mapping[str, object], *, record_type: str = "outcome-finding") -> Tier:
    """Process traces are always silver, regardless of disposition. Nondecisive findings are task-only
    and go to adjudication.

    Decisive findings require a nonempty evidence list or raise GoldGateError. Evidence after as_of
    is retained as silver; eligible decisive findings are gold.
    """
    if not isinstance(resolution, Mapping):
        raise TypeError(f"resolution must be a mapping, got {type(resolution).__name__}")

    if record_type == "process-trace":
        return "silver"

    disposition = resolution.get("disposition")
    if not isinstance(disposition, str):
        raise TypeError(f"disposition must be a string, got {type(disposition).__name__}")

    if disposition in _NON_DECISIVE_DISPOSITIONS:
        return "task-only"

    if disposition in _DECISIVE_DISPOSITIONS:
        evidence = resolution.get("evidence")
        if not isinstance(evidence, list):
            raise TypeError(f"evidence must be a list, got {type(evidence).__name__}")
        if not evidence:
            raise GoldGateError(
                f"decisive disposition {disposition!r} for fingerprint "
                f"{resolution.get('fingerprint')!r} has empty evidence; "
                "gold requires human/developer reply evidence"
            )
        if resolution.get("evidence_after_as_of") is True:
            return "silver"
        return "gold"

    raise TypeError(f"unknown disposition {disposition!r}")
