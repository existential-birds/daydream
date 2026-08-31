"""Tier + eligibility classification for corpus v2 records.

The gold gate is human-evidence-only and structural (C5): no intrinsic
reward or LLM self-score input exists in this module's signature, so a
score can never promote a record to gold. The module deliberately
imports nothing from ``daydream.training.reward``.
"""

from typing import Literal, Mapping

from daydream.training.adjudication.dispositions import (
    DECISIVE_DISPOSITIONS as _DECISIVE_DISPOSITIONS,
)
from daydream.training.adjudication.dispositions import (
    NON_DECISIVE_DISPOSITIONS as _NON_DECISIVE_DISPOSITIONS,
)

__all__ = ["GoldGateError", "classify_tier", "_NON_DECISIVE_DISPOSITIONS"]

Tier = Literal["gold", "silver", "task-only"]

class GoldGateError(ValueError):
    """Raised when a decisive disposition lacks human/developer evidence.

    Fail-closed: a record that claims ``accepted``/``rejected`` but carries
    no evidence is never silently demoted to silver — it is an error.
    """


def classify_tier(resolution: Mapping[str, object], *, record_type: str = "outcome-finding") -> Tier:
    """Classify a per-finding resolution into a disjoint tier.

    - ``record_type == "process-trace"`` is always ``silver``: ATIF process
      data is a separate class from outcome decisions, whatever its
      disposition says.
    - ``outcome-finding`` records are ``gold`` only when the disposition is
      decisive (``accepted``/``rejected``) *and* the evidence list is
      non-empty. Decisive-but-evidenceless raises :class:`GoldGateError`.
    - Non-decisive dispositions are ``task-only`` (the projector routes
      them to adjudication).
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
        # C5/M9 temporal gate: evidence observed after the record's as_of
        # pin cannot establish gold eligibility — keep the evidence but
        # classify silver, never gold.
        if resolution.get("evidence_after_as_of") is True:
            return "silver"
        return "gold"

    raise TypeError(f"unknown disposition {disposition!r}")
