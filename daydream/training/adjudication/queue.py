"""Build deterministic record_id-keyed queues from validated source findings.

Projection and adjudication share finding validation and disposition policy.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from daydream.training.adjudication.snapshot import FindingRecord
from daydream.training.dispositions import (
    NON_DECISIVE_DISPOSITIONS as _NON_DECISIVE_DISPOSITIONS,
)
from daydream.training.labeler_versions import ADJUDICATION_LABELER_VERSION

__all__ = ["build_queue", "_NON_DECISIVE_DISPOSITIONS"]


def build_queue(
    sessions: Sequence[Mapping[str, object]],
    *,
    rubric_version: str = ADJUDICATION_LABELER_VERSION,
    prior_observations: Mapping[str, Mapping[str, Any]] | None = None,
    include_decisive: bool = False,
) -> list[dict[str, object]]:
    """Rebuild a record_id-sorted queue with evidence-drift reopening.

    Select non-decisive source findings. Findings with prior observations also remain
    available for human review, even when automatically decisive. include_decisive
    adds the complete record set for drift checks using the same item shape,
    status, and observation logic.

    A completed human judgment with an old evidence digest reopens with its prior
    disposition retained. Other items start open. Missing fresh digests or invalid
    dispositions raise ValueError naming the fingerprint. Identical inputs produce
    identical queues regardless of session order.
    """
    prior_observations = prior_observations or {}
    items = [
        finding.queue(rubric_version, prior_observations.get(finding.record_id))
        for session in sessions
        for finding in FindingRecord.from_session(session)
        if include_decisive or finding.resolution["disposition"] in _NON_DECISIVE_DISPOSITIONS
        or finding.record_id in prior_observations
    ]
    return sorted(items, key=lambda item: str(item["record_id"]))
