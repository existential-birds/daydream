"""Serialize posterior evidence and derive conservative finding/run labels.

PR merge state is contextual; per-finding human dispositions decide outcome.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Literal

from daydream.training.dispositions import (
    DECISIVE_DISPOSITIONS,
    NON_DECISIVE_DISPOSITIONS,
)
from daydream.training.labeler_signals import (
    CommentResolutionSignal,
    FixAppliedSignal,
    LocalCommitAppliedSignal,
    PerFindingResolution,
    PRMergeSignal,
    resolution_to_dict,
)

PosteriorSource = Literal["pr_review", "local_branch", "none"]

PerFindingLabel = Literal["accepted", "rejected", "ambiguous", "unanswered", "missing", "unknown"]


@dataclass(frozen=True)
class Rubric:
    """Posterior signals with an authoritative-source discriminator.

    Per-finding resolutions serialize as full provenance records and a separate
    labels-only view. None means no per-finding join was performed."""

    pr_merge: PRMergeSignal
    fix_applied: FixAppliedSignal
    comment_resolution: CommentResolutionSignal
    local_commit_applied: LocalCommitAppliedSignal | None
    posterior_source: PosteriorSource
    per_finding_resolutions: Sequence[PerFindingResolution] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation with explicit key order.

        ``local_commit_applied`` is omitted entirely when ``None`` so the
        emitted JSON stays compact for PR-sourced rows.
        """
        out: dict[str, Any] = {
            "posterior_source": self.posterior_source,
            "pr_merge": {
                "merged": self.pr_merge.merged,
                "merged_at": self.pr_merge.merged_at,
                "state": self.pr_merge.state,
                "draft": self.pr_merge.draft,
            },
            "fix_applied": {**asdict(self.fix_applied), "window_commits": list(self.fix_applied.window_commits)},
            "comment_resolution": asdict(self.comment_resolution),
        }
        if self.local_commit_applied is not None:
            out["local_commit_applied"] = {"verdict": self.local_commit_applied.verdict}
        if self.per_finding_resolutions is not None:
            out["per_finding_resolutions"] = [resolution_to_dict(r) for r in self.per_finding_resolutions]
            out["per_finding_outcomes"] = derive_per_finding_labels(self, self.per_finding_resolutions)
        return out


def derive_outcome_label(rubric: Rubric) -> str:
    """Aggregate findings conservatively; merge state never supplies a label.

    PR evidence is accepted/rejected only when decisive findings agree without
    non-decisive findings. Mixed evidence is contested; no decisive findings is
    unknown. Local-branch verdicts map independently, and source none is unknown.
    Extractors guarantee signal invariants; this reducer does not repair them."""
    if rubric.posterior_source == "pr_review":
        dispositions = {r.disposition for r in (rubric.per_finding_resolutions or [])}
        decisive = dispositions & DECISIVE_DISPOSITIONS
        if not decisive:
            return "unknown"
        if len(decisive) == 1 and dispositions.isdisjoint(NON_DECISIVE_DISPOSITIONS):
            return next(iter(decisive))
        return "contested"
    if rubric.posterior_source == "local_branch":
        # Extractor invariant: posterior_source="local_branch" implies
        # local_commit_applied is not None.
        if rubric.local_commit_applied is None:
            raise RuntimeError(
                "Extractor invariant violated: posterior_source='local_branch' but local_commit_applied is None"
            )
        verdict = rubric.local_commit_applied.verdict
        if verdict == "applied":
            return "accepted"
        if verdict == "rejected":
            return "rejected"
        return "unknown"
    return "unknown"


def derive_per_finding_labels(
    rubric: Rubric,
    per_finding: Sequence[PerFindingResolution],
) -> list[PerFindingLabel]:
    """Pass PR finding dispositions through in order; other sources yield unknown.

    Finding-level labels never derive from merge state."""
    if rubric.posterior_source != "pr_review":
        return ["unknown" for _ in per_finding]
    return [resolution.disposition for resolution in per_finding]
