"""Rubric: bundle posterior signals + derive outcome label.

The labeler (Task 13) gathers four posterior signals from
:mod:`daydream.training.labeler_signals` and packages them into a
:class:`Rubric` along with a ``posterior_source`` discriminator that
tells callers which sub-signal carries the authoritative outcome.

A :class:`Rubric` knows two things:

* How to serialize itself to a JSON-friendly ``dict`` for the exporter
  to embed in the manifest / JSONL row (``Rubric.to_dict``).
* How its fields combine into a single outcome label via
  :func:`derive_outcome_label`. Both are pure functions — invalid
  invariants (e.g. ``unresolved > total``) are not validated here;
  upstream extractors guarantee them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from daydream.training.labeler_signals import (
    CommentResolutionSignal,
    FixAppliedSignal,
    LocalCommitAppliedSignal,
    PerFindingResolution,
    PRMergeSignal,
)

PosteriorSource = Literal["pr_review", "local_branch", "none"]

PerFindingLabel = Literal["accepted", "contested", "rejected", "unknown", "missing"]


@dataclass(frozen=True)
class Rubric:
    """Bundle of posterior signals + the discriminator for outcome derivation.

    Attributes:
        pr_merge: Whether the originating PR was merged.
        fix_applied: Layered-cascade verdict on whether the recommended
            diff landed upstream within the review window.
        comment_resolution: Proxy for "review comments addressed".
        local_commit_applied: PR-less branch signal; ``None`` when the
            row originated from a PR.
        posterior_source: Discriminator selecting which sub-signal
            carries the authoritative outcome label.
        per_finding_labels: One outcome label per recorded finding
            (fingerprint-joined), or ``None`` when no per-finding join was
            performed.
    """

    pr_merge: PRMergeSignal
    fix_applied: FixAppliedSignal
    comment_resolution: CommentResolutionSignal
    local_commit_applied: LocalCommitAppliedSignal | None
    posterior_source: PosteriorSource
    per_finding_labels: list[str] | None = None

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
            },
            "fix_applied": {
                "verdict": self.fix_applied.verdict,
                "hunks_applied": self.fix_applied.hunks_applied,
                "hunks_total": self.fix_applied.hunks_total,
                "window_commits": list(self.fix_applied.window_commits),
            },
            "comment_resolution": {
                "total": self.comment_resolution.total,
                "replied": self.comment_resolution.replied,
                "unresolved": self.comment_resolution.unresolved,
            },
        }
        if self.local_commit_applied is not None:
            out["local_commit_applied"] = {"verdict": self.local_commit_applied.verdict}
        if self.per_finding_labels is not None:
            out["per_finding_outcomes"] = list(self.per_finding_labels)
        return out


def derive_outcome_label(rubric: Rubric) -> str:
    """Reduce a rubric to a single outcome label.

    Selection follows ``rubric.posterior_source``:

    * ``"pr_review"`` — merge-state plus comment resolution decide:
      ``"accepted"`` (merged, no unresolved bot comments), ``"contested"``
      (merged but unresolved > 0), or ``"rejected"`` (not merged).
    * ``"local_branch"`` — passes through the verdict on
      :attr:`Rubric.local_commit_applied`.
    * ``"none"`` — always ``"unknown"``.

    Returns:
        One of ``"accepted"``, ``"contested"``, ``"rejected"``, or
        ``"unknown"``.
    """
    if rubric.posterior_source == "pr_review":
        if rubric.pr_merge.merged:
            if rubric.comment_resolution.unresolved == 0:
                return "accepted"
            return "contested"
        return "rejected"
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
    per_finding: list[PerFindingResolution],
) -> list[PerFindingLabel]:
    """Reduce per-finding resolutions to one outcome label per finding.

    Only ``posterior_source == "pr_review"`` yields decisive per-finding
    labels; any other source is inconclusive at finding granularity:

    * ``"pr_review"`` on a **merged** PR:
      - resolved (replied) → ``"accepted"``
      - unresolved with a surviving comment → ``"contested"``
      - unresolved with no comment (deleted / never posted) → ``"missing"``
    * ``"pr_review"`` on an **unmerged** PR → all ``"rejected"``.
    * any other posterior source → all ``"unknown"``.

    Args:
        rubric: The rubric whose posterior source and merge state decide
            the mapping.
        per_finding: The per-finding resolutions to label, in order.

    Returns:
        One :data:`PerFindingLabel` per entry in ``per_finding``, order
        preserved.
    """
    if rubric.posterior_source != "pr_review":
        return ["unknown" for _ in per_finding]
    if not rubric.pr_merge.merged:
        return ["rejected" for _ in per_finding]

    labels: list[PerFindingLabel] = []
    for resolution in per_finding:
        if resolution.resolved:
            labels.append("accepted")
        elif resolution.comment_id is not None:
            labels.append("contested")
        else:
            labels.append("missing")
    return labels
