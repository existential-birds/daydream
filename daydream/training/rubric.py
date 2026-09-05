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

Per-finding label vocabulary: ``accepted`` / ``rejected`` (decisive
classifier dispositions), ``ambiguous`` / ``unanswered`` /
``missing`` (non-decisive), and ``unknown`` (non-PR posterior sources).
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
    resolution_to_dict,
)

PosteriorSource = Literal["pr_review", "local_branch", "none"]

PerFindingLabel = Literal["accepted", "rejected", "ambiguous", "unanswered", "missing", "unknown"]

_DECISIVE = ("accepted", "rejected")
_NON_DECISIVE = ("ambiguous", "unanswered", "missing")


@dataclass(frozen=True)
class Rubric:
    """Bundle of posterior signals + the discriminator for outcome derivation.

    Attributes:
        pr_merge: Whether the originating PR was merged (plus preserved
            PR ``state``/``draft`` context).
        fix_applied: Layered-cascade verdict on whether the recommended
            diff landed upstream within the review window.
        comment_resolution: Proxy for "review comments addressed".
        local_commit_applied: PR-less branch signal; ``None`` when the
            row originated from a PR.
        posterior_source: Discriminator selecting which sub-signal
            carries the authoritative outcome label.
        per_finding_resolutions: Per-finding dispositions joined by
            fingerprint, or ``None`` when no per-finding join was
            performed. Serialized two ways: the full resolution objects
            (fingerprint, disposition, evidence, evidence digest) under
            ``per_finding_resolutions``, and the derived labels-only view
            under ``per_finding_outcomes`` for existing consumers.
    """

    pr_merge: PRMergeSignal
    fix_applied: FixAppliedSignal
    comment_resolution: CommentResolutionSignal
    local_commit_applied: LocalCommitAppliedSignal | None
    posterior_source: PosteriorSource
    per_finding_resolutions: list[PerFindingResolution] | None = None

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
        if self.per_finding_resolutions is not None:
            out["per_finding_resolutions"] = [resolution_to_dict(r) for r in self.per_finding_resolutions]
            out["per_finding_outcomes"] = derive_per_finding_labels(self, self.per_finding_resolutions)
        return out


def derive_outcome_label(rubric: Rubric) -> str:
    """Reduce a rubric to a single outcome label.

    Selection follows ``rubric.posterior_source``:

    * ``"pr_review"`` — conservatively aggregate the per-finding
      dispositions (never bare merge state): ``"accepted"`` when every
      disposition is ``accepted`` and at least one finding was mapped,
      ``"rejected"`` when every disposition is ``rejected`` (also with at
      least one mapped finding), ``"contested"`` when decisive
      dispositions are mixed or decisive evidence coexists with
      ambiguous/unanswered/missing findings, and ``"unknown"`` when no
      finding was mapped or none of the dispositions is decisive. Merge
      state is context only — it never decides the label.
    * ``"local_branch"`` — passes through the verdict on
      :attr:`Rubric.local_commit_applied`.
    * ``"none"`` — always ``"unknown"``.

    Returns:
        One of ``"accepted"``, ``"contested"``, ``"rejected"``, or
        ``"unknown"``.
    """
    if rubric.posterior_source == "pr_review":
        dispositions = [r.disposition for r in (rubric.per_finding_resolutions or [])]
        if not dispositions:
            return "unknown"
        decisive = [d for d in dispositions if d in _DECISIVE]
        if not decisive:
            return "unknown"
        if any(d in _NON_DECISIVE for d in dispositions):
            return "contested"
        if all(d == "accepted" for d in decisive):
            return "accepted"
        if all(d == "rejected" for d in decisive):
            return "rejected"
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
    per_finding: list[PerFindingResolution],
) -> list[PerFindingLabel]:
    """Reduce per-finding resolutions to one outcome label per finding.

    Only ``posterior_source == "pr_review"`` yields dispositions as
    labels, passed through verbatim in order — the classifier already
    decided per finding, and merge state is context only (never mapped
    onto a disposition). Any other posterior source is inconclusive at
    finding granularity: all ``"unknown"``.

    Args:
        rubric: The rubric whose posterior source decides whether
            dispositions may be trusted.
        per_finding: The per-finding resolutions to label, in order.

    Returns:
        One :data:`PerFindingLabel` per entry in ``per_finding``, order
        preserved.
    """
    if rubric.posterior_source != "pr_review":
        return ["unknown" for _ in per_finding]
    return [resolution.disposition for resolution in per_finding]
