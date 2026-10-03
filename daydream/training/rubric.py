"""Serialize posterior evidence and derive conservative finding/run labels.

PR merge state is contextual; per-finding human dispositions decide outcome.
The Stage-0 scoring rubric separately combines learned outcome, subtractive FP
penalty and intrinsic evidence, with golden overlap retained as telemetry.
"""

from __future__ import annotations

import hashlib
import json
import types
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
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
from daydream.training.reward import DEFAULT_WEIGHTS, ScoringInputs, _clip, score_trajectory

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
        return {"applied": "accepted", "rejected": "rejected"}.get(rubric.local_commit_applied.verdict, "unknown")
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


# Missing signals are renormalized out, never imputed as zero. Intrinsic and
# golden signals cannot replace the learned outcome term or its Stage-0 gate.

REWARD_VERSION_RUBRIC = "2026.10.01-rubric-1"
"""Bump on any change to rubric weights, penalty semantics, or composite shape.

Read at call time so a test can monkeypatch
``daydream.training.rubric.REWARD_VERSION_RUBRIC`` and have
:func:`score_review` observe the override. Stamped verbatim on breakdowns
scored under :data:`DEFAULT_RUBRIC_WEIGHTS`; custom weights get a
``+custom-{fingerprint}`` suffix (same convention as ``reward.py``).
"""

_PROTOCOL_ATTR = "score_comment"
"""The one method :func:`score_review` requires of the outcome model."""


@dataclass(frozen=True)
class RubricV2Weights:
    """Stage-0 term weights; the subtractive FP weight must be positive.

    Custom instances receive a fingerprinted version instead of the canonical
    DEFAULT_RUBRIC_WEIGHTS stamp. Intrinsic evidence is a signal, never a
    substitute for the learned outcome term."""

    w_learned_outcome: float = 0.4
    w_false_positive: float = 0.3
    w_intrinsic: float = 0.5

    def __post_init__(self) -> None:
        if self.w_false_positive <= 0:
            raise ValueError(
                f"w_false_positive must be > 0 (got {self.w_false_positive!r}); "
                "the CR-Bench false-positive penalty is load-bearing (M2)."
            )


DEFAULT_RUBRIC_WEIGHTS = RubricV2Weights()
"""The golden-locked rubric weights; only this instance earns the canonical
:data:`REWARD_VERSION_RUBRIC` stamp (identity-checked, as in ``reward.py``)."""

def _rubric_fingerprint(weights: RubricV2Weights) -> str:
    """Stable 8-hex SHA-256 fingerprint of rubric weights (sorted-key JSON).

    Mirrors ``reward._weights_fingerprint``. Pure; no I/O.
    """
    payload = {field.name: getattr(weights, field.name) for field in fields(RubricV2Weights)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:8]


def _validate_findings(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Raise ``ValueError`` naming the finding id on any malformed finding.

    A well-formed finding is a mapping with a string ``text``. No silent skip.
    """
    checked: list[dict[str, Any]] = []
    for i, f in enumerate(findings):
        if not isinstance(f, dict) or not isinstance(f.get("text"), str):
            fid = f.get("id") if isinstance(f, dict) else None
            raise ValueError(f"Malformed finding at index {i} (id={fid!r}): expected a dict with a string 'text'.")
        checked.append(f)
    return checked


@dataclass(frozen=True)
class RubricV2Breakdown:
    """Stage-0 review score with explicit term presence and telemetry.

    learned_outcome is the mean model score; FP penalty is fp/total and
    signal_to_noise is (total-fp)/total. Ratios are absent at zero total.
    Golden overlap is telemetry only. intrinsic_composite comes from the shipped
    intrinsic scorer and may be absent.

    terms carries negative penalties, None for missing signals, and unweighted
    telemetry. The weighted mean renormalizes present terms, clips to [0, 1],
    and rounds to four places."""

    learned_outcome: float | None
    false_positive_penalty: float | None
    signal_to_noise: float | None
    golden_overlap: float
    intrinsic_composite: float | None
    terms: dict[str, float | None]
    composite: float
    reward_version: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation with explicit key order."""
        return asdict(self)


def score_review(
    model: Any,
    *,
    findings: list[dict[str, Any]],
    fp_count: int,
    total_findings: int,
    breakdown: bool = False,
    gold_texts: frozenset[str] | set[str] | None = None,
    weights: RubricV2Weights = DEFAULT_RUBRIC_WEIGHTS,
) -> float | RubricV2Breakdown:
    """Score a review using model.score_comment(text), returning a scalar or breakdown.

    Reject malformed findings with their index/id and FP counts outside
    [0, total_findings]. gold_texts affects overlap telemetry only. Missing
    composite terms are renormalized out; no present term or nonpositive total
    weight raises ValueError."""
    if not hasattr(model, _PROTOCOL_ATTR):
        raise TypeError(f"model must expose {_PROTOCOL_ATTR}(text) -> float; got {type(model).__name__!r}.")
    if not 0 <= fp_count <= total_findings:
        raise ValueError(
            f"fp_count must be in [0, total_findings] (got fp_count={fp_count!r}, total={total_findings!r})."
        )
    checked = _validate_findings(findings)

    version = (
        REWARD_VERSION_RUBRIC
        if weights is DEFAULT_RUBRIC_WEIGHTS
        else f"{REWARD_VERSION_RUBRIC}+custom-{_rubric_fingerprint(weights)}"
    )

    # Learned outcome term (M6 anchor): mean model score over finding texts.
    learned: float | None = None
    if checked:
        learned = sum(float(model.score_comment(str(f["text"]))) for f in checked) / len(checked)

    # CR-Bench terms: penalty magnitude fp/total (subtractive), usefulness
    # rate (total - fp)/total as Signal-to-Noise telemetry.
    false_positive_penalty = fp_count / total_findings if total_findings else None
    signal_to_noise = (total_findings - fp_count) / total_findings if total_findings else None

    # Golden overlap: telemetry only (never a composite substitute, M6).
    gold = gold_texts or set()
    golden_overlap = len([f for f in checked if str(f["text"]) in gold]) / len(checked) if checked else 0.0

    # Intrinsic composite: read via score_trajectory, never recomputed here.
    verdicts = [{"verdict": str(f["verdict"])} for f in checked if f.get("verdict")]
    total_chars = sum(len(str(f["text"])) for f in checked)
    intrinsic = score_trajectory(
        ScoringInputs(
            verifier_verdicts=verdicts or None,
            format_valid=True,
            length=total_chars or None,
        ),
        weights=DEFAULT_WEIGHTS,
    )
    intrinsic_composite = intrinsic.composite

    terms: dict[str, float | None] = {
        "learned_outcome": learned,
        "fp_penalty": -false_positive_penalty if false_positive_penalty is not None else None,
        "intrinsic_composite": intrinsic_composite,
        "golden_overlap": golden_overlap,  # telemetry only — never contributes (M6)
    }

    # Weighted mean over present composite terms, renormalized (missing
    # signals are None, never 0.0).
    present: list[tuple[float, float]] = []
    for name, value in terms.items():
        if value is None or name == "golden_overlap":
            continue
        w = getattr(weights, _TERM_WEIGHTS[name])
        present.append((w, value))
    if not present:
        raise ValueError("No rubric term is present; composite is uncomputable.")
    weight_sum = sum(w for w, _ in present)
    if weight_sum <= 0:
        raise ValueError(f"Invalid RubricV2Weights: sum of present term weights must be > 0 (got {weight_sum!r}).")
    composite = round(_clip(sum((w / weight_sum) * v for w, v in present), 0.0, 1.0), 4)

    result = RubricV2Breakdown(
        learned_outcome=learned,
        false_positive_penalty=false_positive_penalty,
        signal_to_noise=signal_to_noise,
        golden_overlap=golden_overlap,
        intrinsic_composite=intrinsic_composite,
        terms=terms,
        composite=composite,
        reward_version=version,
    )
    return result if breakdown else result.composite


_TERM_WEIGHTS: types.MappingProxyType[str, str] = types.MappingProxyType(
    {
        "learned_outcome": "w_learned_outcome",
        "fp_penalty": "w_false_positive",
        "intrinsic_composite": "w_intrinsic",
    }
)
"""Composite-term name → :class:`RubricV2Weights` field name."""
