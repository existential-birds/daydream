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

from collections.abc import Sequence
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
    per_finding: Sequence[PerFindingResolution],
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


# ---------------------------------------------------------------------------
# Stage-0 scoring rubric (M2, M5, M6, M7): learned outcome term + CR-Bench FP
# penalty.
#
# Composes, never rewrites, the shipped intrinsic composite from
# :mod:`daydream.training.reward`. This section adds two components the
# intrinsic composite does not have:
#
# - a **learned outcome term** from the Stage-0 two-class model
#   (:func:`daydream.training.reward_model.score_comment`) — the model's
#   ``[0, 1]`` probability that a finished comment reads as gold-accepted
#   review prose rather than noise;
# - an explicit **false-positive penalty** with CR-Bench Usefulness Rate
#   semantics (``(total − fp)/total``; KD3): the penalty magnitude is
#   ``fp/total`` and it enters the composite subtractively, so a run whose
#   findings are all noise can never outrank a clean run at the same recall
#   (M5).
#
# The intrinsic composite and the golden-overlap telemetry are carried as
# **signals only**: neither can substitute for the learned outcome term or the
# Stage-0 gate (M6). Missing signals are ``None`` and renormalized out of the
# composite — never imputed ``0.0`` (mirrors ``reward.py``'s ``axes_present``
# rule).
#
# Version discipline (M7 / PATTERN golden-update): :data:`REWARD_VERSION_RUBRIC`
# stamps every breakdown; scoring under a non-default :class:`RubricV2Weights`
# appends a ``+custom-{fingerprint}`` suffix so an analysis-time override can
# never be mistaken for the canonical rubric score. Changing any default weight
# is a deliberate golden-update: re-pin the golden values and bump
# :data:`REWARD_VERSION_RUBRIC`. So is redefining what an input label *means*
# while the weights hold still — the stamp identifies the label semantics as
# much as the algebra.
#
# Pure: no filesystem, network, or subprocess access.
# ---------------------------------------------------------------------------

import hashlib  # noqa: E402  (rubric-scoring section)
import json  # noqa: E402  (rubric-scoring section)
import types  # noqa: E402  (rubric-scoring section)

from daydream.training.reward import DEFAULT_WEIGHTS, ScoringInputs, score_trajectory  # noqa: E402

REWARD_VERSION_RUBRIC = "2026.09.04-rubric-1"
"""Bump on any change to rubric weights, penalty semantics, or composite shape.

``2026.09.04-rubric-1`` is **not** a weight change: :class:`RubricV2Weights`
defaults, the penalty semantics and the composite shape are all identical to
``2026.05.28-rubric-1``. The bump records an upstream *label* redefinition
(issue #1106): ``daydream.eval.analyzer.analyze_grounding`` tightened its
grounding predicate from "the finding's cited file was read" to that plus "the
finding's cited line resolves inside (or within tolerance of) a diff hunk". That
predicate produces the ``grounded`` count this rubric divides by
``total_findings`` for the localization term, and the ``grounding_rate`` it hands
to :func:`~daydream.training.reward.score_trajectory`, so scores stamped before
and after are not comparable and must not share a stamp.

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
    """Tunable weights for :func:`score_review`.

    Defaults reproduce the golden-locked rubric; overriding any field is an
    analysis-time choice stamped with a ``+custom-`` suffix (never stored as
    the canonical rubric score).

    Attributes:
        w_learned_outcome: Weight of the learned outcome term (the M6 anchor —
            the term that cannot be substituted by intrinsic or golden signals).
        w_false_positive: Weight of the subtractive CR-Bench false-positive
            penalty. Must be positive (M2).
        w_localization: Weight of the localization term (grounded findings /
            total findings).
        w_tool_grounded: Weight of the tool-grounded term (fraction of
            findings carrying tool evidence); absent when no finding reports
            tool usage.
        w_intrinsic: Weight of the shipped intrinsic composite (signal only).
    """

    w_learned_outcome: float = 0.4
    w_false_positive: float = 0.3
    w_localization: float = 0.2
    w_tool_grounded: float = 0.1
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

_WEIGHT_FIELDS = (
    "w_learned_outcome",
    "w_false_positive",
    "w_localization",
    "w_tool_grounded",
    "w_intrinsic",
)


def _rubric_fingerprint(weights: RubricV2Weights) -> str:
    """Stable 8-hex SHA-256 fingerprint of rubric weights (sorted-key JSON).

    Mirrors ``reward._weights_fingerprint``. Pure; no I/O.
    """
    payload = {name: getattr(weights, name) for name in _WEIGHT_FIELDS}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:8]


def _clip(value: float, low: float, high: float) -> float:
    """Clamp ``value`` to the closed interval ``[low, high]``."""
    return max(low, min(high, value))


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
    """Per-term decomposition + composite for one review under rubric v2.

    Attributes:
        learned_outcome: Mean learned outcome score over finding texts
            (``[0, 1]``), or ``None`` when there are no findings to score.
        false_positive_penalty: CR-Bench penalty magnitude ``fp/total``
            (``[0, 1]``), or ``None`` when ``total`` is zero; the composite
            term is ``−w_false_positive ×`` this.
        signal_to_noise: CR-Bench Usefulness Rate ``(total − fp)/total``
            telemetry (not a composite term), or ``None`` when ``total`` is
            zero.
        localization: Fraction of findings grounded (``grounded/total``), or
            ``None`` when ``total`` is zero.
        tool_grounded: Fraction of findings carrying tool evidence, or
            ``None`` when no finding reports tool usage.
        golden_overlap: Telemetry — fraction of findings present in the
            supplied gold evidence set (``0.0`` when none supplied). Carried
            as a signal only; never a substitute for the learned term (M6).
        intrinsic_composite: The shipped pure-intrinsic composite from
            ``reward.score_trajectory`` (read, never recomputed), or ``None``
            when the intrinsic score was uncomputable.
        terms: Composite-term map (negative values are penalties; ``None``
            marks an absent signal that was renormalized out; telemetry-only
            entries such as ``golden_overlap`` are present but excluded from
            the weighted mean).
        composite: The rubric composite — weighted mean of present terms,
            renormalized, clipped to ``[0, 1]``, rounded to 4 places.
        reward_version: The version stamp at scoring time.
    """

    learned_outcome: float | None
    false_positive_penalty: float | None
    signal_to_noise: float | None
    localization: float | None
    tool_grounded: float | None
    golden_overlap: float
    intrinsic_composite: float | None
    terms: dict[str, float | None]
    composite: float
    reward_version: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation with explicit key order."""
        return {
            "learned_outcome": self.learned_outcome,
            "false_positive_penalty": self.false_positive_penalty,
            "signal_to_noise": self.signal_to_noise,
            "localization": self.localization,
            "tool_grounded": self.tool_grounded,
            "golden_overlap": self.golden_overlap,
            "intrinsic_composite": self.intrinsic_composite,
            "terms": dict(self.terms),
            "composite": self.composite,
            "reward_version": self.reward_version,
        }


def score_review(
    model: Any,
    *,
    findings: list[dict[str, Any]],
    fp_count: int,
    total_findings: int,
    grounded: int,
    breakdown: bool = False,
    gold_texts: frozenset[str] | set[str] | None = None,
    weights: RubricV2Weights = DEFAULT_RUBRIC_WEIGHTS,
) -> float | RubricV2Breakdown:
    """Score one finished review under the Stage-0 rubric v2.

    Args:
        model: The trained outcome model (or any object exposing
            ``score_comment(text) -> float``).
        findings: Well-formed finding dicts (each a mapping with a string
            ``text``); malformed entries raise :class:`ValueError` naming the
            finding id.
        fp_count: Number of the findings judged false positives.
        total_findings: Total number of findings reported.
        grounded: Number of findings grounded in real code.
        breakdown: When ``True`` return the full :class:`RubricV2Breakdown`;
            otherwise return the composite scalar only.
        gold_texts: Optional set of finding texts known to overlap gold
            accepted evidence; drives the golden-overlap telemetry only.
        weights: The :class:`RubricV2Weights` to score under; defaults to
            :data:`DEFAULT_RUBRIC_WEIGHTS`.

    Returns:
        The composite ``[0, 1]`` scalar, or the frozen breakdown when
        ``breakdown=True``.
    """
    if not hasattr(model, _PROTOCOL_ATTR):
        raise TypeError(f"model must expose {_PROTOCOL_ATTR}(text) -> float; got {type(model).__name__!r}.")
    if not 0 <= fp_count <= total_findings:
        raise ValueError(
            f"fp_count must be in [0, total_findings] (got fp_count={fp_count!r}, total={total_findings!r})."
        )
    if not 0 <= grounded <= total_findings:
        raise ValueError(
            f"grounded must be in [0, total_findings] (got grounded={grounded!r}, total={total_findings!r})."
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

    # Localization term: grounded / total.
    localization: float | None = grounded / total_findings if total_findings > 0 else None

    # Tool-grounded term: present only when at least one finding reports
    # tool evidence; absent otherwise (None, never imputed 0.0).
    with_tools = [f for f in checked if f.get("tools")]
    tool_grounded: float | None = len(with_tools) / len(checked) if with_tools else None

    # Golden overlap: telemetry only (never a composite substitute, M6).
    gold = gold_texts or set()
    golden_overlap = len([f for f in checked if str(f["text"]) in gold]) / len(checked) if checked else 0.0

    # Intrinsic composite: read via score_trajectory, never recomputed here.
    verdicts = [{"verdict": str(f["verdict"])} for f in checked if f.get("verdict")]
    total_chars = sum(len(str(f["text"])) for f in checked)
    intrinsic = score_trajectory(
        ScoringInputs(
            verifier_verdicts=verdicts or None,
            grounding_rate=grounded / total_findings if total_findings else None,
            format_valid=True,
            length=total_chars or None,
        ),
        weights=DEFAULT_WEIGHTS,
    )
    intrinsic_composite = intrinsic.composite

    terms: dict[str, float | None] = {
        "learned_outcome": learned,
        "fp_penalty": -false_positive_penalty if false_positive_penalty is not None else None,
        "localization": localization,
        "tool_grounded": tool_grounded,
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
        localization=localization,
        tool_grounded=tool_grounded,
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
        "localization": "w_localization",
        "tool_grounded": "w_tool_grounded",
        "intrinsic_composite": "w_intrinsic",
    }
)
"""Composite-term name → :class:`RubricV2Weights` field name."""
