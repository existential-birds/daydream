"""Reward reducer for code-review trajectories.

This module scores a *single* run across both intrinsic (capture-time)
axes and the posterior false-positive axis derived from maintainer
accept/reject outcomes. It implements the reducer described in the
corpus-pipeline plan (Task 3), including the posterior axis introduced
alongside it; #88 owns empirical calibration of the defaults, behind
this same interface plus a :data:`REWARD_VERSION` bump.

Formula (golden-locked; full justification + citations in
``.beagle/concepts/corpus-pipeline-architecture/research/reward-formula-recommendation.md``):

Axes and their rules:

All weights and ramp parameters live on :class:`RewardWeights`; scoring under
:data:`DEFAULT_WEIGHTS` reproduces the golden-locked formula. The default
values cited below are :class:`RewardWeights` fields.

* **correctness** — mean over per-finding verifier verdicts mapped by
  :attr:`RewardWeights.verdict_map` (``consistent → 1.0``,
  ``uncertain → 0.5``, ``contradicts → 0.0``). A positive credit axis.
  Default weight :attr:`RewardWeights.w_correctness` ``= 0.6``
  (correctness-dominant). Source: arXiv:2509.15557 ``+1/0/−1`` ternary,
  rescaled to ``[0, 1]``.
* **grounding** — ``grounding_rate ∈ [0, 1]`` passed through unchanged. A
  positive credit axis. Default weight :attr:`RewardWeights.w_grounding`
  ``= 0.4`` (secondary guardrail). Source: HalluJudge reference-free
  grounding (F1 0.85).
* **format_valid** — a *dominating* gate, not an additive term. When
  ``False`` the composite floors to ``0.0`` regardless of every other axis.
  Source: arXiv:2509.15557 "improper format … maximum penalty regardless of
  correctness".
* **length** — a bounded saturating ramp over the char-count proxy:
  ``len_norm = clip((length − len_tau) / len_scale, 0, 1)`` subtracted after
  the credit mean with weight :attr:`RewardWeights.w_len` ``= 0.2`` (strictly
  smaller than every credit weight, so verbosity can shave but never
  dominate). Source: A-DLP / Leash bounded length penalty.
* **false_positive_penalty** — the posterior axis, derived from the
  maintainer accept/reject outcome (``rejected → 1.0``, ``contested → 0.5``,
  ``accepted → 0.0`` via :attr:`RewardWeights.fp_penalty_map`). The reported
  ``posterior_cost`` is the *calibrated surprise* ``max(0.0, observed − prior)``
  on the ``[0, 1]`` penalty scale: only the deviation above the reviewers'
  mean observed penalty (``outcome_prior``) is penalized, retaining the
  non-negative clamp. An uncalibrated prior falls back to the ``0.5``
  maximum-entropy midpoint. It is a *sibling* of the composite, carried on
  :class:`PosteriorBreakdown`, and is **never** subtracted inside the composite
  (C5: Safe-RLHF documents a safety-compensation pathology for fixed-weight
  subtractive scalarization of constraint-style signals — arXiv:2509.15557 is a
  subtractive composite, not a gate, and does not establish a "penalty < credit"
  rule). :attr:`RewardWeights.w_fp` is **not** applied here — it survives as a
  documented training-time combination weight (pending recalibration #114). The
  posterior is present only when a mapped maintainer label is supplied; absent
  at capture time and for ``"unknown"``/unmapped labels — then the result is a
  plain :class:`RewardBreakdown` with no posterior fields.

Composite = ``round(clip(credit − w_len·len_norm, 0, 1), 4)`` — a pure
intrinsic score. ``credit`` is the weighted mean over the *present* credit
axes only, renormalized so present weights sum to one
(``w_i' = w_i / Σ_present w_j``). A missing/empty/unparseable signal makes
that axis ``None`` and ``axes_present[axis] = False`` — never impute ``0.0``
for a missing axis, never raise. If no credit axis is present while
``format_valid`` is ``True``, the composite is ``None`` (uncomputable).
:attr:`RewardWeights.w_fp` is **not** applied here — it survives as a
documented training-time combination weight (pending recalibration #114).

Changing any default weight is a deliberate golden-update: it requires
re-pinning the golden test values *and* bumping :data:`REWARD_VERSION`.

:data:`REWARD_VERSION` fully identifies the formula *only* under
:data:`DEFAULT_WEIGHTS`. Passing a custom :class:`RewardWeights` is an
analysis-time override (e.g. sensitivity sweeps); its output is **not** the
canonical corpus reward and must not be stored as such — only scores produced
under :data:`DEFAULT_WEIGHTS` carry the meaning stamped by
:data:`REWARD_VERSION`.
"""

from __future__ import annotations

import hashlib
import json
import types
from dataclasses import dataclass, field
from typing import Any

REWARD_VERSION = "2026.05.28-1"
"""Bump on any change to axis weights, verdict map, gate, or composite shape.

Read at call time (not captured in a default argument) so a test can
monkeypatch ``daydream.training.reward.REWARD_VERSION`` and have
:func:`score_trajectory` observe the override.

Stamped verbatim on breakdowns scored under :data:`DEFAULT_WEIGHTS`. Scoring
under a custom :class:`RewardWeights` stamps a
``f"{REWARD_VERSION}+custom-{_weights_fingerprint(weights)}"`` suffix instead,
so a non-default (analysis-time override) score can never be mistaken for the
canonical corpus reward (OpenAI Evals / lm-eval-harness convention: a
scoring-config change forces a version bump).
"""

_VERDICT_MAP: dict[str, float] = {"consistent": 1.0, "uncertain": 0.5, "contradicts": 0.0}
"""Per-finding verdict → ``[0, 1]`` correctness sub-score (rescaled ternary)."""

_FP_PENALTY_MAP: dict[str, float] = {"accepted": 0.0, "contested": 0.5, "rejected": 1.0}
"""Maintainer outcome label → posterior false-positive penalty."""

FLOOR = 0.0
"""Composite floor — the ``[0, 1]`` range minimum, the format-gate override."""


@dataclass(frozen=True)
class RewardWeights:
    """Tunable weights and ramp parameters for :func:`score_trajectory`.

    Defaults reproduce the golden-locked formula exactly; overriding any
    field is an analysis-time choice, not a change to the canonical corpus
    reward (which is defined by :data:`REWARD_VERSION` under
    :data:`DEFAULT_WEIGHTS`).

    Attributes:
        w_correctness: Credit weight for the correctness axis
            (correctness-dominant).
        w_grounding: Credit weight for the grounding axis (secondary
            guardrail).
        w_len: Length-penalty weight; strictly smaller than every credit
            weight, so verbosity can shave but never dominate.
        w_fp: False-positive (posterior reject) penalty weight. **No longer
            applied inside the composite** (C5 made the posterior a sibling
            field, not a subtracted term); retained as a documented
            training-time combination weight pending recalibration (#114).
        len_tau: Length-ramp baseline (chars): no penalty at or below this
            proxy value.
        len_scale: Length-ramp scale (chars): the penalty saturates at
            ``len_tau + len_scale``.
        verdict_map: Per-finding verdict → ``[0, 1]`` correctness sub-score.
        fp_penalty_map: Maintainer outcome label → posterior penalty
            (``accepted → 0.0``, ``contested → 0.5``, ``rejected → 1.0``).
            An unmapped/``"unknown"`` label leaves the axis absent.
        is_default: Identity flag, ``True`` only on :data:`DEFAULT_WEIGHTS`.
            Set at construction, never mutated; excluded from
            :func:`_weights_fingerprint` (it is identity metadata, not a
            scoring parameter).
    """

    w_correctness: float = 0.6
    w_grounding: float = 0.4
    w_len: float = 0.2
    w_fp: float = 0.3
    len_tau: float = 2000.0
    len_scale: float = 8000.0
    verdict_map: types.MappingProxyType = field(
        default_factory=lambda: types.MappingProxyType(dict(_VERDICT_MAP))
    )
    fp_penalty_map: types.MappingProxyType = field(
        default_factory=lambda: types.MappingProxyType(dict(_FP_PENALTY_MAP))
    )
    is_default: bool = False

    def __post_init__(self) -> None:
        if self.len_scale <= 0:
            raise ValueError(
                f"len_scale must be > 0 (got {self.len_scale!r}); "
                "a zero or negative value causes ZeroDivisionError at the length-ramp computation."
            )


DEFAULT_WEIGHTS = RewardWeights(is_default=True)
"""The golden-locked weights; scoring under these is byte-identical to the
canonical corpus reward stamped by :data:`REWARD_VERSION`. The only
:class:`RewardWeights` instance with ``is_default=True``."""


def _weights_fingerprint(weights: RewardWeights) -> str:
    """Return a stable 8-char fingerprint of a :class:`RewardWeights`.

    Serializes the six scalar fields plus the two map fields (as plain
    ``dict``) via sorted-key JSON, then takes the leading 8 hex chars of the
    SHA-256 digest. ``is_default`` is excluded — it is identity metadata, not
    a scoring parameter. Pure; no I/O.

    Args:
        weights: The weights to fingerprint.

    Returns:
        The first 8 hex characters of the SHA-256 digest of the canonical
        JSON serialization of the scoring parameters.
    """
    payload = {
        "w_correctness": weights.w_correctness,
        "w_grounding": weights.w_grounding,
        "w_len": weights.w_len,
        "w_fp": weights.w_fp,
        "len_tau": weights.len_tau,
        "len_scale": weights.len_scale,
        "verdict_map": dict(weights.verdict_map),
        "fp_penalty_map": dict(weights.fp_penalty_map),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:8]


def _clip(value: float, low: float, high: float) -> float:
    """Clamp ``value`` to the closed interval ``[low, high]``."""
    return max(low, min(high, value))


@dataclass(frozen=True)
class ScoringInputs:
    """Intrinsic, capture-time signals for one trajectory.

    Attributes:
        verifier_verdicts: Per-finding verifier verdict records (each a
            dict with a ``"verdict"`` key), or ``None`` when the run has no
            structured verdicts (e.g. a shallow run).
        grounding_rate: Fraction of findings grounded in real code, in
            ``[0, 1]``, or ``None`` when unavailable.
        format_valid: Whether the structured bronze artifacts parsed
            cleanly. ``False`` floors the composite (dominating gate).
        length: Char-count length proxy, or ``None`` when absent.
    """

    verifier_verdicts: list[dict[str, Any]] | None
    grounding_rate: float | None
    format_valid: bool
    length: int | None


@dataclass(frozen=True)
class RewardBreakdown:
    """Per-axis decomposition + composite for one *intrinsic-only* trajectory.

    Represents a row with no maintainer outcome label. The posterior
    false-positive axis lives on the :class:`PosteriorBreakdown` subclass, not
    here — keeping the two populations type-separated (C3).

    Attributes:
        correctness_per_finding: Mapped verdict scores per finding, or
            ``None`` when the correctness axis is absent.
        grounding: Grounding rate passed through, or ``None`` when absent.
        format_valid: The dominating format gate flag.
        length_penalty: Bounded length ramp ``len_norm ∈ [0, 1]``, or
            ``None`` when no length proxy was available.
        composite: Pure-intrinsic ``[0, 1]`` score (``round(..., 4)``),
            ``0.0`` when format-invalid, or ``None`` when uncomputable (no
            present credit axis while format-valid).
        axes_present: Per-axis presence flags (``correctness``,
            ``grounding``, ``length``).
        reward_version: The :data:`REWARD_VERSION` stamped at scoring time.
    """

    correctness_per_finding: list[float] | None
    grounding: float | None
    format_valid: bool
    length_penalty: float | None
    composite: float | None
    axes_present: dict[str, bool]
    reward_version: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation with explicit key order."""
        return {
            "correctness_per_finding": (
                list(self.correctness_per_finding) if self.correctness_per_finding is not None else None
            ),
            "grounding": self.grounding,
            "format_valid": self.format_valid,
            "length_penalty": self.length_penalty,
            "composite": self.composite,
            "axes_present": dict(self.axes_present),
            "reward_version": self.reward_version,
        }


@dataclass(frozen=True)
class PosteriorBreakdown(RewardBreakdown):
    """Intrinsic breakdown extended with the posterior false-positive axis.

    Produced by :func:`score_trajectory` only when a mapped maintainer outcome
    label is supplied. The ``composite`` it inherits is the pure intrinsic
    score (C5: the posterior is a sibling, never folded in). The presence of
    ``posterior_cost`` in :meth:`to_dict` is the population discriminator for
    downstream consumers splitting labeled from unlabeled rows.

    Attributes:
        false_positive_penalty: Raw observed maintainer outcome mapped via
            :attr:`RewardWeights.fp_penalty_map` (``accepted → 0.0``,
            ``contested → 0.5``, ``rejected → 1.0``).
        posterior_cost: The surprise component ``max(0.0, observed − prior)``
            on the ``[0, 1]`` penalty scale — only the deviation above the
            reviewers' prior is penalized.
        outcome_prior: The reviewers' mean observed penalty used as the prior,
            or ``None`` when uncalibrated (the reducer then applies the ``0.5``
            maximum-entropy default).
        outcome_prior_n: The pooled count of prior outcomes behind
            ``outcome_prior`` (for audit; recorded regardless of threshold).
    """

    false_positive_penalty: float
    posterior_cost: float
    outcome_prior: float | None
    outcome_prior_n: int

    def to_dict(self) -> dict[str, Any]:
        """Return the intrinsic dict extended with the four posterior keys."""
        base = super().to_dict()
        base["false_positive_penalty"] = self.false_positive_penalty
        base["posterior_cost"] = self.posterior_cost
        base["outcome_prior"] = self.outcome_prior
        base["outcome_prior_n"] = self.outcome_prior_n
        return base


def score_trajectory(
    inputs: ScoringInputs,
    *,
    pr_feedback: Any | None = None,
    outcome_prior: float | None = None,
    outcome_prior_n: int = 0,
    weights: RewardWeights = DEFAULT_WEIGHTS,
) -> RewardBreakdown | PosteriorBreakdown:
    """Reduce intrinsic (+ optional posterior) signals to a breakdown.

    Pure: no filesystem, network, or subprocess access; identical inputs
    yield identical output. The ``composite`` is always a pure intrinsic
    score (correctness + grounding − length penalty); the posterior
    false-positive axis is a *sibling* field, never folded in (C5).

    ``pr_feedback`` carries the maintainer outcome label. When it maps to a
    measured penalty via :attr:`RewardWeights.fp_penalty_map`, the result is a
    :class:`PosteriorBreakdown` carrying the posterior fields. Otherwise
    (``None``/``"unknown"``/unmapped) the result is a plain
    :class:`RewardBreakdown` — the composite is identical either way.

    Args:
        inputs: The intrinsic, capture-time signals to score.
        pr_feedback: Maintainer outcome label (``accepted``/``contested``/
            ``rejected``); mapped via :attr:`RewardWeights.fp_penalty_map`.
            ``None``/``"unknown"``/unmapped yields a base
            :class:`RewardBreakdown`.
        outcome_prior: The reviewers' mean observed penalty on the ``[0, 1]``
            penalty scale, used as the prior the posterior surprise is measured
            against. ``None`` (uncalibrated) falls back to the ``0.5``
            maximum-entropy default for the cost, and is stored verbatim on the
            breakdown as the audit trail of calibration status. Meaningful only
            on the mapped-label (:class:`PosteriorBreakdown`) path.
        outcome_prior_n: The pooled count of prior outcomes behind
            ``outcome_prior`` (audit only; stored verbatim). Meaningful only on
            the mapped-label path.
        weights: The :class:`RewardWeights` to score under; defaults to
            :data:`DEFAULT_WEIGHTS` (the golden-locked, canonical weights).

    Returns:
        A frozen :class:`PosteriorBreakdown` when ``pr_feedback`` maps to a
        penalty, else a frozen :class:`RewardBreakdown`. Both are stamped with
        :data:`REWARD_VERSION` (read at call time); ``composite`` is ``0.0``
        when ``format_valid`` is ``False``, ``None`` when no credit axis is
        present, else the rounded ``[0, 1]`` pure-intrinsic composite.
    """
    version = REWARD_VERSION if weights.is_default else f"{REWARD_VERSION}+custom-{_weights_fingerprint(weights)}"

    # Correctness axis: present only when verdicts parse to a non-empty list.
    correctness_per_finding: list[float] | None = None
    correctness: float | None = None
    if inputs.verifier_verdicts:
        scores = [weights.verdict_map.get(str(v.get("verdict")), 0.0) for v in inputs.verifier_verdicts]
        correctness_per_finding = scores
        correctness = sum(scores) / len(scores)

    # Grounding axis: present only when a rate was supplied.
    grounding = inputs.grounding_rate

    # Length penalty: bounded ramp; absent when no length proxy.
    length_penalty: float | None = None
    if inputs.length is not None:
        length_penalty = _clip((inputs.length - weights.len_tau) / weights.len_scale, 0.0, 1.0)

    # Posterior false-positive penalty: present only when the maintainer
    # outcome label maps to a measured penalty. Absent/"unknown"/unmapped ⇒
    # a plain RewardBreakdown — never impute 0.0 as if measured, never raise.
    fp_penalty: float | None = None
    if pr_feedback is not None:
        fp_penalty = weights.fp_penalty_map.get(str(pr_feedback))

    axes_present = {
        "correctness": correctness is not None,
        "grounding": grounding is not None,
        "length": length_penalty is not None,
    }

    # Pure-intrinsic composite (posterior is a sibling, not subtracted here).
    composite: float | None
    if not inputs.format_valid:
        # Format gate dominates everything below it.
        composite = FLOOR
    else:
        # Weighted credit mean, renormalized over PRESENT credit axes only.
        present_weights: dict[str, float] = {}
        present_values: dict[str, float] = {}
        if correctness is not None:
            present_weights["correctness"] = weights.w_correctness
            present_values["correctness"] = correctness
        if grounding is not None:
            present_weights["grounding"] = weights.w_grounding
            present_values["grounding"] = grounding

        if not present_weights:
            # No present credit axis while format-valid ⇒ uncomputable.
            composite = None
        else:
            weight_sum = sum(present_weights.values())
            if weight_sum <= 0:
                raise ValueError(
                    "Invalid RewardWeights for present credit axes: "
                    f"sum of present credit weights must be > 0 (got {weight_sum!r})."
                )
            credit = sum((w / weight_sum) * present_values[axis] for axis, w in present_weights.items())
            ramp = length_penalty if length_penalty is not None else 0.0
            composite = round(_clip(credit - weights.w_len * ramp, FLOOR, 1.0), 4)

    # Mapped maintainer label ⇒ PosteriorBreakdown carrying the sibling axis.
    if fp_penalty is not None:
        # Calibrated surprise: penalize only the deviation above the reviewers'
        # prior. An uncalibrated (None) prior falls back to the 0.5 max-entropy
        # midpoint for the cost, but is stored verbatim as the audit trail.
        effective_prior = outcome_prior if outcome_prior is not None else 0.5
        posterior_cost = max(0.0, fp_penalty - effective_prior)
        return PosteriorBreakdown(
            correctness_per_finding=correctness_per_finding,
            grounding=grounding,
            format_valid=inputs.format_valid,
            length_penalty=length_penalty,
            composite=composite,
            axes_present={**axes_present, "false_positive": True},
            reward_version=version,
            false_positive_penalty=fp_penalty,
            posterior_cost=posterior_cost,
            outcome_prior=outcome_prior,
            outcome_prior_n=outcome_prior_n,
        )

    return RewardBreakdown(
        correctness_per_finding=correctness_per_finding,
        grounding=grounding,
        format_valid=inputs.format_valid,
        length_penalty=length_penalty,
        composite=composite,
        axes_present=axes_present,
        reward_version=version,
    )
