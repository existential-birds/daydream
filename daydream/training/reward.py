"""Score intrinsic correctness/length and a separate maintainer-outcome penalty.

Composite = round(clip(correctness - w_len * len_norm, 0, 1), 4).
Missing/empty verifier evidence leaves correctness absent and a valid-format
composite uncomputable. PosteriorBreakdown carries posterior cost beside the
intrinsic composite, never subtracting it.

Changing default weights or label meanings requires golden updates and a
REWARD_VERSION bump. Only DEFAULT_WEIGHTS identifies the canonical corpus
formula; custom weight outputs are analysis overrides.
"""

from __future__ import annotations

import hashlib
import json
import types
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

REWARD_VERSION = "2026.10.01-1"
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

FP_PENALTY_MAP: dict[str, float] = {"accepted": 0.0, "contested": 0.5, "rejected": 1.0}
"""Maintainer outcome label → posterior false-positive penalty; also the archive reviewer-prior scale."""

FLOOR = 0.0
"""Composite floor — the ``[0, 1]`` range minimum, the format-gate override."""


@dataclass(frozen=True)
class RewardWeights:
    """Canonical reward weights and length-ramp parameters.

    Length penalty rises from zero at len_tau to one at len_tau + len_scale.
    w_fp is reserved for training-time combination; it never changes the
    intrinsic composite. Unmapped feedback leaves the posterior axis absent.
    Only DEFAULT_WEIGHTS earns the canonical version stamp."""

    w_len: float = 0.2
    w_fp: float = 0.3
    len_tau: float = 2000.0
    len_scale: float = 8000.0
    verdict_map: types.MappingProxyType[str, float] = field(
        default_factory=lambda: types.MappingProxyType(dict(_VERDICT_MAP))
    )
    fp_penalty_map: types.MappingProxyType[str, float] = field(
        default_factory=lambda: types.MappingProxyType(dict(FP_PENALTY_MAP))
    )

    def __post_init__(self) -> None:
        if self.len_scale <= 0:
            raise ValueError(
                f"len_scale must be > 0 (got {self.len_scale!r}); "
                "a zero or negative value causes ZeroDivisionError at the length-ramp computation."
            )


DEFAULT_WEIGHTS = RewardWeights()
"""The golden-locked weights; scoring under these is byte-identical to the
canonical corpus reward stamped by :data:`REWARD_VERSION`. Only this instance
earns the canonical stamp, keyed by object identity in
:func:`score_trajectory`."""


def _weights_fingerprint(weights: RewardWeights) -> str:
    """Hash sorted-key JSON of the scoring parameters to eight hexadecimal characters."""
    payload = {
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
    """Capture-time signals: optional structured verifier verdicts and character count.

    format_valid=False dominates every credit axis and floors the composite."""

    verifier_verdicts: Sequence[Mapping[str, Any]] | None
    format_valid: bool
    length: int | None


@dataclass(frozen=True)
class RewardBreakdown:
    """Intrinsic score and axis presence for one trajectory.

    Missing verdicts leave correctness absent; missing length leaves its penalty
    absent. The composite is rounded to four places in [0, 1], zero for invalid
    format, or None when valid format has no correctness credit. Posterior
    fields exist only on PosteriorBreakdown."""

    correctness_per_finding: list[float] | None
    format_valid: bool
    length_penalty: float | None
    composite: float | None
    axes_present: dict[str, bool]
    reward_version: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation with explicit key order."""
        return asdict(self)


@dataclass(frozen=True)
class PosteriorBreakdown(RewardBreakdown):
    """Intrinsic score plus a separate observed maintainer penalty.

    posterior_cost is abs(observed_penalty - prior), penalizing deviation in
    both directions. A missing prior uses 0.5 for cost while preserving None and
    outcome_prior_n for audit. Presence of posterior_cost distinguishes labeled
    rows; it never changes the inherited intrinsic composite."""

    false_positive_penalty: float
    posterior_cost: float
    outcome_prior: float | None
    outcome_prior_n: int



def score_trajectory(
    inputs: ScoringInputs,
    *,
    pr_feedback: Any | None = None,
    outcome_prior: float | None = None,
    outcome_prior_n: int = 0,
    weights: RewardWeights = DEFAULT_WEIGHTS,
) -> RewardBreakdown | PosteriorBreakdown:
    """Reduce capture-time signals, optionally adding a separate posterior penalty.

    The format gate dominates; missing correctness makes a valid-format composite
    uncomputable. A mapped pr_feedback label returns PosteriorBreakdown with
    absolute deviation from the supplied prior (0.5 when absent). Unmapped
    feedback returns RewardBreakdown without imputing a measured penalty.

    Read REWARD_VERSION at call time. Only the DEFAULT_WEIGHTS instance receives
    its canonical stamp; all other weights receive a custom fingerprint."""
    version = (
        REWARD_VERSION
        if weights is DEFAULT_WEIGHTS
        else f"{REWARD_VERSION}+custom-{_weights_fingerprint(weights)}"
    )

    # Correctness axis: present only when verdicts parse to a non-empty list.
    correctness_per_finding: list[float] | None = None
    correctness: float | None = None
    if inputs.verifier_verdicts:
        scores = [weights.verdict_map.get(str(v.get("verdict")), 0.0) for v in inputs.verifier_verdicts]
        correctness_per_finding = scores
        correctness = sum(scores) / len(scores)

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
        "length": length_penalty is not None,
    }

    # Pure-intrinsic composite (posterior is a sibling, not subtracted here).
    composite: float | None
    if not inputs.format_valid:
        # Format gate dominates everything below it.
        composite = FLOOR
    elif correctness is None:
        composite = None
    else:
        ramp = length_penalty if length_penalty is not None else 0.0
        composite = round(_clip(correctness - weights.w_len * ramp, FLOOR, 1.0), 4)

    # Mapped maintainer label ⇒ PosteriorBreakdown carrying the sibling axis.
    if fp_penalty is not None:
        # Calibrated surprise: penalize the absolute deviation from the reviewers'
        # prior (both under- and over-rejection). An uncalibrated (None) prior
        # falls back to the 0.5 max-entropy midpoint; stored verbatim as audit trail.
        effective_prior = outcome_prior if outcome_prior is not None else 0.5
        posterior_cost = abs(fp_penalty - effective_prior)
        return PosteriorBreakdown(
            correctness_per_finding=correctness_per_finding,
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
        format_valid=inputs.format_valid,
        length_penalty=length_penalty,
        composite=composite,
        axes_present=axes_present,
        reward_version=version,
    )
