"""Pure intrinsic (capture-time) reward reducer for code-review trajectories.

This module scores a *single* run from intrinsic axes only — the signals
that are observable at capture time, before any posterior accept/reject
outcome is known. It is the minimal reducer described in the corpus-pipeline
plan (Task 3); #88 owns empirical calibration of the defaults and the
posterior axis, behind this same interface plus a :data:`REWARD_VERSION` bump.

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
  ``accepted → 0.0`` via :attr:`RewardWeights.fp_penalty_map`) and subtracted
  after the credit mean with weight :attr:`RewardWeights.w_fp` ``= 0.3``.
  ``0.3`` follows the May-2026 report's published penalty-ordering
  ``w_a=0.5 > w_s=0.3`` (arXiv:2509.15557, ``w_b=1.0 > w_a=0.5 > w_s=0.3``):
  a penalty term strictly smaller than every credit weight. It sits strictly
  below the grounding weight (``0.4``), so a rejected outcome deducts but
  never swamps a genuinely good review (KD2 drown-out guard). Absent at
  capture time and for ``"unknown"``/unmapped labels — then the axis stays
  ``None`` and the composite is byte-identical to the intrinsic-only score.

Composite = ``round(clip(credit − w_len·len_norm − w_fp·fp_penalty, 0, 1), 4)``
where ``fp_penalty`` is ``0.0`` when the posterior axis is absent and
``credit`` is the weighted mean over the *present* credit axes only,
renormalized so present weights sum to one
(``w_i' = w_i / Σ_present w_j``). A missing/empty/unparseable signal makes
that axis ``None`` and ``axes_present[axis] = False`` — never impute ``0.0``
for a missing axis, never raise. If no credit axis is present while
``format_valid`` is ``True``, the composite is ``None`` (uncomputable).

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

from dataclasses import dataclass, field
from typing import Any

REWARD_VERSION = "2026.05.27-1"
"""Bump on any change to axis weights, verdict map, gate, or composite shape.

Read at call time (not captured in a default argument) so a test can
monkeypatch ``daydream.training.reward.REWARD_VERSION`` and have
:func:`score_trajectory` observe the override.
"""

VERDICT_MAP: dict[str, float] = {"consistent": 1.0, "uncertain": 0.5, "contradicts": 0.0}
"""Per-finding verdict → ``[0, 1]`` correctness sub-score (rescaled ternary)."""

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
        w_fp: False-positive (posterior reject) penalty weight; sits strictly
            below the grounding weight so a rejected outcome deducts but never
            swamps a genuinely good review (KD2 drown-out guard).
        len_tau: Length-ramp baseline (chars): no penalty at or below this
            proxy value.
        len_scale: Length-ramp scale (chars): the penalty saturates at
            ``len_tau + len_scale``.
        verdict_map: Per-finding verdict → ``[0, 1]`` correctness sub-score.
        fp_penalty_map: Maintainer outcome label → posterior penalty
            (``accepted → 0.0``, ``contested → 0.5``, ``rejected → 1.0``).
            An unmapped/``"unknown"`` label leaves the axis absent.
    """

    w_correctness: float = 0.6
    w_grounding: float = 0.4
    w_len: float = 0.2
    w_fp: float = 0.3
    len_tau: float = 2000.0
    len_scale: float = 8000.0
    verdict_map: dict[str, float] = field(default_factory=lambda: dict(VERDICT_MAP))
    fp_penalty_map: dict[str, float] = field(
        default_factory=lambda: {"accepted": 0.0, "contested": 0.5, "rejected": 1.0}
    )


DEFAULT_WEIGHTS = RewardWeights()
"""The golden-locked weights; scoring under these is byte-identical to the
canonical corpus reward stamped by :data:`REWARD_VERSION`."""


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
    """Per-axis decomposition + composite for one trajectory.

    Attributes:
        correctness_per_finding: Mapped verdict scores per finding, or
            ``None`` when the correctness axis is absent.
        grounding: Grounding rate passed through, or ``None`` when absent.
        format_valid: The dominating format gate flag.
        false_positive_penalty: Posterior axis — the maintainer outcome
            mapped via :attr:`RewardWeights.fp_penalty_map`
            (``accepted → 0.0``, ``contested → 0.5``, ``rejected → 1.0``), or
            ``None`` when no outcome label was supplied or it was
            absent/``"unknown"``/unmapped (never imputed as ``0.0``).
        length_penalty: Bounded length ramp ``len_norm ∈ [0, 1]``, or
            ``None`` when no length proxy was available.
        composite: Final ``[0, 1]`` score (``round(..., 4)``), ``0.0`` when
            format-invalid, or ``None`` when uncomputable (no present credit
            axis while format-valid).
        axes_present: Per-axis presence flags (``correctness``,
            ``grounding``, ``length``, ``false_positive``).
        reward_version: The :data:`REWARD_VERSION` stamped at scoring time.
    """

    correctness_per_finding: list[float] | None
    grounding: float | None
    format_valid: bool
    false_positive_penalty: float | None
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
            "false_positive_penalty": self.false_positive_penalty,
            "length_penalty": self.length_penalty,
            "composite": self.composite,
            "axes_present": dict(self.axes_present),
            "reward_version": self.reward_version,
        }


def score_trajectory(
    inputs: ScoringInputs,
    *,
    pr_feedback: Any | None = None,
    weights: RewardWeights = DEFAULT_WEIGHTS,
) -> RewardBreakdown:
    """Reduce intrinsic + posterior signals to a :class:`RewardBreakdown`.

    Pure: no filesystem, network, or subprocess access; identical inputs
    yield identical output. ``pr_feedback`` carries the maintainer outcome
    label; when present and mapped it populates the posterior
    false-positive axis, otherwise that axis stays ``None`` and the
    composite is byte-identical to the intrinsic-only score.

    Args:
        inputs: The intrinsic, capture-time signals to score.
        pr_feedback: Maintainer outcome label (``accepted``/``contested``/
            ``rejected``); mapped via :attr:`RewardWeights.fp_penalty_map`.
            ``None``/``"unknown"``/unmapped leaves the posterior axis absent.
        weights: The :class:`RewardWeights` to score under; defaults to
            :data:`DEFAULT_WEIGHTS` (the golden-locked, canonical weights).

    Returns:
        A frozen :class:`RewardBreakdown` stamped with :data:`REWARD_VERSION`
        (read at call time). ``composite`` is ``0.0`` when ``format_valid``
        is ``False``, ``None`` when no credit axis is present, else the
        rounded ``[0, 1]`` composite.
    """
    version = REWARD_VERSION

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
    # None — never impute 0.0 as if measured, never raise.
    fp_penalty: float | None = None
    if pr_feedback is not None:
        fp_penalty = weights.fp_penalty_map.get(str(pr_feedback))

    axes_present = {
        "correctness": correctness is not None,
        "grounding": grounding is not None,
        "length": length_penalty is not None,
        "false_positive": fp_penalty is not None,
    }

    # Format gate dominates everything below it.
    if not inputs.format_valid:
        return RewardBreakdown(
            correctness_per_finding=correctness_per_finding,
            grounding=grounding,
            format_valid=False,
            false_positive_penalty=fp_penalty,
            length_penalty=length_penalty,
            composite=FLOOR,
            axes_present=axes_present,
            reward_version=version,
        )

    # Weighted credit mean, renormalized over PRESENT credit axes only.
    present_weights: dict[str, float] = {}
    present_values: dict[str, float] = {}
    if correctness is not None:
        present_weights["correctness"] = weights.w_correctness
        present_values["correctness"] = correctness
    if grounding is not None:
        present_weights["grounding"] = weights.w_grounding
        present_values["grounding"] = grounding

    composite: float | None
    if not present_weights:
        # No present credit axis while format-valid ⇒ uncomputable.
        composite = None
    else:
        weight_sum = sum(present_weights.values())
        credit = sum((w / weight_sum) * present_values[axis] for axis, w in present_weights.items())
        ramp = length_penalty if length_penalty is not None else 0.0
        fp = fp_penalty if fp_penalty is not None else 0.0
        composite = round(_clip(credit - weights.w_len * ramp - weights.w_fp * fp, FLOOR, 1.0), 4)

    return RewardBreakdown(
        correctness_per_finding=correctness_per_finding,
        grounding=grounding,
        format_valid=True,
        false_positive_penalty=fp_penalty,
        length_penalty=length_penalty,
        composite=composite,
        axes_present=axes_present,
        reward_version=version,
    )
