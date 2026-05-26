"""Pure intrinsic (capture-time) reward reducer for code-review trajectories.

This module scores a *single* run from intrinsic axes only — the signals
that are observable at capture time, before any posterior accept/reject
outcome is known. It is the minimal reducer described in the corpus-pipeline
plan (Task 3); #88 owns empirical calibration of the defaults and the
posterior axis, behind this same interface plus a :data:`REWARD_VERSION` bump.

Formula (golden-locked; full justification + citations in
``.beagle/concepts/corpus-pipeline-architecture/research/reward-formula-recommendation.md``):

Axes and their rules:

* **correctness** — mean over per-finding verifier verdicts mapped by
  :data:`VERDICT_MAP` (``consistent → 1.0``, ``uncertain → 0.5``,
  ``contradicts → 0.0``). A positive credit axis. Default weight
  ``w_correctness = 0.6`` (correctness-dominant). Source: arXiv:2509.15557
  ``+1/0/−1`` ternary, rescaled to ``[0, 1]``.
* **grounding** — ``grounding_rate ∈ [0, 1]`` passed through unchanged. A
  positive credit axis. Default weight ``w_grounding = 0.4`` (secondary
  guardrail). Source: HalluJudge reference-free grounding (F1 0.85).
* **format_valid** — a *dominating* gate, not an additive term. When
  ``False`` the composite floors to ``0.0`` regardless of every other axis.
  Source: arXiv:2509.15557 "improper format … maximum penalty regardless of
  correctness".
* **length** — a bounded saturating ramp over the char-count proxy:
  ``len_norm = clip((length − TAU) / SCALE, 0, 1)`` subtracted after the
  credit mean with weight ``w_len = 0.2`` (strictly smaller than every
  credit weight, so verbosity can shave but never dominate). Source:
  A-DLP / Leash bounded length penalty.
* **false_positive_penalty** — the posterior axis. Structurally absent at
  capture time; always ``None`` in this minimal reducer.

Composite = ``round(clip(credit − w_len·len_norm, 0, 1), 4)`` where
``credit`` is the weighted mean over the *present* credit axes only,
renormalized so present weights sum to one
(``w_i' = w_i / Σ_present w_j``). A missing/empty/unparseable signal makes
that axis ``None`` and ``axes_present[axis] = False`` — never impute ``0.0``
for a missing axis, never raise. If no credit axis is present while
``format_valid`` is ``True``, the composite is ``None`` (uncomputable).

Changing any default weight is a deliberate golden-update: it requires
re-pinning the golden test values *and* bumping :data:`REWARD_VERSION`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

REWARD_VERSION = "2026.05.25-1"
"""Bump on any change to axis weights, verdict map, gate, or composite shape.

Read at call time (not captured in a default argument) so a test can
monkeypatch ``daydream.training.reward.REWARD_VERSION`` and have
:func:`score_trajectory` observe the override.
"""

VERDICT_MAP: dict[str, float] = {"consistent": 1.0, "uncertain": 0.5, "contradicts": 0.0}
"""Per-finding verdict → ``[0, 1]`` correctness sub-score (rescaled ternary)."""

W_CORRECTNESS = 0.6
"""Default credit weight for the correctness axis (correctness-dominant)."""

W_GROUNDING = 0.4
"""Default credit weight for the grounding axis (secondary guardrail)."""

W_LEN = 0.2
"""Default length-penalty weight; strictly smaller than every credit weight."""

LEN_TAU = 2000.0
"""Length-ramp baseline (chars): no penalty at or below this proxy value."""

LEN_SCALE = 8000.0
"""Length-ramp scale (chars): the penalty saturates at ``TAU + SCALE``."""

FLOOR = 0.0
"""Composite floor — the ``[0, 1]`` range minimum, the format-gate override."""


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
        false_positive_penalty: Posterior axis — always ``None`` in the
            minimal intrinsic reducer.
        length_penalty: Bounded length ramp ``len_norm ∈ [0, 1]``, or
            ``None`` when no length proxy was available.
        composite: Final ``[0, 1]`` score (``round(..., 4)``), ``0.0`` when
            format-invalid, or ``None`` when uncomputable (no present credit
            axis while format-valid).
        axes_present: Per-axis presence flags (``correctness``,
            ``grounding``, ``length``).
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


def score_trajectory(inputs: ScoringInputs, *, pr_feedback: Any | None = None) -> RewardBreakdown:
    """Reduce intrinsic signals to a :class:`RewardBreakdown`.

    Pure: no filesystem, network, or subprocess access; identical inputs
    yield identical output. ``pr_feedback`` is accepted but unused in the
    minimal reducer (posterior axes stay ``None``); it reserves the
    signature for #88's posterior axis.

    Args:
        inputs: The intrinsic, capture-time signals to score.
        pr_feedback: Reserved posterior signal; unused here.

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
        scores = [VERDICT_MAP.get(str(v.get("verdict")), 0.0) for v in inputs.verifier_verdicts]
        correctness_per_finding = scores
        correctness = sum(scores) / len(scores)

    # Grounding axis: present only when a rate was supplied.
    grounding = inputs.grounding_rate

    # Length penalty: bounded ramp; absent when no length proxy.
    length_penalty: float | None = None
    if inputs.length is not None:
        length_penalty = _clip((inputs.length - LEN_TAU) / LEN_SCALE, 0.0, 1.0)

    axes_present = {
        "correctness": correctness is not None,
        "grounding": grounding is not None,
        "length": length_penalty is not None,
    }

    # Format gate dominates everything below it.
    if not inputs.format_valid:
        return RewardBreakdown(
            correctness_per_finding=correctness_per_finding,
            grounding=grounding,
            format_valid=False,
            false_positive_penalty=None,
            length_penalty=length_penalty,
            composite=FLOOR,
            axes_present=axes_present,
            reward_version=version,
        )

    # Weighted credit mean, renormalized over PRESENT credit axes only.
    present_weights: dict[str, float] = {}
    present_values: dict[str, float] = {}
    if correctness is not None:
        present_weights["correctness"] = W_CORRECTNESS
        present_values["correctness"] = correctness
    if grounding is not None:
        present_weights["grounding"] = W_GROUNDING
        present_values["grounding"] = grounding

    composite: float | None
    if not present_weights:
        # No present credit axis while format-valid ⇒ uncomputable.
        composite = None
    else:
        weight_sum = sum(present_weights.values())
        credit = sum((w / weight_sum) * present_values[axis] for axis, w in present_weights.items())
        ramp = length_penalty if length_penalty is not None else 0.0
        composite = round(_clip(credit - W_LEN * ramp, FLOOR, 1.0), 4)

    return RewardBreakdown(
        correctness_per_finding=correctness_per_finding,
        grounding=grounding,
        format_valid=True,
        false_positive_penalty=None,
        length_penalty=length_penalty,
        composite=composite,
        axes_present=axes_present,
        reward_version=version,
    )
