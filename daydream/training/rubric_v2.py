"""Stage-0 rubric v2 (M2, M5, M6, M7): learned outcome term + CR-Bench FP penalty.

Composes, never rewrites, the shipped intrinsic composite from
:mod:`daydream.training.reward`. The rubric adds two components the intrinsic
composite does not have:

- a **learned outcome term** from the Stage-0 two-class model
  (:func:`daydream.training.reward_model.score_comment`) — the model's
  ``[0, 1]`` probability that a finished comment reads as gold-accepted
  review prose rather than noise;
- an explicit **false-positive penalty** with CR-Bench Usefulness Rate
  semantics (``(total − fp)/total``; KD3): the penalty magnitude is
  ``fp/total`` and it enters the composite subtractively, so a run whose
  findings are all noise can never outrank a clean run at the same recall
  (M5).

The intrinsic composite and the golden-overlap telemetry are carried as
**signals only**: neither can substitute for the learned outcome term or the
Stage-0 gate (M6). Missing signals are ``None`` and renormalized out of the
composite — never imputed ``0.0`` (mirrors ``reward.py``'s ``axes_present``
rule).

Version discipline (M7 / PATTERN golden-update): :data:`REWARD_VERSION_RUBRIC`
stamps every breakdown; scoring under a non-default :class:`RubricV2Weights`
appends a ``+custom-{fingerprint}`` suffix so an analysis-time override can
never be mistaken for the canonical rubric score. Changing any default weight
is a deliberate golden-update: re-pin the golden values and bump
:data:`REWARD_VERSION_RUBRIC`.

Pure module: no filesystem, network, or subprocess access.
"""

from __future__ import annotations

import hashlib
import json
import types
from dataclasses import dataclass
from typing import Any

from daydream.training.reward import DEFAULT_WEIGHTS, ScoringInputs, score_trajectory

REWARD_VERSION_RUBRIC = "2026.05.28-rubric-1"
"""Bump on any change to rubric weights, penalty semantics, or composite shape.

Read at call time so a test can monkeypatch
``daydream.training.rubric_v2.REWARD_VERSION_RUBRIC`` and have
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
