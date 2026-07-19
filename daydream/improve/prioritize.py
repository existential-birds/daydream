"""Prioritize vetted improve-flow findings."""

from __future__ import annotations

from typing import Any

_IMPACT = {"HIGH": 3.0, "MED": 2.0, "LOW": 1.0}
_EFFORT = {"S": 1.0, "M": 2.0, "L": 3.0}
_CONFIDENCE = {"HIGH": 1.0, "MED": 0.7, "LOW": 0.4}
_RISK = {"LOW": 1.0, "MED": 0.8, "HIGH": 0.6}


def leverage_score(finding: dict[str, Any]) -> float:
    """Return impact over effort, discounted by confidence and fix risk."""
    impact = _axis_value(_IMPACT, finding.get("impact"), min(_IMPACT.values()))
    effort = _axis_value(_EFFORT, finding.get("effort"), max(_EFFORT.values()))
    confidence = _axis_value(
        _CONFIDENCE, finding.get("confidence"), min(_CONFIDENCE.values())
    )
    risk = _axis_value(_RISK, finding.get("risk"), min(_RISK.values()))
    return (impact / effort) * confidence * risk


def order_by_leverage(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Set leverage values and return findings in descending priority order."""
    for finding in findings:
        finding["leverage"] = round(leverage_score(finding), 2)

    return sorted(
        findings,
        key=lambda finding: (
            -leverage_score(finding),
            -int(
                finding.get("category") == "security"
                and finding.get("confidence") == "HIGH"
            ),
        ),
    )


def partition_direction(
    findings: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split defect findings from separately presented direction findings."""
    defects: list[dict[str, Any]] = []
    direction: list[dict[str, Any]] = []
    for finding in findings:
        (direction if finding.get("category") == "direction" else defects).append(finding)
    return defects, direction


def _axis_value(weights: dict[str, float], value: Any, worst: float) -> float:
    return weights.get(value, worst) if isinstance(value, str) else worst
