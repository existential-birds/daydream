"""Prioritize vetted improve-flow findings."""

from __future__ import annotations

from typing import Any

from daydream.deep.dedup import _bigrams, _jaccard, _normalize_title

_IMPACT = {"HIGH": 3.0, "MED": 2.0, "LOW": 1.0}
_EFFORT = {"S": 1.0, "M": 2.0, "L": 3.0}
_CONFIDENCE = {"HIGH": 1.0, "MED": 0.7, "LOW": 0.4}
_RISK = {"LOW": 1.0, "MED": 0.8, "HIGH": 0.6}
_CROSS_SERVICE_SIMILARITY = 0.5


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


def aggregate_cross_service(
    findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge the same finding pattern when it occurs in distinct services."""
    aggregated: list[dict[str, Any]] = []
    consumed: set[int] = set()

    for index, finding in enumerate(findings):
        if index in consumed:
            continue
        services = _finding_services(finding)
        title_bigrams = _bigrams(_normalize_title(str(finding.get("title", ""))))
        group = [finding]

        if services and title_bigrams:
            for candidate_index in range(index + 1, len(findings)):
                if candidate_index in consumed:
                    continue
                candidate = findings[candidate_index]
                candidate_services = _finding_services(candidate)
                if (
                    candidate.get("category") != finding.get("category")
                    or not candidate_services
                    or services & candidate_services
                ):
                    continue
                candidate_bigrams = _bigrams(
                    _normalize_title(str(candidate.get("title", "")))
                )
                if (
                    _jaccard(title_bigrams, candidate_bigrams)
                    < _CROSS_SERVICE_SIMILARITY
                ):
                    continue
                group.append(candidate)
                services.update(candidate_services)
                consumed.add(candidate_index)

        if len(group) == 1:
            aggregated.append(finding)
            continue
        aggregated.append(_merge_cross_service_group(group))

    return aggregated


def _finding_services(finding: dict[str, Any]) -> set[str]:
    services = finding.get("services")
    if not isinstance(services, list):
        return set()
    return {service for service in services if isinstance(service, str) and service}


def _merge_cross_service_group(
    findings: list[dict[str, Any]],
) -> dict[str, Any]:
    representative = max(findings, key=leverage_score)
    merged = dict(representative)
    merged["services"] = _ordered_unique(
        service for finding in findings for service in _service_list(finding)
    )
    merged["evidence"] = _ordered_unique(
        entry for finding in findings for entry in _evidence_list(finding)
    )
    locations = _cross_service_locations(findings)
    body = str(representative.get("body", "")).rstrip()
    merged["body"] = f"{body}\n\nCross-service locations:\n" + "\n".join(
        f"- **{service}** — `{location}`" for service, location in locations
    )
    return merged


def _service_list(finding: dict[str, Any]) -> list[str]:
    services = finding.get("services")
    if not isinstance(services, list):
        return []
    return [service for service in services if isinstance(service, str) and service]


def _evidence_list(finding: dict[str, Any]) -> list[str]:
    evidence = finding.get("evidence")
    if not isinstance(evidence, list):
        return []
    return [entry for entry in evidence if isinstance(entry, str) and entry]


def _ordered_unique(values: Any) -> list[str]:
    return list(dict.fromkeys(values))


def _cross_service_locations(
    findings: list[dict[str, Any]],
) -> list[tuple[str, str]]:
    locations: list[tuple[str, str]] = []
    for finding in findings:
        path = str(finding.get("path", ""))
        line = finding.get("line")
        location = f"{path}:{line}" if line is not None else path
        locations.extend((service, location) for service in _service_list(finding))
    return list(dict.fromkeys(locations))


def _axis_value(weights: dict[str, float], value: Any, worst: float) -> float:
    return weights.get(value, worst) if isinstance(value, str) else worst
