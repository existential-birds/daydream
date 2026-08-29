"""Prioritize vetted improve-flow findings."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

from daydream.deep.dedup import bigrams, jaccard, normalize_title
from daydream.severity import normalize_severity

_IMPACT = {"HIGH": 3.0, "MED": 2.0, "LOW": 1.0}
_EFFORT = {"S": 1.0, "M": 2.0, "L": 3.0}
_CONFIDENCE = {"HIGH": 1.0, "MED": 0.7, "LOW": 0.4}
_RISK = {"LOW": 1.0, "MED": 0.8, "HIGH": 0.6}
_SAME_LOCATION_SIMILARITY = 0.5
_CROSS_CATEGORY_SIMILARITY = 0.7
_WORK_PACKAGE_MAX_FINDINGS = 5
_WORK_PACKAGE_MAX_PATHS = 8
_IMPACT_ORDER = {"LOW": 0, "MED": 1, "HIGH": 2}
_EFFORT_UNITS = {"S": 1, "M": 2, "L": 3}
_RISK_ORDER = {"LOW": 0, "MED": 1, "HIGH": 2}
_CONFIDENCE_ORDER = {"LOW": 0, "MED": 1, "HIGH": 2}
_SEVERITY_ORDER = {"LOW": 0, "MED": 1, "HIGH": 2, "CRITICAL": 3}
_CHANGE_SHAPES = {
    "delete",
    "reuse",
    "consolidate",
    "neutral",
    "additive",
    "unknown",
}
_CHANGE_SHAPE_PREFERENCE = {
    "delete": 0,
    "reuse": 1,
    "consolidate": 2,
    "neutral": 3,
    "unknown": 3,
    "additive": 4,
}
# HIGH impact still clears P1 at medium effort; the P2 floor is a MED/MED/MED
# finding, so only the low-impact or high-risk tail lands in P3.
_P1_LEVERAGE = 1.2
_P2_LEVERAGE = 0.8


def leverage_score(finding: dict[str, Any]) -> float:
    """Return impact over effort, discounted by confidence and fix risk."""
    impact = _axis_value(_IMPACT, finding.get("impact"), min(_IMPACT.values()))
    effort = _axis_value(_EFFORT, finding.get("effort"), max(_EFFORT.values()))
    confidence = _axis_value(
        _CONFIDENCE, finding.get("confidence"), min(_CONFIDENCE.values())
    )
    risk = _axis_value(_RISK, finding.get("risk"), min(_RISK.values()))
    return (impact / effort) * confidence * risk


def plan_priority(finding: dict[str, Any]) -> str:
    """Rank a finding's plan for a human picking what to work on next.

    Derived from the same impact/effort/risk/confidence axes as the leverage
    score, so the plan index and the audit report cannot disagree.
    """
    score = round(leverage_score(finding), 2)
    if score >= _P1_LEVERAGE:
        return "P1"
    return "P2" if score >= _P2_LEVERAGE else "P3"


def order_by_leverage(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Set leverage values and return findings in descending priority order."""
    for finding in findings:
        finding["leverage"] = round(leverage_score(finding), 2)

    return sorted(
        findings,
        key=lambda finding: (
            -leverage_score(finding),
            -int(finding.get("category") == "security" and finding.get("confidence") == "HIGH"),
            _CHANGE_SHAPE_PREFERENCE.get(
                str(finding.get("change_shape") or "unknown"),
                _CHANGE_SHAPE_PREFERENCE["unknown"],
            ),
        ),
    )


def aggregate_cross_service(
    findings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build conservative, issue-sized work packages before user selection.

    The historical name remains as a compatibility surface, but aggregation is
    no longer limited to identical findings in distinct services. Exact aliases,
    a shared canonical reuse target, and corroborated title similarity can form
    a package. Every new member must be compatible with *every* existing member;
    this deliberately avoids transitive ``A ~ B ~ C`` bridge merges.
    """
    # Aggregates can be fed back through this function by callers composing
    # audit partitions. Flatten them first and collapse exact member aliases so
    # an identical finding cannot consume a package cap more than once.
    ordered = sorted(
        (member for finding in findings for member in _package_members(finding)),
        key=_finding_sort_key,
    )
    unique: list[dict[str, Any]] = []
    seen_fingerprints: set[str] = set()
    for finding in ordered:
        fingerprint = _primary_fingerprint(finding)
        if fingerprint and fingerprint in seen_fingerprints:
            continue
        if fingerprint:
            seen_fingerprints.add(fingerprint)
        unique.append(finding)
    groups: list[list[dict[str, Any]]] = []

    for finding in unique:
        for group in groups:
            if _can_join_work_package(finding, group):
                group.append(finding)
                break
        else:
            groups.append([finding])

    return _with_unique_package_fingerprints([_merge_work_package(group) for group in groups])


def _finding_sort_key(finding: dict[str, Any]) -> tuple[Any, ...]:
    """Return an input-order-independent package construction key."""
    return (
        -leverage_score(finding),
        normalize_title(str(finding.get("title") or "")),
        str(finding.get("path") or ""),
        int(finding.get("line") or 0),
        str(finding.get("fingerprint") or ""),
    )


def _can_join_work_package(finding: dict[str, Any], group: list[dict[str, Any]]) -> bool:
    members = [member for item in group for member in _package_members(item)]
    candidate_members = _package_members(finding)
    if len(members) + len(candidate_members) > _WORK_PACKAGE_MAX_FINDINGS:
        return False
    paths = {path for member in [*members, *candidate_members] if (path := _finding_path(member))}
    if len(paths) > _WORK_PACKAGE_MAX_PATHS:
        return False
    return all(_findings_are_compatible(candidate, existing) for candidate in candidate_members for existing in members)


def _findings_are_compatible(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Return whether two findings describe one coherent implementation unit."""
    left_target = _reuse_target_key(left)
    right_target = _reuse_target_key(right)
    # A finding cannot coherently belong to two different reuse migrations,
    # even if an upstream model accidentally stamped the same volatile ID.
    if left_target and right_target and left_target != right_target:
        return False

    if _finding_aliases(left) & _finding_aliases(right):
        return True

    if left_target and left_target == right_target:
        return True

    left_title = bigrams(normalize_title(str(left.get("title") or "")))
    right_title = bigrams(normalize_title(str(right.get("title") or "")))
    if not left_title or not right_title:
        return False
    similarity = jaccard(left_title, right_title)
    same_category = left.get("category") == right.get("category")
    signal_overlap = bool(_maintenance_signals(left) & _maintenance_signals(right))
    same_path = bool(_finding_path(left) and _finding_path(left) == _finding_path(right))
    localities = _finding_services(left), _finding_services(right)
    distinct_localities = bool(localities[0] and localities[1] and not localities[0] & localities[1])

    if same_category and similarity >= _SAME_LOCATION_SIMILARITY:
        return same_path or signal_overlap or distinct_localities
    return similarity >= _CROSS_CATEGORY_SIMILARITY and (same_path or signal_overlap)


def _package_members(finding: dict[str, Any]) -> list[dict[str, Any]]:
    members = finding.get("members")
    if isinstance(members, list):
        valid = [member for member in members if isinstance(member, dict)]
        if valid:
            return [_without_package_fields(member) for member in valid]
    return [_without_package_fields(finding)]


def _without_package_fields(finding: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in finding.items()
        if key
        not in {
            "members",
            "member_fingerprints",
            "member_aliases",
            "package_fingerprint",
            "categories",
            "locations",
            "partitions",
        }
    }


def _finding_aliases(finding: dict[str, Any]) -> set[str]:
    aliases = {
        value
        for value in (
            finding.get("fingerprint"),
            finding.get("package_fingerprint"),
        )
        if isinstance(value, str) and value
    }
    member_fingerprints = finding.get("member_fingerprints")
    if isinstance(member_fingerprints, list):
        aliases.update(value for value in member_fingerprints if isinstance(value, str) and value)
    return aliases


def _primary_fingerprint(finding: dict[str, Any]) -> str:
    fingerprint = finding.get("fingerprint")
    return fingerprint if isinstance(fingerprint, str) and fingerprint else ""


def _concern_for(target: str, signals: list[str], title: str) -> dict[str, Any]:
    """Select the shared three-way concern anchor for member and package IDs.

    A canonical reuse target wins, then structured maintenance signals, then
    the normalized title as the safe fallback. The same ladder backs both the
    per-member alias and the package fingerprint, so a reworded finding keeps
    one semantic anchor across both identities.
    """
    if target:
        return {"kind": "reuse-target", "value": target}
    if signals:
        return {"kind": "maintenance-signals", "value": signals}
    return {"kind": "title", "value": normalize_title(title)}


def member_alias(finding: dict[str, Any]) -> str:
    """Return a wording-resistant semantic alias for one finding.

    Audit fingerprints deliberately include model prose. This alias instead
    uses the repository location and the host-normalized maintenance concern,
    allowing a later run to recognize a reworded member. The line keeps two
    independent concerns in one large file from becoming the same alias; when
    no structured concern exists, normalized title text is the safe fallback.
    """
    concern = _concern_for(
        _reuse_target_key(finding),
        sorted(_maintenance_signals(finding)),
        str(finding.get("title") or ""),
    )
    line = finding.get("line")
    canonical = json.dumps(
        {
            "kind": "daydream-improve-member-v1",
            "path": _finding_path(finding),
            "line": line if isinstance(line, int) and not isinstance(line, bool) else None,
            "category": str(finding.get("category") or ""),
            "concern": concern,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"member-v1:{hashlib.sha256(canonical.encode()).hexdigest()}"


def _reuse_target_key(finding: dict[str, Any]) -> str:
    target = finding.get("reuse_target")
    if not isinstance(target, str):
        return ""
    normalized = " ".join(target.strip().split()).casefold()
    return "" if normalized in {"", "none", "null", "n/a"} else normalized


def _maintenance_signals(finding: dict[str, Any]) -> set[str]:
    signals = finding.get("maintenance_signals")
    if not isinstance(signals, list):
        return set()
    return {signal for signal in signals if isinstance(signal, str) and signal}


def _finding_path(finding: dict[str, Any]) -> str:
    path = finding.get("path")
    return path if isinstance(path, str) else ""


def _merge_work_package(findings: list[dict[str, Any]]) -> dict[str, Any]:
    members = sorted(
        (member for finding in findings for member in _package_members(finding)),
        key=_finding_sort_key,
    )
    representative = members[0]
    merged = dict(representative)
    member_fingerprints = sorted({fingerprint for member in members if (fingerprint := _primary_fingerprint(member))})
    # Preserve one alias per member. Repeated values are meaningful: they tell
    # reconciliation that this semantic alias is ambiguous and cannot by
    # itself prove coverage of multiple distinct members.
    member_aliases = sorted(member_alias(member) for member in members)
    package_fingerprint = _work_package_fingerprint(members)
    merged["fingerprint"] = package_fingerprint
    merged["package_fingerprint"] = package_fingerprint
    merged["member_fingerprints"] = member_fingerprints
    merged["member_aliases"] = member_aliases
    merged["members"] = members
    merged["categories"] = sorted(
        {category for member in members if isinstance((category := member.get("category")), str) and category}
    )
    merged["services"] = sorted({service for member in members for service in _service_list(member)})
    merged["partitions"] = sorted(
        {partition for member in members if isinstance((partition := member.get("partition")), str) and partition}
    )
    merged["evidence"] = sorted({entry for member in members for entry in _evidence_list(member)})
    merged["locations"] = sorted({_finding_location(member) for member in members if _finding_path(member)})
    merged["maintenance_signals"] = sorted({signal for member in members for signal in _maintenance_signals(member)})
    merged["reuse_target"] = _common_reuse_target(members)
    merged["change_shape"] = _combined_change_shape(members)
    merged["impact"] = _conservative_axis(members, "impact", _IMPACT_ORDER, "LOW")
    merged["effort"] = _combined_effort(members)
    merged["risk"] = _conservative_axis(members, "risk", _RISK_ORDER, "HIGH")
    merged["confidence"] = _conservative_axis(
        members,
        "confidence",
        _CONFIDENCE_ORDER,
        "LOW",
        highest=False,
    )
    # Unified fallback policy (issue #972 R3.1): when no member carries a
    # known severity the axis is omitted entirely — no conservative "HIGH"
    # default is fabricated for severity (the one permitted "high" default is
    # the structural-lens setdefault in phases.py).
    severity_values = [
        axis
        for member in members
        if "severity" in member and (axis := _map_axis_severity(member["severity"])) is not None
    ]
    if severity_values:
        merged["severity"] = max(severity_values, key=_SEVERITY_ORDER.__getitem__)
    merged["leverage"] = round(leverage_score(merged), 2)

    if len(members) > 1:
        body = str(representative.get("body") or "").rstrip()
        locations = "\n".join(
            f"- `{_finding_location(member)}` — {member.get('title', 'Finding')}" for member in members
        )
        merged["body"] = f"{body}\n\nCombined work-package locations:\n{locations}"
    return merged


def _work_package_fingerprint(members: list[dict[str, Any]]) -> str:
    """Hash semantic member anchors, leaving volatile audit IDs as aliases."""
    paths = sorted({_finding_path(member) for member in members if _finding_path(member)})
    reuse_target = _common_reuse_target(members)
    signals = sorted({signal for member in members for signal in _maintenance_signals(member)})
    categories = sorted(
        {category for member in members if isinstance((category := member.get("category")), str) and category}
    )
    semantic_members = sorted(member_alias(member) for member in members)
    alias_counts = Counter(semantic_members)
    # Two semantically distinct findings can share the same structured anchor
    # (most often the same line and a generic maintenance signal). In that
    # ambiguous case, volatile IDs are the only safe discriminator. Ordinary
    # rewording remains stable because unique semantic anchors need no such
    # fallback.
    collision_members = (
        sorted(_primary_fingerprint(member) or normalize_title(str(member.get("title") or "")) for member in members)
        if any(count > 1 for count in alias_counts.values())
        else []
    )
    concern = _concern_for(reuse_target or "", signals, str(members[0].get("title") or ""))
    canonical = json.dumps(
        {
            "kind": "daydream-improve-work-package-v2",
            "concern": concern,
            "paths": paths,
            "members": semantic_members,
            **({"collision_members": collision_members} if collision_members else {}),
            **({} if reuse_target else {"categories": categories}),
            **(
                {}
                if paths
                else {"fallback_members": sorted({alias for member in members for alias in _finding_aliases(member)})}
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _with_unique_package_fingerprints(
    packages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Disambiguate the rare case where capped packages share one base ID."""
    by_fingerprint: dict[str, list[dict[str, Any]]] = {}
    for package in packages:
        by_fingerprint.setdefault(str(package["package_fingerprint"]), []).append(package)
    for base, collisions in by_fingerprint.items():
        if len(collisions) < 2:
            continue
        for package in collisions:
            canonical = json.dumps(
                {
                    "kind": "daydream-improve-work-package-split-v1",
                    "base": base,
                    "members": sorted(str(value) for value in package.get("member_fingerprints", [])),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
            package["fingerprint"] = fingerprint
            package["package_fingerprint"] = fingerprint
    return packages


def _finding_location(finding: dict[str, Any]) -> str:
    path = _finding_path(finding)
    line = finding.get("line")
    return f"{path}:{line}" if line is not None else path


def _common_reuse_target(findings: list[dict[str, Any]]) -> str | None:
    targets = {target for finding in findings if (target := _reuse_target_key(finding))}
    return next(iter(targets)) if len(targets) == 1 else None


def _combined_change_shape(findings: list[dict[str, Any]]) -> str:
    shapes = {
        shape
        for finding in findings
        if isinstance((shape := finding.get("change_shape")), str) and shape in _CHANGE_SHAPES
    }
    if not shapes:
        return "unknown"
    if len(shapes) == 1:
        return next(iter(shapes))
    if "additive" in shapes:
        return "additive"
    if "unknown" in shapes:
        return "unknown"
    if "neutral" in shapes:
        return "neutral"
    return "consolidate"


def _map_axis_severity(value: object) -> str | None:
    """Map a finding's severity to the audit-axis vocabulary, or ``None``.

    Canonical lowercase levels (``daydream.severity``) map to their uppercase
    axis names; values already in the axis vocabulary pass through as known
    vocabulary (not an unknown passthrough). Anything else — unknown, absent,
    ``None`` — maps to ``None`` so the caller can omit the axis instead of
    promoting it to a conservative fallback (P-BOUNDARY, issue #972 R3.1/R6.2).
    """
    normalized = normalize_severity(value)
    if normalized is not None:
        return {"low": "LOW", "medium": "MED", "high": "HIGH"}[normalized]
    if isinstance(value, str) and value in _SEVERITY_ORDER:
        return value
    return None


def _conservative_axis(
    findings: list[dict[str, Any]],
    field: str,
    order: dict[str, int],
    fallback: str,
    *,
    highest: bool = True,
) -> str:
    values = [value for finding in findings if isinstance((value := finding.get(field)), str) and value in order]
    if not values:
        return fallback
    choose = max if highest else min
    return choose(values, key=order.__getitem__)


def _combined_effort(findings: list[dict[str, Any]]) -> str:
    units = sum(_EFFORT_UNITS.get(str(finding.get("effort") or ""), 3) for finding in findings)
    if units <= 1:
        return "S"
    return "M" if units <= 3 else "L"


def _finding_services(finding: dict[str, Any]) -> set[str]:
    """Return a finding's locality keys: its services plus its partition.

    A finding in an uncovered tree has no service, so the partition is what
    makes it comparable to the same pattern found elsewhere.
    """
    services = finding.get("services")
    keys = (
        {service for service in services if isinstance(service, str) and service}
        if isinstance(services, list)
        else set()
    )
    partition = finding.get("partition")
    if isinstance(partition, str) and partition:
        keys.add(partition)
    return keys


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


def _axis_value(weights: dict[str, float], value: Any, worst: float) -> float:
    return weights.get(value, worst) if isinstance(value, str) else worst
