"""Extract native review-profile, optional legacy skill, and stack provenance."""

from __future__ import annotations

from typing import Any, Mapping

_PROFILE_FIELDS = (
    "profile_schema_version",
    "profile_name",
    "profile_source_kind",
    "profile_digest",
)


def extract_provenance(manifest_or_record: Mapping[str, Any]) -> dict[str, Any]:
    """Prefer flat profile fields, falling back to a canonical nested profile when all are absent.
    Include skill only when present; stack falls back to the legacy skill mapping.
    """
    prov: dict[str, Any] = {
        "profile": {field: manifest_or_record.get(field) for field in _PROFILE_FIELDS},
    }

    nested = manifest_or_record.get("profile")
    if not any(prov["profile"].values()) and isinstance(nested, Mapping):
        prov["profile"] = {field: nested.get(field) for field in _PROFILE_FIELDS}

    skill = manifest_or_record.get("skill")
    if skill is not None:
        prov["skill"] = skill

    stack = manifest_or_record.get("stack")
    if stack is None and skill is not None:
        from daydream.training.corpus import _stack_for_skill

        stack = _stack_for_skill(skill)
    prov["stack"] = stack

    return prov
