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
    """Carry all four native profile fields verbatim, using None when absent. Include skill only when
    present. Stack falls back to the shared legacy skill-to-stack mapping, remaining None if
    unresolved.
    """
    prov: dict[str, Any] = {
        "profile": {field: manifest_or_record.get(field) for field in _PROFILE_FIELDS},
    }

    nested = manifest_or_record.get("profile")
    if isinstance(nested, Mapping):
        prov["profile"].update({name: nested.get(name) for name in _PROFILE_FIELDS if name in nested})
    effective = manifest_or_record.get("effective_configuration")
    if isinstance(effective, Mapping):
        native = effective.get("profile")
        if isinstance(native, Mapping) and not any(prov["profile"].values()):
            prov["profile"] = {f"profile_{name}": native.get(name)
                               for name in ("schema_version", "name", "source_kind", "digest")}

    skill = manifest_or_record.get("skill")
    if skill is not None:
        prov["skill"] = skill

    stack = manifest_or_record.get("stack")
    if stack is None and skill is not None:
        from daydream.training.corpus import _stack_for_skill

        stack = _stack_for_skill(skill)
    prov["stack"] = stack

    return prov
