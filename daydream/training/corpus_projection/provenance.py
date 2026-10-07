"""Extract native review-profile and explicit stack provenance."""

from __future__ import annotations

from typing import Any, Mapping

_PROFILE_FIELDS = (
    "profile_schema_version",
    "profile_name",
    "profile_source_kind",
    "profile_digest",
)


def extract_provenance(manifest_or_record: Mapping[str, Any]) -> dict[str, Any]:
    """Carry native profile fields and an explicit stack, using None when absent."""
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

    prov["stack"] = manifest_or_record.get("stack")

    return prov
