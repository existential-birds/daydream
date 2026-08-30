"""Profile + stack provenance extraction for corpus v2 records (Req 8).

Reads the four native review-profile fields (issue #885, R12) verbatim from a
manifest row / trajectory record and assembles the provenance block: the
``profile`` mapping, optional legacy ``skill`` provenance, and the detected
``stack`` label. Composes v1's legacy skill→stack map
(``daydream.training.corpus``) rather than duplicating it.
"""

from __future__ import annotations

from typing import Any, Mapping

_PROFILE_FIELDS = (
    "profile_schema_version",
    "profile_name",
    "profile_source_kind",
    "profile_digest",
)


def extract_provenance(manifest_or_record: Mapping[str, Any]) -> dict[str, Any]:
    """Extract provenance fields from a manifest row or v2 record.

    The four profile fields are carried verbatim (``null`` when absent —
    never dropped, never substituted). Legacy ``skill`` is included only when
    a value exists; ``stack`` falls back to the legacy skill→stack mapping and
    is ``None`` when unresolvable.
    """
    prov: dict[str, Any] = {
        "profile": {field: manifest_or_record.get(field) for field in _PROFILE_FIELDS},
    }

    skill = manifest_or_record.get("skill")
    if skill is not None:
        prov["skill"] = skill

    stack = manifest_or_record.get("stack")
    if stack is None and skill is not None:
        from daydream.training.corpus import _stack_for_skill

        stack = _stack_for_skill(skill)
    prov["stack"] = stack

    return prov
