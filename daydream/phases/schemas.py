"""Structured phase outputs, sharing the canonical finding and verdict fields."""

import copy
import re
from typing import Any

from daydream.output_schema import result_array_schema, severity_enum_schema, strict_object
from daydream.repository_paths import REPOSITORY_FILE_PATH_SCHEMA
from daydream.review_source_refs import SOURCE_REFS_SCHEMA, SOURCE_SELECTORS_SCHEMA

_FINDING_FIELDS = {
    "id": {"type": "integer"},
    "description": {"type": "string"},
    "file": REPOSITORY_FILE_PATH_SCHEMA,
    "line": {"type": "integer"},
    "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM"]},
    "rationale": {"type": "string"},
    "evidence": {"type": "string"},
}
FEEDBACK_SCHEMA = result_array_schema("issues", dict(_FINDING_FIELDS))
PER_STACK_RECORD_SCHEMA = result_array_schema("issues", {
    **copy.deepcopy(_FINDING_FIELDS), "severity": severity_enum_schema(),
})
PER_STACK_RECORD_SCHEMA["properties"]["issues"]["items"]["required"] = [
    "id", "description", "file", "line", "severity", "confidence", "rationale", "evidence"
]
# Private progress protocol. Terminal records keep their public contract unchanged.
REVIEW_STAGE_SCHEMA = strict_object({
    "targets": {"type": "array", "items": strict_object({
        "target_id": {"type": "string"},
        "status": {"type": "string", "enum": ["reviewed", "not_reviewed"]},
        "reason": {"type": "string"},
        "evidence_refs": copy.deepcopy(SOURCE_REFS_SCHEMA),
    })},
    "notes": {"type": "string"},
    # Optional associations are confined by the host; an unusable reference
    # cannot invalidate otherwise valid target/candidate decisions.
    "supporting_files": {"type": ["array", "null"], "items": {"type": "string"}},
    "candidates": {"type": "array", "items": strict_object({
        "candidate_id": {"type": "string"},
        "file": REPOSITORY_FILE_PATH_SCHEMA,
        "line": {"type": "integer"},
        **{name: {"type": "string"} for name in ("trigger", "consequence", "grounds")},
        "disposition": {"type": "string", "enum": ["open", "confirmed", "rejected", "unresolved"]},
        "finding": {"anyOf": [copy.deepcopy(PER_STACK_RECORD_SCHEMA["properties"]["issues"]["items"]),
                               {"type": "null"}]},
    })},
    "contradictions": {"type": "array", "items": {"type": "string"}},
})


def review_stage_schema(target_ids: list[str], candidate_ids: list[str], *, triage: bool) -> dict[str, Any]:
    """Specialize output transport to the assignment; semantic admission stays independent."""
    schema = copy.deepcopy(REVIEW_STAGE_SCHEMA)
    targets = schema['properties']['targets']
    targets.update(minItems=len(target_ids), maxItems=len(target_ids))
    if target_ids:
        targets['items']['properties']['target_id']['enum'] = target_ids
    candidates = schema['properties']['candidates']
    candidates['items']['properties']['candidate_id']['enum'] = candidate_ids if triage else ['']
    if triage:
        candidates.update(minItems=len(candidate_ids), maxItems=len(candidate_ids))
        candidates['items']['properties']['disposition']['enum'] = ['confirmed', 'rejected', 'unresolved']
    return schema


def review_stage_admission_schema(transport_schema: dict[str, Any]) -> dict[str, Any]:
    """Optional relevance cannot invalidate required primary stage judgments.

    Native transports retain their strict array-or-null field; host admission
    ignores absent or unusable supporting metadata without repairing decisions.
    """
    schema = copy.deepcopy(transport_schema)
    schema['required'].remove('supporting_files')
    schema['properties']['supporting_files'] = {}
    return schema


def review_stage_generation_schema(canonical_schema: dict[str, Any], revision: dict[str, Any]) -> dict[str, Any]:
    """Constrain native generation to frozen commits without changing admission."""
    schema = copy.deepcopy(canonical_schema)
    revisions = [revision.get(key) for key in ('head_sha', 'merge_base_sha')]
    if all(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', value) for value in revisions):
        schema['properties']['targets']['items']['properties']['evidence_refs']['items']['properties'][
            'revision']['enum'] = list(dict.fromkeys(revisions))
    return schema


def review_stage_submission_schema(canonical_schema: dict[str, Any]) -> dict[str, Any]:
    """Pi models select sources; only the host constructs retained references."""
    schema = copy.deepcopy(canonical_schema)
    schema['properties']['targets']['items']['properties']['evidence_refs'] = copy.deepcopy(SOURCE_SELECTORS_SCHEMA)
    return schema

ALTERNATIVE_REVIEW_SCHEMA = result_array_schema("issues", {
    "id": {"type": "integer"},
    "title": {"type": "string"},
    "description": {"type": "string"},
    "recommendation": {"type": "string"},
    "severity": severity_enum_schema(),
    "files": {"type": "array", "items": REPOSITORY_FILE_PATH_SCHEMA},
    **{name: copy.deepcopy(_FINDING_FIELDS[name]) for name in ("confidence", "rationale", "evidence")},
})
MERGED_ITEMS_SCHEMA = result_array_schema("items", {
    **_FINDING_FIELDS,
    "lens": {"type": "string", "enum": ["per-stack", "cross-stack", "structural", "wonder"]},
    "severity": severity_enum_schema(),
    # Strict structured output requires both keys. Null means no known related
    # file or provenance; the host validates source_uids against actual records.
    "related_files": {"type": ["array", "null"], "items": REPOSITORY_FILE_PATH_SCHEMA},
    "source_uids": {"type": ["array", "null"], "items": {"type": "string"}},
})
RECOMMENDATION_VERDICTS_SCHEMA = result_array_schema("verdicts", {
    "issue_id": {"type": "integer"},
    "verdict": {"type": "string", "enum": ["consistent", "contradicts", "uncertain"]},
    "evidence": {"type": "string"},
    "unverified_assumptions": {"type": "array", "items": {"type": "string"}},
})

# The verifier and fix-loop routing share this vocabulary. Only retargetable
# verdicts may provide a corrected path; phase_fix_verify enforces that rule.
FIX_VERIFY_VERDICTS: tuple[str, ...] = ("resolved", "unresolved", "wrong_target", "regressed")
FIX_VERIFY_ACTIONABLE_VERDICTS: tuple[str, ...] = ("unresolved", "wrong_target", "regressed")
FIX_VERIFY_RETARGETABLE_VERDICTS: tuple[str, ...] = ("wrong_target", "regressed")
FIX_VERIFY_VERDICTS_SCHEMA = result_array_schema("verdicts", {
    "issue_id": {"type": "integer"},
    "verdict": {"type": "string", "enum": list(FIX_VERIFY_VERDICTS)},
    "path": {"type": ["string", "null"]},
    "reason": {"type": "string"},
    "check_command": {"type": ["string", "null"]},
    "check_only": {"type": "boolean"},
})


def _adjudication_schema(id_key: str, *, allow_low: bool = False) -> dict[str, Any]:
    return result_array_schema("findings", {
        id_key: {"type": "integer"},
        "keep": {"type": "boolean"},
        "severity": severity_enum_schema(),
        "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"] if allow_low else ["HIGH", "MEDIUM"]},
        **{name: {"type": "string"} for name in ("description", "rationale", "evidence")},
    })


ARBITER_SCHEMA = _adjudication_schema("arb_id")
# Suppression sees LOW-confidence findings that the arbiter never receives.
SUPPRESSION_SCHEMA = _adjudication_schema("sup_id", allow_low=True)
SUPERVISE_SCHEMA = result_array_schema("verdicts", {
    "id": {"type": "integer"},
    "action": {"type": "string", "enum": ["allow", "drop", "edit", "hold"]},
    "reason": {"type": "string"},
    "severity": severity_enum_schema(nullable=True),
    "confidence": {"anyOf": [{"type": "string", "enum": ["HIGH", "MEDIUM", "LOW"]}, {"type": "null"}]},
    **{name: {"anyOf": [{"type": "string"}, {"type": "null"}]} for name in ("description", "rationale", "evidence")},
})
