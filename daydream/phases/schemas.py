"""Structured phase outputs, sharing the canonical finding and verdict fields."""

import copy

from daydream.output_schema import result_array_schema, severity_enum_schema
from daydream.repository_paths import REPOSITORY_FILE_PATH_SCHEMA

_FINDING_FIELDS = {
    "id": {"type": "integer"},
    "description": {"type": "string"},
    "file": REPOSITORY_FILE_PATH_SCHEMA,
    "line": {"type": "integer"},
    "confidence": {"type": "string", "enum": ["HIGH", "MEDIUM"]},
    "rationale": {"type": "string"},
    "evidence": {"type": "string"},
}
PER_STACK_RECORD_SCHEMA = result_array_schema("issues", {
    **copy.deepcopy(_FINDING_FIELDS), "severity": severity_enum_schema(),
})
PER_STACK_RECORD_SCHEMA["properties"]["issues"]["items"]["required"] = [
    "id", "description", "file", "line", "severity", "confidence", "rationale", "evidence"
]


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
