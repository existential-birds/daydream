"""Review for review and fix phases."""

from typing import Any

from daydream.json_utils import validates_schema
from daydream.phases.schemas import PER_STACK_RECORD_SCHEMA


def valid_record_artifact(
    value: Any, *, scope_id: str, analyzed_revision: dict[str, Any],
) -> bool:
    """Validate persisted provider records, permitting only host identity metadata."""
    if not isinstance(value, dict) or set(value) - {
        "issues", "incomplete", "scope_id", "analyzed_revision", "originating_run_id",
    }:
        return False
    if value.get("scope_id") != scope_id or value.get("analyzed_revision") != analyzed_revision:
        return False
    if not isinstance(value.get("originating_run_id"), str) or not value["originating_run_id"]:
        return False
    if "incomplete" in value and value["incomplete"] is not True:
        return False
    issues = value.get("issues")
    if not isinstance(issues, list) or any(not isinstance(issue, dict) for issue in issues):
        return False
    for issue in issues:
        uid = issue.get("uid")
        if not isinstance(uid, str):
            return False
        scope, _, ordinal = uid.rpartition(':')
        if scope != scope_id or not ordinal.isascii() or not ordinal.isdigit() or ordinal.startswith('0'):
            return False
    cleaned = [{key: field for key, field in issue.items() if key != "uid"} for issue in issues]
    return validates_schema({"issues": cleaned}, PER_STACK_RECORD_SCHEMA)


# Deep-mode: per-stack fan-out


