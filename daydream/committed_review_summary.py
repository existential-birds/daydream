"""One small, schema-versioned summary of a sealed committed review, derived only from retained evidence.

The summary reads the sealed result, the config-resolved per-phase settings and the
host-retained local evidence of the recorder that recorded the review; it never calls
a model and never changes what it summarizes. A value that was not recorded is
``null``, never ``0``. Only the current ``schema_version`` validates.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import jsonschema

import daydream
from daydream.committed_review_model import RESULT_VERSIONS
from daydream.json_utils import validate_json_schema
from daydream.output_schema import strict_object
from daydream.retry_policy import FailureClass
from daydream.review_result import BUDGET_REASON_CODES, TERMINAL_RESULT_SCHEMA
from daydream.run_config import RunConfig, resolved_phase_settings
from daydream.trajectory.timing import phase_wall_seconds
from daydream.trajectory.types import DaydreamPhase

if TYPE_CHECKING:
    from daydream.trajectory.recorder import EvidenceRecorder

SUMMARY_VERSION = 1
#: Each configuration phase users set (``[phases.review]``, ``[phases.arbiter]``) and the evidence phase of its work.
SUMMARY_PHASES = {"review": DaydreamPhase.REVIEW, "arbiter": DaydreamPhase.CHALLENGE}
_USAGE = ("prompt_tokens", "cached_tokens", "completion_tokens", "cost_usd")
_COUNT = {"type": ["integer", "null"], "minimum": 0}
_AMOUNT = {"type": ["number", "null"], "minimum": 0}
# Recorded settings are copied as they are: a profile without a name records an empty one.
_TEXT = {"type": ["string", "null"]}


def _by_reason(reasons: list[str]) -> dict[str, Any]:
    return {"type": ["object", "null"], "propertyNames": {"enum": reasons},
            "additionalProperties": {"type": "integer", "minimum": 1}}


SUMMARY_SCHEMA = strict_object({
    "schema_version": {"const": SUMMARY_VERSION},
    "run_id": {"type": "string", "minLength": 1},
    "result_schema_version": {"enum": sorted(RESULT_VERSIONS.values())},
    "daydream_version": {"type": "string", "minLength": 1},
    "profile": strict_object({"name": _TEXT, "source_kind": _TEXT, "digest": _TEXT}),
    "terminal": strict_object({key: TERMINAL_RESULT_SCHEMA["properties"][key]
                               for key in ("analysis_state", "pipeline_state", "reason_codes")}),
    "counts": strict_object({
        "assignments": _COUNT,
        "targets": strict_object({key: _COUNT for key in ("planned", "reviewed", "not_reviewed")}),
        "claims": strict_object({key: _COUNT for key in ("proposed", "confirmed", "rejected", "unresolved")}),
        "findings": strict_object({key: _COUNT for key in ("confirmed", "delivered")}),
    }),
    "budget_stops": _by_reason(sorted(code.value for code in BUDGET_REASON_CODES)),
    "retries": strict_object({"total": _COUNT, "by_reason": _by_reason([member.value for member in FailureClass])}),
    "phases": strict_object({name: strict_object({
        **{key: _TEXT for key in ("backend", "model", "reasoning_effort")},
        **{key: _AMOUNT if key == "cost_usd" else _COUNT for key in _USAGE}, "wall_seconds": _AMOUNT,
    }) for name in SUMMARY_PHASES}),
})


def validate_review_summary(value: dict[str, Any]) -> None:
    """Refuse any other summary version or shape."""
    try:
        validate_json_schema(value, SUMMARY_SCHEMA)
    except jsonschema.ValidationError as exc:
        raise ValueError(f"invalid committed review summary: {exc.message}") from None


def build_review_summary(
    sealed: Mapping[str, Any], *, config: RunConfig, recorder: EvidenceRecorder | None, delivered: int,
) -> dict[str, Any]:
    """Summarize a validated sealed result; without a recorder, evidence-derived values are ``null``.

    The output writer validates the summary with the sealed result it stages.
    """
    settings, terminal = sealed["settings"], sealed["terminal_result"]
    decisions = Counter(decision["status"] for outcome in terminal["stack_outcomes"]
                        for decision in outcome.get("target_decisions", []))
    dispositions = Counter(claim["disposition"] for claim in sealed["claims"])
    stops = [entry["reason"] for entry in sealed["execution"] if entry["reason"] in BUDGET_REASON_CODES]
    retried = [] if recorder is None else [
        summary["retry_failure_class"] for summary in recorder.invocation_summaries()
        if summary.get("phase") in SUMMARY_PHASES.values() and "retry_failure_class" in summary]
    events = [] if recorder is None else recorder.phase_event_summaries()
    resolved = resolved_phase_settings(config, SUMMARY_PHASES)
    phases = {}
    for name, phase in SUMMARY_PHASES.items():
        usage = {} if recorder is None or phase not in recorder.usage.phases else recorder.usage.phases[phase].usage()
        phases[name] = {**resolved[name], **{key: usage.get(key) for key in _USAGE},
                        "wall_seconds": phase_wall_seconds(events, phase)}
    return {
        "schema_version": SUMMARY_VERSION,
        "run_id": sealed["run_id"],
        "result_schema_version": sealed["schema_version"],
        "daydream_version": daydream.__version__,
        "profile": {"name": settings.get("profile_name"), "source_kind": settings.get("profile_source"),
                    "digest": settings.get("profile_digest")},
        "terminal": {key: terminal[key] for key in ("analysis_state", "pipeline_state", "reason_codes")},
        "counts": {
            "assignments": (settings.get("assignment_policy") or {}).get("planned_assignments"),
            "targets": {"planned": len(sealed["assignments"]), "reviewed": decisions["reviewed"],
                        "not_reviewed": decisions["not_reviewed"]},
            "claims": {"proposed": len(sealed["claims"]), **{key: dispositions[key]
                                                             for key in ("confirmed", "rejected", "unresolved")}},
            "findings": {"confirmed": len(sealed["findings"]), "delivered": delivered},
        },
        "budget_stops": None if recorder is None else dict(Counter([*stops, *recorder.schedule_budget_stops])),
        "retries": {"total": None if recorder is None else len(retried),
                    "by_reason": None if recorder is None else dict(Counter(retried))},
        "phases": phases,
    }
