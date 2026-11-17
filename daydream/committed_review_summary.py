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

import daydream
from daydream.committed_review_model import INVESTIGATION_KINDS, RESULT_VERSIONS
from daydream.json_utils import validate_json_schema
from daydream.output_schema import strict_object
from daydream.retry_policy import FailureClass
from daydream.review_result import TERMINAL_RESULT_SCHEMA, ReasonCode
from daydream.trajectory.timing import phase_wall_seconds
from daydream.trajectory.types import DaydreamPhase

if TYPE_CHECKING:
    from daydream.trajectory.recorder import EvidenceRecorder

SUMMARY_VERSION = 1
#: Each configuration phase users set (``[phases.review]``, ``[phases.arbiter]``), its evidence phase and work kinds.
SUMMARY_PHASES: dict[str, tuple[DaydreamPhase, list[str]]] = {
    "review": (DaydreamPhase.REVIEW, sorted(INVESTIGATION_KINDS)),
    "arbiter": (DaydreamPhase.CHALLENGE, ["challenge"]),
}
_BUDGET_REASONS = [ReasonCode.HOST_WALL_BUDGET_EXHAUSTION.value, ReasonCode.HOST_TOOL_BUDGET_EXHAUSTION.value,
                   ReasonCode.HOST_PIPELINE_BUDGET_EXHAUSTION.value]
_COUNT = {"type": ["integer", "null"], "minimum": 0}
_AMOUNT = {"type": ["number", "null"], "minimum": 0}
_NAME = {"type": ["string", "null"], "minLength": 1, "maxLength": 512}


def _by_reason(reasons: list[str]) -> dict[str, Any]:
    return {"type": "object", "propertyNames": {"enum": reasons}, "additionalProperties": {"type": "integer",
                                                                                           "minimum": 1}}


SUMMARY_SCHEMA = strict_object({
    "schema_version": {"const": SUMMARY_VERSION},
    "run_id": {"type": "string", "minLength": 1, "maxLength": 512},
    "result_schema_version": {"enum": sorted(RESULT_VERSIONS.values())},
    "daydream_version": {"type": "string", "minLength": 1, "maxLength": 128},
    "profile": strict_object({"name": _NAME, "source_kind": _NAME, "digest": _NAME}),
    "terminal": strict_object({key: TERMINAL_RESULT_SCHEMA["properties"][key]
                               for key in ("analysis_state", "pipeline_state", "reason_codes")}),
    "counts": strict_object({
        "assignments": _COUNT,
        "targets": strict_object({key: _COUNT for key in ("planned", "reviewed", "not_reviewed")}),
        "claims": strict_object({key: _COUNT for key in ("proposed", "confirmed", "rejected", "unresolved")}),
        "findings": strict_object({key: _COUNT for key in ("confirmed", "delivered")}),
    }),
    "budget_stops": {**_by_reason(_BUDGET_REASONS), "type": ["object", "null"]},
    "retries": strict_object({"total": _COUNT, "by_reason": {**_by_reason([member.value for member in FailureClass]),
                                                             "type": ["object", "null"]}}),
    "phases": strict_object({name: strict_object({
        "kinds": {"const": kinds}, "backend": _NAME, "model": _NAME, "reasoning_effort": _NAME,
        "prompt_tokens": _COUNT, "cached_tokens": _COUNT, "completion_tokens": _COUNT, "cost_usd": _AMOUNT,
        "wall_seconds": _AMOUNT,
    }) for name, (_phase, kinds) in SUMMARY_PHASES.items()}),
})


def validate_review_summary(value: dict[str, Any]) -> None:
    """Refuse any other summary version, then validate the current shape."""
    version = value.get("schema_version")
    if type(version) is not int or version != SUMMARY_VERSION:
        raise ValueError("unsupported committed review summary version")
    validate_json_schema(value, SUMMARY_SCHEMA)


def build_review_summary(
    sealed: Mapping[str, Any], *, resolved_phases: Mapping[str, Mapping[str, str | None]],
    recorder: EvidenceRecorder | None, delivered: int,
) -> dict[str, Any]:
    """Summarize a validated sealed result; without a recorder, evidence-derived values are ``null``.

    The output writer validates the summary with the sealed result it stages.
    """
    settings, terminal = sealed["settings"], sealed["terminal_result"]
    decisions = Counter(decision["status"] for outcome in terminal["stack_outcomes"]
                        for decision in outcome.get("target_decisions", []))
    dispositions = Counter(claim["disposition"] for claim in sealed["claims"])
    stops = [entry["reason"] for entry in sealed["execution"] if entry["reason"] in _BUDGET_REASONS]
    invocations = [] if recorder is None else [
        summary for summary in recorder.invocation_summaries()
        if summary.get("phase") in {phase.value for phase, _kinds in SUMMARY_PHASES.values()}]
    retried = [summary["retry_failure_class"] for summary in invocations if "retry_failure_class" in summary]
    events = [] if recorder is None else recorder.phase_event_summaries()
    phases = {}
    for name, (phase, kinds) in SUMMARY_PHASES.items():
        share = None if recorder is None else recorder.usage.phases.get(phase.value)
        phases[name] = {
            "kinds": kinds, **{key: resolved_phases[name][key] for key in ("backend", "model", "reasoning_effort")},
            **{key: None if share is None else getattr(share, attribute) for key, attribute in (
                ("prompt_tokens", "prompt"), ("cached_tokens", "cached"), ("completion_tokens", "completion"),
                ("cost_usd", "cost"))},
            "wall_seconds": phase_wall_seconds(events, phase.value),
        }
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
