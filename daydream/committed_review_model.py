"""Canonical claims, revision-sensitive challenge evidence and sealed output."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import jsonschema

from daydream.host_tools import _MAX_ARGUMENT_BYTES
from daydream.immutable_review import IMMUTABLE_REVIEW_INPUT_SCHEMA, validate_immutable_input
from daydream.json_utils import canonical_json
from daydream.output_schema import strict_object
from daydream.review_result import TERMINAL_RESULT_SCHEMA, ReasonCode, validate_terminal_result
from daydream.review_source_refs import (
    SOURCE_REF_SCHEMA,
    SOURCE_REFS_SCHEMA,
    SOURCE_SELECTORS_SCHEMA,
    source_revision,
    validate_source_refs,
)

_TEXT = {"type": "string", "minLength": 1, "maxLength": 4096}
_ID = {"type": "string", "maxLength": 512}
_NULLABLE_SELECTOR = {"type": ["string", "null"], "maxLength": 4096}
# Each native submission admits a bounded batch; the population itself is unbounded
# here and limited only by capture, retained-coverage and transport bounds below.
COMMITTED_BATCH_MAX_ITEMS = 128
# One maximal claim must stay submittable; reuse the existing host argument bound.
COMMITTED_SUBMISSION_MAX_BYTES = _MAX_ARGUMENT_BYTES
# Reasons stay small enough that a full shard of worst-case decisions (four refs
# each) always fits the shared 256 KiB per-scope retained attribution bound.
COMMITTED_REASON_MAX_CHARS = 1024
COVERAGE_SHARD_TARGETS = 12
COVERAGE_MAX_SHARDS = 10000
CATALOG_PAGE_MAX_BYTES = 16 * 1024
LOOKUP_PAGE_MAX_BYTES = 32 * 1024
# Preflight bound for any single tool result: stock Pi read/grep cap output at 50 KB,
# and a lookup page plus its compact mapping stays below this.
LARGEST_TOOL_RESULT_BYTES = 64 * 1024
LOOKUP_TOOL = "review_source"
ADMIT_TOOL = "review_admit"
LOOKUP_SCHEMA = strict_object({
    "target_id": _NULLABLE_SELECTOR, "path": _NULLABLE_SELECTOR,
    "cursor": {"type": ["string", "null"], "maxLength": 256},
})


def _array(items: dict[str, Any], limit: int = 128, **constraints: Any) -> dict[str, Any]:
    return {"type": "array", "maxItems": limit, "items": items, **constraints}


def _claim_schema(*, selectors: bool) -> dict[str, Any]:
    return strict_object({
        "uid": _ID,
        "revision": {"type": "integer", "minimum": 0},
        "source_uids": _array(_ID, uniqueItems=True),
        "title": _TEXT,
        "payload": strict_object({
            **{key: _TEXT for key in ("claim", "trigger", "consequence", "grounds", "file")},
            "line": {"type": ["integer", "null"], "minimum": 1},
            "evidence_refs": SOURCE_SELECTORS_SCHEMA if selectors else SOURCE_REFS_SCHEMA,
        }),
    })


def assignment_schema(kind: str, *, selectors: bool, retained: bool = False) -> dict[str, Any]:
    """Use the same assignment shape for model selectors and canonical host output."""
    fields: dict[str, Any] = (
        {"target_id": _ID, "status": {"enum": ["reviewed", "not_reviewed"]},
         "reason": {"type": "string", "maxLength": COMMITTED_REASON_MAX_CHARS}}
        if kind == "discovery" else
        {"uid": _ID, "revision": {"type": "integer", "minimum": 1},
         "verdict": {"enum": ["confirmed", "rejected", "unresolved"]}, "grounds": _TEXT}
    )
    fields["evidence_refs"] = SOURCE_SELECTORS_SCHEMA if selectors else SOURCE_REFS_SCHEMA
    return strict_object({
        "targets" if kind == "discovery" else "verdicts": _array(strict_object(fields), COMMITTED_BATCH_MAX_ITEMS),
        "claims": _array(_RECORD_SCHEMA if retained else _claim_schema(selectors=selectors)),
    })


def receipt_schema(kind: str) -> dict[str, Any]:
    """Compact canonical answer to one admitted batch; Review alone retains whole state."""
    count = {"type": "integer", "minimum": 0}
    return strict_object({
        "assignment": {"const": kind},
        "final": {"type": "boolean"},
        "admitted": strict_object({"decisions": count, "claims": _array(strict_object({
            "uid": _ID, "revision": {"type": "integer", "minimum": 1}, "title": _TEXT}))}),
        "progress": strict_object({
            "decided": count, "total": count, "remaining": count,
            "next_undecided": _array(_ID, 8), "pending_challenge": count,
        }),
    })


def coverage_shards(assignments: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Partition planned targets in inventory order into fixed retained-coverage scopes."""
    return [assignments[index:index + COVERAGE_SHARD_TARGETS]
            for index in range(0, len(assignments), COVERAGE_SHARD_TARGETS)]


def shard_scope_id(index: int) -> str:
    return f"committed:{index:05d}"


RESULT_VERSIONS = {"committed": 4, "immutable": 5, "local": 6}


def planned_work_id(revision: dict[str, Any], assignments: list[dict[str, Any]]) -> str:
    return hashlib.sha256(canonical_json({"revision": revision, "assignments": assignments}).encode()).hexdigest()


_RECORD_SCHEMA = strict_object({
    **_claim_schema(selectors=False)["properties"],
    "item_uid": _ID,
    "id": {"type": "integer", "minimum": 1},
    "disposition": {"enum": ["confirmed", "rejected", "unresolved"]},
    "challenge_evidence": assignment_schema("challenge", selectors=False)["properties"]["verdicts"],
})
COMMITTED_RESULT_SCHEMA = strict_object({
    "schema_version": {"const": 4},
    "run_id": _TEXT,
    "analyzed_revision": TERMINAL_RESULT_SCHEMA["properties"]["analyzed_revision"],
    "planned_work_id": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    "assignments": _array({
        "type": "object",
        "required": ["target_id", "file", "source_refs"],
        "properties": {
            "target_id": _ID, "file": _TEXT,
            "source_refs": {"type": "array", "minItems": 1, "items": SOURCE_REF_SCHEMA},
        },
    }, COVERAGE_SHARD_TARGETS * COVERAGE_MAX_SHARDS),
    **{key: _array(_RECORD_SCHEMA) for key in ("claims", "findings")},
    "terminal_result": TERMINAL_RESULT_SCHEMA,
    "settings": {"type": "object"},
    "execution": {"type": "array", "items": strict_object({
        "assignment": _TEXT,
        **{key: {"type": "boolean"} for key in ("accepted", "settled", "cleanup_complete")},
        "reason": {"enum": [None, *[reason.value for reason in ReasonCode]]},
    })},
})
IMMUTABLE_COMMITTED_RESULT_SCHEMA = strict_object({
    **COMMITTED_RESULT_SCHEMA["properties"],
    "schema_version": {"const": 5},
    "immutable_input": IMMUTABLE_REVIEW_INPUT_SCHEMA,
})
LOCAL_REVIEW_RESULT_SCHEMA = strict_object({
    **COMMITTED_RESULT_SCHEMA['properties'], 'schema_version': {'const': 6},
})


def validate_committed_result(value: dict[str, Any]) -> None:
    """Reject identity drift, unchecked publication and manufactured completion."""
    try:
        version = value.get("schema_version")
        if type(version) is not int or version not in RESULT_VERSIONS.values():
            raise ValueError("unsupported committed review result version; restart review")
        jsonschema.validate(value, {RESULT_VERSIONS["committed"]: COMMITTED_RESULT_SCHEMA,
                                    RESULT_VERSIONS["immutable"]: IMMUTABLE_COMMITTED_RESULT_SCHEMA,
                                    RESULT_VERSIONS["local"]: LOCAL_REVIEW_RESULT_SCHEMA}[version])
    except jsonschema.ValidationError as exc:
        raise ValueError(f"invalid committed review at {exc.json_path}") from exc
    terminal = value["terminal_result"]
    revision = value["analyzed_revision"]
    if ('source_id' in revision) != (version == RESULT_VERSIONS["local"]):
        raise ValueError('result version does not match captured source identity; restart review')
    if version == RESULT_VERSIONS["immutable"]:
        validate_immutable_input(value["immutable_input"])
        materialized = value["immutable_input"]["materialization"]
        if revision != {"head_sha": materialized["head_sha"],
                        "merge_base_sha": materialized["comparison_base_sha"],
                        "diff_key": materialized["review_diff_key"]}:
            raise ValueError("committed result differs from immutable materialization")
    validate_terminal_result(terminal, expected_head_sha=revision["head_sha"])
    if terminal["run_id"] != value["run_id"] or terminal["analyzed_revision"] != revision:
        raise ValueError("committed result differs from terminal run/source identity")
    assignments = value["assignments"]
    if value["planned_work_id"] != planned_work_id(revision, assignments):
        raise ValueError("committed result differs from complete planned work")
    by_target = {part["target_id"]: part for part in assignments}
    if len(by_target) != len(assignments):
        raise ValueError("duplicate assignment identity")
    shards = coverage_shards(assignments)
    if (
        len(shards) > COVERAGE_MAX_SHARDS
        or [(scope["scope_id"], scope["stack"], scope["shard"], scope["files"])
            for scope in terminal["planned_scopes"]]
        != [(shard_scope_id(index), "committed", index, sorted({part["file"] for part in shard}))
            for index, shard in enumerate(shards)]
        or terminal["required_phases"] != (["challenge"] if assignments else ["no_diff"])
    ):
        raise ValueError("terminal coverage differs from complete planned work")
    shard_targets = {shard_scope_id(index): {part["target_id"] for part in shard}
                     for index, shard in enumerate(shards)}
    expected_refs = [ref for part in assignments for ref in part["source_refs"]]

    def check_refs(refs: list[dict[str, Any]], expected: list[dict[str, Any]], *, assigned: bool = True) -> None:
        if any(
            ref["revision"] != source_revision(revision, ref['side'])
            or ref["role"] == "assigned"
            and ref["artifact_sha256"] != revision["diff_key"]
            for ref in refs
        ):
            raise ValueError("source reference differs from captured revision/diff")
        primary = [ref for ref in refs if ref["role"] == "assigned"]
        if not validate_source_refs(primary, expected, revision=revision, require_assigned=assigned):
            raise ValueError("source reference differs from planned assignment")

    for part in assignments:
        # Inventories retain every planned source; only submitted citations
        # share the four-reference bound. Validate each inventory identity.
        for ref in part["source_refs"]:
            check_refs([ref], [ref])
    for outcome in terminal["stack_outcomes"]:
        decisions = outcome.get("target_decisions", [])
        actual = {decision["target_id"] for decision in decisions}
        expected_targets = shard_targets[outcome["scope_id"]]
        # Every planned target retains a typed decision: unfinished work is explicit per target.
        if actual != expected_targets:
            raise ValueError("terminal target decisions differ from planned assignment")
        for decision in decisions:
            check_refs(
                decision["evidence_refs"],
                by_target[decision["target_id"]]["source_refs"],
                assigned=decision["status"] == "reviewed",
            )
    claims = value["claims"]
    known = {c["uid"] for c in claims}
    if len(known) != len(claims) or len({c["item_uid"] for c in claims}) != len(claims):
        raise ValueError("duplicate claim identity")
    for claim in claims:
        if (
            not claim["uid"]
            or not claim["item_uid"]
            or claim["revision"] < 1
            or claim["uid"] in claim["source_uids"]
            or not set(claim["source_uids"]) <= known
        ):
            raise ValueError("invalid retained claim identity/provenance")
        check_refs(claim["payload"]["evidence_refs"], expected_refs)
        for verdict in claim["challenge_evidence"]:
            check_refs(verdict["evidence_refs"], expected_refs)
            if (
                verdict["uid"] != claim["uid"]
                or verdict["revision"] > claim["revision"]
                or not verdict["grounds"].strip()
            ):
                raise ValueError("invalid retained challenge evidence")
        if claim["disposition"] != "unresolved" and not any(
            verdict["revision"] == claim["revision"] and verdict["verdict"] == claim["disposition"]
            for verdict in claim["challenge_evidence"]
        ):
            raise ValueError("publication/rejection requires current-revision challenge evidence")
    if value["findings"] != [claim for claim in claims if claim["disposition"] == "confirmed"]:
        raise ValueError("findings differ from challenged publication claims")
    native_assignments = {e["assignment"] for e in value["execution"]}
    if terminal["analysis_state"] == "complete" and (
        assignments and "discovery" not in native_assignments
        or claims and "challenge" not in native_assignments
        or any(c["disposition"] == "unresolved" for c in claims)
        or any(
            not e["settled"] or not e["accepted"] or not e["cleanup_complete"] or e["reason"] is not None
            for e in value["execution"]
        )
    ):
        raise ValueError("unfinished obligations/native execution cannot be complete")


@dataclass(frozen=True)
class CommittedReviewResult:
    """Sealed JSON bytes plus independent installation, capture and delivery facts."""

    json_text: str
    output_installed: bool
    capture_state: str
    delivery_state: str = "not_requested"
    output_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = json.loads(self.json_text)
        return value

    @property
    def complete(self) -> bool:
        return self.output_installed and self.to_dict()["terminal_result"]["analysis_state"] == "complete"
