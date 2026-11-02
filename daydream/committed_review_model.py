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
# One maximal claim must stay submittable; reuse the existing host argument bound.
COMMITTED_SUBMISSION_MAX_BYTES = _MAX_ARGUMENT_BYTES
# Instructions ask for batches this size; preflight requires one to fit native output.
COMMITTED_BATCH_OUTPUT_BYTES = 32 * 1024
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
        "targets" if kind == "discovery" else "verdicts": _array(strict_object(fields)),
        "claims": _array(_RECORD_SCHEMA if retained else _claim_schema(selectors=selectors)),
    })


def receipt_schema(kind: str) -> dict[str, Any]:
    """Compact canonical answer to one admitted batch; Review alone retains whole state."""
    count = {"type": "integer", "minimum": 0}
    return strict_object({
        "assignment": {"const": kind},
        "work_id": _ID,
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


# Work records, attribution, per-execution capacity evidence and the cross_assignment phase (#1628).
RESULT_VERSIONS = {"committed": 7, "immutable": 8, "local": 9}
REQUIRED_PHASES = ["challenge", "cross_assignment"]
ASSIGNMENT_POLICIES = frozenset({"broad", "adaptive"})
# Investigation kinds decide owned targets and propose new concerns; challenge judges claim revisions.
INVESTIGATION_KINDS = frozenset({"discovery", "follow_up", "cross_assignment"})
WORK_KINDS = [*sorted(INVESTIGATION_KINDS), "challenge"]


def planned_work_id(revision: dict[str, Any], assignments: list[dict[str, Any]]) -> str:
    return hashlib.sha256(canonical_json({"revision": revision, "assignments": assignments}).encode()).hexdigest()


_VERDICT_ITEMS = assignment_schema("challenge", selectors=False)["properties"]["verdicts"]
_RECORD_SCHEMA = strict_object({
    **_claim_schema(selectors=False)["properties"],
    "item_uid": _ID,
    "id": {"type": "integer", "minimum": 1},
    "disposition": {"enum": ["confirmed", "rejected", "unresolved"]},
    # Originating work and the work that authored the current revision.
    "work_id": _ID, "revised_by": _ID,
    "challenge_evidence": {**_VERDICT_ITEMS, "items": strict_object({
        **_VERDICT_ITEMS["items"]["properties"], "work_id": _ID})},
})
_REVISION_REF = strict_object({"uid": _ID, "revision": {"type": "integer", "minimum": 1}})
_WORK_SCHEMA = strict_object({
    "work_id": _ID, "kind": {"enum": WORK_KINDS}, "parents": {"type": "array", "items": _ID},
    "state": {"enum": ["planned", "open", "accepted", "closed", "superseded"]},
    "targets": {"type": "array", "items": _ID},
    "claims": {"type": "array", "items": _REVISION_REF},
    "superseded_by": {"type": "array", "items": _ID},
    "admitted_targets": {"type": "array", "items": _ID},
    "admitted_claims": {"type": "array", "items": _REVISION_REF},
    "verdicts": {"type": "array", "items": _REVISION_REF},
})
_COUNT = {"type": ["integer", "null"], "minimum": 0}
COMMITTED_RESULT_SCHEMA = strict_object({
    "schema_version": {"const": RESULT_VERSIONS["committed"]},
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
    "work": {"type": "array", "items": _WORK_SCHEMA},
    "terminal_result": TERMINAL_RESULT_SCHEMA,
    "settings": {"type": "object"},
    "execution": {"type": "array", "items": strict_object({
        "assignment": _TEXT, "work_id": {"type": ["string", "null"], "maxLength": 512},
        **{key: {"type": "boolean"} for key in ("accepted", "settled", "cleanup_complete")},
        "reason": {"enum": [None, *[reason.value for reason in ReasonCode]]},
        # Dispatch allocation, every observed native start, and starts beyond the charged allocation.
        **{key: _COUNT for key in ("allocation", "observed_starts", "overshoot")},
    })},
})
IMMUTABLE_COMMITTED_RESULT_SCHEMA = strict_object({
    **COMMITTED_RESULT_SCHEMA["properties"],
    "schema_version": {"const": RESULT_VERSIONS["immutable"]},
    "immutable_input": IMMUTABLE_REVIEW_INPUT_SCHEMA,
})
LOCAL_REVIEW_RESULT_SCHEMA = strict_object({
    **COMMITTED_RESULT_SCHEMA['properties'], 'schema_version': {'const': RESULT_VERSIONS['local']},
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
        [(scope["scope_id"], scope["stack"], scope["shard"], scope["files"])
            for scope in terminal["planned_scopes"]]
        != [(shard_scope_id(index), "committed", index, sorted({part["file"] for part in shard}))
            for index, shard in enumerate(shards)]
        or terminal["required_phases"] != (REQUIRED_PHASES if assignments else ["no_diff"])
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
    work = {record["work_id"]: record for record in value["work"]}
    _validate_work(value["work"], work, by_target, terminal)
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
            or {"uid": claim["uid"], "revision": 1} not in work.get(claim["work_id"], {}).get("admitted_claims", [])
            or {"uid": claim["uid"], "revision": claim["revision"]}
            not in work.get(claim["revised_by"], {}).get("admitted_claims", [])
        ):
            raise ValueError("invalid retained claim identity/provenance")
        check_refs(claim["payload"]["evidence_refs"], expected_refs)
        for verdict in claim["challenge_evidence"]:
            check_refs(verdict["evidence_refs"], expected_refs)
            if (
                verdict["uid"] != claim["uid"]
                or verdict["revision"] > claim["revision"]
                or not verdict["grounds"].strip()
                or work.get(verdict["work_id"], {}).get("kind") != "challenge"
                or {"uid": verdict["uid"], "revision": verdict["revision"]} not in work[verdict["work_id"]]["verdicts"]
                # A current-revision verdict comes from a context distinct from that revision's author.
                or verdict["revision"] == claim["revision"] and verdict["work_id"] == claim["revised_by"]
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
    superseded = {wid for wid, record in work.items() if record["state"] == "superseded"}
    if any(e["work_id"] is not None and e["work_id"] not in work for e in value["execution"]):
        raise ValueError("execution evidence differs from retained work")
    # Current accepted work decides completion; superseded attempts remain evidence only.
    if terminal["analysis_state"] == "complete" and (
        assignments and not native_assignments & INVESTIGATION_KINDS
        or claims and "challenge" not in native_assignments
        or any(c["disposition"] == "unresolved" for c in claims)
        or any(record["state"] != "accepted" for wid, record in work.items() if wid not in superseded)
        or any(
            not e["settled"] or not e["accepted"] or not e["cleanup_complete"] or e["reason"] is not None
            for e in value["execution"] if e["work_id"] not in superseded
        )
    ):
        raise ValueError("unfinished obligations/native execution cannot be complete")


def _validate_work(records: list[dict[str, Any]], work: dict[str, dict[str, Any]],
                   by_target: dict[str, dict[str, Any]], terminal: dict[str, Any]) -> None:
    """Responsibilities are disjoint, transfers are exact and each admission is credited once."""
    if len(work) != len(records):
        raise ValueError("duplicate work identity")
    order = {record["work_id"]: index for index, record in enumerate(records)}
    admitted: dict[str, str] = {}
    held: set[str] = set()
    for index, record in enumerate(records):
        if record["state"] != "superseded":
            if held & set(record["targets"]):
                raise ValueError("current work responsibilities overlap")
            held.update(record["targets"])
        if (
            any(order.get(parent, index) >= index for parent in record["parents"])
            or any(order.get(child, index) <= index for child in record["superseded_by"])
            or (record["state"] == "superseded") != bool(record["superseded_by"])
            or any(record["work_id"] not in work[child]["parents"] for child in record["superseded_by"])
            or not set(record["admitted_targets"]) <= set(record["targets"]) <= by_target.keys()
            or record["kind"] == "challenge" and record["targets"]
            or record["kind"] != "challenge" and (record["claims"] or record["verdicts"])
        ):
            raise ValueError("invalid retained work responsibility or transfer")
        for target in record["admitted_targets"]:
            if admitted.setdefault(target, record["work_id"]) != record["work_id"]:
                raise ValueError("duplicate completion credit for a target decision")
    decided = {d["target_id"]: d for outcome in terminal["stack_outcomes"] for d in outcome.get("target_decisions", [])}
    for target, decision in decided.items():
        if target not in admitted and (decision["status"] != "not_reviewed" or decision["evidence_refs"]
                                       or not decision["reason"].startswith("unsubmitted:")):
            raise ValueError("target decision lacks admitting work attribution")


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
