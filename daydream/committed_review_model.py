"""Canonical claims and the sealed committed-review contract.

Semantic payloads identify a claim revision. Titles are presentation; neither a
native receipt nor a previous revision's verdict grants publication authority.
"""

from __future__ import annotations

import hashlib
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import jsonschema

from daydream.json_utils import canonical_json
from daydream.output_schema import strict_object
from daydream.review_result import TERMINAL_RESULT_SCHEMA, ReasonCode, validate_terminal_result
from daydream.review_source_refs import (
    SOURCE_REF_SCHEMA,
    SOURCE_REFS_SCHEMA,
    SOURCE_SELECTORS_SCHEMA,
    validate_source_refs,
)

_TEXT = {"type": "string", "minLength": 1, "maxLength": 4096}
_ID = {"type": "string", "maxLength": 512}
COMMITTED_ASSIGNMENT_LIMIT = 512


def claim_schema(*, selectors: bool) -> dict[str, Any]:
    return strict_object(
        {
            "uid": _ID,
            "revision": {"type": "integer", "minimum": 0},
            "source_uids": {"type": "array", "maxItems": 128, "uniqueItems": True, "items": _ID},
            "title": _TEXT,
            "payload": strict_object(
                {
                    **{key: _TEXT for key in ("claim", "trigger", "consequence", "grounds", "file")},
                    "line": {"type": ["integer", "null"], "minimum": 1},
                    "evidence_refs": SOURCE_SELECTORS_SCHEMA if selectors else SOURCE_REFS_SCHEMA,
                }
            ),
        }
    )


def discovery_schema(*, selectors: bool) -> dict[str, Any]:
    return strict_object(
        {
            "targets": {
                "type": "array",
                "maxItems": COMMITTED_ASSIGNMENT_LIMIT,
                "items": strict_object(
                    {
                        "target_id": _ID,
                        "status": {"enum": ["reviewed", "not_reviewed"]},
                        "reason": {"type": "string", "maxLength": 4096},
                        "evidence_refs": SOURCE_SELECTORS_SCHEMA if selectors else SOURCE_REFS_SCHEMA,
                    }
                ),
            },
            "claims": {"type": "array", "maxItems": 128, "items": claim_schema(selectors=selectors)},
        }
    )


def challenge_schema(*, selectors: bool) -> dict[str, Any]:
    return strict_object(
        {
            "verdicts": {
                "type": "array",
                "maxItems": 128,
                "items": strict_object(
                    {
                        "uid": _ID,
                        "revision": {"type": "integer", "minimum": 1},
                        "verdict": {"enum": ["confirmed", "rejected", "unresolved"]},
                        "grounds": _TEXT,
                        "evidence_refs": SOURCE_SELECTORS_SCHEMA if selectors else SOURCE_REFS_SCHEMA,
                    }
                ),
            },
            "claims": {"type": "array", "maxItems": 128, "items": claim_schema(selectors=selectors)},
        }
    )


def semantic_revision(payload: dict[str, Any]) -> str:
    """All canonical semantic fields participate; presentation is outside this hash."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def planned_work_id(revision: dict[str, Any], assignments: list[dict[str, Any]]) -> str:
    return hashlib.sha256(canonical_json({"revision": revision, "assignments": assignments}).encode()).hexdigest()


_RECORD_SCHEMA = claim_schema(selectors=False)
_RECORD_SCHEMA["properties"].update(
    {
        "item_uid": _ID,
        "id": {"type": "integer", "minimum": 1},
        "disposition": {"enum": ["confirmed", "rejected", "unresolved"]},
        "challenge_evidence": challenge_schema(selectors=False)["properties"]["verdicts"],
    }
)
_RECORD_SCHEMA["required"] = list(_RECORD_SCHEMA["properties"])
COMMITTED_RESULT_SCHEMA = strict_object(
    {
        "schema_version": {"const": 1},
        "run_id": _TEXT,
        "analyzed_revision": TERMINAL_RESULT_SCHEMA["properties"]["analyzed_revision"],
        "planned_work_id": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "assignments": {
            "type": "array",
            "maxItems": COMMITTED_ASSIGNMENT_LIMIT,
            "items": {
                "type": "object",
                "required": ["target_id", "file", "source_refs"],
                "properties": {
                    "target_id": _ID, "file": _TEXT,
                    "source_refs": {"type": "array", "minItems": 1, "items": SOURCE_REF_SCHEMA},
                },
            },
        },
        "claims": {"type": "array", "maxItems": 128, "items": _RECORD_SCHEMA},
        "findings": {"type": "array", "maxItems": 128, "items": _RECORD_SCHEMA},
        "terminal_result": TERMINAL_RESULT_SCHEMA,
        "settings": {"type": "object"},
        "execution": {
            "type": "array",
            "items": strict_object(
                {
                    "assignment": _TEXT,
                    **{key: {"type": "boolean"} for key in ("accepted", "settled", "cleanup_complete")},
                    "reason": {"enum": [None, *[reason.value for reason in ReasonCode]]},
                }
            ),
        },
    }
)


def validate_committed_result(value: dict[str, Any]) -> None:
    """Reject identity drift, unchecked publication and manufactured completion."""
    try:
        jsonschema.validate(value, COMMITTED_RESULT_SCHEMA)
    except jsonschema.ValidationError as exc:
        raise ValueError(f"invalid committed review at {exc.json_path}") from exc
    terminal = value["terminal_result"]
    revision = value["analyzed_revision"]
    validate_terminal_result(terminal, expected_head_sha=revision["head_sha"])
    if terminal["run_id"] != value["run_id"] or terminal["analyzed_revision"] != revision:
        raise ValueError("committed result differs from terminal run/source identity")
    assignments = value["assignments"]
    if value["planned_work_id"] != planned_work_id(revision, assignments):
        raise ValueError("committed result differs from complete planned work")
    by_target = {part["target_id"]: part for part in assignments}
    if len(by_target) != len(assignments):
        raise ValueError("duplicate assignment identity")
    planned_files = sorted({part["file"] for part in assignments})
    if (
        [scope["files"] for scope in terminal["planned_scopes"]] != ([planned_files] if assignments else [])
        or [scope["scope_id"] for scope in terminal["planned_scopes"]] != (["committed"] if assignments else [])
        or terminal["required_phases"] != (["challenge"] if assignments else ["no_diff"])
    ):
        raise ValueError("terminal coverage differs from complete planned work")
    expected_refs = [ref for part in assignments for ref in part["source_refs"]]

    def check_refs(refs: list[dict[str, Any]], expected: list[dict[str, Any]], *, assigned: bool = True) -> None:
        if any(
            ref["revision"] != revision["merge_base_sha" if ref["side"] == "old" else "head_sha"]
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
        if not actual <= by_target.keys() or outcome["status"] == "complete" and actual != by_target.keys():
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
        import json

        return deepcopy(json.loads(self.json_text))

    @property
    def complete(self) -> bool:
        return self.output_installed and self.to_dict()["terminal_result"]["analysis_state"] == "complete"
