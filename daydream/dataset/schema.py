"""Versioned JSON contracts for raw run evidence and typed observation history."""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from typing import Any

from jsonschema import Draft202012Validator
from pydantic import JsonValue, TypeAdapter

from daydream.atif import Trajectory
from daydream.json_utils import canonical_json
from daydream.training.labeler_versions import reply_evidence_digest

Record = dict[str, Any]
_JSON: TypeAdapter[JsonValue] = TypeAdapter(JsonValue, config={"strict": True, "allow_inf_nan": False})
_MODEL_AUTHOR_RE = re.compile(
    r"(?:^|[-_])(?:claude|gpt|llm|model|classifier|anthropic|openai|codex|gemini)(?:$|[-_0-9])", re.IGNORECASE)


def _timestamp(value: str) -> str:
    if datetime.fromisoformat(value).tzinfo is None:
        raise ValueError("timestamp must include a UTC offset")
    return value


def _object(*, optional: tuple[str, ...] = (), extra: bool = False, **properties: Any) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "additionalProperties": extra,
            "required": [name for name in properties if name not in optional]}


def _array(items: dict[str, Any]) -> dict[str, Any]:
    return {"type": "array", "items": items}


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


_STRING = {"type": "string"}
_ID = {"type": "string", "minLength": 1}
_SHA = {"type": "string", "pattern": "^[a-f0-9]{64}$"}
_NUMBER = {"type": "number"}
_BOOL = {"type": "boolean"}
_MAP = {"type": "object"}
_COUNT = {"type": "integer", "minimum": 0}
_CLAIM = _object(extra=True, uid=_ID)
_ITEM = _object(extra=True, item_uid=_ID, source_uids=_array(_ID), fingerprint=_SHA)
_BREAKDOWN = _object(optional=("false_positive_penalty", "posterior_cost", "outcome_prior", "outcome_prior_n"),
    correctness_per_finding=_nullable(_array(_NUMBER)), format_valid=_BOOL, length_penalty=_nullable(_NUMBER),
    composite=_nullable(_NUMBER), axes_present={"type": "object", "additionalProperties": _BOOL}, reward_version=_ID,
    false_positive_penalty=_nullable(_NUMBER), posterior_cost=_nullable(_NUMBER),
    outcome_prior=_nullable(_NUMBER), outcome_prior_n=_nullable(_COUNT))
_RUN_PAYLOADS = {
    "original_task": _object(optional=("source_diff_sha256", "redaction"),
        analyzed_revision=_object(optional=("pr_base_sha",), head_sha=_ID, merge_base_sha=_ID, diff_key=_ID,
                                  pr_base_sha=_nullable(_STRING)),
        diff=_STRING, diff_sha256=_SHA, source_diff_sha256=_nullable(_SHA), redaction=_nullable(_MAP),
        repository=_object(repo_slug=_nullable(_STRING), remote_url=_nullable(_STRING), host=_nullable(_STRING)),
        pr=_object(number=_nullable({"type": "integer", "minimum": 1}), repo=_nullable(_STRING)),
        changed_files=_array(_STRING), input_scope={"enum": ["committed", "committed_and_tracked_worktree"]},
        dirty_tracked=_nullable(_BOOL), untracked_files_included={"const": False}),
    "final_state": _object(head_sha=_nullable(_STRING)),
    "recommended_patch": _object(patch=_STRING, sha256=_SHA, source_sha256=_nullable(_SHA), capture=_nullable(_MAP)),
    "trajectories": _object(documents={**_array(_MAP), "minItems": 1}, root_trajectory_id=_ID,
                            status={"enum": ["complete", "partial"]}, cutoff_at=_STRING),
    "findings": _object(claims=_array(_object(stack=_ID, records={"anyOf": [
        _object(extra=True, issues=_array(_CLAIM)), _array(_CLAIM)]})),
        items=_array(_ITEM), derivation=_MAP, terminal_coverage=_nullable(_MAP)),
    "verification": {**_object(extra=True, item_associations=_array(_MAP)), "minProperties": 2},
    "scoring": _object(optional=("review_text_redaction", "source_review_sha256"),
        verifier_verdicts=_nullable(_array(_MAP)), format_valid=_BOOL, review_text=_nullable(_STRING),
        length=_nullable(_COUNT), persisted_breakdown=_BREAKDOWN, posterior_cost=_nullable(_NUMBER),
        reward_policy=_object(version=_ID, configuration=_object(
            w_len=_NUMBER, w_fp=_NUMBER, len_tau=_NUMBER, len_scale={"type": "number", "exclusiveMinimum": 0},
            verdict_map={"type": "object", "additionalProperties": _NUMBER},
            fp_penalty_map={"type": "object", "additionalProperties": _NUMBER})),
        review_text_redaction=_nullable(_MAP), source_review_sha256=_nullable(_SHA)),
}
_AVAILABILITY = {"enum": ["available", "unavailable", "unproduced", "withheld", "failed"]}
_EVIDENCE = _object(optional=("value", "reason"), status=_AVAILABILITY, value={}, reason=_nullable(_STRING))
_EVIDENCE["if"] = {"properties": {"status": {"const": "available"}}}
_EVIDENCE["then"] = {"required": ["value"], "properties": {"value": {"not": {"type": "null"}}}}
_EVIDENCE["else"] = {"properties": {"value": {"type": "null"}}}
_RUN_SCHEMA = _object(schema_version={"const": "daydream.run.v1"}, run_id=_ID, captured_at=_STRING,
    outcome={"enum": ["success", "failed", "interrupted"]}, provenance=_MAP, completeness=_MAP,
    trace_id=_nullable(_STRING))
_RUN_SCHEMA["properties"].update({name: {"allOf": [_EVIDENCE, {
    "if": {"properties": {"status": {"const": "available"}}},
    "then": {"properties": {"value": payload}}}]} for name, payload in _RUN_PAYLOADS.items()})
_RUN_SCHEMA["required"].extend(_RUN_PAYLOADS)
_CORRECTION = _object(optional=("body_sha256", "text", "captured_sha256", "redaction_provenance", "reason"),
    status=_AVAILABILITY, source_reply_id=_ID, body_sha256=_nullable(_SHA),
    text=_nullable(_STRING), captured_sha256=_nullable(_SHA), redaction_provenance=_MAP, reason=_nullable(_STRING))
_OBSERVATION_SCHEMA = _object(
    optional=("item_uid", "classifier_version", "correction", "evidence_digest_scheme", "review_required"),
    schema_version={"const": "daydream.observation.v1"}, observation_id=_ID, run_id=_ID,
    item_uid=_nullable(_STRING), valid_at=_STRING, observed_at=_STRING, source=_ID, author=_ID,
    role={"enum": ["rater", "adjudicator", "model-suggested", "automatic"]}, policy_version=_ID, rubric_version=_ID,
    classifier_version=_nullable(_STRING), evidence_digest=_SHA, semantic_evidence={},
    correction=_nullable(_CORRECTION),
    evidence_digest_scheme={"enum": ["canonical-json-v1", "reply-evidence-v1"]}, review_required=_BOOL,
    payload={"oneOf": [
        _object(optional=("reviewer_logins", "outcome_prior", "outcome_prior_n", "rubric"),
            type={"const": "run-label"}, label={"enum": ["accepted", "rejected", "contested", "unknown"]},
            reviewer_logins=_array(_STRING), outcome_prior=_nullable(_NUMBER), outcome_prior_n=_COUNT, rubric=_MAP),
        _object(optional=("record_id",), type={"const": "finding-judgment"}, disposition={"enum": [
            "accepted", "rejected", "ambiguous", "unanswered", "missing", "unknown"]},
            rationale=_ID, record_id=_nullable(_SHA)),
        _object(type={"const": "enrichment"}, kind={"enum": ["pr", "base", "license"]}, evidence=_EVIDENCE)]})
_MEMBER = _object(identity=_ID, record_digest=_SHA, shard={"type": "string", "pattern": "^[a-f0-9]{64}\\.jsonl$"},
                  shard_digest=_SHA)
_SNAPSHOT_SCHEMA = _object(schema_version={"const": "daydream.snapshot.v1"}, snapshot_id=_SHA,
    observed_before=_STRING, valid_before=_nullable(_STRING), runs=_array(_MEMBER), observations=_array(_MEMBER))
_VALIDATORS = {"daydream.run.v1": Draft202012Validator(_RUN_SCHEMA),
               "daydream.observation.v1": Draft202012Validator(_OBSERVATION_SCHEMA),
               "daydream.snapshot.v1": Draft202012Validator(_SNAPSHOT_SCHEMA)}


def _defaults(value: Any, defaults: Record) -> Any:
    return {**defaults, **value} if isinstance(value, dict) else value


def _parse(raw: Mapping[str, Any] | str | bytes, version: str) -> Record:
    name = {"daydream.run.v1": "RunRecord", "daydream.observation.v1": "ObservationRecord"}.get(version, "Snapshot")
    try:
        value = json.loads(raw) if isinstance(raw, (str, bytes)) else dict(raw)
        if not isinstance(value, dict) or value.get("schema_version") != version or version not in _VALIDATORS:
            raise ValueError("schema_version")
        evidence_default = {"status": "unproduced", "value": None, "reason": None}
        if version == "daydream.run.v1":
            value = _defaults(value, {"provenance": {}, "completeness": {}, "trace_id": None})
            for field in _RUN_PAYLOADS:
                value[field] = _defaults(value.get(field, {}), evidence_default)
            _timestamp(value["captured_at"])
        elif version == "daydream.observation.v1":
            _timestamp(value["valid_at"])
            _timestamp(value["observed_at"])
        _JSON.validate_python(value)
        if not _VALIDATORS[version].is_valid(value):
            raise ValueError("record")
        if version == "daydream.run.v1":
            _validate_run_evidence(value)
        elif version == "daydream.observation.v1":
            _validate_observation(value)
        else:
            _timestamp(value["observed_before"])
            if value["valid_before"] is not None:
                _timestamp(value["valid_before"])
            for kind in ("runs", "observations"):
                if len({member["identity"] for member in value[kind]}) != len(value[kind]):
                    raise ValueError("duplicate snapshot members")
        return dict(json.loads(canonical_json(value)))
    except (ValueError, TypeError, KeyError, UnicodeError) as error:
        field = "schema_version" if str(error) == "schema_version" else "record"
        raise ValueError(f"invalid {name}: {field}") from None


def parse_run(raw: Mapping[str, Any] | str | bytes) -> Record:
    return _parse(raw, "daydream.run.v1")


def parse_observation(raw: Mapping[str, Any] | str | bytes) -> Record:
    return _parse(raw, "daydream.observation.v1")


def parse_snapshot(raw: Mapping[str, Any] | str | bytes) -> Record:
    return _parse(raw, "daydream.snapshot.v1")


def semantic_evidence_digest(value: Any) -> str:
    _JSON.validate_python(value)
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _validate_observation(record: Record) -> None:
    evidence, correction = record["semantic_evidence"], record.get("correction")
    if correction is not None:
        if correction["status"] == "available":
            if correction.get("text") is None or not correction.get("redaction_provenance") or hashlib.sha256(
                correction["text"].encode()).hexdigest() != correction.get("captured_sha256"):
                raise ValueError("invalid captured correction")
        elif correction.get("text") is not None or correction.get("captured_sha256") is not None:
            raise ValueError("absent correction contains captured content")
    if record.get("evidence_digest_scheme", "canonical-json-v1") == "reply-evidence-v1":
        if not isinstance(evidence, list) or any(not isinstance(reply, dict) for reply in evidence):
            raise ValueError("reply semantic evidence must be a list of objects")
        expected = reply_evidence_digest(evidence)
        if correction is not None:
            matching = [reply for reply in evidence if str(reply.get("reply_id", reply.get("id", "")))
                        == correction["source_reply_id"]]
            if len(matching) != 1 or matching[0].get("body_sha256") != correction.get("body_sha256"):
                raise ValueError("correction conflicts with source evidence")
    else:
        expected = semantic_evidence_digest(evidence)
    if expected != record["evidence_digest"]:
        raise ValueError("semantic evidence digest mismatch")
    if record["payload"]["type"] == "finding-judgment":
        if evidence is None or not record.get("item_uid"):
            raise ValueError("finding judgment requires evidence and an item target")
    elif record.get("item_uid") is not None:
        raise ValueError("run-level observation cannot target a finding")
    if record["role"] in ("rater", "adjudicator") and _MODEL_AUTHOR_RE.search(record["author"]):
        raise ValueError("model author cannot hold a human role")
    if record["role"] == "model-suggested":
        record["review_required"] = True


def _validate_run_evidence(run: Record) -> None:
    """Validate complete raw sections without rewriting producer payloads."""
    for name, text_key, digest_key in (("original_task", "diff", "diff_sha256"),
                                      ("recommended_patch", "patch", "sha256")):
        section = run[name]
        if section["status"] == "available" and section["value"][digest_key] != hashlib.sha256(
            section["value"][text_key].encode()).hexdigest():
            raise ValueError(f"{name} digest does not match captured content")
    if run["trajectories"]["status"] == "available":
        payload = run["trajectories"]["value"]
        _timestamp(payload["cutoff_at"])
        documents: dict[str, dict[str, Any]] = {}
        for document in payload["documents"]:
            Trajectory.model_validate(document)
            identity = document.get("trajectory_id")
            if not isinstance(identity, str) or not identity or identity in documents:
                raise ValueError("trajectory documents need unique identities")
            if document.get("session_id") != run["run_id"]:
                raise ValueError("trajectory session_id differs from run_id")
            documents[identity] = document
        if payload["root_trajectory_id"] != run["run_id"] or run["run_id"] not in documents:
            raise ValueError("trajectory root identity differs from run_id")
        invocation_ids: dict[str, dict[str, Any]] = {}
        for identity, document in documents.items():
            summaries = (document.get("extra") or {}).get("subtrajectories", [])
            if not isinstance(summaries, list):
                raise ValueError("trajectory invocation summaries must be an ordered array")
            for summary in summaries:
                if not isinstance(summary, dict):
                    raise ValueError("trajectory invocation summary must be a mapping")
                if "invocation_id" in summary:
                    _validate_invocation(summary, documents, invocation_ids)
                elif "trajectory_id" in summary:
                    # A registered fork can fail before producing a document.
                    # Keep its recorded identity without inventing a child.
                    for invocation in summary.get("invocations", []):
                        _validate_invocation(invocation, documents, invocation_ids)
    items: set[str] = set()
    if run["findings"]["status"] == "available":
        findings = run["findings"]["value"]
        claims = [claim["uid"] for stack in findings["claims"] for claim in (
            stack["records"]["issues"] if isinstance(stack["records"], dict) else stack["records"])]
        items = {item["item_uid"] for item in findings["items"]}
        if len(set(claims)) != len(claims) or len(items) != len(findings["items"]):
            raise ValueError("findings contain duplicate identities")
        if any(len(set(item["source_uids"])) != len(item["source_uids"]) for item in findings["items"]):
            raise ValueError("finding source_uids contain duplicate identities")
    if run["verification"]["status"] == "available":
        verification = run["verification"]["value"]
        for association in verification["item_associations"]:
            if association.get("item_uid") not in items:
                raise ValueError("verification association references an absent finding item")
    if run["scoring"]["status"] == "available":
        scoring = run["scoring"]["value"]
        if scoring["persisted_breakdown"]["reward_version"] != scoring["reward_policy"]["version"]:
            raise ValueError("reward breakdown version differs from persisted policy")
        if scoring["persisted_breakdown"]["format_valid"] != scoring["format_valid"]:
            raise ValueError("reward breakdown format gate differs from scoring input")
        if not {"correctness", "length"} <= scoring["persisted_breakdown"]["axes_present"].keys():
            raise ValueError("reward breakdown requires explicit correctness/length presence")
        if scoring["review_text"] is not None and scoring["length"] != len(scoring["review_text"]):
            provenance = scoring.get("review_text_redaction") or {}
            if not provenance.get("applied") or not provenance.get("policy"):
                raise ValueError("scoring length differs from retained review text without redaction provenance")


def _validate_invocation(
    invocation: dict[str, Any], documents: dict[str, dict[str, Any]],
    known: dict[str, dict[str, Any]],
) -> None:
    invocation_id = invocation.get("invocation_id")
    document = documents.get(invocation.get("trajectory_id", ""))
    if not isinstance(invocation_id, str) or not invocation_id or document is None:
        raise ValueError("invocation requires identity and an existing document")
    if invocation_id in known and known[invocation_id] != invocation:
        raise ValueError("invocation identity has conflicting summaries")
    step_ids = invocation.get("step_ids")
    actual = {step["step_id"] for step in document["steps"]}
    if (not isinstance(step_ids, list) or any(type(step_id) is not int for step_id in step_ids)
            or len(set(step_ids)) != len(step_ids) or not set(step_ids) <= actual):
        raise ValueError("invocation step membership references absent or duplicate steps")
    known[invocation_id] = invocation


def run_record_schema() -> Record:
    return deepcopy(_RUN_SCHEMA)


def observation_record_schema() -> Record:
    return deepcopy(_OBSERVATION_SCHEMA)
