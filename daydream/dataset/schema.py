"""Versioned raw run and observation contracts for newly collected evidence.

Raw producer payloads retain their existing identities and ordering; these models
never reconstruct a corpus record or reinterpret a verifier/outcome label.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from datetime import datetime
from typing import Annotated, Any, Literal, TypeVar

from jsonschema import Draft202012Validator
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    TypeAdapter,
    ValidationError,
    model_validator,
)

from daydream.atif import Trajectory
from daydream.json_utils import canonical_json
from daydream.training.labeler_versions import reply_evidence_digest

Availability = Literal["available", "unavailable", "unproduced", "withheld", "failed"]


def _timestamp(value: str) -> str:
    if datetime.fromisoformat(value).tzinfo is None:
        raise ValueError("timestamp must include a UTC offset")
    return value


Timestamp = Annotated[str, AfterValidator(_timestamp)]
Digest = Annotated[str, Field(pattern=r"^[a-f0-9]{64}$")]


class _RecordModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)


class Evidence(_RecordModel):
    """Explicit evidence availability, including an intentionally empty value."""

    status: Availability = "unproduced"
    value: Any = None
    reason: str | None = None

    @model_validator(mode="after")
    def check_availability(self) -> Evidence:
        if self.status == "available" and self.value is None:
            raise ValueError("available evidence requires a value")
        if self.status != "available" and self.value is not None:
            raise ValueError("absent evidence must not contain a value")
        _json_value(self.value)
        return self


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
_PAYLOAD_VALIDATORS = {name: Draft202012Validator(shape) for name, shape in _RUN_PAYLOADS.items()}
_JSON: TypeAdapter[JsonValue] = TypeAdapter(JsonValue, config=ConfigDict(strict=True, allow_inf_nan=False))


class RunRecord(_RecordModel):
    """Immutable run identity plus selected frozen producer evidence."""

    schema_version: Literal["daydream.run.v1"] = "daydream.run.v1"
    run_id: str = Field(min_length=1)
    captured_at: Timestamp
    outcome: Literal["success", "failed", "interrupted"]
    original_task: Evidence = Field(default_factory=Evidence)
    final_state: Evidence = Field(default_factory=Evidence)
    recommended_patch: Evidence = Field(default_factory=Evidence)
    trajectories: Evidence = Field(default_factory=Evidence)
    findings: Evidence = Field(default_factory=Evidence)
    verification: Evidence = Field(default_factory=Evidence)
    scoring: Evidence = Field(default_factory=Evidence)
    provenance: dict[str, Any] = Field(default_factory=dict)
    completeness: dict[str, Any] = Field(default_factory=dict)
    trace_id: str | None = None

    @model_validator(mode="after")
    def check_task(self) -> RunRecord:
        _json_value(self.provenance)
        _json_value(self.completeness)
        _validate_run_evidence(self)

        return self


def _json_value(value: Any) -> None:
    _JSON.validate_python(value)


class CapturedCorrection(_RecordModel):
    """Captured redacted content has a digest distinct from the original body."""

    status: Availability
    source_reply_id: str = Field(min_length=1)
    body_sha256: Digest | None = None
    text: str | None = None
    captured_sha256: Digest | None = None
    redaction_provenance: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None

    @model_validator(mode="after")
    def check_content(self) -> CapturedCorrection:
        _json_value(self.redaction_provenance)
        if self.status == "available":
            if self.text is None or not self.redaction_provenance:
                raise ValueError("captured text requires redaction provenance")
            if hashlib.sha256(self.text.encode()).hexdigest() != self.captured_sha256:
                raise ValueError("captured_sha256 does not match redacted text")
        elif self.text is not None or self.captured_sha256 is not None:
            raise ValueError("absent correction text must not contain captured content")
        return self


class RunLabelPayload(_RecordModel):
    """Run outcome inputs; PR merge state never supplies a finding judgment."""

    type: Literal["run-label"] = "run-label"
    label: Literal["accepted", "rejected", "contested", "unknown"]
    reviewer_logins: list[str] = Field(default_factory=list)
    outcome_prior: float | None = None
    outcome_prior_n: int = Field(default=0, ge=0)
    rubric: dict[str, Any] = Field(default_factory=dict)


class FindingJudgmentPayload(_RecordModel):
    """Existing disposition vocabulary with optional derived corpus identity."""

    type: Literal["finding-judgment"] = "finding-judgment"
    disposition: Literal["accepted", "rejected", "ambiguous", "unanswered", "missing", "unknown"]
    rationale: str = Field(min_length=1)
    record_id: Digest | None = None

class EnrichmentPayload(_RecordModel):
    """Later PR/base/license evidence, preserving explicit missing evidence."""

    type: Literal["enrichment"] = "enrichment"
    kind: Literal["pr", "base", "license"]
    evidence: Evidence


ObservationPayload = Annotated[
    RunLabelPayload | FindingJudgmentPayload | EnrichmentPayload,
    Field(discriminator="type"),
]

_MODEL_AUTHOR_RE = re.compile(
    r"(?:^|[-_])(?:claude|gpt|llm|model|classifier|anthropic|openai|codex|gemini)(?:$|[-_0-9])",
    re.IGNORECASE,
)


class ObservationRecord(_RecordModel):
    """Append identity and bitemporal evidence, separate from captured reply text."""

    schema_version: Literal["daydream.observation.v1"] = "daydream.observation.v1"
    observation_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    item_uid: str | None = None
    valid_at: Timestamp
    observed_at: Timestamp
    source: str = Field(min_length=1)
    author: str = Field(min_length=1)
    role: Literal["rater", "adjudicator", "model-suggested", "automatic"]
    policy_version: str = Field(min_length=1)
    rubric_version: str = Field(min_length=1)
    classifier_version: str | None = None
    evidence_digest: Digest
    evidence_digest_scheme: Literal["canonical-json-v1", "reply-evidence-v1"] = "canonical-json-v1"
    semantic_evidence: Any
    correction: CapturedCorrection | None = None
    payload: ObservationPayload
    review_required: bool = False

    @property
    def labeler(self) -> str:
        """The unchanged adjudication reducer names observation authors labelers."""
        return self.author

    @model_validator(mode="after")
    def check_observation(self) -> ObservationRecord:
        _json_value(self.semantic_evidence)
        _json_value(self.payload.model_dump(mode="json"))
        if self.evidence_digest_scheme == "reply-evidence-v1":
            if not isinstance(self.semantic_evidence, list) or any(
                not isinstance(reply, dict) for reply in self.semantic_evidence
            ):
                raise ValueError("reply semantic evidence must be a list of objects")
            expected = reply_evidence_digest(self.semantic_evidence)
            if self.correction is not None:
                matching = [reply for reply in self.semantic_evidence
                            if str(reply.get("reply_id", reply.get("id", "")))
                            == self.correction.source_reply_id]
                if len(matching) != 1:
                    raise ValueError("correction references an unknown or conflicting source reply")
                original_hash = matching[0].get("body_sha256")
                if original_hash != self.correction.body_sha256:
                    raise ValueError("correction original body hash differs from source evidence")
        else:
            expected = semantic_evidence_digest(self.semantic_evidence)
        if expected != self.evidence_digest:
            raise ValueError("semantic evidence_digest does not match its projection")
        if self.payload.type == "finding-judgment":
            if self.semantic_evidence is None:
                raise ValueError("finding judgment requires explicit semantic evidence")
            if not self.item_uid:
                raise ValueError("finding judgment requires item_uid")
        elif self.item_uid is not None:
            raise ValueError("run label/enrichment cannot target an item")
        if self.role in ("rater", "adjudicator") and _MODEL_AUTHOR_RE.search(self.author):
            raise ValueError("model author cannot hold a human role")
        if self.role == "model-suggested":
            object.__setattr__(self, "review_required", True)
        return self


def semantic_evidence_digest(value: Any) -> str:
    """New canonical projection digest; existing replies use their own scheme."""
    _json_value(value)
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


Record = RunRecord | ObservationRecord
_Model = TypeVar("_Model", bound=_RecordModel)


def _parse(raw: Mapping[str, Any] | str | bytes, model: type[_Model]) -> _Model:
    try:
        value = json.loads(raw) if isinstance(raw, (str, bytes)) else dict(raw)
        if not isinstance(value, dict) or "schema_version" not in value:
            raise ValueError(f"invalid {model.__name__}: schema_version is required")
        return model.model_validate(value)
    except ValidationError as exc:
        # Report only declared top-level fields, never rejected keys or values.
        fields = ", ".join(str(error["loc"][0]) for error in exc.errors(include_input=False)
                           if error["loc"] and error["loc"][0] in model.model_fields)
        raise ValueError(f"invalid {model.__name__}: {fields or 'record'}") from None
    except (TypeError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"invalid {model.__name__}: malformed JSON object") from exc


def parse_run(raw: Mapping[str, Any] | str | bytes) -> RunRecord:
    """Read a validated current-version run; unknown versions fail closed."""
    return _parse(raw, RunRecord)


def parse_observation(raw: Mapping[str, Any] | str | bytes) -> ObservationRecord:
    """Read a discriminated current-version observation without legacy coercion."""
    return _parse(raw, ObservationRecord)


def serialize_record(record: Record) -> dict[str, Any]:
    """Return owned JSON containers after validating the complete record again."""
    payload = record.model_dump(mode="json")
    type(record).model_validate(payload)
    return payload


def canonical_record_json(record: Record) -> str:
    """Canonical complete-record bytes shared by identity and shard pinning."""
    return canonical_json(serialize_record(record))


def _validate_run_evidence(run: RunRecord) -> None:
    """Validate complete raw sections without rewriting producer payloads."""
    for name, validator in _PAYLOAD_VALIDATORS.items():
        section = getattr(run, name)
        if section.status == "available" and not validator.is_valid(section.value):
            raise ValueError(f"invalid {name} payload")
    for name, text_key, digest_key in (("original_task", "diff", "diff_sha256"),
                                      ("recommended_patch", "patch", "sha256")):
        section = getattr(run, name)
        if section.status == "available" and section.value[digest_key] != hashlib.sha256(
            section.value[text_key].encode()).hexdigest():
            raise ValueError(f"{name} digest does not match captured content")
    if run.trajectories.status == "available":
        payload = run.trajectories.value
        _timestamp(payload["cutoff_at"])
        documents: dict[str, dict[str, Any]] = {}
        for document in payload["documents"]:
            Trajectory.model_validate(document)
            identity = document.get("trajectory_id")
            if not isinstance(identity, str) or not identity or identity in documents:
                raise ValueError("trajectory documents need unique identities")
            if document.get("session_id") != run.run_id:
                raise ValueError("trajectory session_id differs from run_id")
            documents[identity] = document
        if payload["root_trajectory_id"] != run.run_id or run.run_id not in documents:
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
    if run.findings.status == "available":
        findings = run.findings.value
        claims = [claim["uid"] for stack in findings["claims"] for claim in (
            stack["records"]["issues"] if isinstance(stack["records"], dict) else stack["records"])]
        items = {item["item_uid"] for item in findings["items"]}
        if len(set(claims)) != len(claims) or len(items) != len(findings["items"]):
            raise ValueError("findings contain duplicate identities")
        if any(len(set(item["source_uids"])) != len(item["source_uids"]) for item in findings["items"]):
            raise ValueError("finding source_uids contain duplicate identities")
    if run.verification.status == "available":
        verification = run.verification.value
        for association in verification["item_associations"]:
            if association.get("item_uid") not in items:
                raise ValueError("verification association references an absent finding item")
    if run.scoring.status == "available":
        scoring = run.scoring.value
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


def run_record_schema() -> dict[str, Any]:
    """Publish the run contract; parsing also validates links and digests."""
    schema = RunRecord.model_json_schema()
    schema["required"].append("schema_version")
    for name, payload in _RUN_PAYLOADS.items():
        schema["properties"][name] = {"allOf": [schema["properties"][name], {
            "if": {"properties": {"status": {"const": "available"}}, "required": ["status"]},
            "then": {"properties": {"value": payload}, "required": ["value"]},
            "else": {"properties": {"value": {"type": "null"}}}}]}
    return schema


def observation_record_schema() -> dict[str, Any]:
    """Publish the typed history contract; parsing also verifies evidence digests."""
    schema = ObservationRecord.model_json_schema()
    schema["required"].append("schema_version")
    return schema
