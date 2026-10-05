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

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from daydream.atif import Trajectory
from daydream.training.labeler_versions import reply_evidence_digest

Availability = Literal["available", "unavailable", "unproduced", "withheld", "failed"]


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


class OriginalTaskEvidence(Evidence):
    """Original analyzed task evidence, never finalization-time Git state."""


class AnalyzedRevisionPayload(_RecordModel):
    head_sha: str = Field(min_length=1)
    merge_base_sha: str = Field(min_length=1)
    diff_key: str = Field(min_length=1)
    pr_base_sha: str | None = None


class RepositoryPayload(_RecordModel):
    repo_slug: str | None
    remote_url: str | None
    host: str | None


class PullRequestPayload(_RecordModel):
    number: int | None = Field(ge=1)
    repo: str | None


class OriginalTaskPayload(_RecordModel):
    analyzed_revision: AnalyzedRevisionPayload
    diff: str
    diff_sha256: str
    repository: RepositoryPayload
    pr: PullRequestPayload
    changed_files: list[str]
    input_scope: Literal["committed", "committed_and_tracked_worktree"]
    dirty_tracked: bool | None
    untracked_files_included: Literal[False]
    source_diff_sha256: str | None = None
    redaction: dict[str, Any] | None = None


class TrajectoriesPayload(_RecordModel):
    documents: list[dict[str, Any]] = Field(min_length=1)
    root_trajectory_id: str = Field(min_length=1)
    status: Literal["complete", "partial"]
    cutoff_at: str


class StackClaimsPayload(_RecordModel):
    stack: str = Field(min_length=1)
    records: dict[str, Any] | list[dict[str, Any]]


class CanonicalItemPayload(BaseModel):
    # Existing producer fields stay raw, including verifier and fix decisions.
    model_config = ConfigDict(extra="allow", strict=True)
    item_uid: str = Field(min_length=1)
    source_uids: list[str]
    fingerprint: str


class FindingsPayload(_RecordModel):
    claims: list[StackClaimsPayload]
    items: list[dict[str, Any]]
    derivation: dict[str, Any]
    terminal_coverage: dict[str, Any] | None


class RewardConfigurationPayload(_RecordModel):
    w_len: float
    w_fp: float
    len_tau: float
    len_scale: float = Field(gt=0)
    verdict_map: dict[str, float]
    fp_penalty_map: dict[str, float]


class RewardPolicyPayload(_RecordModel):
    version: str = Field(min_length=1)
    configuration: RewardConfigurationPayload


class RewardBreakdownPayload(_RecordModel):
    correctness_per_finding: list[float] | None
    format_valid: bool
    length_penalty: float | None
    composite: float | None
    axes_present: dict[str, bool]
    reward_version: str = Field(min_length=1)
    false_positive_penalty: float | None = None
    posterior_cost: float | None = None
    outcome_prior: float | None = None
    outcome_prior_n: int | None = Field(default=None, ge=0)


class ScoringPayload(_RecordModel):
    verifier_verdicts: list[dict[str, Any]] | None
    format_valid: bool
    review_text: str | None
    length: int | None = Field(ge=0)
    reward_policy: RewardPolicyPayload
    persisted_breakdown: RewardBreakdownPayload
    posterior_cost: float | None
    review_text_redaction: dict[str, Any] | None = None
    source_review_sha256: str | None = None


class RecommendedPatchPayload(_RecordModel):
    patch: str
    sha256: str
    source_sha256: str | None
    capture: dict[str, Any] | None


class FinalStatePayload(_RecordModel):
    head_sha: str | None


class RunRecord(_RecordModel):
    """Immutable run identity plus selected frozen producer evidence."""

    schema_version: Literal["daydream.run.v1"] = "daydream.run.v1"
    run_id: str = Field(min_length=1)
    captured_at: str
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

    @field_validator("captured_at")
    @classmethod
    def check_time(cls, value: str) -> str:
        _timestamp(value)
        return value

    @model_validator(mode="after")
    def check_task(self) -> RunRecord:
        _json_value(self.provenance)
        _json_value(self.completeness)
        _validate_run_evidence(self)

        return self


def _timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("timestamp must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a UTC offset")
    return parsed


def _json_value(value: Any) -> None:
    """Reject Python-only and non-finite values before canonical serialization."""
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        for child in value.values():
            _json_value(child)
    elif isinstance(value, list):
        for child in value:
            _json_value(child)
    elif value is not None and type(value) not in (str, bool, int, float):
        raise ValueError("evidence must contain only JSON values")
    try:
        json.dumps(value, allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValueError("evidence must contain finite JSON values") from exc


class CapturedCorrection(_RecordModel):
    """Captured redacted content has a digest distinct from the original body."""

    status: Availability
    source_reply_id: str = Field(min_length=1)
    body_sha256: str | None = None
    text: str | None = None
    captured_sha256: str | None = None
    redaction_provenance: dict[str, Any] = Field(default_factory=dict)
    reason: str | None = None

    @model_validator(mode="after")
    def check_content(self) -> CapturedCorrection:
        _json_value(self.redaction_provenance)
        if self.body_sha256 is not None:
            _digest(self.body_sha256)
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
    record_id: str | None = None

    @field_validator("record_id")
    @classmethod
    def check_identity(cls, value: str | None) -> str | None:
        if value is not None:
            _digest(value)
        return value


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
    valid_at: str
    observed_at: str
    source: str = Field(min_length=1)
    author: str = Field(min_length=1)
    role: Literal["rater", "adjudicator", "model-suggested", "automatic"]
    policy_version: str = Field(min_length=1)
    rubric_version: str = Field(min_length=1)
    classifier_version: str | None = None
    evidence_digest: str
    evidence_digest_scheme: Literal["canonical-json-v1", "reply-evidence-v1"] = "canonical-json-v1"
    semantic_evidence: Any
    correction: CapturedCorrection | None = None
    payload: ObservationPayload
    review_required: bool = False

    @property
    def labeler(self) -> str:
        """The unchanged adjudication reducer names observation authors labelers."""
        return self.author

    @field_validator("valid_at", "observed_at")
    @classmethod
    def check_time(cls, value: str) -> str:
        _timestamp(value)
        return value

    @model_validator(mode="after")
    def check_observation(self) -> ObservationRecord:
        _json_value(self.semantic_evidence)
        _json_value(self.payload.model_dump(mode="json"))
        _digest(self.evidence_digest)
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


def _digest(value: str) -> None:
    if not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("digest must be a lowercase SHA-256 digest")


def semantic_evidence_digest(value: Any) -> str:
    """New canonical projection digest; existing replies use their own scheme."""
    _json_value(value)
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                            allow_nan=False)
    return hashlib.sha256(serialized.encode()).hexdigest()


Record = RunRecord | ObservationRecord
_Model = TypeVar("_Model", bound=_RecordModel)


def _parse(raw: Mapping[str, Any] | str | bytes, model: type[_Model]) -> _Model:
    try:
        value = json.loads(raw) if isinstance(raw, (str, bytes)) else dict(raw)
        if not isinstance(value, dict) or "schema_version" not in value:
            raise ValueError(f"invalid {model.__name__}: schema_version is required")
        return model.model_validate(value)
    except ValidationError as exc:
        # Never echo rejected evidence or credentials in diagnostics.
        allowed = {field for value in globals().values()
                   if isinstance(value, type) and issubclass(value, BaseModel)
                   for field in value.model_fields}
        fields = ", ".join(".".join(
            str(part) if isinstance(part, int) or part in allowed else "field"
            for part in item["loc"]) or "record" for item in exc.errors(include_input=False))
        raise ValueError(f"invalid {model.__name__}: {fields}") from None
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
    return json.dumps(serialize_record(record), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def record_digest(record: Record) -> str:
    """Content digest independent of its immutable identity."""
    return hashlib.sha256(canonical_record_json(record).encode()).hexdigest()


def _validate_run_evidence(run: RunRecord) -> None:
    """Validate semantic sections without rewriting retained producer payloads."""
    if run.original_task.status == "available":
        task = OriginalTaskPayload.model_validate(run.original_task.value)
        if task.diff_sha256 != hashlib.sha256(task.diff.encode()).hexdigest():
            raise ValueError("original_task diff_sha256 does not match diff")
        if task.source_diff_sha256 is not None:
            _digest(task.source_diff_sha256)
    if run.final_state.status == "available":
        FinalStatePayload.model_validate(run.final_state.value)
    if run.recommended_patch.status == "available":
        patch = RecommendedPatchPayload.model_validate(run.recommended_patch.value)
        if patch.sha256 != hashlib.sha256(patch.patch.encode()).hexdigest():
            raise ValueError("recommended_patch sha256 does not match patch")
        if patch.source_sha256 is not None:
            _digest(patch.source_sha256)
    if run.trajectories.status == "available":
        payload = TrajectoriesPayload.model_validate(run.trajectories.value)
        _timestamp(payload.cutoff_at)
        documents: dict[str, dict[str, Any]] = {}
        for document in payload.documents:
            Trajectory.model_validate(document)
            identity = document.get("trajectory_id")
            if not isinstance(identity, str) or not identity or identity in documents:
                raise ValueError("trajectory documents need unique identities")
            if document.get("session_id") != run.run_id:
                raise ValueError("trajectory session_id differs from run_id")
            documents[identity] = document
        if payload.root_trajectory_id != run.run_id or run.run_id not in documents:
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
        findings = FindingsPayload.model_validate(run.findings.value)
        claim_ids: set[str] = set()
        for stack in findings.claims:
            records = stack.records.get("issues") if isinstance(stack.records, dict) else stack.records
            if not isinstance(records, list):
                raise ValueError("claim records require an issues array")
            for claim in records:
                if not isinstance(claim, dict) or not isinstance(claim.get("uid"), str) or not claim["uid"]:
                    raise ValueError("claim requires its host-assigned uid")
                if claim["uid"] in claim_ids:
                    raise ValueError("claims contain duplicate uid")
                claim_ids.add(claim["uid"])
        for raw_item in findings.items:
            item = CanonicalItemPayload.model_validate(raw_item)
            _digest(item.fingerprint)
            if item.item_uid in items:
                raise ValueError("canonical findings contain duplicate item_uid")
            if any(not uid for uid in item.source_uids) or len(set(item.source_uids)) != len(item.source_uids):
                raise ValueError("finding source_uids must be nonempty unique identities")
            items.add(item.item_uid)
    if run.verification.status == "available":
        verification = run.verification.value
        if not isinstance(verification, dict) or not isinstance(verification.get("item_associations"), list):
            raise ValueError("verification requires item associations")
        if len(verification) < 2:
            raise ValueError("verification requires producer evidence")
        for association in verification["item_associations"]:
            if not isinstance(association, dict) or association.get("item_uid") not in items:
                raise ValueError("verification association references an absent finding item")
    if run.scoring.status == "available":
        scoring = ScoringPayload.model_validate(run.scoring.value)
        if scoring.persisted_breakdown.reward_version != scoring.reward_policy.version:
            raise ValueError("reward breakdown version differs from persisted policy")
        if scoring.persisted_breakdown.format_valid != scoring.format_valid:
            raise ValueError("reward breakdown format gate differs from scoring input")
        if not {"correctness", "length"} <= scoring.persisted_breakdown.axes_present.keys():
            raise ValueError("reward breakdown requires explicit correctness/length presence")
        if scoring.source_review_sha256 is not None:
            _digest(scoring.source_review_sha256)
        if scoring.review_text is not None and scoring.length != len(scoring.review_text):
            provenance = scoring.review_text_redaction or {}
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
    """Publish the envelope and producer payload shapes as JSON Schema.

    Cross-record links, digests and ordered step membership additionally require
    ``parse_run`` validation. The schema describes newly collected records only.
    """
    schema = RunRecord.model_json_schema()
    schema["required"].append("schema_version")
    definitions = schema.setdefault("$defs", {})
    for field, model in (
        ("original_task", OriginalTaskPayload), ("final_state", FinalStatePayload),
        ("recommended_patch", RecommendedPatchPayload), ("trajectories", TrajectoriesPayload),
        ("findings", FindingsPayload), ("scoring", ScoringPayload),
    ):
        payload = model.model_json_schema()
        definitions.update(payload.pop("$defs", {}))
        envelope = schema["properties"][field]
        schema["properties"][field] = {"allOf": [envelope, {
            "if": {"properties": {"status": {"const": "available"}}, "required": ["status"]},
            "then": {"properties": {"value": payload}, "required": ["value"]},
            "else": {"properties": {"value": {"type": "null"}}},
        }]}
    schema["properties"]["verification"] = {"allOf": [
        schema["properties"]["verification"], {
            "if": {"properties": {"status": {"const": "available"}}, "required": ["status"]},
            "then": {"properties": {"value": {
                "type": "object", "minProperties": 2, "required": ["item_associations"],
                "properties": {"item_associations": {"type": "array", "items": {"type": "object"}}},
            }}, "required": ["value"]},
            "else": {"properties": {"value": {"type": "null"}}},
        },
    ]}
    return schema


def observation_record_schema() -> dict[str, Any]:
    """Publish the typed history contract; parsing also verifies evidence digests."""
    schema = ObservationRecord.model_json_schema()
    schema["required"].append("schema_version")
    return schema
