"""Versioned JSONL evidence and content-pinned private local snapshots."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from jsonschema import Draft202012Validator
from pydantic import JsonValue, TypeAdapter

from daydream.artifacts.filesystem import _create_private_directory, _projection_path
from daydream.artifacts.models import ArtifactVisibilityError
from daydream.atif import Trajectory
from daydream.deep.render import without_review_context
from daydream.json_utils import _publish_staged, atomic_write_bytes, canonical_json
from daydream.training.adjudication.precedence import effective_adjudication
from daydream.training.labeler_versions import reply_evidence_digest


class StoreError(ValueError):
    """Value-free, actionable failure at the public record-store boundary."""
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"local record store: {code}")


def _timestamp(value: str) -> datetime:
    try:
        result = datetime.fromisoformat(value)
        if result.tzinfo is None:
            raise ValueError
        return result
    except (TypeError, ValueError):
        raise StoreError("invalid_temporal_cutoff") from None


Record = dict[str, Any]
_JSON: TypeAdapter[JsonValue] = TypeAdapter(JsonValue, config={"strict": True, "allow_inf_nan": False})
_MODEL_AUTHOR_RE = re.compile(
    r"(?:^|[-_])(?:claude|gpt|llm|model|classifier|anthropic|openai|codex|gemini)(?:$|[-_0-9])", re.IGNORECASE)


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
        analyzed_revision=_object(optional=("pr_base_sha", "source_id"), head_sha=_ID, merge_base_sha=_ID, diff_key=_ID,
                                  pr_base_sha=_nullable(_STRING), source_id=_SHA),
        diff=_STRING, diff_sha256=_SHA, source_diff_sha256=_nullable(_SHA), redaction=_nullable(_MAP),
        repository=_object(repo_slug=_nullable(_STRING), remote_url=_nullable(_STRING), host=_nullable(_STRING)),
        pr=_object(number=_nullable({"type": "integer", "minimum": 1}), repo=_nullable(_STRING)),
        changed_files=_array(_STRING),
        input_scope={"enum": ["committed", "committed_and_tracked_worktree", "captured_working"]},
        dirty_tracked=_nullable(_BOOL), untracked_files_included=_BOOL),
    "final_state": _object(optional=("source_id",), head_sha=_nullable(_STRING), source_id=_SHA),
    "recommended_patch": _object(optional=("source_sha256",),
        patch=_STRING, sha256=_SHA, source_sha256=_nullable(_SHA), capture=_nullable(_MAP)),
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
_REPLY_CAPTURE = _object(optional=("text", "captured_sha256", "redaction_provenance", "reason",
    "created_at", "updated_at", "html_url", "in_reply_to_id"),
    status=_AVAILABILITY, source_reply_id=_ID, body_sha256=_SHA,
    text=_nullable(_STRING), captured_sha256=_nullable(_SHA), redaction_provenance=_MAP, reason=_nullable(_STRING),
    created_at=_STRING, updated_at=_STRING, html_url=_STRING, in_reply_to_id=_ID)
_OBSERVATION_SCHEMA = _object(
    optional=("item_uid", "classifier_version", "correction", "reply_captures", "evidence_digest_scheme",
              "review_required"),
    schema_version={"const": "daydream.observation.v3"}, observation_id=_ID, run_id=_ID,
    item_uid=_nullable(_STRING), valid_at=_STRING, observed_at=_STRING, source=_ID, author=_ID,
    role={"enum": ["rater", "adjudicator", "model-suggested", "automatic"]}, policy_version=_ID, rubric_version=_ID,
    classifier_version=_nullable(_STRING), evidence_digest=_SHA, semantic_evidence={},
    correction=_nullable(_CORRECTION),
    reply_captures=_array(_REPLY_CAPTURE),
    evidence_digest_scheme={"enum": ["canonical-json-v1", "reply-evidence-v1"]}, review_required=_BOOL,
    payload={"oneOf": [
        _object(optional=("reviewer_logins", "outcome_prior", "outcome_prior_n", "rubric"),
            type={"const": "run-label"}, label={"enum": ["accepted", "rejected", "contested", "unknown"]},
            reviewer_logins=_array(_STRING), outcome_prior=_nullable(_NUMBER), outcome_prior_n=_COUNT, rubric=_MAP),
        _object(optional=("record_id",), type={"const": "finding-judgment"}, disposition={"enum": [
            "accepted", "rejected", "ambiguous", "unanswered", "missing", "unknown"]},
            rationale=_ID, record_id=_nullable(_SHA)),
        _object(type={"const": "enrichment"}, kind={"enum": ["pr", "base"]}, evidence=_EVIDENCE)]})
_HARVEST_ANNOTATION = _object(
    labels=_array(_STRING), pr_state=_nullable(_STRING), valid_at=_nullable(_STRING),
    reward_version=_ID, reward_json=_STRING, composite_reward=_nullable(_NUMBER),
    evidence_sha=_nullable(_STRING), rubric_json=_nullable(_STRING), reviewer_logins=_array(_STRING),
    has_posterior=_BOOL, reply_classifier_version=_nullable(_STRING), reply_evidence_digest=_nullable(_SHA))
_OBSERVATION_SCHEMA["properties"]["payload"]["oneOf"].append(_object(
    type={"const": "harvest-annotation"}, annotation=_HARVEST_ANNOTATION, labeler_policy_version=_ID))
_MEMBER = _object(identity=_ID, record_digest=_SHA, shard={"type": "string", "pattern": "^[a-f0-9]{64}\\.jsonl$"},
                  shard_digest=_SHA)
_SNAPSHOT_SCHEMA = _object(schema_version={"const": "daydream.snapshot.v1"}, snapshot_id=_SHA,
    observed_before=_STRING, valid_before=_nullable(_STRING), runs=_array(_MEMBER), observations=_array(_MEMBER))
_SNAPSHOT_V2_SCHEMA = deepcopy(_SNAPSHOT_SCHEMA)
_SNAPSHOT_V2_SCHEMA["properties"].update(
    schema_version={"const": "daydream.snapshot.v2"},
    source=_nullable(_object(repository=_ID, revision={"type": "string", "pattern": "^[a-f0-9]{40}$"},
                            manifest_sha256=_nullable(_SHA))))
_SNAPSHOT_V2_SCHEMA["required"].append("source")
_VALIDATORS = {"daydream.run.v1": Draft202012Validator(_RUN_SCHEMA),
               "daydream.observation.v3": Draft202012Validator(_OBSERVATION_SCHEMA),
               "daydream.snapshot.v1": Draft202012Validator(_SNAPSHOT_SCHEMA),
               "daydream.snapshot.v2": Draft202012Validator(_SNAPSHOT_V2_SCHEMA)}


class _InvalidRecord(ValueError):
    """A record failed current-format validation; the message stays value-free."""


_REJECTED = (_InvalidRecord, RecursionError, OverflowError)


@contextmanager
def _invalid(name: str) -> Iterator[None]:
    try:
        yield
    except (ValueError, TypeError, KeyError, UnicodeError) as error:
        field = "schema_version" if str(error) == "schema_version" else "record"
        raise _InvalidRecord(f"invalid {name}: {field}") from None


def _parse(raw: Mapping[str, Any] | str | bytes, version: str) -> Record:
    name = "ObservationRecord" if version.startswith("daydream.observation.") else (
        "RunRecord" if version == "daydream.run.v1" else "Snapshot")
    with _invalid(name):
        value = json.loads(raw) if isinstance(raw, (str, bytes)) else dict(raw)
        if not isinstance(value, dict) or value.get("schema_version") != version or version not in _VALIDATORS:
            raise ValueError("schema_version")
        _JSON.validate_python(value)
        evidence_default = {"status": "unproduced", "value": None, "reason": None}
        if version == "daydream.run.v1":
            value = {"provenance": {}, "completeness": {}, "trace_id": None, **value}
            for field in _RUN_PAYLOADS:
                value[field] = {**evidence_default, **value.get(field, {})}
            _timestamp(value["captured_at"])
        elif version.startswith("daydream.observation."):
            _timestamp(value["valid_at"])
            _timestamp(value["observed_at"])
        if not _VALIDATORS[version].is_valid(value):
            raise ValueError("record")
        if version == "daydream.run.v1":
            _validate_run_evidence(value)
        elif version.startswith("daydream.observation."):
            _validate_observation(value)
        else:
            _timestamp(value["observed_before"])
            if value["valid_before"] is not None:
                _timestamp(value["valid_before"])
            for kind in ("runs", "observations"):
                if len({member["identity"] for member in value[kind]}) != len(value[kind]):
                    raise ValueError("duplicate snapshot members")
        return dict(json.loads(canonical_json(value)))


def _run_envelope(
    raw: Mapping[str, Any] | str | bytes, documents: Iterable[Any] | None = None,
) -> tuple[Record, Iterable[Any] | None]:
    """Validate a run record with its trajectory documents split out for streaming.

    ``documents`` supplies them separately for a record whose trajectory value omits
    them. The returned envelope holds a unique placeholder where they splice back in.
    """
    with _invalid("RunRecord"):
        value = json.loads(raw) if isinstance(raw, (str, bytes)) else dict(raw)
        if not isinstance(value, dict):
            raise ValueError("schema_version")
        section = value.get("trajectories")
        payload = section.get("value") if isinstance(section, dict) else None
        if isinstance(section, dict) and isinstance(payload, dict):
            if documents is None and isinstance(payload.get("documents"), list):
                documents = payload["documents"]
            elif documents is not None and "documents" in payload:
                raise ValueError("record")
            if documents is not None:
                # Each document is validated as it streams; the schema sees one stand-in object.
                value["trajectories"] = {**section, "value": {**payload, "documents": [{}]}}
        elif documents is not None:
            raise ValueError("record")
    envelope = _parse(value, "daydream.run.v1")
    if documents is not None:
        envelope["trajectories"]["value"]["documents"] = f"documents-{uuid4().hex}"
    return envelope, documents


def _write_run_text(envelope: Record, documents: Iterable[Any] | None, write: Callable[[str], None]) -> None:
    """Write a validated envelope's canonical text, checking and splicing in each streamed document.

    The text equals ``canonical_json`` of the whole record. Validation failures raise the
    value-free ``invalid RunRecord`` error; errors from ``documents`` or ``write`` propagate.
    """
    text = canonical_json(envelope)
    if documents is None:
        write(text)
        return
    trajectories = envelope["trajectories"]["value"]
    prefix, suffix = text.split(json.dumps(trajectories["documents"], ensure_ascii=False))
    checks = _TrajectoryChecks(envelope["run_id"], trajectories["root_trajectory_id"])
    write(prefix + "[")
    for index, document in enumerate(documents):
        with _invalid("RunRecord"):
            checks.admit(document)
            text = canonical_json(document)
        if index:
            write(",")
        write(text)
        del document, text  # a loop variable would keep this document alive while the next is read
    with _invalid("RunRecord"):
        checks.finish()
    write("]" + suffix)


def _parse_typed(raw: Mapping[str, Any] | str | bytes, *, versions: tuple[str, ...], name: str) -> Record:
    """Decode and version-gate ``raw`` before the full ``_parse`` validation.

    A non-mapping payload and an undecodable one both surface as ``invalid
    <name>: record``; a decoded object whose ``schema_version`` is outside
    ``versions`` surfaces as ``invalid <name>: schema_version``.
    """
    with _invalid(name):
        value = json.loads(raw) if isinstance(raw, (str, bytes)) else dict(raw)
        version = value.get("schema_version") if isinstance(value, dict) else None
    if version not in versions:
        raise _InvalidRecord(f"invalid {name}: schema_version")
    return _parse(value, version)


def parse_run(raw: Mapping[str, Any] | str | bytes) -> Record:
    envelope, documents = _run_envelope(raw)
    parts: list[str] = []
    _write_run_text(envelope, documents, parts.append)
    return dict(json.loads("".join(parts)))


def parse_observation(raw: Mapping[str, Any] | str | bytes) -> Record:
    return _parse_typed(raw, versions=("daydream.observation.v3",), name="ObservationRecord")


def parse_snapshot(raw: Mapping[str, Any] | str | bytes) -> Record:
    return _parse_typed(raw, versions=("daydream.snapshot.v1", "daydream.snapshot.v2"), name="Snapshot")


def validate_record_targets(runs: Sequence[Record], observations: Sequence[Record]) -> None:
    """Validate observation references against a complete collection of typed runs."""
    known = {run["run_id"]: run for run in runs}
    for record in observations:
        run = known.get(record["run_id"])
        if run is None:
            raise StoreError("unknown_run_reference")
        if record.get("item_uid") is not None:
            items = (run["findings"]["value"] or {}).get("items", [])
            if not any(item["item_uid"] == record["item_uid"] for item in items):
                raise StoreError("orphan_finding_reference")


def semantic_evidence_digest(value: Any) -> str:
    _JSON.validate_python(value)
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def _validate_observation(record: Record) -> None:
    evidence, correction = record["semantic_evidence"], record.get("correction")
    captures = record.get("reply_captures", [])
    contents = ([correction] if correction is not None else []) + captures
    for content in contents:
        if content["status"] == "available":
            if content.get("text") is None or hashlib.sha256(
                content["text"].encode()).hexdigest() != content.get("captured_sha256"):
                raise ValueError("invalid captured correction")
        elif content.get("text") is not None or content.get("captured_sha256") is not None:
            raise ValueError("absent correction contains captured content")
    if len({capture["source_reply_id"] for capture in captures}) != len(captures):
        raise ValueError("duplicate source reply capture")
    if captures and (record["payload"]["type"] != "finding-judgment"
                     or record.get("evidence_digest_scheme") != "reply-evidence-v1"):
        raise ValueError("reply captures require source-bound finding evidence")
    if record.get("evidence_digest_scheme", "canonical-json-v1") == "reply-evidence-v1":
        if not isinstance(evidence, list) or any(not isinstance(reply, dict) for reply in evidence):
            raise ValueError("reply semantic evidence must be a list of objects")
        expected = reply_evidence_digest(evidence)
        for content in contents:
            matching = [reply for reply in evidence if str(reply.get("reply_id", reply.get("id", "")))
                        == content["source_reply_id"]]
            if len(matching) != 1 or matching[0].get("body_sha256") != content.get("body_sha256"):
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
    task = run['original_task']
    if task['status'] == 'available':
        value = task['value']
        local = 'source_id' in value['analyzed_revision']
        if local != (value['input_scope'] == 'captured_working') or local != value['untracked_files_included']:
            raise ValueError('captured source identity differs from input population policy')
        final = run['final_state']
        if final['status'] == 'available':
            if local != ('source_id' in final['value']) or local and (
                    final['value']['source_id'] != value['analyzed_revision']['source_id']
                    or final['value']['head_sha'] != value['analyzed_revision']['head_sha']):
                raise ValueError('captured final state differs from reviewed source identity')
    for name, text_key, digest_key in (("original_task", "diff", "diff_sha256"),
                                      ("recommended_patch", "patch", "sha256")):
        section = run[name]
        if section["status"] == "available" and section["value"][digest_key] != hashlib.sha256(
            section["value"][text_key].encode()).hexdigest():
            raise ValueError(f"{name} digest does not match captured content")
    if run["trajectories"]["status"] == "available":
        _timestamp(run["trajectories"]["value"]["cutoff_at"])
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
        if scoring["review_text"] is not None and scoring["length"] not in {
            len(scoring["review_text"]), len(without_review_context(scoring["review_text"])),
        }:
            provenance = scoring.get("review_text_redaction") or {}
            if not provenance.get("applied") or not provenance.get("policy"):
                raise ValueError("scoring length differs from retained review text without redaction provenance")


class _TrajectoryChecks:
    """Per-document trajectory checks that retain only the facts the run-level checks need."""

    def __init__(self, run_id: str, root_trajectory_id: Any) -> None:
        self.run_id, self.root_trajectory_id = run_id, root_trajectory_id
        self.step_ids: dict[str, set[Any]] = {}
        self.summaries: list[Any] = []

    def admit(self, document: Any) -> None:
        _JSON.validate_python(document)
        Trajectory.model_validate(document)
        identity = document.get("trajectory_id")
        if not isinstance(identity, str) or not identity or identity in self.step_ids:
            raise ValueError("trajectory documents need unique identities")
        if document.get("session_id") != self.run_id:
            raise ValueError("trajectory session_id differs from run_id")
        self.step_ids[identity] = {step["step_id"] for step in document["steps"]}
        self.summaries.append((document.get("extra") or {}).get("subtrajectories", []))

    def finish(self) -> None:
        if self.root_trajectory_id != self.run_id or self.run_id not in self.step_ids:
            raise ValueError("trajectory root identity differs from run_id")
        invocation_ids: dict[str, dict[str, Any]] = {}
        for summaries in self.summaries:
            if not isinstance(summaries, list):
                raise ValueError("trajectory invocation summaries must be an ordered array")
            for summary in summaries:
                if not isinstance(summary, dict):
                    raise ValueError("trajectory invocation summary must be a mapping")
                if "invocation_id" in summary:
                    self._validate_invocation(summary, invocation_ids)
                elif "trajectory_id" in summary:
                    # A registered fork can fail before producing a document.
                    # Keep its recorded identity without inventing a child.
                    for invocation in summary.get("invocations", []):
                        self._validate_invocation(invocation, invocation_ids)

    def _validate_invocation(self, invocation: dict[str, Any], known: dict[str, dict[str, Any]]) -> None:
        invocation_id = invocation.get("invocation_id")
        actual = self.step_ids.get(invocation.get("trajectory_id", ""))
        if not isinstance(invocation_id, str) or not invocation_id or actual is None:
            raise ValueError("invocation requires identity and an existing document")
        if invocation_id in known and known[invocation_id] != invocation:
            raise ValueError("invocation identity has conflicting summaries")
        step_ids = invocation.get("step_ids")
        if (not isinstance(step_ids, list) or any(type(step_id) is not int for step_id in step_ids)
                or len(set(step_ids)) != len(step_ids) or not set(step_ids) <= actual):
            raise ValueError("invocation step membership references absent or duplicate steps")
        known[invocation_id] = invocation


def run_record_schema() -> Record:
    return deepcopy(_RUN_SCHEMA)


_DEFAULT_MAX_RECORD_BYTES = 64 * 1024 * 1024
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_DIRECTORIES = ("runs", "observations", "snapshots")


@dataclass(frozen=True)
class CommitResult:
    identity: str
    digest: str
    committed: bool
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True)
class SnapshotRecords:
    snapshot: Record
    runs: tuple[Record, ...]
    observations: tuple[Record, ...]
    eligible_observations: tuple[Record, ...]
    diagnostics: tuple[str, ...] = ()

    def effective_judgment(self, run_id: str, item_uid: str) -> dict[str, Any]:
        """Resolve eligible finding history with the unchanged precedence reducer."""
        history = [{**record, **record["payload"], "record_id": item_uid, "labeler": record["author"],
                    "evidence": record["semantic_evidence"]} for record in self.eligible_observations
                   if record["run_id"] == run_id and record.get("item_uid") == item_uid
                   and record["payload"]["type"] == "finding-judgment"]
        if not history:
            raise StoreError("missing_eligible_finding_judgment")
        return effective_adjudication(history)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _abandoned(staged: Path) -> bool:
    """Whether a staged temp is still linked and no live writer holds its lock."""
    try:
        handle = open(staged, "rb")
    except FileNotFoundError:
        return False
    with handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return os.fstat(handle.fileno()).st_nlink > 0


class LocalRecordStore:
    """Persist validated new records and read explicitly selected snapshots."""
    def __init__(self, root: Path, *, max_record_bytes: int = _DEFAULT_MAX_RECORD_BYTES) -> None:
        if max_record_bytes <= 0:
            raise ValueError("max_record_bytes must be positive")
        self.root = Path(root).expanduser().absolute()
        self.max_record_bytes = max_record_bytes

    def commit_run(
        self, record: Mapping[str, Any] | str | bytes, documents: Iterable[Mapping[str, Any]] | None = None,
    ) -> CommitResult:
        """Commit one immutable run; an identical retry is a no-op.

        ``documents`` streams the trajectory documents of a record whose trajectory value
        omits them. Either way, each document is validated and staged as it streams, outside
        the store lock; an ``OSError`` while staging reports ``persistence_failed``.
        """
        self._check_size(record)
        try:
            envelope, documents = _run_envelope(record, documents)
        except _REJECTED:
            raise StoreError("invalid_or_unknown_record_schema") from None
        return self._commit_run(envelope, documents)

    def append_observation(self, record: Mapping[str, Any] | str | bytes) -> CommitResult:
        """Append history without overwriting prior decisions or source evidence."""
        parsed = self._parse(record, run=False)
        return self._commit_observation(parsed, parsed["observation_id"])

    def read_records(self) -> dict[str, tuple[Record, ...]]:
        """Read all validated current history without applying a temporal cutoff."""
        with self._locked():
            runs = tuple(record for _member, record in self._members("runs"))
            observations = tuple(record for _member, record in self._members("observations"))
            for record in observations:
                self._validate_target(record)
            return {"runs": runs, "observations": observations}

    def record_download_source(
        self, source: Mapping[str, Any], *, runs: Sequence[Record], observations: Sequence[Record],
    ) -> None:
        """Persist download provenance only while exact validated membership still matches."""
        with self._locked():
            current = {kind: tuple(record for _member, record in self._members(kind))
                       for kind in ("runs", "observations")}
            validate_record_targets(current["runs"], current["observations"])
            for kind, expected in (("runs", runs), ("observations", observations)):
                identity = "run_id" if kind == "runs" else "observation_id"
                if {record[identity]: record for record in current[kind]} != {
                    record[identity]: record for record in expected
                }:
                    raise StoreError("download_destination_conflict")
            path = self.root / "source.json"
            if path.is_symlink() or (path.exists() and not stat.S_ISREG(path.lstat().st_mode)):
                raise StoreError("unsafe_storage_path")
            self._atomic_write(path, self._payload(dict(source)))

    def download_source(self) -> Record | None:
        """Read the authenticated exact-commit provenance saved by canonical download."""
        with self._locked():
            return self._download_source()

    def _download_source(self) -> Record | None:
        path = self.root / "source.json"
        if not path.exists() and not path.is_symlink():
            return None
        try:
            value = json.loads(self._read_bytes(path))
            if (not isinstance(value, dict) or not isinstance(value.get("repository"), str)
                    or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*/[A-Za-z0-9_][A-Za-z0-9_.-]*",
                                        value["repository"])
                    or not isinstance(value.get("revision"), str)
                    or not re.fullmatch(r"[a-f0-9]{40}", value["revision"])
                    or value.get("manifest_sha256") is not None and (
                        not isinstance(value["manifest_sha256"], str)
                        or not _DIGEST.fullmatch(value["manifest_sha256"]))):
                raise ValueError
            return value
        except (ValueError, UnicodeError, RecursionError):
            raise StoreError("invalid_download_source") from None

    def _check_size(self, value: Any) -> None:
        size = len(value.encode() if isinstance(value, str) else value) if isinstance(value, (str, bytes)) else 0
        if size > self.max_record_bytes:
            raise StoreError("record_too_large")

    def _parse(self, value: Any, *, run: bool) -> Record:
        self._check_size(value)
        try:
            return parse_run(value) if run else parse_observation(value)
        except _REJECTED:
            raise StoreError("invalid_or_unknown_record_schema") from None

    def _payload(self, record: Record) -> bytes:
        text = canonical_json(record)
        data = (text + "\n").encode("utf-8")
        if len(data) > self.max_record_bytes:
            raise StoreError("record_too_large")
        return data

    @contextmanager
    def _locked(self) -> Iterator[tuple[str, ...]]:
        try:
            for path in (self.root, *(self.root / name for name in _DIRECTORIES)):
                _create_private_directory(_projection_path(path), exist_ok=True)
            descriptor = os.open(self.root / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            with os.fdopen(descriptor, "rb") as lock:
                os.fchmod(lock.fileno(), 0o600)
                fcntl.flock(lock, fcntl.LOCK_EX)
                diagnostics: list[str] = []
                for staged in (path for name in _DIRECTORIES for path in (self.root / name).glob("*.tmp")):
                    try:
                        metadata = staged.lstat()
                    except FileNotFoundError:
                        continue
                    if not stat.S_ISREG(metadata.st_mode):
                        raise StoreError("unsafe_storage_path")
                    if _abandoned(staged):
                        staged.unlink(missing_ok=True)
                        diagnostics.append("recovered_interrupted_write")
                yield tuple(diagnostics)
        except ArtifactVisibilityError:
            raise StoreError("unsafe_storage_path") from None
        except OSError:
            raise StoreError("persistence_failed") from None

    def _atomic_write(self, path: Path, data: bytes) -> None:
        if len(data) > self.max_record_bytes:
            raise StoreError("record_too_large")
        try:
            atomic_write_bytes(path, data, fsync=True, dir_fsync=True, mode=0o600)
        except OSError:
            path.unlink(missing_ok=True)
            raise

    def _read_bytes(self, path: Path) -> bytes:
        metadata = path.lstat()
        if not stat.S_ISREG(metadata.st_mode):
            raise StoreError("unsafe_storage_path")
        if metadata.st_size > self.max_record_bytes:
            raise StoreError("record_too_large")
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as handle:
            data = handle.read(self.max_record_bytes + 1)
        if len(data) > self.max_record_bytes:
            raise StoreError("record_too_large")
        if not data.endswith(b"\n") or data.count(b"\n") != 1:
            raise StoreError("malformed_or_interrupted_record")
        return data

    def _shard(self, kind: str, identity: str) -> Path:
        return self.root / kind / (_sha(identity.encode()) + ".jsonl")

    def _commit_run(self, envelope: Record, documents: Iterable[Any] | None) -> CommitResult:
        """Stage one run record outside the store lock, then compare or publish it under the lock.

        The stage is a ``*.tmp`` file created under the lock and locked by its writer until
        it is published or removed, so interrupted-write recovery skips it while the writer lives.
        """
        identity = envelope["run_id"]
        path = self._shard("runs", identity)
        with self._locked() as recovered:
            stage = tempfile.NamedTemporaryFile("wb", dir=path.parent, suffix=".tmp", delete=False)
            fcntl.flock(stage, fcntl.LOCK_EX)
        staged = Path(stage.name)
        digest = hashlib.sha256()
        size = 0

        def write(text: str) -> None:
            nonlocal size
            chunk = text.encode("utf-8")
            size += len(chunk)
            if size >= self.max_record_bytes:  # the trailing newline must still fit
                raise StoreError("record_too_large")
            digest.update(chunk)
            stage.write(chunk)

        with stage:
            try:
                try:
                    _write_run_text(envelope, documents, write)
                    stage.write(b"\n")
                    stage.flush()
                    os.fsync(stage.fileno())
                except _REJECTED:
                    raise StoreError("invalid_or_unknown_record_schema") from None
                except OSError:
                    raise StoreError("persistence_failed") from None
                recorded = digest.copy()
                recorded.update(b"\n")
                with self._locked() as also_recovered:
                    diagnostics = recovered + also_recovered
                    if path.exists():
                        if not self._holds(path, size + 1, recorded.hexdigest()):
                            raise StoreError("immutable_identity_conflict")
                        return CommitResult(identity, digest.hexdigest(), False, diagnostics)
                    try:
                        _publish_staged(path, staged, dir_fsync=True, mode=0o600)
                    except OSError:
                        path.unlink(missing_ok=True)
                        raise
                    return CommitResult(identity, digest.hexdigest(), True, diagnostics)
            finally:
                staged.unlink(missing_ok=True)  # before closing the stage releases its lock

    @staticmethod
    def _holds(path: Path, size: int, sha256: str) -> bool:
        """Whether the shard at *path* holds exactly *size* bytes with this SHA-256, read in chunks."""
        if not stat.S_ISREG(path.lstat().st_mode):
            raise StoreError("unsafe_storage_path")
        with open(path, "rb") as shard:
            return (os.fstat(shard.fileno()).st_size == size
                    and hashlib.file_digest(shard, "sha256").hexdigest() == sha256)

    def _commit_observation(self, record: Record, identity: str) -> CommitResult:
        data = self._payload(record)
        digest = _sha(data[:-1])
        with self._locked() as diagnostics:
            path = self._shard("observations", identity)
            if path.exists():
                previous = self._read_bytes(path)
                self._parse(previous, run=False)
                if previous == data:
                    return CommitResult(identity, digest, False, diagnostics)
                raise StoreError("immutable_identity_conflict")
            self._validate_target(record)
            self._atomic_write(path, data)
            return CommitResult(identity, digest, True, diagnostics)

    def _validate_target(self, record: Record) -> None:
        path = self._shard("runs", record["run_id"])
        if not path.exists():
            raise StoreError("unknown_run_reference")
        run = self._load("runs", path)[1]
        self._validate_finding_target(record, run)

    @staticmethod
    def _validate_finding_target(record: Record, run: Record) -> None:
        validate_record_targets((run,), (record,))

    def _load(
        self, kind: str, path: Path, expected: Record | None = None,
    ) -> tuple[Record, Record]:
        if not re.fullmatch(r"[0-9a-f]{64}\.jsonl", path.name):
            raise StoreError("unknown_shard")
        data = self._read_bytes(path)
        if expected is not None and (
            _sha(data) != expected["shard_digest"] or _sha(data[:-1]) != expected["record_digest"]
        ):
            raise StoreError("snapshot_record_digest_mismatch")
        record = self._parse(data, run=kind == "runs")
        identity = record["run_id" if kind == "runs" else "observation_id"]
        if (path.name != _sha(identity.encode()) + ".jsonl"
                or (expected is not None and identity != expected["identity"])):
            raise StoreError("conflicting_record_identity")
        if self._payload(record) != data:
            raise StoreError("noncanonical_record")
        member = {"identity": identity, "record_digest": _sha(data[:-1]),
                  "shard": path.name, "shard_digest": _sha(data)}
        return member, record

    def _members(self, kind: str) -> list[tuple[Record, Record]]:
        return [self._load(kind, path) for path in sorted((self.root / kind).iterdir())]

    def select_snapshot(
        self, *, observed_before: str, valid_before: str | None = None,
        run_ids: Sequence[str] | None = None,
    ) -> Record:
        """Pin observe-time membership; valid-time only controls eligible evidence."""
        observed_cutoff = _timestamp(observed_before)
        if valid_before is not None:
            _timestamp(valid_before)
        with self._locked() as diagnostics:
            all_runs = self._members("runs")
            if run_ids is not None and set(run_ids) - {member["identity"] for member, _ in all_runs}:
                raise StoreError("unknown_run_reference")
            runs = tuple(member for member, record in all_runs if
                         _timestamp(record["captured_at"]) <= observed_cutoff
                         and (run_ids is None or member["identity"] in run_ids))
            selected = {member["identity"] for member in runs}
            all_observations = self._members("observations")
            for _member, record in all_observations:
                self._validate_target(record)
            observations = tuple(member for member, record in all_observations if
                                 record["run_id"] in selected and _timestamp(record["observed_at"]) <= observed_cutoff)
            source = self._download_source()
            frozen_source = None if source is None else {
                "repository": source["repository"], "revision": source["revision"],
                "manifest_sha256": source.get("manifest_sha256")}
            value = {"schema_version": "daydream.snapshot.v2", "observed_before": observed_before,
                     "valid_before": valid_before, "runs": list(runs),
                     "observations": list(observations), "source": frozen_source}
            snapshot_id = _sha(canonical_json(value).encode())
            document = parse_snapshot({"snapshot_id": snapshot_id, **value})
            data = (canonical_json(document) + "\n").encode()
            path = self.root / "snapshots" / (snapshot_id + ".jsonl")
            if path.exists():
                if self._read_bytes(path) != data:
                    raise StoreError("snapshot_content_conflict")
            else:
                self._atomic_write(path, data)
            return {**document, "diagnostics": diagnostics}

    def read_snapshot(self, snapshot: Mapping[str, Any] | str) -> SnapshotRecords:
        """Read only the immutable shards pinned by a saved snapshot."""
        snapshot_id = snapshot.get("snapshot_id") if isinstance(snapshot, Mapping) else snapshot
        if not isinstance(snapshot_id, str) or not _DIGEST.fullmatch(snapshot_id):
            raise StoreError("invalid_snapshot_identity")
        with self._locked() as diagnostics:
            path = self.root / "snapshots" / (snapshot_id + ".jsonl")
            if not path.exists():
                raise StoreError("unknown_snapshot")
            try:
                value = json.loads(self._read_bytes(path))
                if not isinstance(value, dict) or "diagnostics" in value:
                    raise StoreError("malformed_snapshot")
                expected_id = value.pop("snapshot_id")
                if expected_id != snapshot_id or _sha(canonical_json(value).encode()) != snapshot_id:
                    raise StoreError("snapshot_digest_mismatch")
                if value["schema_version"] not in ("daydream.snapshot.v1", "daydream.snapshot.v2"):
                    raise StoreError("unknown_snapshot_schema")
                pinned = parse_snapshot({"snapshot_id": snapshot_id, **value})
            except (TypeError, KeyError, ValueError, RecursionError) as error:
                if isinstance(error, StoreError):
                    raise
                raise StoreError("malformed_snapshot") from None
            if isinstance(snapshot, Mapping) and {k: v for k, v in snapshot.items() if k != "diagnostics"} != pinned:
                raise StoreError("snapshot_content_conflict")
            typed_runs = tuple(self._load("runs", self.root / "runs" / member["shard"], member)[1]
                               for member in pinned["runs"])
            typed_observations = tuple(
                self._load("observations", self.root / "observations" / member["shard"], member)[1]
                for member in pinned["observations"])
            selected_runs = {run["run_id"]: run for run in typed_runs}
            cutoff = _timestamp(pinned["observed_before"])
            if any(_timestamp(run["captured_at"]) > cutoff for run in typed_runs):
                raise StoreError("snapshot_temporal_membership_conflict")
            for record in typed_observations:
                if _timestamp(record["observed_at"]) > cutoff:
                    raise StoreError("snapshot_temporal_membership_conflict")
                if record["run_id"] not in selected_runs:
                    raise StoreError("orphan_snapshot_observation")
                self._validate_finding_target(record, selected_runs[record["run_id"]])
            valid_cutoff = _timestamp(pinned["valid_before"]) if pinned["valid_before"] else None
            eligible = tuple(record for record in typed_observations
                             if valid_cutoff is None or _timestamp(record["valid_at"]) <= valid_cutoff)
            return SnapshotRecords({**pinned, "diagnostics": diagnostics},
                                   typed_runs, typed_observations, eligible, diagnostics)
