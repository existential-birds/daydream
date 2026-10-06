"""Versioned JSONL evidence and content-pinned private local snapshots."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator
from pydantic import JsonValue, TypeAdapter

from daydream.artifacts.filesystem import _create_private_directory, _projection_path
from daydream.artifacts.models import ArtifactVisibilityError
from daydream.atif import Trajectory
from daydream.json_utils import atomic_write_bytes, canonical_json
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
        analyzed_revision=_object(optional=("pr_base_sha",), head_sha=_ID, merge_base_sha=_ID, diff_key=_ID,
                                  pr_base_sha=_nullable(_STRING)),
        diff=_STRING, diff_sha256=_SHA, source_diff_sha256=_nullable(_SHA), redaction=_nullable(_MAP),
        repository=_object(repo_slug=_nullable(_STRING), remote_url=_nullable(_STRING), host=_nullable(_STRING)),
        pr=_object(number=_nullable({"type": "integer", "minimum": 1}), repo=_nullable(_STRING)),
        changed_files=_array(_STRING), input_scope={"enum": ["committed", "committed_and_tracked_worktree"]},
        dirty_tracked=_nullable(_BOOL), untracked_files_included={"const": False}),
    "final_state": _object(head_sha=_nullable(_STRING)),
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
_HARVEST_ANNOTATION = _object(
    labels=_array(_STRING), pr_state=_nullable(_STRING), valid_at=_nullable(_STRING),
    reward_version=_ID, reward_json=_STRING, composite_reward=_nullable(_NUMBER),
    evidence_sha=_nullable(_STRING), rubric_json=_nullable(_STRING), reviewer_logins=_array(_STRING),
    has_posterior=_BOOL, reply_classifier_version=_nullable(_STRING), reply_evidence_digest=_nullable(_SHA))
_OBSERVATION_V2_SCHEMA = deepcopy(_OBSERVATION_SCHEMA)
_OBSERVATION_V2_SCHEMA["properties"]["schema_version"] = {"const": "daydream.observation.v2"}
_OBSERVATION_V2_SCHEMA["properties"]["payload"]["oneOf"].append(_object(
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
               "daydream.observation.v1": Draft202012Validator(_OBSERVATION_SCHEMA),
               "daydream.observation.v2": Draft202012Validator(_OBSERVATION_V2_SCHEMA),
               "daydream.snapshot.v1": Draft202012Validator(_SNAPSHOT_SCHEMA),
               "daydream.snapshot.v2": Draft202012Validator(_SNAPSHOT_V2_SCHEMA)}


def _parse(raw: Mapping[str, Any] | str | bytes, version: str) -> Record:
    name = "ObservationRecord" if version.startswith("daydream.observation.") else (
        "RunRecord" if version == "daydream.run.v1" else "Snapshot")
    try:
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
    except (ValueError, TypeError, KeyError, UnicodeError) as error:
        field = "schema_version" if str(error) == "schema_version" else "record"
        raise ValueError(f"invalid {name}: {field}") from None


def parse_run(raw: Mapping[str, Any] | str | bytes) -> Record:
    return _parse(raw, "daydream.run.v1")


def parse_observation(raw: Mapping[str, Any] | str | bytes) -> Record:
    try:
        value = json.loads(raw) if isinstance(raw, (str, bytes)) else dict(raw)
        version = value.get("schema_version") if isinstance(value, dict) else None
    except (ValueError, TypeError, UnicodeError):
        raise ValueError("invalid ObservationRecord: record") from None
    if version not in ("daydream.observation.v1", "daydream.observation.v2"):
        raise ValueError("invalid ObservationRecord: schema_version")
    return _parse(value, version)


def parse_snapshot(raw: Mapping[str, Any] | str | bytes) -> Record:
    try:
        value = json.loads(raw) if isinstance(raw, (str, bytes)) else dict(raw)
        version = value.get("schema_version") if isinstance(value, dict) else None
    except (ValueError, TypeError, UnicodeError):
        raise ValueError("invalid Snapshot: record") from None
    if version not in ("daydream.snapshot.v1", "daydream.snapshot.v2"):
        raise ValueError("invalid Snapshot: schema_version")
    return _parse(value, version)


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
    if correction is not None:
        if correction["status"] == "available":
            if correction.get("text") is None or hashlib.sha256(
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
    return deepcopy(_OBSERVATION_V2_SCHEMA)


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


class LocalRecordStore:
    """Persist validated new records and read explicitly selected snapshots."""
    def __init__(self, root: Path, *, max_record_bytes: int = _DEFAULT_MAX_RECORD_BYTES) -> None:
        if max_record_bytes <= 0:
            raise ValueError("max_record_bytes must be positive")
        self.root = Path(root).expanduser().absolute()
        self.max_record_bytes = max_record_bytes

    def commit_run(self, record: Mapping[str, Any] | str | bytes) -> CommitResult:
        """Commit one immutable run; an identical retry is a no-op."""
        parsed = self._parse(record, run=True)
        return self._commit(parsed, "runs", parsed["run_id"])

    def append_observation(self, record: Mapping[str, Any] | str | bytes) -> CommitResult:
        """Append history without overwriting prior decisions or source evidence."""
        parsed = self._parse(record, run=False)
        return self._commit(parsed, "observations", parsed["observation_id"])

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

    def _parse(self, value: Any, *, run: bool) -> Record:
        size = len(value.encode() if isinstance(value, str) else value) if isinstance(value, (str, bytes)) else 0
        if size > self.max_record_bytes:
            raise StoreError("record_too_large")
        try:
            return parse_run(value) if run else parse_observation(value)
        except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError):
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
                for abandoned in (path for name in _DIRECTORIES for path in (self.root / name).glob("*.tmp")):
                    if abandoned.is_file() and not abandoned.is_symlink():
                        abandoned.unlink()
                        diagnostics.append("recovered_interrupted_write")
                    else:
                        raise StoreError("unsafe_storage_path")
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

    def _commit(self, record: Record, kind: str, identity: str) -> CommitResult:
        data = self._payload(record)
        digest = _sha(data[:-1])
        shard = _sha(identity.encode()) + ".jsonl"
        with self._locked() as diagnostics:
            path = self.root / kind / shard
            if path.exists():
                previous = self._read_bytes(path)
                self._parse(previous, run=kind == "runs")
                if previous == data:
                    return CommitResult(identity, digest, False, diagnostics)
                raise StoreError("immutable_identity_conflict")
            if kind == "observations":
                self._validate_target(record)
            self._atomic_write(path, data)
            return CommitResult(identity, digest, True, diagnostics)

    def _validate_target(self, record: Record) -> None:
        path = self.root / "runs" / (_sha(record["run_id"].encode()) + ".jsonl")
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
