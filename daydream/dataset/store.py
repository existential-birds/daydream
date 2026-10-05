"""Atomic, private JSONL records and immutable, content-pinned local snapshots.

Each complete record is its own immutable JSONL shard. Mutations and selection
share an OS file lock, including across processes. Record/shard names are hashes,
never untrusted identities. A snapshot reads exactly its retained membership;
valid-time eligibility is exposed separately from observation-time membership.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from daydream.dataset.privacy import record_is_private
from daydream.dataset.schema import (
    ObservationRecord,
    RunRecord,
    canonical_record_json,
    parse_observation,
    parse_run,
)
from daydream.timeutil import parse_iso_timestamp
from daydream.training.adjudication.precedence import effective_adjudication

_DEFAULT_MAX_RECORD_BYTES = 64 * 1024 * 1024
_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class StoreError(ValueError):
    """Value-free, actionable failure at the public record-store boundary."""
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"local record store: {code}")


@dataclass(frozen=True)
class CommitResult:
    identity: str
    digest: str
    committed: bool
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True)
class RecordMember:
    identity: str
    record_digest: str
    shard: str
    shard_digest: str


@dataclass(frozen=True)
class DatasetSnapshot:
    snapshot_id: str
    observed_before: str
    valid_before: str | None
    runs: tuple[RecordMember, ...]
    observations: tuple[RecordMember, ...]
    schema_version: str = "daydream.snapshot.v1"
    diagnostics: tuple[str, ...] = field(default=(), compare=False)


@dataclass(frozen=True)
class SnapshotRecords:
    snapshot: DatasetSnapshot
    runs: tuple[RunRecord, ...]
    observations: tuple[ObservationRecord, ...]
    eligible_observations: tuple[ObservationRecord, ...]
    diagnostics: tuple[str, ...] = ()

    def effective_judgment(self, run_id: str, item_uid: str) -> dict[str, Any]:
        """Resolve eligible finding history with the unchanged precedence reducer."""
        history = []
        for record in self.eligible_observations:
            if record.run_id != run_id or record.item_uid != item_uid or record.payload.type != "finding-judgment":
                continue
            history.append({
                "record_id": item_uid,
                "disposition": record.payload.disposition,
                "labeler": record.labeler, "role": record.role,
                "observed_at": record.observed_at,
                "evidence": record.semantic_evidence, "evidence_digest": record.evidence_digest,
                "review_required": record.review_required,
            })
        if not history:
            raise StoreError("missing_eligible_finding_judgment")
        return effective_adjudication(history)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _timestamp(value: str) -> datetime:
    try:
        result = parse_iso_timestamp(value)
        if result.tzinfo is None:
            raise ValueError
        return result
    except (TypeError, ValueError):
        raise StoreError("invalid_temporal_cutoff") from None


class LocalRecordStore:
    """Persist validated new records and read explicitly selected snapshots."""
    def __init__(self, root: Path, *, max_record_bytes: int = _DEFAULT_MAX_RECORD_BYTES) -> None:
        if max_record_bytes <= 0:
            raise ValueError("max_record_bytes must be positive")
        self.root = Path(root).expanduser().absolute()
        self.max_record_bytes = max_record_bytes

    def commit_run(self, record: RunRecord | Mapping[str, Any] | str | bytes) -> CommitResult:
        """Commit one immutable run; an identical retry is a no-op."""
        parsed = self._parse(record, run=True)
        assert isinstance(parsed, RunRecord)
        return self._commit(parsed, "runs", parsed.run_id)

    def append_observation(self, record: ObservationRecord | Mapping[str, Any] | str | bytes) -> CommitResult:
        """Append history without overwriting prior decisions or source evidence."""
        parsed = self._parse(record, run=False)
        assert isinstance(parsed, ObservationRecord)
        return self._commit(parsed, "observations", parsed.observation_id)

    def _parse(self, value: Any, *, run: bool) -> RunRecord | ObservationRecord:
        size = len(value.encode() if isinstance(value, str) else value) if isinstance(value, (str, bytes)) else 0
        if size > self.max_record_bytes:
            raise StoreError("record_too_large")
        try:
            if isinstance(value, (RunRecord, ObservationRecord)):
                value = value.model_dump(mode="json")
            return parse_run(value) if run else parse_observation(value)
        except (ValidationError, ValueError, TypeError, UnicodeError, RecursionError, OverflowError):
            raise StoreError("invalid_or_unknown_record_schema") from None

    def _payload(self, record: RunRecord | ObservationRecord) -> bytes:
        text = canonical_record_json(record)
        data = (text + "\n").encode("utf-8")
        if len(data) > self.max_record_bytes:
            raise StoreError("record_too_large")
        if not record_is_private(record.model_dump(mode="json"), text):
            raise StoreError("privacy_refused")
        return data

    @contextmanager
    def _locked(self) -> Iterator[tuple[str, ...]]:
        try:
            self._directory(self.root)
            for name in ("runs", "observations", "snapshots", "staging"):
                self._directory(self.root / name)
            descriptor = os.open(self.root / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            with os.fdopen(descriptor, "rb") as lock:
                os.fchmod(lock.fileno(), 0o600)
                fcntl.flock(lock, fcntl.LOCK_EX)
                diagnostics: list[str] = []
                for abandoned in (self.root / "staging").iterdir():
                    if abandoned.is_file() and not abandoned.is_symlink():
                        abandoned.unlink()
                        diagnostics.append("recovered_interrupted_write")
                    else:
                        raise StoreError("unsafe_storage_path")
                yield tuple(diagnostics)
        except OSError:
            raise StoreError("persistence_failed") from None

    @staticmethod
    def _directory(path: Path) -> None:
        if any(ancestor.is_symlink() for ancestor in (path, *path.parents)):
            raise StoreError("unsafe_storage_path")
        missing = []
        ancestor = path
        while not ancestor.exists():
            missing.append(ancestor)
            ancestor = ancestor.parent
        for directory in reversed(missing):
            directory.mkdir(mode=0o700, exist_ok=True)
            os.chmod(directory, 0o700)
            LocalRecordStore._fsync_directory(directory)
            LocalRecordStore._fsync_directory(directory.parent)
        os.chmod(path, 0o700)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _atomic_write(self, path: Path, data: bytes) -> None:
        if len(data) > self.max_record_bytes:
            raise StoreError("record_too_large")
        descriptor, staging = tempfile.mkstemp(dir=self.root / "staging", prefix="record-")
        temporary = Path(staging)
        renamed = False
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            renamed = True
            self._fsync_directory(path.parent)
            self._fsync_directory(self.root / "staging")
        except OSError:
            if renamed:
                path.unlink(missing_ok=True)
            raise
        finally:
            temporary.unlink(missing_ok=True)

    def _read_bytes(self, path: Path) -> bytes:
        if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):
            raise StoreError("unsafe_storage_path")
        if path.stat().st_size > self.max_record_bytes:
            raise StoreError("record_too_large")
        with path.open("rb") as handle:
            data = handle.read(self.max_record_bytes + 1)
        if len(data) > self.max_record_bytes:
            raise StoreError("record_too_large")
        if not data.endswith(b"\n") or data.count(b"\n") != 1:
            raise StoreError("malformed_or_interrupted_record")
        return data

    def _commit(self, record: RunRecord | ObservationRecord, kind: str, identity: str) -> CommitResult:
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
            if isinstance(record, ObservationRecord):
                self._validate_target(record)
            self._atomic_write(path, data)
            return CommitResult(identity, digest, True, diagnostics)

    def _validate_target(self, record: ObservationRecord) -> None:
        path = self.root / "runs" / (_sha(record.run_id.encode()) + ".jsonl")
        if not path.exists():
            raise StoreError("unknown_run_reference")
        run = self._parse(self._read_bytes(path), run=True)
        assert isinstance(run, RunRecord)
        if run.run_id != record.run_id:
            raise StoreError("conflicting_run_reference")
        self._validate_finding_target(record, run)

    @staticmethod
    def _validate_finding_target(record: ObservationRecord, run: RunRecord) -> None:
        if record.item_uid is not None:
            value = run.findings.value
            items = value.get("items", []) if isinstance(value, dict) else []
            if not any(isinstance(item, dict) and item.get("item_uid") == record.item_uid for item in items):
                raise StoreError("orphan_finding_reference")

    def _members(self, kind: str) -> list[tuple[RecordMember, RunRecord | ObservationRecord]]:
        members = []
        for path in sorted((self.root / kind).iterdir()):
            if not re.fullmatch(r"[0-9a-f]{64}\.jsonl", path.name):
                raise StoreError("unknown_shard")
            data = self._read_bytes(path)
            record = self._parse(data, run=kind == "runs")
            if self._payload(record) != data:
                raise StoreError("noncanonical_record")
            identity = record.run_id if isinstance(record, RunRecord) else record.observation_id
            if path.name != _sha(identity.encode()) + ".jsonl":
                raise StoreError("conflicting_record_identity")
            members.append((RecordMember(identity, _sha(data[:-1]), path.name, _sha(data)), record))
        return members

    def select_snapshot(
        self, *, observed_before: str, valid_before: str | None = None,
        run_ids: Sequence[str] | None = None,
    ) -> DatasetSnapshot:
        """Pin observe-time membership; valid-time only controls eligible evidence."""
        observed_cutoff = _timestamp(observed_before)
        if valid_before is not None:
            _timestamp(valid_before)
        with self._locked() as diagnostics:
            all_runs = self._members("runs")
            if run_ids is not None and set(run_ids) - {member.identity for member, _ in all_runs}:
                raise StoreError("unknown_run_reference")
            runs = tuple(member for member, record in all_runs if
                         isinstance(record, RunRecord) and _timestamp(record.captured_at) <= observed_cutoff
                         and (run_ids is None or member.identity in run_ids))
            selected = {member.identity for member in runs}
            all_observations = self._members("observations")
            for _member, record in all_observations:
                assert isinstance(record, ObservationRecord)
                self._validate_target(record)
            observations = tuple(member for member, record in all_observations
                                 if isinstance(record, ObservationRecord)
                                 and record.run_id in selected and _timestamp(record.observed_at) <= observed_cutoff)
            value = {"schema_version": "daydream.snapshot.v1", "observed_before": observed_before,
                     "valid_before": valid_before, "runs": [asdict(member) for member in runs],
                     "observations": [asdict(member) for member in observations]}
            snapshot_id = _sha(_canonical(value).encode())
            snapshot = DatasetSnapshot(snapshot_id, observed_before, valid_before, runs, observations)
            document = asdict(snapshot)
            document.pop("diagnostics")
            data = (_canonical(document) + "\n").encode()
            path = self.root / "snapshots" / (snapshot_id + ".jsonl")
            if path.exists():
                if self._read_bytes(path) != data:
                    raise StoreError("snapshot_content_conflict")
            else:
                self._atomic_write(path, data)
            return DatasetSnapshot(snapshot_id, observed_before, valid_before, runs, observations,
                                   diagnostics=diagnostics)

    def read_snapshot(self, snapshot: DatasetSnapshot | str) -> SnapshotRecords:
        """Read only the immutable shards pinned by a saved snapshot."""
        snapshot_id = snapshot.snapshot_id if isinstance(snapshot, DatasetSnapshot) else snapshot
        if not isinstance(snapshot_id, str) or not _DIGEST.fullmatch(snapshot_id):
            raise StoreError("invalid_snapshot_identity")
        with self._locked() as diagnostics:
            path = self.root / "snapshots" / (snapshot_id + ".jsonl")
            if not path.exists():
                raise StoreError("unknown_snapshot")
            try:
                value = json.loads(self._read_bytes(path))
                if not isinstance(value, dict):
                    raise StoreError("malformed_snapshot")
                expected_id = value.pop("snapshot_id")
                if expected_id != snapshot_id or _sha(_canonical(value).encode()) != snapshot_id:
                    raise StoreError("snapshot_digest_mismatch")
                if value["schema_version"] != "daydream.snapshot.v1":
                    raise StoreError("unknown_snapshot_schema")
                self._validate_snapshot(value)
                pinned = DatasetSnapshot(snapshot_id, value["observed_before"], value["valid_before"],
                    tuple(RecordMember(**member) for member in value["runs"]),
                    tuple(RecordMember(**member) for member in value["observations"]))
            except (TypeError, KeyError, ValueError, RecursionError) as error:
                if isinstance(error, StoreError):
                    raise
                raise StoreError("malformed_snapshot") from None
            if isinstance(snapshot, DatasetSnapshot) and (
                snapshot.snapshot_id, snapshot.observed_before, snapshot.valid_before,
                snapshot.runs, snapshot.observations, snapshot.schema_version
            ) != (pinned.snapshot_id, pinned.observed_before, pinned.valid_before,
                  pinned.runs, pinned.observations, pinned.schema_version):
                raise StoreError("snapshot_content_conflict")
            runs = tuple(self._read_member(member, "runs") for member in pinned.runs)
            observations = tuple(self._read_member(member, "observations") for member in pinned.observations)
            typed_runs = tuple(record for record in runs if isinstance(record, RunRecord))
            typed_observations = tuple(record for record in observations if isinstance(record, ObservationRecord))
            selected_runs = {run.run_id for run in typed_runs}
            cutoff = _timestamp(pinned.observed_before)
            if any(_timestamp(run.captured_at) > cutoff for run in typed_runs):
                raise StoreError("snapshot_temporal_membership_conflict")
            for record in typed_observations:
                if _timestamp(record.observed_at) > cutoff:
                    raise StoreError("snapshot_temporal_membership_conflict")
                if record.run_id not in selected_runs:
                    raise StoreError("orphan_snapshot_observation")
                run = next(run for run in typed_runs if run.run_id == record.run_id)
                self._validate_finding_target(record, run)
            eligible = tuple(record for record in typed_observations if pinned.valid_before is None
                             or _timestamp(record.valid_at) <= _timestamp(pinned.valid_before))
            return SnapshotRecords(pinned, typed_runs, typed_observations, eligible, diagnostics)

    @staticmethod
    def _validate_snapshot(value: Any) -> None:
        if not isinstance(value, dict) or set(value) != {
            "schema_version", "observed_before", "valid_before", "runs", "observations"
        }:
            raise StoreError("malformed_snapshot")
        if not isinstance(value["observed_before"], str):
            raise StoreError("malformed_snapshot")
        _timestamp(value["observed_before"])
        if value["valid_before"] is not None:
            if not isinstance(value["valid_before"], str):
                raise StoreError("malformed_snapshot")
            _timestamp(value["valid_before"])
        for kind in ("runs", "observations"):
            members = value[kind]
            if not isinstance(members, list):
                raise StoreError("malformed_snapshot")
            identities = set()
            for member in members:
                if not isinstance(member, dict) or set(member) != {
                    "identity", "record_digest", "shard", "shard_digest"
                } or any(not isinstance(field, str) for field in member.values()):
                    raise StoreError("malformed_snapshot")
                if not member["identity"] or member["identity"] in identities:
                    raise StoreError("malformed_snapshot")
                identities.add(member["identity"])
                if not _DIGEST.fullmatch(member["record_digest"]) or not _DIGEST.fullmatch(member["shard_digest"]):
                    raise StoreError("malformed_snapshot")
                if member["shard"] != _sha(member["identity"].encode()) + ".jsonl":
                    raise StoreError("malformed_snapshot")

    def _read_member(self, member: RecordMember, kind: str) -> RunRecord | ObservationRecord:
        if not re.fullmatch(r"[0-9a-f]{64}\.jsonl", member.shard):
            raise StoreError("invalid_snapshot_shard")
        data = self._read_bytes(self.root / kind / member.shard)
        if _sha(data) != member.shard_digest or _sha(data[:-1]) != member.record_digest:
            raise StoreError("snapshot_record_digest_mismatch")
        record = self._parse(data, run=kind == "runs")
        if self._payload(record) != data:
            raise StoreError("noncanonical_record")
        identity = record.run_id if isinstance(record, RunRecord) else record.observation_id
        if identity != member.identity or member.shard != _sha(identity.encode()) + ".jsonl":
            raise StoreError("conflicting_record_identity")
        return record
