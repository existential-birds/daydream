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
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from daydream.artifacts.filesystem import _create_private_directory
from daydream.artifacts.models import ArtifactVisibilityError
from daydream.dataset.privacy import record_is_private
from daydream.dataset.schema import (
    Record,
    parse_observation,
    parse_run,
    parse_snapshot,
)
from daydream.json_utils import atomic_write_bytes, canonical_json as _canonical
from daydream.timeutil import parse_iso_timestamp
from daydream.training.adjudication.precedence import effective_adjudication

_DEFAULT_MAX_RECORD_BYTES = 64 * 1024 * 1024
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_DIRECTORIES = ("runs", "observations", "snapshots")


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
class SnapshotRecords:
    snapshot: Record
    runs: tuple[Record, ...]
    observations: tuple[Record, ...]
    eligible_observations: tuple[Record, ...]
    diagnostics: tuple[str, ...] = ()

    def effective_judgment(self, run_id: str, item_uid: str) -> dict[str, Any]:
        """Resolve eligible finding history with the unchanged precedence reducer."""
        history = []
        for record in self.eligible_observations:
            if (record["run_id"] != run_id or record.get("item_uid") != item_uid
                    or record["payload"]["type"] != "finding-judgment"):
                continue
            history.append({
                "record_id": item_uid,
                "disposition": record["payload"]["disposition"],
                "labeler": record["author"], "role": record["role"],
                "observed_at": record["observed_at"],
                "evidence": record["semantic_evidence"], "evidence_digest": record["evidence_digest"],
                "review_required": record.get("review_required", False),
            })
        if not history:
            raise StoreError("missing_eligible_finding_judgment")
        return effective_adjudication(history)


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

    def commit_run(self, record: Mapping[str, Any] | str | bytes) -> CommitResult:
        """Commit one immutable run; an identical retry is a no-op."""
        parsed = self._parse(record, run=True)
        return self._commit(parsed, "runs", parsed["run_id"])

    def append_observation(self, record: Mapping[str, Any] | str | bytes) -> CommitResult:
        """Append history without overwriting prior decisions or source evidence."""
        parsed = self._parse(record, run=False)
        return self._commit(parsed, "observations", parsed["observation_id"])

    def _parse(self, value: Any, *, run: bool) -> Record:
        size = len(value.encode() if isinstance(value, str) else value) if isinstance(value, (str, bytes)) else 0
        if size > self.max_record_bytes:
            raise StoreError("record_too_large")
        try:
            return parse_run(value) if run else parse_observation(value)
        except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError):
            raise StoreError("invalid_or_unknown_record_schema") from None

    def _payload(self, record: Record) -> bytes:
        text = _canonical(record)
        data = (text + "\n").encode("utf-8")
        if len(data) > self.max_record_bytes:
            raise StoreError("record_too_large")
        if not record_is_private(record, text):
            raise StoreError("privacy_refused")
        return data

    @contextmanager
    def _locked(self) -> Iterator[tuple[str, ...]]:
        try:
            self._directory(self.root)
            for name in _DIRECTORIES:
                self._directory(self.root / name)
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
        except OSError:
            raise StoreError("persistence_failed") from None

    @staticmethod
    def _directory(path: Path) -> None:
        try:
            _create_private_directory(path, exist_ok=True)
        except ArtifactVisibilityError:
            raise StoreError("unsafe_storage_path") from None

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
        run = self._parse(self._read_bytes(path), run=True)
        if run["run_id"] != record["run_id"]:
            raise StoreError("conflicting_run_reference")
        self._validate_finding_target(record, run)

    @staticmethod
    def _validate_finding_target(record: Record, run: Record) -> None:
        if record.get("item_uid") is not None:
            value = run["findings"].get("value")
            items = value.get("items", []) if isinstance(value, dict) else []
            if not any(isinstance(item, dict) and item.get("item_uid") == record["item_uid"] for item in items):
                raise StoreError("orphan_finding_reference")

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
            value = {"schema_version": "daydream.snapshot.v1", "observed_before": observed_before,
                     "valid_before": valid_before, "runs": list(runs),
                     "observations": list(observations)}
            snapshot_id = _sha(_canonical(value).encode())
            document = parse_snapshot({"snapshot_id": snapshot_id, **value})
            data = (_canonical(document) + "\n").encode()
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
                if expected_id != snapshot_id or _sha(_canonical(value).encode()) != snapshot_id:
                    raise StoreError("snapshot_digest_mismatch")
                if value["schema_version"] != "daydream.snapshot.v1":
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
