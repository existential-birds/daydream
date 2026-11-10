"""Private, content-addressed HF JSONL publication from the local record store."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
from collections.abc import Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from daydream.archive.scan import SEVERITY_BLOCKING, _scan_file, _scan_text
from daydream.artifacts.filesystem import _create_private_directory, _projection_path
from daydream.artifacts.models import ArtifactVisibilityError
from daydream.dataset import (
    LocalRecordStore,
    Record,
    StoreError,
    parse_observation,
    parse_run,
    record_name,
    validate_record_targets,
)
from daydream.hub import DatasetHub, HfDatasetHub, HubConflict, HubError, _revision
from daydream.json_utils import atomic_write_bytes, canonical_json

MANIFEST_PATH = "manifest.json"
_SCHEMAS = {"runs": ["daydream.run.v1"], "observations": ["daydream.observation.v3"]}
_IDENTITY_FIELDS = {"runs": "run_id", "observations": "observation_id"}
_PARSERS = {"runs": parse_run, "observations": parse_observation}
_VERSION = "daydream.hub.v3"
_DIGEST = re.compile(r"[a-f0-9]{64}")
_REPO = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*/[A-Za-z0-9_][A-Za-z0-9_.-]*")
_MAX_BYTES = 64 * 1024 * 1024


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _empty_manifest() -> Record:
    return {"schema_version": _VERSION, "record_schemas": dict(_SCHEMAS),
            "shards": {kind: [] for kind in _SCHEMAS}}


def _validate_repo(repo_id: str) -> None:
    if not _REPO.fullmatch(repo_id) or ".." in repo_id:
        raise StoreError("invalid_repository")


def _manifest(raw: bytes) -> Record:
    """Validate paths, schema versions and unique logical identities before any fetch."""
    try:
        if len(raw) > _MAX_BYTES:
            raise ValueError
        value = json.loads(raw)
        if (not isinstance(value, dict) or set(value) != {"schema_version", "record_schemas", "shards"}
                or value["schema_version"] != _VERSION
                or value["record_schemas"] != _SCHEMAS
                or not isinstance(value["shards"], dict) or set(value["shards"]) != set(_SCHEMAS)):
            raise ValueError
        for kind in _SCHEMAS:
            shards = value["shards"][kind]
            if not isinstance(shards, list):
                raise ValueError
            paths: set[str] = set()
            identities: set[str] = set()
            for shard in shards:
                if not isinstance(shard, dict) or set(shard) != {"path", "sha256", "bytes", "count", "records"}:
                    raise ValueError
                digest = shard["sha256"]
                if (not isinstance(digest, str) or not _DIGEST.fullmatch(digest)
                        or shard["path"] != f"{kind}/{digest}.jsonl" or shard["path"] in paths
                        or type(shard["bytes"]) is not int or not 0 < shard["bytes"] <= _MAX_BYTES
                        or type(shard["count"]) is not int or shard["count"] <= 0
                        or not isinstance(shard["records"], list) or len(shard["records"]) != shard["count"]):
                    raise ValueError
                paths.add(shard["path"])
                for record in shard["records"]:
                    if (not isinstance(record, dict) or set(record) != {"identity", "sha256"}
                            or not isinstance(record["identity"], str) or not record["identity"]
                            or record["identity"] in identities or not isinstance(record["sha256"], str)
                            or not _DIGEST.fullmatch(record["sha256"])):
                        raise ValueError
                    identities.add(record["identity"])
        return value
    except (ValueError, TypeError, KeyError, RecursionError, UnicodeError):
        raise StoreError("invalid_or_unknown_manifest") from None


def _members(manifest: Record, kind: str) -> set[str]:
    """The local file names of each published record."""
    return {record_name(record["identity"]) for shard in manifest["shards"][kind] for record in shard["records"]}


def _scan(record: Record) -> None:
    """Decode JSON leaves for multiline secrets, and scan keys in serialized evidence too."""
    try:
        text = canonical_json(record)
        if (any(finding.severity == SEVERITY_BLOCKING for finding in _scan_file("record.json", text))
                or any(severity == SEVERITY_BLOCKING for _, _, _, severity in _scan_text(text))):
            raise StoreError("blocking_secret_findings")
    except StoreError:
        raise
    except Exception:
        raise StoreError("secret_scan_failed") from None


def _identity(record: Record, kind: str) -> str:
    return str(record[_IDENTITY_FIELDS[kind]])


def _validate_shard(data: bytes, shard: Record, kind: str) -> Iterator[Record]:
    """Yield each record after checking it against its manifest entry, one line at a time."""
    if len(data) != shard["bytes"] or _sha(data) != shard["sha256"]:
        raise StoreError("shard_digest_mismatch")
    if data.count(b"\n") != shard["count"] or not data.endswith(b"\n"):
        raise StoreError("malformed_or_interrupted_shard")
    end = 0
    for member in shard["records"]:
        start, end = end, data.index(b"\n", end) + 1
        line = data[start:end]
        try:
            record = _PARSERS[kind](line)
        except (ValueError, TypeError, RecursionError, OverflowError):
            raise StoreError("invalid_or_unknown_record_schema") from None
        if (_identity(record, kind) != member["identity"] or _sha(line[:-1]) != member["sha256"]
                or _json_bytes(record) != line):
            raise StoreError("shard_record_identity_mismatch")
        yield record


@dataclass(frozen=True)
class SkippedRecord:
    """A pending record left out of publication: its kind, identity SHA-256 and error code."""
    kind: str
    identity_sha256: str
    error: str


@dataclass(frozen=True)
class UploadStatus:
    queued: int
    published: int
    failed: int = 0
    error: str | None = None
    revision: str | None = None
    skipped: tuple[SkippedRecord, ...] = ()


class DatasetUploader:
    """Local records are the queue; receipts acknowledge only confirmed atomic HF commits.

    Pending records are those missing from the remote manifest, the only remote file read.
    Each commit carries at most ``max_commit_bytes`` of shards. A pending record that fails
    validation or the scanner is left out and reported; the rest still publish.
    """
    def __init__(
        self, store: LocalRecordStore, repo_id: str, *, backend: DatasetHub | None = None,
        max_shard_records: int = 100, max_shard_bytes: int = 16 * 1024 * 1024, max_commit_bytes: int = _MAX_BYTES,
    ) -> None:
        if max_shard_records <= 0 or not 0 < max_shard_bytes <= min(_MAX_BYTES, max_commit_bytes):
            raise ValueError("invalid shard limits")
        self.store, self.repo_id, self.backend = store, repo_id, backend
        self.max_shard_records, self.max_shard_bytes = max_shard_records, max_shard_bytes
        self.max_commit_bytes = max_commit_bytes
        self.queue = store.root / "uploads" / _sha(repo_id.encode())

    @contextmanager
    def _locked(self) -> Iterator[None]:
        try:
            _create_private_directory(_projection_path(self.queue), exist_ok=True)
            descriptor = os.open(self.queue / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            with os.fdopen(descriptor, "rb") as lock:
                os.fchmod(lock.fileno(), 0o600)
                fcntl.flock(lock, fcntl.LOCK_EX)
                yield
        except ArtifactVisibilityError:
            raise StoreError("unsafe_storage_path") from None
        except OSError:
            raise StoreError("queue_persistence_failed") from None

    def _read(self, path: Path) -> bytes | None:
        if not path.exists() and not path.is_symlink():
            return None
        if not stat.S_ISREG(path.lstat().st_mode):
            raise StoreError("unsafe_storage_path")
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as handle:
            data = handle.read(min(os.fstat(handle.fileno()).st_size, _MAX_BYTES) + 1)
        if len(data) > _MAX_BYTES:
            raise StoreError("queue_content_too_large")
        return data

    def _write(self, path: Path, data: bytes) -> None:
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise StoreError("unsafe_storage_path")
        atomic_write_bytes(path, data, fsync=True, dir_fsync=True, mode=0o600)

    def _receipt(self) -> tuple[Record, str | None, str | None]:
        raw = self._read(self.queue / "receipt.json")
        if raw is None:
            return _empty_manifest(), None, None
        try:
            value = json.loads(raw)
            if (set(value) != {"manifest", "revision", "error"}
                    or value["error"] is not None and not re.fullmatch(r"[a-z_]+", value["error"])):
                raise ValueError
            revision = None if value["revision"] is None else _revision(value["revision"])
            return _manifest(_json_bytes(value["manifest"])), revision, value["error"]
        except (ValueError, TypeError, KeyError, RecursionError):
            raise StoreError("invalid_upload_receipt") from None

    def _status(self, index: Mapping[str, Sequence[str]], manifest: Record, revision: str | None,
                error: str | None = None, skipped: Mapping[tuple[str, str], str] | None = None) -> UploadStatus:
        published = sum(len(_members(manifest, kind).intersection(index[kind])) for kind in _SCHEMAS)
        queued = sum(len(index[kind]) for kind in _SCHEMAS) - published
        refused = tuple(SkippedRecord(kind, name.removesuffix(".jsonl"), code)
                        for (kind, name), code in sorted((skipped or {}).items()))
        return UploadStatus(queued, published, queued if error else len(refused), error, revision, refused)

    def status(self) -> UploadStatus:
        """Offline receipt status; credentials do not activate an uploader."""
        with self._locked():
            manifest, revision, error = self._receipt()
            return self._status(self.store.record_index(), manifest, revision, error)

    def _seal(
        self, pending: Sequence[tuple[str, str]], skipped: dict[tuple[str, str], str], shards: dict[str, list[Record]],
        runs: Collection[str],
    ) -> tuple[dict[str, bytes], bool]:
        """Seal the next commit's pending records into ``shards``, read one at a time, and say whether more remain.

        A refused record is left out and recorded in ``skipped``, with any observation whose run is not among the
        local ``runs`` or was refused. Runs precede observations, so every other target run is published or sealed
        by then; the store validated each observation's finding target against its immutable run when appending it.
        """
        files: dict[str, bytes] = {}
        batch: list[bytes] = []
        members: list[Record] = []
        kind, size, total = "runs", 0, 0

        def seal() -> None:
            if not batch:
                return
            data = b"".join(batch)
            digest = _sha(data)
            path = f"{kind}/{digest}.jsonl"
            shards[kind].append({"path": path, "sha256": digest, "bytes": len(data),
                                 "count": len(batch), "records": list(members)})
            files[path] = data
            batch.clear()
            members.clear()

        for record_kind, name in pending:
            try:
                data, record = self.store.read_record(record_kind, name)
                _scan(record)
                if len(data) > self.max_shard_bytes:
                    raise StoreError("record_exceeds_shard_limit")
                run = record_name(record["run_id"]) if record_kind == "observations" else None
                if run is not None and (run not in runs or ("runs", run) in skipped):
                    raise StoreError("unknown_run_reference")
            except StoreError as refusal:
                skipped[record_kind, name] = refusal.code
                continue
            if total + len(data) > self.max_commit_bytes:
                seal()
                return files, True
            if batch and (record_kind != kind or len(batch) >= self.max_shard_records
                          or size + len(data) > self.max_shard_bytes):
                seal()
                size = 0
            kind = record_kind
            batch.append(data)
            members.append({"identity": _identity(record, kind), "sha256": _sha(data[:-1])})
            size += len(data)
            total += len(data)
        seal()
        return files, False

    def upload(self) -> UploadStatus:
        """Retry from current remote membership; retain local evidence after any failure."""
        index: dict[str, list[str]] = {kind: [] for kind in _SCHEMAS}
        skipped: dict[tuple[str, str], str] = {}
        receipt, revision = _empty_manifest(), None
        try:
            with self._locked():
                index = self.store.record_index()
                _validate_repo(self.repo_id)
                backend = self.backend or HfDatasetHub()
                more = True
                while more:
                    for _attempt in range(5):
                        head = backend.private_revision(self.repo_id)
                        raw = backend.read_file(self.repo_id, MANIFEST_PATH, head)
                        merged = _empty_manifest() if raw is None else _manifest(raw)
                        known = {kind: _members(merged, kind) for kind in _SCHEMAS}
                        pending = [(kind, name) for kind in _SCHEMAS for name in index[kind]
                                   if name not in known[kind] and (kind, name) not in skipped]
                        files, more = self._seal(pending, skipped, merged["shards"], set(index["runs"]))
                        if files:
                            files[MANIFEST_PATH] = _json_bytes(merged)
                            _manifest(files[MANIFEST_PATH])
                            try:
                                head = backend.commit(self.repo_id, files, head)
                            except HubConflict:
                                continue
                            del files
                            raw = backend.read_file(self.repo_id, MANIFEST_PATH, head)
                            if raw is None or _manifest(raw) != merged:
                                raise StoreError("publication_not_confirmed")
                        self._write(self.queue / "receipt.json", _json_bytes(
                            {"manifest": merged, "revision": head, "error": "records_skipped" if skipped else None}))
                        receipt, revision = merged, head
                        break
                    else:
                        raise StoreError("concurrent_update_retry_exhausted")
                return self._status(index, receipt, revision, skipped=skipped)
        except (StoreError, HubError) as failure:
            error = failure.code
        except Exception:
            error = "upload_failed"
        # Persist a value-free failure without ever discarding a prior acknowledgement.
        try:
            with self._locked():
                receipt, revision, _ = self._receipt()
                self._write(self.queue / "receipt.json", _json_bytes(
                    {"manifest": receipt, "revision": revision, "error": error}))
        except (StoreError, OSError):
            pass
        return self._status(index, receipt, revision, error, skipped)


def download_snapshot(
    repo_id: str, revision: str, destination: Path, *, backend: DatasetHub | None = None,
) -> LocalRecordStore:
    """Fetch exactly one private HF commit and check every shard against its manifest before exposing records."""
    _validate_repo(repo_id)
    try:
        _revision(revision)
        client = backend or HfDatasetHub()
        client.private_revision(repo_id)
        raw = client.read_file(repo_id, MANIFEST_PATH, revision)
        if raw is None:
            raise StoreError("missing_manifest")
        manifest = _manifest(raw)
        records: dict[str, list[Record]] = {kind: [] for kind in _SCHEMAS}
        for kind in _SCHEMAS:
            for shard in manifest["shards"][kind]:
                data = client.read_file(repo_id, shard["path"], revision)
                if data is None:
                    raise StoreError("missing_shard")
                records[kind].extend(_validate_shard(data, shard, kind))
        validate_record_targets(records["runs"], records["observations"])
        store = LocalRecordStore(destination)
        # Do not contaminate a pinned download with unrelated local evidence on reuse.
        if store.root.exists():
            existing = store.read_records()
            for kind in _SCHEMAS:
                desired = {_identity(record, kind): record for record in records[kind]}
                if any(desired.get(_identity(record, kind)) != record for record in existing[kind]):
                    raise StoreError("download_destination_conflict")
        for record in records["runs"]:
            store.commit_run(record)
        for record in records["observations"]:
            store.append_observation(record)
        # Store the transport provenance alongside records, never a per-run archive or index.
        store.record_download_source({
            "repository": repo_id, "revision": revision, "manifest_sha256": _sha(raw), "manifest": manifest,
        }, runs=records["runs"], observations=records["observations"])
        return store
    except HubError as error:
        raise StoreError(error.code) from None
    except (OSError, ArtifactVisibilityError):
        raise StoreError("download_persistence_failed") from None
