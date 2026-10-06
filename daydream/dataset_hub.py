"""Private, content-addressed HF JSONL publication with a durable local retry queue."""
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
from pathlib import Path
from typing import Any

from daydream.archive.scan import SEVERITY_BLOCKING, _scan_file, _scan_text
from daydream.artifacts.filesystem import _create_private_directory, _projection_path
from daydream.artifacts.models import ArtifactVisibilityError
from daydream.dataset import LocalRecordStore, Record, StoreError, parse_observation, parse_run, validate_record_targets
from daydream.dataset_hub_client import DatasetHub, HfDatasetHub, HubConflict, HubError
from daydream.json_utils import atomic_write_bytes, canonical_json

DEFAULT_HUB_REPO = "existentialbirds/daydream-trajectories"
MANIFEST_PATH = "manifest.json"
_SCHEMAS = {"runs": "daydream.run.v1", "observations": "daydream.observation.v1"}
_IDENTITY_FIELDS = {"runs": "run_id", "observations": "observation_id"}
_PARSERS = {"runs": parse_run, "observations": parse_observation}
_VERSION = "daydream.hub.v1"
_DIGEST = re.compile(r"[a-f0-9]{64}")
_REVISION = re.compile(r"[a-f0-9]{40}")
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


def _validate_revision(revision: str) -> str:
    if not _REVISION.fullmatch(revision):
        raise StoreError("invalid_revision")
    return revision


def _manifest(raw: bytes) -> Record:
    """Validate paths, schema versions and unique logical identities before any fetch."""
    try:
        if len(raw) > _MAX_BYTES:
            raise ValueError
        value = json.loads(raw)
        if (not isinstance(value, dict) or set(value) != {"schema_version", "record_schemas", "shards"}
                or value["schema_version"] != _VERSION or value["record_schemas"] != _SCHEMAS
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


def _members(manifest: Record, kind: str) -> dict[str, str]:
    return {record["identity"]: record["sha256"] for shard in manifest["shards"][kind]
            for record in shard["records"]}


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


def _validate_shard(data: bytes, shard: Record, kind: str) -> tuple[Record, ...]:
    if len(data) != shard["bytes"] or _sha(data) != shard["sha256"]:
        raise StoreError("shard_digest_mismatch")
    lines = data.splitlines(keepends=True)
    if len(lines) != shard["count"] or any(not line.endswith(b"\n") for line in lines):
        raise StoreError("malformed_or_interrupted_shard")
    records = []
    for line, member in zip(lines, shard["records"], strict=True):
        try:
            record = _PARSERS[kind](line)
        except (ValueError, TypeError, RecursionError, OverflowError):
            raise StoreError("invalid_or_unknown_record_schema") from None
        if (_identity(record, kind) != member["identity"] or _sha(line[:-1]) != member["sha256"]
                or _json_bytes(record) != line):
            raise StoreError("shard_record_identity_mismatch")
        _scan(record)
        records.append(record)
    return tuple(records)


def _remote_records(
    backend: DatasetHub, repo_id: str, revision: str, manifest: Record,
) -> dict[str, tuple[Record, ...]]:
    records: dict[str, tuple[Record, ...]] = {}
    for kind in _SCHEMAS:
        loaded: list[Record] = []
        for shard in manifest["shards"][kind]:
            data = backend.read_file(repo_id, shard["path"], revision)
            if data is None:
                raise StoreError("missing_shard")
            loaded.extend(_validate_shard(data, shard, kind))
        records[kind] = tuple(loaded)
    validate_record_targets(records["runs"], records["observations"])
    return records


@dataclass(frozen=True)
class UploadStatus:
    queued: int
    published: int
    failed: int = 0
    error: str | None = None
    revision: str | None = None


class DatasetUploader:
    """Local records are the queue; receipts acknowledge only verified atomic HF commits."""
    def __init__(
        self, store: LocalRecordStore, repo_id: str | None = None, *, backend: DatasetHub | None = None,
        max_shard_records: int = 100, max_shard_bytes: int = 16 * 1024 * 1024,
    ) -> None:
        if max_shard_records <= 0 or not 0 < max_shard_bytes <= _MAX_BYTES:
            raise ValueError("invalid shard limits")
        self.store, self.repo_id, self.backend = store, repo_id, backend
        self.max_shard_records, self.max_shard_bytes = max_shard_records, max_shard_bytes
        key = _sha((repo_id or "disabled").encode())
        self.queue = store.root / "uploads" / key

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
            data = handle.read(_MAX_BYTES + 1)
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
            revision = None if value["revision"] is None else _validate_revision(value["revision"])
            return _manifest(_json_bytes(value["manifest"])), revision, value["error"]
        except (ValueError, TypeError, KeyError, RecursionError):
            raise StoreError("invalid_upload_receipt") from None

    def _status(self, records: Mapping[str, Sequence[Record]], manifest: Record,
                revision: str | None, error: str | None = None) -> UploadStatus:
        published = queued = 0
        for kind in _SCHEMAS:
            known = _members(manifest, kind)
            for record in records[kind]:
                if known.get(_identity(record, kind)) == _sha(_json_bytes(record)[:-1]):
                    published += 1
                else:
                    queued += 1
        return UploadStatus(queued, published, queued if error else 0, error, revision)

    def status(self) -> UploadStatus:
        """Offline receipt status; credentials do not activate an uploader."""
        with self._locked():
            manifest, revision, error = self._receipt()
            return self._status(self.store.read_records(), manifest, revision, error)

    def _seal(self, records: Sequence[Record], kind: str) -> tuple[list[Record], dict[str, bytes]]:
        shards: list[Record] = []
        files: dict[str, bytes] = {}
        batch: list[bytes] = []
        members: list[Record] = []
        size = 0

        def seal() -> None:
            if not batch:
                return
            data = b"".join(batch)
            digest = _sha(data)
            path = f"{kind}/{digest}.jsonl"
            local = self.queue / f"{kind}-{digest}.jsonl"
            previous = self._read(local)
            if previous is not None and previous != data:
                raise StoreError("queued_shard_digest_mismatch")
            if previous is None:
                self._write(local, data)
            shards.append({"path": path, "sha256": digest, "bytes": len(data),
                           "count": len(batch), "records": list(members)})
            files[path] = data
            batch.clear()
            members.clear()

        for record in sorted(records, key=lambda item: _identity(item, kind)):
            _scan(record)
            data = _json_bytes(record)
            if len(data) > self.max_shard_bytes:
                raise StoreError("record_exceeds_shard_limit")
            if batch and (len(batch) >= self.max_shard_records or size + len(data) > self.max_shard_bytes):
                seal()
                size = 0
            batch.append(data)
            members.append({"identity": _identity(record, kind), "sha256": _sha(data[:-1])})
            size += len(data)
        seal()
        return shards, files

    def upload(self) -> UploadStatus:
        """Retry from current remote membership; retain local evidence after any failure."""
        records: dict[str, tuple[Record, ...]] = {kind: () for kind in _SCHEMAS}
        receipt, revision, error = _empty_manifest(), None, None
        try:
            with self._locked():
                receipt, revision, error = self._receipt()
                records = self.store.read_records()
                if self.repo_id is None:
                    return self._status(records, receipt, revision)
                _validate_repo(self.repo_id)
                # Seal privately before any network operation; incomplete staging never acknowledges records.
                for kind in _SCHEMAS:
                    self._seal(records[kind], kind)
                backend = self.backend or HfDatasetHub()
                for _attempt in range(5):
                    head = _validate_revision(backend.private_revision(self.repo_id))
                    raw = backend.read_file(self.repo_id, MANIFEST_PATH, head)
                    remote = _empty_manifest() if raw is None else _manifest(raw)
                    _remote_records(backend, self.repo_id, head, remote)
                    files: dict[str, bytes] = {}
                    merged = json.loads(canonical_json(remote))
                    for kind in _SCHEMAS:
                        known = _members(remote, kind)
                        pending = []
                        for record in records[kind]:
                            identity = _identity(record, kind)
                            digest = _sha(_json_bytes(record)[:-1])
                            if identity in known:
                                if known[identity] != digest:
                                    raise StoreError("immutable_identity_conflict")
                            else:
                                pending.append(record)
                        shards, additions = self._seal(pending, kind)
                        merged["shards"][kind].extend(shards)
                        files.update(additions)
                    if files:
                        files[MANIFEST_PATH] = _json_bytes(merged)
                        _manifest(files[MANIFEST_PATH])
                        try:
                            head = _validate_revision(backend.commit(self.repo_id, files, head))
                        except HubConflict:
                            continue
                        raw = backend.read_file(self.repo_id, MANIFEST_PATH, head)
                        if raw is None or _manifest(raw) != merged:
                            raise StoreError("publication_not_confirmed")
                        _remote_records(backend, self.repo_id, head, merged)
                    self._write(self.queue / "receipt.json", _json_bytes(
                        {"manifest": merged, "revision": head, "error": None}))
                    # Evidence remains immutable; only acknowledged sealed staging is removed.
                    for path in self.queue.glob("*.jsonl"):
                        if not path.is_symlink() and path.is_file():
                            path.unlink()
                    return self._status(records, merged, head)
                raise StoreError("concurrent_update_retry_exhausted")
        except (StoreError, HubError) as failure:
            error = failure.code
        except Exception:
            error = "upload_failed"
        # Persist a value-free failure without ever discarding a prior acknowledgement.
        try:
            with self._locked():
                current_receipt, current_revision, _ = self._receipt()
                receipt, revision = current_receipt, current_revision
                self._write(self.queue / "receipt.json", _json_bytes(
                    {"manifest": receipt, "revision": revision, "error": error}))
        except (StoreError, OSError):
            pass
        return self._status(records, receipt, revision, error)


def download_snapshot(
    repo_id: str, revision: str, destination: Path, *, backend: DatasetHub | None = None,
) -> LocalRecordStore:
    """Fetch exactly one private HF commit and validate all bytes before exposing records."""
    _validate_repo(repo_id)
    _validate_revision(revision)
    try:
        client = backend or HfDatasetHub()
        _validate_revision(client.private_revision(repo_id))
        raw = client.read_file(repo_id, MANIFEST_PATH, revision)
        if raw is None:
            raise StoreError("missing_manifest")
        manifest = _manifest(raw)
        records = _remote_records(client, repo_id, revision, manifest)
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
