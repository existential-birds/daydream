"""Immutable adjudication checkpoints and complete final-bundle publication.

State checkpoints use immutable batches plus a curation-scoped stable pointer.
Final publication uses a verified data commit followed by a canonical success
marker commit. Every mutable-head write is a branch-aware compare-and-swap;
all reads are pinned to one validated private-repository commit.

Credentials (C1): this module never reads or embeds tokens; ``HF_TOKEN`` is
consumed only by the real ``HfHubClient`` wiring.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Protocol, runtime_checkable

from daydream.archive.hydrate import (
    ANNOTATION_BRANCH,
    HubConcurrentUpdateError,
    HubUnavailableError,
    HydrationError,
    PublicDestinationError,
    RepoInfo,
)
from daydream.archive.hydrate_rules import derive_curation_id_v2
from daydream.training.adjudication.final_bundle import (
    FINAL_IDENTITY_FILES,
    _bundle_input_names,
)
from daydream.training.labeler_versions import ANNOTATION_SNAPSHOT_SCHEMA_VERSION
from daydream.trajectory import redact_text

__all__ = [
    "annotation_prefix",
    "download_final_annotation_bundle",
    "publish_annotation_state",
    "publish_final_annotation_bundle",
    "resume_annotation_state",
]

# State files published under the snapshot prefix (stable order for digests).
_STATE_FILES = ("queue.json", "observations.jsonl", "preview-ledger.json")
# The archive index (imported label_observations rows) is published alongside
# the state files whenever it exists in the state dir: a fresh-VM resume must
# restore the import itself, not just the adjudication state.
_ARCHIVE_INDEX_FILE = "index.db"
_MANIFEST_FILENAME = "preview-manifest.json"
_CHECKPOINT_RELPATH = "checkpoints/batch-latest.json"
_CHECKPOINT_SCHEMA = "annotation-checkpoint/v1"
_ATOMIC_ATTEMPTS = 6
_FINAL_SEGMENT = "final"
_SUCCESS_FILENAME = "_SUCCESS"
_SUMS_FILENAME = "SHA256SUMS"
_PUBLICATION_MANIFEST_FILENAME = "publication-manifest.json"
_PUBLICATION_SCHEMA = "annotation-publication/v1"
_SUCCESS_SCHEMA = "annotation-success/v1"

# HF_TOKEN-shaped values (fail-closed secret scan, S1). A hit means a live
# credential leaked into a payload; publication is refused, never scrubbed.
# The threshold is deliberately low ({8,}) — a false positive merely blocks
# publication, while a miss would leak a live token to a remote dataset repo.
_SECRET_SHAPES = re.compile(r"(?:hf_[0-9A-Za-z]{8,}|github_pat_[0-9A-Za-z_]{8,}|ghp_[0-9A-Za-z]{8,})")


def _read_manifest_data(manifest: Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(manifest, Mapping):
        return dict(manifest)
    try:
        data: dict[str, Any] = json.loads(Path(manifest).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"publish: unreadable preview manifest {manifest}: {exc}") from exc
    return data


def annotation_prefix(manifest: Path | Mapping[str, Any]) -> str:
    """Content-addressed remote prefix ``annotations/<curation-id>/<snapshot-id>/``.

    Both pin components must be present and non-empty, else ``ValueError``
    naming the offending field (M3).
    """
    data = _read_manifest_data(manifest)
    curation_id = data.get("curation_id")
    snapshot_id = data.get("snapshot_id")
    if not isinstance(curation_id, str) or not curation_id:
        raise ValueError("annotation_prefix: manifest is missing required field 'curation_id'")
    if not isinstance(snapshot_id, str) or not snapshot_id:
        raise ValueError("annotation_prefix: manifest is missing required field 'snapshot_id'")
    return f"annotations/{curation_id}/{snapshot_id}/"


# SQLite archives (index.db) are binary, not UTF-8; scan them as latin-1
# (lossless byte->char mapping, so regex hits still fire on ASCII credentials
# embedded in the file) instead of rejecting the payload outright.
_BINARY_STATE_FILES = frozenset({"index.db"})


def _scan_for_secrets(name: str, data: bytes) -> None:
    """Fail-closed secret scan (S1): refuse to upload credential-shaped payloads."""
    try:
        if name in _BINARY_STATE_FILES:
            # latin-1 maps every byte to a code point losslessly; the ASCII
            # secret shapes the regex targets are still detectable in SQLite pages.
            text = data.decode("latin-1")
        else:
            text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PublicDestinationError(f"{name}: payload is not valid UTF-8: {exc}") from exc
    hit = _SECRET_SHAPES.search(text)
    if hit is not None:
        raise PublicDestinationError(f"{name}: refusing to publish: credential-shaped value detected in payload")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _merge_observations(remote: bytes | None, local: bytes) -> bytes:
    """Union canonical-row identities while retaining first-observed bytes."""

    def rows(data: bytes) -> list[tuple[str, bytes]]:
        result: list[tuple[str, bytes]] = []
        try:
            lines = data.decode("utf-8").splitlines(keepends=True)
        except UnicodeDecodeError as exc:
            raise ValueError(f"observations.jsonl is not valid UTF-8: {exc}") from None
        for line_no, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"observations.jsonl line {line_no} is invalid JSON: {exc}") from None
            if not isinstance(record, dict):
                raise ValueError(f"observations.jsonl line {line_no} is not a JSON object")
            canonical = _canonical_json_bytes(record)
            raw = line.encode("utf-8")
            if not raw.endswith((b"\n", b"\r")):
                raw += b"\n"
            result.append((_digest(canonical), raw))
        return result

    merged: list[bytes] = []
    seen: set[str] = set()
    for key, row in [*rows(remote or b""), *rows(local)]:
        if key not in seen:
            seen.add(key)
            merged.append(row)
    return b"".join(merged)


@runtime_checkable
class AnnotationHubClient(Protocol):
    def repo_info(self, revision: str | None = None) -> RepoInfo: ...

    def list_repo_files(self, revision: str) -> list[str]: ...

    def download_file(self, path_in_repo: str, revision: str) -> bytes: ...

    def commit_files_atomic(
        self,
        mapping: dict[str | Path, Path],
        commit_message: str,
        *,
        parent_commit: str,
        branch: str,
    ) -> str: ...


def _canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _validate_identifier(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value or value in {".", ".."}:
        raise ValueError(f"{name} must be a non-empty identifier")
    if "/" in value or "\\" in value or not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise ValueError(f"{name} contains an invalid path character")
    _scan_for_secrets(name, value.encode("utf-8"))
    return value


def _validate_oid(value: Any, *, what: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
        raise HydrationError(f"{what} is not a lowercase 40-hex commit OID")
    return value


def _validate_remote_path(path: Any, *, allowed: set[str], what: str) -> str:
    if not isinstance(path, str) or not path or "\\" in path or path.endswith("/"):
        raise ValueError(f"{what}: invalid remote path")
    pure = PurePosixPath(path)
    if pure.is_absolute() or str(pure) != path or any(part in {"", ".", ".."} for part in path.split("/")):
        raise ValueError(f"{what}: invalid remote path")
    if path not in allowed:
        raise ValueError(f"{what}: remote path {path!r} is outside the exact allowlist")
    _scan_for_secrets(what, path.encode("utf-8"))
    return path


def _read_regular_file(path: Path, *, label: str) -> bytes:
    # Operator paths may legitimately live below platform aliases such as
    # macOS /tmp -> /private/tmp.  The declared leaf remains non-following:
    # state/bundle roots are checked by their callers and file leaves here.
    if path.is_symlink():
        raise PublicDestinationError(f"{label}: refusing symlinked input")
    if not path.is_file():
        raise FileNotFoundError(f"{label}: required regular file is missing")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise HydrationError(redact_text(f"{label}: cannot read input: {exc}")) from None


def _manifest_payload(manifest: Path | Mapping[str, Any]) -> tuple[dict[str, Any], bytes]:
    if isinstance(manifest, Mapping):
        data = dict(manifest)
        raw = _canonical_json_bytes(data)
    else:
        declared = Path(manifest)
        if declared.is_symlink():
            raise PublicDestinationError(f"{_MANIFEST_FILENAME}: refusing symlinked input")
        try:
            anchored_parent = declared.parent.resolve(strict=True)
        except OSError as exc:
            raise HydrationError(
                redact_text(f"{_MANIFEST_FILENAME}: cannot resolve input parent: {exc}")
            ) from None
        raw = _read_regular_file(anchored_parent / declared.name, label=_MANIFEST_FILENAME)
        try:
            parsed = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{_MANIFEST_FILENAME}: invalid JSON: {exc}") from None
        if not isinstance(parsed, dict):
            raise ValueError(f"{_MANIFEST_FILENAME}: expected a JSON object")
        data = parsed
    return data, raw


def _read_local_state(state_dir: Path, manifest: Path | Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, bytes]]:
    if state_dir.is_symlink() or not state_dir.is_dir():
        raise PublicDestinationError("state_dir must be a real directory, not a symlink")
    manifest_data, manifest_bytes = _manifest_payload(manifest)
    payloads = {
        name: _read_regular_file(state_dir / name, label=name)
        for name in _STATE_FILES
    }
    payloads[_MANIFEST_FILENAME] = manifest_bytes
    index_path = state_dir / _ARCHIVE_INDEX_FILE
    if index_path.is_symlink():
        raise PublicDestinationError(f"{_ARCHIVE_INDEX_FILE}: refusing symlinked input")
    if index_path.exists():
        payloads[_ARCHIVE_INDEX_FILE] = _read_regular_file(index_path, label=_ARCHIVE_INDEX_FILE)
    for name, data in payloads.items():
        _scan_for_secrets(name, data)
    return manifest_data, payloads


def _pin_repository(client: AnnotationHubClient, *, revision: str | None = None) -> str:
    requested = revision or ANNOTATION_BRANCH
    if revision is not None:
        _validate_oid(revision, what="requested revision")
    try:
        info = client.repo_info(revision=requested)
    except Exception as exc:
        raise HubUnavailableError(redact_text(f"annotation repo metadata unavailable: {exc}")) from None
    if info.private is not True:
        raise PublicDestinationError("refusing to publish or restore annotations from a public Hub repository")
    sha = _validate_oid(info.sha, what="annotation repository SHA")
    if revision is not None and sha != revision:
        raise HydrationError("annotation repository resolved a different commit than requested")
    return sha


def _list_remote(client: AnnotationHubClient, revision: str) -> set[str]:
    try:
        return set(client.list_repo_files(revision=revision))
    except Exception as exc:
        raise HubUnavailableError(redact_text(f"annotation repository listing failed: {exc}")) from None


def _download_remote(client: AnnotationHubClient, path: str, revision: str) -> bytes:
    try:
        return client.download_file(path, revision=revision)
    except Exception as exc:
        raise HubUnavailableError(redact_text(f"annotation repository download failed: {exc}")) from None


def _state_root(curation_id: str) -> str:
    return f"annotations/{curation_id}/"


def _stable_pointer_path(curation_id: str) -> str:
    return f"{_state_root(curation_id)}{_CHECKPOINT_RELPATH}"


def _batch_identity(payloads: Mapping[str, bytes]) -> tuple[str, list[dict[str, str]]]:
    entries = [
        {"path": name, "sha256": _digest(data)}
        for name, data in sorted(payloads.items())
    ]
    return _digest(_canonical_json_bytes({"files": entries})), entries


def _build_checkpoint(
    curation_id: str,
    snapshot_id: str,
    payloads: Mapping[str, bytes],
) -> tuple[str, str, bytes]:
    required = {*_STATE_FILES, _MANIFEST_FILENAME}
    allowed = {*required, _ARCHIVE_INDEX_FILE}
    names = set(payloads)
    if not required.issubset(names) or not names.issubset(allowed):
        raise HydrationError("refusing to commit an incomplete or foreign checkpoint batch")
    batch_id, entries = _batch_identity(payloads)
    batch_prefix = f"{_state_root(curation_id)}checkpoints/batches/{batch_id}/"
    observations = [json.loads(line) for line in payloads["observations.jsonl"].splitlines() if line.strip()]
    latest = max(
        (row.get("observed_at") for row in observations if isinstance(row.get("observed_at"), str)),
        default=None,
    )
    pointer = {
        "schema_version": _CHECKPOINT_SCHEMA,
        "curation_id": curation_id,
        "snapshot_id": snapshot_id,
        "batch_id": batch_id,
        "batch_prefix": batch_prefix,
        "files": entries,
        "observation_count": len(observations),
        "latest_observed_at": latest,
    }
    return batch_id, batch_prefix, _canonical_json_bytes(pointer)


def _parse_checkpoint(
    client: AnnotationHubClient,
    *,
    revision: str,
    curation_id: str,
    remote_files: set[str],
) -> tuple[dict[str, Any], dict[str, bytes], bytes] | None:
    stable_path = _stable_pointer_path(curation_id)
    _validate_remote_path(stable_path, allowed={stable_path}, what="checkpoint pointer")
    if stable_path not in remote_files:
        return None
    raw = _download_remote(client, stable_path, revision)
    _scan_for_secrets("checkpoint pointer", raw)
    try:
        pointer = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{_CHECKPOINT_RELPATH}: corrupt checkpoint JSON: {exc}") from None
    if not isinstance(pointer, dict) or pointer.get("schema_version") != _CHECKPOINT_SCHEMA:
        raise ValueError(f"{_CHECKPOINT_RELPATH}: unsupported checkpoint schema")
    if pointer.get("curation_id") != curation_id:
        raise ValueError(f"{_CHECKPOINT_RELPATH}: curation identity mismatch")
    snapshot_id = _validate_identifier("snapshot_id", pointer.get("snapshot_id"))
    batch_id = pointer.get("batch_id")
    if not isinstance(batch_id, str) or re.fullmatch(r"[0-9a-f]{64}", batch_id) is None:
        raise ValueError(f"{_CHECKPOINT_RELPATH}: invalid batch identity")
    expected_prefix = f"{_state_root(curation_id)}checkpoints/batches/{batch_id}/"
    if pointer.get("batch_prefix") != expected_prefix:
        raise ValueError(f"{_CHECKPOINT_RELPATH}: invalid batch path")
    entries = pointer.get("files")
    if not isinstance(entries, list):
        raise ValueError(f"{_CHECKPOINT_RELPATH}: files must be a list")
    allowed_names = {*_STATE_FILES, _MANIFEST_FILENAME, _ARCHIVE_INDEX_FILE}
    required_names = {*_STATE_FILES, _MANIFEST_FILENAME}
    seen: set[str] = set()
    payloads: dict[str, bytes] = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise ValueError(f"{_CHECKPOINT_RELPATH}: invalid file entry")
        name = _validate_remote_path(entry.get("path"), allowed=allowed_names, what="checkpoint file")
        if name in seen:
            raise ValueError(f"{_CHECKPOINT_RELPATH}: duplicate normalized path {name!r}")
        seen.add(name)
        expected_digest = entry.get("sha256")
        if not isinstance(expected_digest, str) or re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None:
            raise ValueError(f"{_CHECKPOINT_RELPATH}: invalid digest for {name!r}")
        remote_path = f"{expected_prefix}{name}"
        allowed_remote = {f"{expected_prefix}{allowed}" for allowed in allowed_names}
        _validate_remote_path(remote_path, allowed=allowed_remote, what="checkpoint payload")
        if remote_path not in remote_files:
            raise ValueError(f"{_CHECKPOINT_RELPATH}: missing remote payload {name!r}")
        data = _download_remote(client, remote_path, revision)
        _scan_for_secrets(name, data)
        if _digest(data) != expected_digest:
            raise ValueError(f"digest mismatch for {name!r}")
        payloads[name] = data
    if not required_names.issubset(seen) or not seen.issubset(allowed_names):
        raise ValueError(f"{_CHECKPOINT_RELPATH}: incomplete or foreign file set")
    derived_id, _ = _batch_identity(payloads)
    if derived_id != batch_id:
        raise ValueError(f"{_CHECKPOINT_RELPATH}: batch identity mismatch")
    manifest_data = json.loads(payloads[_MANIFEST_FILENAME])
    if not isinstance(manifest_data, dict) or manifest_data.get("curation_id") != curation_id:
        raise ValueError(f"{_CHECKPOINT_RELPATH}: preview manifest curation mismatch")
    if manifest_data.get("snapshot_id") != snapshot_id:
        raise ValueError(f"{_CHECKPOINT_RELPATH}: preview manifest snapshot mismatch")
    return pointer, payloads, raw


_ABSENT = object()


def _three_way_state(
    base: Mapping[str, bytes],
    remote: Mapping[str, bytes],
    local: Mapping[str, bytes],
) -> dict[str, bytes]:
    resolved: dict[str, bytes] = {
        "observations.jsonl": _merge_observations(remote.get("observations.jsonl"), local["observations.jsonl"])
    }
    for name in sorted(({*base, *remote, *local} - {"observations.jsonl"})):
        base_value = base.get(name, _ABSENT)
        remote_value = remote.get(name, _ABSENT)
        local_value = local.get(name, _ABSENT)
        if remote_value == local_value:
            selected = remote_value
        elif remote_value == base_value:
            selected = local_value
        elif local_value == base_value:
            selected = remote_value
        else:
            raise HydrationError(f"concurrent state conflict for {name}")
        if isinstance(selected, bytes):
            resolved[name] = selected
    return resolved


def _state_result(pointer: Mapping[str, Any], revision: str, uploaded: list[str]) -> dict[str, Any]:
    return {
        "prefix": _state_root(str(pointer["curation_id"])),
        "batch_id": pointer["batch_id"],
        "batch_prefix": pointer["batch_prefix"],
        "checkpoint_revision": revision,
        "observation_count": pointer["observation_count"],
        "uploaded": uploaded,
    }


def publish_annotation_state(
    client: AnnotationHubClient,
    state_dir: Path,
    *,
    manifest: Path | Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically publish one immutable state batch and its stable pointer."""
    state_dir = Path(state_dir)
    if state_dir.is_symlink():
        raise PublicDestinationError("state_dir must be a real directory, not a symlink")
    try:
        state_dir = state_dir.resolve(strict=True)
    except OSError as exc:
        raise PublicDestinationError(redact_text(f"state_dir cannot be resolved: {exc}")) from None
    if not state_dir.is_dir():
        raise PublicDestinationError("state_dir must be a real directory, not a symlink")
    manifest_data, local_payloads = _read_local_state(state_dir, manifest)
    curation_id = _validate_identifier("curation_id", manifest_data.get("curation_id"))
    snapshot_id = _validate_identifier("snapshot_id", manifest_data.get("snapshot_id"))
    initial_revision = _pin_repository(client)
    initial_files = _list_remote(client, initial_revision)
    initial_checkpoint = _parse_checkpoint(
        client,
        revision=initial_revision,
        curation_id=curation_id,
        remote_files=initial_files,
    )
    base_payloads = initial_checkpoint[1] if initial_checkpoint is not None else {}
    current_revision = initial_revision
    current_files = initial_files
    current_checkpoint = initial_checkpoint

    for _attempt in range(_ATOMIC_ATTEMPTS):
        if current_checkpoint is not None and current_checkpoint[0]["snapshot_id"] != snapshot_id:
            raise HydrationError("published checkpoint was superseded by a different snapshot identity")
        current_payloads = current_checkpoint[1] if current_checkpoint is not None else {}
        resolved = _three_way_state(base_payloads, current_payloads, local_payloads)
        for name, data in resolved.items():
            _scan_for_secrets(name, data)
        batch_id, batch_prefix, pointer_bytes = _build_checkpoint(curation_id, snapshot_id, resolved)
        _scan_for_secrets("checkpoint pointer", pointer_bytes)
        pointer = json.loads(pointer_bytes)
        if current_checkpoint is not None and current_checkpoint[2] == pointer_bytes:
            return _state_result(pointer, current_revision, [])

        batch_mapping_bytes = {f"{batch_prefix}{name}": data for name, data in resolved.items()}
        stable_path = _stable_pointer_path(curation_id)
        allowed_paths = {*batch_mapping_bytes, stable_path}
        for path in allowed_paths:
            _validate_remote_path(path, allowed=allowed_paths, what="annotation checkpoint upload")
            _scan_for_secrets("annotation checkpoint upload path", path.encode("utf-8"))
        commit_message = f"daydream adjudication checkpoint {curation_id} {batch_id}"
        _scan_for_secrets("annotation checkpoint commit message", commit_message.encode("utf-8"))
        for path, expected in batch_mapping_bytes.items():
            if path in current_files and _download_remote(client, path, current_revision) != expected:
                raise HydrationError(f"immutable checkpoint collision for batch {batch_id}")

        with tempfile.TemporaryDirectory(prefix=".daydream-checkpoint-", dir=state_dir.parent) as temp:
            staging = Path(temp)
            mapping: dict[str | Path, Path] = {}
            for index, (path, data) in enumerate(sorted(batch_mapping_bytes.items())):
                local = staging / f"payload-{index}"
                local.write_bytes(data)
                mapping[path] = local
            pointer_path = staging / "pointer"
            pointer_path.write_bytes(pointer_bytes)
            mapping[stable_path] = pointer_path
            try:
                revision = client.commit_files_atomic(
                    mapping,
                    commit_message,
                    parent_commit=current_revision,
                    branch=ANNOTATION_BRANCH,
                )
            except HubConcurrentUpdateError:
                current_revision = _pin_repository(client)
                current_files = _list_remote(client, current_revision)
                current_checkpoint = _parse_checkpoint(
                    client,
                    revision=current_revision,
                    curation_id=curation_id,
                    remote_files=current_files,
                )
                if initial_checkpoint is not None and current_checkpoint is None:
                    raise HydrationError(
                        "published checkpoint pointer disappeared during concurrent update"
                    )
                continue
            except Exception as exc:
                raise HubUnavailableError(redact_text(f"annotation checkpoint commit failed: {exc}")) from None
        _validate_oid(revision, what="checkpoint revision")
        return _state_result(pointer, revision, sorted(map(str, mapping)))
    raise HubUnavailableError("annotation checkpoint could not win the bounded concurrent-update race")


def _final_prefix(curation_id: str, final_id: str) -> str:
    return f"annotations/{curation_id}/{final_id}/{_FINAL_SEGMENT}/"


def _parse_json_object(name: str, data: bytes) -> dict[str, Any]:
    _scan_for_secrets(name, data)
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name}: invalid JSON: {exc}") from None
    if not isinstance(value, dict):
        raise ValueError(f"{name}: expected a JSON object")
    return value


def _report_count(value: Any, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"coverage-report.json: {field} must be a non-negative integer")
    return value


def _validate_coverage_report(data: bytes) -> None:
    report = _parse_json_object("coverage-report.json", data)
    expected = {
        "outcome_coverage",
        "silver_task_only_count",
        "class_balance",
        "unresolved",
        "inter_rater",
        "strata",
        "evidence_after_as_of",
        "admission_gate",
    }
    if set(report) != expected:
        raise ValueError("coverage-report.json: invalid producer field set")
    if data != _canonical_json_bytes(report):
        raise ValueError("coverage-report.json: non-canonical producer encoding")

    coverage = report["outcome_coverage"]
    balance = report["class_balance"]
    inter_rater = report["inter_rater"]
    gate = report["admission_gate"]
    if not isinstance(coverage, dict) or set(coverage) != {"adjudicated", "total"}:
        raise ValueError("coverage-report.json: invalid outcome_coverage")
    if not isinstance(balance, dict) or set(balance) != {"accepted", "rejected"}:
        raise ValueError("coverage-report.json: invalid class_balance")
    if not isinstance(inter_rater, dict) or set(inter_rater) != {"items", "agreeing"}:
        raise ValueError("coverage-report.json: invalid inter_rater")
    if not isinstance(gate, dict) or set(gate) != {
        "outcome_bearing_total",
        "total",
        "passes_80pct",
        "class_balance_ok",
        "gate_version",
    }:
        raise ValueError("coverage-report.json: invalid admission_gate")

    adjudicated = _report_count(coverage["adjudicated"], field="outcome_coverage.adjudicated")
    total = _report_count(coverage["total"], field="outcome_coverage.total")
    accepted = _report_count(balance["accepted"], field="class_balance.accepted")
    rejected = _report_count(balance["rejected"], field="class_balance.rejected")
    unresolved = _report_count(report["unresolved"], field="unresolved")
    inter_items = _report_count(inter_rater["items"], field="inter_rater.items")
    inter_agreeing = _report_count(inter_rater["agreeing"], field="inter_rater.agreeing")
    _report_count(report["silver_task_only_count"], field="silver_task_only_count")
    gate_outcomes = _report_count(gate["outcome_bearing_total"], field="admission_gate.outcome_bearing_total")
    gate_total = _report_count(gate["total"], field="admission_gate.total")
    gate_version = _report_count(gate["gate_version"], field="admission_gate.gate_version")
    if type(gate["passes_80pct"]) is not bool or type(gate["class_balance_ok"]) is not bool:
        raise ValueError("coverage-report.json: admission gate flags must be booleans")
    strata = report["strata"]
    if not isinstance(strata, dict) or any(
        not isinstance(key, str) or not key or type(value) is not int or value < 0
        for key, value in strata.items()
    ):
        raise ValueError("coverage-report.json: invalid strata counts")
    evidence = report["evidence_after_as_of"]
    if not isinstance(evidence, list) or any(not isinstance(value, str) for value in evidence):
        raise ValueError("coverage-report.json: invalid evidence_after_as_of")
    if evidence != sorted(evidence):
        raise ValueError("coverage-report.json: evidence_after_as_of is not sorted")

    expected_passes = total > 0 and adjudicated * 5 >= total * 4
    expected_balance = accepted > 0 and rejected > 0
    if (
        adjudicated != total
        or accepted + rejected != total
        or unresolved > total
        or inter_agreeing > inter_items
        or gate_outcomes != adjudicated
        or gate_total != total
        or gate["passes_80pct"] is not expected_passes
        or gate["class_balance_ok"] is not expected_balance
        or gate_version != 1
    ):
        raise ValueError("coverage-report.json: contradictory producer counts or admission gate")
    if gate["passes_80pct"] is not True:
        raise PublicDestinationError(
            "coverage-report.json: refusing to publish final bundle that fails the 80% admission gate"
        )


def _validate_final_semantics(payloads: Mapping[str, bytes]) -> tuple[str, str]:
    preview = _parse_json_object(_MANIFEST_FILENAME, payloads[_MANIFEST_FILENAME])
    binding = _parse_json_object("policy-binding.json", payloads["policy-binding.json"])
    lineage = _parse_json_object("lineage.json", payloads["lineage.json"])
    curation_id = _validate_identifier("curation_id", preview.get("curation_id"))
    source = preview.get("source_hub_commit")
    sanitized = preview.get("sanitized_hub_commit")
    if not isinstance(source, str) or re.fullmatch(r"[0-9a-f]{40}", source) is None:
        raise ValueError("preview-manifest.json: invalid source_hub_commit")
    if sanitized != source:
        raise ValueError("preview-manifest.json: sanitized/source commit mismatch")
    source_snapshot_id = preview.get("snapshot_id")
    if not isinstance(source_snapshot_id, str) or re.fullmatch(r"[0-9a-f]{64}", source_snapshot_id) is None:
        raise ValueError("preview-manifest.json: invalid snapshot_id")
    shared_pins = ("labeler_version", "rubric_version", "classifier_version")
    if any(not isinstance(preview.get(field), str) or not preview[field] for field in shared_pins):
        raise ValueError("preview-manifest.json: invalid producer version pin")
    if "as_of" not in preview or not (preview["as_of"] is None or isinstance(preview["as_of"], str)):
        raise ValueError("preview-manifest.json: invalid as_of pin")
    expected_fields = {
        "schema_version",
        "policy_digest",
        "policy_version",
        "allow_copyleft",
        "exclusions_digest",
        "resolved_decisions_digest",
        "distribution_digest",
    }
    if set(binding) != expected_fields or binding.get("schema_version") != "2":
        raise ValueError("policy-binding.json: invalid v2 field set")
    canonical_binding = (json.dumps(binding, sort_keys=True) + "\n").encode("utf-8")
    if payloads["policy-binding.json"] != canonical_binding:
        raise ValueError("policy-binding.json: non-canonical encoding")
    digests = (
        "policy_digest",
        "exclusions_digest",
        "resolved_decisions_digest",
        "distribution_digest",
    )
    if any(
        not isinstance(binding.get(name), str)
        or re.fullmatch(r"[0-9a-f]{64}", binding[name]) is None
        for name in digests
    ):
        raise ValueError("policy-binding.json: invalid digest")
    policy_version = binding.get("policy_version")
    allow_copyleft = binding.get("allow_copyleft")
    if (
        not isinstance(policy_version, str)
        or not policy_version
        or not isinstance(allow_copyleft, list)
        or any(
            not isinstance(slug, str) or not slug or slug != slug.casefold()
            for slug in allow_copyleft
        )
        or allow_copyleft != sorted(set(allow_copyleft))
    ):
        raise ValueError("policy-binding.json: invalid policy fields")
    derived = derive_curation_id_v2(
        source,
        binding["policy_digest"],
        policy_version,
        frozenset(allow_copyleft),
        binding["exclusions_digest"],
        binding["resolved_decisions_digest"],
        binding["distribution_digest"],
    )
    if derived != curation_id:
        raise ValueError("policy-binding.json: curation identity mismatch")
    lineage_fields = {
        "curation_id",
        "sanitized_hub_commit",
        "snapshot_id",
        "labeler_version",
        "rubric_version",
        "classifier_version",
        "schema_version",
        "batch_fileset_digest",
        "as_of",
    }
    if set(lineage) != lineage_fields:
        raise ValueError("lineage.json: invalid producer field set")
    if payloads["lineage.json"] != _canonical_json_bytes(lineage):
        raise ValueError("lineage.json: non-canonical producer encoding")
    expected_schema = f"annotation-snapshot/{ANNOTATION_SNAPSHOT_SCHEMA_VERSION}"
    if lineage.get("schema_version") != expected_schema:
        raise ValueError("lineage.json: unsupported producer schema")
    if (
        lineage.get("curation_id") != curation_id
        or lineage.get("sanitized_hub_commit") != source
        or lineage.get("snapshot_id") != source_snapshot_id
        or any(lineage.get(field) != preview[field] for field in shared_pins)
        or lineage.get("as_of") != ("" if preview["as_of"] is None else preview["as_of"])
    ):
        raise ValueError("lineage.json: producer pin linkage mismatch")
    batch_digest = lineage.get("batch_fileset_digest")
    if not isinstance(batch_digest, str) or re.fullmatch(r"[0-9a-f]{64}", batch_digest) is None:
        raise ValueError("lineage.json: invalid batch_fileset_digest")
    _validate_coverage_report(payloads["coverage-report.json"])
    return curation_id, source_snapshot_id


def _prepare_final_bundle(bundle_dir: Path) -> tuple[str, str, str, dict[str, bytes]]:
    local_names = _bundle_input_names(bundle_dir)
    if local_names != set(FINAL_IDENTITY_FILES):
        raise ValueError("final publication input must contain exactly the seven semantic files")
    payloads: dict[str, bytes] = {}
    for name in FINAL_IDENTITY_FILES:
        path = bundle_dir / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"final bundle identity input {name!r} must be a regular file")
        payloads[name] = _read_regular_file(path, label=name)
    # Bind identity, validation, checksums, and upload to one captured byteset.
    # A concurrent local edit must not separate an immutable ID from its data.
    digests = {name: _digest(data) for name, data in payloads.items()}
    final_id = _digest(_canonical_json_bytes(digests))
    for name, data in payloads.items():
        _scan_for_secrets(name, data)
    curation_id, source_snapshot_id = _validate_final_semantics(payloads)
    publication = _canonical_json_bytes(
        {
            "schema_version": _PUBLICATION_SCHEMA,
            "curation_id": curation_id,
            "source_snapshot_id": source_snapshot_id,
            "final_snapshot_id": final_id,
            "files": digests,
        }
    )
    _scan_for_secrets(_PUBLICATION_MANIFEST_FILENAME, publication)
    payloads[_PUBLICATION_MANIFEST_FILENAME] = publication
    sums = "".join(
        f"{_digest(data)}  {name}\n" for name, data in sorted(payloads.items())
    ).encode("utf-8")
    _scan_for_secrets(_SUMS_FILENAME, sums)
    payloads[_SUMS_FILENAME] = sums
    return curation_id, source_snapshot_id, final_id, payloads


def _prefix_names(remote_files: set[str], prefix: str) -> set[str]:
    names: set[str] = set()
    for path in remote_files:
        if not path.startswith(prefix):
            continue
        name = path[len(prefix) :]
        allowed = {
            *FINAL_IDENTITY_FILES,
            _PUBLICATION_MANIFEST_FILENAME,
            _SUMS_FILENAME,
            _SUCCESS_FILENAME,
        }
        _validate_remote_path(name, allowed=allowed, what="final bundle file")
        if name in names:
            raise ValueError(f"duplicate normalized final bundle path {name!r}")
        names.add(name)
    return names


def _verify_prefix(
    client: AnnotationHubClient,
    *,
    revision: str,
    remote_files: set[str],
    prefix: str,
    expected: Mapping[str, bytes],
) -> dict[str, bytes]:
    names = _prefix_names(remote_files, prefix)
    if names != set(expected):
        raise HydrationError(
            f"immutable final bundle collision: expected {sorted(expected)}, found {sorted(names)}"
        )
    actual: dict[str, bytes] = {}
    allowed_paths = {f"{prefix}{name}" for name in expected}
    for name, wanted in sorted(expected.items()):
        path = f"{prefix}{name}"
        _validate_remote_path(path, allowed=allowed_paths, what="final bundle payload")
        data = _download_remote(client, path, revision)
        _scan_for_secrets(name, data)
        if data != wanted:
            raise HydrationError(f"immutable final bundle collision for {name}")
        actual[name] = data
    return actual


def _success_bytes(final_id: str, data_commit_oid: str) -> bytes:
    marker = _canonical_json_bytes(
        {
            "schema_version": _SUCCESS_SCHEMA,
            "final_snapshot_id": final_id,
            "data_commit_oid": _validate_oid(data_commit_oid, what="final data commit"),
        }
    )
    _scan_for_secrets(_SUCCESS_FILENAME, marker)
    return marker


def _parse_success(data: bytes, *, final_id: str) -> str:
    marker = _parse_json_object(_SUCCESS_FILENAME, data)
    if set(marker) != {"schema_version", "final_snapshot_id", "data_commit_oid"}:
        raise HydrationError("invalid final success marker field set")
    if marker.get("schema_version") != _SUCCESS_SCHEMA or marker.get("final_snapshot_id") != final_id:
        raise HydrationError("final success marker identity mismatch")
    data_oid = _validate_oid(marker.get("data_commit_oid"), what="success marker data commit")
    if data != _success_bytes(final_id, data_oid):
        raise HydrationError("final success marker is not canonically encoded")
    return data_oid


def _fresh_destination(path: Path, *, label: str) -> tuple[Path, Path]:
    requested = Path(path)
    if requested.exists() or requested.is_symlink():
        raise ValueError(f"{label} destination {requested} must not exist")
    try:
        parent = requested.parent.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} destination parent must be an existing directory: {exc}") from None
    if not parent.is_dir():
        raise ValueError(f"{label} destination parent must be an existing directory")
    anchored = parent / requested.name
    if anchored.exists() or anchored.is_symlink():
        raise ValueError(f"{label} destination {requested} must not exist")
    return requested, anchored


def _directory_identity(path: Path) -> tuple[int, int]:
    info = path.stat(follow_symlinks=False)
    if not stat.S_ISDIR(info.st_mode):
        raise HydrationError(f"installation staging path {path} is not a directory")
    return info.st_dev, info.st_ino


def _remove_owned_tree(path: Path, identity: tuple[int, int]) -> None:
    """Best-effort collision avoidance while the caller holds the owned inode.

    This is not a sandbox against a hostile same-parent writer: the pathname
    can still change between the identity check and recursive removal.
    """
    try:
        info = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError:
        return
    if stat.S_ISDIR(info.st_mode) and (info.st_dev, info.st_ino) == identity:
        shutil.rmtree(path, ignore_errors=True)


def _commit_final_mapping(
    client: AnnotationHubClient,
    *,
    bundle_dir: Path,
    prefix: str,
    payloads: Mapping[str, bytes],
    parent: str,
    message: str,
) -> str:
    with tempfile.TemporaryDirectory(prefix=".daydream-final-", dir=bundle_dir.parent) as temp:
        root = Path(temp)
        mapping: dict[str | Path, Path] = {}
        for index, (name, data) in enumerate(sorted(payloads.items())):
            local = root / str(index)
            local.write_bytes(data)
            mapping[f"{prefix}{name}"] = local
        try:
            revision = client.commit_files_atomic(
                mapping,
                message,
                parent_commit=parent,
                branch=ANNOTATION_BRANCH,
            )
        except HubConcurrentUpdateError:
            raise
        except Exception as exc:
            raise HubUnavailableError(redact_text(f"final annotation commit failed: {exc}")) from None
    return _validate_oid(revision, what="final annotation revision")


def _verified_existing_success(
    client: AnnotationHubClient,
    *,
    revision: str,
    remote_files: set[str],
    prefix: str,
    final_id: str,
    data_payloads: Mapping[str, bytes],
) -> tuple[str, bytes]:
    marker_path = f"{prefix}{_SUCCESS_FILENAME}"
    marker_bytes = _download_remote(client, marker_path, revision)
    data_oid = _parse_success(marker_bytes, final_id=final_id)
    expected_success = {**data_payloads, _SUCCESS_FILENAME: marker_bytes}
    _verify_prefix(
        client,
        revision=revision,
        remote_files=remote_files,
        prefix=prefix,
        expected=expected_success,
    )
    pinned_data = _pin_repository(client, revision=data_oid)
    data_files = _list_remote(client, pinned_data)
    _verify_prefix(
        client,
        revision=pinned_data,
        remote_files=data_files,
        prefix=prefix,
        expected=data_payloads,
    )
    return data_oid, marker_bytes


def _final_result(
    *,
    success_oid: str,
    data_oid: str,
    final_id: str,
    prefix: str,
    files: Mapping[str, bytes],
) -> dict[str, Any]:
    return {
        "hub_commit_sha": success_oid,
        "data_commit_sha": data_oid,
        "final_snapshot_id": final_id,
        "prefix": prefix,
        "files": sorted([*files, _SUCCESS_FILENAME]),
    }


def publish_final_annotation_bundle(
    client: AnnotationHubClient,
    bundle_dir: Path,
) -> dict[str, Any]:
    """Publish and verify a complete semantic bundle in data then success commits."""
    bundle_dir = Path(bundle_dir)
    if bundle_dir.is_symlink():
        raise ValueError("final bundle must be a real directory")
    try:
        bundle_dir = bundle_dir.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"final bundle cannot be resolved: {exc}") from None
    if not bundle_dir.is_dir():
        raise ValueError("final bundle must be a real directory")
    curation_id, _source_snapshot_id, final_id, data_payloads = _prepare_final_bundle(bundle_dir)
    prefix = _final_prefix(curation_id, final_id)
    commit_message = f"daydream annotation final data {curation_id} {final_id}"
    _scan_for_secrets("final commit message", commit_message.encode("utf-8"))

    data_oid: str | None = None
    for _attempt in range(_ATOMIC_ATTEMPTS):
        current = _pin_repository(client)
        remote_files = _list_remote(client, current)
        names = _prefix_names(remote_files, prefix)
        if _SUCCESS_FILENAME in names:
            existing_data_oid, _marker = _verified_existing_success(
                client,
                revision=current,
                remote_files=remote_files,
                prefix=prefix,
                final_id=final_id,
                data_payloads=data_payloads,
            )
            return _final_result(
                success_oid=current,
                data_oid=existing_data_oid,
                final_id=final_id,
                prefix=prefix,
                files=data_payloads,
            )
        if names:
            _verify_prefix(
                client,
                revision=current,
                remote_files=remote_files,
                prefix=prefix,
                expected=data_payloads,
            )
            data_oid = current
            break
        try:
            data_oid = _commit_final_mapping(
                client,
                bundle_dir=bundle_dir,
                prefix=prefix,
                payloads=data_payloads,
                parent=current,
                message=commit_message,
            )
        except HubConcurrentUpdateError:
            continue
        pinned_data = _pin_repository(client, revision=data_oid)
        data_files = _list_remote(client, pinned_data)
        _verify_prefix(
            client,
            revision=pinned_data,
            remote_files=data_files,
            prefix=prefix,
            expected=data_payloads,
        )
        break
    if data_oid is None:
        raise HubUnavailableError("final data publication could not win the concurrent-update race")

    for _attempt in range(_ATOMIC_ATTEMPTS):
        current = _pin_repository(client)
        remote_files = _list_remote(client, current)
        names = _prefix_names(remote_files, prefix)
        if _SUCCESS_FILENAME in names:
            existing_data_oid, _marker = _verified_existing_success(
                client,
                revision=current,
                remote_files=remote_files,
                prefix=prefix,
                final_id=final_id,
                data_payloads=data_payloads,
            )
            return _final_result(
                success_oid=current,
                data_oid=existing_data_oid,
                final_id=final_id,
                prefix=prefix,
                files=data_payloads,
            )
        _verify_prefix(
            client,
            revision=current,
            remote_files=remote_files,
            prefix=prefix,
            expected=data_payloads,
        )
        marker = _success_bytes(final_id, data_oid)
        success_message = f"daydream annotation final success {curation_id} {final_id}"
        _scan_for_secrets("final success commit message", success_message.encode("utf-8"))
        try:
            success_oid = _commit_final_mapping(
                client,
                bundle_dir=bundle_dir,
                prefix=prefix,
                payloads={_SUCCESS_FILENAME: marker},
                parent=current,
                message=success_message,
            )
        except HubConcurrentUpdateError:
            continue
        pinned_success = _pin_repository(client, revision=success_oid)
        success_files = _list_remote(client, pinned_success)
        verified_data_oid, _verified_marker = _verified_existing_success(
            client,
            revision=pinned_success,
            remote_files=success_files,
            prefix=prefix,
            final_id=final_id,
            data_payloads=data_payloads,
        )
        return _final_result(
            success_oid=pinned_success,
            data_oid=verified_data_oid,
            final_id=final_id,
            prefix=prefix,
            files=data_payloads,
        )
    raise HubUnavailableError("final success publication could not win the concurrent-update race")


def download_final_annotation_bundle(
    client: AnnotationHubClient,
    *,
    curation_id: str,
    snapshot_id: str,
    revision: str,
    destination: Path,
) -> dict[str, Any]:
    """Verify and atomically install a complete final bundle at an exact commit."""
    curation_id = _validate_identifier("curation_id", curation_id)
    final_id = _validate_identifier("snapshot_id", snapshot_id)
    if re.fullmatch(r"[0-9a-f]{64}", final_id) is None:
        raise ValueError("snapshot_id must be lowercase 64-hex")
    requested_destination, destination = _fresh_destination(Path(destination), label="download")
    pinned = _pin_repository(client, revision=revision)
    remote_files = _list_remote(client, pinned)
    prefix = _final_prefix(curation_id, final_id)
    names = _prefix_names(remote_files, prefix)
    expected_names = {
        *FINAL_IDENTITY_FILES,
        _PUBLICATION_MANIFEST_FILENAME,
        _SUMS_FILENAME,
        _SUCCESS_FILENAME,
    }
    if names != expected_names:
        raise HydrationError("final annotation bundle is incomplete or contains foreign files")
    downloaded = {
        name: _download_remote(client, f"{prefix}{name}", pinned)
        for name in sorted(names)
    }
    publication = _parse_json_object(
        _PUBLICATION_MANIFEST_FILENAME,
        downloaded[_PUBLICATION_MANIFEST_FILENAME],
    )
    semantic = {name: downloaded[name] for name in FINAL_IDENTITY_FILES}
    expected_curation, source_snapshot = _validate_final_semantics(semantic)
    if expected_curation != curation_id:
        raise HydrationError("final semantic bundle curation mismatch")
    if set(publication) != {
        "schema_version",
        "curation_id",
        "source_snapshot_id",
        "final_snapshot_id",
        "files",
    } or publication.get("schema_version") != _PUBLICATION_SCHEMA:
        raise HydrationError("final publication manifest field set or schema mismatch")
    if (
        publication.get("final_snapshot_id") != final_id
        or publication.get("curation_id") != curation_id
        or publication.get("source_snapshot_id") != source_snapshot
    ):
        raise HydrationError("final publication manifest identity mismatch")
    digests = {name: _digest(data) for name, data in semantic.items()}
    if publication.get("files") != digests:
        raise HydrationError("final publication manifest digest map mismatch")
    expected_publication = _canonical_json_bytes(publication)
    if downloaded[_PUBLICATION_MANIFEST_FILENAME] != expected_publication:
        raise HydrationError("final publication manifest is not canonically encoded")
    derived_id = _digest(_canonical_json_bytes(digests))
    if derived_id != final_id:
        raise HydrationError("final semantic bundle identity mismatch")
    expected_sums = "".join(
        f"{_digest(data)}  {name}\n"
        for name, data in sorted(
            {**semantic, _PUBLICATION_MANIFEST_FILENAME: downloaded[_PUBLICATION_MANIFEST_FILENAME]}.items()
        )
    ).encode("utf-8")
    if downloaded[_SUMS_FILENAME] != expected_sums:
        raise HydrationError("final SHA256SUMS mismatch")
    data_oid = _parse_success(downloaded[_SUCCESS_FILENAME], final_id=final_id)
    data_revision = _pin_repository(client, revision=data_oid)
    data_files = _list_remote(client, data_revision)
    _verify_prefix(
        client,
        revision=data_revision,
        remote_files=data_files,
        prefix=prefix,
        expected={
            **semantic,
            _PUBLICATION_MANIFEST_FILENAME: downloaded[_PUBLICATION_MANIFEST_FILENAME],
            _SUMS_FILENAME: expected_sums,
        },
    )

    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    stage_fd: int | None = None
    owned_identity: tuple[int, int] | None = None
    installed = False
    try:
        stage_fd = os.open(stage, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        stage_info = os.fstat(stage_fd)
        if not stat.S_ISDIR(stage_info.st_mode):
            raise HydrationError("installation staging descriptor is not a directory")
        owned_identity = (stage_info.st_dev, stage_info.st_ino)
        if _directory_identity(stage) != owned_identity:
            raise HydrationError("installation staging directory changed")
        for name, data in sorted(downloaded.items()):
            target = stage / name
            with target.open("wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        os.fsync(stage_fd)
        if _directory_identity(stage) != owned_identity:
            raise HydrationError("installation staging directory changed")
        os.replace(stage, destination)
        installed = True
        if _directory_identity(destination) != owned_identity:
            raise HydrationError("installation destination changed")
        parent_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except Exception as exc:
        if owned_identity is not None:
            _remove_owned_tree(destination if installed else stage, owned_identity)
        raise HydrationError(redact_text(f"final bundle installation failed: {exc}")) from None
    finally:
        if stage_fd is not None:
            os.close(stage_fd)
    return {
        "hub_commit_sha": pinned,
        "data_commit_sha": data_oid,
        "final_snapshot_id": final_id,
        "prefix": prefix,
        "files": sorted(downloaded),
        "destination": str(requested_destination),
    }


def resume_annotation_state(
    client: AnnotationHubClient,
    *,
    curation_id: str,
    destination: Path,
    expected_snapshot_id: str | None = None,
    revision: str | None = None,
) -> dict[str, Any]:
    """Restore one pinned checkpoint into a fresh destination atomically."""
    curation_id = _validate_identifier("curation_id", curation_id)
    _requested_destination, destination = _fresh_destination(Path(destination), label="resume")
    pinned = _pin_repository(client, revision=revision)
    remote_files = _list_remote(client, pinned)
    checkpoint = _parse_checkpoint(
        client,
        revision=pinned,
        curation_id=curation_id,
        remote_files=remote_files,
    )
    if checkpoint is None:
        raise HydrationError(f"no published checkpoint exists for curation {curation_id}")
    pointer, payloads, _raw = checkpoint
    if expected_snapshot_id is not None:
        expected_snapshot_id = _validate_identifier("expected_snapshot_id", expected_snapshot_id)
        if pointer["snapshot_id"] != expected_snapshot_id:
            raise HydrationError("published checkpoint snapshot does not match the expected snapshot")

    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    stage_fd: int | None = None
    owned_identity: tuple[int, int] | None = None
    installed = False
    try:
        stage_fd = os.open(stage, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        stage_info = os.fstat(stage_fd)
        if not stat.S_ISDIR(stage_info.st_mode):
            raise HydrationError("installation staging descriptor is not a directory")
        owned_identity = (stage_info.st_dev, stage_info.st_ino)
        if _directory_identity(stage) != owned_identity:
            raise HydrationError("installation staging directory changed")
        for name, data in sorted(payloads.items()):
            target = stage / name
            with target.open("wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        os.fsync(stage_fd)
        if _directory_identity(stage) != owned_identity:
            raise HydrationError("installation staging directory changed")
        os.replace(stage, destination)
        installed = True
        if _directory_identity(destination) != owned_identity:
            raise HydrationError("installation destination changed")
        parent_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except Exception as exc:
        if owned_identity is not None:
            _remove_owned_tree(destination if installed else stage, owned_identity)
        if isinstance(exc, (HydrationError, ValueError, PublicDestinationError)):
            raise
        raise HydrationError(redact_text(f"checkpoint installation failed: {exc}")) from None
    finally:
        if stage_fd is not None:
            os.close(stage_fd)
    return {
        "curation_id": curation_id,
        "snapshot_id": pointer["snapshot_id"],
        "batch_id": pointer["batch_id"],
        "batch_prefix": pointer["batch_prefix"],
        "checkpoint_revision": pinned,
        "observation_count": pointer["observation_count"],
        "restored": sorted(payloads),
    }
