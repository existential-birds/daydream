"""Durable adjudication-state publication + fresh-VM resume (issue #1055, Task 3).

Mirrors ``archive/hydrate.py``'s publication machinery (``HubClient``,
additive content-addressed upload, ``PublicDestinationError``,
``_retry_upload`` retry seam) under a new
``annotations/<curation-id>/<snapshot-id>/`` prefix. The Hub checkpoint under
``checkpoints/batch-latest.json`` is the canonical resume state — the VM-local
``--state-dir`` is scratch only.

Credentials (C1): this module never reads or embeds tokens; ``HF_TOKEN`` is
consumed only by the real ``HfHubClient`` wiring.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

from daydream.archive.hydrate import (
    HubClient,
    HubUnavailableError,
    HydrationError,
    PublicDestinationError,
    _retry_upload,
)
from daydream.trajectory import redact_text

__all__ = [
    "annotation_prefix",
    "publish_annotation_state",
    "resume_annotation_state",
]

# State files published under the snapshot prefix (stable order for digests).
_STATE_FILES = ("queue.json", "observations.jsonl", "preview-ledger.json")
_MANIFEST_FILENAME = "preview-manifest.json"
_CHECKPOINT_RELPATH = "checkpoints/batch-latest.json"

# HF_TOKEN-shaped values (fail-closed secret scan, S1). A hit means a live
# credential leaked into a payload; publication is refused, never scrubbed.
_SECRET_SHAPES = re.compile(r"(?:hf_[0-9A-Za-z]{20,}|github_pat_[0-9A-Za-z_]{20,}|ghp_[0-9A-Za-z]{20,})")


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


def _scan_for_secrets(name: str, data: bytes) -> None:
    """Fail-closed secret scan (S1): refuse to upload credential-shaped payloads."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PublicDestinationError(f"{name}: payload is not valid UTF-8: {exc}") from exc
    hit = _SECRET_SHAPES.search(text)
    if hit is not None:
        raise PublicDestinationError(f"{name}: refusing to publish: credential-shaped value detected in payload")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _observation_key(line: str, record: dict[str, Any]) -> str:
    observed_at = record.get("observed_at")
    record_id = record.get("record_id")
    if isinstance(record_id, str) and record_id and isinstance(observed_at, str) and observed_at:
        return f"{record_id}\x00{observed_at}"
    return hashlib.sha256(line.encode("utf-8")).hexdigest()


def _merge_observations(remote: bytes | None, local: bytes) -> bytes:
    """Append-only merge (M4): remote lines first, then new local lines.

    Dedup key is ``record_id + observed_at`` (falling back to line digest when
    either is absent); local lines already present remotely are dropped so a
    re-publish never duplicates, and remote history is never truncated.
    """
    if not local.strip():
        return remote or b""
    local_lines = [ln for ln in local.decode("utf-8").splitlines() if ln.strip()]
    remote_lines: list[str] = []
    seen: set[str] = set()
    if remote is not None:
        for line in remote.decode("utf-8").splitlines():
            if not line.strip():
                continue
            remote_lines.append(line)
            try:
                seen.add(_observation_key(line, json.loads(line)))
            except ValueError:
                seen.add(_observation_key(line, {}))
    out_lines = list(remote_lines)
    for line in local_lines:
        try:
            key = _observation_key(line, json.loads(line))
        except ValueError:
            key = _observation_key(line, {})
        if key in seen:
            continue
        seen.add(key)
        out_lines.append(line)
    return ("".join(ln + "\n" for ln in out_lines)).encode("utf-8")


def _read_state_file(state_dir: Path, name: str) -> bytes:
    return (state_dir / name).read_bytes()


def _upload(client: HubClient, mapping: dict[str | Path, Path], commit_message: str) -> None:
    """Upload through hydrate's retry seam; failures become redacted HubUnavailableError."""
    try:
        _retry_upload(client, mapping, commit_message)
    except HydrationError as exc:
        raise HubUnavailableError(redact_text(str(exc))) from exc


def publish_annotation_state(
    client: HubClient,
    state_dir: Path,
    *,
    manifest: Path | Mapping[str, Any],
    batch_complete: bool = False,
) -> dict[str, Any]:
    """Publish adjudication state additively under the snapshot prefix (M3/M4).

    Mirrors ``hydrate.publish_batches``: hard-fails with
    :class:`PublicDestinationError` when the repo is not private **before any
    byte is written**, runs the S1 secret scan over every payload, then uploads
    ``queue.json``, ``observations.jsonl``, ``preview-ledger.json`` and the
    preview manifest additively (re-upload idempotent; observations merged
    append-only). When ``batch_complete`` a remote checkpoint
    ``checkpoints/batch-latest.json`` is written with the observation count,
    latest ``observed_at`` and sha256 digests of every published file — the
    Hub-side canonical state for fresh-VM resume.
    """
    if not client.repo_private:
        raise PublicDestinationError(
            "refusing to publish: the Hub repo is not private; adjudication state "
            "publishes annotation content only to private repos (M17)"
        )
    prefix = annotation_prefix(manifest)
    state_dir = Path(state_dir)

    manifest_bytes = (
        Path(manifest).read_bytes()
        if isinstance(manifest, (str, Path))
        else json.dumps(dict(manifest), sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    payloads: dict[str, bytes] = {
        "queue.json": _read_state_file(state_dir, "queue.json"),
        "observations.jsonl": _read_state_file(state_dir, "observations.jsonl"),
        "preview-ledger.json": _read_state_file(state_dir, "preview-ledger.json"),
        _MANIFEST_FILENAME: manifest_bytes,
    }
    for name, data in payloads.items():
        _scan_for_secrets(name, data)

    # Append-only observations merge (M4): fetch the remote file and union it
    # with the local one before upload — never truncate remote history.
    try:
        remote_observations = client.download_file(f"{prefix}observations.jsonl")
    except (HydrationError, FileNotFoundError, OSError):
        remote_observations = None
    merged_observations = _merge_observations(remote_observations, payloads["observations.jsonl"])

    # Stage payloads to disk so the upload mapping carries real file paths.
    stage_dir = state_dir / ".publish-stage"
    stage_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str | Path, Path] = {}
    staged_bytes: dict[str, bytes] = {}
    for name, data in payloads.items():
        if name == "observations.jsonl":
            data = merged_observations
        staged_bytes[name] = data
        staged = stage_dir / name
        staged.write_bytes(data)
        mapping[f"{prefix}{name}"] = staged
    _scan_for_secrets("observations.jsonl (merged)", staged_bytes["observations.jsonl"])

    checkpoint: dict[str, Any] | None = None
    if batch_complete:
        observation_count = sum(
            1 for line in staged_bytes["observations.jsonl"].decode("utf-8").splitlines() if line.strip()
        )
        latest_observed_at: str | None = None
        for line in staged_bytes["observations.jsonl"].decode("utf-8").splitlines():
            if not line.strip():
                continue
            try:
                observed_at = json.loads(line).get("observed_at")
            except ValueError:
                observed_at = None
            if isinstance(observed_at, str) and (latest_observed_at is None or observed_at > latest_observed_at):
                latest_observed_at = observed_at
        checkpoint = {
            "curation_id": prefix.split("/")[1],
            "snapshot_id": prefix.split("/")[2],
            "observation_count": observation_count,
            "latest_observed_at": latest_observed_at,
            "digests": {name: _digest(data) for name, data in staged_bytes.items()},
        }
        data = json.dumps(checkpoint, sort_keys=True, separators=(",", ":")).encode("utf-8")
        staged = stage_dir / "batch-latest.json"
        staged.write_bytes(data)
        mapping[f"{prefix}{_CHECKPOINT_RELPATH}"] = staged

    _upload(client, mapping, f"daydream adjudication: annotation state publication {prefix}")

    summary: dict[str, Any] = {"prefix": prefix, "uploaded": sorted(mapping)}
    if checkpoint is not None:
        summary["observation_count"] = checkpoint["observation_count"]
    return summary


def resume_annotation_state(
    client: HubClient,
    *,
    manifest: Path | Mapping[str, Any],
    stage_dir: Path,
) -> dict[str, Any]:
    """Restore adjudication state onto a fresh VM from the Hub checkpoint (M3).

    Mirrors ``hydrate.resume_state``: the remote checkpoint is the canonical
    state. Every published file is downloaded and verified against the sha256
    digest recorded in the checkpoint (the manifest additionally against its
    recorded digest); any mismatch raises ``ValueError`` naming the file —
    fail closed, never a silent partial restore. A missing checkpoint yields
    an empty-state summary (nothing published yet).
    """
    prefix = annotation_prefix(manifest)
    stage_dir = Path(stage_dir)
    try:
        checkpoint_raw = client.download_file(f"{prefix}{_CHECKPOINT_RELPATH}")
    except (HydrationError, FileNotFoundError, OSError):
        return {"observation_count": 0, "restored": []}
    try:
        checkpoint = json.loads(checkpoint_raw)
    except ValueError as exc:
        raise ValueError(f"{_CHECKPOINT_RELPATH}: corrupt checkpoint JSON: {exc}") from exc
    digests = checkpoint.get("digests")
    if not isinstance(digests, dict):
        raise ValueError(f"{_CHECKPOINT_RELPATH}: checkpoint is missing 'digests'")

    stage_dir.mkdir(parents=True, exist_ok=True)
    restored: list[str] = []
    for name in [*_STATE_FILES, _MANIFEST_FILENAME]:
        expected = digests.get(name)
        if not isinstance(expected, str):
            raise ValueError(f"{_CHECKPOINT_RELPATH}: checkpoint has no digest for {name!r}")
        data = client.download_file(f"{prefix}{name}")
        actual = _digest(data)
        if actual != expected:
            raise ValueError(f"digest mismatch for {name!r}: checkpoint recorded {expected}, remote is {actual}")
        target = stage_dir / name
        tmp = target.with_name(target.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, target)
        restored.append(name)
    return {
        "observation_count": int(checkpoint.get("observation_count", 0)),
        "restored": restored,
    }
