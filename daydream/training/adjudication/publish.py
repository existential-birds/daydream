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
from collections.abc import Callable
from pathlib import Path
from typing import Any, Mapping

from daydream.archive.hydrate import (
    HubClient,
    HubUnavailableError,
    HydrationError,
    PublicDestinationError,
    _retry_upload,
    resolve_source_revision,
)
from daydream.trajectory import redact_text

__all__ = [
    "annotation_prefix",
    "publish_annotation_state",
    "publish_final_annotation_bundle",
    "resume_annotation_state",
]

# State files published under the snapshot prefix (stable order for digests).
_STATE_FILES = ("queue.json", "observations.jsonl", "preview-ledger.json")
_MANIFEST_FILENAME = "preview-manifest.json"
_CHECKPOINT_RELPATH = "checkpoints/batch-latest.json"
_FINAL_SEGMENT = "final"
_SUCCESS_FILENAME = "_SUCCESS"
_SUMS_FILENAME = "SHA256SUMS"

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


def _bundle_payloads(bundle_dir: Path) -> dict[str, bytes]:
    """Read every file in ``bundle_dir`` (sorted, deterministic order).

    ``_SUCCESS`` and ``SHA256SUMS`` are generated fresh below and must never
    leak in from a prior publication: a stale local marker in the payload set
    would skew the pinned checksums and break clean-download verification
    (the marker actually uploaded is the freshly staged one).
    """
    payloads = {
        p.name: p.read_bytes()
        for p in sorted(bundle_dir.iterdir())
        if p.is_file() and p.name not in {_SUCCESS_FILENAME, _SUMS_FILENAME}
    }
    if not payloads:
        raise ValueError(f"publish final bundle: {bundle_dir} contains no files")
    return payloads


def _enforce_admission_gate(bundle_dir: Path) -> None:
    """Refuse to publish a bundle whose own coverage report fails the 80%
    human-adjudication admission gate (issue #336 finding 2).

    The gate is read from the bundle's ``coverage-report.json`` (written by
    ``final_bundle.build_final_bundle`` over the same enrichment as
    ``corpus adjudicate report``): fails closed with
    :class:`PublicDestinationError` before any byte is uploaded when the
    report is missing, unreadable, or says ``passes_80pct`` is not true — so
    publish-final can never upload identically to a fully adjudicated run.
    """
    report_path = bundle_dir / "coverage-report.json"
    if not report_path.is_file():
        raise PublicDestinationError(
            f"refusing to publish: final bundle has no coverage report at {report_path} "
            "(run `daydream corpus adjudicate publish-final --dry-run` to build and "
            "inspect the bundle first)"
        )
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PublicDestinationError(
            f"refusing to publish: unreadable coverage report at {report_path}: {exc}"
        ) from exc
    if not isinstance(report, dict):
        raise PublicDestinationError(
            f"refusing to publish: coverage report at {report_path} is not a JSON object"
        )
    gate = report.get("admission_gate")
    passes_80pct = gate.get("passes_80pct") if isinstance(gate, dict) else None
    if passes_80pct is not True:
        raise PublicDestinationError(
            "refusing to publish: final bundle fails the 80% human-adjudication "
            f"admission gate (coverage report at {report_path}); complete the "
            "outcome-bearing adjudication backlog or re-materialize before publishing"
        )


def _resolve_hub_commit(client: HubClient, manifest_data: Mapping[str, Any], sums_bytes: bytes) -> str:
    """Resolve the Hub commit SHA under the pinned-revision policy (M6).

    The manifest's ``index_revision`` is used when present; otherwise the
    publication revision is derived from the published ``SHA256SUMS`` content
    (stable across idempotent re-publication of identical bytes, C2).
    ``resolve_source_revision(..., exploratory=False)`` fail-closes on symbolic
    refs: a run without a verified Hub commit is a failure, never a best effort.
    """
    revision = manifest_data.get("index_revision")
    if not isinstance(revision, str) or not revision:
        revision = _digest(sums_bytes)[:40]
    return resolve_source_revision(client, revision, exploratory=False)


def publish_final_annotation_bundle(
    client: HubClient,
    bundle_dir: Path,
    *,
    manifest: Path | Mapping[str, Any],
    verify_download: bool = True,
    _download_verifier: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    """Publish the final immutable annotation bundle under ``<prefix>final/`` (M6/C3).

    Strict order, each step fail-closed before anything is uploaded:

    1. :class:`PublicDestinationError` when the Hub repo is not private.
    2. 80% human-adjudication admission gate read from the bundle's
       ``coverage-report.json`` (:func:`_enforce_admission_gate`) — a missing,
       unreadable, or failing report refuses publication (issue #336 finding 2),
       before any byte is uploaded.
    3. S1 secret scan over every bundle file (a hit raises naming the file).
    4. ``SHA256SUMS`` over the file set, self-excluded (mirror
       ``hydrate.publish_batches``).
    5. Additive upload of the bundle + manifest + sums.
    6. Clean-download verification: every uploaded file is read back, re-hashed
       and compared (``_download_verifier`` replaces this step when injected;
       returning ``False`` raises ``ValueError``).
    7. Hub commit SHA resolved via :func:`hydrate.resolve_source_revision`
       (pinned-revision policy) — a resolution failure propagates and
       ``_SUCCESS`` is never uploaded.
    8. ``_SUCCESS`` uploaded **last** — its presence is the commitment that the
       bundle is complete and verified.

    Immutability (C2): publication is additive; re-publishing identical bytes
    re-uploads the same content and the module never deletes or rewrites prior
    snapshot prefixes.
    """
    if not client.repo_private:
        raise PublicDestinationError(
            "refusing to publish: the Hub repo is not private; annotation bundles "
            "publish only to private repos (M17)"
        )
    prefix = f"{annotation_prefix(manifest)}{_FINAL_SEGMENT}/"
    bundle_dir = Path(bundle_dir)
    manifest_data = _read_manifest_data(manifest)

    # 80% admission gate (issue #336 finding 2): refuse before any byte is
    # uploaded — and before any Hub revision lookup — when the bundle's own
    # coverage report says the gate fails or is missing, so a half-adjudicated
    # snapshot can never be committed under the final/ prefix.
    _enforce_admission_gate(bundle_dir)

    payloads = _bundle_payloads(bundle_dir)
    for name, data in payloads.items():
        _scan_for_secrets(name, data)
    manifest_bytes = (
        Path(manifest).read_bytes()
        if isinstance(manifest, (str, Path))
        else json.dumps(manifest_data, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    _scan_for_secrets(_MANIFEST_FILENAME, manifest_bytes)
    payloads[_MANIFEST_FILENAME] = manifest_bytes

    # SHA256SUMS covers every published file except itself (self-inclusion would
    # make the checksum file unstable across idempotent re-publishes).
    sums_bytes = "".join(
        f"{_digest(data)}  {name}\n" for name, data in sorted(payloads.items())
    ).encode("utf-8")

    stage_dir = bundle_dir / ".publish-stage"
    stage_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[str | Path, Path] = {}
    for name, data in payloads.items():
        staged = stage_dir / name
        staged.write_bytes(data)
        mapping[f"{prefix}{name}"] = staged
    staged_sums = stage_dir / _SUMS_FILENAME
    staged_sums.write_bytes(sums_bytes)
    mapping[f"{prefix}{_SUMS_FILENAME}"] = staged_sums
    _upload(client, mapping, f"daydream adjudication: final annotation bundle {prefix}")

    if verify_download:
        expected: dict[str, bytes] = {**payloads, _SUMS_FILENAME: sums_bytes}
        if _download_verifier is not None:
            if not _download_verifier(prefix):
                raise ValueError(
                    f"final bundle download verification failed for {prefix}: "
                    "the verifier refused the freshly uploaded bundle"
                )
        else:
            for name, data in sorted(expected.items()):
                remote = client.download_file(f"{prefix}{name}")
                if _digest(remote) != _digest(data):
                    raise ValueError(
                        f"final bundle download verification failed for {prefix}: "
                        f"re-downloaded {name!r} does not match the uploaded bytes"
                    )

    hub_commit_sha = _resolve_hub_commit(client, manifest_data, sums_bytes)

    staged_success = stage_dir / _SUCCESS_FILENAME
    staged_success.write_bytes(b"")
    _upload(
        client,
        {f"{prefix}{_SUCCESS_FILENAME}": staged_success},
        f"daydream adjudication: final annotation bundle committed {prefix}",
    )
    return {
        "hub_commit_sha": hub_commit_sha,
        "prefix": prefix,
        "files": sorted([*payloads, _SUMS_FILENAME, _SUCCESS_FILENAME]),
    }


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
