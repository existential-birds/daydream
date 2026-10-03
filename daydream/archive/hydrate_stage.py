"""Download and admit sanitized session bundles into the local staging tree."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from daydream.archive import sanitize
from daydream.archive.git_safe import normalize_remote_url
from daydream.archive.hydrate_discovery import (
    _discover_snapshot,
    _is_bare_segment,
    _validate_relpath,
)
from daydream.archive.hydrate_rules import (
    REASON_CODE_BUNDLE_UNREADABLE,
    REASON_CODE_PATH_TRAVERSAL,
    REASON_CODE_SANITIZE_FAILED,
    REASON_CODE_SECRETS_SCAN_DIRTY,
    REASON_CODE_UNTRUSTED_REMOTE_HOST,
)
from daydream.archive.hydrate_types import (
    DownloadResult,
    HubClient,
    HydrationError,
    IngestResult,
    NoSessionCandidatesError,
    StageError,
)
from daydream.json_utils import atomic_write_json
from daydream.redaction import redact_text
from daydream.timeutil import now_iso_utc
from daydream.trajectory import RUNS_DIRNAME


def download_snapshot(
    client: HubClient,
    *,
    revision: str,
    stage_dir: Path,
    expect: dict[str, str] | None = None,
) -> DownloadResult:
    """Download a pinned snapshot into ``stage_dir/<revision>/bundles/<session-id>``."""
    revision = str(revision)
    root = stage_dir / revision
    manifest_path = root / "_download_manifest.json"
    records: dict[str, dict[str, Any]] = {}
    if manifest_path.exists():
        try:
            loaded = json.loads(manifest_path.read_text())
            records = {a["relpath"]: a for a in loaded.get("artifacts", [])}
        except (OSError, ValueError, KeyError, TypeError):
            records = {}  # corrupt ledger: rebuild from disk state

    relpaths = list(client.list_repo_files(revision=revision))
    for relpath in relpaths:
        # Validate every Hub path, including ignored metadata and derived
        # output, before applying discovery filters. A malicious ignored path
        # must not become an escape hatch around the staging trust boundary.
        _validate_relpath(relpath, root)
    discovery = _discover_snapshot(relpaths)
    if not discovery.sessions:
        details = "; ".join(discovery.incomplete_manifests) or "required artifacts were not found"
        raise NoSessionCandidatesError(
            redact_text(
                f"source revision {revision!r} contains {discovery.run_shaped_manifests} "
                "run-shaped manifest(s) but discovery produced zero candidates "
                f"(canonical={discovery.canonical_manifests}, legacy={discovery.legacy_manifests}); "
                "expected <session-id>/manifest.json plus trajectory.json or "
                f"bundles/<session-id>/...; {details}"
            )
        )
    downloaded = 0
    skipped = 0
    digests: dict[str, str] = {}
    artifacts: list[dict[str, Any]] = []

    for normalized, source_relpath in discovery.normalized_paths:
        target = _validate_relpath(normalized, root)
        expected_sha = (expect or {}).get(source_relpath) or (expect or {}).get(normalized)
        try:
            if target.exists():
                existing = hashlib.sha256(target.read_bytes()).hexdigest()
                record_sha = records.get(normalized, {}).get("sha256")
                if expected_sha in (None, existing) and record_sha in (None, existing):
                    digests[normalized] = existing
                    skipped += 1
                    artifacts.append(
                        {
                            **records.get(normalized, {}),
                            "relpath": normalized,
                            "source_relpath": source_relpath,
                            "sha256": existing,
                        }
                    )
                    continue
            data = client.download_file(source_relpath, revision=revision)
        except HydrationError as exc:
            raise StageError(redact_text(str(exc))) from exc
        sha = hashlib.sha256(data).hexdigest()
        if expected_sha is not None and sha != expected_sha:
            raise StageError(
                redact_text(f"digest mismatch for {source_relpath!r}: expected {expected_sha}, got {sha}")
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(target.name + ".partial")
        try:
            tmp.write_bytes(data)
            tmp.replace(target)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            raise StageError(redact_text(f"write failed for {normalized!r}: {exc}")) from exc
        downloaded += 1
        digests[normalized] = sha
        artifacts.append(
            {
                "relpath": normalized,
                "source_relpath": source_relpath,
                "sha256": sha,
                "size": len(data),
                "fetched_at": now_iso_utc(),
            }
        )

    root.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "revision": revision,
                "artifacts": artifacts,
                "candidate_sessions": [session.session_id for session in discovery.sessions],
                "discovery": {
                    "run_shaped_manifests": discovery.run_shaped_manifests,
                    "canonical_manifests": discovery.canonical_manifests,
                    "legacy_manifests": discovery.legacy_manifests,
                    "incomplete_manifests": list(discovery.incomplete_manifests),
                },
            },
            indent=2,
        )
    )
    return DownloadResult(
        downloaded=downloaded,
        skipped=skipped,
        digests=digests,
        discovered=len(discovery.sessions),
        run_shaped_manifests=discovery.run_shaped_manifests,
        incomplete_manifests=discovery.incomplete_manifests,
    )


def _read_manifest_dict(bundle_dir: Path) -> dict[str, Any] | None:
    """Read ``manifest.json`` from ``bundle_dir``; ``None`` when absent/unparseable."""
    path = bundle_dir / "manifest.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _require_manifest_dict(bundle_dir: Path, *, label: str) -> dict[str, Any]:
    """Read a derivative manifest fail-closed; ``label`` names it in the error."""
    data = _read_manifest_dict(bundle_dir)
    if data is None:
        raise HydrationError(redact_text(f"{label} has an unreadable manifest"))
    return data


def _bundle_dirs(root: Path) -> list[Path]:
    """Snapshot the bundle directories in deterministic order before callers move them."""
    return sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []


def _derivative_manifests(stage: Path, root: str = RUNS_DIRNAME) -> Iterator[tuple[Path, dict[str, Any]]]:
    label = "admitted" if root == RUNS_DIRNAME else root
    for derivative in _bundle_dirs(stage / root):
        yield derivative, _require_manifest_dict(derivative, label=f"{label} derivative {derivative.name}")


def _admitted_session_id(derivative: Path, data: dict[str, Any]) -> str:
    sid = str(data.get("session_id") or derivative.name)
    if not _is_bare_segment(sid):
        raise HydrationError(
            redact_text(f"admitted derivative {derivative.name} has an unsafe session id {sid!r}")
        )
    return sid


def _read_manifest_field(data: dict[str, Any], key: str) -> Any:
    """Read canonical ``git`` provenance first, then the legacy flat field."""
    git = data.get("git")
    if isinstance(git, dict) and key in git:
        return git[key]
    return data.get(key)


def _manifest_remote_fields(data: dict[str, Any]) -> tuple[bool, str | None, str | None]:
    """Normalize ``remote_url`` to ``(has_url, slug, canonical_url)``; blank is all-None."""
    raw_url = _read_manifest_field(data, "remote_url")
    if isinstance(raw_url, str) and raw_url.strip():
        return True, *normalize_remote_url(raw_url)
    return False, None, None


def _read_download_manifest(stage: Path, revision: str) -> dict[str, Any] | None:
    """Read and parse the download manifest, or ``None`` when it is absent."""
    path = stage / "downloads" / str(revision) / "_download_manifest.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HydrationError(redact_text(f"invalid download discovery ledger {path}: {exc}")) from exc
    if not isinstance(payload, dict):
        raise HydrationError(redact_text(f"invalid download discovery ledger {path}"))
    return payload


def _download_discovery_block(stage: Path, revision: str) -> dict[str, Any]:
    """Read discovery diagnostics; legacy manifests return empty, while invalid JSON fails closed."""
    payload = _read_download_manifest(stage, revision)
    if payload is None:
        return {}
    discovery = payload.get("discovery", {})
    if not isinstance(discovery, dict):
        path = stage / "downloads" / str(revision) / "_download_manifest.json"
        raise HydrationError(redact_text(f"invalid download discovery ledger {path}"))
    return discovery


def _discovered_session_ids(stage: Path, revision: str) -> list[str] | None:
    """Read the authoritative candidate list, rejecting unsafe or duplicate ids."""
    payload = _read_download_manifest(stage, revision)
    if payload is None:
        return None
    if "candidate_sessions" not in payload:
        return None
    path = stage / "downloads" / str(revision) / "_download_manifest.json"
    candidates = payload["candidate_sessions"]
    if not isinstance(candidates, list) or not all(
        isinstance(session_id, str) and _is_bare_segment(session_id) for session_id in candidates
    ):
        raise HydrationError(redact_text(f"invalid candidate session ids in {path}"))
    if len(set(candidates)) != len(candidates):
        raise HydrationError(redact_text(f"duplicate candidate session ids in {path}"))
    return list(candidates)


def ingest_bundles(stage: Path, *, revision: str) -> list[IngestResult]:
    """Admit discovered bundles through the shared import and sanitization gates."""
    bundles_root = stage / "downloads" / str(revision) / "bundles"
    discovered_ids = _discovered_session_ids(stage, revision)
    bundle_items = (
        [(bundle_dir.name, bundle_dir) for bundle_dir in _bundle_dirs(bundles_root)]
        if discovered_ids is None
        else [(session_id, bundles_root / session_id) for session_id in discovered_ids]
    )
    results: list[IngestResult] = []
    for name, bundle_dir in bundle_items:
        data = _read_manifest_dict(bundle_dir)
        if data is None:
            results.append(IngestResult(name, "quarantined", REASON_CODE_BUNDLE_UNREADABLE))
            continue
        session_id = str(data.get("session_id") or name)
        if not _is_bare_segment(session_id):
            # M4: the manifest's session id is Hub-provided path data; reject
            # traversal/absolute ids before any write — the sanitize gate
            # (which mkdirs from the session id) never sees them.
            results.append(IngestResult(session_id, "quarantined", REASON_CODE_PATH_TRAVERSAL))
            continue
        has_url, identity, _canonical = _manifest_remote_fields(data)
        if has_url and identity is None:
            # Non-allowlisted host: rejected as admission data before any
            # gate work; the raw copy stays in downloads, never indexed.
            results.append(
                IngestResult(session_id, "quarantined", REASON_CODE_UNTRUSTED_REMOTE_HOST)
            )
            continue
        gate = sanitize.import_bundle(bundle_dir, stage)
        if gate.quarantined or not gate.imported:
            results.append(IngestResult(session_id, "quarantined", REASON_CODE_SECRETS_SCAN_DIRTY))
            continue
        # Task 0B constraint: hydrated staging bundles must exclude .git (the
        # raw download copy is daydream-staged data, safe to prune locally).
        for git_dir in list(bundle_dir.rglob(".git")):
            if git_dir.is_dir():
                shutil.rmtree(git_dir)
        try:
            sanitized = sanitize.sanitize_bundle(bundle_dir, stage)
        except Exception:
            results.append(IngestResult(session_id, "quarantined", REASON_CODE_SANITIZE_FAILED))
            continue
        if not sanitized.released:
            results.append(IngestResult(session_id, "quarantined", REASON_CODE_SECRETS_SCAN_DIRTY))
            continue
        derivative = stage / "sanitized" / session_id
        target = stage / RUNS_DIRNAME / session_id
        _move_dir(derivative, target)  # staging layout only, not the gate
        results.append(IngestResult(session_id, "admitted"))
    atomic_write_json(
        bundles_root.parent / "_ingest_results.json",
        {
            "revision": str(revision),
            "results": [
                {"session_id": r.session_id, "status": r.status, "reason_code": r.reason_code}
                for r in results
            ],
        },
    )
    return results


def _move_dir(source: Path, target: Path) -> None:
    """Move a directory, replacing any prior occupant of ``target``."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(source), str(target))
