"""Legacy bronze bundle sanitizer (issue #981 M14/M15/M19).

Copies legacy bronze run bundles into content-addressed, credential-free
derivatives under ``<archive_dir>/sanitized/<session_id>/``. Sources are never
modified (M14): the bundle is copied first, then the copy is transformed.

Transformation pipeline per file:

* ``manifest.json`` (and any JSON file): credential-bearing URL string leaves
  are rewritten through :func:`daydream.archive.git_safe.normalize_remote_url`
  (the sole URL authority); the whole document is then passed through
  :func:`daydream.trajectory.redact_value`.
* Text files: :func:`daydream.trajectory.redact_text` plus the scanner's
  extended userinfo/query-param substitutions.

Release gate: every derivative is re-scanned with
:func:`daydream.archive.scan.scan_run_dir`; a derivative that does not come
back clean is quarantined (removed from ``sanitized/``, recorded with
``status="quarantined"``) — never released (fail-closed).

``derivative_digest`` is a SHA-256 over a canonical manifest of
``(relative path, per-file SHA-256)`` pairs, stable across runs on identical
input (M15). Bulk passes via :func:`sanitize_archive` are resumable: a
``progress.jsonl`` marker records completed session ids + digests (mirroring
the ``BackfillCache`` resume-marker pattern), and a session is only skipped
when its derivative still hashes to the recorded digest (M19).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from daydream.archive.git_safe import classify_remote_url, normalize_remote_url
from daydream.archive.scan import scan_run_dir
from daydream.trajectory import redact_text, redact_value

__all__ = ["SanitizeResult", "sanitize_archive", "sanitize_bundle"]

_PROGRESS_FILENAME = "progress.jsonl"
_AUDIT_FILENAME = "audit.jsonl"


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class SanitizeResult:
    """Outcome of sanitizing one bundle."""

    session_id: str
    source: Path
    derivative_digest: str
    status: str  # "sanitized" | "quarantined"


def _derivative_digest(derivative_dir: Path) -> str:
    """SHA-256 over a canonical (relpath, file digest) manifest — M15 stable."""
    entries: list[tuple[str, str]] = []
    for file_path in sorted(derivative_dir.rglob("*")):
        if not file_path.is_file():
            continue
        rel = file_path.relative_to(derivative_dir).as_posix()
        digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
        entries.append((rel, digest))
    canonical = "".join(f"{rel}\t{digest}\n" for rel, digest in entries)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sanitize_url_string(value: str) -> str:
    """Rewrite a credential-bearing URL string to its canonical form."""
    if classify_remote_url(value):
        _identity, canonical = normalize_remote_url(value)
        if canonical is not None:
            return canonical
    return value


def _sanitize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize_json_value(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_sanitize_json_value(child) for child in value]
    if isinstance(value, str):
        return _sanitize_url_string(value)
    return value


def _sanitize_text(text: str) -> str:
    """Redact one text file body, then re-check with the scanner's extra rules."""
    return redact_text(text)


def _sanitize_derivative(derivative_dir: Path) -> None:
    """Transform every file of a copied bundle in place."""
    for file_path in sorted(derivative_dir.rglob("*")):
        if not file_path.is_file():
            continue
        text = file_path.read_text(encoding="utf-8")
        if file_path.suffix == ".json":
            try:
                doc = json.loads(text)
            except ValueError:
                doc = None
            if isinstance(doc, (dict, list)):
                sanitized = redact_value(_sanitize_json_value(doc))
                file_path.write_text(
                    json.dumps(sanitized, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8",
                )
                continue
        file_path.write_text(_sanitize_text(text), encoding="utf-8")


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _read_progress(sanitized_dir: Path) -> dict[str, str]:
    """Return {session_id: derivative_digest} from the resume marker file."""
    progress_path = sanitized_dir / _PROGRESS_FILENAME
    if not progress_path.exists():
        return {}
    completed: dict[str, str] = {}
    for line in progress_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue  # append-only log; a partial last line is the only failure mode
        session_id = record.get("session_id")
        digest = record.get("derivative_digest")
        if isinstance(session_id, str) and isinstance(digest, str):
            completed[session_id] = digest
    return completed


def _mark_done(sanitized_dir: Path, session_id: str, derivative_digest: str) -> None:
    _append_jsonl(
        sanitized_dir / _PROGRESS_FILENAME,
        {
            "session_id": session_id,
            "derivative_digest": derivative_digest,
            "completed_at": _now_iso_utc(),
        },
    )


def sanitize_bundle(run_dir: Path, archive_dir: Path) -> SanitizeResult:
    """Sanitize one legacy bundle into ``archive_dir/sanitized/<session_id>/``.

    The source bundle is never modified. On any failure the derivative is
    removed, a ``status="quarantined"`` audit record is appended, and the
    exception is re-raised so a bulk caller can continue with the next bundle.
    """
    sanitized_dir = archive_dir / "sanitized"
    manifest: dict[str, Any] = {}
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                manifest = loaded
        except ValueError:
            manifest = {}
    session_id = str(manifest.get("session_id") or run_dir.name)
    derivative_dir = sanitized_dir / session_id

    try:
        if derivative_dir.exists():
            shutil.rmtree(derivative_dir)
        derivative_dir.mkdir(parents=True)
        for item in sorted(run_dir.iterdir()):
            target = derivative_dir / item.name
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)
        _sanitize_derivative(derivative_dir)

        # Fail-closed release gate: only a clean scan releases the derivative.
        scan_result = scan_run_dir(derivative_dir)
        if not scan_result.clean:
            raise ValueError(f"derivative scan found {scan_result.summary()}")

        digest = _derivative_digest(derivative_dir)
    except Exception:
        if derivative_dir.exists():
            shutil.rmtree(derivative_dir, ignore_errors=True)
        _append_jsonl(
            sanitized_dir / _AUDIT_FILENAME,
            {
                "source": str(run_dir),
                "session_id": session_id,
                "derivative_digest": "",
                "status": "quarantined",
                "completed_at": _now_iso_utc(),
            },
        )
        raise

    _append_jsonl(
        sanitized_dir / _AUDIT_FILENAME,
        {
            "source": str(run_dir),
            "session_id": session_id,
            "derivative_digest": digest,
            "status": "sanitized",
            "completed_at": _now_iso_utc(),
        },
    )
    return SanitizeResult(
        session_id=session_id,
        source=run_dir,
        derivative_digest=digest,
        status="sanitized",
    )


def sanitize_archive(archive_dir: Path) -> list[SanitizeResult]:
    """Sanitize every bundle under ``archive_dir/runs/*``; resumable (M19).

    A session recorded in ``progress.jsonl`` is skipped only when its
    derivative still exists and hashes to the recorded digest — partial or
    corrupted derivatives are re-processed. One bad bundle never stops the
    pass: its failure is quarantined and the loop continues.
    """
    sanitized_dir = archive_dir / "sanitized"
    sanitized_dir.mkdir(parents=True, exist_ok=True)
    completed = _read_progress(sanitized_dir)
    results: list[SanitizeResult] = []
    runs_dir = archive_dir / "runs"
    if not runs_dir.is_dir():
        return results
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        session_id = run_dir.name
        recorded_digest = completed.get(session_id)
        if recorded_digest is not None and (sanitized_dir / session_id).is_dir():
            try:
                current_digest = _derivative_digest(sanitized_dir / session_id)
            except OSError:
                current_digest = None
            if current_digest == recorded_digest:
                continue  # M19: completed items are not re-processed
        result = sanitize_bundle(run_dir, archive_dir)
        _mark_done(sanitized_dir, result.session_id, result.derivative_digest)
        results.append(result)
    return results
