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
back clean is moved to ``<archive_dir>/quarantine/<session_id>/`` and recorded
with ``status="quarantined"`` — never released (fail-closed).

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

from daydream.archive import scan
from daydream.archive.git_safe import classify_remote_url, normalize_remote_url
from daydream.trajectory import redact_text, redact_value

__all__ = ["ImportResult", "SanitizeResult", "import_bundle", "sanitize_archive", "sanitize_bundle"]

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

    @property
    def released(self) -> bool:
        """True only when the derivative passed the release scan (M16)."""
        return self.status == "sanitized"


@dataclass(frozen=True)
class ImportResult:
    """Outcome of the fail-closed Hub-bundle ingest gate (M18)."""

    source: Path
    imported: bool
    quarantined: bool


class _DerivativeUncleanError(Exception):
    """Internal sentinel: the release scan found the derivative unclean."""


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
    """Redact one text file body, then re-check with the scanner's extra rules.

    The scanner's two local gaps (token-only userinfo, credential query params)
    are applied from scan.py's own patterns, so a rule added there applies here
    too and the derivative passes the release scan (M16).
    """
    text = redact_text(text)
    text = scan._TOKEN_ONLY_USERINFO_PATTERN.sub(r"\1[REDACTED_USER]@", text)
    text = scan._QUERY_CREDENTIAL_PATTERN.sub(r"\1\2=[REDACTED_CREDENTIAL]", text)
    return text


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


def _quarantine_derivative(
    derivative_dir: Path, sanitized_dir: Path, archive_dir: Path, run_dir: Path, session_id: str
) -> None:
    """Move a failed derivative to quarantine and record it (M16, fail-closed).

    Nothing is deleted: the derivative (a copy, never the bronze original) is
    moved under ``<archive_dir>/quarantine/<session_id>/`` for human review.
    """
    quarantine_dir = archive_dir / "quarantine" / session_id
    quarantine_dir.parent.mkdir(parents=True, exist_ok=True)
    if derivative_dir.exists():
        if quarantine_dir.exists():
            shutil.rmtree(quarantine_dir)  # our own prior derivative copy, not a source
        shutil.move(str(derivative_dir), str(quarantine_dir))
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


def sanitize_bundle(run_dir: Path, archive_dir: Path) -> SanitizeResult:
    """Sanitize one legacy bundle into ``archive_dir/sanitized/<session_id>/``.

    The source bundle is never modified. The derivative is released only when
    the post-transform release scan comes back clean; otherwise it is moved to
    ``archive_dir/quarantine/<session_id>/``, a ``status="quarantined"`` audit
    record is appended, and a quarantined (``released=False``) result is
    returned — nothing under ``sanitized/`` (M16, fail-closed). Unexpected
    failures clean the partial derivative, record the quarantine, and re-raise
    so a bulk caller can continue with the next bundle.
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
        scan_result = scan.scan_run_dir(derivative_dir)
        if not scan_result.clean:
            raise _DerivativeUncleanError(f"derivative scan found {scan_result.summary()}")

        digest = _derivative_digest(derivative_dir)
    except _DerivativeUncleanError:
        _quarantine_derivative(derivative_dir, sanitized_dir, archive_dir, run_dir, session_id)
        return SanitizeResult(
            session_id=session_id,
            source=run_dir,
            derivative_digest="",
            status="quarantined",
        )
    except Exception:
        if derivative_dir.exists():
            shutil.rmtree(derivative_dir, ignore_errors=True)
        _append_jsonl(  # unexpected failure: record quarantine, re-raise for bulk loop
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
        try:
            result = sanitize_bundle(run_dir, archive_dir)
        except Exception:
            # sanitize_bundle already recorded the quarantine audit line and
            # cleaned the partial derivative; one bad bundle never stops the pass.
            continue
        if result.released:
            _mark_done(sanitized_dir, result.session_id, result.derivative_digest)
        results.append(result)
    return results


def import_bundle(run_dir: Path, archive_dir: Path) -> ImportResult:
    """Fail-closed ingest gate for a downloaded Hub bundle (M18).

    The incoming bundle is scanned before ingestion. A clean bundle is
    imported in place; an affected bundle is moved to
    ``<archive_dir>/quarantine/<session_id>/`` and skipped — never imported
    raw, even when a released derivative exists. The move is never a deletion
    of a source bundle; when the quarantine slot is already occupied the move
    is skipped and the bundle is still reported quarantined.
    """
    scan_result = scan.scan_run_dir(run_dir)
    if scan_result.clean:
        return ImportResult(source=run_dir, imported=True, quarantined=False)
    quarantine_dir = archive_dir / "quarantine" / run_dir.name
    quarantine_dir.parent.mkdir(parents=True, exist_ok=True)
    if not quarantine_dir.exists():
        shutil.move(str(run_dir), str(quarantine_dir))
    return ImportResult(source=run_dir, imported=False, quarantined=True)


def report_inventory(archive_dir: Path) -> dict[str, int]:
    """Value-free inventory mode (M11): classify each bundle's remote URL.

    For every ``runs/*`` bundle, ``manifest.json``'s ``git.remote_url`` is
    classified via :func:`classify_remote_url`. Prints and returns counts by
    category (session counts only — never a URL fragment or matched value).
    A malformed manifest counts under ``"unparseable"``. Never raises.
    """
    counts: dict[str, int] = {}
    runs_dir = archive_dir / "runs"
    if runs_dir.is_dir():
        for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
            manifest = run_dir / "manifest.json"
            categories: list[str]
            try:
                data = json.loads(manifest.read_text())
                raw = data["git"]["remote_url"]
            except (OSError, json.JSONDecodeError, KeyError, TypeError):
                categories = ["unparseable"]
            else:
                categories = classify_remote_url(raw) if isinstance(raw, str) else ["unparseable"]
            for category in categories or ["clean"]:
                counts[category] = counts.get(category, 0) + 1
    for category in sorted(counts):
        print(f"{category}: {counts[category]}")
    return counts
