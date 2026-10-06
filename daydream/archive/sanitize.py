"""Copy legacy bundles into credential-free derivatives without modifying sources.

JSON URL leaves use the shared Git URL normalizer; all string content uses
shared text/value redaction. A publication scan gates release: blocking
findings quarantine the derivative, while advisory findings remain visible
in a value-free report.

Digests hash canonical (relative path, file SHA-256) pairs for released derivatives.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from daydream.archive import scan
from daydream.archive._console import warn as _warn
from daydream.archive.git_safe import classify_remote_url, normalize_remote_url
from daydream.timeutil import now_iso_utc
from daydream.trajectory import redact_text, redact_value

__all__ = ["SanitizeResult", "sanitize_bundle", "sanitize_bundle_files"]

_AUDIT_FILENAME = "audit.jsonl"
# Marker written inside a quarantined *derivative* so a later replacement can
# tell our own copy from an imported source bundle parked in the same
# ``quarantine/<name>`` namespace (M14).
_DERIVATIVE_MARKER = ".daydream_derivative_marker"


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


def _sanitize_json_document(doc: Any) -> Any:
    """Canonicalize URL leaves, then run every string leaf through the text pipeline.

    Tool observations routinely embed URLs inside prose, so URL normalization
    alone is insufficient. Text and JSON leaves use the same redaction rules.
    """
    if isinstance(doc, dict):
        return {key: _sanitize_json_document(child) for key, child in doc.items()}
    if isinstance(doc, list):
        return [_sanitize_json_document(child) for child in doc]
    if isinstance(doc, str):
        return redact_text(_sanitize_url_string(doc))
    return doc


def sanitize_bundle_files(derivative_dir: Path) -> None:
    """Transform a private bundle copy in place; callers must scan before release.

    This transformation does not copy, publish, or record archive audit state.
    Never pass an original or frozen evidence directory.
    """
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
                sanitized = redact_value(_sanitize_json_document(doc))
                file_path.write_text(
                    json.dumps(sanitized, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8",
                )
                continue
        file_path.write_text(redact_text(text), encoding="utf-8")


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _resolve_session_id(run_dir: Path) -> str:
    """Use manifest identity, falling back to the directory name, consistently with bundle sanitization."""
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        except ValueError:
            loaded = {}
        session_id = loaded.get("session_id") if isinstance(loaded, dict) else None
        if session_id:
            return str(session_id)
    return run_dir.name


def _append_audit(
    sanitized_dir: Path, run_dir: Path, session_id: str, *, status: str, derivative_digest: str = ""
) -> None:
    """Append the audit record for a sanitized or quarantined bundle."""
    _append_jsonl(
        sanitized_dir / _AUDIT_FILENAME,
        {
            "source": str(run_dir),
            "session_id": session_id,
            "derivative_digest": derivative_digest,
            "status": status,
            "completed_at": now_iso_utc(),
        },
    )


def _quarantine_derivative(
    derivative_dir: Path, sanitized_dir: Path, archive_dir: Path, run_dir: Path, session_id: str
) -> None:
    """Quarantine a derivative without overwriting imported source bundles.

    Only an ownership marker permits replacing a prior derivative slot;
    unowned collisions use a sibling slot.
    """
    quarantine_dir = archive_dir / "quarantine" / session_id
    quarantine_dir.parent.mkdir(parents=True, exist_ok=True)
    if derivative_dir.exists():
        if (quarantine_dir / _DERIVATIVE_MARKER).exists():
            shutil.rmtree(quarantine_dir)  # our own prior derivative copy, not a source
        if quarantine_dir.exists():
            # The slot is not provably our derivative (e.g. an imported source
            # retained at quarantine/<name>). Never delete it;
            # park this failed derivative in a unique sibling slot instead.
            quarantine_dir = quarantine_dir.with_name(f"{session_id}.{int(time.time())}")
        shutil.move(str(derivative_dir), str(quarantine_dir))
        (quarantine_dir / _DERIVATIVE_MARKER).write_text(now_iso_utc(), encoding="utf-8")
    _append_audit(sanitized_dir, run_dir, session_id, status="quarantined")


def sanitize_bundle(run_dir: Path, archive_dir: Path) -> SanitizeResult:
    """Copy, sanitize, and scan one bundle; release only nonblocking derivatives.

    The source remains untouched. Blocked output moves to quarantine with an
    audit record. Unexpected failures remove partial output, record quarantine,
    and propagate to the caller.
    """
    sanitized_dir = archive_dir / "sanitized"
    session_id = _resolve_session_id(run_dir)
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
        sanitize_bundle_files(derivative_dir)

        # Fail-closed release gate: a blocking finding (or a scanner error)
        # withholds the derivative. An advisory finding is by construction not
        # a credential (#1170) — a rule that cannot identify a secret must not
        # gate egress either — so it is reported and the derivative released.
        scan_result = scan.scan_run_dir(derivative_dir)
        if scan_result.blocking:
            raise _DerivativeUncleanError(f"derivative scan found {scan_result.summary()}")
        if scan_result.findings:
            _warn(
                f"Sanitized derivative for {session_id} carries advisory-only scan "
                f"findings ({scan_result.summary()})"
            )

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
        # Unexpected failure: record quarantine, re-raise to the caller.
        _append_audit(sanitized_dir, run_dir, session_id, status="quarantined")
        raise

    _append_audit(sanitized_dir, run_dir, session_id, status="sanitized", derivative_digest=digest)
    return SanitizeResult(
        session_id=session_id,
        source=run_dir,
        derivative_digest=digest,
        status="sanitized",
    )
