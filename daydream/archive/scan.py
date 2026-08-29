"""Fail-closed bundle secret scanner (issue #981 M9/M11/M13).

Walks a serialized run directory and reports secret-shaped content before any
egress path (Hub upload, ``--dump-artifacts``, sanitizer release) touches it.
Safe-only reporting (M11): findings carry the file name, JSON key path or line
number, category, and a short digest of the matched region — never the matched
value or its surrounding content. Any scanner error is absorbed into a
``scan_error`` finding so a broken scan can never return clean (fail-closed).

Redaction rule shapes are imported from :mod:`daydream.trajectory` (reuse, not
copy); the two pattern gaps unique to serialized bundles (token-only userinfo,
credential-bearing query params) are local additions.
"""

import hashlib
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from daydream.archive.git_safe import _CREDENTIAL_QUERY_KEYS
from daydream.trajectory import (
    _API_KEY_PATTERN,
    _ENV_VAR_PATTERN,
    _JWT_PATTERN,
    _PEM_KEY_PATTERN,
    _URL_CREDENTIAL_PATTERN,
)

__all__ = ["Finding", "ScanResult", "scan_run_dir"]

# Token-only userinfo: ``https://x-access-token@github.com/...`` — the
# trajectory URL-credential rule only matches ``user:pass@``, so this closes
# the single-token gap (matches the pinned inventory's x-access-token rows).
_TOKEN_ONLY_USERINFO_PATTERN = re.compile(r"(https?://)[^@/\s]+@", re.IGNORECASE)
# Credential-like query params: ``?token=...`` / ``&access_token=...`` etc.
# The key set comes from git_safe._CREDENTIAL_QUERY_KEYS (single source: a key
# added there widens this scan gate automatically).
_QUERY_CREDENTIAL_PATTERN = re.compile(
    r"([?&])(" + "|".join(sorted(_CREDENTIAL_QUERY_KEYS)) + r")=[^&\s]+",
    re.IGNORECASE,
)

# (pattern, category) pairs applied in order to every scanned text. The
# trajectory rules are imported; the two local rules close the userinfo/query
# gaps without touching the shared trajectory module.
_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (_URL_CREDENTIAL_PATTERN, "url_credential"),
    (_TOKEN_ONLY_USERINFO_PATTERN, "url_credential"),
    (_QUERY_CREDENTIAL_PATTERN, "query_credential"),
    (_PEM_KEY_PATTERN, "pem_key"),
    (_ENV_VAR_PATTERN, "env_var"),
    (_API_KEY_PATTERN, "api_key"),
    (_JWT_PATTERN, "jwt"),
)


@dataclass(frozen=True)
class Finding:
    """A safe-only secret finding: never carries the matched value (M11)."""

    path: str
    location: str
    category: str
    digest: str


@dataclass
class ScanResult:
    clean: bool = True
    findings: list[Finding] = field(default_factory=list)

    def summary(self) -> str:
        """Human-readable counts + paths only — never finding values."""
        if self.clean:
            return "clean"
        by_path: dict[str, int] = {}
        for f in self.findings:
            by_path[f.path] = by_path.get(f.path, 0) + 1
        parts = ", ".join(f"{p}: {n}" for p, n in sorted(by_path.items()))
        return f"{len(self.findings)} finding(s) — {parts}"


def _digest(matched: str) -> str:
    return hashlib.sha256(matched.encode()).hexdigest()[:12]


def _scan_text(text: str) -> Iterator[tuple[str, str, str]]:
    """Yield (category, matched_value, digest) for every rule hit in *text*."""
    for pattern, category in _RULES:
        for match in pattern.finditer(text):
            # Redaction markers are already-safe output, not secrets; scanning
            # sanitized text must not flag its own markers.
            if "[REDACTED_" in match.group(0):
                continue
            yield category, match.group(0), _digest(match.group(0))


def _json_string_leaves(value: object) -> Iterator[tuple[str, str]]:
    """Walk parsed JSON, yielding (key path, string leaf) pairs."""
    if isinstance(value, dict):
        for key, child in value.items():
            for path, leaf in _json_string_leaves(child):
                yield f"{key}.{path}" if path else str(key), leaf
    elif isinstance(value, list):
        for index, child in enumerate(value):
            for path, leaf in _json_string_leaves(child):
                yield f"[{index}].{path}" if path else f"[{index}]", leaf
    elif isinstance(value, str):
        yield "", value


def _scan_file(rel_path: str, text: str) -> list[Finding]:
    """Scan one decoded file, preferring JSON key paths for location."""
    try:
        parsed = json.loads(text)
    except ValueError:
        parsed = None
    if isinstance(parsed, (dict, list)):
        findings = []
        for key_path, leaf in _json_string_leaves(parsed):
            for category, _matched, digest in _scan_text(leaf):
                findings.append(
                    Finding(
                        path=rel_path,
                        location=f"{key_path} (json)" if key_path else "(json)",
                        category=category,
                        digest=digest,
                    )
                )
        return findings
    findings = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for category, _matched, digest in _scan_text(line):
            findings.append(
                Finding(
                    path=rel_path,
                    location=f"line {line_number}",
                    category=category,
                    digest=digest,
                )
            )
    return findings


def scan_run_dir(run_dir: Path) -> ScanResult:
    """Recursively scan every file under *run_dir*; fail closed on any error.

    Files that fail UTF-8 decode are findings themselves (no binary exemption
    in v1). Exceptions from decoding, regex execution, or I/O are absorbed
    per-file into a ``scan_error`` finding — this function never raises and
    never returns a clean result when something went wrong.
    """
    result = ScanResult()
    if not run_dir.is_dir():
        result.clean = False
        result.findings.append(
            Finding(path=str(run_dir), location="(missing)", category="scan_error", digest="")
        )
        return result
    for file_path in sorted(run_dir.rglob("*")):
        if not file_path.is_file():
            continue
        rel_path = file_path.relative_to(run_dir).as_posix()
        try:
            text = file_path.read_text(encoding="utf-8")
            result.findings.extend(_scan_file(rel_path, text))
        except Exception:  # noqa: BLE001 - fail closed on any per-file error
            result.findings.append(
                Finding(path=rel_path, location="(unreadable)", category="scan_error", digest="")
            )
    result.clean = not result.findings
    return result
