"""Fail-closed secret scanning before bundle egress.

Findings contain only paths, locations, categories, and short digests, never
matched values or excerpts. Per-file scanner errors become blocking findings.
Patterns are shared with ``daydream.redaction`` and ``credential_patterns``.
Value-constrained credential shapes block egress; name/template heuristics are
advisory because names such as ``SORT_KEY`` alone do not identify credentials.
Both tiers report findings; severity determines whether publication is refused.
"""

import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from daydream.credential_patterns import (
    _QUERY_CREDENTIAL_PATTERN,
    _SCP_USERINFO_PATTERN,
    _TOKEN_ONLY_USERINFO_PATTERN,
    _URL_CREDENTIAL_PATTERN,
)
from daydream.redaction import _API_KEY_PATTERN, _ENV_VAR_PATTERN, _JWT_PATTERN, _PEM_KEY_PATTERN

__all__ = ["SEVERITY_ADVISORY", "SEVERITY_BLOCKING", "Finding", "ScanResult", "scan_run_dir"]

#: A rule whose match is a credential by value shape. Refuses egress.
SEVERITY_BLOCKING = "blocking"
#: A rule whose match is only a name/template shape. Reported, never refused.
SEVERITY_ADVISORY = "advisory"

# Ordered value-constrained rules shared with live redaction.
_BLOCKING_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (_URL_CREDENTIAL_PATTERN, "url_credential"),
    (_TOKEN_ONLY_USERINFO_PATTERN, "url_credential"),
    (_SCP_USERINFO_PATTERN, "url_credential"),
    (_QUERY_CREDENTIAL_PATTERN, "query_credential"),
    (_PEM_KEY_PATTERN, "pem_key"),
    (_API_KEY_PATTERN, "api_key"),
    (_JWT_PATTERN, "jwt"),
)
# Name-only heuristics also match SORT_KEY="created_at" and API_TOKEN=${API_TOKEN};
# useful for redaction, insufficient to refuse publication.
_ADVISORY_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (_ENV_VAR_PATTERN, "env_var"),
)
_RULES: tuple[tuple[re.Pattern[str], str, str], ...] = tuple(
    [(p, c, SEVERITY_BLOCKING) for p, c in _BLOCKING_RULES]
    + [(p, c, SEVERITY_ADVISORY) for p, c in _ADVISORY_RULES]
)

# Whole template slots are safe, including string prefixes/quotes captured in userinfo.
# Require the prefix's quote so mixed literal passwords such as pw{n} still block.
_PLACEHOLDER_PART_PATTERN = re.compile(r"""(?:[A-Za-z]{0,2}["'`])?\$?\{[^{}]*\}["'`]?""")


@dataclass(frozen=True)
class Finding:
    """A safe-only secret finding: never carries the matched value (M11)."""

    path: str
    location: str
    category: str
    digest: str
    #: Default to blocking, including scan errors and callers omitting severity.
    severity: str = SEVERITY_BLOCKING


@dataclass
class ScanResult:
    clean: bool = True
    findings: list[Finding] = field(default_factory=list)

    @property
    def blocking(self) -> bool:
        """Refuse egress for blocking findings or ``clean=False`` without findings."""
        if not self.clean and not self.findings:
            return True
        return any(f.severity == SEVERITY_BLOCKING for f in self.findings)

    def summary(self) -> str:
        """Human-readable counts, paths and categories — never finding values."""
        if self.clean:
            return "clean"
        if not self.findings:
            return "not clean (no findings recorded)"
        blocking = sum(1 for f in self.findings if f.severity == SEVERITY_BLOCKING)
        by_path: dict[str, list[str]] = {}
        for f in self.findings:
            by_path.setdefault(f.path, []).append(f.category)
        parts = ", ".join(
            f"{p}: {len(cats)} ({'/'.join(sorted(set(cats)))})" for p, cats in sorted(by_path.items())
        )
        return (
            f"{len(self.findings)} finding(s) "
            f"({blocking} blocking, {len(self.findings) - blocking} advisory) — {parts}"
        )


def _digest(matched: str) -> str:
    return hashlib.sha256(matched.encode()).hexdigest()[:12]


def _userinfo_parts(pattern: re.Pattern[str], match: re.Match[str]) -> list[str] | None:
    """Normalize each userinfo rule's capture into colon-separated parts."""
    if pattern is _URL_CREDENTIAL_PATTERN:
        return [match.group(2), match.group(3)]
    if pattern is _TOKEN_ONLY_USERINFO_PATTERN:
        return match.group(0)[len(match.group(1)) :].removesuffix("@").split(":")
    if pattern is _SCP_USERINFO_PATTERN:
        return match.group(1).removesuffix("@").split(":")
    return None


def _is_placeholder_userinfo(parts: list[str]) -> bool:
    """Recognize userinfo made entirely of interpolation slots, including quoted literals."""
    return bool(parts) and all(_PLACEHOLDER_PART_PATTERN.fullmatch(part) for part in parts)


def _scan_text(text: str) -> Iterator[tuple[str, str, str, str]]:
    """Yield (category, matched_value, digest, severity) per rule hit in *text*."""
    for pattern, category, severity in _RULES:
        for match in pattern.finditer(text):
            # Redaction markers are already-safe output, not secrets; scanning
            # sanitized text must not flag its own markers.
            if "[REDACTED_" in match.group(0):
                continue
            effective = severity
            if severity == SEVERITY_BLOCKING:
                parts = _userinfo_parts(pattern, match)
                if parts is not None and _is_placeholder_userinfo(parts):
                    effective = SEVERITY_ADVISORY
            yield category, match.group(0), _digest(match.group(0)), effective


def _dedupe(findings: Iterable[Finding]) -> list[Finding]:
    """Deduplicate by ``(path, location, category, digest)``, retaining insertion order."""
    resolved: dict[tuple[str, str, str, str], Finding] = {}
    for finding in findings:
        key = (finding.path, finding.location, finding.category, finding.digest)
        existing = resolved.get(key)
        if existing is None or (
            existing.severity != SEVERITY_BLOCKING and finding.severity == SEVERITY_BLOCKING
        ):
            resolved[key] = finding
    return list(resolved.values())


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
            for category, _matched, digest, severity in _scan_text(leaf):
                findings.append(
                    Finding(
                        path=rel_path,
                        location=f"{key_path} (json)" if key_path else "(json)",
                        category=category,
                        digest=digest,
                        severity=severity,
                    )
                )
        return _dedupe(findings)
    findings = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for category, _matched, digest, severity in _scan_text(line):
            findings.append(
                Finding(
                    path=rel_path,
                    location=f"line {line_number}",
                    category=category,
                    digest=digest,
                    severity=severity,
                )
            )
    findings.extend(_scan_multiline_pem(rel_path, text))
    return _dedupe(findings)


def _scan_multiline_pem(rel_path: str, text: str) -> Iterator[Finding]:
    """Find PEM armor spanning lines that the non-JSON per-line pass cannot match."""
    for match in _PEM_KEY_PATTERN.finditer(text):
        matched = match.group(0)
        if "[REDACTED_" in matched:
            continue
        start_line = text.count("\n", 0, match.start()) + 1
        yield Finding(
            path=rel_path,
            location=f"line {start_line}",
            category="pem_key",
            digest=_digest(matched),
            severity=SEVERITY_BLOCKING,
        )


def scan_run_dir(run_dir: Path) -> ScanResult:
    """Recursively scan every file under *run_dir*; fail closed on any error."""
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
