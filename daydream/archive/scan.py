"""Fail-closed bundle secret scanner (issue #981 M9/M11/M13).

Walks a serialized run directory and reports secret-shaped content before any
egress path (Hub upload, ``--dump-artifacts``, sanitizer release) touches it.
Safe-only reporting (M11): findings carry the file name, JSON key path or line
number, category, and a short digest of the matched region — never the matched
value or its surrounding content. Any scanner error is absorbed into a
``scan_error`` finding so a broken scan can never return clean (fail-closed).

Redaction rule shapes are imported from :mod:`daydream.trajectory` (reuse, not
copy); the pattern gaps unique to serialized bundles (token-only userinfo,
scheme-less SCP userinfo, credential-bearing query params) are local additions.

Rules are tiered by severity (issue #1170). A redactor needs recall — over-
matching ``SORT_KEY = "created_at"`` costs one value in a log. A publication
gate needs precision — over-matching costs the whole run. So the name-shape
heuristics that carry no value constraint report as ``advisory`` and only the
high-confidence value shapes are ``blocking``. Every rule still reports; the
severity decides whether egress is refused.
"""

import hashlib
import json
import re
from collections.abc import Iterable, Iterator
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

__all__ = ["SEVERITY_ADVISORY", "SEVERITY_BLOCKING", "Finding", "ScanResult", "scan_run_dir"]

#: A rule whose match is a credential by value shape. Refuses egress.
SEVERITY_BLOCKING = "blocking"
#: A rule whose match is only a name/template shape. Reported, never refused.
SEVERITY_ADVISORY = "advisory"

# Token-only userinfo: ``https://x-access-token@github.com/...`` — the
# trajectory URL-credential rule only matches ``user:pass@``, so this closes
# the single-token gap (matches the pinned inventory's x-access-token rows).
_TOKEN_ONLY_USERINFO_PATTERN = re.compile(r"(https?://)[^@/\s]+@", re.IGNORECASE)
# Scheme-less SCP userinfo: ``user:pass@host:path``. The trajectory URL rule
# and the token-only rule above both anchor on ``https?://``, so this closes
# the SCP gap that git_safe.classify_remote_url labels a credential (':' in
# the pre-@ user group). A lone login user (git@host:path) is not a credential
# and is intentionally not matched, matching git_safe's classification.
_SCP_USERINFO_PATTERN = re.compile(
    r"([^\s@/:]+:[^\s@/:]+@)([^/\s@:]+:)(?=[^\s])", re.IGNORECASE,
)
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
#
# Blocking tier: every rule here constrains the matched *value*, so a hit is a
# credential (a known token prefix, key armor, a literal ``user:pass@``, a
# credential-bearing query param) rather than a name that merely sounds like
# one.
_BLOCKING_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (_URL_CREDENTIAL_PATTERN, "url_credential"),
    (_TOKEN_ONLY_USERINFO_PATTERN, "url_credential"),
    (_SCP_USERINFO_PATTERN, "url_credential"),
    (_QUERY_CREDENTIAL_PATTERN, "query_credential"),
    (_PEM_KEY_PATTERN, "pem_key"),
    (_API_KEY_PATTERN, "api_key"),
    (_JWT_PATTERN, "jwt"),
)
# Advisory tier: pure name-shape heuristics. ``_ENV_VAR_PATTERN``'s value group
# is ``[^\s\n\r;]+`` — any non-space token — so it matches ``SORT_KEY =
# "created_at"``, ``AUTH=none`` and ``API_TOKEN=${API_TOKEN}``. Correct for
# redaction, far too coarse to refuse a publication.
_ADVISORY_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (_ENV_VAR_PATTERN, "env_var"),
)
_RULES: tuple[tuple[re.Pattern[str], str, str], ...] = tuple(
    [(p, c, SEVERITY_BLOCKING) for p, c in _BLOCKING_RULES]
    + [(p, c, SEVERITY_ADVISORY) for p, c in _ADVISORY_RULES]
)

# A userinfo part that is entirely an interpolation slot (``{cfg.DB_USER}``,
# ``{password}``, ``${PGPASSWORD}``) is a template, not a value. The optional
# opener/closer absorbs the string literal the slot is written inside: the
# userinfo character classes are greedy over quotes, so the first captured part
# of ``f"{cfg.DB_USER}:{cfg.DB_PASSWORD}@..."`` is ``f"{cfg.DB_USER}``, not
# ``{cfg.DB_USER}``. A quote is never a credential character (RFC 3986 userinfo
# excludes it), and the prefix letters are only accepted when a quote follows,
# so ``pw{n}`` stays blocking.
_PLACEHOLDER_PART_PATTERN = re.compile(r"""(?:[A-Za-z]{0,2}["'`])?\$?\{[^{}]*\}["'`]?""")


@dataclass(frozen=True)
class Finding:
    """A safe-only secret finding: never carries the matched value (M11)."""

    path: str
    location: str
    category: str
    digest: str
    #: ``SEVERITY_BLOCKING`` or ``SEVERITY_ADVISORY``. Defaults to blocking so
    #: every construction site that predates the tiering — and ``scan_error``
    #: in particular — stays fail-closed without naming the field.
    severity: str = SEVERITY_BLOCKING


@dataclass
class ScanResult:
    clean: bool = True
    findings: list[Finding] = field(default_factory=list)

    @property
    def blocking(self) -> bool:
        """Whether egress must be refused.

        Fail-closed tie-break: a result that is not clean but carries no
        findings blocks, so a caller that only knows ``clean=False`` (a
        hand-built stub, a future producer) can never publish by omission.
        """
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
    """Colon-separated userinfo parts a userinfo rule captured, else ``None``.

    Each rule captures the userinfo differently: ``_URL_CREDENTIAL_PATTERN``
    has separate user/password groups, ``_TOKEN_ONLY_USERINFO_PATTERN`` only
    captures the scheme, and ``_SCP_USERINFO_PATTERN``'s group 1 is the
    *combined* ``user:pass@`` with no separate groups at all — hence the split.
    """
    if pattern is _URL_CREDENTIAL_PATTERN:
        return [match.group(2), match.group(3)]
    if pattern is _TOKEN_ONLY_USERINFO_PATTERN:
        return match.group(0)[len(match.group(1)) :].removesuffix("@").split(":")
    if pattern is _SCP_USERINFO_PATTERN:
        return match.group(1).removesuffix("@").split(":")
    return None


def _is_placeholder_userinfo(parts: list[str]) -> bool:
    """Whether every userinfo part is wholly a ``{...}`` interpolation slot.

    ``f"postgresql://{cfg.DB_USER}:{cfg.DB_PASSWORD}@{cfg.DB_HOST}:..."`` has
    the credential shape but carries no credential. The guard covers all three
    userinfo rules, not just the scheme-bearing one: ``_URL_CREDENTIAL_PATTERN``
    is a strict *subset* of ``_TOKEN_ONLY_USERINFO_PATTERN`` (both match
    ``https://{a}:{b}@``), so guarding one and not the other leaves every
    ``https://``-scheme template blocking through the wider rule.
    """
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
    """Collapse identical findings, resolving to the strictest severity.

    Keyed on ``(path, location, category, digest)``. The URL-credential and
    token-only rules both fire on every ``scheme://user:pass@`` with the same
    matched string, so without an explicit blocking-wins tie-break a text's
    severity would depend on rule iteration order. Insertion order is kept so
    the finding list stays reproducible.
    """
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
    """Yield PEM findings the per-line pass structurally cannot see (#1170 D4).

    ``_PEM_KEY_PATTERN`` is ``re.DOTALL`` and spans BEGIN..END lines, but the
    non-JSON pass above scans line by line, so real key armor in ``diff.patch``
    was invisible while the same key inside a JSON string leaf was caught. The
    location is the line the block starts on. A single-line block matches both
    passes with the same span, and ``_dedupe`` collapses it.
    """
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
