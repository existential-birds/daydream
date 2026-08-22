"""Pure review scoring core for the Daydream Harbor verifier.

Stdlib-only (dataclasses, hashlib, json, re, math) so this module can be
copied byte-for-byte into a judge-free Harbor verifier image. No daydream
source, no pydantic, no third-party imports.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

MAX_ARTIFACT_BYTES = 1_048_576
MAX_CANDIDATE_FINDINGS = 100
MAX_GOLD_FINDINGS = 50
CONFIDENCE_THRESHOLD = 0.7

_SEP = "\x1f"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class VerifierError(Exception):
    """Raised on any invalid verifier input (mirrors a pydantic error)."""


# ---------------------------------------------------------------------------
# field validators
# ---------------------------------------------------------------------------


def _validate_hex64(value: object, field: str) -> str:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise VerifierError(f"{field} must be 64-hex, got {value!r}")
    return value


def _validate_title(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise VerifierError(f"title must not be blank, got {value!r}")
    if "\x00" in value:
        raise VerifierError("title must not contain NUL")
    if len(value.encode("utf-8")) > 500:
        raise VerifierError("title exceeds 500 UTF-8 bytes")
    return value


def _validate_body(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise VerifierError(f"body must not be blank, got {value!r}")
    if "\x00" in value:
        raise VerifierError("body must not contain NUL")
    if len(value.encode("utf-8")) > 8 * 1024:
        raise VerifierError("body exceeds 8 KiB")
    return value


def _validate_severity(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or value not in ("high", "medium", "low"):
        raise VerifierError(f"severity must be high/medium/low or null, got {value!r}")
    return value


def _validate_path(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise VerifierError(f"path must not be blank, got {value!r}")
    if value.startswith("/"):
        raise VerifierError(f"path must be relative, got {value!r}")
    if value == ".." or value.startswith("../") or "/../" in value or value.endswith("/.."):
        raise VerifierError(f"path must not contain '..' segments: {value!r}")
    if "\x00" in value:
        raise VerifierError("path must not contain NUL")
    return value


def _validate_lines(start_line: object, end_line: object) -> tuple[int, int]:
    if not isinstance(start_line, int) or isinstance(start_line, bool):
        raise VerifierError(f"start_line must be an integer, got {start_line!r}")
    if not isinstance(end_line, int) or isinstance(end_line, bool):
        raise VerifierError(f"end_line must be an integer, got {end_line!r}")
    if start_line < 1 or end_line < 1:
        raise VerifierError("start_line/end_line must be positive")
    if start_line > end_line:
        raise VerifierError("start_line must be <= end_line")
    return start_line, end_line


def _content_fields(raw: dict[str, object]) -> dict[str, object]:
    """Validate the content fields shared by gold and candidate findings."""
    try:
        return {
            "title": _validate_title(raw["title"]),
            "body": _validate_body(raw["body"]),
            "severity": _validate_severity(raw.get("severity")),
            "path": _validate_path(raw["path"]),
            "start_line": raw["start_line"],
            "end_line": raw["end_line"],
        }
    except KeyError as exc:
        raise VerifierError(f"missing required field {exc.args[0]}") from exc


@dataclass(frozen=True)
class GoldFinding:
    """A hidden gold finding, validated with schema.py Finding limits."""

    finding_id: str
    title: str
    body: str
    severity: str | None
    path: str
    start_line: int
    end_line: int

    def __post_init__(self) -> None:
        _validate_hex64(self.finding_id, "finding id")
        _validate_title(self.title)
        _validate_body(self.body)
        _validate_severity(self.severity)
        _validate_path(self.path)
        _validate_lines(self.start_line, self.end_line)


@dataclass(frozen=True)
class CandidateFinding:
    """A daydream candidate finding, validated with schema.py Finding limits."""

    candidate_id: str
    title: str
    body: str
    severity: str | None
    path: str
    start_line: int
    end_line: int

    def __post_init__(self) -> None:
        _validate_hex64(self.candidate_id, "candidate id")
        _validate_title(self.title)
        _validate_body(self.body)
        _validate_severity(self.severity)
        _validate_path(self.path)
        _validate_lines(self.start_line, self.end_line)


def parse_gold_finding(raw: dict[str, object]) -> GoldFinding:
    """Validate a raw gold-finding dict and return a GoldFinding."""
    if not isinstance(raw, dict):
        raise VerifierError("gold finding must be a dict")
    try:
        ident = _validate_hex64(raw["finding_id"], "finding_id")
    except KeyError as exc:
        raise VerifierError(f"missing required field {exc.args[0]}") from exc
    fields = _content_fields(raw)
    return GoldFinding(
        finding_id=ident,
        title=fields["title"],  # type: ignore[arg-type]
        body=fields["body"],  # type: ignore[arg-type]
        severity=fields["severity"],  # type: ignore[arg-type]
        path=fields["path"],  # type: ignore[arg-type]
        start_line=fields["start_line"],  # type: ignore[arg-type]
        end_line=fields["end_line"],  # type: ignore[arg-type]
    )


def parse_candidate_finding(raw: dict[str, object]) -> CandidateFinding:
    """Validate a raw candidate-finding dict and return a CandidateFinding."""
    if not isinstance(raw, dict):
        raise VerifierError("candidate finding must be a dict")
    try:
        ident = _validate_hex64(raw["candidate_id"], "candidate_id")
    except KeyError as exc:
        raise VerifierError(f"missing required field {exc.args[0]}") from exc
    fields = _content_fields(raw)
    return CandidateFinding(
        candidate_id=ident,
        title=fields["title"],  # type: ignore[arg-type]
        body=fields["body"],  # type: ignore[arg-type]
        severity=fields["severity"],  # type: ignore[arg-type]
        path=fields["path"],  # type: ignore[arg-type]
        start_line=fields["start_line"],  # type: ignore[arg-type]
        end_line=fields["end_line"],  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# deterministic candidate-ID derivation
# ---------------------------------------------------------------------------


def _finding_component(finding: object, name: str) -> object:
    """Read a field from either a dataclass attribute or a dict key."""
    if isinstance(finding, dict):
        try:
            return finding[name]
        except KeyError as exc:
            raise VerifierError(f"missing required field {name}") from exc
    return getattr(finding, name)


def derive_candidate_id(
    case_key: str,
    finding: CandidateFinding | dict[str, object],
    ordinal: int,
) -> str:
    """Return the deterministic sha256 candidate id for a finding."""
    title = str(_finding_component(finding, "title") or "")
    body = str(_finding_component(finding, "body") or "")
    severity = str(_finding_component(finding, "severity") or "")
    path = str(_finding_component(finding, "path"))
    start_line = _component_int(finding, "start_line")
    end_line = _component_int(finding, "end_line")
    canonical = _SEP.join(
        [title, body, severity, path, str(start_line), str(end_line)]
    )
    payload = _SEP.join([str(case_key), canonical, str(ordinal)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _component_int(finding: object, name: str) -> int:
    value = _finding_component(finding, name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise VerifierError(f"{name} must be an integer, got {value!r}")
    return value
