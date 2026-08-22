"""Pure review scoring core for the Daydream Harbor verifier.

Stdlib-only (dataclasses, hashlib, json, re, math) so this module can be
copied byte-for-byte into a judge-free Harbor verifier image. No daydream
source, no pydantic, no third-party imports.
"""

from __future__ import annotations

import hashlib
import json
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


# ---------------------------------------------------------------------------
# candidate artifact + gold set validation
# ---------------------------------------------------------------------------


def _canonical_tuple(finding: object) -> tuple[object, ...]:
    return (
        str(_finding_component(finding, "title") or ""),
        str(_finding_component(finding, "body") or ""),
        str(_finding_component(finding, "severity") or ""),
        str(_finding_component(finding, "path")),
        _component_int(finding, "start_line"),
        _component_int(finding, "end_line"),
    )


def validate_candidate_artifact(raw: dict[str, object]) -> list[CandidateFinding]:
    """Validate a §9 candidate artifact and return its parsed findings."""
    if not isinstance(raw, dict):
        raise VerifierError("candidate artifact must be a dict")
    try:
        schema_version = raw["schema_version"]
        case_id = raw["case_id"]
        base_ref = raw["base_ref"]
        head_ref = raw["head_ref"]
        findings = raw["findings"]
    except KeyError as exc:
        raise VerifierError(f"missing artifact field {exc.args[0]}") from exc
    if schema_version != 1:
        raise VerifierError(f"unsupported schema_version {schema_version!r}")
    if not isinstance(case_id, str) or not isinstance(base_ref, str) or not isinstance(head_ref, str):
        raise VerifierError("case_id/base_ref/head_ref must be strings")
    if len(json.dumps(raw).encode("utf-8")) > MAX_ARTIFACT_BYTES:
        raise VerifierError("candidate artifact exceeds 1 MiB")
    if not isinstance(findings, list):
        raise VerifierError("artifact findings must be a list")
    if len(findings) > MAX_CANDIDATE_FINDINGS:
        raise VerifierError("candidate artifact exceeds 100 findings")

    parsed = [parse_candidate_finding(f) for f in findings]  # type: ignore[arg-type]

    seen: dict[tuple[object, ...], int] = {}
    ids: set[str] = set()
    for finding in parsed:
        canon = _canonical_tuple(finding)
        ordinal = seen.get(canon, 0)
        seen[canon] = ordinal + 1
        expected = derive_candidate_id(case_id, finding, ordinal)
        if finding.candidate_id != expected:
            raise VerifierError("candidate_id does not match the derived id")
        if finding.candidate_id in ids:
            raise VerifierError("duplicate candidate_id in artifact")
        ids.add(finding.candidate_id)
    return parsed


def validate_gold_set(raw: list[dict[str, object]]) -> list[GoldFinding]:
    """Validate a gold set against the 50-finding cap and field limits."""
    if len(raw) > MAX_GOLD_FINDINGS:
        raise VerifierError("gold set exceeds 50 findings")
    return [parse_gold_finding(f) for f in raw]


# ---------------------------------------------------------------------------
# verdicts + edge retention
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    """A caller-injected verdict for one gold/candidate pair."""

    gold_id: str
    candidate_id: str
    match: bool
    confidence: float
    reasoning: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise VerifierError(f"confidence must be in [0,1], got {self.confidence!r}")


def retained_edges(
    verdicts: list[Verdict],
    gold_ids: list[str],
    candidate_ids: list[str],
) -> list[Verdict]:
    """Return verdicts retained as edges: match true and confidence >= 0.7."""
    gold_set = set(gold_ids)
    cand_set = set(candidate_ids)
    return [
        v
        for v in verdicts
        if v.match
        and v.confidence >= CONFIDENCE_THRESHOLD
        and v.gold_id in gold_set
        and v.candidate_id in cand_set
    ]


# ---------------------------------------------------------------------------
# maximum-cardinality one-to-one matching
# ---------------------------------------------------------------------------


def _edge_key(v: Verdict) -> tuple[float, str, str]:
    return (-v.confidence, v.gold_id, v.candidate_id)


def maximum_matching(
    verdicts: list[Verdict],
    gold_ids: list[str],
    candidate_ids: list[str],
) -> set[tuple[str, str]]:
    """Maximum-cardinality one-to-one matching over retained verdict edges.

    Edges are ordered deterministically (descending confidence, then gold ID,
    then candidate ID); golds are visited in sorted-id order and candidates in
    the fixed adjacency order, so the result is stable run-to-run. Never
    iterates a set/dict for an ordering decision.
    """
    ordered = sorted(verdicts, key=_edge_key)
    adjacency: dict[str, list[str]] = {}
    for v in ordered:
        adjacency.setdefault(v.gold_id, []).append(v.candidate_id)
    gold_order = sorted(gold_ids)

    match_candidate: dict[str, str] = {}  # candidate_id -> gold_id

    def _first_free(gold: str) -> str | None:
        for cand in adjacency.get(gold, []):
            if cand not in match_candidate:
                return cand
        return None

    def _augment(gold: str, seen: set[str]) -> bool:
        for cand in adjacency.get(gold, []):
            if cand in seen:
                continue
            seen.add(cand)
            owner = match_candidate.get(cand)
            if owner is None or _augment(owner, seen):
                match_candidate[cand] = gold
                return True
        return False

    for gold in gold_order:
        free = _first_free(gold)
        if free is not None:
            match_candidate[free] = gold
        else:
            _augment(gold, set())

    return {(match_candidate[c], c) for c in match_candidate}
