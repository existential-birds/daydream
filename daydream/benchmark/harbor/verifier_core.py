"""Pure review scoring core for the Daydream Harbor verifier.

Stdlib-only (dataclasses, hashlib, json, re) so this module can be
copied byte-for-byte into a judge-free Harbor verifier image. No daydream
source, no pydantic, no third-party imports.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, cast

MAX_ARTIFACT_BYTES = 1_048_576
MAX_CANDIDATE_FINDINGS = 100
MAX_GOLD_FINDINGS = 50
CONFIDENCE_THRESHOLD = 0.7

_SEP = "\x1f"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")

# Exact, versioned key sets (issue #817): a candidate artifact / finding and a
# gold finding must carry exactly these keys -- no more, no fewer.
CANDIDATE_ARTIFACT_KEYS = {"schema_version", "case_id", "base_ref", "head_ref", "findings"}
CANDIDATE_FINDING_KEYS = {"candidate_id", "title", "body", "severity", "path", "start_line", "end_line"}
GOLD_FINDING_KEYS = {"finding_id", "title", "body", "severity", "path", "start_line", "end_line"}


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
    if len(value) > 500:
        raise VerifierError("title exceeds 500 characters")
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


def _validate_location(
    path: object, start_line: object, end_line: object
) -> tuple[str | None, int | None, int | None]:
    """Validate the exact-one-of location triple shared by gold/candidate findings.

    Either all three components are ``None`` (a locationless review finding that
    names a defect without a file or line), or all three are present and fully
    validated via :func:`_validate_path` / :func:`_validate_lines`. A partially
    populated location (exactly one or two of the three ``None``) is rejected
    with a :class:`VerifierError` naming the partial-population rule.
    """
    if path is None and start_line is None and end_line is None:
        return (None, None, None)
    if path is None or start_line is None or end_line is None:
        raise VerifierError(
            "location must be all-null or fully populated (path, start_line, end_line)"
        )
    return (_validate_path(path), *_validate_lines(start_line, end_line))


def _content_fields(raw: dict[str, object]) -> dict[str, object]:
    """Validate the content fields shared by gold and candidate findings."""
    try:
        path, start_line, end_line = _validate_location(
            raw["path"], raw["start_line"], raw["end_line"]
        )
        return {
            "title": _validate_title(raw["title"]),
            "body": _validate_body(raw["body"]),
            "severity": _validate_severity(raw.get("severity")),
            "path": path,
            "start_line": start_line,
            "end_line": end_line,
        }
    except KeyError as exc:
        raise VerifierError(f"missing required field {exc.args[0]}") from exc


@dataclass(frozen=True)
class _FindingContent:
    """Content fields shared by gold and candidate findings."""

    title: str
    body: str
    severity: str | None
    path: str | None
    start_line: int | None
    end_line: int | None

    def __post_init__(self) -> None:
        _validate_title(self.title)
        _validate_body(self.body)
        _validate_severity(self.severity)
        _validate_location(self.path, self.start_line, self.end_line)


@dataclass(frozen=True)
class GoldFinding(_FindingContent):
    """A hidden gold finding, validated with schema.py Finding limits."""

    finding_id: str

    def __post_init__(self) -> None:
        _validate_hex64(self.finding_id, "finding id")
        super().__post_init__()


@dataclass(frozen=True)
class CandidateFinding(_FindingContent):
    """A daydream candidate finding, validated with schema.py Finding limits."""

    candidate_id: str

    def __post_init__(self) -> None:
        _validate_hex64(self.candidate_id, "candidate id")
        super().__post_init__()


def validate_exact_keys(raw: object, allowed: set[str], context: str) -> None:
    """Enforce an exact key set over a dict, naming only keys and context.

    Raises :class:`VerifierError` when *raw* is not a dict, when any allowed
    key is missing, or when any key present is not in *allowed* -- with a
    leak-free message naming only the keys and *context*, never values.
    """
    if not isinstance(raw, dict):
        raise VerifierError(f"{context} must be a JSON object")
    missing = sorted(allowed - set(raw))
    if missing:
        raise VerifierError(f"{context} missing required field(s): {', '.join(missing)}")
    extra = sorted(set(raw) - allowed)
    if extra:
        raise VerifierError(f"{context} contains unknown field(s): {', '.join(extra)}")


def _finding_kwargs(
    raw: dict[str, object], *, side: str, id_key: str
) -> dict[str, object]:
    """Validate a raw finding dict and return its model constructor kwargs."""
    if not isinstance(raw, dict):
        raise VerifierError(f"{side} finding must be a dict")
    validate_exact_keys(
        raw,
        CANDIDATE_FINDING_KEYS if side == "candidate" else GOLD_FINDING_KEYS,
        f"{side} finding",
    )
    try:
        ident = _validate_hex64(raw[id_key], id_key)
    except KeyError as exc:
        raise VerifierError(f"missing required field {exc.args[0]}") from exc
    fields = _content_fields(raw)
    fields[id_key] = ident
    return fields


def parse_gold_finding(raw: dict[str, object]) -> GoldFinding:
    """Validate a raw gold-finding dict and return a GoldFinding."""
    return GoldFinding(**_finding_kwargs(raw, side="gold", id_key="finding_id"))  # type: ignore[arg-type]


def parse_candidate_finding(raw: dict[str, object]) -> CandidateFinding:
    """Validate a raw candidate-finding dict and return a CandidateFinding."""
    return CandidateFinding(**_finding_kwargs(raw, side="candidate", id_key="candidate_id"))  # type: ignore[arg-type]


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
    canonical = _SEP.join(str(part) for part in _canonical_tuple(finding))
    payload = _SEP.join([str(case_key), canonical, str(ordinal)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# candidate artifact + gold set validation
# ---------------------------------------------------------------------------


def _canonical_tuple(finding: object) -> tuple[object, ...]:
    return (
        str(_finding_component(finding, "title") or ""),
        str(_finding_component(finding, "body") or ""),
        str(_finding_component(finding, "severity") or ""),
        str(_finding_component(finding, "path") or ""),
        str(_finding_component(finding, "start_line") or ""),
        str(_finding_component(finding, "end_line") or ""),
    )


def validate_candidate_artifact(raw: dict[str, object]) -> list[CandidateFinding]:
    """Validate a §9 candidate artifact and return its parsed findings."""
    validate_exact_keys(raw, CANDIDATE_ARTIFACT_KEYS, "candidate artifact")
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

    parsed = [parse_candidate_finding(f) for f in findings]

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


def validate_gold_set(
    raw: list[dict[str, object]], *, case_id: str | None = None
) -> list[GoldFinding]:
    """Validate a gold set: 50-finding cap, per-member fields, canonical unique ids.

    ``case_id`` is the schema-scoped case id the gold finding ids were derived
    with (``sha256(case_id, title, body, severity, path, start_line, end_line)``);
    a non-empty gold set requires it unless ``case_id`` is ``None`` (legacy
    back-scoring of tasks compiled before the case-scoped digest), in which case
    the finding ids are validated against the prior content-only digest. An
    empty gold set (pure-clean case) needs no case_id.
    """
    if len(raw) > MAX_GOLD_FINDINGS:
        raise VerifierError("gold set exceeds 50 findings")
    parsed = [parse_gold_finding(f) for f in raw]
    if not parsed:
        return parsed
    seen: set[str] = set()
    for f in parsed:
        digest_tail = (
            _canonical_tuple(f)
            if case_id is None
            else ((case_id,) + _canonical_tuple(f))
        )
        payload = _SEP.join(str(part) for part in digest_tail)
        expected = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        if f.finding_id != expected:
            raise VerifierError("gold finding_id is not the canonical digest")
        if f.finding_id in seen:
            raise VerifierError("duplicate gold finding_id")
        seen.add(f.finding_id)
    return parsed


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


# ---------------------------------------------------------------------------
# reward + score_review
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Reward:
    """The 22-field §10 per-task reward output.

    The 12 headline fields are unchanged; the 10 location/severity axis
    fields (issue #971) are reported-only, computed over matched tp pairs.
    Axis fields default to zero/absent (0/1 presence flags, no imputed
    values) when the task has no pairs scoring that axis.
    """

    reward: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    gold_count: int = 0
    candidate_count: int = 0
    clean_task: int = 0
    clean_pass: int = 0
    verifier_error: int = 0
    location_exact: int = 0
    location_near: int = 0
    location_file: int = 0
    location_miss: int = 0
    location_credit: float = 0.0
    location_present: int = 0
    severity_exact: int = 0
    severity_within_1: int = 0
    severity_mean_distance: float = 0.0
    severity_credit: float = 0.0
    severity_present: int = 0

    def to_dict(self) -> dict[str, float | int]:
        """Numeric-only dict with exactly the 22 §10 keys."""
        return {
            "reward": self.reward,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "gold_count": self.gold_count,
            "candidate_count": self.candidate_count,
            "clean_task": self.clean_task,
            "clean_pass": self.clean_pass,
            "verifier_error": self.verifier_error,
            "location_exact": self.location_exact,
            "location_near": self.location_near,
            "location_file": self.location_file,
            "location_miss": self.location_miss,
            "location_credit": self.location_credit,
            "location_present": self.location_present,
            "severity_exact": self.severity_exact,
            "severity_within_1": self.severity_within_1,
            "severity_mean_distance": self.severity_mean_distance,
            "severity_credit": self.severity_credit,
            "severity_present": self.severity_present,
        }


LOCATION_TOLERANCE: Final = 3
# Minimum boundary tolerance for location-tier classification (issue #971 R2):
# below 3 lines the "near" tier would measure the posting snapper's behavior,
# not the reviewer's actual localization accuracy.

_RANGE_DISTANCE_DOC = """Distance from ``line`` to the inclusive ``[start, end]`` hunk range.

``0`` when ``line`` lies inside the range, else the distance to the nearer
boundary (``start`` when ``line`` is below it, ``end`` when above).

Private stdlib duplicate of the shared primitive in ``daydream/hunk_index.py``
(the source of truth); policed against drift by
``tests/test_benchmark_verifier_assets.py`` (issue #971 R8).
"""


def _range_distance(line: int, start: int, end: int) -> int:
    if start <= line <= end:
        return 0
    if line < start:
        return start - line
    return line - end


_range_distance.__doc__ = _RANGE_DISTANCE_DOC


_SEVERITY_RANK: Final = {"high": 3, "medium": 2, "low": 1}
"""Severity vocabulary rank (issue #971 R3); validated upstream by
``_validate_severity``, so an unknown value here is a programming error."""

_SEVERITY_CREDIT: Final = {0: 1.0, 1: 0.5, 2: 0.0}
"""Severity-agreement credit per rank distance; distance 2 is the maximum
reachable from the validated high/medium/low vocabulary."""


def location_tier(
    g_path: str,
    g_start: int,
    g_end: int,
    c_path: str,
    c_start: int,
    c_end: int,
    tolerance: int,
) -> str:
    """Classify one matched pair's location agreement into exactly one tier.

    Tiers: ``"exact"`` (same path, distance 0), ``"near"`` (same path,
    distance ``<= tolerance``), ``"file"`` (same path, distance beyond
    tolerance), ``"miss"`` (different path). Distance is the candidate
    range's distance to the inclusive gold range ``[g_start, g_end]``: a
    single-line candidate is a point-in-range check; a multi-line candidate
    range (candidate.py normally sets ``start_line == end_line``) scores 0
    when the ranges overlap at all (both endpoints checked against the gold
    range), else the nearer-boundary distance.
    """
    if g_path != c_path:
        return "miss"
    d_start = _range_distance(c_start, g_start, g_end)
    d_end = _range_distance(c_end, g_start, g_end)
    distance = min(d_start, d_end)
    if distance == 0:
        return "exact"
    if distance <= tolerance:
        return "near"
    return "file"


def severity_distance(g_sev: str | None, c_sev: str | None) -> int | None:
    """Return ``abs(rank(gold) - rank(candidate))`` for the severity axis.

    ``None`` when either side is ``None`` (axis absent -- never imputed,
    mirroring the missing-signal doctrine in ``daydream/training/reward.py``).
    Unknown severity strings cannot occur (validated by ``_validate_severity``
    upstream); an unexpected value raises :class:`VerifierError` rather than
    being coerced.
    """
    if g_sev is None or c_sev is None:
        return None
    g_rank = _SEVERITY_RANK.get(g_sev)
    c_rank = _SEVERITY_RANK.get(c_sev)
    if g_rank is None or c_rank is None:
        raise VerifierError(f"unknown severity, got {g_sev!r}/{c_sev!r}")
    return abs(g_rank - c_rank)


def severity_credit(distance: int) -> float:
    """Map a severity rank distance to partial credit: 0->1.0, 1->0.5, 2->0.0."""
    try:
        return _SEVERITY_CREDIT[distance]
    except KeyError:
        raise VerifierError(f"severity distance out of range, got {distance!r}") from None


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 1.0
    return 2 * precision * recall / (precision + recall)


def _finding_id(finding: object) -> str:
    """Read a gold finding's id from either a GoldFinding or a raw dict."""
    if isinstance(finding, dict):
        try:
            value = finding["finding_id"]
        except KeyError as exc:
            raise VerifierError("missing gold finding_id") from exc
    else:
        value = getattr(finding, "finding_id")
    if not isinstance(value, str):
        raise VerifierError("gold finding_id must be a string")
    return value


def _empty_side_error(gold_count: int) -> Reward:
    return Reward(reward=0.0, gold_count=gold_count, verifier_error=0)


def _located(finding: object) -> bool:
    """True when the finding's location is fully populated (all-or-nothing)."""
    return (
        _finding_component(finding, "path") is not None
        and _finding_component(finding, "start_line") is not None
        and _finding_component(finding, "end_line") is not None
    )


def _score_axes(
    gold: list[GoldFinding],
    candidates: list[CandidateFinding],
    matches: set[tuple[str, str]],
) -> Reward:
    """Compute reported location/severity axes over matched tp pairs.

    A pair contributes to an axis only when both sides carry that signal
    (axis-absent doctrine: never imputed, never raised on absence). Unknown
    severity values raise :class:`VerifierError` via the shared helpers.
    """
    gold_by_id = {_finding_id(g): g for g in gold}
    cand_by_id = {c.candidate_id: c for c in candidates}

    loc_tiers = {"exact": 0, "near": 0, "file": 0, "miss": 0}
    loc_credits: list[float] = []
    sev_exact = sev_within_1 = 0
    sev_distances: list[int] = []
    sev_credits: list[float] = []

    for gold_id, cand_id in matches:
        g = gold_by_id.get(gold_id)
        c = cand_by_id.get(cand_id)
        if g is None or c is None:
            raise VerifierError(f"matched pair missing finding: {gold_id!r}/{cand_id!r}")
        g_path = _finding_component(g, "path")
        g_start = _finding_component(g, "start_line")
        g_end = _finding_component(g, "end_line")
        c_path = _finding_component(c, "path")
        c_start = _finding_component(c, "start_line")
        c_end = _finding_component(c, "end_line")
        if g_path is not None and c_path is not None \
                and g_start is not None and c_start is not None \
                and g_end is not None and c_end is not None:
            tier = location_tier(
                cast("str", g_path), cast("int", g_start), cast("int", g_end),
                cast("str", c_path), cast("int", c_start), cast("int", c_end),
                LOCATION_TOLERANCE,
            )
            loc_tiers[tier] += 1
            loc_credits.append(1.0 if tier in ("exact", "near") else 0.0)
        distance = severity_distance(
            cast("str | None", _finding_component(g, "severity")),
            cast("str | None", _finding_component(c, "severity")),
        )
        if distance is not None:
            if distance == 0:
                sev_exact += 1
            if distance <= 1:
                sev_within_1 += 1
            sev_distances.append(distance)
            sev_credits.append(severity_credit(distance))

    n_loc = len(loc_credits)
    n_sev = len(sev_distances)
    return Reward(
        location_exact=loc_tiers["exact"],
        location_near=loc_tiers["near"],
        location_file=loc_tiers["file"],
        location_miss=loc_tiers["miss"],
        location_credit=sum(loc_credits) / n_loc if n_loc else 0.0,
        location_present=1 if n_loc else 0,
        severity_exact=sev_exact,
        severity_within_1=sev_within_1,
        severity_mean_distance=(sum(sev_distances) / n_sev) if n_sev else 0.0,
        severity_credit=(sum(sev_credits) / n_sev) if n_sev else 0.0,
        severity_present=1 if n_sev else 0,
    )


def score_review(
    gold: list[GoldFinding],
    candidate_artifact: dict[str, object],
    verdicts: list[Verdict],
) -> Reward:
    """Score one review against hidden gold and injected verdicts."""
    gold_count = len(gold)
    try:
        candidates = validate_candidate_artifact(candidate_artifact)
    except VerifierError:
        return _empty_side_error(gold_count)
    candidate_count = len(candidates)

    if gold_count == 0 and candidate_count == 0:
        return Reward(
            reward=1.0, precision=1.0, recall=1.0, f1=1.0,
            clean_task=1, clean_pass=1,
        )
    if gold_count == 0:
        return Reward(
            fp=candidate_count, recall=1.0,
            candidate_count=candidate_count, clean_task=1,
        )
    if candidate_count == 0:
        return Reward(
            fn=gold_count, precision=1.0, gold_count=gold_count,
        )

    gold_ids = [_finding_id(g) for g in gold]
    cand_ids = [c.candidate_id for c in candidates]
    retained = retained_edges(verdicts, gold_ids, cand_ids)
    matches = maximum_matching(retained, gold_ids, cand_ids)
    tp = len(matches)
    fp = candidate_count - tp
    fn = gold_count - tp
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 0.0 if tp == 0 else _f1(precision, recall)
    axes = _score_axes(gold, candidates, matches)
    return Reward(
        reward=f1,
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        gold_count=gold_count,
        candidate_count=candidate_count,
        location_exact=axes.location_exact,
        location_near=axes.location_near,
        location_file=axes.location_file,
        location_miss=axes.location_miss,
        location_credit=axes.location_credit,
        location_present=axes.location_present,
        severity_exact=axes.severity_exact,
        severity_within_1=axes.severity_within_1,
        severity_mean_distance=axes.severity_mean_distance,
        severity_credit=axes.severity_credit,
        severity_present=axes.severity_present,
    )


# ---------------------------------------------------------------------------
# reward / reward-details serialization
# ---------------------------------------------------------------------------


def reward_to_json(reward: Reward) -> str:
    """Serialize a Reward to a numeric-only JSON document."""
    return json.dumps(reward.to_dict())


def _candidate_id(finding: object) -> str:
    if isinstance(finding, dict):
        try:
            value = finding["candidate_id"]
        except KeyError as exc:
            raise VerifierError("missing candidate_id") from exc
    else:
        value = getattr(finding, "candidate_id")
    if not isinstance(value, str):
        raise VerifierError("candidate_id must be a string")
    return value


def reward_details(
    gold: Sequence[object],
    candidates: Sequence[object],
    verdicts: list[Verdict],
    matches: set[tuple[str, str]],
) -> dict[str, object]:
    """Capture verdicts, selected matches, and unmatched gold/candidates.

    Never embeds finding title/body/path content, source, or diffs — only ids
    and verdict reasoning.
    """
    matched_gold = {g for g, _ in matches}
    matched_candidates = {c for _, c in matches}
    gold_ids = [_finding_id(g) for g in gold]
    cand_ids = [_candidate_id(c) for c in candidates]
    return {
        "verdicts": [
            {
                "gold_id": v.gold_id,
                "candidate_id": v.candidate_id,
                "match": v.match,
                "confidence": v.confidence,
                "reasoning": v.reasoning,
            }
            for v in verdicts
        ],
        "matches": [
            {"gold_id": g, "candidate_id": c} for g, c in sorted(matches)
        ],
        "unmatched_gold": sorted(gid for gid in gold_ids if gid not in matched_gold),
        "unmatched_candidates": sorted(
            cid for cid in cand_ids if cid not in matched_candidates
        ),
    }


def reward_details_to_json(details: dict[str, object]) -> str:
    """Serialize a reward-details dict to JSON."""
    return json.dumps(details)


# ---------------------------------------------------------------------------
# corpus micro-metric aggregation
# ---------------------------------------------------------------------------


def aggregate_metrics(rows: list[dict[str, object] | None]) -> dict[str, float | int]:
    """Aggregate per-task reward JSONL rows into pooled corpus micro metrics.

    TP/FP/FN are pooled across tasks (never averaged per task). A ``None`` row
    or a row with ``verifier_error == 1`` is an unscored infrastructure
    failure: it increments ``infra_error_task_count`` and contributes nothing
    (no reward, no tp/fp/fn, no clean counts). Scored rows (``verifier_error
    == 0``) pool into ``scored_task_count``/``total_tp/fp/fn`` and the mean,
    which is over scored rows only (zero scored rows -> 1.0). Zero
    denominators evaluate to 1.0 throughout.
    """
    infra_errors = 0
    clean_correct = 0
    clean_total = 0
    scored_rewards: list[float] = []
    total_tp = total_fp = total_fn = 0

    for row in rows:
        if row is None or row.get("verifier_error") == 1:
            infra_errors += 1
            continue
        total_tp += _as_int(row["tp"])
        total_fp += _as_int(row["fp"])
        total_fn += _as_int(row["fn"])
        scored_rewards.append(_as_float(row["reward"]))
        if row.get("clean_task") == 1:
            clean_total += 1
            if _as_int(row["fp"]) == 0:
                clean_correct += 1

    task_count = len(rows)
    scored_task_count = len(scored_rewards)
    mean_task_score = (
        sum(scored_rewards) / scored_task_count if scored_task_count else 1.0
    )
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 1.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 1.0
    if total_tp == 0 and (total_tp + total_fp > 0 or total_tp + total_fn > 0):
        micro_f1 = 0.0
    else:
        micro_f1 = _f1(micro_precision, micro_recall)
    clean_accuracy = clean_correct / clean_total if clean_total else 1.0

    return {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "mean_task_score": mean_task_score,
        "clean_accuracy": clean_accuracy,
        "task_count": task_count,
        "scored_task_count": scored_task_count,
        "infra_error_task_count": infra_errors,
        "clean_task_count": clean_total,
        "total_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
    }


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise VerifierError(f"expected integer, got {value!r}")
    return value


def _as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise VerifierError(f"expected number, got {value!r}")
    return float(value)
