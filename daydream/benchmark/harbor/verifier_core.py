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


# ---------------------------------------------------------------------------
# reward + score_review
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Reward:
    """The 12-field §10 per-task reward output."""

    reward: float
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    gold_count: int
    candidate_count: int
    clean_task: int
    clean_pass: int
    verifier_error: int

    def to_dict(self) -> dict[str, float | int]:
        """Numeric-only dict with exactly the 12 §10 keys."""
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
        }


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 1.0
    return 2 * precision * recall / (precision + recall)


def _finding_id(finding: object) -> str:
    """Read a gold finding's id from either a GoldFinding or a raw dict."""
    if isinstance(finding, dict):
        try:
            return finding["finding_id"]
        except KeyError as exc:
            raise VerifierError("missing gold finding_id") from exc
    return getattr(finding, "finding_id")


def _empty_side_error(gold_count: int) -> Reward:
    return Reward(
        reward=0.0,
        tp=0,
        fp=0,
        fn=0,
        precision=0.0,
        recall=0.0,
        f1=0.0,
        gold_count=gold_count,
        candidate_count=0,
        clean_task=0,
        clean_pass=0,
        verifier_error=1,
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
            reward=1.0,
            tp=0,
            fp=0,
            fn=0,
            precision=1.0,
            recall=1.0,
            f1=1.0,
            gold_count=0,
            candidate_count=0,
            clean_task=1,
            clean_pass=1,
            verifier_error=0,
        )
    if gold_count == 0:
        return Reward(
            reward=0.0,
            tp=0,
            fp=candidate_count,
            fn=0,
            precision=0.0,
            recall=1.0,
            f1=0.0,
            gold_count=0,
            candidate_count=candidate_count,
            clean_task=1,
            clean_pass=0,
            verifier_error=0,
        )
    if candidate_count == 0:
        return Reward(
            reward=0.0,
            tp=0,
            fp=0,
            fn=gold_count,
            precision=1.0,
            recall=0.0,
            f1=0.0,
            gold_count=gold_count,
            candidate_count=0,
            clean_task=0,
            clean_pass=0,
            verifier_error=0,
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
    f1 = _f1(precision, recall)
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
        clean_task=0,
        clean_pass=0,
        verifier_error=0,
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
            return finding["candidate_id"]
        except KeyError as exc:
            raise VerifierError("missing candidate_id") from exc
    return getattr(finding, "candidate_id")


def reward_details(
    gold: list[object],
    candidates: list[object],
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
        "unmatched_gold": [gid for gid in gold_ids if gid not in matched_gold],
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
    or a row with ``verifier_error == 1`` is a failed task: it contributes
    reward 0 to the mean, zero counts, and increments ``failed_task_count``.
    Zero denominators evaluate to 1.0 throughout.
    """
    failed = 0
    clean_correct = 0
    clean_total = 0
    rewards: list[float] = []
    total_tp = total_fp = total_fn = 0

    for row in rows:
        if row is None or row.get("verifier_error") == 1:
            failed += 1
            rewards.append(0.0)
            continue
        total_tp += _as_int(row["tp"])
        total_fp += _as_int(row["fp"])
        total_fn += _as_int(row["fn"])
        rewards.append(_as_float(row["reward"]))
        if row.get("clean_task") == 1:
            clean_total += 1
            if _as_int(row["fp"]) == 0:
                clean_correct += 1

    task_count = len(rows)
    mean_task_score = sum(rewards) / task_count if task_count else 1.0
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 1.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 1.0
    if micro_precision + micro_recall == 0:
        micro_f1 = 1.0
    else:
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
    clean_accuracy = clean_correct / clean_total if clean_total else 1.0

    return {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "mean_task_score": mean_task_score,
        "clean_accuracy": clean_accuracy,
        "task_count": task_count,
        "clean_task_count": clean_total,
        "failed_task_count": failed,
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
