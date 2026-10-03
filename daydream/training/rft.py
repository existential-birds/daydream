"""Replay frozen task identities and write deterministic, breakdown-filtered winners.

Given identical input bytes, model, seed, and rubric, winners are byte-identical:
records and winners are sorted, floats use fixed formatting, and the header
stamps input digest and sampling settings. Candidate findings drive their own
scores. Missing identity raises with the record id; thresholds name axes and
cannot be a bare scalar.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from daydream.training.reward import RewardBreakdown, ScoringInputs, score_trajectory

__all__ = ["RftConfig", "RftWinner", "RftResult", "run_rft", "validate_full_sha"]

# Full-SHA contract: RFT rebuilds every task from full-length hex commit
# SHAs (Req 7, the #714 rebase target). A short, empty, or non-hex value is
# refused with a ValueError naming the record id and field — shared with the
# coordinator's ``_rft_rows`` so Stage 2 and the replay fail closed alike.
# Longer sha256-style content-address stamps are tolerated, matching the
# coordinator's pre-existing acceptance.
_FULL_SHA_RE = re.compile(r"[0-9a-f]{40,}")


def validate_full_sha(record_id: str, field: str, value: object) -> str:
    """Require lowercase hexadecimal identity of at least 40 characters.

    Longer content-address stamps are accepted; invalid values raise ValueError
    naming the record and field without a fabricated fallback.
    """
    if not isinstance(value, str) or not _FULL_SHA_RE.fullmatch(value):
        raise ValueError(
            f"record {record_id!r} carries an invalid {field} {value!r}: "
            "RFT rebuilds every task from a full 40-hex commit sha and never "
            "replays against a truncated or non-hex identity"
        )
    return value

# Breakdown axes the winner spec may constrain: exactly the attribute names of
# ``score_trajectory``'s ``RewardBreakdown`` (reward.py:316). Anything else is a
# typo — fail closed at config time rather than silently filtering on nothing.
_SPEC_AXES = ("composite", "correctness_per_finding", "length_penalty")

# Minimum candidates sampled per record (full finding set + seeded subsets).
DEFAULT_CANDIDATES_PER_TASK = 4


@dataclass(frozen=True)
class RftConfig:
    """Frozen replay inputs, sampling settings, and per-axis winner thresholds.

    Candidate zero always contains every finding. model_id, seed, rubric_version,
    and temperature are stamped into the output; thresholds have no scalar form.
    """

    inputs: str | Path
    seed: int
    rubric_version: str
    output_dir: str | Path
    min_breakdown: Mapping[str, float] = field(default_factory=lambda: {"composite": 0.0})
    model_id: str = "unknown"
    candidates_per_task: int = DEFAULT_CANDIDATES_PER_TASK
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if isinstance(self.min_breakdown, (int, float, bool)) or not isinstance(self.min_breakdown, Mapping):
            raise TypeError(
                "min_breakdown must be a breakdown-shaped mapping of axis names to minimums "
                "(e.g. {'composite': 0.6, 'correctness_per_finding': 0.5}); "
                f"got {type(self.min_breakdown).__name__!r}. "
                "A bare scalar cannot name the axes it constrains (M12)."
            )
        unknown = sorted(set(self.min_breakdown) - set(_SPEC_AXES))
        if unknown:
            raise TypeError(
                f"min_breakdown names unknown axis(es) {unknown}; allowed axes: {', '.join(_SPEC_AXES)}."
            )
        if not 1 <= self.candidates_per_task:
            raise ValueError(f"candidates_per_task must be >= 1 (got {self.candidates_per_task!r}).")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError(f"temperature must be in [0, 2] (got {self.temperature!r}).")


@dataclass(frozen=True)
class RftWinner:
    """One winning candidate: record identity, candidate index, breakdown."""

    record_id: str
    candidate_index: int
    breakdown: RewardBreakdown

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "candidate_index": self.candidate_index,
            "breakdown": self.breakdown.to_dict(),
        }


@dataclass(frozen=True)
class RftResult:
    """Outcome of one replay: the winners file path and parsed winners."""

    winners_path: Path
    records: list[RftWinner]
    inputs_sha256: str


def _reconstruct_task(rec: Mapping[str, Any]) -> str:
    """Validate repo/base/head/diff identity and full SHAs before replaying a record."""
    rid = str(rec.get("id", ""))
    repo_slug = rec.get("repo_slug")
    base_sha = rec.get("base_sha")
    head_sha = rec.get("head_sha")
    diff = rec.get("diff")
    missing = [
        name
        for name, value in (("repo_slug", repo_slug), ("base_sha", base_sha), ("head_sha", head_sha), ("diff", diff))
        if not value
    ]
    if missing:
        raise ValueError(
            f"record {rid!r} is missing frozen task identity field(s) {missing}; "
            "RFT rebuilds every task from repo/base/head/diff identity (M16) and never skips a "
            "record silently."
        )
    for sha_field in ("base_sha", "head_sha"):
        validate_full_sha(rid, sha_field, rec.get(sha_field))
    return rid


def _sample_candidates(rec: Mapping[str, Any], rid: str, cfg: RftConfig) -> list[dict[str, Any]]:
    """Deterministically derive candidate completions from the frozen record.

    Candidate 0 is the full capture; the rest are seeded sub-selections of the
    findings, replaying what a model at temperature 0 would emit against the
    reconstructed task. Identical inputs + seed ⇒ identical candidates.
    """
    findings = [f for f in rec.get("findings", []) if isinstance(f, Mapping)]
    candidates = [dict(rec, findings=list(findings))]
    rng = random.Random(f"{cfg.seed}:{cfg.model_id}:{rid}")
    for _ in range(cfg.candidates_per_task - 1):
        if len(findings) > 1:
            keep = [f for f in findings if rng.random() < 0.75] or [findings[0]]
        else:
            keep = list(findings)
        candidates.append(dict(rec, findings=keep))
    return candidates


def _score_candidate(rec: Mapping[str, Any]) -> RewardBreakdown:
    """Score candidate findings through the canonical intrinsic-reward hook.

    Derive verdicts from sampled findings when available; otherwise retain record
    signals. Candidate variation can therefore affect winner selection.
    """
    findings = [f for f in rec.get("findings", []) if isinstance(f, Mapping)]
    if findings:
        verdicts_derived = [{"verdict": str(f["verdict"])} for f in findings if f.get("verdict")]
        verifier_verdicts: Any = verdicts_derived or rec.get("verifier_verdicts")
    else:
        verifier_verdicts = rec.get("verifier_verdicts")
    breakdown = score_trajectory(
        ScoringInputs(
            verifier_verdicts=verifier_verdicts,
            format_valid=bool(rec.get("format_valid", False)),
            length=rec.get("length"),
        )
    )
    assert isinstance(breakdown, RewardBreakdown)
    return breakdown


def _passes(spec: Mapping[str, float], breakdown: RewardBreakdown) -> bool:
    """Evaluate the breakdown-shaped threshold spec against one breakdown.

    Scalar axes are compared directly. ``correctness_per_finding`` is a list
    of mapped per-finding verdicts: the minimum applies to *every* verdict,
    and an empty list fails (no correctness evidence cannot clear a floor).
    """
    for axis, minimum in spec.items():
        value = getattr(breakdown, axis)
        if isinstance(value, list):
            if not value or any(float(v) < float(minimum) for v in value):
                return False
        elif value is None or float(value) < float(minimum):
            return False
    return True


def _fixed(value: Any) -> Any:
    """Fixed float formatting so JSON serialization is byte-stable."""
    if isinstance(value, float):
        return f"{value:.6f}"
    if isinstance(value, dict):
        return {k: _fixed(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_fixed(v) for v in value]
    return value


def run_rft(config: RftConfig) -> RftResult:
    """Validate frozen task identities, sample/score candidates, and write stable winners.

    Missing or malformed identity raises ValueError naming the record and field.
    """
    inputs_path = Path(config.inputs)
    raw = inputs_path.read_bytes()
    records: list[dict[str, Any]] = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]

    winners: list[RftWinner] = []
    for rec in sorted(records, key=lambda record: str(record.get("id", ""))):
        rid = _reconstruct_task(rec)
        for index, candidate in enumerate(_sample_candidates(rec, rid, config)):
            breakdown = _score_candidate(candidate)
            if _passes(config.min_breakdown, breakdown):
                winners.append(RftWinner(record_id=rid, candidate_index=index, breakdown=breakdown))

    winners.sort(key=lambda w: (w.record_id, w.candidate_index))

    inputs_sha256 = hashlib.sha256(raw).hexdigest()
    payload: dict[str, Any] = {
        "header": {
            "model_id": config.model_id,
            "seed": config.seed,
            "rubric_version": config.rubric_version,
            "temperature": config.temperature,
            "candidates_per_task": config.candidates_per_task,
            "min_breakdown": dict(config.min_breakdown),
            "inputs_sha256": inputs_sha256,
        },
        "winners": [w.to_dict() for w in winners],
    }

    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    winners_path = out_dir / "rft-winners.json"
    winners_path.write_text(json.dumps(_fixed(payload), sort_keys=True, indent=2) + "\n", encoding="utf-8")

    return RftResult(winners_path=winners_path, records=winners, inputs_sha256=inputs_sha256)
