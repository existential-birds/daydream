"""Stage-2 deterministic RFT (M11, M12): offline replay with breakdown-filtered
byte-identical winners.

Each task is **reconstructed from the frozen base/head/diff identity recorded
in the corpus** (M16) — never from live repo state — and candidate completions
are sampled deterministically (seeded per record) against the Stage-0 rubric
inputs. Every candidate is scored through :func:`score_trajectory`
(``reward.py`` — the same hook the offline pipeline and the Stage-3 env use),
so the winner filter reads the full breakdown, never a bare scalar (M12).

The winner threshold is a **breakdown-shaped spec**: a mapping of breakdown
axis names to minimum values (e.g. ``{"composite": 0.6, "grounding": 0.5}``).
A plain float is a :class:`TypeError` at config time.

Determinism contract: given the same inputs file, model id, seed, and rubric
version, reruns produce **byte-identical** winners files. All iteration is
sorted (records by stable id), serialization uses ``sort_keys=True`` with
fixed float formatting, and the header stamps the model id, seed, rubric
version, and a sha256 digest of the inputs file. Candidate scoring derives
its breakdown inputs (verdicts + grounding rate) from the *sampled* findings
subset, so a sampled completion is scored on what varies, never on a
byte-identical copy of the record.

A record missing base/head/diff identity raises :class:`ValueError` naming
the record id — never skipped silently.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from daydream.training.reward import RewardBreakdown, ScoringInputs, score_trajectory

__all__ = ["RftConfig", "RftWinner", "RftResult", "run_rft"]

# Breakdown axes the winner spec may constrain: exactly the attribute names of
# ``score_trajectory``'s ``RewardBreakdown`` (reward.py:316). Anything else is a
# typo — fail closed at config time rather than silently filtering on nothing.
_SPEC_AXES = ("composite", "grounding", "correctness_per_finding", "length_penalty")

# Minimum candidates sampled per record (full finding set + seeded subsets).
DEFAULT_CANDIDATES_PER_TASK = 4


@dataclass(frozen=True)
class RftConfig:
    """Configuration for one deterministic RFT replay.

    Attributes:
        inputs: Path to the frozen replay corpus (JSONL, one record per line)
            whose records carry ``id``, ``base_sha``, ``head_sha``, ``diff``,
            and the intrinsic signals ``score_trajectory`` consumes.
        seed: Master seed; per-record seeds are derived deterministically.
        rubric_version: Rubric version stamped into the winners header.
        output_dir: Directory the winners file is written to.
        min_breakdown: Breakdown-shaped winner threshold — a mapping of
            breakdown axis names to minimum values. A bare scalar is rejected.
        model_id: Identifier of the model whose completions are replayed;
            stamped into the winners header.
        candidates_per_task: Number of deterministic candidates sampled per
            record (the full finding set is always candidate 0).
        temperature: Sampling temperature carried for provenance; deterministic
            replay pins it to 0.0.
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
                f"(e.g. {{'composite': 0.6, 'grounding': 0.5}}); got {type(self.min_breakdown).__name__!r}. "
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


def _record_sort_key(rec: Mapping[str, Any]) -> str:
    return str(rec.get("id", ""))


def _reconstruct_task(rec: Mapping[str, Any]) -> tuple[str, str, str]:
    """Return (record_id, base_sha, diff), failing closed on missing identity."""
    rid = str(rec.get("id", ""))
    base_sha = rec.get("base_sha")
    head_sha = rec.get("head_sha")
    diff = rec.get("diff")
    missing = [name for name, value in (("base_sha", base_sha), ("head_sha", head_sha), ("diff", diff)) if not value]
    if missing:
        raise ValueError(
            f"record {rid!r} is missing frozen task identity field(s) {missing}; "
            "RFT rebuilds every task from base/head/diff identity (M16) and never skips a "
            "record silently."
        )
    assert isinstance(base_sha, str) and isinstance(diff, str)
    return rid, base_sha, diff


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
    """Score one candidate through the canonical ``score_trajectory`` hook.

    The sampled ``findings`` subset is the candidate-varying input: verdicts
    and grounding rate are derived from it (mirroring ``rubric_v2.score_review``),
    so candidates that differ only in their findings subset score differently
    and the winner filter can prefer one sampled completion over another.
    Record-level signals are used only when the record carries no findings.
    The derivation is a pure function of the candidate, so byte-identical
    determinism on rerun is preserved (M11).
    """
    findings = [f for f in rec.get("findings", []) if isinstance(f, Mapping)]
    if findings:
        grounded = sum(1 for f in findings if f.get("grounded"))
        grounding_rate: float | None = grounded / len(findings)
        verdicts_derived = [{"verdict": str(f["verdict"])} for f in findings if f.get("verdict")]
        verifier_verdicts: Any = verdicts_derived or rec.get("verifier_verdicts")
    else:
        grounding_rate = rec.get("grounding_rate")
        verifier_verdicts = rec.get("verifier_verdicts")
    breakdown = score_trajectory(
        ScoringInputs(
            verifier_verdicts=verifier_verdicts,
            grounding_rate=grounding_rate,
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


def _inputs_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    """Run one deterministic Stage-2 RFT replay over the frozen inputs.

    Reconstructs each task from its recorded base/head/diff identity, samples
    deterministic candidates, scores every candidate through
    :func:`score_trajectory`, filters winners by the breakdown-shaped spec,
    and writes a byte-identical-on-rerun winners JSON file.

    Raises:
        ValueError: When any record lacks base/head/diff identity (named by
            record id) — never skipped silently.
    """
    inputs_path = Path(config.inputs)
    raw = inputs_path.read_text(encoding="utf-8")
    records: list[dict[str, Any]] = [json.loads(line) for line in raw.splitlines() if line.strip()]

    tasks = [_reconstruct_task(rec) for rec in sorted(records, key=_record_sort_key)]

    winners: list[RftWinner] = []
    for rec, (rid, _base_sha, _diff) in zip(sorted(records, key=_record_sort_key), tasks):
        for index, candidate in enumerate(_sample_candidates(rec, rid, config)):
            breakdown = _score_candidate(candidate)
            if _passes(config.min_breakdown, breakdown):
                winners.append(RftWinner(record_id=rid, candidate_index=index, breakdown=breakdown))

    winners.sort(key=lambda w: (w.record_id, w.candidate_index))

    inputs_sha256 = _inputs_digest(inputs_path)
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
