"""Offline shadow-calibration harness for Jev / System 1 review gates.

Replays historical targets, proposed claims, and decomposed cross-assignment
boundaries from archived Daydream runs or frozen ``LocalRecordStore`` snapshots
through :class:`~daydream.system1.System1Client`, computes calibration error
(ECE, Brier score) and threshold sweeps against confirmed findings, and
recommends safe abstain thresholds without altering live review behavior.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from daydream.dataset import LocalRecordStore
from daydream.deep.diff import iter_diff_blocks
from daydream.system1 import (
    ChoiceQuestion,
    DecisionOutcome,
    NoulQuestion,
    ScoreQuestion,
    System1Client,
    injection_detected,
    path_is_hard_denied,
    with_injection_guard,
)

# --- Shared review-triage question sets --------------------------------------

TARGET_TRIAGE_QUESTIONS = with_injection_guard({
    "non_behavioral": NoulQuestion(
        instructions=(
            "Is this file diff strictly non-behavioral (comments, documentation, formatting, "
            "lockfiles, snapshots, or pure import/identifier reordering) with no plausible "
            "runtime, security, or contract defect?"
        ),
        true_criteria=(
            "Only comments, markdown/docs, whitespace/formatting, lockfile/snapshot churn, "
            "or mechanical edits that cannot alter runtime logic, contracts, or security."
        ),
        false_criteria=(
            "Modifies executable code, control flow, data handling, types/schemas, configuration, "
            "queries, permissions, tests asserting behavior, or anything that could introduce a defect."
        ),
    ),
    "cross_file_contract": NoulQuestion(
        instructions=(
            "Does this diff modify an exported symbol signature, shared schema, public API, "
            "configuration key, or cross-file invariant?"
        ),
        true_criteria="Changes a contract, signature, schema, or shared state consumed by other files.",
        false_criteria="Changes remain purely internal to this file with no external contract change.",
    ),
    "risk_tier": ChoiceQuestion(
        instructions="Classify the defect-review risk tier of this changed file.",
        criteria={
            "mechanical": "Formatting, comments, docs, snapshots, or trivial renames.",
            "low": "Small isolated helper, simple test update, or straightforward plumbing.",
            "medium": "Meaningful business logic, error handling, state updates, or API changes.",
            "high_critical": "Auth, security, concurrency, data integrity, persistence, or subtle invariants.",
        },
    ),
    "defect_risk": ScoreQuestion(
        instructions="Rate how likely this diff is to harbor a real correctness, security, or contract defect.",
        criteria=[
            "No plausible defect risk (purely mechanical/docs)",
            "Very low risk (trivial isolated change)",
            "Moderate risk (standard behavioral change)",
            "Elevated risk (complex state, edge cases, or boundary changes)",
            "High risk (security, concurrency, data loss, or subtle cross-cutting invariants)",
        ],
    ),
})

CLAIM_TRIAGE_QUESTIONS = with_injection_guard({
    "obviously_invalid": NoulQuestion(
        instructions=(
            "Given the proposed code-review claim and its cited diff, is the claim clearly "
            "contradicted by the code, purely stylistic/speculative, or missing a concrete "
            "trigger and observable consequence?"
        ),
        true_criteria=(
            "The cited code directly refutes the claim, or the claim is a stylistic nit, "
            "speculative 'what-if', or lacks a concrete triggering input and observable bug."
        ),
        false_criteria=(
            "The claim identifies a plausible, concrete defect supported by or consistent "
            "with the cited code that merits verification."
        ),
    ),
    "needs_repo_search": NoulQuestion(
        instructions=(
            "Does confirming or refuting this claim require reading callers, definitions, "
            "configuration, or other files outside the provided diff snippet?"
        ),
        true_criteria="Truth of the claim depends on external callers, types, config, or sibling modules.",
        false_criteria="The defect or refutation is self-contained inside the provided diff snippet.",
    ),
    "severity": ChoiceQuestion(
        instructions="Classify the severity of this claim if it holds.",
        criteria={
            "high": "Breaks correctness, security, or data integrity and must be fixed before merge.",
            "medium": "Real defect with limited reach or moderate impact.",
            "low": "Minor defect or edge-case polish.",
        },
    ),
})

CROSS_ASSIGNMENT_QUESTIONS = with_injection_guard({
    "cross_boundary_needed": NoulQuestion(
        instructions=(
            "Could interactions across these separately reviewed file groups introduce a defect "
            "(broken interface, mismatched schema/config, shared state bug, or caller/callee drift) "
            "that would be invisible when reviewing each group in isolation?"
        ),
        true_criteria="Multiple groups touch coupled interfaces, shared state, schemas, or contracts.",
        false_criteria="Groups are independent mechanical, doc, or isolated module changes.",
    ),
})

TARGET_SKIP_THRESHOLDS: tuple[float, ...] = (0.50, 0.70, 0.80, 0.85, 0.90, 0.95, 0.98, 0.99)
CLAIM_REJECT_THRESHOLDS: tuple[float, ...] = (0.50, 0.70, 0.80, 0.85, 0.90, 0.95, 0.98, 0.99)
CROSS_NOOP_THRESHOLDS: tuple[float, ...] = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50)
MAX_CROSS_FILE_PROBABILITY_FOR_SKIP = 0.05
_ECE_BINS = 10


# --- Shadow dataset models & loaders -----------------------------------------


@dataclass(frozen=True)
class TargetShadowCase:
    run_id: str
    path: str
    diff: str
    has_confirmed_finding: bool
    hard_denied: bool


@dataclass(frozen=True)
class ClaimShadowCase:
    run_id: str
    uid: str
    file: str
    line: int | None
    severity: str
    claim: dict[str, Any]
    file_diff: str
    confirmed: bool


@dataclass(frozen=True)
class CrossAssignmentShadowCase:
    run_id: str
    changed_files: tuple[str, ...]
    batch_summary: dict[str, Any]
    has_cross_finding: bool


@dataclass(frozen=True)
class ShadowDataset:
    run_ids: tuple[str, ...]
    targets: tuple[TargetShadowCase, ...]
    claims: tuple[ClaimShadowCase, ...]
    cross_assignments: tuple[CrossAssignmentShadowCase, ...]


def _is_binary_block(block: str) -> bool:
    lines = block.splitlines()
    return any(line.startswith("Binary files ") for line in lines) and not any(
        line.startswith("@@ ") for line in lines
    )


def _find_committed_review(run_dir: Path) -> Path | None:
    for candidate in (run_dir / "deep" / "committed-review.json", run_dir / "committed-review.json"):
        if candidate.is_file():
            return candidate
    return None


def _find_diff_patch(run_dir: Path) -> Path | None:
    for candidate in (run_dir / "diff.patch", run_dir / "committed" / "diff.patch"):
        if candidate.is_file():
            return candidate
    return None


def discover_run_dirs(paths: Sequence[Path]) -> list[Path]:
    """Resolve explicit run directories or archive roots into sorted, unique run directories."""
    found: list[Path] = []
    seen: set[Path] = set()
    for root in paths:
        resolved = root.expanduser().resolve()
        if not resolved.is_dir():
            continue
        if _find_committed_review(resolved) is not None:
            if resolved not in seen:
                seen.add(resolved)
                found.append(resolved)
            continue
        for match in sorted(resolved.rglob("committed-review.json")):
            run_dir = match.parent.parent if match.parent.name == "deep" else match.parent
            if run_dir not in seen:
                seen.add(run_dir)
                found.append(run_dir)
    return found


def _diff_blocks_by_file(diff_text: str) -> dict[str, str]:
    blocks: dict[str, list[str]] = {}
    for path, block in iter_diff_blocks(diff_text):
        blocks.setdefault(path, []).append(block)
    return {path: "".join(parts) for path, parts in blocks.items()}


def load_shadow_cases_from_runs(paths: Sequence[Path]) -> ShadowDataset:
    """Extract target, claim, and cross-assignment calibration cases from archived review runs."""
    run_ids: list[str] = []
    targets: list[TargetShadowCase] = []
    claims: list[ClaimShadowCase] = []
    cross_assignments: list[CrossAssignmentShadowCase] = []

    for run_dir in discover_run_dirs(paths):
        review_path = _find_committed_review(run_dir)
        if review_path is None:
            continue
        sealed = json.loads(review_path.read_text(encoding="utf-8"))
        run_id = str(sealed.get("run_id") or run_dir.name)
        run_ids.append(run_id)

        diff_path = _find_diff_patch(run_dir)
        diff_by_file = _diff_blocks_by_file(diff_path.read_text(encoding="utf-8")) if diff_path is not None else {}

        sealed_findings = sealed.get("findings") or []
        confirmed_files = {
            item["payload"]["file"]
            for item in sealed_findings
            if isinstance(item, Mapping) and isinstance(item.get("payload"), Mapping) and item["payload"].get("file")
        }

        for path, block in sorted(diff_by_file.items()):
            if _is_binary_block(block):
                continue
            targets.append(TargetShadowCase(
                run_id=run_id,
                path=path,
                diff=block,
                has_confirmed_finding=path in confirmed_files,
                hard_denied=path_is_hard_denied(path),
            ))

        for claim in sealed.get("claims") or []:
            disposition = claim.get("disposition")
            if disposition not in ("confirmed", "rejected"):
                continue
            payload = claim.get("payload") or {}
            file_path = str(payload.get("file") or "")
            claims.append(ClaimShadowCase(
                run_id=run_id,
                uid=str(claim.get("uid") or ""),
                file=file_path,
                line=payload.get("line"),
                severity=str(payload.get("severity") or ""),
                claim={
                    "title": claim.get("title"),
                    "claim": payload.get("claim"),
                    "trigger": payload.get("trigger"),
                    "consequence": payload.get("consequence"),
                    "grounds": payload.get("grounds"),
                    "file": file_path,
                    "line": payload.get("line"),
                    "severity": payload.get("severity"),
                },
                file_diff=diff_by_file.get(file_path, ""),
                confirmed=disposition == "confirmed",
            ))

        policy = (sealed.get("settings") or {}).get("assignment_policy") or {}
        if policy.get("decomposed"):
            work_items = sealed.get("work") or []
            work_kind = {str(w.get("work_id")): str(w.get("kind")) for w in work_items if isinstance(w, Mapping)}
            local_batches = [
                {"work_id": w.get("work_id"), "targets": list(w.get("targets") or [])}
                for w in work_items
                if isinstance(w, Mapping) and w.get("kind") == "discovery"
            ]
            has_cross = any(
                work_kind.get(str(item.get("work_id"))) == "cross_assignment"
                for item in sealed_findings
                if isinstance(item, Mapping)
            )
            cross_assignments.append(CrossAssignmentShadowCase(
                run_id=run_id,
                changed_files=tuple(sorted(diff_by_file)),
                batch_summary={
                    "batches": local_batches,
                    "changed_files": sorted(diff_by_file),
                    "local_claim_titles": [
                        {"file": (c.get("payload") or {}).get("file"), "title": c.get("title")}
                        for c in (sealed.get("claims") or [])
                        if work_kind.get(str(c.get("work_id"))) != "cross_assignment"
                    ],
                },
                has_cross_finding=has_cross,
            ))

    return ShadowDataset(
        run_ids=tuple(run_ids),
        targets=tuple(targets),
        claims=tuple(claims),
        cross_assignments=tuple(cross_assignments),
    )


def load_shadow_cases_from_store(store_dir: Path, snapshot_id: str) -> ShadowDataset:
    """Extract target shadow cases from a frozen ``LocalRecordStore`` snapshot."""
    snapshot = LocalRecordStore(store_dir).read_snapshot(snapshot_id)
    run_ids: list[str] = []
    targets: list[TargetShadowCase] = []
    for run in snapshot.runs:
        task_ev = run.get("original_task") or {}
        findings_ev = run.get("findings") or {}
        if task_ev.get("status") != "available" or findings_ev.get("status") != "available":
            continue
        diff_text = (task_ev.get("value") or {}).get("diff")
        items = (findings_ev.get("value") or {}).get("items") or []
        if not isinstance(diff_text, str) or not diff_text.strip():
            continue
        run_id = str(run["run_id"])
        run_ids.append(run_id)
        confirmed_files = {str(item["file"]) for item in items if isinstance(item, Mapping) and item.get("file")}
        for path, block in sorted(_diff_blocks_by_file(diff_text).items()):
            if _is_binary_block(block):
                continue
            targets.append(TargetShadowCase(
                run_id=run_id,
                path=path,
                diff=block,
                has_confirmed_finding=path in confirmed_files,
                hard_denied=path_is_hard_denied(path),
            ))
    return ShadowDataset(run_ids=tuple(run_ids), targets=tuple(targets), claims=(), cross_assignments=())


# --- Calibration metrics & shadow runner -------------------------------------


@dataclass(frozen=True)
class CalibrationPoint:
    threshold: float
    gated_count: int
    total_evaluated: int
    gated_fraction: float
    missed_positives: int
    total_positives: int
    recall: float | None
    gate_precision: float | None


@dataclass(frozen=True)
class GateCalibration:
    name: str
    total_cases: int
    evaluated_cases: int
    hard_denied_cases: int
    injection_flagged_cases: int
    fallback_cases: int
    total_positives: int
    brier_score: float | None
    ece: float | None
    recall_floor: float
    recommended_threshold: float | None
    recommended_gated_fraction: float | None
    sweep: tuple[CalibrationPoint, ...]


@dataclass(frozen=True)
class CalibrationReport:
    model: str
    runs_scanned: int
    total_calls: int
    fallback_calls: int
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float | None
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    target_skip: GateCalibration
    claim_fast_reject: GateCalibration
    cross_assignment_noop: GateCalibration

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_brier_and_ece(samples: Sequence[tuple[float, bool]]) -> tuple[float | None, float | None]:
    """Compute Brier score and 10-bin Expected Calibration Error over (p_positive, is_positive) pairs."""
    n = len(samples)
    if n == 0:
        return None, None
    brier = sum((prob - float(actual)) ** 2 for prob, actual in samples) / n
    bins: list[list[tuple[float, float]]] = [[] for _ in range(_ECE_BINS)]
    for prob, actual in samples:
        index = min(_ECE_BINS - 1, max(0, int(prob * _ECE_BINS)))
        bins[index].append((prob, float(actual)))
    ece = 0.0
    for bucket in bins:
        if not bucket:
            continue
        avg_conf = sum(p for p, _ in bucket) / len(bucket)
        avg_acc = sum(y for _, y in bucket) / len(bucket)
        ece += (len(bucket) / n) * abs(avg_acc - avg_conf)
    return round(brier, 6), round(ece, 6)


def _percentile_ms(latencies_s: Sequence[float], quantile: float) -> float | None:
    if not latencies_s:
        return None
    ordered = sorted(latencies_s)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return round(ordered[index] * 1000.0, 2)


def _summarize_gate(
    *,
    name: str,
    total_cases: int,
    hard_denied_cases: int,
    injection_flagged_cases: int,
    fallback_cases: int,
    total_positives: int,
    scored: Sequence[tuple[float, bool, bool]],
    thresholds: Sequence[float],
    gate_when_below: bool,
    recall_floor: float,
    brier_samples: Sequence[tuple[float, bool]],
) -> GateCalibration:
    """Sweep thresholds over ``(score, is_positive, eligible_to_gate)`` tuples."""
    brier, ece = compute_brier_and_ece(brier_samples)
    evaluated = len(scored)
    sweep: list[CalibrationPoint] = []
    for threshold in thresholds:
        gated_count = 0
        missed = 0
        for value, is_positive, eligible in scored:
            gated = eligible and (value < threshold if gate_when_below else value >= threshold)
            if gated:
                gated_count += 1
                if is_positive:
                    missed += 1
        recall = None if total_positives == 0 else round((total_positives - missed) / total_positives, 6)
        gate_precision = None if gated_count == 0 else round((gated_count - missed) / gated_count, 6)
        sweep.append(CalibrationPoint(
            threshold=threshold,
            gated_count=gated_count,
            total_evaluated=evaluated,
            gated_fraction=round(gated_count / total_cases, 6) if total_cases > 0 else 0.0,
            missed_positives=missed,
            total_positives=total_positives,
            recall=recall,
            gate_precision=gate_precision,
        ))
    eligible_points = [
        point for point in sweep
        if point.recall is not None and point.recall >= recall_floor and point.gated_count > 0
    ]
    best = max(eligible_points, key=lambda p: (p.gated_fraction, p.recall or 0.0), default=None)
    return GateCalibration(
        name=name,
        total_cases=total_cases,
        evaluated_cases=evaluated,
        hard_denied_cases=hard_denied_cases,
        injection_flagged_cases=injection_flagged_cases,
        fallback_cases=fallback_cases,
        total_positives=total_positives,
        brier_score=brier,
        ece=ece,
        recall_floor=recall_floor,
        recommended_threshold=None if best is None else best.threshold,
        recommended_gated_fraction=None if best is None else best.gated_fraction,
        sweep=tuple(sweep),
    )


def _accumulate_usage(outcomes: Sequence[DecisionOutcome]) -> tuple[int, int, int, float | None, list[float]]:
    fallbacks = 0
    input_tokens = 0
    output_tokens = 0
    cost_total: float | None = None
    latencies: list[float] = []
    for outcome in outcomes:
        if not outcome.ok or outcome.response is None:
            fallbacks += 1
            continue
        usage = outcome.response.usage
        input_tokens += usage.input_tokens
        output_tokens += usage.output_tokens
        if usage.cost_usd is not None:
            cost_total = (cost_total or 0.0) + usage.cost_usd
        latencies.append(outcome.latency_s)
    return fallbacks, input_tokens, output_tokens, (None if cost_total is None else round(cost_total, 8)), latencies


def _calibrate_gate[T](
    *,
    name: str,
    cases: Sequence[T],
    outcomes: Sequence[DecisionOutcome],
    extract: Callable[[T, DecisionOutcome], tuple[float, float, bool, bool] | None],
    is_positive: Callable[[T], bool],
    hard_denied_cases: int,
    thresholds: Sequence[float],
    gate_when_below: bool,
    recall_floor: float,
) -> GateCalibration:
    """Collect valid outcomes through *extract* → ``(gate_prob, positive_prob, positive, eligible)``."""
    injections = 0
    fallbacks = 0
    scored: list[tuple[float, bool, bool]] = []
    brier_samples: list[tuple[float, bool]] = []
    for case, outcome in zip(cases, outcomes, strict=True):
        if not outcome.ok:
            fallbacks += 1
            continue
        extracted = extract(case, outcome)
        if extracted is None:
            fallbacks += 1
            continue
        gate_prob, pos_prob, positive, pre_eligible = extracted
        injected = injection_detected(outcome)
        if injected:
            injections += 1
        scored.append((gate_prob, positive, pre_eligible and not injected))
        brier_samples.append((pos_prob, positive))
    return _summarize_gate(
        name=name,
        total_cases=len(cases),
        hard_denied_cases=hard_denied_cases,
        injection_flagged_cases=injections,
        fallback_cases=fallbacks,
        total_positives=sum(is_positive(case) for case in cases),
        scored=scored,
        thresholds=thresholds,
        gate_when_below=gate_when_below,
        recall_floor=recall_floor,
        brier_samples=brier_samples,
    )


async def run_shadow_calibration(
    dataset: ShadowDataset,
    client: System1Client,
    *,
    target_recall_floor: float = 0.99,
    claim_recall_floor: float = 0.98,
    cross_recall_floor: float = 0.99,
) -> CalibrationReport:
    """Replay *dataset* through *client* in shadow mode and compute calibrated gate recommendations."""
    target_outcomes = await client.evaluate_many(
        [({"path": case.path, "diff": case.diff}, TARGET_TRIAGE_QUESTIONS) for case in dataset.targets],
        record_usage=False,
    )
    claim_outcomes = await client.evaluate_many(
        [({"claim": case.claim, "file_diff": case.file_diff}, CLAIM_TRIAGE_QUESTIONS) for case in dataset.claims],
        record_usage=False,
    )
    cross_outcomes = await client.evaluate_many(
        [(case.batch_summary, CROSS_ASSIGNMENT_QUESTIONS) for case in dataset.cross_assignments],
        record_usage=False,
    )

    def _extract_target(case: TargetShadowCase, outcome: DecisionOutcome) -> tuple[float, float, bool, bool] | None:
        p_no_risk = outcome.noul("non_behavioral")
        p_cross = outcome.noul("cross_file_contract")
        if p_no_risk is None or p_cross is None:
            return None
        return (
            p_no_risk,
            1.0 - p_no_risk,
            case.has_confirmed_finding,
            (not case.hard_denied) and p_cross <= MAX_CROSS_FILE_PROBABILITY_FOR_SKIP,
        )

    def _extract_claim(case: ClaimShadowCase, outcome: DecisionOutcome) -> tuple[float, float, bool, bool] | None:
        p_invalid = outcome.noul("obviously_invalid")
        if p_invalid is None:
            return None
        return (p_invalid, 1.0 - p_invalid, case.confirmed, True)

    def _extract_cross(
        case: CrossAssignmentShadowCase, outcome: DecisionOutcome,
    ) -> tuple[float, float, bool, bool] | None:
        p_needed = outcome.noul("cross_boundary_needed")
        if p_needed is None:
            return None
        return (p_needed, p_needed, case.has_cross_finding, True)

    target_gate = _calibrate_gate(
        name="target_skip",
        cases=dataset.targets,
        outcomes=target_outcomes,
        extract=_extract_target,
        is_positive=lambda case: case.has_confirmed_finding,
        hard_denied_cases=sum(case.hard_denied for case in dataset.targets),
        thresholds=TARGET_SKIP_THRESHOLDS,
        gate_when_below=False,
        recall_floor=target_recall_floor,
    )
    claim_gate = _calibrate_gate(
        name="claim_fast_reject",
        cases=dataset.claims,
        outcomes=claim_outcomes,
        extract=_extract_claim,
        is_positive=lambda case: case.confirmed,
        hard_denied_cases=0,
        thresholds=CLAIM_REJECT_THRESHOLDS,
        gate_when_below=False,
        recall_floor=claim_recall_floor,
    )
    cross_gate = _calibrate_gate(
        name="cross_assignment_noop",
        cases=dataset.cross_assignments,
        outcomes=cross_outcomes,
        extract=_extract_cross,
        is_positive=lambda case: case.has_cross_finding,
        hard_denied_cases=0,
        thresholds=CROSS_NOOP_THRESHOLDS,
        gate_when_below=True,
        recall_floor=cross_recall_floor,
    )

    all_outcomes = [*target_outcomes, *claim_outcomes, *cross_outcomes]
    fallbacks, in_tok, out_tok, total_cost, latencies = _accumulate_usage(all_outcomes)
    return CalibrationReport(
        model=client.config.model,
        runs_scanned=len(dataset.run_ids),
        total_calls=len(all_outcomes),
        fallback_calls=fallbacks,
        total_input_tokens=in_tok,
        total_output_tokens=out_tok,
        total_cost_usd=total_cost,
        p50_latency_ms=_percentile_ms(latencies, 0.50),
        p95_latency_ms=_percentile_ms(latencies, 0.95),
        target_skip=target_gate,
        claim_fast_reject=claim_gate,
        cross_assignment_noop=cross_gate,
    )
