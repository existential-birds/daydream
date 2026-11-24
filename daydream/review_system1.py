"""Committed-review System 1 pre-dispatch ranking, supporting-file discovery, and triage questions.

Single authoritative home for review-domain Jev / System 1 question sets so live
committed reviews and offline shadow calibration evaluate identical rubrics.
All hints are strictly advisory, confined to repository paths, and fail open to
existing deterministic order when System 1 is disabled, unavailable, or times out.
"""

from __future__ import annotations

import codecs
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from itertools import combinations, islice
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import anyio

from daydream.deep.dependency import _LANG_BY_EXT, _resolve_import
from daydream.prompt_budget import truncate_utf8_to_budget
from daydream.repository_paths import git_observed_path_is_confined
from daydream.system1 import (
    ChoiceQuestion,
    DecisionOutcome,
    DecisionQuestion,
    DecisionUsage,
    NoulQuestion,
    ScoreQuestion,
    System1Client,
    System1Config,
    injection_detected,
    path_is_hard_denied,
    with_injection_guard,
)
from daydream.trajectory import DaydreamPhase, get_current_recorder
from daydream.tree_sitter_index import extract_imports, get_parser
from daydream.tree_sitter_index.runtime import _query_for_language

# --- Shared review question sets (used by live dispatch and offline calibration) ---

TARGET_TRIAGE_QUESTIONS: dict[str, DecisionQuestion] = with_injection_guard({
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

HISTORICAL_CONTEXT_QUESTIONS: dict[str, DecisionQuestion] = with_injection_guard({
    "substantive_intent": NoulQuestion(
        instructions=(
            "Does this PR title, body, or author description state non-obvious behavioral "
            "requirements, edge-case constraints, invariants, compatibility guarantees, or "
            "intentional design tradeoffs that matter when judging whether code changes are correct?"
        ),
        true_criteria=(
            "Explains specific requirements, constraints, invariants, backwards-compatibility "
            "rules, or non-obvious tradeoffs that affect defect evaluation."
        ),
        false_criteria=(
            "Empty, boilerplate PR template/checklist, generic title/summary restating file names, "
            "or prose containing no correctness-relevant constraints."
        ),
    ),
})

SUPPORTING_FILE_QUESTIONS: dict[str, DecisionQuestion] = with_injection_guard({
    "relevant_to_diff": NoulQuestion(
        instructions=(
            "Is inspecting this supporting repository file likely necessary to verify the "
            "correctness, invariants, or contracts of the changed diff?"
        ),
        true_criteria=(
            "Directly defines, consumes, configures, or tests interfaces, types, state, "
            "or invariants modified by the diff."
        ),
        false_criteria=(
            "Unrelated or generic utility whose contents are unlikely to affect whether the diff has a bug."
        ),
    ),
    "relevance_score": ScoreQuestion(
        instructions="Rate how useful reading this supporting file is for verifying the changed diff.",
        criteria=[
            "Not relevant to verifying the diff",
            "Tangentially related background only",
            "Useful context on conventions or sibling behavior",
            "Directly defines, calls, or tests modified interfaces or invariants",
        ],
    ),
})

CLAIM_TRIAGE_QUESTIONS: dict[str, DecisionQuestion] = with_injection_guard({
    "obviously_invalid": NoulQuestion(
        instructions=(
            "Is this proposed code-review claim clearly invalid, purely stylistic/speculative, "
            "lacking a concrete trigger/consequence, or directly contradicted by the cited diff?"
        ),
        true_criteria=(
            "Clearly contradicted by the cited code, purely stylistic preference, speculative "
            "without concrete failure mode, or trivial noise."
        ),
        false_criteria=(
            "Plausible correctness, security, data-integrity, or contract bug that warrants "
            "independent verification."
        ),
    ),
    "needs_repo_search": NoulQuestion(
        instructions=(
            "Does verifying or refuting this claim require reading callers, definitions, or files "
            "outside the cited diff hunk and enclosing function?"
        ),
        true_criteria="Requires checking external callers, callees, config, or cross-file invariants.",
        false_criteria="Self-contained within the cited hunk and enclosing symbol.",
    ),
    "severity": ChoiceQuestion(
        instructions="Classify the calibrated severity of this claim under Daydream's consequence rubric.",
        criteria={
            "high": "Security vulnerability, data loss/corruption, crash, or broken critical contract.",
            "medium": "Real functional defect, edge-case bug, or degraded reliability under realistic inputs.",
            "low": "Minor edge case, localized non-critical bug, or low-impact defect.",
        },
    ),
})

CLAIM_DUPLICATE_QUESTIONS: dict[str, DecisionQuestion] = with_injection_guard({
    "is_duplicate": NoulQuestion(
        instructions=(
            "Do these two proposed code-review claims describe the same underlying defect or "
            "root cause such that one challenge session should evaluate and deduplicate them together?"
        ),
        true_criteria="Both claims target the same bug, failure mode, or overlapping root cause.",
        false_criteria="The two claims describe independent defects with distinct failure modes.",
    ),
})

CROSS_ASSIGNMENT_QUESTIONS: dict[str, DecisionQuestion] = with_injection_guard({
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

BATCH_PAIR_COUPLING_QUESTIONS: dict[str, DecisionQuestion] = with_injection_guard({
    "coupled_boundary": NoulQuestion(
        instructions=(
            "Do the changes in Batch A and Batch B share an interface, invariant, schema, "
            "configuration contract, or shared state boundary that should be cross-checked together?"
        ),
        true_criteria="Batch A and Batch B modify interacting APIs, types, callers/callees, or shared state.",
        false_criteria="Batch A and Batch B are independent changes with no meaningful cross-batch interaction.",
    ),
})

MAX_SUPPORTING_CANDIDATES = 12
MAX_SUPPORTING_HINTS = 3
SUPPORTING_PREVIEW_MAX_BYTES = 2048
GROUP_DIFF_EXCERPT_MAX_BYTES = 8192
HISTORICAL_EXCERPT_MAX_BYTES = 16384
SUPPORTING_RELEVANCE_THRESHOLD = 0.40
HISTORICAL_SUBSTANTIVE_THRESHOLD = 0.35
# Hard-denied or injection-flagged targets outrank normal 0..4 defect_risk scores.
PRIORITY_MAX_RISK = 5.0
# Neutral-high priority for an unscored target when siblings succeeded, so a transient error never demotes it.
UNSCORED_NEUTRAL_RISK = 3.0
_CANDIDATE_SCAN_WALL_BUDGET_S = 2.0

RiskTier = Literal["mechanical", "low", "medium", "high_critical"]
RISK_TIER_WEIGHTS: Mapping[RiskTier, float] = {
    "mechanical": 1.0,
    "low": 1.5,
    "medium": 3.0,
    "high_critical": 5.0,
}
_RISK_TIERS: Mapping[str, RiskTier] = {str(tier): tier for tier in RISK_TIER_WEIGHTS}
FAST_TIER_CROSS_FILE_MAX = 0.25
FAST_TIER_RISK_SCORE_MAX = 2.0
SKIP_TIER_RISK_SCORE_MAX = 1.0


@dataclass(frozen=True)
class TargetTriage:
    """Calibrated System 1 triage outcome for one changed review target."""

    target_id: str
    file_path: str
    risk_score: float
    risk_tier: RiskTier
    p_non_behavioral: float | None = None
    p_cross_file: float | None = None
    hard_denied: bool = False
    injection_flagged: bool = False
    fallback: bool = False
    model: str | None = None

    def qualifies_for_fast_tier(self) -> bool:
        """Whether this target may safely use a fast/low-effort System 2 discovery tier."""
        if self.hard_denied or self.injection_flagged or self.fallback:
            return False
        if self.risk_tier not in ("mechanical", "low"):
            return False
        if self.risk_score >= FAST_TIER_RISK_SCORE_MAX:
            return False
        return self.p_cross_file is None or self.p_cross_file <= FAST_TIER_CROSS_FILE_MAX

    def qualifies_for_skip(self, *, skip_threshold: float, cross_file_max: float) -> bool:
        """Whether this target satisfies the strict opt-in Mode 2 non-behavioral fast-path."""
        if self.hard_denied or self.injection_flagged or self.fallback:
            return False
        if path_is_hard_denied(self.file_path):
            return False
        if self.p_non_behavioral is None or self.p_non_behavioral < skip_threshold:
            return False
        if self.p_cross_file is None or self.p_cross_file > cross_file_max:
            return False
        if self.risk_tier != "mechanical" or self.risk_score > SKIP_TIER_RISK_SCORE_MAX:
            return False
        return True


def allocate_weighted_calls(total: int, weights: Sequence[float], *, floor: int = 2) -> list[int]:
    """Allocate *total* tool calls proportionally to *weights* with Hamilton's largest-remainder method.

    Guarantees exact sum == *total*, respects *floor* per slot whenever ``total >= floor * len(weights)``,
    and falls back cleanly to uniform division when weights are uniform, invalid, or budget is tight.
    """
    count = len(weights)
    if count == 0:
        return []
    if count == 1:
        return [total]

    def _uniform() -> list[int]:
        base, extra = divmod(total, count)
        return [base + (index < extra) for index in range(count)]

    if floor < 0 or total <= floor * count:
        return _uniform()
    cleaned = [float(w) for w in weights]
    if any(w <= 0 for w in cleaned) or len(set(cleaned)) == 1:
        return _uniform()

    weight_sum = sum(cleaned)
    surplus = total - floor * count
    exact = [surplus * (w / weight_sum) for w in cleaned]
    shares = [floor + int(val) for val in exact]
    remainder = total - sum(shares)
    order = sorted(
        range(count),
        key=lambda idx: (-(exact[idx] - int(exact[idx])), -cleaned[idx], idx),
    )
    for idx in order[:remainder]:
        shares[idx] += 1
    return shares


def resolve_fast_phase_spec(
    config: Any,
    system1_config: System1Config,
    phase: Literal["review", "arbiter"],
    *,
    hosted: bool,
) -> tuple[str, str | None, str | None] | None:
    """Resolve optional fast ``(backend, model, effort)`` for discovery or arbiter System 1 tier routing."""
    if not system1_config.enabled:
        return None
    from daydream.run_config import resolved_backend_name, resolved_model, resolved_reasoning_effort

    override_backend, override_model, override_effort = (
        (system1_config.fast_review_backend, system1_config.fast_review_model, system1_config.fast_review_effort)
        if phase == "review"
        else (
            system1_config.fast_arbiter_backend,
            system1_config.fast_arbiter_model,
            system1_config.fast_arbiter_effort,
        )
    )
    file_fast: Mapping[str, str] = {}
    if not hosted and config.file_config is not None:
        raw_fast = config.file_config.phases.get(f"{phase}_fast")
        if isinstance(raw_fast, Mapping):
            file_fast = raw_fast
    eff_backend = override_backend or file_fast.get("backend")
    eff_model = override_model or file_fast.get("model")
    eff_effort = override_effort or file_fast.get("reasoning_effort")
    if not any((eff_backend, eff_model, eff_effort)):
        return None
    base_backend = resolved_backend_name(config, phase)
    backend_name = eff_backend or base_backend
    model = eff_model or (resolved_model(config, phase) if backend_name == base_backend else None)
    effort = eff_effort or resolved_reasoning_effort(config, phase)
    return backend_name, model, effort


@dataclass(frozen=True)
class SupportingCandidate:
    """One confined unchanged repository file candidate for System 1 relevance scoring."""

    path: str
    relation: Literal["import", "test"]
    preview: str


def _conventional_test_paths(file_path: str) -> tuple[str, ...]:
    """Generate conventional test-file counterpart paths for a changed source file."""
    posix = PurePosixPath(file_path)
    suffix = posix.suffix.lower()
    stem = posix.stem
    parent = "" if str(posix.parent) == "." else str(posix.parent)
    candidates: list[str] = []

    def _join(base: str, name: str) -> str:
        return f"{base}/{name}" if base else name

    if suffix == ".py" and not stem.startswith("test_") and not stem.endswith("_test"):
        candidates.extend((
            f"tests/test_{stem}.py",
            _join(parent, f"test_{stem}.py"),
            _join(parent, f"tests/test_{stem}.py"),
        ))
        if parent and "/" in parent:
            sub = parent.split("/", 1)[1]
            candidates.append(f"tests/{sub}/test_{stem}.py")
    elif suffix in (".ts", ".tsx", ".js", ".jsx") and not stem.endswith((".test", ".spec")):
        candidates.extend((
            _join(parent, f"{stem}.test{suffix}"),
            _join(parent, f"{stem}.spec{suffix}"),
            _join(parent, f"__tests__/{stem}.test{suffix}"),
        ))
    elif suffix == ".go" and not stem.endswith("_test"):
        candidates.append(_join(parent, f"{stem}_test.go"))
    elif suffix == ".rs" and not stem.endswith("_test"):
        candidates.extend((f"tests/{stem}_test.rs", f"tests/{stem}.rs"))
    return tuple(dict.fromkeys(candidates))


def _direct_import_paths(repo: Path, changed_file: str) -> list[str]:
    """Resolve direct import edges out of *changed_file* using tree-sitter where supported."""
    lang_id = _LANG_BY_EXT.get(PurePosixPath(changed_file).suffix.lower())
    if lang_id is None or not git_observed_path_is_confined(repo, changed_file):
        return []
    parser = get_parser(lang_id)
    query = _query_for_language(lang_id)
    if parser is None or query is None:
        return []
    try:
        source = (repo / changed_file).read_bytes()
    except OSError:
        return []
    resolved: list[str] = []
    for imp in extract_imports(parser, source, query):
        resolved.extend(_resolve_import(imp, changed_file))
    return resolved


def _read_confined_utf8(repo: Path, rel_path: str, *, max_bytes: int | None = None) -> str | None:
    """Read a confined regular UTF-8 file (optionally prefix-bounded); reject symlinks, escapes, and binaries."""
    if not git_observed_path_is_confined(repo, rel_path):
        return None
    full = repo / rel_path
    try:
        if full.is_symlink() or not full.is_file():
            return None
        with full.open("rb") as handle:
            raw = handle.read() if max_bytes is None else handle.read(max_bytes + 1)
    except OSError:
        return None
    if b"\x00" in raw:
        return None
    truncated = max_bytes is not None and len(raw) > max_bytes
    chunk = raw[:max_bytes] if truncated else raw
    try:
        return codecs.getincrementaldecoder("utf-8")("strict").decode(chunk, final=not truncated)
    except UnicodeDecodeError:
        return None


def collect_supporting_candidates(
    repo: Path,
    changed_files: Sequence[str],
    *,
    max_candidates: int = MAX_SUPPORTING_CANDIDATES,
    preview_bytes: int = SUPPORTING_PREVIEW_MAX_BYTES,
) -> list[SupportingCandidate]:
    """Collect bounded, confined unchanged neighbor files (direct imports and conventional tests)."""
    changed_set = set(changed_files)
    seen: set[str] = set()
    results: list[SupportingCandidate] = []
    deadline = time.monotonic() + _CANDIDATE_SCAN_WALL_BUDGET_S

    def _try_add(path: str, relation: Literal["import", "test"]) -> bool:
        if path in changed_set or path in seen:
            return False
        seen.add(path)
        preview = _read_confined_utf8(repo, path, max_bytes=preview_bytes)
        if preview is None:
            return False
        results.append(SupportingCandidate(path=path, relation=relation, preview=preview))
        return len(results) >= max_candidates

    for changed_file in changed_files:
        if time.monotonic() > deadline:
            break
        for imported in _direct_import_paths(repo, changed_file):
            if _try_add(imported, "import"):
                return results
        for test_path in _conventional_test_paths(changed_file):
            if _try_add(test_path, "test"):
                return results
    return results


@dataclass(frozen=True)
class ReviewPrefetchHints:
    """Advisory System 1 pre-dispatch hints, target triage classifications, and risk priority scores."""

    historical_context: dict[str, Any] | None = None
    supporting_files_by_group: Mapping[frozenset[str], tuple[dict[str, Any], ...]] = field(default_factory=dict)
    target_triages: Mapping[str, TargetTriage] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> ReviewPrefetchHints:
        return cls()

    def _risk_score(self, target_id: str) -> float:
        triage = self.target_triages.get(target_id)
        return UNSCORED_NEUTRAL_RISK if triage is None or triage.fallback else triage.risk_score

    def order_targets(
        self,
        targets: Sequence[str],
        inventory_index: Mapping[str, int],
    ) -> tuple[str, ...]:
        """Sort *targets* highest risk first; preserve inventory order on ties or when unscored."""
        if not self.target_triages:
            return tuple(targets)
        return tuple(
            sorted(
                targets,
                key=lambda target_id: (-self._risk_score(target_id), inventory_index.get(target_id, 0)),
            )
        )

    def group_priority(self, targets: Sequence[str], fallback_index: int) -> tuple[float, int]:
        """Priority key for ordering batches (highest max target risk first)."""
        if not self.target_triages or not targets:
            return (0.0, fallback_index)
        return (-max(self._risk_score(target_id) for target_id in targets), fallback_index)

    def group_weight(self, targets: Sequence[str]) -> float:
        """Risk-weighted tool-call budget weight for one batch (neutral medium weight when unscored)."""
        if not self.target_triages or not targets:
            return RISK_TIER_WEIGHTS["medium"]
        return max(
            RISK_TIER_WEIGHTS["medium"]
            if (triage := self.target_triages.get(target_id)) is None or triage.fallback
            else RISK_TIER_WEIGHTS[triage.risk_tier]
            for target_id in targets
        )

    def is_fast_tier_group(self, targets: Sequence[str]) -> bool:
        """Whether every target in *targets* safely qualifies for a fast/low-effort discovery tier."""
        return bool(self.target_triages and targets) and all(
            (triage := self.target_triages.get(target_id)) is not None and triage.qualifies_for_fast_tier()
            for target_id in targets
        )

    def work_advisory_hints(self, targets: Sequence[str]) -> dict[str, Any] | None:
        """Build the bounded advisory hints payload for one investigation dispatch, or None."""
        hints: dict[str, Any] = {}
        if self.historical_context is not None:
            hints["historical_context"] = self.historical_context
        target_set = frozenset(targets)
        supporting = self.supporting_files_by_group.get(target_set)
        if supporting is None and target_set:
            # Follow-up work holds a subset of its parent group's targets; reuse matching parent hints.
            for group_targets, group_hints in self.supporting_files_by_group.items():
                if target_set <= group_targets:
                    supporting = group_hints
                    break
        if supporting:
            hints["supporting_files"] = list(supporting)
        return hints or None


def _group_diff_excerpt(
    group: Sequence[str],
    target_diffs: Mapping[str, tuple[str, str]],
    *,
    max_bytes: int = GROUP_DIFF_EXCERPT_MAX_BYTES,
) -> tuple[list[str], str]:
    files = list(dict.fromkeys(target_diffs[t][0] for t in group if t in target_diffs))
    combined = "\n".join(target_diffs[t][1] for t in group if t in target_diffs)
    return files, truncate_utf8_to_budget(combined, max_bytes)


def _partition_target_triage_items(
    target_diffs: Mapping[str, tuple[str, str]],
    *,
    precomputed: Mapping[str, TargetTriage] | None = None,
) -> tuple[
    dict[str, TargetTriage],
    list[tuple[str, str]],
    list[tuple[Any, Mapping[str, DecisionQuestion]]],
]:
    """Separate precomputed/hard-denied targets from targets queued for System 1 triage."""
    triages: dict[str, TargetTriage] = {}
    scored_ids: list[tuple[str, str]] = []
    items: list[tuple[Any, Mapping[str, DecisionQuestion]]] = []
    for target_id, (file_path, diff_text) in target_diffs.items():
        if precomputed is not None and target_id in precomputed:
            triages[target_id] = precomputed[target_id]
            continue
        if path_is_hard_denied(file_path):
            triages[target_id] = TargetTriage(
                target_id=target_id,
                file_path=file_path,
                risk_score=PRIORITY_MAX_RISK,
                risk_tier="high_critical",
                hard_denied=True,
            )
            continue
        scored_ids.append((target_id, file_path))
        items.append(({"path": file_path, "diff": diff_text}, TARGET_TRIAGE_QUESTIONS))
    return triages, scored_ids, items


def _triage_from_outcome(target_id: str, file_path: str, outcome: DecisionOutcome) -> TargetTriage:
    if not outcome.ok or outcome.response is None:
        return TargetTriage(
            target_id=target_id,
            file_path=file_path,
            risk_score=UNSCORED_NEUTRAL_RISK,
            risk_tier="medium",
            fallback=True,
        )
    model_name = outcome.response.model
    if injection_detected(outcome):
        return TargetTriage(
            target_id=target_id,
            file_path=file_path,
            risk_score=PRIORITY_MAX_RISK,
            risk_tier="high_critical",
            injection_flagged=True,
            model=model_name,
        )
    p_non_behavioral = outcome.noul("non_behavioral")
    p_cross_file = outcome.noul("cross_file_contract")
    risk_ans = outcome.score("defect_risk")
    if risk_ans is not None:
        risk_score = risk_ans.score
    elif p_non_behavioral is not None:
        risk_score = (1.0 - p_non_behavioral) * 4.0
    else:
        return TargetTriage(
            target_id=target_id,
            file_path=file_path,
            risk_score=UNSCORED_NEUTRAL_RISK,
            risk_tier="medium",
            fallback=True,
            model=model_name,
        )
    tier_ans = outcome.choice("risk_tier")
    risk_tier: RiskTier = _RISK_TIERS.get(tier_ans.choice, "medium") if tier_ans is not None else "medium"
    return TargetTriage(
        target_id=target_id,
        file_path=file_path,
        risk_score=risk_score,
        risk_tier=risk_tier,
        p_non_behavioral=p_non_behavioral,
        p_cross_file=p_cross_file,
        model=model_name,
    )


async def triage_targets(
    *,
    client: System1Client,
    target_diffs: Mapping[str, tuple[str, str]],
    session_id: str | None = None,
    remaining_s: float | None = None,
) -> tuple[dict[str, TargetTriage], tuple[DecisionUsage, ...]]:
    """Score non-hard-denied targets in one parallel pass; return triages plus any unbilled usage."""
    ready = client.with_remaining_budget(remaining_s)
    if ready is None:
        return {}, ()
    triages, scored_ids, items = _partition_target_triage_items(target_diffs)

    has_recorder = get_current_recorder() is not None
    outcomes = await ready.evaluate_many(
        items,
        phase=DaydreamPhase.REVIEW,
        session_id=session_id,
        record_usage=has_recorder,
    )
    pending_usage: list[DecisionUsage] = []
    for (target_id, file_path), outcome in zip(scored_ids, outcomes, strict=True):
        triages[target_id] = _triage_from_outcome(target_id, file_path, outcome)
        if not has_recorder and outcome.ok and outcome.response is not None:
            pending_usage.append(outcome.response.usage)
    return triages, tuple(pending_usage)


async def prepare_review_prefetch_hints(
    *,
    client: System1Client,
    repo: Path,
    target_diffs: Mapping[str, tuple[str, str]],
    groups: Sequence[tuple[str, ...]],
    historical_fields: Mapping[str, bytes],
    session_id: str | None = None,
    remaining_s: float | None = None,
    precomputed_triages: Mapping[str, TargetTriage] | None = None,
) -> ReviewPrefetchHints:
    """Evaluate targets, historical context, and supporting files in one parallel System 1 pass."""
    ready = client.with_remaining_budget(remaining_s)
    if ready is None:
        return ReviewPrefetchHints.empty()

    # Discover confined supporting candidates off the event loop so timers/host tools stay responsive.
    group_files_and_excerpts = [_group_diff_excerpt(group, target_diffs) for group in groups]
    candidates_by_group = await anyio.to_thread.run_sync(
        lambda: [collect_supporting_candidates(repo, files) for files, _ in group_files_and_excerpts]
    )

    # 1. Target triage items (reuse Mode 2 precomputed triages if already evaluated in plan_review).
    target_triages, scored_target_ids, items = _partition_target_triage_items(
        target_diffs,
        precomputed=precomputed_triages,
    )

    # 2. Historical context relevance items (author text stays host-side; only advisory scores surface).
    historical_keys: list[str] = []
    for field_name, raw_bytes in sorted(historical_fields.items()):
        if not raw_bytes:
            continue
        text = truncate_utf8_to_budget(raw_bytes.decode("utf-8", errors="replace"), HISTORICAL_EXCERPT_MAX_BYTES)
        historical_keys.append(field_name)
        items.append(({"field": field_name, "text": text}, HISTORICAL_CONTEXT_QUESTIONS))

    # 3. Supporting-file relevance items per planned group.
    support_slots: list[tuple[int, SupportingCandidate]] = []
    for group_idx, candidates in enumerate(candidates_by_group):
        files, excerpt = group_files_and_excerpts[group_idx]
        for candidate in candidates:
            support_slots.append((group_idx, candidate))
            items.append((
                {
                    "changed_files": files,
                    "diff_excerpt": excerpt,
                    "supporting_file": candidate.path,
                    "relation": candidate.relation,
                    "supporting_preview": candidate.preview,
                },
                SUPPORTING_FILE_QUESTIONS,
            ))

    outcomes = await ready.evaluate_many(
        items,
        phase=DaydreamPhase.REVIEW,
        session_id=session_id,
        record_usage=True,
    )

    cursor = 0
    for target_id, file_path in scored_target_ids:
        target_triages[target_id] = _triage_from_outcome(target_id, file_path, outcomes[cursor])
        cursor += 1
    has_scored_targets = any(not triage.fallback for triage in target_triages.values())

    hist_fields_summary: dict[str, dict[str, Any]] = {}
    for field_name in historical_keys:
        outcome = outcomes[cursor]
        cursor += 1
        if not outcome.ok or injection_detected(outcome):
            continue
        prob = outcome.noul("substantive_intent")
        if prob is None:
            continue
        hist_fields_summary[field_name] = {
            "substantive": prob >= HISTORICAL_SUBSTANTIVE_THRESHOLD,
            "probability": round(prob, 4),
        }
    historical_hint: dict[str, Any] | None = None
    if hist_fields_summary:
        any_substantive = any(entry["substantive"] for entry in hist_fields_summary.values())
        historical_hint = {
            "advisory": "fetch_recommended" if any_substantive else "likely_boilerplate",
            "fields": hist_fields_summary,
        }

    ranked_by_group_idx: dict[int, list[tuple[float, float, SupportingCandidate]]] = {}
    for group_idx, candidate in support_slots:
        outcome = outcomes[cursor]
        cursor += 1
        if not outcome.ok or injection_detected(outcome):
            continue
        rel_prob = outcome.noul("relevant_to_diff")
        if rel_prob is None or rel_prob < SUPPORTING_RELEVANCE_THRESHOLD:
            continue
        rel_score_ans = outcome.score("relevance_score")
        rel_score = rel_score_ans.score if rel_score_ans is not None else rel_prob * 3.0
        ranked_by_group_idx.setdefault(group_idx, []).append((rel_score, rel_prob, candidate))

    supporting_by_group: dict[frozenset[str], tuple[dict[str, Any], ...]] = {}
    for group_idx, group in enumerate(groups):
        ranked = ranked_by_group_idx.get(group_idx)
        if not ranked:
            continue
        ranked.sort(key=lambda item: (-item[0], -item[1], item[2].path))
        supporting_by_group[frozenset(group)] = tuple(
            {
                "path": candidate.path,
                "relation": candidate.relation,
                "relevance": round(rel_prob, 4),
            }
            for _score, rel_prob, candidate in ranked[:MAX_SUPPORTING_HINTS]
        )

    return ReviewPrefetchHints(
        historical_context=historical_hint,
        supporting_files_by_group=supporting_by_group,
        target_triages=target_triages if has_scored_targets else {},
    )


# --- Pre-challenge claim triage, duplicate clustering, and 2-tier arbiter routing (#1972) ---

SeverityLevel = Literal["high", "medium", "low"]
_SEVERITY_LEVELS: Mapping[str, SeverityLevel] = {"high": "high", "medium": "medium", "low": "low"}
CLAIM_ENCLOSING_RADIUS_LINES = 30
CLAIM_ENCLOSING_MAX_BYTES = 4096
CLAIM_DIFF_EXCERPT_MAX_BYTES = 6144
CLAIM_DUPLICATE_THRESHOLD = 0.80
MAX_DUPLICATE_PAIRS = 16
FAST_CHALLENGE_CALL_CAP = 6


@dataclass(frozen=True)
class ClaimTriage:
    """Calibrated System 1 pre-challenge triage for one pending claim revision."""

    revision: int
    p_invalid: float | None = None
    p_needs_repo_search: float | None = None
    calibrated_severity: SeverityLevel | None = None
    injection_flagged: bool = False
    fallback: bool = False
    model: str | None = None
    cited_diff: str = ""
    enclosing_source: str = ""

    def qualifies_for_fast_challenge(self, *, search_max: float) -> bool:
        """Whether this self-contained local claim can use the fast/low-effort arbiter tier."""
        if self.injection_flagged or self.fallback:
            return False
        return self.p_needs_repo_search is not None and self.p_needs_repo_search <= search_max

    def qualifies_for_fast_reject(self, *, reject_threshold: float, search_max: float) -> bool:
        """Whether this claim qualifies for opt-in asymmetric host fast-rejection (never confirmation)."""
        return (
            self.qualifies_for_fast_challenge(search_max=search_max)
            and self.p_invalid is not None
            and self.p_invalid >= reject_threshold
        )


@dataclass(frozen=True)
class ChallengeTriagePlan:
    """Host System 1 pre-challenge triage, duplicate clusters, and 2-tier routing plan."""

    triages: Mapping[str, ClaimTriage] = field(default_factory=dict)
    duplicate_clusters: tuple[tuple[str, ...], ...] = ()

    @classmethod
    def empty(cls) -> ChallengeTriagePlan:
        return cls()

    def calibrated_severities(self, claim_uids: Sequence[str] | None = None) -> dict[str, str]:
        uids = self.triages if claim_uids is None else claim_uids
        return {
            uid: triage.calibrated_severity
            for uid in uids
            if (triage := self.triages.get(uid)) is not None
            and not triage.injection_flagged
            and not triage.fallback
            and triage.calibrated_severity is not None
        }

    def is_fast_challenge_group(self, claim_uids: Sequence[str], *, search_max: float) -> bool:
        if not self.triages or not claim_uids:
            return False
        return all(
            (triage := self.triages.get(uid)) is not None and triage.qualifies_for_fast_challenge(search_max=search_max)
            for uid in claim_uids
        )

    def plan_challenge_groups(
        self,
        available: Mapping[str, int],
        *,
        allowance_by_uid: Mapping[str, int],
        size_by_uid: Mapping[str, int],
        max_bytes: int,
        has_fast_backend: bool,
        search_max: float,
    ) -> list[dict[str, int]]:
        """Partition pending claims keeping duplicate clusters together and separating fast local groups."""
        if not available:
            return []
        cluster_of: dict[str, tuple[str, ...]] = {}
        for cluster in self.duplicate_clusters:
            present = tuple(uid for uid in cluster if uid in available)
            if len(present) >= 2:
                for uid in present:
                    cluster_of[uid] = present

        # Build atomic units: each duplicate cluster stays together; non-clustered claims are singletons.
        units: list[tuple[str, ...]] = []
        seen: set[str] = set()
        for uid in available:
            if uid in seen:
                continue
            unit = cluster_of.get(uid, (uid,))
            units.append(unit)
            seen.update(unit)

        chunks: dict[tuple[int, int], list[dict[str, int]]] = {}
        sizes: dict[tuple[int, int], int] = {}
        for unit in units:
            is_fast = int(has_fast_backend and self.is_fast_challenge_group(unit, search_max=search_max))
            base_group = min(allowance_by_uid.get(uid, 0) for uid in unit)
            bucket = (is_fast, base_group)
            unit_size = sum(size_by_uid[uid] for uid in unit)
            if bucket not in chunks or (chunks[bucket][-1] and sizes[bucket] + unit_size > max_bytes):
                chunks.setdefault(bucket, []).append({})
                sizes[bucket] = 1
            for uid in unit:
                chunks[bucket][-1][uid] = available[uid]
            sizes[bucket] += unit_size
        return [group for bucket_groups in chunks.values() for group in bucket_groups]

    def work_advisory_hints(self, claim_uids: Sequence[str]) -> dict[str, Any] | None:
        uid_set = set(claim_uids)
        severities = self.calibrated_severities(claim_uids)
        clusters = [
            list(cluster)
            for cluster in self.duplicate_clusters
            if len(uid_set.intersection(cluster)) >= 2
        ]
        hints: dict[str, Any] = {}
        if severities:
            hints["calibrated_severity"] = severities
        if clusters:
            hints["possible_duplicate_clusters"] = clusters
        return hints or None

    def preinlined_context_blocks(self, claim_uids: Sequence[str]) -> list[dict[str, Any]]:
        """Bounded untrusted diff + enclosing source snippets for fast local challenge dispatches."""
        blocks: list[dict[str, Any]] = []
        for uid in claim_uids:
            triage = self.triages.get(uid)
            if triage is None:
                continue
            if triage.cited_diff or triage.enclosing_source:
                blocks.append({
                    "uid": uid,
                    "revision": triage.revision,
                    "cited_diff": triage.cited_diff,
                    "enclosing_source": triage.enclosing_source,
                })
        return blocks


def extract_claim_enclosing_source(
    repo: Path,
    file_path: str,
    line: int | None,
    *,
    radius: int = CLAIM_ENCLOSING_RADIUS_LINES,
    max_bytes: int = CLAIM_ENCLOSING_MAX_BYTES,
) -> str:
    """Read bounded line-numbered enclosing source around *line* from a confined captured file."""
    text = _read_confined_utf8(repo, file_path)
    if not text:
        return ""
    lines = text.splitlines()
    if not lines:
        return ""
    if line is None or line < 1:
        start, end = 1, min(len(lines), 2 * radius + 1)
    else:
        center = min(line, len(lines))
        start = max(1, center - radius)
        end = min(len(lines), center + radius)
    excerpt = "\n".join(f"L{lineno}: {lines[lineno - 1]}" for lineno in range(start, end + 1))
    return truncate_utf8_to_budget(excerpt, max_bytes)


def _claim_triage_from_outcome(
    claim: Mapping[str, Any],
    outcome: DecisionOutcome,
    *,
    cited_diff: str,
    enclosing_source: str,
) -> ClaimTriage:
    revision = int(claim["revision"])
    if not outcome.ok or outcome.response is None:
        return ClaimTriage(
            revision=revision,
            fallback=True,
            cited_diff=cited_diff,
            enclosing_source=enclosing_source,
        )
    model_name = outcome.response.model
    if injection_detected(outcome):
        return ClaimTriage(
            revision=revision,
            injection_flagged=True,
            model=model_name,
            cited_diff=cited_diff,
            enclosing_source=enclosing_source,
        )
    sev_ans = outcome.choice("severity")
    calibrated = _SEVERITY_LEVELS.get(sev_ans.choice) if sev_ans is not None else None
    return ClaimTriage(
        revision=revision,
        p_invalid=outcome.noul("obviously_invalid"),
        p_needs_repo_search=outcome.noul("needs_repo_search"),
        calibrated_severity=calibrated,
        model=model_name,
        cited_diff=cited_diff,
        enclosing_source=enclosing_source,
    )


def _connected_clusters(pairs: Sequence[tuple[str, str]]) -> tuple[tuple[str, ...], ...]:
    adjacency: dict[str, set[str]] = {}
    for left, right in pairs:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    visited: set[str] = set()
    clusters: list[tuple[str, ...]] = []
    for node in sorted(adjacency):
        if node in visited:
            continue
        component: list[str] = []
        stack = [node]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            component.append(current)
            stack.extend(adjacency[current] - visited)
        if len(component) >= 2:
            clusters.append(tuple(sorted(component)))
    return tuple(clusters)


def _claim_brief(claim: Mapping[str, Any]) -> dict[str, Any]:
    payload = claim["payload"]
    return {
        "title": claim.get("title", ""),
        "line": payload.get("line"),
        "claim": payload.get("claim", ""),
        "trigger": payload.get("trigger", ""),
    }


async def prepare_challenge_triage(
    *,
    client: System1Client,
    repo: Path,
    pending_claims: Sequence[Mapping[str, Any]],
    diff_by_file: Mapping[str, str],
    session_id: str | None = None,
    remaining_s: float | None = None,
) -> ChallengeTriagePlan:
    """Evaluate pending claims + bounded same-file duplicate pairs in one parallel System 1 pass."""
    ready = client.with_remaining_budget(remaining_s)
    if not pending_claims or ready is None:
        return ChallengeTriagePlan.empty()

    def _collect_contexts() -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for claim in pending_claims:
            payload = claim["payload"]
            file_path = str(payload["file"])
            line = payload.get("line")
            cited_diff = truncate_utf8_to_budget(diff_by_file.get(file_path, ""), CLAIM_DIFF_EXCERPT_MAX_BYTES)
            enclosing = extract_claim_enclosing_source(
                repo,
                file_path,
                line if isinstance(line, int) else None,
            )
            out.append((cited_diff, enclosing))
        return out

    contexts = await anyio.to_thread.run_sync(_collect_contexts)

    items: list[tuple[Any, Mapping[str, DecisionQuestion]]] = []
    for claim, (cited_diff, enclosing) in zip(pending_claims, contexts, strict=True):
        payload = claim["payload"]
        items.append((
            {
                "title": claim.get("title", ""),
                "file": payload["file"],
                "line": payload.get("line"),
                "severity": payload.get("severity"),
                "claim": payload.get("claim", ""),
                "trigger": payload.get("trigger", ""),
                "consequence": payload.get("consequence", ""),
                "grounds": payload.get("grounds", ""),
                "cited_diff": cited_diff,
                "enclosing_source": enclosing,
            },
            CLAIM_TRIAGE_QUESTIONS,
        ))

    # Pairwise duplicate checks across pending claims touching the same file (bounded).
    by_file: dict[str, list[Mapping[str, Any]]] = {}
    for claim in pending_claims:
        by_file.setdefault(str(claim["payload"]["file"]), []).append(claim)
    dup_pairs: list[tuple[str, str]] = []
    same_file_pairs = (
        (left, right)
        for file_claims in by_file.values()
        for left, right in combinations(file_claims, 2)
    )
    for left, right in islice(same_file_pairs, MAX_DUPLICATE_PAIRS):
        dup_pairs.append((str(left["uid"]), str(right["uid"])))
        items.append((
            {
                "file": left["payload"]["file"],
                "claim_a": _claim_brief(left),
                "claim_b": _claim_brief(right),
            },
            CLAIM_DUPLICATE_QUESTIONS,
        ))

    outcomes = await ready.evaluate_many(
        items,
        phase=DaydreamPhase.CHALLENGE,
        session_id=session_id,
        record_usage=True,
    )

    triages: dict[str, ClaimTriage] = {}
    claim_count = len(pending_claims)
    for idx, (claim, (cited_diff, enclosing)) in enumerate(zip(pending_claims, contexts, strict=True)):
        triages[str(claim["uid"])] = _claim_triage_from_outcome(
            claim,
            outcomes[idx],
            cited_diff=cited_diff,
            enclosing_source=enclosing,
        )

    matched_pairs: list[tuple[str, str]] = []
    for pair_idx, (uid_a, uid_b) in enumerate(dup_pairs):
        outcome = outcomes[claim_count + pair_idx]
        if not outcome.ok or injection_detected(outcome):
            continue
        prob = outcome.noul("is_duplicate")
        if prob is not None and prob >= CLAIM_DUPLICATE_THRESHOLD:
            matched_pairs.append((uid_a, uid_b))

    return ChallengeTriagePlan(
        triages=triages,
        duplicate_clusters=_connected_clusters(matched_pairs),
    )


# --- Cross-assignment boundary gating and focused pair ranking (#1973) ---

MAX_BATCH_PAIR_CHECKS = 12
MAX_FOCUSED_CROSS_PAIRS = 4
CROSS_SUMMARY_DIFF_PER_BATCH_BYTES = 2048
CROSS_PAIR_COUPLING_THRESHOLD = 0.25


@dataclass(frozen=True)
class CoupledBatchPair:
    """One ranked cross-batch boundary candidate for focused cross_assignment review."""

    batch_a: int
    batch_b: int
    files_a: tuple[str, ...]
    files_b: tuple[str, ...]
    import_edges: tuple[tuple[str, str], ...]
    p_coupled: float
    excerpt_a: str
    excerpt_b: str

    def advisory_entry(self) -> dict[str, Any]:
        return {
            "batch_a": self.batch_a,
            "batch_b": self.batch_b,
            "files_a": list(self.files_a),
            "files_b": list(self.files_b),
            "import_edges": [[src, dst] for src, dst in self.import_edges],
            "coupling_probability": round(self.p_coupled, 4),
        }


@dataclass(frozen=True)
class CrossAssignmentPlan:
    """Host System 1 cross-assignment risk gate and focused boundary-pair ranking."""

    p_cross_needed: float | None = None
    cross_batch_edge_count: int = 0
    focused_pairs: tuple[CoupledBatchPair, ...] = ()
    injection_flagged: bool = False
    fallback: bool = False
    model: str | None = None

    @classmethod
    def empty(cls, *, cross_batch_edge_count: int = 0) -> CrossAssignmentPlan:
        return cls(
            p_cross_needed=None,
            cross_batch_edge_count=cross_batch_edge_count,
            focused_pairs=(),
            fallback=True,
        )

    def qualifies_for_noop(
        self,
        *,
        gate_enabled: bool,
        threshold: float,
        unfinished_targets: int,
    ) -> bool:
        """Whether decomposed cross_assignment may safely record host noop=True."""
        if not gate_enabled or unfinished_targets > 0 or self.cross_batch_edge_count > 0:
            return False
        if self.injection_flagged or self.fallback or self.p_cross_needed is None:
            return False
        return self.p_cross_needed < threshold

    def work_advisory_hints(self) -> dict[str, Any] | None:
        if not self.focused_pairs:
            return None
        return {"coupled_batch_pairs": [pair.advisory_entry() for pair in self.focused_pairs]}


def find_cross_batch_edges(
    batch_files: Sequence[Sequence[str]],
    import_graph: Mapping[str, set[str]],
) -> dict[tuple[int, int], list[tuple[str, str]]]:
    """Return directed import edges crossing between distinct batches, keyed by ``(min_idx, max_idx)``."""
    batch_of: dict[str, int] = {}
    for idx, files in enumerate(batch_files):
        for path in files:
            batch_of.setdefault(path, idx)
    edges_by_pair: dict[tuple[int, int], list[tuple[str, str]]] = {}
    for src, dsts in sorted(import_graph.items()):
        src_batch = batch_of.get(src)
        if src_batch is None:
            continue
        for dst in sorted(dsts):
            dst_batch = batch_of.get(dst)
            if dst_batch is None or dst_batch == src_batch:
                continue
            pair = (min(src_batch, dst_batch), max(src_batch, dst_batch))
            edges_by_pair.setdefault(pair, []).append((src, dst))
    return edges_by_pair


async def prepare_cross_assignment_plan(
    *,
    client: System1Client,
    batches: Sequence[Sequence[str]],
    target_diffs: Mapping[str, tuple[str, str]],
    import_graph: Mapping[str, set[str]],
    admitted_claims: Sequence[Mapping[str, Any]],
    session_id: str | None = None,
    remaining_s: float | None = None,
) -> CrossAssignmentPlan:
    """Score PR-level cross-batch defect risk and rank coupled batch pairs in one System 1 pass."""
    batch_files: list[tuple[str, ...]] = [
        tuple(dict.fromkeys(target_diffs[t][0] for t in batch if t in target_diffs))
        for batch in batches
    ]
    edges_by_pair = find_cross_batch_edges(batch_files, import_graph)
    total_cross_edges = sum(len(edges) for edges in edges_by_pair.values())

    ready = client.with_remaining_budget(remaining_s)
    if len(batches) < 2 or ready is None:
        return CrossAssignmentPlan.empty(cross_batch_edge_count=total_cross_edges)

    batch_excerpts: list[str] = [
        truncate_utf8_to_budget(
            "\n".join(target_diffs[t][1] for t in batch if t in target_diffs),
            CROSS_SUMMARY_DIFF_PER_BATCH_BYTES,
        )
        for batch in batches
    ]
    pr_state = {
        "batches": [
            {"batch": idx, "files": list(files), "diff_excerpt": batch_excerpts[idx]}
            for idx, files in enumerate(batch_files)
        ],
        "cross_batch_import_edges": [
            {"batch_a": a, "batch_b": b, "edges": [[src, dst] for src, dst in edges]}
            for (a, b), edges in sorted(edges_by_pair.items())
        ],
        "admitted_local_claims": [
            {
                "file": claim.get("payload", {}).get("file"),
                "title": claim.get("title", ""),
                "severity": claim.get("payload", {}).get("severity"),
            }
            for claim in admitted_claims[:24]
        ],
    }

    # Prioritize batch pairs with known import edges first, then fill up to MAX_BATCH_PAIR_CHECKS.
    candidate_pairs: list[tuple[int, int]] = sorted(edges_by_pair)
    if len(candidate_pairs) < MAX_BATCH_PAIR_CHECKS:
        candidate_pairs.extend(
            islice(
                (pair for pair in combinations(range(len(batches)), 2) if pair not in edges_by_pair),
                MAX_BATCH_PAIR_CHECKS - len(candidate_pairs),
            )
        )

    items: list[tuple[Any, Mapping[str, DecisionQuestion]]] = [(pr_state, CROSS_ASSIGNMENT_QUESTIONS)]
    for a, b in candidate_pairs:
        items.append((
            {
                "batch_a": {"index": a, "files": list(batch_files[a]), "diff_excerpt": batch_excerpts[a]},
                "batch_b": {"index": b, "files": list(batch_files[b]), "diff_excerpt": batch_excerpts[b]},
                "import_edges": [[src, dst] for src, dst in edges_by_pair.get((a, b), ())],
            },
            BATCH_PAIR_COUPLING_QUESTIONS,
        ))

    outcomes = await ready.evaluate_many(
        items,
        phase=DaydreamPhase.REVIEW,
        session_id=session_id,
        record_usage=True,
    )

    pr_outcome = outcomes[0]
    if not pr_outcome.ok or pr_outcome.response is None:
        return CrossAssignmentPlan.empty(cross_batch_edge_count=total_cross_edges)
    pr_injection = injection_detected(pr_outcome)
    p_cross_needed = pr_outcome.noul("cross_boundary_needed")
    model_name = pr_outcome.response.model

    ranked_pairs: list[CoupledBatchPair] = []
    for pair_idx, (a, b) in enumerate(candidate_pairs):
        outcome = outcomes[1 + pair_idx]
        edges = tuple(edges_by_pair.get((a, b), ()))
        if not outcome.ok or injection_detected(outcome):
            # Deterministic import edges stay visible even if a pair call failed open.
            p_coupled = 1.0 if edges else None
        else:
            raw_p = outcome.noul("coupled_boundary")
            p_coupled = raw_p if raw_p is not None and (edges or raw_p >= CROSS_PAIR_COUPLING_THRESHOLD) else None
        if p_coupled is not None:
            ranked_pairs.append(
                CoupledBatchPair(
                    batch_a=a,
                    batch_b=b,
                    files_a=batch_files[a],
                    files_b=batch_files[b],
                    import_edges=edges,
                    p_coupled=p_coupled,
                    excerpt_a=batch_excerpts[a],
                    excerpt_b=batch_excerpts[b],
                )
            )
    ranked_pairs.sort(key=lambda p: (not bool(p.import_edges), -p.p_coupled, p.batch_a, p.batch_b))
    return CrossAssignmentPlan(
        p_cross_needed=p_cross_needed,
        cross_batch_edge_count=total_cross_edges,
        focused_pairs=tuple(ranked_pairs[:MAX_FOCUSED_CROSS_PAIRS]),
        injection_flagged=pr_injection,
        fallback=p_cross_needed is None,
        model=model_name,
    )


# --- Diagram auto semantic pre-gate & Improve xhigh vetting triage (#1974) ---

DIAGRAM_SEQUENCE_QUESTION = NoulQuestion(
    instructions=(
        "Does this change introduce or materially modify a multi-component interaction flow "
        "that a sequence diagram would clarify for a reviewer?"
    ),
    true_criteria=(
        "Introduces or meaningfully alters cross-component calls, async handoffs, protocol flows, "
        "or request/response orchestration across modules/services."
    ),
    false_criteria=(
        "Mechanical refactor, bulk rename, repetitive boilerplate, or flat changes where a "
        "sequence diagram adds no clarifying insight."
    ),
)

DIAGRAM_FLOWCHART_QUESTION = NoulQuestion(
    instructions=(
        "Does this change introduce or materially modify non-trivial branching or state-machine "
        "logic that a flowchart would clarify for a reviewer?"
    ),
    true_criteria=(
        "Introduces or alters meaningful decision trees, fallback paths, retry/error states, "
        "or non-trivial control flow in a changed function."
    ),
    false_criteria=(
        "Simple input validation guards, mechanical table/switch additions, or routine boilerplate "
        "that needs no flowchart."
    ),
)

IMPROVE_VET_TRIAGE_QUESTIONS: dict[str, DecisionQuestion] = with_injection_guard({
    "speculative_noise": NoulQuestion(
        instructions=(
            "Is this audit candidate clearly speculative, purely stylistic/nitpicky, or unsupported "
            "by the cited source excerpt?"
        ),
        true_criteria=(
            "Clearly speculative, stylistic preference without real failure mode, or contradicted "
            "by the cited source excerpt."
        ),
        false_criteria=(
            "Plausible, concrete maintainability, correctness, security, or performance issue "
            "warranting deep verification."
        ),
    ),
    "concrete_impact": ScoreQuestion(
        instructions="Rate the concrete engineering impact and trigger strength of this audit candidate.",
        criteria=[
            "Noise or unsupported speculation",
            "Minimal/cosmetic impact only",
            "Moderate localized improvement",
            "Strong concrete issue with clear trigger",
            "High-leverage correctness, security, or architectural risk",
        ],
    ),
})

DIAGRAM_DIFF_EXCERPT_MAX_BYTES = 12_288
IMPROVE_PREFILTER_IMPACT_MAX = 1.0


_DIAGRAM_GATE_SPECS: tuple[tuple[str, str, DecisionQuestion, str], ...] = (
    ("sequence", "sequence_clarifies", DIAGRAM_SEQUENCE_QUESTION, "no clarifying cross-component flow"),
    ("flowchart", "flowchart_clarifies", DIAGRAM_FLOWCHART_QUESTION, "no clarifying branching logic"),
)


async def apply_system1_diagram_gate(
    eligibility: Any,
    *,
    client: System1Client,
    diff_text: str,
    import_graph: Mapping[str, set[str]],
    session_id: str | None = None,
) -> Any:
    """Apply System 1 semantic pre-checks to ``--diagram auto`` eligibility; fail open on errors."""
    ready = client.with_remaining_budget()
    if eligibility.force != "auto" or ready is None:
        return eligibility
    questions: dict[str, DecisionQuestion] = {
        qkey: question
        for attr, qkey, question, _detail in _DIAGRAM_GATE_SPECS
        if getattr(eligibility, attr).eligible
    }
    if not questions:
        return eligibility

    from daydream.deep.diagram_trigger import KindDecision

    state = {
        "code_files": list(eligibility.code_files[:40]),
        "modules": dict(list(eligibility.modules.items())[:40]),
        "services": dict(list(eligibility.services.items())[:40]),
        "cross_module_edges": eligibility.cross_module_edges,
        "import_edges": [
            [src, dst]
            for src, dsts in sorted(import_graph.items())
            for dst in sorted(dsts)
        ][:40],
        "candidate_roots": [
            {
                "file": root.file,
                "name": root.name,
                "line": root.line,
                "branch_points": root.branch_points,
            }
            for root in eligibility.candidate_roots[:10]
        ],
        "diff_excerpt": truncate_utf8_to_budget(diff_text, DIAGRAM_DIFF_EXCERPT_MAX_BYTES),
    }
    outcome = await ready.evaluate(
        state,
        with_injection_guard(questions),
        phase=DaydreamPhase.DIAGRAM,
        session_id=session_id,
        record_usage=True,
    )
    if not outcome.ok or injection_detected(outcome):
        return eligibility

    threshold = ready.config.diagram_min_probability
    updates: dict[str, Any] = {}
    for attr, qkey, _question, detail in _DIAGRAM_GATE_SPECS:
        if qkey not in questions:
            continue
        prob = outcome.noul(qkey)
        if prob is not None and prob < threshold:
            updates[attr] = KindDecision(
                eligible=False,
                rule="system1-gate",
                reason=(
                    f"System 1 semantic gate skipped {attr} diagram "
                    f"(p={prob:.3f} < {threshold:.2f}; {detail})."
                ),
            )
    return replace(eligibility, **updates) if updates else eligibility


async def triage_improve_vet_candidates(
    candidates: Sequence[dict[str, Any]],
    *,
    client: System1Client,
    repo: Path,
    session_id: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Score Improve audit candidates with System 1 for signal-ordered batching and opt-in pre-filtering.

    Reads only bounded excerpts confined to *repo* (preserving snapshot-root isolation).
    Returns ``(ordered_for_vet, prefiltered_audit_records)``.
    """
    ready = client.with_remaining_budget()
    if not candidates or ready is None:
        return list(candidates), []

    def _collect_excerpts() -> list[str]:
        excerpts: list[str] = []
        for finding in candidates:
            path = str(finding.get("path") or "")
            raw_line = finding.get("line")
            line = raw_line if isinstance(raw_line, int) else None
            excerpts.append(
                extract_claim_enclosing_source(repo, path, line, radius=20, max_bytes=3072)
                if path
                else ""
            )
        return excerpts

    excerpts = await anyio.to_thread.run_sync(_collect_excerpts)
    items: list[tuple[Any, Mapping[str, DecisionQuestion]]] = [
        (
            {
                "category": finding.get("category"),
                "title": finding.get("title", ""),
                "path": finding.get("path", ""),
                "line": finding.get("line"),
                "severity": finding.get("severity"),
                "confidence": finding.get("confidence"),
                "summary": finding.get("summary") or finding.get("description") or finding.get("claim", ""),
                "evidence": finding.get("evidence", []),
                "source_excerpt": excerpt,
            },
            IMPROVE_VET_TRIAGE_QUESTIONS,
        )
        for finding, excerpt in zip(candidates, excerpts, strict=True)
    ]

    outcomes = await ready.evaluate_many(
        items,
        phase=DaydreamPhase.VET,
        session_id=session_id,
        record_usage=True,
    )

    kept_indexed: list[tuple[tuple[float, float, int], dict[str, Any]]] = []
    prefiltered: list[dict[str, Any]] = []
    threshold = ready.config.improve_reject_threshold
    for idx, (finding, outcome) in enumerate(zip(candidates, outcomes, strict=True)):
        if not outcome.ok or outcome.response is None or injection_detected(outcome):
            kept_indexed.append(((-PRIORITY_MAX_RISK, 0.0, idx), finding))
            continue
        p_noise = outcome.noul("speculative_noise")
        impact_ans = outcome.score("concrete_impact")
        concrete_impact = impact_ans.score if impact_ans is not None else None
        model = outcome.response.model or ready.config.model
        if (
            ready.config.improve_prefilter
            and p_noise is not None
            and p_noise >= threshold
            and concrete_impact is not None
            and concrete_impact <= IMPROVE_PREFILTER_IMPACT_MAX
        ):
            prefiltered.append({
                "fingerprint": str(finding.get("fingerprint") or ""),
                "category": finding.get("category", "unknown"),
                "title": finding.get("title", ""),
                "path": finding.get("path", ""),
                "line": finding.get("line"),
                "p_noise": round(p_noise, 4),
                "concrete_impact": round(concrete_impact, 4),
                "threshold": threshold,
                "model": model,
                "reason": (
                    f"system1_prefiltered: speculative/unsupported noise "
                    f"(p={p_noise:.3f} >= {threshold:.2f}, impact={concrete_impact:.2f}, model={model})"
                ),
            })
            continue
        impact = concrete_impact if concrete_impact is not None else 2.0
        noise = p_noise if p_noise is not None else 0.5
        kept_indexed.append(((-impact, noise, idx), finding))

    kept_indexed.sort(key=lambda item: item[0])
    return [finding for _, finding in kept_indexed], prefiltered
