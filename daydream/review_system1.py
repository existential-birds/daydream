"""Committed-review System 1 pre-dispatch ranking, supporting-file discovery, and triage questions.

Single authoritative home for review-domain Jev / System 1 question sets so live
committed reviews and offline shadow calibration evaluate identical rubrics.
All hints are strictly advisory, confined to repository paths, and fail open to
existing deterministic order when System 1 is disabled, unavailable, or times out.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
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


def resolve_fast_discovery_spec(
    config: Any,
    system1_config: System1Config,
    *,
    hosted: bool,
) -> tuple[str, str | None, str | None] | None:
    """Resolve optional fast discovery ``(backend, model, effort)`` when System 1 Mode 1 routing is configured."""
    return _resolve_fast_phase_spec(
        config,
        system1_config,
        base_phase="review",
        fast_phase="review_fast",
        override_backend=system1_config.fast_review_backend,
        override_model=system1_config.fast_review_model,
        override_effort=system1_config.fast_review_effort,
        hosted=hosted,
    )


def resolve_fast_arbiter_spec(
    config: Any,
    system1_config: System1Config,
    *,
    hosted: bool,
) -> tuple[str, str | None, str | None] | None:
    """Resolve optional fast arbiter ``(backend, model, effort)`` when System 1 2-tier challenge routing is set."""
    return _resolve_fast_phase_spec(
        config,
        system1_config,
        base_phase="arbiter",
        fast_phase="arbiter_fast",
        override_backend=system1_config.fast_arbiter_backend,
        override_model=system1_config.fast_arbiter_model,
        override_effort=system1_config.fast_arbiter_effort,
        hosted=hosted,
    )


def _resolve_fast_phase_spec(
    config: Any,
    system1_config: System1Config,
    *,
    base_phase: str,
    fast_phase: str,
    override_backend: str | None,
    override_model: str | None,
    override_effort: str | None,
    hosted: bool,
) -> tuple[str, str | None, str | None] | None:
    if not system1_config.enabled:
        return None
    from daydream.run_config import resolved_backend_name, resolved_model, resolved_reasoning_effort

    file_fast: Mapping[str, str] = {}
    if not hosted and config.file_config is not None:
        raw_fast = config.file_config.phases.get(fast_phase)
        if isinstance(raw_fast, Mapping):
            file_fast = raw_fast
    eff_backend = override_backend or file_fast.get("backend")
    eff_model = override_model or file_fast.get("model")
    eff_effort = override_effort or file_fast.get("reasoning_effort")
    if not any((eff_backend, eff_model, eff_effort)):
        return None
    base_backend = resolved_backend_name(config, base_phase)
    backend_name = eff_backend or base_backend
    model = eff_model or (
        resolved_model(config, base_phase) if backend_name == base_backend else None
    )
    effort = eff_effort or resolved_reasoning_effort(config, base_phase)
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


def _read_confined_preview(repo: Path, rel_path: str, *, max_bytes: int) -> str | None:
    """Read a bounded UTF-8 prefix of a confined regular file; reject symlinks, escapes, and binaries."""
    if not git_observed_path_is_confined(repo, rel_path):
        return None
    full = repo / rel_path
    try:
        if full.is_symlink() or not full.is_file():
            return None
        with full.open("rb") as handle:
            raw = handle.read(max_bytes + 1)
    except OSError:
        return None
    if b"\x00" in raw:
        return None
    truncated = len(raw) > max_bytes
    chunk = raw[:max_bytes]
    # Trim any split trailing UTF-8 sequence only when we truncated mid-file.
    if truncated:
        while chunk and (chunk[-1] & 0xC0) == 0x80:
            chunk = chunk[:-1]
        if chunk and chunk[-1] >= 0xC0:
            chunk = chunk[:-1]
    try:
        return chunk.decode("utf-8")
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
        preview = _read_confined_preview(repo, path, max_bytes=preview_bytes)
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

    target_risk_scores: Mapping[str, float]
    historical_context: dict[str, Any] | None = None
    supporting_files_by_group: Mapping[frozenset[str], tuple[dict[str, Any], ...]] = field(default_factory=dict)
    target_triages: Mapping[str, TargetTriage] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> ReviewPrefetchHints:
        return cls(target_risk_scores={}, historical_context=None, supporting_files_by_group={}, target_triages={})

    def order_targets(
        self,
        targets: Sequence[str],
        inventory_index: Mapping[str, int],
    ) -> tuple[str, ...]:
        """Sort *targets* highest risk first; preserve inventory order on ties or when unscored."""
        if not self.target_risk_scores:
            return tuple(targets)
        return tuple(
            sorted(
                targets,
                key=lambda target_id: (
                    -self.target_risk_scores.get(target_id, UNSCORED_NEUTRAL_RISK),
                    inventory_index.get(target_id, 0),
                ),
            )
        )

    def group_priority(self, targets: Sequence[str], fallback_index: int) -> tuple[float, int]:
        """Priority key for ordering batches (highest max target risk first)."""
        if not self.target_risk_scores or not targets:
            return (0.0, fallback_index)
        max_risk = max(self.target_risk_scores.get(target_id, UNSCORED_NEUTRAL_RISK) for target_id in targets)
        return (-max_risk, fallback_index)

    def group_weight(self, targets: Sequence[str]) -> float:
        """Risk-weighted tool-call budget weight for one batch (neutral medium weight when unscored)."""
        if not self.target_triages or not targets:
            return RISK_TIER_WEIGHTS["medium"]
        weights: list[float] = []
        for target_id in targets:
            triage = self.target_triages.get(target_id)
            if triage is None or triage.fallback:
                weights.append(RISK_TIER_WEIGHTS["medium"])
            else:
                weights.append(RISK_TIER_WEIGHTS[triage.risk_tier])
        return max(weights)

    def is_fast_tier_group(self, targets: Sequence[str]) -> bool:
        """Whether every target in *targets* safely qualifies for a fast/low-effort discovery tier."""
        if not self.target_triages or not targets:
            return False
        for target_id in targets:
            triage = self.target_triages.get(target_id)
            if triage is None or not triage.qualifies_for_fast_tier():
                return False
        return True

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


def _clamp_client_timeout(client: System1Client, remaining_s: float | None) -> System1Client:
    if remaining_s is not None and remaining_s < client.config.batch_timeout_s:
        effective_timeout = min(client.config.timeout_s, remaining_s)
        return System1Client(
            replace(client.config, timeout_s=effective_timeout, batch_timeout_s=remaining_s),
            transport=client._transport,
        )
    return client


def _triage_from_outcome(target_id: str, file_path: str, outcome: DecisionOutcome) -> TargetTriage:
    if path_is_hard_denied(file_path):
        return TargetTriage(
            target_id=target_id,
            file_path=file_path,
            risk_score=PRIORITY_MAX_RISK,
            risk_tier="high_critical",
            hard_denied=True,
        )
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
    raw_tier = tier_ans.choice if tier_ans is not None else None
    risk_tier: RiskTier
    if raw_tier == "mechanical":
        risk_tier = "mechanical"
    elif raw_tier == "low":
        risk_tier = "low"
    elif raw_tier == "medium":
        risk_tier = "medium"
    elif raw_tier == "high_critical":
        risk_tier = "high_critical"
    else:
        risk_tier = "medium"
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
    if not client.config.enabled or not client.config.api_key:
        return {}, ()
    if remaining_s is not None and remaining_s <= 0:
        return {}, ()
    client = _clamp_client_timeout(client, remaining_s)
    triages: dict[str, TargetTriage] = {}
    scored_ids: list[tuple[str, str]] = []
    items: list[tuple[Any, Mapping[str, DecisionQuestion]]] = []
    for target_id, (file_path, diff_text) in target_diffs.items():
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

    has_recorder = get_current_recorder() is not None
    outcomes = await client.evaluate_many(
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
    if not client.config.enabled or not client.config.api_key:
        return ReviewPrefetchHints.empty()
    if remaining_s is not None and remaining_s <= 0:
        return ReviewPrefetchHints.empty()
    client = _clamp_client_timeout(client, remaining_s)

    # Discover confined supporting candidates off the event loop so timers/host tools stay responsive.
    group_files_and_excerpts = [_group_diff_excerpt(group, target_diffs) for group in groups]
    candidates_by_group = await anyio.to_thread.run_sync(
        lambda: [collect_supporting_candidates(repo, files) for files, _ in group_files_and_excerpts]
    )

    items: list[tuple[Any, Mapping[str, DecisionQuestion]]] = []
    # 1. Target triage items (reuse Mode 2 precomputed triages if already evaluated in plan_review).
    scored_target_ids: list[tuple[str, str]] = []
    target_triages: dict[str, TargetTriage] = {}
    for target_id, (file_path, diff_text) in target_diffs.items():
        if precomputed_triages is not None and target_id in precomputed_triages:
            target_triages[target_id] = precomputed_triages[target_id]
            continue
        if path_is_hard_denied(file_path):
            target_triages[target_id] = TargetTriage(
                target_id=target_id,
                file_path=file_path,
                risk_score=PRIORITY_MAX_RISK,
                risk_tier="high_critical",
                hard_denied=True,
            )
            continue
        scored_target_ids.append((target_id, file_path))
        items.append(({"path": file_path, "diff": diff_text}, TARGET_TRIAGE_QUESTIONS))

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

    outcomes = await client.evaluate_many(
        items,
        phase=DaydreamPhase.REVIEW,
        session_id=session_id,
        record_usage=True,
    )

    cursor = 0
    for target_id, file_path in scored_target_ids:
        target_triages[target_id] = _triage_from_outcome(target_id, file_path, outcomes[cursor])
        cursor += 1
    target_scores: dict[str, float] = {
        target_id: triage.risk_score
        for target_id, triage in target_triages.items()
        if not triage.fallback
    }

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
        if not git_observed_path_is_confined(repo, candidate.path):
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
        target_risk_scores=target_scores,
        historical_context=historical_hint,
        supporting_files_by_group=supporting_by_group,
        target_triages=target_triages if target_scores else {},
    )


# --- Pre-challenge claim triage, duplicate clustering, and 2-tier arbiter routing (#1972) ---

SeverityLevel = Literal["high", "medium", "low"]
CLAIM_ENCLOSING_RADIUS_LINES = 30
CLAIM_ENCLOSING_MAX_BYTES = 4096
CLAIM_DIFF_EXCERPT_MAX_BYTES = 6144
CLAIM_DUPLICATE_THRESHOLD = 0.80
MAX_DUPLICATE_PAIRS = 16
FAST_CHALLENGE_CALL_CAP = 6


@dataclass(frozen=True)
class ClaimTriage:
    """Calibrated System 1 pre-challenge triage for one pending claim revision."""

    uid: str
    revision: int
    p_invalid: float | None = None
    p_needs_repo_search: float | None = None
    calibrated_severity: SeverityLevel | None = None
    injection_flagged: bool = False
    fallback: bool = False
    model: str | None = None
    cited_diff: str = ""
    enclosing_source: str = ""

    def qualifies_for_fast_reject(self, *, reject_threshold: float, search_max: float) -> bool:
        """Whether this claim qualifies for opt-in asymmetric host fast-rejection (never confirmation)."""
        if self.injection_flagged or self.fallback:
            return False
        if self.p_invalid is None or self.p_invalid < reject_threshold:
            return False
        return self.p_needs_repo_search is not None and self.p_needs_repo_search <= search_max

    def qualifies_for_fast_challenge(self, *, search_max: float) -> bool:
        """Whether this self-contained local claim can use the fast/low-effort arbiter tier."""
        if self.injection_flagged or self.fallback:
            return False
        return self.p_needs_repo_search is not None and self.p_needs_repo_search <= search_max


@dataclass(frozen=True)
class ChallengeTriagePlan:
    """Host System 1 pre-challenge triage, duplicate clusters, and 2-tier routing plan."""

    triages: Mapping[str, ClaimTriage] = field(default_factory=dict)
    duplicate_clusters: tuple[tuple[str, ...], ...] = ()

    @classmethod
    def empty(cls) -> ChallengeTriagePlan:
        return cls(triages={}, duplicate_clusters=())

    def calibrated_severities(self) -> dict[str, str]:
        return {
            uid: triage.calibrated_severity
            for uid, triage in self.triages.items()
            if not triage.injection_flagged and not triage.fallback and triage.calibrated_severity is not None
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
        severities = {
            uid: triage.calibrated_severity
            for uid in claim_uids
            if (triage := self.triages.get(uid)) is not None
            and not triage.injection_flagged
            and not triage.fallback
            and triage.calibrated_severity is not None
        }
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
    if not git_observed_path_is_confined(repo, file_path):
        return ""
    full = repo / file_path
    try:
        if full.is_symlink() or not full.is_file():
            return ""
        raw = full.read_bytes()
    except OSError:
        return ""
    if b"\x00" in raw:
        return ""
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
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
    uid = str(claim["uid"])
    revision = int(claim["revision"])
    if not outcome.ok or outcome.response is None:
        return ClaimTriage(
            uid=uid,
            revision=revision,
            fallback=True,
            cited_diff=cited_diff,
            enclosing_source=enclosing_source,
        )
    model_name = outcome.response.model
    if injection_detected(outcome):
        return ClaimTriage(
            uid=uid,
            revision=revision,
            injection_flagged=True,
            model=model_name,
            cited_diff=cited_diff,
            enclosing_source=enclosing_source,
        )
    sev_ans = outcome.choice("severity")
    raw_sev = sev_ans.choice if sev_ans is not None else None
    calibrated: SeverityLevel | None
    if raw_sev == "high":
        calibrated = "high"
    elif raw_sev == "medium":
        calibrated = "medium"
    elif raw_sev == "low":
        calibrated = "low"
    else:
        calibrated = None
    return ClaimTriage(
        uid=uid,
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
    if not pending_claims or not client.config.enabled or not client.config.api_key:
        return ChallengeTriagePlan.empty()
    if remaining_s is not None and remaining_s <= 0:
        return ChallengeTriagePlan.empty()
    client = _clamp_client_timeout(client, remaining_s)

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
    for file_claims in by_file.values():
        if len(file_claims) < 2:
            continue
        for i in range(len(file_claims)):
            for j in range(i + 1, len(file_claims)):
                if len(dup_pairs) >= MAX_DUPLICATE_PAIRS:
                    break
                left, right = file_claims[i], file_claims[j]
                dup_pairs.append((str(left["uid"]), str(right["uid"])))
                items.append((
                    {
                        "file": left["payload"]["file"],
                        "claim_a": {
                            "title": left.get("title", ""),
                            "line": left["payload"].get("line"),
                            "claim": left["payload"].get("claim", ""),
                            "trigger": left["payload"].get("trigger", ""),
                        },
                        "claim_b": {
                            "title": right.get("title", ""),
                            "line": right["payload"].get("line"),
                            "claim": right["payload"].get("claim", ""),
                            "trigger": right["payload"].get("trigger", ""),
                        },
                    },
                    CLAIM_DUPLICATE_QUESTIONS,
                ))
            if len(dup_pairs) >= MAX_DUPLICATE_PAIRS:
                break

    outcomes = await client.evaluate_many(
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
