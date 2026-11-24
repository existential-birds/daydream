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
    NoulQuestion,
    ScoreQuestion,
    System1Client,
    injection_detected,
    path_is_hard_denied,
    with_injection_guard,
)
from daydream.trajectory import DaydreamPhase
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
    """Advisory System 1 pre-dispatch hints and target risk priority scores."""

    target_risk_scores: Mapping[str, float]
    historical_context: dict[str, Any] | None = None
    supporting_files_by_group: Mapping[frozenset[str], tuple[dict[str, Any], ...]] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> ReviewPrefetchHints:
        return cls(target_risk_scores={}, historical_context=None, supporting_files_by_group={})

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


def _score_target_outcome(file_path: str, outcome: DecisionOutcome) -> float | None:
    if path_is_hard_denied(file_path):
        return PRIORITY_MAX_RISK
    if not outcome.ok:
        return None
    if injection_detected(outcome):
        return PRIORITY_MAX_RISK
    risk = outcome.score("defect_risk")
    if risk is not None:
        return risk.score
    non_behavioral = outcome.noul("non_behavioral")
    if non_behavioral is not None:
        return (1.0 - non_behavioral) * 4.0
    return None


async def prepare_review_prefetch_hints(
    *,
    client: System1Client,
    repo: Path,
    target_diffs: Mapping[str, tuple[str, str]],
    groups: Sequence[tuple[str, ...]],
    historical_fields: Mapping[str, bytes],
    session_id: str | None = None,
    remaining_s: float | None = None,
) -> ReviewPrefetchHints:
    """Evaluate targets, historical context, and supporting files in one parallel System 1 pass."""
    if not client.config.enabled or not client.config.api_key:
        return ReviewPrefetchHints.empty()
    if remaining_s is not None and remaining_s <= 0:
        return ReviewPrefetchHints.empty()

    if remaining_s is not None and remaining_s < client.config.batch_timeout_s:
        effective_timeout = min(client.config.timeout_s, remaining_s)
        client = System1Client(
            replace(client.config, timeout_s=effective_timeout, batch_timeout_s=remaining_s),
            transport=client._transport,
        )

    # Discover confined supporting candidates off the event loop so timers/host tools stay responsive.
    group_files_and_excerpts = [_group_diff_excerpt(group, target_diffs) for group in groups]
    candidates_by_group = await anyio.to_thread.run_sync(
        lambda: [collect_supporting_candidates(repo, files) for files, _ in group_files_and_excerpts]
    )

    items: list[tuple[Any, Mapping[str, DecisionQuestion]]] = []
    # 1. Target risk scoring items (hard-denied targets get max priority deterministically without spending calls).
    scored_target_ids: list[tuple[str, str]] = []
    target_scores: dict[str, float] = {}
    for target_id, (file_path, diff_text) in target_diffs.items():
        if path_is_hard_denied(file_path):
            target_scores[target_id] = PRIORITY_MAX_RISK
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
        score = _score_target_outcome(file_path, outcomes[cursor])
        cursor += 1
        if score is not None:
            target_scores[target_id] = score

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
    )
