"""Static shape of the native deep flow's canonical extension dictionary."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict

if TYPE_CHECKING:
    from daydream.backends import ContinuationToken
    from daydream.deep.detection import StackAssignment
    from daydream.deep.diff import DeepDiffBoundInfo
    from daydream.deep.fix_state import FixCycleState
    from daydream.deep.latency import ArbiterPlan, LatencyRoute, RiskSummary
    from daydream.deep.records import RecordPool
    from daydream.deep.reuse_store import ReuseCache
    from daydream.exploration_runner import Tier
    from daydream.phases import PushReceipt
    from daydream.review_result import ReviewCoverage
    from daydream.test_execution import TestRecipe


class DeepData(TypedDict, total=False):
    """Built-in producer types; extensions share the same dictionary, never a projection."""

    review_coverage: ReviewCoverage
    mode: str
    diff: str
    diff_path: Path
    diff_truncated: bool
    diff_truncation: DeepDiffBoundInfo
    tier: Tier
    exploration_dir: Path | None
    intent_path: Path
    alts_path: Path
    items_file: Path
    items: list[dict[str, Any]]
    diagrams: dict[str, Any] | None
    import_graph: dict[str, set[str]]
    intent_authoritative: bool
    changed_files: set[str]
    dd: Path
    stacks: list[StackAssignment]
    single_stack_mode: bool
    latency_route: LatencyRoute
    reuse_cache: ReuseCache
    risk_summary: RiskSummary
    arbiter_plan: ArbiterPlan
    log: str
    branch: str
    intent_summary: str
    record_pool: RecordPool
    arbiter_continuation: ContinuationToken
    merged_report: Path
    fix_cycle_state: FixCycleState
    test_recipe: TestRecipe
    iteration: int
    push_receipt: PushReceipt


def validate_extension_inputs(data: dict[str, Any]) -> None:
    """Reject invalid public inputs at native consumption, without copying or changing state.

    Internal values are owned by their producers. Only the documented extension
    fields cross this boundary; receipt and fix-capability checks remain with their
    actual consumers. Nullable advisory fields retain their previous absence policy.
    """
    expected: type[object]
    for key, expected in (
        ("diff", str),
        ("diff_path", Path),
        ("intent_path", Path),
        ("alts_path", Path),
        ("items_file", Path),
        ("items", list),
    ):
        if key in data and not isinstance(data[key], expected):
            raise TypeError(
                f"deep state key '{key}' expected {expected.__name__}, got {type(data[key]).__name__}"
            )
    for key, expected in (
        ("exploration_dir", Path),
        ("diagrams", dict),
        ("import_graph", dict),
        ("intent_authoritative", bool),
    ):
        value = data.get(key)
        if value is not None and not isinstance(value, expected):
            raise TypeError(
                f"deep state key '{key}' expected {expected.__name__} or None, got {type(value).__name__}"
            )
    if "tier" in data and data["tier"] not in ("parallel", "single", "skip"):
        raise TypeError("deep state key 'tier' expected one of: parallel, single, skip")
