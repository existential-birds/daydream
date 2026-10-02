from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from daydream.backends import ContinuationToken
from daydream.deep.detection import StackAssignment
from daydream.deep.diff import DeepDiffBoundInfo
from daydream.deep.fix_state import FixCycleState, RetainedTreeSnapshot
from daydream.deep.records import RecordPool
from daydream.deep.state import DeepState
from daydream.phases import PushReceipt
from tests.harness.review_result import review_coverage


def test_state_is_a_live_view_over_the_original_mapping(tmp_path: Path) -> None:
    original_items = [{"id": 1}]
    original = {"items": original_items, "items_file": tmp_path / "first.json"}
    state = DeepState(original)
    assert state.items is original_items
    extension_items = [{"id": 2, "extension_field": object()}]
    extension_path = tmp_path / "rewritten.json"
    original["items"] = extension_items
    original["items_file"] = extension_path
    assert state.items is extension_items
    assert state.items_file is extension_path
    built_in_items = [{"id": 3}]
    state.items = built_in_items
    assert original["items"] is built_in_items

def test_required_and_optional_accessors_preserve_missing_key_semantics() -> None:
    state = DeepState({})
    with pytest.raises(KeyError, match="diff_path"):
        _ = state.diff_path
    assert state.diff_path_or_none is None
    with pytest.raises(KeyError, match="exploration_dir"):
        _ = state.exploration_dir
    assert state.exploration_dir_or_none is None
    with pytest.raises(KeyError, match="merged_report"):
        _ = state.merged_report
    assert state.merged_report_or_none is None
    with pytest.raises(KeyError, match="record_pool"):
        _ = state.record_pool

def test_required_nullable_paths_accept_a_published_none() -> None:
    state = DeepState({"exploration_dir": None, "record_pool": RecordPool({}, {})})
    assert state.exploration_dir is None
    assert state.record_pool.structural_path is None

def test_documented_defaults_match_current_flow_reads() -> None:
    state = DeepState({})

    assert state.mode == "loop"
    assert state.diff_or_empty == ""
    assert state.intent_authoritative is False
    assert state.diff_truncated is False
    assert state.diagrams is None
    assert state.import_graph == {}
    assert state.arbiter_continuation is None
    assert state.iteration is None
    assert state.fix_outcomes == {}
    assert state.fix_round_snapshot is None
    assert state.push_receipt is None
    assert state.items_or_empty == []
    with pytest.raises(KeyError, match="review_coverage"):
        _ = state.unfinished_scopes

    with pytest.raises(RuntimeError, match="fix cycle was not initialized at the accepted gate"):
        _ = state.fix_cycle_state

def test_mode_preserves_existing_string_conversion() -> None:
    assert DeepState({"mode": None}).mode == "None"
    assert DeepState({"mode": 3}).mode == "3"

@pytest.mark.parametrize(("key", "property_name", "value", "expected_type"),
    [("diff", "diff", 1, "str"), ("diff", "diff_or_empty", 1, "str"),
        ("diff_path", "diff_path", "diff.patch", "Path"), ("tier", "tier", "large", "one of: parallel, single, skip"),
        ("exploration_dir", "exploration_dir", "exploration", "Path or None"),
        ("items_file", "items_file", "items.json", "Path"), ("items", "items", {}, "list"),
        ("diagrams", "diagrams", [], "dict or None"), ("import_graph", "import_graph", [], "dict"),
        ("intent_authoritative", "intent_authoritative", 1, "bool"), ("changed_files", "changed_files", [], "set"),
        ("stacks", "stacks", {}, "list"), ("review_coverage", "unfinished_scopes", [], "ReviewCoverage"),
        ("record_pool", "record_pool", {}, "RecordPool"), ("iteration", "iteration", "1", "int or None"),
        ("diff_truncation", "diff_truncation", {}, "DeepDiffBoundInfo or None",),
        ("arbiter_continuation", "arbiter_continuation", {}, "ContinuationToken or None",),
    ],
)
def test_accessors_reject_wrong_outer_types_at_access(key: str, property_name: str, value: object, expected_type: str,
) -> None:
    original = {key: value}
    state = DeepState(original)
    with pytest.raises(TypeError, match=rf"deep state key '{key}' expected {expected_type}, got {type(value).__name__}",
    ):
        getattr(state, property_name)
    assert original[key] is value

def test_valid_empty_mutable_containers_are_returned_by_reference() -> None:
    records: list[dict[str, Any]] = []
    pool = RecordPool({"python": {"issues": records}}, {})
    import_graph: dict[str, set[str]] = {}
    state = DeepState({"record_pool": pool, "import_graph": import_graph})
    assert state.record_pool is pool and state.import_graph is import_graph
    records.append({"uid": "python:1"})
    assert state.record_pool.language == records
    state.record_pool.replace([])
    assert pool.language == []


def test_concrete_value_types_are_checked_lazily(tmp_path: Path) -> None:
    stack = StackAssignment("python", ["app.py"])
    truncation = DeepDiffBoundInfo(truncated=True)
    continuation = ContinuationToken("codex", {"thread_id": "abc"})
    receipt = PushReceipt("origin", "feature", "a" * 40, "owner/repo")
    state = DeepState({"stacks": [stack], "diff_truncation": truncation, "arbiter_continuation": continuation,
            "push_receipt": receipt, "dd": tmp_path,
        }
    )
    assert state.stacks == [stack]
    assert state.diff_truncation is truncation
    assert state.arbiter_continuation is continuation
    assert state.push_receipt is receipt
    assert state.dd is tmp_path

def test_wrong_push_receipt_preserves_remote_ci_disabled_semantics() -> None:
    assert DeepState({"push_receipt": object()}).push_receipt is None

def test_wrong_fix_cycle_state_preserves_uninitialized_gate_error() -> None:
    state = DeepState({"fix_cycle_state": object()})
    with pytest.raises(RuntimeError, match="fix cycle was not initialized at the accepted gate"):
        _ = state.fix_cycle_state

def test_wrong_retained_snapshot_preserves_recapture_semantics() -> None:
    assert DeepState({"fix_round_snapshot": object()}).fix_round_snapshot is None

def test_fix_dto_accessors_return_the_published_instances() -> None:
    cycle = object.__new__(FixCycleState)
    snapshot = object.__new__(RetainedTreeSnapshot)
    state = DeepState({"fix_cycle_state": cycle, "fix_round_snapshot": snapshot})
    assert state.fix_cycle_state is cycle
    assert state.fix_round_snapshot is snapshot

def test_property_setters_publish_to_the_existing_keys(tmp_path: Path) -> None:
    original: dict[str, Any] = {}
    state = DeepState(original)
    pool = RecordPool({}, {})
    outcomes = {"python:1": {"verdict": "fixed"}}
    snapshot: Any = object()
    cycle: Any = object()

    state.exploration_dir = tmp_path / "exploration"
    state.intent_authoritative = True
    state.record_pool = pool
    state.fix_outcomes = outcomes
    state.fix_round_snapshot = snapshot
    state.fix_cycle_state = cycle

    assert original == {"exploration_dir": tmp_path / "exploration", "intent_authoritative": True, "record_pool": pool,
        "fix_outcomes": outcomes, "fix_round_snapshot": snapshot, "fix_cycle_state": cycle,
    }


def test_unfinished_scopes_derive_from_live_checked_coverage() -> None:
    coverage = review_coverage(scope_ids=('python',))
    state = DeepState({'review_coverage': coverage})
    assert state.unfinished_scopes == {'python': 'coverage_unknown'}
    coverage.record_scope('python', 'failed', reasons=('backend_failure',), diagnostic='provider unavailable')
    assert state.unfinished_scopes == {'python': 'provider unavailable'}
    coverage.record_scope('python', 'complete')
    assert state.unfinished_scopes == {}
