from __future__ import annotations

import pytest

from daydream.deep.state import DeepState


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

def test_wrong_push_receipt_preserves_remote_ci_disabled_semantics() -> None:
    assert DeepState({"push_receipt": object()}).push_receipt is None

def test_wrong_fix_cycle_state_preserves_uninitialized_gate_error() -> None:
    state = DeepState({"fix_cycle_state": object()})
    with pytest.raises(RuntimeError, match="fix cycle was not initialized at the accepted gate"):
        _ = state.fix_cycle_state

def test_wrong_retained_snapshot_preserves_recapture_semantics() -> None:
    assert DeepState({"fix_round_snapshot": object()}).fix_round_snapshot is None
