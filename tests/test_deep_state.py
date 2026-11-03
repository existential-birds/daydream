from __future__ import annotations

import pytest

from daydream.deep.state import DeepState


def test_accessors_preserve_missing_key_semantics_and_documented_defaults() -> None:
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
    # A wrong push receipt preserves remote-CI-disabled semantics.
    assert DeepState({"push_receipt": object()}).push_receipt is None


def test_accessors_reject_wrong_outer_types_at_access() -> None:
    for key, value, expected_type in [("tier", "large", "one of: parallel, single, skip"),
                                      ("diff_truncation", {}, "DeepDiffBoundInfo or None")]:
        original = {key: value}
        state = DeepState(original)
        with pytest.raises(TypeError,
                           match=rf"deep state key '{key}' expected {expected_type}, got {type(value).__name__}"):
            getattr(state, key)
        assert original[key] is value
