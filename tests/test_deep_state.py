"""The native flow admits public inputs into its single live extension dictionary."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from daydream.deep.fix_state import FixCycleState
from daydream.deep.orchestrator import _mode_of, _remote_ci_enabled
from daydream.deep.records import RecordPool
from daydream.extensions import Registry
from daydream.flows.engine import FlowContext
from daydream.phases import PushReceipt
from daydream.run_config import RunConfig
from daydream.workspace import WorkContext
from tests.harness.review_result import review_coverage


def _context(data: dict[str, Any]) -> FlowContext:
    root = Path(".")
    work = WorkContext(
        repo=root, source=root, base_branch="main", base_sha="0" * 40,
        head_branch="feature", head_sha="1" * 40, is_ephemeral=False, run_id="test-state",
    )
    return FlowContext(RunConfig(), work, Registry(), data=data)


def test_native_consumption_preserves_the_original_extension_dictionary(tmp_path: Path) -> None:
    original_items = [{"id": 1}]
    original = {"items": original_items, "items_file": tmp_path / "first.json"}
    ctx = _context(original)
    admitted = ctx.deep_data()
    assert admitted is original
    assert admitted["items"] is original_items
    extension_items = [{"id": 2, "extension_field": object()}]
    extension_path = tmp_path / "rewritten.json"
    original["items"] = extension_items
    original["items_file"] = extension_path
    assert ctx.deep_data() is admitted
    assert admitted["items"] is extension_items
    assert admitted["items_file"] is extension_path
    built_in_items = [{"id": 3}]
    admitted["items"] = built_in_items
    assert original["items"] is built_in_items


def test_native_admission_does_not_fabricate_missing_state() -> None:
    original: dict[str, Any] = {}
    admitted = _context(original).deep_data()
    assert admitted is original and original == {}
    for key in ("diff_path", "exploration_dir", "merged_report", "record_pool"):
        with pytest.raises(KeyError, match=key):
            admitted[key]
    assert admitted.get("diagrams") is None
    assert admitted.get("intent_authoritative", False) is False
    assert admitted.get("items", []) == []


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("diff", 1), ("diff", None), ("diff_path", "diff.patch"),
        ("tier", "large"), ("exploration_dir", "exploration"),
        ("intent_path", "intent.md"), ("alts_path", 1),
        ("items_file", "items.json"), ("items", {}), ("diagrams", []),
        ("import_graph", []), ("intent_authoritative", 1),
    ],
)
def test_native_consumption_rejects_invalid_public_inputs_without_mutation(key: str, value: object) -> None:
    original = {key: value}
    with pytest.raises(TypeError, match=rf"deep state key '{key}' expected"):
        _context(original).deep_data()
    assert original == {key: value}


def test_nullable_advisory_fields_remain_unmodified() -> None:
    original = {
        "exploration_dir": None, "diagrams": None,
        "import_graph": None, "intent_authoritative": None,
    }
    assert _context(original).deep_data() is original
    assert all(value is None for value in original.values())


def test_concrete_native_owners_and_empty_public_containers_are_not_projected() -> None:
    coverage = review_coverage(scope_ids=("python",))
    records: list[dict[str, Any]] = []
    pool = RecordPool({"python": {"issues": records}}, {})
    import_graph: dict[str, set[str]] = {}
    original = {"record_pool": pool, "review_coverage": coverage, "import_graph": import_graph}
    admitted = _context(original).deep_data()
    assert admitted["record_pool"] is pool
    assert admitted["review_coverage"] is coverage
    assert admitted["import_graph"] is import_graph
    coverage.record_scope("python", "complete")
    assert not admitted["review_coverage"].unfinished_scopes
    records.append({"uid": "python:1"})
    assert admitted["record_pool"].language == records


@pytest.mark.parametrize(("mode", "expected"), [(None, "None"), (3, "3"), ("review", "review")])
def test_native_mode_gate_retains_string_conversion(mode: object, expected: str) -> None:
    assert _mode_of(_context({"mode": mode})) == expected
    assert _mode_of(_context({})) == "loop"


def test_remote_ci_admission_requires_the_actual_push_receipt() -> None:
    ctx = _context({"push_receipt": object()})
    assert not _remote_ci_enabled(ctx)
    receipt = PushReceipt("origin", "feature", "a" * 40, "owner/repo")
    ctx.data["push_receipt"] = receipt
    assert _remote_ci_enabled(ctx)


@pytest.mark.parametrize("value", [None, object()])
def test_fix_cycle_capability_is_still_checked_at_its_consumer(value: object) -> None:
    with pytest.raises(RuntimeError, match="fix cycle was not initialized at the accepted gate"):
        FixCycleState.require(_context({"fix_cycle_state": value}))
    cycle = object.__new__(FixCycleState)
    assert FixCycleState.require(_context({"fix_cycle_state": cycle})) is cycle


async def test_fix_verification_recaptures_an_invalid_retained_snapshot_from_real_git(tmp_path: Path) -> None:
    from daydream.deep.fix_state import RetainedTreeSnapshot
    from daydream.deep.fix_steps import _step_fix_verify
    from daydream.extensions import BreakLoop
    from tests.deep_orchestrator.support import _base_repo, _direct_fix_context, _direct_fix_state

    repo = _base_repo(tmp_path, "snapshot-recapture")
    ctx = _direct_fix_context(repo, [], changed_files={"a.py"})
    state = _direct_fix_state(ctx, [], {"a.py"})
    (repo / "a.py").write_text("A = 2\n")
    ctx.data["fix_round_snapshot"] = object()
    assert isinstance(await _step_fix_verify(ctx), BreakLoop)
    assert isinstance(state.latest_retained, RetainedTreeSnapshot)
    assert state.latest_retained.paths == frozenset({"a.py"})
    assert "+A = 2" in state.latest_retained.verifier_patch
    assert (repo / "a.py").read_text() == "A = 2\n"
