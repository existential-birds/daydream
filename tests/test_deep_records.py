"""Pin referential identities used by dedup, arbitration, and record rewriting.

Real-path pipeline coverage lives in tests/deep_orchestrator/."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from daydream.deep.records import (
    ITEM_UID_KEY,
    RECORD_UID_KEY,
    RecordPool,
    duplicate_record_uids,
    item_source_uids,
    item_uid,
    record_uid,
    stack_name_from_uid,
    stamp_item_uids,
    stamp_record_uids,
    union_source_uids,
)
from tests.harness.review_result import records_artifact, review_coverage


def test_record_pool_retains_scope_order_and_metadata_through_adjudication(tmp_path: Path) -> None:
    coverage = review_coverage(scope_ids=("structure", "python", "react"))
    records = [{"uid": f"{scope}:1"} for scope in ("structure", "python", "react")]
    paths = {scope: tmp_path / f"stack-{scope}-records.json" for scope in ("structure", "python", "react")}
    pool = RecordPool({scope: records_artifact(coverage, scope, [record])
                       for scope, record in zip(paths, records, strict=True)}, paths)
    assert pool.records == [records[1], records[2], records[0]]
    pool.replace([records[1], records[0]])
    pool.save()
    assert pool.language == [records[1]] and pool.structural == [records[0]]
    assert pool.scopes["react"]["issues"] == []
    assert all(value["originating_run_id"] == coverage.run_id for value in pool.scopes.values())
    with pytest.raises(ValueError, match="scope"):
        pool.replace([{"uid": "go:1"}])


def test_stack_name_from_uid_is_empty_for_a_separatorless_value() -> None:
    """An empty result is the "no identity" signal, distinguishable from a real name."""
    assert stack_name_from_uid("python") == ""
    assert stack_name_from_uid("") == ""

def test_stack_name_from_uid_splits_on_the_last_separator() -> None:
    """A trailing ordinal leaves any separators within the stack name intact."""
    assert stack_name_from_uid("weird:name:7") == "weird:name"

def test_stamp_assigns_sequential_uids_in_place() -> None:
    """Existing list holders see assigned durable identities without rebinding."""
    records: list[dict[str, Any]] = [{"id": 1}, {"id": 1}, {"id": 2}]
    stamp_record_uids(records, "python")
    assert [record[RECORD_UID_KEY] for record in records] == ["python:1", "python:2", "python:3"]

def test_stamp_preserves_an_existing_uid() -> None:
    """Resume preserves identities even after adjudication compacts their source list."""
    records: list[dict[str, Any]] = [{"id": 9, RECORD_UID_KEY: "python:4"}]
    stamp_record_uids(records, "python")
    assert record_uid(records[0]) == "python:4"

def test_stamp_is_idempotent() -> None:
    """Called at both record birth and record load, the second call must be a no-op."""
    records: list[dict[str, Any]] = [{"id": 1}, {"id": 2}]
    stamp_record_uids(records, "python")
    first = [record_uid(record) for record in records]
    stamp_record_uids(records, "python")
    assert [record_uid(record) for record in records] == first

def test_partial_stamp_does_not_reuse_an_ordinal_already_held() -> None:
    """Mixed stamped/unstamped records cannot receive colliding ordinals."""
    records: list[dict[str, Any]] = [{"id": 1, RECORD_UID_KEY: "python:1"}, {"id": 2}]
    stamp_record_uids(records, "python")
    assert [record_uid(record) for record in records] == ["python:1", "python:2"]
    assert duplicate_record_uids(records) == []

def test_item_source_uids_ignores_malformed_entries() -> None:
    """A hallucinated non-string entry must not reach a consumer as provenance."""
    item: dict[str, Any] = {"source_uids": ["python:1", 7, None, "", "react:1"]}
    assert item_source_uids(item) == ["python:1", "react:1"]

def test_item_source_uids_ignores_a_non_list_value() -> None:
    """A scalar where a list belongs degrades to the uid fallback, never raises."""
    item: dict[str, Any] = {"source_uids": "python:1", RECORD_UID_KEY: "structure:2"}
    assert item_source_uids(item) == ["structure:2"]

def test_union_drops_empty_and_non_string_entries() -> None:
    assert union_source_uids(["python:1", "", None, 3], []) == ["python:1"]

def test_stamp_skips_an_ordinal_a_preserved_record_already_holds() -> None:
    """Minting skips ordinals held by preserved records, even at displaced positions."""
    records: list[dict[str, Any]] = [{RECORD_UID_KEY: "python:2"}, {"id": 1}, {"id": 2}]
    stamp_record_uids(records, "python")
    assert [record_uid(record) for record in records] == ["python:2", "python:1", "python:3"]
    assert duplicate_record_uids(records) == []

def test_stamp_item_uids_skips_an_ordinal_a_preserved_item_already_holds() -> None:
    """Same hazard, same guarantee, on the merged-item side."""
    items: list[dict[str, Any]] = [{ITEM_UID_KEY: "item:2"}, {"id": 1}, {"id": 2}]
    stamp_item_uids(items)
    assert [item_uid(item) for item in items] == ["item:2", "item:1", "item:3"]
    assert len({item_uid(item) for item in items}) == len(items)
