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
    pool = RecordPool(
        {scope: records_artifact(coverage, scope, [record]) for scope, record in zip(paths, records, strict=True)},
        paths,
    )
    assert pool.records == [records[1], records[2], records[0]]
    pool.replace([records[1], records[0]])
    pool.save()
    assert pool.language == [records[1]] and pool.structural == [records[0]]
    assert pool.scopes["react"]["issues"] == []
    assert all(value["originating_run_id"] == coverage.run_id for value in pool.scopes.values())
    with pytest.raises(ValueError, match="scope"):
        pool.replace([{"uid": "go:1"}])


def test_record_birth_and_resume_preserve_unique_durable_identity(subtests: pytest.Subtests) -> None:
    """Birth, partial resume, and reloading share the host ordinal allocator."""
    cases: list[tuple[list[dict[str, Any]], list[str]]] = [
        ([{"id": 1}, {"id": 1}, {"id": 2}], ["python:1", "python:2", "python:3"]),
        ([{"id": 9, RECORD_UID_KEY: "python:4"}], ["python:4"]),
        ([{"id": 1}, {"id": 2}], ["python:1", "python:2"]),
        ([{"id": 1, RECORD_UID_KEY: "python:1"}, {"id": 2}], ["python:1", "python:2"]),
        ([{RECORD_UID_KEY: "python:2"}, {"id": 1}, {"id": 2}], ["python:2", "python:1", "python:3"]),
    ]
    for records, expected in cases:
        with subtests.test(expected=expected):
            stamp_record_uids(records, "python")
            assert [record_uid(record) for record in records] == expected
            assert duplicate_record_uids(records) == []
            # A second load cannot remint or compact an existing identity.
            stamp_record_uids(records, "python")
            assert [record_uid(record) for record in records] == expected
    for uid, expected_stack in (("python", ""), ("", ""), ("weird:name:7", "weird:name")):
        with subtests.test(uid=uid):
            assert stack_name_from_uid(uid) == expected_stack


def test_item_provenance_filters_malformed_values_and_falls_back_to_record_uid(subtests: pytest.Subtests) -> None:
    cases: list[tuple[dict[str, Any], list[str]]] = [
        ({"source_uids": ["python:1", 7, None, "", "react:1"]}, ["python:1", "react:1"]),
        ({"source_uids": "python:1", RECORD_UID_KEY: "structure:2"}, ["structure:2"]),
    ]
    for item, expected in cases:
        with subtests.test(item=item):
            assert item_source_uids(item) == expected
    assert union_source_uids(["python:1", "", None, 3], []) == ["python:1"]


def test_stamp_item_uids_skips_an_ordinal_a_preserved_item_already_holds() -> None:
    """Same hazard, same guarantee, on the merged-item side."""
    items: list[dict[str, Any]] = [{ITEM_UID_KEY: "item:2"}, {"id": 1}, {"id": 2}]
    stamp_item_uids(items)
    assert [item_uid(item) for item in items] == ["item:2", "item:1", "item:3"]
    assert len({item_uid(item) for item in items}) == len(items)
