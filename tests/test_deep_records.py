"""Pin referential identities used by dedup, arbitration, and record rewriting.

Real-path pipeline coverage lives in tests/deep_orchestrator/."""

from __future__ import annotations

from pathlib import Path

import pytest

from daydream.deep.records import (
    RecordPool,
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


def test_union_drops_empty_and_non_string_entries() -> None:
    assert union_source_uids(["python:1", "", None, 3], []) == ["python:1"]
