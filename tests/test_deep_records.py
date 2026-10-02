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
    mint_item_uid,
    mint_record_uid,
    record_uid,
    stack_name_from_records_source,
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


def test_uid_format_is_stack_name_and_one_based_ordinal() -> None:
    """The documented, debuggable format -- readable in artifacts, not a uuid4."""
    assert mint_record_uid("python", 1) == "python:1"
    assert mint_record_uid("structure", 3) == "structure:3"

def test_source_normalization_accepts_both_spellings() -> None:
    """Disk filenames and bare stack names route to the same record identity."""
    assert stack_name_from_records_source("stack-python-records.json") == "python"
    assert stack_name_from_records_source("stack-structure-records.json") == "structure"
    assert stack_name_from_records_source("python") == "python"

def test_unrecognized_source_shape_passes_through_unchanged() -> None:
    """Preserve unroutable source names so callers can report them instead of losing records."""
    assert stack_name_from_records_source("something-else.json") == "something-else.json"

def test_stack_name_recovered_from_uid() -> None:
    assert stack_name_from_uid("python:1") == "python"
    assert stack_name_from_uid("structure:12") == "structure"

def test_stack_name_from_uid_is_empty_for_a_separatorless_value() -> None:
    """An empty result is the "no identity" signal, distinguishable from a real name."""
    assert stack_name_from_uid("python") == ""
    assert stack_name_from_uid("") == ""

def test_stack_name_from_uid_splits_on_the_last_separator() -> None:
    """A trailing ordinal leaves any separators within the stack name intact."""
    assert stack_name_from_uid("weird:name:7") == "weird:name"

def test_record_uid_reads_the_field() -> None:
    assert record_uid({RECORD_UID_KEY: "python:2"}) == "python:2"

def test_record_uid_is_empty_for_a_record_without_one() -> None:
    """Merged items may lack pre-merge identity; accessors return the empty sentinel."""
    assert record_uid({"id": 1, "file": "api.py"}) == ""

def test_record_uid_is_empty_for_a_non_string_value() -> None:
    """A malformed artifact must degrade to "no identity", not leak an int."""
    assert record_uid({RECORD_UID_KEY: 7}) == ""

def test_stamp_assigns_sequential_uids_in_place() -> None:
    """Existing list holders see assigned durable identities without rebinding."""
    records: list[dict[str, Any]] = [{"id": 1}, {"id": 1}, {"id": 2}]
    stamp_record_uids(records, "python")
    assert [record[RECORD_UID_KEY] for record in records] == ["python:1", "python:2", "python:3"]

def test_stamp_disambiguates_records_the_reviewer_numbered_identically() -> None:
    """The whole point: ``id`` restarts at 1 per stack, so ``id: 1`` is the norm."""
    python_records: list[dict[str, Any]] = [{"id": 1, "file": "api.py"}]
    react_records: list[dict[str, Any]] = [{"id": 1, "file": "api.py"}]
    stamp_record_uids(python_records, "python")
    stamp_record_uids(react_records, "react")
    assert record_uid(python_records[0]) != record_uid(react_records[0])

def test_stamp_accepts_a_records_filename_as_the_stack_name() -> None:
    """The load path passes ``records_path.name``; it must not leak into the uid."""
    records: list[dict[str, Any]] = [{"id": 1}]
    stamp_record_uids(records, "stack-python-records.json")
    assert record_uid(records[0]) == "python:1"

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

def test_stamp_handles_an_empty_list() -> None:
    records: list[dict[str, Any]] = []
    stamp_record_uids(records, "python")
    assert records == []

def test_independent_producer_batches_mint_identical_record_uids() -> None:
    """The same producer inventory deterministically mints the same identities."""
    at_birth: list[dict[str, Any]] = [{"id": 1}, {"id": 2}, {"id": 3}]
    stamp_record_uids(at_birth, "python")
    reloaded_without_uids: list[dict[str, Any]] = [{"id": 1}, {"id": 2}, {"id": 3}]
    stamp_record_uids(reloaded_without_uids, "stack-python-records.json")
    assert [record_uid(r) for r in reloaded_without_uids] == [record_uid(r) for r in at_birth]

def test_duplicate_detection_reports_collisions_sorted() -> None:
    """A duplicate uid would reintroduce the over-delete, so it must be detectable."""
    records: list[dict[str, Any]] = [
        {RECORD_UID_KEY: "python:1"}, {RECORD_UID_KEY: "python:1"}, {RECORD_UID_KEY: "react:2"},
        {RECORD_UID_KEY: "react:2"}, {RECORD_UID_KEY: "python:3"},
    ]
    assert duplicate_record_uids(records) == ["python:1", "react:2"]

def test_duplicate_detection_is_empty_for_a_sound_pool() -> None:
    records: list[dict[str, Any]] = [{RECORD_UID_KEY: "python:1"}, {RECORD_UID_KEY: "react:1"}]
    assert duplicate_record_uids(records) == []

def test_duplicate_detection_ignores_records_without_a_uid() -> None:
    """Multiple absent identities are not a collision; absence is reported separately."""
    records: list[dict[str, Any]] = [{"id": 1}, {"id": 2}, {RECORD_UID_KEY: "python:1"}]
    assert duplicate_record_uids(records) == []

def test_item_source_uids_prefers_the_explicit_attribution() -> None:
    """A merged item's provenance is a list: one item may consolidate several records."""
    item: dict[str, Any] = {"id": 3, "source_uids": ["python:2", "react:2"]}
    assert item_source_uids(item) == ["python:2", "react:2"]

def test_item_source_uids_preserves_order_and_dedupes() -> None:
    item: dict[str, Any] = {"source_uids": ["python:1", "react:1", "python:1"]}
    assert item_source_uids(item) == ["python:1", "react:1"]

def test_item_source_uids_falls_back_to_the_items_own_uid() -> None:
    """Structural, bypass, and salvage items retain their own record UID as provenance."""
    item: dict[str, Any] = {"id": 4, RECORD_UID_KEY: "structure:1"}
    assert item_source_uids(item) == ["structure:1"]

def test_item_source_uids_is_empty_when_the_agent_declined_to_attribute() -> None:
    """Missing attribution remains empty rather than inventing a source record."""
    assert item_source_uids({"id": 1, "source_uids": []}) == []
    assert item_source_uids({"id": 1}) == []

def test_item_source_uids_ignores_malformed_entries() -> None:
    """A hallucinated non-string entry must not reach a consumer as provenance."""
    item: dict[str, Any] = {"source_uids": ["python:1", 7, None, "", "react:1"]}
    assert item_source_uids(item) == ["python:1", "react:1"]

def test_item_source_uids_ignores_a_non_list_value() -> None:
    """A scalar where a list belongs degrades to the uid fallback, never raises."""
    item: dict[str, Any] = {"source_uids": "python:1", RECORD_UID_KEY: "structure:2"}
    assert item_source_uids(item) == ["structure:2"]

def test_item_uid_format_is_a_one_based_ordinal() -> None:
    """Readable in artifacts and reproducible from position, like the record uid."""
    assert mint_item_uid(1) == "item:1"
    assert mint_item_uid(3) == "item:3"

def test_item_uid_reads_the_field() -> None:
    assert item_uid({ITEM_UID_KEY: "item:3"}) == "item:3"

def test_item_uid_is_empty_for_an_item_without_one() -> None:
    """The empty sentinel instructs normalization to mint an absent item identity."""
    assert item_uid({"id": 1, "file": "api.py"}) == ""

def test_item_uid_is_empty_for_a_non_string_value() -> None:
    """Malformed values must not be preserved as durable item identities."""
    assert item_uid({ITEM_UID_KEY: 7}) == ""

def test_record_and_item_identities_are_reported_independently() -> None:
    """An item may retain both its birth record UID and a distinct item UID.

    Conflating these would let an item identity masquerade as source provenance."""
    item: dict[str, Any] = {
        "id": 4, RECORD_UID_KEY: "structure:1", ITEM_UID_KEY: "item:7", "source_uids": ["python:2", "react:2"],
    }
    assert record_uid(item) == "structure:1"
    assert item_uid(item) == "item:7"
    assert item_source_uids(item) == ["python:2", "react:2"]

def test_item_uid_is_never_reported_as_provenance() -> None:
    """Source attribution may repeat when one record splits into several items.

    Its fallback may use a record UID, never the new item's identity."""
    born_as_a_record: dict[str, Any] = {"id": 4, RECORD_UID_KEY: "structure:1", ITEM_UID_KEY: "item:7"}
    synthesized: dict[str, Any] = {"id": 5, ITEM_UID_KEY: "item:8"}
    assert item_source_uids(born_as_a_record) == ["structure:1"]
    assert item_source_uids(synthesized) == []

def test_union_merges_two_provenances_survivor_first() -> None:
    """Stable survivor-first order keeps structural-fold audit provenance readable."""
    assert union_source_uids(["python:1"], ["structure:1"]) == ["python:1", "structure:1"]

def test_union_dedupes_across_groups() -> None:
    assert union_source_uids(["python:1", "react:1"], ["react:1"]) == ["python:1", "react:1"]

def test_union_drops_empty_and_non_string_entries() -> None:
    assert union_source_uids(["python:1", "", None, 3], []) == ["python:1"]

def test_union_of_nothing_is_empty() -> None:
    assert union_source_uids([], []) == []

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

def test_stamp_item_uids_leaves_a_fully_stamped_list_untouched() -> None:
    """Idempotence: the merge write runs once, but the helper must not renumber."""
    items: list[dict[str, Any]] = [{ITEM_UID_KEY: "item:1"}, {ITEM_UID_KEY: "item:2"}]
    stamp_item_uids(items)
    assert [item_uid(item) for item in items] == ["item:1", "item:2"]
