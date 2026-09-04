"""Unit tests for per-record and merged-item referential identity (issue #1111).

These are supplementary to the real-path coverage in ``tests/test_deep_orchestrator.py``
that drives ``runner.run`` end to end; what they pin here are the exact
guarantees the rest of the pipeline is allowed to rely on, because six call
sites key destructive decisions (dedup drops, arbitration verdicts, disk
rewrites) on this one field.
"""

from __future__ import annotations

from typing import Any

from daydream.deep.records import (
    ITEM_UID_KEY,
    RECORD_UID_KEY,
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


def test_uid_format_is_stack_name_and_one_based_ordinal() -> None:
    """The documented, debuggable format -- readable in artifacts, not a uuid4."""
    assert mint_record_uid("python", 1) == "python:1"
    assert mint_record_uid("structure", 3) == "structure:3"


def test_source_normalization_accepts_both_spellings() -> None:
    """A records filename and a bare stack name must collapse to one key.

    The pipeline has two producers: the loop that loads records off disk tags
    them with the filename, while the uncovered sweep tags its records with a
    bare stack name. Both have to route to the same file.
    """
    assert stack_name_from_records_source("stack-python-records.json") == "python"
    assert stack_name_from_records_source("stack-structure-records.json") == "structure"
    assert stack_name_from_records_source("uncovered") == "uncovered"


def test_unrecognized_source_shape_passes_through_unchanged() -> None:
    """An unroutable source is returned as-is so the caller can detect and report it.

    Silently coercing it would recreate the exact failure mode issue #1111's
    latent bug 2 describes: a record that routes nowhere and vanishes without
    a warning.
    """
    assert stack_name_from_records_source("something-else.json") == "something-else.json"


def test_stack_name_recovered_from_uid() -> None:
    assert stack_name_from_uid("python:1") == "python"
    assert stack_name_from_uid("structure:12") == "structure"


def test_stack_name_from_uid_is_empty_for_a_separatorless_value() -> None:
    """An empty result is the "no identity" signal, distinguishable from a real name."""
    assert stack_name_from_uid("python") == ""
    assert stack_name_from_uid("") == ""


def test_stack_name_from_uid_splits_on_the_last_separator() -> None:
    """The ordinal half is always the trailing integer.

    Splitting on the last separator keeps this correct even if a stack name
    ever gained a separator character, where splitting on the first would
    truncate the name.
    """
    assert stack_name_from_uid("weird:name:7") == "weird:name"


def test_record_uid_reads_the_field() -> None:
    assert record_uid({RECORD_UID_KEY: "python:2"}) == "python:2"


def test_record_uid_is_empty_for_a_record_without_one() -> None:
    """Post-merge items legitimately carry no uid; that must not raise.

    The cross-stack merge agent re-emits items from scratch, so a merged item
    has no pre-merge identity. Returning "" keeps the accessor usable on both
    sides of that boundary.
    """
    assert record_uid({"id": 1, "file": "api.py"}) == ""


def test_record_uid_is_empty_for_a_non_string_value() -> None:
    """A malformed artifact must degrade to "no identity", not leak an int."""
    assert record_uid({RECORD_UID_KEY: 7}) == ""


def test_stamp_assigns_sequential_uids_in_place() -> None:
    """Mutation is in place because the surrogate this replaces was ``id(record)``.

    A stage that rebuilt a record as a fresh dict silently escaped those
    object-identity sets. Stamping in place means a caller holding the same
    list sees the field without re-binding anything.
    """
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
    """Never overwrite: a resume must keep the uids the producing run minted.

    ``_rewrite_stack_records`` writes back a list that adjudication may have
    compacted, so the on-disk list can be shorter than the one that produced
    those uids. Re-minting by position would hand out different values than the
    run that created the artifacts.
    """
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
    """Ordinals count every entry, stamped or not, so a mixed list cannot collide.

    Counting only the unstamped records would re-issue ``python:1`` here and
    silently reintroduce the collision the field exists to prevent.
    """
    records: list[dict[str, Any]] = [{"id": 1, RECORD_UID_KEY: "python:1"}, {"id": 2}]

    stamp_record_uids(records, "python")

    assert [record_uid(record) for record in records] == ["python:1", "python:2"]
    assert duplicate_record_uids(records) == []


def test_stamp_handles_an_empty_list() -> None:
    records: list[dict[str, Any]] = []

    stamp_record_uids(records, "python")

    assert records == []


def test_backfill_reproduces_the_uids_the_producing_run_minted() -> None:
    """Back-compat: a pre-``uid`` artifact resumes to the same identities.

    This determinism is why the format is ``(stack_name, position)`` rather than
    a uuid4 -- a uuid4 could not be regenerated for records written before the
    field existed.
    """
    at_birth: list[dict[str, Any]] = [{"id": 1}, {"id": 2}, {"id": 3}]
    stamp_record_uids(at_birth, "python")

    reloaded_without_uids: list[dict[str, Any]] = [{"id": 1}, {"id": 2}, {"id": 3}]
    stamp_record_uids(reloaded_without_uids, "stack-python-records.json")

    assert [record_uid(r) for r in reloaded_without_uids] == [record_uid(r) for r in at_birth]


def test_duplicate_detection_reports_collisions_sorted() -> None:
    """A duplicate uid would reintroduce the over-delete, so it must be detectable."""
    records: list[dict[str, Any]] = [
        {RECORD_UID_KEY: "python:1"},
        {RECORD_UID_KEY: "python:1"},
        {RECORD_UID_KEY: "react:2"},
        {RECORD_UID_KEY: "react:2"},
        {RECORD_UID_KEY: "python:3"},
    ]

    assert duplicate_record_uids(records) == ["python:1", "react:2"]


def test_duplicate_detection_is_empty_for_a_sound_pool() -> None:
    records: list[dict[str, Any]] = [{RECORD_UID_KEY: "python:1"}, {RECORD_UID_KEY: "react:1"}]

    assert duplicate_record_uids(records) == []


def test_duplicate_detection_ignores_records_without_a_uid() -> None:
    """Absence is a separate, separately reported condition -- not a collision.

    Several records sharing "no uid" must not be reported as sharing an
    identity, or the guard would fire on every post-merge item list.
    """
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
    """Items that never met the merge agent keep their birth identity.

    The host-appended structural items, the single-stack bypass and the salvage
    path all carry a record's own ``uid`` rather than a synthesized list, and
    must report the same way as an attributed merge item.
    """
    item: dict[str, Any] = {"id": 4, RECORD_UID_KEY: "structure:1"}

    assert item_source_uids(item) == ["structure:1"]


def test_item_source_uids_is_empty_when_the_agent_declined_to_attribute() -> None:
    """Empty is a real answer, not an error.

    An item the merge agent could not tie to a record reports nothing rather
    than a fabricated link -- inventing provenance here would be worse than
    admitting there is none.
    """
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
    """Items written before the field existed must read as "no identity", not raise.

    ``normalize_items`` mints with preserve-if-present semantics, so "" is the
    signal that tells it to mint rather than keep.
    """
    assert item_uid({"id": 1, "file": "api.py"}) == ""


def test_item_uid_is_empty_for_a_non_string_value() -> None:
    """A malformed artifact must degrade to "no identity", never leak an int.

    Preserve-if-present keys off this accessor, so returning the raw ``7`` would
    preserve a non-string as an item's durable identity instead of replacing it.
    """
    assert item_uid({ITEM_UID_KEY: 7}) == ""


def test_record_and_item_identities_are_reported_independently() -> None:
    """One item can carry both identities, and each accessor must return only its own.

    This is why ``ITEM_UID_KEY`` is deliberately not ``RECORD_UID_KEY``. The
    host-appended structural items keep the ``uid`` of the record they were born
    as *and* need an item identity on top. Overloading one key would make
    ``record_uid`` return an item identity for some items and a record identity
    for others -- and ``item_source_uids`` falls back to ``record_uid``, so the
    item identity would silently masquerade as provenance.
    """
    item: dict[str, Any] = {
        "id": 4,
        RECORD_UID_KEY: "structure:1",
        ITEM_UID_KEY: "item:7",
        "source_uids": ["python:2", "react:2"],
    }

    assert record_uid(item) == "structure:1"
    assert item_uid(item) == "item:7"
    assert item_source_uids(item) == ["python:2", "react:2"]


def test_item_uid_is_never_reported_as_provenance() -> None:
    """Identity is not provenance, so the fallback chain must not reach ``item_uid``.

    ``source_uids`` is not unique -- the merge agent may split one record into
    two findings, so two items can cite the same record. An item identity
    appearing in a provenance list would tie a finding to a record that was
    never its source.
    """
    born_as_a_record: dict[str, Any] = {"id": 4, RECORD_UID_KEY: "structure:1", ITEM_UID_KEY: "item:7"}
    synthesized: dict[str, Any] = {"id": 5, ITEM_UID_KEY: "item:8"}

    assert item_source_uids(born_as_a_record) == ["structure:1"]
    assert item_source_uids(synthesized) == []


def test_union_merges_two_provenances_survivor_first() -> None:
    """The structural fold's survivor represents a defect two lenses reported.

    First-seen order puts the survivor's own provenance first, which is what
    keeps the fold's audit sidecar readable.
    """
    assert union_source_uids(["python:1"], ["structure:1"]) == ["python:1", "structure:1"]


def test_union_dedupes_across_groups() -> None:
    assert union_source_uids(["python:1", "react:1"], ["react:1"]) == ["python:1", "react:1"]


def test_union_drops_empty_and_non_string_entries() -> None:
    assert union_source_uids(["python:1", "", None, 3], []) == ["python:1"]


def test_union_of_nothing_is_empty() -> None:
    assert union_source_uids([], []) == []


def test_stamp_skips_an_ordinal_a_preserved_record_already_holds() -> None:
    """Counting positions alone can emit the duplicate this field exists to prevent.

    A preserved record sitting anywhere other than its original position --
    ``python:2`` first -- would hand position 2's unstamped record ``python:2``
    as well. That is the exact defect class issue #1111 removes, reintroduced by
    the helper meant to enforce it, so minting skips taken ordinals.
    """
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
