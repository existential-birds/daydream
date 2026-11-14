"""Pin referential identities used by dedup, arbitration, and record rewriting.

Real-path pipeline coverage lives in tests/deep_orchestrator/."""

from __future__ import annotations

from daydream.deep.records import (
    union_source_uids,
)


def test_union_drops_empty_and_non_string_entries() -> None:
    assert union_source_uids(["python:1", "", None, 3], []) == ["python:1"]
