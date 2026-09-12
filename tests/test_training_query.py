"""Tests for the legacy skill decoder that survives in daydream.training.corpus.

The v1 query/filter pipeline (``_query_index`` / ``CorpusFilters``) was deleted
with the legacy records-builder (#1093); the one retained helper this module
covers is the historical skill-to-stack decoder. The archive fixture keeps the
§9 fixture matrix materialized via
``tests.fixtures.training.build_archive.build_fixture_archive`` — a real
SQLite index built with the production ``upsert_run`` helper.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.fixtures.training.build_archive import build_fixture_archive


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    """Build the §9 fixture matrix into ``tmp_path`` and return the root."""
    build_fixture_archive(tmp_path)
    return tmp_path


def test_legacy_decoder_still_classifies(archive: Path) -> None:
    """M21/M22: a historical beagle skill still decodes to its stack via the private map."""
    from daydream.training.corpus import _stack_for_skill

    assert _stack_for_skill("beagle-python:review-python") == "python"
    assert _stack_for_skill("beagle-zig:review-zig") is None  # unknown → None (warned)
