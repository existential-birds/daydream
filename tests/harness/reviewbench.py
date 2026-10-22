"""Explicit opt-in source admission for supplied ReviewBench integration tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from daydream.benchmark.reviewbench.contract import verify_source


@pytest.fixture
def supplied_source() -> Path:
    configured = os.environ.get("DAYDREAM_REVIEWBENCH_SOURCE")
    if not configured:
        pytest.skip("set DAYDREAM_REVIEWBENCH_SOURCE to the pinned checkout with npm ci installed")
    source = Path(configured).resolve()
    verify_source(source)
    if not (source / "node_modules" / ".bin" / "tsx").is_file():
        pytest.fail("Run npm ci in the explicitly selected ReviewBench source checkout")
    return source
