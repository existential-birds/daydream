"""Real-path tests for content-addressed reuse of deep review results.

These tests enter from ``runner.run`` with the real filesystem and event loop,
mocking only the external backend through the ``create_backend`` seam. They
assert observable outcomes of the ``.daydream/review-cache/`` store: that it is
published through the artifact-visibility anchors and survives a fresh run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from daydream.runner import run
from tests.harness.stub_backend import install_stub_backend
from tests.test_deep_orchestrator import MakeConfig


async def test_store_directory_survives_a_fresh_run_and_is_readable_by_the_next(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig,
) -> None:
    """The `.daydream/review-cache/` sibling is published back into the tree, so a
    later run (which detaches and re-seeds its live root) can read what an earlier
    run wrote."""
    install_stub_backend(monkeypatch, multi_stack_target)
    assert await run(make_config(multi_stack_target)) == 0
    store = multi_stack_target / ".daydream" / "review-cache"
    assert store.is_dir(), "store must be published beside .daydream/deep"
    (store / "entries").mkdir(exist_ok=True)
    (store / "entries" / ("a" * 64)).mkdir()
    assert await run(make_config(multi_stack_target)) == 0
    assert (store / "entries" / ("a" * 64)).is_dir(), "a fresh run must not wipe the store"
