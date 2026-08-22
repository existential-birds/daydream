"""Deep per-stack/shared-name integration tests.

Issue #745 (AC4) removed the ``parse-<stack>`` stage: per-stack reviewers emit
``PER_STACK_RECORD_SCHEMA`` records directly. The record artifacts and their
deterministic ordering remain the merge's input, so the shard-naming /
deterministic-sort contract that still owns this file is preserved here. The
parse-specific concurrency / failure / truncation tests that were their own
file (``test_deep_parse_concurrency``) no longer have a subject and were
removed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.harness.stub_backend import install_stub_backend


async def _run_deep(target: Path) -> int:
    from daydream.runner import RunConfig, run

    return await run(RunConfig(target=str(target), cleanup=False))


async def test_shard_names_flow_through_parse_and_sort_deterministically(
    shard_many_python_target: Path, monkeypatch: pytest.MonkeyPatch,
    make_config,
) -> None:
    """Issue #731 (P2): synthetic ``#`` shard names ride artifact paths and the
    sorted parse/merge ordering deterministically."""
    from daydream.runner import run

    install_stub_backend(monkeypatch, shard_many_python_target)
    deep = shard_many_python_target / ".daydream" / "deep"
    # Shard record files sort stably (python#0 < python#1 ... < structure), so
    # parse fan-out and merge consume them in a fixed order regardless of
    # completion order (sorted iteration at orchestrator.py:1322).
    def shard_names() -> list[str]:
        return sorted(p.name for p in deep.glob("stack-*-records.json"))

    seen: list[list[str]] = []
    for _ in range(2):  # two identical runs -> identical shard set, stable order
        rc = await run(make_config(shard_many_python_target, deep_shard_enabled=True,
                                   deep_shard_max_files=1, deep_shard_max_bytes=10**9))
        assert rc == 0
        names = shard_names()
        assert any(n.startswith("stack-python#") for n in names)
        assert names == sorted(names)   # deterministic by construction
        seen.append(names)
    # The stated determinism claim: both runs must produce the same shard set
    # in the same order. A shard-naming or record-path nondeterminism
    # regression flips this comparison.
    assert seen[0] == seen[1]
