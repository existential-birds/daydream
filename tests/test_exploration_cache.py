"""Cross-run reuse of the deep pipeline's exploration pre-scan.

``.daydream/exploration/`` now survives a run and is keyed by
``head sha + diff + tier + depth`` (``daydream.exploration.exploration_cache_key``).
A second run with an identical key reuses the directory verbatim and fires zero
specialist agents; any key change re-runs the pre-scan and rewrites the files.

Every test drives the real ``runner.run`` -> deep orchestrator path with only the
backend seam stubbed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.harness.git_helpers import git as _git
from tests.harness.stub_backend import StubBackend, install_stub_backend, silence

_SPECIALIST_MARKER = "specialist"


def _install(monkeypatch: pytest.MonkeyPatch, target: Path, sentinel: str) -> StubBackend:
    stub = install_stub_backend(monkeypatch, target, enable_exploration=True)
    stub.exploration_sentinel = sentinel
    return stub


def _count_specialist_calls(stub: StubBackend) -> int:
    return sum(1 for c in stub.calls if _SPECIALIST_MARKER in c["prompt"].lower())


async def _run_deep(target: Path) -> int:
    from daydream.runner import RunConfig, run

    exclude = target / ".git" / "info" / "exclude"
    exclude.write_text(f"{exclude.read_text()}\n.daydream/\n.review-output.md\n")
    return await run(RunConfig(target=str(target), start_at="review", cleanup=False))


async def test_second_run_reuses_exploration(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exact key match reuses the directory and fires zero specialists."""
    silence(monkeypatch)
    stub1 = _install(monkeypatch, multi_stack_target, "RUN1 SENTINEL")
    assert await _run_deep(multi_stack_target) == 0
    assert _count_specialist_calls(stub1) > 0

    exploration = multi_stack_target / ".daydream" / "exploration"
    assert exploration.is_dir(), "exploration must survive the run"
    assert (exploration / "cache-key").read_text().strip()

    stub2 = _install(monkeypatch, multi_stack_target, "RUN2 SENTINEL")
    assert await _run_deep(multi_stack_target) == 0
    assert _count_specialist_calls(stub2) == 0, "cache hit must fire no specialists"

    # Reviewers are still grounded by the pointer.
    review_prompt = next(
        c["prompt"] for c in stub2.calls if "you are reviewing the" in c["prompt"].lower()
    )
    assert ".daydream/exploration" in review_prompt

    # Run 1's content survived: the hit did NOT clobber the cache with the
    # empty-context stubs a naive hit path would write.
    dependencies = (exploration / "dependencies.md").read_text()
    assert "RUN1 SENTINEL" in dependencies
    assert "RUN2 SENTINEL" not in dependencies
    assert "No data collected" not in dependencies


async def test_diff_change_invalidates_cache(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new commit changes head+diff, so the pre-scan re-runs and rewrites."""
    silence(monkeypatch)
    stub1 = _install(monkeypatch, multi_stack_target, "RUN1 SENTINEL")
    assert await _run_deep(multi_stack_target) == 0
    assert _count_specialist_calls(stub1) > 0

    exploration = multi_stack_target / ".daydream" / "exploration"
    key_after_run1 = (exploration / "cache-key").read_text().strip()

    (multi_stack_target / "api.py").write_text("def hello():\n    return 'galaxy'\n")
    _git(multi_stack_target, "add", "api.py")
    _git(multi_stack_target, "commit", "-m", "change again")

    stub2 = _install(monkeypatch, multi_stack_target, "RUN2 SENTINEL")
    assert await _run_deep(multi_stack_target) == 0
    assert _count_specialist_calls(stub2) > 0, "changed diff must re-fire specialists"

    assert (exploration / "cache-key").read_text().strip() != key_after_run1
    dependencies = (exploration / "dependencies.md").read_text()
    assert "RUN2 SENTINEL" in dependencies
    assert "RUN1 SENTINEL" not in dependencies


async def test_uncommitted_edit_reuses_an_exact_cache_key(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exact key hit remains reusable when the worktree is dirty."""
    silence(monkeypatch)
    (multi_stack_target / "api.py").write_text("def hello():\n    return 'galaxy'\n")

    stub1 = _install(monkeypatch, multi_stack_target, "RUN1 SENTINEL")
    assert await _run_deep(multi_stack_target) == 0
    assert _count_specialist_calls(stub1) > 0

    exploration = multi_stack_target / ".daydream" / "exploration"
    stub2 = _install(monkeypatch, multi_stack_target, "RUN2 SENTINEL")
    assert await _run_deep(multi_stack_target) == 0
    assert _count_specialist_calls(stub2) == 0, "exact key hit must skip specialists"

    assert (exploration / "cache-key").exists()
    dependencies = (exploration / "dependencies.md").read_text()
    assert "RUN1 SENTINEL" in dependencies
    assert "RUN2 SENTINEL" not in dependencies


async def test_daydream_artifacts_do_not_block_writing_a_rebuilt_cache_key(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unignored Daydream output alone does not make a rebuilt cache ineligible."""
    from daydream.runner import RunConfig, run

    silence(monkeypatch)
    stub1 = _install(monkeypatch, multi_stack_target, "RUN1 SENTINEL")
    assert await run(RunConfig(target=str(multi_stack_target), start_at="review", cleanup=False)) == 0
    assert _count_specialist_calls(stub1) > 0

    exploration = multi_stack_target / ".daydream" / "exploration"
    (exploration / "cache-key").write_text("stale")

    stub2 = _install(monkeypatch, multi_stack_target, "RUN2 SENTINEL")
    assert await run(RunConfig(target=str(multi_stack_target), start_at="review", cleanup=False)) == 0
    assert _count_specialist_calls(stub2) > 0
    assert (exploration / "cache-key").read_text().strip() != "stale"


async def test_depth_change_invalidates_cache(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """exploration_depth is part of the key, so changing it re-runs the pre-scan."""
    from daydream.runner import RunConfig, run

    silence(monkeypatch)
    stub1 = _install(monkeypatch, multi_stack_target, "RUN1 SENTINEL")
    assert await run(
        RunConfig(target=str(multi_stack_target), start_at="review", cleanup=False)
    ) == 0
    assert _count_specialist_calls(stub1) > 0

    stub2 = _install(monkeypatch, multi_stack_target, "RUN2 SENTINEL")
    assert await run(
        RunConfig(
            target=str(multi_stack_target),
            start_at="review",
            cleanup=False,
            exploration_depth=3,
        )
    ) == 0
    assert _count_specialist_calls(stub2) > 0, "changed depth must re-fire specialists"


async def test_corrupt_key_file_is_a_miss_not_a_crash(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A truncated/garbage key file re-runs the pre-scan instead of failing."""
    silence(monkeypatch)
    _install(monkeypatch, multi_stack_target, "RUN1 SENTINEL")
    assert await _run_deep(multi_stack_target) == 0

    exploration = multi_stack_target / ".daydream" / "exploration"
    (exploration / "cache-key").write_text("not-a-real-key")

    stub2 = _install(monkeypatch, multi_stack_target, "RUN2 SENTINEL")
    assert await _run_deep(multi_stack_target) == 0
    assert _count_specialist_calls(stub2) > 0
    assert "RUN2 SENTINEL" in (exploration / "dependencies.md").read_text()


async def test_missing_key_file_is_a_miss(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pre-upgrade exploration dir (no key file) is treated as stale."""
    silence(monkeypatch)
    _install(monkeypatch, multi_stack_target, "RUN1 SENTINEL")
    assert await _run_deep(multi_stack_target) == 0

    exploration = multi_stack_target / ".daydream" / "exploration"
    (exploration / "cache-key").unlink()

    stub2 = _install(monkeypatch, multi_stack_target, "RUN2 SENTINEL")
    assert await _run_deep(multi_stack_target) == 0
    assert _count_specialist_calls(stub2) > 0


async def test_failed_exploration_is_not_durably_cached(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A degraded pre-scan is materialized for this run but cannot be reused."""
    from daydream.deep import orchestrator

    silence(monkeypatch)
    _install(monkeypatch, multi_stack_target, "unused")

    async def failing_pre_scan(*args: object, **kwargs: object) -> object:
        raise RuntimeError("exploration unavailable")

    monkeypatch.setattr(orchestrator, "pre_scan", failing_pre_scan)

    assert await _run_deep(multi_stack_target) == 0

    exploration = multi_stack_target / ".daydream" / "exploration"
    assert exploration.is_dir()
    assert not (exploration / "cache-key").exists()


def test_cache_key_is_sensitive_to_every_component() -> None:
    """Each of head, diff, tier, depth changes the key."""
    from daydream.exploration import exploration_cache_key

    base = exploration_cache_key("sha1", "diff", "standard", 2)
    assert base == exploration_cache_key("sha1", "diff", "standard", 2)
    assert base != exploration_cache_key("sha2", "diff", "standard", 2)
    assert base != exploration_cache_key("sha1", "other", "standard", 2)
    assert base != exploration_cache_key("sha1", "diff", "deep", 2)
    assert base != exploration_cache_key("sha1", "diff", "standard", 3)


def test_cache_key_components_cannot_be_confused_by_delimiters() -> None:
    """Shifting content across the newline boundary changes the key."""
    from daydream.exploration import exploration_cache_key

    assert exploration_cache_key("a", "b", "standard", 2) != exploration_cache_key(
        "a\nb", "", "standard", 2
    )
