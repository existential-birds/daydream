"""Cross-run reuse of the deep pipeline's exploration pre-scan.

``.daydream/exploration/`` now survives a run and is keyed by
``head sha + diff + tier + format version``
(``daydream.exploration.exploration_cache_key``).
A second run with an identical key reuses the directory verbatim and fires zero
specialist agents; any key change re-runs the pre-scan and rewrites the files.

Every test drives the real ``runner.run`` -> deep orchestrator path with only the
backend seam stubbed.
"""

from __future__ import annotations

import json
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


def _drift_cached_key_between_runs(
    artifact_runtime_root: Path, target: Path, *, drop: bool = False, content: str = "",
) -> None:
    """Drift the cached exploration key between two runs, consistently.

    Published artifacts live in two synchronized copies: the public tree, and
    the canonical recovery copy under the private state root that the next run
    seeds its live tree from (where the exploration cache is actually read).
    Artifact sessions fail closed when those two disagree, so a staleness
    simulation must drift BOTH and refresh the canonical manifest — leaving a
    consistent state the next run reads as a stale/corrupt/missing cache key.
    """
    from daydream.artifact_visibility import _atomic_json, _manifest, _manifest_payload

    roots = [entry for entry in artifact_runtime_root.iterdir() if entry.is_dir()]
    assert len(roots) == 1, f"expected exactly one workspace state root, got {roots}"
    state_root = roots[0]
    canonical = state_root / "canonical"
    for base in (target / ".daydream", canonical / ".daydream"):
        key = base / "exploration" / "cache-key"
        if drop:
            key.unlink()
        else:
            key.write_text(content)
    _atomic_json(state_root / "canonical-manifest.json", _manifest_payload(_manifest(canonical)))


async def _run_deep(target: Path) -> int:
    from daydream.runner import RunConfig, run

    exclude = target / ".git" / "info" / "exclude"
    exclude.write_text(f"{exclude.read_text()}\n.daydream/\n.review-output.md\n")
    return await run(RunConfig(target=str(target), start_at="review", cleanup=False))


def _add_one_hop_graph_on_main(multi_stack_target: Path) -> None:
    """Make a committed one-hop Python graph before the feature-only edit."""
    _git(multi_stack_target, "checkout", "main")
    (multi_stack_target / "changed.py").write_text(
        "from dep import direct\n\n\ndef subject() -> str:\n    return direct()\n"
    )
    (multi_stack_target / "dep.py").write_text(
        "from secondhop import indirect\n\n\ndef direct() -> str:\n    return indirect()\n"
    )
    (multi_stack_target / "secondhop.py").write_text(
        "def indirect() -> str:\n    return 'base'\n"
    )
    (multi_stack_target / "consumer.py").write_text(
        "from changed import subject\n\n\ndef consume() -> str:\n    return subject()\n"
    )
    (multi_stack_target / "outer_consumer.py").write_text(
        "from consumer import consume\n\n\ndef outer() -> str:\n    return consume()\n"
    )
    _git(multi_stack_target, "add", "changed.py", "dep.py", "secondhop.py", "consumer.py", "outer_consumer.py")
    _git(multi_stack_target, "commit", "-m", "add one-hop exploration graph")
    _git(multi_stack_target, "checkout", "feature")
    _git(multi_stack_target, "rebase", "main")
    (multi_stack_target / "changed.py").write_text(
        "from dep import direct\n\n\ndef subject() -> str:\n    return direct() + '-changed'\n"
    )
    _git(multi_stack_target, "add", "changed.py")
    _git(multi_stack_target, "commit", "-m", "change one-hop graph root")


async def test_second_run_reuses_exploration(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exact key match reuses the directory and fires zero specialists."""
    silence(monkeypatch)
    _add_one_hop_graph_on_main(multi_stack_target)
    stub1 = _install(monkeypatch, multi_stack_target, "RUN1 SENTINEL")
    assert await _run_deep(multi_stack_target) == 0
    assert _count_specialist_calls(stub1) > 0

    exploration = multi_stack_target / ".daydream" / "exploration"
    assert exploration.is_dir(), "exploration must survive the run"
    first_key = (exploration / "cache-key").read_text().strip()
    first_exploration = (exploration / "exploration.json").read_bytes()
    assert first_key
    affected = {
        (row["path"], row["role"])
        for row in json.loads(first_exploration)["affected_files"]
    }
    assert ("changed.py", "modified") in affected
    assert ("dep.py", "imports") in affected
    assert ("consumer.py", "imported_by") in affected
    assert all(path not in {"secondhop.py", "outer_consumer.py"} for path, _ in affected)

    stub2 = _install(monkeypatch, multi_stack_target, "RUN2 SENTINEL")
    assert await _run_deep(multi_stack_target) == 0
    assert _count_specialist_calls(stub2) == 0, "cache hit must fire no specialists"
    assert (exploration / "cache-key").read_text().strip() == first_key
    assert (exploration / "exploration.json").read_bytes() == first_exploration

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
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    multi_stack_target: Path,
    artifact_runtime_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unignored Daydream output alone does not make a rebuilt cache ineligible."""
    from daydream.runner import RunConfig, run

    silence(monkeypatch)
    stub1 = _install(monkeypatch, multi_stack_target, "RUN1 SENTINEL")
    assert await run(RunConfig(target=str(multi_stack_target), start_at="review", cleanup=False)) == 0
    assert _count_specialist_calls(stub1) > 0

    _drift_cached_key_between_runs(
        artifact_runtime_root, multi_stack_target, content="stale"
    )

    stub2 = _install(monkeypatch, multi_stack_target, "RUN2 SENTINEL")
    assert await run(RunConfig(target=str(multi_stack_target), start_at="review", cleanup=False)) == 0
    assert _count_specialist_calls(stub2) > 0
    exploration = multi_stack_target / ".daydream" / "exploration"
    assert (exploration / "cache-key").read_text().strip() != "stale"


async def test_cache_version_change_invalidates_cache(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cache-format bump forces a real second-run pre-scan."""
    from daydream import exploration as exploration_mod

    silence(monkeypatch)
    stub1 = _install(monkeypatch, multi_stack_target, "RUN1 SENTINEL")
    assert await _run_deep(multi_stack_target) == 0
    assert _count_specialist_calls(stub1) > 0
    exploration = multi_stack_target / ".daydream" / "exploration"
    first_key = (exploration / "cache-key").read_text().strip()

    monkeypatch.setattr(exploration_mod, "_CACHE_VERSION", exploration_mod._CACHE_VERSION + 1)
    stub2 = _install(monkeypatch, multi_stack_target, "RUN2 SENTINEL")
    assert await _run_deep(multi_stack_target) == 0
    assert _count_specialist_calls(stub2) > 0, "version change must re-fire specialists"
    assert (exploration / "cache-key").read_text().strip() != first_key
    assert "RUN2 SENTINEL" in (exploration / "dependencies.md").read_text()


async def test_corrupt_key_file_is_a_miss_not_a_crash(
    multi_stack_target: Path,
    artifact_runtime_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A truncated/garbage key file re-runs the pre-scan instead of failing."""
    silence(monkeypatch)
    _install(monkeypatch, multi_stack_target, "RUN1 SENTINEL")
    assert await _run_deep(multi_stack_target) == 0

    _drift_cached_key_between_runs(
        artifact_runtime_root, multi_stack_target, content="not-a-real-key"
    )

    stub2 = _install(monkeypatch, multi_stack_target, "RUN2 SENTINEL")
    assert await _run_deep(multi_stack_target) == 0
    assert _count_specialist_calls(stub2) > 0
    exploration = multi_stack_target / ".daydream" / "exploration"
    assert "RUN2 SENTINEL" in (exploration / "dependencies.md").read_text()


async def test_missing_key_file_is_a_miss(
    multi_stack_target: Path,
    artifact_runtime_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-upgrade exploration dir (no key file) is treated as stale."""
    silence(monkeypatch)
    _install(monkeypatch, multi_stack_target, "RUN1 SENTINEL")
    assert await _run_deep(multi_stack_target) == 0

    _drift_cached_key_between_runs(artifact_runtime_root, multi_stack_target, drop=True)

    stub2 = _install(monkeypatch, multi_stack_target, "RUN2 SENTINEL")
    assert await _run_deep(multi_stack_target) == 0
    assert _count_specialist_calls(stub2) > 0


async def test_failed_exploration_is_not_durably_cached(
    multi_stack_target: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A degraded pre-scan is materialized for this run but cannot be reused."""
    silence(monkeypatch)
    stub = _install(monkeypatch, multi_stack_target, "unused")
    stub.fail_exploration = True

    assert await _run_deep(multi_stack_target) == 0

    exploration = multi_stack_target / ".daydream" / "exploration"
    assert exploration.is_dir()
    assert not (exploration / "cache-key").exists()


def test_cache_key_is_sensitive_to_every_component(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each of head, diff, tier, and the format version changes the key."""
    from daydream import exploration as exploration_mod
    from daydream.exploration import exploration_cache_key

    base = exploration_cache_key("sha1", "diff", "standard")
    assert base == exploration_cache_key("sha1", "diff", "standard")
    assert base != exploration_cache_key("sha2", "diff", "standard")
    assert base != exploration_cache_key("sha1", "other", "standard")
    assert base != exploration_cache_key("sha1", "diff", "deep")

    # The format version is part of the key: a bump must change it. The override
    # value is arbitrary (never assert the production value), so a legitimate
    # upgrade stays green while a removal from the payload goes red.
    monkeypatch.setattr(exploration_mod, "_CACHE_VERSION", 999)
    assert base != exploration_cache_key("sha1", "diff", "standard")


def test_cache_key_components_cannot_be_confused_by_delimiters() -> None:
    """Shifting content across the newline boundary changes the key."""
    from daydream.exploration import exploration_cache_key

    assert exploration_cache_key("a", "b", "standard") != exploration_cache_key(
        "a\nb", "", "standard"
    )
