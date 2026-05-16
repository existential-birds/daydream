"""Tests for daydream.runner.RunConfig and the unified ``run`` dispatch."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest

from daydream import runner
from daydream.exploration import ExplorationContext
from daydream.runner import RunConfig
from daydream.workspace import WorkContext


def test_run_config_exploration_depth():
    assert RunConfig().exploration_depth == 1
    cfg = RunConfig(exploration_depth=2)
    assert cfg.exploration_depth == 2


def test_run_config_exploration_context_defaults_to_none():
    cfg = RunConfig()
    assert cfg.exploration_context is None
    explicit = ExplorationContext()
    cfg2 = RunConfig(exploration_context=explicit)
    assert cfg2.exploration_context is explicit


# --- Stage 4.1b dispatch tests ---------------------------------------------


def _fake_work(repo: Path) -> WorkContext:
    """Build a synthetic ``WorkContext`` for dispatch unit tests."""
    return WorkContext(
        repo=repo,
        source=repo,
        base_branch="main",
        base_sha="DEADBEEF",
        head_branch="feat/x",
        head_sha="CAFEBABE",
        is_ephemeral=False,
        run_id="20260101000000-deadbeef",
    )


@pytest.fixture
def patch_workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Stub ``open_workspace`` and the in-place fallback so dispatch tests
    don't touch git. Yields the synthetic ``WorkContext`` callers will see.
    """
    work = _fake_work(tmp_path)

    @asynccontextmanager
    async def _fake_open_workspace(*_args: Any, **_kwargs: Any) -> AsyncIterator[WorkContext]:
        yield work

    monkeypatch.setattr("daydream.runner.open_workspace", _fake_open_workspace)
    # Force the in-place fallback path off — every call goes through the fake
    # context manager. Avoids ambiguity about which code path was exercised.
    monkeypatch.setattr("daydream.runner.git_ops.is_inside_worktree", lambda _p: True)
    return work


@pytest.fixture
def silence_runner_ui(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop the noisy ``print_phase_hero`` banner during dispatch tests."""
    monkeypatch.setattr("daydream.runner.print_phase_hero", lambda *a, **kw: None)


@pytest.mark.asyncio
async def test_run_dispatches_to_pr_feedback_when_pr_number_set(
    monkeypatch, patch_workspace, silence_runner_ui, tmp_path
):
    called: dict[str, Any] = {}

    async def stub(work, config):
        called["work"] = work
        called["pr"] = config.pr_number
        return 0

    monkeypatch.setattr("daydream.runner._run_pr_feedback", stub)
    config = RunConfig(target=str(tmp_path), pr_number=42, bot="botname")

    exit_code = await runner.run(config)
    assert exit_code == 0
    assert called["pr"] == 42
    assert called["work"] is patch_workspace


@pytest.mark.asyncio
async def test_run_dispatches_to_comment_mode(
    monkeypatch, patch_workspace, silence_runner_ui, tmp_path
):
    called: dict[str, Any] = {}

    async def stub(work, config):
        called["mode"] = config.output_mode
        return 0

    monkeypatch.setattr("daydream.runner._run_comment", stub)
    config = RunConfig(target=str(tmp_path), output_mode="comment")

    exit_code = await runner.run(config)
    assert exit_code == 0
    assert called["mode"] == "comment"


@pytest.mark.asyncio
async def test_run_dispatches_to_review_mode(
    monkeypatch, patch_workspace, silence_runner_ui, tmp_path
):
    called: dict[str, Any] = {}

    async def stub(work, config):
        called["mode"] = config.output_mode
        return 0

    monkeypatch.setattr("daydream.runner._run_review", stub)
    config = RunConfig(target=str(tmp_path), output_mode="review")

    exit_code = await runner.run(config)
    assert exit_code == 0
    assert called["mode"] == "review"


@pytest.mark.asyncio
async def test_run_dispatches_to_shallow_loop(
    monkeypatch, patch_workspace, silence_runner_ui, tmp_path
):
    called: dict[str, bool] = {}

    async def stub(work, config):
        called["shallow"] = config.shallow
        return 0

    monkeypatch.setattr("daydream.runner._run_loop_shallow", stub)
    config = RunConfig(target=str(tmp_path), output_mode="loop", shallow=True)

    exit_code = await runner.run(config)
    assert exit_code == 0
    assert called["shallow"] is True


@pytest.mark.asyncio
async def test_run_dispatches_to_deep_loop(
    monkeypatch, patch_workspace, silence_runner_ui, tmp_path
):
    """Stage 4.2: deep is the default. No flags required to route here."""
    called: dict[str, bool] = {}

    async def stub(work, config):
        called["routed"] = True
        called["shallow"] = config.shallow
        return 0

    monkeypatch.setattr("daydream.runner._run_loop_deep", stub)
    # Default RunConfig (no shallow, no deep) goes deep — that's the new default.
    config = RunConfig(target=str(tmp_path), output_mode="loop")

    exit_code = await runner.run(config)
    assert exit_code == 0
    assert called["routed"] is True
    assert called["shallow"] is False


@pytest.mark.asyncio
async def test_run_dispatches_to_shallow_loop_when_explicit(
    monkeypatch, patch_workspace, silence_runner_ui, tmp_path
):
    """Shallow runs only when ``--shallow`` is explicitly set (paired w/ deep default)."""
    called: dict[str, bool] = {}

    async def shallow_stub(work, config):
        called["shallow"] = True
        return 0

    async def deep_stub(work, config):
        called["deep"] = True
        return 0

    monkeypatch.setattr("daydream.runner._run_loop_shallow", shallow_stub)
    monkeypatch.setattr("daydream.runner._run_loop_deep", deep_stub)
    config = RunConfig(target=str(tmp_path), output_mode="loop", shallow=True)

    exit_code = await runner.run(config)
    assert exit_code == 0
    assert called == {"shallow": True}


@pytest.mark.asyncio
async def test_comment_mode_errors_when_no_open_pr_for_branch(
    monkeypatch, patch_workspace, silence_runner_ui, tmp_path
):
    """``--comment --branch X`` with no open PR for X exits 1 with a clear error."""
    monkeypatch.setattr(
        "daydream.runner.git_ops.gh_pr_list_for_branch", lambda _repo, _branch: []
    )
    captured: dict[str, str] = {}

    def fake_print_error(_console, title, body):
        captured["title"] = title
        captured["body"] = body

    monkeypatch.setattr("daydream.runner.print_error", fake_print_error)

    config = RunConfig(
        target=str(tmp_path),
        output_mode="comment",
        branch="feat/missing",
    )
    exit_code = await runner.run(config)

    assert exit_code == 1
    assert "No Open PR" == captured["title"]
    assert "no open PR for branch feat/missing" in captured["body"]


@pytest.mark.asyncio
async def test_run_feedback_routes_through_pr_feedback(
    monkeypatch, patch_workspace, silence_runner_ui, tmp_path
):
    """``run_feedback`` sets ``pr_number`` and re-enters dispatch."""
    called: dict[str, Any] = {}

    async def stub(work, config):
        called["pr"] = config.pr_number
        return 0

    monkeypatch.setattr("daydream.runner._run_pr_feedback", stub)
    config = RunConfig(target=str(tmp_path), bot="botname")

    exit_code = await runner.run_feedback(config, 99)
    assert exit_code == 0
    assert called["pr"] == 99
    # The wrapper should have populated ``pr_number`` on the shared config.
    assert config.pr_number == 99


@pytest.mark.asyncio
async def test_pr_feedback_banner_echoes_resolved_backend_model(
    monkeypatch, tmp_path
):
    """The PR-feedback banner reports the model id that the resolved backend
    actually carries — not a parallel literal hardcoded in the runner. Tests
    propagation, not a specific id.
    """
    work = _fake_work(tmp_path)

    class _StubBackend:
        def __init__(self, model: str):
            self.model = model

    chosen_model = "fixture-model-xyz"

    monkeypatch.setattr(
        "daydream.runner._resolve_backend",
        lambda _config, _phase, _cache=None: _StubBackend(chosen_model),
    )

    async def _no_op_fetch(*_args, **_kwargs):
        return None

    async def _empty_parse(*_args, **_kwargs):
        return []

    monkeypatch.setattr("daydream.runner.phase_fetch_pr_feedback", _no_op_fetch)
    monkeypatch.setattr("daydream.runner.phase_parse_feedback", _empty_parse)

    captured: list[str] = []
    monkeypatch.setattr(
        "daydream.runner.print_info",
        lambda _console, message: captured.append(message),
    )

    config = RunConfig(
        target=str(tmp_path),
        pr_number=42,
        bot="botname",
    )

    exit_code = await runner._run_pr_feedback(work, config)
    assert exit_code == 0
    assert f"Model: {chosen_model}" in captured, (
        f"Banner did not echo backend.model; got {captured!r}"
    )


# --- Per-phase model resolution tests (Task 2) -----------------------------


class TestResolveBackendPhaseModel:
    def test_explicit_phase_flag_wins_over_table(self):
        config = RunConfig(backend="claude", review_model="claude-haiku-4-5")
        backend = runner._resolve_backend(config, "review")
        assert backend.model == "claude-haiku-4-5"

    def test_table_default_used_when_no_flag(self):
        config = RunConfig(backend="claude")  # no review_model override
        backend = runner._resolve_backend(config, "review")
        assert backend.model == "claude-opus-4-6"  # claude REVIEW default

    def test_table_default_for_phase_without_flag(self):
        # WONDER has no override flag but should still get the table default.
        config = RunConfig(backend="claude")
        backend = runner._resolve_backend(config, "wonder")
        assert backend.model == "claude-opus-4-6"

    def test_codex_table_default(self):
        config = RunConfig(backend="codex")
        backend = runner._resolve_backend(config, "parse")
        assert backend.model == "gpt-5.5"

    def test_backend_override_uses_overridden_backends_table(self):
        # --review-backend codex while default is claude: resolver should look up
        # the codex table for review, not the claude one.
        config = RunConfig(backend="claude", review_backend="codex")
        backend = runner._resolve_backend(config, "review")
        assert backend.model == "gpt-5.5"  # codex REVIEW default (v1)

    def test_cache_returns_same_instance_for_same_phase_and_backend(self):
        cache: dict = {}
        config = RunConfig(backend="claude")
        b1 = runner._resolve_backend(config, "review", cache)
        b2 = runner._resolve_backend(config, "review", cache)
        assert b1 is b2

    def test_cache_returns_distinct_instances_for_different_phases(self):
        # Different models -> different backends, even on the same backend kind.
        cache: dict = {}
        config = RunConfig(backend="claude")
        review_backend = runner._resolve_backend(config, "review", cache)
        parse_backend = runner._resolve_backend(config, "parse", cache)
        assert review_backend is not parse_backend
        assert review_backend.model != parse_backend.model


# --- Task 6: HEAL hero is followed by Model: dim line ----------------------


@pytest.mark.asyncio
async def test_run_loop_shallow_heal_hero_followed_by_model_line(
    monkeypatch, tmp_path
):
    """The HEAL phase hero in ``_run_loop_shallow`` must be followed by a dim
    ``Model: <name>`` line scoped to the fix backend.

    Drives single-pass shallow loop with ``start_at="fix"`` to skip review,
    exploration, and skill resolution. Only the fix and test phases run, and
    HEAL renders before phase_fix. Asserts hero + dim call ordering.
    """
    from daydream.config import REVIEW_OUTPUT_FILE

    # Pre-create the review file so the check_review_file_exists guard passes.
    (tmp_path / REVIEW_OUTPUT_FILE).write_text("## Issues\n\n1. foo.py:1 - Bug\n")

    work = _fake_work(tmp_path)

    # Stub backends to carry distinct phase-specific models so the dim line's
    # source is unambiguous.
    class _StubBackend:
        def __init__(self, model: str):
            self.model = model

    backends_by_phase = {
        "review": _StubBackend("review-model-xyz"),
        "fix": _StubBackend("fix-model-xyz"),
        "test": _StubBackend("test-model-xyz"),
    }

    monkeypatch.setattr(
        "daydream.runner._resolve_backend",
        lambda _config, phase, _cache=None: backends_by_phase[phase],
    )

    async def _stub_phase_parse_feedback(_backend, _work):
        return [{"id": 1, "description": "Bug", "file": "foo.py", "line": 1}]

    async def _stub_phase_fix(*_args, **_kwargs):
        return None

    async def _stub_phase_test_and_heal(*_args, **_kwargs):
        return (True, 0)

    async def _stub_phase_commit_push(*_args, **_kwargs):
        return None

    monkeypatch.setattr("daydream.runner.phase_parse_feedback", _stub_phase_parse_feedback)
    monkeypatch.setattr("daydream.runner.phase_fix", _stub_phase_fix)
    monkeypatch.setattr("daydream.runner.phase_test_and_heal", _stub_phase_test_and_heal)
    monkeypatch.setattr("daydream.runner.phase_commit_push", _stub_phase_commit_push)

    # Capture hero + dim calls in order.
    calls: list[tuple[str, str]] = []  # (kind, payload)

    def _hero_spy(_console, title, _description):
        calls.append(("hero", title))

    def _dim_spy(_console, message):
        calls.append(("dim", message))

    monkeypatch.setattr("daydream.runner.print_phase_hero", _hero_spy)
    monkeypatch.setattr("daydream.runner.print_dim", _dim_spy)

    # Silence misc UI noise.
    monkeypatch.setattr("daydream.runner.print_info", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.runner.print_warning", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.runner.print_error", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.runner.print_summary", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.runner.print_skipped_phases", lambda *a, **kw: None)

    config = RunConfig(
        target=str(tmp_path),
        start_at="fix",
        cleanup=False,
        loop=False,
        shallow=True,
    )

    exit_code = await runner._run_loop_shallow(work, config)
    assert exit_code == 0

    # Find the HEAL hero in call order.
    heal_idx = next(
        (i for i, (kind, payload) in enumerate(calls) if kind == "hero" and payload == "HEAL"),
        None,
    )
    assert heal_idx is not None, f"HEAL hero never rendered; calls={calls!r}"
    # The very next call must be a dim Model: line carrying the fix backend's id.
    assert heal_idx + 1 < len(calls), "HEAL hero has no following call"
    next_kind, next_payload = calls[heal_idx + 1]
    assert next_kind == "dim", (
        f"Call after HEAL hero was not a dim line; got {(next_kind, next_payload)!r}"
    )
    assert next_payload == "Model: fix-model-xyz", (
        f"Dim line after HEAL hero did not echo the fix backend's model; got {next_payload!r}"
    )
