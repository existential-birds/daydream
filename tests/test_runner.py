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


def test_run_config_skill_availability_defaults_to_none():
    assert RunConfig().skill_availability is None
    cfg = RunConfig(skill_availability=frozenset({"python"}))
    assert cfg.skill_availability == frozenset({"python"})


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
    # Force the in-place fallback off so every call goes through the fake CM.
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
async def test_auto_detected_pr_number_dispatches_to_deep_loop(
    monkeypatch, patch_workspace, silence_runner_ui, tmp_path
):
    """Auto-detected pr_number (no bot) routes to the deep loop, not PR feedback."""
    called: dict[str, Any] = {}

    async def stub(work, config):
        called["pr_number"] = config.pr_number
        return 0

    monkeypatch.setattr("daydream.runner._run_loop_deep", stub)
    config = RunConfig(target=str(tmp_path), pr_number=42, bot=None)

    exit_code = await runner.run(config)
    assert exit_code == 0
    assert called["pr_number"] == 42


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
        fanout_concurrency = 4

        def __init__(self, model: str):
            self.model = model

    chosen_model = "fixture-model-xyz"

    # ``FlowContext.backend_for`` resolves through ``daydream.runner._resolve_backend``
    # (late import) and passes ``cache=`` by keyword.
    monkeypatch.setattr(
        "daydream.runner._resolve_backend",
        lambda _config, _phase, cache=None, **_kwargs: _StubBackend(chosen_model),
    )

    async def _no_op_fetch(*_args, **_kwargs):
        return None

    async def _empty_parse(*_args, **_kwargs):
        return []

    # The fetch/parse call sites moved into the registered pr-feedback flow steps.
    monkeypatch.setattr("daydream.flows.pr_feedback.phase_fetch_pr_feedback", _no_op_fetch)
    monkeypatch.setattr("daydream.flows.pr_feedback.phase_parse_feedback", _empty_parse)

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
        assert backend.model == "claude-opus-4-8"  # claude REVIEW default

    def test_table_default_for_phase_without_flag(self):
        # WONDER has no override flag but should still get the table default.
        config = RunConfig(backend="claude")
        backend = runner._resolve_backend(config, "wonder")
        assert backend.model == "claude-opus-4-8"

    def test_codex_table_default(self):
        config = RunConfig(backend="codex")
        backend = runner._resolve_backend(config, "parse")
        assert backend.model == "gpt-5.6-luna"  # codex PARSE default (cheap tier)

    def test_backend_override_uses_overridden_backends_table(self):
        # review_backend=codex while default is claude: resolver must use the codex table.
        config = RunConfig(backend="claude", review_backend="codex")
        backend = runner._resolve_backend(config, "review")
        assert backend.model == "gpt-5.6-sol"  # codex REVIEW default (heavy tier)

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

    def test_codex_backend_receives_resolved_reasoning_effort_and_cache_splits_on_it(self):
        cache: dict = {}
        config = RunConfig(backend="codex", reasoning_effort="low")
        low_backend = runner._resolve_backend(config, "review", cache)
        assert low_backend.reasoning_effort == "low"
        config.reasoning_effort = "high"
        high_backend = runner._resolve_backend(config, "review", cache)
        assert high_backend.reasoning_effort == "high"
        assert low_backend is not high_backend  # different effort -> distinct cached instance


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
        fanout_concurrency = 4

        def __init__(self, model: str):
            self.model = model

    backends_by_phase = {
        "review": _StubBackend("review-model-xyz"),
        "parse": _StubBackend("parse-model-xyz"),
        "fix": _StubBackend("fix-model-xyz"),
        "test": _StubBackend("test-model-xyz"),
    }

    monkeypatch.setattr(
        "daydream.runner._resolve_backend",
        lambda _config, phase, cache=None, **_kwargs: backends_by_phase[phase],
    )

    async def _stub_phase_parse_feedback(_backend, _work):
        return [{"id": 1, "description": "Bug", "file": "foo.py", "line": 1}]

    async def _stub_phase_fix(*_args, **_kwargs):
        return None

    async def _stub_phase_test_and_heal(*_args, **_kwargs):
        return (True, 0)

    async def _stub_phase_commit_push(*_args, **_kwargs):
        return None

    monkeypatch.setattr("daydream.flows.shallow.phase_parse_feedback", _stub_phase_parse_feedback)
    monkeypatch.setattr("daydream.flows.shallow.phase_fix", _stub_phase_fix)
    monkeypatch.setattr("daydream.flows.shallow.phase_test_and_heal", _stub_phase_test_and_heal)
    monkeypatch.setattr("daydream.flows.shallow.phase_commit_push", _stub_phase_commit_push)

    # Capture hero + dim calls in order.
    calls: list[tuple[str, str]] = []  # (kind, payload)

    def _hero_spy(_console, title, _description):
        calls.append(("hero", title))

    def _dim_spy(_console, message):
        calls.append(("dim", message))

    monkeypatch.setattr("daydream.flows.shallow.print_phase_hero", _hero_spy)
    monkeypatch.setattr("daydream.flows.shallow.print_dim", _dim_spy)

    # Silence misc UI noise.
    monkeypatch.setattr("daydream.runner.print_info", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.runner.print_warning", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.runner.print_error", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.flows.shallow.print_summary", lambda *a, **kw: None)
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


async def test_shallow_items_canonicalized_and_severity_ordered(monkeypatch, tmp_path):
    """Shallow items carry ``lens="per-stack"`` + a ``severity`` derived from
    confidence, and ``phase_fix`` receives them severity-sorted.

    parse_feedback returns confidence-tagged items in order [LOW, HIGH]. After
    canonicalization + severity-sort, the HIGH-confidence item must be fixed
    first. Asserts on the severity ``phase_fix`` actually receives (observable
    consequence), never on dispatch.
    """
    from daydream.config import REVIEW_OUTPUT_FILE

    # Pre-create the review file so the check_review_file_exists guard passes.
    (tmp_path / REVIEW_OUTPUT_FILE).write_text("## Issues\n\n1. low.py:1 - L\n2. high.py:2 - H\n")

    work = _fake_work(tmp_path)

    class _StubBackend:
        fanout_concurrency = 4

        def __init__(self, model: str):
            self.model = model

    monkeypatch.setattr(
        "daydream.runner._resolve_backend",
        lambda _config, _phase, cache=None, **_kwargs: _StubBackend("stub-model"),
    )

    async def _stub_phase_parse_feedback(_backend, _work):
        # Intentionally out of severity order: LOW then HIGH.
        return [
            {"id": 1, "description": "L", "file": "low.py", "line": 1, "confidence": "LOW"},
            {"id": 2, "description": "H", "file": "high.py", "line": 2, "confidence": "HIGH"},
        ]

    order: list[str] = []

    async def _spy_phase_fix(_b, _w, item, _n, _t):
        order.append(item["severity"])

    async def _stub_phase_test_and_heal(*_args, **_kwargs):
        return (True, 0)

    async def _stub_phase_commit_push(*_args, **_kwargs):
        return None

    monkeypatch.setattr("daydream.flows.shallow.phase_parse_feedback", _stub_phase_parse_feedback)
    monkeypatch.setattr("daydream.flows.shallow.phase_fix", _spy_phase_fix)
    monkeypatch.setattr("daydream.flows.shallow.phase_test_and_heal", _stub_phase_test_and_heal)
    monkeypatch.setattr("daydream.flows.shallow.phase_commit_push", _stub_phase_commit_push)

    # Silence UI noise.
    monkeypatch.setattr("daydream.runner.print_phase_hero", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.runner.print_dim", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.runner.print_info", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.runner.print_warning", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.runner.print_error", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.flows.shallow.print_summary", lambda *a, **kw: None)
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
    assert order == ["high", "low"], f"phase_fix did not receive severity-ordered items; got {order!r}"


# --- Task 4: non_interactive threading -------------------------------------


def test_runconfig_defaults_non_interactive_false():
    assert RunConfig().non_interactive is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("dispatch_target", "config_kwargs"),
    [
        ("daydream.runner._run_loop_deep", {"output_mode": "loop"}),
        ("daydream.runner._run_loop_shallow", {"output_mode": "loop", "shallow": True}),
        ("daydream.runner._run_pr_feedback", {"pr_number": 7, "bot": "botname"}),
        ("daydream.runner._run_comment", {"output_mode": "comment"}),
    ],
    ids=["deep_loop", "shallow", "pr_feedback", "comment"],
)
async def test_run_threads_non_interactive_into_agent_state(
    dispatch_target, config_kwargs, monkeypatch, patch_workspace, silence_runner_ui, tmp_path
):
    """``config.non_interactive=True`` flips the agent singleton flag before any
    promptable phase, on every dispatch branch ``run()`` can take. Each case
    patches one dispatch fn so ``run()`` reaches the run-start setup (where
    ``set_non_interactive`` fires) without executing real phases.
    """
    from daydream.agent import get_non_interactive, reset_state

    reset_state()
    try:

        async def stub(work, config):
            return 0

        monkeypatch.setattr(dispatch_target, stub)
        config = RunConfig(target=str(tmp_path), non_interactive=True, **config_kwargs)

        exit_code = await runner.run(config)
        assert exit_code == 0
        assert get_non_interactive() is True
    finally:
        reset_state()


# --- non_interactive commit branch in _run_loop_shallow --------------------


@pytest.mark.asyncio
async def test_non_interactive_shallow_calls_phase_commit_push_auto(monkeypatch, tmp_path):
    """When non_interactive=True and tests pass, _run_loop_shallow must call
    phase_commit_push_auto — not the interactive phase_commit_push.

    This is a real-path test: it drives _run_loop_shallow directly and asserts
    on which commit function was actually invoked (observable side effect),
    rather than asserting on dispatch bookkeeping.
    """
    from daydream.agent import set_non_interactive
    from daydream.config import REVIEW_OUTPUT_FILE

    # The commit gate reads the agent singleton (set by run() from config), not
    # config.non_interactive directly. This test enters at _run_loop_shallow,
    # below run()'s set_non_interactive call, so establish the global here.
    set_non_interactive(True)

    (tmp_path / REVIEW_OUTPUT_FILE).write_text("## Issues\n\n1. foo.py:1 - X\n")

    work = _fake_work(tmp_path)

    class _StubBackend:
        model = "stub-model"
        fanout_concurrency = 4

    monkeypatch.setattr(
        "daydream.runner._resolve_backend",
        lambda _config, _phase, cache=None, **_kwargs: _StubBackend(),
    )

    async def _stub_phase_parse_feedback(_backend, _work):
        return [{"id": 1, "description": "X", "file": "foo.py", "line": 1}]

    async def _stub_phase_fix(*_args, **_kwargs):
        return None

    async def _stub_phase_test_and_heal(*_args, **_kwargs):
        return (True, 0)

    auto_calls: list[bool] = []
    interactive_calls: list[bool] = []

    async def _spy_phase_commit_push_auto(*_args, **_kwargs):
        auto_calls.append(True)

    async def _spy_phase_commit_push(*_args, **_kwargs):
        interactive_calls.append(True)

    monkeypatch.setattr("daydream.flows.shallow.phase_parse_feedback", _stub_phase_parse_feedback)
    monkeypatch.setattr("daydream.flows.shallow.phase_fix", _stub_phase_fix)
    monkeypatch.setattr("daydream.flows.shallow.phase_test_and_heal", _stub_phase_test_and_heal)
    monkeypatch.setattr("daydream.flows.shallow.phase_commit_push_auto", _spy_phase_commit_push_auto)
    monkeypatch.setattr("daydream.flows.shallow.phase_commit_push", _spy_phase_commit_push)

    monkeypatch.setattr("daydream.runner.print_phase_hero", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.runner.print_dim", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.runner.print_info", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.runner.print_warning", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.runner.print_error", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.flows.shallow.print_summary", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.runner.print_skipped_phases", lambda *a, **kw: None)

    config = RunConfig(
        target=str(tmp_path),
        start_at="fix",
        cleanup=False,
        loop=False,
        shallow=True,
        non_interactive=True,
    )

    exit_code = await runner._run_loop_shallow(work, config)
    assert exit_code == 0
    assert auto_calls == [True], (
        "phase_commit_push_auto was not called when non_interactive=True"
    )
    assert interactive_calls == [], (
        "phase_commit_push was called despite non_interactive=True"
    )


@pytest.mark.asyncio
async def test_non_interactive_shallow_failing_tests_write_handoff_no_fix(monkeypatch, tmp_path):
    """Real-path: a non-interactive shallow run whose tests FAIL writes a handoff
    and exits 1 without launching the fix agent or reading stdin.

    Drives ``_run_loop_shallow`` (the ``--shallow`` dispatch target) with the
    REAL ``phase_test_and_heal`` -- not stubbed -- against a failing test run.
    With the agent singleton's ``non_interactive`` flag set, ``phase_test_and_heal``
    must take choice-"4" semantics: skip the menu, skip stdin, run the read-only
    failure-summarizer, write ``handoff.md`` to the live run directory, and
    return ``(False, 0)``. If the non-interactive guard in ``phase_test_and_heal``
    were reverted, the menu's default "2" would launch the mutating heal fix agent
    -- whose ``_build_fix_prompt`` text ("Analyze the failures and fix them",
    asserted absent) -- and ``prompt_user`` would read stdin (rigged to fail).
    """
    import importlib
    import sys
    from pathlib import Path as _Path

    _tests_dir = str(_Path(__file__).parent)
    if _tests_dir not in sys.path:
        sys.path.insert(0, _tests_dir)
    _SummarizerBackend = importlib.import_module("test_phases")._SummarizerBackend

    from daydream.agent import reset_state, set_non_interactive
    from daydream.config import REVIEW_OUTPUT_FILE

    (tmp_path / REVIEW_OUTPUT_FILE).write_text("## Issues\n\n1. foo.py:1 - X\n")

    work = _fake_work(tmp_path)

    # The test phase's backend yields a failing test run, then the read-only
    # summarizer's handoff body -- exactly the choice-"4" path the guard takes.
    test_backend = _SummarizerBackend([
        "fail",
        ("handoff", "# Handoff\n\nnon-interactive failure context"),
    ])

    class _StubBackend:
        model = "stub-model"
        fanout_concurrency = 4

    def _resolve(_config, phase, cache=None, **_kwargs):
        return test_backend if phase == "test" else _StubBackend()

    monkeypatch.setattr("daydream.runner._resolve_backend", _resolve)

    async def _stub_phase_parse_feedback(_backend, _work):
        return [{"id": 1, "description": "X", "file": "foo.py", "line": 1}]

    async def _stub_phase_fix(*_args, **_kwargs):
        return None

    commit_calls: list[bool] = []

    async def _spy_phase_commit_push_auto(*_args, **_kwargs):
        commit_calls.append(True)

    async def _spy_phase_commit_push(*_args, **_kwargs):
        commit_calls.append(True)

    monkeypatch.setattr("daydream.flows.shallow.phase_parse_feedback", _stub_phase_parse_feedback)
    monkeypatch.setattr("daydream.flows.shallow.phase_fix", _stub_phase_fix)
    monkeypatch.setattr("daydream.flows.shallow.phase_commit_push_auto", _spy_phase_commit_push_auto)
    monkeypatch.setattr("daydream.flows.shallow.phase_commit_push", _spy_phase_commit_push)

    def _forbidden_input(*_a: Any, **_kw: Any) -> str:
        raise AssertionError("input() was called in non-interactive mode -- stdin must not be touched")

    monkeypatch.setattr("builtins.input", _forbidden_input)

    monkeypatch.setattr("daydream.runner.print_phase_hero", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.runner.print_dim", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.runner.print_info", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.runner.print_warning", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.runner.print_error", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.flows.shallow.print_summary", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.runner.print_skipped_phases", lambda *a, **kw: None)
    monkeypatch.setattr(
        "daydream.phases.console",
        type("C", (), {"print": lambda *a, **kw: None})(),
    )

    config = RunConfig(
        target=str(tmp_path),
        start_at="fix",
        cleanup=False,
        loop=False,
        shallow=True,
        non_interactive=True,
    )

    reset_state()
    set_non_interactive(True)
    try:
        exit_code = await runner._run_loop_shallow(work, config)
    finally:
        reset_state()

    assert exit_code == 1

    handoffs = list(tmp_path.glob(".daydream/runs/*/handoff.md"))
    assert len(handoffs) == 1, f"expected exactly one handoff.md, got {handoffs!r}"
    assert handoffs[0].read_text(encoding="utf-8") == "# Handoff\n\nnon-interactive failure context"

    assert commit_calls == [], "a commit ran despite tests failing"

    # Exactly two test-backend calls: the failing test run + the read-only
    # summarizer. The mutating heal fix agent was never launched.
    assert len(test_backend.captured_prompts) == 2
    assert "read-only failure-summarizer" in test_backend.captured_prompts[1]
    assert all(
        "Analyze the failures and fix them" not in p for p in test_backend.captured_prompts
    ), test_backend.captured_prompts


@pytest.mark.asyncio
async def test_yes_shallow_failing_tests_bounded_fix_and_abort(monkeypatch, tmp_path):
    """Real-path: a --yes shallow run whose tests FAIL runs ONE fix attempt then aborts.

    Drives ``_run_loop_shallow`` with the REAL ``phase_test_and_heal`` (not stubbed)
    against a scripted backend: fail → fix → fail → handoff (summarizer).

    With ``assume="yes"`` the bounded-loop guard at phases.py line 1777
    (``decision is True and retries_used > 0``) must fire after the first auto
    fix attempt, writing ``handoff.md`` and returning exit code 1. If the guard
    were absent, the fix agent would loop forever (the backend raises on call 5).
    The fix prompt text ("Analyze the failures and fix them") must appear in
    exactly one call (the fix agent), proving the fix ran once and only once.
    stdin must never be touched.
    """
    import importlib
    import sys
    from pathlib import Path as _Path

    _tests_dir = str(_Path(__file__).parent)
    if _tests_dir not in sys.path:
        sys.path.insert(0, _tests_dir)
    _SummarizerBackend = importlib.import_module("test_phases")._SummarizerBackend

    from daydream.agent import reset_state, set_assume
    from daydream.config import REVIEW_OUTPUT_FILE

    (tmp_path / REVIEW_OUTPUT_FILE).write_text("## Issues\n\n1. foo.py:1 - X\n")

    work = _fake_work(tmp_path)

    # Script: fail → fix (one bounded attempt) → fail → handoff (summarizer).
    # Any 5th call raises via _SummarizerBackend's guard, proving the loop ends.
    test_backend = _SummarizerBackend([
        "fail",
        "fix",
        "fail",
        ("handoff", "# Handoff\n\n--yes bounded fix failure"),
    ])

    class _StubBackend:
        model = "stub-model"
        fanout_concurrency = 4

    def _resolve(_config, phase, cache=None, **_kwargs):
        return test_backend if phase == "test" else _StubBackend()

    monkeypatch.setattr("daydream.runner._resolve_backend", _resolve)

    async def _stub_phase_parse_feedback(_backend, _work):
        return [{"id": 1, "description": "X", "file": "foo.py", "line": 1}]

    async def _stub_phase_fix(*_args, **_kwargs):
        return None

    commit_calls: list[bool] = []

    async def _spy_phase_commit_push_auto(*_args, **_kwargs):
        commit_calls.append(True)

    async def _spy_phase_commit_push(*_args, **_kwargs):
        commit_calls.append(True)

    monkeypatch.setattr("daydream.flows.shallow.phase_parse_feedback", _stub_phase_parse_feedback)
    monkeypatch.setattr("daydream.flows.shallow.phase_fix", _stub_phase_fix)
    monkeypatch.setattr("daydream.flows.shallow.phase_commit_push_auto", _spy_phase_commit_push_auto)
    monkeypatch.setattr("daydream.flows.shallow.phase_commit_push", _spy_phase_commit_push)

    def _forbidden_input(*_a: Any, **_kw: Any) -> str:
        raise AssertionError("input() must not be called under --yes")

    monkeypatch.setattr("builtins.input", _forbidden_input)

    monkeypatch.setattr("daydream.runner.print_phase_hero", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.runner.print_dim", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.runner.print_info", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.runner.print_warning", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.runner.print_error", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.flows.shallow.print_summary", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.runner.print_skipped_phases", lambda *a, **kw: None)
    monkeypatch.setattr(
        "daydream.phases.console",
        type("C", (), {"print": lambda *a, **kw: None})(),
    )

    config = RunConfig(
        target=str(tmp_path),
        start_at="fix",
        cleanup=False,
        loop=False,
        shallow=True,
        assume="yes",
    )

    reset_state()
    set_assume("yes")
    try:
        exit_code = await runner._run_loop_shallow(work, config)
    finally:
        reset_state()

    assert exit_code == 1

    handoffs = list(tmp_path.glob(".daydream/runs/*/handoff.md"))
    assert len(handoffs) == 1, f"expected exactly one handoff.md, got {handoffs!r}"
    assert handoffs[0].read_text(encoding="utf-8") == "# Handoff\n\n--yes bounded fix failure"

    assert commit_calls == [], "a commit ran despite tests failing"

    # Exactly four test-backend calls: fail → fix → fail → summarizer. No 5th
    # call (bounded-loop guard fired); call 4 ran read-only.
    assert len(test_backend.captured_prompts) == 4, test_backend.captured_prompts
    assert "Analyze the failures and fix them" in test_backend.captured_prompts[1]
    assert "read-only failure-summarizer" in test_backend.captured_prompts[3]
    assert test_backend.read_only_calls == [False, False, False, True], (
        test_backend.read_only_calls
    )
