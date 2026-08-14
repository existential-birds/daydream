"""Tests for daydream.runner.RunConfig and the unified ``run`` dispatch."""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import threading
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anyio
import pytest

from daydream import git_ops, runner
from daydream.backends import AgentEvent, ResultEvent, TextEvent
from daydream.exploration import ExplorationContext
from daydream.runner import RunConfig
from daydream.workspace import WorkContext
from tests.harness.backend import ScriptedBackend, Turn
from tests.harness.git_helpers import commit as _commit
from tests.harness.git_helpers import git as _git
from tests.harness.git_helpers import init_repo as _init_repo
from tests.test_deep_pr_comment_integration import (
    FakeAssistantMessage,
    FakeResultMessage,
    FakeTextBlock,
    FakeThinkingBlock,
    FakeToolResultBlock,
    FakeToolUseBlock,
    FakeUserMessage,
    _answer_prompts,
    _FakeSDKClient,
    _silence_ui,
)


@pytest.fixture
def deep_target(tmp_path: Path) -> Path:
    """Real git repo on a feature branch with one Python file changed.

    Mirrors ``tests/test_deep_pr_comment_integration.py``'s fixture so the
    real-path App-identity test drives the identical single-file deep path
    (tier ``"skip"``) with the shared fake SDK.
    """
    repo = tmp_path / "deep_repo"
    _init_repo(repo)
    (repo / "foo.py").write_text("def foo():\n    return 1\n")
    _git(repo, "add", ".")
    _commit(repo, "init")
    _git(repo, "checkout", "-b", "feature")
    (repo / "foo.py").write_text("def foo():\n    return 2\n")
    _git(repo, "add", ".")
    _commit(repo, "tweak foo")
    return repo


@pytest.fixture
def patch_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch every SDK symbol that ``ClaudeBackend.execute`` does isinstance on."""
    for symbol, fake in (
        ("ClaudeSDKClient", _FakeSDKClient),
        ("AssistantMessage", FakeAssistantMessage),
        ("UserMessage", FakeUserMessage),
        ("ResultMessage", FakeResultMessage),
        ("TextBlock", FakeTextBlock),
        ("ThinkingBlock", FakeThinkingBlock),
        ("ToolUseBlock", FakeToolUseBlock),
        ("ToolResultBlock", FakeToolResultBlock),
    ):
        monkeypatch.setattr(f"daydream.backends.claude.{symbol}", fake)

_RESULT = ResultEvent(structured_output=None, continuation=None)
# A failing test run, then the heal fix agent's turn.
_FAIL_TURN: tuple[AgentEvent, ...] = (TextEvent(text="1 failed, 0 passed"), _RESULT)
_FIX_TURN: tuple[AgentEvent, ...] = (TextEvent(text="Applied fix attempt"), _RESULT)
# Raised if the heal loop calls the backend past its script -- the bounded-loop guard.
_BEYOND_SCRIPT: Turn = (AssertionError("backend invoked beyond scripted call count"),)


def _handoff_turn(body: str) -> Turn:
    """The read-only failure-summarizer's structured handoff response."""
    return (ResultEvent(structured_output={"handoff_prompt": body}, continuation=None),)


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


@pytest.fixture
def patch_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, make_work: Callable[..., WorkContext]
):
    """Stub ``open_workspace`` and the in-place fallback so dispatch tests
    don't touch git. Yields the synthetic ``WorkContext`` callers will see.
    """
    work = make_work(tmp_path)

    @asynccontextmanager
    async def _fake_open_workspace(*_args: Any, **_kwargs: Any) -> AsyncIterator[WorkContext]:
        yield work

    monkeypatch.setattr("daydream.runner.open_workspace", _fake_open_workspace)
    # Force the in-place fallback off so every call goes through the fake CM.
    monkeypatch.setattr("daydream.runner.git_ops.is_inside_worktree", lambda _p: True)
    return work


@pytest.fixture
def silence_runner_ui(silence_console: Callable[..., None]) -> None:
    """Drop ``daydream.runner``'s UI helpers (notably the ``print_phase_hero``
    banner). ``daydream.deep.orchestrator``'s own ``print_phase_hero`` /
    ``print_dim`` bindings are deliberately left live: the AWAKEN-hero ordering
    test spies on them via ``daydream.phases``.
    """
    silence_console("daydream.runner")


_DISPATCH_TARGETS = (
    "_run_loop_deep",
    "_run_improve",
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("expected_target", "config_kwargs", "expected_attr", "expected_value"),
    [
        ("_run_loop_deep", {"pr_number": 42, "bot": "botname"}, "pr_number", 42),
        # Auto-detected pr_number (no bot) routes to the deep loop, not PR feedback.
        ("_run_loop_deep", {"pr_number": 42, "bot": None}, "pr_number", 42),
        ("_run_loop_deep", {"output_mode": "comment"}, "output_mode", "comment"),
        ("_run_loop_deep", {"output_mode": "review"}, "output_mode", "review"),
        ("_run_loop_deep", {"output_mode": "loop", "shallow": True}, "shallow", True),
        # Stage 4.2: deep is the default. No flags required to route here.
        ("_run_loop_deep", {"output_mode": "loop"}, "shallow", False),
        ("_run_improve", {"flow_name": "improve"}, "flow_name", "improve"),
    ],
    ids=[
        "pr_feedback_when_pr_number_set",
        "auto_detected_pr_number_goes_deep",
        "comment_mode",
        "review_mode",
        "shallow_mode",
        "deep_loop_by_default",
        "improve_flow",
    ],
)
async def test_run_dispatches_to_expected_flow(
    expected_target,
    config_kwargs,
    expected_attr,
    expected_value,
    monkeypatch,
    patch_workspace,
    silence_runner_ui,
    tmp_path,
    make_config,
):
    """``run()`` routes each flag combination to exactly one flow entrypoint.

    Every dispatch function is stubbed, so the recorded call list also proves
    exclusivity: every PR-process mode (feedback / comment / review / shallow /
    deep loop) lands on the single deep flow, and an auto-detected ``pr_number``
    (no ``bot``) must not route to a separate PR-feedback path.
    """
    called: list[tuple[str, WorkContext, RunConfig]] = []

    def _record(name: str):
        async def stub(work, config):
            called.append((name, work, config))
            return 0

        return stub

    for name in _DISPATCH_TARGETS:
        monkeypatch.setattr(f"daydream.runner.{name}", _record(name))

    config = make_config(tmp_path, **config_kwargs)

    exit_code = await runner.run(config)
    assert exit_code == 0
    assert [name for name, _work, _config in called] == [expected_target]
    _name, work, seen_config = called[0]
    assert work is patch_workspace
    observed = getattr(seen_config, expected_attr)
    assert observed == expected_value and type(observed) is type(expected_value)


@pytest.mark.asyncio
@pytest.mark.parametrize("flow_name", [None, "deep"], ids=["default_deep", "explicit_deep"])
async def test_deep_run_mints_app_identity_before_posting_path(
    flow_name: str | None,
    monkeypatch: pytest.MonkeyPatch,
    deep_target: Path,
    patch_sdk: None,
) -> None:
    """Real-path: deep runs mint the App token before their PR-posting path.

    Drives ``daydream.runner.run`` end-to-end in deep mode — both the default
    dispatch (``flow_name=None``) and an explicit ``--flow deep`` — on a real
    temp git worktree with GitHub App credentials set. Only the external
    network/API seams are mocked: the App installation-token mint, the Claude
    SDK transport (``patch_sdk``), and the final ``gh`` PR-posting transport
    (``find_open_pr`` + ``_submit_review``). The real deep orchestrator,
    ``ClaudeBackend.execute``, every phase, ``_post``, ``classify``, and
    ``build_payload`` run unmodified.

    Asserts the observable outcome rather than a stubbed loop: the App token
    is minted before the posting path is reached, ``config.identity`` resolves
    to the App bot identity, and the minted token is injected as ``GH_TOKEN``
    into every ``gh`` subprocess for the duration of the run.
    """
    from daydream import pr_review
    from daydream.runner import RunConfig

    _silence_ui(monkeypatch)
    _answer_prompts(monkeypatch)

    monkeypatch.setenv("DAYDREAM_APP_ID", "12345")
    monkeypatch.setenv("DAYDREAM_APP_PRIVATE_KEY", "test-private-key")

    events: list[str] = []
    payloads: list[dict[str, Any]] = []

    def fake_mint(*_args: object) -> SimpleNamespace:
        events.append("mint")
        return SimpleNamespace(
            token="installation-token",
            identity="daydream-review[bot]",
            expires_at=4_102_444_800.0,
        )

    fake_pr = pr_review.PRInfo(
        number=123,
        head_sha="0" * 40,
        base_sha="1" * 40,
        base_ref="main",
        owner="test-owner",
        repo="test-repo",
        url="https://example/pr/123",
    )

    def fake_find_open_pr(_target_dir: object) -> pr_review.PRInfo:
        events.append("find-open-pr")
        return fake_pr

    def fake_submit_review(
        _target_dir: object, _pr: object, payload: dict[str, Any]
    ) -> tuple[str, None]:
        events.append("post")
        payloads.append(payload)
        return "https://example/pr/123#review-1", None

    monkeypatch.setattr("daydream.github_app._mint_installation_token", fake_mint)
    monkeypatch.setattr("daydream.pr_review.find_open_pr", fake_find_open_pr)
    monkeypatch.setattr("daydream.pr_review._submit_review", fake_submit_review)

    config = RunConfig(
        target=str(deep_target),
        flow_name=flow_name,
        pr_repo="acme/widgets",
        cleanup=False,
        archive=False,
    )

    rc = await runner.run(config)

    assert rc == 0, f"run() returned {rc}"
    # Mint strictly precedes the posting path: find_open_pr is _post()'s first
    # action, and the review is submitted only after classify + build_payload.
    assert events == ["mint", "find-open-pr", "post"], events
    assert config.identity == "daydream-review[bot]"
    assert git_ops.get_gh_token_env() == {"GH_TOKEN": "installation-token"}
    assert payloads, "deep flow never reached _submit_review"


@pytest.mark.asyncio
async def test_review_run_does_not_mint_app_identity(
    monkeypatch: pytest.MonkeyPatch,
    patch_workspace: WorkContext,
    silence_runner_ui: None,
    tmp_path: Path,
    make_config: Callable[..., RunConfig],
) -> None:
    """``--review`` remains report-only when App credentials are configured."""
    monkeypatch.setenv("DAYDREAM_APP_ID", "12345")
    monkeypatch.setenv("DAYDREAM_APP_PRIVATE_KEY", "test-private-key")
    monkeypatch.setattr("daydream.github_app.resolve_user_identity", lambda _target: "operator")

    def mint_forbidden(*_args: object) -> SimpleNamespace:
        pytest.fail("report-only --review must not mint an App installation token")

    async def post_forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("report-only --review must not post a PR review")

    async def fake_review(_work: WorkContext, config: RunConfig) -> int:
        assert config.identity == "operator"
        return 0

    monkeypatch.setattr("daydream.github_app._mint_installation_token", mint_forbidden)
    monkeypatch.setattr("daydream.pr_review.post_review_to_pr_from_report", post_forbidden)
    monkeypatch.setattr("daydream.runner._run_loop_deep", fake_review)

    rc = await runner.run(make_config(tmp_path, output_mode="review", pr_repo="acme/widgets"))

    assert rc == 0


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (RunConfig(bot="review-bot"), True),
        (RunConfig(output_mode="comment"), True),
        (RunConfig(output_mode="loop"), True),
        (RunConfig(flow_name="deep"), True),
        (RunConfig(output_mode="review"), False),
        (RunConfig(flow_name="review"), False),
        (RunConfig(output_mode="loop", shallow=True), False),
        (RunConfig(flow_name="shallow"), False),
        (RunConfig(flow_name="custom-audit"), False),
    ],
    ids=[
        "pr_feedback",
        "comment",
        "default_deep",
        "explicit_deep",
        "review",
        "explicit_review",
        "shallow_loop",
        "explicit_shallow",
        "custom_flow",
    ],
)
def test_run_posts_to_github_matches_dispatch(config: RunConfig, expected: bool) -> None:
    """The identity classifier follows the runner's known write-capable paths."""
    assert runner._run_posts_to_github(config) is expected


@pytest.mark.asyncio
async def test_comment_mode_without_open_pr_dispatches_to_deep_flow(
    monkeypatch, patch_workspace, silence_runner_ui, tmp_path, make_config
):
    """``--comment --branch X`` with no open PR for X runs the deep flow.

    The old review flow refused up front ("No Open PR" error, exit 1); the
    collapsed deep flow dropped that pre-flight — ``_step_post_review`` warns and
    skips the post when no PR is resolvable (covered at the seam by
    ``test_post_skips_when_no_pr``). The runner-level contract is now: comment
    mode reaches the deep flow regardless of PR existence.
    """
    monkeypatch.setattr(
        "daydream.runner.git_ops.gh_pr_list_for_branch", lambda _repo, _branch: []
    )
    seen: dict[str, Any] = {}

    async def fake_deep(work: WorkContext, config: RunConfig) -> int:
        seen["output_mode"] = config.output_mode
        seen["branch"] = config.branch
        return 0

    monkeypatch.setattr("daydream.runner._run_loop_deep", fake_deep)

    config = make_config(tmp_path, output_mode="comment", branch="feat/missing")
    exit_code = await runner.run(config)

    assert exit_code == 0
    assert seen == {"output_mode": "comment", "branch": "feat/missing"}, seen


@pytest.mark.asyncio
async def test_run_feedback_routes_through_deep_flow(
    monkeypatch, patch_workspace, silence_runner_ui, tmp_path, make_config
):
    """``run_feedback`` sets ``pr_number`` and re-enters dispatch (deep loop)."""
    called: dict[str, Any] = {}

    async def stub(work, config):
        called["pr"] = config.pr_number
        return 0

    monkeypatch.setattr("daydream.runner._run_loop_deep", stub)
    config = make_config(tmp_path, bot="botname")

    exit_code = await runner.run_feedback(config, 99)
    assert exit_code == 0
    assert called["pr"] == 99
    # The wrapper should have populated ``pr_number`` on the shared config.
    assert config.pr_number == 99


@pytest.mark.asyncio
async def test_pr_feedback_banner_echoes_resolved_backend_model(
    monkeypatch, tmp_path, make_work, make_config
):
    """The PR-feedback banner reports the model id that the resolved backend
    actually carries — not a parallel literal hardcoded in the runner. Tests
    propagation, not a specific id.
    """
    from daydream.deep import orchestrator
    from daydream.extensions import build_registry, set_registry

    work = make_work(tmp_path)
    chosen_model = "fixture-model-xyz"

    set_registry(build_registry())

    # ``FlowContext.backend_for`` resolves through ``daydream.runner._resolve_backend``
    # (late import) and passes ``cache=`` by keyword.
    monkeypatch.setattr(
        "daydream.runner._resolve_backend",
        lambda _config, _phase, cache=None, **_kwargs: ScriptedBackend(model=chosen_model),
    )

    async def _no_op_fetch(*_args, **_kwargs):
        return None

    async def _empty_parse(*_args, **_kwargs):
        return []

    # The fetch/parse call sites live in the feedback-mode prefix of the deep flow.
    monkeypatch.setattr(
        "daydream.deep.orchestrator.phase_fetch_pr_feedback", _no_op_fetch
    )
    monkeypatch.setattr("daydream.deep.orchestrator.phase_parse_feedback", _empty_parse)

    captured: list[str] = []
    monkeypatch.setattr(
        "daydream.deep.orchestrator.print_info",
        lambda _console, message: captured.append(message),
    )

    config = make_config(tmp_path, pr_number=42, bot="botname")

    exit_code = await orchestrator._run_feedback_flow(config, work)
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
        assert backend.model == "claude-opus-5"  # claude REVIEW default

    def test_table_default_for_phase_without_flag(self):
        # WONDER has no override flag but should still get the table default.
        config = RunConfig(backend="claude")
        backend = runner._resolve_backend(config, "wonder")
        assert backend.model == "claude-opus-5"

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


# --- Task 6: deep fix-cycle hero is followed by Model: dim line -------------


def _seed_fix_resume(target: Path, items: list[dict[str, Any]]) -> Path:
    """Prime the deep artifacts a ``--start-at fix`` resume reads.

    ``merged-items.json`` is the canonical items source the fix gate reads, and
    ``diff-key`` must match the current diff so ``check_deep_artifacts`` does not
    treat the primed set as stale. Key written first so every prerequisite is
    strictly newer than the key (the resume freshness gate requires it).

    Returns:
        The ``.daydream/deep`` directory.
    """
    from daydream import git_ops
    from daydream.deep.artifacts import diff_key, diff_key_path, merged_items_path
    from daydream.workspace import _resolve_base

    deep = target / ".daydream" / "deep"
    deep.mkdir(parents=True, exist_ok=True)
    base = _resolve_base(target, None, None)
    diff = git_ops.diff(target, base)
    diff_key_path(deep).write_text(diff_key(diff or ""), encoding="utf-8")
    merged_items_path(deep).write_text(json.dumps({"items": items}))
    return deep


def _fix_item(item_id: int = 1, *, severity: str = "medium") -> dict[str, Any]:
    """Build a validated merged item targeting the fixture's tracked file."""
    return {
        "id": item_id,
        "lens": "per-stack",
        "file": "main.py",
        "line": 1,
        "severity": severity,
        "description": f"{severity} issue in main.py",
        "confidence": "MEDIUM",
        "rationale": "rationale",
        "evidence": "main.py:1",
    }


def _silence_fix_cycle_ui(silence_console: Callable[..., None]) -> None:
    """Silence the runner / deep orchestrator / phases noise; keeps phase hero
    and dim bindings alive so the hero-ordering test can spy on them."""
    silence_console("daydream.runner")
    silence_console("daydream.deep.orchestrator")
    silence_console("daydream.phases", keep=("print_phase_hero", "print_dim"))


async def _stub_verify(*_a: Any, **_k: Any) -> tuple[Path, dict[str, Any]]:
    """Empty-verdicts stand-in for ``phase_verify_recommendations``."""
    return Path("/nonexistent"), {"verdicts": []}


@pytest.mark.asyncio
async def test_fix_cycle_awaken_hero_followed_by_model_line(
    monkeypatch, feature_branch_repo, make_config, silence_console
):
    """The AWAKEN test-phase hero must be followed by a dim ``Model: <name>``
    line scoped to the test backend.

    The shallow flow's HEAL hero was dropped in the single-flow collapse (#330);
    the surviving phase-hero + Model-line pair in the deep fix cycle is
    ``phase_test_and_heal``'s AWAKEN hero followed by the dim Model line. Drives
    the deep fix cycle (``shallow=True, start_at="fix"``) real-path with the REAL
    ``phase_test_and_heal`` over a passing scripted suite, and asserts hero + dim
    call ordering through the module bindings that actually render them.
    """
    _seed_fix_resume(feature_branch_repo, [_fix_item()])
    _silence_fix_cycle_ui(silence_console)

    test_backend = ScriptedBackend(
        script=[(TextEvent(text="All 1 tests passed. 0 failed."), _RESULT)],
        model="test-model-xyz",
    )
    monkeypatch.setattr(
        "daydream.runner._resolve_backend",
        lambda _config, phase, cache=None, **_kwargs: (
            test_backend if phase == "test" else ScriptedBackend(model="stub-model")
        ),
    )
    monkeypatch.setattr("daydream.deep.orchestrator.phase_verify_recommendations", _stub_verify)

    async def _noop_fix(*_a: Any, **_k: Any) -> dict[str, str]:
        return {}

    async def _noop_commit(*_a: Any, **_k: Any) -> None:
        return None

    monkeypatch.setattr("daydream.deep.orchestrator.phase_fix_parallel", _noop_fix)
    monkeypatch.setattr("daydream.deep.orchestrator.phase_commit_push", _noop_commit)

    # Capture hero + dim calls in order.
    calls: list[tuple[str, str]] = []  # (kind, payload)

    def _hero_spy(_console, title, _description):
        calls.append(("hero", title))

    def _dim_spy(_console, message):
        calls.append(("dim", message))

    monkeypatch.setattr("daydream.phases.print_phase_hero", _hero_spy)
    monkeypatch.setattr("daydream.phases.print_dim", _dim_spy)

    exit_code = await runner.run(
        make_config(feature_branch_repo, start_at="fix", shallow=True, assume="yes")
    )
    assert exit_code == 0

    # Find the AWAKEN hero in call order.
    awaken_idx = next(
        (
            i
            for i, (kind, payload) in enumerate(calls)
            if kind == "hero" and payload == "AWAKEN"
        ),
        None,
    )
    assert awaken_idx is not None, f"AWAKEN hero never rendered; calls={calls!r}"
    # The very next call must be a dim Model: line carrying the test backend's id.
    assert awaken_idx + 1 < len(calls), "AWAKEN hero has no following call"
    next_kind, next_payload = calls[awaken_idx + 1]
    assert next_kind == "dim", (
        f"Call after AWAKEN hero was not a dim line; got {(next_kind, next_payload)!r}"
    )
    assert next_payload == "Model: test-model-xyz", (
        f"Dim line after AWAKEN hero did not echo the test backend's model; got {next_payload!r}"
    )


@pytest.mark.asyncio
async def test_fix_cycle_items_severity_ordered(
    monkeypatch, feature_branch_repo, make_config, silence_console
):
    """Merged items are severity-sorted (high before low) before ``phase_fix_parallel``.

    The fix gate seeds ``ctx.data["items"]`` with canonical merged items in an
    out-of-order shape [low, high]; after ``severity_sorted`` the HIGH item must
    be fixed first. Asserts on the severity ``phase_fix_parallel`` actually
    receives (observable consequence), never on dispatch bookkeeping.
    """
    _seed_fix_resume(
        feature_branch_repo,
        [_fix_item(1, severity="low"), _fix_item(2, severity="high")],
    )
    _silence_fix_cycle_ui(silence_console)

    monkeypatch.setattr(
        "daydream.runner._resolve_backend",
        lambda _config, _phase, cache=None, **_kwargs: ScriptedBackend(model="stub-model"),
    )
    monkeypatch.setattr("daydream.deep.orchestrator.phase_verify_recommendations", _stub_verify)

    order: list[str] = []

    async def _spy_fix_parallel(_b, _w, items, **_k):
        order.extend(item["severity"] for item in items)
        return {}

    async def _noop_test(*_a: Any, **_k: Any) -> tuple[bool, int, bool]:
        return (True, 0, True)

    async def _noop_commit(*_a: Any, **_k: Any) -> None:
        return None

    monkeypatch.setattr("daydream.deep.orchestrator.phase_fix_parallel", _spy_fix_parallel)
    monkeypatch.setattr("daydream.deep.orchestrator.phase_test_and_heal", _noop_test)
    monkeypatch.setattr("daydream.deep.orchestrator.phase_commit_push", _noop_commit)

    exit_code = await runner.run(
        make_config(feature_branch_repo, start_at="fix", shallow=True, assume="yes")
    )
    assert exit_code == 0
    assert order == ["high", "low"], (
        f"phase_fix_parallel did not receive severity-ordered items; got {order!r}"
    )


# --- Task 4: non_interactive threading -------------------------------------


def test_runconfig_defaults_non_interactive_false():
    assert RunConfig().non_interactive is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("dispatch_target", "config_kwargs"),
    [
        ("daydream.runner._run_loop_deep", {"output_mode": "loop"}),
        ("daydream.runner._run_loop_deep", {"output_mode": "loop", "shallow": True}),
        ("daydream.runner._run_loop_deep", {"pr_number": 7, "bot": "botname"}),
        ("daydream.runner._run_loop_deep", {"output_mode": "comment"}),
        ("daydream.runner._run_improve", {"flow_name": "improve"}),
    ],
    ids=["deep_loop", "shallow", "pr_feedback", "comment", "improve"],
)
async def test_run_threads_non_interactive_into_agent_state(
    dispatch_target, config_kwargs, monkeypatch, patch_workspace, silence_runner_ui, tmp_path,
    make_config,
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
        config = make_config(tmp_path, non_interactive=True, **config_kwargs)

        exit_code = await runner.run(config)
        assert exit_code == 0
        assert get_non_interactive() is True
    finally:
        reset_state()


# --- Deep fix-cycle commit gate semantics ----------------------------------


class _CommitWritingBackend:
    """Scripted fake whose test-suite and commit turns really touch the worktree.

    The test-suite turn reports a green run; the commit turn runs a REAL ``git
    commit`` carrying the Daydream trailers, so ``_do_commit``'s post-commit
    trailer verification sees a new HEAD. The backend is the only mocked seam —
    exactly the shape an agent with tools would take.
    """

    model = "mock-model"

    def __init__(self, repo: Path) -> None:
        self.repo = repo
        self.commit_prompts: list[str] = []

    async def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: Any = None,
        continuation: Any = None,
        agents: Any = None,
        max_turns: Any = None,
        read_only: bool = False,
    ):
        pl = prompt.lower()
        if "run the project's test suite" in pl:
            yield TextEvent(text="All 1 tests passed. 0 failed.")
            yield ResultEvent(structured_output=None, continuation=None)
            return
        if "stage all changes and commit" in pl:
            self.commit_prompts.append(prompt)
            run_id = re.search(r"Daydream-Run: (\S+)", prompt)
            version = re.search(r"Daydream-Version: (\S+)", prompt)
            message = (
                "fix: apply daydream fix\n\n"
                f"Daydream-Run: {run_id.group(1) if run_id else 'unknown'}\n"
                f"Daydream-Version: {version.group(1) if version else 'unknown'}\n"
            )
            _git(cwd, "add", "-A")
            _git(cwd, "commit", "-m", message)
            yield TextEvent(text="Committed.")
            yield ResultEvent(structured_output=None, continuation=None)
            return
        yield TextEvent(text="ok")
        yield ResultEvent(structured_output=None, continuation=None)

    async def cancel(self) -> None:
        pass

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
        return f"/{skill_key}" + (f" {args}" if args else "")


@pytest.mark.asyncio
async def test_fix_cycle_yes_commits_fixes(
    monkeypatch, feature_branch_repo, make_config, silence_console
):
    """Real-path: a --yes shallow deep run whose tests pass commits its fixes.

    The deep fix cycle's commit step is ``phase_commit_push`` (interactive gate);
    under ``assume="yes"`` the gate auto-approves and the commit agent runs. The
    observable outcome is a NEW git commit carrying the Daydream trailer.
    """
    from daydream.config import REVIEW_OUTPUT_FILE

    _seed_fix_resume(feature_branch_repo, [_fix_item()])
    _silence_fix_cycle_ui(silence_console)

    commit_backend = _CommitWritingBackend(feature_branch_repo)
    monkeypatch.setattr(
        "daydream.runner._resolve_backend",
        lambda _config, _phase, cache=None, **_kwargs: commit_backend,
    )
    monkeypatch.setattr("daydream.deep.orchestrator.phase_verify_recommendations", _stub_verify)

    async def _fix_writes(*_a: Any, **_k: Any) -> dict[str, str]:
        main_py = feature_branch_repo / "main.py"
        main_py.write_text(main_py.read_text() + "\n# daydream fix\n")
        return {}

    monkeypatch.setattr("daydream.deep.orchestrator.phase_fix_parallel", _fix_writes)

    # Pre-create the review output so the fix-gate decline path never triggers.
    (feature_branch_repo / REVIEW_OUTPUT_FILE).write_text("# Review\n")
    head_before = _git(feature_branch_repo, "rev-parse", "HEAD")

    exit_code = await runner.run(
        make_config(feature_branch_repo, start_at="fix", shallow=True, assume="yes")
    )
    assert exit_code == 0

    head_after = _git(feature_branch_repo, "rev-parse", "HEAD")
    assert head_after != head_before, "the --yes run never committed"
    assert len(commit_backend.commit_prompts) == 1
    assert "Daydream-Run:" in _git(feature_branch_repo, "log", "-1", "--format=%B")
    assert "# daydream fix" in _git(feature_branch_repo, "show", "HEAD:main.py")


@pytest.mark.asyncio
async def test_fix_cycle_non_interactive_declines_fix_and_commit(
    monkeypatch, feature_branch_repo, make_config, silence_console
):
    """Real-path: a non-interactive shallow deep run with no ``--yes`` declines at
    the apply-fixes gate — no fix, no test, no commit (observable: HEAD unchanged).
    """
    _seed_fix_resume(feature_branch_repo, [_fix_item()])
    _silence_fix_cycle_ui(silence_console)

    monkeypatch.setattr(
        "daydream.runner._resolve_backend",
        lambda _config, _phase, cache=None, **_kwargs: ScriptedBackend(model="stub-model"),
    )
    head_before = _git(feature_branch_repo, "rev-parse", "HEAD")

    exit_code = await runner.run(
        make_config(feature_branch_repo, start_at="fix", shallow=True)
    )

    assert exit_code == 0
    assert _git(feature_branch_repo, "rev-parse", "HEAD") == head_before, (
        "a non-interactive run without --yes committed"
    )
    assert not (feature_branch_repo / ".daydream-fix-applied").exists(), (
        "a non-interactive run without --yes applied a fix"
    )


async def _drive_fix_cycle_failing(
    monkeypatch: pytest.MonkeyPatch,
    target: Path,
    config: RunConfig,
    *,
    script: list[Turn],
    stdin_guard_message: str | None = None,
    stdin_answers: list[str] | None = None,
    clipboard_is_available: bool = False,
) -> tuple[int, ScriptedBackend, list[bool]]:
    """Drive the deep fix-cycle (``shallow=True, start_at="fix"``) with the REAL
    ``phase_test_and_heal`` over a scripted failing test run.

    Holds everything the two failing-fix-cycle real-path tests share: the
    fix-resume artifact seed, the scripted backend bound to the ``test`` phase,
    stubbed verify/fix so only the test phase touches the backend, one commit
    spy, and either a stdin trap (unattended) or a fed stdin queue (interactive
    gate answers). ``_BEYOND_SCRIPT`` is appended so a heal loop that runs past
    its script raises instead of silently replaying the last turn forever.

    Returns:
        ``(exit_code, test_backend, commit_calls)``.
    """
    from daydream.agent import reset_state

    _seed_fix_resume(target, [_fix_item()])

    # Silence the flow's terminal noise; the test phase is observed through the
    # backend script, not scraped rendering.
    monkeypatch.setattr("daydream.deep.orchestrator.print_preflight_notice", lambda *a, **k: None)
    monkeypatch.setattr("daydream.deep.orchestrator.print_stage_progress", lambda *a, **k: None)
    monkeypatch.setattr("daydream.deep.orchestrator.print_verification_summary", lambda *a, **k: None)
    monkeypatch.setattr(
        "daydream.phases.console",
        type("C", (), {"print": lambda *a, **kw: None})(),
    )
    monkeypatch.setattr(
        "daydream.runner.console",
        type("C", (), {"print": lambda *a, **kw: None})(),
    )

    test_backend = ScriptedBackend(script=[*script, _BEYOND_SCRIPT], model="test-model-xyz")
    stub_backend = ScriptedBackend(model="stub-model")

    monkeypatch.setattr(
        "daydream.runner._resolve_backend",
        lambda _config, phase, cache=None, **_kwargs: (
            test_backend if phase == "test" else stub_backend
        ),
    )
    monkeypatch.setattr("daydream.deep.orchestrator.phase_verify_recommendations", _stub_verify)

    async def _noop_fix(*_a: Any, **_k: Any) -> dict[str, str]:
        return {}

    commit_calls: list[bool] = []

    async def _spy_commit(*_a: Any, **_k: Any) -> None:
        commit_calls.append(True)

    monkeypatch.setattr("daydream.deep.orchestrator.phase_fix_parallel", _noop_fix)
    monkeypatch.setattr("daydream.deep.orchestrator.phase_commit_push", _spy_commit)
    monkeypatch.setattr("daydream.phases.clipboard_available", lambda: clipboard_is_available)

    if stdin_answers is not None:
        answers = iter(stdin_answers)

        def _feed_input(*_a: Any, **_k: Any) -> str:
            return next(answers)

        monkeypatch.setattr("builtins.input", _feed_input)
        monkeypatch.setattr("daydream.runner._stdin_isatty", lambda: True)
        monkeypatch.delenv("CI", raising=False)
    else:

        def _forbidden_input(*_a: Any, **_k: Any) -> str:
            raise AssertionError(stdin_guard_message or "stdin must not be touched")

        monkeypatch.setattr("builtins.input", _forbidden_input)

    reset_state()
    try:
        exit_code = await runner.run(config)
    finally:
        reset_state()

    return exit_code, test_backend, commit_calls


@pytest.mark.asyncio
async def test_fix_cycle_failing_tests_abort_writes_handoff(
    monkeypatch, feature_branch_repo, make_config
):
    """Real-path: an interactive deep shallow run whose tests FAIL and the operator
    aborts (heal-menu "4") writes a handoff and exits 1 without committing.

    Drives the deep fix cycle with the REAL ``phase_test_and_heal`` over a
    scripted failing test run. Fix-gate answered "y"; the heal menu answered "4"
    (abort): the read-only failure-summarizer runs, ``handoff.md`` lands in the
    live run directory, and the run exits 1. The mutating heal fix agent is never
    launched ("Analyze the failures and fix them" absent).
    """
    exit_code, test_backend, commit_calls = await _drive_fix_cycle_failing(
        monkeypatch,
        feature_branch_repo,
        make_config(feature_branch_repo, start_at="fix", shallow=True, non_interactive=False),
        script=[_FAIL_TURN, _handoff_turn("# Handoff\n\ninteractive abort")],
        stdin_answers=["y", "4"],
    )

    assert exit_code == 1

    handoffs = list(feature_branch_repo.glob(".daydream/runs/*/handoff.md"))
    assert len(handoffs) == 1, f"expected exactly one handoff.md, got {handoffs!r}"
    assert handoffs[0].read_text(encoding="utf-8") == "# Handoff\n\ninteractive abort"

    assert commit_calls == [], "a commit ran despite tests failing"

    # Exactly two test-backend calls: the failing test run + the read-only
    # summarizer. The mutating heal fix agent was never launched.
    assert len(test_backend.prompts) == 2
    assert "read-only failure-summarizer" in test_backend.prompts[1]
    assert all(
        "Analyze the failures and fix them" not in p for p in test_backend.prompts
    ), test_backend.prompts


@pytest.mark.asyncio
async def test_fix_cycle_clipboard_timeout_keeps_event_loop_responsive_and_shows_manual_copy_guidance(
    monkeypatch, feature_branch_repo, make_config
):
    """Real-path: a hung clipboard utility during the confirmed failure handoff is bounded to
    5s, runs off the event loop, degrades to the manual-copy warning, and the run still exits 1.

    Drives the deep fix cycle with the REAL phase_test_and_heal over a scripted failing test
    run. The operator confirms the fix gate, aborts the heal menu, and confirms the clipboard
    copy. The clipboard subprocess fake blocks 0.3s (a compressed stand-in for the 5s bound),
    raises TimeoutExpired, and the real copy_to_clipboard returns False -> the manual-copy
    warning fires. An event-loop ticker must keep ticking during the worker-thread block —
    proving the copy is offloaded, not run on the loop.
    """
    from daydream import clipboard

    warnings: list[str] = []
    monkeypatch.setattr(
        "daydream.phases.print_warning",
        lambda console_arg, message: warnings.append(message),
    )
    successes: list[str] = []
    monkeypatch.setattr(
        "daydream.phases.print_success",
        lambda console_arg, message: successes.append(message),
    )

    observed_timeouts: list[Any] = []
    state = {"ticks": 0, "at_entry": -1, "at_release": -1}
    stop_tick = threading.Event()

    def _blocking_run(argv: list[str], **kwargs: Any) -> None:
        observed_timeouts.append(kwargs.get("timeout"))
        state["at_entry"] = state["ticks"]
        time.sleep(0.3)  # simulated hung clipboard utility, bounded (stands in for 5s)
        state["at_release"] = state["ticks"]
        raise subprocess.TimeoutExpired(cmd=argv, timeout=kwargs.get("timeout", 5.0))

    monkeypatch.setattr(
        clipboard,
        "subprocess",
        SimpleNamespace(run=_blocking_run, SubprocessError=subprocess.SubprocessError),
    )
    monkeypatch.setattr("daydream.clipboard._detect_clipboard_command", lambda: ["pbcopy"])

    async def _ticker() -> None:
        while not stop_tick.is_set():
            state["ticks"] += 1
            await anyio.sleep(0.001)

    ticker_task = asyncio.create_task(_ticker())

    exit_code, _backend, commit_calls = await _drive_fix_cycle_failing(
        monkeypatch,
        feature_branch_repo,
        make_config(feature_branch_repo, start_at="fix", shallow=True, non_interactive=False),
        script=[_FAIL_TURN, _handoff_turn("# Handoff\n\nclipboard timeout")],
        stdin_answers=["y", "4", "y"],
        clipboard_is_available=True,
    )
    stop_tick.set()
    await ticker_task

    assert exit_code == 1
    assert observed_timeouts == [5], f"expected timeout exactly 5, got {observed_timeouts}"
    assert state["at_release"] > state["at_entry"], (
        "event loop did not tick during the blocked clipboard copy — the copy is running "
        "synchronously on the loop, not offloaded"
    )
    assert any(
        "Clipboard copy failed; copy manually from path above" in m for m in warnings
    ), f"manual-copy warning missing; got {warnings!r}"
    assert successes == [], f"expected no success message, got {successes!r}"
    assert commit_calls == [], "a commit ran despite tests failing"

    handoffs = list(feature_branch_repo.glob(".daydream/runs/*/handoff.md"))
    assert len(handoffs) == 1, f"expected exactly one handoff.md, got {handoffs!r}"
    assert handoffs[0].read_text(encoding="utf-8") == "# Handoff\n\nclipboard timeout"


@pytest.mark.asyncio
async def test_fix_cycle_failing_tests_bounded_fix_then_handoff(
    monkeypatch, feature_branch_repo, make_config
):
    """Real-path: a --yes shallow deep run whose tests FAIL runs ONE fix attempt
    then aborts.

    Drives the deep fix cycle with the REAL ``phase_test_and_heal`` against a
    scripted backend: fail → fix → fail → handoff (summarizer). With
    ``assume="yes"`` the bounded-loop guard (``decision is True and
    retries_used > 0``) fires after the first auto fix attempt, writing
    ``handoff.md`` and returning exit code 1. The fix prompt text ("Analyze the
    failures and fix them") appears in exactly one call (the fix agent), proving
    the fix ran once and only once. stdin must never be touched.
    """
    exit_code, test_backend, commit_calls = await _drive_fix_cycle_failing(
        monkeypatch,
        feature_branch_repo,
        make_config(feature_branch_repo, start_at="fix", shallow=True, assume="yes"),
        # Script: fail → fix (one bounded attempt) → fail → handoff (summarizer).
        # Any 5th call raises via the beyond-script turn, proving the loop ends.
        script=[
            _FAIL_TURN,
            _FIX_TURN,
            _FAIL_TURN,
            _handoff_turn("# Handoff\n\n--yes bounded fix failure"),
        ],
        stdin_guard_message="input() must not be called under --yes",
    )

    assert exit_code == 1

    handoffs = list(feature_branch_repo.glob(".daydream/runs/*/handoff.md"))
    assert len(handoffs) == 1, f"expected exactly one handoff.md, got {handoffs!r}"
    assert handoffs[0].read_text(encoding="utf-8") == "# Handoff\n\n--yes bounded fix failure"

    assert commit_calls == [], "a commit ran despite tests failing"

    # Exactly four test-backend calls: fail → fix → fail → summarizer. No 5th
    # call (bounded-loop guard fired); call 4 ran read-only.
    assert len(test_backend.prompts) == 4, test_backend.prompts
    assert "Analyze the failures and fix them" in test_backend.prompts[1]
    assert "read-only failure-summarizer" in test_backend.prompts[3]
    assert test_backend.read_only_calls == [False, False, False, True], (
        test_backend.read_only_calls
    )
