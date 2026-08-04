"""Tests for daydream.runner.RunConfig and the unified ``run`` dispatch."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
def silence_runner_ui(silence_console: Callable[..., None], monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop ``daydream.runner``'s UI helpers (notably the ``print_phase_hero``
    banner) plus the shallow flow's end-of-run summary.

    ``daydream.flows.shallow``'s own ``print_phase_hero``/``print_dim`` bindings
    are deliberately left live: the HEAL-hero ordering test spies on them.
    """
    silence_console("daydream.runner")
    monkeypatch.setattr("daydream.flows.shallow.print_summary", lambda *a, **kw: None)


_DISPATCH_TARGETS = (
    "_run_pr_feedback",
    "_run_comment",
    "_run_review",
    "_run_loop_shallow",
    "_run_loop_deep",
    "_run_improve",
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("expected_target", "config_kwargs", "expected_attr", "expected_value"),
    [
        ("_run_pr_feedback", {"pr_number": 42, "bot": "botname"}, "pr_number", 42),
        # Auto-detected pr_number (no bot) routes to the deep loop, not PR feedback.
        ("_run_loop_deep", {"pr_number": 42, "bot": None}, "pr_number", 42),
        ("_run_comment", {"output_mode": "comment"}, "output_mode", "comment"),
        ("_run_review", {"output_mode": "review"}, "output_mode", "review"),
        ("_run_loop_shallow", {"output_mode": "loop", "shallow": True}, "shallow", True),
        # Stage 4.2: deep is the default. No flags required to route here.
        ("_run_loop_deep", {"output_mode": "loop"}, "shallow", False),
        ("_run_improve", {"flow_name": "improve"}, "flow_name", "improve"),
    ],
    ids=[
        "pr_feedback_when_pr_number_set",
        "auto_detected_pr_number_goes_deep",
        "comment_mode",
        "review_mode",
        "shallow_loop_when_explicit",
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
    exclusivity: shallow runs only when ``--shallow`` is explicitly set, and an
    auto-detected ``pr_number`` (no ``bot``) must not reach PR feedback.
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
    monkeypatch.setattr("daydream.runner._run_review", fake_review)

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
async def test_comment_mode_errors_when_no_open_pr_for_branch(
    monkeypatch, patch_workspace, silence_runner_ui, tmp_path, make_config
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

    config = make_config(tmp_path, output_mode="comment", branch="feat/missing")
    exit_code = await runner.run(config)

    assert exit_code == 1
    assert "No Open PR" == captured["title"]
    assert "no open PR for branch feat/missing" in captured["body"]


@pytest.mark.asyncio
async def test_run_feedback_routes_through_pr_feedback(
    monkeypatch, patch_workspace, silence_runner_ui, tmp_path, make_config
):
    """``run_feedback`` sets ``pr_number`` and re-enters dispatch."""
    called: dict[str, Any] = {}

    async def stub(work, config):
        called["pr"] = config.pr_number
        return 0

    monkeypatch.setattr("daydream.runner._run_pr_feedback", stub)
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
    work = make_work(tmp_path)
    chosen_model = "fixture-model-xyz"

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

    # The fetch/parse call sites moved into the registered pr-feedback flow steps.
    monkeypatch.setattr("daydream.flows.pr_feedback.phase_fetch_pr_feedback", _no_op_fetch)
    monkeypatch.setattr("daydream.flows.pr_feedback.phase_parse_feedback", _empty_parse)

    captured: list[str] = []
    monkeypatch.setattr(
        "daydream.runner.print_info",
        lambda _console, message: captured.append(message),
    )

    config = make_config(tmp_path, pr_number=42, bot="botname")

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


# --- Task 6: HEAL hero is followed by Model: dim line ----------------------


@pytest.mark.asyncio
async def test_run_loop_shallow_heal_hero_followed_by_model_line(
    monkeypatch, tmp_path, make_work, make_config, silence_runner_ui
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

    work = make_work(tmp_path)

    # Stub backends to carry distinct phase-specific models so the dim line's
    # source is unambiguous.
    backends_by_phase = {
        phase: ScriptedBackend(model=f"{phase}-model-xyz")
        for phase in ("review", "parse", "fix", "test")
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
        return (True, 0, True)

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

    config = make_config(tmp_path, start_at="fix", loop=False, shallow=True)

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


async def test_shallow_items_canonicalized_and_severity_ordered(
    monkeypatch, tmp_path, make_work, make_config, silence_runner_ui
):
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

    work = make_work(tmp_path)

    monkeypatch.setattr(
        "daydream.runner._resolve_backend",
        lambda _config, _phase, cache=None, **_kwargs: ScriptedBackend(model="stub-model"),
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
        return (True, 0, True)

    async def _stub_phase_commit_push(*_args, **_kwargs):
        return None

    monkeypatch.setattr("daydream.flows.shallow.phase_parse_feedback", _stub_phase_parse_feedback)
    monkeypatch.setattr("daydream.flows.shallow.phase_fix", _spy_phase_fix)
    monkeypatch.setattr("daydream.flows.shallow.phase_test_and_heal", _stub_phase_test_and_heal)
    monkeypatch.setattr("daydream.flows.shallow.phase_commit_push", _stub_phase_commit_push)

    config = make_config(tmp_path, start_at="fix", loop=False, shallow=True)

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


# --- non_interactive commit branch in _run_loop_shallow --------------------


@pytest.mark.asyncio
async def test_non_interactive_shallow_calls_phase_commit_push_auto(
    monkeypatch, tmp_path, make_work, make_config, silence_runner_ui
):
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

    work = make_work(tmp_path)

    monkeypatch.setattr(
        "daydream.runner._resolve_backend",
        lambda _config, _phase, cache=None, **_kwargs: ScriptedBackend(model="stub-model"),
    )

    async def _stub_phase_parse_feedback(_backend, _work):
        return [{"id": 1, "description": "X", "file": "foo.py", "line": 1}]

    async def _stub_phase_fix(*_args, **_kwargs):
        return None

    async def _stub_phase_test_and_heal(*_args, **_kwargs):
        return (True, 0, True)

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

    config = make_config(
        tmp_path, start_at="fix", loop=False, shallow=True, non_interactive=True
    )

    exit_code = await runner._run_loop_shallow(work, config)
    assert exit_code == 0
    assert auto_calls == [True], (
        "phase_commit_push_auto was not called when non_interactive=True"
    )
    assert interactive_calls == [], (
        "phase_commit_push was called despite non_interactive=True"
    )


async def _drive_shallow_failing_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    work: WorkContext,
    config: RunConfig,
    *,
    script: list[Turn],
    stdin_guard_message: str,
    apply_agent_state: Callable[[], None],
) -> tuple[int, ScriptedBackend, list[bool]]:
    """Drive ``_run_loop_shallow`` with the REAL ``phase_test_and_heal`` over a
    scripted failing test run.

    Holds everything the two failing-shallow real-path tests share: the review
    file seed, the scripted backend bound to the ``test`` phase, stubbed
    parse/fix, one commit spy across BOTH commit variants, a stdin trap, and the
    agent-state reset bracket. The differing script, config, agent axis, and
    assertions stay at the call sites.

    ``_BEYOND_SCRIPT`` is appended so a heal loop that runs past its script
    raises instead of silently replaying the last turn forever.

    Returns:
        ``(exit_code, test_backend, commit_calls)``.
    """
    from daydream.agent import reset_state
    from daydream.config import REVIEW_OUTPUT_FILE

    (tmp_path / REVIEW_OUTPUT_FILE).write_text("## Issues\n\n1. foo.py:1 - X\n")

    test_backend = ScriptedBackend(script=[*script, _BEYOND_SCRIPT], model="test-model")
    stub_backend = ScriptedBackend(model="stub-model")

    monkeypatch.setattr(
        "daydream.runner._resolve_backend",
        lambda _config, phase, cache=None, **_kwargs: (
            test_backend if phase == "test" else stub_backend
        ),
    )

    async def _stub_phase_parse_feedback(_backend, _work):
        return [{"id": 1, "description": "X", "file": "foo.py", "line": 1}]

    async def _stub_phase_fix(*_args, **_kwargs):
        return None

    commit_calls: list[bool] = []

    async def _spy_commit(*_args, **_kwargs):
        commit_calls.append(True)

    monkeypatch.setattr("daydream.flows.shallow.phase_parse_feedback", _stub_phase_parse_feedback)
    monkeypatch.setattr("daydream.flows.shallow.phase_fix", _stub_phase_fix)
    monkeypatch.setattr("daydream.flows.shallow.phase_commit_push_auto", _spy_commit)
    monkeypatch.setattr("daydream.flows.shallow.phase_commit_push", _spy_commit)

    def _forbidden_input(*_a: Any, **_kw: Any) -> str:
        raise AssertionError(stdin_guard_message)

    monkeypatch.setattr("builtins.input", _forbidden_input)
    monkeypatch.setattr(
        "daydream.phases.console",
        type("C", (), {"print": lambda *a, **kw: None})(),
    )

    reset_state()
    apply_agent_state()
    try:
        exit_code = await runner._run_loop_shallow(work, config)
    finally:
        reset_state()

    return exit_code, test_backend, commit_calls


@pytest.mark.asyncio
async def test_non_interactive_shallow_failing_tests_write_handoff_no_fix(
    monkeypatch, tmp_path, make_work, make_config, silence_runner_ui
):
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
    from daydream.agent import set_non_interactive

    exit_code, test_backend, commit_calls = await _drive_shallow_failing_run(
        monkeypatch,
        tmp_path,
        make_work(tmp_path),
        make_config(tmp_path, start_at="fix", loop=False, shallow=True, non_interactive=True),
        # The test phase's backend yields a failing test run, then the read-only
        # summarizer's handoff body -- exactly the choice-"4" path the guard takes.
        script=[_FAIL_TURN, _handoff_turn("# Handoff\n\nnon-interactive failure context")],
        stdin_guard_message=(
            "input() was called in non-interactive mode -- stdin must not be touched"
        ),
        apply_agent_state=lambda: set_non_interactive(True),
    )

    assert exit_code == 1

    handoffs = list(tmp_path.glob(".daydream/runs/*/handoff.md"))
    assert len(handoffs) == 1, f"expected exactly one handoff.md, got {handoffs!r}"
    assert handoffs[0].read_text(encoding="utf-8") == "# Handoff\n\nnon-interactive failure context"

    assert commit_calls == [], "a commit ran despite tests failing"

    # Exactly two test-backend calls: the failing test run + the read-only
    # summarizer. The mutating heal fix agent was never launched.
    assert len(test_backend.prompts) == 2
    assert "read-only failure-summarizer" in test_backend.prompts[1]
    assert all(
        "Analyze the failures and fix them" not in p for p in test_backend.prompts
    ), test_backend.prompts


@pytest.mark.asyncio
async def test_yes_shallow_failing_tests_bounded_fix_and_abort(
    monkeypatch, tmp_path, make_work, make_config, silence_runner_ui
):
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
    from daydream.agent import set_assume

    exit_code, test_backend, commit_calls = await _drive_shallow_failing_run(
        monkeypatch,
        tmp_path,
        make_work(tmp_path),
        make_config(
            tmp_path,
            start_at="fix",
            loop=False,
            shallow=True,
            non_interactive=False,
            assume="yes",
        ),
        # Script: fail → fix (one bounded attempt) → fail → handoff (summarizer).
        # Any 5th call raises via the beyond-script turn, proving the loop ends.
        script=[
            _FAIL_TURN,
            _FIX_TURN,
            _FAIL_TURN,
            _handoff_turn("# Handoff\n\n--yes bounded fix failure"),
        ],
        stdin_guard_message="input() must not be called under --yes",
        apply_agent_state=lambda: set_assume("yes"),
    )

    assert exit_code == 1

    handoffs = list(tmp_path.glob(".daydream/runs/*/handoff.md"))
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
