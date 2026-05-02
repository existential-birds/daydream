"""End-to-end deep-mode PR-comment integration test.

Drives the FULL ``daydream.runner.run`` pipeline with ``deep=True``, mocking
ONLY at the Claude SDK boundary and the final ``gh`` PR-posting transport.
Captures the markdown produced by ``pr_review.build_payload`` and asserts
on the production-bug symptoms the user reported:

    Mode: deep review
    Model: unknown
    Cost: $0.00
    Tokens: 0 in -> 0 out
    Per-phase breakdown rows: Model = unknown, Cost = $0.00

The test is intentionally red until the runner / backend / trajectory
recorder thread the real SDK model id and per-step usage data into every
agent step. No production code is modified by this file.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

REAL_MODEL_ID = "claude-opus-4-7-20251101"


# ---------------------------------------------------------------------------
# Fake SDK message types (real-shape: AssistantMessage carries .model, no
# .usage; ResultMessage carries .total_cost_usd + .usage). ClaudeBackend.execute
# does isinstance() against the symbols imported into daydream.backends.claude;
# we monkeypatch those symbols to point at these so isinstance returns True.
# ---------------------------------------------------------------------------


@dataclass
class FakeTextBlock:
    text: str


@dataclass
class FakeThinkingBlock:
    thinking: str


@dataclass
class FakeToolUseBlock:
    id: str
    name: str
    input: dict[str, Any] | None = None


@dataclass
class FakeToolResultBlock:
    tool_use_id: str
    content: str | None = None
    is_error: bool = False


@dataclass
class FakeAssistantMessage:
    content: list[Any]
    model: str
    parent_tool_use_id: str | None = None
    error: object | None = None


@dataclass
class FakeUserMessage:
    content: list[Any] = field(default_factory=list)


@dataclass
class FakeResultMessage:
    total_cost_usd: float | None = None
    usage: dict[str, Any] | None = None
    structured_output: Any = None
    subtype: str = "success"
    duration_ms: int = 0
    duration_api_ms: int = 0
    is_error: bool = False
    num_turns: int = 1
    session_id: str = "fake-session"
    result: str | None = None


# ---------------------------------------------------------------------------
# Fake ClaudeSDKClient. Inspects each query() prompt, picks a canned response,
# and emulates the agent's tool-use side effects (writing per-stack review
# files and the merged review report) so downstream phases do not crash on
# missing files. We mock only the SDK boundary; the side-effect file writes
# stand in for the Bash/Write tool calls a real agent would have made.
# ---------------------------------------------------------------------------


_OUTPUT_PATH_RE = re.compile(r"to ([^\s]+\.(?:md|json))")


def _extract_output_path(prompt: str) -> Path | None:
    """Pull the first ``... to <path>.md|.json`` reference out of a prompt."""
    m = _OUTPUT_PATH_RE.search(prompt)
    if not m:
        return None
    return Path(m.group(1))


_PER_STACK_REPORT = (
    "# Per-stack review\n"
    "\n"
    "## Issues\n"
    "\n"
    "1. [foo.py:1] Use a more descriptive function name\n"
    "   The current name is ambiguous.\n"
)

_MERGED_REPORT = (
    "# Daydream merged review\n"
    "\n"
    "## Issues\n"
    "\n"
    "1. [foo.py:1] Use a more descriptive function name\n"
    "   The current name is ambiguous.\n"
)


class _FakeSDKClient:
    """Per-call canned response + simulated tool-use side effects."""

    def __init__(self, options: Any = None) -> None:
        self.options = options
        self._prompt: str = ""

    async def __aenter__(self) -> "_FakeSDKClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        return None

    async def query(self, prompt: str) -> None:
        self._prompt = prompt
        self._maybe_write_artifact(prompt)

    @staticmethod
    def _maybe_write_artifact(prompt: str) -> None:
        """Simulate the real agent's Write tool: produce stub report files
        whenever the prompt instructs the agent to write one."""
        out = _extract_output_path(prompt)
        if out is None:
            return
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            return
        name = out.name
        if name == "review-output.md":
            out.write_text(_MERGED_REPORT)
        elif name.startswith("stack-") and name.endswith("-review.md"):
            out.write_text(_PER_STACK_REPORT)
        # alternatives.json / intent.md / records.json are written by Python.

    async def receive_response(self) -> Any:
        for msg in self._build_messages(self._prompt):
            yield msg

    @staticmethod
    def _exploration_messages(
        structured: dict[str, Any],
        *,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> list[Any]:
        """Build the response stream for one exploration specialist call.

        Emits an AssistantMessage with a tool-use block, a UserMessage with
        the matching tool-result, then a final AssistantMessage + Result so
        ``ClaudeBackend.execute`` records a non-zero ``tool_calls`` count
        AND surfaces real cost + usage on the trailing CostEvent. The
        CostEvent is what carries ``model_name`` to the recorder, so the
        fork's child trajectory should pick up REAL_MODEL_ID.
        """
        tool_id = f"toolu_{tool_name.lower()}_01"
        return [
            FakeAssistantMessage(
                content=[
                    FakeToolUseBlock(id=tool_id, name=tool_name, input=tool_input),
                ],
                model=REAL_MODEL_ID,
            ),
            FakeUserMessage(
                content=[
                    FakeToolResultBlock(
                        tool_use_id=tool_id,
                        content="ok",
                        is_error=False,
                    ),
                ],
            ),
            FakeAssistantMessage(
                content=[FakeTextBlock(text="exploration complete")],
                model=REAL_MODEL_ID,
            ),
            FakeResultMessage(
                structured_output=structured,
                total_cost_usd=0.12,
                usage={
                    "input_tokens": 5000,
                    "output_tokens": 400,
                    "cache_read_input_tokens": 1200,
                },
            ),
        ]

    @staticmethod
    def _build_messages(prompt: str) -> list[Any]:
        """Pick the per-prompt canned response.

        Every response carries (a) a real SDK model id on AssistantMessage
        and (b) real cost + usage on ResultMessage, so the trajectory has
        non-zero metrics to render even though no real Claude API was hit.
        """
        pl = prompt.lower()

        # Exploration specialist prompts (pattern-scanner, dependency-tracer,
        # test-mapper). These run inside ``maybe_fork`` from
        # ``exploration_runner.pre_scan`` and are the production-bug path:
        # the parent recorder's ``agent_model_name`` is still the generic
        # backend label when ``create_dispatch_step`` runs, so the resulting
        # 'Exploration' row in the rollup table renders 'unknown' / '$0.00'.
        if "pattern-scanner" in pl:
            return _FakeSDKClient._exploration_messages(
                {"conventions": [], "guidelines": []},
                tool_name="Read",
                tool_input={"file_path": "CLAUDE.md"},
            )
        if "dependency-tracer" in pl:
            return _FakeSDKClient._exploration_messages(
                {"affected_files": [], "dependencies": []},
                tool_name="Grep",
                tool_input={"pattern": "import foo"},
            )
        if "test-mapper" in pl:
            return _FakeSDKClient._exploration_messages(
                {"affected_files": []},
                tool_name="Read",
                tool_input={"file_path": "tests/test_foo.py"},
            )

        # phase_alternative_review uses ALTERNATIVE_REVIEW_SCHEMA: the
        # ResultMessage MUST carry structured_output with the right shape
        # or phase_alternative_review crashes / returns an empty list.
        if "architectural alternatives" in pl or (
            "alternative" in pl and "intent" in pl and "given" in pl
        ):
            return [
                FakeAssistantMessage(
                    content=[FakeTextBlock(text="evaluating alternatives")],
                    model=REAL_MODEL_ID,
                ),
                FakeResultMessage(
                    structured_output={"issues": []},
                    total_cost_usd=0.10,
                    usage={
                        "input_tokens": 3000,
                        "output_tokens": 250,
                        "cache_read_input_tokens": 1000,
                    },
                ),
            ]

        # phase_parse_feedback uses FEEDBACK_SCHEMA: ResultMessage must
        # carry structured_output with the right shape.
        if "extract only actionable issues" in pl or "read the review output file" in pl:
            return [
                FakeAssistantMessage(
                    content=[FakeTextBlock(text="parsing")],
                    model=REAL_MODEL_ID,
                ),
                FakeResultMessage(
                    structured_output={"issues": []},
                    total_cost_usd=0.05,
                    usage={
                        "input_tokens": 1500,
                        "output_tokens": 100,
                        "cache_read_input_tokens": 500,
                    },
                ),
            ]

        # phase_understand_intent: free-form text.
        if "understand" in pl and "intent" in pl:
            return [
                FakeAssistantMessage(
                    content=[FakeTextBlock(text="The PR refactors foo() for clarity.")],
                    model=REAL_MODEL_ID,
                ),
                FakeResultMessage(
                    total_cost_usd=0.08,
                    usage={
                        "input_tokens": 2000,
                        "output_tokens": 150,
                        "cache_read_input_tokens": 800,
                    },
                ),
            ]

        # phase_per_stack_reviews + phase_cross_stack_merge: free-form text;
        # report files are written by _maybe_write_artifact via prompt scan.
        return [
            FakeAssistantMessage(
                content=[FakeTextBlock(text="ok, wrote the review")],
                model=REAL_MODEL_ID,
            ),
            FakeResultMessage(
                total_cost_usd=0.20,
                usage={
                    "input_tokens": 4000,
                    "output_tokens": 600,
                    "cache_read_input_tokens": 1500,
                },
            ),
        ]


# ---------------------------------------------------------------------------
# Repo + monkeypatch fixtures.
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> None:
    subprocess.run(  # noqa: S603 - args are not user-controlled
        ["git", *args],  # noqa: S607
        cwd=repo,
        capture_output=True,
        check=True,
    )


@pytest.fixture
def deep_target(tmp_path: Path) -> Path:
    """Real git repo on a feature branch with one Python file changed."""
    repo = tmp_path / "deep_repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "foo.py").write_text("def foo():\n    return 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "init")
    _git(repo, "checkout", "-b", "feature")
    (repo / "foo.py").write_text("def foo():\n    return 2\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "tweak foo")
    return repo


@pytest.fixture
def deep_target_multi(tmp_path: Path) -> Path:
    """Real git repo with >=4 Python files changed between main and feature.

    Production bug repro: ``select_tier(count_changed_files(diff))`` returns
    ``"parallel"`` for 4+ files, which is the path that spawns the three
    exploration specialist subagents (pattern_scanner, dependency_tracer,
    test_mapper) under ``maybe_fork``. The single-file fixture above
    short-circuits with ``"skip"`` and never exercises the broken row.
    """
    repo = tmp_path / "deep_repo_multi"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    # Seed five Python files on main so the tree-sitter index has something
    # to resolve; enough room to mutate four of them on the branch.
    for name in ("foo.py", "bar.py", "baz.py", "qux.py", "quux.py"):
        (repo / name).write_text(f"def {name[:-3]}():\n    return 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "init")
    _git(repo, "checkout", "-b", "feature")
    # Mutate four of the five so count_changed_files >= 4 -> tier="parallel".
    for name in ("foo.py", "bar.py", "baz.py", "qux.py"):
        (repo / name).write_text(f"def {name[:-3]}():\n    return 2\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "tweak four files")
    # Sanity: verify diff count up-front so a future refactor that breaks the
    # threshold trips loudly here, not silently downstream.
    diff_out = subprocess.run(  # noqa: S603 - controlled args
        ["git", "diff", "--name-only", "main..HEAD"],  # noqa: S607
        cwd=repo,
        capture_output=True,
        check=True,
        text=True,
    )
    changed = [ln for ln in diff_out.stdout.splitlines() if ln]
    assert len(changed) >= 4, (
        f"deep_target_multi fixture should produce >=4 changed files; "
        f"got {len(changed)}: {changed!r}"
    )
    return repo


@pytest.fixture
def patch_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch every SDK symbol that ClaudeBackend.execute does isinstance on."""
    monkeypatch.setattr(
        "daydream.backends.claude.ClaudeSDKClient", _FakeSDKClient,
    )
    monkeypatch.setattr(
        "daydream.backends.claude.AssistantMessage", FakeAssistantMessage,
    )
    monkeypatch.setattr(
        "daydream.backends.claude.UserMessage", FakeUserMessage,
    )
    monkeypatch.setattr(
        "daydream.backends.claude.ResultMessage", FakeResultMessage,
    )
    monkeypatch.setattr(
        "daydream.backends.claude.TextBlock", FakeTextBlock,
    )
    monkeypatch.setattr(
        "daydream.backends.claude.ThinkingBlock", FakeThinkingBlock,
    )
    monkeypatch.setattr(
        "daydream.backends.claude.ToolUseBlock", FakeToolUseBlock,
    )
    monkeypatch.setattr(
        "daydream.backends.claude.ToolResultBlock", FakeToolResultBlock,
    )


# ---------------------------------------------------------------------------
# gh / PR plumbing patches. _post() short-circuits without an open PR; we
# patch find_open_pr to return a fake PRInfo and _submit_review to capture
# the payload (the ONLY allowed non-SDK mock per the brief — it stands in
# for the gh-CLI subprocess that would otherwise post to GitHub).
# ---------------------------------------------------------------------------


@dataclass
class _CapturedPost:
    payloads: list[dict[str, Any]] = field(default_factory=list)


@pytest.fixture
def captured_post(monkeypatch: pytest.MonkeyPatch) -> _CapturedPost:
    """Wire find_open_pr + _submit_review so build_payload runs and we see
    the rendered markdown without ever touching GitHub."""
    from daydream import pr_review

    captured = _CapturedPost()

    fake_pr = pr_review.PRInfo(
        number=123,
        head_sha="0" * 40,
        base_sha="1" * 40,
        base_ref="main",
        owner="test-owner",
        repo="test-repo",
        url="https://example/pr/123",
    )

    monkeypatch.setattr(
        "daydream.pr_review.find_open_pr",
        lambda target_dir: fake_pr,
    )

    def _capture(
        target_dir: Path, pr: pr_review.PRInfo, payload: dict[str, Any]
    ) -> tuple[str, None]:
        captured.payloads.append(payload)
        return "https://example/pr/123#review-1", None

    monkeypatch.setattr(
        "daydream.pr_review._submit_review", _capture,
    )

    return captured


# ---------------------------------------------------------------------------
# Misc: silence Rich UI noise + answer interactive prompts.
# ---------------------------------------------------------------------------


_UI_FUNCS: tuple[str, ...] = (
    "print_stage_progress",
    "print_preflight_notice",
    "print_phase_hero",
    "print_info",
    "print_success",
    "print_warning",
    "print_error",
    "print_dim",
    "print_issues_table",
    "print_iteration_divider",
    "print_skipped_phases",
    "print_menu",
    "print_summary",
    "print_fix_progress",
    "print_fix_complete",
)


def _silence_ui(monkeypatch: pytest.MonkeyPatch) -> None:
    noop: Callable[..., None] = lambda *a, **kw: None  # noqa: E731
    for module in (
        "daydream.deep.orchestrator",
        "daydream.phases",
        "daydream.runner",
        "daydream.pr_review",
    ):
        for name in _UI_FUNCS:
            monkeypatch.setattr(f"{module}.{name}", noop, raising=False)


def _answer_prompts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Phase prompts: y for intent confirmation; n for fix gate + PR post.

    The PR-post prompt lives in ``daydream.pr_review.prompt_user`` and asks
    "Post these as a PR review? [y/N]" — we MUST answer ``y`` or the test
    never reaches ``_submit_review``.
    """
    monkeypatch.setattr(
        "daydream.phases.prompt_user", lambda *a, **kw: "y", raising=False,
    )
    monkeypatch.setattr(
        "daydream.deep.orchestrator.prompt_user",
        lambda *a, **kw: "n",
        raising=False,
    )
    monkeypatch.setattr(
        "daydream.pr_review.prompt_user",
        lambda *a, **kw: "y",
        raising=False,
    )
    monkeypatch.setattr(
        "daydream.runner.prompt_user", lambda *a, **kw: "n", raising=False,
    )


# ---------------------------------------------------------------------------
# Markdown extraction helpers.
# ---------------------------------------------------------------------------


def _line_starting(markdown: str, prefix: str) -> str:
    for line in markdown.splitlines():
        if line.startswith(prefix):
            return line
    raise AssertionError(
        f"No line starting with {prefix!r} in markdown:\n{markdown}"
    )


def _phase_rows(markdown: str) -> list[str]:
    rows: list[str] = []
    for line in markdown.splitlines():
        # Header / separator rows start with "| Phase" or "|---".
        if line.startswith("|") and not line.startswith("| Phase") and not line.startswith("|---"):
            rows.append(line)
    return rows


def _row_cells(row: str) -> list[str]:
    return [c.strip() for c in row.strip("|").split("|")]


# ---------------------------------------------------------------------------
# The test.
# ---------------------------------------------------------------------------


async def test_deep_run_produces_pr_comment_with_real_model_and_metrics(
    deep_target: Path,
    patch_sdk: None,
    captured_post: _CapturedPost,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Drive ``daydream.runner.run`` end-to-end in deep mode.

    Mocks:
      - ``daydream.backends.claude.ClaudeSDKClient`` and the SDK message-type
        symbols imported into that module (isinstance-pinning).
      - ``pr_review.find_open_pr`` returns a synthetic ``PRInfo`` (avoids
        ``gh pr list`` subprocess).
      - ``pr_review._submit_review`` captures the payload (avoids ``gh api``
        subprocess that would actually POST the comment).

    Everything else — RunConfig dispatch, _run_loop_deep, run_deep, the
    real ``TrajectoryRecorder``, ``ClaudeBackend.execute``, ``run_agent``,
    every phase function, ``Invocation._dispatch``, ``build_payload``, the
    ``pr_comment_renderer`` — runs unmodified.
    """
    from daydream.exploration import ExplorationContext
    from daydream.runner import RunConfig, run

    _silence_ui(monkeypatch)
    _answer_prompts(monkeypatch)

    config = RunConfig(
        target=str(deep_target),
        # Production cli sets deep=True via --deep but Stage 4.2 makes deep
        # the default for output_mode="loop" + non-shallow. Set it explicitly
        # for clarity even though the default would route the same way.
        deep=True,
        cleanup=False,
        archive=False,
    )
    # The single-file diff means count_changed_files == 1 -> select_tier
    # returns "skip", so the orchestrator constructs a fresh
    # ExplorationContext() without invoking the backend. We do NOT pre-
    # populate exploration_context — letting the orchestrator's natural
    # pre-scan path run is part of the integration coverage. (Reference to
    # ExplorationContext silenced for the linter below.)
    _ = ExplorationContext

    exit_code = await run(config)

    assert exit_code == 0, f"run() returned {exit_code}"
    assert captured_post.payloads, (
        "build_payload was never called: deep flow did not reach _post"
    )
    payload = captured_post.payloads[-1]
    body = payload["body"]
    assert isinstance(body, str)

    # --- Mode line: should reflect deep review --------------------------
    mode_line = _line_starting(body, "- **Mode:**")
    assert "deep review" in mode_line.lower(), (
        f"expected 'deep review' in Mode line, got: {mode_line!r}\n\n"
        f"full body:\n{body}"
    )

    # --- Model line: real SDK id, not 'unknown' / backend alias ---------
    model_line = _line_starting(body, "- **Model:**")
    assert REAL_MODEL_ID in model_line, (
        f"BUG: rollup Model line is missing the real SDK model id "
        f"({REAL_MODEL_ID!r}).\n  got: {model_line!r}\n\n"
        f"full body:\n{body}"
    )
    assert "unknown" not in model_line, (
        f"BUG: rollup Model line still says 'unknown' (no model id "
        f"propagated from SDK).\n  got: {model_line!r}"
    )

    # --- Cost line: non-zero ---------------------------------------------
    cost_line = _line_starting(body, "- **Cost:**")
    assert "$0.00" not in cost_line, (
        f"BUG: rollup Cost line shows $0.00 — per-step cost never landed "
        f"on Step.metrics.\n  got: {cost_line!r}"
    )

    # --- Tokens line: non-zero in / out ---------------------------------
    tokens_line = _line_starting(body, "- **Tokens:**")
    assert not re.search(r"(?<!\d)0 in\b", tokens_line), (
        f"BUG: rollup Tokens line shows '0 in'.\n  got: {tokens_line!r}"
    )
    assert not re.search(r"(?<!\d)0 out\b", tokens_line), (
        f"BUG: rollup Tokens line shows '0 out'.\n  got: {tokens_line!r}"
    )

    # --- Per-phase breakdown: at least 2 rows, each with real model + cost
    rows = _phase_rows(body)
    assert len(rows) >= 2, (
        f"expected >= 2 per-phase rows, got {len(rows)}.\n"
        f"  rows: {rows}\n\nfull body:\n{body}"
    )
    for row in rows:
        cells = _row_cells(row)
        # Layout: | Phase | Model | Steps | Tools | Input | Cached | Output | Cost |
        assert len(cells) >= 8, f"unexpected row layout: {row!r}"
        phase_name, model_cell, steps_cell, _tools, input_cell, _cached, _out, cost_cell = cells[:8]
        assert model_cell != "unknown", (
            f"BUG: row {phase_name!r} has Model='unknown' "
            f"(SDK model id never propagated to the per-phase rollup).\n"
            f"  row: {row!r}"
        )
        assert REAL_MODEL_ID in model_cell, (
            f"BUG: row {phase_name!r} Model cell missing real SDK id "
            f"{REAL_MODEL_ID!r}.\n  row: {row!r}"
        )
        # Steps cell is a stringified int. A row with 0 steps would not
        # have rendered, but be defensive.
        try:
            steps_n = int(steps_cell)
        except ValueError:
            steps_n = 0
        assert steps_n >= 1, (
            f"BUG: row {phase_name!r} has Steps={steps_cell} (expected >=1)."
        )
        # Input tokens may render with thousand-separator (>=1000).
        # Anything other than literal '0' counts as non-zero.
        assert input_cell != "0", (
            f"BUG: row {phase_name!r} has Input=0 (per-step metrics empty).\n"
            f"  row: {row!r}"
        )
        assert cost_cell != "$0.00", (
            f"BUG: row {phase_name!r} has Cost=$0.00 "
            f"(per-step cost never landed).\n  row: {row!r}"
        )


# ---------------------------------------------------------------------------
# Exploration-row reproduction test.
#
# The single-file fixture above exercises ``select_tier(...) == "skip"`` and
# never spawns the exploration sub-recorders. Production reports the
# Exploration row rendering as ``unknown / 4 / 53 / 0 / 0 / 0 / $0.00`` —
# that lives behind ``select_tier(...) in ("single", "parallel")``. This
# test forces the parallel tier and asserts on the broken row.
# ---------------------------------------------------------------------------


async def test_deep_run_exploration_row_has_real_model_and_metrics(
    deep_target_multi: Path,
    patch_sdk: None,
    captured_post: _CapturedPost,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reproduce the production 'Exploration ... unknown ... $0.00' row.

    With >=4 changed Python files, ``select_tier`` returns ``"parallel"``,
    which spawns three exploration specialists under ``maybe_fork``. Each
    specialist runs through ``ClaudeBackend.execute`` -> ``Invocation``
    inside its own forked ``TrajectoryRecorder``. The fork's child recorder
    inherits the parent's ``agent_model_name`` (still ``""``) and is
    supposed to upgrade it once the SDK CostEvent surfaces a real model id.

    What the renderer sees today, per the user's bug report:

        Exploration       unknown  4 53  0  0  0  $0.00

    The assertions below pin every column the user called out as broken.
    """
    from daydream.runner import RunConfig, run

    _silence_ui(monkeypatch)
    _answer_prompts(monkeypatch)

    config = RunConfig(
        target=str(deep_target_multi),
        deep=True,
        cleanup=False,
        archive=False,
    )

    exit_code = await run(config)

    assert exit_code == 0, f"run() returned {exit_code}"
    assert captured_post.payloads, (
        "build_payload was never called: deep flow did not reach _post"
    )
    payload = captured_post.payloads[-1]
    body = payload["body"]
    assert isinstance(body, str)

    # Print rendered markdown so any future failure trace shows what we
    # observed. (Kept as -s diagnostics; harmless when assertions pass.)
    print("\n=== RENDERED PR COMMENT BODY ===")
    print(body)
    print("=== END RENDERED PR COMMENT BODY ===\n")

    # --- Top-level rollup must reflect a real model + non-zero cost -------
    model_line = _line_starting(body, "- **Model:**")
    assert "unknown" not in model_line.lower(), (
        f"BUG: rollup Model line still says 'unknown' even when exploration "
        f"forks observed real model ids.\n  got: {model_line!r}\n\n"
        f"full body:\n{body}"
    )

    cost_line = _line_starting(body, "- **Cost:**")
    assert "$0.00" not in cost_line and "—" not in cost_line, (
        f"BUG: rollup Cost line is degraded.\n  got: {cost_line!r}"
    )

    # --- Locate the Exploration row in the per-phase breakdown ------------
    rows = _phase_rows(body)
    exploration_rows = [
        r for r in rows if re.search(r"\|\s*Exploration\s*\|", r, re.IGNORECASE)
    ]
    assert exploration_rows, (
        "BUG: per-phase breakdown is missing an 'Exploration' row entirely.\n"
        f"  rows: {rows!r}\n\nfull body:\n{body}"
    )
    assert len(exploration_rows) == 1, (
        f"unexpected: {len(exploration_rows)} Exploration rows in breakdown."
    )
    exploration_row = exploration_rows[0]
    cells = _row_cells(exploration_row)
    # Layout: | Phase | Model | Steps | Tools | Input | Cached | Output | Cost |
    assert len(cells) >= 8, f"unexpected row layout: {exploration_row!r}"
    (
        _phase_name,
        model_cell,
        _steps_cell,
        tools_cell,
        input_cell,
        _cached_cell,
        _output_cell,
        cost_cell,
    ) = cells[:8]

    # --- The four production-bug symptoms, asserted one at a time. --------

    # 1. Model column — production shows 'unknown'. The fork's child
    #    recorder should have upgraded this to the SDK id.
    assert model_cell.lower() != "unknown", (
        f"BUG: Exploration row Model='unknown' (matches production bug).\n"
        f"  row: {exploration_row!r}\n\nfull body:\n{body}"
    )
    assert REAL_MODEL_ID in model_cell, (
        f"BUG: Exploration row Model cell is missing real SDK id "
        f"{REAL_MODEL_ID!r}.\n  got Model cell: {model_cell!r}\n"
        f"  row: {exploration_row!r}"
    )

    # 2. Cost column — production shows '$0.00' (or '—'). Per-step cost
    #    should be aggregated from the fork CostEvents.
    assert cost_cell != "$0.00", (
        f"BUG: Exploration row Cost='$0.00' (matches production bug — "
        f"per-step cost from fork trajectories never aggregated).\n"
        f"  row: {exploration_row!r}"
    )
    assert cost_cell != "—", (
        f"BUG: Exploration row Cost='—' (cost_unknown flipped True).\n"
        f"  row: {exploration_row!r}"
    )

    # 3. Tools column — production shows 53; even one tool call per fork
    #    should produce a non-zero count here.
    assert tools_cell != "0", (
        f"BUG: Exploration row Tools='0' but the forks issued tool calls.\n"
        f"  row: {exploration_row!r}"
    )

    # 4. Input column — production shows 0; CostEvent usage from each
    #    fork should aggregate to a non-zero input count.
    assert input_cell != "0", (
        f"BUG: Exploration row Input='0' (matches production bug — "
        f"per-step token usage from fork trajectories never aggregated).\n"
        f"  row: {exploration_row!r}"
    )
