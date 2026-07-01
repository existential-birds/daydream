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

FIXTURE_MODEL_ID = "fixture-model-id"


# Fake SDK message types (real-shape: AssistantMessage carries .model, no .usage;
# ResultMessage carries .total_cost_usd + .usage). Monkeypatched over the symbols
# ClaudeBackend.execute isinstance-checks so they match.


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


# Fake ClaudeSDKClient: picks a canned response per query() prompt and emulates
# the agent's tool-use file writes (per-stack + merged reports) so downstream
# phases don't crash on missing files. Only the SDK boundary is mocked.


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
        fork's child trajectory should pick up FIXTURE_MODEL_ID.
        """
        tool_id = f"toolu_{tool_name.lower()}_01"
        return [
            FakeAssistantMessage(
                content=[
                    FakeToolUseBlock(id=tool_id, name=tool_name, input=tool_input),
                ],
                model=FIXTURE_MODEL_ID,
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
                model=FIXTURE_MODEL_ID,
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

        # Exploration specialists run under maybe_fork; the production-bug path
        # where the 'Exploration' rollup row renders 'unknown' / '$0.00'.
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

        # phase_alternative_review (ALTERNATIVE_REVIEW_SCHEMA): ResultMessage
        # must carry structured_output of the right shape or it returns [].
        if "architectural alternatives" in pl or (
            "alternative" in pl and "intent" in pl and "given" in pl
        ):
            return [
                FakeAssistantMessage(
                    content=[FakeTextBlock(text="evaluating alternatives")],
                    model=FIXTURE_MODEL_ID,
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

        # phase_parse_feedback (FEEDBACK_SCHEMA / PER_STACK_RECORD_SCHEMA):
        # ResultMessage must carry structured_output of the right shape.
        # Issue #172: the tiny-diff short-circuit writes ``merged-items.json``
        # directly from these parsed records (no merge agent) for ≤2-file
        # diffs, so the per-stack parse MUST yield a non-empty issue for the
        # PR-comment pipeline to have content on the single-file fixture.
        # The structural parse (FEEDBACK_SCHEMA, no ``severity`` in the prompt)
        # returns empty — the per-stack record drives the assertion below.
        if "extract only actionable issues" in pl or "read the review output file" in pl:
            # phase_parse_feedback injects the ``severity`` hint/field into the
            # prompt ONLY when PER_STACK_RECORD_SCHEMA is passed (per-stack parse);
            # the structural parse uses FEEDBACK_SCHEMA, whose prompt omits
            # ``severity`` entirely. So the word ``severity`` in the prompt text
            # is the fingerprint that distinguishes the two schemas.
            is_per_stack_parse = "severity" in pl
            if is_per_stack_parse:  # PER_STACK_RECORD_SCHEMA (per-stack parse)
                issues = [
                    {
                        "id": 1,
                        "description": "Use a more descriptive function name",
                        "file": "foo.py",
                        "line": 1,
                        "severity": "medium",
                        "confidence": "MEDIUM",
                        "rationale": "The current name is ambiguous.",
                        "evidence": "foo.py:1",
                    }
                ]
            else:  # FEEDBACK_SCHEMA (structural parse)
                issues = []
            return [
                FakeAssistantMessage(
                    content=[FakeTextBlock(text="parsing")],
                    model=FIXTURE_MODEL_ID,
                ),
                FakeResultMessage(
                    structured_output={"issues": issues},
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
                    model=FIXTURE_MODEL_ID,
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

        # phase_cross_stack_merge (MERGED_ITEMS_SCHEMA): ResultMessage carries
        # the structured item list the host renders review-output.md from. One
        # item keeps the rendered report (and PR comment) non-empty.
        if "cross-stack merge agent" in pl:
            return [
                FakeAssistantMessage(
                    content=[FakeTextBlock(text="merging")],
                    model=FIXTURE_MODEL_ID,
                ),
                FakeResultMessage(
                    structured_output={
                        "items": [
                            {
                                "id": 1,
                                "lens": "per-stack",
                                "file": "foo.py",
                                "line": 1,
                                "severity": "medium",
                                "description": "Use a more descriptive function name",
                                "confidence": "MEDIUM",
                                "rationale": "The current name is ambiguous.",
                                "evidence": "foo.py:1",
                            }
                        ]
                    },
                    total_cost_usd=0.20,
                    usage={
                        "input_tokens": 4000,
                        "output_tokens": 600,
                        "cache_read_input_tokens": 1500,
                    },
                ),
            ]

        # phase_per_stack_reviews: free-form text; reports written by
        # _maybe_write_artifact via prompt scan.
        return [
            FakeAssistantMessage(
                content=[FakeTextBlock(text="ok, wrote the review")],
                model=FIXTURE_MODEL_ID,
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


# Repo + monkeypatch fixtures.


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
    # Seed five Python files on main so the tree-sitter index can resolve them.
    for name in ("foo.py", "bar.py", "baz.py", "qux.py", "quux.py"):
        (repo / name).write_text(f"def {name[:-3]}():\n    return 1\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "init")
    _git(repo, "checkout", "-b", "feature")
    # Mutate four so count_changed_files >= 4 -> tier="parallel".
    for name in ("foo.py", "bar.py", "baz.py", "qux.py"):
        (repo / name).write_text(f"def {name[:-3]}():\n    return 2\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "tweak four files")
    # Assert diff count up-front so a threshold-breaking refactor trips here.
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


# gh / PR plumbing patches: find_open_pr returns a fake PRInfo and _submit_review
# captures the payload (the only allowed non-SDK mock per the brief — it stands
# in for the gh-CLI subprocess that would post to GitHub).


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


# Misc: silence Rich UI noise + answer interactive prompts.


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
    """Phase prompts: y for intent confirmation + PR post; n for fix gate.

    In pytest stdin is not a TTY, so ``runner._resolve_interactive`` returns
    False and ``set_non_interactive(True)`` is called. Every gate that uses
    ``resolve_or_prompt(safe_default=False)`` then auto-declines without
    prompting -- including the PR-post gate. We must force interactive mode
    the same way ``_force_interactive`` does (in test_deep_orchestrator.py).

    Both the orchestrator fix gate ("Apply fixes now?") and the PR-post gate
    ("Post these as a PR review?") route through resolve_or_prompt in agent.py,
    which calls prompt_user from its own (agent) namespace. We dispatch on the
    question text so the fix gate gets "n" (skip fixes) and PR-post gets "y".
    """
    # Force interactive mode so resolve_gate defers to prompt_user.
    monkeypatch.setattr("daydream.runner._stdin_isatty", lambda: True)
    monkeypatch.delenv("CI", raising=False)

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

    def _agent_prompt(console, message: str, default: str = "") -> str:  # noqa: ARG001
        # Decline the fix gate (proceed to PR post without fixes); approve PR post.
        if "apply fixes" in message.lower() or "apply fix" in message.lower():
            return "n"
        if "post these" in message.lower() or "pr review" in message.lower():
            return "y"
        return "y"  # all other gates (intent confirmation, etc.)

    monkeypatch.setattr(
        "daydream.agent.prompt_user", _agent_prompt, raising=False,
    )


# Markdown extraction helpers.


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
        if line.startswith("|") and not line.startswith("| Phase") and not line.startswith("|---"):
            rows.append(line)
    return rows


def _row_cells(row: str) -> list[str]:
    return [c.strip() for c in row.strip("|").split("|")]


# The test.


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
        cleanup=False,
        archive=False,
    )
    # Single-file diff -> select_tier "skip": the orchestrator's natural
    # pre-scan path runs unpopulated. (ExplorationContext kept for the linter.)
    _ = ExplorationContext

    exit_code = await run(config)

    assert exit_code == 0, f"run() returned {exit_code}"
    assert captured_post.payloads, (
        "build_payload was never called: deep flow did not reach _post"
    )
    payload = captured_post.payloads[-1]
    body = payload["body"]
    assert isinstance(body, str)

    # --- Mode line: removed everywhere ----------------------------------
    assert "**Mode:**" not in body, (
        f"BUG: Mode line should be gone from PR-comment body.\n\nfull body:\n{body}"
    )

    # --- Model line: real SDK id, not 'unknown' / backend alias ---------
    model_line = _line_starting(body, "- **Model:**")
    assert FIXTURE_MODEL_ID in model_line, (
        f"BUG: rollup Model line is missing the real SDK model id "
        f"({FIXTURE_MODEL_ID!r}).\n  got: {model_line!r}\n\n"
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
        # Layout: | Phase | Model | Tools | Input (cached) | Output | Cost |
        assert len(cells) >= 6, f"unexpected row layout: {row!r}"
        phase_name, model_cell, _tools, input_cell, _out, cost_cell = cells[:6]
        assert input_cell != "0", (
            f"BUG: row {phase_name!r} has Input='0' "
            f"(per-step token metrics never propagated).\n  row: {row!r}"
        )
        assert model_cell != "unknown", (
            f"BUG: row {phase_name!r} has Model='unknown' "
            f"(SDK model id never propagated to the per-phase rollup).\n"
            f"  row: {row!r}"
        )
        assert FIXTURE_MODEL_ID in model_cell, (
            f"BUG: row {phase_name!r} Model cell missing real SDK id "
            f"{FIXTURE_MODEL_ID!r}.\n  row: {row!r}"
        )
        assert cost_cell != "$0.00", (
            f"BUG: row {phase_name!r} has Cost=$0.00 "
            f"(per-step cost never landed).\n  row: {row!r}"
        )


# Exploration-row reproduction test: forces the parallel tier (which the
# single-file fixture skips) so the broken Exploration rollup row is exercised.


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

    # Print rendered markdown so a future failure trace shows what we observed.
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
    # Layout: | Phase | Model | Tools | Input (cached) | Output | Cost |
    assert len(cells) >= 6, f"unexpected row layout: {exploration_row!r}"
    (
        _phase_name,
        model_cell,
        tools_cell,
        _input_cell,
        _output_cell,
        cost_cell,
    ) = cells[:6]

    # --- The three production-bug symptoms, asserted one at a time. -------

    # 1. Model column — production shows 'unknown'. The fork's child
    #    recorder should have upgraded this to the SDK id.
    assert model_cell.lower() != "unknown", (
        f"BUG: Exploration row Model='unknown' (matches production bug).\n"
        f"  row: {exploration_row!r}\n\nfull body:\n{body}"
    )
    assert FIXTURE_MODEL_ID in model_cell, (
        f"BUG: Exploration row Model cell is missing real SDK id "
        f"{FIXTURE_MODEL_ID!r}.\n  got Model cell: {model_cell!r}\n"
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
