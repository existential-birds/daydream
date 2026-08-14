"""Integration tests for the full review-fix-test flow."""

import json
import re
import time
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from daydream.backends import (
    CostEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from daydream.runner import run
from daydream.trajectory import DaydreamPhase
from daydream.ui import NEON_THEME
from tests.harness.backend import ScriptedBackend
from tests.harness.git_helpers import commit as _commit
from tests.harness.git_helpers import git as _git
from tests.harness.git_helpers import init_repo as _init_repo
from tests.harness.phase_backend import PhaseDispatchBackend

# ANSI escape code pattern for stripping terminal colors
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    """Strip ANSI escape codes from text for assertion comparisons."""
    return _ANSI_ESCAPE.sub("", text)


# Mock Backends


# The prompt-dispatching MockBackend was consolidated onto the shared
# PhaseDispatchBackend (tests/harness/phase_backend.py). emit_cost=True preserves
# the old CostEvent-on-non-structured-turns shape; parse_results=[[issue]] yields
# one issue on the first parse, matching the old single-issue dispatch.
_FULL_FLOW_ISSUE = {"id": 1, "description": "Add type hints to function", "file": "main.py", "line": 1}


async def render_agent(
    monkeypatch: pytest.MonkeyPatch,
    events: list[Any],
    *,
    quiet: bool,
    prompt: str = "Test prompt",
    color_system: str | None = None,
) -> str:
    """Drive ``run_agent`` over *events* and return the raw terminal output.

    The tool-panel / quiet-mode tests all repeated the same harness: a scripted
    event-yielding backend, a ``StringIO``-backed ``Console`` bound over
    ``daydream.agent.console``, and ``set_quiet_mode``. The console is pinned
    (``force_terminal=True``, ``width=120``, ``NEON_THEME``) so wrapping and
    styling are identical regardless of the host terminal.

    Returns the output with ANSI codes INTACT -- the border/styling assertions
    read them. Callers comparing plain text pass the result to ``strip_ansi``.
    """
    from daydream.agent import run_agent, set_quiet_mode

    output = StringIO()
    extra: dict[str, Any] = {} if color_system is None else {"color_system": color_system}
    monkeypatch.setattr(
        "daydream.agent.console",
        Console(file=output, force_terminal=True, width=120, theme=NEON_THEME, **extra),
    )
    set_quiet_mode(quiet)

    await run_agent(
        ScriptedBackend(events=events, model="mock-model"),
        Path("/tmp"),
        prompt,
        phase=DaydreamPhase.REVIEW,
    )
    return output.getvalue()


@pytest.mark.asyncio
async def test_five_thinking_panels_render_in_under_two_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Five consecutive thinking panels render immediately, in order.

    Regression for issue #377: the old LiveThinkingPanel.show slept 0.5s per
    thought inside the event loop, so five thoughts blocked the loop for
    >= 2.5s of wall time. render_agent no longer stubs out time.sleep, so this
    exercises the real production path. The < 2.0s bound is discriminating:
    the old code needs >= 2.5s; the fixed code renders all five immediately.
    """
    thoughts = [f"thought {i} of the stream" for i in range(5)]
    events: list[Any] = [
        *[ThinkingEvent(text=t) for t in thoughts],
        ResultEvent(structured_output=None, continuation=None),
    ]

    # Warm up run_agent/ScriptedBackend setup and Console construction so that
    # setup cost is not counted against the wall-clock bound: under loaded
    # parallel CI (pytest -n auto) that setup can push elapsed past 2.0s and
    # flake the regression (issue #336).
    await render_agent(monkeypatch, events, quiet=False)

    start = time.perf_counter()
    plain_text = strip_ansi(await render_agent(monkeypatch, events, quiet=False))
    elapsed = time.perf_counter() - start

    assert elapsed < 2.0, f"5 thoughts took {elapsed:.2f}s (old code >= 2.5s)"

    assert plain_text.count("Thinking") == 5, "each thought must render its Thinking title once"

    cursor = -1
    for t in thoughts:
        idx = plain_text.find(t)
        assert idx != -1, f"thought {t!r} did not render"
        assert idx > cursor, f"thought {t!r} rendered out of order"
        cursor = idx


# Fixtures


@pytest.fixture
def mock_backend(install_backend):
    """Patch create_backend to return the shared phase-dispatch fake."""
    return install_backend(
        PhaseDispatchBackend(parse_results=[[_FULL_FLOW_ISSUE]], emit_cost=True)
    )


@pytest.fixture
def mock_ui(monkeypatch):
    """Patch UI functions that require user input."""
    monkeypatch.setattr("daydream.phases.prompt_user", lambda *args, **kwargs: "n")
    monkeypatch.setattr("daydream.runner.prompt_user", lambda *args, **kwargs: "n")


@pytest.fixture
def target_project(tmp_path: Path) -> Path:
    """Create a minimal project structure for testing.

    Stage 4.2: ``open_workspace`` requires a real worktree. Initialise a
    fresh repo with one initial commit on ``main`` and a ``feature`` branch
    so the WrongBranchError guard does not fire for default-branch runs.
    """
    project = tmp_path / "test_project"
    project.mkdir()

    (project / "main.py").write_text("def hello():\n    return 'world'\n")

    # Pre-create the review output the harness mock does not write.
    review_content = """# Code Review

## Issues Found

1. **Missing type hints** in `main.py:1`
   - Add type hints to the `hello` function

## Summary

Found 1 issue to address.
"""
    (project / ".review-output.md").write_text(review_content)

    _init_repo(project)
    _git(project, "add", "main.py")
    _commit(project, "init")
    # Move off main so the WrongBranchError guard doesn't fire on default runs.
    _git(project, "checkout", "-b", "feature")

    return project


@pytest.mark.asyncio
async def test_full_fix_flow(mock_backend, mock_ui, target_project: Path, make_config):
    """Test the complete review -> parse -> fix -> test flow."""
    config = make_config(target_project, skill="python", quiet=True, shallow=True)

    exit_code = await run(config)

    assert exit_code == 0
    assert (target_project / ".review-output.md").exists()


class _WorktreeMutatingBackend(PhaseDispatchBackend):
    """Phase-dispatch fake whose fix and commit turns really touch the worktree.

    The backend is the only mocked seam, so the edit and the commit the real
    prompts ask for have to happen here -- exactly what the agent would do with
    its tools -- for the run to leave observable Git state behind.
    """

    async def execute(
        self,
        cwd,
        prompt,
        output_schema=None,
        continuation=None,
        agents=None,
        max_turns=None,
        read_only=False,
    ):
        if prompt.startswith("Fix this issue") or prompt.startswith("Fix these"):
            (cwd / "main.py").write_text("def hello() -> str:\n    return 'world'\n")
        elif prompt.startswith("Stage all changes and commit"):
            run_id = prompt.split("Daydream-Run: ", 1)[1].splitlines()[0]
            version = prompt.split("Daydream-Version: ", 1)[1].splitlines()[0]
            _git(cwd, "add", "main.py")
            _commit(
                cwd,
                f"fix: add type hints\n\nDaydream-Run: {run_id}\nDaydream-Version: {version}",
            )

        async for event in super().execute(
            cwd,
            prompt,
            output_schema=output_schema,
            continuation=continuation,
            agents=agents,
            max_turns=max_turns,
            read_only=read_only,
        ):
            yield event


@pytest.mark.asyncio
async def test_shallow_commits_when_operator_ignores_red_suite(
    monkeypatch,
    feature_branch_repo: Path,
    install_backend,
    make_config,
    silence_console,
):
    """Heal-menu choice "3" keeps the shallow deep run going all the way to a real commit.

    Drives the deep shallow flow through the REAL ``phase_test_and_heal`` and
    ``phase_commit_push`` against a permanently red suite, with the backend as
    the only mocked seam. Choice "3" (ignore and continue) reports ``passed``
    False but ``proceed`` True, and the deep fix cycle's commit step reads the
    operator's "y" at the commit gate -- so the run exits 0 and the fix lands
    in the real worktree instead of being abandoned with the failure.
    """
    # stdin answers, in order: intent confirmation, the apply-fixes gate, the
    # heal menu ("3" = ignore and continue), and the commit gate.
    monkeypatch.setattr("sys.stdin", StringIO("y\ny\n3\ny\n"))
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.delenv("CI", raising=False)
    silence_console("daydream.runner")
    silence_console("daydream.deep.orchestrator")
    silence_console("daydream.phases")
    install_backend(
        _WorktreeMutatingBackend(parse_results=[[_FULL_FLOW_ISSUE]], tests_pass=False)
    )

    head_before = _git(feature_branch_repo, "rev-parse", "HEAD")

    config = make_config(
        feature_branch_repo,
        skill="python",
        quiet=True,
        shallow=True,
        non_interactive=False,
        output_mode="loop",
    )
    exit_code = await run(config)

    assert exit_code == 0, "choice '3' must continue the run, not abort it"
    assert _git(feature_branch_repo, "rev-parse", "HEAD") != head_before, (
        "the ignored-red-suite run never committed"
    )
    assert "-> str" in _git(feature_branch_repo, "show", "HEAD:main.py"), (
        "the fix was reverted instead of committed"
    )
    assert "Daydream-Run:" in _git(feature_branch_repo, "log", "-1", "--format=%B")


@pytest.mark.asyncio
async def test_glob_tool_panel_displays_file_count_and_list(monkeypatch):
    """Test the full tool panel lifecycle in normal mode shows file count and list.

    This test exercises the actual run_agent() flow by providing a scripted backend
    that yields events. Normal mode (quiet=False) shows both header and output section.

    Also tests that:
    - AgentTextRenderer displays streamed text with spinner cursor effect
    - LiveThinkingPanel displays thinking blocks with stable title
    """
    tool_use_id = "test-glob-lifecycle-123"
    glob_result = """/project/src/main.py
/project/src/utils/helper.py
/project/tests/test_main.py"""

    events = [
        ThinkingEvent(text="Analyzing the project structure..."),
        TextEvent(text="I'll search for Python files in the project."),
        ToolStartEvent(id=tool_use_id, name="Glob", input={"pattern": "**/*.py", "path": "/project"}),
        ToolResultEvent(id=tool_use_id, output=glob_result, is_error=False),
        CostEvent(cost_usd=0.001, input_tokens=None, output_tokens=None),
        ResultEvent(structured_output=None, continuation=None),
    ]

    plain_text = strip_ansi(
        await render_agent(monkeypatch, events, quiet=False, prompt="Test prompt for Glob tool")
    )

    assert "Thinking" in plain_text
    assert "Analyzing the project structure" in plain_text

    assert "I'll search for Python files" in plain_text

    assert "Glob" in plain_text
    assert "**/*.py" in plain_text

    # Normal mode shows the output section with the file count.
    assert "Found 3 files" in plain_text

    assert "main.py" in plain_text
    assert "helper.py" in plain_text
    assert "test_main.py" in plain_text


@pytest.mark.asyncio
async def test_glob_tool_panel_singular_file_count(monkeypatch):
    """Test that LiveToolPanel shows singular 'file' for 1 result in normal mode."""
    tool_use_id = "test-glob-singular-456"
    glob_result = "/project/main.py"

    events = [
        ToolStartEvent(id=tool_use_id, name="Glob", input={"pattern": "*.py"}),
        ToolResultEvent(id=tool_use_id, output=glob_result, is_error=False),
        CostEvent(cost_usd=0.001, input_tokens=None, output_tokens=None),
        ResultEvent(structured_output=None, continuation=None),
    ]

    # Normal mode shows the output section.
    output_text = await render_agent(monkeypatch, events, quiet=False)

    # Singular "file", not "files".
    assert "Found 1 file" in output_text
    assert "Found 1 files" not in output_text

    assert "main.py" in output_text


@pytest.mark.asyncio
async def test_glob_tool_panel_truncates_long_results(monkeypatch):
    """Test that LiveToolPanel truncates long Glob results in normal mode."""
    tool_use_id = "test-glob-truncate-789"
    # 25 files exceeds max_lines=20 from _build_result_content_internal.
    mock_files = [f"/project/src/module{i}.py" for i in range(25)]
    glob_result = "\n".join(mock_files)

    events = [
        ToolStartEvent(id=tool_use_id, name="Glob", input={"pattern": "**/*.py"}),
        ToolResultEvent(id=tool_use_id, output=glob_result, is_error=False),
        CostEvent(cost_usd=0.001, input_tokens=None, output_tokens=None),
        ResultEvent(structured_output=None, continuation=None),
    ]

    # Normal mode shows the output section.
    output_text = await render_agent(monkeypatch, events, quiet=False)

    assert "Found 25 files" in output_text
    assert "and 5 more" in output_text  # 25 total - 20 displayed


@pytest.mark.asyncio
async def test_quiet_mode_shows_header_only(monkeypatch):
    """Test that quiet mode shows header only (no output section)."""
    tool_use_id = "test-output-panel-001"
    read_result = "def hello():\n    return 'world'"

    events = [
        ToolStartEvent(id=tool_use_id, name="Read", input={"file_path": "/project/main.py"}),
        ToolResultEvent(id=tool_use_id, output=read_result, is_error=False),
        CostEvent(cost_usd=0.001, input_tokens=None, output_tokens=None),
        ResultEvent(structured_output=None, continuation=None),
    ]

    output_text = await render_agent(monkeypatch, events, quiet=True)
    plain_text = strip_ansi(output_text)

    assert "Read" in plain_text
    assert "/project/main.py" in plain_text

    # Quiet mode: header only — no output section, no content.
    assert "Output" not in plain_text
    assert "hello" not in plain_text
    assert "world" not in plain_text

    assert "╭" in output_text or "│" in output_text


@pytest.mark.asyncio
async def test_quiet_mode_empty_result_shows_header_only(monkeypatch):
    """Test that quiet mode shows header only for empty results (no output section)."""
    tool_use_id = "test-empty-result-002"

    events = [
        ToolStartEvent(id=tool_use_id, name="Bash", input={"command": "true"}),
        ToolResultEvent(id=tool_use_id, output="", is_error=False),
        CostEvent(cost_usd=0.001, input_tokens=None, output_tokens=None),
        ResultEvent(structured_output=None, continuation=None),
    ]

    output_text = await render_agent(monkeypatch, events, quiet=True)

    assert "Bash" in output_text
    assert "Output" not in output_text  # quiet mode: header only
    assert "╭" in output_text or "│" in output_text


@pytest.mark.asyncio
async def test_quiet_mode_error_shows_header_with_red_border(monkeypatch):
    """Test that quiet mode shows header only with red border for errors."""
    tool_use_id = "test-error-result-003"

    events = [
        ToolStartEvent(id=tool_use_id, name="Bash", input={"command": "false"}),
        ToolResultEvent(id=tool_use_id, output="Command failed with exit code 1", is_error=True),
        CostEvent(cost_usd=0.001, input_tokens=None, output_tokens=None),
        ResultEvent(structured_output=None, continuation=None),
    ]

    # Force truecolor for consistent RGB color codes across environments.
    output_text = await render_agent(
        monkeypatch, events, quiet=True, color_system="truecolor"
    )

    assert "Bash" in output_text
    assert "Command failed" not in output_text  # quiet mode: header only, no error body
    assert "╭" in output_text or "│" in output_text
    assert "\x1b[" in output_text  # ANSI styling present (red border)


@pytest.mark.asyncio
async def test_skill_tool_panel_collapses_output(monkeypatch):
    """Test that Skill tool calls don't show an Output panel.

    The skill name already appears in the tool call header, so the
    "Launching skill: X" output is redundant and should be suppressed.
    """
    tool_use_id = "test-skill-collapse-001"

    events = [
        ToolStartEvent(id=tool_use_id, name="Skill", input={"skill": "review-python"}),
        ToolResultEvent(id=tool_use_id, output="Launching skill: review-python", is_error=False),
        CostEvent(cost_usd=0.001, input_tokens=None, output_tokens=None),
        ResultEvent(structured_output=None, continuation=None),
    ]

    plain_text = strip_ansi(await render_agent(monkeypatch, events, quiet=True))

    assert "Skill" in plain_text
    assert "review-python" in plain_text

    # No Output panel: the header already shows the skill name, so the redundant
    # "Launching skill:" result is suppressed.
    assert "Output" not in plain_text
    assert "Launching skill:" not in plain_text


@pytest.mark.asyncio
async def test_concurrent_tool_panels_display_results(monkeypatch):
    """Test that concurrent tool panels (e.g. Codex parallel commands) all show results.

    When multiple ToolStartEvents arrive before any ToolResultEvents (as happens
    with the Codex backend's parallel command execution), all panels should
    eventually display their results without display corruption.
    """
    # 3 concurrent commands: all started before any complete.
    events = [
        ToolStartEvent(id="cmd-1", name="shell", input={"command": "git diff -- file1.py"}),
        ToolStartEvent(id="cmd-2", name="shell", input={"command": "git diff -- file2.py"}),
        ToolStartEvent(id="cmd-3", name="shell", input={"command": "git diff -- file3.py"}),
        ToolResultEvent(id="cmd-1", output="+added line in file1", is_error=False),
        ToolResultEvent(id="cmd-2", output="+added line in file2", is_error=False),
        ToolResultEvent(id="cmd-3", output="+added line in file3", is_error=False),
        CostEvent(cost_usd=0.001, input_tokens=None, output_tokens=None),
        ResultEvent(structured_output=None, continuation=None),
    ]

    plain_text = strip_ansi(await render_agent(monkeypatch, events, quiet=False))

    # All three results and commands appear.
    assert "+added line in file1" in plain_text
    assert "+added line in file2" in plain_text
    assert "+added line in file3" in plain_text

    assert "git diff -- file1.py" in plain_text
    assert "git diff -- file2.py" in plain_text
    assert "git diff -- file3.py" in plain_text


def _two_commit_repo(repo: Path, filename: str, before: str, after: str, branch: str) -> Path:
    """A repo with *filename* committed on ``main`` and modified on *branch*."""
    _init_repo(repo)
    (repo / filename).write_text(before)
    _git(repo, "add", ".")
    _commit(repo, "init")
    _git(repo, "checkout", "-b", branch)
    (repo / filename).write_text(after)
    _git(repo, "add", ".")
    _commit(repo, "change")
    return repo


@pytest.mark.asyncio
async def test_run_comment_full_flow(tmp_path, monkeypatch, make_config):
    """Integration test: full --comment flow through the deep pipeline."""
    from tests.test_deep_orchestrator import _install_stub_backend, _silence

    _two_commit_repo(tmp_path, "app.py", "print('hello')", "print('world')", "feat/test")

    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, tmp_path)

    # Capture the canonical items that reach the PR-posting step — the comment path.
    posted: list[dict[str, Any]] = []
    posted_posts: list[bool] = []

    async def fake_post(target_dir, merged_items_path, *, console, post, approve_on_clean=False):
        posted.extend(json.loads(merged_items_path.read_text())["items"])
        posted_posts.append(post)

    monkeypatch.setattr("daydream.pr_review.post_review_to_pr_from_report", fake_post)

    config = make_config(tmp_path, output_mode="comment")

    exit_code = await run(config)

    assert exit_code == 0
    # Comment mode auto-posts (post=True) with the canonical merged items.
    assert posted_posts == [True]
    assert posted, "the --comment post step never received merged items"
    # The review spine ran: a per-stack finding reached the post step.
    assert any(item.get("file") for item in posted), posted
    # The diff was materialised for the review prompts.
    assert (tmp_path / ".daydream" / "diff.patch").exists()


@pytest.mark.asyncio
async def test_run_comment_does_not_prompt_for_skill(
    tmp_path, monkeypatch, install_backend, silence_console, make_config
):
    """--comment mode should never prompt for skill selection."""
    _two_commit_repo(tmp_path, "f.txt", "a", "b", "feat")

    install_backend(ScriptedBackend(
        events=[
            TextEvent(text="Intent: changes f.txt."),
            ResultEvent(structured_output={"issues": []}, continuation=None),
        ],
        model="mock-model",
    ))
    silence_console("daydream.phases")
    silence_console("daydream.runner")

    monkeypatch.setattr("daydream.phases.prompt_user", lambda *a, **kw: "y")  # confirm intent

    # Trap: skill selection must never prompt in --comment mode.
    def runner_prompt_trap(*args, **kwargs):
        raise AssertionError("Should not prompt for skill selection in --comment mode")
    monkeypatch.setattr("daydream.runner.prompt_user", runner_prompt_trap)

    config = make_config(tmp_path, output_mode="comment")
    exit_code = await run(config)
    assert exit_code == 0


@pytest.mark.asyncio
async def test_run_comment_missing_pr_exits_nonzero(
    tmp_path, monkeypatch, make_config
):
    """Comment mode chose posting as its deliverable: no open PR -> exit 1.

    Drives ``runner.run`` for real (real temp worktree, stub backend only);
    only ``pr_review.find_open_pr`` is mocked to report no PR, so the missing-PR
    warning path runs production code end to end.
    """
    from tests.test_deep_orchestrator import _install_stub_backend, _silence

    _two_commit_repo(tmp_path, "app.py", "print('hello')", "print('world')", "feat/test")

    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, tmp_path)
    monkeypatch.setattr("daydream.pr_review.find_open_pr", lambda _td: None)

    config = make_config(tmp_path, output_mode="comment")

    exit_code = await run(config)

    assert exit_code == 1, "comment mode must fail when no open PR exists"


@pytest.mark.asyncio
async def test_run_comment_submission_failure_exits_nonzero(
    tmp_path, monkeypatch, make_config
):
    """Comment mode: a failed GitHub review post -> exit 1.

    Only ``_submit_review`` is mocked to fail; everything else (the review
    pipeline, ``_post``, classification, payload build) runs production code.
    """
    from daydream.pr_review import PRInfo
    from tests.test_deep_orchestrator import _install_stub_backend, _silence

    _two_commit_repo(tmp_path, "app.py", "print('hello')", "print('world')", "feat/test")

    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, tmp_path)

    fake_pr = PRInfo(
        number=7,
        head_sha="0" * 40,
        base_sha="1" * 40,
        base_ref="main",
        owner="acme",
        repo="widgets",
        url="https://example/pr/7",
    )
    monkeypatch.setattr("daydream.pr_review.find_open_pr", lambda _td: fake_pr)
    monkeypatch.setattr(
        "daydream.pr_review._submit_review",
        lambda _td, _pr, _payload: (None, "gh api failed: HTTP 500"),
    )

    config = make_config(tmp_path, output_mode="comment")

    exit_code = await run(config)

    assert exit_code == 1, "comment mode must fail when the review post fails"


@pytest.mark.asyncio
async def test_run_loop_submission_failure_warns_and_continues(
    tmp_path, monkeypatch, make_config
):
    """Default deep loop: a failed review post warns-and-continues (exit 0).

    Posting is optional in loop mode, so a failed GitHub post must not abort the
    run. The post gate is approved (interactive prompt path), the fix gate
    declines, and the run still exits 0 with the report written.
    """
    from daydream.pr_review import PRInfo
    from tests.harness.stub_backend import force_interactive, install_stub_backend, silence

    _two_commit_repo(tmp_path, "app.py", "print('hello')", "print('world')", "feat/test")

    silence(monkeypatch, prompts=False)
    install_stub_backend(monkeypatch, tmp_path)
    force_interactive(monkeypatch)

    fake_pr = PRInfo(
        number=7,
        head_sha="0" * 40,
        base_sha="1" * 40,
        base_ref="main",
        owner="acme",
        repo="widgets",
        url="https://example/pr/7",
    )
    monkeypatch.setattr("daydream.pr_review.find_open_pr", lambda _td: fake_pr)
    monkeypatch.setattr(
        "daydream.pr_review._submit_review",
        lambda _td, _pr, _payload: (None, "gh api failed: HTTP 500"),
    )

    # Approve the PR-post gate but decline the apply-fixes gate so the run ends
    # after the report is written (no fix cycle / commit).
    def _gate_prompt(console, message: str, default: str = "") -> str:
        if "apply fix" in message.lower():
            return "n"
        return "y"

    monkeypatch.setattr("daydream.agent.prompt_user", _gate_prompt)
    monkeypatch.setattr("daydream.phases.prompt_user", lambda *a, **kw: "y")

    config = make_config(tmp_path, output_mode="loop")

    exit_code = await run(config)

    assert exit_code == 0, "deep loop must warn-and-continue on a failed PR post"
    # The report the review produced is still on disk (the fix gate declined).
    assert (tmp_path / ".review-output.md").exists()


# Phase 02-04: Pre-scan exploration wiring


@pytest.mark.asyncio
async def test_run_populates_exploration_context(
    monkeypatch, multi_stack_target, make_config
):
    """run() populates config.exploration_context before the review fan-out fires.

    Drives the deep shallow flow through ``runner.run`` with exploration left
    enabled (4 changed files -> "parallel" tier so the real ``pre_scan`` runs)
    and asserts the wired consequence: ``config.exploration_context`` is set and
    the per-stack review receives the on-disk ``exploration_dir``.
    """
    from daydream.exploration import ExplorationContext
    from tests.test_deep_orchestrator import _install_stub_backend, _silence

    (multi_stack_target / "extra.py").write_text("VALUE = 2\n")
    _git(multi_stack_target, "add", ".")
    _commit(multi_stack_target, "add extra")

    _silence(monkeypatch)
    _install_stub_backend(monkeypatch, multi_stack_target, enable_exploration=True)

    captured: dict[str, Any] = {}

    async def fake_per_stack_reviews(backend, work, stacks, **kwargs):
        captured["exploration_dir"] = kwargs.get("exploration_dir")
        return {}, {}

    monkeypatch.setattr(
        "daydream.deep.orchestrator.phase_per_stack_reviews", fake_per_stack_reviews
    )

    config = make_config(multi_stack_target, shallow=True)
    exit_code = await run(config)

    assert exit_code == 0
    assert isinstance(config.exploration_context, ExplorationContext)
    assert "exploration_dir" in captured
    assert captured["exploration_dir"] is not None


@pytest.mark.asyncio
async def test_codex_backend_raises_on_agents(tmp_path: Path):
    """CodexBackend.execute() refuses agents= with NotImplementedError."""
    from daydream.backends.codex import CodexBackend

    backend = CodexBackend(model="fixture-model")
    with pytest.raises(NotImplementedError, match="Codex backend does not support exploration"):
        async for _ in backend.execute(tmp_path, "prompt", agents={"x": object()}):
            pass


async def test_exploration_enriched_output_both_flows(tmp_path, make_work):
    """Both normal and TTT flows surface confidence + rationale on parsed issues.

    Exercises `phase_parse_feedback` (normal flow) and `phase_alternative_review`
    (TTT flow) directly: both return parsed issue lists, and both must carry the
    schema-enforced confidence/rationale fields per QUAL-02.
    """
    from daydream.phases import phase_alternative_review, phase_parse_feedback

    enriched_normal_issue = {
        "id": 1,
        "description": "x",
        "file": "a.py",
        "line": 1,
        "confidence": "HIGH",
        "rationale": "verified by Convention snake_case_modules",
        "evidence": "a.py:1",
    }
    enriched_trust_issue = {
        "id": 1,
        "title": "t",
        "description": "x",
        "recommendation": "y",
        "severity": "high",
        "files": ["a.py"],
        "confidence": "HIGH",
        "rationale": "verified by Convention snake_case_modules",
    }

    def _issue_backend(payload: dict) -> ScriptedBackend:
        return ScriptedBackend(
            events=[
                TextEvent(text="ok"),
                ResultEvent(structured_output=payload, continuation=None),
            ],
            model="test-model",
        )

    work = make_work(tmp_path)
    # Normal flow: phase_parse_feedback returns list of validated issues
    (tmp_path / ".review-output.md").write_text("# Review\n")
    normal_backend = _issue_backend({"issues": [enriched_normal_issue]})
    normal_issues = await phase_parse_feedback(normal_backend, work)

    # TTT flow: phase_alternative_review returns list of issues
    diff_path = tmp_path / "diff.txt"
    diff_path.write_text("diff")
    trust_backend = _issue_backend({"issues": [enriched_trust_issue]})
    trust_issues = await phase_alternative_review(
        trust_backend,
        work,
        diff_path,
        "intent summary",
        exploration_dir=tmp_path,
    )

    for issues in (normal_issues, trust_issues):
        assert issues
        assert "confidence" in issues[0]
        assert "rationale" in issues[0]
