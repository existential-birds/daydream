"""Tests for continuous loop mode."""

import re
import subprocess
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from daydream.runner import RunConfig, run
from daydream.ui import NEON_THEME, SummaryData, print_iteration_divider, print_summary
from tests.harness.phase_backend import PhaseDispatchBackend

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def test_runconfig_loop_defaults():
    config = RunConfig()
    assert config.loop is False
    assert config.max_iterations == 5


def strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE.sub("", text)


def test_print_iteration_divider():
    output = StringIO()
    test_console = Console(file=output, force_terminal=True, width=80, theme=NEON_THEME)
    print_iteration_divider(test_console, 2, 5)
    plain = strip_ansi(output.getvalue())
    assert "Iteration 2 of 5" in plain
    assert "━" in plain


def test_summary_data_loop_fields_default():
    data = SummaryData(
        skill="python", target="/tmp", feedback_count=3,
        fixes_applied=3, test_retries=0, tests_passed=True,
    )
    assert data.loop_mode is False
    assert data.iterations_used == 1


def test_summary_loop_mode_shows_iterations():
    output = StringIO()
    test_console = Console(file=output, force_terminal=True, width=80, theme=NEON_THEME)
    data = SummaryData(
        skill="python", target="/tmp", feedback_count=8,
        fixes_applied=8, test_retries=1, tests_passed=True,
        loop_mode=True, iterations_used=3,
    )
    print_summary(test_console, data)
    plain = strip_ansi(output.getvalue())
    assert "Iterations" in plain
    assert "3" in plain


def loop_mock_backend(review_results: list[list[dict[str, Any]]], tests_pass: bool = True) -> PhaseDispatchBackend:
    """Build a shared ``PhaseDispatchBackend`` configured for loop-mode tests.

    Migrated onto the consolidated dispatch fake. ``review_results`` maps to the
    fake's per-iteration ``parse_results`` queue; the prompt-heuristic dispatch,
    ``_parse_call``/``commit_calls``/``review_prompts`` tracking, and observable
    outcomes are byte-for-byte the same as the former local class.
    """
    return PhaseDispatchBackend(parse_results=review_results, tests_pass=tests_pass)


@pytest.fixture
def loop_target(feature_branch_repo: Path) -> Path:
    """Loop-mode target: a clean repo on a feature branch with a committed diff.

    Consumes the shared ``feature_branch_repo`` fixture (tests/conftest.py)
    instead of re-rolling git setup inline.
    """
    return feature_branch_repo


def test_feature_branch_repo_has_committed_diff(feature_branch_repo):
    """The shared fixture yields a clean repo on a non-main branch with a committed diff."""
    out = subprocess.run(  # noqa: S603 - arguments are not user-controlled
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],  # noqa: S607 - git is a trusted command
        cwd=feature_branch_repo, capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() != "main"
    assert (feature_branch_repo / "main.py").exists()
    diff_out = subprocess.run(  # noqa: S603 - arguments are not user-controlled
        ["git", "diff", "--name-only", "main...HEAD"],  # noqa: S607 - git is a trusted command
        cwd=feature_branch_repo, capture_output=True, text=True, check=True,
    )
    assert "main.py" in diff_out.stdout.splitlines()


@pytest.fixture
def mock_ui_loop(monkeypatch):
    monkeypatch.setattr("daydream.phases.prompt_user", lambda *a, **kw: "n")
    monkeypatch.setattr("daydream.runner.prompt_user", lambda *a, **kw: "n")


@pytest.mark.asyncio
async def test_loop_exits_on_zero_issues(loop_target, mock_ui_loop, monkeypatch):
    """Issues on iteration 1, zero on iteration 2 -> exits 0."""
    issue = {"id": 1, "description": "Add type hints", "file": "main.py", "line": 1}
    backend = loop_mock_backend(review_results=[[issue], []])

    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: backend)

    config = RunConfig(
        target=str(loop_target), skill="python", quiet=True,
        cleanup=False, loop=True, max_iterations=5,
        shallow=True,
    )
    exit_code = await run(config)

    assert exit_code == 0
    assert backend._parse_call == 2


@pytest.mark.asyncio
async def test_loop_respects_max_iterations(loop_target, mock_ui_loop, monkeypatch):
    """Always returns issues -> stops at max_iterations, exits 1."""
    issue = {"id": 1, "description": "Persistent issue", "file": "main.py", "line": 1}
    backend = loop_mock_backend(review_results=[[issue], [issue], [issue]])

    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: backend)

    config = RunConfig(
        target=str(loop_target), skill="python", quiet=True,
        cleanup=False, loop=True, max_iterations=3,
        shallow=True,
    )
    exit_code = await run(config)

    assert exit_code == 1
    assert backend._parse_call == 3


@pytest.mark.asyncio
async def test_loop_stops_on_test_failure(loop_target, mock_ui_loop, monkeypatch):
    """Tests fail mid-loop -> reverts changes, stops immediately, exits 1."""
    issue = {"id": 1, "description": "Issue", "file": "main.py", "line": 1}
    backend = loop_mock_backend(review_results=[[issue], [issue]], tests_pass=False)

    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: backend)

    reverted = []
    monkeypatch.setattr(
        "daydream.flows.shallow.revert_uncommitted_changes", lambda cwd: (reverted.append(cwd) or True)
    )

    config = RunConfig(
        target=str(loop_target), skill="python", quiet=True,
        cleanup=False, loop=True, max_iterations=5,
        shallow=True,
    )
    exit_code = await run(config)

    assert exit_code == 1
    assert backend._parse_call == 1  # stopped after first iteration
    assert len(reverted) == 1


@pytest.mark.asyncio
async def test_loop_accumulates_stats(loop_target, mock_ui_loop, monkeypatch):
    """Stats accumulate across iterations."""
    issue1 = {"id": 1, "description": "Issue A", "file": "main.py", "line": 1}
    issue2 = {"id": 2, "description": "Issue B", "file": "main.py", "line": 2}
    backend = loop_mock_backend(review_results=[[issue1, issue2], [issue1], []])

    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: backend)

    captured_summary = {}

    import daydream.flows.shallow as shallow_mod
    original_print_summary = shallow_mod.print_summary

    def capture_summary(console, data):
        captured_summary["feedback_count"] = data.feedback_count
        captured_summary["fixes_applied"] = data.fixes_applied
        captured_summary["iterations_used"] = data.iterations_used
        captured_summary["loop_mode"] = data.loop_mode
        original_print_summary(console, data)

    monkeypatch.setattr("daydream.flows.shallow.print_summary", capture_summary)

    config = RunConfig(
        target=str(loop_target), skill="python", quiet=True,
        cleanup=False, loop=True, max_iterations=5,
        shallow=True,
    )
    exit_code = await run(config)

    assert exit_code == 0
    assert captured_summary["feedback_count"] == 3  # 2 + 1
    assert captured_summary["fixes_applied"] == 3  # 2 + 1
    assert captured_summary["iterations_used"] == 3
    assert captured_summary["loop_mode"] is True


@pytest.mark.asyncio
async def test_loop_false_single_pass(loop_target, mock_ui_loop, monkeypatch):
    """loop=False behaves identically to existing single-pass flow."""
    issue = {"id": 1, "description": "Issue", "file": "main.py", "line": 1}
    backend = loop_mock_backend(review_results=[[issue]])

    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: backend)

    config = RunConfig(
        target=str(loop_target), skill="python", quiet=True,
        cleanup=False, loop=False,
        shallow=True,
    )
    exit_code = await run(config)

    assert exit_code == 0
    assert backend._parse_call == 1  # single pass only
    assert backend.commit_calls == []


@pytest.mark.asyncio
async def test_loop_commits_between_iterations(loop_target, mock_ui_loop, monkeypatch):
    """Each successful iteration commits changes before the next review."""
    issue = {"id": 1, "description": "Add type hints", "file": "main.py", "line": 1}
    backend = loop_mock_backend(review_results=[[issue], [issue], []])

    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: backend)

    config = RunConfig(
        target=str(loop_target), skill="python", quiet=True,
        cleanup=False, loop=True, max_iterations=5,
        shallow=True,
    )
    exit_code = await run(config)

    assert exit_code == 0
    # Two successful iterations with fixes -> two commits
    assert len(backend.commit_calls) == 2
    assert "iteration: 1" in backend.commit_calls[0]
    assert "iteration: 2" in backend.commit_calls[1]


@pytest.mark.asyncio
async def test_loop_no_commit_on_test_failure(loop_target, mock_ui_loop, monkeypatch):
    """No iteration commit when tests fail; changes are reverted."""
    issue = {"id": 1, "description": "Issue", "file": "main.py", "line": 1}
    backend = loop_mock_backend(review_results=[[issue]], tests_pass=False)

    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: backend)
    monkeypatch.setattr("daydream.flows.shallow.revert_uncommitted_changes", lambda cwd: True)

    config = RunConfig(
        target=str(loop_target), skill="python", quiet=True,
        cleanup=False, loop=True, max_iterations=5,
        shallow=True,
    )
    exit_code = await run(config)

    assert exit_code == 1
    assert backend.commit_calls == []


@pytest.mark.asyncio
async def test_loop_reverted_fixes_not_counted(loop_target, mock_ui_loop, monkeypatch):
    """Fixes from a failed iteration are not counted in the summary."""
    issue = {"id": 1, "description": "Issue", "file": "main.py", "line": 1}
    backend = loop_mock_backend(review_results=[[issue]], tests_pass=False)

    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: backend)
    monkeypatch.setattr("daydream.flows.shallow.revert_uncommitted_changes", lambda cwd: True)

    captured_summary: dict[str, Any] = {}

    import daydream.flows.shallow as shallow_mod
    original_print_summary = shallow_mod.print_summary

    def capture_summary(console, data):
        captured_summary["fixes_applied"] = data.fixes_applied
        original_print_summary(console, data)

    monkeypatch.setattr("daydream.flows.shallow.print_summary", capture_summary)

    config = RunConfig(
        target=str(loop_target), skill="python", quiet=True,
        cleanup=False, loop=True, max_iterations=5,
        shallow=True,
    )
    exit_code = await run(config)

    assert exit_code == 1
    assert captured_summary["fixes_applied"] == 0


@pytest.mark.asyncio
async def test_loop_no_commit_on_clean_first_iteration(loop_target, mock_ui_loop, monkeypatch):
    """No commit when first iteration is already clean (no fixes applied)."""
    backend = loop_mock_backend(review_results=[[]])

    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: backend)

    config = RunConfig(
        target=str(loop_target), skill="python", quiet=True,
        cleanup=False, loop=True, max_iterations=5,
        shallow=True,
    )
    exit_code = await run(config)

    assert exit_code == 0
    assert backend.commit_calls == []


@pytest.mark.asyncio
async def test_loop_rejects_dirty_working_tree(loop_target, mock_ui_loop, monkeypatch):
    """Loop mode aborts with exit code 1 when the working tree is dirty."""

    (loop_target / "untracked.py").write_text("dirty")

    backend = loop_mock_backend(review_results=[[]])
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: backend)

    config = RunConfig(
        target=str(loop_target), skill="python", quiet=True,
        cleanup=False, loop=True, max_iterations=5,
        shallow=True,
    )
    exit_code = await run(config)

    assert exit_code == 1
    assert backend._parse_call == 0  # never reached review phase


def test_revert_uncommitted_changes(tmp_path):
    """revert_uncommitted_changes restores tracked files and removes untracked."""
    import subprocess

    from daydream.phases import revert_uncommitted_changes

    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=tmp_path, capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path, capture_output=True, check=True,
    )
    tracked = tmp_path / "tracked.py"
    tracked.write_text("original")
    subprocess.run(["git", "add", "."], cwd=tmp_path, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=tmp_path, capture_output=True, check=True,
    )

    tracked.write_text("modified")
    untracked = tmp_path / "new_file.py"
    untracked.write_text("junk")

    assert revert_uncommitted_changes(tmp_path) is True
    assert tracked.read_text() == "original"
    assert not untracked.exists()


def test_revert_uncommitted_changes_not_a_repo(tmp_path):
    """Returns False when cwd is not a git repo."""
    from daydream.phases import revert_uncommitted_changes

    assert revert_uncommitted_changes(tmp_path) is False


@pytest.mark.asyncio
async def test_loop_uses_incremental_diff_on_iteration_2(loop_target, mock_ui_loop, monkeypatch):
    """Iteration 1 diffs against main; iteration 2 diffs against the SHA from iteration 1's commit."""
    import subprocess

    issue = {"id": 1, "description": "Add type hints", "file": "main.py", "line": 1}
    backend = loop_mock_backend(review_results=[[issue], []])

    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: backend)

    config = RunConfig(
        target=str(loop_target), skill="python", quiet=True,
        cleanup=False, loop=True, max_iterations=5,
        shallow=True,
    )
    exit_code = await run(config)
    assert exit_code == 0

    assert len(backend.review_prompts) == 2

    assert "git diff main...HEAD" in backend.review_prompts[0]

    prompt2 = backend.review_prompts[1]
    assert "main...HEAD" not in prompt2
    assert "git diff " in prompt2
    import re
    sha_match = re.search(r"git diff ([0-9a-f]{40})\.\.\.HEAD", prompt2)
    assert sha_match is not None, f"Expected SHA-based diff in prompt: {prompt2}"

    # Verify the SHA is a real commit in the repo
    result = subprocess.run(
        ["git", "cat-file", "-t", sha_match.group(1)],
        capture_output=True, text=True, cwd=loop_target,
    )
    assert result.stdout.strip() == "commit"


@pytest.mark.asyncio
async def test_fix_phase_receives_fix_max_turns(loop_target, mock_ui_loop, monkeypatch):
    """Real-path: the fix agent's backend.execute receives max_turns == FIX_MAX_TURNS (40).

    Drives the production entrypoint (runner.run, shallow single pass) through a
    real temp git worktree and asserts the turn budget the backend ACTUALLY
    receives on the fix dispatch — not that the literal exists in source.
    """
    from daydream.phases import FIX_MAX_TURNS

    captured_fix_turns: list[int | None] = []

    class TurnCapturingBackend(PhaseDispatchBackend):
        async def execute(self, cwd, prompt, output_schema=None, continuation=None,
                          agents=None, max_turns=None, read_only=False):
            if "fix this issue" in prompt.lower():
                captured_fix_turns.append(max_turns)
            async for event in super().execute(
                cwd, prompt, output_schema, continuation, agents, max_turns, read_only
            ):
                yield event

    issue = {"id": 1, "description": "Add type hints", "file": "main.py", "line": 1}
    backend = TurnCapturingBackend(parse_results=[[issue]])

    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: backend)

    config = RunConfig(
        target=str(loop_target), skill="python", quiet=True,
        cleanup=False, loop=False, shallow=True,
    )
    exit_code = await run(config)

    assert exit_code == 0
    assert captured_fix_turns == [FIX_MAX_TURNS]
    assert FIX_MAX_TURNS == 40


@pytest.mark.asyncio
async def test_loop_diff_base_unchanged_on_test_failure(loop_target, mock_ui_loop, monkeypatch):
    """When tests fail, diff_base stays None — next iteration (if any) uses full branch diff."""
    issue = {"id": 1, "description": "Issue", "file": "main.py", "line": 1}
    backend = loop_mock_backend(review_results=[[issue], [issue]], tests_pass=False)

    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None, **kwargs: backend)
    monkeypatch.setattr("daydream.flows.shallow.revert_uncommitted_changes", lambda cwd: True)

    sha_calls: list[Path] = []
    original_get_head_sha = None

    import daydream.runner as runner_mod
    original_get_head_sha = runner_mod._get_head_sha

    def tracking_get_head_sha(cwd: Path) -> str | None:
        sha_calls.append(cwd)
        return original_get_head_sha(cwd)

    monkeypatch.setattr("daydream.runner._get_head_sha", tracking_get_head_sha)

    config = RunConfig(
        target=str(loop_target), skill="python", quiet=True,
        cleanup=False, loop=True, max_iterations=5,
        shallow=True,
    )
    exit_code = await run(config)

    assert exit_code == 1
    # Tests fail -> early return before commit -> _get_head_sha never called
    assert len(sha_calls) == 0
