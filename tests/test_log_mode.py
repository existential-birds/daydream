"""Integration tests for --log mode (bypass Rich UI, emit plain text).

Tests the log mode implementation using real-path tests through runner.run()
with real filesystem/git/event-loop, mocking only the backend seam.

These tests verify that --log mode:
1. Bypasses all Rich UI components and emits plain text to stdout
2. Dumps tool events with proper markers ([tool:bash], [tool:bash result])
3. Emits cost events with proper formatting ([cost] $0.0042)
4. Works with other flags like --non-interactive
5. Still records full trajectory (recorder unaffected)
6. Default behavior unchanged (Rich UI when --log not used)
"""

from __future__ import annotations

import io
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from daydream.backends import (
    AgentEvent,
    CostEvent,
    MetricsEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from daydream.runner import RunConfig, run
from tests.harness.backend import ScriptedBackend

MakeConfig = Callable[..., RunConfig]
InstallBackend = Callable[[object], object]


def _capture_stdout_and_run(config: RunConfig, monkeypatch: pytest.MonkeyPatch) -> str:
    """Run daydream with the given config and capture stdout output."""
    # Mock external dependencies
    monkeypatch.setattr("daydream.runner.print_phase_hero", lambda *a, **kw: None)
    monkeypatch.setattr("daydream.git_ops.gh_repo_view", lambda repo: ("test", "repo"))
    monkeypatch.setattr("daydream.git_ops.gh_pr_view", lambda repo, _branch: None)

    # Capture stdout
    old_stdout = sys.stdout
    captured_output = io.StringIO()
    sys.stdout = captured_output

    try:
        # Run the actual runner.run function
        import anyio

        exit_code = anyio.run(run, config)
        assert exit_code == 0, "Expected successful run"
    finally:
        sys.stdout = old_stdout

    return captured_output.getvalue()


@pytest.mark.parametrize(
    ("events", "config_overrides", "required", "forbidden"),
    [
        pytest.param(
            [TextEvent("hello world")],
            {"log_mode": True, "quiet": True, "output_mode": "review"},
            ("hello world",),
            ("\x1b[", "[bold]", "[dim]"),
            id="plain-text",
        ),
        pytest.param(
            [
                ToolStartEvent(
                    id="test-id",
                    name="bash",
                    input={"command": "echo hello", "description": "test command"},
                ),
                ToolResultEvent(id="test-id", output="hello\nworld", is_error=False),
            ],
            {"log_mode": True, "quiet": True, "output_mode": "review"},
            ("[tool:bash] echo hello", "[tool:bash result] hello"),
            (),
            id="tool-events",
        ),
        pytest.param(
            [CostEvent(cost_usd=0.0042, input_tokens=100, output_tokens=50)],
            {"log_mode": True, "quiet": True, "output_mode": "review"},
            ("[cost] $0.0042",),
            (),
            id="cost",
        ),
        pytest.param(
            [
                MetricsEvent(
                    message_id="test-msg",
                    prompt_tokens=100,
                    completion_tokens=50,
                    cached_tokens=None,
                    cost_usd=None,
                )
            ],
            {"log_mode": True, "quiet": True, "output_mode": "review"},
            ("[metrics] prompt=100 completion=50",),
            (),
            id="metrics",
        ),
        pytest.param(
            [ThinkingEvent("I need to analyze this code")],
            {"log_mode": True, "quiet": True, "output_mode": "review"},
            ("[thinking] I need to analyze this code",),
            (),
            id="thinking",
        ),
        pytest.param(
            [
                ResultEvent(
                    structured_output={"status": "complete", "findings": ["issue1", "issue2"]},
                    continuation=None,
                )
            ],
            {"log_mode": True, "quiet": True, "output_mode": "review"},
            ("[result]", '"status": "complete"', '"findings"'),
            (),
            id="result-event",
        ),
        pytest.param(
            [
                ToolStartEvent(
                    id="test-id",
                    name="bash",
                    input={"command": "false", "description": "failing command"},
                ),
                ToolResultEvent(
                    id="test-id",
                    output="command failed with exit code 1",
                    is_error=True,
                ),
            ],
            {"log_mode": True, "quiet": True, "output_mode": "review"},
            ("[tool:bash ERROR] command failed with exit code 1",),
            (),
            id="tool-error",
        ),
        pytest.param(
            [
                TextEvent("hello world"),
                CostEvent(cost_usd=0.0042, input_tokens=100, output_tokens=50),
            ],
            {"log_mode": False, "quiet": False, "output_mode": "review"},
            (),
            ("[cost] $0.0042",),
            id="default-off",
        ),
    ],
)
def test_log_mode_rendering(
    multi_stack_target: Path,
    make_config: MakeConfig,
    install_backend: InstallBackend,
    monkeypatch: pytest.MonkeyPatch,
    events: list[AgentEvent],
    config_overrides: dict[str, object],
    required: tuple[str, ...],
    forbidden: tuple[str, ...],
) -> None:
    install_backend(ScriptedBackend(events=events, retryable=False))
    config = make_config(
        multi_stack_target,
        non_interactive=True,
        **config_overrides,
    )
    output = _capture_stdout_and_run(config, monkeypatch)

    for substring in required:
        assert substring in output
    for substring in forbidden:
        assert substring not in output


def test_log_mode_trajectory_still_written(
    multi_stack_target: Path,
    make_config: MakeConfig,
    install_backend: InstallBackend,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test that --log mode still writes trajectory file (recorder unaffected)."""
    install_backend(ScriptedBackend(events=[TextEvent("generating trajectory")], retryable=False))

    trajectory_path = tmp_path / "trajectory.json"

    config = make_config(
        multi_stack_target,
        log_mode=True,
        quiet=True,
        output_mode="review",
        trajectory_path=trajectory_path,
    )

    output = _capture_stdout_and_run(config, monkeypatch)

    # Verify trajectory file was created
    assert trajectory_path.exists(), "Trajectory file should be written even in log mode"

    # Verify log output still works
    assert "generating trajectory" in output
