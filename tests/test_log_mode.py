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
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from daydream.backends import (
    Backend,
    CostEvent,
    MetricsEvent,
    ResultEvent,
    TextEvent,
    ThinkingEvent,
    ToolResultEvent,
    ToolStartEvent,
)
from daydream.runner import RunConfig, run


class _LogModeStubBackend:
    """Mock backend for testing log mode output formatting."""

    model = "test-model"

    def __init__(self, events: list[Any]):
        """Initialize with a sequence of events to emit."""
        self.events = events
        self.retryable = False

    async def execute(
        self,
        cwd: Path,
        prompt: str,
        output_schema: dict[str, Any] | None = None,
        continuation: Any = None,
        agents: dict[str, Any] | None = None,
        max_turns: int | None = None,
        read_only: bool = False,
    ) -> AsyncIterator[Any]:
        """Emit the configured events in sequence."""
        for event in self.events:
            yield event

    async def cancel(self) -> None:
        """No-op cancel for testing."""
        pass

    def format_skill_invocation(self, skill_key: str, args: str = "") -> str:
        """Return formatted skill invocation."""
        return f"[skill:{skill_key}] {args}"


def _capture_stdout_and_run(config: RunConfig, backend: Backend, monkeypatch: pytest.MonkeyPatch) -> str:
    """Run daydream with the given config and capture stdout output."""
    # Patch create_backend to return our test backend
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, model=None: backend)

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


def test_log_mode_produces_plain_text(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that --log mode produces plain text output without ANSI escape sequences."""
    # Create backend that emits simple text
    backend = _LogModeStubBackend([
        TextEvent("hello world"),
    ])

    # Configure log mode
    config = RunConfig(
        target=str(multi_stack_target),
        log_mode=True,
        non_interactive=True,
        quiet=True,
        output_mode="review",
    )

    # Capture output
    output = _capture_stdout_and_run(config, backend, monkeypatch)

    # Verify plain text output
    assert "hello world" in output

    # Verify no ANSI escape sequences (Rich markup patterns)
    assert "\x1b[" not in output  # No ANSI color codes
    assert "[bold]" not in output  # No Rich markup
    assert "[dim]" not in output   # No Rich styling


def test_log_mode_dumps_tool_events(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that --log mode dumps tool events with proper markers."""
    # Create backend that emits tool events
    backend = _LogModeStubBackend([
        ToolStartEvent(
            id="test-id",
            name="bash",
            input={"command": "echo hello", "description": "test command"}
        ),
        ToolResultEvent(
            id="test-id",
            output="hello\nworld",
            is_error=False
        ),
    ])

    config = RunConfig(
        target=str(multi_stack_target),
        log_mode=True,
        non_interactive=True,
        quiet=True,
        output_mode="review",
    )

    output = _capture_stdout_and_run(config, backend, monkeypatch)

    # Verify tool start marker
    assert "[tool:bash] echo hello" in output

    # Verify tool result marker
    assert "[tool:bash result] hello" in output


def test_log_mode_dumps_cost(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that --log mode dumps cost events with proper formatting."""
    backend = _LogModeStubBackend([
        CostEvent(cost_usd=0.0042, input_tokens=100, output_tokens=50),
    ])

    config = RunConfig(
        target=str(multi_stack_target),
        log_mode=True,
        non_interactive=True,
        quiet=True,
        output_mode="review",
    )

    output = _capture_stdout_and_run(config, backend, monkeypatch)

    # Verify cost formatting
    assert "[cost] $0.0042" in output


def test_log_mode_dumps_metrics(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that --log mode dumps metrics events."""
    backend = _LogModeStubBackend([
        MetricsEvent(
            message_id="test-msg",
            prompt_tokens=100,
            completion_tokens=50,
            cached_tokens=None,
            cost_usd=None
        ),
    ])

    config = RunConfig(
        target=str(multi_stack_target),
        log_mode=True,
        non_interactive=True,
        quiet=True,
        output_mode="review",
    )

    output = _capture_stdout_and_run(config, backend, monkeypatch)

    # Verify metrics formatting
    assert "[metrics] prompt=100 completion=50" in output


def test_log_mode_dumps_thinking(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that --log mode dumps thinking events."""
    backend = _LogModeStubBackend([
        ThinkingEvent("I need to analyze this code"),
    ])

    config = RunConfig(
        target=str(multi_stack_target),
        log_mode=True,
        non_interactive=True,
        quiet=True,
        output_mode="review",
    )

    output = _capture_stdout_and_run(config, backend, monkeypatch)

    # Verify thinking marker
    assert "[thinking] I need to analyze this code" in output


def test_log_mode_dumps_result_event(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that --log mode dumps structured result events."""
    backend = _LogModeStubBackend([
        ResultEvent(
            structured_output={"status": "complete", "findings": ["issue1", "issue2"]},
            continuation=None
        ),
    ])

    config = RunConfig(
        target=str(multi_stack_target),
        log_mode=True,
        non_interactive=True,
        quiet=True,
        output_mode="review",
    )

    output = _capture_stdout_and_run(config, backend, monkeypatch)

    # Verify result formatting (truncated to 500 chars)
    assert "[result]" in output
    assert '"status": "complete"' in output
    assert '"findings"' in output


def test_log_mode_tool_error_handling(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that --log mode handles tool errors with ERROR prefix."""
    backend = _LogModeStubBackend([
        ToolStartEvent(
            id="test-id",
            name="bash",
            input={"command": "false", "description": "failing command"}
        ),
        ToolResultEvent(
            id="test-id",
            output="command failed with exit code 1",
            is_error=True
        ),
    ])

    config = RunConfig(
        target=str(multi_stack_target),
        log_mode=True,
        non_interactive=True,
        quiet=True,
        output_mode="review",
    )

    output = _capture_stdout_and_run(config, backend, monkeypatch)

    # Verify error prefix
    assert "[tool:bash ERROR] command failed with exit code 1" in output


def test_log_mode_with_non_interactive(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that --log works with --non-interactive (orthogonal flags)."""
    backend = _LogModeStubBackend([
        TextEvent("processing in non-interactive mode"),
    ])

    config = RunConfig(
        target=str(multi_stack_target),
        log_mode=True,
        non_interactive=True,
        quiet=True,
        output_mode="review",
    )

    output = _capture_stdout_and_run(config, backend, monkeypatch)

    # Verify the flags work together
    assert "processing in non-interactive mode" in output
    assert "\x1b[" not in output  # Still no ANSI codes


def test_log_mode_default_off(multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that default behavior (no --log) still uses Rich UI."""
    # This test is more challenging since we can't easily capture Rich output
    # but we can verify the backend was called and no plain text markers appear
    backend = _LogModeStubBackend([
        TextEvent("hello world"),
        CostEvent(cost_usd=0.0042, input_tokens=100, output_tokens=50),
    ])

    config = RunConfig(
        target=str(multi_stack_target),
        log_mode=False,  # Default off
        non_interactive=True,
        quiet=False,  # Allow Rich UI
        output_mode="review",
    )

    output = _capture_stdout_and_run(config, backend, monkeypatch)

    # In Rich mode, we should not see the raw log markers
    # (The Rich UI would format these differently)
    assert "[cost] $0.0042" not in output  # Raw log format should not appear


def test_log_mode_trajectory_still_written(
    multi_stack_target: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Test that --log mode still writes trajectory file (recorder unaffected)."""
    backend = _LogModeStubBackend([
        TextEvent("generating trajectory"),
    ])

    trajectory_path = tmp_path / "trajectory.json"

    config = RunConfig(
        target=str(multi_stack_target),
        log_mode=True,
        non_interactive=True,
        quiet=True,
        output_mode="review",
        trajectory_path=trajectory_path,
    )

    output = _capture_stdout_and_run(config, backend, monkeypatch)

    # Verify trajectory file was created
    assert trajectory_path.exists(), "Trajectory file should be written even in log mode"

    # Verify log output still works
    assert "generating trajectory" in output
