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

# Synthetic credential-shaped sentinel: the ``{6,}`` API-key rule requires at
# least 6 chars after the ``ghp_`` prefix, so a truncated ``ghp_yyy`` fragment
# (only 3) is unmatchable — the discriminating boundary tests rely on this.
REDACTION_SENTINEL = "ghp_" + "x" * 16

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
        # Sentinel-absence is asserted at the agent boundary in
        # test_log_mode_emission_redacts_sentinels_on_agent_path; marker presence
        # here proves the agent's `_print_log` redaction runs on the real path.
        # The forbidden tuples keep the log-mode markup-free invariant guarded:
        # captured stdout on this path is plain text (no Rich escape sequences).
        pytest.param(
            [TextEvent(f"token=hello {REDACTION_SENTINEL} world")],
            {"log_mode": True, "quiet": True, "output_mode": "review"},
            ("hello", "[REDACTED_API_KEY]"),
            ("\x1b[", "[bold]", "[dim]"),
            id="plain-text",
        ),
        pytest.param(
            [ThinkingEvent(f"thinking about {REDACTION_SENTINEL}")],
            {"log_mode": True, "quiet": True, "output_mode": "review"},
            ("[thinking]", "[REDACTED_API_KEY]"),
            (),
            id="thinking-sentinel",
        ),
        pytest.param(
            [
                ToolStartEvent(
                    id="test-id",
                    name="bash",
                    input={"command": f"echo {REDACTION_SENTINEL}"},
                ),
                ToolResultEvent(id="test-id", output=f"token={REDACTION_SENTINEL}", is_error=False),
            ],
            {"log_mode": True, "quiet": True, "output_mode": "review"},
            ("[tool:bash]", "[REDACTED_API_KEY]"),
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
                    structured_output={"status": "complete", "token": REDACTION_SENTINEL},
                    continuation=None,
                )
            ],
            {"log_mode": True, "quiet": True, "output_mode": "review"},
            ("[result]", "[REDACTED_CREDENTIAL]"),
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
    tiny_diff_target: Path,
    make_config: MakeConfig,
    install_backend: InstallBackend,
    monkeypatch: pytest.MonkeyPatch,
    events: list[AgentEvent],
    config_overrides: dict[str, object],
    required: tuple[str, ...],
    forbidden: tuple[str, ...],
) -> None:
    """Render only the event fields appropriate to each log-mode scenario."""
    install_backend(ScriptedBackend(events=events, retryable=False))
    config = make_config(
        tiny_diff_target,
        non_interactive=True,
        **config_overrides,
    )
    output = _capture_stdout_and_run(config, monkeypatch)

    for substring in required:
        assert substring in output
    for substring in forbidden:
        assert substring not in output


def test_log_mode_redacts_tool_summary_before_200_truncation() -> None:
    """A token straddling the 200-char summary boundary is redacted, not truncated
    into an unmatchable fragment. Redact-after-slice would leak the raw 'ghp_'."""
    from daydream.agent import _summarize_input

    # The space before the token is REQUIRED: `_API_KEY_PATTERN` anchors on `\b`,
    # so a token glued directly after a word char is never matchable and the test
    # would fail against BOTH implementations. With the space, redact_text matches
    # the complete ``ghp_`` + 8 alnum token, while ``[:200]`` still cuts 9 chars
    # into it — a redact-after implementation leaks the unmatchable ``ghp_yyyyy``
    # fragment (the ``{6,}`` rule needs 6+ chars after the prefix).
    command = "x" * 190 + " " + "ghp_" + "y" * 8 + "z" * 10   # [:200] cuts into the token
    out = _summarize_input({"command": command})
    assert "ghp_" not in out        # no raw fragment survives the 200-cut
    assert "[REDACTED" in out       # redaction marker (even truncated) present

    # Same boundary check on the output summary: a token straddling the [:200]
    # first-line cut must be caught before strip/first-line/slice.
    from daydream.agent import _summarize_output

    out = _summarize_output("x" * 190 + " " + "ghp_" + "y" * 8 + "z" * 10)
    assert "ghp_" not in out
    assert "[REDACTED" in out


def test_log_mode_summaries_redact_structured_credentials() -> None:
    """Log-mode summaries must use the structured redactor, not the flat one.

    Issue #455 broadens structured credential redaction. Flat ``redact_text``
    leaks structured credentials that ``redact_structured_text`` catches -- e.g.
    a nested ``key=value`` assignment (``token=opaque-test-12345``) and a Basic
    auth header's base64 credential. These flow through the real agent summary
    paths (``_summarize_input`` / ``_summarize_output`` / ``_print_log``), so a
    flat-only redactor would print the secret in --log mode.
    """
    from daydream.agent import _print_log, _summarize_input, _summarize_output

    # Nested assignment under a sensitive key: flat redaction leaks the token.
    out = _summarize_input({"command": "the config: token=opaque-test-12345"})
    assert "opaque-test-12345" not in out
    assert "[REDACTED" in out

    # Basic auth header value (base64): flat redaction leaks it.
    out = _summarize_output("Authorization: Basic dXNlcjpwYXNzd29yZA== more")
    assert "dXNlcjpwYXNzd29yZA==" not in out
    assert "[REDACTED" in out

    # And the direct _print_log emitter on a command with a structured pair.
    import io
    from contextlib import redirect_stdout

    buf = io.StringIO()
    with redirect_stdout(buf):
        _print_log("run: token=opaque-test-12345 done")
    assert "opaque-test-12345" not in buf.getvalue()
    assert "[REDACTED" in buf.getvalue()


def test_log_mode_console_redacts_string_payloads() -> None:
    """phases.py imports agent's module-level console; in --log mode that
    console redacts string payloads, so phase/UI output (e.g. the failure
    handoff body) cannot bypass the log-mode redaction boundary."""
    from daydream.agent import _LogRedactingConsole, set_log_mode

    sentinel = REDACTION_SENTINEL
    buffer = io.StringIO()
    rec = _LogRedactingConsole(file=buffer, force_terminal=True, width=100)

    set_log_mode(False)
    try:
        # Normal mode: raw pass-through (Rich UI is unredacted by design).
        rec.print(f"handoff token={sentinel}")
        assert sentinel in buffer.getvalue()

        # --log mode: the same emission path is redacted at the console boundary.
        buffer.truncate(0)
        buffer.seek(0)
        set_log_mode(True)
        rec.print(f"handoff token={sentinel}")
        out = buffer.getvalue()
        # Rich's highlighter splits the marker's brackets into styled spans, so
        # assert on the marker body (and the sentinel's absence) rather than the
        # exact bracketed string.
        assert "REDACTED_API_KEY" in out
        assert sentinel not in out
    finally:
        set_log_mode(False)


def test_log_mode_trajectory_still_written(
    tiny_diff_target: Path,
    make_config: MakeConfig,
    install_backend: InstallBackend,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Test that --log mode still writes trajectory file (recorder unaffected)."""
    install_backend(ScriptedBackend(events=[TextEvent("generating trajectory")], retryable=False))

    trajectory_path = tmp_path / "trajectory.json"

    config = make_config(
        tiny_diff_target,
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
