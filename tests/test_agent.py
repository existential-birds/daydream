"""Tests for daydream.agent module-level state accessors."""

from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from daydream.agent import (
    get_non_interactive,
    is_environmental_failure,
    reset_state,
    run_agent,
    set_non_interactive,
)
from daydream.backends import DiagnosticEvent, ResultEvent
from daydream.extensions import ToolDecision, get_registry, set_registry
from daydream.extensions.registry import Registry
from daydream.trajectory import DaydreamPhase
from tests.harness.backend import ScriptedBackend
from tests.harness.trajectory import make_recorder, read_trajectory


def test_set_and_get_non_interactive() -> None:
    try:
        set_non_interactive(True)
        assert get_non_interactive() is True
    finally:
        reset_state()


def test_reset_state_clears_non_interactive() -> None:
    set_non_interactive(True)
    reset_state()
    assert get_non_interactive() is False


def test_is_environmental_failure_both_directions() -> None:
    environmental = [
        "The dev Postgres container is not running",
        "could not connect to server: Connection refused",
        "localhost:5432",
        "ECONNREFUSED",
    ]
    for output in environmental:
        assert is_environmental_failure(output) is True, output

    ordinary = [
        "AssertionError: assert 1 == 2",
        "1 failed, 3 passed",
        "ValueError: bad input",
    ]
    for output in ordinary:
        assert is_environmental_failure(output) is False, output


def test_scrubbed_supervisor_error_scrubs_all_str_surfaces() -> None:
    """_scrubbed_supervisor_error must never re-surface a redactable value.

    Regression for issue #702 round 2: the args-scrub must hold for
    OSError-family types (whose str() is built from errno/strerror, not args)
    and for types overriding __str__/__repr__, and must preserve the
    retryable discriminator on the reconstruction path too.
    """
    from daydream.agent import (
        _RedactedSupervisorError,
        _scrubbed_supervisor_error,
    )

    credential = "ZAI_API_KEY=credential-shaped-supervisor-value"

    # OSError-family: real (sub)type preserved, str() scrubbed
    err = OSError(2, f"failed auth with {credential}")
    rebuilt = _scrubbed_supervisor_error(err)
    assert type(rebuilt) is type(err)
    assert isinstance(rebuilt, OSError)
    assert credential not in str(rebuilt)
    assert "[REDACTED_ENV_VAR]" in str(rebuilt)

    # custom __str__ override: fail closed to the already-redacted stand-in
    class CustomLeak(Exception):
        def __str__(self) -> str:
            return f"boom {self.args}"

    custom = CustomLeak((credential,))
    stand_in = _scrubbed_supervisor_error(custom)
    assert type(stand_in) is _RedactedSupervisorError
    assert credential not in str(stand_in)
    assert stand_in.original_type_name == "CustomLeak"

    # reconstruction path must preserve retryable even when it is an
    # instance attribute set from a non-args kwarg (e.g. BackendError).
    class RetryableBackendError(RuntimeError):
        def __init__(self, message: str, *, retryable: bool = False) -> None:
            super().__init__(message)
            self.retryable = retryable

    backend = RetryableBackendError(f"boom {credential}", retryable=True)
    rebuilt_retryable = _scrubbed_supervisor_error(backend)
    assert type(rebuilt_retryable) is RetryableBackendError
    assert credential not in str(rebuilt_retryable)
    assert getattr(rebuilt_retryable, "retryable", False) is True


@pytest.mark.anyio
async def test_diagnostic_event_is_recorder_only_and_has_no_agent_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_agent forwards diagnostics without UI, callback, supervision, or budget effects."""
    output = StringIO()
    monkeypatch.setattr("daydream.agent.console", Console(file=output, force_terminal=False))
    callback_events: list[object] = []
    supervisor_events: list[tuple[str, dict[str, Any]]] = []

    def callback(value: object) -> None:
        callback_events.append(value)

    def supervisor(
        tool_name: str, tool_input: dict[str, Any], *, phase: DaydreamPhase
    ) -> ToolDecision:
        supervisor_events.append((tool_name, tool_input))
        return ToolDecision(False, "")

    registry = Registry()
    registry.register_tool_supervisor(supervisor)
    previous_registry = get_registry()
    set_registry(registry)
    recorder = make_recorder(tmp_path)
    try:
        async with recorder:
            result = await run_agent(
                ScriptedBackend(
                    events=[
                        DiagnosticEvent(
                            code="codex_parser_coverage",
                            message="bounded parser evidence",
                            metadata={"count": 1},
                        ),
                        ResultEvent(structured_output=None, continuation=None),
                    ]
                ),
                tmp_path,
                "inspect",
                phase=DaydreamPhase.REVIEW,
                progress_callback=callback,
                tool_call_budget=0,
            )
    finally:
        set_registry(previous_registry)
        reset_state()

    assert result == ("", None, None)
    assert callback_events == []
    assert supervisor_events == []
    assert output.getvalue() == ""
    trajectory = read_trajectory(recorder.path)
    agent_steps = [step for step in trajectory["steps"] if step["source"] == "agent"]
    assert len(agent_steps) == 1
    assert agent_steps[0]["message"] == ""
    assert agent_steps[0]["extra"]["backend_diagnostics"] == [
        {
            "code": "codex_parser_coverage",
            "message": "bounded parser evidence",
            "metadata": {"count": 1},
        }
    ]
