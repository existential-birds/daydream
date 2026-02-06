# tests/rlm/test_errors.py
"""Tests for RLM error types."""

from daydream.rlm.errors import (
    RLMError,
    REPLCrashError,
    REPLTimeoutError,
    HeartbeatFailedError,
    ContainerError,
)


def test_rlm_error_is_exception():
    """RLMError should be a base Exception."""
    err = RLMError("test message")
    assert isinstance(err, Exception)
    assert str(err) == "test message"


def test_repl_crash_error_inherits_rlm_error():
    """REPLCrashError should inherit from RLMError."""
    err = REPLCrashError("process died")
    assert isinstance(err, RLMError)
    assert "process died" in str(err)


def test_repl_timeout_error_inherits_rlm_error():
    """REPLTimeoutError should inherit from RLMError."""
    err = REPLTimeoutError("exceeded 300s")
    assert isinstance(err, RLMError)


def test_heartbeat_failed_error_inherits_rlm_error():
    """HeartbeatFailedError should inherit from RLMError."""
    err = HeartbeatFailedError("no pong received")
    assert isinstance(err, RLMError)


def test_container_error_inherits_rlm_error():
    """ContainerError should inherit from RLMError."""
    err = ContainerError("failed to start")
    assert isinstance(err, RLMError)
