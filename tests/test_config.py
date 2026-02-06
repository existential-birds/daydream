# tests/test_config.py
"""Tests for daydream configuration constants."""

from daydream.config import (
    RLM_REPL_INIT_TIMEOUT,
    RLM_CODE_EXEC_TIMEOUT,
    RLM_HEARTBEAT_TIMEOUT,
    RLM_LLM_QUERY_TIMEOUT,
    RLM_CONTAINER_STARTUP_TIMEOUT,
    RLM_HEARTBEAT_INTERVAL,
    RLM_OUTPUT_TRUNCATION_LIMIT,
)


def test_rlm_timeout_constants_exist():
    """RLM timeout constants should be defined."""
    assert RLM_REPL_INIT_TIMEOUT == 60
    assert RLM_CODE_EXEC_TIMEOUT == 300
    assert RLM_HEARTBEAT_TIMEOUT == 5
    assert RLM_LLM_QUERY_TIMEOUT == 60
    assert RLM_CONTAINER_STARTUP_TIMEOUT == 120


def test_rlm_heartbeat_interval():
    """RLM heartbeat interval should be defined."""
    assert RLM_HEARTBEAT_INTERVAL == 10


def test_rlm_output_truncation_limit():
    """RLM output truncation limit should be 50k chars."""
    assert RLM_OUTPUT_TRUNCATION_LIMIT == 50_000
