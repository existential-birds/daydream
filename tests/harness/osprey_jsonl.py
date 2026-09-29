"""Shared Osprey session JSONL fixture.

The minimal valid Osprey protocol stream: a ``protocol`` line, a
``session_start``, the caller's turn events, and a completed ``session_end``.
Both the backend contract tests and the lifecycle tests consume this so the
session envelope has a single owner.
"""

from __future__ import annotations


def osprey_session(*events: dict[str, object]) -> list[dict[str, object]]:
    """Return one valid Osprey session with *events* spliced before ``session_end``."""
    return [
        {"event": "protocol", "version": 2},
        {
            "event": "session_start",
            "session_id": "s-137",
            "started_at": "2026-08-15T00:00:00Z",
            "model": "custom-model",
            "provider": "openai-compatible",
        },
        *events,
        {
            "event": "session_end",
            "total_turns": 1,
            "session_wallclock_ms": 15,
            "total_cost_usd": None,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_cached_tokens": None,
            "total_cache_write_tokens": None,
            "total_thinking_tokens": 0,
            "total_oom_kills": 0,
            "p50_turn_ms": 15,
            "p99_turn_ms": 15,
            "avg_turn_cost_usd": None,
            "structured_output": None,
            "outcome": "completed",
            "verification": None,
            "exit_code": 0,
        },
    ]
