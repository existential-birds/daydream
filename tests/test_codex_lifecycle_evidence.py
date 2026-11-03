"""Closed lifecycle evidence contracts; real process/runner tests own stall outcomes."""
from __future__ import annotations

import pytest

from daydream.backends.lifecycle import (
    BackendLifecycleCollector,
    BackendLifecycleFailure,
    BackendLifecycleSnapshot,
    lifecycle_failure_diagnostic,
)


async def test_lifecycle_evidence_is_bounded_frozen_and_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Content-bearing scalars, post-freeze updates, invalid failures and broken redaction cannot leak evidence."""
    for values in (
        {"last_native_event_kind": "private-payload"}, {"expiry_stage": "private-path"},
        {"process_returncode": True}, {"process_returncode": 2**40},
        {"raw_line_count": 2**40}, {"last_raw_line_s": float("nan")},
        {"loop_max_lag_s": float("inf")}, {"consumer_event_count": -1},
    ):
        with pytest.raises(ValueError, match="invalid backend lifecycle scalar"):
            BackendLifecycleSnapshot(attempt=1, backend="codex", **values)

    lifecycle = BackendLifecycleCollector(attempt=1, started_at=0.0, wall_allowance_s=None, deadline=None)
    lifecycle.activate_codex(stream_idle_timeout_s=None, terminal_drain_timeout_s=5.0,
                             process_reap_timeout_s=5.0, cleanup_timeout_s=10.0)
    lifecycle.consumer_requested()
    await lifecycle.stop_sampling()
    frozen = lifecycle.finalize()
    assert frozen is not None
    assert frozen.consumer_event_count == 0
    assert frozen.raw_line_count == 0
    lifecycle.raw_line_received()
    assert lifecycle.finalize() == frozen

    monkeypatch.setenv("FIXTURE_SECRET", "turn.completed")
    failure = BackendLifecycleFailure("terminal_drain", 5.0, True, False, None)
    warning = lifecycle_failure_diagnostic(failure)
    assert warning is not None
    assert "turn.completed" not in warning and "[REDACTED" in warning
    invalid = BackendLifecycleFailure("secret\x1b[31m" * 300, 5.0, True, False, None)  # type: ignore[arg-type]
    assert lifecycle_failure_diagnostic(invalid) is None
    assert lifecycle_failure_diagnostic(RuntimeError("terminal_drain")) is None

    def broken_redactor(_text: str) -> str:
        raise ValueError("private raw payload")

    monkeypatch.setattr("daydream.backends.lifecycle.sanitize_verbose_message", broken_redactor)
    assert lifecycle_failure_diagnostic(failure) is None
