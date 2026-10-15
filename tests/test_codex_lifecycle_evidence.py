"""Closed lifecycle evidence contracts; real process/runner tests own stall outcomes."""
from __future__ import annotations

import pytest

from daydream.backends.lifecycle import (
    BackendLifecycleCollector,
    BackendLifecycleFailure,
    BackendLifecycleSnapshot,
    lifecycle_failure_diagnostic,
)


@pytest.mark.parametrize("returncode", [0, -9])
def test_codex_snapshot_preserves_exit_code_and_absent_milestones(returncode: int) -> None:
    snapshot = BackendLifecycleSnapshot(attempt=1, backend="codex", process_returncode=returncode, process_exit_s=0.0)
    evidence = snapshot.to_dict()
    assert evidence["process_returncode"] == returncode
    assert evidence["process_exit_s"] == 0.0
    assert evidence["stdout_eof_s"] is None


@pytest.mark.parametrize("values", [
    {"last_native_event_kind": "private-payload"}, {"expiry_stage": "private-path"},
    {"process_returncode": True}, {"process_returncode": 2**40},
    {"raw_line_count": 2**40}, {"last_raw_line_s": float("nan")},
    {"loop_max_lag_s": float("inf")}, {"consumer_event_count": -1},
])
def test_codex_snapshot_rejects_unbounded_or_content_bearing_scalars(values: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="invalid backend lifecycle scalar"):
        BackendLifecycleSnapshot(attempt=1, backend="codex", **values)  # type: ignore[arg-type]


async def test_closed_collector_cannot_change_frozen_evidence() -> None:
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


@pytest.mark.parametrize("stage,unmet", [
    ("terminal_drain", "stdout EOF was not observed"),
    ("process_reap", "actual child exit was not observed"),
    ("cleanup", "process-group cleanup did not complete"),
])
def test_trusted_lifecycle_warning_reports_only_checked_host_facts(stage: str, unmet: str) -> None:
    """This public warning projection has a separate privacy contract from scalar persistence."""
    failure = BackendLifecycleFailure(stage, 5.0, True, stage != "terminal_drain", None)  # type: ignore[arg-type]
    warning = lifecycle_failure_diagnostic(failure)
    assert warning is not None
    assert f"Codex {stage} timeout after 5s" in warning and unmet in warning
    assert "turn.completed received" in warning and "child exit not observed" in warning
    assert "Automatic retry disabled after completion" in warning


def test_trusted_lifecycle_warning_redacts_and_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neither invalid content-bearing fields nor a broken redactor can leak a raw fallback."""
    monkeypatch.setenv("FIXTURE_SECRET", "turn.completed")
    failure = BackendLifecycleFailure("terminal_drain", 5.0, True, False, None)
    warning = lifecycle_failure_diagnostic(failure)
    assert warning is not None
    assert "turn.completed" not in warning and "[REDACTED" in warning
    invalid = BackendLifecycleFailure("secret\x1b[31m" * 300, 5.0, True, False, None)  # type: ignore[arg-type]
    assert lifecycle_failure_diagnostic(invalid) is None

    def broken_redactor(_text: str) -> str:
        raise ValueError("private raw payload")

    monkeypatch.setattr("daydream.backends.lifecycle.sanitize_verbose_message", broken_redactor)
    assert lifecycle_failure_diagnostic(failure) is None
