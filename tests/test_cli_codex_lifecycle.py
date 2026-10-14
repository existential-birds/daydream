"""The real CLI preserves external Codex evidence on bounded reap failure."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from tests.harness.codex_lifecycle_cli import install_codex_lifecycle_cli
from tests.test_cli_main_integration import _cli_main_exit, _silence_all


def test_cli_records_terminal_evidence_when_codex_closes_stdout_but_does_not_exit(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _silence_all(monkeypatch)
    fixture = install_codex_lifecycle_cli(tmp_path / "codex", mode="terminal_eof_live_child")
    monkeypatch.setenv("PATH", f"{fixture.bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("DAYDREAM_CODEX_PROCESS_REAP_TIMEOUT_S", "0.2")
    monkeypatch.setenv("DAYDREAM_CODEX_CLEANUP_TIMEOUT_S", "1")
    monkeypatch.setenv("DAYDREAM_PI_RETRY_ATTEMPTS", "0")
    monkeypatch.setattr("daydream.deep.review_steps.EXPLORATION_AVAILABLE", False)
    trajectory = tmp_path / "trajectory.json"
    before = (multi_stack_target / "api.py").read_bytes()
    code = _cli_main_exit(monkeypatch, "--review", "--backend", "codex", "--model", "fixture-model",
                          "--test-command", f"{sys.executable} -c pass", "--trajectory", str(trajectory),
                          str(multi_stack_target))
    assert code == 1
    assert (multi_stack_target / "api.py").read_bytes() == before
    payload = json.loads(trajectory.read_text())
    assert "Synthetic lifecycle response" in json.dumps(payload)
    attempts = [attempt for invocation in payload["extra"]["subtrajectories"]
                for attempt in invocation.get("invocations", [invocation])]
    lifecycles = [attempt["backend_lifecycle"] for attempt in attempts if "backend_lifecycle" in attempt]
    assert len(lifecycles) == 1
    lifecycle = lifecycles[0]
    assert lifecycle["terminal_received_s"] is not None
    assert lifecycle["stdout_eof_s"] is not None
    assert lifecycle["expiry_stage"] == "process_reap"
    assert lifecycle["teardown_completed_s"] is not None
    assert payload["final_metrics"]["total_prompt_tokens"] == 10
    assert payload["final_metrics"]["total_completion_tokens"] == 5
