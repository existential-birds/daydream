"""Trusted Codex shutdown warnings through the real staged runner and frozen record."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from daydream import runner
from daydream.config import REVIEW_OUTPUT_FILE
from daydream.config_file import DaydreamFileConfig
from daydream.dataset import LocalRecordStore
from tests.harness.codex_lifecycle_cli import install_codex_lifecycle_cli
from tests.harness.dataset import read_records
from tests.harness.trajectory import read_trajectory, root_trajectory
from tests.test_deep_orchestrator import MakeConfig, _silence


async def test_structure_shutdown_failure_retains_siblings_and_trusted_folded_warning(
    multi_stack_target: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig,
) -> None:
    """Only the real external CLI is substituted; Git, stages, recorder and freezing stay real."""
    target = multi_stack_target
    _silence(monkeypatch)
    fixture = install_codex_lifecycle_cli(tmp_path / "cli", structure_mode="terminal_open_stdout")
    monkeypatch.setenv("PATH", f"{fixture.bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("DAYDREAM_CODEX_TERMINAL_DRAIN_TIMEOUT_S", "0.5")
    monkeypatch.setenv("DAYDREAM_CODEX_CLEANUP_TIMEOUT_S", "1")
    monkeypatch.setenv("DAYDREAM_PI_RETRY_ATTEMPTS", "0")
    monkeypatch.setattr("daydream.deep.review_steps.EXPLORATION_AVAILABLE", False)
    store = tmp_path / "records"
    before = {file: (target / file).read_bytes() for file in ("api.py", "App.tsx", "README.md")}
    async with asyncio.timeout(25):
        assert await runner.run(make_config(
            target, output_mode="review", backend="codex", model="fixture-model", shallow_fanout_threshold=0,
            review_cache_enabled=False, file_config=DaydreamFileConfig(quality_gate_enabled=False),
            dataset_capture=True, dataset_store_path=store, run_eval=False,
        )) == 0
    assert {file: (target / file).read_bytes() for file in before} == before
    deep = target / ".daydream" / "deep"
    coverage = json.loads((deep / "review-coverage.json").read_text())
    scopes = {row["scope_id"]: row for row in coverage["stack_outcomes"]}
    assert scopes["structure"]["status"] == "failed"
    assert scopes["structure"]["reason_codes"] == ["backend_failure"]
    assert all(scopes[scope]["status"] == "complete" for scope in ("python", "react", "generic"))
    warning = coverage["diagnostics"]["scopes"]["structure"]
    assert "Codex terminal_drain timeout after 0.5s" in warning
    assert "stdout EOF was not observed" in warning and "child exit not observed" in warning
    assert "Automatic retry disabled after completion" in warning
    assert warning in coverage["diagnostics"]["phases"]["alternatives"]
    report = (target / REVIEW_OUTPUT_FILE).read_text()
    assert warning in report and "Review incomplete" in report
    items = json.loads((deep / "merged-items.json").read_text())["items"]
    assert {item["file"] for item in items} == set(before)
    root = root_trajectory(target)
    refs = root["extra"]["subtrajectories"]
    assert not any("alternatives" in row.get("descriptor", "") for row in refs)
    structure = next(row for row in refs if row.get("descriptor") == "deep-structure")
    child = read_trajectory(target / ".daydream" / structure["sibling_trajectory_ref"])
    attempts = child["extra"]["subtrajectories"]
    assert len(attempts) == 1
    evidence = attempts[0]["backend_lifecycle"]
    assert evidence["expiry_stage"] == "terminal_drain" and evidence["terminal_drain_timeout_s"] == 0.5
    assert evidence["terminal_received_s"] is not None and evidence["stdout_eof_s"] is None
    assert evidence["teardown_completed_s"] is not None
    assert child["final_metrics"]["total_prompt_tokens"] == 10
    captured = read_records(LocalRecordStore(store)).runs
    assert len(captured) == 1
    frozen = captured[0]
    assert frozen["findings"]["value"]["terminal_coverage"]["diagnostics"] == coverage["diagnostics"]
    documents = frozen["trajectories"]["value"]["documents"]
    assert any(document.get("trajectory_id") == child["trajectory_id"]
               and evidence == document["extra"]["subtrajectories"][0]["backend_lifecycle"]
               for document in documents)
