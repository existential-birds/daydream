"""Real runner coverage with opaque reviews and independently resolved sweep backends."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from daydream.backends import AgentEvent, ToolResultEvent, ToolStartEvent
from daydream.config_file import DaydreamFileConfig
from daydream.eval.analyzer import analyze_coverage, load_trajectories
from daydream.runner import run
from tests.deep_orchestrator.support import _uncovered_sweep_target
from tests.harness.stub_backend import StubBackend
from tests.test_deep_orchestrator import MakeConfig


class CodexBackend(StubBackend):
    """Expose aggregate successful shell results after an early loop read failure."""

    async def execute(self, cwd: Path, prompt: str, *args: Any, **kwargs: Any) -> AsyncIterator[AgentEvent]:
        async for event in super().execute(cwd, prompt, *args, **kwargs):
            if isinstance(event, ToolStartEvent) and event.name == "Read":
                path = event.input["file_path"]
                event = ToolStartEvent(
                    id=event.id, name="Bash",
                    input={"command": f'for task_file in missing.py {path}; do nl -ba "$task_file"; done'},
                )
            elif isinstance(event, ToolResultEvent):
                event = ToolResultEvent(
                    id=event.id, output="nl: missing.py: No such file or directory\nfile content",
                    is_error=False, exit_code=0, status="completed",
                )
            yield event


class ClaudeBackend(StubBackend):
    """Emit successful paired native reads through the same provider seam."""


@pytest.mark.parametrize("review_backend", ["codex", "claude"])
async def test_runner_uses_review_evidence_independently_of_the_sweep_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, make_config: MakeConfig,
    capsys: pytest.CaptureFixture[str], review_backend: str,
) -> None:
    target = _uncovered_sweep_target(tmp_path)
    backends = {"codex": CodexBackend(target), "claude": ClaudeBackend(target)}
    for backend in backends.values():
        backend.per_stack_emit_reads = True
        backend.per_stack_unread = frozenset({"notes.txt"})
        backend.merge_echo_records = True
        backend.parse_declared_verdicts = [{"path": "README.md", "lines_read": 3, "verdict": "clean", "n_findings": 0}]
    monkeypatch.setattr("daydream.runner.create_backend", lambda name, **kwargs: backends[name])
    config = make_config(
        target, output_mode="review", assume="yes",
        file_config=DaydreamFileConfig(phases={
            "per_stack_review": {"backend": review_backend},
            "parse": {"backend": "claude" if review_backend == "codex" else "codex"},
        }),
    )
    assert await run(config) == 0
    deep = target / ".daydream" / "deep"
    stats = json.loads((deep / "coverage-stats.json").read_text())
    records = json.loads((deep / "stack-generic-records.json").read_text())
    readme_verdict = next(v for v in records["verdicts"] if v["path"] == "README.md")
    merged = json.loads((deep / "merged-items.json").read_text())["items"]
    assert any(item["description"] == "Sample issue" for item in merged)
    report = (target / ".review-output.md").read_text()
    output = " ".join(capsys.readouterr().out.split())
    evaluation = analyze_coverage(load_trajectories(target / ".daydream"), target / ".daydream")

    if review_backend == "codex":
        assert stats["pre_sweep"]["coverage_status"] == "unverifiable"
        assert stats["pre_sweep"]["coverage_ratio"] is None
        assert stats["pre_sweep"]["files_read_by_reviewers"] is None
        assert stats["pre_sweep"]["uncovered_files"] is None
        assert stats["post_sweep"]["coverage_ratio"] is None
        assert stats["sweep_unavailable"] is True
        assert stats["attempted_files"] == []
        assert not (deep / "stack-uncovered-records.json").exists()
        assert readme_verdict["verdict"] == "unknown"
        assert readme_verdict["lines_read"] is None
        assert evaluation["coverage_ratio"] is None
        assert "Source-read coverage: unverifiable" in report
        assert "Coverage-targeted catch-up unavailable" in report
        assert "Coverage ratio:" not in report
        assert "Files read by reviewers: 0" not in report
        assert "unverifiable" in output.lower()
        for label in stats["pre_sweep"]["unverifiable_agents"]:
            assert output.count(f"Source-read coverage unverifiable for {label};") == 1
    else:
        assert stats["pre_sweep"]["coverage_ratio"] == 0.75
        assert stats["pre_sweep"]["uncovered_files"] == ["notes.txt"]
        assert stats["attempted_files"] == ["notes.txt"]
        assert readme_verdict["verdict"] == "clean"
        sweep = json.loads((deep / "stack-uncovered-records.json").read_text())
        assert sweep[0]["file"] == "notes.txt"
        assert any(item["file"] == "notes.txt" for item in merged)
        assert stats["post_sweep"]["coverage_ratio"] is None
        assert evaluation["coverage_ratio"] is None
        assert "notes.txt" in evaluation["unverifiable_files"]
        assert {"api.py", "App.tsx", "README.md"} <= set(evaluation["verified_files"])
        assert "Source-read coverage: unverifiable" in report
        assert "Files with verified source evidence: 3" in report
        assert "Second-pass sweep completed without verified source read: notes.txt" in report
        assert "Coverage-targeted catch-up unavailable" not in report
        assert "Coverage ratio:" not in report
