"""Benchmark report CLI contracts, with only the external gh API substituted.

Planner/shard tests cannot detect misleading benchmark medians or treatment
identity drift. These cases protect the persisted evidence used for acceptance.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from tests.harness.fake_gh import FakeGh

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/ci_benchmark.py"
_REPO = "acme/widgets"
_SHA = "a" * 40


def _time(seconds: int) -> str:
    return (datetime(2026, 10, 19, tzinfo=timezone.utc) + timedelta(seconds=seconds)).isoformat()


def _job(name: str, start: int, end: int) -> dict[str, Any]:
    return {"name": name, "status": "completed", "conclusion": "success", "labels": ["ubuntu-24.04"],
            "started_at": _time(start), "completed_at": _time(end), "steps": []}


def _sample(run_id: int, count: int, gate: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run = {"id": run_id, "head_sha": _SHA, "head_branch": "benchmark", "event": "workflow_dispatch",
           "path": ".github/workflows/ci.yml", "repository": {"full_name": _REPO}, "run_attempt": 1,
           "html_url": f"https://github.com/{_REPO}/actions/runs/{run_id}", "created_at": _time(0),
           "status": "completed", "conclusion": "success"}
    jobs = [_job("plan", 10, 20)]
    if count != 1:
        jobs.append(_job("static-checks", 25, 45))
    durations = {1: [120], 2: [60, 40], 4: [40, 30, 20, 10]}[count]
    for index, duration in enumerate(durations):
        shard = _job("legacy-check" if count == 1 else f"test-shard ({index})", 30, 42 + duration)
        shard["steps"] = [{"name": "Run tests" if count == 1 else "Run pytest shard",
                           "started_at": _time(40), "completed_at": _time(40 + duration)}]
        jobs.append(shard)
    if count != 1:
        jobs.append(_job("coverage", 45 + max(durations), 55 + max(durations)))
    jobs.append(_job("check", gate - 2, gate))
    # A skipped job has no runner cost and must not affect queue/overlap metrics.
    skipped = _job("rl-check", 1, 999)
    skipped["conclusion"] = "skipped"
    jobs.append(skipped)
    return run, jobs


def _report(
    tmp_path: Path, fake_gh: FakeGh, samples: list[tuple[int, dict[str, Any], list[dict[str, Any]]]],
) -> tuple[subprocess.CompletedProcess[str], Path]:
    entries = []
    for count, run, jobs in samples:
        run_id = run["id"]
        endpoint = f"repos/{_REPO}/actions/runs/{run_id}"
        fake_gh.set_response("GET", endpoint, run)
        # Two pages exercise the report's actual flattening of --paginate --slurp.
        fake_gh.set_response("GET", endpoint + "/jobs", [{"jobs": jobs[:2]}, {"jobs": jobs[2:]}])
        entries.append({"id": run_id, "jobs": count, "attempt": 1})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": 1, "repository": _REPO, "ref": "benchmark", "sha": _SHA,
                                    "runs": entries}))
    output = tmp_path / "report"
    result = subprocess.run([sys.executable, str(_SCRIPT), "report", "--manifest", str(manifest),
                             "--output", str(output)], capture_output=True, text=True, check=False)
    return result, output


def test_report_measures_same_revision_treatments_and_preserves_raw_evidence(tmp_path: Path, fake_gh: FakeGh) -> None:
    samples: list[tuple[int, dict[str, Any], list[dict[str, Any]]]] = []
    for count, gates in ((1, [210, 190, 250]), (2, [120, 120, 120]), (4, [100, 100, 100])):
        for gate in gates:
            run, jobs = _sample(len(samples) + 1, count, gate)
            samples.append((count, run, jobs))
    result, output = _report(tmp_path, fake_gh, samples)
    assert result.returncode == 0, result.stderr
    report = (output / "report.md").read_text()
    assert "| 1 / 8 | 3 | 210.0 | 2.4 | 0.0% |" in report
    assert "| 2 / 4 | 3 | 120.0 | 2.8 | 42.9% |" in report
    assert "| 4 / 4 | 3 | 100.0 | 3.2 | 52.4% |" in report
    assert "Timing sample requirement: satisfied." in report
    measurements = json.loads((output / "measurements.json").read_text())["measurements"]
    two = measurements[3]
    assert two["initial_queue_seconds"] == 10
    assert two["peak_job_overlap"] == 3
    assert two["runner_minutes"] == pytest.approx(166 / 60)
    assert two["imbalance_max_over_mean"] == pytest.approx(1.2)
    assert two["shards"][0] == {"name": "test-shard (0)", "job_seconds": 72, "pytest_seconds": 60,
                                "setup_seconds": 10, "dispatch_wait_seconds": 10}
    assert measurements[6]["peak_job_overlap"] == 5
    assert measurements[6]["imbalance_max_over_mean"] == pytest.approx(1.6)
    assert json.loads((output / "run-4.json").read_text()) == {"run": samples[3][1], "jobs": samples[3][2]}


def test_report_retains_failed_and_active_runs_without_counting_them_as_successes(
    tmp_path: Path, fake_gh: FakeGh,
) -> None:
    samples: list[tuple[int, dict[str, Any], list[dict[str, Any]]]] = []
    for run_id, status, conclusion in ((1, "completed", "failure"), (2, "in_progress", None)):
        run, jobs = _sample(run_id, 4, 100)
        run.update(status=status, conclusion=conclusion)
        jobs[1].update(status=status, conclusion=conclusion)
        samples.append((4, run, jobs))
    result, output = _report(tmp_path, fake_gh, samples)
    assert result.returncode == 0, result.stderr
    measurements = json.loads((output / "measurements.json").read_text())["measurements"]
    assert [(item["conclusion"], item["complete"]) for item in measurements] == [("failure", True), (None, False)]
    assert all("gate_seconds" not in item for item in measurements)
    report = (output / "report.md").read_text()
    assert "| 4 / 4 | 0 | pending | pending | pending |" in report
    assert "4 jobs: failure." in report
    assert "4 jobs: in progress." in report
    assert "Timing sample requirement: PENDING" in report
    assert (output / "run-1.json").is_file() and (output / "run-2.json").is_file()


@pytest.mark.parametrize("fault", ["sha", "attempt", "runner"])
def test_report_rejects_different_revision_attempt_or_runner_class(
    tmp_path: Path, fake_gh: FakeGh, fault: str,
) -> None:
    run, jobs = _sample(1, 4, 100)
    if fault == "sha":
        run["head_sha"] = "b" * 40
    elif fault == "attempt":
        run["run_attempt"] = 2
    else:
        jobs[1]["labels"] = ["paid-larger-runner"]
    result, output = _report(tmp_path, fake_gh, [(4, run, jobs)])
    assert result.returncode != 0
    expected = "used a different runner class" if fault == "runner" else "does not match the benchmark identity"
    assert expected in result.stderr
    assert not (output / "report.md").exists()
