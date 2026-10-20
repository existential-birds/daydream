"""Benchmark report CLI contracts, with only the external gh API substituted.

Planner/shard tests cannot detect dispatch substitution, misleading medians or treatment
identity drift. These cases protect the persisted evidence used for acceptance.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from tests.harness.fake_gh import FakeGh

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/ci_benchmark.py"
_REPO = "acme/widgets"
_SHA = "a" * 40
_DISPATCH_ID = "cc40ab14-9829-457d-9916-2b858e645c60"


def _time(seconds: int) -> str:
    return (datetime(2026, 10, 19, tzinfo=timezone.utc) + timedelta(seconds=seconds)).isoformat()


def _job(name: str, start: int, end: int) -> dict[str, Any]:
    return {"name": name, "status": "completed", "conclusion": "success", "labels": ["ubuntu-24.04"],
            "started_at": _time(start), "completed_at": _time(end), "steps": []}


def _sample(
    run_id: int, count: int, gate: int, *, durations: list[int] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run = {"id": run_id, "head_sha": _SHA, "head_branch": "benchmark", "event": "workflow_dispatch",
           "path": ".github/workflows/ci.yml", "repository": {"full_name": _REPO}, "run_attempt": 1,
           "html_url": f"https://github.com/{_REPO}/actions/runs/{run_id}", "created_at": _time(0),
           "status": "completed", "conclusion": "success"}
    jobs = [_job("plan", 10, 20)]
    if count != 1:
        jobs.append(_job("static-checks", 25, 45))
    if durations is None:
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
    *, dispatch_id: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    entries = []
    for count, run, jobs in samples:
        run_id = run["id"]
        endpoint = f"repos/{_REPO}/actions/runs/{run_id}"
        fake_gh.set_response("GET", endpoint, run)
        # Two pages exercise the report's actual flattening of --paginate --slurp.
        fake_gh.set_response("GET", endpoint + "/jobs", [{"jobs": jobs[:2]}, {"jobs": jobs[2:]}])
        entry: dict[str, Any] = {"id": run_id, "jobs": count, "attempt": 1}
        if dispatch_id is not None:
            entry["dispatch_id"] = dispatch_id
        entries.append(entry)
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
    for count, gates, durations in ((16, [110, 100, 120], [40] + [20] * 15), (20, [95, 90, 105], [30] * 20)):
        for gate in gates:
            run, jobs = _sample(len(samples) + 1, count, gate, durations=durations)
            samples.append((count, run, jobs))
    result, output = _report(tmp_path, fake_gh, samples)
    assert result.returncode == 0, result.stderr
    report = (output / "report.md").read_text()
    assert "| 1 / 8 | 3 | 210.0 | 2.4 | 0.0% |" in report
    assert "| 2 / 4 | 3 | 120.0 | 2.8 | 42.9% |" in report
    assert "| 4 / 4 | 3 | 100.0 | 3.2 | 52.4% |" in report
    assert "| 16 / 4 | 3 | 110.0 | 9.6 | 47.6% |" in report
    assert "| 20 / 4 | 3 | 95.0 | 14.7 | 54.8% |" in report
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
    sixteen, twenty = measurements[9], measurements[12]
    assert len(sixteen["shards"]) == 16 and len(twenty["shards"]) == 20
    assert sixteen["runner_minutes"] == pytest.approx(574 / 60)
    assert twenty["runner_minutes"] == pytest.approx(882 / 60)
    assert sixteen["shards"][0]["job_seconds"] == 52
    assert sixteen["shards"][0]["pytest_seconds"] == 40
    assert sixteen["peak_job_overlap"] == 17 and twenty["peak_job_overlap"] == 21
    assert sixteen["imbalance_max_over_mean"] == pytest.approx(40 / 21.25)
    assert json.loads((output / "run-10.json").read_text()) == {"run": samples[9][1], "jobs": samples[9][2]}


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


@pytest.mark.parametrize("count", [4, 8, 20], ids=["legacy", "default-treatment", "maximum-treatment"])
def test_launch_ignores_foreign_dispatch_until_its_persisted_marker_is_visible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, count: int,
) -> None:
    """Same-SHA, same-size treatments must still belong to this invocation."""
    manifest = tmp_path / "manifest.json"
    receipt = tmp_path / "dispatch.json"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    shim = bin_dir / "gh"
    shim.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, sys\n"
        f"manifest = pathlib.Path({str(manifest)!r})\n"
        f"receipt = pathlib.Path({str(receipt)!r})\n"
        "args = sys.argv[1:]\n"
        "if args[:2] == ['workflow', 'run']:\n"
        "    receipt.write_text(json.dumps({'args': args, 'pending': json.loads(manifest.read_text())['pending']}))\n"
        "elif args[0] == 'api' and '/commits/' in args[-1]:\n"
        f"    print(json.dumps({{'sha': {_SHA!r}}}))\n"
        "elif args[0] == 'api' and '/runs?' in args[-1]:\n"
        "    items = []\n"
        "    if receipt.exists():\n"
        "        sent = json.loads(receipt.read_text())\n"
        f"        foreign = {{'id': 222, 'head_sha': {_SHA!r}, 'run_attempt': 1,\n"
        "                   'html_url': 'https://github.test/222',\n"
        "                   'display_title': 'CI benchmark somebody-else'}\n"
        "        items = [foreign]\n"
        "        polls = receipt.with_suffix('.polls')\n"
        "        if polls.exists():\n"
        "            fields = [arg.split('=', 1)[1] for arg in sent['args'] if arg.startswith('benchmark_id=')]\n"
        "            marker = fields[0] if fields else ''\n"
        "            items.append(dict(foreign, id=111, html_url='https://github.test/111',\n"
        "                              display_title='CI benchmark ' + marker))\n"
        "        polls.touch()\n"
        "    print(json.dumps({'workflow_runs': items}))\n"
        "else:\n"
        "    raise SystemExit('unexpected gh call: ' + repr(args))\n"
    )
    shim.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ["PATH"])
    result = subprocess.run(
        [sys.executable, str(_SCRIPT), "launch", "--repo", _REPO, "--ref", "benchmark", "--sha", _SHA,
         "--jobs", str(count), "--manifest", str(manifest)], capture_output=True, text=True, check=False, timeout=15,
    )
    assert result.returncode == 0, result.stderr
    recorded = json.loads(manifest.read_text())
    assert [entry["id"] for entry in recorded["runs"]] == [111]
    sent = json.loads(receipt.read_text())
    marker = sent["pending"]["dispatch_id"]
    assert str(UUID(marker)) == marker and UUID(marker).version == 4
    assert f"benchmark_id={marker}" in sent["args"]
    assert f"test_jobs={count}" in sent["args"]
    assert recorded["runs"] == [{"id": 111, "jobs": count, "attempt": 1, "dispatch_id": marker}]
    assert "pending" not in recorded


@pytest.mark.parametrize("scenario", ["foreign", "missing-title", "explicit-foreign", "ambiguous", "legacy-pending"])
def test_resolve_preserves_pending_when_dispatch_identity_is_not_proven(
    tmp_path: Path, fake_gh: FakeGh, scenario: str,
) -> None:
    pending: dict[str, Any] = {"jobs": 4, "before": [], "dispatch_id": _DISPATCH_ID}
    run, _ = _sample(1, 4, 100)
    run["display_title"] = f"CI benchmark {_DISPATCH_ID}"
    if scenario in ("foreign", "explicit-foreign"):
        run["display_title"] = "CI benchmark cc40ab14-9829-457d-9916-2b858e645c61"
    elif scenario == "missing-title":
        del run["display_title"]
    elif scenario == "legacy-pending":
        del pending["dispatch_id"]
    candidates = [run, {**run, "id": 2}] if scenario == "ambiguous" else [run]
    fake_gh.set_response("GET", f"repos/{_REPO}/actions/workflows/ci.yml/runs", {"workflow_runs": candidates})
    manifest = tmp_path / "manifest.json"
    original = json.dumps({"repository": _REPO, "ref": "benchmark", "sha": _SHA, "runs": [], "pending": pending})
    manifest.write_text(original)
    command = [sys.executable, str(_SCRIPT), "resolve", "--manifest", str(manifest)]
    if scenario == "explicit-foreign":
        command += ["--run-id", "1"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "dispatch" in result.stderr
    assert manifest.read_text() == original


def test_resolve_explicit_run_still_accepts_the_owned_dispatch(tmp_path: Path, fake_gh: FakeGh) -> None:
    run, _ = _sample(1, 4, 100)
    run["display_title"] = f"CI benchmark {_DISPATCH_ID}"
    fake_gh.set_response("GET", f"repos/{_REPO}/actions/workflows/ci.yml/runs", {"workflow_runs": [run]})
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"repository": _REPO, "ref": "benchmark", "sha": _SHA, "runs": [],
                                    "pending": {"jobs": 4, "before": [], "dispatch_id": _DISPATCH_ID}}))
    result = subprocess.run([sys.executable, str(_SCRIPT), "resolve", "--manifest", str(manifest), "--run-id", "1"],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    resolved = json.loads(manifest.read_text())
    assert resolved["runs"] == [{"id": 1, "jobs": 4, "attempt": 1, "dispatch_id": _DISPATCH_ID}]
    assert "pending" not in resolved


@pytest.mark.parametrize("fault", [None, "foreign", "missing-title", "invalid-uuid"])
def test_report_requires_recorded_dispatch_identity_for_marked_entries(
    tmp_path: Path, fake_gh: FakeGh, fault: str | None,
) -> None:
    run, jobs = _sample(1, 4, 100)
    marker = "invalid" if fault == "invalid-uuid" else _DISPATCH_ID
    if fault != "missing-title":
        run["display_title"] = "CI benchmark " + ("foreign" if fault == "foreign" else marker)
    result, output = _report(tmp_path, fake_gh, [(4, run, jobs)], dispatch_id=marker)
    if fault is None:
        assert result.returncode == 0, result.stderr
        assert json.loads((output / "measurements.json").read_text())["measurements"][0]["id"] == 1
    else:
        assert result.returncode != 0
        assert "dispatch" in result.stderr
        assert (output / "run-1.json").exists()
        assert not (output / "report.md").exists()


@pytest.mark.parametrize("incomplete_count", [4, 20], ids=["original-configuration", "additional-configuration"])
def test_report_requires_three_successes_for_original_and_additional_configurations(
    tmp_path: Path, fake_gh: FakeGh, incomplete_count: int,
) -> None:
    """Failed/active samples cannot make a newly measured group or an original group ready."""
    samples: list[tuple[int, dict[str, Any], list[dict[str, Any]]]] = []
    for count in (1, 2, 4, 16, 20):
        for gate in ([100, 120] if count == incomplete_count else [190, 210, 250]):
            durations = [40] + [20] * 15 if count == 16 else [30] * 20 if count == 20 else None
            run, jobs = _sample(len(samples) + 1, count, gate, durations=durations)
            samples.append((count, run, jobs))
    for status, conclusion in (("completed", "failure"), ("in_progress", None)):
        durations = [30] * 20 if incomplete_count == 20 else None
        run, jobs = _sample(len(samples) + 1, incomplete_count, 120, durations=durations)
        run.update(status=status, conclusion=conclusion)
        jobs[2].update(status=status, conclusion=conclusion)
        samples.append((incomplete_count, run, jobs))
    result, output = _report(tmp_path, fake_gh, samples)
    assert result.returncode == 0, result.stderr
    report = (output / "report.md").read_text()
    assert "Timing sample requirement: PENDING" in report
    measurements = json.loads((output / "measurements.json").read_text())["measurements"]
    assert [(row["conclusion"], row["complete"]) for row in measurements[-2:]] == [("failure", True), (None, False)]
    assert all("gate_seconds" not in row for row in measurements[-2:])
