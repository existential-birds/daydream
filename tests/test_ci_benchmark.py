"""Benchmark report CLI contracts, with only the external gh boundary substituted.

Planner/shard tests cannot detect dispatch substitution, misleading medians or treatment
identity drift. These cases protect the persisted evidence used for acceptance.
"""

from __future__ import annotations

import json
import runpy
import sys
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

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



class _Gh:
    """Serve gh calls in-process: API reads, paginated job pages and workflow dispatch."""

    def __init__(self) -> None:
        self.responses: dict[str, Any] = {}
        self.dispatches: list[list[str]] = []
        self.on_runs: Callable[[], dict[str, Any]] | None = None

    def __call__(self, *args: str) -> str:
        if args[:2] == ("workflow", "run"):
            self.dispatches.append(list(args))
            return ""
        assert args[0] == "api", args
        endpoint = args[-1]
        if "/actions/workflows/ci.yml/runs?" in endpoint and self.on_runs is not None:
            return json.dumps(self.on_runs())
        return json.dumps(self.responses[endpoint.split("?", 1)[0] if "--paginate" not in args else endpoint])


@pytest.fixture
def bench(monkeypatch: pytest.MonkeyPatch) -> tuple[_Gh, Callable[..., None]]:
    namespace = runpy.run_path(str(_SCRIPT))
    gh = _Gh()
    globals_ = namespace["main"].__globals__
    monkeypatch.setitem(globals_, "gh", gh)
    # Discovery polling must not spend wall time in tests.
    monkeypatch.setitem(globals_, "time", type("Clock", (), {
        "monotonic": staticmethod(time.monotonic), "sleep": staticmethod(lambda seconds: None)}))

    def invoke(*args: str) -> None:
        monkeypatch.setattr(sys, "argv", [str(_SCRIPT), *args])
        namespace["main"]()

    return gh, invoke


def _report(
    tmp_path: Path, bench: tuple[_Gh, Callable[..., None]],
    samples: list[tuple[int, dict[str, Any], list[dict[str, Any]]]], *, dispatch_id: str | None = None,
) -> Path:
    gh, invoke = bench
    entries = []
    for count, run, jobs in samples:
        run_id = run["id"]
        endpoint = f"repos/{_REPO}/actions/runs/{run_id}"
        gh.responses[endpoint] = run
        # Two pages exercise the report's actual flattening of --paginate --slurp.
        gh.responses[endpoint + "/jobs?per_page=100"] = [{"jobs": jobs[:2]}, {"jobs": jobs[2:]}]
        entry: dict[str, Any] = {"id": run_id, "jobs": count, "attempt": 1}
        if dispatch_id is not None:
            entry["dispatch_id"] = dispatch_id
        entries.append(entry)
    tmp_path.mkdir(parents=True, exist_ok=True)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"schema": 1, "repository": _REPO, "ref": "benchmark", "sha": _SHA,
                                    "runs": entries}))
    output = tmp_path / "report"
    invoke("report", "--manifest", str(manifest), "--output", str(output))
    return output


def test_report_measures_same_revision_treatments_and_preserves_raw_evidence(
    tmp_path: Path, bench: tuple[_Gh, Callable[..., None]],
) -> None:
    samples: list[tuple[int, dict[str, Any], list[dict[str, Any]]]] = []
    for count, gates in ((1, [210, 190, 250]), (2, [120, 120, 120]), (4, [100, 100, 100])):
        for gate in gates:
            run, jobs = _sample(len(samples) + 1, count, gate)
            samples.append((count, run, jobs))
    for count, gates, durations in ((16, [110, 100, 120], [40] + [20] * 15), (20, [95, 90, 105], [30] * 20)):
        for gate in gates:
            run, jobs = _sample(len(samples) + 1, count, gate, durations=durations)
            samples.append((count, run, jobs))
    output = _report(tmp_path, bench, samples)
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
    tmp_path: Path, bench: tuple[_Gh, Callable[..., None]],
) -> None:
    """Failed/active samples cannot make a newly measured group or an original group ready."""
    for incomplete_count in (4, 20):
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
        output = _report(tmp_path / str(incomplete_count), bench, samples)
        report = (output / "report.md").read_text()
        assert "Timing sample requirement: PENDING" in report
        assert f"{incomplete_count} jobs: failure." in report and f"{incomplete_count} jobs: in progress." in report
        measurements = json.loads((output / "measurements.json").read_text())["measurements"]
        assert [(row["conclusion"], row["complete"]) for row in measurements[-2:]] == [
            ("failure", True), (None, False)]
        assert all("gate_seconds" not in row for row in measurements[-2:])
        assert (output / f"run-{len(samples)}.json").is_file()


def test_report_rejects_foreign_revision_attempt_runner_or_dispatch(
    tmp_path: Path, bench: tuple[_Gh, Callable[..., None]],
) -> None:
    run, jobs = _sample(1, 4, 100)
    run["display_title"] = f"CI benchmark {_DISPATCH_ID}"
    output = _report(tmp_path / "owned", bench, [(4, run, jobs)], dispatch_id=_DISPATCH_ID)
    assert json.loads((output / "measurements.json").read_text())["measurements"][0]["id"] == 1
    for fault in ("sha", "attempt", "runner", "foreign", "missing-title", "invalid-uuid"):
        run, jobs = _sample(1, 4, 100)
        marker = "invalid" if fault == "invalid-uuid" else _DISPATCH_ID
        if fault != "missing-title":
            run["display_title"] = "CI benchmark " + ("foreign" if fault == "foreign" else marker)
        if fault == "sha":
            run["head_sha"] = "b" * 40
        elif fault == "attempt":
            run["run_attempt"] = 2
        elif fault == "runner":
            jobs[1]["labels"] = ["paid-larger-runner"]
        dispatch = marker if fault in {"foreign", "missing-title", "invalid-uuid"} else None
        expected = {"runner": "used a different runner class", "sha": "does not match the benchmark identity",
                    "attempt": "does not match the benchmark identity"}.get(fault, "dispatch")
        with pytest.raises(ValueError, match=expected):
            _report(tmp_path / fault, bench, [(4, run, jobs)], dispatch_id=dispatch)
        # Raw evidence is retained; no report is published for a rejected treatment.
        assert (tmp_path / fault / "report/run-1.json").exists()
        assert not (tmp_path / fault / "report/report.md").exists()


def test_launch_ignores_foreign_dispatch_until_its_persisted_marker_is_visible(
    tmp_path: Path, bench: tuple[_Gh, Callable[..., None]],
) -> None:
    """Same-SHA, same-size treatments must still belong to this invocation."""
    gh, invoke = bench
    manifest = tmp_path / "manifest.json"
    gh.responses[f"repos/{_REPO}/commits/benchmark"] = {"sha": _SHA}
    foreign = {"id": 222, "head_sha": _SHA, "run_attempt": 1, "html_url": "https://github.test/222",
               "display_title": "CI benchmark somebody-else"}
    polls: list[dict[str, Any]] = []

    def runs() -> dict[str, Any]:
        if not gh.dispatches:
            return {"workflow_runs": []}
        items = [foreign]
        if polls:
            # The persisted pending marker must already name the dispatched UUID.
            marker = json.loads(manifest.read_text())["pending"]["dispatch_id"]
            items.append(dict(foreign, id=111, html_url="https://github.test/111",
                              display_title="CI benchmark " + marker))
        polls.append({})
        return {"workflow_runs": items}

    gh.on_runs = runs
    invoke("launch", "--repo", _REPO, "--ref", "benchmark", "--sha", _SHA, "--jobs", "20",
           "--manifest", str(manifest))
    recorded = json.loads(manifest.read_text())
    sent, = gh.dispatches
    marker = recorded["runs"][0]["dispatch_id"]
    assert str(UUID(marker)) == marker and UUID(marker).version == 4
    assert f"benchmark_id={marker}" in sent and "test_jobs=20" in sent
    assert recorded["runs"] == [{"id": 111, "jobs": 20, "attempt": 1, "dispatch_id": marker}]
    assert "pending" not in recorded


def test_resolve_records_only_the_proven_owned_dispatch(
    tmp_path: Path, bench: tuple[_Gh, Callable[..., None]],
) -> None:
    gh, invoke = bench
    candidates: list[dict[str, Any]] = []
    gh.on_runs = lambda: {"workflow_runs": candidates}
    for scenario in ("foreign", "missing-title", "explicit-foreign", "ambiguous", "legacy-pending", "owned"):
        pending: dict[str, Any] = {"jobs": 4, "before": [], "dispatch_id": _DISPATCH_ID}
        run, _ = _sample(1, 4, 100)
        run["display_title"] = f"CI benchmark {_DISPATCH_ID}"
        if scenario in ("foreign", "explicit-foreign"):
            run["display_title"] = "CI benchmark cc40ab14-9829-457d-9916-2b858e645c61"
        elif scenario == "missing-title":
            del run["display_title"]
        elif scenario == "legacy-pending":
            del pending["dispatch_id"]
        candidates[:] = [run, {**run, "id": 2}] if scenario == "ambiguous" else [run]
        manifest = tmp_path / f"{scenario}.json"
        original = json.dumps({"repository": _REPO, "ref": "benchmark", "sha": _SHA, "runs": [],
                               "pending": pending})
        manifest.write_text(original)
        command = ["resolve", "--manifest", str(manifest)]
        if scenario in ("explicit-foreign", "owned"):
            command += ["--run-id", "1"]
        if scenario == "owned":
            invoke(*command)
            resolved = json.loads(manifest.read_text())
            assert resolved["runs"] == [{"id": 1, "jobs": 4, "attempt": 1, "dispatch_id": _DISPATCH_ID}]
            assert "pending" not in resolved
            continue
        with pytest.raises(ValueError, match="dispatch"):
            invoke(*command)
        assert manifest.read_text() == original, scenario
