#!/usr/bin/env python3
"""Dispatch and measure same-revision CI configurations using authenticated gh.

Launch never waits for a gate to finish. Report preserves raw GitHub metadata
and includes incomplete/failed runs instead of silently dropping them.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from uuid import UUID, uuid4


def gh(*args: str) -> str:
    return subprocess.check_output(["gh", *args], text=True, timeout=60)


def api(endpoint: str) -> Any:
    return json.loads(gh("api", endpoint))


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def runs(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    query = urlencode({"branch": manifest["ref"], "event": "workflow_dispatch", "per_page": 100})
    result = api(f"repos/{manifest['repository']}/actions/workflows/ci.yml/runs?{query}")
    return [run for run in result["workflow_runs"] if run["head_sha"] == manifest["sha"]]


def dispatch_title(dispatch_id: Any) -> str:
    if not isinstance(dispatch_id, str):
        raise ValueError("Missing benchmark dispatch UUID; cannot prove dispatch identity")
    try:
        valid = str(UUID(dispatch_id)) == dispatch_id
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("Invalid benchmark dispatch UUID")
    return f"CI benchmark {dispatch_id}"


def resolve(manifest: dict[str, Any], path: Path, run_id: int | None = None) -> None:
    pending = manifest.get("pending")
    if pending is None:
        raise ValueError("No unresolved dispatch")
    title = dispatch_title(pending.get("dispatch_id"))
    candidates = [run for run in runs(manifest) if run["id"] not in pending["before"] and
                  run.get("display_title") == title]
    if run_id is not None:
        candidates = [run for run in candidates if run["id"] == run_id]
    if len(candidates) != 1:
        raise ValueError(f"Expected one new matching dispatch, found {len(candidates)}; retry resolve --run-id ID")
    run = candidates[0]
    manifest["runs"].append({"id": run["id"], "jobs": pending["jobs"], "attempt": run["run_attempt"],
                             "dispatch_id": pending["dispatch_id"]})
    del manifest["pending"]
    save(path, manifest)
    print(run["html_url"])


def launch(args: argparse.Namespace) -> None:
    if args.manifest.exists():
        manifest = json.loads(args.manifest.read_text())
        if (manifest["repository"], manifest["ref"], manifest["sha"]) != (args.repo, args.ref, args.sha):
            raise ValueError("Manifest belongs to another repository/ref/revision")
    else:
        manifest = {"schema": 1, "repository": args.repo, "ref": args.ref, "sha": args.sha, "runs": []}
    if manifest.get("pending"):
        raise ValueError("Resolve the previous dispatch before launching another run")
    # Resolve the remote branch immediately before each dispatch. A moved branch
    # must never silently create a different-revision benchmark treatment.
    for _ in range(args.repeat):
        branch = api(f"repos/{args.repo}/commits/{args.ref}")
        if branch["sha"] != args.sha:
            raise ValueError("Remote ref moved from the requested benchmark SHA")
        before = [run["id"] for run in runs(manifest)]
        dispatch_id = str(uuid4())
        manifest["pending"] = {"jobs": args.jobs, "before": before, "dispatch_id": dispatch_id}
        save(args.manifest, manifest)
        gh("workflow", "run", "ci.yml", "--repo", args.repo, "--ref", args.ref,
           "-f", f"test_jobs={args.jobs}", "-f", f"benchmark_id={dispatch_id}")
        # Discovery only: this bounded wait never waits for pytest or CI.
        deadline = time.monotonic() + 30
        while True:
            try:
                resolve(manifest, args.manifest)
                break
            except ValueError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(2)


def timestamp(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def elapsed(item: dict[str, Any]) -> float:
    return max(0.0, timestamp(item["completed_at"]) - timestamp(item["started_at"]))


def overlap(jobs: list[dict[str, Any]]) -> int:
    events = [(timestamp(job[key]), change) for job in jobs for key, change in
              (("started_at", 1), ("completed_at", -1)) if job.get(key) and job["conclusion"] != "skipped"]
    active = peak = 0
    for _, change in sorted(events):
        active += change
        peak = max(peak, active)
    return peak


def measure(run: dict[str, Any], jobs: list[dict[str, Any]], count: int) -> dict[str, Any]:
    shards = [job for job in jobs if (job["name"] == "legacy-check" if count == 1 else
                                    job["name"].startswith("test-shard"))]
    checks = [job for job in jobs if job["name"] == "check"]
    live = [job for job in jobs if job["conclusion"] != "skipped"]
    complete = run["status"] == "completed" and all(job["status"] == "completed" for job in live)
    result: dict[str, Any] = {"id": run["id"], "jobs": count, "url": run["html_url"],
                              "conclusion": run["conclusion"], "complete": complete}
    if not complete or run["conclusion"] != "success":
        return result
    if len(shards) != count or len(checks) != 1:
        raise ValueError(f"Run {run['id']} has unexpected shard/final gate jobs")
    if any("ubuntu-24.04" not in job["labels"] for job in live):
        raise ValueError(f"Run {run['id']} used a different runner class")
    shard_metrics = []
    plan = next(job for job in jobs if job["name"] == "plan")
    for shard in shards:
        test_name = "Run tests" if count == 1 else "Run pytest shard"
        tests = [step for step in shard["steps"] if step["name"] == test_name]
        if len(tests) != 1:
            raise ValueError(f"Run {run['id']} missing pytest step in {shard['name']}")
        test = tests[0]
        shard_metrics.append({"name": shard["name"], "job_seconds": elapsed(shard),
                              "pytest_seconds": elapsed(test),
                              "setup_seconds": timestamp(test["started_at"]) - timestamp(shard["started_at"]),
                              "dispatch_wait_seconds": max(0, timestamp(shard["started_at"]) -
                                                           timestamp(plan["completed_at"]))})
    durations = [shard["pytest_seconds"] for shard in shard_metrics]
    result.update({"gate_seconds": timestamp(checks[0]["completed_at"]) - timestamp(run["created_at"]),
                   "initial_queue_seconds": min(timestamp(job["started_at"]) for job in live) -
                                            timestamp(run["created_at"]),
                   "runner_minutes": sum(elapsed(job) for job in live) / 60,
                   "peak_job_overlap": overlap(live), "shards": shard_metrics,
                   "imbalance_max_over_mean": max(durations) / statistics.mean(durations) if any(durations) else 0})
    return result


def report(args: argparse.Namespace) -> None:
    manifest = json.loads(args.manifest.read_text())
    measurements = []
    for entry in manifest["runs"]:
        run_id = entry["id"]
        run = api(f"repos/{manifest['repository']}/actions/runs/{run_id}")
        pages = json.loads(gh("api", "--paginate", "--slurp",
                             f"repos/{manifest['repository']}/actions/runs/{run_id}/jobs?per_page=100"))
        jobs = [job for page in pages for job in page["jobs"]]
        save(args.output / f"run-{run_id}.json", {"run": run, "jobs": jobs})
        if (run["head_sha"] != manifest["sha"] or run["head_branch"] != manifest["ref"] or
                run["event"] != "workflow_dispatch" or run["path"] != ".github/workflows/ci.yml" or
                run["repository"]["full_name"] != manifest["repository"] or run["run_attempt"] != entry["attempt"]):
            raise ValueError(f"Run {run_id} does not match the benchmark identity")
        if "dispatch_id" in entry and run.get("display_title") != dispatch_title(entry["dispatch_id"]):
            raise ValueError(f"Run {run_id} does not match the benchmark dispatch identity")
        measurements.append(measure(run, jobs, entry["jobs"]))
    save(args.output / "measurements.json", {"manifest": manifest, "measurements": measurements})
    lines = ["# CI sharding benchmark", "", f"Revision: `{manifest['sha']}` in `{manifest['repository']}`.", "",
             "| Jobs / workers | Successful runs | Median gate seconds | Median runner minutes | Reduction |",
             "|---|---:|---:|---:|---:|"]
    counts = sorted({1, 2, 4} | {m["jobs"] for m in measurements})
    groups = {count: [m for m in measurements if m["jobs"] == count and m["complete"] and
                     m["conclusion"] == "success"] for count in counts}
    baseline = statistics.median(m["gate_seconds"] for m in groups[1]) if groups[1] else None
    for count, group in groups.items():
        gate = statistics.median(m["gate_seconds"] for m in group) if group else None
        minutes = statistics.median(m["runner_minutes"] for m in group) if group else None
        reduction = f"{100 * (1 - gate / baseline):.1f}%" if gate is not None and baseline else "pending"
        lines.append(f"| {count} / {8 if count == 1 else 4} | {len(group)} | "
                     f"{f'{gate:.1f}' if gate is not None else 'pending'} | "
                     f"{f'{minutes:.1f}' if minutes is not None else 'pending'} | {reduction} |")
    lines += ["", "## Individual runs", ""]
    for m in measurements:
        lines.append(f"- [{m['id']}]({m['url']}), {m['jobs']} jobs: {m['conclusion'] or 'in progress'}.")
        if "gate_seconds" not in m:
            continue
        lines.append(f"  Gate {m['gate_seconds']:.1f}s; initial queue {m['initial_queue_seconds']:.1f}s; "
                     f"runner minutes {m['runner_minutes']:.1f}; observed peak overlap {m['peak_job_overlap']}; "
                     f"pytest max/mean imbalance {m['imbalance_max_over_mean']:.3f}.")
        for shard in m["shards"]:
            lines.append(f"  {shard['name']}: pytest {shard['pytest_seconds']:.1f}s; "
                         f"setup {shard['setup_seconds']:.1f}s; dispatch wait {shard['dispatch_wait_seconds']:.1f}s; "
                         f"job {shard['job_seconds']:.1f}s.")
    ready = not manifest.get("pending") and all(len(group) >= 3 for group in groups.values())
    sample_status = "satisfied." if ready else "PENDING (three successes per configuration)."
    lines += ["", "Timing sample requirement: " + sample_status,
              "", "Raw metadata and metrics accompany this report. Validate selection, skips, coverage and runner "
              "hardware from artifacts/logs before treating timing as acceptance evidence. Peak overlap is observed "
              "utilization, not proof of the account's configured concurrency limit."]
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    print(args.output / "report.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    dispatch = sub.add_parser("launch")
    dispatch.add_argument("--repo", required=True)
    dispatch.add_argument("--ref", required=True)
    dispatch.add_argument("--sha", required=True)
    dispatch.add_argument("--jobs", type=int, choices=range(1, 21), required=True)
    dispatch.add_argument("--repeat", type=int, choices=(1, 2, 3), default=1)
    pending = sub.add_parser("resolve")
    pending.add_argument("--run-id", type=int)
    summary = sub.add_parser("report")
    summary.add_argument("--output", type=Path, required=True)
    for command in (dispatch, pending, summary):
        command.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "launch":
        launch(args)
    elif args.command == "resolve":
        resolve(json.loads(args.manifest.read_text()), args.manifest, args.run_id)
    else:
        report(args)


if __name__ == "__main__":
    main()
