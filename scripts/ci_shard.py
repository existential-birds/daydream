"""Run deterministic pytest shards and reject incomplete CI gate evidence.

Loaded as a pytest plugin only by the ``run`` subcommand. Collection is sorted
and dealt round-robin; every worker independently verifies the same selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_CONFIG_ENV = "DAYDREAM_CI_SHARD"
_STATE: dict[str, Any] = {}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity() -> dict[str, str]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], check=False).returncode != 0:
        raise ValueError("tracked checkout differs from HEAD")
    tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], text=True).strip()
    if os.environ.get("GITHUB_SHA", head) != head:
        raise ValueError("checkout HEAD differs from GITHUB_SHA")
    return {"head": head, "tree": tree, "run_id": os.environ.get("GITHUB_RUN_ID", "local"),
            "run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", "local")}


def _selection(collection: list[str], index: int, count: int) -> list[str]:
    if count < 1 or not 0 <= index < count:
        raise ValueError("invalid shard index/count")
    if not collection or collection != sorted(set(collection)):
        raise ValueError("collection must be nonempty, sorted, and unique")
    selected = collection[index::count]
    if not selected:
        raise ValueError("empty shard selection")
    return selected


def pytest_configure(config: Any) -> None:
    if _CONFIG_ENV not in os.environ:
        return
    _STATE.clear()
    _STATE.update(json.loads(os.environ[_CONFIG_ENV]))
    _STATE.update(collections=[], reports={}, errors=[])


def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    if not _STATE:
        return
    collection = sorted(item.nodeid for item in items)
    selected = _selection(collection, _STATE["index"], _STATE["count"])
    if hasattr(config, "workeroutput"):
        config.workeroutput["ci_collection"] = collection
    else:
        _STATE["collections"].append(collection)
    admitted = set(selected)
    items[:] = sorted((item for item in items if item.nodeid in admitted), key=lambda item: item.nodeid)


def pytest_testnodedown(node: Any, error: Any) -> None:
    if not _STATE:
        return
    if error is not None:
        _STATE["errors"].append(str(error))
    collection = node.workeroutput.get("ci_collection")
    if not isinstance(collection, list):
        _STATE["errors"].append("worker did not return its full collection")
    else:
        _STATE["collections"].append(collection)


def pytest_runtest_logreport(report: Any) -> None:
    if not _STATE:
        return
    phases = _STATE["reports"].setdefault(report.nodeid, {})
    if report.when in phases:
        _STATE["errors"].append(f"duplicate {report.when} report: {report.nodeid}")
    phases[report.when] = {"outcome": report.outcome, "seconds": report.duration}


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    if not _STATE or hasattr(session.config, "workerinput"):
        return
    receipt = {key: _STATE[key] for key in ("collections", "reports", "errors")}
    receipt["exitstatus"] = int(exitstatus)
    Path(_STATE["receipt"]).write_text(json.dumps(receipt), encoding="utf-8")


def _execution(
    manifest: dict[str, Any], count: int, index: int, *, allow_failure: bool = False,
) -> list[str]:
    collection = manifest["collection"]
    if not isinstance(collection, list) or any(not isinstance(item, str) for item in collection):
        raise ValueError("invalid collection")
    selected = _selection(collection, index, count)
    if manifest["selected"] != selected:
        raise ValueError("inconsistent shard selection")
    if not allow_failure and (manifest["exitstatus"] != 0 or manifest["errors"]):
        raise ValueError("failed shard execution")
    reports = manifest["reports"]
    if not isinstance(reports, dict) or set(reports) != set(selected):
        raise ValueError("execution does not match shard selection")
    for nodeid, phases in reports.items():
        if not isinstance(phases, dict) or not {"setup", "teardown"} <= phases.keys():
            raise ValueError(f"incomplete execution: {nodeid}")
        if set(phases) - {"setup", "call", "teardown"}:
            raise ValueError(f"unexpected execution phase: {nodeid}")
        for phase in phases.values():
            outcomes = {"passed", "skipped", "failed"} if allow_failure else {"passed", "skipped"}
            if phase["outcome"] not in outcomes:
                raise ValueError(f"unsuccessful execution: {nodeid}")
            seconds = phase["seconds"]
            if not isinstance(seconds, (int, float)) or not 0 <= seconds < float("inf"):
                raise ValueError(f"invalid duration: {nodeid}")
        if phases["setup"]["outcome"] == "passed" and "call" not in phases:
            raise ValueError(f"missing test call: {nodeid}")
    return collection


def _run(args: argparse.Namespace) -> None:
    identity = _identity()
    if args.workers < 1 or args.count < 1 or not 0 <= args.index < args.count:
        raise ValueError("invalid shard index/count/workers")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if list(output.iterdir()):
        raise ValueError("shard output directory must be empty")
    receipt = output / "execution.json"
    raw_coverage = output / ".coverage"
    config = {"index": args.index, "count": args.count, "receipt": str(receipt)}
    env = {**os.environ, _CONFIG_ENV: json.dumps(config), "COVERAGE_FILE": str(raw_coverage)}
    started = time.monotonic()
    result = subprocess.run([
        sys.executable, "-m", "pytest", "-p", "scripts.ci_shard", "-n", str(args.workers),
        "--cov", "--cov-branch", "--cov-report=", "--cov-fail-under=0",
    ], env=env, check=False)
    elapsed = time.monotonic() - started
    if _identity() != identity:
        raise ValueError("checkout identity changed while pytest was running")
    if not receipt.is_file():
        raise ValueError("pytest did not publish an execution receipt")
    evidence = json.loads(receipt.read_text(encoding="utf-8"))
    collections = evidence.pop("collections")
    if not collections or any(collection != collections[0] for collection in collections):
        raise ValueError("workers did not agree on the full collection")
    if result.returncode != evidence["exitstatus"]:
        raise ValueError("pytest process and receipt exit status differ")
    manifest = {"schema": 1, **identity, "index": args.index, "count": args.count, "workers": args.workers,
                "collection": collections[0], "selected": collections[0][args.index::args.count],
                "elapsed_seconds": elapsed, **evidence}
    # Preserve complete failed executions for diagnostic coverage without admitting them as success.
    _execution(manifest, args.count, args.index, allow_failure=True)
    if not raw_coverage.is_file():
        raise ValueError("pytest did not produce coverage data")
    coverage = output / "coverage"
    raw_coverage.rename(coverage)
    manifest["coverage_sha256"] = _digest(coverage)
    (output / "manifest.json").write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    receipt.unlink()
    _execution(manifest, args.count, args.index)


def _combine(args: argparse.Namespace) -> None:
    from coverage import Coverage, CoverageData
    from coverage.exceptions import CoverageException

    identity = _identity()
    # A failed validation must not leave a previous report available for publication.
    Path("coverage.xml").unlink(missing_ok=True)
    root = args.input.resolve()
    if args.count < 1 or {path.name for path in root.iterdir()} != {f"shard-{i}" for i in range(args.count)}:
        raise ValueError("coverage artifacts do not match the complete expected shard set")
    full_collection: list[str] | None = None
    manifests: list[dict[str, Any]] = []
    execution_errors: list[str] = []
    with tempfile.TemporaryDirectory(prefix="daydream-coverage-") as temporary:
        staging = Path(temporary)
        for index in range(args.count):
            shard = root / f"shard-{index}"
            if not shard.is_dir() or {path.name for path in shard.iterdir()} != {"manifest.json", "coverage"}:
                raise ValueError("unexpected or missing shard files")
            manifest = json.loads((shard / "manifest.json").read_text(encoding="utf-8"))
            if manifest["schema"] != 1 or manifest["index"] != index or manifest["count"] != args.count:
                raise ValueError("incorrect shard identity")
            if any(manifest.get(key) != value for key, value in identity.items()):
                raise ValueError("shard belongs to a different checkout/run/attempt")
            collection = _execution(manifest, args.count, index, allow_failure=args.diagnostic)
            if args.diagnostic:
                try:
                    _execution(manifest, args.count, index)
                except ValueError as error:
                    execution_errors.append(f"shard-{index}: {error}")
            if full_collection is not None and collection != full_collection:
                raise ValueError("shards disagree on the full collection")
            full_collection = collection
            coverage = shard / "coverage"
            if _digest(coverage) != manifest["coverage_sha256"]:
                raise ValueError("coverage checksum mismatch")
            data = CoverageData(basename=str(coverage))
            try:
                data.read()
                if not data.has_arcs() or not data.measured_files():
                    raise ValueError("shard lacks branch coverage data")
            except CoverageException as error:
                raise ValueError(f"invalid coverage database: {error}") from error
            shutil.copyfile(coverage, staging / f".coverage.shard-{index}")
            manifests.append(manifest)
        combined = Coverage()
        combined.combine(data_paths=[str(staging)], strict=True, keep=True)
        combined.save()
        total = combined.report(show_missing=True)
        combined.xml_report(outfile="coverage.xml")
        if execution_errors:
            raise ValueError("diagnostic coverage only; " + "; ".join(execution_errors))
        if total < combined.config.fail_under:
            raise ValueError(f"combined coverage {total:.2f} is below {combined.config.fail_under}")
    durations = [manifest["elapsed_seconds"] for manifest in manifests]
    print(json.dumps({"shards": args.count, "tests": len(full_collection or []),
                      "shard_seconds": durations, "coverage": total}, sort_keys=True))


def _gate(args: argparse.Namespace) -> None:
    if args.mode == "baseline":
        if args.baseline_result != "success" or any(
            result != "skipped" for result in (args.static_result, args.shard_result, args.coverage_result)
        ):
            raise ValueError("baseline gate requires successful legacy job and skipped shard jobs")
        return
    if args.baseline_result != "skipped":
        raise ValueError("unexpected legacy baseline job")
    if args.static_result != "success":
        raise ValueError("static gate did not succeed")
    if args.mode == "full":
        if args.shard_result != "success" or args.coverage_result != "success":
            raise ValueError("full gate requires successful shards and combined coverage")
    elif args.mode in {"basic", "reuse"}:
        if args.shard_result != "skipped" or args.coverage_result != "skipped":
            raise ValueError("basic/reuse gate requires skipped full-suite jobs")
    else:
        raise ValueError("unknown CI gate mode")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--index", type=int, required=True)
    run.add_argument("--count", type=int, required=True)
    run.add_argument("--workers", type=int, required=True)
    run.add_argument("--output", type=Path, required=True)
    combine = commands.add_parser("combine")
    combine.add_argument("--count", type=int, required=True)
    combine.add_argument("--input", type=Path, required=True)
    combine.add_argument("--diagnostic", action="store_true",
                         help="write complete failed-suite coverage, then reject unsuccessful execution")
    gate = commands.add_parser("gate")
    gate.add_argument("--mode", required=True)
    gate.add_argument("--baseline-result", default="skipped")
    gate.add_argument("--static-result", required=True)
    gate.add_argument("--shard-result", required=True)
    gate.add_argument("--coverage-result", required=True)
    args = parser.parse_args()
    try:
        {"run": _run, "combine": _combine, "gate": _gate}[args.command](args)
    except (KeyError, TypeError, ValueError, OSError) as error:
        parser.exit(1, f"ci-shard: {error}\n")


if __name__ == "__main__":
    main()
