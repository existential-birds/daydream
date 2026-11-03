"""CI shard CLI contracts: duration partition, real pytest receipts, and combined coverage.

Planner tests cannot detect lost pytest execution or incomplete coverage artifacts.
One real shard run proves the pytest plugin and coverage persistence; the shard
split and every combine rejection then run in-process against copies of that evidence.
Each real run starts several interpreters, which is what made per-module fixtures
and repeated rounds explode under a loaded parallel gate.
"""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/ci_shard.py"
_PROFILE = "scripts/ci_test_durations.json"
_WEIGHTS = dict(zip("abcdefgh", [28.0, 1.0, 40.0, 1.0, 32.0, 1.0, 36.0, 1.0], strict=True))


def _write_profile(repo: Path, durations: dict[str, Any], *, fallback: Any = 2.0) -> None:
    """Keep the real CLI's profile inside its immutable, test-owned Git checkout."""
    (repo / _PROFILE).write_text(json.dumps({
        "schema": 1,
        "source": {
            "head": "a" * 40, "run_ids": [101], "samples_per_node": 1,
            "aggregation": "median_setup_call_teardown",
        },
        "fallback_seconds": fallback,
        "durations": durations,
    }))


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-c", "user.name=CI", "-c", "user.email=ci@example.invalid", *args],
                   cwd=repo, check=True, capture_output=True)


def _project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict[str, Any]]:
    """Commit a tiny project with weighted, unknown and skipped tests; load its script copy."""
    repo = tmp_path / "project"
    (repo / "scripts").mkdir(parents=True)
    shutil.copyfile(_SCRIPT, repo / "scripts/ci_shard.py")
    _write_profile(repo, {f"tests/test_sample.py::test_{name}": weight for name, weight in _WEIGHTS.items()})
    (repo / "sample").mkdir()
    (repo / "sample/__init__.py").write_text(
        "def positive(value):\n    if value > 0:\n        return 1\n    return 0\n"
        "def negative(value):\n    if value < 0:\n        return -1\n    return 0\n"
    )
    (repo / "tests").mkdir()
    # Each test records actual invocation order, separate from controller report order.
    (repo / "tests/test_sample.py").write_text(
        "import pytest\nfrom pathlib import Path\nfrom sample import positive, negative\n"
        "def _record(name):\n"
        "    with Path('execution-order.txt').open('a') as stream:\n"
        "        stream.write('tests/test_sample.py::' + name + '\\n')\n"
        "    assert (positive(1), negative(-1)) == (1, -1)\n"
        + "".join(f"def test_{name}(): _record('test_{name}')\n" for name in _WEIGHTS)
        # Full branch coverage needs every selected test, including the unknown-duration one.
        + "def test_z0():\n    _record('test_z0')\n    assert (positive(0), negative(0)) == (0, 0)\n"
        + "@pytest.mark.skip(reason='same selection includes skips')\ndef test_z1(): pass\n"
    )
    (repo / ".gitignore").write_text("execution-order*.txt\nartifacts/\ncoverage.xml\n.coverage*\n")
    (repo / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n'
        '[tool.coverage.run]\nbranch = true\nrelative_files = true\nsource = ["sample"]\n'
        '[tool.coverage.report]\nfail_under = 99\n'
    )
    for name in ("GITHUB_SHA", "COVERAGE_FILE", "DAYDREAM_CI_SHARD", "PYTEST_ADDOPTS", "PYTHONPATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GITHUB_RUN_ID", "100")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    _git(repo, "init", "-b", "main")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "fixture")
    monkeypatch.chdir(repo)
    return repo, runpy.run_path(str(repo / "scripts/ci_shard.py"))


def _cli(namespace: dict[str, Any], monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["ci_shard.py", *args])
    namespace["main"]()


def _rejects(namespace: dict[str, Any], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
             *args: str) -> str:
    with pytest.raises(SystemExit) as exited:
        _cli(namespace, monkeypatch, *args)
    assert exited.value.code == 1
    return capsys.readouterr().err


def test_partition_balances_estimates_and_admits_unknown_tests(tmp_path: Path) -> None:
    """Count balance misses costly skew; timing hints must still admit unknown tests."""
    namespace = runpy.run_path(str(_SCRIPT))
    load, partition = namespace["load_timing_profile"], namespace["partition"]
    (tmp_path / "scripts").mkdir()
    _write_profile(tmp_path, {f"t::{name}": weight for name, weight in _WEIGHTS.items()})
    profile = load(tmp_path / _PROFILE)
    assert profile.sha256 == hashlib.sha256((tmp_path / _PROFILE).read_bytes()).hexdigest()
    collection = sorted([f"t::{name}" for name in _WEIGHTS] + ["t::z0", "t::z1"])
    for count in (1, 2, 4, 8):
        shards = partition(collection, count, profile)
        assert shards == partition(list(reversed(collection)), count, profile)
        assert sorted(node for shard in shards for node in shard) == collection
        assert all(shards)
        for shard in shards:
            costs = [profile.durations.get(node, profile.fallback_seconds) for node in shard]
            assert costs == sorted(costs, reverse=True), costs
    two = partition(collection, 2, profile)
    assert all({"t::z0", "t::z1"} & set(shard) for shard in two)
    loads = [sum(profile.durations.get(node, profile.fallback_seconds) for node in shard) for shard in two]
    # An independent load bound rejects the lexical round-robin split without copying a partition.
    assert max(loads) <= sum(loads) / 2 + profile.fallback_seconds, loads
    for invalid in (["t::a"] * 2, ["t::a", ""]):
        with pytest.raises(ValueError):
            partition(invalid, 1, profile)
    with pytest.raises(ValueError):
        partition(collection, len(collection) + 1, profile)
    # A zero fallback hides future work; invalid costs cannot define a reproducible layout.
    for durations, fallback in (({}, 0), ({"t::a": -1.0}, 1.0), ({"t::a": float("nan")}, 1.0),
                                ({"t::a": True}, 1.0), ({"": 1.0}, 1.0)):
        _write_profile(tmp_path, durations, fallback=fallback)
        with pytest.raises(ValueError):
            load(tmp_path / _PROFILE)


def test_shards_execute_every_selected_test_and_combine_real_branch_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    from coverage import CoverageData

    repo, namespace = _project(tmp_path, monkeypatch)
    profile = namespace["load_timing_profile"]()
    # One real CLI round drives a real pytest-xdist worker through the shipped plugin.
    # Repeating it per shard only multiplies interpreter startups under a loaded gate.
    result = subprocess.run([sys.executable, "scripts/ci_shard.py", "run", "--index", "0", "--count", "1",
                             "--workers", "1", "--output", "real/shard-0"], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    real = json.loads((repo / "real/shard-0/manifest.json").read_text())
    collection = real["collection"]
    assert len(collection) == 10 and real["selected"] == namespace["partition"](collection, 1, profile)[0]
    assert real["partition_algorithm"] == "duration-lpt-v1" and real["partition_profile_sha256"] == profile.sha256
    assert set(real["reports"]) == set(collection) and real["exitstatus"] == 0 and not real["errors"]
    # Workers run the selection heaviest-first, and skips still report every phase.
    executed = (repo / "execution-order.txt").read_text().splitlines()
    assert executed == [node for node in real["selected"] if not node.endswith("test_z1")]
    assert real["reports"]["tests/test_sample.py::test_z1"]["setup"]["outcome"] == "skipped"

    # The plugin keeps exactly its shard, in partition order, and returns the full collection.
    shards = namespace["partition"](collection, 2, profile)
    state = namespace["_STATE"]
    for index in range(2):
        state.update(index=index, count=2, profile=profile, collections=[])
        items = [SimpleNamespace(nodeid=node) for node in reversed(collection)]
        namespace["pytest_collection_modifyitems"](SimpleNamespace(), items)
        assert [item.nodeid for item in items] == shards[index]
        assert state["collections"] == [collection]
    state.clear()

    # Split the real receipt into the two shards CI would publish, sharing its real coverage database.
    for index, selected in enumerate(shards):
        shard = repo / f"artifacts/shard-{index}"
        shard.mkdir(parents=True)
        shutil.copyfile(repo / "real/shard-0/coverage", shard / "coverage")
        (shard / "manifest.json").write_text(json.dumps({
            **real, "index": index, "count": 2, "selected": selected,
            "reports": {node: real["reports"][node] for node in selected},
        }))
    manifests = [json.loads((repo / f"artifacts/shard-{index}/manifest.json").read_text()) for index in range(2)]
    _cli(namespace, monkeypatch, "combine", "--count", "2", "--input", "artifacts")
    assert '"coverage": 100.0' in capsys.readouterr().out
    assert 'branch-rate="1"' in (repo / "coverage.xml").read_text()

    def corrupt(fault: str, artifacts: Path, shard: Path, manifest: dict[str, Any]) -> None:
        first = manifest["selected"][0]
        if fault == "missing-shard":
            shutil.rmtree(artifacts / "shard-1")
        elif fault == "unexpected-shard":
            (artifacts / "shard-2").mkdir()
        elif fault == "missing-coverage":
            (shard / "coverage").unlink()
        elif fault == "corrupt-coverage":
            (shard / "coverage").write_bytes(b"not sqlite")
        elif fault == "wrong-checksum":
            manifest["coverage_sha256"] = "wrong"
        elif fault == "wrong-profile":
            manifest["partition_profile_sha256"] = "0" * 64
        elif fault in {"wrong-head", "wrong-run", "wrong-index"}:
            manifest[{"wrong-head": "head", "wrong-run": "run_id", "wrong-index": "index"}[fault]] = "wrong"
        elif fault == "different-collection":
            # Change an unselected node while keeping this shard internally consistent.
            manifest["collection"][manifest["collection"].index(manifests[1]["selected"][0])] += "_changed"
        elif fault == "wrong-selection":
            manifest["selected"] = []
        elif fault == "missing-test":
            manifest["reports"].pop(first)
        elif fault == "missing-call":
            manifest["reports"][first].pop("call")
        elif fault == "failed-test":
            manifest["reports"][first]["call"]["outcome"] = "failed"
        elif fault == "worker-crash":
            manifest["errors"] = ["worker died"]
        elif fault == "line-only":
            (shard / "coverage").unlink()
            data = CoverageData(basename=str(shard / "coverage"))
            data.add_lines({"sample/__init__.py": {1}})
            data.write()
        if fault in {"corrupt-coverage", "line-only"}:
            manifest["coverage_sha256"] = hashlib.sha256((shard / "coverage").read_bytes()).hexdigest()

    # CI's diagnostic mode must reject each corrupt receipt from the real execution.
    for fault in ("missing-shard", "unexpected-shard", "missing-coverage", "corrupt-coverage", "wrong-checksum",
                  "wrong-profile", "wrong-head", "wrong-run", "wrong-index", "different-collection",
                  "wrong-selection", "missing-test", "missing-call", "failed-test", "worker-crash", "line-only",
                  "low-coverage"):
        artifacts = tmp_path / fault
        shutil.copytree(repo / "artifacts", artifacts)
        shard = artifacts / "shard-0"
        manifest_path = shard / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        corrupt(fault, artifacts, shard, manifest)
        manifest_path.write_text(json.dumps(manifest))
        if fault == "low-coverage":
            # Valid, checksummed branch data that misses the project's coverage floor.
            for index in range(2):
                low = artifacts / f"shard-{index}"
                (low / "coverage").unlink()
                data = CoverageData(basename=str(low / "coverage"))
                data.add_arcs({"sample/__init__.py": {(-1, 1), (1, -1)}})
                data.write()
                low_manifest = json.loads((low / "manifest.json").read_text())
                low_manifest["coverage_sha256"] = hashlib.sha256((low / "coverage").read_bytes()).hexdigest()
                (low / "manifest.json").write_text(json.dumps(low_manifest))
        error = _rejects(namespace, monkeypatch, capsys, "combine", "--count", "2", "--input", str(artifacts),
                         "--diagnostic")
        assert error.startswith("ci-shard:"), (fault, error)
        if fault in {"failed-test", "worker-crash"}:
            # A complete failed suite keeps aggregate XML while the gate remains red.
            assert "diagnostic coverage only" in error
            assert 'branch-rate="1"' in (repo / "coverage.xml").read_text()
        elif fault != "low-coverage":
            assert not (repo / "coverage.xml").exists(), fault
    # Outside diagnostic mode a failed suite publishes no coverage report at all.
    error = _rejects(namespace, monkeypatch, capsys, "combine", "--count", "2",
                     "--input", str(tmp_path / "failed-test"))
    assert "unsuccessful execution" in error
    assert not (repo / "coverage.xml").exists()


def test_run_cannot_publish_success_without_complete_unchanged_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    repo, namespace = _project(tmp_path, monkeypatch)
    real_run = subprocess.run
    pytest_effect: list[Callable[[], None]] = []

    def run(command: list[str], **kwargs: Any) -> Any:
        if command[1:3] == ["-m", "pytest"]:
            # Stand in for pytest only; the identity and receipt checks stay real.
            # Group dispatch lets the heaviest-first order spread across workers.
            assert "--dist=loadgroup" in command
            pytest_effect[0]()
            return subprocess.CompletedProcess(command, 0)
        return real_run(command, **kwargs)

    monkeypatch.setitem(namespace["main"].__globals__, "subprocess", SimpleNamespace(
        run=run, check_output=subprocess.check_output, CompletedProcess=subprocess.CompletedProcess))

    def mutate_source() -> None:
        source = repo / "sample/__init__.py"
        source.write_text(source.read_text() + "# changed\n")

    def run_shard(output: str) -> str:
        error = _rejects(namespace, monkeypatch, capsys, "run", "--index", "0", "--count", "1", "--workers", "1",
                         "--output", output)
        assert not (repo / output / "manifest.json").exists()
        return error

    pytest_effect[:] = [lambda: None]
    assert "did not publish an execution receipt" in run_shard("no-receipt")
    # Source mutated while pytest ran: the post-run identity check rejects the evidence.
    pytest_effect[:] = [mutate_source]
    assert "tracked checkout differs" in run_shard("mutated")

    def must_not_run() -> None:
        raise AssertionError("pytest ran from a dirty checkout")

    pytest_effect[:] = [must_not_run]
    assert "tracked checkout differs" in run_shard("dirty")
    _git(repo, "checkout", "--", "sample/__init__.py")
    (repo / "stale").mkdir()
    (repo / "stale/sentinel").write_text("prior run")
    assert "must be empty" in run_shard("stale")
    assert (repo / "stale/sentinel").read_text() == "prior run"
    _write_profile(repo, {}, fallback=0)
    _git(repo, "commit", "-am", "invalid timing")
    assert "finite and positive" in run_shard("invalid-timing")


def test_final_gate_propagates_dependency_outcomes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    namespace = runpy.run_path(str(_SCRIPT))
    rows = [
        ("full", "skipped", "success", "success", "success", True),
        ("basic", "skipped", "success", "skipped", "skipped", True),
        ("reuse", "skipped", "success", "skipped", "skipped", True),
        ("baseline", "success", "skipped", "skipped", "skipped", True),
        ("unknown", "skipped", "success", "skipped", "skipped", False),
        ("full", "skipped", "failure", "success", "success", False),
        ("full", "skipped", "success", "skipped", "success", False),
        ("full", "skipped", "success", "success", "failure", False),
        ("basic", "skipped", "cancelled", "skipped", "skipped", False),
        ("reuse", "skipped", "success", "failure", "skipped", False),
        ("basic", "skipped", "success", "success", "success", False),
        ("baseline", "skipped", "skipped", "skipped", "skipped", False),
        ("baseline", "success", "success", "skipped", "skipped", False),
        ("basic", "success", "success", "skipped", "skipped", False),
    ]
    for mode, baseline, static, shards, coverage, success in rows:
        args = ("gate", "--mode", mode, "--baseline-result", baseline, "--static-result", static,
                "--shard-result", shards, "--coverage-result", coverage)
        if success:
            _cli(namespace, monkeypatch, *args)
        else:
            assert _rejects(namespace, monkeypatch, capsys, *args).startswith("ci-shard:"), args
    # The final status job runs with plain python3 and no installed dependencies.
    result = subprocess.run([sys.executable, "-S", str(_SCRIPT), "gate", "--mode", "full", "--static-result",
                             "success", "--shard-result", "success", "--coverage-result", "success"],
                            capture_output=True, text=True, check=False, env={"PATH": os.environ["PATH"]})
    assert result.returncode == 0, result.stderr
