"""CI shard CLI contracts with real pytest workers, Git, and coverage databases.

Existing planner tests cannot detect lost pytest execution or incomplete coverage
artifacts. These tests exercise that new persistence/process boundary directly.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/ci_shard.py"
_PROFILE = "scripts/ci_test_durations.json"


@dataclass(frozen=True)
class ShardProject:
    """Preserve subprocess environment without exposing credentials in pytest reprs."""

    repo: Path
    env: dict[str, str] = field(repr=False)


def _write_profile(repo: Path, durations: dict[str, float], *, fallback: float = 1.0) -> None:
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


def _cli(repo: Path, *args: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "scripts/ci_shard.py", *args], cwd=repo, env=env,
                          capture_output=True, text=True, check=False)


@pytest.fixture(scope="module")
def shard_artifacts(tmp_path_factory: pytest.TempPathFactory) -> ShardProject:
    """Generate both artifacts through the shipped CLI, retaining real xdist execution."""
    repo = tmp_path_factory.mktemp("ci-shard-project")
    (repo / "scripts").mkdir()
    shutil.copyfile(_SCRIPT, repo / "scripts/ci_shard.py")
    _write_profile(repo, {})
    (repo / "sample").mkdir()
    (repo / "sample/__init__.py").write_text(
        "def positive(value):\n    if value > 0:\n        return 1\n    return 0\n"
        "def negative(value):\n    if value < 0:\n        return -1\n    return 0\n"
    )
    (repo / "tests").mkdir()
    (repo / "tests/test_sample.py").write_text(
        "import pytest\nfrom sample import positive, negative\n"
        "def test_a(): assert positive(1) == 1\n"
        "def test_b(): assert negative(-1) == -1\n"
        "def test_c(): assert positive(0) == 0\n"
        "def test_d(): assert negative(0) == 0\n"
        "@pytest.mark.skip(reason='same selection includes skips')\ndef test_e(): pass\n"
    )
    (repo / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n'
        '[tool.coverage.run]\nbranch = true\nrelative_files = true\nsource = ["sample"]\n'
        '[tool.coverage.report]\nfail_under = 99\n'
    )
    env = {key: value for key, value in os.environ.items() if key not in {
        "GITHUB_SHA", "COVERAGE_FILE", "DAYDREAM_CI_SHARD", "PYTEST_ADDOPTS", "PYTHONPATH",
    }}
    env.update(GITHUB_RUN_ID="100", GITHUB_RUN_ATTEMPT="1")
    for command in (["init", "-b", "main"], ["add", "."],
                    ["-c", "user.name=CI", "-c", "user.email=ci@example.invalid", "commit", "-m", "fixture"]):
        subprocess.run(["git", *command], cwd=repo, env=env, check=True, capture_output=True)
    for index in range(2):
        result = _cli(repo, "run", "--index", str(index), "--count", "2", "--workers", "1",
                      "--output", f"artifacts/shard-{index}", env=env)
        assert result.returncode == 0, result.stdout + result.stderr
    return ShardProject(repo, env)


@pytest.fixture(scope="module")
def weighted_shard_artifacts(
    shard_artifacts: ShardProject, tmp_path_factory: pytest.TempPathFactory,
) -> ShardProject:
    """Lexical round-robin puts all expensive nodes together; real receipts expose it."""
    original, env = shard_artifacts.repo, shard_artifacts.env
    repo = tmp_path_factory.mktemp("weighted-ci-shard-project") / "project"
    shutil.copytree(original, repo)
    weights = dict(zip("abcdefgh", [28.0, 1.0, 40.0, 1.0, 32.0, 1.0, 36.0, 1.0], strict=True))
    _write_profile(repo, {f"tests/test_sample.py::test_{name}": weight for name, weight in weights.items()},
                   fallback=2.0)
    # No sleeps: this fixture checks the scheduling promise against supplied timing hints.
    # The worker records actual invocation order separately from controller report order.
    (repo / "tests/test_sample.py").write_text(
        "from pathlib import Path\nfrom sample import positive, negative\n"
        "def _record(name):\n"
        "    with Path('execution-order.txt').open('a') as stream:\n"
        "        stream.write('tests/test_sample.py::' + name + '\\n')\n"
        "    assert (positive(1), positive(0), negative(-1), negative(0)) == (1, 0, -1, 0)\n"
        + "".join(f"def test_{name}(): _record('test_{name}')\n" for name in [*weights, "z0", "z1"])
    )
    subprocess.run(["git", "add", "tests/test_sample.py", _PROFILE], cwd=repo, env=env, check=True)
    subprocess.run(["git", "-c", "user.name=CI", "-c", "user.email=ci@example.invalid",
                    "commit", "-m", "weighted fixture"], cwd=repo, env=env, check=True, capture_output=True)
    for index in range(2):
        result = _cli(repo, "run", "--index", str(index), "--count", "2", "--workers", "1",
                      "--output", f"weighted/shard-{index}", env=env)
        assert result.returncode == 0, result.stdout + result.stderr
        (repo / "execution-order.txt").rename(repo / f"execution-order-{index}.txt")
    return ShardProject(repo, env)


def test_duration_partition_balances_real_execution_without_losing_new_tests(
    weighted_shard_artifacts: ShardProject,
) -> None:
    """Count balance misses costly skew; timing hints must still admit unknown tests."""
    repo, env = weighted_shard_artifacts.repo, weighted_shard_artifacts.env
    profile = json.loads((repo / _PROFILE).read_text())
    manifests = [json.loads((repo / f"weighted/shard-{index}/manifest.json").read_text()) for index in range(2)]
    collection = manifests[0]["collection"]
    selected = [node for manifest in manifests for node in manifest["selected"]]
    assert sorted(selected) == collection
    assert len(selected) == len(set(selected))
    assert all(set(manifest["reports"]) == set(manifest["selected"]) for manifest in manifests)
    unknown = set(collection) - profile["durations"].keys()
    assert len(unknown) == 2
    assert all(unknown.intersection(manifest["selected"]) for manifest in manifests)
    loads = [sum(profile["durations"].get(node, profile["fallback_seconds"])
                 for node in manifest["selected"]) for manifest in manifests]
    # An independent load bound rejects the 138/6 round-robin split without copying a partition.
    assert max(loads) <= sum(loads) / 2 + profile["fallback_seconds"], loads
    result = _cli(repo, "combine", "--count", "2", "--input", "weighted", env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'branch-rate="1"' in (repo / "coverage.xml").read_text()


def test_duration_partition_executes_expensive_nodes_first(
    weighted_shard_artifacts: ShardProject,
) -> None:
    """Correct membership alone permits a long tail when xdist receives lexical order."""
    repo = weighted_shard_artifacts.repo
    profile = json.loads((repo / _PROFILE).read_text())
    for index in range(2):
        manifest = json.loads((repo / f"weighted/shard-{index}/manifest.json").read_text())
        observed = (repo / f"execution-order-{index}.txt").read_text().splitlines()
        assert observed == manifest["selected"]
        costs = [profile["durations"].get(node, profile["fallback_seconds"]) for node in observed]
        assert costs == sorted(costs, reverse=True), costs


def test_duration_partition_receipts_bind_the_shared_algorithm_and_exact_profile(
    weighted_shard_artifacts: ShardProject,
) -> None:
    """Runtime, verifier and combiner need one reproducible selection contract."""
    repo = weighted_shard_artifacts.repo
    manifests = [json.loads((repo / f"weighted/shard-{index}/manifest.json").read_text()) for index in range(2)]
    for manifest in manifests:
        assert manifest.get("partition_algorithm") == "duration-lpt-v1"
        assert manifest.get("partition_profile_sha256") == hashlib.sha256((repo / _PROFILE).read_bytes()).hexdigest()

    from scripts.ci_shard import load_timing_profile, partition

    profile = load_timing_profile(repo / _PROFILE)
    collection = manifests[0]["collection"]
    assert partition(collection, 2, profile) == [manifest["selected"] for manifest in manifests]
    for count in (1, 2, 4, 8):
        shards = partition(collection, count, profile)
        assert shards == partition(list(reversed(collection)), count, profile)
        assert sorted(node for shard in shards for node in shard) == collection
        assert all(shards)


@pytest.mark.parametrize("cheap_count", [3, 60], ids=["less-than-two-per-worker", "batched-load"])
def test_four_workers_admit_the_four_heaviest_cases_before_any_finishes(
    shard_artifacts: ShardProject, tmp_path: Path, cheap_count: int,
) -> None:
    """W1 ordering misses xdist batching several expensive cases onto one worker."""
    original, env = shard_artifacts.repo, shard_artifacts.env
    repo = tmp_path / "project"
    shutil.copytree(original, repo)
    heavy = [f"heavy_{index}" for index in range(4)]
    _write_profile(repo, {f"tests/test_sample.py::test_{name}": 100.0 - index
                          for index, name in enumerate(heavy)})
    (repo / "tests/test_sample.py").write_text(
        "import os\nimport time\nfrom pathlib import Path\nfrom sample import positive, negative\n"
        "def _cover():\n"
        "    assert (positive(1), positive(0), negative(-1), negative(0)) == (1, 0, -1, 0)\n"
        "def _rendezvous(name):\n"
        "    root = Path('admitted')\n"
        "    root.mkdir(exist_ok=True)\n"
        "    (root / (name + '-' + os.environ['PYTEST_XDIST_WORKER'])).mkdir()\n"
        "    failed = root / 'failed'\n"
        "    deadline = time.monotonic() + 15\n"
        "    while len(list(root.glob('heavy_*'))) < 4 and not failed.exists():\n"
        "        if time.monotonic() >= deadline:\n"
        "            failed.touch()\n"
        "            break\n"
        "        time.sleep(0.01)\n"
        "    assert not failed.exists(), 'expensive cases were queued behind another expensive case'\n"
        "    assert len(list(root.glob('heavy_*'))) == 4\n"
        "    _cover()\n"
        + "".join(f"def test_{name}(): _rendezvous('{name}')\n" for name in heavy)
        + "".join(f"def test_cheap_{index:02}(): _cover()\n" for index in range(cheap_count))
    )
    subprocess.run(["git", "add", "tests/test_sample.py", _PROFILE], cwd=repo, env=env, check=True)
    subprocess.run(["git", "-c", "user.name=CI", "-c", "user.email=ci@example.invalid",
                    "commit", "-m", "concurrent scheduling fixture"],
                   cwd=repo, env=env, check=True, capture_output=True)
    result = _cli(repo, "run", "--index", "0", "--count", "1", "--workers", "4",
                  "--output", "concurrent/shard-0", env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    admitted = list((repo / "admitted").glob("heavy_*"))
    assert len(admitted) == 4
    assert len({path.name.rsplit("-", 1)[1] for path in admitted}) == 4
    assert not (repo / "admitted/failed").exists()
    manifest = json.loads((repo / "concurrent/shard-0/manifest.json").read_text())
    assert len(manifest["collection"]) == 4 + cheap_count
    assert set(manifest["selected"]) == set(manifest["reports"]) == set(manifest["collection"])
    combined = _cli(repo, "combine", "--count", "1", "--input", "concurrent", env=env)
    assert combined.returncode == 0, combined.stdout + combined.stderr
    assert 'branch-rate="1"' in (repo / "coverage.xml").read_text()


@pytest.mark.parametrize("fault", ["zero-fallback", "negative-duration", "nonfinite-duration"])
def test_run_rejects_invalid_timing_hints_before_publishing_evidence(
    shard_artifacts: ShardProject, tmp_path: Path, fault: str,
) -> None:
    """A zero fallback hides future work; invalid costs cannot define a reproducible layout."""
    original, env = shard_artifacts.repo, shard_artifacts.env
    repo = tmp_path / "project"
    shutil.copytree(original, repo)
    profile = json.loads((repo / _PROFILE).read_text())
    if fault == "zero-fallback":
        profile["fallback_seconds"] = 0
    else:
        profile["durations"]["tests/test_sample.py::test_a"] = (
            -1.0 if fault == "negative-duration" else float("nan")
        )
    (repo / _PROFILE).write_text(json.dumps(profile))
    subprocess.run(["git", "add", _PROFILE], cwd=repo, env=env, check=True)
    subprocess.run(["git", "-c", "user.name=CI", "-c", "user.email=ci@example.invalid",
                    "commit", "-m", "invalid timing fixture"], cwd=repo, env=env, check=True, capture_output=True)
    result = _cli(repo, "run", "--index", "0", "--count", "1", "--workers", "1",
                  "--output", "invalid-timing", env=env)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "ci-shard:" in result.stderr
    assert not (repo / "invalid-timing/manifest.json").exists()


def test_shards_cover_collection_and_combine_real_branch_data(
    shard_artifacts: ShardProject,
) -> None:
    repo, env = shard_artifacts.repo, shard_artifacts.env
    manifests = [json.loads((repo / f"artifacts/shard-{index}/manifest.json").read_text()) for index in range(2)]
    # Explicit independent selection proves no omitted, duplicated, or unselected test ran.
    assert [item["selected"] for item in manifests] == [
        ["tests/test_sample.py::test_a", "tests/test_sample.py::test_c", "tests/test_sample.py::test_e"],
        ["tests/test_sample.py::test_b", "tests/test_sample.py::test_d"],
    ]
    assert [set(item["reports"]) for item in manifests] == [set(item["selected"]) for item in manifests]
    assert manifests[0]["reports"]["tests/test_sample.py::test_e"]["setup"]["outcome"] == "skipped"
    result = _cli(repo, "combine", "--count", "2", "--input", "artifacts", env=env)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"coverage": 100.0' in result.stdout
    assert 'branch-rate="1"' in (repo / "coverage.xml").read_text()


@pytest.mark.parametrize("fault", [
    "missing-shard", "unexpected-shard", "missing-coverage", "corrupt-coverage", "wrong-checksum",
    "wrong-head", "wrong-tree", "wrong-run", "wrong-attempt", "wrong-index", "wrong-profile", "wrong-algorithm",
    "different-collection", "duplicate-test",
    "wrong-selection", "missing-test", "missing-teardown", "missing-call", "failed-test", "failed-process",
    "worker-crash", "line-only", "low-coverage",
])
def test_combine_rejects_incomplete_or_wrong_run_artifacts(
    shard_artifacts: ShardProject, tmp_path: Path, fault: str,
) -> None:
    """CI's diagnostic mode must reject each corrupt receipt from a real execution."""
    from coverage import CoverageData

    repo, env = shard_artifacts.repo, shard_artifacts.env
    artifacts = tmp_path / "artifacts"
    shutil.copytree(repo / "artifacts", artifacts)
    shard = artifacts / "shard-0"
    path = shard / "manifest.json"
    manifest = json.loads(path.read_text())
    if fault == "missing-shard":
        shutil.rmtree(artifacts / "shard-1")
    elif fault == "unexpected-shard":
        (artifacts / "shard-2").mkdir()
    elif fault == "missing-coverage":
        (shard / "coverage").unlink()
    elif fault == "corrupt-coverage":
        (shard / "coverage").write_bytes(b"not sqlite")
        manifest["coverage_sha256"] = hashlib.sha256((shard / "coverage").read_bytes()).hexdigest()
    elif fault == "wrong-checksum":
        manifest["coverage_sha256"] = "wrong"
    elif fault == "wrong-profile":
        manifest["partition_profile_sha256"] = "0" * 64
    elif fault == "wrong-algorithm":
        manifest["partition_algorithm"] = "unrecognized-partitioner"
    elif fault in {"wrong-head", "wrong-tree", "wrong-run", "wrong-attempt", "wrong-index"}:
        keys = {"wrong-head": "head", "wrong-tree": "tree", "wrong-run": "run_id",
                "wrong-attempt": "run_attempt", "wrong-index": "index"}
        key = keys[fault]
        manifest[key] = "wrong"
    elif fault in {"different-collection", "duplicate-test"}:
        if fault == "duplicate-test":
            manifest["collection"].append(manifest["collection"][0])
        else:
            # Change an unselected node while keeping this shard internally consistent.
            manifest["collection"][1] = "tests/test_sample.py::test_b_changed"
    elif fault == "wrong-selection":
        manifest["selected"] = []
    elif fault == "missing-test":
        manifest["reports"].pop("tests/test_sample.py::test_a")
    elif fault == "missing-teardown":
        manifest["reports"]["tests/test_sample.py::test_a"].pop("teardown")
    elif fault == "missing-call":
        manifest["reports"]["tests/test_sample.py::test_a"].pop("call")
    elif fault == "failed-test":
        manifest["reports"]["tests/test_sample.py::test_a"]["call"]["outcome"] = "failed"
    elif fault == "failed-process":
        manifest["exitstatus"] = 1
    elif fault == "worker-crash":
        manifest["errors"] = ["worker died"]
    else:
        (shard / "coverage").unlink()
        data = CoverageData(basename=str(shard / "coverage"))
        if fault == "line-only":
            data.add_lines({"sample/__init__.py": {1}})
        else:
            data.add_arcs({"sample/__init__.py": {(-1, 1), (1, -1)}})
        data.write()
        manifest["coverage_sha256"] = hashlib.sha256((shard / "coverage").read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))
    result = _cli(repo, "combine", "--count", "2", "--input", str(artifacts), "--diagnostic", env=env)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "ci-shard:" in result.stderr or "coverage.exceptions.DataError" in result.stderr
    if fault not in {"failed-test", "failed-process", "worker-crash", "low-coverage"}:
        assert not (repo / "coverage.xml").exists()


@pytest.mark.parametrize(("mode", "static", "shards", "coverage", "success"), [
    ("full", "success", "success", "success", True),
    ("basic", "success", "skipped", "skipped", True),
    ("reuse", "success", "skipped", "skipped", True),
    ("unknown", "success", "skipped", "skipped", False),
    ("full", "failure", "success", "success", False),
    ("full", "success", "cancelled", "skipped", False),
    ("full", "success", "skipped", "skipped", False),
    ("full", "success", "success", "failure", False),
    ("basic", "cancelled", "skipped", "skipped", False),
    ("reuse", "success", "failure", "skipped", False),
    ("basic", "success", "success", "success", False),
])
def test_final_gate_propagates_dependency_outcomes(
    mode: str, static: str, shards: str, coverage: str, success: bool,
) -> None:
    """Run with -S to prove the final status does not require installed dependencies."""
    result = subprocess.run([
        sys.executable, "-S", str(_SCRIPT), "gate", "--mode", mode, "--static-result", static,
        "--shard-result", shards, "--coverage-result", coverage,
    ], capture_output=True, text=True, check=False)
    assert (result.returncode == 0) == success, result.stderr


@pytest.mark.parametrize(("mode", "baseline", "static", "success"), [
    ("baseline", "success", "skipped", True),
    ("baseline", "skipped", "skipped", False),
    ("baseline", "cancelled", "skipped", False),
    ("baseline", "success", "success", False),
    ("basic", "success", "success", False),
])
def test_final_gate_requires_the_selected_baseline_job(
    mode: str, baseline: str, static: str, success: bool,
) -> None:
    result = subprocess.run([
        sys.executable, "-S", str(_SCRIPT), "gate", "--mode", mode, "--baseline-result", baseline,
        "--static-result", static, "--shard-result", "skipped", "--coverage-result", "skipped",
    ], capture_output=True, text=True, check=False)
    assert (result.returncode == 0) == success, result.stderr


@pytest.mark.parametrize("fault", [
    "empty-selection", "stale-output", "dirty-checkout", "source-mutation",
])
def test_run_cannot_publish_success_for_failed_or_empty_execution(
    shard_artifacts: ShardProject, tmp_path: Path, fault: str,
) -> None:
    repo, env = shard_artifacts.repo, shard_artifacts.env
    project = tmp_path / "project"
    shutil.copytree(repo, project)
    output = project / "rejected"
    index, count = 0, 1
    if fault in {"failing-test", "dirty-checkout", "source-mutation"}:
        body = "def test_failure(): assert False\n"
        if fault == "source-mutation":
            body = ("from pathlib import Path\nfrom sample import positive\n"
                    "def test_mutation():\n    assert positive(1) == 1\n"
                    "    source = Path('sample/__init__.py')\n"
                    "    source.write_text(source.read_text() + '# changed\\n')\n")
        (project / "tests/test_sample.py").write_text(body)
        if fault != "dirty-checkout":
            subprocess.run(["git", "add", "tests/test_sample.py"], cwd=project, env=env, check=True)
            subprocess.run(["git", "-c", "user.name=CI", "-c", "user.email=ci@example.invalid",
                            "commit", "-m", "failing fixture"], cwd=project, env=env, check=True, capture_output=True)
    elif fault == "empty-selection":
        index, count = 5, 6
    else:
        output.mkdir()
        (output / "sentinel").write_text("prior run")
    result = _cli(project, "run", "--index", str(index), "--count", str(count), "--workers", "1",
                  "--output", str(output), env=env)
    assert result.returncode != 0, result.stdout + result.stderr
    assert not (output / "manifest.json").exists()
    if fault == "stale-output":
        assert (output / "sentinel").read_text() == "prior run"


def test_failed_suite_retains_diagnostic_coverage_without_success(
    shard_artifacts: ShardProject, tmp_path: Path,
) -> None:
    """An actual assertion failure keeps aggregate XML while every gate remains red."""
    repo, env = shard_artifacts.repo, shard_artifacts.env
    project = tmp_path / "project"
    shutil.copytree(repo, project)
    tests = project / "tests/test_sample.py"
    tests.write_text(tests.read_text() + "def test_f(): assert False\n")
    subprocess.run(["git", "add", "tests/test_sample.py"], cwd=project, env=env, check=True)
    subprocess.run(["git", "-c", "user.name=CI", "-c", "user.email=ci@example.invalid",
                    "commit", "-m", "failing fixture"], cwd=project, env=env, check=True, capture_output=True)
    for index in range(2):
        result = _cli(project, "run", "--index", str(index), "--count", "2", "--workers", "1",
                      "--output", f"diagnostic/shard-{index}", env=env)
        assert (result.returncode == 0) == (index == 0), result.stdout + result.stderr
        assert (project / f"diagnostic/shard-{index}/manifest.json").is_file()
        assert (project / f"diagnostic/shard-{index}/coverage").is_file()
    strict = _cli(project, "combine", "--count", "2", "--input", "diagnostic", env=env)
    assert strict.returncode != 0, strict.stdout + strict.stderr
    assert not (project / "coverage.xml").exists()
    diagnostic = _cli(project, "combine", "--diagnostic", "--count", "2", "--input", "diagnostic", env=env)
    assert diagnostic.returncode != 0, diagnostic.stdout + diagnostic.stderr
    assert "diagnostic coverage only" in diagnostic.stderr
    assert 'branch-rate="1"' in (project / "coverage.xml").read_text()
