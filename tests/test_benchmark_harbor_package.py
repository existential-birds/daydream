"""Packaging and Harbor 0.25 integration tests for compiled benchmarks."""
import hashlib
import importlib.metadata
import importlib.resources
import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

import pytest
import yaml

from daydream.benchmark.harbor import build, package as pkg
from daydream.benchmark.harbor.build import TEMPLATE_VERSION
from tests.harness.fake_gh import FakeGh
from tests.test_benchmark_harbor_build import (
    _harbor_tree_bytes,
    _seed_clean_workspace,
    _seed_ready_workspace,
    _stub_wheel,
)

# The runtime-requirements lock is committed, not generated at package time: nothing in the
# production packaging path runs `uv export`. These helpers ARE that regeneration recipe, so
# they live with the only test that uses them and with the committed lock they must reproduce.
_GENERATION_COMMAND = "uv export --frozen --no-dev --no-emit-project --format requirements-txt"


def _strip_uv_header(text: str) -> str:
    """Strip uv's leading generated-comment block while preserving requirement comments."""
    lines = text.splitlines(keepends=True)
    index = 0
    while index < len(lines) and (lines[index].startswith("#") or not lines[index].strip()):
        index += 1
    return "".join(lines[index:])


def _uv_export_body(uv_lock_path: Path) -> str:
    """Export hash-pinned runtime requirements from *uv_lock_path*."""
    command = _GENERATION_COMMAND.split()
    try:
        result = subprocess.run(
            command, cwd=uv_lock_path.resolve().parent, capture_output=True, text=True, check=False,
        )
    except OSError as exc:
        raise pkg.PackageError(f"cannot run `{_GENERATION_COMMAND}`: {exc}") from exc
    if result.returncode != 0:
        raise pkg.PackageError(
            f"`{_GENERATION_COMMAND}` failed with exit {result.returncode}: {result.stderr.strip()}"
        )
    body = _strip_uv_header(result.stdout)
    if "--hash=sha256:" not in body:
        raise pkg.PackageError(f"`{_GENERATION_COMMAND}` did not produce hash-pinned requirements")
    return body


def _render_runtime_lock(uv_lock_path: Path, *, daydream_version: str) -> tuple[str, str]:
    """Return the deterministic header and exported requirements body."""
    source_sha256 = hashlib.sha256(uv_lock_path.read_bytes()).hexdigest()
    body = _uv_export_body(uv_lock_path)
    if f"daydream=={daydream_version}" in body:
        raise pkg.PackageError("runtime lock unexpectedly includes the Daydream project")
    header = (
        "# Daydream Harbor runtime requirements (generated; do not edit)\n"
        f"# daydream_version: {daydream_version}\n"
        f"# source_uv_lock_sha256: {source_sha256}\n"
        f"# generation_command: {_GENERATION_COMMAND}\n"
        f"# template_version: {TEMPLATE_VERSION}\n"
        "\n"
    )
    return header, body


def test_runtime_lock_regeneration_is_noop_on_unchanged(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    repo = tmp_path / "repo"
    repo.mkdir()
    uv_lock = repo / "uv.lock"
    uv_lock.write_bytes((root / "uv.lock").read_bytes())
    (repo / "pyproject.toml").write_bytes((root / "pyproject.toml").read_bytes())
    ver = importlib.metadata.version("daydream")
    committed = (root / "daydream/benchmark/harbor/runtime-requirements.lock").read_bytes()
    # Byte-identity is the whole header contract: every field, in order, plus the pinned body.
    assert "".join(_render_runtime_lock(uv_lock, daydream_version=ver)).encode() == committed
    uv_lock.write_bytes(uv_lock.read_bytes() + b"\n# drift\n")
    assert "".join(_render_runtime_lock(uv_lock, daydream_version=ver)).encode() != committed

def test_validate_wheel_accepts_matching_and_rejects_mismatch(tmp_path: Path) -> None:
    good, ver = _stub_wheel(tmp_path)
    info = pkg.validate_wheel(good, daydream_version=ver)
    assert info.distribution == "daydream" and info.version == ver
    assert len(info.sha256) == 64

    bad = tmp_path / "daydream-0.99.0-py3-none-any.whl"
    bad.write_bytes(b"x")
    with pytest.raises(pkg.PackageError) as mismatch:
        pkg.validate_wheel(bad, daydream_version=ver)
    assert bad.name in str(mismatch.value)
    assert "daydream-" + ver in str(mismatch.value)

    with pytest.raises(pkg.PackageError) as missing:
        pkg.validate_wheel(tmp_path / "absent.whl", daydream_version=ver)
    assert "absent.whl" in str(missing.value)

def test_resolve_harbor_checks_same_interpreter_and_version(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("harbor")

    exe = pkg.resolve_harbor()
    assert exe == str(Path(sys.executable).parent / "harbor")

    def absent(distribution: Any) -> None:
        raise importlib.metadata.PackageNotFoundError(distribution)

    monkeypatch.setattr(importlib.metadata, "version", absent)
    with pytest.raises(pkg.PackageError) as missing:
        pkg.resolve_harbor()
    assert "pip install 'daydream[benchmark]'" in str(missing.value)

    for unsupported in ("0.24.0", "0.26.0", "invalid"):
        monkeypatch.setattr(importlib.metadata, "version", lambda d, version=unsupported: version)
        with pytest.raises(pkg.PackageError) as wrong:
            pkg.resolve_harbor()
        assert "[0.25, 0.26)" in str(wrong.value)

def test_render_task_toml_fails_closed_on_empty_or_missing_hosts() -> None:
    bad_cases: tuple[dict[str, Any], ...] = (
        {"reviewer_hosts": [], "judge_hosts": ["h.example"]}, {"reviewer_hosts": ["h.example"], "judge_hosts": []},
        {"reviewer_hosts": [], "judge_hosts": []},
    )
    for kw in bad_cases:
        with pytest.raises(pkg.PackageError):
            pkg.render_task_toml("case-abc", **kw)
    with pytest.raises(pkg.PackageError):
        pkg.render_task_toml("case-abc")   # missing required kwargs

def test_render_task_toml_normalizes_and_sorts_hosts() -> None:
    doc = tomllib.loads(pkg.render_task_toml(
        "case-abc", reviewer_hosts=["HTTPS://B.Example.com", "a.example.com"], judge_hosts=["openrouter.ai"],
    ).decode())
    assert doc["agent"]["allowed_hosts"] == ["a.example.com", "b.example.com"]
    assert doc["verifier"]["environment"]["allowed_hosts"] == ["openrouter.ai"]
    with pytest.raises(pkg.PackageError):
        pkg.render_task_toml("case-abc", reviewer_hosts=["not-a-host"], judge_hosts=["openrouter.ai"])
    with pytest.raises(pkg.PackageError):
        pkg.render_task_toml("case-abc", reviewer_hosts=["api.anthropic.com"], judge_hosts=["*.example.com"])

def test_render_job_config_resolves_with_only_selected_provider_credential(monkeypatch: pytest.MonkeyPatch,) -> None:
    """An unset *alternative* credential never aborts rendering (issue #979)."""

    pytest.importorskip("harbor")

    from harbor.utils.env import resolve_env_vars

    # Judge: anthropic selected -> CLAUDE_CODE_OAUTH_TOKEN must be optional.
    for var in (
        "DAYDREAM_JUDGE_API_KEY", "DAYDREAM_JUDGE_BASE_URL", "CLAUDE_CODE_OAUTH_TOKEN", "DAYDREAM_REVIEW_API_KEY",
        "DAYDREAM_REVIEW_BASE_URL", "ANTHROPIC_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DAYDREAM_JUDGE_PROVIDER", "anthropic")
    monkeypatch.setenv("DAYDREAM_JUDGE_MODEL", "claude-x")
    monkeypatch.setenv("DAYDREAM_JUDGE_API_KEY", "sk-judge")
    monkeypatch.setenv("DAYDREAM_REVIEW_BACKEND", "pi")
    monkeypatch.setenv("DAYDREAM_REVIEW_MODEL", "pi-model")
    monkeypatch.setenv("DAYDREAM_REVIEW_API_KEY", "sk-review")

    job = yaml.safe_load(pkg.render_job_config(oracle=False))
    agent_env = resolve_env_vars(dict(job["agents"][0]["env"]))
    verifier_env = resolve_env_vars(dict(job["verifier"]["env"]))
    assert agent_env["DAYDREAM_REVIEW_API_KEY"] == "sk-review"
    assert agent_env["ANTHROPIC_API_KEY"] == ""          # claude alternative, unused, unset
    assert verifier_env["DAYDREAM_JUDGE_API_KEY"] == "sk-judge"
    assert verifier_env["CLAUDE_CODE_OAUTH_TOKEN"] == ""  # alternative, unused, unset
    assert verifier_env["DAYDREAM_JUDGE_MODEL"] == "claude-x"

def test_render_job_config_still_requires_selection_vars(monkeypatch: pytest.MonkeyPatch,) -> None:
    """Selection vars stay bare: unset provider/model still abort rendering."""

    pytest.importorskip("harbor")

    from harbor.utils.env import resolve_env_vars

    monkeypatch.delenv("DAYDREAM_JUDGE_PROVIDER", raising=False)
    monkeypatch.setenv("DAYDREAM_JUDGE_MODEL", "m")
    job = yaml.safe_load(pkg.render_job_config(oracle=False))
    with pytest.raises(ValueError, match="DAYDREAM_JUDGE_PROVIDER"):
        resolve_env_vars(dict(job["verifier"]["env"]))

def test_build_harbor_refuses_without_ready_workspace(tmp_path: Path, fake_gh: FakeGh) -> None:
    ws, _, _ = _seed_clean_workspace(tmp_path, fake_gh, ready=False)
    wheel, _ = _stub_wheel(tmp_path, content=b"x")
    with pytest.raises(pkg.PackageError) as rejected:
        pkg.build_harbor(ws, wheel=wheel)
    assert "validate" in str(rejected.value).lower() or "ready" in str(rejected.value).lower()

def test_validate_compiled_rejects_missing_harbor_with_remediation(monkeypatch: pytest.MonkeyPatch) -> None:
    def absent(distribution: Any) -> None:
        raise importlib.metadata.PackageNotFoundError(distribution)

    monkeypatch.setattr(importlib.metadata, "version", absent)
    with pytest.raises(pkg.PackageError) as rejected:
        pkg.validate_compiled(None)
    assert "pip install 'daydream[benchmark]'" in str(rejected.value)

def test_validate_compiled_instantiates_harbor_tasks_and_job_configs(tmp_path: Path, fake_gh: FakeGh) -> None:
    pytest.importorskip("harbor")
    from harbor.models.job.config import JobConfig

    try:
        from harbor.models.task import Task
    except ImportError:  # Harbor exposes task as a namespace package in some wheels.
        from harbor.models.task.task import Task

    ws, case_id, _ = _seed_ready_workspace(tmp_path, fake_gh)
    wheel, version = _stub_wheel(tmp_path)
    pkg.build_harbor(ws, wheel=wheel)
    assert pkg.validate_compiled(ws) == 0
    key = build.derive_task_key(case_id)
    case = ws / "harbor" / key
    assert (case / "environment" / wheel.name).read_bytes() == wheel.read_bytes()
    assert (case / "environment/runtime-requirements.lock").read_bytes() == pkg.lock_text().encode()
    lock = json.loads((ws / "harbor/benchmark.lock.json").read_text())
    assert (lock["daydream"]["version"], lock["daydream"]["sha256"]) == (
        version, hashlib.sha256(wheel.read_bytes()).hexdigest(),
    )
    doc = tomllib.loads((case / "task.toml").read_text())
    assert doc["schema_version"] == "1.4"
    assert doc["metadata"] == {"benchmark_case_key": key, "source_kind": "historic-github-pr"}
    assert doc["agent"] == {"timeout_sec": 1800.0, "network_mode": "allowlist", "allowed_hosts": ["h1.example.com"]}
    assert doc["environment"] == {
        "network_mode": "no-network", "build_timeout_sec": 1200.0, "workdir": "/workspace/repo", "cpus": 2,
        "memory_mb": 4096, "storage_mb": 10240,
        "env": {"DAYDREAM_REVIEW_CASE_ID": key, "DAYDREAM_REVIEW_BASE_REF": "base", "DAYDREAM_REVIEW_HEAD_REF": "head"},
    }
    assert doc["verifier"]["timeout_sec"] == 900.0 and doc["verifier"]["environment_mode"] == "separate"
    assert doc["verifier"]["environment"] == {
        "network_mode": "allowlist", "allowed_hosts": ["h2.example.com"], "build_timeout_sec": 1200.0, "cpus": 1,
        "memory_mb": 2048, "storage_mb": 4096,
    }
    assert "dataset.toml" not in (case / "task.toml").read_text() and "registry" not in (case / "task.toml").read_text()
    environment = (case / "environment/Dockerfile").read_text()
    assert environment.startswith("FROM " + pkg.ENV_BASE_IMAGE)
    assert "git clone" in environment and "repository.bundle" in environment and "remote remove" in environment
    assert "checkout" in environment and "base" in environment and "head" in environment
    assert "rm" in environment and "WORKDIR /workspace/repo" in environment
    assert "@earendil-works/pi-coding-agent@1.4.0" in environment
    assert "node --version" in environment and "pi --version" in environment
    assert "--require-hashes" in environment and "--no-deps" in environment
    assert not any(forbidden in environment for forbidden in ("Task.md", "solution/", "tests/score_review", "COPY .."))
    assert "import daydream" not in (case / "tests/score_review.py").read_text()
    assert "import daydream" not in (ws / "harbor/metric.py").read_text()
    verifier = (case / "tests/Dockerfile").read_text()
    assert verifier.startswith("FROM " + pkg.VERIFIER_BASE_IMAGE)
    assert not any(forbidden in verifier for forbidden in ("ENTRYPOINT", "CMD", "/verifier"))
    assert "WORKDIR /tests" in verifier and "test.sh" in verifier and "score_review.py" in verifier
    assert "verifier-metadata.json" in verifier and "httpx==0.28.1" in verifier and "httpx>=" not in verifier
    assert "node-v22." in verifier and "npm ci" in verifier and "package-lock.json" in verifier
    assert "@anthropic-ai/claude-code" in verifier and "2.1.250" in verifier
    job = yaml.safe_load((ws / "harbor/harbor-job.yaml").read_text())
    oracle = yaml.safe_load((ws / "harbor/harbor-oracle.yaml").read_text())
    assert (job["jobs_dir"], job["n_attempts"], job["n_concurrent_trials"]) == ("jobs", 1, 4)
    assert job["environment"] == {"type": "docker", "delete": True}
    assert job["agents"][0]["import_path"] == "daydream.benchmark.harbor.agent:DaydreamReviewAgent"
    assert job["datasets"] == [{"path": "."}]
    assert job["metrics"] == [{"type": "uv-script", "kwargs": {"script_path": "metric.py"}}]
    assert oracle["agents"] == [{"name": "oracle"}]
    assert oracle["verifier"] == job["verifier"] and oracle["metrics"] == job["metrics"]
    tree = _harbor_tree_bytes(ws)
    pkg.build_harbor(ws, wheel=wheel)
    assert _harbor_tree_bytes(ws) == tree
    task = Task(str(case), disable_verification=True)
    # Harbor must retain the network boundary, not merely accept our task schema.
    assert task.config.environment.network_mode == "no-network"
    assert task.config.agent.network_mode == "allowlist"
    assert task.config.agent.allowed_hosts == ["h1.example.com"]
    assert task.config.verifier.environment_mode == "separate"
    assert task.config.verifier.environment is not None
    assert task.config.verifier.environment.network_mode == "allowlist"
    assert task.config.verifier.environment.allowed_hosts == ["h2.example.com"]
    for name in ("harbor-job.yaml", "harbor-oracle.yaml"):
        assert JobConfig.model_validate(yaml.safe_load((ws / "harbor" / name).read_text())) is not None
    (ws / "harbor/harbor-job.yaml").write_text("n_attempts: not-an-int\n")
    with pytest.raises(pkg.PackageError) as rejected:
        pkg.validate_compiled(ws)
    assert "harbor-job.yaml" in str(rejected.value)

def test_audit_execution_proofs_harbor_gated(tmp_path: Path, fake_gh: FakeGh) -> None:
    pytest.importorskip("harbor")
    if shutil.which("docker") is None or subprocess.run(["docker", "info"], capture_output=True).returncode:
        if os.environ.get("DAYDREAM_REQUIRE_DOCKER"):
            pytest.fail("DAYDREAM_REQUIRE_DOCKER is set but the Docker daemon is not available")
        pytest.skip("Docker daemon is not available")

    root = Path(__file__).resolve().parents[1]
    wheels = tmp_path / "wheels"
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(wheels), str(root)], check=True, capture_output=True, text=True,
    )
    ver = importlib.metadata.version("daydream")
    wheel = wheels / f"daydream-{ver}-py3-none-any.whl"
    ws, case_id, _ = _seed_ready_workspace(tmp_path, fake_gh)
    pkg.build_harbor(ws, wheel=wheel)
    key = build.derive_task_key(case_id)
    case = ws / "harbor" / key

    env_df = (case / "environment/Dockerfile").read_text()
    for forbidden in ("Task.md", "solution/", "tests/score_review", "tests/test.sh"):
        assert forbidden not in env_df
    ver_df = (case / "tests/Dockerfile").read_text()
    assert "repository.bundle" not in ver_df and "instruction.md" not in ver_df
    assert "DAYDREAM_REVIEW" not in ver_df

    verifier_tag = f"dd-verifier-{key}"
    environment_tag = f"dd-env-{key}"
    subprocess.run(["docker", "build", "-t", verifier_tag, str(case / "tests")], check=True)
    subprocess.run(["docker", "build", "-t", environment_tag, str(case / "environment")], check=True)
    proof = tmp_path / "packaged-proof"
    harness = proof / "tests/harness"
    harness.mkdir(parents=True)
    (proof / "tests/__init__.py").touch()
    (harness / "__init__.py").touch()
    for name in ("native_pi.py", "harbor_review.py", "packaged_harbor_review.py"):
        shutil.copyfile(root / "tests/harness" / name, harness / name)
    native = subprocess.run([
        "docker", "run", "--rm", "--network", "none", "-v", f"{proof}:/proof:ro", "-w", "/proof",
        environment_tag, "python", "-m", "tests.harness.packaged_harbor_review",
    ], check=True, capture_output=True, text=True)
    assert '"packaged_native_review": "complete"' in native.stdout
    subprocess.run(["docker", "run", "--rm", environment_tag, "sh", "-c",
            "test ! -e /workspace/repo/Task.md && test ! -e /workspace/repo/solution "
            "&& test ! -e /workspace/repo/tests",
        ], check=True,
    )
    subprocess.run(["docker", "run", "--rm", verifier_tag, "sh", "-c",
            "test ! -e /workspace/repo && test ! -e /instruction.md "
            "&& test -z \"${DAYDREAM_REVIEW_API_KEY:-}\"",
        ], check=True,
    )
    container = subprocess.run(
        ["docker", "run", "-d", verifier_tag, "sh", "-c", "sleep infinity"], check=True, capture_output=True, text=True,
    ).stdout.strip()
    try:
        state = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Status}}", container], check=True, capture_output=True, text=True,
        ).stdout.strip()
        assert state == "running"
    finally:
        subprocess.run(["docker", "rm", "-f", container], check=False, capture_output=True)

    rewards = tmp_path / "rewards.jsonl"
    rewards.write_text(json.dumps({"verifier_error": 0, "tp": 1, "fp": 0, "fn": 0, "reward": 1.0, "clean_task": 0
    }) + "\n")
    out = tmp_path / "metric.json"
    subprocess.run(
        ["uv", "run", str(ws / "harbor/metric.py"), "-i", str(rewards), "-o", str(out)], check=True, cwd=ws / "harbor",
    )
    metric = json.loads(out.read_text())
    assert metric["task_count"] == 1 and metric["micro_f1"] == 1.0
