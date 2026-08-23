"""Packaging and Harbor 0.21 integration tests for compiled benchmarks."""

from pathlib import Path


def test_benchmark_extra_pins_harbor_021_and_not_base():
    import tomllib

    root = Path(__file__).resolve().parents[1]
    data = tomllib.loads((root / "pyproject.toml").read_text())
    deps = data["project"]["dependencies"]
    assert "harbor" not in " ".join(deps)
    extra = data["project"]["optional-dependencies"]["benchmark"]
    assert "harbor>=0.21,<0.22" in extra
    include = data["tool"]["hatch"]["build"]["targets"]["wheel"]["include"]
    assert "daydream/benchmark/harbor/templates/**" in include
    assert "daydream/benchmark/harbor/runtime-requirements.lock" in include


def test_runtime_lock_header_and_render(tmp_path, monkeypatch):
    import hashlib
    import importlib.metadata

    from daydream.benchmark.harbor import package as pkg
    from daydream.benchmark.harbor.build import TEMPLATE_VERSION

    ver = importlib.metadata.version("daydream")
    uv_lock = tmp_path / "uv.lock"
    uv_lock.write_text("LOCKBODY\n")
    monkeypatch.setattr(pkg, "_uv_export_body", lambda path: "httpx==0.28.1 \\\n    --hash=sha256:abc\n")
    header, body = pkg.render_runtime_lock(uv_lock, daydream_version=ver)
    assert f"daydream=={ver}" not in body
    assert "--hash=sha256:" in body
    assert "uv export --frozen --no-dev --no-emit-project" in header
    assert f"template_version: {TEMPLATE_VERSION}" in header
    assert f"source_uv_lock_sha256: {hashlib.sha256(uv_lock.read_bytes()).hexdigest()}" in header
    assert f"daydream_version: {ver}" in header


def test_runtime_lock_regeneration_is_noop_on_unchanged(tmp_path):
    import importlib.metadata

    from daydream.benchmark.harbor import package as pkg

    root = Path(__file__).resolve().parents[1]
    repo = tmp_path / "repo"
    repo.mkdir()
    uv_lock = repo / "uv.lock"
    uv_lock.write_bytes((root / "uv.lock").read_bytes())
    (repo / "pyproject.toml").write_bytes((root / "pyproject.toml").read_bytes())
    ver = importlib.metadata.version("daydream")
    committed = (root / "daydream/benchmark/harbor/runtime-requirements.lock").read_bytes()
    regenerated = pkg.generate_runtime_lock(uv_lock, daydream_version=ver)
    assert regenerated == committed
    uv_lock.write_bytes(uv_lock.read_bytes() + b"\n# drift\n")
    regenerated2 = pkg.generate_runtime_lock(uv_lock, daydream_version=ver)
    assert regenerated2 != committed


def test_validate_wheel_accepts_matching_and_rejects_mismatch(tmp_path):
    import importlib.metadata

    import pytest

    from daydream.benchmark.harbor import package as pkg

    ver = importlib.metadata.version("daydream")
    good = tmp_path / f"daydream-{ver}-py3-none-any.whl"
    good.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
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


def test_resolve_harbor_checks_same_interpreter_and_version(monkeypatch):
    import importlib.metadata
    import sys

    import pytest

    from daydream.benchmark.harbor import package as pkg

    monkeypatch.setattr(pkg.importlib.metadata, "version", lambda d: "0.21.0")
    exe = pkg.resolve_harbor()
    assert exe == str(Path(sys.executable).parent / "harbor")

    def absent(distribution):
        raise importlib.metadata.PackageNotFoundError(distribution)

    monkeypatch.setattr(pkg.importlib.metadata, "version", absent)
    with pytest.raises(pkg.PackageError) as missing:
        pkg.resolve_harbor()
    assert "pip install 'daydream[benchmark]'" in str(missing.value)

    monkeypatch.setattr(pkg.importlib.metadata, "version", lambda d: "0.20.0")
    with pytest.raises(pkg.PackageError) as wrong:
        pkg.resolve_harbor()
    assert "[0.21, 0.22)" in str(wrong.value)


def test_render_task_toml_matches_plan_s8():
    import tomllib

    from daydream.benchmark.harbor import package as pkg

    text = pkg.render_task_toml("case-4f7c81d922a0").decode()
    doc = tomllib.loads(text)
    assert doc["schema_version"] == "1.4"
    assert doc["metadata"] == {
        "benchmark_case_key": "case-4f7c81d922a0",
        "source_kind": "historic-github-pr",
    }
    assert doc["agent"] == {
        "timeout_sec": 1800.0,
        "network_mode": "allowlist",
        "allowed_hosts": ["api.anthropic.com"],
    }
    env = doc["environment"]
    assert env == {
        "network_mode": "no-network",
        "build_timeout_sec": 1200.0,
        "workdir": "/workspace/repo",
        "cpus": 2,
        "memory_mb": 4096,
        "storage_mb": 10240,
    }
    assert doc["verifier"]["timeout_sec"] == 900.0
    assert doc["verifier"]["environment_mode"] == "separate"
    assert doc["verifier"]["environment"] == {
        "network_mode": "allowlist",
        "allowed_hosts": ["api.anthropic.com"],
        "build_timeout_sec": 1200.0,
        "cpus": 1,
        "memory_mb": 2048,
        "storage_mb": 4096,
    }
    assert "dataset.toml" not in text and "registry" not in text


def test_render_environment_dockerfile_clones_bundle_no_remote():
    from daydream.benchmark.harbor import package as pkg

    dockerfile = pkg.render_environment_dockerfile(
        base_image=pkg.ENV_BASE_IMAGE, daydream_version="0.27.0"
    ).decode()
    assert dockerfile.startswith("FROM " + pkg.ENV_BASE_IMAGE)
    assert "git clone" in dockerfile and "repository.bundle" in dockerfile
    assert "/workspace/repo" in dockerfile
    assert "checkout" in dockerfile and "base" in dockerfile and "head" in dockerfile
    assert "rm" in dockerfile and "repository.bundle" in dockerfile
    assert "WORKDIR /workspace/repo" in dockerfile
    assert "remote remove" in dockerfile
    assert "--require-hashes" in dockerfile and "--no-deps" in dockerfile
    for forbidden in ("Task.md", "solution/", "tests/score_review", "COPY .."):
        assert forbidden not in dockerfile


def test_verifier_dockerfile_is_entrypoint_free_and_digest_pinned():
    from daydream.benchmark.harbor import package as pkg

    text = pkg.render_verifier_dockerfile(base_image=pkg.VERIFIER_BASE_IMAGE).decode()
    assert text.startswith("FROM " + pkg.VERIFIER_BASE_IMAGE)
    assert "ENTRYPOINT" not in text and "CMD" not in text
    assert "/verifier" not in text
    assert "test.sh" in text and "score_review.py" in text
    assert "httpx" in text and "httpx>=" not in text and "httpx==0.28.1" in text


def test_render_job_config_matches_plan_s8_and_oracle_differs():
    import yaml

    from daydream.benchmark.harbor import package as pkg

    job = yaml.safe_load(pkg.render_job_config(oracle=False))
    assert job["jobs_dir"] == "jobs" and job["n_attempts"] == 1
    assert job["n_concurrent_trials"] == 4
    assert job["environment"] == {"type": "docker", "delete": True}
    agent = job["agents"][0]
    assert agent["import_path"] == "daydream.benchmark.harbor.agent:DaydreamReviewAgent"
    assert "DAYDREAM_REVIEW_API_KEY" in agent["env"]
    assert job["datasets"] == [{"path": "."}]
    assert job["metrics"] == [{"type": "uv-script", "kwargs": {"script_path": "metric.py"}}]
    assert "DAYDREAM_JUDGE_API_KEY" in job["verifier"]["env"]

    oracle = yaml.safe_load(pkg.render_job_config(oracle=True))
    assert oracle["agents"] == [{"name": "oracle"}]
    assert oracle["verifier"] == job["verifier"]
    assert oracle["metrics"] == job["metrics"]


def test_compile_with_wheel_emits_full_packaged_tree(tmp_path, fake_gh):
    import importlib.metadata

    from daydream.benchmark.harbor import build
    from daydream.benchmark.harbor import package as pkg
    from tests.test_benchmark_harbor_build import _harbor_tree_bytes, _seed_ready_workspace

    ws, case_id, _ = _seed_ready_workspace(tmp_path, fake_gh)
    ver = importlib.metadata.version("daydream")
    wheel = tmp_path / f"daydream-{ver}-py3-none-any.whl"
    wheel.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    lock = build.compile_workspace(ws, wheel=wheel)
    key = build.derive_task_key(case_id)
    case = ws / "harbor" / key
    assert (case / "task.toml").is_file()
    assert (case / "environment/Dockerfile").is_file()
    assert (case / "environment/runtime-requirements.lock").is_file()
    assert (case / "environment" / wheel.name).read_bytes() == wheel.read_bytes()
    assert (ws / "harbor/harbor-job.yaml").is_file()
    assert (ws / "harbor/harbor-oracle.yaml").is_file()
    assert lock["daydream"]["version"] == ver
    assert lock["daydream"]["sha256"] == pkg.validate_wheel(wheel, daydream_version=ver).sha256
    tree1 = _harbor_tree_bytes(ws)
    build.compile_workspace(ws, wheel=wheel)
    assert _harbor_tree_bytes(ws) == tree1


def test_build_harbor_refuses_without_ready_workspace(tmp_path, fake_gh):
    import importlib.metadata

    import pytest

    from daydream.benchmark.harbor import package as pkg
    from tests.test_benchmark_harbor_build import _seed_clean_workspace

    ws, _, _ = _seed_clean_workspace(tmp_path, fake_gh, ready=False)
    ver = importlib.metadata.version("daydream")
    wheel = tmp_path / f"daydream-{ver}-py3-none-any.whl"
    wheel.write_bytes(b"x")
    with pytest.raises(pkg.PackageError) as rejected:
        pkg.build_harbor(ws, wheel=wheel)
    assert "validate" in str(rejected.value).lower() or "ready" in str(rejected.value).lower()


def test_validate_compiled_rejects_missing_harbor_with_remediation(monkeypatch):
    import importlib.metadata

    import pytest

    from daydream.benchmark.harbor import package as pkg

    def absent(distribution):
        raise importlib.metadata.PackageNotFoundError(distribution)

    monkeypatch.setattr(pkg.importlib.metadata, "version", absent)
    with pytest.raises(pkg.PackageError) as rejected:
        pkg.validate_compiled(None)
    assert "pip install 'daydream[benchmark]'" in str(rejected.value)


def test_validate_compiled_instantiates_harbor_tasks_and_job_configs(tmp_path, fake_gh):
    import importlib.metadata

    import pytest
    import yaml

    pytest.importorskip("harbor")
    from harbor.models.job.config import JobConfig
    try:
        from harbor.models.task import Task
    except ImportError:  # Harbor 0.21 wheel exposes task as a namespace package.
        from harbor.models.task.task import Task

    from daydream.benchmark.harbor import build
    from daydream.benchmark.harbor import package as pkg
    from tests.test_benchmark_harbor_build import _seed_ready_workspace

    ws, case_id, _ = _seed_ready_workspace(tmp_path, fake_gh)
    ver = importlib.metadata.version("daydream")
    wheel = tmp_path / f"daydream-{ver}-py3-none-any.whl"
    wheel.write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    pkg.build_harbor(ws, wheel=wheel)
    assert pkg.validate_compiled(ws) == 0
    case = ws / "harbor" / build.derive_task_key(case_id)
    assert Task(str(case), disable_verification=True) is not None
    for name in ("harbor-job.yaml", "harbor-oracle.yaml"):
        assert JobConfig.model_validate(yaml.safe_load((ws / "harbor" / name).read_text())) is not None
    (ws / "harbor/harbor-job.yaml").write_text("n_attempts: not-an-int\n")
    with pytest.raises(pkg.PackageError) as rejected:
        pkg.validate_compiled(ws)
    assert "harbor-job.yaml" in str(rejected.value)


def test_templates_and_lock_readable_via_importlib_resources():
    import importlib.resources

    from daydream.benchmark.harbor import package as pkg

    assert pkg.template_text("tests/Dockerfile")
    assert pkg.template_text("environment/Dockerfile")
    assert pkg.lock_text()
    resource = importlib.resources.files("daydream.benchmark.harbor.templates").joinpath(
        "tests/Dockerfile"
    )
    assert "FROM" in resource.read_text()


def test_audit_execution_proofs_harbor_gated(tmp_path, fake_gh):
    import importlib.metadata
    import json
    import subprocess

    import pytest

    pytest.importorskip("harbor")
    from daydream.benchmark.harbor import build
    from daydream.benchmark.harbor import package as pkg
    from tests.test_benchmark_harbor_build import _seed_ready_workspace

    root = Path(__file__).resolve().parents[1]
    wheels = tmp_path / "wheels"
    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(wheels), str(root)],
        check=True,
        capture_output=True,
        text=True,
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
    subprocess.run(
        [
            "docker", "run", "--rm", environment_tag, "sh", "-c",
            "test ! -e /workspace/repo/Task.md && test ! -e /workspace/repo/solution "
            "&& test ! -e /workspace/repo/tests",
        ],
        check=True,
    )
    subprocess.run(
        [
            "docker", "run", "--rm", verifier_tag, "sh", "-c",
            "test ! -e /workspace/repo && test ! -e /instruction.md "
            "&& test -z \"${DAYDREAM_REVIEW_API_KEY:-}\"",
        ],
        check=True,
    )
    container = subprocess.run(
        ["docker", "run", "-d", verifier_tag, "sh", "-c", "sleep infinity"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    try:
        state = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Status}}", container],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert state == "running"
    finally:
        subprocess.run(["docker", "rm", "-f", container], check=False, capture_output=True)

    rewards = tmp_path / "rewards.jsonl"
    rewards.write_text(json.dumps({
        "verifier_error": 0, "tp": 1, "fp": 0, "fn": 0, "reward": 1.0, "clean_task": 0
    }) + "\n")
    out = tmp_path / "metric.json"
    subprocess.run(
        ["uv", "run", str(ws / "harbor/metric.py"), "-i", str(rewards), "-o", str(out)],
        check=True,
        cwd=ws / "harbor",
    )
    metric = json.loads(out.read_text())
    assert metric["task_count"] == 1 and metric["micro_f1"] == 1.0
