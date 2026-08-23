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
