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
