import tomllib
from pathlib import Path


def test_pyyaml_is_a_base_runtime_dependency():
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    deps = data["project"]["dependencies"]
    dev = data["dependency-groups"]["dev"]
    assert any(d == "pyyaml>=6.0" or d.startswith("pyyaml") for d in deps)
    assert "types-pyyaml>=6.0" in dev  # type stubs stay dev-only
