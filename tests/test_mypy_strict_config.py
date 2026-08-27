"""Regression guard: strict mypy stays enabled, globally tolerant config does not return."""
import tomllib
from pathlib import Path


def _cfg() -> dict:
    return tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))["tool"]["mypy"]


def test_strict_is_enabled():
    assert _cfg()["strict"] is True


def test_no_global_ignore_errors():
    assert "ignore_errors" not in _cfg()


def test_global_ignore_missing_imports_gone():
    assert _cfg().get("ignore_missing_imports") is not True


def test_overrides_are_narrow_and_documented():
    blocks = _cfg().get("overrides", [])
    assert blocks, "boundary overrides must exist"
    for b in blocks:
        mods = b["module"]
        assert isinstance(mods, list), "each override names an explicit module list"
        assert len(mods) <= 6, f"override too broad: {mods}"


def test_types_jsonschema_in_dev_group():
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    assert any(s.startswith("types-jsonschema") for s in data["dependency-groups"]["dev"])
