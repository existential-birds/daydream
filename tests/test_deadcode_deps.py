import tomllib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_RL_PROJECT = _ROOT / "rl" / "daydream_review_v1"


def _dev_names(pyproject: Path) -> list[str]:
    data = tomllib.loads(pyproject.read_text())
    return [d.split(">=")[0].split("==")[0] for d in data["dependency-groups"]["dev"]]


def test_vulture_in_root_dev_group():
    assert "vulture" in _dev_names(_ROOT / "pyproject.toml")


def test_vulture_in_rl_dev_group():
    assert "vulture" in _dev_names(_RL_PROJECT / "pyproject.toml")


def _tool_block(pyproject: Path) -> dict:
    return tomllib.loads(pyproject.read_text()).get("tool", {}).get("vulture") or {}


def test_root_vulture_config_pins_conventions():
    cfg = _tool_block(_ROOT / "pyproject.toml")
    assert cfg["min_confidence"] == 80
    assert cfg["exclude"] == ["*/atif/*"]
    assert "silence_ui" in cfg["ignore_names"]
    assert any("console_arg" in n for n in cfg["ignore_names"])


def test_rl_vulture_config_is_scoped_to_own_tree():
    cfg = _tool_block(_RL_PROJECT / "pyproject.toml")
    assert cfg["min_confidence"] == 80
    joined = " ".join(cfg.get("exclude", []))
    assert "daydream/" not in joined  # locality invariant: never reaches into root package
