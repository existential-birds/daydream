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
