"""Pin enforcement: tree-sitter must stay at the last known-good release.

py-tree-sitter#472 (native Point getter use-after-free, SIGSEGV) ships in
0.26.0; the in-repo guard in daydream/_tree_sitter_safety.py blocks it. The
upstream fix (#466) is merged but unreleased, so the pin cannot be loosened
until a fixed version lands on PyPI.
"""

import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _tree_sitter_requirement() -> str:
    deps = tomllib.loads(PYPROJECT.read_text())["project"]["dependencies"]
    return str(next(d for d in deps if d.startswith("tree-sitter")))


def test_tree_sitter_is_pinned_to_known_good_version() -> None:
    assert _tree_sitter_requirement() == "tree-sitter==0.25.2"
