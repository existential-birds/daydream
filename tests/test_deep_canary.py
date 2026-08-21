"""Deterministic deep-sharding + sibling-frontier coverage canary (issue #763).

Real-path, model-free tests proving that enabled deep sharding and
sibling-frontier (``dependency_frontier_read``) coverage bind and hold their
regression invariants through the real ``runner.run`` pipeline, using the
cross-importing ``sibling_frontier_target`` fixture.
"""

from __future__ import annotations

import re

from tests.harness.git_helpers import git as _git
from tests.harness.stub_backend import install_stub_backend


def _numeric(name: str) -> tuple[int, ...]:
    """Natural sort: mod10.py orders after mod9.py (lexicographic would place
    mod10 before mod2, diverging from the plan's numeric ordering)."""
    return tuple(
        int(p) if p.isdigit() else float("inf")
        for p in re.split(r"(\d+)", name)
        if p
    )


def test_sibling_frontier_target_shape(sibling_frontier_target: Path) -> None:
    """The canary fixture has 13 changed python files, all importing core.py."""
    repo = sibling_frontier_target
    pys = sorted((p.name for p in repo.glob("*.py")), key=_numeric)
    assert pys == ["core.py"] + [f"mod{i}.py" for i in range(12)]
    # All files changed on the feature branch (diff vs main is non-empty).
    changed = set(
        _git(repo, "diff", "--name-only", "main..HEAD").splitlines()
    )
    assert changed == {p.name for p in repo.glob("*.py")}
    # Real, tree-sitter-parseable cross-file import edges are physically present.
    assert all(
        "from core import core_helper" in (repo / f"mod{i}.py").read_text()
        for i in range(12)
    )