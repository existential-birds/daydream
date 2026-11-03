"""Ground diagram eligibility in real Git diffs, stack/service discovery, and imports.

Only two native-parser failure injections are mocked, to verify graceful fallback."""

from __future__ import annotations

from pathlib import Path

import pytest

from daydream._tree_sitter_safety import TreeSitterBadVersionError
from daydream.config_file import DaydreamFileConfig
from daydream.deep import diagram_trigger
from daydream.deep.dependency import build_import_graph
from daydream.deep.detection import StackAssignment, detect_stacks
from daydream.deep.diagram_trigger import (
    DiagramThresholds,
    Eligibility,
    count_function_branch_points,
    decide_eligibility,
)
from daydream.hunk_index import head_side_ranges_by_file, parse_hunks
from daydream.services import Service, enumerate_services
from tests.harness.git_helpers import commit, git, init_repo

PYTHON_SOURCE = """\
def outer(a, b):
    if a:
        return 1
    elif b:
        for x in range(3):
            print(x)
    else:
        while b:
            b -= 1
    def inner(c):
        if c:
            return 2
        try:
            pass
        except ValueError:
            pass
        return 3
    return inner(a)
"""

TYPESCRIPT_SOURCE = """\
export function handle(a: number, b: number): number {
  if (a > 0) {
    return 1;
  } else if (b > 0) {
    switch (b) {
      case 1:
        return 2;
      default:
        break;
    }
  } else {
    for (const x of [1, 2]) {
      console.log(x);
    }
  }
  return 0;
}
"""

GO_SOURCE = """\
package main

func Handle(a int, b int) int {
\tif a > 0 {
\t\treturn 1
\t} else if b > 0 {
\t\tswitch b {
\t\tcase 1:
\t\t\treturn 2
\t\tdefault:
\t\t\treturn 3
\t\t}
\t}
\tfor i := 0; i < 3; i++ {
\t\ta += i
\t}
\treturn a
}
"""

RUST_SOURCE = """\
pub fn handle(a: i32, b: i32) -> i32 {
    if a > 0 {
        return 1;
    } else if b > 0 {
        match b {
            1 => return 2,
            _ => return 3,
        }
    }
    for i in 0..3 {
        let _ = i;
    }
    while a > 0 {
        return a;
    }
    0
}
"""


def _write(repo: Path, files: dict[str, str]) -> None:
    """Write every ``path: text`` under ``repo``, creating parents."""
    for path, text in files.items():
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)


def _diff_repo(
    repo: Path, base: dict[str, str], head: dict[str, str]
) -> tuple[list[str], dict[str, list[tuple[int, int]]]]:
    """Commit base/head files with a seed, then parse the real diff into hunk ranges."""
    init_repo(repo)
    _write(repo, {"seed.txt": "seed\n", **base})
    git(repo, "add", ".")
    commit(repo, "base")
    _write(repo, head)
    git(repo, "add", ".")
    commit(repo, "change")
    parsed = parse_hunks(git(repo, "diff", "HEAD~1", "HEAD"))
    return sorted(parsed), head_side_ranges_by_file(parsed)


def _decide(
    repo: Path, changed_files: list[str], hunk_ranges: dict[str, list[tuple[int, int]]], *,
    services: list[Service] | None = None, stacks: list[StackAssignment] | None = None,
    thresholds: DiagramThresholds | None = None, force: str = "auto",
) -> Eligibility:
    """Call ``decide_eligibility`` with real stacks and a real import graph."""
    return decide_eligibility(
        repo_root=repo, changed_files=changed_files, hunk_ranges=hunk_ranges,
        stacks=detect_stacks(sorted(changed_files), target_dir=repo) if stacks is None else stacks,
        services=[] if services is None else services, import_graph=build_import_graph(sorted(changed_files), repo),
        thresholds=DiagramThresholds() if thresholds is None else thresholds, force=force,
    )


# --- Sequence eligibility ---------------------------------------------------

def test_single_module_diff_is_below_the_sequence_threshold(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    head = {
        "core/a.py": "from core.b import b\n\n\ndef a():\n    return b()\n",
        "core/b.py": "def b():\n    return 2\n",
        "core/c.py": "def c():\n    return 3\n",
        "core/test_c.py": "def test_c():\n    assert True\n",
    }
    changed, ranges = _diff_repo(repo, {}, head)
    eligibility = _decide(repo, changed, ranges)

    assert eligibility.code_files == ["core/a.py", "core/b.py", "core/c.py"], "changed tests are not code files"

    assert set(eligibility.modules.values()) == {"core"}
    assert eligibility.cross_module_edges == 0
    assert not eligibility.sequence.eligible
    assert eligibility.sequence.rule is None
    assert "1 module(s), 0 cross-module edge(s)" in eligibility.sequence.reason
    assert eligibility.eligible_kinds() == []

# --- Flowchart eligibility and branch counting ------------------------------

def test_multilingual_function_inventory_preserves_inner_ownership_thresholds_and_tie_order(tmp_path: Path) -> None:
    """One native sweep sorts real language siblings without counting nested branches twice."""
    two_branches = "def {name}(a, b):\n    if a:\n        return 1\n    if b:\n        return 2\n    return 3\n"
    head = {
        "python/sample.py": PYTHON_SOURCE,
        "python/ties.py": two_branches.format(name="first") + "\n\n" + two_branches.format(name="second"),
        "python/plain.py": (
            "def cheap():\n    return 1\n\n\ndef weak(value):\n    if value:\n        return 1\n    return 2\n"
        ),
        "typescript/sample.ts": TYPESCRIPT_SOURCE,
        "go/sample.go": GO_SOURCE,
        "rust/sample.rs": RUST_SOURCE,
    }
    repo = tmp_path / "repo"
    changed, ranges = _diff_repo(repo, {}, head)
    eligibility = _decide(repo, changed, ranges, thresholds=DiagramThresholds(min_branch_points=2))
    assert [(root.file, root.name, root.line, root.branch_points) for root in eligibility.function_branch_counts] == [
        ("rust/sample.rs", "handle", 1, 6), ("go/sample.go", "Handle", 3, 5),
        ("typescript/sample.ts", "handle", 1, 5), ("python/sample.py", "outer", 1, 4),
        ("python/sample.py", "inner", 10, 3), ("python/ties.py", "first", 1, 2),
        ("python/ties.py", "second", 9, 2), ("python/plain.py", "weak", 5, 1),
    ]
    assert [(root.name, root.end_line) for root in eligibility.function_branch_counts
            if root.file == "python/sample.py"] == [("outer", 18), ("inner", 17)]
    # The native sweep retains zero-branch definitions; persisted diagnostics omit
    # zero but retain the one-branch sibling below the two-branch candidate floor.
    assert [(root.name, root.branch_points) for root in
            count_function_branch_points(repo, "python/plain.py", ranges["python/plain.py"])] == [
        ("cheap", 0), ("weak", 1),
    ]
    assert eligibility.candidate_roots == eligibility.function_branch_counts[:-1]
    assert eligibility.code_files == sorted(head)
    assert eligibility.flowchart.rule == "branch-points" and eligibility.flowchart.eligible
    assert "most: 6 in handle" in eligibility.flowchart.reason
    assert eligibility.eligible_kinds() == ["flowchart"]

def test_branch_points_count_only_supported_changed_functions(tmp_path: Path) -> None:
    """Unsupported languages, pure deletions and missing files contribute nothing; closures own their branches."""
    repo = tmp_path / "repo"
    (repo / "api").mkdir(parents=True)
    (repo / "api/sample.rb").write_text("def handle(a)\n  if a\n    1\n  end\nend\n")
    assert count_function_branch_points(repo, "api/sample.rb", [(1, 5)]) == []
    (repo / "api/sample.py").write_text(PYTHON_SOURCE)

    # Lines 1-5 hold ``if`` (2), ``elif`` (4) and ``for`` (5); ``inner``'s range
    # (10-17) does not overlap, so it is not a changed function at all.
    roots = count_function_branch_points(repo, "api/sample.py", [(1, 5)])
    assert [(root.name, root.branch_points) for root in roots] == [("outer", 3)]

    # A hunk confined to the closure leaves ``outer`` a changed function with
    # zero changed branch points of its own.
    roots = count_function_branch_points(repo, "api/sample.py", [(10, 17)])
    assert [(root.name, root.branch_points) for root in roots] == [("outer", 0), ("inner", 3)]
    # A file whose only hunk was a pure deletion contributes nothing.
    assert count_function_branch_points(repo, "api/sample.py", []) == []
    assert count_function_branch_points(tmp_path, "api/gone.py", [(1, 10)]) == []


def test_forcing_and_tree_sitter_failures_fail_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Force flags override thresholds; known-bad installs, query errors and budget exhaustion drop roots only."""
    repo = tmp_path / "repo"
    changed, ranges = _diff_repo(repo, {}, {"api/sample.py": PYTHON_SOURCE})
    forced = _decide(repo, changed, ranges, force="both")
    assert forced.eligible_kinds() == ["sequence", "flowchart"]
    assert forced.sequence.rule == forced.flowchart.rule == "forced"
    # Forcing with no function over the threshold still offers every changed function as a root.
    unqualified = _decide(repo, changed, ranges, force="both", thresholds=DiagramThresholds(min_branch_points=99))
    assert unqualified.candidate_roots and unqualified.flowchart.rule == "forced"

    eligibility = _decide(repo, changed, ranges, force="off")
    assert eligibility.eligible_kinds() == []
    assert eligibility.sequence == eligibility.flowchart
    assert eligibility.sequence.rule is None
    assert eligibility.sequence.reason == "Diagram mode is off."
    assert eligibility.candidate_roots == []
    assert eligibility.function_branch_counts == []
    # Signals that cost no parsing are still recorded for the artifact.
    assert eligibility.code_files == ["api/sample.py"]

    assert _decide(repo, changed, ranges).candidate_roots
    with monkeypatch.context() as patch:
        def _bad_install(*_args: object, **_kwargs: object) -> list[dict[str, object]]:
            raise TreeSitterBadVersionError("installed tree-sitter is known-bad")

        patch.setattr(diagram_trigger, "definitions_in_file", _bad_install)
        bad = _decide(repo, changed, ranges)
        assert bad.candidate_roots == [] and not bad.flowchart.eligible
    with monkeypatch.context() as patch:
        def _query_error(*_args: object, **_kwargs: object) -> list[int]:
            raise ValueError("query blew up")

        patch.setattr(diagram_trigger, "branch_statement_lines", _query_error)
        assert _decide(repo, changed, ranges).candidate_roots == []
    monkeypatch.setattr(diagram_trigger, "_BRANCH_COUNT_WALL_BUDGET_S", -1.0)
    assert _decide(repo, changed, ranges).candidate_roots == []


# --- Artifact form ----------------------------------------------------------

# --- Service ownership --------------------------------------------------------

def test_service_ownership_is_the_deepest_root_and_a_repo_root_service_owns_nothing(tmp_path: Path,) -> None:
    repo = tmp_path / "repo"
    base = {
        "services/api/pyproject.toml": "[project]\nname = 'api'\n",
        "services/api/inner/pyproject.toml": "[project]\nname = 'inner'\n",
    }
    head = {
        "services/api/handler.py": "def handle():\n    return 1\n",
        "services/api/inner/main.py": "def main():\n    return 2\n",
        "scripts/tool.py": "def tool():\n    return 3\n",
    }
    changed, ranges = _diff_repo(repo, base, head)
    services = enumerate_services(
        repo, DaydreamFileConfig(improve_service_roots=["services/*", "services/api/inner", "."])
    )

    eligibility = _decide(repo, changed, ranges, services=services)

    assert eligibility.services == {"services/api/handler.py": "api", "services/api/inner/main.py": "inner"}
    assert eligibility.modules == {
        "scripts/tool.py": "scripts", "services/api/handler.py": "services/api",
        "services/api/inner/main.py": "services/api/inner",
    }
