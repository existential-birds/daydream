"""Grounded-diagram eligibility (issue #1113).

Covers ``daydream.deep.diagram_trigger`` against real inputs: a real temporary
git repository, a real ``git diff`` parsed by the production hunk index, real
``detect_stacks`` routing, real ``enumerate_services`` discovery and a real
``build_import_graph``. Nothing here is mocked except the two failure
injections that prove the tree-sitter path fails open, because the whole point
of this module is that a diagram decision is reproducible from facts about the
tree rather than from anything a model said.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from daydream._tree_sitter_safety import TreeSitterBadVersionError
from daydream.config_file import DaydreamFileConfig
from daydream.deep import diagram_trigger
from daydream.deep.dependency import build_import_graph
from daydream.deep.detection import StackAssignment, detect_stacks
from daydream.deep.diagram_trigger import (
    CandidateRoot,
    DiagramThresholds,
    Eligibility,
    count_function_branch_points,
    decide_eligibility,
)
from daydream.deep.orchestrator import _collapse_stacks_for_tiny_diff
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
    """Build a real two-commit repo and return its changed files and hunk ranges.

    ``base`` is committed first (plus an unrelated seed file so the base commit
    is never empty), then ``head`` is written over it and committed. The
    returned ranges come from the production ``parse_hunks`` /
    ``head_side_ranges_by_file`` pair over a real ``git diff``.
    """
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
    repo: Path,
    changed_files: list[str],
    hunk_ranges: dict[str, list[tuple[int, int]]],
    *,
    services: list[Service] | None = None,
    stacks: list[StackAssignment] | None = None,
    thresholds: DiagramThresholds | None = None,
    force: str = "auto",
) -> Eligibility:
    """Call ``decide_eligibility`` with real stacks and a real import graph."""
    return decide_eligibility(
        repo_root=repo,
        changed_files=changed_files,
        hunk_ranges=hunk_ranges,
        stacks=detect_stacks(sorted(changed_files)) if stacks is None else stacks,
        services=[] if services is None else services,
        import_graph=build_import_graph(sorted(changed_files), repo),
        thresholds=DiagramThresholds() if thresholds is None else thresholds,
        force=force,
    )


# --- Sequence eligibility ---------------------------------------------------


def test_cross_module_rule_fires_on_a_real_import_edge(tmp_path: Path) -> None:
    """Three code files across two modules with one cross-module import edge."""
    repo = tmp_path / "repo"
    head = {
        "api/handler.py": "from core.engine import run\n\n\ndef handle(x):\n    return run(x)\n",
        "core/engine.py": "def run(x):\n    return x + 1\n",
        "core/util.py": "def util(x):\n    return x\n",
    }
    changed, ranges = _diff_repo(repo, {}, head)
    eligibility = _decide(repo, changed, ranges)

    assert eligibility.code_files == ["api/handler.py", "core/engine.py", "core/util.py"]
    assert eligibility.modules == {
        "api/handler.py": "api",
        "core/engine.py": "core",
        "core/util.py": "core",
    }
    assert eligibility.cross_module_edges == 1
    assert eligibility.sequence.eligible
    assert eligibility.sequence.rule == "cross-module"
    assert "2 modules" in eligibility.sequence.reason
    assert eligibility.eligible_kinds() == ["sequence"]


def test_cross_service_rule_fires_without_any_import_edge(tmp_path: Path) -> None:
    """Two services and no import edge still qualify: HTTP boundaries have no edge."""
    repo = tmp_path / "repo"
    # The service manifests are pre-existing, not part of the change: a
    # ``pyproject.toml`` in the diff would itself route to the python stack.
    base = {
        "services/a/pyproject.toml": "[project]\nname = 'a'\n",
        "services/b/pyproject.toml": "[project]\nname = 'b'\n",
    }
    head = {
        "services/a/main.py": "def a():\n    return 1\n",
        "services/b/main.py": "def b():\n    return 2\n",
    }
    changed, ranges = _diff_repo(repo, base, head)
    services = enumerate_services(repo, DaydreamFileConfig())
    assert [service.name for service in services] == ["a", "b"]

    eligibility = _decide(repo, changed, ranges, services=services)

    # Only two code files (below min_code_files=3) and zero import edges, so the
    # cross-module rule cannot be what fired.
    assert eligibility.code_files == ["services/a/main.py", "services/b/main.py"]
    assert eligibility.cross_module_edges == 0
    assert eligibility.services == {"services/a/main.py": "a", "services/b/main.py": "b"}
    assert eligibility.modules == {
        "services/a/main.py": "services/a",
        "services/b/main.py": "services/b",
    }
    assert eligibility.sequence.rule == "cross-service"
    assert "2 services (a, b)" in eligibility.sequence.reason


def test_single_module_diff_is_below_the_sequence_threshold(tmp_path: Path) -> None:
    """Three files in one module with an import edge is not a cross-module change."""
    repo = tmp_path / "repo"
    head = {
        "core/a.py": "from core.b import b\n\n\ndef a():\n    return b()\n",
        "core/b.py": "def b():\n    return 2\n",
        "core/c.py": "def c():\n    return 3\n",
    }
    changed, ranges = _diff_repo(repo, {}, head)
    eligibility = _decide(repo, changed, ranges)

    assert set(eligibility.modules.values()) == {"core"}
    assert eligibility.cross_module_edges == 0
    assert not eligibility.sequence.eligible
    assert eligibility.sequence.rule is None
    assert "1 module(s), 0 cross-module edge(s)" in eligibility.sequence.reason
    assert eligibility.eligible_kinds() == []


def test_lowered_thresholds_admit_a_two_file_cross_module_change(tmp_path: Path) -> None:
    """The cross-module floors are thresholds, not constants."""
    repo = tmp_path / "repo"
    head = {
        "api/handler.py": "from core.engine import run\n\n\ndef handle(x):\n    return run(x)\n",
        "core/engine.py": "def run(x):\n    return x\n",
    }
    changed, ranges = _diff_repo(repo, {}, head)

    assert not _decide(repo, changed, ranges).sequence.eligible
    lowered = _decide(
        repo, changed, ranges, thresholds=DiagramThresholds(min_code_files=2, min_modules=2)
    )
    assert lowered.sequence.rule == "cross-module"
    assert lowered.cross_module_edges == 1


def test_root_level_file_module_is_dot(tmp_path: Path) -> None:
    """A repository-root code file's module is ``"."``, never the empty string."""
    repo = tmp_path / "repo"
    head = {"main.py": "def main():\n    return 1\n", "api/a.py": "def a():\n    return 1\n"}
    changed, ranges = _diff_repo(repo, {}, head)
    eligibility = _decide(repo, changed, ranges)
    assert eligibility.modules == {"main.py": ".", "api/a.py": "api"}


def test_tests_and_docs_are_not_code_files(tmp_path: Path) -> None:
    """Markdown lands in the generic bucket and test paths are excluded outright."""
    repo = tmp_path / "repo"
    head = {
        "api/handler.py": "def handle():\n    return 1\n",
        "tests/test_handler.py": "def test_handle():\n    assert True\n",
        "api/__tests__/helper.py": "def helper():\n    return 1\n",
        "api/handler.spec.ts": "it('x', () => {});\n",
        "README.md": "# docs\n",
    }
    changed, ranges = _diff_repo(repo, {}, head)
    eligibility = _decide(repo, changed, ranges)
    assert eligibility.code_files == ["api/handler.py"]


def test_structure_and_generic_stacks_never_contribute_code_files(tmp_path: Path) -> None:
    """A docs-only diff has no code files even though ``structure`` holds every file."""
    repo = tmp_path / "repo"
    changed, ranges = _diff_repo(repo, {}, {"docs/guide.md": "# guide\n"})
    stacks = detect_stacks(sorted(changed))
    assert "docs/guide.md" in {path for stack in stacks for path in stack.files}
    assert _decide(repo, changed, ranges).code_files == []


def test_sharded_stack_names_classify_like_unsharded_ones(tmp_path: Path) -> None:
    """A ``python#2`` shard is still the python stack."""
    repo = tmp_path / "repo"
    changed, ranges = _diff_repo(repo, {}, {"api/a.py": "def a():\n    return 1\n"})
    sharded = [StackAssignment(stack_name="python#2", files=["api/a.py"])]
    assert _decide(repo, changed, ranges, stacks=sharded).code_files == ["api/a.py"]


def test_cross_module_edges_are_counted_per_direction(tmp_path: Path) -> None:
    """A mutual import between two modules counts as two directed edges."""
    repo = tmp_path / "repo"
    head = {
        "api/a.py": "from core.b import b\n\n\ndef a():\n    return b()\n",
        "core/b.py": "from api.a import a\n\n\ndef b():\n    return a\n",
        "core/c.py": "def c():\n    return 3\n",
    }
    changed, ranges = _diff_repo(repo, {}, head)
    assert _decide(repo, changed, ranges).cross_module_edges == 2


# --- Flowchart eligibility and branch counting ------------------------------


@pytest.mark.parametrize(
    "path,source,name,branch_points",
    [
        pytest.param("api/sample.py", PYTHON_SOURCE, "outer", 4, id="python"),
        pytest.param("api/sample.ts", TYPESCRIPT_SOURCE, "handle", 5, id="typescript"),
        pytest.param("api/sample.go", GO_SOURCE, "Handle", 5, id="go"),
        pytest.param("api/sample.rs", RUST_SOURCE, "handle", 6, id="rust"),
    ],
)
def test_branch_points_are_counted_per_language(
    tmp_path: Path, path: str, source: str, name: str, branch_points: int
) -> None:
    """Each grammar's conditional and loop constructs are counted once each."""
    repo = tmp_path / "repo"
    changed, ranges = _diff_repo(repo, {}, {path: source})
    eligibility = _decide(repo, changed, ranges)

    top = eligibility.candidate_roots[0]
    assert (top.file, top.name, top.branch_points) == (path, name, branch_points)
    assert eligibility.flowchart.eligible
    assert eligibility.flowchart.rule == "branch-points"
    assert f"most: {branch_points} in {name}" in eligibility.flowchart.reason
    assert eligibility.eligible_kinds() == ["flowchart"]


def test_unsupported_language_contributes_no_branch_points(tmp_path: Path) -> None:
    """A language with no installed grammar has no branch points to count."""
    repo = tmp_path / "repo"
    (repo / "api").mkdir(parents=True)
    (repo / "api/sample.rb").write_text("def handle(a)\n  if a\n    1\n  end\nend\n")
    assert count_function_branch_points(repo, "api/sample.rb", [(1, 5)]) == []


def test_branch_points_belong_to_the_innermost_function(tmp_path: Path) -> None:
    """A branch inside a nested closure counts for the closure, not its host."""
    repo = tmp_path / "repo"
    (repo / "api").mkdir(parents=True)
    (repo / "api/sample.py").write_text(PYTHON_SOURCE)

    roots = count_function_branch_points(repo, "api/sample.py", [(1, 18)])
    assert [(root.name, root.line, root.end_line, root.branch_points) for root in roots] == [
        ("outer", 1, 18, 4),
        ("inner", 10, 17, 3),
    ]


def test_only_branch_points_inside_a_changed_hunk_are_counted(tmp_path: Path) -> None:
    """The count is over changed hunks, not over the whole function."""
    repo = tmp_path / "repo"
    (repo / "api").mkdir(parents=True)
    (repo / "api/sample.py").write_text(PYTHON_SOURCE)

    # Lines 1-5 hold ``if`` (2), ``elif`` (4) and ``for`` (5); ``inner``'s range
    # (10-17) does not overlap, so it is not a changed function at all.
    roots = count_function_branch_points(repo, "api/sample.py", [(1, 5)])
    assert [(root.name, root.branch_points) for root in roots] == [("outer", 3)]

    # A hunk confined to the closure leaves ``outer`` a changed function with
    # zero changed branch points of its own.
    roots = count_function_branch_points(repo, "api/sample.py", [(10, 17)])
    assert [(root.name, root.branch_points) for root in roots] == [("outer", 0), ("inner", 3)]


def test_no_hunks_means_no_changed_functions(tmp_path: Path) -> None:
    """A file whose only hunk was a pure deletion contributes nothing."""
    repo = tmp_path / "repo"
    (repo / "api").mkdir(parents=True)
    (repo / "api/sample.py").write_text(PYTHON_SOURCE)
    assert count_function_branch_points(repo, "api/sample.py", []) == []


def test_missing_file_contributes_no_branch_points(tmp_path: Path) -> None:
    """A path that is not on disk fails open rather than raising."""
    assert count_function_branch_points(tmp_path, "api/gone.py", [(1, 10)]) == []


def test_candidate_roots_are_ordered_by_branch_points_then_file_then_line(
    tmp_path: Path,
) -> None:
    """Ordering is deterministic and puts the busiest function first."""
    repo = tmp_path / "repo"
    two_branches = "def {name}(a, b):\n    if a:\n        return 1\n    if b:\n        return 2\n    return 3\n"
    head = {
        "api/b.py": PYTHON_SOURCE,
        "api/a.py": two_branches.format(name="first") + "\n\n" + two_branches.format(name="second"),
    }
    changed, ranges = _diff_repo(repo, {}, head)
    eligibility = _decide(
        repo, changed, ranges, thresholds=DiagramThresholds(min_branch_points=2)
    )

    assert [(root.file, root.name, root.branch_points) for root in eligibility.candidate_roots] == [
        ("api/b.py", "outer", 4),
        ("api/b.py", "inner", 3),
        ("api/a.py", "first", 2),
        ("api/a.py", "second", 2),
    ]
    assert eligibility.function_branch_counts == eligibility.candidate_roots


def test_functions_below_the_branch_threshold_are_reported_but_not_candidates(
    tmp_path: Path,
) -> None:
    """``function_branch_counts`` records the near misses the decision rejected."""
    repo = tmp_path / "repo"
    head = {"api/a.py": "def a(x):\n    if x:\n        return 1\n    return 2\n"}
    changed, ranges = _diff_repo(repo, {}, head)
    eligibility = _decide(repo, changed, ranges)

    assert [(root.name, root.branch_points) for root in eligibility.function_branch_counts] == [
        ("a", 1)
    ]
    assert eligibility.candidate_roots == []
    assert not eligibility.flowchart.eligible
    assert "No changed function has >= 3 changed branch points" in eligibility.flowchart.reason


def test_raised_branch_threshold_disables_the_flowchart(tmp_path: Path) -> None:
    """``min_branch_points`` is honored, not baked in."""
    repo = tmp_path / "repo"
    changed, ranges = _diff_repo(repo, {}, {"api/sample.py": PYTHON_SOURCE})

    assert _decide(repo, changed, ranges).flowchart.eligible
    raised = _decide(repo, changed, ranges, thresholds=DiagramThresholds(min_branch_points=6))
    assert not raised.flowchart.eligible
    assert raised.candidate_roots == []


# --- Forcing ----------------------------------------------------------------


def test_force_sequence_leaves_the_flowchart_skipped(tmp_path: Path) -> None:
    """A forcing mode names kinds; the kinds it does not name are skipped."""
    repo = tmp_path / "repo"
    changed, ranges = _diff_repo(repo, {}, {"api/sample.py": PYTHON_SOURCE})
    eligibility = _decide(repo, changed, ranges, force="sequence")

    assert eligibility.sequence.eligible
    assert eligibility.sequence.rule == "forced"
    # The auto rules would have made the flowchart eligible here.
    assert not eligibility.flowchart.eligible
    assert eligibility.flowchart.rule is None
    assert "does not name this kind" in eligibility.flowchart.reason
    assert eligibility.eligible_kinds() == ["sequence"]


def test_force_both_makes_both_kinds_eligible(tmp_path: Path) -> None:
    """``both`` forces the two kinds on a diff no rule would have admitted."""
    repo = tmp_path / "repo"
    changed, ranges = _diff_repo(repo, {}, {"api/a.py": "def a():\n    return 1\n"})
    eligibility = _decide(repo, changed, ranges, force="both")

    assert eligibility.eligible_kinds() == ["sequence", "flowchart"]
    assert eligibility.sequence.rule == eligibility.flowchart.rule == "forced"


def test_forced_flowchart_falls_back_to_every_changed_function(tmp_path: Path) -> None:
    """With no qualifying candidate, a forced flowchart may root anywhere changed."""
    repo = tmp_path / "repo"
    head = {"api/a.py": "def a(x):\n    if x:\n        return 1\n    return 2\n\n\ndef b():\n    return 3\n"}
    changed, ranges = _diff_repo(repo, {}, head)
    eligibility = _decide(repo, changed, ranges, force="flowchart")

    assert eligibility.flowchart.eligible
    assert eligibility.flowchart.rule == "forced"
    assert "every changed function is offered" in eligibility.flowchart.reason
    assert [(root.name, root.branch_points) for root in eligibility.candidate_roots] == [
        ("a", 1),
        ("b", 0),
    ]


def test_forced_flowchart_keeps_the_qualifying_candidates_when_there_are_any(
    tmp_path: Path,
) -> None:
    """The fallback is a fallback: a qualifying candidate list is not widened."""
    repo = tmp_path / "repo"
    head = {"api/sample.py": PYTHON_SOURCE + "\n\ndef plain():\n    return 0\n"}
    changed, ranges = _diff_repo(repo, {}, head)
    eligibility = _decide(repo, changed, ranges, force="flowchart")

    assert [root.name for root in eligibility.candidate_roots] == ["outer", "inner"]
    assert "every changed function is offered" not in eligibility.flowchart.reason


def test_force_off_skips_both_kinds(tmp_path: Path) -> None:
    """``off`` denies both kinds and skips the branch sweep entirely."""
    repo = tmp_path / "repo"
    changed, ranges = _diff_repo(repo, {}, {"api/sample.py": PYTHON_SOURCE})
    eligibility = _decide(repo, changed, ranges, force="off")

    assert eligibility.eligible_kinds() == []
    assert eligibility.sequence == eligibility.flowchart
    assert eligibility.sequence.rule is None
    assert eligibility.sequence.reason == "Diagram mode is off."
    assert eligibility.candidate_roots == []
    assert eligibility.function_branch_counts == []
    # Signals that cost no parsing are still recorded for the artifact.
    assert eligibility.code_files == ["api/sample.py"]


def test_unrecognized_force_behaves_like_auto(tmp_path: Path) -> None:
    """The vocabulary gate is the CLI; this function never raises on a bad mode."""
    repo = tmp_path / "repo"
    changed, ranges = _diff_repo(repo, {}, {"api/sample.py": PYTHON_SOURCE})

    bogus = _decide(repo, changed, ranges, force="nonsense")
    auto = _decide(repo, changed, ranges, force="auto")
    assert bogus.flowchart == auto.flowchart
    assert bogus.sequence == auto.sequence
    assert bogus.force == "nonsense"


# --- Tiny-diff stack collapse (why stacks are an argument) -------------------


def test_decide_eligibility_is_immune_to_the_tiny_diff_stack_collapse(
    tmp_path: Path,
) -> None:
    """A collapsed stack list would classify a real code diff as no code at all.

    The orchestrator's published ``ctx.data["stacks"]`` is post-collapse: on a
    two-file diff spanning two languages every file lands in one ``generic``
    assignment. Passing that list in produces zero code files and therefore no
    diagram; passing a fresh ``detect_stacks`` produces both files. This is why
    ``_step_diagram`` must call ``detect_stacks`` itself.
    """
    repo = tmp_path / "repo"
    head = {
        "api/handler.py": "def handle():\n    return 1\n",
        "web/App.tsx": "export const App = () => <div>hi</div>;\n",
    }
    changed, ranges = _diff_repo(repo, {}, head)

    fresh = detect_stacks(sorted(changed))
    collapsed, single_stack_mode = _collapse_stacks_for_tiny_diff(
        fresh, sorted(changed), threshold=2
    )
    assert single_stack_mode
    assert {stack.stack_name for stack in collapsed} == {"generic", "structure"}

    assert _decide(repo, changed, ranges, stacks=collapsed).code_files == []
    assert _decide(repo, changed, ranges).code_files == ["api/handler.py", "web/App.tsx"]


# --- Fail-open and budget ---------------------------------------------------


def test_bad_tree_sitter_install_fails_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A known-bad tree-sitter yields no branch points, never an exception."""
    repo = tmp_path / "repo"
    changed, ranges = _diff_repo(repo, {}, {"api/sample.py": PYTHON_SOURCE})

    def _explode(*_args: object, **_kwargs: object) -> list[dict[str, object]]:
        raise TreeSitterBadVersionError("installed tree-sitter is known-bad")

    monkeypatch.setattr(diagram_trigger, "definitions_in_file", _explode)
    eligibility = _decide(repo, changed, ranges)

    assert eligibility.candidate_roots == []
    assert not eligibility.flowchart.eligible


def test_unexpected_tree_sitter_error_fails_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Anything the tree-sitter helpers let through is absorbed too."""
    repo = tmp_path / "repo"
    changed, ranges = _diff_repo(repo, {}, {"api/sample.py": PYTHON_SOURCE})

    def _explode(*_args: object, **_kwargs: object) -> list[int]:
        raise ValueError("query blew up")

    monkeypatch.setattr(diagram_trigger, "branch_statement_lines", _explode)
    assert _decide(repo, changed, ranges).candidate_roots == []


def test_exhausted_wall_budget_yields_partial_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The branch sweep is wall-bounded; on expiry it stops instead of stalling."""
    repo = tmp_path / "repo"
    changed, ranges = _diff_repo(repo, {}, {"api/sample.py": PYTHON_SOURCE})

    assert _decide(repo, changed, ranges).candidate_roots
    monkeypatch.setattr(diagram_trigger, "_BRANCH_COUNT_WALL_BUDGET_S", -1.0)
    assert _decide(repo, changed, ranges).candidate_roots == []


# --- Artifact form ----------------------------------------------------------


def test_to_dict_is_json_serializable_and_complete(tmp_path: Path) -> None:
    """``diagram.json`` carries every signal behind the decision."""
    repo = tmp_path / "repo"
    head = {
        "api/handler.py": "from core.engine import run\n\n\ndef handle(x):\n    return run(x)\n",
        "core/engine.py": "def run(x):\n    return x\n",
        "core/util.py": PYTHON_SOURCE,
    }
    changed, ranges = _diff_repo(repo, {}, head)
    payload: dict[str, Any] = _decide(repo, changed, ranges).to_dict()

    assert set(payload) == {
        "code_files",
        "modules",
        "services",
        "cross_module_edges",
        "function_branch_counts",
        "candidate_roots",
        "sequence",
        "flowchart",
        "thresholds",
        "force",
    }
    assert set(payload["sequence"]) == {"eligible", "rule", "reason"}
    assert payload["thresholds"] == {
        "min_code_files": 3,
        "min_modules": 2,
        "min_branch_points": 3,
    }
    assert payload["candidate_roots"][0] == {
        "file": "core/util.py",
        "name": "outer",
        "line": 1,
        "end_line": 18,
        "branch_points": 4,
    }
    assert payload["force"] == "auto"
    assert json.loads(json.dumps(payload)) == payload


def test_candidate_root_and_thresholds_are_re_exported() -> None:
    """The eligibility call sites import both dataclasses from this module."""
    root = CandidateRoot(file="a.py", name="f", line=1, end_line=9, branch_points=3)
    assert (root.file, root.branch_points) == ("a.py", 3)
    assert DiagramThresholds() == DiagramThresholds(
        min_code_files=3, min_modules=2, min_branch_points=3
    )
