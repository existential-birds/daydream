"""Deterministic-grounding tests for the diagram pipeline (issue #1113).

Every test runs against a **real** git repository on disk with committed files:
the grounder reads the head tree, parses it with tree-sitter and shells out to
``git grep`` for its symbol fallback, so a mocked filesystem would exercise
none of the behavior that matters. Fixture line numbers below are pinned by
``test_fixture_definition_ranges_are_pinned`` -- if a grammar's definition range
ever shifts, that test fails first and explains every other failure.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from daydream.config import (
    DIAGRAM_MAX_BLOCKS,
    DIAGRAM_MAX_EDGES,
    DIAGRAM_MAX_MESSAGES,
    DIAGRAM_MAX_NODES,
    DIAGRAM_MAX_PARTICIPANTS,
)
from daydream.deep.diagram_grounding import (
    OMIT_REASONS,
    REASON_CODES,
    ElementCheck,
    GroundingReport,
    RepoSymbols,
    ground_flowchart,
    ground_sequence,
)
from daydream.deep.diagram_types import CandidateRoot
from daydream.tree_sitter_index import definitions_in_file
from tests.harness.git_helpers import commit, git, init_repo

# --- Fixture sources ---------------------------------------------------------

_API_PY = """from pkg import service


def handle(request):
    result = service.resolve(request)
    return result
"""

_SERVICE_PY = """def resolve(request):
    return request


def store(value):
    return value
"""

# 1 def resolve_identity   2 if   3 raise   4 if   5 verify_jwt call
# 6 return   7 for   8 assignment   9 return   12 def verify_jwt   13 return
_FLOW_PY = """def resolve_identity(token):
    if token is None:
        raise ValueError("missing")
    if token.startswith("Bearer"):
        claims = verify_jwt(token)
        return claims
    for part in token.split():
        result = part
    return result


def verify_jwt(token):
    return token
"""

_LEGACY_RB = """def resolve_legacy(request)
  request
end
"""

_CALLER_RB = """def call_legacy(request)
  resolve_legacy(request)
end
"""

#: ``pkg/flow.py``'s ``resolve_identity``, exactly as ``decide_eligibility``
#: would publish it.
FLOW_ROOT = CandidateRoot(
    file="pkg/flow.py", name="resolve_identity", line=1, end_line=9, branch_points=4
)
#: ``pkg/big.py``'s ``big``, used by the node/edge cap tests.
BIG_ROOT = CandidateRoot(file="pkg/big.py", name="big", line=1, end_line=39, branch_points=1)

#: Head-side changed ranges covering every fixture file.
HUNKS: dict[str, list[tuple[int, int]]] = {
    "pkg/api.py": [(4, 6)],
    "pkg/service.py": [(1, 2)],
    "pkg/flow.py": [(1, 9)],
    "pkg/big.py": [(1, 39)],
    "pkg/caller.rb": [(1, 3)],
    "pkg/legacy.rb": [(1, 3)],
    **{f"pkg/p{index:02d}.py": [(1, 3)] for index in range(12)},
}

_BIG_LAST_STATEMENT = 38
_BIG_RETURN_LINE = 39


def _big_py() -> str:
    """Return a 39-line single-function source with one branch and one return."""
    lines = ["def big(flag):", "    if flag:", "        pass"]
    lines.extend(f"    step{index} = {index}" for index in range(4, _BIG_LAST_STATEMENT + 1))
    lines.append("    return step4")
    return "\n".join(lines) + "\n"


@pytest.fixture(scope="module")
def repo(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A committed git repo holding every grounding fixture file.

    Committed, not merely written: ``git grep`` -- the symbol fallback behind
    ``SUBROUTINE_NOT_DEFINED`` and the token-strength path -- only sees tracked
    content.
    """
    root = tmp_path_factory.mktemp("grounding-repo")
    init_repo(root)
    (root / "pkg").mkdir()
    files = {
        "pkg/api.py": _API_PY,
        "pkg/service.py": _SERVICE_PY,
        "pkg/flow.py": _FLOW_PY,
        "pkg/big.py": _big_py(),
        "pkg/legacy.rb": _LEGACY_RB,
        "pkg/caller.rb": _CALLER_RB,
    }
    for index in range(12):
        body = [
            f"from pkg import p{index + 1:02d}" if index < 11 else "# terminal participant",
            f"def fn{index:02d}(arg):",
            f"    return p{index + 1:02d}.fn{index + 1:02d}(arg)" if index < 11 else "    return arg",
        ]
        files[f"pkg/p{index:02d}.py"] = "\n".join(body) + "\n"
    for name, text in files.items():
        (root / name).write_text(text)
    git(root, "add", "-A")
    commit(root, "fixtures")
    return root


@pytest.fixture
def symbols(repo: Path) -> RepoSymbols:
    """A fresh definition index per test (memoization must not leak)."""
    return RepoSymbols(repo)


def reads(repo: Path, *relative: str) -> set[str]:
    """Return absolute read receipts, the shape a real tool call records."""
    return {str(repo / name) for name in relative}


ALL_READS = (
    "pkg/api.py",
    "pkg/service.py",
    "pkg/flow.py",
    "pkg/big.py",
    "pkg/caller.rb",
    "pkg/legacy.rb",
    *(f"pkg/p{index:02d}.py" for index in range(12)),
)


def check_for(report: GroundingReport, element: str, ref: str) -> ElementCheck:
    """Return the one check with this element type and ref."""
    matches = [c for c in report.elements if c.element == element and c.ref == ref]
    assert len(matches) == 1, f"{element}/{ref}: {matches}"
    return matches[0]


def reasons(report: GroundingReport) -> dict[str, str | None]:
    """Return ``{"element:ref": reason}`` for every failing check."""
    return {f"{c.element}:{c.ref}": c.reason for c in report.ungrounded()}


# --- Spec builders -----------------------------------------------------------


def base_sequence() -> dict[str, Any]:
    """A fully groundable three-message sequence spec over the fixture repo."""
    return {
        "participants": [
            {"name": "Client", "kind": "external", "files": [], "service": None},
            {"name": "API", "kind": "internal", "files": ["pkg/api.py"], "service": None},
            {
                "name": "Service",
                "kind": "internal",
                "files": ["pkg/service.py"],
                "service": "svc",
            },
        ],
        "messages": [
            {
                "from": "Client",
                "to": "API",
                "label": "request",
                "kind": "call",
                "changed": True,
                "evidence": {"file": "pkg/api.py", "line": 4, "symbol": "handle"},
            },
            {
                "from": "API",
                "to": "Service",
                "label": "resolve identity",
                "kind": "call",
                "changed": True,
                "evidence": {"file": "pkg/api.py", "line": 5, "symbol": "resolve"},
            },
            {
                # A reply names its enclosing function, and the grounder proves
                # only that the token is on the cited line -- here the
                # definition line of ``resolve`` in the replying participant.
                "from": "Service",
                "to": "API",
                "label": "payload",
                "kind": "reply",
                "changed": False,
                "evidence": {"file": "pkg/service.py", "line": 1, "symbol": "resolve"},
            },
        ],
        "blocks": [],
    }


def base_flowchart() -> dict[str, Any]:
    """A fully groundable eight-node flowchart over ``pkg/flow.py``."""
    return {
        "root": {"file": "pkg/flow.py", "name": "resolve_identity", "line": 1},
        "nodes": [
            {
                "id": "N1",
                "kind": "start",
                "label": "resolve identity",
                "evidence": {"file": "pkg/flow.py", "line": 1, "symbol": "resolve_identity"},
            },
            {
                "id": "N2",
                "kind": "decision",
                "label": "token missing?",
                "evidence": {"file": "pkg/flow.py", "line": 2, "symbol": None},
            },
            {
                "id": "N3",
                "kind": "end",
                "label": "raise ValueError",
                "evidence": {"file": "pkg/flow.py", "line": 3, "symbol": None},
            },
            {
                "id": "N4",
                "kind": "decision",
                "label": "bearer token?",
                "evidence": {"file": "pkg/flow.py", "line": 4, "symbol": None},
            },
            {
                "id": "N5",
                "kind": "subroutine",
                "label": "verify jwt",
                "evidence": {"file": "pkg/flow.py", "line": 5, "symbol": "verify_jwt"},
            },
            {
                "id": "N6",
                "kind": "end",
                "label": "return claims",
                "evidence": {"file": "pkg/flow.py", "line": 6, "symbol": None},
            },
            {
                "id": "N7",
                "kind": "process",
                "label": "scan parts",
                "evidence": {"file": "pkg/flow.py", "line": 8, "symbol": None},
            },
            {
                "id": "N8",
                "kind": "end",
                "label": "return result",
                "evidence": {"file": "pkg/flow.py", "line": 9, "symbol": None},
            },
        ],
        "edges": [
            {"from": "N1", "to": "N2", "label": None},
            {"from": "N2", "to": "N3", "label": "missing"},
            {"from": "N2", "to": "N4", "label": "present"},
            {"from": "N4", "to": "N5", "label": "bearer"},
            {"from": "N4", "to": "N7", "label": "other"},
            {"from": "N5", "to": "N6", "label": None},
            {"from": "N7", "to": "N8", "label": None},
        ],
    }


def run_sequence(
    repo: Path, symbols: RepoSymbols, spec: dict[str, Any], *, read: tuple[str, ...] = ALL_READS
) -> GroundingReport:
    """Ground ``spec`` as a sequence diagram against the fixture repo."""
    return ground_sequence(
        spec,
        repo_root=repo,
        hunk_ranges=HUNKS,
        read_paths=reads(repo, *read),
        symbols=symbols,
    )


def run_flowchart(
    repo: Path,
    symbols: RepoSymbols,
    spec: dict[str, Any],
    *,
    read: tuple[str, ...] = ALL_READS,
    candidate_roots: list[CandidateRoot] | None = None,
) -> GroundingReport:
    """Ground ``spec`` as a flowchart against the fixture repo."""
    return ground_flowchart(
        spec,
        repo_root=repo,
        hunk_ranges=HUNKS,
        read_paths=reads(repo, *read),
        candidate_roots=[FLOW_ROOT, BIG_ROOT] if candidate_roots is None else candidate_roots,
        symbols=symbols,
    )


# --- Fixture pinning ---------------------------------------------------------


def test_fixture_definition_ranges_are_pinned(repo: Path) -> None:
    """The candidate ranges every other test hardcodes come from tree-sitter."""
    flow = {record["name"]: record for record in definitions_in_file(repo, "pkg/flow.py")}
    assert (flow["resolve_identity"]["line"], flow["resolve_identity"]["end_line"]) == (
        FLOW_ROOT.line,
        FLOW_ROOT.end_line,
    )
    assert flow["verify_jwt"]["line"] == 12
    big = definitions_in_file(repo, "pkg/big.py")
    assert (big[0]["line"], big[0]["end_line"]) == (BIG_ROOT.line, BIG_ROOT.end_line)


# --- Happy paths -------------------------------------------------------------


def test_sequence_happy_path_grounds_every_element(repo: Path, symbols: RepoSymbols) -> None:
    report = run_sequence(repo, symbols, base_sequence())

    assert report.ungrounded() == []
    assert report.omit_reasons == []
    assert report.capped == {}
    assert report.root_range is None
    assert report.rejected is None
    assert report.summary == {"proposed": 6, "grounded": 6, "pruned": 0}
    assert [m["label"] for m in report.spec_final["messages"]] == [
        "request",
        "resolve identity",
        "payload",
    ]
    # The callee of an internal call is resolved to a real definition, which is
    # what the evidence table's "Callee defined at" column renders.
    assert check_for(report, "message", "0").strength == "definition"
    assert check_for(report, "message", "0").defined_at == "pkg/api.py:4"
    assert check_for(report, "message", "1").defined_at == "pkg/service.py:1"
    # A reply proves its symbol only as a token on the cited line.
    assert check_for(report, "message", "2").strength == "token"
    assert check_for(report, "message", "2").defined_at is None
    assert [check_for(report, "message", str(i)).final_index for i in range(3)] == [0, 1, 2]
    assert [check_for(report, "message", str(i)).in_changed_hunk for i in range(3)] == [
        True,
        True,
        True,
    ]


def test_flowchart_happy_path_grounds_every_element(repo: Path, symbols: RepoSymbols) -> None:
    report = run_flowchart(repo, symbols, base_flowchart())

    assert report.ungrounded() == []
    assert report.omit_reasons == []
    assert report.capped == {}
    assert report.rejected is None
    assert report.root_range == (1, 9)
    assert [node["id"] for node in report.spec_final["nodes"]] == [f"N{i}" for i in range(1, 9)]
    assert len(report.spec_final["edges"]) == 7
    assert report.spec_final["root"] == {
        "file": "pkg/flow.py",
        "name": "resolve_identity",
        "line": 1,
    }
    # The subroutine's symbol resolves to a real definition in the repo.
    subroutine = check_for(report, "node", "N5")
    assert (subroutine.strength, subroutine.defined_at) == ("definition", "pkg/flow.py:12")
    assert check_for(report, "edge", "N2->N3").final_index == 1
    # Both decisions keep two distinctly labeled branches, so neither is demoted.
    kinds = {node["id"]: node["kind"] for node in report.spec_final["nodes"]}
    assert kinds["N2"] == "decision" and kinds["N4"] == "decision"


def test_spec_final_key_sets_match_the_schemas(repo: Path, symbols: RepoSymbols) -> None:
    """``spec_final`` is annotation-free: Phase B re-validates it strictly."""
    spec = base_sequence()
    spec["blocks"] = [
        {
            "kind": "alt",
            "branches": [
                {
                    "condition": "token missing",
                    "evidence": {"file": "pkg/flow.py", "line": 2},
                    "messages": [0],
                },
                {
                    "condition": "bearer token",
                    "evidence": {"file": "pkg/flow.py", "line": 4},
                    "messages": [1, 2],
                },
            ],
        }
    ]
    # Extra keys a model might volunteer must not survive into spec_final.
    spec["participants"][1]["notes"] = "ignored"
    spec["messages"][0]["confidence"] = "high"
    sequence = run_sequence(repo, symbols, spec).spec_final

    assert set(sequence) == {"participants", "messages", "blocks"}
    for participant in sequence["participants"]:
        assert set(participant) == {"name", "kind", "files", "service"}
    for message in sequence["messages"]:
        assert set(message) == {"from", "to", "label", "kind", "changed", "evidence"}
        assert set(message["evidence"]) == {"file", "line", "symbol"}
        assert isinstance(message["evidence"]["symbol"], str)
    assert len(sequence["blocks"]) == 1
    for block in sequence["blocks"]:
        assert set(block) == {"kind", "branches"}
        for branch in block["branches"]:
            assert set(branch) == {"condition", "evidence", "messages"}
            assert set(branch["evidence"]) == {"file", "line"}
    assert [branch["messages"] for branch in sequence["blocks"][0]["branches"]] == [[0], [1, 2]]

    flowchart = run_flowchart(repo, symbols, base_flowchart()).spec_final
    assert set(flowchart) == {"root", "nodes", "edges"}
    assert set(flowchart["root"]) == {"file", "name", "line"}
    for node in flowchart["nodes"]:
        assert set(node) == {"id", "kind", "label", "evidence"}
        assert set(node["evidence"]) == {"file", "line", "symbol"}
    for edge in flowchart["edges"]:
        assert set(edge) == {"from", "to", "label"}


def test_element_check_to_dict_exposes_exactly_nine_keys(repo: Path, symbols: RepoSymbols) -> None:
    payload = check_for(run_sequence(repo, symbols, base_sequence()), "message", "0").to_dict()
    assert set(payload) == {
        "element",
        "ref",
        "grounded",
        "reason",
        "strength",
        "snapped_line",
        "in_changed_hunk",
        "defined_at",
        "final_index",
    }


# --- Shared reason codes, sequence side --------------------------------------


def test_sequence_path_escapes_repo(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_sequence()
    spec["messages"][1]["evidence"]["file"] = "../outside/secrets.py"
    report = run_sequence(repo, symbols, spec)
    assert check_for(report, "message", "1").reason == "PATH_ESCAPES_REPO"


def test_sequence_file_missing(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_sequence()
    spec["participants"][1]["files"] = ["pkg/api.py", "pkg/ghost.py"]
    spec["messages"][1]["evidence"]["file"] = "pkg/ghost.py"
    report = run_sequence(repo, symbols, spec)
    # The participant fails on its own missing file, and the message that cites
    # it fails on the citation.
    assert check_for(report, "participant", "API").reason == "PARTICIPANT_FILE_MISSING"
    assert check_for(report, "message", "1").reason == "EVIDENCE_NOT_IN_SOURCE_PARTICIPANT"

    spec = base_sequence()
    spec["messages"][1]["evidence"]["file"] = "pkg/ghost.py"
    spec["participants"][1]["files"] = ["pkg/api.py"]
    lone = run_sequence(repo, symbols, spec)
    assert check_for(lone, "message", "1").reason == "FILE_MISSING"


def test_sequence_line_out_of_range(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_sequence()
    spec["messages"][1]["evidence"]["line"] = 999
    report = run_sequence(repo, symbols, spec)
    assert check_for(report, "message", "1").reason == "LINE_OUT_OF_RANGE"


def test_sequence_symbol_not_on_line_beyond_snap_range(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_sequence()
    # "resolve" is on line 5; line 1 is four lines away, outside the +/-3 window.
    spec["messages"][1]["evidence"]["line"] = 1
    report = run_sequence(repo, symbols, spec)
    assert check_for(report, "message", "1").reason == "SYMBOL_NOT_ON_LINE"
    assert check_for(report, "message", "1").snapped_line is None


def test_sequence_symbol_snap_rewrites_the_citation(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_sequence()
    # "resolve" is not on pkg/service.py:2 but is one line above it.
    spec["messages"][2]["evidence"]["line"] = 2
    report = run_sequence(repo, symbols, spec)

    check = check_for(report, "message", "2")
    assert check.grounded and check.reason is None
    assert check.snapped_line == 1
    assert report.spec_final["messages"][2]["evidence"]["line"] == 1


def test_sequence_file_not_read_by_model(repo: Path, symbols: RepoSymbols) -> None:
    report = run_sequence(repo, symbols, base_sequence(), read=("pkg/api.py",))
    assert check_for(report, "message", "2").reason == "FILE_NOT_READ_BY_MODEL"
    # Fail-closed with no receipts at all: the missing-trajectory case.
    blind = ground_sequence(
        base_sequence(),
        repo_root=repo,
        hunk_ranges=HUNKS,
        read_paths=set(),
        symbols=symbols,
    )
    assert {c.reason for c in blind.ungrounded()} == {"FILE_NOT_READ_BY_MODEL"}


def test_read_receipt_matches_on_path_components_only(repo: Path, symbols: RepoSymbols) -> None:
    """A read of ``notapi.py`` must not cover ``pkg/api.py``."""
    spec = base_sequence()
    report = ground_sequence(
        spec,
        repo_root=repo,
        hunk_ranges=HUNKS,
        read_paths={str(repo / "pkg/notapi.py"), str(repo / "pkg/service.py")},
        symbols=symbols,
    )
    assert check_for(report, "message", "0").reason == "FILE_NOT_READ_BY_MODEL"
    assert check_for(report, "message", "2").grounded


def test_sequence_branch_not_a_branch_statement(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_sequence()
    spec["blocks"] = [
        {
            "kind": "opt",
            "branches": [
                {
                    "condition": "assignment is not a branch",
                    "evidence": {"file": "pkg/flow.py", "line": 5},
                    "messages": [0],
                }
            ],
        }
    ]
    report = run_sequence(repo, symbols, spec)
    assert check_for(report, "branch", "b0.0").reason == "NOT_A_BRANCH_STATEMENT"
    assert check_for(report, "block", "b0").reason == "NOT_A_BRANCH_STATEMENT"
    # The block is gone but its message is not: it renders flat.
    assert report.spec_final["blocks"] == []
    assert len(report.spec_final["messages"]) == 3


# --- Sequence-specific reason codes ------------------------------------------


def test_sequence_evidence_not_in_source_participant(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_sequence()
    spec["messages"][1]["evidence"] = {"file": "pkg/service.py", "line": 1, "symbol": "resolve"}
    report = run_sequence(repo, symbols, spec)
    assert check_for(report, "message", "1").reason == "EVIDENCE_NOT_IN_SOURCE_PARTICIPANT"


def test_sequence_callee_not_defined_in_target(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_sequence()
    # "handle" is on the cited line but is defined in the caller, not in Service.
    spec["messages"][1]["evidence"] = {"file": "pkg/api.py", "line": 4, "symbol": "handle"}
    report = run_sequence(repo, symbols, spec)
    assert check_for(report, "message", "1").reason == "CALLEE_NOT_DEFINED_IN_TARGET"


def test_sequence_participant_no_files(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_sequence()
    spec["participants"][2]["files"] = []
    report = run_sequence(repo, symbols, spec)
    assert check_for(report, "participant", "Service").reason == "PARTICIPANT_NO_FILES"
    # Its messages go with it.
    assert check_for(report, "message", "1").reason == "CALLEE_NOT_DEFINED_IN_TARGET"
    assert check_for(report, "message", "2").reason == "EVIDENCE_NOT_IN_SOURCE_PARTICIPANT"


def test_sequence_participant_file_missing(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_sequence()
    spec["participants"][2]["files"] = ["pkg/vanished.py"]
    report = run_sequence(repo, symbols, spec)
    assert check_for(report, "participant", "Service").reason == "PARTICIPANT_FILE_MISSING"


def test_sequence_participant_path_escapes_repo(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_sequence()
    spec["participants"][2]["files"] = ["../elsewhere/service.py"]
    report = run_sequence(repo, symbols, spec)
    assert check_for(report, "participant", "Service").reason == "PATH_ESCAPES_REPO"


def test_sequence_external_misused_by_declaring_files(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_sequence()
    spec["participants"][0]["files"] = ["pkg/api.py"]
    report = run_sequence(repo, symbols, spec)
    assert check_for(report, "participant", "Client").reason == "EXTERNAL_MISUSED"


def test_sequence_external_misused_by_sourcing_a_later_message(
    repo: Path, symbols: RepoSymbols
) -> None:
    spec = base_sequence()
    spec["messages"][1]["from"] = "Client"
    report = run_sequence(repo, symbols, spec)
    assert check_for(report, "participant", "Client").reason == "EXTERNAL_MISUSED"
    # Message 0 loses its (now ungrounded) source too.
    assert check_for(report, "message", "0").reason == "EVIDENCE_NOT_IN_SOURCE_PARTICIPANT"


def test_sequence_malformed_elements(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_sequence()
    spec["participants"].append({"name": "Ghost", "kind": "spectral", "files": [], "service": None})
    spec["participants"].append(
        {"name": "API", "kind": "internal", "files": ["pkg/api.py"], "service": None}
    )
    spec["messages"].append("not a message")
    spec["messages"].append({**base_sequence()["messages"][1], "kind": "telepathy"})
    spec["blocks"].append({"kind": "whenever", "branches": [{"condition": "x"}]})
    report = run_sequence(repo, symbols, spec)

    assert check_for(report, "participant", "Ghost").reason == "MALFORMED_ELEMENT"
    # The duplicate name is rejected, not silently allowed to shadow the first.
    assert [c.reason for c in report.elements if c.element == "participant"].count(
        "MALFORMED_ELEMENT"
    ) == 2
    assert check_for(report, "message", "3").reason == "MALFORMED_ELEMENT"
    assert check_for(report, "message", "4").reason == "MALFORMED_ELEMENT"
    assert check_for(report, "block", "b0").reason == "MALFORMED_ELEMENT"
    assert check_for(report, "branch", "b0.0").reason == "MALFORMED_ELEMENT"


def test_sequence_token_strength_fallback_for_a_language_without_a_grammar(
    repo: Path, symbols: RepoSymbols
) -> None:
    """Ruby has no tree-sitter grammar here, so a callee is proven by token only."""
    spec = {
        "participants": [
            {"name": "Caller", "kind": "internal", "files": ["pkg/caller.rb"], "service": None},
            {"name": "Legacy", "kind": "internal", "files": ["pkg/legacy.rb"], "service": None},
        ],
        "messages": [
            {
                "from": "Caller",
                "to": "Legacy",
                "label": "resolve legacy",
                "kind": "call",
                "changed": True,
                "evidence": {"file": "pkg/caller.rb", "line": 2, "symbol": "resolve_legacy"},
            }
        ],
        "blocks": [],
    }
    report = run_sequence(repo, symbols, spec)

    check = check_for(report, "message", "0")
    assert check.grounded
    assert (check.strength, check.defined_at) == ("token", None)
    # One message is below the floor, which is the honest outcome.
    assert report.omit_reasons == ["TOO_FEW_MESSAGES"]


def test_repo_symbols_survives_a_repo_without_git_history(tmp_path: Path) -> None:
    """No commits (and no git dir at all) must degrade to "not found", never raise."""
    plain = tmp_path / "plain"
    (plain / "pkg").mkdir(parents=True)
    (plain / "pkg" / "mod.py").write_text("def helper():\n    return 1\n")
    symbols = RepoSymbols(plain)
    assert symbols.token_defined_anywhere("helper") is False
    assert [r["line"] for r in symbols.definitions("helper", ["pkg/mod.py"])] == [1]

    init_repo(plain)
    assert RepoSymbols(plain).token_defined_anywhere("helper") is False


# --- Sequence prune semantics ------------------------------------------------


def test_sequence_prune_flattens_a_block_whose_condition_is_ungrounded(
    repo: Path, symbols: RepoSymbols
) -> None:
    spec = base_sequence()
    spec["blocks"] = [
        {
            "kind": "alt",
            "branches": [
                {
                    "condition": "token missing",
                    "evidence": {"file": "pkg/flow.py", "line": 2},
                    "messages": [0],
                },
                {
                    # Line 5 is an assignment: not a branch statement.
                    "condition": "fabricated",
                    "evidence": {"file": "pkg/flow.py", "line": 5},
                    "messages": [1, 2],
                },
            ],
        }
    ]
    report = run_sequence(repo, symbols, spec)

    assert check_for(report, "branch", "b0.1").reason == "NOT_A_BRANCH_STATEMENT"
    # An ``alt`` with one surviving branch is not an alternative any more, so
    # the whole block is dropped -- and every message stays, rendered flat.
    assert report.spec_final["blocks"] == []
    assert len(report.spec_final["messages"]) == 3
    assert check_for(report, "block", "b0").grounded
    assert check_for(report, "block", "b0").final_index is None
    assert check_for(report, "branch", "b0.0").final_index is None


def test_sequence_prune_drops_participants_with_no_remaining_messages(
    repo: Path, symbols: RepoSymbols
) -> None:
    spec = base_sequence()
    spec["participants"].append(
        {"name": "Store", "kind": "internal", "files": ["pkg/service.py"], "service": None}
    )
    report = run_sequence(repo, symbols, spec)

    store = check_for(report, "participant", "Store")
    assert store.grounded and store.final_index is None
    assert [p["name"] for p in report.spec_final["participants"]] == ["Client", "API", "Service"]
    # A structural drop is not an ungrounded drop.
    assert report.summary["pruned"] == 0


def test_sequence_final_index_is_dense_after_a_mid_list_prune(
    repo: Path, symbols: RepoSymbols
) -> None:
    spec = base_sequence()
    spec["messages"].insert(
        1,
        {
            "from": "API",
            "to": "Service",
            "label": "fabricated",
            "kind": "call",
            "changed": True,
            "evidence": {"file": "pkg/api.py", "line": 999, "symbol": "resolve"},
        },
    )
    report = run_sequence(repo, symbols, spec)

    assert check_for(report, "message", "1").reason == "LINE_OUT_OF_RANGE"
    assert check_for(report, "message", "1").final_index is None
    assert [check_for(report, "message", str(i)).final_index for i in range(4)] == [0, None, 1, 2]
    assert [m["label"] for m in report.spec_final["messages"]] == [
        "request",
        "resolve identity",
        "payload",
    ]
    assert report.summary["pruned"] == 1


def test_sequence_block_message_indices_are_remapped_after_a_prune(
    repo: Path, symbols: RepoSymbols
) -> None:
    spec = base_sequence()
    # After the entrypoint, so message 0 keeps its "first message" status -- an
    # external participant may only source the entrypoint.
    spec["messages"].insert(
        1,
        {
            "from": "API",
            "to": "Service",
            "label": "fabricated",
            "kind": "call",
            "changed": True,
            "evidence": {"file": "pkg/ghost.py", "line": 1, "symbol": "resolve"},
        },
    )
    spec["blocks"] = [
        {
            "kind": "opt",
            "branches": [
                {
                    "condition": "token missing",
                    "evidence": {"file": "pkg/flow.py", "line": 2},
                    "messages": [1, 2, 3],
                }
            ],
        }
    ]
    report = run_sequence(repo, symbols, spec)

    assert check_for(report, "message", "1").reason == "FILE_MISSING"
    # Proposed indices 2 and 3 became final positions 1 and 2; index 1 is gone.
    assert report.spec_final["blocks"][0]["branches"][0]["messages"] == [1, 2]
    assert check_for(report, "branch", "b0.0").final_index == 0
    assert check_for(report, "block", "b0").final_index == 0


def test_sequence_opt_block_keeps_only_its_first_branch(
    repo: Path, symbols: RepoSymbols
) -> None:
    spec = base_sequence()
    spec["blocks"] = [
        {
            "kind": "opt",
            "branches": [
                {
                    "condition": "first",
                    "evidence": {"file": "pkg/flow.py", "line": 2},
                    "messages": [0],
                },
                {
                    "condition": "second",
                    "evidence": {"file": "pkg/flow.py", "line": 4},
                    "messages": [1],
                },
            ],
        }
    ]
    report = run_sequence(repo, symbols, spec)

    assert [b["condition"] for b in report.spec_final["blocks"][0]["branches"]] == ["first"]
    # The surplus branch was grounded; it is normalized away, not pruned.
    assert check_for(report, "branch", "b0.1").grounded
    assert check_for(report, "branch", "b0.1").final_index is None
    assert report.summary["pruned"] == 0


# --- Sequence floors ---------------------------------------------------------


def test_sequence_floor_too_few_messages(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_sequence()
    spec["messages"] = spec["messages"][:2]
    report = run_sequence(repo, symbols, spec)
    assert report.omit_reasons == ["TOO_FEW_MESSAGES"]
    assert len(report.spec_final["messages"]) == 2


def test_sequence_floor_too_few_participants(repo: Path, symbols: RepoSymbols) -> None:
    spec = {
        "participants": [
            {"name": "API", "kind": "internal", "files": ["pkg/api.py"], "service": None}
        ],
        "messages": [
            {
                "from": "API",
                "to": "API",
                "label": f"self step {index}",
                "kind": "self",
                "changed": True,
                "evidence": {"file": "pkg/api.py", "line": 4, "symbol": "handle"},
            }
            for index in range(3)
        ],
        "blocks": [],
    }
    report = run_sequence(repo, symbols, spec)
    assert report.omit_reasons == ["TOO_FEW_PARTICIPANTS"]


def test_sequence_floor_no_changed_interaction(repo: Path, symbols: RepoSymbols) -> None:
    report = ground_sequence(
        base_sequence(),
        repo_root=repo,
        hunk_ranges={},
        read_paths=reads(repo, *ALL_READS),
        symbols=symbols,
    )
    assert report.omit_reasons == ["NO_CHANGED_INTERACTION"]
    assert len(report.spec_final["messages"]) == 3
    assert all(not c.in_changed_hunk for c in report.elements if c.element == "message")


# --- Sequence caps -----------------------------------------------------------


def _wide_sequence(count: int) -> dict[str, Any]:
    """A groundable spec with ``count`` participants chained by one call each."""
    return {
        "participants": [
            {
                "name": f"P{index:02d}",
                "kind": "internal",
                "files": [f"pkg/p{index:02d}.py"],
                "service": None,
            }
            for index in range(count)
        ],
        "messages": [
            {
                "from": f"P{index:02d}",
                "to": f"P{index + 1:02d}",
                "label": f"step {index}",
                "kind": "call",
                "changed": True,
                "evidence": {
                    "file": f"pkg/p{index:02d}.py",
                    "line": 3,
                    "symbol": f"fn{index + 1:02d}",
                },
            }
            for index in range(count - 1)
        ],
        "blocks": [],
    }


def test_sequence_participant_cap_drops_orphaned_messages(
    repo: Path, symbols: RepoSymbols
) -> None:
    report = run_sequence(repo, symbols, _wide_sequence(12))

    assert report.ungrounded() == []
    assert len(report.spec_final["participants"]) == DIAGRAM_MAX_PARTICIPANTS
    assert report.capped == {"participants": 2, "messages": 2}
    assert len(report.spec_final["messages"]) == 9
    assert report.omit_reasons == []
    # Cap drops are not ungrounded drops.
    assert report.summary["pruned"] == 0
    assert check_for(report, "participant", "P11").final_index is None
    assert check_for(report, "message", "10").final_index is None
    assert check_for(report, "message", "10").grounded


def test_sequence_message_cap_truncates_the_tail(repo: Path, symbols: RepoSymbols) -> None:
    spec = _wide_sequence(2)
    template = spec["messages"][0]
    spec["messages"] = [
        {**template, "evidence": dict(template["evidence"]), "label": f"step {index}"}
        for index in range(DIAGRAM_MAX_MESSAGES + 5)
    ]
    report = run_sequence(repo, symbols, spec)

    assert report.capped == {"messages": 5}
    assert len(report.spec_final["messages"]) == DIAGRAM_MAX_MESSAGES
    assert report.spec_final["messages"][-1]["label"] == f"step {DIAGRAM_MAX_MESSAGES - 1}"
    assert report.omit_reasons == []


def test_sequence_block_cap_truncates_the_tail(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_sequence()
    spec["blocks"] = [
        {
            "kind": "alt",
            "branches": [
                {
                    "condition": f"block {index} first",
                    "evidence": {"file": "pkg/flow.py", "line": 2},
                    "messages": [0],
                },
                {
                    "condition": f"block {index} second",
                    "evidence": {"file": "pkg/flow.py", "line": 4},
                    "messages": [1],
                },
            ],
        }
        for index in range(DIAGRAM_MAX_BLOCKS + 2)
    ]
    report = run_sequence(repo, symbols, spec)

    assert report.capped == {"blocks": 2}
    assert len(report.spec_final["blocks"]) == DIAGRAM_MAX_BLOCKS
    assert check_for(report, "block", f"b{DIAGRAM_MAX_BLOCKS}").final_index is None


def test_sequence_cap_can_push_a_kind_below_its_floor(repo: Path, symbols: RepoSymbols) -> None:
    """The floor is evaluated on the capped spec, so a cap drop cannot hide.

    The only two interactions inside a changed hunk sit past the message cap.
    Trimming them in the renderer would have drawn a diagram whose ``<sub>``
    line claims a changed interaction it no longer shows; trimming them here
    omits the kind instead.
    """
    spec = _wide_sequence(3)
    unchanged, changed = spec["messages"][0], spec["messages"][1]
    spec["messages"] = [
        {**unchanged, "evidence": dict(unchanged["evidence"]), "label": f"step {index}"}
        for index in range(DIAGRAM_MAX_MESSAGES)
    ] + [
        {**changed, "evidence": dict(changed["evidence"]), "label": f"changed {index}"}
        for index in range(2)
    ]
    report = ground_sequence(
        spec,
        repo_root=repo,
        hunk_ranges={"pkg/p01.py": [(1, 3)]},
        read_paths=reads(repo, *ALL_READS),
        symbols=symbols,
    )

    assert report.capped == {"messages": 2}
    assert len(report.spec_final["messages"]) == DIAGRAM_MAX_MESSAGES
    assert not any(m["label"].startswith("changed") for m in report.spec_final["messages"])
    assert report.omit_reasons == ["NO_CHANGED_INTERACTION"]


# --- Flowchart root ----------------------------------------------------------


def test_flowchart_root_not_candidate_rejects_the_whole_spec(
    repo: Path, symbols: RepoSymbols
) -> None:
    spec = base_flowchart()
    spec["root"] = {"file": "pkg/flow.py", "name": "verify_jwt", "line": 12}
    report = run_flowchart(repo, symbols, spec)

    assert report.rejected == "ROOT_NOT_CANDIDATE"
    assert check_for(report, "root", "verify_jwt").reason == "ROOT_NOT_CANDIDATE"
    assert report.spec_final["nodes"] == [] and report.spec_final["edges"] == []
    assert report.spec_final["root"] == {"file": "pkg/flow.py", "name": "verify_jwt", "line": 12}
    assert report.root_range is None
    # No node was even adjudicated, so the omission is not silent.
    assert report.omit_reasons == ["TOO_FEW_NODES"]
    assert len(report.elements) == 1


def test_flowchart_root_must_still_overlap_a_changed_hunk(
    repo: Path, symbols: RepoSymbols
) -> None:
    report = ground_flowchart(
        base_flowchart(),
        repo_root=repo,
        hunk_ranges={"pkg/api.py": [(4, 6)]},
        read_paths=reads(repo, *ALL_READS),
        candidate_roots=[FLOW_ROOT],
        symbols=symbols,
    )
    assert report.rejected == "ROOT_NOT_CANDIDATE"


def test_flowchart_root_range_comes_from_the_candidate_not_the_model(
    repo: Path, symbols: RepoSymbols
) -> None:
    report = run_flowchart(repo, symbols, base_flowchart())
    assert report.root_range == (FLOW_ROOT.line, FLOW_ROOT.end_line)
    assert check_for(report, "root", "resolve_identity").final_index == 0


# --- Flowchart node reason codes ---------------------------------------------


def test_flowchart_node_path_escapes_repo(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_flowchart()
    spec["nodes"][6]["evidence"]["file"] = "../outside/flow.py"
    report = run_flowchart(repo, symbols, spec)
    assert check_for(report, "node", "N7").reason == "PATH_ESCAPES_REPO"


def test_flowchart_node_file_missing(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_flowchart()
    spec["nodes"][6]["evidence"]["file"] = "pkg/ghost.py"
    report = run_flowchart(repo, symbols, spec)
    assert check_for(report, "node", "N7").reason == "FILE_MISSING"


def test_flowchart_node_line_out_of_range(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_flowchart()
    spec["nodes"][6]["evidence"]["line"] = 9999
    report = run_flowchart(repo, symbols, spec)
    assert check_for(report, "node", "N7").reason == "LINE_OUT_OF_RANGE"


def test_flowchart_node_file_not_read_by_model(repo: Path, symbols: RepoSymbols) -> None:
    report = run_flowchart(repo, symbols, base_flowchart(), read=("pkg/api.py",))
    assert {c.reason for c in report.elements if c.element == "node"} == {
        "FILE_NOT_READ_BY_MODEL"
    }
    assert report.spec_final["nodes"] == []


def test_flowchart_node_symbol_not_on_line(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_flowchart()
    spec["nodes"][6]["evidence"]["symbol"] = "resolve_identity"
    report = run_flowchart(repo, symbols, spec)
    assert check_for(report, "node", "N7").reason == "SYMBOL_NOT_ON_LINE"


def test_flowchart_node_symbol_snap_stays_inside_the_root(
    repo: Path, symbols: RepoSymbols
) -> None:
    spec = base_flowchart()
    # "verify_jwt" is on line 5; the node cites line 6 and snaps back one line.
    spec["nodes"][4]["evidence"]["line"] = 6
    report = run_flowchart(repo, symbols, spec)

    check = check_for(report, "node", "N5")
    assert check.grounded and check.snapped_line == 5
    assert report.spec_final["nodes"][4]["evidence"]["line"] == 5


def test_flowchart_node_outside_root(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_flowchart()
    # Line 13 is inside verify_jwt, not inside the root function.
    spec["nodes"][6]["evidence"]["line"] = 13
    report = run_flowchart(repo, symbols, spec)
    assert check_for(report, "node", "N7").reason == "NODE_OUTSIDE_ROOT"

    other_file = base_flowchart()
    other_file["nodes"][6]["evidence"] = {"file": "pkg/api.py", "line": 5, "symbol": None}
    assert (
        check_for(run_flowchart(repo, symbols, other_file), "node", "N7").reason
        == "NODE_OUTSIDE_ROOT"
    )


def test_flowchart_not_a_terminal_statement(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_flowchart()
    spec["nodes"][7]["evidence"]["line"] = 8  # an assignment, not a return
    report = run_flowchart(repo, symbols, spec)
    assert check_for(report, "node", "N8").reason == "NOT_A_TERMINAL_STATEMENT"


def test_flowchart_decision_not_a_branch_statement(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_flowchart()
    spec["nodes"][3]["evidence"]["line"] = 5  # a call, not a branch
    report = run_flowchart(repo, symbols, spec)
    assert check_for(report, "node", "N4").reason == "NOT_A_BRANCH_STATEMENT"


def test_flowchart_subroutine_not_called_here(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_flowchart()
    spec["nodes"][4]["evidence"]["symbol"] = "nonexistent_helper"
    report = run_flowchart(repo, symbols, spec)
    assert check_for(report, "node", "N5").reason == "SUBROUTINE_NOT_CALLED_HERE"


def test_flowchart_subroutine_not_defined(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_flowchart()
    # "ValueError" really is on line 3, but this repository defines no such
    # symbol -- being on the call-site line is not being defined.
    spec["nodes"][4]["evidence"] = {"file": "pkg/flow.py", "line": 3, "symbol": "ValueError"}
    report = run_flowchart(repo, symbols, spec)
    assert check_for(report, "node", "N5").reason == "SUBROUTINE_NOT_DEFINED"


def test_flowchart_subroutine_without_a_symbol_is_malformed(
    repo: Path, symbols: RepoSymbols
) -> None:
    spec = base_flowchart()
    spec["nodes"][4]["evidence"]["symbol"] = None
    report = run_flowchart(repo, symbols, spec)
    assert check_for(report, "node", "N5").reason == "MALFORMED_ELEMENT"


def test_flowchart_malformed_nodes_and_edges(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_flowchart()
    spec["nodes"].append({**spec["nodes"][6], "kind": "hologram", "id": "N9"})
    spec["nodes"].append(spec["nodes"][0])  # duplicate id
    spec["nodes"].append("not a node")
    spec["edges"].append({"from": "N7", "to": "N8", "label": "duplicate ref"})
    spec["edges"].append({"from": "", "to": "N1", "label": None})
    report = run_flowchart(repo, symbols, spec)

    assert check_for(report, "node", "N9").reason == "MALFORMED_ELEMENT"
    malformed = [c for c in report.elements if c.reason == "MALFORMED_ELEMENT"]
    assert {c.element for c in malformed} == {"node", "edge"}
    assert len(malformed) == 5


def test_flowchart_multiple_start(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_flowchart()
    spec["nodes"].append(
        {
            "id": "N9",
            "kind": "start",
            "label": "second entry",
            "evidence": {"file": "pkg/flow.py", "line": 1, "symbol": "resolve_identity"},
        }
    )
    spec["edges"].append({"from": "N9", "to": "N2", "label": None})
    report = run_flowchart(repo, symbols, spec)

    assert check_for(report, "node", "N9").reason == "MULTIPLE_START"
    assert check_for(report, "node", "N1").grounded
    assert check_for(report, "edge", "N9->N2").reason == "EDGE_ENDPOINT_UNGROUNDED"


def test_flowchart_edge_endpoint_ungrounded(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_flowchart()
    spec["nodes"][6]["evidence"]["line"] = 9999
    report = run_flowchart(repo, symbols, spec)

    assert check_for(report, "node", "N7").reason == "LINE_OUT_OF_RANGE"
    assert check_for(report, "edge", "N4->N7").reason == "EDGE_ENDPOINT_UNGROUNDED"
    assert check_for(report, "edge", "N7->N8").reason == "EDGE_ENDPOINT_UNGROUNDED"


def test_flowchart_decision_edges_invalid(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_flowchart()
    spec["edges"][2]["label"] = None  # N2 -> N4 loses its label
    spec["edges"].append({"from": "N4", "to": "N6", "label": "bearer"})  # duplicate label
    report = run_flowchart(repo, symbols, spec)

    assert check_for(report, "edge", "N2->N4").reason == "DECISION_EDGES_INVALID"
    assert check_for(report, "edge", "N4->N6").reason == "DECISION_EDGES_INVALID"


def test_flowchart_decision_with_one_branch_is_demoted_not_dropped(
    repo: Path, symbols: RepoSymbols
) -> None:
    spec = base_flowchart()
    spec["edges"] = [edge for edge in spec["edges"] if edge != {"from": "N2", "to": "N3", "label": "missing"}]
    report = run_flowchart(repo, symbols, spec)

    # N2 keeps one labeled branch, so it can no longer claim to be a decision.
    kinds = {node["id"]: node["kind"] for node in report.spec_final["nodes"]}
    assert kinds["N2"] == "process"
    assert check_for(report, "node", "N2").grounded
    # N3 was only reachable through the removed edge.
    assert "N3" not in kinds
    assert check_for(report, "node", "N3").grounded
    assert check_for(report, "node", "N3").final_index is None
    assert kinds["N4"] == "decision"


def test_flowchart_unreachable_nodes_are_removed(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_flowchart()
    spec["nodes"].append(
        {
            "id": "N9",
            "kind": "process",
            "label": "orphan",
            "evidence": {"file": "pkg/flow.py", "line": 8, "symbol": None},
        }
    )
    report = run_flowchart(repo, symbols, spec)

    assert [node["id"] for node in report.spec_final["nodes"]] == [f"N{i}" for i in range(1, 9)]
    orphan = check_for(report, "node", "N9")
    assert orphan.grounded and orphan.final_index is None
    assert report.summary["pruned"] == 0


def test_flowchart_final_index_is_dense_after_a_mid_list_prune(
    repo: Path, symbols: RepoSymbols
) -> None:
    spec = base_flowchart()
    spec["nodes"].insert(
        2,
        {
            "id": "NX",
            "kind": "process",
            "label": "fabricated",
            "evidence": {"file": "pkg/flow.py", "line": 9999, "symbol": None},
        },
    )
    spec["edges"].insert(1, {"from": "N2", "to": "NX", "label": "bogus"})
    report = run_flowchart(repo, symbols, spec)

    assert check_for(report, "node", "NX").final_index is None
    assert [check_for(report, "node", f"N{i}").final_index for i in range(1, 9)] == list(range(8))
    assert check_for(report, "edge", "N2->NX").reason == "EDGE_ENDPOINT_UNGROUNDED"
    assert check_for(report, "edge", "N2->N3").final_index == 1


# --- Flowchart floors --------------------------------------------------------


def test_flowchart_floor_too_few_nodes(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_flowchart()
    spec["nodes"] = spec["nodes"][:3]
    spec["edges"] = spec["edges"][:2]
    report = run_flowchart(repo, symbols, spec)
    assert report.omit_reasons == ["TOO_FEW_NODES", "NO_DECISION"]
    assert len(report.spec_final["nodes"]) == 3


def test_flowchart_floor_no_end(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_flowchart()
    for node in spec["nodes"]:
        if node["kind"] == "end":
            node["kind"] = "process"
    report = run_flowchart(repo, symbols, spec)
    assert report.omit_reasons == ["NO_END"]


def test_flowchart_floor_no_decision(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_flowchart()
    for node in spec["nodes"]:
        if node["kind"] == "decision":
            node["kind"] = "process"
    report = run_flowchart(repo, symbols, spec)
    assert report.omit_reasons == ["NO_DECISION"]
    assert {node["kind"] for node in report.spec_final["nodes"]} == {
        "start",
        "process",
        "subroutine",
        "end",
    }


def test_flowchart_unlabeled_decision_edges_cascade_into_an_omission(
    repo: Path, symbols: RepoSymbols
) -> None:
    """Stripping a decision's labels invalidates its edges, not just its kind."""
    spec = base_flowchart()
    for edge in spec["edges"]:
        edge["label"] = None
    report = run_flowchart(repo, symbols, spec)

    assert reasons(report) == {
        "edge:N2->N3": "DECISION_EDGES_INVALID",
        "edge:N2->N4": "DECISION_EDGES_INVALID",
        "edge:N4->N5": "DECISION_EDGES_INVALID",
        "edge:N4->N7": "DECISION_EDGES_INVALID",
        "edge:N5->N6": "EDGE_ENDPOINT_UNGROUNDED",
        "edge:N7->N8": "EDGE_ENDPOINT_UNGROUNDED",
    }
    assert [node["id"] for node in report.spec_final["nodes"]] == ["N1", "N2"]
    assert report.omit_reasons == ["TOO_FEW_NODES", "NO_END", "NO_DECISION"]


# --- Flowchart caps ----------------------------------------------------------


def _tall_flowchart(*, end_last: bool) -> dict[str, Any]:
    """A single-root flowchart with more nodes than the render cap allows.

    ``end_last`` puts the only terminal node at the end of spec order, where the
    node cap will trim it -- the case that must reach the floor as ``NO_END``
    instead of rendering an endless diagram.
    """
    process_lines = list(range(4, _BIG_LAST_STATEMENT + 1))[: DIAGRAM_MAX_NODES + 1]
    end_node = {
        "id": "NEND",
        "kind": "end",
        "label": "return",
        "evidence": {"file": "pkg/big.py", "line": _BIG_RETURN_LINE, "symbol": None},
    }
    nodes: list[dict[str, Any]] = [
        {
            "id": "NSTART",
            "kind": "start",
            "label": "big",
            "evidence": {"file": "pkg/big.py", "line": 1, "symbol": "big"},
        },
        {
            "id": "NDEC",
            "kind": "decision",
            "label": "flag?",
            "evidence": {"file": "pkg/big.py", "line": 2, "symbol": None},
        },
    ]
    if not end_last:
        nodes.append(end_node)
    nodes.extend(
        {
            "id": f"NP{index}",
            "kind": "process",
            "label": f"step {index}",
            "evidence": {"file": "pkg/big.py", "line": line, "symbol": None},
        }
        for index, line in enumerate(process_lines)
    )
    if end_last:
        nodes.append(end_node)
    edges: list[dict[str, Any]] = [
        {"from": "NSTART", "to": "NDEC", "label": None},
        {"from": "NDEC", "to": "NEND", "label": "done"},
        {"from": "NDEC", "to": "NP0", "label": "work"},
    ]
    edges.extend(
        {"from": f"NP{index}", "to": f"NP{index + 1}", "label": None}
        for index in range(len(process_lines) - 1)
    )
    return {"root": {"file": "pkg/big.py", "name": "big", "line": 1}, "nodes": nodes, "edges": edges}


def test_flowchart_node_cap_trims_the_tail_and_keeps_rendering(
    repo: Path, symbols: RepoSymbols
) -> None:
    report = run_flowchart(repo, symbols, _tall_flowchart(end_last=False))

    assert report.ungrounded() == []
    assert len(report.spec_final["nodes"]) == DIAGRAM_MAX_NODES
    assert report.capped["nodes"] == 4
    assert report.capped["edges"] == 4
    assert report.omit_reasons == []
    assert report.spec_final["nodes"][0]["id"] == "NSTART"
    assert len(report.spec_final["edges"]) <= DIAGRAM_MAX_EDGES


def test_flowchart_cap_can_push_a_kind_below_its_floor(
    repo: Path, symbols: RepoSymbols
) -> None:
    report = run_flowchart(repo, symbols, _tall_flowchart(end_last=True))

    assert report.capped["nodes"] == 4
    # Losing the terminal node also strips the decision's second branch, so the
    # demotion pass runs again on the capped graph.
    assert report.omit_reasons == ["NO_END", "NO_DECISION"]
    # The trimmed end node was grounded; it was cap-dropped, not pruned.
    assert check_for(report, "node", "NEND").grounded
    assert check_for(report, "node", "NEND").final_index is None
    assert report.summary["pruned"] == 0


def test_flowchart_node_cap_never_trims_the_start_node(
    repo: Path, symbols: RepoSymbols
) -> None:
    spec = _tall_flowchart(end_last=False)
    start = spec["nodes"].pop(0)
    spec["nodes"].append(start)  # start now sits past the cap in spec order
    report = run_flowchart(repo, symbols, spec)

    ids = [node["id"] for node in report.spec_final["nodes"]]
    assert ids[0] == "NSTART"
    assert len(ids) == DIAGRAM_MAX_NODES


# --- Vocabulary contracts ----------------------------------------------------


def test_every_emitted_reason_code_is_declared(repo: Path, symbols: RepoSymbols) -> None:
    """Nothing may leak a reason code that is not in the published vocabulary."""
    specs: list[GroundingReport] = []
    broken_sequence = base_sequence()
    broken_sequence["participants"][0]["files"] = ["pkg/api.py"]
    broken_sequence["participants"][2]["files"] = ["pkg/gone.py"]
    broken_sequence["messages"][1]["evidence"] = {
        "file": "../outside.py",
        "line": 0,
        "symbol": "nope",
    }
    broken_sequence["blocks"] = [
        {
            "kind": "loop",
            "branches": [
                {"condition": "", "evidence": {"file": "pkg/flow.py", "line": 5}, "messages": [0]}
            ],
        }
    ]
    specs.append(run_sequence(repo, symbols, broken_sequence))

    broken_flowchart = base_flowchart()
    broken_flowchart["nodes"][1]["evidence"]["line"] = 5
    broken_flowchart["nodes"][4]["evidence"]["symbol"] = "nonexistent_helper"
    broken_flowchart["nodes"][6]["evidence"]["file"] = "pkg/ghost.py"
    broken_flowchart["edges"][1]["label"] = None
    specs.append(run_flowchart(repo, symbols, broken_flowchart))
    specs.append(run_flowchart(repo, symbols, {"root": None, "nodes": [], "edges": []}))

    emitted = {c.reason for report in specs for c in report.ungrounded()}
    assert emitted, "the deliberately broken specs must fail something"
    assert emitted <= REASON_CODES
    assert {r for report in specs for r in report.omit_reasons} <= OMIT_REASONS


def test_reason_and_omit_vocabularies_are_disjoint_and_complete() -> None:
    assert "NO_END" in OMIT_REASONS and "NO_END" not in REASON_CODES
    assert "MULTIPLE_START" in REASON_CODES
    assert REASON_CODES.isdisjoint(OMIT_REASONS)
