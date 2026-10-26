"""Ground diagrams against real committed Git files and tree-sitter parses.

Fixture ranges are pinned separately so grammar drift has a clear failure.
Git-backed symbol fallback requires tracked content."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from daydream.config import (
    DIAGRAM_MAX_BLOCKS,
    DIAGRAM_MAX_MESSAGES,
    DIAGRAM_MAX_NODES,
    DIAGRAM_MAX_PARTICIPANTS,
)
from daydream.deep.diagram_grounding import (
    ElementCheck,
    GroundingReport,
    RepoSymbols,
    ground_flowchart,
    ground_sequence,
)
from daydream.deep.diagram_render import render_sequence_mermaid
from daydream.deep.diagram_types import CandidateRoot
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

_NON_EXECUTABLE_PY = '''def classify(value):
    """Describe the flow."""
    # perform the work

    result = value
    if result:
        return result
    return None
'''

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
FLOW_ROOT = CandidateRoot(file="pkg/flow.py", name="resolve_identity", line=1, end_line=9, branch_points=4)
#: ``pkg/big.py``'s ``big``, used by the node/edge cap tests.
BIG_ROOT = CandidateRoot(file="pkg/big.py", name="big", line=1, end_line=39, branch_points=1)
NON_EXECUTABLE_ROOT = CandidateRoot(file="pkg/non_executable.py", name="classify", line=1, end_line=8, branch_points=1)

#: Head-side changed ranges covering every fixture file.
HUNKS: dict[str, list[tuple[int, int]]] = {
    "pkg/api.py": [(4, 6)], "pkg/service.py": [(1, 2)], "pkg/flow.py": [(1, 9)], "pkg/big.py": [(1, 39)],
    "pkg/caller.rb": [(1, 3)], "pkg/legacy.rb": [(1, 3)], **{f"pkg/p{index:02d}.py": [(1, 3)] for index in range(12)},
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
    """Commit grounding fixtures so the Git symbol fallback can read them."""
    root = tmp_path_factory.mktemp("grounding-repo")
    init_repo(root)
    (root / "pkg").mkdir()
    files = {"pkg/api.py": _API_PY, "pkg/service.py": _SERVICE_PY, "pkg/flow.py": _FLOW_PY,
        "pkg/non_executable.py": _NON_EXECUTABLE_PY, "pkg/big.py": _big_py(), "pkg/legacy.rb": _LEGACY_RB,
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
    return {"participants": [{"name": "Client", "kind": "external", "files": [], "service": None},
            {"name": "API", "kind": "internal", "files": ["pkg/api.py"], "service": None},
            {"name": "Service", "kind": "internal", "files": ["pkg/service.py"], "service": "svc"},
        ], "messages": [{"from": "Client", "to": "API", "label": "request", "kind": "call", "changed": True,
                "evidence": {"file": "pkg/api.py", "line": 4, "symbol": "handle"},
            }, {"from": "API", "to": "Service", "label": "resolve identity", "kind": "call", "changed": True,
                "evidence": {"file": "pkg/api.py", "line": 5, "symbol": "resolve"},
            },
            {
                # A reply cites the return statement in its enclosing function.
                "from": "Service", "to": "API", "label": "payload", "kind": "reply", "changed": False,
                "evidence": {"file": "pkg/service.py", "line": 2, "symbol": "resolve"},
            },
        ], "blocks": [],
    }


def base_flowchart() -> dict[str, Any]:
    """A fully groundable eight-node flowchart over ``pkg/flow.py``."""
    return {"root": {"file": "pkg/flow.py", "name": "resolve_identity", "line": 1},
        "nodes": [{"id": "N1", "kind": "start", "label": "resolve identity",
                "evidence": {"file": "pkg/flow.py", "line": 1, "symbol": "resolve_identity"},
            }, {"id": "N2", "kind": "decision", "label": "token missing?",
                "evidence": {"file": "pkg/flow.py", "line": 2, "symbol": None},
            }, {"id": "N3", "kind": "end", "label": "raise ValueError",
                "evidence": {"file": "pkg/flow.py", "line": 3, "symbol": None},
            }, {"id": "N4", "kind": "decision", "label": "bearer token?",
                "evidence": {"file": "pkg/flow.py", "line": 4, "symbol": None},
            }, {"id": "N5", "kind": "subroutine", "label": "verify jwt",
                "evidence": {"file": "pkg/flow.py", "line": 5, "symbol": "verify_jwt"},
            }, {"id": "N6", "kind": "end", "label": "return claims",
                "evidence": {"file": "pkg/flow.py", "line": 6, "symbol": None},
            }, {"id": "N7", "kind": "process", "label": "scan parts",
                "evidence": {"file": "pkg/flow.py", "line": 8, "symbol": None},
            }, {"id": "N8", "kind": "end", "label": "return result",
                "evidence": {"file": "pkg/flow.py", "line": 9, "symbol": None},
            },
        ], "edges": [{"from": "N1", "to": "N2", "label": None}, {"from": "N2", "to": "N3", "label": "missing"},
            {"from": "N2", "to": "N4", "label": "present"}, {"from": "N4", "to": "N5", "label": "bearer"},
            {"from": "N4", "to": "N7", "label": "other"}, {"from": "N5", "to": "N6", "label": None},
            {"from": "N7", "to": "N8", "label": None},
        ],
    }


def run_sequence(repo: Path, symbols: RepoSymbols, spec: dict[str, Any]) -> GroundingReport:
    """Ground ``spec`` as a sequence diagram against the fixture repo."""
    return ground_sequence(spec, repo_root=repo, hunk_ranges=HUNKS, symbols=symbols,)


def run_flowchart(
    repo: Path, symbols: RepoSymbols, spec: dict[str, Any], *, candidate_roots: list[CandidateRoot] | None = None,
) -> GroundingReport:
    """Ground ``spec`` as a flowchart against the fixture repo."""
    return ground_flowchart(spec, repo_root=repo, hunk_ranges=HUNKS,
        candidate_roots=[FLOW_ROOT, BIG_ROOT] if candidate_roots is None else candidate_roots, symbols=symbols,
    )


# --- Fixture pinning ---------------------------------------------------------

# --- Happy paths -------------------------------------------------------------


def test_sequence_graph_isolates_bad_elements_and_preserves_live_order_and_blocks(
    repo: Path,
    symbols: RepoSymbols,
) -> None:
    """Independent local faults cannot erase valid calls or borrow another participant's evidence."""
    spec = base_sequence()
    spec["participants"].extend(
        [
            {"name": name, "kind": "internal", "files": files, "service": None}
            for name, files in [
                ("Caller", ["pkg/api.py"]),
                ("Callee", ["pkg/service.py"]),
                ("SelfAPI", ["pkg/api.py"]),
                ("SelfService", ["pkg/service.py"]),
                ("Missing", ["pkg/api.py", "pkg/ghost.py"]),
                ("Empty", []),
                ("Escape", ["../elsewhere/service.py"]),
            ]
        ]
    )
    spec["participants"].extend(
        [
            {"name": "BadExternal", "kind": "external", "files": [], "service": None},
            {"name": "ExternalFiles", "kind": "external", "files": ["pkg/api.py"], "service": None},
            {"name": "Ghost", "kind": "spectral", "files": [], "service": None},
            dict(spec["participants"][1]),
        ]
    )
    call = base_sequence()["messages"][1]
    reply = base_sequence()["messages"][2]
    messages = [
        {**spec["messages"][0], "from": "BadExternal"},
        {**call, "label": "mid-list fault", "evidence": {**call["evidence"], "line": 999}},
        *spec["messages"][1:],
    ]
    expected = {"message:0": "EVIDENCE_NOT_IN_SOURCE_PARTICIPANT",
                "message:1": "LINE_OUT_OF_RANGE"}
    faults: list[tuple[str, dict[str, Any], dict[str, Any], dict[str, Any], str]] = [
        ("escape", call, {"file": "../outside/secrets.py"}, {}, "PATH_ESCAPES_REPO"),
        ("range", call, {"line": 999}, {}, "LINE_OUT_OF_RANGE"),
        (
            "source",
            call,
            {"file": "pkg/service.py", "line": 1, "symbol": "resolve"},
            {},
            "EVIDENCE_NOT_IN_SOURCE_PARTICIPANT",
        ),
        ("callee", call, {"file": "pkg/api.py", "line": 4, "symbol": "handle"}, {}, "CALLEE_NOT_DEFINED_IN_TARGET"),
        ("return", reply, {"line": 1}, {}, "NOT_A_REPLY_STATEMENT"),
        ("orphan", reply, {"file": "pkg/api.py", "line": 6, "symbol": "handle"},
         {"from": "API", "to": "Service"}, "REPLY_NOT_PRECEDED_BY_CALL"),
        ("bad preceding call", call, {"line": 999}, {"from": "Caller", "to": "Callee"}, "LINE_OUT_OF_RANGE"),
        ("bad call reply", reply, {}, {"from": "Callee", "to": "Caller"}, "REPLY_NOT_PRECEDED_BY_CALL"),
        ("missing file", call, {"file": "pkg/ghost.py"}, {}, "FILE_MISSING"),
        (
            "missing participant",
            call,
            {"file": "pkg/ghost.py"},
            {"from": "Missing"},
            "EVIDENCE_NOT_IN_SOURCE_PARTICIPANT",
        ),
        ("empty target", call, {}, {"to": "Empty"}, "CALLEE_NOT_DEFINED_IN_TARGET"),
        ("empty source", reply, {}, {"from": "Empty"}, "EVIDENCE_NOT_IN_SOURCE_PARTICIPANT"),
        ("external misuse", call, {}, {"from": "BadExternal"}, "EVIDENCE_NOT_IN_SOURCE_PARTICIPANT"),
    ]
    for label, template, evidence, fields, reason in faults:
        if label in {"return", "orphan"}:
            messages.append({**call, "label": f"{label} check call"})
        expected[f"message:{len(messages)}"] = reason
        messages.append({**template, **fields, "label": label, "evidence": {**template["evidence"], **evidence}})
    messages.append({**call, "label": "symbol snap", "evidence": {**call["evidence"], "line": 4}})
    snap_index = len(messages) - 1
    messages.append({**call, "from": "SelfAPI", "to": "SelfService", "kind": "self", "label": "self interaction"})
    expected[f"message:{len(messages)}"] = "REPLY_NOT_PRECEDED_BY_CALL"
    messages.append({**reply, "from": "SelfService", "to": "SelfAPI"})
    expected[f"message:{len(messages)}"] = "MALFORMED_ELEMENT"
    messages.append("not a message")
    expected[f"message:{len(messages)}"] = "MALFORMED_ELEMENT"
    messages.append({**call, "kind": "telepathy"})
    spec["messages"] = messages
    spec["blocks"] = [
        {
            "kind": "opt",
            "branches": [
                {"condition": "not a branch", "evidence": {"file": "pkg/flow.py", "line": 5}, "messages": [0]}
            ],
        },
        {
            "kind": "alt",
            "branches": [
                {"condition": "real alternative", "evidence": {"file": "pkg/flow.py", "line": 2}, "messages": [0]},
                {"condition": "fabricated", "evidence": {"file": "pkg/flow.py", "line": 5}, "messages": [2, 3]},
            ],
        },
        {"kind": "whenever", "branches": [{"condition": "x"}]},
        {
            "kind": "opt",
            "branches": [
                {"condition": "retained", "evidence": {"file": "pkg/flow.py", "line": 2}, "messages": [1, 2, 3]},
                {"condition": "surplus", "evidence": {"file": "pkg/flow.py", "line": 4}, "messages": [0]},
            ],
        },
    ]

    report = run_sequence(repo, symbols, spec)

    for ref, reason in expected.items():
        element, index = ref.split(":")
        assert check_for(report, element, index).reason == reason
    assert check_for(report, "participant", "Missing").reason == "PARTICIPANT_FILE_MISSING"
    assert check_for(report, "participant", "Empty").reason == "PARTICIPANT_NO_FILES"
    assert check_for(report, "participant", "Escape").reason == "PATH_ESCAPES_REPO"
    assert check_for(report, "participant", "ExternalFiles").reason == "EXTERNAL_MISUSED"
    assert check_for(report, "participant", "BadExternal").reason == "EXTERNAL_MISUSED"
    assert [check.reason for check in report.elements if check.element == "participant"].count("MALFORMED_ELEMENT") == 2
    assert check_for(report, "branch", "b0.0").reason == "NOT_A_BRANCH_STATEMENT"
    assert check_for(report, "block", "b0").reason == "NOT_A_BRANCH_STATEMENT"
    assert check_for(report, "branch", "b1.1").reason == "NOT_A_BRANCH_STATEMENT"
    assert check_for(report, "block", "b1").grounded
    assert check_for(report, "block", "b2").reason == check_for(report, "branch", "b2.0").reason == "MALFORMED_ELEMENT"
    assert check_for(report, "branch", "b3.1").grounded
    assert report.spec_final["blocks"] == [
        {
            "kind": "opt",
            "branches": [{"condition": "retained", "evidence": {"file": "pkg/flow.py", "line": 2}, "messages": [0, 1]}],
        }
    ]
    assert [message["label"] for message in report.spec_final["messages"]] == [
        "resolve identity",
        "payload",
        "return check call",
        "orphan check call",
        "symbol snap",
        "self interaction",
    ]
    assert check_for(report, "message", str(snap_index)).snapped_line == 5
    assert report.spec_final["messages"][4]["evidence"]["line"] == 5
    assert [check_for(report, "message", str(i)).defined_at for i in (2, 3)] == [
        "pkg/service.py:1",
        "pkg/service.py:1",
    ]
    assert all(
        check_for(report, "message", str(i)).strength == "definition"
        and check_for(report, "message", str(i)).in_changed_hunk
        for i in (2, 3)
    )
    assert report.rejected is None and report.omit_reasons == []
    # The two surviving interactions stay in one remapped branch, without an
    # extra opt/else boundary between the valid call and its corresponding reply.
    assert (
        "    opt retained\n        P1->>P2: resolve identity\n        P2-->>P1: payload\n    end\n"
    ) in render_sequence_mermaid(report.spec_final)


# --- Shared reason codes, sequence side --------------------------------------


# --- Sequence-specific reason codes ------------------------------------------


def test_sequence_reply_requires_an_enclosing_function(repo: Path, symbols: RepoSymbols, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = base_sequence()
    original_definitions = symbols.definitions

    def definitions(symbol: str, files: list[str] | None = None) -> list[dict[str, object]]:
        if symbol == "resolve" and files == ["pkg/service.py"]:
            return [{"file": "pkg/service.py", "line": 1, "end_line": 2, "kind": "class"}]
        return original_definitions(symbol, files)

    monkeypatch.setattr(symbols, "definitions", definitions)
    report = run_sequence(repo, symbols, spec)

    assert check_for(report, "message", "2").reason == "REPLY_NOT_IN_ENCLOSING_FUNCTION"


def test_sequence_token_strength_fallback_for_a_language_without_a_grammar(repo: Path, symbols: RepoSymbols) -> None:
    """Ruby has no tree-sitter grammar here, so a callee is proven by token only."""
    spec = {"participants": [{"name": "Caller", "kind": "internal", "files": ["pkg/caller.rb"], "service": None},
            {"name": "Legacy", "kind": "internal", "files": ["pkg/legacy.rb"], "service": None},
        ], "messages": [{"from": "Caller", "to": "Legacy", "label": "resolve legacy", "kind": "call", "changed": True,
                "evidence": {"file": "pkg/caller.rb", "line": 2, "symbol": "resolve_legacy"},
            }
        ], "blocks": [],
    }
    report = run_sequence(repo, symbols, spec)

    check = check_for(report, "message", "0")
    assert check.grounded
    assert (check.strength, check.defined_at) == ("token", None)
    # One message is below the floor, which is the honest outcome.
    assert report.omit_reasons == ["TOO_FEW_MESSAGES"]


def test_repo_symbols_survives_a_repo_without_git_history(tmp_path: Path) -> None:
    plain = tmp_path / "plain"
    (plain / "pkg").mkdir(parents=True)
    (plain / "pkg" / "mod.py").write_text("def helper():\n    return 1\n")
    symbols = RepoSymbols(plain)
    assert [r["line"] for r in symbols.definitions("helper", ["pkg/mod.py"])] == [1]

    init_repo(plain)
    assert RepoSymbols(plain).definitions("helper", ["pkg/mod.py"])


# --- Sequence prune semantics ------------------------------------------------


# --- Sequence floors ---------------------------------------------------------


def test_sequence_floor_no_changed_interaction(repo: Path, symbols: RepoSymbols) -> None:
    report = ground_sequence(base_sequence(), repo_root=repo, hunk_ranges={}, symbols=symbols,)
    assert report.omit_reasons == ["NO_CHANGED_INTERACTION"]
    assert len(report.spec_final["messages"]) == 3
    assert all(not c.in_changed_hunk for c in report.elements if c.element == "message")


# --- Sequence caps -----------------------------------------------------------


def _wide_sequence(count: int) -> dict[str, Any]:
    """A groundable spec with ``count`` participants chained by one call each."""
    return {"participants": [
            {"name": f"P{index:02d}", "kind": "internal", "files": [f"pkg/p{index:02d}.py"], "service": None}
            for index in range(count)
        ], "messages": [{"from": f"P{index:02d}", "to": f"P{index + 1:02d}", "label": f"step {index}", "kind": "call",
                "changed": True,
                "evidence": {"file": f"pkg/p{index:02d}.py", "line": 3, "symbol": f"fn{index + 1:02d}"},
            }
            for index in range(count - 1)
        ], "blocks": [],
    }


def test_sequence_participant_cap_drops_orphaned_messages(repo: Path, symbols: RepoSymbols) -> None:
    report = run_sequence(repo, symbols, _wide_sequence(12))

    assert report.ungrounded() == []
    assert len(report.spec_final["participants"]) == DIAGRAM_MAX_PARTICIPANTS
    assert len(report.spec_final["messages"]) == 9
    participants = {p["name"] for p in report.spec_final["participants"]}
    assert all(m["from"] in participants and m["to"] in participants for m in report.spec_final["messages"])
    assert report.omit_reasons == []
    # Cap drops are not ungrounded drops.
    assert check_for(report, "message", "10").grounded




def test_sequence_block_cap_truncates_the_tail(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_sequence()
    spec["blocks"] = [{"kind": "alt",
            "branches": [{"condition": f"block {index} first", "evidence": {"file": "pkg/flow.py", "line": 2},
                    "messages": [0],
                }, {"condition": f"block {index} second", "evidence": {"file": "pkg/flow.py", "line": 4},
                    "messages": [1],
                },
            ],
        }
        for index in range(DIAGRAM_MAX_BLOCKS + 2)
    ]
    report = run_sequence(repo, symbols, spec)

    assert len(report.spec_final["blocks"]) == DIAGRAM_MAX_BLOCKS
    assert check_for(report, "participant", "Client").grounded
    entrypoint = check_for(report, "message", "0")
    assert entrypoint.grounded and entrypoint.in_changed_hunk
    assert (entrypoint.strength, entrypoint.defined_at) == ("definition", "pkg/api.py:4")


def test_sequence_cap_can_push_a_kind_below_its_floor(repo: Path, symbols: RepoSymbols) -> None:
    spec = _wide_sequence(3)
    unchanged, changed = spec["messages"][0], spec["messages"][1]
    spec["messages"] = [{**unchanged, "evidence": dict(unchanged["evidence"]), "label": f"step {index}"}
        for index in range(DIAGRAM_MAX_MESSAGES)
    ] + [{**changed, "evidence": dict(changed["evidence"]), "label": f"changed {index}"} for index in range(2)]
    report = ground_sequence(spec, repo_root=repo, hunk_ranges={"pkg/p01.py": [(1, 3)]}, symbols=symbols,)

    assert len(report.spec_final["messages"]) == DIAGRAM_MAX_MESSAGES
    assert not any(m["label"].startswith("changed") for m in report.spec_final["messages"])
    assert report.spec_final["messages"][-1]["label"] == f"step {DIAGRAM_MAX_MESSAGES - 1}"
    assert report.omit_reasons == ["NO_CHANGED_INTERACTION"]


# --- Flowchart root ----------------------------------------------------------


def test_flowchart_root_must_still_overlap_a_changed_hunk(repo: Path, symbols: RepoSymbols) -> None:
    report = ground_flowchart(
        base_flowchart(), repo_root=repo, hunk_ranges={"pkg/api.py": [(4, 6)]}, candidate_roots=[FLOW_ROOT],
        symbols=symbols,
    )
    assert report.rejected == "ROOT_NOT_CANDIDATE"


# --- Flowchart node reason codes ---------------------------------------------


def test_flowchart_graph_isolates_evidence_topology_and_malformed_element_failures(
    repo: Path,
    symbols: RepoSymbols,
) -> None:
    """One grounded graph retains its live path beside independently rejected nodes and edges."""
    spec = base_flowchart()
    spec["nodes"][4]["evidence"]["line"] = 6
    faults: list[tuple[int, dict[str, Any], str]] = [
        (6, {"file": "../outside/flow.py"}, "PATH_ESCAPES_REPO"),
        (6, {"file": "pkg/ghost.py"}, "FILE_MISSING"),
        (6, {"line": 9999}, "LINE_OUT_OF_RANGE"),
        (6, {"symbol": "resolve_identity"}, "SYMBOL_NOT_ON_LINE"),
        (7, {"line": 8}, "NOT_A_TERMINAL_STATEMENT"),
        (3, {"line": 5}, "NOT_A_BRANCH_STATEMENT"),
        (4, {"symbol": "nonexistent_helper"}, "SUBROUTINE_NOT_CALLED_HERE"),
        (4, {"file": "pkg/flow.py", "line": 3, "symbol": "ValueError"}, "SUBROUTINE_NOT_DEFINED"),
        (4, {"symbol": None}, "MALFORMED_ELEMENT"),
    ]
    expected = {}
    for index, (template, evidence, reason) in enumerate(faults):
        node = base_flowchart()["nodes"][template]
        identifier = f"BAD{index}"
        spec["nodes"].append({**node, "id": identifier, "evidence": {**node["evidence"], **evidence}})
        expected[identifier] = reason
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
    spec["nodes"].extend(
        [
            {**base_flowchart()["nodes"][6], "kind": "hologram", "id": "HOLOGRAM"},
            spec["nodes"][0],
            "not a node",
            {
                "id": "SECOND",
                "kind": "start",
                "label": "second entry",
                "evidence": {"file": "pkg/flow.py", "line": 1, "symbol": "resolve_identity"},
            },
            {**base_flowchart()["nodes"][1], "id": "D1"},
            {**base_flowchart()["nodes"][3], "id": "D2"},
        ]
    )
    spec["edges"].extend(
        [
            {"from": "N7", "to": "N8", "label": "duplicate ref"},
            {"from": "", "to": "N1", "label": None},
            {"from": "SECOND", "to": "N2", "label": None},
            {"from": "N1", "to": "D1", "label": None},
            {"from": "D1", "to": "N3", "label": None},
            {"from": "D1", "to": "N7", "label": "yes"},
            {"from": "N1", "to": "D2", "label": None},
            {"from": "D2", "to": "N5", "label": "bearer"},
            {"from": "D2", "to": "N6", "label": "bearer"},
        ]
    )

    report = run_flowchart(repo, symbols, spec)

    assert {identifier: check_for(report, "node", identifier).reason for identifier in expected} == expected
    assert check_for(report, "node", "NX").reason == "LINE_OUT_OF_RANGE"
    assert check_for(report, "edge", "N2->NX").reason == "EDGE_ENDPOINT_UNGROUNDED"
    assert check_for(report, "node", "SECOND").reason == "MULTIPLE_START"
    assert [(check.grounded, check.reason) for check in report.elements
            if check.element == "node" and check.ref == "N1"] == [(True, None), (False, "MALFORMED_ELEMENT")]
    assert check_for(report, "edge", "SECOND->N2").reason == "EDGE_ENDPOINT_UNGROUNDED"
    assert check_for(report, "node", "HOLOGRAM").reason == "MALFORMED_ELEMENT"
    malformed = [check for check in report.elements if check.reason == "MALFORMED_ELEMENT"]
    assert len(malformed) == 6 and {check.element for check in malformed} == {"node", "edge"}
    assert all(check_for(report, "edge", edge).reason == "DECISION_EDGES_INVALID"
               for edge in ("D1->N3", "D2->N6"))
    assert all(check_for(report, "edge", edge).grounded for edge in ("D1->N7", "D2->N5"))
    assert {node["id"]: node["kind"] for node in report.spec_final["nodes"]
            if node["id"] in {"D1", "D2"}} == {"D1": "process", "D2": "process"}
    assert check_for(report, "node", "N5").snapped_line == 5
    live = {node["id"] for node in base_flowchart()["nodes"]}
    assert [node for node in report.spec_final["nodes"] if node["id"] in live] == base_flowchart()["nodes"]
    assert [edge for edge in report.spec_final["edges"] if edge["from"] in live and edge["to"] in live] == (
        base_flowchart()["edges"]
    )
    assert not ({*expected, "NX", "SECOND", "HOLOGRAM"} & {node["id"] for node in report.spec_final["nodes"]})
    assert report.rejected is None and report.omit_reasons == []


def test_flowchart_executable_nodes_reject_non_executable_lines(repo: Path, symbols: RepoSymbols) -> None:
    spec = {"root": {"file": "pkg/non_executable.py", "name": "classify", "line": 1},
        "nodes": [{"id": "N1", "kind": "start", "label": "describe the flow",
                "evidence": {"file": "pkg/non_executable.py", "line": 2, "symbol": None},
            }, {"id": "N2", "kind": "process", "label": "perform the work",
                "evidence": {"file": "pkg/non_executable.py", "line": 3, "symbol": None},
            }, {"id": "N3", "kind": "io", "label": "read the value",
                "evidence": {"file": "pkg/non_executable.py", "line": 4, "symbol": None},
            }, {"id": "N4", "kind": "decision", "label": "has a result?",
                "evidence": {"file": "pkg/non_executable.py", "line": 6, "symbol": None},
            }, {"id": "N5", "kind": "end", "label": "return the result",
                "evidence": {"file": "pkg/non_executable.py", "line": 7, "symbol": None},
            },
        ], "edges": [{"from": "N1", "to": "N2", "label": None}, {"from": "N2", "to": "N3", "label": None},
            {"from": "N3", "to": "N4", "label": None}, {"from": "N4", "to": "N5", "label": "yes"},
        ],
    }

    report = ground_flowchart(
        spec, repo_root=repo, hunk_ranges={"pkg/non_executable.py": [(1, 8)]}, candidate_roots=[NON_EXECUTABLE_ROOT],
        symbols=symbols,
    )

    assert {check_for(report, "node", node_id).reason for node_id in ("N1", "N2", "N3")
    } == {"NOT_AN_EXECUTABLE_STATEMENT"}


# --- Flowchart floors --------------------------------------------------------


def test_flowchart_unlabeled_decision_edges_cascade_into_an_omission(repo: Path, symbols: RepoSymbols) -> None:
    """Stripping a decision's labels invalidates its edges, not just its kind."""
    spec = base_flowchart()
    for edge in spec["edges"]:
        edge["label"] = None
    report = run_flowchart(repo, symbols, spec)

    assert reasons(report) == {"edge:N2->N3": "DECISION_EDGES_INVALID", "edge:N2->N4": "DECISION_EDGES_INVALID",
        "edge:N4->N5": "DECISION_EDGES_INVALID", "edge:N4->N7": "DECISION_EDGES_INVALID",
        "edge:N5->N6": "EDGE_ENDPOINT_UNGROUNDED", "edge:N7->N8": "EDGE_ENDPOINT_UNGROUNDED",
    }
    assert [node["id"] for node in report.spec_final["nodes"]] == ["N1", "N2"]
    assert report.omit_reasons == ["TOO_FEW_NODES", "NO_END", "NO_DECISION"]


# --- Flowchart caps ----------------------------------------------------------


def _tall_flowchart(*, end_last: bool) -> dict[str, Any]:
    """Exceed the node cap; end_last makes trimming remove the sole terminal node."""
    process_lines = list(range(4, _BIG_LAST_STATEMENT + 1))[: DIAGRAM_MAX_NODES + 1]
    end_node = {"id": "NEND", "kind": "end", "label": "return",
        "evidence": {"file": "pkg/big.py", "line": _BIG_RETURN_LINE, "symbol": None},
    }
    nodes: list[dict[str, Any]] = [{"id": "NSTART", "kind": "start", "label": "big",
            "evidence": {"file": "pkg/big.py", "line": 1, "symbol": "big"},
        }, {"id": "NDEC", "kind": "decision", "label": "flag?",
            "evidence": {"file": "pkg/big.py", "line": 2, "symbol": None},
        },
    ]
    if not end_last:
        nodes.append(end_node)
    nodes.extend({"id": f"NP{index}", "kind": "process", "label": f"step {index}",
            "evidence": {"file": "pkg/big.py", "line": line, "symbol": None},
        }
        for index, line in enumerate(process_lines)
    )
    if end_last:
        nodes.append(end_node)
    edges: list[dict[str, Any]] = [
        {"from": "NSTART", "to": "NDEC", "label": None}, {"from": "NDEC", "to": "NEND", "label": "done"},
        {"from": "NDEC", "to": "NP0", "label": "work"},
    ]
    edges.extend({"from": f"NP{index}", "to": f"NP{index + 1}", "label": None}
        for index in range(len(process_lines) - 1)
    )
    return {"root": {"file": "pkg/big.py", "name": "big", "line": 1}, "nodes": nodes, "edges": edges}




def test_flowchart_cap_can_push_a_kind_below_its_floor(repo: Path, symbols: RepoSymbols) -> None:
    report = run_flowchart(repo, symbols, _tall_flowchart(end_last=True))

    # Losing the terminal node also strips the decision's second branch, so the
    # demotion pass runs again on the capped graph.
    assert report.omit_reasons == ["NO_END", "NO_DECISION"]
    # The trimmed end node was grounded; it was cap-dropped, not pruned.
    assert check_for(report, "node", "NEND").grounded
    assert "NEND" not in {n["id"] for n in report.spec_final["nodes"]}


def test_flowchart_node_cap_never_trims_the_start_node(repo: Path, symbols: RepoSymbols) -> None:
    spec = _tall_flowchart(end_last=False)
    start = spec["nodes"].pop(0)
    spec["nodes"].append(start)  # start now sits past the cap in spec order
    report = run_flowchart(repo, symbols, spec)

    ids = [node["id"] for node in report.spec_final["nodes"]]
    assert ids[0] == "NSTART"
    assert len(ids) == DIAGRAM_MAX_NODES

    assert report.ungrounded() == []
    assert report.omit_reasons == []
    assert len(report.spec_final["edges"]) == DIAGRAM_MAX_NODES - 1


# --- Vocabulary contracts ----------------------------------------------------
