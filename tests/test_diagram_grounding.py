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

def test_sequence_happy_path_grounds_every_element(repo: Path, symbols: RepoSymbols) -> None:
    report = run_sequence(repo, symbols, base_sequence())

    assert report.ungrounded() == []
    assert report.omit_reasons == []
    assert report.rejected is None
    assert [m["label"] for m in report.spec_final["messages"]] == ["request", "resolve identity", "payload"]
    # Internal calls resolve to real definitions.
    assert check_for(report, "message", "0").strength == "definition"
    assert check_for(report, "message", "0").defined_at == "pkg/api.py:4"
    assert check_for(report, "message", "1").defined_at == "pkg/service.py:1"
    # A reply is proven by a return statement within its enclosing function.
    assert check_for(report, "message", "2").strength == "definition"
    assert check_for(report, "message", "2").defined_at == "pkg/service.py:1"
    assert [check_for(report, "message", str(i)).in_changed_hunk for i in range(3)] == [True, True, True]

# --- Shared reason codes, sequence side --------------------------------------

@pytest.mark.parametrize("index, evidence, reason", [
    pytest.param(1, {"file": "../outside/secrets.py"}, "PATH_ESCAPES_REPO", id="path-escapes"),
    pytest.param(1, {"line": 999}, "LINE_OUT_OF_RANGE", id="line-range"),
    pytest.param(1, {"file": "pkg/service.py", "line": 1, "symbol": "resolve"},
                 "EVIDENCE_NOT_IN_SOURCE_PARTICIPANT", id="wrong-source"),
    # handle is defined by the caller rather than the Service participant.
    pytest.param(1, {"file": "pkg/api.py", "line": 4, "symbol": "handle"},
                 "CALLEE_NOT_DEFINED_IN_TARGET", id="wrong-callee"),
    pytest.param(2, {"line": 1}, "NOT_A_REPLY_STATEMENT", id="reply-without-return"),
])
def test_sequence_message_evidence_failures(
    repo: Path, symbols: RepoSymbols, index: int, evidence: dict[str, Any], reason: str,
) -> None:
    spec = base_sequence()
    spec["messages"][index]["evidence"].update(evidence)
    report = run_sequence(repo, symbols, spec)
    assert check_for(report, "message", str(index)).reason == reason

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


def test_sequence_symbol_snap_rewrites_the_citation(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_sequence()
    # "resolve" is not on pkg/api.py:4 but is one line below it.
    spec["messages"][1]["evidence"]["line"] = 4
    report = run_sequence(repo, symbols, spec)

    check = check_for(report, "message", "1")
    assert check.grounded and check.reason is None
    assert check.snapped_line == 5
    assert report.spec_final["messages"][1]["evidence"]["line"] == 5

def test_sequence_branch_not_a_branch_statement(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_sequence()
    spec["blocks"] = [{"kind": "opt",
            "branches": [{"condition": "assignment is not a branch", "evidence": {"file": "pkg/flow.py", "line": 5},
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




def test_sequence_reply_requires_a_reversed_preceding_call(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_sequence()
    spec["messages"][1]["kind"] = "self"

    report = run_sequence(repo, symbols, spec)

    assert check_for(report, "message", "2").reason == "REPLY_NOT_PRECEDED_BY_CALL"

def test_sequence_reply_requires_a_grounded_preceding_call(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_sequence()
    spec["messages"][1]["evidence"]["line"] = 999

    report = run_sequence(repo, symbols, spec)

    assert check_for(report, "message", "1").reason == "LINE_OUT_OF_RANGE"
    assert check_for(report, "message", "2").reason == "REPLY_NOT_PRECEDED_BY_CALL"
    assert all(message["kind"] != "reply" for message in report.spec_final["messages"])

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

def test_sequence_participant_no_files(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_sequence()
    spec["participants"][2]["files"] = []
    report = run_sequence(repo, symbols, spec)
    assert check_for(report, "participant", "Service").reason == "PARTICIPANT_NO_FILES"
    # Its messages go with it.
    assert check_for(report, "message", "1").reason == "CALLEE_NOT_DEFINED_IN_TARGET"
    assert check_for(report, "message", "2").reason == "EVIDENCE_NOT_IN_SOURCE_PARTICIPANT"

@pytest.mark.parametrize("index, files, ref, reason", [
    pytest.param(2, ["pkg/vanished.py"], "Service", "PARTICIPANT_FILE_MISSING", id="missing-file"),
    pytest.param(2, ["../elsewhere/service.py"], "Service", "PATH_ESCAPES_REPO", id="path-escapes"),
    pytest.param(0, ["pkg/api.py"], "Client", "EXTERNAL_MISUSED", id="external-declares-files"),
])
def test_sequence_participant_file_failures(
    repo: Path, symbols: RepoSymbols, index: int, files: list[str], ref: str, reason: str,
) -> None:
    spec = base_sequence()
    spec["participants"][index]["files"] = files
    report = run_sequence(repo, symbols, spec)
    assert check_for(report, "participant", ref).reason == reason



def test_sequence_external_misused_by_sourcing_a_later_message(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_sequence()
    spec["messages"][1]["from"] = "Client"
    report = run_sequence(repo, symbols, spec)
    assert check_for(report, "participant", "Client").reason == "EXTERNAL_MISUSED"
    # Message 0 loses its (now ungrounded) source too.
    assert check_for(report, "message", "0").reason == "EVIDENCE_NOT_IN_SOURCE_PARTICIPANT"

def test_sequence_malformed_elements(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_sequence()
    spec["participants"].append({"name": "Ghost", "kind": "spectral", "files": [], "service": None})
    spec["participants"].append({"name": "API", "kind": "internal", "files": ["pkg/api.py"], "service": None})
    spec["messages"].append("not a message")
    spec["messages"].append({**base_sequence()["messages"][1], "kind": "telepathy"})
    spec["blocks"].append({"kind": "whenever", "branches": [{"condition": "x"}]})
    report = run_sequence(repo, symbols, spec)

    assert check_for(report, "participant", "Ghost").reason == "MALFORMED_ELEMENT"
    # The duplicate name is rejected, not silently allowed to shadow the first.
    assert [c.reason for c in report.elements if c.element == "participant"].count("MALFORMED_ELEMENT") == 2
    assert check_for(report, "message", "3").reason == "MALFORMED_ELEMENT"
    assert check_for(report, "message", "4").reason == "MALFORMED_ELEMENT"
    assert check_for(report, "block", "b0").reason == "MALFORMED_ELEMENT"
    assert check_for(report, "branch", "b0.0").reason == "MALFORMED_ELEMENT"

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

def test_sequence_prune_flattens_a_block_whose_condition_is_ungrounded(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_sequence()
    spec["blocks"] = [{"kind": "alt",
            "branches": [
                {"condition": "token missing", "evidence": {"file": "pkg/flow.py", "line": 2}, "messages": [0]},
                {
                    # Line 5 is an assignment: not a branch statement.
                    "condition": "fabricated", "evidence": {"file": "pkg/flow.py", "line": 5}, "messages": [1, 2],
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

def test_sequence_preserves_message_order_after_a_mid_list_prune(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_sequence()
    spec["messages"].insert(1,
        {"from": "API", "to": "Service", "label": "fabricated", "kind": "call", "changed": True,
            "evidence": {"file": "pkg/api.py", "line": 999, "symbol": "resolve"},
        },
    )
    report = run_sequence(repo, symbols, spec)

    assert check_for(report, "message", "1").reason == "LINE_OUT_OF_RANGE"
    assert [m["label"] for m in report.spec_final["messages"]] == ["request", "resolve identity", "payload"]

def test_sequence_block_message_indices_are_remapped_after_a_prune(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_sequence()
    # After the entrypoint, so message 0 keeps its "first message" status -- an
    # external participant may only source the entrypoint.
    spec["messages"].insert(1,
        {"from": "API", "to": "Service", "label": "fabricated", "kind": "call", "changed": True,
            "evidence": {"file": "pkg/ghost.py", "line": 1, "symbol": "resolve"},
        },
    )
    spec["blocks"] = [{"kind": "opt",
            "branches": [
                {"condition": "token missing", "evidence": {"file": "pkg/flow.py", "line": 2}, "messages": [1, 2, 3]}
            ],
        }
    ]
    report = run_sequence(repo, symbols, spec)

    assert check_for(report, "message", "1").reason == "FILE_MISSING"
    # Proposed indices 2 and 3 became final positions 1 and 2; index 1 is gone.
    assert report.spec_final["blocks"][0]["branches"][0]["messages"] == [1, 2]

def test_sequence_opt_block_keeps_only_its_first_branch(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_sequence()
    spec["blocks"] = [{"kind": "opt",
            "branches": [{"condition": "first", "evidence": {"file": "pkg/flow.py", "line": 2}, "messages": [0]},
                {"condition": "second", "evidence": {"file": "pkg/flow.py", "line": 4}, "messages": [1]},
            ],
        }
    ]
    report = run_sequence(repo, symbols, spec)

    assert [b["condition"] for b in report.spec_final["blocks"][0]["branches"]] == ["first"]
    # The surplus branch was grounded; it is normalized away, not pruned.
    assert check_for(report, "branch", "b0.1").grounded


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

def test_sequence_message_cap_truncates_the_tail(repo: Path, symbols: RepoSymbols) -> None:
    spec = _wide_sequence(2)
    template = spec["messages"][0]
    spec["messages"] = [{**template, "evidence": dict(template["evidence"]), "label": f"step {index}"}
        for index in range(DIAGRAM_MAX_MESSAGES + 5)
    ]
    report = run_sequence(repo, symbols, spec)

    assert len(report.spec_final["messages"]) == DIAGRAM_MAX_MESSAGES
    assert report.spec_final["messages"][-1]["label"] == f"step {DIAGRAM_MAX_MESSAGES - 1}"
    assert report.omit_reasons == []

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

def test_sequence_cap_can_push_a_kind_below_its_floor(repo: Path, symbols: RepoSymbols) -> None:
    spec = _wide_sequence(3)
    unchanged, changed = spec["messages"][0], spec["messages"][1]
    spec["messages"] = [{**unchanged, "evidence": dict(unchanged["evidence"]), "label": f"step {index}"}
        for index in range(DIAGRAM_MAX_MESSAGES)
    ] + [{**changed, "evidence": dict(changed["evidence"]), "label": f"changed {index}"} for index in range(2)]
    report = ground_sequence(spec, repo_root=repo, hunk_ranges={"pkg/p01.py": [(1, 3)]}, symbols=symbols,)

    assert len(report.spec_final["messages"]) == DIAGRAM_MAX_MESSAGES
    assert not any(m["label"].startswith("changed") for m in report.spec_final["messages"])
    assert report.omit_reasons == ["NO_CHANGED_INTERACTION"]


# --- Flowchart root ----------------------------------------------------------

def test_flowchart_root_must_still_overlap_a_changed_hunk(repo: Path, symbols: RepoSymbols) -> None:
    report = ground_flowchart(
        base_flowchart(), repo_root=repo, hunk_ranges={"pkg/api.py": [(4, 6)]}, candidate_roots=[FLOW_ROOT],
        symbols=symbols,
    )
    assert report.rejected == "ROOT_NOT_CANDIDATE"


# --- Flowchart node reason codes ---------------------------------------------

@pytest.mark.parametrize("index, evidence, reason", [
    pytest.param(6, {"file": "../outside/flow.py"}, "PATH_ESCAPES_REPO", id="path-escapes"),
    pytest.param(6, {"file": "pkg/ghost.py"}, "FILE_MISSING", id="missing-file"),
    pytest.param(6, {"line": 9999}, "LINE_OUT_OF_RANGE", id="line-range"),
    pytest.param(6, {"symbol": "resolve_identity"}, "SYMBOL_NOT_ON_LINE", id="symbol-location"),
    pytest.param(7, {"line": 8}, "NOT_A_TERMINAL_STATEMENT", id="assignment-not-terminal"),
    pytest.param(3, {"line": 5}, "NOT_A_BRANCH_STATEMENT", id="call-not-branch"),
    pytest.param(4, {"symbol": "nonexistent_helper"}, "SUBROUTINE_NOT_CALLED_HERE", id="uncalled-symbol"),
    # ValueError occurs at the call site but has no definition in this repository.
    pytest.param(4, {"file": "pkg/flow.py", "line": 3, "symbol": "ValueError"},
                 "SUBROUTINE_NOT_DEFINED", id="undefined-subroutine"),
    pytest.param(4, {"symbol": None}, "MALFORMED_ELEMENT", id="subroutine-without-symbol"),
])
def test_flowchart_node_evidence_failures(
    repo: Path, symbols: RepoSymbols, index: int, evidence: dict[str, Any], reason: str,
) -> None:
    spec = base_flowchart()
    spec["nodes"][index]["evidence"].update(evidence)
    report = run_flowchart(repo, symbols, spec)
    assert check_for(report, "node", f"N{index + 1}").reason == reason




def test_flowchart_node_symbol_snap_stays_inside_the_root(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_flowchart()
    # "verify_jwt" is on line 5; the node cites line 6 and snaps back one line.
    spec["nodes"][4]["evidence"]["line"] = 6
    report = run_flowchart(repo, symbols, spec)

    check = check_for(report, "node", "N5")
    assert check.grounded and check.snapped_line == 5
    assert report.spec_final["nodes"][4]["evidence"]["line"] == 5

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
    spec["nodes"].append({"id": "N9", "kind": "start", "label": "second entry",
            "evidence": {"file": "pkg/flow.py", "line": 1, "symbol": "resolve_identity"},
        }
    )
    spec["edges"].append({"from": "N9", "to": "N2", "label": None})
    report = run_flowchart(repo, symbols, spec)

    assert check_for(report, "node", "N9").reason == "MULTIPLE_START"
    assert check_for(report, "node", "N1").grounded
    assert check_for(report, "edge", "N9->N2").reason == "EDGE_ENDPOINT_UNGROUNDED"

def test_flowchart_decision_edges_invalid(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_flowchart()
    spec["edges"][2]["label"] = None  # N2 -> N4 loses its label
    spec["edges"].append({"from": "N4", "to": "N6", "label": "bearer"})  # duplicate label
    report = run_flowchart(repo, symbols, spec)

    assert check_for(report, "edge", "N2->N4").reason == "DECISION_EDGES_INVALID"
    assert check_for(report, "edge", "N4->N6").reason == "DECISION_EDGES_INVALID"

def test_flowchart_preserves_live_nodes_and_edges_after_a_mid_list_prune(repo: Path, symbols: RepoSymbols) -> None:
    spec = base_flowchart()
    spec["nodes"].insert(2,
        {"id": "NX", "kind": "process", "label": "fabricated",
            "evidence": {"file": "pkg/flow.py", "line": 9999, "symbol": None},
        },
    )
    spec["edges"].insert(1, {"from": "N2", "to": "NX", "label": "bogus"})
    report = run_flowchart(repo, symbols, spec)

    assert check_for(report, "edge", "N2->NX").reason == "EDGE_ENDPOINT_UNGROUNDED"
    assert report.spec_final["nodes"] == base_flowchart()["nodes"]
    assert report.spec_final["edges"] == base_flowchart()["edges"]


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

def test_flowchart_node_cap_trims_the_tail_and_keeps_rendering(repo: Path, symbols: RepoSymbols) -> None:
    report = run_flowchart(repo, symbols, _tall_flowchart(end_last=False))

    assert report.ungrounded() == []
    assert len(report.spec_final["nodes"]) == DIAGRAM_MAX_NODES
    assert report.omit_reasons == []
    assert report.spec_final["nodes"][0]["id"] == "NSTART"
    assert len(report.spec_final["edges"]) == DIAGRAM_MAX_NODES - 1

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


# --- Vocabulary contracts ----------------------------------------------------
