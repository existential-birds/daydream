"""Tests for runtime findings and tool supervision."""

from __future__ import annotations

from daydream.extensions import ToolDecision
from daydream.supervision import (
    RuleBasedSupervisor,
    RuleBasedToolSupervisor,
    apply_findings_verdicts,
    revise_finding_fields,
)
from daydream.trajectory import DaydreamPhase


def test_revise_finding_fields_updates_whitelist_only() -> None:
    item = {
        "id": 7,
        "severity": "high",
        "confidence": "medium",
        "description": "original",
        "rationale": "because",
        "evidence": ["line 1"],
        "file": "safe.py",
        "line": 12,
    }
    item_id = id(item)

    revise_finding_fields(
        item,
        {"severity": "low", "file": "hacked.py", "line": 999, "reason": "x"},
    )

    assert id(item) == item_id
    assert item["severity"] == "low"
    assert item["file"] == "safe.py"
    assert item["line"] == 12
    assert item["id"] == 7


def test_apply_findings_verdicts_handles_actions_and_fails_open() -> None:
    items = [
        {"id": 1, "description": "allowed", "severity": "low"},
        {"id": 2, "description": "duplicate", "severity": "medium"},
        {"id": 3, "description": "needs edit", "severity": "high"},
        {"id": 4, "description": "held", "severity": "low"},
        {"id": 5, "description": "missing verdict", "severity": "medium"},
    ]

    kept, held, events = apply_findings_verdicts(
        items,
        {
            1: {"id": 1, "action": "allow", "reason": "confirmed"},
            2: {"id": 2, "action": "drop", "reason": "dup"},
            3: {"id": 3, "action": "edit", "reason": "more precise", "severity": "low"},
            4: {"id": 4, "action": "hold", "reason": "needs review"},
            99: {"id": 99, "action": "drop", "reason": "unknown"},
        },
    )

    assert [item["id"] for item in kept] == [1, 3, 5]
    assert [item["id"] for item in held] == [4]
    assert kept[1]["severity"] == "low"
    assert events == [
        (2, "drop", "dup"),
        (3, "edit", "more precise"),
        (4, "hold", "needs review"),
    ]


def test_rule_based_supervisor_drops_matching_repo_relative_files() -> None:
    items = [
        {"id": 1, "file": "vendor/x.py", "description": "vendored"},
        {"id": 2, "file": "src/app.py", "description": "application"},
    ]

    verdicts = RuleBasedSupervisor(deny_globs=["vendor/**"]).review_findings(items)

    assert verdicts == {
        1: {"id": 1, "action": "drop", "reason": "denied by glob 'vendor/**'"}
    }


def test_rule_based_tool_supervisor_vetoes_paths_and_bash() -> None:
    supervisor = RuleBasedToolSupervisor(
        deny_globs=["vendor/**"],
        bash_deny=[r"rm -rf"],
    )

    write_decision = supervisor(
        "Write",
        {"file_path": "/repo/vendor/x.py"},
        phase=DaydreamPhase.FIX,
    )
    edit_decision = supervisor(
        "Edit",
        {"path": "/repo/vendor/y.py"},
        phase=DaydreamPhase.FIX,
    )
    allowed_decision = supervisor(
        "Write",
        {"file_path": "/repo/src/app.py"},
        phase=DaydreamPhase.FIX,
    )
    bash_decision = supervisor(
        "Bash",
        {"command": "rm -rf /"},
        phase=DaydreamPhase.FIX,
    )
    unknown_decision = supervisor("Read", {}, phase=DaydreamPhase.FIX)

    assert isinstance(write_decision, ToolDecision) and write_decision.veto
    assert "vendor/**" in write_decision.reason
    assert edit_decision.veto
    assert not allowed_decision.veto
    assert bash_decision.veto and "rm -rf" in bash_decision.reason
    assert not unknown_decision.veto
