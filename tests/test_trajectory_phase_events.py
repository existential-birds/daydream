"""Tests for trajectory phase timing events."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import anyio
import pytest

import daydream.trajectory as trajectory_module
from daydream import remote_ci
from daydream.atif import validate as atif_validate
from daydream.runner import run
from daydream.trajectory import (
    DaydreamPhase,
    RunWriteSnapshot,
    TrajectoryDocumentSnapshot,
    compute_timing_summary,
    phase_scope,
)
from tests.harness.execution import host_test_command
from tests.harness.git_helpers import bare_remote, git
from tests.harness.remote_ci import NoCIRemote
from tests.harness.review_profile import independent_exploration_profile
from tests.harness.trajectory import make_recorder, observe_text_and_result, read_trajectory
from tests.test_deep_orchestrator import _install_stub_backend


def _snapshot_document(path: Path, payload: dict[str, Any]) -> TrajectoryDocumentSnapshot:
    return TrajectoryDocumentSnapshot(
        trajectory_id=str(payload["trajectory_id"]), path=path, json_bytes=json.dumps(payload, sort_keys=True).encode(),
    )

def test_timing_diagnostics_clip_scoped_events_without_synthesizing_forks(tmp_path: Path) -> None:
    session = "diagnostics"
    def event(kind: str, scope: str | None, timestamp: str, **extra: Any) -> dict[str, Any]:
        return {"phase": "review", "event": kind, "session_id": session,
            "scope_id": scope, "timestamp": timestamp, **extra,
        }
    start = "2026-01-01T00:00:00Z"
    end = "2026-01-01T00:00:10Z"
    invocation = {"trajectory_id": session, "invocation_id": "duplicate", "phase": "review",
        "started_at": start, "ended_at": end,
    }
    root = {"session_id": session, "trajectory_id": session, "steps": [],
        "extra": {"run_started_at": start, "run_ended_at": end, "phase_events": [
            event("phase_start", "clipped", "2025-12-31T23:59:59Z"),
            event("phase_end", "clipped", "2026-01-01T00:00:11Z", status="succeeded"),
            event("phase_start", "duplicate", start), event("phase_start", "duplicate", start),
            event("phase_end", "duplicate", end, status="succeeded"),
            event("phase_start", "orphan", start),
            event("phase_start", "backward", end), event("phase_end", "backward", start, status="succeeded"),
            event("phase_start", None, start), event("phase_end", None, end, status="succeeded"),
            event("phase_start", "invalid-time", "not-a-time"),
            event("phase_end", "invalid-time", end, status="succeeded"),
        ], "subtrajectories": [invocation, invocation,
            {"trajectory_id": session, "invocation_id": "malformed", "phase": "review",
                "started_at": "invalid", "ended_at": end},
        ]},
    }
    child = {"session_id": session, "trajectory_id": "child",
        "steps": [{"timestamp": start}, {"timestamp": end}], "extra": {},
    }
    snapshot = RunWriteSnapshot(status="complete", cutoff_at=end, root_trajectory_id=session,
        documents=(_snapshot_document(tmp_path / "trajectory.json", root),
            _snapshot_document(tmp_path / "child.json", child)),
    )

    summary = compute_timing_summary(snapshot)

    assert summary is not None
    assert summary.phase_timings == {"review": {"wall_clock_seconds": 10.0, "occurrences": 1}}
    assert summary.coverage_ratio == 1.0
    assert summary.diagnostics == {"malformed_interval": 4, "duplicate_interval": 1, "orphaned_interval": 2,
        "malformed_invocation": 1, "duplicate_invocation": 1,
    }
    assert summary.agent_completeness == {"total": 2, "attributed": 0, "unattributed": 2}


# --- PhaseEvent.to_dict ----------------------------------------------------


# --- phase_scope -----------------------------------------------------------


async def test_phase_scope_id_rejects_second_or_post_close_decision(tmp_path: Path,) -> None:
    """A phase handle accepts exactly one caller terminal decision."""
    assert hasattr(trajectory_module, "LifecycleStatus")
    rec = make_recorder(tmp_path)
    async with rec:
        async with phase_scope(DaydreamPhase.REVIEW) as handle:
            handle.finish(
                trajectory_module.LifecycleStatus.PARTIAL, trajectory_module.LifecycleReasonCode.SOME_CHILDREN_FAILED,
            )
            with pytest.raises(RuntimeError, match="terminal decision"):
                handle.finish(trajectory_module.LifecycleStatus.FAILED)
        with pytest.raises(RuntimeError, match="closed"):
            handle.finish(trajectory_module.LifecycleStatus.FAILED)
        async with rec.invocation(phase=DaydreamPhase.REVIEW) as inv:
            observe_text_and_result(inv, "seed")

        # FIX budget exhaustion keeps its wire shape: elapsed_s appears only when measured.
        rec.emit_file_group_budget_exceeded(file="api.py", reason="wall", items_processed=1, items_skipped=2)
        rec.emit_file_group_budget_exceeded(file="ui.tsx", reason="tools", items_processed=0, items_skipped=1,
                                            elapsed_s=1.5)

    events = read_trajectory(rec.path)["extra"]["phase_events"]
    terminal = events[1]
    assert terminal["status"] == "partial"
    assert terminal["reason_code"] == "some_children_failed"
    budgets = [event["metadata"] for event in events if event["event"] == "file_group_budget_exceeded"]
    assert budgets == [
        {"file": "api.py", "reason": "wall", "items_processed": 1, "items_skipped": 2},
        {"file": "ui.tsx", "reason": "tools", "items_processed": 0, "items_skipped": 1, "elapsed_s": 1.5},
    ]


# --- Per-Invocation subtrajectory timestamps -------------------------------


# --- Real-path: deep run via runner.run ------------------------------------


async def test_real_fix_fallback_records_multiple_invocations_in_one_fork(
    multi_stack_target: Path, tmp_path: Path, archive_dir: Path, monkeypatch: pytest.MonkeyPatch, make_config: Any,
    no_ci_remote: NoCIRemote,
) -> None:
    """A failed batch plus serial fallback stays one exact FIX child document."""

    target = multi_stack_target
    origin = bare_remote(tmp_path / "origin.git")
    no_ci_remote.connect(target, origin)
    backend = _install_stub_backend(monkeypatch, target, pin_skill_availability=False)
    backend.fail_batched_fix_file = "api.py"
    backend.fix_edit_line = "\n"
    suite_dir = tmp_path / "suite"

    # Allow discovery/operation time beyond the independently bounded CI wait.
    with anyio.fail_after(30 + remote_ci.DEFAULT_LIMITS.completion_seconds):
        exit_code = await run(make_config(
                target, archive=True, assume="yes", cleanup=False, output_mode="loop", run_eval=True,
                review_profile=independent_exploration_profile(), pr_number=no_ci_remote.pr_number,
                pr_repo=no_ci_remote.base_repository,
                test_command=host_test_command(suite_dir),
            )
        )

    assert exit_code == 0
    assert (suite_dir / "suite-count.txt").read_text() == "1"
    assert backend.test_suite_calls == 0
    assert git(origin, "rev-parse", "refs/heads/feature") == git(target, "rev-parse", "HEAD")
    fix_calls = [call for call in backend.calls if call["prompt"].lower().startswith(("fix this issue", "fix these"))]
    api_batch = [call
        for call in fix_calls
        if call["prompt"].startswith("Fix these 3 issues in ")
        and "api.py:" in call["prompt"].splitlines()[0]
    ]
    api_serial = [call
        for call in fix_calls
        if call["prompt"].startswith("Fix this issue:")
        and any(line.startswith("File: ") and Path(line.removeprefix("File: ")).name == "api.py"
            for line in call["prompt"].splitlines()
        )
    ]
    assert len(api_batch) == 1
    assert len(api_serial) == 3

    roots = list((target / ".daydream" / "runs").glob("*/trajectory.json"))
    assert len(roots) == 1
    root = json.loads(roots[0].read_bytes())
    assert atif_validate(root, validate_images=False) is True
    commit_ends = [event for event in root["extra"]["phase_events"]
        if event["phase"] == "commit" and event["event"] == "phase_end"]
    assert len(commit_ends) == 1
    assert commit_ends[0]["metadata"]["stop_reason"] == "completed"
    assert commit_ends[0]["metadata"]["duration_ms"] >= 0
    fix_summaries = [summary for summary in root["extra"]["subtrajectories"] if summary["phase"] == "fix"]
    assert len(fix_summaries) == 2
    for summary in fix_summaries:
        assert summary["descriptor"].startswith("fix-")
        assert summary["started_at"]
        assert summary["ended_at"]
        assert summary["sibling_trajectory_ref"]
        assert "step_ids" not in summary
    fix_dispatches = [step
        for step in root["steps"]
        if step.get("extra", {}).get("daydream_phase") == "fix"
        and "dispatch_id" in step.get("extra", {})
    ]
    assert len(fix_dispatches) == 1
    dispatch = fix_dispatches[0]
    assert dispatch["extra"]["dispatch_status"] == "succeeded"
    assert dispatch["extra"]["planned_count"] == 2
    assert dispatch["extra"]["attempted_count"] == 2
    assert dispatch["extra"]["completed_count"] == 2
    assert [result["content"] for result in dispatch["observation"]["results"]
    ] == ["Dispatched to fix-api.py", "Dispatched to fix-App.tsx"]
    api_result = dispatch["observation"]["results"][0]
    assert len(api_result["subagent_trajectory_ref"]) == 1
    api_ref = api_result["subagent_trajectory_ref"][0]
    assert Path(api_ref["trajectory_path"]).name.startswith("fix-api-py--")
    api_child = json.loads((target / ".daydream" / api_ref["trajectory_path"]).read_bytes())
    assert api_child["trajectory_id"] == api_ref["trajectory_id"]
    assert api_child["session_id"] == root["session_id"]

    api_summaries = [summary
        for summary in root["extra"]["subtrajectories"]
        if summary.get("dispatch_id") == dispatch["extra"]["dispatch_id"]
        and summary.get("descriptor") == "fix-api.py"
    ]
    assert len(api_summaries) == 1
    api_summary = api_summaries[0]
    assert "invocation_id" not in api_summary
    assert api_summary["trajectory_id"] == api_ref["trajectory_id"]
    assert api_summary["sibling_trajectory_ref"] == api_ref["trajectory_path"]
    assert api_summary["invocations"] == api_child["extra"]["subtrajectories"]
    api_invocations = api_summary["invocations"]
    assert len(api_invocations) == 4
    assert all(invocation["phase"] == "fix" for invocation in api_invocations)
    assert all(invocation["trajectory_id"] == api_child["trajectory_id"] for invocation in api_invocations)
    assert len({(invocation["trajectory_id"], invocation["invocation_id"]) for invocation in api_invocations}) == 4
    assert all(api_child["extra"]["run_started_at"]
        <= invocation["started_at"]
        <= invocation["ended_at"]
        <= api_child["extra"]["run_ended_at"]
        for invocation in api_invocations
    )

    fix_events = [event for event in root["extra"]["phase_events"] if event["phase"] == "fix"]
    assert [event["event"] for event in fix_events] == ["phase_start", "phase_end"]
    assert fix_events[0]["scope_id"] == fix_events[1]["scope_id"]
    assert fix_events[1]["status"] == "succeeded"
    assert dispatch["extra"]["dispatch_started_at"] <= api_child["extra"]["run_started_at"]
    assert api_child["extra"]["run_ended_at"] <= dispatch["extra"]["dispatch_completed_at"]
    assert all(fix_events[0]["timestamp"]
        <= invocation["started_at"]
        <= invocation["ended_at"]
        <= fix_events[1]["timestamp"]
        for invocation in api_invocations
    )

    documents = {root["trajectory_id"]: root}
    document_descriptors = {root["trajectory_id"]: "root"}
    pending = [root]
    wrapper_count = 0
    while pending:
        document = pending.pop()
        for entry in document["extra"].get("subtrajectories", []):
            sibling_path = entry.get("sibling_trajectory_ref")
            if sibling_path is None:
                continue
            wrapper_count += 1
            child = json.loads((target / ".daydream" / sibling_path).read_bytes())
            if child["trajectory_id"] not in documents:
                documents[child["trajectory_id"]] = child
                document_descriptors[child["trajectory_id"]] = Path(sibling_path).name.split("--", 1)[0]
                pending.append(child)
    invocation_keys = {(document_id, invocation["invocation_id"])
        for document_id, document in documents.items()
        for invocation in document["extra"].get("subtrajectories", [])
        if "invocation_id" in invocation
    }
    assert wrapper_count == len(documents) - 1 > 0
    assert len(invocation_keys) == len(backend.calls)
    assert len(invocation_keys) + wrapper_count > len(invocation_keys)
    qualified_counts: dict[tuple[str, str], int] = {}
    for document_id, document in documents.items():
        for invocation in document["extra"].get("subtrajectories", []):
            if "invocation_id" not in invocation:
                continue
            key = (document_descriptors[document_id], invocation["phase"])
            qualified_counts[key] = qualified_counts.get(key, 0) + 1
    assert qualified_counts == {("root", "intent"): 1, ("root", "merge"): 1, ("root", "verify"): 2,
        ("deep-generic", "deep"): 1, ("deep-python", "deep"): 1, ("deep-react", "deep"): 1,
        ("deep-structure", "deep"): 1, ("explore-dependency-tracer", "exploration"): 1, ("fix-api-py", "fix"): 4,
        ("fix-app-tsx", "fix"): 1,
    }

    # The suite is a host operation within the TEST flow scope; it creates no
    # agent invocation and carries its own duration and terminal metadata.
    host_events = [event for event in root["extra"]["phase_events"] if event["phase"] == "test-execution"]
    test_events = [event for event in root["extra"]["phase_events"] if event["phase"] == "test"]
    assert [event["event"] for event in host_events] == ["phase_start", "phase_end"]
    assert [event["event"] for event in test_events] == ["phase_start", "phase_end"]
    assert host_events[0]["scope_id"] == host_events[1]["scope_id"]
    assert host_events[0]["scope_id"] != test_events[0]["scope_id"]
    assert host_events[1]["metadata"]["duration_ms"] >= 0
    assert host_events[1]["metadata"]["stop_reason"] == "completed"
    assert test_events[0]["timestamp"] <= host_events[0]["timestamp"]
    assert host_events[0]["timestamp"] <= host_events[1]["timestamp"] <= test_events[1]["timestamp"]

    verify_invocations = [invocation
        for invocation in root["extra"]["subtrajectories"]
        if invocation.get("phase") == "verify" and "invocation_id" in invocation
    ]
    verify_events = [event for event in root["extra"]["phase_events"] if event["phase"] == "verify"]
    assert len(verify_invocations) == 2
    assert [event["event"] for event in verify_events] == ["phase_start", "phase_end", "phase_start", "phase_end"]
    for start, end, invocation in zip(verify_events[::2], verify_events[1::2], verify_invocations, strict=True):
        assert start["scope_id"] == end["scope_id"]
        assert end["status"] == "succeeded"
        assert start["timestamp"] <= invocation["started_at"]
        assert invocation["ended_at"] <= end["timestamp"]

    archive = archive_dir / "runs" / root["session_id"]
    manifest = json.loads((archive / "manifest.json").read_bytes())
    evaluation = json.loads((archive / "evaluation.json").read_bytes())
    phase_timings = manifest["metrics"]["phase_timings"]
    assert {"intent", "verify", "fix", "test", "deep"} <= phase_timings.keys()
    assert "alternatives" not in phase_timings
    completeness = {"total": len(invocation_keys), "attributed": len(invocation_keys), "unattributed": 0}
    assert manifest["metrics"]["timing_coverage"]["agent_completeness"] == completeness
    assert evaluation["timing"]["agent_completeness"] == completeness
