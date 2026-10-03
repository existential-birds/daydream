"""Shared recorder, trajectory, manifest, and unified-diff test builders."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from daydream.backends import (
    CostEvent,
    MetricsEvent,
    ResultEvent,
    TextEvent,
    TurnEndEvent,
)
from daydream.trajectory import (
    DaydreamRunFlow,
    Invocation,
    TrajectoryRecorder,
)


def make_recorder(tmp_path: Path, *, run_flow: DaydreamRunFlow = DaydreamRunFlow.NORMAL, agent_model_name: str = "opus",
    on_write: Any = None, path: Path | None = None, session_id: str = "test", **overrides: Any,
) -> TrajectoryRecorder:
    """Build a tmp_path recorder with explicit identity/path and field overrides."""
    return TrajectoryRecorder(
        path=path if path is not None else tmp_path / ".daydream" / "trajectory.json", run_flow=run_flow,
        target_dir=tmp_path, agent_model_name=agent_model_name, session_id=session_id, on_write=on_write, **overrides,
    )


def read_trajectory(path: Path) -> dict[str, Any]:
    """Load the produced trajectory JSON from disk."""
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def trajectory_payload(trajectory_id: str, *, session_id: str = "11111111-2222-3333-4444-555555555555",) -> bytes:
    """Encode an empty trajectory document as a run snapshot stores it."""
    return json.dumps({"session_id": session_id, "trajectory_id": trajectory_id, "steps": []}, sort_keys=True,
    ).encode()


def root_trajectory(repo: Path) -> dict[str, Any]:
    """Load the single run trajectory written under ``repo/.daydream/runs``."""
    paths = list((repo / ".daydream" / "runs").glob("*/trajectory.json"))
    assert len(paths) == 1
    payload = read_trajectory(paths[0])
    assert isinstance(payload, dict)
    return payload


def dispatch_descriptors(step: dict[str, Any]) -> list[str]:
    return [result["content"].removeprefix("Dispatched to ") for result in step["observation"]["results"]]


def dispatch_encloses_children(step: dict[str, Any], target_dir: Path) -> bool:
    children = [read_trajectory(target_dir / ".daydream" / ref["trajectory_path"])
        for result in step["observation"]["results"]
        for ref in result["subagent_trajectory_ref"]
    ]
    return bool(children) and all(step["timestamp"] <= child["extra"]["run_started_at"]
        and step["extra"]["dispatch_completed_at"] >= child["extra"]["run_ended_at"]
        for child in children
    )


def assert_dispatch_children(target_dir: Path, dispatch: dict[str, Any], phase: str, descriptors: list[str],
) -> list[dict[str, Any]]:
    """Assert exact counts, ordered child refs, invocation identity, and time enclosure.

    Read the root and referenced child documents; return the validated children.
    """
    expected_count = len(descriptors)
    assert dispatch["extra"]["planned_count"] == expected_count
    assert dispatch["extra"]["attempted_count"] == expected_count
    assert dispatch["extra"]["completed_count"] == expected_count
    root = root_trajectory(target_dir)
    results = dispatch["observation"]["results"]
    assert [result["content"] for result in results] == [f"Dispatched to {descriptor}" for descriptor in descriptors]
    assert all(len(result["subagent_trajectory_ref"]) == 1 for result in results)
    refs = [result["subagent_trajectory_ref"][0] for result in results]
    assert len({ref["trajectory_id"] for ref in refs}) == expected_count
    assert {ref["session_id"] for ref in refs} == {root["session_id"]}

    summaries = [summary
        for summary in root["extra"]["subtrajectories"]
        if summary.get("dispatch_id") == dispatch["extra"]["dispatch_id"]
    ]
    assert [summary["descriptor"] for summary in summaries] == descriptors
    assert all("invocation_id" not in summary for summary in summaries)

    children: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    for descriptor, ref, summary in zip(descriptors, refs, summaries, strict=True):
        assert Path(ref["trajectory_path"]).name.startswith(f"{descriptor}--")
        child = cast(
            dict[str, Any], json.loads((target_dir / ".daydream" / ref["trajectory_path"]).read_text(encoding="utf-8")),
        )
        assert child["trajectory_id"] == ref["trajectory_id"]
        assert child["session_id"] == root["session_id"]
        assert summary["trajectory_id"] == ref["trajectory_id"]
        assert summary["sibling_trajectory_ref"] == ref["trajectory_path"]
        assert summary["fork_id"] == ref["trajectory_id"]
        assert summary["phase"] == phase
        assert summary["invocations"] == child["extra"]["subtrajectories"]
        assert len(summary["invocations"]) == 1
        invocation = summary["invocations"][0]
        assert invocation["phase"] == phase
        assert invocation["trajectory_id"] == child["trajectory_id"]
        assert (child["extra"]["run_started_at"]
            <= invocation["started_at"]
            <= invocation["ended_at"]
            <= child["extra"]["run_ended_at"]
        )
        identities.add((invocation["trajectory_id"], invocation["invocation_id"]))
        children.append(child)

    assert len(identities) == expected_count
    assert dispatch_encloses_children(dispatch, target_dir)
    return children


def step_token_sum(traj: dict[str, Any], key: str) -> int:
    """Sum present nonzero step metrics for comparison with recorder final totals."""
    return sum(s["metrics"][key] for s in traj["steps"] if s.get("metrics") and s["metrics"].get(key))


def observe_claude_shape(inv: Invocation) -> None:
    """Emit five understated message metrics followed by an authoritative session total.

    This SDK-shaped discrepancy exercises residual token reconciliation.
    """
    for i, c in enumerate((12, 9, 11, 8, 10)):
        inv.observe(TextEvent(text=f"turn {i}"))
        inv.observe(
            MetricsEvent(message_id=f"m{i}", prompt_tokens=100, completion_tokens=c, cached_tokens=None, cost_usd=None)
        )
        inv.observe(TurnEndEvent(message_id=f"m{i}"))
    inv.observe(CostEvent(cost_usd=0.5, input_tokens=600, output_tokens=66_737, cached_tokens=None))
    inv.observe(ResultEvent(structured_output=None, continuation=None))


def observe_text_and_result(inv: Invocation, text: str = "output") -> None:
    """Observe a TextEvent + ResultEvent to produce a minimal agent step."""
    inv.observe(TextEvent(text=text))
    inv.observe(ResultEvent(structured_output=None, continuation=None))


def observe_metrics_and_result(
    inv: Invocation, text: str, *, message_id: str, prompt_tokens: int, completion_tokens: int,
    cached_tokens: int | None, cost_usd: float | None,
) -> None:
    """Observe a TextEvent + MetricsEvent + ResultEvent to produce one agent step."""
    inv.observe(TextEvent(text=text))
    inv.observe(MetricsEvent(message_id=message_id, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
            cached_tokens=cached_tokens, cost_usd=cost_usd,
        )
    )
    inv.observe(ResultEvent(structured_output=None, continuation=None))


def make_manifest(session_id: str = "sess-0001", **overrides: Any) -> dict[str, Any]:
    """Build a minimal indexed manifest with optional field overrides, including PR identity."""
    defaults: dict[str, Any] = {
        "session_id": session_id, "archived_at": "2026-04-29T00:00:00+00:00", "status": "complete",
        "run_flow": "normal", "skill": "python", "model": "opus", "backend": "claude",
        "archive_path": "/tmp/archive/runs/sess-0001",
    }
    defaults.update(overrides)
    return defaults


def diff_adding(line: str, *, file: str = "app.py") -> str:
    """One-hunk unified diff that adds ``line`` to ``file``."""
    return (
        f"diff --git a/{file} b/{file}\n"
        "index 1111111..2222222 100644\n"
        f"--- a/{file}\n"
        f"+++ b/{file}\n"
        "@@ -1,1 +1,2 @@\n"
        " existing\n"
        f"+{line}\n"
    )
