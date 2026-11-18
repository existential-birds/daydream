"""Shared recorder, evidence document, manifest, and unified-diff test builders."""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

from daydream.archive.manifest import Manifest
from daydream.backends import (
    ResultEvent,
    TextEvent,
)
from daydream.dataset import portable_evidence_document
from daydream.trajectory import (
    RUN_DOCUMENT_NAME,
    DaydreamRunFlow,
    EvidenceRecorder,
    Invocation,
    TrajectoryDocumentReference,
    read_document_view,
)
from daydream.trajectory.journal import JournalWriter
from daydream.trajectory.view import OPEN_EVENT, PHASE_EVENT, SNAPSHOT_EVENT, STEP_EVENT, SUBTRAJECTORY_EVENT


def make_recorder(tmp_path: Path, *, run_flow: DaydreamRunFlow = DaydreamRunFlow.NORMAL, agent_model_name: str = "opus",
    on_write: Any = None, path: Path | None = None, session_id: str = "test", **overrides: Any,
) -> EvidenceRecorder:
    """Build a tmp_path recorder with explicit identity/path and field overrides."""
    return EvidenceRecorder(
        path=path if path is not None else tmp_path / ".daydream" / RUN_DOCUMENT_NAME, run_flow=run_flow,
        target_dir=tmp_path, agent_model_name=agent_model_name, session_id=session_id, on_write=on_write, **overrides,
    )


def read_trajectory(path: Path | TrajectoryDocumentReference) -> dict[str, Any]:
    """Reduce the evidence document at *path* (or a reference's prefix) to its view."""
    return read_document_view(path.read_bytes())


def write_evidence(path: Path, payload: Mapping[str, Any], *, parent: str | None = None, sealed: bool = True,
) -> TrajectoryDocumentReference:
    """Write a document view's content as real evidence records at *path* and reference the whole file.

    The view's ``agent.model_name``, ``steps`` (numbered by position when they carry no ``step_id``),
    ``final_metrics`` and ``extra`` become records: ``extra.phase_events`` and ``extra.subtrajectories``
    their own records, ``extra.partial`` an aborted seal. ``extra.snapshot_at`` leaves the document
    unsealed behind a snapshot record; otherwise it is sealed at ``extra.run_ended_at`` (default
    ``2026-01-01T00:00:01Z``) unless *sealed* is false."""
    extra = dict(payload.get("extra") or {})
    ended = extra.pop("run_ended_at", "2026-01-01T00:00:01Z")
    snapshot_at = extra.pop("snapshot_at", None)
    phase_events = extra.pop("phase_events", [])
    subtrajectories = extra.pop("subtrajectories", [])
    aborted = bool(extra.pop("partial", False))
    metrics = dict(payload.get("final_metrics") or {})
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = JournalWriter(path, resource={}, version="test")
    writer.append(OPEN_EVENT, {
        "session_id": payload["session_id"], "trajectory_id": payload["trajectory_id"], "descriptor": "",
        "parent_trajectory_id": parent, "run_flow": "normal", "agent_version": "test",
        "agent_model_name": (payload.get("agent") or {}).get("model_name", ""), "extra": extra,
    })
    for index, step in enumerate(payload.get("steps") or [], start=1):
        writer.append(STEP_EVENT, {"step_id": index, **step})
    for event in phase_events:
        writer.append(PHASE_EVENT, event)
    for summary in subtrajectories:
        writer.append(SUBTRAJECTORY_EVENT, summary)
    if snapshot_at is not None:
        writer.append(SNAPSHOT_EVENT, {"snapshot_at": snapshot_at, "final_metrics": metrics})
    elif sealed:
        writer.seal({"run_ended_at": ended, "aborted": aborted, "capture_complete": True, "failed_documents": [],
                     "final_metrics": metrics})
    writer.close()
    size, sha256 = writer.reference()
    return TrajectoryDocumentReference(payload["trajectory_id"], path, path, size, sha256)


def evidence_reference(payload: Mapping[str, Any], private_path: Path, *, logical_path: Path | None = None,
) -> TrajectoryDocumentReference:
    """:func:`write_evidence`, optionally naming the reference by a logical path other than its file."""
    reference = write_evidence(private_path, payload)
    return reference if logical_path is None else replace(reference, path=logical_path)


def evidence_document(
    trajectory_id: str, *, session_id: str = "11111111-2222-3333-4444-555555555555",
    steps: tuple[dict[str, Any], ...] = (), extra: dict[str, Any] | None = None,
    subtrajectories: tuple[dict[str, Any], ...] = (), parent: str | None = None, sealed: bool = True,
) -> bytes:
    """The bytes of one evidence document written through the real journal writer."""
    with tempfile.TemporaryDirectory() as raw:
        path = Path(raw) / RUN_DOCUMENT_NAME
        write_evidence(path, {
            "session_id": session_id, "trajectory_id": trajectory_id, "agent": {"model_name": "opus"},
            "steps": list(steps), "final_metrics": {"total_steps": len(steps)},
            "extra": {"run_started_at": "2026-01-01T00:00:00Z", "run_ended_at": "2026-01-01T00:01:00Z",
                      **(extra or {}), "subtrajectories": list(subtrajectories)},
        }, parent=parent, sealed=sealed)
        return path.read_bytes()


def portable_document(trajectory_id: str, *, session_id: str, **fields: Any) -> dict[str, Any]:
    """One evidence document as a dataset run record embeds it."""
    return portable_evidence_document(evidence_document(trajectory_id, session_id=session_id, **fields),
                                      trajectory_id=trajectory_id, session_id=session_id)


def root_trajectory(repo: Path) -> dict[str, Any]:
    """Reduce the single root evidence document written under ``repo/.daydream/runs``."""
    paths = list((repo / ".daydream" / "runs").glob(f"*/{RUN_DOCUMENT_NAME}"))
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
    # Summaries land when each child finalizes; dispatch refs retain planned
    # order. Bind these independent representations by document identity.
    assert len(summaries) == expected_count
    summaries_by_id = {summary["trajectory_id"]: summary for summary in summaries}
    assert set(summaries_by_id) == {ref["trajectory_id"] for ref in refs}
    assert all("invocation_id" not in summary for summary in summaries)

    children: list[dict[str, Any]] = []
    identities: set[tuple[str, str]] = set()
    for descriptor, ref in zip(descriptors, refs, strict=True):
        summary = summaries_by_id[ref["trajectory_id"]]
        assert summary["descriptor"] == descriptor
        assert Path(ref["trajectory_path"]).name.startswith(f"{descriptor}--")
        child = read_trajectory(target_dir / ".daydream" / ref["trajectory_path"])
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


def observe_text_and_result(inv: Invocation, text: str = "output") -> None:
    """Observe a TextEvent + ResultEvent to produce a minimal agent step."""
    inv.observe(TextEvent(text=text))
    inv.observe(ResultEvent(structured_output=None, continuation=None))


def make_manifest(session_id: str = "sess-0001", **overrides: Any) -> Manifest:
    """Build a minimal indexed manifest with optional field overrides, including PR identity."""
    defaults: dict[str, Any] = {
        "session_id": session_id, "archived_at": "2026-04-29T00:00:00+00:00", "status": "complete",
        "run_flow": "normal", "backend": "claude",
        "archive_path": "/tmp/archive/runs/sess-0001",
    }
    defaults.update(overrides)
    return Manifest(**defaults)


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

