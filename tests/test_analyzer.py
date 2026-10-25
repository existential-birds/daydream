"""Trajectory resolution, finding/location arithmetic, and source-quality metrics."""

import json
import math
import os
import uuid
from pathlib import Path
from typing import Any, Literal

import pytest

from daydream import _tree_sitter_safety as safety
from daydream.artifact_visibility import ArtifactEvidenceProvenance
from daydream.backends import MetricsEvent, ResultEvent, TextEvent
from daydream.deep.records import RECORD_SOURCE_UIDS_KEY, mint_record_uid
from daydream.eval import quality as quality_mod
from daydream.eval.analyzer import (
    _agent_label,
    analyze_costs,
    analyze_findings,
    analyze_location,
    analyze_quality,
    analyze_session,
    analyze_shipped_duplication,
)
from daydream.eval.quality import _quality_python_parser
from daydream.git_ops import GitError
from daydream.trajectory import (
    DaydreamPhase,
    RunWriteSnapshot,
    TrajectoryDocumentSnapshot,
)
from tests.harness.git_helpers import git
from tests.harness.trajectory import make_recorder


def _snapshot(root: dict[str, Any], *forks: dict[str, Any], cutoff_at: str = "2026-01-01T00:00:10Z",
    status: Literal["complete", "partial"] = "complete",
) -> RunWriteSnapshot:
    """Prepare explicitly identified fixture documents without reading trajectory files."""
    return RunWriteSnapshot(status=status, cutoff_at=cutoff_at, root_trajectory_id=root["trajectory_id"],
        documents=tuple(TrajectoryDocumentSnapshot(payload["trajectory_id"], Path(payload["_source_file"]),
                json.dumps({key: value for key, value in payload.items() if key != "_source_file"}).encode(),
            ) for payload in (root, *forks)),
    )


@pytest.mark.parametrize(("status", "cutoff_at", "wall"), [
    ("partial", "2026-01-01T00:00:10Z", None),
    ("complete", "2026-01-01T00:00:04Z", None),
])
def test_evaluation_uses_frozen_bytes_status_and_cutoff(tmp_path: Path,
    status: Literal["complete", "partial"], cutoff_at: str, wall: float | None,
) -> None:
    root: dict[str, Any] = {"session_id": "frozen", "trajectory_id": "frozen", "_source_file": "trajectory.json",
        "steps": [{"timestamp": "2026-01-01T00:00:02Z"}, {"timestamp": "2026-01-01T00:00:03Z"}],
        "final_metrics": {"total_cost_usd": 0.5},
        "extra": {"partial": False, "run_started_at": "2026-01-01T00:00:00Z",
            "run_ended_at": "2026-01-01T00:00:10Z", "snapshot_at": "2026-01-01T00:00:04Z"},
    }
    snapshot = _snapshot(root, status=status, cutoff_at=cutoff_at)
    snapshot.validate("frozen")
    live_path = tmp_path / ".daydream" / "runs" / "frozen" / "trajectory.json"
    live_path.parent.mkdir(parents=True)
    live_path.write_bytes(snapshot.documents[0].json_bytes)
    root["final_metrics"]["total_cost_usd"] = 99.0
    root["extra"]["run_ended_at"] = "2026-01-01T00:01:00Z"
    live_path.write_text(json.dumps(root))

    result = analyze_session(tmp_path / ".daydream", write_snapshot=snapshot)

    assert result["cost"]["total_cost_usd"] == 0.5
    assert result["timing"]["total_wall_clock_seconds"] == wall
    assert result["timing"]["by_agent"] == [{"agent": "main", "duration_seconds": 1.0}]


def test_training_signals_preserve_independent_fork_outcomes_in_archive(tmp_path: Path, archive_dir: Path) -> None:
    from daydream.run_config import RunConfig
    from tests.test_archive_integration import _finalize_strict_archive

    call = {"tool_call_id": "call-1", "function_name": "Read", "arguments": {}}
    step = {"step_id": 1, "tool_calls": [call]}
    completed = {"source_call_id": "call-1", "content": "output"}
    population: dict[str, list[dict[str, Any]]] = {
        "failed": [step | {"observation": {"results": [completed | {"extra": {"is_error": True}}]}}],
        "interrupted": [step | {"observation": {"results": [completed | {"extra": {"status": "interrupted"}}]}}],
        "cancelled": [step | {"observation": {"results": [completed | {"extra": {"cancelled": True}}]}}],
        "unpaired": [step | {"observation": {"results": [{"source_call_id": None, "content": "interrupted",
            "extra": {"is_error": True, "status": "interrupted"}}]}}],
        "step-local": [step, {"step_id": 2,
            "observation": {"results": [completed | {"extra": {"is_error": False}}]}}],
        "explicit-unmatched": [{"step_id": 1, "extra": {"unmatched_tool_results": ["another"]}}],
        "legacy-missing": [step | {"observation": {"results": [completed]}}],
        "legacy-empty": [step | {"observation": {"results": [completed | {"extra": {}}]}}],
        "legacy-error-string": [step | {"observation": {"results": [completed | {"extra": {"is_error": "true"}}]}}],
        "legacy-cancel-int": [step | {"observation": {"results": [completed | {"extra": {"cancelled": 1}}]}}],
        "legacy-list": [step | {"observation": {"results": [completed | {"extra": []}]}}],
    }
    target = tmp_path / "workspace"
    target.mkdir()
    recorder = make_recorder(target, session_id="training-outcomes")
    root = _root_trajectory(recorder.session_id)
    snapshot = _snapshot(root, *(
        root | {"_source_file": f"{name}.json", "trajectory_id": name, "steps": steps}
        for name, steps in population.items()
    ))
    _finalize_strict_archive(recorder, snapshot, RunConfig(run_eval=True, archive=True), target)
    result = json.loads((archive_dir / "runs/training-outcomes/evaluation.json").read_text())["training_signals"]
    assert {row["trajectory"]: (row["noise_flags"], row["training_quality"]) for row in result["trajectories"]} == {
        "failed": (["failed_tool_result"], "review"),
        "interrupted": (["incomplete_tool_call"], "review"),
        "cancelled": (["incomplete_tool_call"], "review"),
        "unpaired": (["incomplete_tool_call"], "review"),
        "step-local": (["incomplete_tool_call", "unmatched_tool_result"], "review"),
        "explicit-unmatched": (["unmatched_tool_result"], "review"),
        "legacy-missing": ([], "clean"), "legacy-empty": ([], "clean"), "legacy-error-string": ([], "clean"),
        "legacy-cancel-int": ([], "clean"), "legacy-list": ([], "clean"),
    }
    assert (result["total_trajectories"], result["clean_for_training"], result["needs_review"]) == (11, 5, 6)



def test_analyze_costs_aggregates_legacy_fork_metrics() -> None:
    trajectories = {"main": None,
        "forked": [{"_source_file": "legacy-main.json", "final_metrics": {"total_cost_usd": 1.0}},
            {"_source_file": "fork.json", "final_metrics": {"total_cost_usd": 0.5}}],
    }

    result = analyze_costs(trajectories)

    assert result["total_cost_usd"] == 1.5
    assert [(row["agent"], row["cost_usd"]) for row in result["by_agent"]] == [("legacy-main", 1.0), ("fork", 0.5)]


async def test_analyze_costs_assigns_nested_forks_their_own_metrics(tmp_path: Path, archive_dir: Path) -> None:
    """One recorded workload owns raw-tool accounting, transitive billing and persisted evaluation.

    Regressions in alias classification or fork subtraction must change this run's exact histogram,
    cache ratio or index row; isolated analyzer dictionaries cannot prove the recorder's folded scope.
    """
    from daydream.backends import DiagnosticEvent, ToolResultEvent, ToolStartEvent
    from daydream.run_config import RunConfig
    from tests.test_archive_integration import _finalize_strict_archive

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    daydream_dir = workspace / ".daydream"
    snapshots: list[RunWriteSnapshot] = []
    recorder = make_recorder(
        workspace,
        on_write=lambda _rec, snapshot: snapshots.append(snapshot),
        path=daydream_dir / "runs" / "nested-forks" / "trajectory.json",
        agent_model_name="opus",
        session_id="nested-forks",
    )
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="main"))
            # One native event stream contains aliases from the supported adapters; production
            # records and analyzes the mixed workload together, preserving the original names.
            for index, name in enumerate(
                (
                    "Write",
                    "write",
                    "Edit",
                    "edit",
                    "MultiEdit",
                    "multiedit",
                    "NotebookEdit",
                    "notebookedit",
                    "patch",
                    "apply_patch",
                    "Read",
                    "read",
                    "shell",
                    "bash",
                    "custom",
                )
            ):
                inv.observe(ToolStartEvent(str(index), name, {}))
                inv.observe(ToolResultEvent(str(index), "recorded output", False))
            inv.observe(MetricsEvent("main", 140, 14, 14, 0.00006))
            inv.observe(ResultEvent(structured_output=None, continuation=None))
        async with recorder.fork("empty-child") as empty:
            pass
        async with recorder.fork("outer") as outer:
            async with outer.invocation(phase=DaydreamPhase.DEEP) as inv:
                inv.observe(TextEvent(text="outer"))
                inv.observe(DiagnosticEvent("codex_parser_coverage", "bounded parser evidence", {}))
                inv.observe(DiagnosticEvent("arbitrary_backend_note", "retained raw evidence", {}))
                inv.observe(DiagnosticEvent("codex_transport_coverage", "bounded transport evidence", {}))
                inv.observe(DiagnosticEvent("codex_parser_coverage", "repeated parser evidence", {}))
                inv.observe(MetricsEvent("outer", 20, 2, 2, 0.2))
                inv.observe(ResultEvent(structured_output=None, continuation=None))
            async with outer.fork("inner") as inner:
                async with inner.invocation(phase=DaydreamPhase.DEEP) as inv:
                    inv.observe(TextEvent(text="inner"))
                    inv.observe(MetricsEvent("inner", 30, 3, 3, 0.3))
                    inv.observe(ResultEvent(structured_output=None, continuation=None))
    snapshot = snapshots[-1]
    snapshot.validate(recorder.session_id)
    assert not empty.path.exists()
    root = json.loads(snapshot.documents[0].json_bytes)
    assert root["final_metrics"]["total_prompt_tokens"] == 190
    assert root["final_metrics"]["total_cost_usd"] == pytest.approx(0.50006)
    assert not any(item.get("descriptor") == "empty-child" for item in root["extra"]["subtrajectories"])

    _finalize_strict_archive(recorder, snapshot, RunConfig(run_eval=True, archive=True), workspace)
    archived = archive_dir / "runs" / recorder.session_id
    result = json.loads((archived / "evaluation.json").read_text())
    costs = result["cost"]
    by_agent = {agent["agent"]: agent for agent in costs["by_agent"]}
    assert {name: agent["cost_usd"] for name, agent in by_agent.items()} == {
        "main": pytest.approx(0.00006),
        "outer": pytest.approx(0.2),
        "inner": pytest.approx(0.3),
    }
    assert {name: agent["steps"] for name, agent in by_agent.items()} == {"main": 1, "outer": 1, "inner": 1}
    assert costs["total_cost_usd"] == pytest.approx(0.50006)
    assert costs["total_input_tokens"] == costs["total_prompt_tokens_raw"] == 190
    assert costs["total_completion_tokens"] == 19
    assert costs["total_cached_tokens"] == 19 and costs["cache_hit_rate"] == 0.1
    assert sum(agent["cost_usd"] for agent in by_agent.values()) == pytest.approx(costs["total_cost_usd"])
    expected_tools = {
        "Write": 1,
        "write": 1,
        "Edit": 1,
        "edit": 1,
        "MultiEdit": 1,
        "multiedit": 1,
        "NotebookEdit": 1,
        "notebookedit": 1,
        "patch": 1,
        "apply_patch": 1,
        "Read": 1,
        "read": 1,
        "shell": 1,
        "bash": 1,
        "custom": 1,
    }
    assert result["tools"] == {
        "total_calls": 15,
        "by_type": expected_tools,
        "by_agent": {"main": expected_tools, "outer": {}, "inner": {}},
        "write_ratio": 0.6667,
    }
    outer_signals = next(row for row in result["training_signals"]["trajectories"] if row["trajectory"] == "outer")
    assert outer_signals["noise_flags"] == ["incomplete_telemetry", "parser_coverage_gap"]
    manifest = json.loads((archived / "manifest.json").read_text())
    assert manifest["metrics"]["total_cost_usd"] == pytest.approx(0.50006)
    import sqlite3
    from contextlib import closing

    with closing(sqlite3.connect(archive_dir / "index.db")) as connection:
        assert connection.execute("SELECT total_cost_usd, total_prompt_tokens FROM runs").fetchone() == (
            pytest.approx(0.50006),
            190,
        )


def _read_traj(source_file: str, *read_paths: str, pi_style: bool = False) -> dict[str, Any]:
    """Build a fork with Read calls; pi_style uses read/arguments.path instead."""
    steps = []
    for i, path in enumerate(read_paths):
        if pi_style:
            tc = {"function_name": "read", "arguments": {"path": path}}
        else:
            tc = {"function_name": "Read", "arguments": {"file_path": path}}
        tc["tool_call_id"] = f"read-{i}"
        steps.append({"step_id": f"s{i}", "tool_calls": [tc],
            "observation": {"results": [{"source_call_id": f"read-{i}", "content": "source"}]},
        })
    return {"_source_file": source_file, "steps": steps}


def _artifact_provenance(
    *, public_source: Path, private_base: Path, workspace_key: str = "workspace-key", session_id: str = "session-id",
) -> Any:
    """Construct exact current-owner provenance without assuming a default root."""

    live = private_base / workspace_key / "runs" / session_id / "live"
    return ArtifactEvidenceProvenance(
        workspace_key=workspace_key, session_id=session_id, public_source=public_source, live_root=live,
    )


def _owned_source(tmp_path: Path, **kwargs: Any) -> tuple[Path, Path, Any]:
    """(public source, its ``.daydream`` dir, current-owner provenance) for one run."""
    public_source = tmp_path / "source"
    provenance = _artifact_provenance(public_source=public_source, private_base=tmp_path / "private", **kwargs)
    return public_source, public_source / ".daydream", provenance


def _deep_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Create and return ``(daydream_dir, deep_dir)`` under ``tmp_path``."""
    dd = tmp_path / ".daydream"
    deep = dd / "deep"
    deep.mkdir(parents=True)
    return dd, deep


def _seed_diff(daydream_dir: Path, *files: str) -> None:
    """Write a ``diff.patch`` naming exactly *files* as the reviewed diff."""
    daydream_dir.mkdir(parents=True, exist_ok=True)
    (daydream_dir / "diff.patch").write_text(
        "".join(f"diff --git a/{path} b/{path}\n" for path in files), encoding="utf-8"
    )


# Other workspaces and other sessions under the same key remain eligible repository reads.


def _write_records(deep: Path, **overrides: Any) -> None:
    """Write one per-stack python record, defaulted to a grounded src/api.py claim."""
    record: dict[str, Any] = {
        "id": "py-1", "file": "src/api.py", "line": 1, "confidence": "HIGH", "rationale": "src/api.py needs a guard",
    }
    record.update(overrides)
    (deep / "stack-python-records.json").write_text(json.dumps([record]), encoding="utf-8")


def _root_trajectory(session_id: str) -> dict[str, Any]:
    """A minimal ATIF root trajectory ``analyze_session`` accepts as frozen input."""
    return {"_source_file": "trajectory.json", "session_id": session_id, "trajectory_id": session_id,
        "agent": {"name": "daydream", "model_name": "test"}, "steps": [], "final_metrics": {},
    }


_READ_TOOL_SHAPES = {"claude": ("Read", "file_path"), "pi": ("read", "path"), "osprey": ("read", "path")}


def test_analyze_session_preserves_source_quality_without_read_metrics(tmp_path: Path,) -> None:
    public_source, _, provenance = _owned_source(tmp_path, session_id="session")
    (public_source / "src").mkdir(parents=True)
    (public_source / "src/api.py").write_text("def api(value):\n    return value\n", encoding="utf-8")
    # Frozen evaluation input lives apart from the post-fix source workspace.
    daydream_dir = tmp_path / "frozen" / ".daydream"
    deep = daydream_dir / "deep"
    deep.mkdir(parents=True)
    _seed_diff(daydream_dir, "src/api.py")
    private_live = provenance.live_root
    artifact_ref = str(private_live / ".daydream/deep/stack-python-review.md")
    exploration_ref = str(private_live / ".daydream/exploration/summary.md")
    _write_records(deep, rationale=f"Evidence came from {artifact_ref}")
    trajectories: dict[str, Any] = {"main": _root_trajectory("session"),
        "forked": [{**_read_traj("deep-python.json", str(public_source / "src/api.py"), artifact_ref, exploration_ref,),
            "session_id": "session", "trajectory_id": "deep-python"}],
    }

    result = analyze_session(
        daydream_dir, write_snapshot=_snapshot(trajectories["main"], *trajectories["forked"]),
        artifact_provenance=provenance,
        code_workspace=public_source,
    )

    assert not {"coverage", "grounding", "exploration_utilization"} & result.keys()
    assert result["tools"]["total_calls"] == 3


def _quality_workspace(tmp_path: Path, files: dict[str, str], name: str = "workspace") -> Path:
    """Create a temp workspace with the given ``{relative_path: content}`` files."""
    ws = tmp_path / name
    ws.mkdir(parents=True)
    for rel, content in files.items():
        target = ws / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    return ws


def _quality(tmp_path: Path, files: dict[str, str], *, name: str = "workspace", **kwargs: Any,) -> dict[str, Any]:
    """Build a workspace with ``files`` and analyze its ``.daydream`` quality."""
    return analyze_quality(_quality_workspace(tmp_path, files, name) / ".daydream", **kwargs)


def _big_function(max_x: int) -> str:
    """Build an if/elif chain with max_x+1 complexity and 2*max_x+2 SLOC."""
    lines = ["def big(x):", "    if x == 1:", "        return 1"]
    for i in range(2, max_x + 1):
        lines.append(f"    elif x == {i}:")
        lines.append(f"        return {i}")
    lines.append("    return 0")
    return "\n".join(lines) + "\n"


def test_quality_returns_none_when_no_python_files(tmp_path: Path) -> None:
    result = _quality(tmp_path, {"README.md": "# nothing here\n"})

    assert result["erosion"] is None
    assert result["verbosity"] is None
    assert result["scoped_files"] == 0
    assert result["per_file"] == {}
    assert result["calibration"] == {"human_verbosity": 0.19, "human_erosion": 0.34, "paper": "arXiv:2603.24755"}


def test_quality_excludes_vendored_and_internal_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Standalone scope prunes excluded trees and preserves leaf-file symlink compatibility."""
    ws = _quality_workspace(tmp_path,
        {
            "app.py": "def f(x):\n    return x\n",
            ".daydream/deep/fixture.py": "def g(x):\n    return x\n",
            "node_modules/pkg/index.py": "def h(x):\n    return x\n",
            "sub/venv/lib/x.py": "def i(x):\n    return x\n",
            "daydream/atif/models.py": "def j(x):\n    return x\n",
            "sub/vendor/lib/x.py": "def k(x):\n    return x\n",
            "vendorish/mod.py": "VALUE = 1\n",
            "eligible.py/child.py": "CHILD = 2\n",
            "api_generated.py": "GENERATED = 3\n", "svc.pb.py": "PROTOBUF = 4\n",
            "third_party/lib/t.py": "THIRD_PARTY = 5\n", "migrations/0001_init.py": "MIGRATION = 6\n",
            "gen_tool.py": "# Code generated by protoc. DO NOT EDIT.\nGENERATED = 7\n",
        },
    )
    outside = tmp_path / "external"
    outside.mkdir()
    (outside / "external.py").write_text("EXTERNAL = 3\n")
    (ws / "linked.py").symlink_to(outside, target_is_directory=True)
    (ws / "linked_file.py").symlink_to(outside / "external.py")
    excluded = {ws / ".daydream", ws / "node_modules", ws / "sub/venv", ws / "daydream/atif", ws / "sub/vendor"}
    excluded.add(ws / "linked.py")
    scandir = os.scandir

    def scan(path: Any) -> Any:
        if isinstance(path, (str, bytes, os.PathLike)) and Path(os.fsdecode(path)) in excluded:
            raise AssertionError(f"excluded subtree must be pruned before traversal: {path!r}")
        return scandir(path)

    monkeypatch.setattr(os, "scandir", scan)
    result = analyze_quality(ws / ".daydream")

    assert result["scoped_files"] == 4
    assert list(result["per_file"]) == ["app.py", "eligible.py/child.py", "linked_file.py", "vendorish/mod.py"]
    assert result["per_file"]["linked_file.py"]["sloc"] == 1


@pytest.mark.parametrize("layout", ["root", "nested", "linked"])
@pytest.mark.parametrize("candidate_paths", [None, {"app.py", "solo.py"}], ids=["workspace", "candidate"])
def test_quality_git_scope_confines_targets_and_clone_peers(
    git_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    layout: str, candidate_paths: set[str] | None,
) -> None:
    """Git membership owns reads and clones; standalone peer tests cannot detect ignored pollution."""
    workspace = git_repo
    if layout == "nested":
        workspace = git_repo / "service"
        workspace.mkdir()
        (git_repo / "outside.py").write_text("OUTSIDE = 1\n")
    elif layout == "linked":
        workspace = tmp_path / "linked"
        git(git_repo, "worktree", "add", "--detach", str(workspace), "HEAD")
    block = "    if x > 1:\n        return 1\n    return 0\n"
    solo_block = "    if y > 7:\n        return 8\n    return 9\n"
    contents = {
        "app.py": "def app(x):\n" + block,
        "solo.py": "def solo(y):\n" + solo_block,
        "peer.py": "def peer(x):\n" + block,
        "deps/tracked.py": "def tracked(x):\n" + block,
        ".rules/hidden.py": "HIDDEN = 2\n",
        "ignored/artifact.py": "def artifact(y):\n" + solo_block,
        "cache/dependency.py": "def dependency(y):\n" + solo_block,
        "vendor/excluded.py": "VENDOR = 3\n",
        "generated.py": "# Code generated by protoc. DO NOT EDIT.\nGENERATED = 4\n",
        "replaced/secret.py": "SECRET = 5\n",
    }
    (workspace / ".gitignore").write_text("deps/\nignored/\ncache/\n")
    for relative, content in contents.items():
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    git(workspace, "add", "app.py", "solo.py", ".rules", "vendor", "generated.py", "replaced")
    git(workspace, "add", "-f", "deps/tracked.py")
    outside = tmp_path / "external-source"
    outside.mkdir()
    (outside / "secret.py").write_text("SECRET = 99\n")
    (workspace / "linked.py").symlink_to(outside / "secret.py")
    git(workspace, "add", "linked.py")
    (workspace / "replaced/secret.py").unlink()
    (workspace / "replaced").rmdir()
    (workspace / "replaced").symlink_to(outside, target_is_directory=True)
    os.mkfifo(workspace / "pipe.py")

    allowed_reads = {"app.py", "solo.py", "peer.py", "deps/tracked.py", ".rules/hidden.py", "generated.py"}
    reads: list[str] = []
    open_path, scandir = Path.open, os.scandir

    def open_source(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path.is_relative_to(workspace) and path.suffix == ".py":
            relative = str(path.relative_to(workspace))
            assert relative in allowed_reads, f"out-of-scope Python read: {relative}"
            reads.append(relative)
        return open_path(path, *args, **kwargs)

    def scan(path: Any) -> Any:
        if isinstance(path, (str, bytes, os.PathLike)):
            assert not Path(os.fsdecode(path)).is_relative_to(workspace), "Git quality must not recursively traverse"
        return scandir(path)

    monkeypatch.setattr(Path, "open", open_source)
    monkeypatch.setattr(os, "scandir", scan)
    result = analyze_quality(workspace / ".daydream", candidate_paths=candidate_paths)
    expected = [".rules/hidden.py", "app.py", "deps/tracked.py", "peer.py", "solo.py"]
    if candidate_paths is not None:
        expected = ["app.py", "solo.py"]
    assert result["scoped_files"] == len(expected)
    assert list(result["per_file"]) == expected
    assert sorted(reads) == sorted(allowed_reads)
    assert result["per_file"]["app.py"]["verbosity"] == 0.75
    assert result["per_file"]["solo.py"]["verbosity"] == 0.0

    (workspace / "peer.py").write_text("PEER = 6\n")
    assert analyze_quality(workspace / ".daydream", candidate_paths={"app.py"})["verbosity"] == 0.75
    (workspace / "deps/tracked.py").write_text("TRACKED = 7\n")
    assert analyze_quality(workspace / ".daydream", candidate_paths={"app.py"})["verbosity"] == 0.0


def test_quality_git_inventory_failure_never_recurses(
    git_repo: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A corrupt real index must surface, rather than admitting ignored experiment files."""
    (git_repo / "app.py").write_text("VALUE = 1\n")
    (git_repo / ".git/index").write_bytes(b"corrupt-index")
    scandir = os.scandir

    def scan(path: Any) -> Any:
        if isinstance(path, (str, bytes, os.PathLike)):
            assert not Path(os.fsdecode(path)).is_relative_to(git_repo), "failed inventory must not widen scope"
        return scandir(path)

    monkeypatch.setattr(os, "scandir", scan)
    with pytest.raises(GitError):
        analyze_quality(git_repo / ".daydream")


def test_analyze_session_reads_quality_from_explicit_code_workspace(tmp_path: Path,) -> None:
    """Analyze one package before/after source edits, independently of its frozen review root.

    Literal per-file metrics expose rule regressions within a mixed source tree. The later edits
    prove post-fix measurement rather than cached pre-fix results or reconstructed public artifacts.
    """
    from textwrap import dedent

    files = {
        "pkg/views.py": dedent("""\
            def copied(items):
                return [item for item in items]
            def streamed(items):
                return (item for item in items)
            def filtered(items):
                return [item for item in items if item > 0]
            def paired(left, right):
                return [item for item in left for other in right]
            def selected_list(values):
                return [x for x in values FILTERS]
            def selected_set(values):
                return {x for x in values FILTERS}
            def selected_dict(values):
                return {x: x for x in values FILTERS}
            def selected_stream(values):
                return (x for x in values FILTERS)
            """).replace("FILTERS", _TEN_COMPREHENSION_FILTERS),
        "pkg/forward.py": dedent("""\
            def add(left, right):
                return left + right
            def documented(left, right):
                "Pass through."
                return add(left, right)
            def plain(left, right):


                return add(left, right)
            def annotated(left: int, right: int):
                return add(left, right)
            def with_literal(left):
                return add(left, 42)
            def with_keywords(left, right):
                return add(left=left, right=right)
            def with_splat(*values):
                return add(*values)
            def with_default(left=1, right=2):
                return add(left, right)
            """),
        "pkg/queue.py": dedent("""\
            def process(items):
                for item in items:
                    if len(items) == 0:
                        return None
                    consume(item)
            def nonempty(items):
                while items:
                    if not items:
                        break
            def counted(items):
                while len(items) > 0:
                    if not items:
                        break
            def conditional(items):
                while should_continue(items):
                    if not items:
                        break
            def draining(items):
                while items:
                    item = items.pop()
                    if not items:
                        break
            def clearing(items):
                while items:
                    items.clear()
                    if not items:
                        break
            def watched(items):
                while items:
                    observed = inspect()
                    if not items:
                        break
            def ladder(a, b, c):
                if a:
                    if b:
                        if c:
                            return 1
                return 0
            def repeated_first(value):
                if value > 1:
                    return 1
                return 0
            def repeated_second(value):
                if value > 1:
                    return 1
                return 0
            """),
        "pkg/calculate.py": dedent("""\
            def simple(x):
                pending = x + 1
                return pending * 2
            def same_row(x):
                local = x + 1; return local * 2
            def same_and_later(x):
                candidate = x + 1; print(candidate)
                return candidate * 2
            def reassigned(x):
                updated = x + 1
                updated = x + 2
                return updated * 2
            def with_nested(x):
                outer = x + 1
                def nested(default=outer):
                    return outer
                return outer * 2
            def with_shadow(x):
                staged = x + 1
                def nested(y):
                    staged = y + 2
                    return staged * 2
                return staged * 3
            def with_decorator(x):
                context = x + 1
                @decorate(context)
                def nested():
                    return context
                return context * 2
            def multiline(x):
                intermediate = (
                    x + intermediate
                )
                return x * 2
            """),
        "pkg/dispatch.py": "def fallback(value):\n    match value:\n        case _:\n            return 0\n\n"
        + "def dispatch(value):\n    match value:\n"
        + "".join(f"        case {index}:\n            return {index}\n" for index in range(1, 12)),
    }
    workspace = _quality_workspace(tmp_path, files, name="operational")
    public_source = tmp_path / "public-source"
    frozen = tmp_path / "frozen" / ".daydream"
    snapshot = run_snapshot("quality-split", schema_version="ATIF-v1.6", model_name="test",)
    provenance = _artifact_provenance(
        public_source=public_source, private_base=tmp_path / "private", workspace_key="workspace",
        session_id="quality-split",
    )
    result = analyze_session(frozen, write_snapshot=snapshot, artifact_provenance=provenance, code_workspace=workspace)
    quality = result["quality"]
    assert quality["scoped_files"] == 5
    assert quality["per_file"] == {
        "pkg/views.py": {"erosion": 0.8276, "verbosity": 0.125, "sloc": 16, "functions": 8, "high_cc_functions": 4},
        "pkg/forward.py": {"erosion": 0.0, "verbosity": 0.4118, "sloc": 17, "functions": 8, "high_cc_functions": 0},
        "pkg/queue.py": {"erosion": 0.0, "verbosity": 0.3478, "sloc": 46, "functions": 10, "high_cc_functions": 0},
        "pkg/calculate.py": {"erosion": 0.0, "verbosity": 0.2059, "sloc": 34, "functions": 11, "high_cc_functions": 0},
        "pkg/dispatch.py": {"erosion": 0.9671, "verbosity": 0.0, "sloc": 28, "functions": 2, "high_cc_functions": 1},
    }
    # Independent function-span/decision arithmetic, including nested functions and blank rows.
    high_mass = 48 * math.sqrt(2) + 12 * math.sqrt(24)
    mass = (
        58 * math.sqrt(2)
        + 6 * math.sqrt(2)
        + math.sqrt(3)
        + 2
        + 12 * math.sqrt(5)
        + 26
        + 4 * math.sqrt(6)
        + 3 * math.sqrt(3)
        + 3 * math.sqrt(2)
        + 2 * math.sqrt(5)
        + 2 * math.sqrt(6)
        + 2
        + 2
        + 12 * math.sqrt(24)
    )
    assert quality["erosion"] == round(high_mass / mass, 4)
    assert quality["verbosity"] == round(32 / 141, 4)
    assert quality["calibration"] == {"human_verbosity": 0.19, "human_erosion": 0.34, "paper": "arXiv:2603.24755"}
    assert result["daydream_dir"] == str(public_source / ".daydream")

    (workspace / "pkg/dispatch.py").write_text(_big_function(13))
    eroded = analyze_session(frozen, write_snapshot=snapshot, code_workspace=workspace)["quality"]
    assert eroded["erosion"] > quality["erosion"]
    for path in workspace.rglob("*.py"):
        path.write_text("def value():\n    return 1\n")
    cleaned = analyze_session(frozen, write_snapshot=snapshot, code_workspace=workspace)["quality"]
    assert cleaned["erosion"] == cleaned["verbosity"] == 0.0
    assert cleaned["scoped_files"] == 5


_TEN_COMPREHENSION_FILTERS = " ".join(f"if x != {i}" for i in range(10))


@pytest.mark.parametrize("candidate_paths", [None, {"app.py"}], ids=["workspace", "candidate"])
def test_quality_reuses_eligibility_bytes_for_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, candidate_paths: set[str] | None,
) -> None:
    """Eligibility and parsing share one read, including peers and failed inputs."""
    block = "    if x > 1:\n        return 1\n    return 0\n"
    ws = _quality_workspace(tmp_path, {
        "app.py": "def a():\n" + block,
        "peer.py": "def b():\n" + block,
        "broken.py": "def broken(:\n" + block,
        "gen_tool.py": "# Code generated by protoc. DO NOT EDIT.\ndef c():\n    return 3\n",
        "denied.py": "def denied():\n    return 4\n",
    })
    (ws / "undecodable.py").write_bytes(b'def byte_string():\n    return "\xff"\n')
    opens: dict[str, int] = {}
    open_path = Path.open

    def open_source(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path.parent == ws and path.suffix == ".py":
            opens[path.name] = opens.get(path.name, 0) + 1
            if path.name == "denied.py":
                raise PermissionError("source is unreadable")
        return open_path(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_source)
    result = analyze_quality(ws / ".daydream", candidate_paths=candidate_paths)

    assert opens == {name: 1 for name in (
        "app.py", "peer.py", "broken.py", "gen_tool.py", "denied.py", "undecodable.py",
    )}
    assert result["scoped_files"] == (4 if candidate_paths is None else 1)
    expected_paths = ["app.py", "peer.py", "undecodable.py"] if candidate_paths is None else ["app.py"]
    assert list(result["per_file"]) == expected_paths
    assert result["per_file"]["app.py"] == {
        "erosion": 0.0, "verbosity": 0.75, "sloc": 4, "functions": 1, "high_cc_functions": 0,
    }
    assert result["erosion"] == 0.0
    assert result["verbosity"] == (0.6 if candidate_paths is None else 0.75)
    if candidate_paths is None:
        assert result["per_file"]["undecodable.py"] == {
            "erosion": 0.0, "verbosity": 0.0, "sloc": 2, "functions": 1, "high_cc_functions": 0,
        }
    # Remove the valid clone peer while leaving the broken file's identical body in place.
    # A malformed tree must not create a false clone in either workspace or candidate scope.
    (ws / "peer.py").write_text("PEER = 7\n")
    opens.clear()
    changed = analyze_quality(ws / ".daydream", candidate_paths=candidate_paths)
    assert changed["per_file"]["app.py"]["verbosity"] == 0.0
    assert "broken.py" not in changed["per_file"]
    assert opens == {
        name: 1 for name in ("app.py", "peer.py", "broken.py", "gen_tool.py", "denied.py", "undecodable.py")
    }


def test_quality_candidate_empty_set_returns_empty_without_enumeration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(_workspace: Path) -> None:
        raise AssertionError("workspace must not be enumerated for an empty candidate set")

    monkeypatch.setattr(quality_mod, "_scoped_python_files", _boom)
    result = _quality(tmp_path, {"app.py": "def a():\n    return 1\n"}, candidate_paths=set())

    assert result["scoped_files"] == 0
    assert result["per_file"] == {}
    assert result["erosion"] is None
    assert result["verbosity"] is None


def test_quality_candidate_ineligible_path_not_reported(tmp_path: Path) -> None:
    result = _quality(tmp_path,
        {
            "app.py": "def a():\n    return 1\n",
            "schema_generated.py": "def x():\n    return 1\n",  # *_generated.py glob excludes it
        }, candidate_paths={"schema_generated.py", "app.py"},
    )
    assert result["scoped_files"] == 1
    assert set(result["per_file"]) == {"app.py"}


@pytest.mark.parametrize("failure", ["unavailable", "empty-tree", "exception"])
def test_quality_parser_failure_keeps_scope_without_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    """Binding failure excludes metrics without pretending the eligible source was absent."""
    class FailedParser:
        def parse(self, _source: bytes) -> None:
            if failure == "exception":
                raise ValueError("native parser failed")
            return None

    parser = None if failure == "unavailable" else FailedParser()
    monkeypatch.setattr(quality_mod, "_quality_python_parser", lambda: parser)
    result = _quality(tmp_path, {"app.py": "def f(x):\n    return x\n"})

    assert result == {
        "erosion": None,
        "verbosity": None,
        "per_file": {},
        "calibration": {"human_verbosity": 0.19, "human_erosion": 0.34, "paper": "arXiv:2603.24755"},
        "scoped_files": 1,
    }


def test_quality_per_file_erosion_none_without_functions(tmp_path: Path) -> None:
    """No function mass means undefined erosion, distinct from zero high-complexity mass."""
    result = _quality(tmp_path, {"m.py": "import os\nX = 1\n", "app.py": "def f(x):\n    return x\n"},)

    assert result["per_file"]["m.py"]["erosion"] is None
    assert result["per_file"]["m.py"]["functions"] == 0
    assert result["per_file"]["app.py"]["erosion"] == 0.0
    assert result["erosion"] == 0.0


def test_quality_per_file_verbosity_none_on_blank_only_file(tmp_path: Path) -> None:
    """A zero nonblank denominator remains undefined in file and workspace verbosity."""
    result = _quality(tmp_path, {"blank.py": "\n\n\n"})

    assert result["per_file"]["blank.py"]["verbosity"] is None
    assert result["per_file"]["blank.py"]["sloc"] == 0
    assert result["verbosity"] is None


# Count authoritative merged items, falling back to review text, then pre-merge stack totals.


def seed_shipped_items(deep: Path, *, high: int, med: int) -> None:
    """Write deep/\"merged-items.json\" = {\"items\": [high+med schema-valid items]}."""
    items = [_item(i, file="a.py", line=i + 1, description=f"high-{i}", confidence="HIGH", severity="high")
        for i in range(high)
    ]
    items += [_item(high + i, file="b.py", line=i + 1, description=f"med-{i}") for i in range(med)]
    seed_merged_items(deep, items)


def seed_stack_records(deep: Path, stack_name: str, *, n: int) -> None:
    """Seed HIGH records with host UIDs using 1-based birth ordinals.

    Reviewer ids deliberately remain 0-based: display ids are not record identity."""
    records: list[dict[str, Any]] = [{"id": i, "confidence": "HIGH", "uid": mint_record_uid(stack_name, i + 1)}
        for i in range(n)
    ]
    (deep / f"stack-{stack_name}-records.json").write_text(json.dumps({"issues": records}))


def seed_review_output(deep: Path, *, count: int) -> None:
    """Write deep/\"review-output.md\" with `count` lines matching ^\\d+\\.\\s+\\[."""
    lines = [f"{i}. [HIGH] finding {i}" for i in range(1, count + 1)]
    (deep / "review-output.md").write_text("\n".join(lines) + "\n")


def test_per_lens_malformed_alternatives_does_not_crash(tmp_path: Path) -> None:
    dd, deep = _deep_dirs(tmp_path)
    (deep / "alternatives.json").write_text("{not json")
    out = analyze_findings(dd)
    assert out["per_lens"]["wonder"] == 0


def run_snapshot(
    session_id: str, *, total_cost_usd: float | None = None, schema_version: str = "ATIF-v1.7",
    model_name: str | None = None, steps: list[dict[str, Any]] | None = None,
) -> RunWriteSnapshot:
    """Prepare immutable run bytes with the requested fields."""
    agent: dict[str, Any] = {"name": "test"}
    if model_name is not None:
        agent["model_name"] = model_name
    document: dict[str, Any] = {"schema_version": schema_version, "session_id": session_id, "trajectory_id": session_id,
        "_source_file": "trajectory.json", "agent": agent,
        "steps": [] if steps is None else steps, "extra": {},
    }
    if total_cost_usd is not None:
        document["final_metrics"] = {"total_cost_usd": total_cost_usd}
    return _snapshot(document)


def test_analyze_session_shipped_metrics_match_a80b9373(tmp_path: Path) -> None:
    """One persisted finding population owns canonical counts and each missing-artifact fallback."""
    sid = "a80b9373-56d6-4062-9ab5-4c75e475ab67"
    dd, deep = _deep_dirs(tmp_path)
    snapshot = run_snapshot(sid, total_cost_usd=18.2056)
    seed_shipped_items(deep, high=4, med=4)
    shipped = json.loads((deep / "merged-items.json").read_text())["items"]
    shipped[4]["lens"] = shipped[5]["lens"] = "wonder"
    seed_merged_items(deep, shipped)
    seed_stack_records(deep, "python", n=3)
    seed_stack_records(deep, "structure", n=2)
    seed_stack_records(deep, "uncovered", n=1)
    (deep / "alternatives.json").write_text(json.dumps([{"id": i} for i in range(6)]))
    res = analyze_session(dd, write_snapshot=snapshot)
    assert res["findings"]["total"] == 8
    assert res["findings"]["by_confidence"] == {"HIGH": 4, "MEDIUM": 4}
    assert res["findings"]["per_lens"] == {"wonder": 6, "per-stack": 3, "structure": 2}
    assert res["findings"]["stacks"] == [
        {"name": "python", "finding_count": 3}, {"name": "structure", "finding_count": 2}
    ]
    assert res["derived"]["cost_per_finding_usd"] == pytest.approx(18.2056 / 8, rel=1e-4)
    # Shipped rendering wins over raw reviewer populations when canonical JSON is unavailable.
    (deep / "merged-items.json").unlink()
    seed_review_output(deep, count=8)
    rendered = analyze_session(dd, write_snapshot=snapshot)
    assert rendered["findings"]["total"] == 8 and rendered["findings"]["by_confidence"] == {}
    assert rendered["derived"]["cost_per_finding_usd"] == pytest.approx(18.2056 / 8, rel=1e-4)
    (deep / "review-output.md").unlink()
    premerge = analyze_session(dd, write_snapshot=snapshot)
    assert premerge["findings"]["total"] == 5 and premerge["findings"]["by_confidence"] == {"HIGH": 5}
    assert premerge["derived"]["cost_per_finding_usd"] == pytest.approx(18.2056 / 5, rel=1e-4)
    (deep / "stack-python-records.json").unlink()
    (deep / "stack-structure-records.json").unlink()
    wonder_only = analyze_session(dd, write_snapshot=snapshot)
    assert wonder_only["findings"]["per_lens"] == {"wonder": 6, "per-stack": 0, "structure": 0}
    assert wonder_only["findings"]["total"] == 0


def test_analyze_quality_refuses_known_bad_tree_sitter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """#1087: on a known-bad tree-sitter install the analyzer refuses native
    analysis by raising the typed guard error — the orchestrator's fail-open
    wrapper converts that into (None, reason); it must not silently skip.
    """

    monkeypatch.setattr(safety, "installed_tree_sitter_version", lambda: "0.26.0")
    # Clear the parser cache so the bad-version guard sees the patched installation.
    _quality_python_parser.cache_clear()
    with pytest.raises(safety.TreeSitterBadVersionError):
        _quality(tmp_path, {"mod.py": "def f():\n    return 1\n"})


def test_analyze_quality_unchanged_on_good_install(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """#1087 (M5): the guard is a no-op on valid installs — behavior identical
    to pre-regression, including the parser cache being consulted.
    """

    monkeypatch.setattr(safety, "installed_tree_sitter_version", lambda: "0.25.2")
    result = _quality(tmp_path, {"mod.py": "def f():\n    return 1\n"})
    assert result["scoped_files"] == 1
    entry = result["per_file"]["mod.py"]
    assert entry["functions"] == 1
    assert entry["sloc"] > 0


def test_analyze_session_degrades_quality_on_known_bad_tree_sitter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """#1087: a known-bad install degrades only the quality section of
    analyze_session -- the rest of the evaluation (and evaluation.json)
    survives instead of the typed escape dropping the whole run.
    """

    monkeypatch.setattr(safety, "installed_tree_sitter_version", lambda: "0.26.0")
    ws = _quality_workspace(tmp_path, {"app.py": "def small(x):\n    return x * 2\n\n" + _big_function(11)},)
    daydream_dir = ws / ".daydream"
    snapshot = run_snapshot("quality-bad", schema_version="ATIF-v1.6", model_name="claude-sonnet-4-5",)

    result = analyze_session(daydream_dir, write_snapshot=snapshot)

    quality = result["quality"]
    assert quality["unavailable"] is True
    assert quality["error"]
    assert quality["per_file"] == {}
    assert quality["scoped_files"] == 0
    assert result["session_id"] == "quality-bad"
    assert result["trajectory_count"] == 1


# Seed the exact production artifact shapes for location and duplication analysis.

WORKED_A = "New helper duplicates the existing loader"
WORKED_B = "Config reading is implemented twice in this module"


def seed_diff_patch(dd: Path, file: str = "svc/loader.py", *, start: int = 85, count: int = 8) -> None:
    """Write one hunk with inclusive new-side range [start, start+count-1]."""
    dd.mkdir(parents=True, exist_ok=True)
    (dd / "diff.patch").write_text(
        f"diff --git a/{file} b/{file}\n"
        f"--- a/{file}\n"
        f"+++ b/{file}\n"
        f"@@ -{start},{count} +{start},{count} @@ def load():\n"
        "-old\n"
        "+new\n"
    )


def seed_hunk_index(dd: Path, ranges: dict[str, list[tuple[int, int]]]) -> None:
    """Write ``.daydream/hunk-index.json`` in the persisted production shape."""
    dd.mkdir(parents=True, exist_ok=True)
    (dd / "hunk-index.json").write_text(json.dumps({path: {"hunks": [{
                            "old_start": start, "old_end": end, "new_start": start, "new_end": end, "added": 1,
                            "removed": 1,
                        }
                        for start, end in file_ranges
                    ], "added_total": len(file_ranges), "removed_total": len(file_ranges),
                }
                for path, file_ranges in ranges.items()
            }
        )
    )


def seed_merged_items(deep: Path, items: list[dict[str, Any]]) -> None:
    """Write ``deep/merged-items.json`` from an explicit item list."""
    deep.mkdir(parents=True, exist_ok=True)
    (deep / "merged-items.json").write_text(json.dumps({"items": items}))


def seed_dedup_candidates(deep: Path, *, record_alt_pairs: list[dict[str, Any]] | None = None,
    record_duplicate_pairs: list[dict[str, Any]] | None = None,
) -> None:
    """Write production-shaped pre-merge candidate pairs, which are inputs rather than escapes."""
    deep.mkdir(parents=True, exist_ok=True)
    (deep / "dedup-candidates.json").write_text(json.dumps(
            {"record_alt_pairs": record_alt_pairs or [], "record_duplicate_pairs": record_duplicate_pairs or []}
        )
    )


def _provenance(*uids: str) -> dict[str, Any]:
    """Provide a typed source_uids keyword payload for unpacking into _item.

    No arguments means explicit attribution refusal, distinct from an absent key."""
    return {RECORD_SOURCE_UIDS_KEY: list(uids)}


def _item(
    item_id: int, *, file: str = "svc/loader.py", line: Any = 88, description: str = WORKED_A, lens: str = "per-stack",
    **extra: Any,
) -> dict[str, Any]:
    """A shipped ``merged-items.json`` item in the canonical shape."""
    item: dict[str, Any] = {
        "id": item_id, "file": file, "line": line, "lens": lens, "severity": "medium", "confidence": "MEDIUM",
        "description": description, "rationale": f"see {file}", "evidence": f"{file}:{line}",
    }
    item.update(extra)
    return item


def _worked_example_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """``(.daydream, .daydream/deep)`` with the issue's diff already seeded."""
    dd, deep = _deep_dirs(tmp_path)
    seed_diff_patch(dd)  # svc/loader.py, single hunk (85, 92)
    return dd, deep


def test_issue_1106_worked_example_is_distinguishable_from_the_clean_run(tmp_path: Path,) -> None:
    """Distinguish a correct line-88 finding from its line-4 structural restatement.

    The sole hunk is lines 85–92; location must expose the 81-line miss, and duplication
    must expose the pair even though similarity 0.1538 is below the 0.5 threshold."""
    dd, deep = _worked_example_dirs(tmp_path)
    seed_merged_items(
        deep,
        [
            _item(1, line=88, description=WORKED_A),
            _item(
                2,
                line=4,
                description=WORKED_B,
                lens="structural",
                location_distrust=True,
                location_cited_line=4,
            ),
        ],
    )

    location = analyze_location(dd)
    assert location["hunk_source"] == "diff.patch"
    assert location["shipped_items"] == 2
    assert location["scored_items"] == 2
    assert location["tiers"] == {"in_hunk": 1, "within_tolerance": 0,
        "beyond_tolerance": 1,   # the mis-anchored twin
        "file_absent": 0,
    }
    assert location["in_hunk_rate"] == 0.5  # NOT 1.0 -- the run is no longer perfect
    assert location["distrusted_items"] == 1
    beyond = [row for row in location["items"] if row["tier"] == "beyond_tolerance"]
    assert [row["id"] for row in beyond] == [2]
    assert beyond[0]["distance"] == 81  # exactly the issue's arithmetic

    duplication = analyze_shipped_duplication(dd)
    assert duplication["shipped_items"] == 2
    assert duplication["comparable_pairs"] == 1
    assert duplication["same_file_pairs"] == 1  # the escape IS visible
    assert duplication["near_duplicate_pairs"] == 0  # ...but under the 0.5 bar
    assert duplication["max_similarity"] == 0.1538  # the issue's own number
    pair = duplication["pairs"][0]
    assert (pair["a_id"], pair["b_id"]) == ("1", "2")
    assert (pair["a_lens"], pair["b_lens"]) == ("per-stack", "structural")
    assert pair["same_file"] is True
    # Unattributed merge-authored items keep empty provenance; the duplicate pair still counts.
    assert (pair["a_source_uids"], pair["b_source_uids"]) == ([], [])

    seed_merged_items(
        deep,
        [
            _item(1, line=88, description=WORKED_A),
            _item(2, line=4, description=WORKED_B, lens="structural"),
            _item(3, file="other/untouched.py", line=1, description="A wholly unrelated concern"),
        ],
    )
    expanded = analyze_session(
        dd, write_snapshot=run_snapshot("expanded", schema_version="ATIF-v1.7", model_name="test")
    )
    distribution = expanded["findings"]["shipped_duplication"]
    assert distribution["comparable_pairs"] == 3 and distribution["near_duplicate_pairs"] == 0
    assert distribution["same_file_pairs"] == 1 and distribution["max_similarity"] == 0.1538
    similarities = [row["similarity"] for row in distribution["pairs"]]
    assert similarities == sorted(similarities, reverse=True)
    assert similarities[0] == distribution["max_similarity"]

    clean_dd, clean_deep = _worked_example_dirs(tmp_path / "clean")
    seed_merged_items(clean_deep, [_item(1, line=88, description=WORKED_A)])
    clean_location = analyze_location(clean_dd)
    assert clean_location["in_hunk_rate"] == 1.0
    assert clean_location["tiers"]["beyond_tolerance"] == 0
    clean_duplication = analyze_shipped_duplication(clean_dd)
    assert clean_duplication["comparable_pairs"] == 0
    assert clean_duplication["same_file_pairs"] == 0
    assert clean_duplication["max_similarity"] is None
    assert clean_location["in_hunk_rate"] > location["in_hunk_rate"]


def test_location_hunk_source_falls_back_to_diff_patch(tmp_path: Path) -> None:
    """Archived runs carry only ``diff.patch``; the axis still scores them."""
    dd, deep = _worked_example_dirs(tmp_path)
    assert not (dd / "hunk-index.json").exists()
    seed_merged_items(deep, [_item(1, line=88)])

    location = analyze_location(dd)

    assert location["hunk_source"] == "diff.patch"
    assert location["in_hunk_rate"] == 1.0


def test_location_hunk_source_none_reports_not_measured(tmp_path: Path) -> None:
    """With neither artifact, "not measured" must be distinguishable from "clean"."""
    dd, deep = _deep_dirs(tmp_path)
    seed_merged_items(deep, [_item(1, line=4)])   # would be beyond_tolerance

    location = analyze_location(dd)

    assert location["hunk_source"] == "none"
    assert location["shipped_items"] == 1
    assert location["scored_items"] == 0
    assert location["tiers"] == {"in_hunk": 0, "within_tolerance": 0, "beyond_tolerance": 0, "file_absent": 0}
    assert location["tier_rates"] == {}
    assert location["in_hunk_rate"] is None       # undefined, not perfect
    assert location["items"] == []


def test_location_and_duplication_are_zeroed_when_merged_items_absent(tmp_path: Path,) -> None:
    """An existing deep directory does not prove merge produced a shipped set.

    Absent merged-items keeps duplicate counts undefined; zero means merge completed
    with a genuinely empty set."""
    dd, _deep = _worked_example_dirs(tmp_path)

    location = analyze_location(dd)
    duplication = analyze_shipped_duplication(dd)

    assert location["shipped_items"] == 0
    assert location["scored_items"] == 0
    assert location["in_hunk_rate"] is None
    assert location["hunk_source"] == "diff.patch"   # honest: hunks WERE readable
    assert duplication["shipped_items"] == 0
    assert duplication["comparable_pairs"] == 0
    assert duplication["near_duplicate_pairs"] is None
    assert duplication["max_similarity"] is None
    assert duplication["mean_similarity"] is None
    assert duplication["pairs"] == []


def test_duplication_near_duplicate_pairs_is_none_when_deep_never_ran(tmp_path: Path,) -> None:
    dd = tmp_path / ".daydream"
    dd.mkdir(parents=True)

    duplication = analyze_shipped_duplication(dd)

    assert duplication["shipped_items"] == 0
    assert duplication["comparable_pairs"] == 0
    assert duplication["near_duplicate_pairs"] is None
    assert duplication["max_similarity"] is None
    assert duplication["mean_similarity"] is None
    assert duplication["pairs"] == []


def test_shipped_duplication_input_is_capped_to_bound_the_on2_scan(tmp_path: Path,) -> None:
    """Place the only duplicate pair beyond the 200-item quadratic-comparison cap.

    It must remain uncounted while shipped_items still reports all 202 entries."""
    dd, deep = _worked_example_dirs(tmp_path)
    # Random hex descriptions avoid accidentally crossing the 0.5 similarity threshold.
    items = [_item(i, description=uuid.uuid4().hex) for i in range(200)]
    items.append(_item(200, description="the exact same duplicate description text"))
    items.append(_item(201, description="the exact same duplicate description text"))
    seed_merged_items(deep, items)

    duplication = analyze_shipped_duplication(dd)

    assert duplication["shipped_items"] == 202
    assert duplication["near_duplicate_pairs"] == 0  # the tail pair is beyond the cap


@pytest.mark.parametrize(("payload", "expected_error"),
    [pytest.param("{not json", json.JSONDecodeError, id="syntax-invalid"),
        pytest.param('{"items": {"a": 1}}', ValueError, id="wrong-shape"),
        pytest.param("{}", ValueError, id="missing-items"),
    ],
)
def test_location_and_duplication_propagate_corrupt_merged_items(
    tmp_path: Path, payload: str, expected_error: type[Exception]
) -> None:
    """The shared loader's raise-on-corrupt contract holds for the new axes too.

    A bogus shipped set is never silently counted -- and never silently scored."""
    dd, deep = _worked_example_dirs(tmp_path)
    seed_stack_records(deep, "python", n=4)
    (deep / "merged-items.json").write_text(payload)

    with pytest.raises(expected_error):
        analyze_findings(dd)
    with pytest.raises(expected_error):
        analyze_location(dd)
    with pytest.raises(expected_error):
        analyze_shipped_duplication(dd)


def _grounding_finding(**extra: Any) -> dict[str, Any]:
    """A pre-merge per-stack record tagged for the ``deep-python`` reader."""
    finding: dict[str, Any] = {
        "id": "py-1", "_stack": "python", "file": "svc/loader.py", "line": 88, "confidence": "HIGH",
        "rationale": "svc/loader.py needs a guard",
    }
    finding.update(extra)
    return finding


def test_analyze_session_reports_location_and_shipped_duplication(tmp_path: Path,) -> None:
    """Analyze one persisted shipped population with independently countable pair/location output.

    Mixed attribution, exemptions and relocated citations must survive the same join; unrelated
    findings cannot hide a wrong per-item tier or fabricated UID in an aggregate count.
    """
    dd, deep = _deep_dirs(tmp_path)
    seed_diff_patch(dd)  # This competing patch has only 85–92; the saved index is authoritative.
    seed_hunk_index(dd, {"svc/loader.py": [(200, 210)]})
    items = [
        _item(1, line=205, description="Loader guard", lens="structural", uid="structure:3"),
        _item(
            2,
            line=210,
            location_cited_line=212,
            description="Loader guards",
            lens="cross-stack",
            **_provenance("python:1", "react:2"),
        ),
        _item(3, line=4, description="Missing mutex", location_distrust=True, location_cited_line=4, **_provenance()),
        _item(4, file="other/untouched.py", line=205, description="ZIP checksum", uid="python:2"),
        _item(5, line=0, description="Loader guard", lens="structural", uid="structure:5"),
        _item(6, line="not-an-int", description=""),
        _item(7, line=True, description=""),
    ]
    assert all(RECORD_SOURCE_UIDS_KEY not in item for item in (items[0], items[3], items[4]))
    seed_merged_items(deep, items)
    seed_dedup_candidates(
        deep,
        record_alt_pairs=[{"similarity": 0.75}, {"similarity": 0.55}],
        record_duplicate_pairs=[{"similarity": 0.9}],
    )
    result = analyze_session(
        dd, write_snapshot=run_snapshot("loc-session", schema_version="ATIF-v1.7", model_name="test")
    )
    location = result["location"]
    assert location["hunk_source"] == "hunk-index.json"
    assert (
        location["shipped_items"],
        location["scored_items"],
        location["whole_file_anchors"],
        location["unscorable_items"],
        location["relocated_items"],
        location["distrusted_items"],
    ) == (7, 4, 1, 2, 1, 1)
    assert location["tiers"] == {"in_hunk": 1, "within_tolerance": 1, "beyond_tolerance": 1, "file_absent": 1}
    assert location["tier_rates"] == {
        "in_hunk": 0.25,
        "within_tolerance": 0.25,
        "beyond_tolerance": 0.25,
        "file_absent": 0.25,
    }
    assert location["in_hunk_rate"] == 0.25
    assert [(row["id"], row["line"], row["cited_line"], row["tier"], row["distance"]) for row in location["items"]] == [
        (1, 205, None, "in_hunk", 0),
        (2, 210, 212, "within_tolerance", 2),
        (3, 4, 4, "beyond_tolerance", 196),
        (4, 205, None, "file_absent", None),
    ]
    duplication = result["findings"]["shipped_duplication"]
    assert (
        duplication["shipped_items"],
        duplication["comparable_pairs"],
        duplication["near_duplicate_pairs"],
        duplication["same_file_pairs"],
        duplication["same_file_near_duplicate_pairs"],
    ) == (7, 10, 3, 6, 3)
    # 'loader guard' has 11 distinct character bigrams; its plural adds one. The other
    # two titles have disjoint bigrams, so the 10-pair mean is (1 +11/12 +11/12) /10.
    assert duplication["max_similarity"] == 1.0 and duplication["mean_similarity"] == 0.2833
    assert [(row["a_id"], row["b_id"], row["similarity"]) for row in duplication["pairs"]] == [
        ("1", "5", 1.0),
        ("1", "2", 0.9167),
        ("2", "5", 0.9167),
        ("1", "3", 0.0),
        ("1", "4", 0.0),
        ("2", "3", 0.0),
        ("2", "4", 0.0),
        ("3", "4", 0.0),
        ("3", "5", 0.0),
        ("4", "5", 0.0),
    ]
    sources = {"1": ["structure:3"], "2": ["python:1", "react:2"], "3": [], "4": ["python:2"], "5": ["structure:5"]}
    lenses = {"1": "structural", "2": "cross-stack", "3": "per-stack", "4": "per-stack", "5": "structural"}
    for row in duplication["pairs"]:
        assert row["a_source_uids"] == sources[row["a_id"]] and row["b_source_uids"] == sources[row["b_id"]]
        assert (row["a_lens"], row["b_lens"]) == (lenses[row["a_id"]], lenses[row["b_id"]])
        assert row["same_file"] == ("4" not in (row["a_id"], row["b_id"]))
    assert result["findings"]["dedup"] == {
        "record_duplicate_candidates": 1,
        "record_alt_overlaps": 2,
        "avg_overlap_similarity": 0.65,
    }


@pytest.mark.parametrize(("filename", "label"),
    [("trajectory.json", "main"), ("deep-python.json", "deep-python"),
        ("trajectory-20260101T000000-abc123.json", "main"), ("deadbeef.deep-python.json", "deep-python"),
    ],
)
def test_agent_label_keeps_its_legacy_filename_tolerance(filename: str, label: str) -> None:
    """The legacy shapes are retained deliberately — pin them instead of guessing they are dead."""
    assert _agent_label(filename) == label
