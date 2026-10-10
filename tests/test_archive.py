"""Unit tests for the daydream.archive package.

Covers git_context, manifest, index, and the strict ``finalize_archive_run`` flow.
"""
import json
import sqlite3
from collections.abc import Callable, Mapping
from contextlib import closing
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import pytest

from daydream.archive import (
    ArchiveFinalizationError,
    finalize_archive_run,
    get_archive_dir,
    pipeline,
)
from daydream.archive.bundle import _copy_snapshot_bundle, _project_documents
from daydream.archive.git_context import GitContext, capture_git_context
from daydream.archive.index import upsert_run
from daydream.archive.manifest import (
    Manifest,
    archive_recorder_provenance_from_snapshot,
    build_manifest_from_snapshot,
)
from daydream.archive.pipeline import (
    _read_session_bound_json_artifact,
    derive_phase_states,
    derive_pipeline_status,
)
from daydream.artifact_visibility import (
    ArtifactEvidenceProvenance,
    ArtifactTreeSnapshot,
    RoutedDestination,
)
from daydream.artifacts.filesystem import manifest_tree
from daydream.backends import ResultEvent, TextEvent
from daydream.deep.artifacts import DeepArtifact
from daydream.remote_ci import (
    CIObservation,
    PRCIBinding,
    RemoteCILimits,
    RemoteCISnapshot,
    RemoteCITarget,
    RemoteCIVerdict,
    RequiredContext,
    RequiredPolicy,
    evaluate_remote_ci,
    write_remote_ci_verdict,
)
from daydream.run_config import RunConfig
from daydream.run_snapshot import (
    ArchiveRunSnapshot,
    ManifestRunIdentity,
    RunPhaseCapabilities,
)
from daydream.runner import run
from daydream.trajectory import (
    DaydreamPhase,
    DaydreamRunFlow,
    PhaseEvent,
    RunWriteSnapshot,
    TrajectoryDocumentSnapshot,
    partial_document_path,
    run_directory,
    run_document_path,
    sibling_document_path,
)
from tests.harness.backend import ScriptedBackend
from tests.harness.improve_backend import install_improve_stub
from tests.harness.review_result import review_coverage
from tests.harness.trajectory import make_manifest, make_recorder
from tests.test_extension_seam_integration import CUSTOM_FLOW_EXT

MakeConfig = Callable[..., RunConfig]
InstallBackend = Callable[[object], object]

_DEFAULT_FINAL_METRICS: dict[str, Any] = {
    "total_prompt_tokens": 100, "total_completion_tokens": 50, "total_cached_tokens": 20, "total_cost_usd": 0.05,
}

def _write_snapshot(recorder: Any, *, status: str = "complete", phase_events: list[dict[str, Any]] | None = None,
    final_metrics: dict[str, Any] | None = None, lifecycle: tuple[str, str] | None = None,
) -> RunWriteSnapshot:
    """Freeze the recorder's root with optional metric and lifecycle overrides.

    A supplied lifecycle also pins the cutoff to its end, making complete timing valid.
    """
    path = Path(recorder.path)
    payload: dict[str, Any] = {}
    if path.is_file():
        loaded = json.loads(path.read_text())
        if isinstance(loaded, dict):
            payload = loaded
    trajectory_id = str(getattr(recorder, "session_id"))
    # A frozen root document always carries the archived run's own identity.
    payload["session_id"] = trajectory_id
    payload["trajectory_id"] = trajectory_id
    payload.setdefault("steps", [])
    extra: dict[str, Any] = payload.setdefault("extra", {})
    if phase_events is not None:
        extra["phase_events"] = phase_events
    if getattr(recorder, "pr_number", None) is not None:
        extra["pr_number"] = recorder.pr_number
    if getattr(recorder, "pr_repo", None) is not None:
        extra["pr_repo"] = recorder.pr_repo
    cutoff_at = "2026-01-01T00:00:01Z"
    if lifecycle is not None:
        extra["run_started_at"], extra["run_ended_at"] = lifecycle
        cutoff_at = lifecycle[1]
    if final_metrics is not None:
        payload["final_metrics"] = final_metrics
    else:
        payload.setdefault("final_metrics", dict(_DEFAULT_FINAL_METRICS))
    document = TrajectoryDocumentSnapshot(
        trajectory_id=trajectory_id, path=path, json_bytes=json.dumps(payload).encode(),
    )
    return RunWriteSnapshot(
        status=cast(Any, status), cutoff_at=cutoff_at, root_trajectory_id=trajectory_id, documents=(document,),
    )

@dataclass
class _MockRecorder:
    """The recorder identity fields a frozen snapshot is stamped from."""
    session_id: str = "abcd1234-0000-0000-0000-000000000000"
    path: Path = Path("/nonexistent/trajectory.json")
    run_flow: DaydreamRunFlow = DaydreamRunFlow.NORMAL
    pr_number: int | None = None
    pr_repo: str | None = None

def _frozen_target(tmp_path: Path) -> Path:
    """A frozen artifact root that never contains the archive directory itself.

    ``finalize_archive_run`` re-attests the tree it was handed after every
    stage, so the archive (which the ``archive_dir`` fixture puts under
    ``tmp_path``) must live outside the root under attestation.
    """
    target = tmp_path / "frozen"
    target.mkdir()
    return target

@pytest.fixture
def target(tmp_path: Path) -> Path:
    """A frozen artifact root isolated from the archive directory."""
    return _frozen_target(tmp_path)



def _strict_archive(*, target: Path, session_id: str, config: Any, write_snapshot: RunWriteSnapshot,
    run_flow: DaydreamRunFlow = DaydreamRunFlow.NORMAL, identity: ManifestRunIdentity | None = None,
    destinations: tuple[RoutedDestination, ...] = (), work: Any = None,
    dump_path: Path | None = None,
) -> None:
    """Finalize a frozen run; target must exclude the archive directory, as in `_frozen_target`."""
    finalize_archive_run(run=_archive_snapshot(write_snapshot, run_flow=run_flow, identity=identity),
        artifacts=ArtifactTreeSnapshot(
            session_id=session_id, workspace_key="workspace", root=target, manifest=manifest_tree(target),
            destinations=destinations,
        ),
        artifact_provenance=ArtifactEvidenceProvenance(
            workspace_key="workspace", session_id=session_id, public_source=target,
            # The frozen root is a copy of the live root, so route paths relative
            # to one resolve unchanged inside the other.
            live_root=target,
        ), config=config, work=work, dump_path=dump_path,
    )

def _manifest_identity(**overrides: Any) -> ManifestRunIdentity:
    """Build the public, already-resolved identity supplied by the runner."""
    identity = ManifestRunIdentity(
        skill="python", model=None, backend="claude", review_backend=None, fix_backend="claude", test_backend="claude",
        per_stack_review_backend="claude", per_stack_review_model="sonnet", review_only=False, deep=True, profile=None,
        phases=RunPhaseCapabilities(per_stack_review=True, merge=True, fix=True, test=True, push=True, remote_ci=True,),
    )
    return replace(identity, **overrides)

def _manifest_write_snapshot(
    *, session_id: str = "abcd1234-0000-0000-0000-000000000000", final_metrics: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None, path: Path = Path("/frozen/trajectory.json"),
) -> RunWriteSnapshot:
    """Build explicit immutable root bytes for manifest-reducer tests."""
    snapshot_extra = dict(extra or {})
    payload = {"session_id": session_id, "trajectory_id": session_id, "steps": [],
        "final_metrics": dict(_DEFAULT_FINAL_METRICS if final_metrics is None else final_metrics),
        "extra": snapshot_extra,
    }
    return RunWriteSnapshot(
        status="complete", cutoff_at=str(snapshot_extra.get("run_ended_at", "2026-01-01T00:00:01Z")),
        root_trajectory_id=session_id,
        documents=(
            TrajectoryDocumentSnapshot(trajectory_id=session_id, path=path, json_bytes=json.dumps(payload).encode(),),
        ),
    )

def _archive_snapshot(trajectories: RunWriteSnapshot, *, run_flow: DaydreamRunFlow = DaydreamRunFlow.NORMAL,
    identity: ManifestRunIdentity | None = None,
) -> ArchiveRunSnapshot:
    """Join frozen trajectory provenance with the runner's public identity."""
    return ArchiveRunSnapshot(
        recorder_provenance=archive_recorder_provenance_from_snapshot(write_snapshot=trajectories, run_flow=run_flow,),
        identity=identity or _manifest_identity(), trajectories=trajectories,
    )




def test_capture_git_context_no_repo(tmp_path: Path) -> None:
    ctx = capture_git_context(tmp_path)
    assert ctx.head_sha is None
    assert ctx.remote_url is None
    assert ctx.branch is None
    assert ctx.base_sha is None
    assert ctx.changed_files == []


def _build(tmp_path: Path, *, git_ctx: GitContext | None = None, write_snapshot: RunWriteSnapshot | None = None,
    run_flow: DaydreamRunFlow = DaydreamRunFlow.NORMAL, identity: ManifestRunIdentity | None = None, **kw: Any,
) -> Manifest:
    """Build a manifest from immutable public archive inputs."""
    snapshot = write_snapshot or _manifest_write_snapshot()
    return build_manifest_from_snapshot(run=_archive_snapshot(snapshot, run_flow=run_flow, identity=identity,),
        git_ctx=git_ctx if git_ctx is not None else GitContext(), status="complete", archive_path=tmp_path, **kw,
    )





def _assert_archive_omits_fix_test_backend(archive_dir: Path, flow: str) -> None:
    manifests = list((archive_dir / "runs").glob("*/manifest.json"))
    assert len(manifests) == 1, f"expected exactly one archived run, found {len(manifests)}"
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["run"]["flow"] == flow
    assert "fix_backend" not in manifest["run"]
    assert "test_backend" not in manifest["run"]
    with closing(sqlite3.connect(f"{(archive_dir / 'index.db').as_uri()}?mode=ro", uri=True)) as conn:
        assert conn.execute("SELECT fix_backend, test_backend FROM runs").fetchall() == [(None, None)]

async def test_improve_archive_real_path_omits_fix_test_backend(
    improve_monorepo_target: Path, monkeypatch: pytest.MonkeyPatch, archive_dir: Path, make_config: MakeConfig,
) -> None:
    monkeypatch.delenv("DAYDREAM_TRAJECTORY_HUB_REPO", raising=False)
    install_improve_stub(monkeypatch, improve_monorepo_target)
    rc = await run(make_config(improve_monorepo_target, flow_name="improve", archive=True, run_eval=False,))
    assert rc == 0
    _assert_archive_omits_fix_test_backend(archive_dir, "improve")

async def test_custom_flow_archive_real_path_omits_fix_test_backend(
    ext_dir: Any, multi_stack_target: Path, install_backend: InstallBackend, monkeypatch: pytest.MonkeyPatch,
    archive_dir: Path, make_config: MakeConfig,
) -> None:
    """Issue #648 real-path: a fork-registered custom flow archives no fix/test backend.

    Same production entry (``runner.run``) with a real git worktree and
    recorder; only the backend is mocked. The custom flow is extension-defined,
    so its pipeline has no fix/test step and the archive must not invent labels.
    """
    monkeypatch.delenv("DAYDREAM_TRAJECTORY_HUB_REPO", raising=False)
    ext_dir.write_module(CUSTOM_FLOW_EXT)
    install_backend(
        ScriptedBackend(
            events=(
                TextEvent(text=""),
                ResultEvent(structured_output=None, continuation=None),
            ),
            model="mock-model",
        )
    )

    rc = await run(make_config(multi_stack_target, flow_name="ro-audit", archive=True, run_eval=False,))

    assert rc == 0
    _assert_archive_omits_fix_test_backend(archive_dir, "custom")



def test_build_manifest_with_evaluation(tmp_path: Path) -> None:
    m = _build(tmp_path,
        evaluation={"timing": {"total_wall_clock_seconds": 42.5}, "findings": {"total": 7},
            "grounding": {"grounding_rate": 0.85}, "coverage": {"coverage_ratio": 0.6},
            "derived": {"cost_per_finding_usd": 0.007},
        },
    )
    assert m.wall_clock_seconds == 42.5
    assert m.total_findings == 7
    assert "grounding_rate" not in m.to_dict()["metrics"]
    assert "coverage_ratio" not in m.to_dict()["metrics"]
    assert m.cost_per_finding_usd == 0.007

def test_build_manifest_with_quality(tmp_path: Path) -> None:
    m = _build(tmp_path, evaluation={"quality": {"erosion": 0.34, "verbosity": 0.19},},)
    assert m.erosion == 0.34
    assert m.verbosity == 0.19
    d = m.to_dict()
    assert d["metrics"]["erosion"] == 0.34
    assert d["metrics"]["verbosity"] == 0.19



def test_build_manifest_snapshot_timing_overrides_conflicting_evaluation(tmp_path: Path,) -> None:
    session_id = "snapshot-session"
    payload = {"session_id": session_id, "trajectory_id": session_id, "steps": [],
        "final_metrics": {"total_prompt_tokens": 7, "total_steps": 0},
        "extra": {"run_started_at": "2026-01-01T00:00:00Z", "run_ended_at": "2026-01-01T00:00:10Z",
            "phase_events": [
                _phase_start_event("review", session_id, timestamp="2026-01-01T00:00:02Z", scope_id="review"),
                {"phase": "review", "event": "phase_end", "timestamp": "2026-01-01T00:00:08Z",
                    "session_id": session_id, "scope_id": "review", "status": "succeeded",
                },
            ],
        },
    }
    snapshot = RunWriteSnapshot(status="complete", cutoff_at="2026-01-01T00:00:10Z", root_trajectory_id=session_id,
        documents=(TrajectoryDocumentSnapshot(
                session_id, Path("/frozen/snapshot-session.json"), json.dumps(payload).encode(),
            ),
        ),
    )

    manifest = _build(tmp_path, write_snapshot=snapshot, evaluation={"timing": {"total_wall_clock_seconds": 42.5}},)

    assert manifest.wall_clock_seconds == 10.0
    assert manifest.phase_timings == {"review": {"wall_clock_seconds": 6.0, "occurrences": 1}}
    assert manifest.timing_coverage == {
        "attributed_wall_clock_seconds": 6.0, "unattributed_wall_clock_seconds": 4.0, "coverage_ratio": 0.6,
        "agent_completeness": {"total": 0, "attributed": 0, "unattributed": 0},
        "diagnostics": {
            "malformed_interval": 0, "duplicate_interval": 0, "orphaned_interval": 0, "malformed_invocation": 0,
            "duplicate_invocation": 0,
        },
    }
    assert manifest.total_prompt_tokens == 7

def _stored_manifest_row(archive_dir: Path, session_id: str, **fields: Any) -> dict[str, Any]:
    upsert_run(archive_dir, make_manifest(session_id=session_id, **fields))
    with closing(sqlite3.connect(f"{(archive_dir / 'index.db').as_uri()}?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        return dict(conn.execute("SELECT * FROM runs WHERE session_id = ?", (session_id,)).fetchone())

def test_upsert_run_never_persists_credential_bearing_url(tmp_path: Path) -> None:
    # M4: even if upstream normalization is bypassed, the row is clean.
    m = make_manifest(remote_url="https://user:ghp_bypassfake@github.com/o/r.git")
    upsert_run(tmp_path, m)
    with closing(sqlite3.connect(f"{(tmp_path / 'index.db').as_uri()}?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        row = dict(conn.execute("SELECT * FROM runs").fetchone())
    assert row["remote_url"] == "https://github.com/o/r"
    assert row["repo_slug"] == "o/r"
    assert "ghp_bypassfake" not in (row["remote_url"] or "")



def test_upsert_run_persists_erosion_verbosity(tmp_path: Path) -> None:
    upsert_run(tmp_path, make_manifest(session_id="s-q1", erosion=0.34, verbosity=0.19))
    upsert_run(tmp_path, make_manifest(session_id="s-q2", archive_path="/tmp/s-q2"))
    upsert_run(tmp_path, make_manifest(session_id="s-q3", erosion=0.51, verbosity=0.07, archive_path="/tmp/s-q3"),)

    with closing(sqlite3.connect(f"{(tmp_path / 'index.db').as_uri()}?mode=ro", uri=True)) as conn:
        row = conn.execute("SELECT erosion, verbosity FROM runs WHERE session_id = ?", ("s-q1",)).fetchone()
        assert row == pytest.approx((0.34, 0.19))
        scored = conn.execute("SELECT session_id FROM runs WHERE erosion IS NOT NULL").fetchall()
        assert {session_id for (session_id,) in scored} == {"s-q1", "s-q3"}
        # The columns are sortable (NULLs sort first in SQLite).
        ordered = conn.execute("SELECT session_id FROM runs ORDER BY erosion").fetchall()
        assert ordered == [("s-q2",), ("s-q1",), ("s-q3",)]




def test_null_in_hunk_rate_survives_manifest_and_db_as_null(tmp_path: Path) -> None:
    """Undefined location accuracy stays NULL so reward reduction can omit the absent axis."""
    m = _build(tmp_path,
        evaluation={"location": {"hunk_source": "none", "scored_items": 0, "in_hunk_rate": None,},
            "findings": {"total": 0, "shipped_duplication": {"near_duplicate_pairs": 0},},
        },
    )
    assert m.location_in_hunk_rate is None
    assert m.to_dict()["metrics"]["location_in_hunk_rate"] is None

    row = _stored_manifest_row(tmp_path, "s-loc-null", location_in_hunk_rate=m.location_in_hunk_rate,
        shipped_duplicate_pairs=m.shipped_duplicate_pairs,
    )
    assert row["location_in_hunk_rate"] is None  # SQL NULL, not 0.0
    assert row["shipped_duplicate_pairs"] == 0  # a real zero is still a zero







@pytest.mark.parametrize(
    ("resolver", "artifact", "payload"),
    [
        pytest.param(
            DeepArtifact.FIX_QUALITY_GATE.at,
            "fix-quality-gate.json",
            {
                "enabled": True,
                "session_id": "sess-42",
                "rounds": [{"round": 1, "per_file": {"api.py": {"flagged": True}}}],
            },
            id="fix-quality-gate",
        ),
        pytest.param(
            DeepArtifact.RECOMMENDED_CAPTURE.at,
            "recommended-capture.json",
            {"session_id": "sess-42", "capture_point": "post_test"},
            id="recommended-capture",
        ),
    ],
)
def test_session_bound_artifact_requires_matching_session(
    tmp_path: Path, resolver: Callable[[Path], Path], artifact: str, payload: dict[str, Any]
) -> None:
    p = tmp_path / ".daydream" / "deep" / artifact
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(payload))
    assert _read_session_bound_json_artifact(tmp_path, "sess-42", resolver) == payload
    assert _read_session_bound_json_artifact(tmp_path, "sess-other", resolver) is None
    assert _read_session_bound_json_artifact(tmp_path, None, resolver) is None

@pytest.mark.parametrize(
    "resolver", [DeepArtifact.FIX_QUALITY_GATE.at, DeepArtifact.RECOMMENDED_CAPTURE.at],
    ids=["fix-quality-gate", "recommended-capture"],
)
def test_session_bound_unbound_artifact_is_none(tmp_path: Path, resolver: Callable[[Path], Path]) -> None:
    p = resolver(tmp_path / ".daydream" / "deep")
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"enabled": True, "rounds": [{"round": 1, "per_file": {}}]}))
    assert _read_session_bound_json_artifact(tmp_path, "sess-42", resolver) is None

def test_session_bound_absent_artifact_is_none(tmp_path: Path) -> None:
    assert _read_session_bound_json_artifact(tmp_path, "sess-42", DeepArtifact.RECOMMENDED_CAPTURE.at) is None




def test_feedback_run_leaves_recommended_patch_capture_none() -> None:
    """Feedback never writes recommended.patch, so absence cannot imply a pre-test capture."""
    m = _build(tmp_path=Path("/tmp"), run_flow=DaydreamRunFlow.PR)
    assert m.recommended_patch_capture is None
    assert "recommended_patch_capture" not in m.to_dict()


def test_get_archive_dir_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("DAYDREAM_ARCHIVE_DIR", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))
    result = get_archive_dir()
    expected = tmp_path / ".daydream" / "archive"
    assert result == expected
    assert expected.is_dir()

def _setup_bundle(tmp_path: Path, session_id: str = "abcd1234-0000-0000-0000-000000000000",
) -> tuple[Path, Path, _MockRecorder]:
    """Create a frozen artifact tree with one run's artifacts, plus an empty run dir.

    Layout mirrors the frozen tree the host hands the archive:
    ``.daydream/runs/<session_id>/trajectory.json`` alongside the deep
    artifacts, the diff, and the public review output in the tree root.
    """
    target = tmp_path / "target"
    daydream = target / ".daydream"
    daydream.mkdir(parents=True)
    frozen_run_dir = daydream / "runs" / session_id
    frozen_run_dir.mkdir(parents=True)

    traj = frozen_run_dir / "trajectory.json"
    traj.write_text(json.dumps({"session_id": session_id, "trajectory_id": session_id}))

    deep = daydream / "deep"
    deep.mkdir()
    (deep / "intent.md").write_text("intent")

    (daydream / "diff.patch").write_text("diff content")

    # Review output lives in target root, not .daydream/.
    (target / ".review-output.md").write_text("review findings")

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    return target, run_dir, _MockRecorder(session_id=session_id, path=traj)

def _assemble_bundle(
    target: Path, run_dir: Path, recorder: _MockRecorder, *, write_snapshot: RunWriteSnapshot | None = None,
    destinations: tuple[RoutedDestination, ...] = (),
) -> None:
    """Run the production bundle assembler over one frozen tree + snapshot."""
    snapshot = write_snapshot if write_snapshot is not None else _write_snapshot(recorder)
    _copy_snapshot_bundle(run=_archive_snapshot(snapshot, run_flow=recorder.run_flow),
        artifacts=ArtifactTreeSnapshot(
            session_id=recorder.session_id, workspace_key="workspace", root=target, manifest=manifest_tree(target),
            destinations=destinations,
        ),
        artifact_provenance=ArtifactEvidenceProvenance(
            workspace_key="workspace", session_id=recorder.session_id, public_source=target, live_root=target,
        ), run_dir=run_dir,
    )


def test_bundle_rejects_a_sibling_document_bound_to_another_session(tmp_path: Path,) -> None:
    target, run_dir, recorder = _setup_bundle(tmp_path)
    root = _write_snapshot(recorder).documents[0]
    foreign = json.dumps({"session_id": "other-session", "trajectory_id": "fork-1"}).encode()
    snapshot = RunWriteSnapshot(
        status="complete", cutoff_at="2026-01-01T00:00:01Z", root_trajectory_id=recorder.session_id,
        documents=(root, TrajectoryDocumentSnapshot("fork-1", target / "fork.json", foreign)),
    )
    with pytest.raises(ArchiveFinalizationError, match="frozen trajectory projection failed"):
        _assemble_bundle(target, run_dir, recorder, write_snapshot=snapshot)
    assert not (run_dir / "trajectories").exists()



def test_bundle_diagram_flow_excludes_stale_review_artifacts(tmp_path: Path,) -> None:
    target, run_dir, recorder = _setup_bundle(tmp_path)
    recorder.run_flow = DaydreamRunFlow.DIAGRAM
    deep_dir = target / ".daydream" / "deep"
    (deep_dir / "merged-items.json").write_text('{"items": []}')
    (deep_dir / "fix-failures.json").write_text('{"src/old.py": "reverted"}')
    (deep_dir / "diagram.json").write_text('{"results": {}}')
    (deep_dir / "diagram.md").write_text("current diagram")
    (target / ".daydream" / "recommended.patch").write_text("stale recommendation")
    _assemble_bundle(target, run_dir, recorder)
    assert sorted(path.name for path in (run_dir / "deep").iterdir()) == ["diagram.json", "diagram.md",]
    assert (run_dir / "deep" / "diagram.md").read_text() == "current diagram"
    assert not (run_dir / "review-output.md").exists()
    assert not (run_dir / "recommended.patch").exists()


def test_bundle_skips_missing(tmp_path: Path) -> None:
    target = tmp_path / "empty_target"
    target.mkdir()
    (target / ".daydream").mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    recorder = _MockRecorder(session_id="no-match-session-id-here", path=tmp_path / "nonexistent.json")
    _assemble_bundle(target, run_dir, recorder)
    assert (run_dir / "trajectory.json").is_file()  # the frozen document always lands
    assert not (run_dir / "review-output.md").exists()
    assert not (run_dir / "deep").exists()
    assert not (run_dir / "diff.patch").exists()







# Canonical UTC timestamp contract: one spelling at write time, strict as_of
# validation at the entry boundary, and legacy "Z" rows preserved at bootstrap.




def test_legacy_manifest_reads_new_fields_as_unknown(tmp_path: Path) -> None:
    upsert_run(tmp_path, Manifest())
    with closing(sqlite3.connect(f"{(tmp_path / 'index.db').as_uri()}?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        row = dict(conn.execute("SELECT * FROM runs").fetchone())
    assert row["pipeline_status"] == "unknown"
    assert row["archive_status"] == "complete"
    assert row["daydream_version"] is None

def _write_deep(target: Path, name: str, data: Any) -> None:
    deep = target / ".daydream" / "deep"
    deep.mkdir(parents=True, exist_ok=True)
    (deep / name).write_text(json.dumps(data), encoding="utf-8")


def _write_review_coverage(target: Path, *, state: str = "failed") -> None:
    coverage = review_coverage(run_id="prior", scope_ids=("python",))
    coverage.record_scope("python", "complete")
    coverage.record_phase("merge", state, reasons=() if state == "complete" else ("synthesis_failure",))
    _write_deep(target, "review-coverage.json", coverage.to_dict())


_PUSHED_SHA = "a" * 40
_MERGE_SHA = "b" * 40

def _phase_event(phase: DaydreamPhase, event: str = "phase_start") -> PhaseEvent:
    return PhaseEvent(phase=phase, event=event, timestamp="2026-09-06T12:00:00Z",)

def _write_push_verdict(target: Path, *, session_id: str = "current", status: str = "succeeded", remote: str = "origin",
    branch: str = "feature/remote-ci", sha: str = _PUSHED_SHA, repository: str | None = "contributor/fork",
) -> None:
    payload: dict[str, Any] = {
        "schema_version": 1, "session_id": session_id, "status": status, "remote": remote, "branch": branch,
        "pushed_sha": sha, "pushed_repository": repository, "started_at": "2026-09-06T12:00:00Z",
        "updated_at": "2026-09-06T12:00:01Z",
    }
    if status == "failed":
        payload["diagnostic"] = "ordinary push rejected"
    _write_deep(target, "push-verdict.json", payload)

def _write_remote_verdict(
    target_dir: Path, *, status: str = "passed", session_id: str = "current", advisory: tuple[CIObservation, ...] = (),
) -> None:
    target = RemoteCITarget(
        target_dir=target_dir, base_repository="example/project", base_ref="main", head_repository="contributor/fork",
        head_ref="feature/remote-ci", pr_number=42, pr_url="https://github.com/example/project/pull/42",
        remote="origin", pushed_sha=_PUSHED_SHA,
    )
    binding = PRCIBinding(
        pr_number=42, pr_url=target.pr_url, base_repository=target.base_repository, base_ref=target.base_ref,
        head_repository=target.head_repository, head_ref=target.head_ref, head_sha=target.pushed_sha,
        merge_sha=_MERGE_SHA, state="open",
    )
    required: tuple[CIObservation, ...] = ()
    if status != "no_ci":
        state = cast(Any, "fail" if status == "failed" else "pending" if status == "pending" else "pass",)
        required = (CIObservation(source="check_run", context="Build", app_id=10, state=state,
                raw_state="failure" if status == "failed" else state,
                url="https://github.com/example/project/actions/runs/7", diagnostic=None,
            ),
        )
    policy = (RequiredPolicy((), False) if status == "no_ci" else RequiredPolicy((RequiredContext("Build", 10),), True))
    verdict = RemoteCIVerdict(
        status=cast(Any, status), reason=f"remote CI {status}", target=target, binding=binding, policy=policy,
        active_workflow_count=0 if status == "no_ci" else 1,
        evidence_sha=_PUSHED_SHA if status == "no_ci" else _MERGE_SHA, required_observations=required,
        advisory_observations=advisory, failing_contexts=("Build (app 10)",) if status == "failed" else (),
        pending_contexts=("Build (app 10)",) if status == "pending" else (),
        missing_contexts=("Build (app 10)",) if status == "missing" else (),
        urls=tuple(item.url for item in (*required, *advisory) if item.url), diagnostic=None, stable_polls=2,
        elapsed_seconds=120 if status == "no_ci" else 20,
    )
    write_remote_ci_verdict(
        target_dir / ".daydream" / "deep" / "remote-ci-verdict.json", verdict, session_id=session_id, poll_count=2,
        started_at="2026-09-06T12:00:00Z",
        updated_at="2026-09-06T12:02:00Z" if status == "no_ci" else "2026-09-06T12:00:20Z", discovery_deadline=120,
        completion_deadline=1800,
    )


def _current_ci_artifact(
    target: Path, *, status: str = "passed", advisory: tuple[CIObservation, ...] = (),
) -> tuple[Path, dict[str, Any]]:
    """Seed a matching current push/CI pair, then expose CI bytes for corruption tests."""
    _write_push_verdict(target)
    _write_remote_verdict(target, status=status, advisory=advisory)
    artifact = target / ".daydream" / "deep" / "remote-ci-verdict.json"
    return artifact, json.loads(artifact.read_text(encoding="utf-8"))


def _derive_push_remote_states(target: Path, *, session_id: str = "current", events: list[PhaseEvent] | None = None,
    pr_repo: str | None = "example/project", pr_number: int | None = 42,
) -> dict[str, dict[str, Any]]:
    return pipeline.derive_phase_states(
        target, phase_events=events or [], runs_merge=False, runs_fix=False, runs_test=True, runs_push=True,
        runs_remote_ci=True, session_id=session_id, pr_repo=pr_repo, pr_number=pr_number,
    )




@pytest.mark.parametrize("remote_status", ["pending", "missing", "unavailable", "timed_out", "superseded", "cancelled"],
)
def test_incomplete_remote_statuses_are_partial(tmp_path: Path, remote_status: str) -> None:
    _write_push_verdict(tmp_path)
    _write_remote_verdict(tmp_path, status=remote_status)
    states = _derive_push_remote_states(tmp_path)
    assert states["remote_ci"]["status"] == "partial"
    assert pipeline.derive_pipeline_status("complete", None, states) == "partial"

@pytest.mark.parametrize(("path", "value"),
    [(("session_id",), "prior"), (("target", "pushed_sha"), "c" * 40), (("target", "remote"), "upstream"),
        (("target", "head_ref"), "other"), (("target", "head_repository"), "other/repo"),
        (("binding", "base_repository"), "other/repo"), (("binding", "base_ref"), "release"),
        (("binding", "head_repository"), "other/repo"), (("binding", "head_ref"), "other"),
        (("binding", "pr_number"), 43), (("binding", "pr_url"), "https://github.com/example/project/pull/43"),
        (("binding", "head_sha"), "c" * 40), (("policy",), None), (("polling", "poll_count"), 0),
        (("active_workflow_count",), True), (("required_observations",), {}), (("limitations",), None),
        (("failing_contexts",), ["Build (app 10)"]),
    ],
)
def test_remote_success_identity_mismatch_is_partial(tmp_path: Path, path: tuple[str, ...], value: object) -> None:
    artifact, payload = _current_ci_artifact(tmp_path)
    cursor = payload
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    assert _derive_push_remote_states(tmp_path)["remote_ci"]["status"] == "partial"

def test_remote_archive_state_field_is_not_outcome_authority(tmp_path: Path) -> None:
    artifact, payload = _current_ci_artifact(tmp_path)
    payload["archive_state"] = "failed"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    assert _derive_push_remote_states(tmp_path)["remote_ci"]["status"] == "succeeded"

@pytest.mark.parametrize(("repo", "number", "expected_status"), [
    ("other/project", 42, "partial"), ("example/project", 43, "partial"),
    ("ExAmPlE/PrOjEcT", 42, "succeeded"),
])
def test_remote_success_matches_configured_pr(
    tmp_path: Path, repo: str, number: int, expected_status: str,
) -> None:
    _write_push_verdict(tmp_path)
    _write_remote_verdict(tmp_path)
    remote = _derive_push_remote_states(tmp_path, pr_repo=repo, pr_number=number)["remote_ci"]
    assert remote["status"] == expected_status

@pytest.mark.parametrize(("artifact_name", "path"),
    [("push-verdict.json", ("pushed_repository",)), ("remote-ci-verdict.json", ("target", "base_repository")),
        ("remote-ci-verdict.json", ("target", "head_repository")),
        ("remote-ci-verdict.json", ("binding", "base_repository")),
        ("remote-ci-verdict.json", ("binding", "head_repository")),
    ],
)
def test_persisted_repository_identities_remain_canonical_lowercase(
    tmp_path: Path, artifact_name: str, path: tuple[str, ...],
) -> None:
    _write_push_verdict(tmp_path)
    _write_remote_verdict(tmp_path)
    artifact = tmp_path / ".daydream" / "deep" / artifact_name
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    cursor = payload
    for key in path[:-1]:
        cursor = cursor[key]
    value = cursor[path[-1]]
    assert isinstance(value, str)
    cursor[path[-1]] = value.upper()
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    states = _derive_push_remote_states(tmp_path, pr_repo="ExAmPlE/PrOjEcT", pr_number=42,)
    if artifact_name == "push-verdict.json":
        assert states["push"]["status"] == "partial"
        assert states["remote_ci"] == {"ran": False, "status": "absent"}
    else:
        assert states["remote_ci"]["status"] == "partial"

def test_archive_rejects_repository_identity_the_producer_cannot_create(
    target: Path, archive_dir: Path, make_config: MakeConfig,
) -> None:
    """Archive success cannot admit a slug rejected by the verdict producer."""
    oversized = f"{'a' * 100}/{'b' * 102}"
    with pytest.raises(ValueError, match="repository"):
        RemoteCITarget(target_dir=target, base_repository=oversized, base_ref="main", head_repository=oversized,
            head_ref="feature/remote-ci", pr_number=42, pr_url="https://github.com/example/project/pull/42",
            remote="origin", pushed_sha=_PUSHED_SHA,
        )

    recorder = _MockRecorder(session_id="oversized-slug-session")
    _write_deep(target, "test-verdict.json", {"session_id": recorder.session_id, "passed": True},)
    _write_push_verdict(target, session_id=recorder.session_id)
    _write_remote_verdict(target, session_id=recorder.session_id)
    push_path = target / ".daydream" / "deep" / "push-verdict.json"
    push = json.loads(push_path.read_text())
    push["pushed_repository"] = oversized
    push_path.write_text(json.dumps(push))
    remote_path = target / ".daydream" / "deep" / "remote-ci-verdict.json"
    remote = json.loads(remote_path.read_text())
    for identity in (remote["target"], remote["binding"]):
        identity["base_repository"] = oversized
        identity["head_repository"] = oversized
    remote_path.write_text(json.dumps(remote))

    _strict_archive(target=target, session_id=recorder.session_id,
        config=make_config(target, archive=True, pr_repo=oversized, pr_number=42,),
        write_snapshot=_write_snapshot(recorder,
            phase_events=[*_merge_events(recorder.session_id, "succeeded"),
                _phase_start_event("fix", recorder.session_id, timestamp="2026-09-06T12:00:00Z"),
                *[_phase_start_event(phase.value, recorder.session_id, timestamp="2026-09-06T12:00:00Z")
                    for phase in (DaydreamPhase.PUSH, DaydreamPhase.REMOTE_CI)
                ],
            ],
        ),
    )

    manifest = json.loads((archive_dir / "runs" / recorder.session_id / "manifest.json").read_text())
    assert manifest["phase_states"]["remote_ci"] == {"ran": True, "status": "partial",}
    assert manifest["pipeline_status"] == "partial"

@pytest.mark.parametrize("malformed_sha", ["A" * 40, "a" * 39, "a" * 41])
def test_remote_archive_rejects_noncanonical_commit_sha(tmp_path: Path, malformed_sha: str,) -> None:
    artifact, payload = _current_ci_artifact(tmp_path)
    payload["target"]["pushed_sha"] = malformed_sha
    payload["binding"]["head_sha"] = malformed_sha
    payload["head_sha"] = malformed_sha
    payload["evidence_sha"] = malformed_sha
    artifact.write_text(json.dumps(payload))
    assert _derive_push_remote_states(tmp_path)["remote_ci"] == {"ran": True, "status": "partial",}

def test_push_failure_is_failed_and_no_receipt_fabricates_no_remote(tmp_path: Path) -> None:
    events = [_phase_event(DaydreamPhase.PUSH)]
    _write_push_verdict(tmp_path, status="failed")
    states = _derive_push_remote_states(tmp_path, events=events)
    assert states["push"]["status"] == "failed"
    assert states["remote_ci"] == {"ran": False, "status": "absent"}
    assert pipeline.derive_pipeline_status("complete", None, states) == "failed"
    (tmp_path / ".daydream" / "deep" / "push-verdict.json").unlink()
    states = _derive_push_remote_states(tmp_path, events=[])
    assert states["push"] == {"ran": False, "status": "absent"}
    assert states["remote_ci"] == {"ran": False, "status": "absent"}

def test_successful_push_without_current_remote_terminal_is_partial(tmp_path: Path) -> None:
    _write_push_verdict(tmp_path)
    states = _derive_push_remote_states(tmp_path)
    assert states["remote_ci"] == {"ran": True, "status": "partial"}
    assert pipeline.derive_pipeline_status("complete", None, states) == "partial"

def test_phase_start_without_terminal_artifact_is_partial(tmp_path: Path) -> None:
    events = [_phase_event(DaydreamPhase.PUSH), _phase_event(DaydreamPhase.REMOTE_CI),]
    states = _derive_push_remote_states(tmp_path, events=events)
    assert states["push"] == {"ran": True, "status": "partial"}
    assert states["remote_ci"] == {"ran": True, "status": "partial"}

def test_remote_advisory_failure_is_detail_not_hard_failure(tmp_path: Path) -> None:
    advisory = CIObservation(source="check_run", context="Optional Linux", app_id=11, state="fail", raw_state="failure",
        url="https://github.com/example/project/actions/runs/8", diagnostic="optional job failed",
    )
    _write_push_verdict(tmp_path)
    _write_remote_verdict(tmp_path, advisory=(advisory,))
    remote = _derive_push_remote_states(tmp_path)["remote_ci"]
    assert remote["status"] == "succeeded"
    assert remote["details"]["advisory_failures"] == ["Optional Linux"]

def test_archive_required_success_with_pending_advisory_is_succeeded(
    target: Path, archive_dir: Path, make_config: MakeConfig,
) -> None:
    recorder = _MockRecorder(session_id="advisory-pending-session")
    config = make_config(target, archive=True, pr_repo="example/project", pr_number=42)
    advisory = CIObservation(
        source="check_run", context="Optional Linux", app_id=11, state="pending", raw_state="in_progress",
        url="https://github.com/example/project/actions/runs/8", diagnostic=None,
    )
    _write_deep(target, "test-verdict.json", {"session_id": recorder.session_id, "passed": True})
    _write_push_verdict(target, session_id=recorder.session_id)
    _write_remote_verdict(target, session_id=recorder.session_id, advisory=(advisory,))
    artifact = target / ".daydream" / "deep" / "remote-ci-verdict.json"
    payload = json.loads(artifact.read_text())
    evaluated = evaluate_remote_ci(RemoteCISnapshot(
            target=RemoteCITarget(target_dir=target, **payload["target"]), binding=PRCIBinding(**payload["binding"]),
            policy=RequiredPolicy((RequiredContext("Build", 10),), True),
            active_workflows=({"id": 7, "state": "active"},), head_observations=(),
            merge_observations=(CIObservation(**payload["required_observations"][0]), advisory),
        ), elapsed=20, stable_polls=2, limits=RemoteCILimits(),
    )
    assert evaluated.status == "passed"
    write_remote_ci_verdict(
        artifact, evaluated, session_id=recorder.session_id, poll_count=2, started_at="2026-09-06T12:00:00Z",
        updated_at="2026-09-06T12:00:20Z", discovery_deadline=120, completion_deadline=1800,
    )

    _strict_archive(target=target, session_id=recorder.session_id, config=config,
        write_snapshot=_write_snapshot(recorder,
            phase_events=[*_merge_events(recorder.session_id, "succeeded"),
                _phase_start_event("fix", recorder.session_id, timestamp="2026-09-06T12:00:00Z"),
            ],
        ),
    )

    run_dir = archive_dir / "runs" / recorder.session_id
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["phase_states"]["remote_ci"]["status"] == "succeeded"
    assert manifest["pipeline_status"] == "succeeded"
    copied = json.loads((run_dir / "deep" / "remote-ci-verdict.json").read_text())
    assert copied["advisory_observations"][0]["state"] == "pending"

def test_archive_no_policy_pending_observation_cannot_be_passed(tmp_path: Path) -> None:
    advisory = CIObservation(source="check_run", context="Build", app_id=10, state="pending", raw_state="in_progress",
        url="https://github.com/example/project/actions/runs/8", diagnostic=None,
    )
    artifact, payload = _current_ci_artifact(tmp_path, advisory=(advisory,))
    payload["policy"] = {"contexts": [], "strict": False}
    payload["required_observations"] = []
    artifact.write_text(json.dumps(payload))
    assert _derive_push_remote_states(tmp_path)["remote_ci"]["status"] == "partial"

@pytest.mark.parametrize("corruption",
    ["missing-required-observations", "missing-one-required-context", "wrong-app-pin", "wrong-check-case",
        "required-marked-advisory", "advisory-marked-required", "duplicated-producer", "policyless-required-partition",
        "policyless-empty-passed", "wrong-failing-context",
    ],
)
def test_archive_rejects_contradictory_terminal_ci_evidence(tmp_path: Path, corruption: str) -> None:
    status = "failed" if corruption == "wrong-failing-context" else "passed"
    artifact, payload = _current_ci_artifact(tmp_path, status=status)
    if corruption == "missing-required-observations":
        payload["required_observations"] = []
        payload["urls"] = []
    elif corruption == "missing-one-required-context":
        payload["policy"]["contexts"].append({"context": "Other", "app_id": 10})
    elif corruption == "wrong-app-pin":
        payload["required_observations"][0]["app_id"] = 11
    elif corruption == "wrong-check-case":
        payload["required_observations"][0]["context"] = "build"
    elif corruption == "required-marked-advisory":
        payload["advisory_observations"] = payload["required_observations"]
        payload["required_observations"] = []
    elif corruption == "advisory-marked-required":
        payload["required_observations"].append({**payload["required_observations"][0], "context": "Other"})
    elif corruption == "duplicated-producer":
        payload["required_observations"].append(payload["required_observations"][0])
    elif corruption == "policyless-required-partition":
        payload["policy"]["contexts"] = []
    elif corruption == "policyless-empty-passed":
        payload["policy"]["contexts"] = []
        payload["required_observations"] = []
        payload["urls"] = []
    else:
        payload["failing_contexts"] = ["Other (app 10)"]
    artifact.write_text(json.dumps(payload))

    assert _derive_push_remote_states(tmp_path)["remote_ci"]["status"] == "partial"

@pytest.mark.parametrize("discovery_seconds", [120.0, 0.5])
def test_archive_no_ci_retains_empty_strict_policy(tmp_path: Path, discovery_seconds: float) -> None:
    """Strictness alone does not declare a required CI context."""
    artifact, payload = _current_ci_artifact(tmp_path, status="no_ci")
    limits = RemoteCILimits(discovery_seconds=discovery_seconds)
    verdict = evaluate_remote_ci(RemoteCISnapshot(
            target=RemoteCITarget(target_dir=tmp_path, **payload["target"]), binding=PRCIBinding(**payload["binding"]),
            policy=RequiredPolicy((), True), active_workflows=(), head_observations=(), merge_observations=(),
        ), elapsed=discovery_seconds, stable_polls=2, limits=limits,
    )
    assert verdict.status == "no_ci"
    write_remote_ci_verdict(artifact, verdict, session_id="current", poll_count=2, started_at="2026-09-06T12:00:00Z",
        updated_at="2026-09-06T12:02:00Z", discovery_deadline=discovery_seconds, completion_deadline=1800,
        limits=limits,
    )

    assert _derive_push_remote_states(tmp_path)["remote_ci"]["status"] == "succeeded"

@pytest.mark.parametrize(("status", "field", "value"),
    [("no_ci", "elapsed_seconds", 0), ("no_ci", "stable_polls", 1), ("passed", "stable_polls", 1),
        ("no_ci", "discovery_seconds", -1), ("no_ci", "required_stable_polls", True),
    ],
)
def test_archive_terminal_ci_requires_declared_discovery_and_stability(
    tmp_path: Path, status: str, field: str, value: object
) -> None:
    artifact, payload = _current_ci_artifact(tmp_path, status=status)
    payload["polling"][field] = value
    artifact.write_text(json.dumps(payload))
    assert _derive_push_remote_states(tmp_path)["remote_ci"]["status"] == "partial"

def test_archive_unpinned_legacy_status_uses_casefolded_context(tmp_path: Path) -> None:
    artifact, payload = _current_ci_artifact(tmp_path)
    payload["policy"]["contexts"][0]["app_id"] = None
    payload["required_observations"][0].update(source="status", app_id=None, context="BUILD", raw_state="success")
    artifact.write_text(json.dumps(payload))
    assert _derive_push_remote_states(tmp_path)["remote_ci"]["status"] == "succeeded"

@pytest.mark.parametrize(("artifact_name", "field_path", "value"),
    [("push-verdict.json", ("schema_version",), True), ("remote-ci-verdict.json", ("schema_version",), True),
        ("push-verdict.json", ("status",), []), ("remote-ci-verdict.json", ("status",), {}),
        ("remote-ci-verdict.json", ("required_observations", 0, "state"), []),
        ("remote-ci-verdict.json", ("required_observations", 0, "source"), {}),
        ("remote-ci-verdict.json", ("binding", "state"), "closed"),
    ],
)
def test_archive_malformed_current_ci_fields_fail_closed(
    tmp_path: Path, artifact_name: str, field_path: tuple[str | int, ...], value: object,
) -> None:
    _write_push_verdict(tmp_path)
    _write_remote_verdict(tmp_path)
    artifact = tmp_path / ".daydream" / "deep" / artifact_name
    payload = json.loads(artifact.read_text())
    cursor = payload
    for component in field_path[:-1]:
        cursor = cursor[component]
    cursor[field_path[-1]] = value
    artifact.write_text(json.dumps(payload))
    states = _derive_push_remote_states(tmp_path)
    assert states["push"]["status"] == ("partial" if artifact_name == "push-verdict.json" else "succeeded")
    assert states["remote_ci"]["status"] != "succeeded"

@pytest.mark.parametrize(("artifact", "expected_push"),
    [({"session_id": "current", "status": "succeeded"}, "partial"),
        ({"schema_version": 1, "session_id": "prior", "status": "succeeded"}, "absent"),
    ],
)
def test_push_artifact_is_strictly_current_session_bound(tmp_path: Path, artifact: dict[str, Any], expected_push: str
) -> None:
    _write_deep(tmp_path, "push-verdict.json", artifact)
    states = _derive_push_remote_states(tmp_path)
    assert states["push"]["status"] == expected_push
    assert states["remote_ci"] == {"ran": False, "status": "absent"}

def test_unbound_archive_session_cannot_adopt_unbound_push_artifact(tmp_path: Path,) -> None:
    _write_push_verdict(tmp_path)
    artifact = tmp_path / ".daydream" / "deep" / "push-verdict.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    del payload["session_id"]
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    states = _derive_push_remote_states(tmp_path, session_id=cast(Any, None))
    assert states["push"] == {"ran": False, "status": "absent"}
    assert states["remote_ci"] == {"ran": False, "status": "absent"}

@pytest.mark.parametrize(("replacement", "value"),
    [("not-json", None), (None, {"schema_version": 1, "session_id": "prior", "status": "passed"}),],
)
def test_successful_push_rejects_malformed_or_stale_remote_artifact(
    tmp_path: Path, replacement: str | None, value: dict[str, Any] | None
) -> None:
    _write_push_verdict(tmp_path)
    artifact = tmp_path / ".daydream" / "deep" / "remote-ci-verdict.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    if replacement is not None:
        artifact.write_text(replacement, encoding="utf-8")
    else:
        artifact.write_text(json.dumps(value), encoding="utf-8")
    assert _derive_push_remote_states(tmp_path)["remote_ci"] == {"ran": True, "status": "partial",}


def test_frozen_mapping_push_and_remote_phase_starts_are_partial_without_artifacts(
    target: Path, archive_dir: Path, make_config: MakeConfig,
) -> None:
    """Only frozen event mappings drive these phase states; the live recorder has no
    events.
    """
    recorder = make_recorder(target, run_flow=DaydreamRunFlow.NORMAL)
    phase_events = [_phase_start_event(phase.value, recorder.session_id, timestamp="2026-09-06T12:00:00Z")
        for phase in (DaydreamPhase.PUSH, DaydreamPhase.REMOTE_CI)
    ]
    assert recorder._phase_events == []

    _strict_archive(target=target, session_id=recorder.session_id,
        config=make_config(target, archive=True, pr_repo="example/project", pr_number=42,),
        write_snapshot=_write_snapshot(recorder, phase_events=phase_events),
    )

    manifest = json.loads((archive_dir / "runs" / recorder.session_id / "manifest.json").read_text())
    assert manifest["phase_states"]["push"] == {"ran": True, "status": "partial"}
    assert manifest["phase_states"]["remote_ci"] == {"ran": True, "status": "partial",}
    assert manifest["pipeline_status"] == "partial"

def _phase_start_event(
    phase: str, session_id: str, *, timestamp: str = "2026-01-01T00:00:00Z", scope_id: str | None = None,
) -> dict[str, Any]:
    return {"phase": phase, "event": "phase_start", "timestamp": timestamp, "session_id": session_id,
        "scope_id": scope_id or f"{phase}-scope",
    }

def _merge_events(session_id: str, status: str, *, scope_id: str = "merge-scope",) -> list[dict[str, Any]]:
    return [_phase_start_event("merge", session_id, scope_id=scope_id),
        {"phase": "merge", "event": "phase_end", "timestamp": "2026-01-01T00:00:01Z", "session_id": session_id,
            "scope_id": scope_id, "status": status,
        },
    ]

@pytest.mark.parametrize(("status", "artifact", "payload"),
    [pytest.param("succeeded", "review-coverage.json", None,
            id="success-despite-stale-failure",
        ), pytest.param("failed", "merged-items.json", {"items": [{"id": 1}]}, id="failure-despite-stale-success"),
        pytest.param("partial", None, None, id="partial"),
    ],
)
def test_current_merge_event_controls_pipeline_status(tmp_path: Path, status: str, artifact: str | None, payload: Any,
) -> None:
    if artifact == "review-coverage.json":
        _write_review_coverage(tmp_path)
    elif artifact is not None:
        _write_deep(tmp_path, artifact, payload)
    states = pipeline.derive_phase_states(
        tmp_path, phase_events=_merge_events("current", status), runs_merge=True, runs_fix=False, runs_test=False,
        session_id="current",
    )
    assert states["merge"] == {"ran": True, "status": status}
    assert pipeline.derive_pipeline_status("complete", None, states, runs_merge=True) == status

def test_stale_merge_event_failure_does_not_override_current_success(tmp_path: Path,) -> None:
    states = pipeline.derive_phase_states(tmp_path,
        phase_events=[*_merge_events("prior", "failed", scope_id="prior-merge"), *_merge_events("current", "succeeded"),
        ], runs_merge=True, runs_fix=False, runs_test=False, session_id="current",
    )
    assert states["merge"] == {"ran": True, "status": "succeeded"}

@pytest.mark.parametrize("events",
    [[_merge_events("current", "succeeded")[1]],
        [*_merge_events("current", "succeeded"), _merge_events("current", "succeeded")[1]],
        [{**_merge_events("current", "succeeded")[0], "timestamp": "2026-01-01T00:00:02Z"},
            _merge_events("current", "succeeded")[1],
        ],
        [{**_merge_events("current", "succeeded")[0], "timestamp": "2026-01-01T00:00:00"},
            _merge_events("current", "succeeded")[1],
        ],
        [{**_merge_events("current", "succeeded")[0], "session_id": None}, _merge_events("current", "succeeded")[1],],
        [_merge_events("current", "succeeded")[0],
            {**_merge_events("current", "succeeded")[1], "status": "not-a-status"},
        ],
    ], ids=("orphan", "duplicate", "reversed", "incomparable-timestamps", "missing-session", "invalid-terminal",),
)
def test_malformed_current_merge_event_is_unknown(tmp_path: Path, events: list[dict[str, Any]],) -> None:
    states = pipeline.derive_phase_states(
        tmp_path, phase_events=events, runs_merge=True, runs_fix=False, runs_test=False, session_id="current",
    )
    assert states["merge"] == {"ran": True, "status": "unknown"}
    assert pipeline.derive_pipeline_status("complete", None, states, runs_merge=True) == "unknown"

@pytest.mark.parametrize("malformed_kind", [[], {}], ids=["list", "mapping"])
def test_current_merge_event_rejects_non_scalar_kind(tmp_path: Path, malformed_kind: Any,) -> None:
    events = _merge_events("current", "succeeded")
    events[1]["event"] = malformed_kind
    states = derive_phase_states(
        tmp_path, phase_events=events, session_id="current", runs_merge=True, runs_fix=False, runs_test=False,
    )
    assert states["merge"] == {"ran": True, "status": "unknown"}
    assert derive_pipeline_status("complete", None, states, runs_merge=True) == "unknown"

def test_missing_current_merge_event_never_uses_stale_success_artifact(tmp_path: Path,) -> None:
    _write_deep(tmp_path, "merged-items.json", {"items": [{"id": 1}]})
    states = pipeline.derive_phase_states(
        tmp_path, phase_events=_merge_events("prior", "succeeded"), runs_merge=True, runs_fix=False, runs_test=False,
        session_id="current",
    )
    assert states["merge"] == {"ran": False, "status": "absent"}
    assert pipeline.derive_pipeline_status("complete", None, states, runs_merge=True) == "partial"

@pytest.mark.parametrize(("coverage_state", "items_payload"), [
    ("complete", {"items": []}), ("failed", {"items": []}),
    ("malformed", {"items": []}), ("complete", {"items": "corrupt"}),
], ids=("success", "failure", "malformed-coverage", "malformed-items"))
def test_merge_artifacts_cannot_supply_missing_run_identity(
    tmp_path: Path, coverage_state: str, items_payload: Any,
) -> None:
    if coverage_state == "malformed":
        _write_deep(tmp_path, "review-coverage.json", {"phases": "corrupt"})
    else:
        _write_review_coverage(tmp_path, state=coverage_state)
    _write_deep(tmp_path, "merged-items.json", items_payload)
    states = pipeline.derive_phase_states(
        tmp_path, phase_events=[], runs_merge=True, runs_fix=False, runs_test=False, session_id=None,
    )
    assert states["merge"] == {"ran": True, "status": "unknown"}

@pytest.mark.parametrize("artifact_name", ["review-coverage.json", "merged-items.json"])
def test_merge_without_identity_ignores_invalid_utf8_artifacts(tmp_path: Path, artifact_name: str) -> None:
    deep = tmp_path / ".daydream" / "deep"
    deep.mkdir(parents=True)
    (deep / artifact_name).write_bytes(b"\xff")
    states = pipeline.derive_phase_states(
        tmp_path, phase_events=[], runs_merge=True, runs_fix=False, runs_test=False, session_id=None,
    )
    assert states["merge"] == {"ran": True, "status": "unknown"}

def test_current_archive_survives_invalid_utf8_fix_failures(target: Path, archive_dir: Path, make_config: MakeConfig,
) -> None:
    session_id = "current-corrupt-fix-sidecar"
    recorder = _MockRecorder(session_id=session_id)
    deep = target / ".daydream" / "deep"
    deep.mkdir(parents=True)
    (deep / "fix-failures.json").write_bytes(b"\xff")
    _write_deep(target, "merged-items.json", {"items": []})
    _write_deep(target, "test-verdict.json", {"session_id": session_id, "passed": True},)
    phase_events = [*_merge_events(session_id, "succeeded"), _phase_start_event("fix", session_id),]

    _strict_archive(target=target, session_id=session_id, config=make_config(target, archive=True, run_eval=True),
        write_snapshot=_write_snapshot(recorder, phase_events=phase_events),
    )

    run_dir = archive_dir / "runs" / session_id
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert (run_dir / "evaluation.json").is_file()
    assert manifest["phase_states"] == {
        "merge": {"ran": True, "status": "succeeded"}, "fix": {"ran": True, "status": "succeeded"},
        "test": {"ran": True, "status": "succeeded"}, "push": {"ran": False, "status": "absent"},
        "remote_ci": {"ran": False, "status": "absent"},
    }
    assert manifest["pipeline_status"] == "succeeded"
    with closing(sqlite3.connect(f"{(archive_dir / 'index.db').as_uri()}?mode=ro", uri=True)) as conn:
        assert conn.execute("SELECT session_id FROM runs").fetchall() == [(session_id,)]

def test_start_at_fix_archive_does_not_require_or_inherit_merge(
    target: Path, archive_dir: Path, make_config: MakeConfig,
) -> None:
    session_id = "fix-resume-session"
    recorder = _MockRecorder(session_id=session_id)
    _write_deep(target, "merged-items.json", {"items": [{"id": 1}]})
    _write_review_coverage(target)
    _write_deep(target, "test-verdict.json", {"session_id": session_id, "passed": True})
    fix_start = _phase_start_event("fix", session_id)

    _strict_archive(target=target, session_id=session_id, config=make_config(target, archive=True, start_at="fix"),
        write_snapshot=_write_snapshot(recorder, phase_events=[fix_start]),
        identity=_manifest_identity(phases=replace(_manifest_identity().phases, merge=False)),
    )

    manifest = json.loads((archive_dir / "runs" / session_id / "manifest.json").read_text())
    assert manifest["phase_states"]["merge"] == {"ran": False, "status": "absent"}
    assert manifest["phase_states"]["fix"] == {"ran": True, "status": "succeeded"}
    assert manifest["phase_states"]["test"] == {"ran": True, "status": "succeeded"}
    assert manifest["phase_states"]["push"] == {"ran": False, "status": "absent"}
    assert manifest["phase_states"]["remote_ci"] == {"ran": False, "status": "absent"}
    assert manifest["pipeline_status"] == "succeeded"

def test_archive_retains_malformed_frozen_merge_evidence_as_unknown(
    target: Path, archive_dir: Path, make_config: MakeConfig,
) -> None:
    session_id = "malformed-merge-session"
    recorder = _MockRecorder(session_id=session_id)
    _write_deep(target, "merged-items.json", {"items": [{"id": 1}]})
    events = _merge_events(session_id, "not-a-status")

    _strict_archive(target=target, session_id=session_id, config=make_config(target, archive=True),
        write_snapshot=_write_snapshot(recorder, phase_events=events),
    )

    manifest = json.loads((archive_dir / "runs" / session_id / "manifest.json").read_text())
    assert manifest["phase_states"]["merge"] == {"ran": True, "status": "unknown"}
    assert manifest["pipeline_status"] == "partial"

def test_archive_rejects_frozen_root_from_another_session(target: Path, archive_dir: Path, make_config: MakeConfig,
) -> None:
    recorder = _MockRecorder(session_id="current-session")
    payload = {"session_id": "other-session", "trajectory_id": recorder.session_id, "steps": [], "extra": {},
        "final_metrics": {},
    }
    snapshot = RunWriteSnapshot(
        status="complete", cutoff_at="2026-01-01T00:00:01Z", root_trajectory_id=recorder.session_id,
        documents=(TrajectoryDocumentSnapshot(
                trajectory_id=recorder.session_id, path=recorder.path, json_bytes=json.dumps(payload).encode(),
            ),
        ),
    )

    with pytest.raises(ValueError, match="frozen root trajectory identity"):
        _strict_archive(target=target, session_id=recorder.session_id, config=make_config(target, archive=True),
            write_snapshot=snapshot,
        )
    assert not (archive_dir / "runs" / recorder.session_id).exists()

def test_archive_rejects_a_sibling_document_from_another_session(
    target: Path, archive_dir: Path, make_config: MakeConfig,
) -> None:
    """The root is valid: bundle projection must reject the foreign fork without publishing a partial archive."""
    recorder = _MockRecorder(session_id="current-session")
    root = _write_snapshot(recorder).documents[0]
    foreign = json.dumps({"session_id": "other-session", "trajectory_id": "fork-1"}).encode()
    snapshot = RunWriteSnapshot(
        status="complete", cutoff_at="2026-01-01T00:00:01Z", root_trajectory_id=recorder.session_id,
        documents=(root, TrajectoryDocumentSnapshot("fork-1", target / "fork.json", foreign)),
    )

    with pytest.raises(ArchiveFinalizationError, match="frozen trajectory projection failed"):
        _strict_archive(target=target, session_id=recorder.session_id, config=make_config(target, archive=True),
            write_snapshot=snapshot,
        )
    assert not (archive_dir / "runs" / recorder.session_id).exists()
    assert not (archive_dir / "index.db").exists()

def test_project_documents_destinations_are_the_layout_surface(tmp_path: Path) -> None:
    """The bundle's destination names come from the owner; `.partial` is stripped there."""
    session_id = "layout-session"
    run_dir = run_directory(tmp_path / "archive", session_id)
    run_dir.mkdir(parents=True)
    root_bytes = json.dumps({"session_id": session_id, "trajectory_id": session_id}).encode()
    sibling_bytes = json.dumps({"session_id": session_id, "trajectory_id": "fork-1"}).encode()
    root_partial = partial_document_path(run_document_path(run_dir))
    snapshot = RunWriteSnapshot(status="partial", cutoff_at="2026-01-01T00:00:00Z", root_trajectory_id=session_id,
        documents=(TrajectoryDocumentSnapshot(session_id, root_partial, root_bytes),
            TrajectoryDocumentSnapshot("fork-1", sibling_document_path(run_dir, "deep-python.json"), sibling_bytes),
        ),
    )
    _project_documents(snapshot, run_dir, session_id=session_id)

    assert run_document_path(run_dir).read_bytes() == root_bytes
    assert sibling_document_path(run_dir, "deep-python.json").read_bytes() == sibling_bytes

def test_merge_failed_discriminates_on_current_event_not_merged_items(tmp_path: Path,) -> None:
    _write_deep(tmp_path, "merged-items.json", {"items": []})
    _write_review_coverage(tmp_path)
    states = pipeline.derive_phase_states(tmp_path, phase_events=_merge_events("current", "failed"),
                                           session_id="current")
    assert states["merge"]["ran"] is True
    assert states["merge"]["status"] == "failed"   # merged-items present is NOT sufficient


def test_test_failed_from_verdict(tmp_path: Path) -> None:
    _write_deep(
        tmp_path, "test-verdict.json", {"session_id": "current", "passed": False, "retries": 1, "ignored": False},
    )
    states = pipeline.derive_phase_states(tmp_path, phase_events=[], session_id="current")
    assert states["test"]["ran"] is True
    assert states["test"]["status"] == "failed"

def test_session_bound_start_at_fix_rejects_prior_green_test_verdict(tmp_path: Path,) -> None:
    _write_deep(tmp_path, "test-verdict.json", {"session_id": "prior", "passed": True})
    states = pipeline.derive_phase_states(tmp_path, phase_events=[], session_id="current")
    assert states["test"] == {"ran": False, "status": "absent"}
    assert pipeline.derive_pipeline_status("complete", None, states, runs_fix=True, runs_test=True) == "partial"

def test_matching_stabilization_failure_overrides_green_test_pipeline(tmp_path: Path,) -> None:
    _write_deep(tmp_path, "test-verdict.json", {"session_id": "current", "passed": True})
    _write_deep(
        tmp_path, "stabilization-failed.json", {"session_id": "current", "reason": "final verifier remains actionable"},
    )
    states = pipeline.derive_phase_states(tmp_path, phase_events=[], session_id="current")
    assert states["fix"] == {"ran": True, "status": "failed"}
    assert states["test"] == {"ran": True, "status": "failed"}
    assert pipeline.derive_pipeline_status("complete", None, states, runs_fix=True, runs_test=True) == "failed"

@pytest.mark.parametrize("payload",
    [{"session_id": "prior", "reason": "stale"}, {"session_id": "current"}, {"session_id": "current", "reason": ""},
        ["malformed"],
    ],
)
def test_stale_or_malformed_stabilization_failure_is_neutral(tmp_path: Path, payload: Any) -> None:
    _write_deep(tmp_path, "test-verdict.json", {"session_id": "current", "passed": True})
    _write_deep(tmp_path, "stabilization-failed.json", payload)
    states = pipeline.derive_phase_states(tmp_path, phase_events=[], session_id="current")
    assert states["fix"] == {"ran": False, "status": "absent"}
    assert states["test"] == {"ran": True, "status": "succeeded"}

def test_archive_manifest_fails_matching_stabilization_session(
    target: Path, archive_dir: Path, make_config: MakeConfig
) -> None:
    session_id = "stabilization-session"
    _write_deep(target, "merged-items.json", {"items": [{"id": 1}]})
    _write_deep(target, "test-verdict.json", {"session_id": session_id, "passed": True})
    _write_deep(
        target, "stabilization-failed.json", {"session_id": session_id, "reason": "post-test tree did not stabilize"},
    )
    recorder = _MockRecorder(session_id=session_id)

    _strict_archive(target=target, session_id=session_id, config=make_config(target, archive=True),
        write_snapshot=_write_snapshot(recorder),
    )

    manifest = json.loads((archive_dir / "runs" / session_id / "manifest.json").read_text())
    assert manifest["archive_status"] == "complete"
    assert manifest["pipeline_status"] == "failed"
    assert manifest["phase_states"]["fix"] == {"ran": True, "status": "failed"}
    assert manifest["phase_states"]["test"] == {"ran": True, "status": "failed"}




def test_non_deep_flow_ignores_stale_deep_artifacts(tmp_path: Path) -> None:
    # Issue #336: derive_phase_states is flow-aware. A prior deep run left
    # session-agnostic merge/fix/test artifacts in target_dir/.daydream/deep;
    # a non-deep flow run afterwards must NOT inherit them as its own pipeline
    # state -- the phases it does not run read absent regardless of disk.
    _write_deep(tmp_path, "merged-items.json", {"items": []})
    _write_review_coverage(tmp_path)
    _write_deep(tmp_path, "test-verdict.json", {"passed": False})
    _write_deep(tmp_path, "fix-failures.json", {"src/a.py": "reverted"})
    # TTT review runs the merge spine but never the fix/test cycle.
    states = pipeline.derive_phase_states(tmp_path, phase_events=_merge_events("current", "failed"),
                                           session_id="current", runs_merge=True, runs_fix=False, runs_test=False)
    assert states["merge"]["status"] == "failed"   # merge ran (spine wrote fresh artifacts)
    assert states["fix"] == {"ran": False, "status": "absent"}    # stale fix ignored
    assert states["test"] == {"ran": False, "status": "absent"}   # stale test ignored
    # An improve-only flow runs none of the deep phases at all.
    states = pipeline.derive_phase_states(tmp_path, phase_events=[], runs_merge=False, runs_fix=False, runs_test=False)
    assert all(s == {"ran": False, "status": "absent"} for s in states.values())







def test_snapshot_manifest_pr_metadata_is_immutable_after_live_inputs_mutate(tmp_path: Path,) -> None:
    session_id = "frozen-pr-session"
    snapshot = _manifest_write_snapshot(session_id=session_id, extra={"pr_number": 7, "pr_repo": "Owner/Repo"},)
    payload = json.loads(snapshot.documents[0].json_bytes)
    payload["extra"] = {"pr_number": 7, "pr_repo": "Owner/Repo"}
    document = TrajectoryDocumentSnapshot(
        trajectory_id=session_id, path=snapshot.documents[0].path, json_bytes=json.dumps(payload).encode(),
    )
    snapshot = RunWriteSnapshot(
        status="complete", cutoff_at=snapshot.cutoff_at, root_trajectory_id=session_id, documents=(document,),
    )
    provenance = archive_recorder_provenance_from_snapshot(write_snapshot=snapshot, run_flow=DaydreamRunFlow.NORMAL,)
    live_recorder = _MockRecorder(pr_number=7, pr_repo="Owner/Repo")
    live_config = RunConfig(target=str(tmp_path), pr_number=7, pr_repo="Owner/Repo")
    live_recorder.pr_number = 99
    live_recorder.pr_repo = "mutated/repo"
    live_config.pr_number = 100
    live_config.pr_repo = "also/mutated"
    manifest = build_manifest_from_snapshot(
        run=ArchiveRunSnapshot(recorder_provenance=provenance, identity=_manifest_identity(), trajectories=snapshot,),
        git_ctx=GitContext(), status="complete", archive_path=tmp_path,
    )

    assert manifest.session_id == snapshot.root_trajectory_id
    assert manifest.pr_number == 7
    assert manifest.pr_repo == "Owner/Repo"
    assert manifest.to_dict()["pr"] == {"number": 7, "repo": "Owner/Repo"}

@pytest.mark.parametrize(("extra", "message"),
    [({"pr_number": True}, "pr_number"), ({"pr_number": 7, "pr_repo": None}, "pr_repo"),
        ({"pr_number": 7, "pr_repo": ""}, "pr_repo"),
    ],
)
def test_snapshot_manifest_provenance_rejects_malformed_present_pr_metadata(
    tmp_path: Path, extra: dict[str, Any], message: str,
) -> None:
    snapshot = _manifest_write_snapshot(
        session_id="session", extra=extra, final_metrics={}, path=tmp_path / "trajectory.json",
    )
    with pytest.raises(ValueError, match=message):
        archive_recorder_provenance_from_snapshot(write_snapshot=snapshot, run_flow=DaydreamRunFlow.NORMAL,)

@pytest.mark.parametrize("session_id", [".", "..", "../escape", "bad\\path", "bad\0id"])
def test_snapshot_manifest_provenance_rejects_unsafe_session_identity(tmp_path: Path, session_id: str,) -> None:
    snapshot = _manifest_write_snapshot(session_id=session_id, final_metrics={}, path=tmp_path / "root.json",)
    with pytest.raises(ValueError, match="session_id"):
        archive_recorder_provenance_from_snapshot(write_snapshot=snapshot, run_flow=DaydreamRunFlow.NORMAL,)

def _finalizer_arguments(tmp_path: Path, session_id: str, *, config: RunConfig,
) -> dict[str, Any]:
    """Freeze real trajectory bytes and bind the finalizer's independent roots."""
    frozen = tmp_path / "frozen"
    path = frozen / ".daydream" / "runs" / session_id / "trajectory.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = _manifest_write_snapshot(session_id=session_id, final_metrics={}, path=path)
    path.write_bytes(snapshot.documents[0].json_bytes)
    return dict(run=_archive_snapshot(snapshot),
        artifacts=ArtifactTreeSnapshot(session_id, "workspace", frozen, manifest_tree(frozen), ()),
        artifact_provenance=ArtifactEvidenceProvenance("workspace", session_id, tmp_path / "source", tmp_path / "live",
        ), config=config, work=None,
    )

def test_strict_archive_evaluation_failure_is_typed_and_never_reports_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The host finalizer cannot infer success from the legacy fail-open wrapper."""
    session_id = "strict-session"
    arguments = _finalizer_arguments(tmp_path, session_id, config=RunConfig(target=str(tmp_path), run_eval=True),)
    monkeypatch.setattr("daydream.eval.analyzer.analyze_session",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("eval failed")),
    )
    with pytest.raises(ArchiveFinalizationError, match="evaluation"):
        finalize_archive_run(**arguments)
    assert not (get_archive_dir() / "runs" / session_id).exists()

@pytest.mark.parametrize("mutate", [False, True])
def test_strict_archive_rejects_frozen_receipt_changed_by_evaluator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutate: bool,
) -> None:
    """A code-running consumer cannot make archive bytes and manifest disagree."""
    session_id = "strict-mutated-evidence"
    frozen = tmp_path / "frozen"
    receipt = frozen / ".daydream" / "deep" / "test-verdict.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({"session_id": session_id, "passed": True}), encoding="utf-8",)
    arguments = _finalizer_arguments(tmp_path, session_id, config=RunConfig(target=str(tmp_path), run_eval=True),)
    public_source = tmp_path / "source"
    public_source.mkdir()

    def mutate_frozen_receipt(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        if mutate:
            receipt.write_text(json.dumps({"session_id": session_id, "passed": False}), encoding="utf-8",)
        return {"quality": {"scoped_files": 0}}

    monkeypatch.setattr("daydream.eval.analyzer.analyze_session", mutate_frozen_receipt,)

    if mutate:
        with pytest.raises(ArchiveFinalizationError, match="frozen artifact tree changed"):
            finalize_archive_run(**arguments)
    else:
        finalize_archive_run(**arguments)

    archive_dir = get_archive_dir()
    if mutate:
        assert not (archive_dir / "runs" / session_id).exists()
        assert not (archive_dir / "index.db").exists()
    else:
        assert (archive_dir / "runs" / session_id / "manifest.json").is_file()
        with closing(sqlite3.connect(f"{(archive_dir / 'index.db').as_uri()}?mode=ro", uri=True)) as conn:
            assert conn.execute("SELECT session_id FROM runs").fetchall() == [(session_id,)]

def test_strict_archive_does_not_publish_historical_bundles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the new JSONL lifecycle may select a remote destination."""
    session_id = "strict-local-only"
    arguments = _finalizer_arguments(
        tmp_path, session_id, config=RunConfig(target=str(tmp_path), run_eval=False, archive=True,
                                               trajectory_hub_repo="private/repo"),
    )

    from tests.harness.dataset_hub import FakeDatasetHub

    hub = FakeDatasetHub()
    monkeypatch.setattr("daydream.dataset_hub.HfDatasetHub", lambda: hub)
    finalize_archive_run(**arguments)
    archive_dir = get_archive_dir()
    assert (archive_dir / "runs" / session_id / "manifest.json").is_file()
    with closing(sqlite3.connect(f"{(archive_dir / 'index.db').as_uri()}?mode=ro", uri=True)) as conn:
        assert conn.execute("SELECT session_id FROM runs").fetchall() == [(session_id,)]
    assert not list(archive_dir.glob("runs/.*.finalizing"))
    assert hub.commits == []


def test_strict_archive_refuses_frozen_tree_mutated_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = "strict-mutated-publication"
    arguments = _finalizer_arguments(
        tmp_path, session_id, config=RunConfig(target=str(tmp_path), run_eval=False, archive=True),
    )
    document = arguments["run"].trajectories.documents[0]
    from daydream.archive.provenance import capture_executable_provenance

    def mutate_then_capture() -> Any:
        document.path.write_bytes(document.json_bytes + b"\n")
        return capture_executable_provenance()

    monkeypatch.setattr("daydream.archive.provenance.capture_executable_provenance", mutate_then_capture)
    with pytest.raises(ArchiveFinalizationError, match="frozen artifact tree changed"):
        finalize_archive_run(**arguments)
    assert not (get_archive_dir() / "runs" / session_id).exists()
