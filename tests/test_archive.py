"""Unit tests for the daydream.archive package.

Covers git_context, manifest, index, and the strict ``finalize_archive_run`` flow.
"""

import json
import sqlite3
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import pytest

from daydream.archive import (
    _read_fix_quality_gate,
    get_archive_dir,
)
from daydream.archive.git_context import GitContext, capture_git_context
from daydream.archive.index import (
    append_label_observation,
    bulk_latest_label_observations,
    canonical_utc_iso,
    delete_runs,
    label_count_summary,
    label_observation_history,
    latest_label_observation,
    normalize_as_of,
    query_runs,
    reviewer_set_penalty_prior,
    set_run_pr_link,
    update_labels,
    upsert_run,
)
from daydream.archive.manifest import (
    Manifest,
    archive_recorder_provenance_from_snapshot,
    build_manifest_from_snapshot,
)
from daydream.artifact_visibility import (
    ArtifactEvidenceProvenance,
    ArtifactTreeSnapshot,
    DestinationDelivery,
    OutputLabel,
    RoutedDestination,
    _manifest,
)
from daydream.remote_ci import (
    CIObservation,
    PRCIBinding,
    RemoteCITarget,
    RemoteCIVerdict,
    RequiredContext,
    RequiredPolicy,
    write_remote_ci_verdict,
)
from daydream.run_snapshot import (
    ArchiveRunSnapshot,
    ManifestRunIdentity,
    RunPhaseCapabilities,
    RunProfileIdentity,
)
from daydream.runner import RunConfig
from daydream.trajectory import (
    DaydreamPhase,
    DaydreamRunFlow,
    PhaseEvent,
    RunWriteSnapshot,
    TrajectoryDocumentSnapshot,
    TrajectoryRecorder,
)
from tests.harness.trajectory import make_manifest

MakeConfig = Callable[..., RunConfig]
InstallBackend = Callable[[object], object]


_DEFAULT_FINAL_METRICS: dict[str, Any] = {
    "total_prompt_tokens": 100,
    "total_completion_tokens": 50,
    "total_cached_tokens": 20,
    "total_cost_usd": 0.05,
}


def _write_snapshot(
    recorder: Any,
    *,
    status: str = "complete",
    phase_events: list[dict[str, Any]] | None = None,
    final_metrics: dict[str, Any] | None = None,
    lifecycle: tuple[str, str] | None = None,
) -> RunWriteSnapshot:
    """Freeze one root trajectory document the way the recorder's writer does.

    ``lifecycle`` stamps the run-span keys the timing reducer needs (and pins the
    snapshot cutoff to the end stamp, which ``compute_timing_summary`` requires
    for a complete write); ``final_metrics`` overrides the whole-run totals the
    manifest projects.
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
        trajectory_id=trajectory_id,
        path=path,
        json_bytes=json.dumps(payload).encode(),
    )
    return RunWriteSnapshot(
        status=cast(Any, status),
        cutoff_at=cutoff_at,
        root_trajectory_id=trajectory_id,
        documents=(document,),
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


def _findings_route(live_root: Path, name: str = "findings.json") -> RoutedDestination:
    """The registered ``--findings-out`` route the strict bundle relocates."""
    private = live_root / ".explicit" / "0000" / name
    return RoutedDestination(
        label=OutputLabel.FINDINGS_OUTPUT,
        requested=Path("findings") / name,
        write_path=private,
        frozen_path=private,
        delivery=DestinationDelivery.DEFERRED,
    )


def _strict_archive(
    *,
    target: Path,
    session_id: str,
    config: Any,
    write_snapshot: RunWriteSnapshot,
    run_flow: DaydreamRunFlow = DaydreamRunFlow.NORMAL,
    identity: ManifestRunIdentity | None = None,
    destinations: tuple[RoutedDestination, ...] = (),
    work: Any = None,
    upload: bool = False,
    dump_path: Path | None = None,
) -> None:
    """Archive one frozen run tree through the production strict finalizer.

    ``target`` is the frozen artifact root, so it must not contain the archive
    directory itself — ``finalize_archive_run`` re-attests the tree after every
    stage and a write inside it would (correctly) be read as tampering.
    """
    from daydream.archive import finalize_archive_run

    finalize_archive_run(
        run=_archive_snapshot(write_snapshot, run_flow=run_flow, identity=identity),
        artifacts=ArtifactTreeSnapshot(
            session_id=session_id,
            workspace_key="workspace",
            root=target,
            manifest=_manifest(target),
            destinations=destinations,
        ),
        artifact_provenance=ArtifactEvidenceProvenance(
            workspace_key="workspace",
            session_id=session_id,
            public_source=target,
            # The frozen root is a copy of the live root, so route paths relative
            # to one resolve unchanged inside the other.
            live_root=target,
        ),
        config=config,
        work=work,
        upload=upload,
        dump_path=dump_path,
    )


def _manifest_identity(**overrides: Any) -> ManifestRunIdentity:
    """Build the public, already-resolved identity supplied by the runner."""
    identity = ManifestRunIdentity(
        flow_name=None,
        skill="python",
        model=None,
        backend="claude",
        review_backend=None,
        fix_backend="claude",
        test_backend="claude",
        per_stack_review_backend="claude",
        per_stack_review_model="sonnet",
        review_only=False,
        deep=True,
        profile=None,
        phases=RunPhaseCapabilities(
            per_stack_review=True,
            merge=True,
            fix=True,
            test=True,
            push=True,
            remote_ci=True,
        ),
    )
    return replace(identity, **overrides)


def _manifest_write_snapshot(
    *,
    session_id: str = "abcd1234-0000-0000-0000-000000000000",
    final_metrics: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> RunWriteSnapshot:
    """Build explicit immutable root bytes for manifest-reducer tests."""
    snapshot_extra = dict(extra or {})
    payload = {
        "session_id": session_id,
        "trajectory_id": session_id,
        "steps": [],
        "final_metrics": dict(_DEFAULT_FINAL_METRICS if final_metrics is None else final_metrics),
        "extra": snapshot_extra,
    }
    return RunWriteSnapshot(
        status="complete",
        cutoff_at=str(snapshot_extra.get("run_ended_at", "2026-01-01T00:00:01Z")),
        root_trajectory_id=session_id,
        documents=(
            TrajectoryDocumentSnapshot(
                trajectory_id=session_id,
                path=Path("/frozen/trajectory.json"),
                json_bytes=json.dumps(payload).encode(),
            ),
        ),
    )


def _archive_snapshot(
    trajectories: RunWriteSnapshot,
    *,
    run_flow: DaydreamRunFlow = DaydreamRunFlow.NORMAL,
    identity: ManifestRunIdentity | None = None,
) -> ArchiveRunSnapshot:
    """Join frozen trajectory provenance with the runner's public identity."""
    return ArchiveRunSnapshot(
        recorder_provenance=archive_recorder_provenance_from_snapshot(
            write_snapshot=trajectories,
            run_flow=run_flow,
        ),
        identity=identity or _manifest_identity(),
        trajectories=trajectories,
    )


def _run_git_init_with_credential_origin(repo: Path, origin_url: str) -> None:
    """git init + one commit + credential-bearing origin remote."""
    for argv in (
        ["git", "init"],
        ["git", "config", "user.email", "test@test.com"],
        ["git", "config", "user.name", "Test"],
        ["git", "commit", "--allow-empty", "-m", "init"],
        ["git", "remote", "add", "origin", origin_url],
    ):
        subprocess.run(argv, cwd=repo, capture_output=True, check=True)  # noqa: S603, S607 - arguments are not user-controlled


def test_capture_git_context_stores_credential_free_remote(tmp_path: Path) -> None:
    # Real git repo whose origin carries credentials (M3, real-path test).
    repo = tmp_path / "repo"
    repo.mkdir()
    _run_git_init_with_credential_origin(repo, "https://user:ghp_canaryfake123@github.com/o/r.git")
    ctx = capture_git_context(repo)
    assert ctx.repo_slug == "o/r"
    assert ctx.remote_url == "https://github.com/o/r"
    assert "ghp_canaryfake123" not in (ctx.remote_url or "")
    assert "@" not in (ctx.remote_url or "")


def test_capture_git_context_real_repo(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True, check=True)  # noqa: S603, S607 - arguments are not user-controlled
    subprocess.run(  # noqa: S603, S607 - arguments are not user-controlled
        ["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True, check=True,
    )
    subprocess.run(  # noqa: S603, S607 - arguments are not user-controlled
        ["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True, check=True,
    )
    subprocess.run(  # noqa: S603, S607 - arguments are not user-controlled
        ["git", "commit", "--allow-empty", "-m", "init"], cwd=tmp_path, capture_output=True, check=True,
    )

    ctx = capture_git_context(tmp_path)
    assert isinstance(ctx, GitContext)
    assert ctx.head_sha is not None and len(ctx.head_sha) == 40
    assert ctx.branch is not None


def test_capture_git_context_no_repo(tmp_path: Path) -> None:
    ctx = capture_git_context(tmp_path)
    assert ctx.head_sha is None
    assert ctx.remote_url is None
    assert ctx.branch is None
    assert ctx.base_sha is None
    assert ctx.changed_files == []


def test_capture_git_context_populates_base_sha_and_changed_files(
    tmp_path: Path,
) -> None:
    """Real repo with a feature branch surfaces merge-base SHA + diff paths."""
    subprocess.run(["git", "init", "-b", "main"], cwd=tmp_path, capture_output=True, check=True)  # noqa: S603, S607 - arguments are not user-controlled
    subprocess.run(  # noqa: S603, S607 - arguments are not user-controlled
        ["git", "config", "user.email", "test@test.com"], cwd=tmp_path, capture_output=True, check=True,
    )
    subprocess.run(  # noqa: S603, S607 - arguments are not user-controlled
        ["git", "config", "user.name", "Test"], cwd=tmp_path, capture_output=True, check=True,
    )
    (tmp_path / "a.py").write_text("print('a')\n")
    subprocess.run(["git", "add", "a.py"], cwd=tmp_path, capture_output=True, check=True)  # noqa: S603, S607 - arguments are not user-controlled
    subprocess.run(  # noqa: S603, S607 - arguments are not user-controlled
        ["git", "commit", "-m", "base"], cwd=tmp_path, capture_output=True, check=True,
    )
    base_sha = subprocess.run(  # noqa: S603, S607 - arguments are not user-controlled
        ["git", "rev-parse", "HEAD"], cwd=tmp_path, capture_output=True, check=True, text=True,
    ).stdout.strip()

    subprocess.run(  # noqa: S603, S607 - arguments are not user-controlled
        ["git", "checkout", "-b", "feat/x"], cwd=tmp_path, capture_output=True, check=True,
    )
    (tmp_path / "b.py").write_text("print('b')\n")
    (tmp_path / "a.py").write_text("print('a-changed')\n")
    subprocess.run(["git", "add", "a.py", "b.py"], cwd=tmp_path, capture_output=True, check=True)  # noqa: S603, S607 - arguments are not user-controlled
    subprocess.run(  # noqa: S603, S607 - arguments are not user-controlled
        ["git", "commit", "-m", "feat"], cwd=tmp_path, capture_output=True, check=True,
    )

    ctx = capture_git_context(tmp_path)
    assert ctx.base_sha == base_sha
    assert sorted(ctx.changed_files) == ["a.py", "b.py"]


def _build(
    tmp_path: Path,
    *,
    git_ctx: GitContext | None = None,
    write_snapshot: RunWriteSnapshot | None = None,
    run_flow: DaydreamRunFlow = DaydreamRunFlow.NORMAL,
    identity: ManifestRunIdentity | None = None,
    **kw: Any,
) -> Manifest:
    """Build a manifest from immutable public archive inputs."""
    snapshot = write_snapshot or _manifest_write_snapshot()
    return build_manifest_from_snapshot(
        run=_archive_snapshot(
            snapshot,
            run_flow=run_flow,
            identity=identity,
        ),
        git_ctx=git_ctx if git_ctx is not None else GitContext(),
        status="complete",
        archive_path=tmp_path,
        **kw,
    )


def test_build_manifest_basic(tmp_path: Path) -> None:
    m = _build(
        tmp_path,
        git_ctx=GitContext(
            remote_url="git@github.com:org/repo.git",
            repo_slug="org/repo",
            branch="main",
            base_branch="main",
            head_sha="a" * 40,
        ),
    )

    assert m.session_id == "abcd1234-0000-0000-0000-000000000000"
    assert m.run_flow == "normal"
    assert m.skill == "python"
    # Per-phase models replaced config.model; the manifest stamps model as None.
    assert m.model is None
    assert m.backend == "claude"
    assert m.review_backend is None
    assert m.total_cost_usd == 0.05
    assert m.total_prompt_tokens == 100
    assert m.total_completion_tokens == 50
    assert m.total_cached_tokens == 20
    assert m.repo_slug == "org/repo"
    assert m.head_sha == "a" * 40


def test_build_manifest_serializes_resolved_profile_identity(tmp_path: Path) -> None:
    """Resolved profile provenance is preserved in the manifest projection."""
    profile = RunProfileIdentity(
        schema_version=7,
        name="focused",
        source_kind="explicit",
        digest="e2e-profile-digest",
    )

    manifest = _build(
        tmp_path,
        identity=_manifest_identity(profile=profile),
    ).to_dict()

    assert {
        key: manifest[key]
        for key in (
            "profile_schema_version",
            "profile_name",
            "profile_source_kind",
            "profile_digest",
        )
    } == {
        "profile_schema_version": 7,
        "profile_name": "focused",
        "profile_source_kind": "explicit",
        "profile_digest": "e2e-profile-digest",
    }


def test_build_manifest_omits_unresolved_profile_identity(tmp_path: Path) -> None:
    """A direct legacy caller with no resolved profile keeps all four keys absent."""
    manifest = _build(tmp_path).to_dict()

    assert {
        "profile_schema_version",
        "profile_name",
        "profile_source_kind",
        "profile_digest",
    }.isdisjoint(manifest)




def test_build_manifest_omits_fix_metadata_for_diagram_flow(tmp_path: Path) -> None:
    m = _build(
        tmp_path,
        run_flow=DaydreamRunFlow.DIAGRAM,
        identity=_manifest_identity(
            phases=replace(_manifest_identity().phases, fix=False, test=False)
        ),
        fix_failures={"src/old.py": "reverted"},
        fix_leftover_untracked=["src/leftover.py"],
        fix_quality_gate={"enabled": True, "rounds": []},
    )

    assert m.fix_failures is None
    assert m.fix_leftover_untracked is None
    assert m.fix_quality_gate is None






def test_fix_cycle_classification_covers_every_run_flow() -> None:
    """Every ``DaydreamRunFlow`` member is explicitly classified: TTT
    (review/comment) is mode-gated never to reach the fix cycle, PR (feedback)
    runs its own fix-items phase (fix yes, test no), and every other label is
    classified by its registered pipeline (issue #648). DIAGRAM (issue #1113)
    is classified by its own two-step registered pipeline, which runs neither
    fix nor test. A future enum member fails this exhaustiveness check instead
    of silently changing which backend fields the manifest emits.
    """
    mode_gated_labels = {DaydreamRunFlow.TTT}
    fix_only_labels = {DaydreamRunFlow.PR}
    fix_cycle_builtins = {
        DaydreamRunFlow.NORMAL,
        DaydreamRunFlow.DEEP,
    }
    assert set(DaydreamRunFlow) == (
        mode_gated_labels
        | fix_only_labels
        | fix_cycle_builtins
        | {
            DaydreamRunFlow.IMPROVE,
            DaydreamRunFlow.CUSTOM,
            DaydreamRunFlow.DIAGRAM,
        }
    )


def _assert_archive_omits_fix_test_backend(archive_dir: Path, flow: str) -> None:
    """Issue #648 observable outcomes: manifest + SQLite carry no fix/test backend."""
    manifests = list((archive_dir / "runs").glob("*/manifest.json"))
    assert len(manifests) == 1, f"expected exactly one archived run, found {len(manifests)}"
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["run"]["flow"] == flow
    assert "fix_backend" not in manifest["run"]
    assert "test_backend" not in manifest["run"]
    rows = query_runs(archive_dir)
    assert len(rows) == 1
    assert rows[0]["fix_backend"] is None
    assert rows[0]["test_backend"] is None


async def test_improve_archive_real_path_omits_fix_test_backend(
    improve_monorepo_target: Path,
    monkeypatch: pytest.MonkeyPatch,
    archive_dir: Path,
    make_config: MakeConfig,
) -> None:
    """Issue #648 real-path: an improve run archives no fix/test backend.

    Enters from the production entrypoint (``runner.run``) with a real temp git
    worktree, real recorder, and real event loop; only the network backend is
    mocked via the ``create_backend`` seam. The archived manifest drops
    ``fix_backend``/``test_backend`` and the SQLite runs row stores NULL.
    """
    from daydream.runner import run
    from tests.harness.improve_backend import install_improve_stub

    monkeypatch.delenv("DAYDREAM_TRAJECTORY_HUB_REPO", raising=False)
    install_improve_stub(monkeypatch, improve_monorepo_target)

    rc = await run(
        make_config(
            improve_monorepo_target, flow_name="improve", archive=True, run_eval=False,
        )
    )

    assert rc == 0
    _assert_archive_omits_fix_test_backend(archive_dir, "improve")


async def test_custom_flow_archive_real_path_omits_fix_test_backend(
    ext_dir: Any,
    multi_stack_target: Path,
    install_backend: InstallBackend,
    monkeypatch: pytest.MonkeyPatch,
    archive_dir: Path,
    make_config: MakeConfig,
) -> None:
    """Issue #648 real-path: a fork-registered custom flow archives no fix/test backend.

    Same production entry (``runner.run``) with a real git worktree and
    recorder; only the backend is mocked. The custom flow is extension-defined,
    so its pipeline has no fix/test step and the archive must not invent labels.
    """
    from daydream.backends import ResultEvent, TextEvent
    from daydream.runner import run
    from tests.harness.backend import ScriptedBackend

    monkeypatch.delenv("DAYDREAM_TRAJECTORY_HUB_REPO", raising=False)
    ext_dir.write_module(
        "from daydream.extensions import FlowStep\n"
        "async def _audit(ctx):\n"
        "    from daydream.agent import run_agent\n"
        "    from daydream.trajectory import DaydreamPhase\n"
        "    await run_agent(ctx.backend_for('ro_audit'), ctx.work.repo, 'CUSTOM-FLOW-PROMPT',\n"
        "                    phase=DaydreamPhase.REVIEW)\n"
        "def register(r):\n"
        "    r.register_phase(FlowStep(name='ro_audit', run=_audit))\n"
        "    r.set_flow('ro-audit', ['ro_audit'])\n"
    )
    install_backend(
        ScriptedBackend(
            events=(
                TextEvent(text=""),
                ResultEvent(structured_output=None, continuation=None),
            ),
            model="mock-model",
        )
    )

    rc = await run(
        make_config(
            multi_stack_target, flow_name="ro-audit", archive=True, run_eval=False,
        )
    )

    assert rc == 0
    _assert_archive_omits_fix_test_backend(archive_dir, "custom")










def test_manifest_to_dict_structure(tmp_path: Path) -> None:
    m = _build(tmp_path)

    d = m.to_dict()
    assert d["schema_version"] == "1.0"
    assert d["session_id"] == "abcd1234-0000-0000-0000-000000000000"
    assert "run" in d and d["run"]["flow"] == "normal"
    assert "git" in d
    assert "pr" in d
    assert "metrics" in d
    assert "outcome" in d
    assert d["outcome"]["labels"] == []
    assert d["code_context"] == {
        "base_sha": None,
        "head_sha": None,
        "base_branch": None,
        "branch": None,
        "changed_files": [],
    }


def test_manifest_to_dict_code_context_carries_git_ctx_fields(tmp_path: Path) -> None:
    m = _build(
        tmp_path,
        git_ctx=GitContext(
            branch="feat/x",
            base_branch="main",
            head_sha="b" * 40,
            base_sha="c" * 40,
            changed_files=["a.py", "b.py"],
        ),
    )

    d = m.to_dict()
    assert d["code_context"] == {
        "base_sha": "c" * 40,
        "head_sha": "b" * 40,
        "base_branch": "main",
        "branch": "feat/x",
        "changed_files": ["a.py", "b.py"],
    }


def test_build_manifest_with_evaluation(tmp_path: Path) -> None:
    m = _build(
        tmp_path,
        evaluation={
            "timing": {"total_wall_clock_seconds": 42.5},
            "findings": {"total": 7},
            "grounding": {"grounding_rate": 0.85},
            "coverage": {"coverage_ratio": 0.6},
            "derived": {"cost_per_finding_usd": 0.007},
        },
    )

    assert m.wall_clock_seconds == 42.5
    assert m.total_findings == 7
    assert m.grounding_rate == 0.85
    assert m.coverage_ratio == 0.6
    assert m.cost_per_finding_usd == 0.007


def test_build_manifest_with_quality(tmp_path: Path) -> None:
    m = _build(
        tmp_path,
        evaluation={
            "quality": {"erosion": 0.34, "verbosity": 0.19},
        },
    )

    assert m.erosion == 0.34
    assert m.verbosity == 0.19
    d = m.to_dict()
    assert d["metrics"]["erosion"] == 0.34
    assert d["metrics"]["verbosity"] == 0.19


def test_build_manifest_without_evaluation(tmp_path: Path) -> None:
    m = _build(tmp_path)

    assert m.total_findings is None
    assert m.grounding_rate is None
    assert m.coverage_ratio is None
    assert m.cost_per_finding_usd is None
    assert m.erosion is None
    assert m.verbosity is None


def test_build_manifest_wall_clock_without_evaluation(tmp_path: Path) -> None:
    """The snapshot's own run span fills wall-clock even when --eval did not run."""
    m = _build(
        tmp_path,
        write_snapshot=_manifest_write_snapshot(
            extra={
                "run_started_at": "2026-01-01T00:00:00Z",
                "run_ended_at": "2026-01-01T00:00:12.300000Z",
            },
        ),
    )

    assert m.wall_clock_seconds == 12.3
    assert m.total_findings is None


def test_build_manifest_snapshot_timing_overrides_conflicting_evaluation(
    tmp_path: Path,
) -> None:
    session_id = "snapshot-session"
    payload = {
        "session_id": session_id,
        "trajectory_id": session_id,
        "steps": [],
        "final_metrics": {"total_prompt_tokens": 7, "total_steps": 0},
        "extra": {
            "run_started_at": "2026-01-01T00:00:00Z",
            "run_ended_at": "2026-01-01T00:00:10Z",
            "phase_events": [
                {
                    "phase": "review",
                    "event": "phase_start",
                    "timestamp": "2026-01-01T00:00:02Z",
                    "session_id": session_id,
                    "scope_id": "review",
                },
                {
                    "phase": "review",
                    "event": "phase_end",
                    "timestamp": "2026-01-01T00:00:08Z",
                    "session_id": session_id,
                    "scope_id": "review",
                    "status": "succeeded",
                },
            ],
        },
    }
    snapshot = RunWriteSnapshot(
        status="complete",
        cutoff_at="2026-01-01T00:00:10Z",
        root_trajectory_id=session_id,
        documents=(
            TrajectoryDocumentSnapshot(
                session_id,
                Path("/frozen/snapshot-session.json"),
                json.dumps(payload).encode(),
            ),
        ),
    )

    manifest = _build(
        tmp_path,
        write_snapshot=snapshot,
        evaluation={"timing": {"total_wall_clock_seconds": 42.5}},
    )

    assert manifest.wall_clock_seconds == 10.0
    assert manifest.phase_timings == {"review": {"wall_clock_seconds": 6.0, "occurrences": 1}}
    assert manifest.timing_coverage == {
        "attributed_wall_clock_seconds": 6.0,
        "unattributed_wall_clock_seconds": 4.0,
        "coverage_ratio": 0.6,
        "agent_completeness": {"total": 0, "attributed": 0, "unattributed": 0},
        "diagnostics": {
            "malformed_interval": 0,
            "duplicate_interval": 0,
            "orphaned_interval": 0,
            "malformed_invocation": 0,
            "duplicate_invocation": 0,
            "legacy_fork_proxy_used": 0,
        },
    }
    assert manifest.total_prompt_tokens == 7


def test_upsert_run_creates_db(tmp_path: Path) -> None:
    m = make_manifest()
    upsert_run(tmp_path, m)
    assert (tmp_path / "index.db").exists()


def test_upsert_run_never_persists_credential_bearing_url(tmp_path: Path) -> None:
    # M4: even if upstream normalization is bypassed, the row is clean.
    m = make_manifest(remote_url="https://user:ghp_bypassfake@github.com/o/r.git")
    upsert_run(tmp_path, m)
    row = query_runs(tmp_path)[0]
    assert row["remote_url"] == "https://github.com/o/r"
    assert row["repo_slug"] == "o/r"
    assert "ghp_bypassfake" not in (row["remote_url"] or "")


def test_upsert_and_query_round_trip(tmp_path: Path) -> None:
    m = make_manifest()
    upsert_run(tmp_path, m)

    rows = query_runs(tmp_path)
    assert len(rows) == 1
    assert rows[0]["session_id"] == "sess-0001"
    assert rows[0]["skill"] == "python"
    assert rows[0]["status"] == "complete"


def test_update_labels_exact(tmp_path: Path) -> None:
    upsert_run(tmp_path, make_manifest())

    ok = update_labels(tmp_path, "sess-0001", ["good", "fast"])
    assert ok is True

    rows = query_runs(tmp_path)
    assert json.loads(rows[0]["outcome_labels"]) == ["good", "fast"]
    assert rows[0]["labeled_at"] is not None


def test_update_labels_prefix(tmp_path: Path) -> None:
    upsert_run(tmp_path, make_manifest(session_id="abcd1234-full-uuid"))

    ok = update_labels(tmp_path, "abcd1234", ["label-a"])
    assert ok is True

    rows = query_runs(tmp_path)
    assert json.loads(rows[0]["outcome_labels"]) == ["label-a"]


def test_update_labels_nonexistent(tmp_path: Path) -> None:
    upsert_run(tmp_path, make_manifest())

    ok = update_labels(tmp_path, "no-such-session", [])
    assert ok is False


def test_update_labels_ambiguous_prefix(tmp_path: Path) -> None:
    upsert_run(tmp_path, make_manifest(session_id="abc-001"))
    upsert_run(tmp_path, make_manifest(session_id="abc-002", archive_path="/tmp/x"))

    with pytest.raises(ValueError, match="matches 2 sessions"):
        update_labels(tmp_path, "abc", ["x"])


def test_set_run_pr_link_backfills_pr_columns(tmp_path: Path) -> None:
    upsert_run(tmp_path, make_manifest(session_id="s-orphan", pr_number=None, pr_repo=None))
    set_run_pr_link(tmp_path, "s-orphan", 7, "org/repo")
    row = query_runs(tmp_path, where="session_id = ?", params=("s-orphan",))[0]
    assert (row["pr_number"], row["pr_repo"]) == (7, "org/repo")


def test_query_runs_with_where(tmp_path: Path) -> None:
    upsert_run(tmp_path, make_manifest(session_id="s1", repo_slug="org/a"))
    upsert_run(
        tmp_path,
        make_manifest(session_id="s2", repo_slug="org/b", archive_path="/tmp/s2"),
    )
    upsert_run(
        tmp_path,
        make_manifest(session_id="s3", repo_slug="org/a", archive_path="/tmp/s3"),
    )

    rows = query_runs(tmp_path, where="repo_slug = ?", params=("org/a",))
    assert len(rows) == 2
    ids = {r["session_id"] for r in rows}
    assert ids == {"s1", "s3"}


def test_upsert_run_persists_erosion_verbosity(tmp_path: Path) -> None:
    """erosion/verbosity round-trip through upsert_run -> query_runs, filterable and sortable."""
    upsert_run(tmp_path, make_manifest(session_id="s-q1", erosion=0.34, verbosity=0.19))
    upsert_run(tmp_path, make_manifest(session_id="s-q2", archive_path="/tmp/s-q2"))
    upsert_run(
        tmp_path,
        make_manifest(session_id="s-q3", erosion=0.51, verbosity=0.07, archive_path="/tmp/s-q3"),
    )

    row = query_runs(tmp_path, where="session_id = ?", params=("s-q1",))[0]
    assert row["erosion"] == pytest.approx(0.34)
    assert row["verbosity"] == pytest.approx(0.19)

    # The query layer's WHERE binds against the new columns.
    scored = query_runs(tmp_path, where="erosion IS NOT NULL")
    assert {r["session_id"] for r in scored} == {"s-q1", "s-q3"}

    # The columns are sortable (NULLs sort first in SQLite).
    conn = sqlite3.connect(str(tmp_path / "index.db"))
    try:
        ordered = [r[0] for r in conn.execute("SELECT session_id FROM runs ORDER BY erosion")]
    finally:
        conn.close()
    assert ordered == ["s-q2", "s-q1", "s-q3"]


def test_runs_erosion_verbosity_columns_migrate_existing_db(tmp_path: Path) -> None:
    """A pre-existing index.db without erosion/verbosity gains them via ALTER-ADD.

    Mirrors the source_path/composite_reward additive migration: a legacy runs
    table (no erosion/verbosity columns) must keep its rows and gain the
    columns on the next production write, never dropping or rewriting data.
    """
    from daydream.archive.index import _CREATE_TABLE

    legacy_ddl = _CREATE_TABLE.replace("    erosion REAL,\n    verbosity REAL,\n", "")
    assert "erosion" not in legacy_ddl
    conn = sqlite3.connect(str(tmp_path / "index.db"))
    conn.execute(legacy_ddl)
    conn.execute(
        "INSERT INTO runs (session_id, archived_at, run_flow, archive_path) VALUES (?, ?, ?, ?)",
        ("legacy-run", "2026-01-01T00:00:00Z", "normal", str(tmp_path / "legacy-run")),
    )
    conn.commit()
    conn.close()

    # The production write path must ALTER-ADD the columns non-destructively.
    upsert_run(tmp_path, make_manifest(session_id="s-mig-q", erosion=0.42, verbosity=0.08))

    legacy = query_runs(tmp_path, where="session_id = ?", params=("legacy-run",))[0]
    assert legacy["erosion"] is None  # pre-existing row preserved, columns nullable
    row = query_runs(tmp_path, where="session_id = ?", params=("s-mig-q",))[0]
    assert row["erosion"] == pytest.approx(0.42)
    assert row["verbosity"] == pytest.approx(0.08)


def test_build_manifest_projects_location_and_duplication_metrics(
    tmp_path: Path,
) -> None:
    """#1106: the location-accuracy and escaped-duplication axes reach the manifest.

    The eval pass computes a location verdict per shipped finding and a
    shipped-set duplication scan; both headline scalars must be projected so a
    change to line resolution or the hunk index is measurable across runs.
    """
    m = _build(
        tmp_path,
        evaluation={
            "location": {
                "hunk_source": "hunk-index.json",
                "scored_items": 4,
                "in_hunk_rate": 0.75,
                "tiers": {
                    "in_hunk": 3,
                    "within_tolerance": 1,
                    "beyond_tolerance": 0,
                    "file_absent": 0,
                },
            },
            "findings": {
                "total": 6,
                "shipped_duplication": {
                    "shipped_items": 6,
                    "comparable_pairs": 15,
                    "near_duplicate_pairs": 2,
                },
            },
        },
    )

    assert m.location_in_hunk_rate == 0.75
    assert m.shipped_duplicate_pairs == 2
    d = m.to_dict()
    assert d["metrics"]["location_in_hunk_rate"] == 0.75
    assert d["metrics"]["shipped_duplicate_pairs"] == 2


def test_build_manifest_location_duplication_metrics_none_when_blocks_absent(
    tmp_path: Path,
) -> None:
    """#1106: an evaluation.json predating the axes yields None, never 0.

    Every already-archived run carries a `findings` block without
    `shipped_duplication` and no `location` block at all. The projection must
    chain defensively and leave both metrics undefined rather than reporting a
    perfect in-hunk rate or zero escaped duplicates.
    """
    m = _build(
        tmp_path,
        evaluation={
            "timing": {"total_wall_clock_seconds": 42.5},
            "findings": {"total": 7},
            "grounding": {"grounding_rate": 0.85},
        },
    )

    assert m.total_findings == 7  # the legacy axes still project
    assert m.location_in_hunk_rate is None
    assert m.shipped_duplicate_pairs is None
    d = m.to_dict()
    assert d["metrics"]["location_in_hunk_rate"] is None
    assert d["metrics"]["shipped_duplicate_pairs"] is None


def test_null_in_hunk_rate_survives_manifest_and_db_as_null(tmp_path: Path) -> None:
    """#1106: `in_hunk_rate: None` (no scorable finding) stays undefined end to end.

    The analyzer emits None — not 0.0 — when `scored_items == 0`, because a run
    that located nothing has no accuracy, and the reward pipeline renormalizes
    over PRESENT axes. Coercion to 0.0 anywhere in the projection would feed it
    an imputed worst score.
    """
    m = _build(
        tmp_path,
        evaluation={
            "location": {
                "hunk_source": "none",
                "scored_items": 0,
                "in_hunk_rate": None,
            },
            "findings": {
                "total": 0,
                "shipped_duplication": {"near_duplicate_pairs": 0},
            },
        },
    )
    assert m.location_in_hunk_rate is None
    assert m.to_dict()["metrics"]["location_in_hunk_rate"] is None

    upsert_run(
        tmp_path,
        make_manifest(
            session_id="s-loc-null",
            location_in_hunk_rate=m.location_in_hunk_rate,
            shipped_duplicate_pairs=m.shipped_duplicate_pairs,
        ),
    )
    row = query_runs(tmp_path, where="session_id = ?", params=("s-loc-null",))[0]
    assert row["location_in_hunk_rate"] is None  # SQL NULL, not 0.0
    assert row["shipped_duplicate_pairs"] == 0  # a real zero is still a zero


def test_upsert_run_persists_location_and_duplication_metrics(tmp_path: Path) -> None:
    """#1106: both new metrics round-trip through upsert_run -> query_runs."""
    upsert_run(
        tmp_path,
        make_manifest(
            session_id="s-loc",
            location_in_hunk_rate=0.6,
            shipped_duplicate_pairs=3,
        ),
    )
    row = query_runs(tmp_path, where="session_id = ?", params=("s-loc",))[0]
    assert row["location_in_hunk_rate"] == pytest.approx(0.6)
    assert row["shipped_duplicate_pairs"] == 3


def test_runs_location_duplication_columns_migrate_existing_db(tmp_path: Path) -> None:
    """#1106: a real pre-existing v7 index.db gains both columns via ALTER-ADD.

    Mirrors the erosion/verbosity additive migration: the legacy runs table (v7
    DDL minus the two new columns, PRAGMA user_version = 7) keeps its rows and
    gains the columns on the next production write, never dropping or
    rewriting data.
    """
    from daydream.archive.index import _CREATE_TABLE, SCHEMA_VERSION

    legacy_ddl = _CREATE_TABLE.replace(
        "    location_in_hunk_rate REAL,\n    shipped_duplicate_pairs INTEGER,\n", ""
    )
    assert "location_in_hunk_rate" not in legacy_ddl
    assert "shipped_duplicate_pairs" not in legacy_ddl
    db_path = tmp_path / "index.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(legacy_ddl)
    conn.execute(
        "INSERT INTO runs (session_id, archived_at, run_flow, archive_path, erosion) VALUES (?, ?, ?, ?, ?)",
        (
            "legacy-loc-run",
            "2026-01-01T00:00:00Z",
            "normal",
            str(tmp_path / "legacy-loc-run"),
            0.5,
        ),
    )
    conn.execute("PRAGMA user_version = 7")
    conn.commit()
    conn.close()

    # The production write path must ALTER-ADD both columns non-destructively.
    upsert_run(
        tmp_path,
        make_manifest(
            session_id="s-mig-loc",
            location_in_hunk_rate=0.25,
            shipped_duplicate_pairs=4,
        ),
    )

    conn = sqlite3.connect(str(db_path))
    cols = {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
    user_version = conn.execute("PRAGMA user_version").fetchone()[0]
    conn.close()
    assert {"location_in_hunk_rate", "shipped_duplicate_pairs"} <= cols
    assert user_version == SCHEMA_VERSION == 8

    legacy = query_runs(tmp_path, where="session_id = ?", params=("legacy-loc-run",))[0]
    assert legacy["erosion"] == pytest.approx(0.5)  # pre-existing row preserved
    assert legacy["location_in_hunk_rate"] is None  # new columns nullable
    assert legacy["shipped_duplicate_pairs"] is None
    row = query_runs(tmp_path, where="session_id = ?", params=("s-mig-loc",))[0]
    assert row["location_in_hunk_rate"] == pytest.approx(0.25)
    assert row["shipped_duplicate_pairs"] == 4


def test_upsert_run_persists_per_stack_review_identity(tmp_path: Path) -> None:
    """Issue #646: per-stack review identity round-trips through the index."""
    m = make_manifest(
        session_id="s-psr",
        per_stack_review_backend="codex",
        per_stack_review_model="gpt-psr",
        review_backend="claude",
    )
    upsert_run(tmp_path, m)
    row = query_runs(tmp_path, where="session_id = ?", params=("s-psr",))[0]
    assert row["per_stack_review_backend"] == "codex"
    assert row["per_stack_review_model"] == "gpt-psr"
    assert row["review_backend"] == "claude"


def test_runs_per_stack_review_columns_migrate_existing_db(tmp_path: Path) -> None:
    """A legacy index.db gains per_stack_review_* via ALTER-ADD, rows preserved."""
    from daydream.archive.index import _CREATE_TABLE

    legacy_ddl = _CREATE_TABLE.replace(
        "    per_stack_review_backend TEXT,\n    per_stack_review_model TEXT,\n", ""
    )
    assert "per_stack_review_backend" not in legacy_ddl
    conn = sqlite3.connect(str(tmp_path / "index.db"))
    conn.execute(legacy_ddl)
    conn.execute(
        "INSERT INTO runs (session_id, archived_at, run_flow, archive_path) VALUES (?, ?, ?, ?)",
        ("legacy-psr", "2026-01-01T00:00:00Z", "normal", str(tmp_path / "legacy-psr")),
    )
    conn.commit()
    conn.close()

    upsert_run(
        tmp_path,
        make_manifest(
            session_id="s-psr-mig",
            per_stack_review_backend="codex",
            per_stack_review_model="gpt-psr",
        ),
    )
    legacy = query_runs(tmp_path, where="session_id = ?", params=("legacy-psr",))[0]
    assert legacy["per_stack_review_backend"] is None   # pre-existing row preserved, nullable
    row = query_runs(tmp_path, where="session_id = ?", params=("s-psr-mig",))[0]
    assert row["per_stack_review_backend"] == "codex"
    assert row["per_stack_review_model"] == "gpt-psr"


def test_build_manifest_carries_fix_quality_gate(tmp_path: Path) -> None:
    """Issue #315: the fix-phase quality-gate verdict round-trips on the manifest."""
    gate = {
        "enabled": True,
        "erosion_delta_threshold": 0.05,
        "verbosity_delta_threshold": 0.05,
        "rounds": [
            {
                "round": 1,
                "per_file": {
                    "api.py": {
                        "erosion_before": 0.0,
                        "erosion_after": 0.0,
                        "erosion_delta": 0.0,
                        "verbosity_before": 0.0,
                        "verbosity_after": 0.8,
                        "verbosity_delta": 0.8,
                        "flagged": True,
                    }
                },
            }
        ],
    }
    m = _build(tmp_path, fix_quality_gate=gate)
    assert m.fix_quality_gate == gate
    d = m.to_dict()
    assert d["fix_quality_gate"] == gate
    assert d["fix_quality_gate"]["rounds"][0]["per_file"]["api.py"]["flagged"] is True


def test_manifest_fix_quality_gate_none_when_absent(tmp_path: Path) -> None:
    """No gate artifact => the manifest field stays null (additive, never invented)."""
    m = _build(tmp_path)
    assert m.fix_quality_gate is None
    assert m.to_dict()["fix_quality_gate"] is None


def test_upsert_run_persists_fix_quality_gate(tmp_path: Path) -> None:
    """Issue #315: fix_quality_gate JSON round-trips through upsert_run -> query_runs."""
    gate = {
        "enabled": True,
        "rounds": [{"round": 1, "per_file": {"api.py": {"flagged": True}}}],
    }
    upsert_run(tmp_path, make_manifest(session_id="s-gate", fix_quality_gate=gate))
    row = query_runs(tmp_path, where="session_id = ?", params=("s-gate",))[0]
    assert json.loads(row["fix_quality_gate"]) == gate


def test_runs_fix_quality_gate_column_migrates_existing_db(tmp_path: Path) -> None:
    """A pre-existing index.db without fix_quality_gate gains it via ALTER-ADD.

    Mirrors the erosion/verbosity additive migration: a legacy runs table keeps
    its rows and gains the column on the next production write, never dropping
    or rewriting data.
    """
    from daydream.archive.index import _CREATE_TABLE

    legacy_ddl = _CREATE_TABLE.replace("    fix_quality_gate TEXT,\n", "")
    assert "fix_quality_gate" not in legacy_ddl
    conn = sqlite3.connect(str(tmp_path / "index.db"))
    conn.execute(legacy_ddl)
    conn.execute(
        "INSERT INTO runs (session_id, archived_at, run_flow, archive_path) VALUES (?, ?, ?, ?)",
        (
            "legacy-gate-run",
            "2026-01-01T00:00:00Z",
            "normal",
            str(tmp_path / "legacy-gate-run"),
        ),
    )
    conn.commit()
    conn.close()

    gate = {"enabled": True, "rounds": []}
    upsert_run(tmp_path, make_manifest(session_id="s-mig-gate", fix_quality_gate=gate))

    legacy = query_runs(tmp_path, where="session_id = ?", params=("legacy-gate-run",))[0]
    assert legacy["fix_quality_gate"] is None  # pre-existing row preserved, column nullable
    row = query_runs(tmp_path, where="session_id = ?", params=("s-mig-gate",))[0]
    assert json.loads(row["fix_quality_gate"]) == gate


def test_upsert_run_persists_recommended_patch_capture(tmp_path: Path) -> None:
    upsert_run(
        tmp_path,
        make_manifest(session_id="s-cap", recommended_patch_capture="post_test"),
    )
    row = query_runs(tmp_path, where="session_id = ?", params=("s-cap",))[0]
    assert row["recommended_patch_capture"] == "post_test"


def test_runs_recommended_patch_capture_column_migrates_existing_db(
    tmp_path: Path,
) -> None:
    from daydream.archive.index import _CREATE_TABLE

    legacy_ddl = _CREATE_TABLE.replace("    recommended_patch_capture TEXT,\n", "")
    assert "recommended_patch_capture" not in legacy_ddl
    conn = sqlite3.connect(str(tmp_path / "index.db"))
    conn.execute(legacy_ddl)
    conn.execute(
        "INSERT INTO runs (session_id, archived_at, run_flow, archive_path) VALUES (?, ?, ?, ?)",
        (
            "legacy-cap-run",
            "2026-01-01T00:00:00Z",
            "normal",
            str(tmp_path / "legacy-cap-run"),
        ),
    )
    conn.commit()
    conn.close()

    upsert_run(
        tmp_path,
        make_manifest(session_id="s-mig-cap", recommended_patch_capture="pre_test"),
    )

    legacy = query_runs(tmp_path, where="session_id = ?", params=("legacy-cap-run",))[0]
    assert legacy["recommended_patch_capture"] is None  # pre-existing row preserved, column nullable
    row = query_runs(tmp_path, where="session_id = ?", params=("s-mig-cap",))[0]
    assert row["recommended_patch_capture"] == "pre_test"


def test_read_fix_quality_gate_requires_matching_session(tmp_path: Path) -> None:
    """#329: only an artifact bound to the current session is read.

    A gate verdict left behind by another session (e.g. a prior deep run on the
    same target repo) must not be attributed to the current run's manifest.
    """
    gate = {
        "enabled": True,
        "session_id": "sess-42",
        "rounds": [{"round": 1, "per_file": {"api.py": {"flagged": True}}}],
    }
    gate_p = tmp_path / ".daydream" / "deep" / "fix-quality-gate.json"
    gate_p.parent.mkdir(parents=True)
    gate_p.write_text(json.dumps(gate))

    assert _read_fix_quality_gate(tmp_path, "sess-42") == gate
    assert _read_fix_quality_gate(tmp_path, "sess-other") is None
    assert _read_fix_quality_gate(tmp_path, None) is None


def test_read_fix_quality_gate_unbound_artifact_is_none(tmp_path: Path) -> None:
    """#329: an artifact with no session_id key cannot be attributed to this run."""
    gate_p = tmp_path / ".daydream" / "deep" / "fix-quality-gate.json"
    gate_p.parent.mkdir(parents=True)
    gate_p.write_text(json.dumps({"enabled": True, "rounds": [{"round": 1, "per_file": {}}]}))

    assert _read_fix_quality_gate(tmp_path, "sess-42") is None


def test_read_recommended_capture_requires_matching_session(tmp_path: Path) -> None:
    from daydream.archive import _read_recommended_capture

    cap = {"session_id": "sess-42", "capture_point": "post_test"}
    p = tmp_path / ".daydream" / "deep" / "recommended-capture.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(cap))

    assert _read_recommended_capture(tmp_path, "sess-42") == cap
    assert _read_recommended_capture(tmp_path, "sess-other") is None
    assert _read_recommended_capture(tmp_path, None) is None


def test_read_recommended_capture_absent_is_none(tmp_path: Path) -> None:
    from daydream.archive import _read_recommended_capture

    assert _read_recommended_capture(tmp_path, "sess-42") is None


def test_manifest_recommended_patch_capture_defaults_pre_test(tmp_path: Path) -> None:
    m = _build(tmp_path)  # no recommended_capture arg => sidecar absent
    assert m.recommended_patch_capture == "pre_test"
    assert m.to_dict()["recommended_patch_capture"] == "pre_test"


def test_manifest_recommended_patch_capture_omitted_when_none() -> None:
    assert "recommended_patch_capture" not in Manifest().to_dict()


def test_manifest_recommended_patch_capture_passes_through(tmp_path: Path) -> None:
    m = _build(tmp_path, recommended_capture="post_test")
    assert m.to_dict()["recommended_patch_capture"] == "post_test"


def test_feedback_run_leaves_recommended_patch_capture_none() -> None:
    """PR/feedback runs never produce a pre-test fix-phase capture, so an
    absent sidecar must leave ``recommended_patch_capture`` ``None`` rather
    than defaulting to ``"pre_test"`` (feedback's ``_step_fix_items`` never
    writes ``recommended.patch``)."""
    m = _build(tmp_path=Path("/tmp"), run_flow=DaydreamRunFlow.PR)
    assert m.recommended_patch_capture is None
    assert "recommended_patch_capture" not in m.to_dict()


def test_get_archive_dir_creates_structure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "custom_archive"
    monkeypatch.setenv("DAYDREAM_ARCHIVE_DIR", str(target))

    result = get_archive_dir()
    assert result == target
    assert target.is_dir()
    assert (target / "runs").is_dir()


def test_get_archive_dir_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("DAYDREAM_ARCHIVE_DIR", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

    result = get_archive_dir()
    expected = tmp_path / ".daydream" / "archive"
    assert result == expected
    assert expected.is_dir()


def _setup_bundle(
    tmp_path: Path,
    session_id: str = "abcd1234-0000-0000-0000-000000000000",
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
    target: Path,
    run_dir: Path,
    recorder: _MockRecorder,
    *,
    write_snapshot: RunWriteSnapshot | None = None,
    destinations: tuple[RoutedDestination, ...] = (),
) -> None:
    """Run the production bundle assembler over one frozen tree + snapshot."""
    from daydream.archive import _copy_snapshot_bundle

    snapshot = write_snapshot if write_snapshot is not None else _write_snapshot(recorder)
    _copy_snapshot_bundle(
        run=_archive_snapshot(snapshot, run_flow=recorder.run_flow),
        artifacts=ArtifactTreeSnapshot(
            session_id=recorder.session_id,
            workspace_key="workspace",
            root=target,
            manifest=_manifest(target),
            destinations=destinations,
        ),
        artifact_provenance=ArtifactEvidenceProvenance(
            workspace_key="workspace",
            session_id=recorder.session_id,
            public_source=target,
            live_root=target,
        ),
        run_dir=run_dir,
    )


def test_bundle_projects_only_frozen_snapshot_trajectory_bytes(
    tmp_path: Path,
) -> None:
    """``trajectory.json`` carries the frozen bytes, never the live tree's copy."""
    target, run_dir, recorder = _setup_bundle(tmp_path)
    live = target / ".daydream" / "runs" / recorder.session_id / "trajectory.json"
    live.write_text('{"session_id":"abcd1234-0000-0000-0000-000000000000","marker":"MUTATED_LIVE"}')
    frozen = json.dumps(
        {
            "session_id": recorder.session_id,
            "trajectory_id": recorder.session_id,
            "marker": "FROZEN",
        }
    ).encode()
    snapshot = RunWriteSnapshot(
        status="complete",
        cutoff_at="2026-01-01T00:00:01Z",
        root_trajectory_id=recorder.session_id,
        documents=(TrajectoryDocumentSnapshot(recorder.session_id, live, frozen),),
    )

    _assemble_bundle(target, run_dir, recorder, write_snapshot=snapshot)

    assert (run_dir / "trajectory.json").read_bytes() == frozen


def test_bundle_rejects_a_sibling_document_bound_to_another_session(
    tmp_path: Path,
) -> None:
    """A fork document whose session is not this run's is refused, not archived."""
    target, run_dir, recorder = _setup_bundle(tmp_path)
    root = _write_snapshot(recorder).documents[0]
    foreign = json.dumps(
        {"session_id": "other-session", "trajectory_id": "fork-1"}
    ).encode()
    snapshot = RunWriteSnapshot(
        status="complete",
        cutoff_at="2026-01-01T00:00:01Z",
        root_trajectory_id=recorder.session_id,
        documents=(root, TrajectoryDocumentSnapshot("fork-1", target / "fork.json", foreign)),
    )

    with pytest.raises(ValueError, match="frozen trajectory document identity"):
        _assemble_bundle(target, run_dir, recorder, write_snapshot=snapshot)

    assert not (run_dir / "trajectories").exists()


@pytest.mark.parametrize(
    ("relative_path", "expected"),
    [
        pytest.param("review-output.md", "review findings", id="review-output"),
        pytest.param("diff.patch", "diff content", id="diff-patch"),
    ],
)
def test_bundle_file_path(tmp_path: Path, relative_path: str, expected: str) -> None:
    """Verify bundle copying preserves parameterized file contents and paths."""
    target, run_dir, recorder = _setup_bundle(tmp_path)
    _assemble_bundle(target, run_dir, recorder)

    assert (run_dir / relative_path).read_text() == expected


def test_bundle_deep_directory(tmp_path: Path) -> None:
    target, run_dir, recorder = _setup_bundle(tmp_path)
    _assemble_bundle(target, run_dir, recorder)

    assert (run_dir / "deep" / "intent.md").read_text() == "intent"


def test_bundle_diagram_flow_excludes_stale_review_artifacts(
    tmp_path: Path,
) -> None:
    target, run_dir, recorder = _setup_bundle(tmp_path)
    recorder.run_flow = DaydreamRunFlow.DIAGRAM
    deep_dir = target / ".daydream" / "deep"
    (deep_dir / "merged-items.json").write_text('{"items": []}')
    (deep_dir / "fix-failures.json").write_text('{"src/old.py": "reverted"}')
    (deep_dir / "diagram.json").write_text('{"results": {}}')
    (deep_dir / "diagram.md").write_text("current diagram")
    (target / ".daydream" / "recommended.patch").write_text("stale recommendation")

    _assemble_bundle(target, run_dir, recorder)

    assert sorted(path.name for path in (run_dir / "deep").iterdir()) == [
        "diagram.json",
        "diagram.md",
    ]
    assert (run_dir / "deep" / "diagram.md").read_text() == "current diagram"
    assert not (run_dir / "review-output.md").exists()
    assert not (run_dir / "recommended.patch").exists()


def test_bundle_sub_trajectories_projected(tmp_path: Path) -> None:
    """Sibling fork documents are projected under ``trajectories/`` by name."""
    target, run_dir, recorder = _setup_bundle(tmp_path)
    fork_path = target / ".daydream" / "runs" / recorder.session_id / "trajectories" / "deep-python.json"
    root = _write_snapshot(recorder).documents[0]
    fork_bytes = json.dumps(
        {"session_id": recorder.session_id, "trajectory_id": "fork-1"}
    ).encode()
    snapshot = RunWriteSnapshot(
        status="complete",
        cutoff_at="2026-01-01T00:00:01Z",
        root_trajectory_id=recorder.session_id,
        documents=(root, TrajectoryDocumentSnapshot("fork-1", fork_path, fork_bytes)),
    )

    _assemble_bundle(target, run_dir, recorder, write_snapshot=snapshot)

    sub = run_dir / "trajectories"
    assert sub.is_dir()
    assert sorted(p.name for p in sub.iterdir()) == ["deep-python.json"]
    assert (sub / "deep-python.json").read_bytes() == fork_bytes


def test_bundle_skips_missing(tmp_path: Path) -> None:
    target = tmp_path / "empty_target"
    target.mkdir()
    (target / ".daydream").mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    recorder = _MockRecorder(
        session_id="no-match-session-id-here", path=tmp_path / "nonexistent.json"
    )
    _assemble_bundle(target, run_dir, recorder)

    assert (run_dir / "trajectory.json").is_file()  # the frozen document always lands
    assert not (run_dir / "review-output.md").exists()
    assert not (run_dir / "deep").exists()
    assert not (run_dir / "diff.patch").exists()


def test_bundle_archives_findings_artifact(tmp_path: Path) -> None:
    """findings.json is relocated from its registered route into the bundle.

    The archive never reconstructs the operator's requested path: it reads the
    route's frozen path relative to the live root and resolves it inside the
    frozen tree, so harvest's per-finding join has a fingerprint source.
    """
    target, run_dir, recorder = _setup_bundle(tmp_path)
    route = _findings_route(target)
    assert route.frozen_path is not None
    route.frozen_path.parent.mkdir(parents=True)
    route.frozen_path.write_text('{"findings": [{"fingerprint": "abc"}]}')

    _assemble_bundle(target, run_dir, recorder, destinations=(route,))

    archived = run_dir / "findings.json"
    assert archived.is_file()
    assert json.loads(archived.read_text())["findings"][0]["fingerprint"] == "abc"


def test_bundle_findings_artifact_skipped_without_route(
    tmp_path: Path,
) -> None:
    """No registered findings destination means no findings.json is archived."""
    target, run_dir, recorder = _setup_bundle(tmp_path)
    _assemble_bundle(target, run_dir, recorder)

    assert not (run_dir / "findings.json").exists()


def test_dump_artifacts_refuses_credential_bearing_bundle(
    tmp_path: Path,
    archive_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """M12: a bundle whose serialized artifacts carry a credential is never published.

    The real scanner (no monkeypatched verdict) reads the assembled bundle. The
    strict finalizer refuses closed: neither the ``--dump-artifacts``
    destination nor the archive receives the dirty bundle, and the failure never
    echoes the credential (M11)."""
    from daydream.archive import ArchiveFinalizationError

    session_id = "abcd1234-0000-0000-0000-000000000000"
    dest = tmp_path / "dump"
    dest.mkdir()
    config = RunConfig(target=str(tmp_path), archive=True, dump_artifacts=str(dest))

    target, _, recorder = _setup_bundle(tmp_path, session_id)
    # Inject a credential into the serialized trajectory the bundle will carry.
    traj = json.loads(recorder.path.read_text())
    traj["remote_url"] = "https://user:ghp_canaryfake123@github.com/o/r"
    recorder.path.write_text(json.dumps(traj))

    with pytest.raises(ArchiveFinalizationError, match="secret scan") as excinfo:
        _strict_archive(
            target=target,
            session_id=session_id,
            config=config,
            write_snapshot=_write_snapshot(recorder),
            dump_path=dest,
        )

    assert list(dest.iterdir()) == []
    assert not (archive_dir / "runs" / session_id).exists()
    assert query_runs(archive_dir) == []

    # The refusal is value-free (M11): the credential never echoes.
    captured = capsys.readouterr()
    out = captured.out + captured.err + str(excinfo.value)
    assert "ghp_canaryfake123" not in out


def test_dump_artifacts_copies_clean_bundle(
    tmp_path: Path, archive_dir: Path
) -> None:
    """The clean path is unchanged: a scan-clean bundle is copied wholesale."""
    session_id = "abcd1234-0000-0000-0000-000000000000"
    dest = tmp_path / "dump"
    dest.mkdir()
    config = RunConfig(target=str(tmp_path), archive=True, dump_artifacts=str(dest))
    target, _, recorder = _setup_bundle(tmp_path, session_id)

    _strict_archive(
        target=target,
        session_id=session_id,
        config=config,
        write_snapshot=_write_snapshot(recorder),
        dump_path=dest,
    )

    assert (dest / "manifest.json").is_file()
    assert (dest / "trajectory.json").is_file()
    run_dir = archive_dir / "runs" / session_id
    assert (dest / "manifest.json").read_text() == (run_dir / "manifest.json").read_text()


def test_finalize_archive_run_round_trip(tmp_path: Path, archive_dir: Path) -> None:
    session_id = "abcd1234-0000-0000-0000-000000000000"
    config = RunConfig(target=str(tmp_path), archive=True)

    target, _, recorder = _setup_bundle(tmp_path, session_id)

    _strict_archive(
        target=target,
        session_id=session_id,
        config=config,
        write_snapshot=_write_snapshot(recorder),
    )

    run_dir = archive_dir / "runs" / session_id
    assert run_dir.is_dir()
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "trajectory.json").is_file()

    manifest_data = json.loads((run_dir / "manifest.json").read_text())
    assert manifest_data["session_id"] == session_id
    assert manifest_data["run"]["flow"] == "normal"
    assert manifest_data["run"]["skill"] == "python"

    rows = query_runs(archive_dir)
    assert len(rows) == 1
    assert rows[0]["session_id"] == session_id


# index: label_observations (Task 12)


def test_delete_runs_removes_matching_rows_and_returns_count(tmp_path: Path) -> None:
    _seed_one_run(tmp_path, "sess-a")
    _seed_one_run(tmp_path, "sess-b")

    deleted = delete_runs(tmp_path, ["sess-a", "sess-missing"])

    assert deleted == 1
    remaining = [r["session_id"] for r in query_runs(tmp_path)]
    assert remaining == ["sess-b"]


def test_delete_runs_empty_collection_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_one_run(tmp_path, "sess-a")

    def _fail_open(archive_dir: Path) -> sqlite3.Connection:
        raise AssertionError("delete_runs must not open the database for an empty collection")

    monkeypatch.setattr("daydream.archive.index._get_connection", _fail_open)

    assert delete_runs(tmp_path, []) == 0

    monkeypatch.undo()
    assert len(query_runs(tmp_path)) == 1


def test_delete_runs_coerces_non_string_members(tmp_path: Path) -> None:
    _seed_one_run(tmp_path, "42")

    assert delete_runs(tmp_path, [42]) == 1
    assert query_runs(tmp_path) == []


def test_delete_runs_matches_exactly_no_like_semantics(tmp_path: Path) -> None:
    _seed_one_run(tmp_path, "sess-a")
    _seed_one_run(tmp_path, "sess-a%")  # LIKE wildcard sibling must survive
    _seed_one_run(tmp_path, "sess-a_x")  # LIKE single-char wildcard sibling

    assert delete_runs(tmp_path, ["sess-a"]) == 1
    remaining = {r["session_id"] for r in query_runs(tmp_path)}
    assert remaining == {"sess-a%", "sess-a_x"}


def test_delete_runs_hydration_rerun_reflects_only_kept_session(tmp_path: Path) -> None:
    # Prior hydration run admitted both sessions.
    _seed_one_run(tmp_path, "sess-kept")
    _seed_one_run(tmp_path, "sess-rejected")
    assert len(query_runs(tmp_path)) == 2

    # Rerun admission: prune the rejected session's index row.
    deleted = delete_runs(tmp_path, ["sess-rejected"])

    assert deleted == 1
    visible = query_runs(tmp_path)
    assert [r["session_id"] for r in visible] == ["sess-kept"]
    # The kept session's harvest-visible row is fully intact.
    assert visible[0]["status"] == "complete"


def test_delete_runs_removes_bundle_directory_under_runs(tmp_path: Path) -> None:
    # Sibling contract (hydrate/sanitize): the on-disk ``runs/`` tree is the
    # source of truth for ``rebuild_index``, so a surviving bundle directory
    # would silently resurrect the pruned row.
    _seed_one_run(tmp_path, "sess-a")
    runs_root = tmp_path / "runs"
    bundle = runs_root / "sess-a"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text("{}", encoding="utf-8")
    conn = sqlite3.connect(str(tmp_path / "index.db"))
    conn.execute(
        "UPDATE runs SET archive_path = ? WHERE session_id = 'sess-a'",
        (str(bundle),),
    )
    conn.commit()
    conn.close()

    assert delete_runs(tmp_path, ["sess-a"]) == 1
    assert query_runs(tmp_path) == []
    assert not bundle.exists()


def test_delete_runs_leaves_bundle_outside_runs_untouched(tmp_path: Path) -> None:
    _seed_one_run(tmp_path, "sess-a")  # archive_path: archive_dir/sess-a
    (tmp_path / "sess-a").mkdir()

    assert delete_runs(tmp_path, ["sess-a"]) == 1
    assert (tmp_path / "sess-a").is_dir()


def _seed_one_run(archive_dir: Path, session_id: str) -> None:
    upsert_run(
        archive_dir,
        Manifest(
            session_id=session_id,
            archived_at="2026-01-01T00:00:00Z",
            run_flow="normal",
            backend="claude",
            archive_path=str(archive_dir / session_id),
        ),
    )


def test_label_observations_has_bitemporal_reward_columns(tmp_path: Path) -> None:
    upsert_run(tmp_path, make_manifest())  # forces _get_connection to build schema
    conn = sqlite3.connect(str(tmp_path / "index.db"))
    lo_cols = {r[1] for r in conn.execute("PRAGMA table_info(label_observations)")}
    runs_cols = {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
    conn.close()
    assert {"valid_at", "reward_version", "reward_json"} <= lo_cols
    assert "composite_reward" in runs_cols


def test_delete_runs_leaves_label_observations_intact(tmp_path: Path) -> None:
    _seed_one_run(tmp_path, "sess-a")
    append_label_observation(
        tmp_path,
        "sess-a",
        labels=["rejected"],
        pr_state=None,
        labeler_version="v1",
        evidence_sha=None,
    )

    assert delete_runs(tmp_path, ["sess-a"]) == 1
    assert query_runs(tmp_path) == []
    history = label_observation_history(tmp_path, "sess-a")
    assert len(history) == 1
    assert json.loads(history[0]["labels"]) == ["rejected"]


_OLD_LABEL_OBSERVATIONS_DDL = """
CREATE TABLE IF NOT EXISTS label_observations (
    session_id       TEXT NOT NULL,
    observed_at      TEXT NOT NULL,
    labels           TEXT NOT NULL,
    pr_state         TEXT,
    labeler_version  TEXT NOT NULL,
    evidence_sha     TEXT,
    rubric_json      TEXT,
    valid_at         TEXT,
    reward_version   TEXT,
    reward_json      TEXT,
    composite_reward REAL,
    reviewer_logins  TEXT,
    has_posterior    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (session_id, observed_at)
)
"""


def _label_obs_columns(archive_dir: Path) -> set[str]:
    conn = sqlite3.connect(str(archive_dir / "index.db"))
    try:
        return {r[1] for r in conn.execute("PRAGMA table_info(label_observations)")}
    finally:
        conn.close()


def _seed_legacy_label_observation(archive_dir: Path, session_id: str) -> None:
    """Insert a label_observations row using the OLD DDL that lacks ``source``."""
    conn = sqlite3.connect(str(archive_dir / "index.db"))
    try:
        conn.execute("DROP TABLE IF EXISTS label_observations")
        conn.execute(_OLD_LABEL_OBSERVATIONS_DDL)
        conn.execute(
            "INSERT INTO label_observations "
            "(session_id, observed_at, labels, pr_state, labeler_version, evidence_sha) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                session_id,
                "2026-01-01T00:00:00+00:00",
                '["accepted"]',
                "merged",
                "v1",
                "sha1",
            ),
        )
        conn.commit()
    finally:
        conn.close()


def test_label_observations_source_column_migrates(tmp_path: Path) -> None:
    # Build schema, then replace the table with the OLD DDL (no `source`) + a legacy row.
    upsert_run(tmp_path, make_manifest(session_id="s-mig"))
    _seed_legacy_label_observation(tmp_path, "s-mig")
    assert "source" not in _label_obs_columns(tmp_path)  # precondition: legacy shape

    # The production connection path must ALTER-ADD `source`.
    upsert_run(tmp_path, make_manifest(session_id="s-mig2"))

    cols = _label_obs_columns(tmp_path)
    assert "source" in cols
    rows = label_observation_history(tmp_path, "s-mig")
    assert rows and rows[0]["source"] == "auto"  # existing row defaulted, non-destructive


def test_human_label_wins_over_newer_auto_in_projection(tmp_path: Path) -> None:
    upsert_run(tmp_path, make_manifest(session_id="s-prec"))
    append_label_observation(tmp_path, "s-prec", labels=["rejected"], pr_state="closed",
                             labeler_version="auto-v1", evidence_sha="sha1", source="auto")
    append_label_observation(tmp_path, "s-prec", labels=["accepted"], pr_state=None,
                             labeler_version="human", evidence_sha=None, source="human")
    # A NEWER auto observation must NOT dethrone the human label:
    append_label_observation(tmp_path, "s-prec", labels=["rejected"], pr_state="closed",
                             labeler_version="auto-v2", evidence_sha="sha2", source="auto")
    prec_obs = latest_label_observation(tmp_path, "s-prec")
    assert prec_obs is not None
    assert prec_obs["labels"] == '["accepted"]'
    assert bulk_latest_label_observations(tmp_path, ["s-prec"])["s-prec"]["labels"] == '["accepted"]'
    assert label_count_summary(tmp_path) == {"accepted": 1}


def test_append_cache_reflects_winning_human_label(tmp_path: Path) -> None:
    upsert_run(tmp_path, make_manifest(session_id="s-cache"))
    append_label_observation(tmp_path, "s-cache", labels=["rejected"], pr_state="closed",
                             labeler_version="auto-v1", evidence_sha="sha1", source="auto")
    append_label_observation(tmp_path, "s-cache", labels=["accepted"], pr_state=None,
                             labeler_version="human", evidence_sha=None, source="human")
    # A later auto append must leave the denormalized runs cache on the human label:
    append_label_observation(tmp_path, "s-cache", labels=["rejected"], pr_state="closed",
                             labeler_version="auto-v2", evidence_sha="sha2", source="auto")
    row = query_runs(tmp_path, "session_id = ?", ("s-cache",))[0]
    assert row["outcome_labels"] == '["accepted"]'


def test_auto_append_dedups_on_unchanged_evidence(tmp_path: Path) -> None:
    upsert_run(tmp_path, make_manifest(session_id="s-dedup"))
    first = append_label_observation(tmp_path, "s-dedup", labels=["accepted"], pr_state="merged",
                                     labeler_version="rv1", evidence_sha="shaA", source="auto")
    second = append_label_observation(tmp_path, "s-dedup", labels=["accepted"], pr_state="merged",
                                      labeler_version="rv1", evidence_sha="shaA", source="auto")
    assert first is True and second is False
    assert len(label_observation_history(tmp_path, "s-dedup")) == 1
    # A labeler_policy_version change DOES append: the M14 auto-dedup tuple is
    # (evidence_sha, labeler_policy_version, reply_evidence_digest, labels,
    # has_posterior, reward_version), so this append fires on the
    # labeler_version bump:
    third = append_label_observation(tmp_path, "s-dedup", labels=["accepted"], pr_state="merged",
                                     labeler_version="rv2", evidence_sha="shaA",
                                     reward_version="rv2", source="auto")
    assert third is True
    assert len(label_observation_history(tmp_path, "s-dedup")) == 2
    # An independent reward_version bump (identical evidence AND policy) also
    # appends: reward_version is part of the M14 tuple, so the freshly
    # computed reward_json/composite_reward must not be silently discarded.
    fourth = append_label_observation(tmp_path, "s-dedup", labels=["accepted"], pr_state="merged",
                                      labeler_version="rv2", evidence_sha="shaA",
                                      reward_version="rv3", source="auto")
    assert fourth is True
    assert len(label_observation_history(tmp_path, "s-dedup")) == 3


def test_auto_append_appends_when_only_has_posterior_changes(tmp_path: Path) -> None:
    """A re-score that moves a row out of the posterior population must append.

    ``has_posterior`` is no longer a function of the label: a ``local_branch``
    outcome carries a label but is not maintainer-PR evidence. If the
    idempotency key ignored it, a re-harvest that demotes such a row would
    silently no-op and leave the stale population flag in place.
    """
    upsert_run(tmp_path, make_manifest(session_id="s-pop"))
    first = append_label_observation(tmp_path, "s-pop", labels=["accepted"], pr_state=None,
                                     labeler_version="rv1", evidence_sha="shaA",
                                     reward_version="rv1", has_posterior=True, source="auto")
    demoted = append_label_observation(tmp_path, "s-pop", labels=["accepted"], pr_state=None,
                                       labeler_version="rv1", evidence_sha="shaA",
                                       reward_version="rv1", has_posterior=False, source="auto")
    assert first is True and demoted is True
    assert len(label_observation_history(tmp_path, "s-pop")) == 2
    latest = latest_label_observation(tmp_path, "s-pop")
    assert latest is not None and latest["has_posterior"] == 0
    # Still idempotent once the demotion has landed.
    assert append_label_observation(tmp_path, "s-pop", labels=["accepted"], pr_state=None,
                                    labeler_version="rv1", evidence_sha="shaA",
                                    reward_version="rv1", has_posterior=False,
                                    source="auto") is False


def test_human_append_never_dedups(tmp_path: Path) -> None:
    upsert_run(tmp_path, make_manifest(session_id="s-h"))
    append_label_observation(
        tmp_path,
        "s-h",
        labels=["accepted"],
        pr_state=None,
        labeler_version="human",
        evidence_sha=None,
        source="human",
    )
    append_label_observation(
        tmp_path,
        "s-h",
        labels=["accepted"],
        pr_state=None,
        labeler_version="human",
        evidence_sha=None,
        source="human",
    )
    assert len(label_observation_history(tmp_path, "s-h")) == 2


def test_append_observation_persists_valid_at_and_reward(tmp_path: Path) -> None:
    upsert_run(tmp_path, make_manifest(session_id="s1"))
    append_label_observation(
        tmp_path, "s1", labels=["accepted"], pr_state="merged",
        labeler_version="v1", evidence_sha=None,
        valid_at="2026-01-02T00:00:00+00:00",
        reward_version="r1", reward_json='{"composite":0.5}', composite_reward=0.5,
    )
    obs = latest_label_observation(tmp_path, "s1")
    assert obs is not None
    assert obs["valid_at"] == "2026-01-02T00:00:00+00:00"
    assert obs["reward_version"] == "r1"
    assert query_runs(tmp_path, "session_id = ?", ("s1",))[0]["composite_reward"] == 0.5


def test_append_observation_defaults_valid_at_to_observed_at(tmp_path: Path) -> None:
    upsert_run(tmp_path, make_manifest(session_id="s2"))
    append_label_observation(
        tmp_path,
        "s2",
        labels=[],
        pr_state=None,
        labeler_version="v1",
        evidence_sha=None,
        valid_at=None,
    )
    obs = latest_label_observation(tmp_path, "s2")
    assert obs is not None
    assert obs["valid_at"] == obs["observed_at"]   # Q2 collapse for local runs


def test_append_label_observation_writes_history_row(tmp_path: Path) -> None:
    _seed_one_run(tmp_path, "sess-1")
    append_label_observation(
        tmp_path,
        "sess-1",
        labels=["accepted"],
        pr_state="merged",
        labeler_version="2026.05.22",
        evidence_sha="abc123",
    )
    hist = label_observation_history(tmp_path, "sess-1")
    assert len(hist) == 1
    assert json.loads(hist[0]["labels"]) == ["accepted"]
    assert hist[0]["pr_state"] == "merged"


def test_append_label_observation_writes_through_to_runs_cache(tmp_path: Path) -> None:
    """The denormalized runs.outcome_labels cache is refreshed on append."""
    _seed_one_run(tmp_path, "sess-2")
    append_label_observation(
        tmp_path,
        "sess-2",
        labels=["contested"],
        pr_state="merged",
        labeler_version="2026.05.22",
        evidence_sha=None,
    )
    rows = query_runs(tmp_path, "session_id = ?", ("sess-2",))
    assert json.loads(rows[0]["outcome_labels"]) == ["contested"]
    assert rows[0]["labeled_at"] is not None


def test_multiple_observations_preserve_history(tmp_path: Path) -> None:
    """Same-session multiple observations all persist; latest wins for the cache."""
    _seed_one_run(tmp_path, "sess-3")
    append_label_observation(
        tmp_path,
        "sess-3",
        labels=["unknown"],
        pr_state="open",
        labeler_version="v1",
        evidence_sha=None,
    )
    append_label_observation(
        tmp_path,
        "sess-3",
        labels=["accepted"],
        pr_state="merged",
        labeler_version="v1",
        evidence_sha="def456",
    )
    hist = label_observation_history(tmp_path, "sess-3")
    assert len(hist) == 2
    assert [json.loads(r["labels"])[0] for r in hist] == ["unknown", "accepted"]
    latest = latest_label_observation(tmp_path, "sess-3")
    assert latest is not None
    assert json.loads(latest["labels"]) == ["accepted"]
    rows = query_runs(tmp_path, "session_id = ?", ("sess-3",))
    assert json.loads(rows[0]["outcome_labels"]) == ["accepted"]


def test_latest_label_observation_filtered_by_as_of(tmp_path: Path) -> None:
    """Snapshot pinning: latest_label_observation(..., as_of=ts) returns the
    latest observation whose observed_at <= as_of."""
    _seed_one_run(tmp_path, "sess-4")
    append_label_observation(
        tmp_path,
        "sess-4",
        labels=["unknown"],
        pr_state="open",
        labeler_version="v1",
        evidence_sha=None,
    )
    early_row = latest_label_observation(tmp_path, "sess-4")
    assert early_row is not None
    early = early_row["observed_at"]
    append_label_observation(
        tmp_path,
        "sess-4",
        labels=["accepted"],
        pr_state="merged",
        labeler_version="v1",
        evidence_sha="def456",
    )
    pinned = latest_label_observation(tmp_path, "sess-4", as_of=early)
    assert pinned is not None
    assert json.loads(pinned["labels"]) == ["unknown"]


def test_same_microsecond_collision_keeps_clean_iso_timestamps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two appends frozen to the same microsecond must both persist with parseable
    ISO 8601 observed_at values, and an exact-boundary as_of must include the
    boundary row (the contract the ~uuid suffix used to break)."""
    from datetime import datetime, timezone

    frozen = datetime(2026, 5, 29, 12, 0, 0, tzinfo=timezone.utc)

    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz: Any=None) -> Any:
            return frozen if tz is None else frozen.astimezone(tz)

    monkeypatch.setattr("daydream.archive.index.datetime", _FrozenDatetime)

    _seed_one_run(tmp_path, "sess-collide")
    append_label_observation(
        tmp_path, "sess-collide", labels=["unknown"], pr_state="open",
        labeler_version="v1", evidence_sha="a",
    )
    append_label_observation(
        tmp_path, "sess-collide", labels=["accepted"], pr_state="merged",
        labeler_version="v1", evidence_sha="b",
    )

    hist = label_observation_history(tmp_path, "sess-collide")
    assert len(hist) == 2
    stamps = [r["observed_at"] for r in hist]
    assert stamps[0] != stamps[1]
    for r in hist:
        datetime.fromisoformat(r["observed_at"])  # parseable, no ~uuid suffix
        assert r["valid_at"] == r["observed_at"]

    runs_row = query_runs(tmp_path, "session_id = ?", ("sess-collide",))[0]
    datetime.fromisoformat(runs_row["labeled_at"])

    boundary = stamps[0]
    pinned = latest_label_observation(tmp_path, "sess-collide", as_of=boundary)
    assert pinned is not None
    assert json.loads(pinned["labels"]) == ["unknown"]  # boundary row included


def test_append_label_observation_persists_reviewer_and_posterior_flag(
    tmp_path: Path,
) -> None:
    """reviewer_logins + has_posterior persist on the observation row and mirror onto runs."""
    _seed_one_run(tmp_path, "s1")
    append_label_observation(
        tmp_path,
        "s1",
        labels=["rejected"],
        pr_state="closed",
        labeler_version="2026.05.28-1",
        evidence_sha="h",
        reviewer_logins=["alice"],
        has_posterior=True,
    )
    obs = latest_label_observation(tmp_path, "s1")
    assert obs is not None
    assert json.loads(obs["reviewer_logins"]) == ["alice"]
    assert obs["has_posterior"] == 1
    runs_row = query_runs(tmp_path, "session_id = ?", ("s1",))[0]
    assert runs_row["has_posterior"] == 1  # SQL consumers split populations without parsing reward_json


def test_existing_db_migrates_to_posterior_columns(tmp_path: Path) -> None:
    """A pre-v4 index.db (runs + label_observations lacking the posterior columns)
    is migrated/recreated on the next connection: runs gains has_posterior via
    ALTER, the stale label_observations is dropped+recreated with both new
    columns, and PRAGMA user_version reaches SCHEMA_VERSION (8)."""
    from daydream.archive.index import _CREATE_TABLE, SCHEMA_VERSION

    db_path = tmp_path / "index.db"
    conn = sqlite3.connect(str(db_path))
    # Pre-v4 runs schema (DDL minus has_posterior); label_observations lacks posterior cols.
    pre_v4_runs_ddl = _CREATE_TABLE.replace(
        "    has_posterior INTEGER NOT NULL DEFAULT 0,\n", ""
    )
    assert "has_posterior" not in pre_v4_runs_ddl
    conn.execute(pre_v4_runs_ddl)
    conn.execute(
        "CREATE TABLE label_observations ("
        "session_id TEXT NOT NULL, observed_at TEXT NOT NULL, labels TEXT NOT NULL, "
        "pr_state TEXT, labeler_version TEXT NOT NULL, evidence_sha TEXT, rubric_json TEXT, "
        "valid_at TEXT, reward_version TEXT, reward_json TEXT, composite_reward REAL, "
        "PRIMARY KEY (session_id, observed_at))"
    )
    conn.execute(
        "INSERT INTO runs (session_id, archived_at, run_flow, archive_path) VALUES (?, ?, ?, ?)",
        ("mig-1", "2026-01-01T00:00:00Z", "normal", str(tmp_path / "mig-1")),
    )
    conn.execute("PRAGMA user_version = 3")
    conn.commit()
    conn.close()

    # First real-path write triggers _migrate_schema + the drop-and-recreate warning.
    with pytest.warns(UserWarning, match="predates bitemporal/posterior columns"):
        append_label_observation(
            tmp_path,
            "mig-1",
            labels=["accepted"],
            pr_state="merged",
            labeler_version="2026.05.28-1",
            evidence_sha=None,
            reviewer_logins=["bob"],
            has_posterior=True,
        )

    conn = sqlite3.connect(str(db_path))
    runs_cols = {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
    lo_cols = {r[1] for r in conn.execute("PRAGMA table_info(label_observations)")}
    user_version = conn.execute("PRAGMA user_version").fetchone()[0]
    conn.close()
    assert "has_posterior" in runs_cols
    assert {"reviewer_logins", "has_posterior"} <= lo_cols
    assert user_version == SCHEMA_VERSION == 8

    obs = latest_label_observation(tmp_path, "mig-1")
    assert obs is not None
    assert json.loads(obs["reviewer_logins"]) == ["bob"]
    assert obs["has_posterior"] == 1
    assert query_runs(tmp_path, "session_id = ?", ("mig-1",))[0]["has_posterior"] == 1


# ISO 8601 valid times stored verbatim in label_observations.valid_at and
# compared lexically with a strict ``<`` cutoff; T1 < T2 < T3 lexically.
T1 = "2026-01-01T00:00:00+00:00"
T2 = "2026-02-01T00:00:00+00:00"
T3 = "2026-03-01T00:00:00+00:00"


def _seed_reviewed_outcomes(archive_dir: Path) -> None:
    """Seed three prior runs (one reviewed outcome each) plus a current run.

    - s_a: reviewers=[alice], rejected (penalty 1.0) @ T1
    - s_b: reviewers=[bob],   accepted (penalty 0.0) @ T2
    - s_c: reviewers=[alice, carol], contested (penalty 0.5) @ T3
    - cur: the current session (excluded from its own prior pool)
    """
    for sid in ("s_a", "s_b", "s_c", "cur"):
        _seed_one_run(archive_dir, sid)
    append_label_observation(
        archive_dir, "s_a", labels=["rejected"], pr_state="closed",
        labeler_version="2026.05.28-1", evidence_sha=None,
        valid_at=T1, reviewer_logins=["alice"], has_posterior=True,
    )
    append_label_observation(
        archive_dir, "s_b", labels=["accepted"], pr_state="merged",
        labeler_version="2026.05.28-1", evidence_sha=None,
        valid_at=T2, reviewer_logins=["bob"], has_posterior=True,
    )
    append_label_observation(
        archive_dir, "s_c", labels=["contested"], pr_state="merged",
        labeler_version="2026.05.28-1", evidence_sha=None,
        valid_at=T3, reviewer_logins=["alice", "carol"], has_posterior=True,
    )


def test_reviewer_set_penalty_prior_pools_shared_reviewer_runs_strict_cutoff(
    tmp_path: Path,
) -> None:
    # Current reviewers={alice}, valid_at==t3 -> pool = alice-sharing runs, valid_at < t3:
    # only s_a (s_c @ t3 excluded by strict <; bob's run shares no reviewer).
    _seed_reviewed_outcomes(tmp_path)
    prior, n = reviewer_set_penalty_prior(tmp_path, ["alice"], before_valid_at=T3, exclude_session="cur")
    assert prior == pytest.approx(1.0) and n == 1
    # widen the set to {alice,bob}: pool now includes s_a(1.0) + s_b(0.0) -> mean 0.5, n=2
    prior2, n2 = reviewer_set_penalty_prior(tmp_path, ["alice", "bob"], before_valid_at=T3, exclude_session="cur")
    assert prior2 == pytest.approx(0.5) and n2 == 2
    # empty reviewer set -> no pool
    assert reviewer_set_penalty_prior(tmp_path, [], before_valid_at=T3, exclude_session="cur") == (None, 0)


def test_reviewer_set_penalty_prior_scoped_to_repo(tmp_path: Path) -> None:
    # Two alice rows in distinct repos (s_a: repo-A rejected@T1; s_b: repo-B accepted@T2)
    # verify per-repo filtering. cur has no repo_slug, excluded by session_id.
    for sid, slug in (("s_a", "org/repo-A"), ("s_b", "org/repo-B"), ("cur", None)):
        upsert_run(
            tmp_path,
            Manifest(
                session_id=sid,
                archived_at="2026-01-01T00:00:00Z",
                run_flow="normal",
                backend="claude",
                repo_slug=slug,
                archive_path=str(tmp_path / sid),
            ),
        )
    append_label_observation(
        tmp_path, "s_a", labels=["rejected"], pr_state="closed",
        labeler_version="2026.05.28-1", evidence_sha=None,
        valid_at=T1, reviewer_logins=["alice"], has_posterior=True,
    )
    append_label_observation(
        tmp_path, "s_b", labels=["accepted"], pr_state="merged",
        labeler_version="2026.05.28-1", evidence_sha=None,
        valid_at=T2, reviewer_logins=["alice"], has_posterior=True,
    )

    # Without repo scoping both alice rows are pooled: mean(1.0, 0.0) = 0.5, n=2
    prior_all, n_all = reviewer_set_penalty_prior(
        tmp_path, ["alice"], before_valid_at=T3, exclude_session="cur"
    )
    assert prior_all == pytest.approx(0.5) and n_all == 2

    # Scoped to org/repo-A: only s_a(rejected,1.0) qualifies
    prior_a, n_a = reviewer_set_penalty_prior(
        tmp_path, ["alice"], before_valid_at=T3, exclude_session="cur",
        repo_slug="org/repo-A",
    )
    assert prior_a == pytest.approx(1.0) and n_a == 1

    # Scoped to org/repo-B: only s_b(accepted,0.0) qualifies
    prior_b, n_b = reviewer_set_penalty_prior(
        tmp_path, ["alice"], before_valid_at=T3, exclude_session="cur",
        repo_slug="org/repo-B",
    )
    assert prior_b == pytest.approx(0.0) and n_b == 1

    # Scoped to an unknown repo: empty pool
    prior_x, n_x = reviewer_set_penalty_prior(
        tmp_path, ["alice"], before_valid_at=T3, exclude_session="cur",
        repo_slug="org/other",
    )
    assert (prior_x, n_x) == (None, 0)


def test_manifest_includes_source_path() -> None:
    """source_path appears in manifest dict under git section."""
    m = Manifest(
        session_id="test-session",
        source_path="/home/user/code/myrepo",
        remote_url="git@github.com:org/repo.git",
        repo_slug="org/repo",
    )
    d = m.to_dict()
    assert d["git"]["source_path"] == "/home/user/code/myrepo"


def test_source_path_indexed_in_sqlite(tmp_path: Path) -> None:
    """source_path round-trips through upsert_run → query_runs."""
    idx_dir = tmp_path / "idx"
    idx_dir.mkdir()
    m = Manifest(
        session_id="sp-test",
        archived_at="2026-01-01T00:00:00Z",
        run_flow="normal",
        backend="claude",
        source_path="/original/repo/path",
        archive_path=str(tmp_path),
    )
    upsert_run(idx_dir, m)
    rows = query_runs(idx_dir)
    assert rows[0]["source_path"] == "/original/repo/path"


def test_source_path_defaults_to_none() -> None:
    """Old manifests without source_path still work."""
    m = Manifest(session_id="old")
    assert m.source_path is None
    assert m.to_dict()["git"]["source_path"] is None


def test_update_labels_is_backward_compat_thin_wrapper(tmp_path: Path) -> None:
    """The legacy update_labels() now writes through append_label_observation
    so existing callers continue to work without source changes."""
    _seed_one_run(tmp_path, "sess-5")
    assert update_labels(tmp_path, "sess-5", ["accepted"]) is True
    hist = label_observation_history(tmp_path, "sess-5")
    assert len(hist) == 1
    rows = query_runs(tmp_path, "session_id = ?", ("sess-5",))
    assert json.loads(rows[0]["outcome_labels"]) == ["accepted"]


# Canonical UTC timestamp contract: one spelling at write time, strict as_of
# validation at the entry boundary, and legacy "Z" rows preserved at bootstrap.


def test_canonical_utc_iso_converts_and_rejects() -> None:
    assert canonical_utc_iso("2026-02-01T00:00:00Z") == "2026-02-01T00:00:00+00:00"
    assert canonical_utc_iso("2026-02-01T00:00:00+00:00") == "2026-02-01T00:00:00+00:00"
    # A foreign offset is an unambiguous instant — converted, not rejected.
    assert canonical_utc_iso("2026-02-01T05:30:00+05:30") == "2026-02-01T00:00:00+00:00"
    # Sub-second precision survives canonically (six digits or absent).
    assert canonical_utc_iso("2026-02-01T00:00:00.500000Z") == "2026-02-01T00:00:00.500000+00:00"
    with pytest.raises(ValueError, match="naive"):
        canonical_utc_iso("2026-02-01T00:00:00")
    with pytest.raises(ValueError):
        canonical_utc_iso("not-a-timestamp")


def test_normalize_as_of_is_strict_utc_only() -> None:
    assert normalize_as_of("2026-04-01T00:00:00Z") == "2026-04-01T00:00:00+00:00"
    assert normalize_as_of("2026-04-01T00:00:00+00:00") == "2026-04-01T00:00:00+00:00"
    with pytest.raises(ValueError, match="must be a UTC timestamp"):
        normalize_as_of("2026-04-01T05:00:00+05:00")
    with pytest.raises(ValueError, match="must be a UTC timestamp"):
        normalize_as_of("2026-04-01T00:00:00")
    with pytest.raises(ValueError, match="not a valid ISO-8601"):
        normalize_as_of("yesterday")


def test_append_label_observation_canonicalizes_valid_at_spelling(
    tmp_path: Path,
) -> None:
    """The write chokepoint converges every caller (GitHub 'Z' merge timestamps
    included) on the '+00:00' isoformat spelling."""
    _seed_one_run(tmp_path, "sess-z")
    append_label_observation(
        tmp_path, "sess-z", labels=["accepted"], pr_state="merged",
        labeler_version="v1", evidence_sha=None,
        valid_at="2026-02-01T00:00:00Z",
    )
    row = latest_label_observation(tmp_path, "sess-z")
    assert row is not None
    assert row["valid_at"] == "2026-02-01T00:00:00+00:00"


def test_append_label_observation_rejects_naive_valid_at(tmp_path: Path) -> None:
    _seed_one_run(tmp_path, "sess-naive")
    with pytest.raises(ValueError, match="naive"):
        append_label_observation(
            tmp_path, "sess-naive", labels=["accepted"], pr_state="merged",
            labeler_version="v1", evidence_sha=None,
            valid_at="2026-02-01T00:00:00",
        )


def test_reviewer_prior_bound_spelling_cannot_misorder(tmp_path: Path) -> None:
    """A 'Z'-spelled before_valid_at bound is canonicalized before the lexical
    SQL cutoff, so it can never mis-order against the '+00:00' stored column.

    Chronology: the pooled row's valid_at is 0.5s AFTER the bound instant, so
    the strict `valid_at < bound` must exclude it. Raw lexical comparison of
    the mixed spellings ("...00.500000+00:00" < "...00Z") would wrongly
    include it.
    """
    _seed_one_run(tmp_path, "s_late")
    _seed_one_run(tmp_path, "cur")
    append_label_observation(
        tmp_path, "s_late", labels=["rejected"], pr_state="closed",
        labeler_version="v1", evidence_sha=None,
        valid_at="2026-03-01T00:00:00.500000+00:00",
        reviewer_logins=["alice"], has_posterior=True,
    )
    prior, n = reviewer_set_penalty_prior(
        tmp_path,
        ["alice"],
        before_valid_at="2026-03-01T00:00:00Z",
        exclude_session="cur",
    )
    assert (prior, n) == (None, 0)
    # And a bound safely after the row still pools it, regardless of spelling.
    prior2, n2 = reviewer_set_penalty_prior(
        tmp_path,
        ["alice"],
        before_valid_at="2026-03-01T00:00:01Z",
        exclude_session="cur",
    )
    assert prior2 == pytest.approx(1.0) and n2 == 1


def test_legacy_z_valid_at_rows_are_left_untouched(tmp_path: Path) -> None:
    """A pre-convergence 'Z'-spelled row survives reconnection unchanged: the
    index never deletes or rewrites history at bootstrap (a destructive
    migration in a library code path would silently eat other users' data)."""
    _seed_one_run(tmp_path, "sess-legacy")
    conn = sqlite3.connect(str(tmp_path / "index.db"))
    conn.execute(
        "INSERT INTO label_observations "
        "(session_id, observed_at, labels, labeler_version, valid_at, source) "
        "VALUES ('sess-legacy', '2026-01-01T00:00:00+00:00', '[\"accepted\"]', 'v0', "
        "'2026-01-01T00:00:00Z', 'auto')"
    )
    conn.commit()
    conn.close()

    hist = label_observation_history(tmp_path, "sess-legacy")
    assert [r["valid_at"] for r in hist] == ["2026-01-01T00:00:00Z"]












async def test_build_manifest_totals_include_fork_trajectories(tmp_path: Path) -> None:
    """Manifest totals are whole-run: the fork's tokens/cost are folded in."""
    from daydream.backends import MetricsEvent, ResultEvent, TextEvent
    from daydream.trajectory import DaydreamPhase, DaydreamRunFlow

    snapshots: list[RunWriteSnapshot] = []
    recorder = TrajectoryRecorder(
        path=tmp_path / ".daydream" / "runs" / "sess-fold" / "trajectory.json",
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=tmp_path,
        agent_model_name="opus",
        session_id="sess-fold",
        on_write=lambda _recorder, snapshot: snapshots.append(snapshot),
    )
    async with recorder:
        async with recorder.invocation(phase=DaydreamPhase.REVIEW) as inv:
            inv.observe(TextEvent(text="parent"))
            inv.observe(MetricsEvent(
                message_id="m-1", prompt_tokens=100, completion_tokens=50,
                cached_tokens=20, cost_usd=0.05,
            ))
            inv.observe(ResultEvent(structured_output=None, continuation=None))
        async with recorder.fork("deep-python") as child:
            async with child.invocation(phase=DaydreamPhase.DEEP) as cinv:
                cinv.observe(TextEvent(text="child"))
                cinv.observe(MetricsEvent(
                    message_id="m-2", prompt_tokens=400, completion_tokens=25,
                    cached_tokens=5, cost_usd=0.20,
                ))
                cinv.observe(ResultEvent(structured_output=None, continuation=None))

    write_snapshot = snapshots[-1]
    m = build_manifest_from_snapshot(
        run=_archive_snapshot(write_snapshot, run_flow=recorder.run_flow),
        git_ctx=GitContext(),
        status="complete",
        archive_path=tmp_path,
    )

    assert m.total_prompt_tokens == 500  # 100 main + 400 fork
    assert m.total_completion_tokens == 75
    assert m.total_cached_tokens == 25
    assert m.total_cost_usd == pytest.approx(0.25)








def test_manifest_splits_status_from_pipeline() -> None:
    from daydream.archive.provenance import ExecutableProvenance
    m = Manifest(
        session_id="s-1", status="complete", archive_status="complete",
        pipeline_status="failed", phase_states={
            "merge": {"ran": True, "status": "failed"},
            "fix": {"ran": False, "status": "absent"},
            "test": {"ran": False, "status": "absent"},
        },
        daydream=ExecutableProvenance(
            version="0.27.0",
            install_source="git",
            commit="abc",
            dirty=False,
            container_digest="unknown",
        ),
    )
    d = m.to_dict()
    assert d["status"] == "complete"
    assert d["archive_status"] == "complete"
    assert d["pipeline_status"] == "failed"
    assert d["phase_states"]["merge"]["status"] == "failed"
    # Namespace separation: executable provenance never merged into git.*
    assert d["daydream"]["version"] == "0.27.0"
    assert d["git"]["head_sha"] is None  # target-repo sha stays in git.*
    assert "commit" not in d["git"]


def test_legacy_manifest_reads_new_fields_as_unknown(tmp_path: Path) -> None:
    # A pre-#762 Manifest carries no archive_status/pipeline_status/phase_states/
    # daydream keys. Indexing it and reading it back through the production
    # query_runs path surfaces the new fields as explicit schema-default
    # sentinels (pipeline_status "unknown"), never a KeyError and never a
    # fabricated value.
    upsert_run(tmp_path, Manifest())
    row = query_runs(tmp_path)[0]
    assert row["pipeline_status"] == "unknown"
    assert row["archive_status"] == "complete"
    assert row["daydream_version"] is None


def _write_deep(target: Path, name: str, data: Any) -> None:
    deep = target / ".daydream" / "deep"
    deep.mkdir(parents=True, exist_ok=True)
    (deep / name).write_text(json.dumps(data), encoding="utf-8")


_PUSHED_SHA = "a" * 40
_MERGE_SHA = "b" * 40


def _phase_event(phase: DaydreamPhase, event: str = "phase_start") -> PhaseEvent:
    return PhaseEvent(
        phase=phase,
        event=event,
        timestamp="2026-09-06T12:00:00Z",
    )


def _write_push_verdict(
    target: Path,
    *,
    session_id: str = "current",
    status: str = "succeeded",
    remote: str = "origin",
    branch: str = "feature/remote-ci",
    sha: str = _PUSHED_SHA,
    repository: str | None = "contributor/fork",
) -> None:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "session_id": session_id,
        "status": status,
        "remote": remote,
        "branch": branch,
        "pushed_sha": sha,
        "pushed_repository": repository,
        "started_at": "2026-09-06T12:00:00Z",
        "updated_at": "2026-09-06T12:00:01Z",
    }
    if status == "failed":
        payload["diagnostic"] = "ordinary push rejected"
    _write_deep(target, "push-verdict.json", payload)


def _write_remote_verdict(
    target_dir: Path,
    *,
    status: str = "passed",
    session_id: str = "current",
    advisory: tuple[CIObservation, ...] = (),
) -> None:
    target = RemoteCITarget(
        target_dir=target_dir,
        base_repository="example/project",
        base_ref="main",
        head_repository="contributor/fork",
        head_ref="feature/remote-ci",
        pr_number=42,
        pr_url="https://github.com/example/project/pull/42",
        remote="origin",
        pushed_sha=_PUSHED_SHA,
    )
    binding = PRCIBinding(
        pr_number=42,
        pr_url=target.pr_url,
        base_repository=target.base_repository,
        base_ref=target.base_ref,
        head_repository=target.head_repository,
        head_ref=target.head_ref,
        head_sha=target.pushed_sha,
        merge_sha=_MERGE_SHA,
        state="open",
    )
    required: tuple[CIObservation, ...] = ()
    if status != "no_ci":
        state = cast(
            Any,
            "fail" if status == "failed" else "pending" if status == "pending" else "pass",
        )
        required = (
            CIObservation(
                source="check_run",
                context="Build",
                app_id=10,
                state=state,
                raw_state="failure" if status == "failed" else state,
                url="https://github.com/example/project/actions/runs/7",
                diagnostic=None,
            ),
        )
    policy = (
        RequiredPolicy((), False)
        if status == "no_ci"
        else RequiredPolicy((RequiredContext("Build", 10),), True)
    )
    verdict = RemoteCIVerdict(
        status=cast(Any, status),
        reason=f"remote CI {status}",
        target=target,
        binding=binding,
        policy=policy,
        active_workflow_count=0 if status == "no_ci" else 1,
        evidence_sha=_PUSHED_SHA if status == "no_ci" else _MERGE_SHA,
        required_observations=required,
        advisory_observations=advisory,
        failing_contexts=("Build (app 10)",) if status == "failed" else (),
        pending_contexts=("Build (app 10)",) if status == "pending" else (),
        missing_contexts=("Build (app 10)",) if status == "missing" else (),
        urls=tuple(item.url for item in (*required, *advisory) if item.url),
        diagnostic=None,
        stable_polls=2,
        elapsed_seconds=120 if status == "no_ci" else 20,
    )
    write_remote_ci_verdict(
        target_dir / ".daydream" / "deep" / "remote-ci-verdict.json",
        verdict,
        session_id=session_id,
        poll_count=2,
        started_at="2026-09-06T12:00:00Z",
        updated_at="2026-09-06T12:02:00Z" if status == "no_ci" else "2026-09-06T12:00:20Z",
        discovery_deadline=120,
        completion_deadline=1800,
    )


def _derive_push_remote_states(
    target: Path,
    *,
    session_id: str = "current",
    events: list[PhaseEvent] | None = None,
    pr_repo: str | None = "example/project",
    pr_number: int | None = 42,
) -> dict[str, dict[str, Any]]:
    from daydream.archive import pipeline

    return pipeline.derive_phase_states(
        target,
        phase_events=events or [],
        runs_merge=False,
        runs_fix=False,
        runs_test=True,
        runs_push=True,
        runs_remote_ci=True,
        session_id=session_id,
        pr_repo=pr_repo,
        pr_number=pr_number,
    )


@pytest.mark.parametrize("remote_status", ["passed", "no_ci"])
def test_current_session_test_push_and_remote_success_are_distinct(
    tmp_path: Path, remote_status: str
) -> None:
    from daydream.archive import pipeline

    _write_deep(tmp_path, "test-verdict.json", {"session_id": "current", "passed": True})
    _write_push_verdict(tmp_path)
    _write_remote_verdict(tmp_path, status=remote_status)

    states = _derive_push_remote_states(tmp_path)

    assert states["test"] == {"ran": True, "status": "succeeded"}
    assert states["push"]["status"] == "succeeded"
    assert states["remote_ci"]["status"] == "succeeded"
    assert pipeline.derive_pipeline_status(
        "complete",
        None,
        states,
        runs_test=True,
    ) == "succeeded"


def test_current_session_remote_required_failure_fails_pipeline(tmp_path: Path) -> None:
    from daydream.archive import pipeline

    _write_deep(tmp_path, "test-verdict.json", {"session_id": "current", "passed": True})
    _write_push_verdict(tmp_path)
    _write_remote_verdict(tmp_path, status="failed")

    states = _derive_push_remote_states(tmp_path)

    assert states["remote_ci"]["status"] == "failed"
    assert pipeline.derive_pipeline_status("complete", None, states) == "failed"


def test_archive_cancellation_precedes_remote_failure(tmp_path: Path) -> None:
    from daydream.archive import pipeline

    _write_push_verdict(tmp_path)
    _write_remote_verdict(tmp_path, status="failed")
    states = _derive_push_remote_states(tmp_path)

    assert pipeline.derive_pipeline_status("partial", None, states) == "cancelled"


@pytest.mark.parametrize(
    "remote_status",
    ["pending", "missing", "unavailable", "timed_out", "superseded", "cancelled"],
)
def test_incomplete_remote_statuses_are_partial(tmp_path: Path, remote_status: str) -> None:
    from daydream.archive import pipeline

    _write_push_verdict(tmp_path)
    _write_remote_verdict(tmp_path, status=remote_status)

    states = _derive_push_remote_states(tmp_path)

    assert states["remote_ci"]["status"] == "partial"
    assert pipeline.derive_pipeline_status("complete", None, states) == "partial"


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("session_id",), "prior"),
        (("target", "pushed_sha"), "c" * 40),
        (("target", "remote"), "upstream"),
        (("target", "head_ref"), "other"),
        (("target", "head_repository"), "other/repo"),
        (("binding", "base_repository"), "other/repo"),
        (("binding", "base_ref"), "release"),
        (("binding", "head_repository"), "other/repo"),
        (("binding", "head_ref"), "other"),
        (("binding", "pr_number"), 43),
        (("binding", "pr_url"), "https://github.com/example/project/pull/43"),
        (("binding", "head_sha"), "c" * 40),
        (("policy",), None),
        (("polling", "poll_count"), 0),
        (("active_workflow_count",), True),
        (("required_observations",), {}),
        (("limitations",), None),
        (("failing_contexts",), ["Build (app 10)"]),
    ],
)
def test_remote_success_identity_mismatch_is_partial(
    tmp_path: Path, path: tuple[str, ...], value: object
) -> None:
    _write_push_verdict(tmp_path)
    _write_remote_verdict(tmp_path)
    artifact = tmp_path / ".daydream" / "deep" / "remote-ci-verdict.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    cursor = payload
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    assert _derive_push_remote_states(tmp_path)["remote_ci"]["status"] == "partial"


def test_remote_archive_state_field_is_not_outcome_authority(tmp_path: Path) -> None:
    _write_push_verdict(tmp_path)
    _write_remote_verdict(tmp_path)
    artifact = tmp_path / ".daydream" / "deep" / "remote-ci-verdict.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["archive_state"] = "failed"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    assert _derive_push_remote_states(tmp_path)["remote_ci"]["status"] == "succeeded"


@pytest.mark.parametrize(
    ("repo", "number"),
    [("other/project", 42), ("example/project", 43)],
)
def test_remote_success_must_match_configured_pr(
    tmp_path: Path, repo: str, number: int
) -> None:
    _write_push_verdict(tmp_path)
    _write_remote_verdict(tmp_path)

    assert _derive_push_remote_states(tmp_path, pr_repo=repo, pr_number=number)["remote_ci"][
        "status"
    ] == "partial"


def test_remote_success_accepts_case_insensitive_configured_pr(tmp_path: Path) -> None:
    _write_push_verdict(tmp_path)
    _write_remote_verdict(tmp_path)

    assert _derive_push_remote_states(
        tmp_path,
        pr_repo="ExAmPlE/PrOjEcT",
        pr_number=42,
    )["remote_ci"]["status"] == "succeeded"


@pytest.mark.parametrize(
    ("artifact_name", "path"),
    [
        ("push-verdict.json", ("pushed_repository",)),
        ("remote-ci-verdict.json", ("target", "base_repository")),
        ("remote-ci-verdict.json", ("target", "head_repository")),
        ("remote-ci-verdict.json", ("binding", "base_repository")),
        ("remote-ci-verdict.json", ("binding", "head_repository")),
    ],
)
def test_persisted_repository_identities_remain_canonical_lowercase(
    tmp_path: Path,
    artifact_name: str,
    path: tuple[str, ...],
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

    states = _derive_push_remote_states(
        tmp_path,
        pr_repo="ExAmPlE/PrOjEcT",
        pr_number=42,
    )
    if artifact_name == "push-verdict.json":
        assert states["push"]["status"] == "partial"
        assert states["remote_ci"] == {"ran": False, "status": "absent"}
    else:
        assert states["remote_ci"]["status"] == "partial"


def test_archive_rejects_repository_identity_the_producer_cannot_create(
    tmp_path: Path,
    archive_dir: Path,
    make_config: MakeConfig,
) -> None:
    """Archive success cannot admit a slug rejected by the verdict producer."""
    target = _frozen_target(tmp_path)
    oversized = f"{'a' * 100}/{'b' * 102}"
    with pytest.raises(ValueError, match="repository"):
        RemoteCITarget(
            target_dir=target,
            base_repository=oversized,
            base_ref="main",
            head_repository=oversized,
            head_ref="feature/remote-ci",
            pr_number=42,
            pr_url="https://github.com/example/project/pull/42",
            remote="origin",
            pushed_sha=_PUSHED_SHA,
        )

    recorder = _MockRecorder(session_id="oversized-slug-session")
    _write_deep(
        target,
        "test-verdict.json",
        {"session_id": recorder.session_id, "passed": True},
    )
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

    _strict_archive(
        target=target,
        session_id=recorder.session_id,
        config=make_config(
            target,
            archive=True,
            pr_repo=oversized,
            pr_number=42,
        ),
        write_snapshot=_write_snapshot(
            recorder,
            phase_events=[
                *_merge_events(recorder.session_id, "succeeded"),
                {
                    "phase": "fix",
                    "event": "phase_start",
                    "timestamp": "2026-09-06T12:00:00Z",
                    "session_id": recorder.session_id,
                    "scope_id": "fix-scope",
                },
                *[
                    {
                        "phase": phase.value,
                        "event": "phase_start",
                        "timestamp": "2026-09-06T12:00:00Z",
                        "session_id": recorder.session_id,
                        "scope_id": f"{phase.value}-scope",
                    }
                    for phase in (DaydreamPhase.PUSH, DaydreamPhase.REMOTE_CI)
                ],
            ],
        ),
    )

    manifest = json.loads(
        (archive_dir / "runs" / recorder.session_id / "manifest.json").read_text()
    )
    assert manifest["phase_states"]["remote_ci"] == {
        "ran": True,
        "status": "partial",
    }
    assert manifest["pipeline_status"] == "partial"


@pytest.mark.parametrize("malformed_sha", ["A" * 40, "a" * 39, "a" * 41])
def test_remote_archive_rejects_noncanonical_commit_sha(
    tmp_path: Path,
    malformed_sha: str,
) -> None:
    _write_push_verdict(tmp_path)
    _write_remote_verdict(tmp_path)
    artifact = tmp_path / ".daydream" / "deep" / "remote-ci-verdict.json"
    payload = json.loads(artifact.read_text())
    payload["target"]["pushed_sha"] = malformed_sha
    payload["binding"]["head_sha"] = malformed_sha
    payload["head_sha"] = malformed_sha
    payload["evidence_sha"] = malformed_sha
    artifact.write_text(json.dumps(payload))

    assert _derive_push_remote_states(tmp_path)["remote_ci"] == {
        "ran": True,
        "status": "partial",
    }


def test_push_failure_is_failed_and_no_receipt_fabricates_no_remote(tmp_path: Path) -> None:
    from daydream.archive import pipeline

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
    from daydream.archive import pipeline

    _write_push_verdict(tmp_path)
    states = _derive_push_remote_states(tmp_path)

    assert states["remote_ci"] == {"ran": True, "status": "partial"}
    assert pipeline.derive_pipeline_status("complete", None, states) == "partial"


def test_phase_start_without_terminal_artifact_is_partial(tmp_path: Path) -> None:
    events = [
        _phase_event(DaydreamPhase.PUSH),
        _phase_event(DaydreamPhase.REMOTE_CI),
    ]

    states = _derive_push_remote_states(tmp_path, events=events)

    assert states["push"] == {"ran": True, "status": "partial"}
    assert states["remote_ci"] == {"ran": True, "status": "partial"}


def test_remote_advisory_failure_is_detail_not_hard_failure(tmp_path: Path) -> None:
    advisory = CIObservation(
        source="check_run",
        context="Optional Linux",
        app_id=11,
        state="fail",
        raw_state="failure",
        url="https://github.com/example/project/actions/runs/8",
        diagnostic="optional job failed",
    )
    _write_push_verdict(tmp_path)
    _write_remote_verdict(tmp_path, advisory=(advisory,))

    remote = _derive_push_remote_states(tmp_path)["remote_ci"]

    assert remote["status"] == "succeeded"
    assert remote["details"]["advisory_failures"] == ["Optional Linux"]


def test_archive_required_success_with_pending_advisory_is_succeeded(
    tmp_path: Path,
    archive_dir: Path,
    make_config: MakeConfig,
) -> None:
    """A real archived verdict preserves the evaluator's nonblocking advisory."""
    from daydream.remote_ci import RemoteCILimits, RemoteCISnapshot, evaluate_remote_ci

    target = _frozen_target(tmp_path)
    recorder = _MockRecorder(session_id="advisory-pending-session")
    config = make_config(
        target, archive=True, pr_repo="example/project", pr_number=42
    )
    advisory = CIObservation(
        source="check_run",
        context="Optional Linux",
        app_id=11,
        state="pending",
        raw_state="in_progress",
        url="https://github.com/example/project/actions/runs/8",
        diagnostic=None,
    )
    _write_deep(
        target, "test-verdict.json", {"session_id": recorder.session_id, "passed": True}
    )
    _write_push_verdict(target, session_id=recorder.session_id)
    _write_remote_verdict(
        target, session_id=recorder.session_id, advisory=(advisory,)
    )
    artifact = target / ".daydream" / "deep" / "remote-ci-verdict.json"
    payload = json.loads(artifact.read_text())
    evaluated = evaluate_remote_ci(
        RemoteCISnapshot(
            target=RemoteCITarget(target_dir=target, **payload["target"]),
            binding=PRCIBinding(**payload["binding"]),
            policy=RequiredPolicy((RequiredContext("Build", 10),), True),
            active_workflows=({"id": 7, "state": "active"},),
            head_observations=(),
            merge_observations=(
                CIObservation(**payload["required_observations"][0]), advisory
            ),
        ),
        elapsed=20,
        stable_polls=2,
        limits=RemoteCILimits(),
    )
    assert evaluated.status == "passed"
    write_remote_ci_verdict(
        artifact,
        evaluated,
        session_id=recorder.session_id,
        poll_count=2,
        started_at="2026-09-06T12:00:00Z",
        updated_at="2026-09-06T12:00:20Z",
        discovery_deadline=120,
        completion_deadline=1800,
    )

    _strict_archive(
        target=target,
        session_id=recorder.session_id,
        config=config,
        write_snapshot=_write_snapshot(
            recorder,
            phase_events=[
                *_merge_events(recorder.session_id, "succeeded"),
                {
                    "phase": "fix",
                    "event": "phase_start",
                    "timestamp": "2026-09-06T12:00:00Z",
                    "session_id": recorder.session_id,
                    "scope_id": "fix-scope",
                },
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
    """Without formal required policy, all observed CI must finish first."""
    advisory = CIObservation(
        source="check_run",
        context="Build",
        app_id=10,
        state="pending",
        raw_state="in_progress",
        url="https://github.com/example/project/actions/runs/8",
        diagnostic=None,
    )
    _write_push_verdict(tmp_path)
    _write_remote_verdict(tmp_path, advisory=(advisory,))
    artifact = tmp_path / ".daydream" / "deep" / "remote-ci-verdict.json"
    payload = json.loads(artifact.read_text())
    payload["policy"] = {"contexts": [], "strict": False}
    payload["required_observations"] = []
    artifact.write_text(json.dumps(payload))

    assert _derive_push_remote_states(tmp_path)["remote_ci"]["status"] == "partial"


@pytest.mark.parametrize(
    "corruption",
    [
        "missing-required-observations",
        "missing-one-required-context",
        "wrong-app-pin",
        "wrong-check-case",
        "required-marked-advisory",
        "advisory-marked-required",
        "duplicated-producer",
        "policyless-required-partition",
        "policyless-empty-passed",
        "wrong-failing-context",
    ],
)
def test_archive_rejects_contradictory_terminal_ci_evidence(
    tmp_path: Path, corruption: str
) -> None:
    _write_push_verdict(tmp_path)
    _write_remote_verdict(
        tmp_path,
        status="failed" if corruption == "wrong-failing-context" else "passed",
    )
    artifact = tmp_path / ".daydream" / "deep" / "remote-ci-verdict.json"
    payload = json.loads(artifact.read_text())
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
        payload["required_observations"].append(
            {**payload["required_observations"][0], "context": "Other"}
        )
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
def test_archive_no_ci_retains_empty_strict_policy(
    tmp_path: Path, discovery_seconds: float
) -> None:
    """Strictness alone does not declare a required CI context."""
    from daydream.remote_ci import RemoteCILimits, RemoteCISnapshot, evaluate_remote_ci

    _write_push_verdict(tmp_path)
    _write_remote_verdict(tmp_path, status="no_ci")
    artifact = tmp_path / ".daydream" / "deep" / "remote-ci-verdict.json"
    payload = json.loads(artifact.read_text())
    limits = RemoteCILimits(discovery_seconds=discovery_seconds)
    verdict = evaluate_remote_ci(
        RemoteCISnapshot(
            target=RemoteCITarget(target_dir=tmp_path, **payload["target"]),
            binding=PRCIBinding(**payload["binding"]),
            policy=RequiredPolicy((), True),
            active_workflows=(),
            head_observations=(),
            merge_observations=(),
        ),
        elapsed=discovery_seconds,
        stable_polls=2,
        limits=limits,
    )
    assert verdict.status == "no_ci"
    write_remote_ci_verdict(
        artifact,
        verdict,
        session_id="current",
        poll_count=2,
        started_at="2026-09-06T12:00:00Z",
        updated_at="2026-09-06T12:02:00Z",
        discovery_deadline=discovery_seconds,
        completion_deadline=1800,
        limits=limits,
    )

    assert _derive_push_remote_states(tmp_path)["remote_ci"]["status"] == "succeeded"


@pytest.mark.parametrize(
    ("status", "field", "value"),
    [
        ("no_ci", "elapsed_seconds", 0),
        ("no_ci", "stable_polls", 1),
        ("passed", "stable_polls", 1),
        ("no_ci", "discovery_seconds", -1),
        ("no_ci", "required_stable_polls", True),
    ],
)
def test_archive_terminal_ci_requires_declared_discovery_and_stability(
    tmp_path: Path, status: str, field: str, value: object
) -> None:
    _write_push_verdict(tmp_path)
    _write_remote_verdict(tmp_path, status=status)
    artifact = tmp_path / ".daydream" / "deep" / "remote-ci-verdict.json"
    payload = json.loads(artifact.read_text())
    payload["polling"][field] = value
    artifact.write_text(json.dumps(payload))

    assert _derive_push_remote_states(tmp_path)["remote_ci"]["status"] == "partial"


def test_archive_unpinned_legacy_status_uses_casefolded_context(tmp_path: Path) -> None:
    _write_push_verdict(tmp_path)
    _write_remote_verdict(tmp_path)
    artifact = tmp_path / ".daydream" / "deep" / "remote-ci-verdict.json"
    payload = json.loads(artifact.read_text())
    payload["policy"]["contexts"][0]["app_id"] = None
    payload["required_observations"][0].update(
        source="status", app_id=None, context="BUILD", raw_state="success"
    )
    artifact.write_text(json.dumps(payload))

    assert _derive_push_remote_states(tmp_path)["remote_ci"]["status"] == "succeeded"


@pytest.mark.parametrize(
    ("artifact_name", "field_path", "value"),
    [
        ("push-verdict.json", ("status",), []),
        ("remote-ci-verdict.json", ("status",), {}),
        ("remote-ci-verdict.json", ("required_observations", 0, "state"), []),
        ("remote-ci-verdict.json", ("required_observations", 0, "source"), {}),
        ("remote-ci-verdict.json", ("binding", "state"), "closed"),
    ],
)
def test_archive_malformed_current_ci_fields_fail_closed(
    tmp_path: Path,
    artifact_name: str,
    field_path: tuple[str | int, ...],
    value: object,
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

    assert states["push"]["status"] == (
        "partial" if artifact_name == "push-verdict.json" else "succeeded"
    )
    assert states["remote_ci"]["status"] != "succeeded"


@pytest.mark.parametrize(
    ("artifact", "expected_push"),
    [
        ({"session_id": "current", "status": "succeeded"}, "partial"),
        ({"schema_version": 1, "session_id": "prior", "status": "succeeded"}, "absent"),
    ],
)
def test_push_artifact_is_strictly_current_session_bound(
    tmp_path: Path, artifact: dict[str, Any], expected_push: str
) -> None:
    _write_deep(tmp_path, "push-verdict.json", artifact)

    states = _derive_push_remote_states(tmp_path)

    assert states["push"]["status"] == expected_push
    assert states["remote_ci"] == {"ran": False, "status": "absent"}


def test_unbound_archive_session_cannot_adopt_unbound_push_artifact(
    tmp_path: Path,
) -> None:
    _write_push_verdict(tmp_path)
    artifact = tmp_path / ".daydream" / "deep" / "push-verdict.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    del payload["session_id"]
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    states = _derive_push_remote_states(tmp_path, session_id=cast(Any, None))

    assert states["push"] == {"ran": False, "status": "absent"}
    assert states["remote_ci"] == {"ran": False, "status": "absent"}


@pytest.mark.parametrize(
    ("replacement", "value"),
    [
        ("not-json", None),
        (None, {"schema_version": 1, "session_id": "prior", "status": "passed"}),
    ],
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

    assert _derive_push_remote_states(tmp_path)["remote_ci"] == {
        "ran": True,
        "status": "partial",
    }


@pytest.mark.parametrize(
    ("remote_status", "expected_status"),
    [("passed", "succeeded"), ("failed", "failed")],
)
def test_archive_run_persists_registry_gated_push_and_remote_states(
    tmp_path: Path,
    archive_dir: Path,
    make_config: MakeConfig,
    remote_status: str,
    expected_status: str,
) -> None:
    target = _frozen_target(tmp_path)
    recorder = _MockRecorder(session_id="registry-gated-session")
    config = make_config(
        target,
        archive=True,
        pr_repo="ExAmPlE/PrOjEcT",
        pr_number=42,
    )
    _write_deep(
        target,
        "test-verdict.json",
        {"session_id": recorder.session_id, "passed": True},
    )
    _write_push_verdict(target, session_id=recorder.session_id)
    _write_remote_verdict(
        target, status=remote_status, session_id=recorder.session_id
    )

    _strict_archive(
        target=target,
        session_id=recorder.session_id,
        config=config,
        write_snapshot=_write_snapshot(
            recorder,
            phase_events=[
                *_merge_events(recorder.session_id, "succeeded"),
                {
                    "phase": "fix",
                    "event": "phase_start",
                    "timestamp": "2026-09-06T12:00:00Z",
                    "session_id": recorder.session_id,
                    "scope_id": "fix-scope",
                },
            ],
        ),
    )

    manifest = json.loads(
        (archive_dir / "runs" / recorder.session_id / "manifest.json").read_text()
    )
    assert manifest["phase_states"]["test"]["status"] == "succeeded"
    assert manifest["phase_states"]["push"]["status"] == "succeeded"
    assert manifest["phase_states"]["remote_ci"]["status"] == expected_status
    assert manifest["pipeline_status"] == expected_status


def test_frozen_mapping_push_and_remote_phase_starts_are_partial_without_artifacts(
    tmp_path: Path,
    archive_dir: Path,
    make_config: MakeConfig,
) -> None:
    """Archived mapping rows, rather than live recorder state, drive phase starts.

    The strict finalizer is handed no recorder at all — the live recorder built
    here records no phase event, and the archived states come only from the
    frozen snapshot's mapping rows."""
    from tests.harness.trajectory import make_recorder

    target = _frozen_target(tmp_path)
    recorder = make_recorder(target, run_flow=DaydreamRunFlow.NORMAL)
    phase_events = [
        {
            "phase": phase.value,
            "event": "phase_start",
            "timestamp": "2026-09-06T12:00:00Z",
            "session_id": recorder.session_id,
            "scope_id": f"{phase.value}-scope",
        }
        for phase in (DaydreamPhase.PUSH, DaydreamPhase.REMOTE_CI)
    ]
    assert recorder.phase_event_dicts() == []

    _strict_archive(
        target=target,
        session_id=recorder.session_id,
        config=make_config(
            target,
            archive=True,
            pr_repo="example/project",
            pr_number=42,
        ),
        write_snapshot=_write_snapshot(recorder, phase_events=phase_events),
    )

    manifest = json.loads(
        (archive_dir / "runs" / recorder.session_id / "manifest.json").read_text()
    )
    assert manifest["phase_states"]["push"] == {"ran": True, "status": "partial"}
    assert manifest["phase_states"]["remote_ci"] == {
        "ran": True,
        "status": "partial",
    }
    assert manifest["pipeline_status"] == "partial"


def _merge_events(
    session_id: str,
    status: str,
    *,
    scope_id: str = "merge-scope",
) -> list[dict[str, Any]]:
    return [
        {
            "phase": "merge",
            "event": "phase_start",
            "timestamp": "2026-01-01T00:00:00Z",
            "session_id": session_id,
            "scope_id": scope_id,
        },
        {
            "phase": "merge",
            "event": "phase_end",
            "timestamp": "2026-01-01T00:00:01Z",
            "session_id": session_id,
            "scope_id": scope_id,
            "status": status,
        },
    ]


def test_current_merge_event_succeeds_despite_stale_failure_artifact(
    tmp_path: Path,
) -> None:
    from daydream.archive import pipeline

    _write_deep(tmp_path, "per-stack-failures.json", {"__merge__": {"message": "prior failure"}})

    states = pipeline.derive_phase_states(
        tmp_path,
        phase_events=_merge_events("current", "succeeded"),
        runs_merge=True,
        runs_fix=False,
        runs_test=False,
        session_id="current",
    )

    assert states["merge"] == {"ran": True, "status": "succeeded"}
    assert pipeline.derive_pipeline_status(
        "complete", None, states, runs_merge=True
    ) == "succeeded"


def test_current_merge_event_failure_beats_stale_success_artifact(
    tmp_path: Path,
) -> None:
    from daydream.archive import pipeline

    _write_deep(tmp_path, "merged-items.json", {"items": [{"id": 1}]})

    states = pipeline.derive_phase_states(
        tmp_path,
        phase_events=_merge_events("current", "failed"),
        runs_merge=True,
        runs_fix=False,
        runs_test=False,
        session_id="current",
    )

    assert states["merge"] == {"ran": True, "status": "failed"}
    assert pipeline.derive_pipeline_status(
        "complete", None, states, runs_merge=True
    ) == "failed"


def test_current_merge_event_partial_produces_partial_pipeline(
    tmp_path: Path,
) -> None:
    from daydream.archive import pipeline

    states = pipeline.derive_phase_states(
        tmp_path,
        phase_events=_merge_events("current", "partial"),
        runs_merge=True,
        runs_fix=False,
        runs_test=False,
        session_id="current",
    )

    assert states["merge"] == {"ran": True, "status": "partial"}
    assert pipeline.derive_pipeline_status(
        "complete", None, states, runs_merge=True
    ) == "partial"


def test_stale_merge_event_failure_does_not_override_current_success(
    tmp_path: Path,
) -> None:
    from daydream.archive import pipeline

    states = pipeline.derive_phase_states(
        tmp_path,
        phase_events=[
            *_merge_events("prior", "failed", scope_id="prior-merge"),
            *_merge_events("current", "succeeded"),
        ],
        runs_merge=True,
        runs_fix=False,
        runs_test=False,
        session_id="current",
    )

    assert states["merge"] == {"ran": True, "status": "succeeded"}


@pytest.mark.parametrize(
    "events",
    [
        [_merge_events("current", "succeeded")[1]],
        [*_merge_events("current", "succeeded"), _merge_events("current", "succeeded")[1]],
        [
            {**_merge_events("current", "succeeded")[0], "timestamp": "2026-01-01T00:00:02Z"},
            _merge_events("current", "succeeded")[1],
        ],
        [
            {**_merge_events("current", "succeeded")[0], "timestamp": "2026-01-01T00:00:00"},
            _merge_events("current", "succeeded")[1],
        ],
        [
            {**_merge_events("current", "succeeded")[0], "session_id": None},
            _merge_events("current", "succeeded")[1],
        ],
        [
            _merge_events("current", "succeeded")[0],
            {**_merge_events("current", "succeeded")[1], "status": "not-a-status"},
        ],
    ],
    ids=(
        "orphan",
        "duplicate",
        "reversed",
        "incomparable-timestamps",
        "missing-session",
        "invalid-terminal",
    ),
)
def test_malformed_current_merge_event_is_unknown(
    tmp_path: Path,
    events: list[dict[str, Any]],
) -> None:
    from daydream.archive import pipeline

    states = pipeline.derive_phase_states(
        tmp_path,
        phase_events=events,
        runs_merge=True,
        runs_fix=False,
        runs_test=False,
        session_id="current",
    )

    assert states["merge"] == {"ran": True, "status": "unknown"}
    assert pipeline.derive_pipeline_status(
        "complete", None, states, runs_merge=True
    ) == "unknown"


@pytest.mark.parametrize("malformed_kind", [[], {}], ids=["list", "mapping"])
def test_current_merge_event_rejects_non_scalar_kind(
    tmp_path: Path, malformed_kind: Any,
) -> None:
    from daydream.archive.pipeline import derive_phase_states, derive_pipeline_status

    events = _merge_events("current", "succeeded")
    events[1]["event"] = malformed_kind
    states = derive_phase_states(
        tmp_path, phase_events=events, session_id="current",
        runs_merge=True, runs_fix=False, runs_test=False,
    )
    assert states["merge"] == {"ran": True, "status": "unknown"}
    assert derive_pipeline_status("complete", None, states, runs_merge=True) == "unknown"


def test_missing_current_merge_event_never_uses_stale_success_artifact(
    tmp_path: Path,
) -> None:
    from daydream.archive import pipeline

    _write_deep(tmp_path, "merged-items.json", {"items": [{"id": 1}]})
    states = pipeline.derive_phase_states(
        tmp_path,
        phase_events=_merge_events("prior", "succeeded"),
        runs_merge=True,
        runs_fix=False,
        runs_test=False,
        session_id="current",
    )

    assert states["merge"] == {"ran": False, "status": "absent"}
    assert pipeline.derive_pipeline_status(
        "complete", None, states, runs_merge=True
    ) == "partial"


@pytest.mark.parametrize(
    ("failure_payload", "items_payload", "expected"),
    [
        (None, {"items": []}, {"ran": True, "status": "succeeded"}),
        ({"__merge__": {"message": "failed"}}, {"items": []}, {"ran": True, "status": "failed"}),
        ({"__merge__": "corrupt"}, {"items": []}, {"ran": True, "status": "unknown"}),
        (None, {"items": "corrupt"}, {"ran": True, "status": "unknown"}),
    ],
    ids=("success", "failure", "malformed-failure", "malformed-items"),
)
def test_legacy_merge_artifact_fallback_is_strict(
    tmp_path: Path,
    failure_payload: Any,
    items_payload: Any,
    expected: dict[str, Any],
) -> None:
    from daydream.archive import pipeline

    if failure_payload is not None:
        _write_deep(tmp_path, "per-stack-failures.json", failure_payload)
    _write_deep(tmp_path, "merged-items.json", items_payload)

    states = pipeline.derive_phase_states(
        tmp_path,
        phase_events=[],
        runs_merge=True,
        runs_fix=False,
        runs_test=False,
        session_id=None,
    )

    assert states["merge"] == expected


@pytest.mark.parametrize(
    "artifact_name",
    ["per-stack-failures.json", "merged-items.json"],
)
def test_legacy_merge_invalid_utf8_is_unknown(
    tmp_path: Path,
    artifact_name: str,
) -> None:
    from daydream.archive import pipeline

    deep = tmp_path / ".daydream" / "deep"
    deep.mkdir(parents=True)
    (deep / artifact_name).write_bytes(b"\xff")

    states = pipeline.derive_phase_states(
        tmp_path,
        phase_events=[],
        runs_merge=True,
        runs_fix=False,
        runs_test=False,
        session_id=None,
    )

    assert states["merge"] == {"ran": True, "status": "unknown"}


def test_current_archive_survives_invalid_utf8_fix_failures(
    tmp_path: Path,
    archive_dir: Path,
    make_config: MakeConfig,
) -> None:
    target = _frozen_target(tmp_path)
    session_id = "current-corrupt-fix-sidecar"
    recorder = _MockRecorder(session_id=session_id)
    deep = target / ".daydream" / "deep"
    deep.mkdir(parents=True)
    (deep / "fix-failures.json").write_bytes(b"\xff")
    _write_deep(target, "merged-items.json", {"items": []})
    _write_deep(
        target,
        "test-verdict.json",
        {"session_id": session_id, "passed": True},
    )
    phase_events = [
        *_merge_events(session_id, "succeeded"),
        {
            "phase": "fix",
            "event": "phase_start",
            "timestamp": "2026-01-01T00:00:00Z",
            "session_id": session_id,
            "scope_id": "fix-scope",
        },
    ]

    _strict_archive(
        target=target,
        session_id=session_id,
        config=make_config(target, archive=True, run_eval=True),
        write_snapshot=_write_snapshot(recorder, phase_events=phase_events),
    )

    run_dir = archive_dir / "runs" / session_id
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert (run_dir / "evaluation.json").is_file()
    assert manifest["phase_states"] == {
        "merge": {"ran": True, "status": "succeeded"},
        "fix": {"ran": True, "status": "succeeded"},
        "test": {"ran": True, "status": "succeeded"},
        "push": {"ran": False, "status": "absent"},
        "remote_ci": {"ran": False, "status": "absent"},
    }
    assert manifest["pipeline_status"] == "succeeded"
    assert [row["session_id"] for row in query_runs(archive_dir)] == [session_id]




def test_start_at_fix_archive_does_not_require_or_inherit_merge(
    tmp_path: Path,
    archive_dir: Path,
    make_config: MakeConfig,
) -> None:
    target = _frozen_target(tmp_path)
    session_id = "fix-resume-session"
    recorder = _MockRecorder(session_id=session_id)
    _write_deep(target, "merged-items.json", {"items": [{"id": 1}]})
    _write_deep(target, "per-stack-failures.json", {"__merge__": {"message": "prior failure"}})
    _write_deep(target, "test-verdict.json", {"session_id": session_id, "passed": True})
    fix_start = {
        "phase": "fix",
        "event": "phase_start",
        "timestamp": "2026-01-01T00:00:00Z",
        "session_id": session_id,
        "scope_id": "fix-scope",
    }

    _strict_archive(
        target=target,
        session_id=session_id,
        config=make_config(target, archive=True, start_at="fix"),
        write_snapshot=_write_snapshot(recorder, phase_events=[fix_start]),
        identity=_manifest_identity(
            phases=replace(_manifest_identity().phases, merge=False)
        ),
    )

    manifest = json.loads(
        (archive_dir / "runs" / session_id / "manifest.json").read_text()
    )
    assert manifest["phase_states"]["merge"] == {"ran": False, "status": "absent"}
    assert manifest["phase_states"]["fix"] == {"ran": True, "status": "succeeded"}
    assert manifest["phase_states"]["test"] == {"ran": True, "status": "succeeded"}
    assert manifest["phase_states"]["push"] == {"ran": False, "status": "absent"}
    assert manifest["phase_states"]["remote_ci"] == {"ran": False, "status": "absent"}
    assert manifest["pipeline_status"] == "succeeded"


def test_archive_retains_malformed_frozen_merge_evidence_as_unknown(
    tmp_path: Path,
    archive_dir: Path,
    make_config: MakeConfig,
) -> None:
    target = _frozen_target(tmp_path)
    session_id = "malformed-merge-session"
    recorder = _MockRecorder(session_id=session_id)
    _write_deep(target, "merged-items.json", {"items": [{"id": 1}]})
    events = _merge_events(session_id, "not-a-status")

    _strict_archive(
        target=target,
        session_id=session_id,
        config=make_config(target, archive=True),
        write_snapshot=_write_snapshot(recorder, phase_events=events),
    )

    manifest = json.loads(
        (archive_dir / "runs" / session_id / "manifest.json").read_text()
    )
    assert manifest["phase_states"]["merge"] == {"ran": True, "status": "unknown"}
    assert manifest["pipeline_status"] == "partial"


def test_archive_rejects_frozen_root_from_another_session(
    tmp_path: Path,
    archive_dir: Path,
    make_config: MakeConfig,
) -> None:
    """A root document carrying another run's session is refused closed."""
    target = _frozen_target(tmp_path)
    recorder = _MockRecorder(session_id="current-session")
    payload = {
        "session_id": "other-session",
        "trajectory_id": recorder.session_id,
        "steps": [],
        "extra": {},
        "final_metrics": {},
    }
    snapshot = RunWriteSnapshot(
        status="complete",
        cutoff_at="2026-01-01T00:00:01Z",
        root_trajectory_id=recorder.session_id,
        documents=(
            TrajectoryDocumentSnapshot(
                trajectory_id=recorder.session_id,
                path=recorder.path,
                json_bytes=json.dumps(payload).encode(),
            ),
        ),
    )

    with pytest.raises(ValueError, match="frozen root trajectory identity"):
        _strict_archive(
            target=target,
            session_id=recorder.session_id,
            config=make_config(target, archive=True),
            write_snapshot=snapshot,
        )
    assert not (archive_dir / "runs" / recorder.session_id).exists()


def test_archive_rejects_a_sibling_document_from_another_session(
    tmp_path: Path,
    archive_dir: Path,
    make_config: MakeConfig,
) -> None:
    """A valid root cannot smuggle a fork document bound to another session.

    The root passes provenance validation, so the refusal has to come from the
    bundle projection — and it must leave no partially assembled archive."""
    from daydream.archive import ArchiveFinalizationError

    target = _frozen_target(tmp_path)
    recorder = _MockRecorder(session_id="current-session")
    root = _write_snapshot(recorder).documents[0]
    foreign = json.dumps(
        {"session_id": "other-session", "trajectory_id": "fork-1"}
    ).encode()
    snapshot = RunWriteSnapshot(
        status="complete",
        cutoff_at="2026-01-01T00:00:01Z",
        root_trajectory_id=recorder.session_id,
        documents=(root, TrajectoryDocumentSnapshot("fork-1", target / "fork.json", foreign)),
    )

    with pytest.raises(ArchiveFinalizationError, match="archive finalization failed"):
        _strict_archive(
            target=target,
            session_id=recorder.session_id,
            config=make_config(target, archive=True),
            write_snapshot=snapshot,
        )
    assert not (archive_dir / "runs" / recorder.session_id).exists()
    assert query_runs(archive_dir) == []


def test_merge_failed_discriminates_on_merge_key_not_merged_items(
    tmp_path: Path,
) -> None:
    from daydream.archive import pipeline
    _write_deep(tmp_path, "merged-items.json", {"items": []})
    _write_deep(tmp_path, "per-stack-failures.json", {"__merge__": {"message": "x"}})
    states = pipeline.derive_phase_states(tmp_path, phase_events=[])
    assert states["merge"]["ran"] is True
    assert states["merge"]["status"] == "failed"   # merged-items present is NOT sufficient


def test_merge_succeeded_when_items_and_no_merge_key(tmp_path: Path) -> None:
    from daydream.archive import pipeline
    _write_deep(tmp_path, "merged-items.json", {"items": []})
    states = pipeline.derive_phase_states(tmp_path, phase_events=[])
    assert states["merge"]["status"] == "succeeded"


def test_test_failed_from_verdict(tmp_path: Path) -> None:
    from daydream.archive import pipeline
    _write_deep(
        tmp_path,
        "test-verdict.json",
        {"session_id": "current", "passed": False, "retries": 1, "ignored": False},
    )
    states = pipeline.derive_phase_states(
        tmp_path, phase_events=[], session_id="current"
    )
    assert states["test"]["ran"] is True
    assert states["test"]["status"] == "failed"


def test_session_bound_start_at_fix_rejects_prior_green_test_verdict(
    tmp_path: Path,
) -> None:
    from daydream.archive import pipeline

    _write_deep(tmp_path, "test-verdict.json", {"session_id": "prior", "passed": True})
    states = pipeline.derive_phase_states(
        tmp_path, phase_events=[], session_id="current"
    )
    assert states["test"] == {"ran": False, "status": "absent"}
    assert pipeline.derive_pipeline_status(
        "complete", None, states, runs_fix=True, runs_test=True
    ) == "partial"


def test_matching_stabilization_failure_overrides_green_test_pipeline(
    tmp_path: Path,
) -> None:
    from daydream.archive import pipeline

    _write_deep(tmp_path, "test-verdict.json", {"session_id": "current", "passed": True})
    _write_deep(
        tmp_path,
        "stabilization-failed.json",
        {"session_id": "current", "reason": "final verifier remains actionable"},
    )

    states = pipeline.derive_phase_states(
        tmp_path, phase_events=[], session_id="current"
    )

    assert states["fix"] == {"ran": True, "status": "failed"}
    assert states["test"] == {"ran": True, "status": "failed"}
    assert pipeline.derive_pipeline_status(
        "complete", None, states, runs_fix=True, runs_test=True
    ) == "failed"


@pytest.mark.parametrize(
    "payload",
    [
        {"session_id": "prior", "reason": "stale"},
        {"session_id": "current"},
        {"session_id": "current", "reason": ""},
        ["malformed"],
    ],
)
def test_stale_or_malformed_stabilization_failure_is_neutral(
    tmp_path: Path, payload: Any
) -> None:
    from daydream.archive import pipeline

    _write_deep(tmp_path, "test-verdict.json", {"session_id": "current", "passed": True})
    _write_deep(tmp_path, "stabilization-failed.json", payload)

    states = pipeline.derive_phase_states(
        tmp_path, phase_events=[], session_id="current"
    )

    assert states["fix"] == {"ran": False, "status": "absent"}
    assert states["test"] == {"ran": True, "status": "succeeded"}


def test_archive_manifest_fails_matching_stabilization_session(
    tmp_path: Path, archive_dir: Path, make_config: MakeConfig
) -> None:
    target = _frozen_target(tmp_path)
    session_id = "stabilization-session"
    _write_deep(target, "merged-items.json", {"items": [{"id": 1}]})
    _write_deep(target, "test-verdict.json", {"session_id": session_id, "passed": True})
    _write_deep(
        target,
        "stabilization-failed.json",
        {"session_id": session_id, "reason": "post-test tree did not stabilize"},
    )
    recorder = _MockRecorder(session_id=session_id)

    _strict_archive(
        target=target,
        session_id=session_id,
        config=make_config(target, archive=True),
        write_snapshot=_write_snapshot(recorder),
    )

    manifest = json.loads(
        (archive_dir / "runs" / session_id / "manifest.json").read_text()
    )
    assert manifest["archive_status"] == "complete"
    assert manifest["pipeline_status"] == "failed"
    assert manifest["phase_states"]["fix"] == {"ran": True, "status": "failed"}
    assert manifest["phase_states"]["test"] == {"ran": True, "status": "failed"}


def test_test_absent_when_no_verdict(tmp_path: Path) -> None:
    from daydream.archive import pipeline
    states = pipeline.derive_phase_states(tmp_path, phase_events=[])
    assert states["test"]["ran"] is False
    assert states["test"]["status"] == "absent"


def test_fix_partial_from_failures(tmp_path: Path) -> None:
    from daydream.archive import pipeline
    _write_deep(tmp_path, "fix-failures.json", {"src/a.py": "reverted"})
    states = pipeline.derive_phase_states(tmp_path, phase_events=[])
    assert states["fix"]["status"] == "partial"


def test_pipeline_status_precedence() -> None:
    from daydream.archive import pipeline
    # cancelled beats everything when archive partial with no fix failures
    assert pipeline.derive_pipeline_status("partial", None,
        {"merge": {"ran": True, "status": "succeeded"},
         "fix": {"ran": True, "status": "succeeded"},
         "test": {"ran": True, "status": "succeeded"}}) == "cancelled"
    # merge failed -> failed even though archive_status complete
    assert pipeline.derive_pipeline_status("complete", None,
        {"merge": {"ran": True, "status": "failed"},
         "fix": {"ran": False, "status": "absent"},
         "test": {"ran": False, "status": "absent"}}, runs_test=True) == "failed"
    # test failed -> failed
    assert pipeline.derive_pipeline_status("complete", None,
        {"merge": {"ran": True, "status": "succeeded"},
         "fix": {"ran": True, "status": "succeeded"},
         "test": {"ran": True, "status": "failed"}}) == "failed"
    # flow runs test but it never ran -> partial
    assert pipeline.derive_pipeline_status("complete", None,
        {"merge": {"ran": True, "status": "succeeded"},
         "fix": {"ran": True, "status": "succeeded"},
         "test": {"ran": False, "status": "absent"}}, runs_test=True) == "partial"
    # clean deep run -> succeeded
    assert pipeline.derive_pipeline_status("complete", None,
        {"merge": {"ran": True, "status": "succeeded"},
         "fix": {"ran": True, "status": "succeeded"},
         "test": {"ran": True, "status": "succeeded"}}) == "succeeded"
    # all-absent (a flow that runs neither fix nor test and surfaced no phase
    # evidence) -> unknown, never succeeded
    assert pipeline.derive_pipeline_status("complete", None,
        {"merge": {"ran": False, "status": "absent"},
         "fix": {"ran": False, "status": "absent"},
         "test": {"ran": False, "status": "absent"}},
        runs_fix=False, runs_test=False) == "unknown"
    # an unexpected/unknown per-phase status drives the _UNKNOWN branch
    assert pipeline.derive_pipeline_status("complete", None,
        {"merge": {"ran": True, "status": "succeeded"},
         "fix": {"ran": True, "status": "unknown"},
         "test": {"ran": True, "status": "succeeded"}}) == "unknown"


def test_non_deep_flow_ignores_stale_deep_artifacts(tmp_path: Path) -> None:
    # Issue #336: derive_phase_states is flow-aware. A prior deep run left
    # session-agnostic merge/fix/test artifacts in target_dir/.daydream/deep;
    # a non-deep flow run afterwards must NOT inherit them as its own pipeline
    # state -- the phases it does not run read absent regardless of disk.
    from daydream.archive import pipeline
    _write_deep(tmp_path, "merged-items.json", {"items": []})
    _write_deep(tmp_path, "per-stack-failures.json", {"__merge__": {"message": "x"}})
    _write_deep(tmp_path, "test-verdict.json", {"passed": False})
    _write_deep(tmp_path, "fix-failures.json", {"src/a.py": "reverted"})
    # TTT review runs the merge spine but never the fix/test cycle.
    states = pipeline.derive_phase_states(tmp_path, phase_events=[],
                                          runs_merge=True, runs_fix=False, runs_test=False)
    assert states["merge"]["status"] == "failed"   # merge ran (spine wrote fresh artifacts)
    assert states["fix"] == {"ran": False, "status": "absent"}    # stale fix ignored
    assert states["test"] == {"ran": False, "status": "absent"}   # stale test ignored
    # An improve-only flow runs none of the deep phases at all.
    states = pipeline.derive_phase_states(tmp_path, phase_events=[],
                                          runs_merge=False, runs_fix=False, runs_test=False)
    assert all(s == {"ran": False, "status": "absent"} for s in states.values())



def test_merge_failed_archives_failed_pipeline(
    tmp_path: Path, archive_dir: Path, make_config: MakeConfig
) -> None:
    target = _frozen_target(tmp_path)
    _write_deep(target, "merged-items.json", {"items": []})
    _write_deep(target, "per-stack-failures.json", {"__merge__": {"message": "x"}})
    _write_deep(target, "test-verdict.json", {"passed": False, "retries": 0, "ignored": False})
    recorder = _MockRecorder(session_id="merge-failed-session")  # run_flow NORMAL
    _strict_archive(
        target=target,
        session_id=recorder.session_id,
        config=make_config(target, archive=True),
        write_snapshot=_write_snapshot(
            recorder,
            phase_events=_merge_events(recorder.session_id, "failed"),
        ),
    )
    manifest_path = archive_dir / "runs" / recorder.session_id / "manifest.json"
    m = json.loads(manifest_path.read_text())
    assert m["archive_status"] == "complete"   # cleanly archived...
    assert m["pipeline_status"] == "failed"    # ...but the pipeline failed
    assert m["phase_states"]["merge"]["status"] == "failed"
    assert m["daydream"]["version"]  # executable provenance recorded
    assert m["daydream"]["commit"]  # a real SHA or the "unknown" sentinel, never blank


def test_schema_additive_columns_and_migration(tmp_path: Path) -> None:
    from daydream.archive import _schema
    db = tmp_path / "index.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE runs (session_id TEXT PRIMARY KEY, status TEXT NOT NULL DEFAULT 'complete')")
    conn.commit()
    conn.close()
    # _migrate_schema adds the new columns idempotently to an existing table
    _schema._migrate_schema(sqlite3.connect(db))
    cols = {r[1] for r in sqlite3.connect(db).execute("PRAGMA table_info(runs)").fetchall()}
    assert "archive_status" in cols and "pipeline_status" in cols and "phase_states" in cols
    assert "daydream_version" in cols and "daydream_commit" in cols and "daydream_dirty" in cols


def test_upsert_run_persists_pipeline_fields(tmp_path: Path) -> None:
    from daydream.archive import index
    from daydream.archive.manifest import Manifest
    from daydream.archive.provenance import ExecutableProvenance

    m = Manifest(
        session_id="s-2",
        status="complete",
        archive_status="complete",
        pipeline_status="failed",
        phase_states={"merge": {"ran": True, "status": "failed"}},
        daydream=ExecutableProvenance(
            version="0.27.0",
            install_source="git",
            commit="abc",
            dirty=False,
            container_digest="unknown",
        ),
    )
    index.upsert_run(tmp_path, m)
    row = index.query_runs(tmp_path, "session_id = ?", ("s-2",))[0]
    assert row["archive_status"] == "complete"
    assert row["pipeline_status"] == "failed"
    assert row["daydream_version"] == "0.27.0"
    assert row["daydream_dirty"] == 0


# Task 7: versioned reply-label columns, digest-keyed dedup, legacy marking


_LEGACY_ROW_OBSERVED_AT_SNAPSHOT = "2026-03-04T05:06:07+00:00"

# Old DDL: everything up to (but excluding) the four reply-label/legacy columns.
_PRE_REPLY_LABEL_DDL = """
CREATE TABLE IF NOT EXISTS label_observations (
    session_id       TEXT NOT NULL,
    observed_at      TEXT NOT NULL,
    labels           TEXT NOT NULL,
    pr_state         TEXT,
    labeler_version  TEXT NOT NULL,
    evidence_sha     TEXT,
    rubric_json      TEXT,
    valid_at         TEXT,
    reward_version   TEXT,
    reward_json      TEXT,
    composite_reward REAL,
    reviewer_logins  TEXT,
    has_posterior    INTEGER NOT NULL DEFAULT 0,
    source           TEXT NOT NULL DEFAULT 'auto',
    PRIMARY KEY (session_id, observed_at)
)
"""


def _seed_pre_reply_label_row(archive_dir: Path, session_id: str) -> None:
    """Insert a label_observations row using the DDL that predates the four new columns."""
    conn = sqlite3.connect(str(archive_dir / "index.db"))
    try:
        conn.execute("DROP TABLE IF EXISTS label_observations")
        conn.execute(_PRE_REPLY_LABEL_DDL)
        conn.execute(
            "INSERT INTO label_observations "
            "(session_id, observed_at, labels, pr_state, labeler_version, evidence_sha) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                session_id,
                _LEGACY_ROW_OBSERVED_AT_SNAPSHOT,
                '["accepted"]',
                "merged",
                "v1",
                "sha1",
            ),
        )
        conn.commit()
    finally:
        conn.close()


def test_append_label_observation_persists_versions_and_digest(tmp_path: Path) -> None:
    _seed_one_run(tmp_path, "sess-1")
    ok = append_label_observation(
        tmp_path, "sess-1", labels=["contested"], pr_state="open",
        labeler_version="980-policy", evidence_sha="abc",
        reply_classifier_version="980-r1", reply_evidence_digest="d" * 64,
    )
    assert ok is True
    row = latest_label_observation(tmp_path, "sess-1")
    assert row is not None
    assert row["reply_classifier_version"] == "980-r1"
    assert row["reply_evidence_digest"] == "d" * 64
    assert row["labeler_policy_version"] == "980-policy"


def test_dedup_includes_policy_version_and_digest(tmp_path: Path) -> None:
    """Unchanged evidence + bumped policy version ⇒ new generation, not dedup (M14/M22)."""
    _seed_one_run(tmp_path, "sess-1")
    kw: dict[str, Any] = dict(labels=["contested"], pr_state="open", labeler_version="980-policy",
              evidence_sha="abc", reply_classifier_version="980-r1", reply_evidence_digest="d" * 64)
    assert append_label_observation(tmp_path, "sess-1", **kw) is True
    assert append_label_observation(tmp_path, "sess-1", **kw) is False       # identical → dedup
    kw2 = {**kw, "labeler_version": "980-policy2"}
    assert append_label_observation(tmp_path, "sess-1", **kw2) is True       # policy bump → append
    kw3 = {**kw2, "reply_evidence_digest": "e" * 64}
    assert append_label_observation(tmp_path, "sess-1", **kw3) is True       # edited reply → append


def test_human_rows_keep_precedence_over_newer_auto(tmp_path: Path) -> None:
    """Human observation wins in the runs cache even after a newer auto append (M14/M22)."""
    _seed_one_run(tmp_path, "sess-1")
    append_label_observation(tmp_path, "sess-1", labels=["contested"], pr_state="open",
                             labeler_version="980-policy", evidence_sha="abc",
                             reply_classifier_version="980-r1", reply_evidence_digest="d" * 64)
    append_label_observation(tmp_path, "sess-1", labels=["rejected"], pr_state="open",
                             labeler_version="human", evidence_sha="abc", source="human")
    append_label_observation(tmp_path, "sess-1", labels=["accepted"], pr_state="open",
                             labeler_version="980-policy2", evidence_sha="abc",
                             reply_classifier_version="980-r1", reply_evidence_digest="f" * 64)
    row = query_runs(tmp_path, "session_id = ?", ("sess-1",))[0]
    assert row["outcome_labels"] == '["rejected"]' and row["labeled_at"] is not None
    obs = latest_label_observation(tmp_path, "sess-1")
    assert obs is not None and obs["source"] == "human" and obs["labels"] == '["rejected"]'


def test_migration_marks_legacy_rows(tmp_path: Path) -> None:
    """Pre-existing auto rows are marked legacy='legacy' and never mutated otherwise (M17/M22)."""
    _seed_one_run(tmp_path, "sess-legacy")
    _seed_pre_reply_label_row(tmp_path, "sess-legacy")

    # The production connection path must ALTER-ADD the new columns and stamp history.
    upsert_run(tmp_path, make_manifest(session_id="sess-legacy-2"))

    rows = label_observation_history(tmp_path, "sess-legacy")
    assert rows
    assert all(r["legacy"] == "legacy" for r in rows if r["labeler_policy_version"] is None)
    # original labels/observed_at untouched:
    assert rows[0]["observed_at"] == _LEGACY_ROW_OBSERVED_AT_SNAPSHOT
    assert rows[0]["labels"] == '["accepted"]'


def test_append_label_observation_preserves_observed_at(tmp_path: Path) -> None:
    """An explicit ``observed_at`` is preserved bitemporally (M3 of the
    local-observations import): the stored row carries the original data
    timestamp verbatim (canonicalized to UTC), not the wall clock."""
    _seed_one_run(tmp_path, "sess-obs")
    original = "2025-06-01T12:00:00+00:00"
    appended = append_label_observation(
        tmp_path, "sess-obs", labels=["accepted"], pr_state="merged",
        labeler_version="1055-human-r1", evidence_sha=None, source="human",
        observed_at=original)
    assert appended is True
    row = latest_label_observation(tmp_path, "sess-obs")
    assert row is not None
    assert row["observed_at"] == "2025-06-01T12:00:00+00:00"


def test_append_label_observation_observed_at_none_uses_wall_clock(
    tmp_path: Path,
) -> None:
    """Default (``observed_at=None``) keeps the existing now() behavior."""
    _seed_one_run(tmp_path, "sess-now")
    appended = append_label_observation(
        tmp_path, "sess-now", labels=["accepted"], pr_state="merged",
        labeler_version="1055-human-r1", evidence_sha=None, source="human")
    assert appended is True
    row = latest_label_observation(tmp_path, "sess-now")
    assert row is not None
    assert row["observed_at"].startswith("2")


def test_append_label_observation_rejects_non_iso_observed_at(tmp_path: Path) -> None:
    """A non-ISO-8601 ``observed_at`` fails closed before any write."""
    _seed_one_run(tmp_path, "sess-bad")
    with pytest.raises(ValueError, match="observed_at"):
        append_label_observation(
            tmp_path, "sess-bad", labels=["accepted"], pr_state="merged",
            labeler_version="1055-human-r1", evidence_sha=None, source="human",
            observed_at="not-a-timestamp")
    assert label_observation_history(tmp_path, "sess-bad") == []


def test_delete_runs_is_exported() -> None:
    import daydream.archive.index as index_module

    assert "delete_runs" in index_module.__all__
    assert "delete_runs:" in index_module.__doc__


def test_diagram_flow_does_not_inherit_a_prior_deep_run_pipeline_state(
    tmp_path: Path, archive_dir: Path, make_config: MakeConfig,
) -> None:
    """#1113 (D21/D22): a diagram-only run deliberately leaves the previous deep
    review's ``.daydream/deep/`` artifacts on disk, so its own flow label must
    answer "runs no merge/fix/test" — otherwise it archives that run's
    ``merged-items.json`` as its own pipeline state."""
    target = _frozen_target(tmp_path)
    _write_deep(target, "merged-items.json", {"items": []})
    _write_deep(target, "per-stack-failures.json", {"__merge__": {"message": "x"}})
    _write_deep(target, "test-verdict.json", {"passed": False, "retries": 0, "ignored": False})
    _write_deep(target, "fix-failures.json", {"src/a.py": "reverted"})

    recorder = _MockRecorder(
        session_id="diagram-only-session", run_flow=DaydreamRunFlow.DIAGRAM
    )
    _strict_archive(
        target=target,
        session_id=recorder.session_id,
        run_flow=DaydreamRunFlow.DIAGRAM,
        config=make_config(target, archive=True),
        write_snapshot=_write_snapshot(recorder),
        identity=_manifest_identity(
            fix_backend=None,
            test_backend=None,
            per_stack_review_backend=None,
            per_stack_review_model=None,
            phases=RunPhaseCapabilities(
                per_stack_review=False,
                merge=False,
                fix=False,
                test=False,
                push=False,
                remote_ci=False,
            ),
        ),
    )

    manifest_path = archive_dir / "runs" / recorder.session_id / "manifest.json"
    m = json.loads(manifest_path.read_text())
    assert m["run"]["flow"] == "diagram"
    assert m["status"] == "complete"
    assert m["archive_status"] == "complete"
    assert m["pipeline_status"] == "unknown"
    # None of the deep phases are claimed, so no stale artifact is adopted.
    assert m["phase_states"]["merge"] == {"ran": False, "status": "absent"}
    assert m["phase_states"]["fix"] == {"ran": False, "status": "absent"}
    assert m["phase_states"]["test"] == {"ran": False, "status": "absent"}
    assert m["phase_states"]["push"] == {"ran": False, "status": "absent"}
    assert m["phase_states"]["remote_ci"] == {
        "ran": False,
        "status": "absent",
    }
    # A two-step flow runs neither fix nor test, so neither backend is labeled.
    assert "fix_backend" not in m["run"]
    assert "test_backend" not in m["run"]
    # No per-stack fan-out either: the diagram flow has no per-stack reviewers.
    assert "per_stack_review_backend" not in m["run"]


def test_diagram_flow_does_not_evaluate_stale_review_artifacts(
    tmp_path: Path, archive_dir: Path, make_config: MakeConfig,
) -> None:
    target = _frozen_target(tmp_path)
    _write_deep(
        target,
        "merged-items.json",
        {"items": [{"file": "src/old.py", "line": 1, "confidence": "HIGH"}]},
    )
    recorder = _MockRecorder(
        session_id="diagram-no-eval-session", run_flow=DaydreamRunFlow.DIAGRAM
    )

    _strict_archive(
        target=target,
        session_id=recorder.session_id,
        run_flow=DaydreamRunFlow.DIAGRAM,
        config=make_config(target, archive=True, run_eval=True),
        write_snapshot=_write_snapshot(recorder),
        identity=_manifest_identity(
            fix_backend=None,
            test_backend=None,
            per_stack_review_backend=None,
            per_stack_review_model=None,
            phases=RunPhaseCapabilities(
                per_stack_review=False,
                merge=False,
                fix=False,
                test=False,
                push=False,
                remote_ci=False,
            ),
        ),
    )

    run_dir = archive_dir / "runs" / recorder.session_id
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert not (run_dir / "evaluation.json").exists()
    assert manifest["metrics"]["total_findings"] is None


def test_snapshot_manifest_pr_metadata_is_immutable_after_live_inputs_mutate(
    tmp_path: Path,
) -> None:
    """The production manifest identity comes only from validated root bytes."""
    from daydream.archive.manifest import (
        archive_recorder_provenance_from_snapshot,
        build_manifest_from_snapshot,
    )

    session_id = "frozen-pr-session"
    snapshot = _manifest_write_snapshot(
        session_id=session_id,
        extra={"pr_number": 7, "pr_repo": "Owner/Repo"},
    )
    payload = json.loads(snapshot.documents[0].json_bytes)
    payload["extra"] = {"pr_number": 7, "pr_repo": "Owner/Repo"}
    document = TrajectoryDocumentSnapshot(
        trajectory_id=session_id,
        path=snapshot.documents[0].path,
        json_bytes=json.dumps(payload).encode(),
    )
    snapshot = RunWriteSnapshot(
        status="complete",
        cutoff_at=snapshot.cutoff_at,
        root_trajectory_id=session_id,
        documents=(document,),
    )
    provenance = archive_recorder_provenance_from_snapshot(
        write_snapshot=snapshot,
        run_flow=DaydreamRunFlow.NORMAL,
    )
    live_recorder = _MockRecorder(pr_number=7, pr_repo="Owner/Repo")
    live_config = RunConfig(target=str(tmp_path), pr_number=7, pr_repo="Owner/Repo")
    live_recorder.pr_number = 99
    live_recorder.pr_repo = "mutated/repo"
    live_config.pr_number = 100
    live_config.pr_repo = "also/mutated"

    manifest = build_manifest_from_snapshot(
        run=ArchiveRunSnapshot(
            recorder_provenance=provenance,
            identity=_manifest_identity(),
            trajectories=snapshot,
        ),
        git_ctx=GitContext(),
        status="complete",
        archive_path=tmp_path,
    )

    assert manifest.session_id == snapshot.root_trajectory_id
    assert manifest.pr_number == 7
    assert manifest.pr_repo == "Owner/Repo"
    assert manifest.to_dict()["pr"] == {"number": 7, "repo": "Owner/Repo"}


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"pr_number": True}, "pr_number"),
        ({"pr_number": 7, "pr_repo": None}, "pr_repo"),
        ({"pr_number": 7, "pr_repo": ""}, "pr_repo"),
    ],
)
def test_snapshot_manifest_provenance_rejects_malformed_present_pr_metadata(
    tmp_path: Path,
    extra: dict[str, Any],
    message: str,
) -> None:
    from daydream.archive.manifest import archive_recorder_provenance_from_snapshot

    payload = {
        "session_id": "session",
        "trajectory_id": "session",
        "steps": [],
        "final_metrics": {},
        "extra": extra,
    }
    snapshot = RunWriteSnapshot(
        status="complete",
        cutoff_at="2026-09-06T00:00:00Z",
        root_trajectory_id="session",
        documents=(
            TrajectoryDocumentSnapshot(
                trajectory_id="session",
                path=tmp_path / "trajectory.json",
                json_bytes=json.dumps(payload).encode(),
            ),
        ),
    )

    with pytest.raises(ValueError, match=message):
        archive_recorder_provenance_from_snapshot(
            write_snapshot=snapshot,
            run_flow=DaydreamRunFlow.NORMAL,
        )


@pytest.mark.parametrize("session_id", [".", "..", "../escape", "bad\\path", "bad\0id"])
def test_snapshot_manifest_provenance_rejects_unsafe_session_identity(
    tmp_path: Path,
    session_id: str,
) -> None:
    from daydream.archive.manifest import archive_recorder_provenance_from_snapshot

    encoded = json.dumps(
        {
            "session_id": session_id,
            "trajectory_id": session_id,
            "steps": [],
            "final_metrics": {},
            "extra": {},
        }
    ).encode()
    snapshot = RunWriteSnapshot(
        status="complete",
        cutoff_at="2026-09-06T00:00:00Z",
        root_trajectory_id=session_id,
        documents=(TrajectoryDocumentSnapshot(session_id, tmp_path / "root.json", encoded),),
    )

    with pytest.raises(ValueError, match="session_id"):
        archive_recorder_provenance_from_snapshot(
            write_snapshot=snapshot,
            run_flow=DaydreamRunFlow.NORMAL,
        )


def test_strict_archive_evaluation_failure_is_typed_and_never_reports_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The host finalizer cannot infer success from the legacy fail-open wrapper."""
    from daydream.archive import ArchiveFinalizationError, finalize_archive_run
    from daydream.artifact_visibility import (
        ArtifactEvidenceProvenance,
        ArtifactTreeSnapshot,
        _manifest,
    )

    session_id = "strict-session"
    frozen = tmp_path / "frozen"
    run_dir = frozen / ".daydream" / "runs" / session_id
    run_dir.mkdir(parents=True)
    payload = {
        "session_id": session_id,
        "trajectory_id": session_id,
        "steps": [],
        "final_metrics": {},
        "extra": {},
    }
    encoded = json.dumps(payload).encode()
    (run_dir / "trajectory.json").write_bytes(encoded)
    write_snapshot = RunWriteSnapshot(
        status="complete",
        cutoff_at="2026-09-06T00:00:00Z",
        root_trajectory_id=session_id,
        documents=(
            TrajectoryDocumentSnapshot(
                trajectory_id=session_id,
                path=run_dir / "trajectory.json",
                json_bytes=encoded,
            ),
        ),
    )
    artifacts = ArtifactTreeSnapshot(
        session_id=session_id,
        workspace_key="workspace",
        root=frozen,
        manifest=_manifest(frozen),
        destinations=(),
    )
    artifact_provenance = ArtifactEvidenceProvenance(
        workspace_key="workspace",
        session_id=session_id,
        public_source=tmp_path / "source",
        live_root=(tmp_path / "live"),
    )
    monkeypatch.setattr(
        "daydream.eval.analyzer.analyze_session",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("eval failed")),
    )

    with pytest.raises(ArchiveFinalizationError, match="evaluation"):
        finalize_archive_run(
            run=_archive_snapshot(write_snapshot),
            artifacts=artifacts,
            artifact_provenance=artifact_provenance,
            config=RunConfig(target=str(tmp_path), run_eval=True),
            work=None,
            upload=False,
        )
    assert not (get_archive_dir() / "runs" / session_id).exists()


@pytest.mark.parametrize("mutate", [False, True])
def test_strict_archive_rejects_frozen_receipt_changed_by_evaluator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: bool,
) -> None:
    """A code-running consumer cannot make archive bytes and manifest disagree."""
    from daydream.archive import ArchiveFinalizationError, finalize_archive_run
    from daydream.artifact_visibility import (
        ArtifactEvidenceProvenance,
        ArtifactTreeSnapshot,
        _manifest,
    )

    session_id = "strict-mutated-evidence"
    frozen = tmp_path / "frozen"
    source_run = frozen / ".daydream" / "runs" / session_id
    source_run.mkdir(parents=True)
    encoded = json.dumps(
        {
            "session_id": session_id,
            "trajectory_id": session_id,
            "steps": [],
            "final_metrics": {},
            "extra": {},
        }
    ).encode()
    (source_run / "trajectory.json").write_bytes(encoded)
    receipt = frozen / ".daydream" / "deep" / "test-verdict.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(
        json.dumps({"session_id": session_id, "passed": True}),
        encoding="utf-8",
    )
    snapshot = RunWriteSnapshot(
        status="complete",
        cutoff_at="2026-09-06T00:00:00Z",
        root_trajectory_id=session_id,
        documents=(
            TrajectoryDocumentSnapshot(
                session_id,
                source_run / "trajectory.json",
                encoded,
            ),
        ),
    )
    artifacts = ArtifactTreeSnapshot(
        session_id,
        "workspace",
        frozen,
        _manifest(frozen),
        (),
    )
    public_source = tmp_path / "source"
    public_source.mkdir()

    def mutate_frozen_receipt(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        if mutate:
            receipt.write_text(
                json.dumps({"session_id": session_id, "passed": False}),
                encoding="utf-8",
            )
        return {"quality": {"scoped_files": 0}}

    monkeypatch.setattr(
        "daydream.eval.analyzer.analyze_session",
        mutate_frozen_receipt,
    )

    arguments: dict[str, Any] = dict(
        run=_archive_snapshot(snapshot),
        artifacts=artifacts,
        artifact_provenance=ArtifactEvidenceProvenance(
            "workspace",
            session_id,
            public_source,
            tmp_path / "live",
        ),
        config=RunConfig(target=str(tmp_path), run_eval=True),
        work=None,
        upload=False,
    )
    if mutate:
        with pytest.raises(ArchiveFinalizationError, match="frozen artifact tree changed"):
            finalize_archive_run(**arguments)
    else:
        finalize_archive_run(
            **arguments,
        )

    archive_dir = get_archive_dir()
    rows = query_runs(archive_dir, "session_id = ?", (session_id,))
    if mutate:
        assert not (archive_dir / "runs" / session_id).exists()
        assert rows == []
    else:
        assert (archive_dir / "runs" / session_id / "manifest.json").is_file()
        assert len(rows) == 1


def test_strict_archive_upload_refusal_removes_incomplete_local_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refused upload withholds the Hub copy and keeps the local archive.

    Issue #981 requires refusing the upload "while preserving the local run",
    and ``upload_run_bundle`` documents that it never raises so the archive
    callback cannot fail the run. The refusal already happened inside the
    callee, before anything reached the Hub, so failing finalization here would
    discard a completed review without containing anything extra.
    """
    from daydream.archive import finalize_archive_run
    from daydream.artifact_visibility import (
        ArtifactEvidenceProvenance,
        ArtifactTreeSnapshot,
        _manifest,
    )

    session_id = "strict-upload"
    frozen = tmp_path / "frozen"
    source_run = frozen / ".daydream" / "runs" / session_id
    source_run.mkdir(parents=True)
    encoded = json.dumps(
        {
            "session_id": session_id,
            "trajectory_id": session_id,
            "steps": [],
            "final_metrics": {},
            "extra": {},
        }
    ).encode()
    (source_run / "trajectory.json").write_bytes(encoded)
    snapshot = RunWriteSnapshot(
        status="complete",
        cutoff_at="2026-09-06T00:00:00Z",
        root_trajectory_id=session_id,
        documents=(TrajectoryDocumentSnapshot(session_id, source_run / "trajectory.json", encoded),),
    )
    uploaded: list[tuple[Any, ...]] = []

    def _refuse(*args: object, **_kwargs: object) -> bool:
        uploaded.append(args)
        return False

    monkeypatch.setattr("daydream.archive.hub.resolve_hub_repo", lambda _config: "private/repo")
    monkeypatch.setattr("daydream.archive.hub.upload_run_bundle", _refuse)

    finalize_archive_run(
        run=_archive_snapshot(snapshot),
        artifacts=ArtifactTreeSnapshot(session_id, "workspace", frozen, _manifest(frozen), ()),
        artifact_provenance=ArtifactEvidenceProvenance(
            "workspace", session_id, tmp_path / "source", tmp_path / "live"
        ),
        config=RunConfig(target=str(tmp_path), run_eval=False, archive=True),
        work=None,
        upload=True,
    )

    assert len(uploaded) == 1, "the upload must still be attempted and refused by the callee"
    archive_dir = get_archive_dir()
    assert (archive_dir / "runs" / session_id / "manifest.json").is_file()
    assert len(query_runs(archive_dir, "session_id = ?", (session_id,))) == 1
    assert not list(archive_dir.glob("runs/.*.finalizing"))


def test_strict_archive_upload_refuses_frozen_tree_mutated_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A frozen tree mutated after finalization starts is caught before the external upload."""
    from daydream.archive import ArchiveFinalizationError, finalize_archive_run
    from daydream.artifact_visibility import (
        ArtifactEvidenceProvenance,
        ArtifactTreeSnapshot,
        _manifest,
    )

    session_id = "strict-upload-mutated"
    frozen = tmp_path / "frozen"
    source_run = frozen / ".daydream" / "runs" / session_id
    source_run.mkdir(parents=True)
    encoded = json.dumps(
        {
            "session_id": session_id,
            "trajectory_id": session_id,
            "steps": [],
            "final_metrics": {},
            "extra": {},
        }
    ).encode()
    (source_run / "trajectory.json").write_bytes(encoded)
    snapshot = RunWriteSnapshot(
        status="complete",
        cutoff_at="2026-09-06T00:00:00Z",
        root_trajectory_id=session_id,
        documents=(TrajectoryDocumentSnapshot(session_id, source_run / "trajectory.json", encoded),),
    )
    artifacts = ArtifactTreeSnapshot(session_id, "workspace", frozen, _manifest(frozen), ())
    uploads: list[Path] = []

    def mutate_then_resolve(_config: Any) -> str:
        (source_run / "trajectory.json").write_bytes(encoded + b"\n")
        return "private/repo"

    def record_upload(run_dir: Path, *_args: Any, **_kwargs: Any) -> bool:
        uploads.append(run_dir)
        return True

    monkeypatch.setattr("daydream.archive.hub.resolve_hub_repo", mutate_then_resolve)
    monkeypatch.setattr("daydream.archive.hub.upload_run_bundle", record_upload)

    with pytest.raises(ArchiveFinalizationError, match="frozen artifact tree changed"):
        finalize_archive_run(
            run=_archive_snapshot(snapshot),
            artifacts=artifacts,
            artifact_provenance=ArtifactEvidenceProvenance(
                "workspace", session_id, tmp_path / "source", tmp_path / "live"
            ),
            config=RunConfig(target=str(tmp_path), run_eval=False, archive=True),
            work=None,
            upload=True,
        )

    assert uploads == []
    assert not (get_archive_dir() / "runs" / session_id).exists()


def test_strict_archive_dump_scan_refusal_leaves_late_stage_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refused secret scan publishes neither an archive nor dump bytes."""
    from daydream.archive import ArchiveFinalizationError, finalize_archive_run
    from daydream.artifact_visibility import (
        ArtifactEvidenceProvenance,
        ArtifactTreeSnapshot,
        _manifest,
    )

    session_id = "strict-dump"
    frozen = tmp_path / "frozen"
    source_run = frozen / ".daydream" / "runs" / session_id
    source_run.mkdir(parents=True)
    encoded = json.dumps(
        {
            "session_id": session_id,
            "trajectory_id": session_id,
            "steps": [],
            "final_metrics": {},
            "extra": {},
        }
    ).encode()
    (source_run / "trajectory.json").write_bytes(encoded)
    snapshot = RunWriteSnapshot(
        status="complete",
        cutoff_at="2026-09-06T00:00:00Z",
        root_trajectory_id=session_id,
        documents=(TrajectoryDocumentSnapshot(session_id, source_run / "trajectory.json", encoded),),
    )
    dump_stage = tmp_path / "late"
    dump_stage.mkdir()
    from daydream.archive.scan import SEVERITY_BLOCKING, Finding, ScanResult

    monkeypatch.setattr(
        "daydream.archive.scan.scan_run_dir",
        lambda _path: ScanResult(
            clean=False,
            findings=[
                Finding(
                    path="trajectory.json",
                    location="steps.[0].observation (json)",
                    category="api_key",
                    digest="0123456789ab",
                    severity=SEVERITY_BLOCKING,
                )
            ],
        ),
    )

    with pytest.raises(ArchiveFinalizationError, match="secret scan"):
        finalize_archive_run(
            run=_archive_snapshot(snapshot),
            artifacts=ArtifactTreeSnapshot(
                session_id, "workspace", frozen, _manifest(frozen), ()
            ),
            artifact_provenance=ArtifactEvidenceProvenance(
                "workspace", session_id, tmp_path / "source", tmp_path / "live"
            ),
            config=RunConfig(
                target=str(tmp_path), run_eval=False, archive=True, dump_artifacts="requested"
            ),
            work=None,
            upload=False,
            dump_path=dump_stage,
        )

    assert list(dump_stage.iterdir()) == []
    assert not (get_archive_dir() / "runs" / session_id).exists()
