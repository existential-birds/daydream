"""Unit tests for the daydream.archive package.

Covers git_context, manifest, index, and the strict ``finalize_archive_run`` flow.
"""

import json
import sqlite3
from collections.abc import Callable, Mapping
from contextlib import ExitStack, closing
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import pytest

from daydream.archive import (
    ArchiveFinalizationError,
    ArchiveIntegrityError,
    finalize_archive_run,
    get_archive_dir,
    pipeline,
)
from daydream.archive.bundle import _copy_snapshot_bundle
from daydream.archive.git_context import GitContext
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
    OutputLabel,
    RoutedDestination,
)
from daydream.artifacts.filesystem import manifest_tree
from daydream.artifacts.models import DestinationDelivery
from daydream.artifacts.ownership import CodeWorkspace
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
from daydream.trajectory import (
    DaydreamPhase,
    DaydreamRunFlow,
    PhaseEvent,
    RunWriteSnapshot,
    TrajectoryDocumentReference,
    TrajectoryDocumentSnapshot,
    sibling_document_path,
    summarize_run,
)
from tests.harness.review_result import review_coverage
from tests.harness.trajectory import make_manifest

MakeConfig = Callable[..., RunConfig]

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


def _build(tmp_path: Path, *, git_ctx: GitContext | None = None, write_snapshot: RunWriteSnapshot | None = None,
    run_flow: DaydreamRunFlow = DaydreamRunFlow.NORMAL, identity: ManifestRunIdentity | None = None, **kw: Any,
) -> Manifest:
    """Build a manifest from immutable public archive inputs."""
    snapshot = write_snapshot or _manifest_write_snapshot()
    return build_manifest_from_snapshot(run=_archive_snapshot(snapshot, run_flow=run_flow, identity=identity,),
        summary=summarize_run(snapshot), git_ctx=git_ctx if git_ctx is not None else GitContext(), status="complete",
        archive_path=tmp_path, **kw,
    )


def _stored_manifest_row(archive_dir: Path, session_id: str, **fields: Any) -> dict[str, Any]:
    upsert_run(archive_dir, make_manifest(session_id=session_id, **fields))
    with closing(sqlite3.connect(f"{(archive_dir / 'index.db').as_uri()}?mode=ro", uri=True)) as conn:
        conn.row_factory = sqlite3.Row
        return dict(conn.execute("SELECT * FROM runs WHERE session_id = ?", (session_id,)).fetchone())


def test_null_in_hunk_rate_survives_manifest_and_db_as_null(tmp_path: Path) -> None:
    """Undefined location accuracy stays NULL so reward reduction can omit the absent axis."""
    m = _build(tmp_path,
        evaluation={"timing": {"total_wall_clock_seconds": 42.5},
            "location": {"hunk_source": "none", "scored_items": 0, "in_hunk_rate": None,},
            "findings": {"total": 0, "shipped_duplication": {"near_duplicate_pairs": 0},},
        },
    )
    assert m.wall_clock_seconds == 42.5  # legacy snapshots without lifecycle use evaluated timing
    assert m.location_in_hunk_rate is None
    assert m.to_dict()["metrics"]["location_in_hunk_rate"] is None

    row = _stored_manifest_row(tmp_path, "s-loc-null", location_in_hunk_rate=m.location_in_hunk_rate,
        shipped_duplicate_pairs=m.shipped_duplicate_pairs,
        wall_clock_seconds=m.wall_clock_seconds,
    )
    assert row["location_in_hunk_rate"] is None  # SQL NULL, not 0.0
    # Reopening the existing current-schema index appends without reinitializing it.
    assert _stored_manifest_row(tmp_path, "s-second")["session_id"] == "s-second"
    assert row["shipped_duplicate_pairs"] == 0  # a real zero is still a zero
    assert row["wall_clock_seconds"] == 42.5


def test_session_bound_artifact_requires_matching_session(tmp_path: Path) -> None:
    resolver = DeepArtifact.RECOMMENDED_CAPTURE.at
    artifact = "recommended-capture.json"
    payload = {"session_id": "sess-42", "capture_point": "post_test"}
    p = tmp_path / ".daydream" / "deep" / artifact
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps(payload))
    assert _read_session_bound_json_artifact(tmp_path, "sess-42", resolver) == payload
    assert _read_session_bound_json_artifact(tmp_path, "sess-other", resolver) is None
    assert _read_session_bound_json_artifact(tmp_path, None, resolver) is None


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


def test_bundle_copies_only_current_flow_artifacts_and_the_registered_findings(tmp_path: Path,) -> None:
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

    # The registered findings destination is relocated from the frozen tree; a malformed route stops the archive.
    findings = target / "explicit" / "findings.json"
    findings.parent.mkdir()
    findings.write_text('{"findings": []}')
    routed = RoutedDestination(OutputLabel.FINDINGS_OUTPUT, findings, findings, findings, DestinationDelivery.DEFERRED)
    recorder.run_flow = DaydreamRunFlow.NORMAL
    normal_run = tmp_path / "normal-run"
    normal_run.mkdir()
    _assemble_bundle(target, normal_run, recorder, destinations=(routed,))
    assert (normal_run / "findings.json").read_text() == '{"findings": []}'
    for malformed in ((routed, routed), (replace(routed, frozen_path=tmp_path / "escaped.json"),)):
        with pytest.raises(ArchiveFinalizationError, match="destination"):
            _assemble_bundle(target, tmp_path / "rejected-run", recorder, destinations=malformed)


# Canonical UTC timestamp contract: one spelling at write time, strict as_of
# validation at the entry boundary, and legacy "Z" rows preserved at bootstrap.


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


def _set_path(payload: dict[str, Any], path: tuple[str | int, ...], value: object) -> None:
    cursor: Any = payload
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value


def _ci_case(*edits: tuple[tuple[str | int, ...], object], status: str = "passed",
    advisory: tuple[CIObservation, ...] = (), edit: Callable[[dict[str, Any]], None] | None = None,
) -> Callable[[Path], None]:
    def seed(target: Path) -> None:
        artifact, payload = _current_ci_artifact(target, status=status, advisory=advisory)
        for path, value in edits:
            _set_path(payload, path, value)
        if edit is not None:
            edit(payload)
        artifact.write_text(json.dumps(payload), encoding="utf-8")
    return seed


def _duplicate_required(payload: dict[str, Any]) -> None:
    payload["required_observations"].append(payload["required_observations"][0])


def _write_unparseable_remote(target: Path) -> None:
    _write_push_verdict(target)
    (target / ".daydream" / "deep" / "remote-ci-verdict.json").write_text("not-json", encoding="utf-8")


def test_remote_ci_evidence_defects_never_report_success(tmp_path: Path) -> None:
    """Every incomplete, mismatched, contradictory or malformed CI receipt degrades to non-success."""
    pending_advisory = CIObservation(source="check_run", context="Build", app_id=10, state="pending",
        raw_state="in_progress", url="https://github.com/example/project/actions/runs/8", diagnostic=None,
    )
    malformed_sha = "a" * 39
    cases: dict[str, tuple[Callable[[Path], None], dict[str, Any]]] = {
        "timed-out": (_ci_case(status="timed_out"), {}),
        **{f"identity-{'.'.join(map(str, path))}": (_ci_case((path, value)), {}) for path, value in (
            (("policy",), None), (("polling", "poll_count"), 0), (("active_workflow_count",), True),
            (("required_observations",), {}), (("limitations",), None),
        )},
        "other-configured-pr": (_ci_case(), {"pr_repo": "other/project"}),
        "noncanonical-base-repository": (_ci_case((("target", "base_repository"), "EXAMPLE/PROJECT"),
            (("binding", "base_repository"), "EXAMPLE/PROJECT")), {"pr_repo": None}),
        "noncanonical-sha": (_ci_case((("binding", "merge_sha"), malformed_sha), (("merge_sha",), malformed_sha),
            (("evidence_sha",), malformed_sha)), {}),
        "no-policy-pending": (_ci_case((("policy",), {"contexts": [], "strict": False}),
            (("required_observations",), []), advisory=(pending_advisory,)), {}),
        "duplicated-producer": (_ci_case(edit=_duplicate_required), {}),
        "wrong-failing-context": (_ci_case((("failing_contexts",), ["Other (app 10)"]), status="failed"), {}),
        **{f"polling-{field}": (_ci_case((("polling", field), value), status="no_ci"), {}) for field, value in (
            ("elapsed_seconds", 0), ("stable_polls", 1), ("discovery_seconds", -1), ("required_stable_polls", True),
        )},
        **{f"malformed-{'.'.join(map(str, path))}": (_ci_case((path, value)), {}) for path, value in (
            (("schema_version",), True), (("status",), {}), (("required_observations", 0, "state"), []),
            (("required_observations", 0, "source"), {}), (("binding", "state"), "closed"),
        )},
        "not-json": (_write_unparseable_remote, {}),
    }
    for status, expected in (("passed", "succeeded"), ("failed", "failed")):
        _ci_case(status=status)(tmp_path / f"control-{status}")
        assert _derive_push_remote_states(tmp_path / f"control-{status}")["remote_ci"]["status"] == expected
    for name, (seed, overrides) in cases.items():
        target = tmp_path / name
        seed(target)
        states = _derive_push_remote_states(target, **overrides)
        assert states["push"]["status"] == "succeeded", name
        assert states["remote_ci"]["ran"] is True, name
        assert states["remote_ci"]["status"] == "partial", name
        assert pipeline.derive_pipeline_status("complete", None, states) == "partial", name


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
    # A prior session's success, or an unbound receipt read by an unbound session, is never adopted.
    _write_deep(tmp_path, "push-verdict.json", {"schema_version": 1, "session_id": "prior", "status": "succeeded"})
    states = _derive_push_remote_states(tmp_path)
    assert states["push"]["status"] == "absent"
    assert states["remote_ci"] == {"ran": False, "status": "absent"}
    _write_push_verdict(tmp_path)
    artifact = tmp_path / ".daydream" / "deep" / "push-verdict.json"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    del payload["session_id"]
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    states = _derive_push_remote_states(tmp_path, session_id=cast(Any, None))
    assert states["push"] == {"ran": False, "status": "absent"}
    assert states["remote_ci"] == {"ran": False, "status": "absent"}


def test_archive_required_success_with_pending_advisory_is_succeeded(
    target: Path, archive_dir: Path, make_config: MakeConfig,
) -> None:
    recorder = _MockRecorder(session_id="advisory-pending-session")
    config = make_config(target, archive=True, pr_repo="example/project", pr_number=42)
    advisory = (CIObservation(
        source="check_run", context="Optional macOS", app_id=11, state="pending", raw_state="in_progress",
        url="https://github.com/example/project/actions/runs/8", diagnostic=None,
    ), CIObservation(
        source="check_run", context="Optional Linux", app_id=11, state="fail", raw_state="failure",
        url="https://github.com/example/project/actions/runs/8", diagnostic="optional job failed",
    ))
    _write_deep(target, "test-verdict.json", {"session_id": recorder.session_id, "passed": True})
    _write_push_verdict(target, session_id=recorder.session_id)
    _write_remote_verdict(target, session_id=recorder.session_id, advisory=advisory)
    artifact = target / ".daydream" / "deep" / "remote-ci-verdict.json"
    payload = json.loads(artifact.read_text())
    evaluated = evaluate_remote_ci(RemoteCISnapshot(
            target=RemoteCITarget(target_dir=target, **payload["target"]), binding=PRCIBinding(**payload["binding"]),
            policy=RequiredPolicy((RequiredContext("Build", 10),), True),
            active_workflows=({"id": 7, "state": "active"},), head_observations=(),
            merge_observations=(CIObservation(**payload["required_observations"][0]), *advisory),
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
    assert manifest["phase_states"]["remote_ci"]["details"]["advisory_failures"] == ["Optional Linux"]
    assert manifest["pipeline_status"] == "succeeded"
    copied = json.loads((run_dir / "deep" / "remote-ci-verdict.json").read_text())
    assert [(row["context"], row["state"]) for row in copied["advisory_observations"]] == [
        ("Optional macOS", "pending"), ("Optional Linux", "fail")]


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


def test_current_merge_event_controls_pipeline_status(tmp_path: Path) -> None:
    def merge_states(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return derive_phase_states(
            tmp_path, phase_events=events, runs_merge=True, runs_fix=False, runs_test=False, session_id="current",
        )

    states = merge_states(_merge_events("current", "partial"))
    assert states["merge"] == {"ran": True, "status": "partial"}
    assert derive_pipeline_status("complete", None, states, runs_merge=True) == "partial"
    stale = merge_states([*_merge_events("prior", "failed", scope_id="prior-merge"),
                          *_merge_events("current", "succeeded")])
    assert stale["merge"] == {"ran": True, "status": "succeeded"}
    start, end = _merge_events("current", "succeeded")
    for events in (
        [start, end, end],
        [{**start, "timestamp": "2026-01-01T00:00:02Z"}, end],
        [{**start, "timestamp": "2026-01-01T00:00:00"}, end],
        [{**start, "session_id": None}, end],
        [start, {**end, "event": {}}],
    ):
        states = merge_states(events)
        assert states["merge"] == {"ran": True, "status": "unknown"}
        assert derive_pipeline_status("complete", None, states, runs_merge=True) == "unknown"


def test_merge_artifacts_cannot_supply_missing_run_identity(tmp_path: Path) -> None:
    _write_review_coverage(tmp_path, state="complete")
    _write_deep(tmp_path, "merged-items.json", {"items": "corrupt"})
    states = pipeline.derive_phase_states(
        tmp_path, phase_events=[], runs_merge=True, runs_fix=False, runs_test=False, session_id=None,
    )
    assert states["merge"] == {"ran": True, "status": "unknown"}


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


def test_test_failed_from_verdict(tmp_path: Path) -> None:
    _write_deep(
        tmp_path, "test-verdict.json", {"session_id": "current", "passed": False, "retries": 1, "ignored": False},
    )
    states = pipeline.derive_phase_states(tmp_path, phase_events=[], session_id="current")
    assert states["test"]["ran"] is True
    assert states["test"]["status"] == "failed"
    # A command that never executed is a failed test phase that ran only if the host attempted it.
    _write_deep(tmp_path, "test-verdict.json", {"session_id": "current", "status": "no-execution",
                                                "attempts": [{"kind": "host"}], "diagnostic": "no_test_command"})
    states = pipeline.derive_phase_states(tmp_path, phase_events=[], session_id="current")
    assert states["test"] == {"ran": True, "status": "failed", "diagnostic": "no_test_command"}
    # Recorded per-group fix failures make a deep run partial even though its trajectory completed.
    fixed = tmp_path / "fix-failures"
    _write_deep(fixed, "fix-failures.json", {"api.py": "verification failed"})
    state = pipeline._manifest_state(
        target_dir=fixed, run=_archive_snapshot(_manifest_write_snapshot(session_id="current")), frozen_extra={},
    )
    assert state["status"] == "partial" and state["fix_failures"] == {"api.py": "verification failed"}
    assert state["phase_states"]["fix"] == {"ran": True, "status": "partial"}
    assert state["pipeline_status"] == "partial"


def test_stale_or_malformed_stabilization_failure_is_neutral(tmp_path: Path) -> None:
    payload = ["malformed"]
    _write_deep(tmp_path, "test-verdict.json", {"session_id": "current", "passed": True})
    _write_deep(tmp_path, "stabilization-failed.json", payload)
    states = pipeline.derive_phase_states(tmp_path, phase_events=[], session_id="current")
    assert states["fix"] == {"ran": False, "status": "absent"}
    assert states["test"] == {"ran": True, "status": "succeeded"}


def test_snapshot_manifest_provenance_rejects_malformed_identity(tmp_path: Path) -> None:
    for session_id, extra, message in (
        ("session", {"pr_number": True}, "pr_number"), ("session", {"pr_number": 7, "pr_repo": None}, "pr_repo"),
        ("..", None, "session_id"),
    ):
        snapshot = _manifest_write_snapshot(
            session_id=session_id, extra=extra, final_metrics={}, path=tmp_path / "trajectory.json",
        )
        with pytest.raises(ValueError, match=message):
            archive_recorder_provenance_from_snapshot(write_snapshot=snapshot, run_flow=DaydreamRunFlow.NORMAL)
    # Finalization refuses artifact evidence frozen for a different session.
    arguments = _finalizer_arguments(tmp_path / "mismatch", "frozen-session", config=RunConfig(run_eval=False))
    arguments["artifacts"] = replace(arguments["artifacts"], session_id="other-session")
    with pytest.raises(ArchiveFinalizationError, match="identity mismatch"):
        finalize_archive_run(**arguments)
    assert not (get_archive_dir() / "runs" / "frozen-session").exists()


def _finalizer_arguments(tmp_path: Path, session_id: str, *, config: RunConfig, child_steps: tuple[Any, ...] = (),
) -> dict[str, Any]:
    """Freeze a prepared root and a completed-child file reference, binding the finalizer's roots."""
    frozen = tmp_path / "frozen"
    path = frozen / ".daydream" / "runs" / session_id / "trajectory.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = _manifest_write_snapshot(session_id=session_id, final_metrics={}, path=path)
    path.write_bytes(snapshot.documents[0].read_bytes())
    # Completed children reach consumers as references into the frozen copy, named by their logical path.
    child = json.dumps({"session_id": session_id, "trajectory_id": "fork-1", "steps": list(child_steps)}).encode()
    private_child = sibling_document_path(path.parent, "frozen-child.json")
    private_child.parent.mkdir()
    private_child.write_bytes(child)
    logical_child = sibling_document_path(tmp_path / "live" / "runs" / session_id, "review.json")
    snapshot = replace(snapshot, documents=(
        *snapshot.documents, TrajectoryDocumentReference.written("fork-1", logical_child, private_child, child),
    ))
    return dict(run=_archive_snapshot(snapshot),
        artifacts=ArtifactTreeSnapshot(session_id, "workspace", frozen, manifest_tree(frozen), ()),
        artifact_provenance=ArtifactEvidenceProvenance("workspace", session_id, tmp_path / "source", tmp_path / "live",
        ), config=config, work=None,
    )


_EVALUATION_FAILURES: tuple[str | tuple[tuple[str, ...], Any], ...] = (
    "error-result", "malformed-result", "empty-result", "wrong-quality", "empty-quality", "invalid-unavailable",
    (("findings",), {}), (("findings", "shipped_duplication"), {}),
    (("findings", "shipped_duplication", "near_duplicate_pairs"), -1), (("location", "in_hunk_rate"), 2.0),
    (("derived", "cost_per_finding_usd"), float("inf")),
)


def test_strict_archive_evaluation_failure_is_typed_and_never_reports_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The host finalizer cannot infer success from the legacy fail-open wrapper."""
    from daydream.eval.analyzer import analyze_session

    session_id = "strict-session"
    for index, failure in enumerate(_EVALUATION_FAILURES):
        root = tmp_path / str(index)
        arguments = _finalizer_arguments(root, session_id, config=RunConfig(target=str(root), run_eval=True))

        def fail_evaluation(*args: Any, failure: Any = failure, **kwargs: Any) -> Any:
            if failure == "error-result":
                return {"error": "no trajectory"}
            if failure == "malformed-result":
                return None
            if failure == "empty-result":
                return {}
            evaluation = analyze_session(*args, **kwargs)
            if isinstance(failure, tuple):
                path, value = failure
                section = evaluation
                for key in path[:-1]:
                    section = section[key]
                section[path[-1]] = value
            elif failure == "empty-quality":
                evaluation["quality"] = {}
            elif failure == "invalid-unavailable":
                evaluation["quality"]["erosion"] = 0.0
            else:
                evaluation["quality"] = None
            return evaluation

        with monkeypatch.context() as patch:
            patch.setattr("daydream.eval.analyzer.analyze_session", fail_evaluation)
            try:
                with pytest.raises(ArchiveFinalizationError, match="evaluation"):
                    finalize_archive_run(**arguments)
            except BaseException as error:
                error.add_note(f"evaluation failure case: {failure!r}")
                raise
        assert not (get_archive_dir() / "runs" / session_id).exists(), failure
        assert not (get_archive_dir() / "index.db").exists(), failure
    # Real malformed child steps fail only the evaluation that reads them; an archive-only run installs.
    archived = get_archive_dir() / "runs" / session_id
    for run_eval in (True, False):
        root = tmp_path / f"malformed-steps-{run_eval}"
        arguments = _finalizer_arguments(root, session_id, config=RunConfig(target=str(root), run_eval=run_eval),
                                         child_steps=({"tool_calls": [{"tool_call_id": "unnamed"}]},))
        if run_eval:
            with pytest.raises(ArchiveFinalizationError, match="evaluation"):
                finalize_archive_run(**arguments)
            assert not archived.exists()
        else:
            finalize_archive_run(**arguments)
            assert json.loads((archived / "manifest.json").read_text())["session_id"] == session_id
            assert not (archived / "evaluation.json").exists()


def test_strict_archive_reattests_live_workspace_before_quality(
    tmp_path: Path, archive_dir: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A source changed during a real frozen-artifact read cannot evade the initial Git check."""
    with ExitStack() as held:
        session_id = "late-source-change"
        receipt = tmp_path / "frozen/.daydream/deep/merged-items.json"
        receipt.parent.mkdir(parents=True)
        receipt.write_text('{"items": [{"confidence": "HIGH"}]}')
        arguments = _finalizer_arguments(tmp_path, session_id, config=RunConfig(run_eval=True))
        source = arguments["artifact_provenance"].public_source
        source.mkdir()
        arguments["code_workspace"] = held.enter_context(CodeWorkspace.standalone(source))
        replacement = tmp_path / "replacement"
        replacement.mkdir()
        marker = replacement / "app.py"
        marker.write_bytes(b"VALUE = 'replacement code'\n")
        original_read = Path.read_text
        moved = tmp_path / "moved-source"

        def read_then_change(path: Path, *args: Any, **kwargs: Any) -> str:
            content = original_read(path, *args, **kwargs)
            if path == receipt and not moved.exists():
                source.rename(moved)
                replacement.rename(source)
            return content

        monkeypatch.setattr(Path, "read_text", read_then_change)
        archived = archive_dir / "runs" / session_id
        with pytest.raises(ArchiveFinalizationError, match="evaluation"):
            finalize_archive_run(**arguments)
        assert not archived.exists()
        assert not (archive_dir / "index.db").exists()
        assert moved.is_dir()
        retained_marker = source / marker.name
        assert retained_marker.read_bytes() == b"VALUE = 'replacement code'\n"
        assert receipt.read_bytes() == b'{"items": [{"confidence": "HIGH"}]}'


def test_strict_archive_rejects_frozen_receipt_changed_by_evaluator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
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
    from daydream.eval.analyzer import analyze_session

    def mutate_frozen_receipt(*args: Any, **kwargs: Any) -> dict[str, Any]:
        evaluation = analyze_session(*args, **kwargs)
        receipt.write_text(json.dumps({"session_id": session_id, "passed": False}), encoding="utf-8",)
        return evaluation

    monkeypatch.setattr("daydream.eval.analyzer.analyze_session", mutate_frozen_receipt,)

    with pytest.raises(ArchiveFinalizationError, match="frozen artifact tree changed"):
        finalize_archive_run(**arguments)

    archive_dir = get_archive_dir()
    assert not (archive_dir / "runs" / session_id).exists()
    assert not (archive_dir / "index.db").exists()


@pytest.mark.parametrize(("mutated", "message"), [
    (0, "frozen artifact tree changed"),
    # A completed child changed after its checked bundle copy fails the summary pass as integrity, not optional.
    (1, "frozen trajectory document changed"),
])
def test_strict_archive_refuses_frozen_tree_mutated_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutated: int, message: str,
) -> None:
    session_id = "strict-mutated-publication"
    arguments = _finalizer_arguments(
        tmp_path, session_id, config=RunConfig(target=str(tmp_path), run_eval=False, archive=True),
    )
    document = arguments["run"].trajectories.documents[mutated]
    private_path = getattr(document, "private_path", document.path)
    from daydream.archive.git_context import capture_git_context

    def mutate_then_capture(*args: Any) -> Any:
        private_path.write_bytes(document.read_bytes() + b"\n")
        return capture_git_context(*args)

    monkeypatch.setattr("daydream.archive.finalize.capture_git_context", mutate_then_capture)
    with pytest.raises(ArchiveIntegrityError, match=message):
        finalize_archive_run(**arguments)
    assert not (get_archive_dir() / "runs" / session_id).exists()
