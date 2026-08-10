"""Tests for the strictly-passive worker artifact envelope (daydream/service/artifact.py)."""

from dataclasses import FrozenInstanceError
from typing import Any

import pytest

from daydream.service.artifact import MAX_FINDINGS, WorkerArtifactV1
from daydream.service.models import ReviewJobV1, ReviewTargetV1


def _job() -> ReviewJobV1:
    return ReviewJobV1(
        job_id="job-1",
        idempotency_key="idem-1",
        target=ReviewTargetV1(
            target_kind="pr_head",
            repo="acme/demo",
            candidate_sha="a" * 40,
            candidate_tree_digest="b" * 40,
            base_sha="c" * 40,
            pr_numbers=(7,),
            full_diff_digest="d" * 64,
            invalidation_id="inv-1",
        ),
        effective_config_digest="e" * 64,
        reviewer_bundle_digest="f" * 64,
        required_lenses=("python",),
        round=1,
        attempt=1,
        deadline="2030-01-01T00:00:00Z",
        created_at="2030-01-01T00:00:00Z",
    )


def _finding(*, severity: str = "low", extra: dict | None = None) -> dict:
    f = {
        "id": 1,
        "lens": "python",
        "file": "main.py",
        "line": 1,
        "severity": severity,
        "confidence": "HIGH",
        "title": "T",
        "body": "B",
    }
    if extra:
        f.update(extra)
    return f


def _clean(job: ReviewJobV1 | None = None) -> WorkerArtifactV1:
    return WorkerArtifactV1.complete(
        job or _job(),
        completed_lenses=("python",),
        findings=(),
        timestamps={"started_at": "2030-01-01T00:00:00Z", "finished_at": "2030-01-01T00:00:01Z"},
    )


def test_complete_helper_produces_clean_for_missing_free_run() -> None:
    a = _clean()
    assert a.terminal == "clean"
    assert a.missing_lenses == ()
    assert a.process_outcome == "exited_0"
    assert a.job_id == "job-1"
    assert a.idempotency_key == "idem-1"


def test_blocking_findings_keep_findings_even_when_process_exited_0() -> None:
    job = _job()
    a = WorkerArtifactV1.complete(
        job,
        completed_lenses=("python",),
        findings=(_finding(severity="high"), _finding(severity="low", extra={"id": 2})),
        process_outcome="exited_0",
    )
    assert a.terminal == "findings"
    assert a.process_outcome == "exited_0"
    assert len(a.findings) == 2


def test_non_blocking_findings_with_exit_0_are_clean() -> None:
    job = _job()
    a = WorkerArtifactV1.complete(
        job,
        completed_lenses=("python",),
        findings=(_finding(severity="low"),),
        process_outcome="exited_0",
    )
    assert a.terminal == "clean"
    assert a.process_outcome == "exited_0"


def test_infra_error_helper_sets_infra_terminal() -> None:
    job = _job()
    a = WorkerArtifactV1.infra_error(
        job,
        process_outcome="budget_exhausted",
        completed_lenses=("python",),
        missing_lenses=(),
    )
    assert a.terminal == "infra_error"
    assert a.process_outcome == "budget_exhausted"


def test_cancelled_helper() -> None:
    job = _job()
    a = WorkerArtifactV1.cancelled(job, completed_lenses=(), missing_lenses=("python",))
    assert a.terminal == "cancelled"
    assert a.process_outcome == "cancelled"
    assert a.missing_lenses == ("python",)


def test_clean_cannot_carry_missing_lenses() -> None:
    with pytest.raises(ValueError):
        WorkerArtifactV1(
            job_id="job-1",
            idempotency_key="idem-1",
            terminal="clean",
            completed_lenses=("python",),
            missing_lenses=("python",),
            process_outcome="exited_0",
            findings=(),
            hashes={},
            timestamps={},
        )


def test_clean_cannot_carry_blocking_findings() -> None:
    with pytest.raises(ValueError):
        WorkerArtifactV1(
            job_id="job-1",
            idempotency_key="idem-1",
            terminal="clean",
            completed_lenses=("python",),
            missing_lenses=(),
            process_outcome="exited_0",
            findings=(_finding(severity="high"),),
            hashes={},
            timestamps={},
        )


def test_findings_terminal_requires_blocking_finding() -> None:
    with pytest.raises(ValueError):
        WorkerArtifactV1(
            job_id="job-1",
            idempotency_key="idem-1",
            terminal="findings",
            completed_lenses=("python",),
            missing_lenses=(),
            process_outcome="exited_0",
            findings=(_finding(severity="low"),),
            hashes={},
            timestamps={},
        )


def test_infra_error_cannot_report_a_clean_process() -> None:
    with pytest.raises(ValueError):
        WorkerArtifactV1.infra_error(_job(), process_outcome="exited_0")


def test_cancelled_requires_cancelled_process_outcome() -> None:
    with pytest.raises(ValueError):
        WorkerArtifactV1(
            job_id="job-1",
            idempotency_key="idem-1",
            terminal="cancelled",
            completed_lenses=(),
            missing_lenses=("python",),
            process_outcome="budget_exhausted",
            findings=(),
            hashes={},
            timestamps={},
        )


def test_unknown_terminal_rejected() -> None:
    with pytest.raises(ValueError):
        WorkerArtifactV1(
            job_id="job-1",
            idempotency_key="idem-1",
            terminal="winsome",
            completed_lenses=(),
            missing_lenses=(),
            process_outcome=None,
            findings=(),
            hashes={},
            timestamps={},
        )


def test_unknown_process_outcome_rejected() -> None:
    with pytest.raises(ValueError):
        WorkerArtifactV1(
            job_id="job-1",
            idempotency_key="idem-1",
            terminal="infra_error",
            completed_lenses=(),
            missing_lenses=(),
            process_outcome="bogus",
            findings=(),
            hashes={},
            timestamps={},
        )


def test_artifact_is_frozen_and_strictly_passive() -> None:
    a = _clean()
    with pytest.raises(FrozenInstanceError):
        a.terminal = "findings"  # type: ignore[misc]
    # No worker-asserted infrastructure identity anywhere on the envelope.
    payload = a.to_dict()
    for banned in ("executor", "lease", "pod", "vm", "attempt_binding", "handle", "kind"):
        assert banned not in payload, f"artifact must not expose {banned!r}"


def test_findings_must_be_strict_and_homogeneous() -> None:
    with pytest.raises(ValueError):
        WorkerArtifactV1.complete(
            _job(),
            completed_lenses=("python",),
            findings=({"id": 1, "lens": "python", "file": "main.py", "line": 1, "severity": "high",
                       "confidence": "HIGH", "title": "T", "body": "B", "surprise": True},),
        )
    with pytest.raises(ValueError):
        WorkerArtifactV1.complete(
            _job(),
            completed_lenses=("python",),
            findings=({"id": 1, "lens": "python", "file": "main.py", "line": 1, "severity": "high",
                       "confidence": "HIGH", "title": "T"},),  # missing body
        )


def test_findings_bounded() -> None:
    too_many = tuple(_finding(extra={"id": i}) for i in range(MAX_FINDINGS + 1))
    with pytest.raises(ValueError):
        WorkerArtifactV1.complete(_job(), completed_lenses=("python",), findings=too_many)


def test_hashes_must_be_digests_not_infra_pointers() -> None:
    with pytest.raises(ValueError):
        WorkerArtifactV1.infra_error(
            _job(), process_outcome="process_loss", hashes={"runner": "https://executor.example/abc"}
        )


def test_timestamps_must_be_iso() -> None:
    with pytest.raises(ValueError):
        WorkerArtifactV1.infra_error(
            _job(), process_outcome="process_loss", timestamps={"started_at": "not-a-date"}
        )


def test_to_dict_from_dict_round_trip() -> None:
    a = _clean()
    assert WorkerArtifactV1.from_dict(a.to_dict()) == a


def test_from_dict_rejects_unknown_field() -> None:
    data = _clean().to_dict()
    data["executor_kind"] = "k8s"
    with pytest.raises(ValueError):
        WorkerArtifactV1.from_dict(data)


def test_from_dict_rejects_invalid_terminal() -> None:
    data = _clean().to_dict()
    data["terminal"] = "winsome"
    with pytest.raises(ValueError):
        WorkerArtifactV1.from_dict(data)


# --- Manifest exact provenance (Plan 008 Step 2 leaf, additive) --------------


class _RecorderStub:
    """Minimal TrajectoryRecorder stand-in for build_manifest provenance tests."""

    session_id = "sess-12345678-0000-0000-0000-000000000000"
    run_flow = type("RunFlow", (), {"value": "deep"})()
    explicit_path = False
    pr_number: int | None = None
    pr_repo: str | None = None
    on_write = None
    _final_totals = {
        "prompt": 0,
        "completion": 0,
        "cached": 0,
        "cost": None,
        "any_cost_seen": False,
    }
    _wall_clock_seconds: float | None = None

    def compute_wall_clock_seconds(self) -> float | None:
        return self._wall_clock_seconds

    def compute_phase_timings(self) -> dict[str, Any] | None:
        return None


def _build_manifest_provenance(
    tmp_path, config, *, status: str = "complete",
) -> dict[str, Any]:
    from typing import cast

    from daydream.archive.git_context import GitContext
    from daydream.archive.manifest import build_manifest
    from daydream.trajectory import TrajectoryRecorder

    m = build_manifest(
        recorder=cast(TrajectoryRecorder, _RecorderStub()),
        config=config,
        git_ctx=GitContext(),
        status=status,
        archive_path=tmp_path,
    )
    return m.to_dict()["provenance"]


def test_manifest_provenance_records_resolved_backend_model_provider(tmp_path, monkeypatch) -> None:
    import platform

    from daydream.runner import RunConfig

    monkeypatch.setenv("PI_PROVIDER", "zai")
    config = RunConfig(
        backend="pi",
        model="deepseek/deepseek-v4-flash-0731",
        skill="python",
        output_mode="review",
        non_interactive=True,
        cleanup=False,
        archive=False,
    )
    p = _build_manifest_provenance(tmp_path, config)
    assert p["backend"] == "pi"
    # Exactly as resolved, not raw config defaults.
    assert p["model"] == "deepseek/deepseek-v4-flash-0731"
    assert p["provider"] == "zai"
    assert p["skill"] == "python"
    assert p["runtime"]["python"] == platform.python_version()
    assert isinstance(p["runtime"]["uv"], (str, type(None)))


def test_manifest_provenance_pi_default_provider_and_none_for_claude(tmp_path, monkeypatch) -> None:
    from daydream.runner import RunConfig

    monkeypatch.delenv("PI_PROVIDER", raising=False)
    pi_config = RunConfig(backend="pi", model="deepseek/deepseek-v4-flash-0731", skill="python",
                          output_mode="review", non_interactive=True, cleanup=False, archive=False)
    assert _build_manifest_provenance(tmp_path, pi_config)["provider"] == "nous"

    claude_config = RunConfig(backend="claude", model="claude-opus-4-5", skill="python",
                              output_mode="review", non_interactive=True, cleanup=False, archive=False)
    p = _build_manifest_provenance(tmp_path, claude_config)
    assert p["backend"] == "claude"
    assert p["provider"] is None
    assert p["model"] == "claude-opus-4-5"


def test_manifest_provenance_config_digest_only_when_file_config_present(tmp_path) -> None:
    from daydream.config_file import DaydreamFileConfig
    from daydream.runner import RunConfig

    bare = RunConfig(backend="claude", skill="python", output_mode="review",
                     non_interactive=True, cleanup=False, archive=False)
    assert _build_manifest_provenance(tmp_path, bare)["config"] is None

    with_config = RunConfig(
        file_config=DaydreamFileConfig(model="custom-model", phases={"fix": {"backend": "codex"}}),
        skill="python",
        output_mode="review",
        non_interactive=True,
        cleanup=False,
        archive=False,
    )
    p = _build_manifest_provenance(tmp_path, with_config)
    assert p["config"] is not None
    assert "digest" in p["config"]
    assert len(p["config"]["digest"]) == 64  # sha256 hex

    # Same effective config -> same digest (stable provenance).
    again = RunConfig(
        file_config=DaydreamFileConfig(model="custom-model", phases={"fix": {"backend": "codex"}}),
        skill="python",
        output_mode="review",
        non_interactive=True,
        cleanup=False,
        archive=False,
    )
    assert _build_manifest_provenance(tmp_path, again)["config"] == p["config"]


def test_manifest_resolves_model_through_file_config_not_raw(tmp_path) -> None:
    from daydream.config_file import DaydreamFileConfig
    from daydream.runner import RunConfig

    # model set ONLY via file config; the manifest must record it as resolved.
    config = RunConfig(
        file_config=DaydreamFileConfig(model="file-configured-model"),
        backend="claude",
        skill="python",
        output_mode="review",
        non_interactive=True,
        cleanup=False,
        archive=False,
    )
    p = _build_manifest_provenance(tmp_path, config)
    assert p["model"] == "file-configured-model"
    assert p["backend"] == "claude"


def test_manifest_provenance_preserves_existing_consumed_keys(tmp_path) -> None:
    from typing import cast

    from daydream.archive.git_context import GitContext
    from daydream.archive.manifest import build_manifest
    from daydream.runner import RunConfig
    from daydream.trajectory import TrajectoryRecorder

    config = RunConfig(backend="claude", skill="python", output_mode="loop",
                       non_interactive=True, cleanup=False, archive=False)
    m = build_manifest(
        recorder=cast(TrajectoryRecorder, _RecorderStub()),
        config=config,
        git_ctx=GitContext(),
        status="complete",
        archive_path=tmp_path,
    )
    d = m.to_dict()
    # Keys consumed by training/labelers keep their exact shapes.
    assert "recommended_patch_supported" in d
    assert set(d["run"]) >= {"flow", "skill", "model", "backend", "review_only", "deep"}
    assert set(d["git"]) >= {
        "head_sha", "source_path", "remote_url", "repo_slug", "branch", "base_branch",
    }
    assert set(d["code_context"]) == {"head_sha", "base_sha", "base_branch", "branch", "changed_files"}
    assert set(d["metrics"]) >= {
        "total_cost_usd", "total_prompt_tokens", "total_completion_tokens", "total_cached_tokens",
    }
    assert set(d["outcome"]) == {"labels", "labeled_at", "composite_reward"}
    # And the additive provenance block is present.
    assert d["provenance"]["backend"] == "claude"
