# tests/test_archive.py
"""Unit tests for the daydream.archive package.

Covers git_context, manifest, index, and the top-level archive_run flow.
"""

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from daydream.archive import _copy_bundle, archive_run, get_archive_dir
from daydream.archive.git_context import GitContext, _parse_repo_slug, capture_git_context
from daydream.archive.index import (
    append_label_observation,
    label_observation_history,
    latest_label_observation,
    query_runs,
    update_labels,
    upsert_run,
)
from daydream.archive.manifest import Manifest, build_manifest

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


@dataclass
class _MockRecorder:
    session_id: str = "abcd1234-0000-0000-0000-000000000000"
    path: Path = field(default_factory=lambda: Path("/nonexistent/trajectory.json"))
    run_flow: MagicMock = field(default_factory=lambda: MagicMock(value="normal"))
    explicit_path: bool = False
    pr_number: int | None = None
    pr_repo: str | None = None
    _final_totals: dict = field(
        default_factory=lambda: {
            "prompt": 100,
            "completion": 50,
            "cached": 20,
            "cost": 0.05,
            "any_cost_seen": True,
        },
    )


@dataclass
class _MockConfig:
    skill: str | None = "python"
    backend: str = "claude"
    review_backend: str | None = None
    fix_backend: str | None = None
    test_backend: str | None = None
    output_mode: str = "loop"
    shallow: bool = False
    loop: bool = False
    archive: bool = True
    run_eval: bool = False


# ---------------------------------------------------------------------------
# git_context: _parse_repo_slug
# ---------------------------------------------------------------------------


def test_parse_repo_slug_ssh():
    assert _parse_repo_slug("git@github.com:org/repo.git") == "org/repo"


def test_parse_repo_slug_https():
    assert _parse_repo_slug("https://github.com/org/repo.git") == "org/repo"


def test_parse_repo_slug_https_no_dot_git():
    assert _parse_repo_slug("https://github.com/org/repo") == "org/repo"


def test_parse_repo_slug_invalid():
    assert _parse_repo_slug("not-a-url") is None


# ---------------------------------------------------------------------------
# git_context: capture_git_context
# ---------------------------------------------------------------------------


def test_capture_git_context_real_repo(tmp_path: Path):
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


def test_capture_git_context_no_repo(tmp_path: Path):
    ctx = capture_git_context(tmp_path)
    assert ctx.head_sha is None
    assert ctx.remote_url is None
    assert ctx.branch is None
    assert ctx.base_sha is None
    assert ctx.changed_files == []


def test_capture_git_context_populates_base_sha_and_changed_files(tmp_path: Path):
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


# ---------------------------------------------------------------------------
# manifest: build_manifest
# ---------------------------------------------------------------------------


def test_build_manifest_basic(tmp_path: Path):
    recorder = _MockRecorder()
    config = _MockConfig()
    git_ctx = GitContext(
        remote_url="git@github.com:org/repo.git",
        repo_slug="org/repo",
        branch="main",
        base_branch="main",
        head_sha="a" * 40,
    )

    m = build_manifest(
        recorder=recorder,
        config=config,
        git_ctx=git_ctx,
        status="complete",
        archive_path=tmp_path,
    )

    assert m.session_id == recorder.session_id
    assert m.run_flow == "normal"
    assert m.skill == "python"
    # Manifest.model is no longer populated from config (per-phase models replaced
    # the single config.model field); build_manifest stamps it as None.
    assert m.model is None
    assert m.backend == "claude"
    assert m.total_cost_usd == 0.05
    assert m.total_prompt_tokens == 100
    assert m.total_completion_tokens == 50
    assert m.total_cached_tokens == 20
    assert m.repo_slug == "org/repo"
    assert m.head_sha == "a" * 40


def test_manifest_to_dict_structure(tmp_path: Path):
    recorder = _MockRecorder()
    config = _MockConfig()
    git_ctx = GitContext()

    m = build_manifest(
        recorder=recorder,
        config=config,
        git_ctx=git_ctx,
        status="complete",
        archive_path=tmp_path,
    )

    d = m.to_dict()
    assert d["schema_version"] == "1.0"
    assert d["session_id"] == recorder.session_id
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


def test_manifest_to_dict_code_context_carries_git_ctx_fields(tmp_path: Path):
    recorder = _MockRecorder()
    config = _MockConfig()
    git_ctx = GitContext(
        branch="feat/x",
        base_branch="main",
        head_sha="b" * 40,
        base_sha="c" * 40,
        changed_files=["a.py", "b.py"],
    )

    m = build_manifest(
        recorder=recorder,
        config=config,
        git_ctx=git_ctx,
        status="complete",
        archive_path=tmp_path,
    )

    d = m.to_dict()
    assert d["code_context"] == {
        "base_sha": "c" * 40,
        "head_sha": "b" * 40,
        "base_branch": "main",
        "branch": "feat/x",
        "changed_files": ["a.py", "b.py"],
    }


def test_build_manifest_with_evaluation(tmp_path: Path):
    recorder = _MockRecorder()
    config = _MockConfig()
    git_ctx = GitContext()
    evaluation = {
        "timing": {"total_wall_clock_seconds": 42.5},
        "findings": {"total": 7},
        "grounding": {"grounding_rate": 0.85},
        "coverage": {"coverage_ratio": 0.6},
        "derived": {"cost_per_finding_usd": 0.007},
    }

    m = build_manifest(
        recorder=recorder,
        config=config,
        git_ctx=git_ctx,
        status="complete",
        archive_path=tmp_path,
        evaluation=evaluation,
    )

    assert m.wall_clock_seconds == 42.5
    assert m.total_findings == 7
    assert m.grounding_rate == 0.85
    assert m.coverage_ratio == 0.6
    assert m.cost_per_finding_usd == 0.007


def test_build_manifest_without_evaluation(tmp_path: Path):
    recorder = _MockRecorder()
    config = _MockConfig()
    git_ctx = GitContext()

    m = build_manifest(
        recorder=recorder,
        config=config,
        git_ctx=git_ctx,
        status="complete",
        archive_path=tmp_path,
    )

    assert m.wall_clock_seconds is None
    assert m.total_findings is None
    assert m.grounding_rate is None
    assert m.coverage_ratio is None
    assert m.cost_per_finding_usd is None


# ---------------------------------------------------------------------------
# index: upsert_run / query_runs
# ---------------------------------------------------------------------------


def _make_manifest(session_id: str = "sess-0001", **overrides) -> Manifest:
    defaults = {
        "session_id": session_id,
        "archived_at": "2026-04-29T00:00:00+00:00",
        "status": "complete",
        "run_flow": "normal",
        "skill": "python",
        "model": "opus",
        "backend": "claude",
        "archive_path": "/tmp/archive/runs/sess-0001",
    }
    defaults.update(overrides)
    return Manifest(**defaults)


def test_upsert_run_creates_db(tmp_path: Path):
    m = _make_manifest()
    upsert_run(tmp_path, m)
    assert (tmp_path / "index.db").exists()


def test_upsert_and_query_round_trip(tmp_path: Path):
    m = _make_manifest()
    upsert_run(tmp_path, m)

    rows = query_runs(tmp_path)
    assert len(rows) == 1
    assert rows[0]["session_id"] == "sess-0001"
    assert rows[0]["skill"] == "python"
    assert rows[0]["status"] == "complete"


def test_update_labels_exact(tmp_path: Path):
    upsert_run(tmp_path, _make_manifest())

    ok = update_labels(tmp_path, "sess-0001", ["good", "fast"])
    assert ok is True

    rows = query_runs(tmp_path)
    assert json.loads(rows[0]["outcome_labels"]) == ["good", "fast"]
    assert rows[0]["labeled_at"] is not None


def test_update_labels_prefix(tmp_path: Path):
    upsert_run(tmp_path, _make_manifest(session_id="abcd1234-full-uuid"))

    ok = update_labels(tmp_path, "abcd1234", ["label-a"])
    assert ok is True

    rows = query_runs(tmp_path)
    assert json.loads(rows[0]["outcome_labels"]) == ["label-a"]


def test_update_labels_nonexistent(tmp_path: Path):
    upsert_run(tmp_path, _make_manifest())

    ok = update_labels(tmp_path, "no-such-session", [])
    assert ok is False


def test_update_labels_ambiguous_prefix(tmp_path: Path):
    upsert_run(tmp_path, _make_manifest(session_id="abc-001"))
    upsert_run(tmp_path, _make_manifest(session_id="abc-002", archive_path="/tmp/x"))

    with pytest.raises(ValueError, match="matches 2 sessions"):
        update_labels(tmp_path, "abc", ["x"])


def test_query_runs_with_where(tmp_path: Path):
    upsert_run(tmp_path, _make_manifest(session_id="s1", repo_slug="org/a"))
    upsert_run(tmp_path, _make_manifest(session_id="s2", repo_slug="org/b", archive_path="/tmp/s2"))
    upsert_run(tmp_path, _make_manifest(session_id="s3", repo_slug="org/a", archive_path="/tmp/s3"))

    rows = query_runs(tmp_path, where="repo_slug = ?", params=("org/a",))
    assert len(rows) == 2
    ids = {r["session_id"] for r in rows}
    assert ids == {"s1", "s3"}


# ---------------------------------------------------------------------------
# __init__: get_archive_dir
# ---------------------------------------------------------------------------


def test_get_archive_dir_creates_structure(monkeypatch, tmp_path: Path):
    target = tmp_path / "custom_archive"
    monkeypatch.setenv("DAYDREAM_ARCHIVE_DIR", str(target))

    result = get_archive_dir()
    assert result == target
    assert target.is_dir()
    assert (target / "runs").is_dir()


def test_get_archive_dir_default(monkeypatch, tmp_path: Path):
    monkeypatch.delenv("DAYDREAM_ARCHIVE_DIR", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

    result = get_archive_dir()
    expected = tmp_path / ".daydream" / "archive"
    assert result == expected
    assert expected.is_dir()


def test_archive_dir_fixture_isolates_env(archive_dir: Path, tmp_path: Path):
    """Verify the autouse archive_dir fixture's contract: env points at tmp_path/archive."""
    assert os.environ.get("DAYDREAM_ARCHIVE_DIR") == str(archive_dir)
    assert archive_dir == tmp_path / "archive"
    assert str(archive_dir).startswith(str(tmp_path))


# ---------------------------------------------------------------------------
# __init__: _copy_bundle
# ---------------------------------------------------------------------------


def _make_recorder_mock(session_id: str, path: Path, *, explicit_path: bool = False) -> MagicMock:
    """Build a mock TrajectoryRecorder with session_id and path attributes."""
    recorder = MagicMock()
    recorder.session_id = session_id
    recorder.path = path
    recorder.explicit_path = explicit_path
    return recorder


def _setup_bundle(
    tmp_path: Path, session_id: str = "abcd1234-0000-0000-0000-000000000000"
) -> tuple[Path, Path, MagicMock]:
    """Create a realistic target directory with artifacts and an empty run dir.

    Layout mirrors live-recorder output: ``.daydream/runs/<session_id>/``
    holds ``trajectory.json`` + a ``trajectories/`` subdir for forks. The
    archive copier copies that subtree wholesale.
    """
    target = tmp_path / "target"
    daydream = target / ".daydream"
    daydream.mkdir(parents=True)
    live_run_dir = daydream / "runs" / session_id
    live_run_dir.mkdir(parents=True)

    # Main trajectory under the run dir.
    traj = live_run_dir / "trajectory.json"
    traj.write_text('{"session_id": "test"}')

    # Sub-trajectories live next to the parent.
    sub_dir = live_run_dir / "trajectories"
    sub_dir.mkdir()
    (sub_dir / "deep-python.json").write_text('{"fork": true}')

    # Deep artifacts
    deep = daydream / "deep"
    deep.mkdir()
    (deep / "intent.md").write_text("intent")

    # Diff patch
    (daydream / "diff.patch").write_text("diff content")

    # Review output (lives in target root, not .daydream/)
    (target / ".review-output.md").write_text("review findings")

    run_dir = tmp_path / "run"
    run_dir.mkdir()

    # Recorder.path points at the live trajectory inside the run dir.
    recorder = _make_recorder_mock(session_id, traj)

    return target, run_dir, recorder


def test_copy_bundle_trajectory(tmp_path: Path):
    target, run_dir, recorder = _setup_bundle(tmp_path)
    _copy_bundle(target, run_dir, recorder)

    assert (run_dir / "trajectory.json").exists()
    assert json.loads((run_dir / "trajectory.json").read_text())["session_id"] == "test"


def test_copy_bundle_partial_trajectory(tmp_path: Path):
    """Partial trajectory file inside the live run dir is copied too."""
    session_id = "abcd1234-0000-0000-0000-000000000000"
    target, run_dir, recorder = _setup_bundle(tmp_path, session_id)

    # Drop a .partial sibling next to the live trajectory.json.
    partial = (
        target / ".daydream" / "runs" / session_id / "trajectory.json.partial"
    )
    partial.write_text('{"partial": true}')

    _copy_bundle(target, run_dir, recorder)

    assert json.loads((run_dir / "trajectory.json.partial").read_text())["partial"] is True


def test_copy_bundle_review_output(tmp_path: Path):
    target, run_dir, recorder = _setup_bundle(tmp_path)
    _copy_bundle(target, run_dir, recorder)

    assert (run_dir / "review-output.md").read_text() == "review findings"


def test_copy_bundle_deep_directory(tmp_path: Path):
    target, run_dir, recorder = _setup_bundle(tmp_path)
    _copy_bundle(target, run_dir, recorder)

    assert (run_dir / "deep" / "intent.md").read_text() == "intent"


def test_copy_bundle_diff_patch(tmp_path: Path):
    target, run_dir, recorder = _setup_bundle(tmp_path)
    _copy_bundle(target, run_dir, recorder)

    assert (run_dir / "diff.patch").read_text() == "diff content"


def test_copy_bundle_sub_trajectories_copied(tmp_path: Path):
    """Sibling trajectories under the live run dir copy verbatim — no prefix filtering."""
    target, run_dir, recorder = _setup_bundle(tmp_path)
    _copy_bundle(target, run_dir, recorder)

    sub = run_dir / "trajectories"
    assert sub.is_dir()
    copied = sorted(p.name for p in sub.iterdir())
    assert copied == ["deep-python.json"]


def test_copy_bundle_explicit_trajectory_path(tmp_path: Path):
    """When --trajectory points outside the live run dir, the file is still archived."""
    session_id = "abcd1234-0000-0000-0000-000000000000"
    target, run_dir, _ = _setup_bundle(tmp_path, session_id)

    # Simulate --trajectory /tmp/custom.json: file lives outside .daydream/runs/
    custom_traj = tmp_path / "custom-trajectory.json"
    custom_traj.write_text('{"custom": true}')

    recorder = _make_recorder_mock(session_id, custom_traj, explicit_path=True)
    _copy_bundle(target, run_dir, recorder)

    # The copytree still copies the live run dir contents (the default trajectory).
    # The custom path is copied on top as trajectory.json.
    archived = json.loads((run_dir / "trajectory.json").read_text())
    assert archived["custom"] is True


def test_copy_bundle_skips_missing(tmp_path: Path):
    target = tmp_path / "empty_target"
    target.mkdir()
    (target / ".daydream").mkdir()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    recorder = _make_recorder_mock("no-match-session-id-here", tmp_path / "nonexistent.json")
    _copy_bundle(target, run_dir, recorder)

    assert not (run_dir / "trajectory.json").exists()
    assert not (run_dir / "review-output.md").exists()
    assert not (run_dir / "deep").exists()
    assert not (run_dir / "diff.patch").exists()


# ---------------------------------------------------------------------------
# __init__: archive_run full round-trip
# ---------------------------------------------------------------------------


def test_archive_run_round_trip(tmp_path: Path, archive_dir: Path):
    archive_root = archive_dir

    session_id = "abcd1234-0000-0000-0000-000000000000"
    config = _MockConfig()

    target, _, _ = _setup_bundle(tmp_path, session_id)
    recorder = _MockRecorder(session_id=session_id)

    archive_run(recorder=recorder, target_dir=target, config=config, status="complete")

    run_dir = archive_root / "runs" / session_id
    assert run_dir.is_dir()
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "trajectory.json").is_file()

    manifest_data = json.loads((run_dir / "manifest.json").read_text())
    assert manifest_data["session_id"] == session_id
    assert manifest_data["run"]["flow"] == "normal"
    assert manifest_data["run"]["skill"] == "python"

    rows = query_runs(archive_root)
    assert len(rows) == 1
    assert rows[0]["session_id"] == session_id


# ---------------------------------------------------------------------------
# index: label_observations (Task 12)
# ---------------------------------------------------------------------------


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
    import time

    _seed_one_run(tmp_path, "sess-3")
    append_label_observation(
        tmp_path,
        "sess-3",
        labels=["unknown"],
        pr_state="open",
        labeler_version="v1",
        evidence_sha=None,
    )
    time.sleep(0.01)
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
    import time

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
    time.sleep(0.01)
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


def test_update_labels_is_backward_compat_thin_wrapper(tmp_path: Path) -> None:
    """The legacy update_labels() now writes through append_label_observation
    so existing callers continue to work without source changes."""
    _seed_one_run(tmp_path, "sess-5")
    assert update_labels(tmp_path, "sess-5", ["accepted"]) is True
    hist = label_observation_history(tmp_path, "sess-5")
    assert len(hist) == 1
    rows = query_runs(tmp_path, "session_id = ?", ("sess-5",))
    assert json.loads(rows[0]["outcome_labels"]) == ["accepted"]
