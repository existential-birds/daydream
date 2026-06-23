# tests/test_archive.py
"""Unit tests for the daydream.archive package.

Covers git_context, manifest, index, and the top-level archive_run flow.
"""

import json
import sqlite3
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from daydream.archive import _copy_bundle, archive_run, get_archive_dir
from daydream.archive.git_context import GitContext, _parse_repo_slug, capture_git_context
from daydream.archive.index import (
    append_label_observation,
    bulk_latest_label_observations,
    label_count_summary,
    label_observation_history,
    latest_label_observation,
    query_runs,
    reviewer_set_penalty_prior,
    set_run_pr_link,
    update_labels,
    upsert_run,
)
from daydream.archive.manifest import Manifest, build_manifest
from daydream.config_file import DaydreamFileConfig
from daydream.runner import RunConfig
from daydream.trajectory import TrajectoryRecorder


@dataclass
class _MockRecorder:
    session_id: str = "abcd1234-0000-0000-0000-000000000000"
    path: Path = field(default_factory=lambda: Path("/nonexistent/trajectory.json"))
    run_flow: MagicMock = field(default_factory=lambda: MagicMock(value="normal"))
    explicit_path: bool = False
    pr_number: int | None = None
    pr_repo: str | None = None
    _wall_clock_seconds: float | None = None
    _final_totals: dict = field(
        default_factory=lambda: {
            "prompt": 100,
            "completion": 50,
            "cached": 20,
            "cost": 0.05,
            "any_cost_seen": True,
        },
    )

    def compute_wall_clock_seconds(self) -> float | None:
        return self._wall_clock_seconds

    def compute_phase_timings(self) -> dict[str, Any] | None:
        return None


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
    file_config: DaydreamFileConfig | None = None


def test_parse_repo_slug_ssh():
    assert _parse_repo_slug("git@github.com:org/repo.git") == "org/repo"


def test_parse_repo_slug_https():
    assert _parse_repo_slug("https://github.com/org/repo.git") == "org/repo"


def test_parse_repo_slug_https_no_dot_git():
    assert _parse_repo_slug("https://github.com/org/repo") == "org/repo"


def test_parse_repo_slug_invalid():
    assert _parse_repo_slug("not-a-url") is None


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
        recorder=cast(TrajectoryRecorder, recorder),
        config=cast(RunConfig, config),
        git_ctx=git_ctx,
        status="complete",
        archive_path=tmp_path,
    )

    assert m.session_id == recorder.session_id
    assert m.run_flow == "normal"
    assert m.skill == "python"
    # Per-phase models replaced config.model; build_manifest stamps model as None.
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
        recorder=cast(TrajectoryRecorder, recorder),
        config=cast(RunConfig, config),
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
        recorder=cast(TrajectoryRecorder, recorder),
        config=cast(RunConfig, config),
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
        recorder=cast(TrajectoryRecorder, recorder),
        config=cast(RunConfig, config),
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
        recorder=cast(TrajectoryRecorder, recorder),
        config=cast(RunConfig, config),
        git_ctx=git_ctx,
        status="complete",
        archive_path=tmp_path,
    )

    assert m.total_findings is None
    assert m.grounding_rate is None
    assert m.coverage_ratio is None
    assert m.cost_per_finding_usd is None


def test_build_manifest_wall_clock_without_evaluation(tmp_path: Path):
    """Wall-clock is derived from step timestamps even when --eval did not run."""
    recorder = _MockRecorder(_wall_clock_seconds=12.3)
    config = _MockConfig()
    git_ctx = GitContext()

    m = build_manifest(
        recorder=cast(TrajectoryRecorder, recorder),
        config=cast(RunConfig, config),
        git_ctx=git_ctx,
        status="complete",
        archive_path=tmp_path,
    )

    assert m.wall_clock_seconds == 12.3
    assert m.total_findings is None


def test_build_manifest_eval_wall_clock_overrides_recorder(tmp_path: Path):
    """When --eval runs, its fork-inclusive timing takes precedence over the recorder span."""
    recorder = _MockRecorder(_wall_clock_seconds=12.3)
    config = _MockConfig()
    git_ctx = GitContext()
    evaluation = {"timing": {"total_wall_clock_seconds": 42.5}}

    m = build_manifest(
        recorder=cast(TrajectoryRecorder, recorder),
        config=cast(RunConfig, config),
        git_ctx=git_ctx,
        status="complete",
        archive_path=tmp_path,
        evaluation=evaluation,
    )

    assert m.wall_clock_seconds == 42.5


def _make_manifest(session_id: str = "sess-0001", **overrides: Any) -> Manifest:
    defaults: dict[str, Any] = {
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


def test_set_run_pr_link_backfills_pr_columns(tmp_path: Path):
    upsert_run(tmp_path, _make_manifest(session_id="s-orphan", pr_number=None, pr_repo=None))
    set_run_pr_link(tmp_path, "s-orphan", 7, "org/repo")
    row = query_runs(tmp_path, where="session_id = ?", params=("s-orphan",))[0]
    assert (row["pr_number"], row["pr_repo"]) == (7, "org/repo")


def test_query_runs_with_where(tmp_path: Path):
    upsert_run(tmp_path, _make_manifest(session_id="s1", repo_slug="org/a"))
    upsert_run(tmp_path, _make_manifest(session_id="s2", repo_slug="org/b", archive_path="/tmp/s2"))
    upsert_run(tmp_path, _make_manifest(session_id="s3", repo_slug="org/a", archive_path="/tmp/s3"))

    rows = query_runs(tmp_path, where="repo_slug = ?", params=("org/a",))
    assert len(rows) == 2
    ids = {r["session_id"] for r in rows}
    assert ids == {"s1", "s3"}


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
    assert get_archive_dir() == archive_dir
    assert archive_dir == tmp_path / "archive"


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

    traj = live_run_dir / "trajectory.json"
    traj.write_text('{"session_id": "test"}')

    # Sub-trajectories (fork output) live next to the parent.
    sub_dir = live_run_dir / "trajectories"
    sub_dir.mkdir()
    (sub_dir / "deep-python.json").write_text('{"fork": true}')

    deep = daydream / "deep"
    deep.mkdir()
    (deep / "intent.md").write_text("intent")

    (daydream / "diff.patch").write_text("diff content")

    # Review output lives in target root, not .daydream/.
    (target / ".review-output.md").write_text("review findings")

    run_dir = tmp_path / "run"
    run_dir.mkdir()

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

    # Simulate --trajectory /tmp/custom.json: file lives outside .daydream/runs/.
    custom_traj = tmp_path / "custom-trajectory.json"
    custom_traj.write_text('{"custom": true}')

    recorder = _make_recorder_mock(session_id, custom_traj, explicit_path=True)
    _copy_bundle(target, run_dir, recorder)

    # The custom path is copied on top of the run dir as trajectory.json.
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


def test_archive_run_round_trip(tmp_path: Path, archive_dir: Path):
    session_id = "abcd1234-0000-0000-0000-000000000000"
    config = _MockConfig()

    target, _, _ = _setup_bundle(tmp_path, session_id)
    recorder = _MockRecorder(session_id=session_id)

    archive_run(
        recorder=cast(TrajectoryRecorder, recorder),
        target_dir=target,
        config=cast(RunConfig, config),
        status="complete",
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


def test_label_observations_has_bitemporal_reward_columns(tmp_path: Path):
    upsert_run(tmp_path, _make_manifest())  # forces _get_connection to build schema
    conn = sqlite3.connect(str(tmp_path / "index.db"))
    lo_cols = {r[1] for r in conn.execute("PRAGMA table_info(label_observations)")}
    runs_cols = {r[1] for r in conn.execute("PRAGMA table_info(runs)")}
    conn.close()
    assert {"valid_at", "reward_version", "reward_json"} <= lo_cols
    assert "composite_reward" in runs_cols


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
            (session_id, "2026-01-01T00:00:00+00:00", '["accepted"]', "merged", "v1", "sha1"),
        )
        conn.commit()
    finally:
        conn.close()


def test_label_observations_source_column_migrates(tmp_path: Path):
    # Build schema, then replace the table with the OLD DDL (no `source`) + a legacy row.
    upsert_run(tmp_path, _make_manifest(session_id="s-mig"))
    _seed_legacy_label_observation(tmp_path, "s-mig")
    assert "source" not in _label_obs_columns(tmp_path)  # precondition: legacy shape

    # The production connection path must ALTER-ADD `source`.
    upsert_run(tmp_path, _make_manifest(session_id="s-mig2"))

    cols = _label_obs_columns(tmp_path)
    assert "source" in cols
    rows = label_observation_history(tmp_path, "s-mig")
    assert rows and rows[0]["source"] == "auto"  # existing row defaulted, non-destructive


def test_human_label_wins_over_newer_auto_in_projection(tmp_path: Path):
    upsert_run(tmp_path, _make_manifest(session_id="s-prec"))
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


def test_append_cache_reflects_winning_human_label(tmp_path: Path):
    upsert_run(tmp_path, _make_manifest(session_id="s-cache"))
    append_label_observation(tmp_path, "s-cache", labels=["rejected"], pr_state="closed",
                             labeler_version="auto-v1", evidence_sha="sha1", source="auto")
    append_label_observation(tmp_path, "s-cache", labels=["accepted"], pr_state=None,
                             labeler_version="human", evidence_sha=None, source="human")
    # A later auto append must leave the denormalized runs cache on the human label:
    append_label_observation(tmp_path, "s-cache", labels=["rejected"], pr_state="closed",
                             labeler_version="auto-v2", evidence_sha="sha2", source="auto")
    row = query_runs(tmp_path, "session_id = ?", ("s-cache",))[0]
    assert row["outcome_labels"] == '["accepted"]'


def test_auto_append_dedups_on_unchanged_evidence(tmp_path: Path):
    upsert_run(tmp_path, _make_manifest(session_id="s-dedup"))
    first = append_label_observation(tmp_path, "s-dedup", labels=["accepted"], pr_state="merged",
                                     labeler_version="rv1", evidence_sha="shaA", source="auto")
    second = append_label_observation(tmp_path, "s-dedup", labels=["accepted"], pr_state="merged",
                                      labeler_version="rv1", evidence_sha="shaA", source="auto")
    assert first is True and second is False
    assert len(label_observation_history(tmp_path, "s-dedup")) == 1
    # A reward_version change DOES append:
    third = append_label_observation(tmp_path, "s-dedup", labels=["accepted"], pr_state="merged",
                                     labeler_version="rv2", evidence_sha="shaA",
                                     reward_version="rv2", source="auto")
    assert third is True
    assert len(label_observation_history(tmp_path, "s-dedup")) == 2


def test_human_append_never_dedups(tmp_path: Path):
    upsert_run(tmp_path, _make_manifest(session_id="s-h"))
    append_label_observation(tmp_path, "s-h", labels=["accepted"], pr_state=None,
                             labeler_version="human", evidence_sha=None, source="human")
    append_label_observation(tmp_path, "s-h", labels=["accepted"], pr_state=None,
                             labeler_version="human", evidence_sha=None, source="human")
    assert len(label_observation_history(tmp_path, "s-h")) == 2


def test_append_observation_persists_valid_at_and_reward(tmp_path: Path):
    upsert_run(tmp_path, _make_manifest(session_id="s1"))
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


def test_append_observation_defaults_valid_at_to_observed_at(tmp_path: Path):
    upsert_run(tmp_path, _make_manifest(session_id="s2"))
    append_label_observation(tmp_path, "s2", labels=[], pr_state=None,
                             labeler_version="v1", evidence_sha=None, valid_at=None)
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


def test_same_microsecond_collision_keeps_clean_iso_timestamps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two appends frozen to the same microsecond must both persist with parseable
    ISO 8601 observed_at values, and an exact-boundary as_of must include the
    boundary row (the contract the ~uuid suffix used to break)."""
    from datetime import datetime, timezone

    frozen = datetime(2026, 5, 29, 12, 0, 0, tzinfo=timezone.utc)

    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
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


def test_append_label_observation_persists_reviewer_and_posterior_flag(tmp_path: Path) -> None:
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
    columns, and PRAGMA user_version reaches SCHEMA_VERSION (4)."""
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
    assert user_version == SCHEMA_VERSION == 5

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


def test_reviewer_set_penalty_prior_pools_shared_reviewer_runs_strict_cutoff(tmp_path):
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


def test_reviewer_set_penalty_prior_scoped_to_repo(tmp_path):
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


def test_manifest_includes_source_path():
    """source_path appears in manifest dict under git section."""
    m = Manifest(
        session_id="test-session",
        source_path="/home/user/code/myrepo",
        remote_url="git@github.com:org/repo.git",
        repo_slug="org/repo",
    )
    d = m.to_dict()
    assert d["git"]["source_path"] == "/home/user/code/myrepo"


def test_source_path_indexed_in_sqlite(tmp_path: Path):
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


def test_source_path_defaults_to_none():
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
