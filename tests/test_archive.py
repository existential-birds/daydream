# tests/test_archive.py
"""Unit tests for the daydream.archive package.

Covers git_context, manifest, index, and the top-level archive_run flow.
"""

import json
import sqlite3
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

from daydream.archive import _copy_bundle, _read_fix_quality_gate, archive_run, get_archive_dir
from daydream.archive.git_context import GitContext, _parse_repo_slug, capture_git_context
from daydream.archive.index import (
    append_label_observation,
    bulk_latest_label_observations,
    canonical_utc_iso,
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
from daydream.archive.manifest import Manifest, build_manifest
from daydream.config_file import DaydreamFileConfig
from daydream.runner import RunConfig
from daydream.trajectory import DaydreamRunFlow, TrajectoryRecorder
from tests.harness.trajectory import make_manifest

MakeConfig = Callable[..., RunConfig]
InstallBackend = Callable[[object], object]


@dataclass
class _MockRecorder:
    session_id: str = "abcd1234-0000-0000-0000-000000000000"
    path: Path = field(default_factory=lambda: Path("/nonexistent/trajectory.json"))
    run_flow: DaydreamRunFlow = DaydreamRunFlow.NORMAL
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
    backend: str | None = None
    model: str | None = None
    review_backend: str | None = None
    fix_backend: str | None = None
    test_backend: str | None = None
    output_mode: str = "loop"
    shallow: bool = False
    bot: str | None = None
    flow_name: str | None = None
    loop: bool = False
    archive: bool = True
    run_eval: bool = False
    file_config: DaydreamFileConfig | None = None
    findings_out: str | None = None


@pytest.mark.parametrize(
    ("remote_url", "expected"),
    [
        pytest.param("git@github.com:org/repo.git", "org/repo", id="ssh"),
        pytest.param("https://github.com/org/repo.git", "org/repo", id="https"),
        pytest.param("https://github.com/org/repo", "org/repo", id="https_no_dot_git"),
        pytest.param("not-a-url", None, id="invalid"),
    ],
)
def test_parse_repo_slug(remote_url: str, expected: str | None):
    assert _parse_repo_slug(remote_url) == expected


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


def _build(
    tmp_path: Path,
    *,
    git_ctx: GitContext | None = None,
    recorder: _MockRecorder | None = None,
    config: Any | None = None,
    **kw: Any,
) -> Manifest:
    """Build a manifest from the mock recorder/config pair."""
    return build_manifest(
        recorder=cast(TrajectoryRecorder, recorder or _MockRecorder()),
        config=cast(RunConfig, _MockConfig() if config is None else config),
        git_ctx=git_ctx if git_ctx is not None else GitContext(),
        status="complete",
        archive_path=tmp_path,
        **kw,
    )


def test_build_manifest_basic(tmp_path: Path):
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

    assert m.session_id == _MockRecorder().session_id
    assert m.run_flow == "normal"
    assert m.skill == "python"
    # Per-phase models replaced config.model; build_manifest stamps model as None.
    assert m.model is None
    assert m.backend == "claude"
    assert m.review_backend is None
    assert m.total_cost_usd == 0.05
    assert m.total_prompt_tokens == 100
    assert m.total_completion_tokens == 50
    assert m.total_cached_tokens == 20
    assert m.repo_slug == "org/repo"
    assert m.head_sha == "a" * 40


@pytest.mark.parametrize(
    "flow",
    list(DaydreamRunFlow),
    ids=[f.value for f in DaydreamRunFlow],
)
def test_build_manifest_fix_test_backend_gated_per_flow(
    tmp_path: Path, flow: DaydreamRunFlow,
) -> None:
    """Issue #648: backend labels track the executed step pipeline.

    Driven over every ``DaydreamRunFlow`` member: the deep family's
    fix-bearing mode labels (NORMAL for shallow, DEEP for loop) resolve
    fix/test backends exactly as today, while TTT (review/comment) gates the
    fix/test STEPS off at runtime (``_fix_cycle_enabled`` is loop/shallow
    only) and drops both labels. PR (feedback) runs its own ``fix-items``
    phase (``_step_fix_items``, ``backend_for("fix")``) but never the ``test``
    step, so it keeps a fix backend and drops only test. IMPROVE's built-in
    pipeline defines no fix/test phase, so it drops both. CUSTOM is classified
    by its registered pipeline; with no fork flow configured here it cannot
    prove a fix/test phase, so it drops both.
    """
    recorder = _MockRecorder(run_flow=flow)
    config = _MockConfig(fix_backend="codex", test_backend="codex")
    m = _build(tmp_path, recorder=recorder, config=config)
    run = m.to_dict()["run"]
    if flow is DaydreamRunFlow.PR:
        # Feedback runs fix-items but never the test step.
        assert m.fix_backend == "codex"
        assert m.test_backend is None
        assert run["fix_backend"] == "codex"
        assert "test_backend" not in run
    elif flow in (
        DaydreamRunFlow.IMPROVE,
        DaydreamRunFlow.TTT,
        DaydreamRunFlow.CUSTOM,
    ):
        assert m.fix_backend is None
        assert m.test_backend is None
        assert "fix_backend" not in run
        assert "test_backend" not in run
    else:
        assert m.fix_backend == "codex"
        assert m.test_backend == "codex"
        assert run["fix_backend"] == "codex"
        assert run["test_backend"] == "codex"


def test_build_manifest_classifies_custom_flow_by_registered_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every ``--flow`` run is stamped CUSTOM regardless of step composition, so
    a fork flow is classified from its registered pipeline, not its label: one
    composing the built-in fix/test steps records backends, one without them
    records none.
    """
    from daydream.extensions import build_registry

    registry = build_registry()
    registry.set_flow("fix-cycle-fork", ["exploration", "intent", "fix", "test", "commit"])
    registry.set_flow("review-only-fork", ["exploration", "intent"])
    monkeypatch.setattr("daydream.archive.manifest.get_registry", lambda: registry)

    for flow_name, fix_backend, test_backend in (
        ("fix-cycle-fork", "codex", "codex"),
        ("review-only-fork", None, None),
    ):
        recorder = _MockRecorder(run_flow=DaydreamRunFlow.CUSTOM)
        config = _MockConfig(
            fix_backend="codex", test_backend="codex", flow_name=flow_name,
        )
        m = _build(tmp_path, recorder=recorder, config=config)
        assert m.fix_backend == fix_backend
        assert m.test_backend == test_backend
        run = m.to_dict()["run"]
        assert ("fix_backend" in run) == (fix_backend is not None)
        assert ("test_backend" in run) == (test_backend is not None)


def test_build_manifest_classifies_fork_override_of_builtin_flow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Issue #648: fork overrides of built-in flow names are classified by
    the registry pipeline actually executed by ``run_flow``, not by the label.

    ``Registry.set_flow`` has no built-in-name guard, so a fork overriding
    ``deep`` (dropping fix/test) or ``improve`` (adding fix/test) runs its own
    pipeline while the recorder label still suggests the built-in. The manifest
    must follow the registry in both directions.
    """
    from daydream.extensions import build_registry

    registry = build_registry()
    # Override built-ins after seeding, exactly as an extension would.
    registry.set_flow("deep", ["exploration", "intent"])
    registry.set_flow("improve", ["exploration", "fix", "test"])
    monkeypatch.setattr("daydream.archive.manifest.get_registry", lambda: registry)

    for flow, flow_name, fix_backend, test_backend in (
        (DaydreamRunFlow.DEEP, None, None, None),
        (DaydreamRunFlow.NORMAL, None, None, None),
        (DaydreamRunFlow.IMPROVE, None, "codex", "codex"),
    ):
        recorder = _MockRecorder(run_flow=flow)
        config = _MockConfig(
            fix_backend="codex", test_backend="codex", flow_name=flow_name,
        )
        m = _build(tmp_path, recorder=recorder, config=config)
        assert m.fix_backend == fix_backend
        assert m.test_backend == test_backend


def test_fix_cycle_classification_covers_every_run_flow() -> None:
    """Every ``DaydreamRunFlow`` member is explicitly classified: TTT
    (review/comment) is mode-gated never to reach the fix cycle, PR (feedback)
    runs its own fix-items phase (fix yes, test no), and every other label is
    classified by its registered pipeline (issue #648). A future enum member
    fails this exhaustiveness check instead of silently changing which backend
    fields the manifest emits.
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
        | {DaydreamRunFlow.IMPROVE, DaydreamRunFlow.CUSTOM}
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


@pytest.mark.parametrize(
    ("flow_name", "shallow"),
    [
        pytest.param(None, False, id="default-deep-loop"),
        pytest.param(None, True, id="shallow-single-stack"),
        pytest.param("deep", False, id="flow-deep"),
        pytest.param("shallow", False, id="flow-shallow"),
        pytest.param("review", False, id="flow-review"),
    ],
)
def test_build_manifest_per_stack_review_tier(
    tmp_path: Path, flow_name: str | None, shallow: bool
) -> None:
    """Issue #646: every deep-flow mode that executes per-stack reviews records the
    per-stack tier resolved from its own key — including shallow, whose collapsed
    single stack is still reviewed by ``phase_per_stack_reviews`` on the
    ``per_stack_review`` tier."""
    fc = DaydreamFileConfig(
        model=None, backend=None,
        phases={"per_stack_review": {"backend": "codex", "model": "gpt-psr"}},
    )
    config = RunConfig(
        target=str(tmp_path), backend=None, model=None,
        file_config=fc, shallow=shallow, flow_name=flow_name,
        review_backend="claude",
    )
    m = build_manifest(
        recorder=cast(TrajectoryRecorder, _MockRecorder()),
        config=config, git_ctx=GitContext(),
        status="complete", archive_path=tmp_path,
    )
    run = m.to_dict()["run"]
    assert m.per_stack_review_backend == "codex"  # per-stack tier resolved from its own key
    assert m.per_stack_review_model == "gpt-psr"
    # Post-#647, review_backend is an override marker (_resolved_review_backend_name),
    # so it stays the explicit review override "claude" — distinct from the
    # per-stack tier resolved from its own key. The per-stack model is the
    # load-bearing part of the identity.
    assert m.review_backend == "claude"
    assert run["per_stack_review_backend"] == "codex"
    assert run["per_stack_review_model"] == "gpt-psr"
    assert run["review_backend"] == "claude"


def test_build_manifest_per_stack_review_gate_tracks_runner_aliases(tmp_path: Path) -> None:
    """Issue #646 finding 3: the per-stack gate derives from the same alias list
    ``runner._dispatch_selected_flow`` routes (no third inline copy), so adding
    or renaming a deep-flow alias surfaces here instead of silently misstating
    "who reviewed" in archived runs."""
    from daydream.runner import _DEEP_FLOW_ALIASES

    assert _DEEP_FLOW_ALIASES  # non-empty; every alias must record the identity
    for flow_name in _DEEP_FLOW_ALIASES:
        m = build_manifest(
            recorder=cast(TrajectoryRecorder, _MockRecorder()),
            config=RunConfig(target=str(tmp_path), backend=None, model=None, flow_name=flow_name),
            git_ctx=GitContext(),
            status="complete", archive_path=tmp_path,
        )
        assert m.per_stack_review_backend is not None
        assert m.per_stack_review_model is not None
        assert "per_stack_review_backend" in m.to_dict()["run"]


def test_build_manifest_omits_per_stack_review_without_spine(tmp_path: Path) -> None:
    """Issue #646: runs that never execute per-stack reviews record no identity —
    feedback mode (the review spine is skipped entirely) and improve/custom flows
    (which never invoke the deep orchestrator's per-stack-reviews step)."""
    for config in (
        RunConfig(target=str(tmp_path), backend=None, model=None,
                  bot="myapp[bot]", pr_number=42),        # feedback mode
        RunConfig(target=str(tmp_path), backend=None, model=None,
                  flow_name="improve"),                   # improve flow
        RunConfig(target=str(tmp_path), backend=None, model=None,
                  flow_name="custom-flow"),               # custom flow
    ):
        m = build_manifest(
            recorder=cast(TrajectoryRecorder, _MockRecorder()),
            config=config, git_ctx=GitContext(),
            status="complete", archive_path=tmp_path,
        )
        run = m.to_dict()["run"]
        assert m.per_stack_review_backend is None
        assert m.per_stack_review_model is None
        assert "per_stack_review_backend" not in run
        assert "per_stack_review_model" not in run


def test_build_manifest_pi_deep_records_backend_default_model(tmp_path: Path) -> None:
    """Issue #646: a Pi deep run with no explicit model override records Pi's
    backend default. Pi's default intentionally lives outside PHASE_DEFAULT_MODELS
    (it is a backend fallback), so _resolved_model alone leaves the load-bearing
    model NULL; the manifest surfaces the backend default instead."""
    from daydream.config import DEFAULT_PI_MODEL

    m = build_manifest(
        recorder=cast(TrajectoryRecorder, _MockRecorder()),
        config=RunConfig(target=str(tmp_path), backend="pi", model=None),
        git_ctx=GitContext(),
        status="complete", archive_path=tmp_path,
    )
    assert m.per_stack_review_backend == "pi"
    assert m.per_stack_review_model == DEFAULT_PI_MODEL
    assert m.to_dict()["run"]["per_stack_review_model"] == DEFAULT_PI_MODEL


def test_manifest_to_dict_structure(tmp_path: Path):
    m = _build(tmp_path)

    d = m.to_dict()
    assert d["schema_version"] == "1.0"
    assert d["session_id"] == _MockRecorder().session_id
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


def test_build_manifest_with_evaluation(tmp_path: Path):
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


def test_build_manifest_with_quality(tmp_path: Path):
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


def test_build_manifest_without_evaluation(tmp_path: Path):
    m = _build(tmp_path)

    assert m.total_findings is None
    assert m.grounding_rate is None
    assert m.coverage_ratio is None
    assert m.cost_per_finding_usd is None
    assert m.erosion is None
    assert m.verbosity is None


def test_build_manifest_wall_clock_without_evaluation(tmp_path: Path):
    """Wall-clock is derived from step timestamps even when --eval did not run."""
    m = _build(tmp_path, recorder=_MockRecorder(_wall_clock_seconds=12.3))

    assert m.wall_clock_seconds == 12.3
    assert m.total_findings is None


def test_build_manifest_eval_wall_clock_overrides_recorder(tmp_path: Path):
    """When --eval runs, its fork-inclusive timing takes precedence over the recorder span."""
    m = _build(
        tmp_path,
        recorder=_MockRecorder(_wall_clock_seconds=12.3),
        evaluation={"timing": {"total_wall_clock_seconds": 42.5}},
    )

    assert m.wall_clock_seconds == 42.5


def test_upsert_run_creates_db(tmp_path: Path):
    m = make_manifest()
    upsert_run(tmp_path, m)
    assert (tmp_path / "index.db").exists()


def test_upsert_and_query_round_trip(tmp_path: Path):
    m = make_manifest()
    upsert_run(tmp_path, m)

    rows = query_runs(tmp_path)
    assert len(rows) == 1
    assert rows[0]["session_id"] == "sess-0001"
    assert rows[0]["skill"] == "python"
    assert rows[0]["status"] == "complete"


def test_update_labels_exact(tmp_path: Path):
    upsert_run(tmp_path, make_manifest())

    ok = update_labels(tmp_path, "sess-0001", ["good", "fast"])
    assert ok is True

    rows = query_runs(tmp_path)
    assert json.loads(rows[0]["outcome_labels"]) == ["good", "fast"]
    assert rows[0]["labeled_at"] is not None


def test_update_labels_prefix(tmp_path: Path):
    upsert_run(tmp_path, make_manifest(session_id="abcd1234-full-uuid"))

    ok = update_labels(tmp_path, "abcd1234", ["label-a"])
    assert ok is True

    rows = query_runs(tmp_path)
    assert json.loads(rows[0]["outcome_labels"]) == ["label-a"]


def test_update_labels_nonexistent(tmp_path: Path):
    upsert_run(tmp_path, make_manifest())

    ok = update_labels(tmp_path, "no-such-session", [])
    assert ok is False


def test_update_labels_ambiguous_prefix(tmp_path: Path):
    upsert_run(tmp_path, make_manifest(session_id="abc-001"))
    upsert_run(tmp_path, make_manifest(session_id="abc-002", archive_path="/tmp/x"))

    with pytest.raises(ValueError, match="matches 2 sessions"):
        update_labels(tmp_path, "abc", ["x"])


def test_set_run_pr_link_backfills_pr_columns(tmp_path: Path):
    upsert_run(tmp_path, make_manifest(session_id="s-orphan", pr_number=None, pr_repo=None))
    set_run_pr_link(tmp_path, "s-orphan", 7, "org/repo")
    row = query_runs(tmp_path, where="session_id = ?", params=("s-orphan",))[0]
    assert (row["pr_number"], row["pr_repo"]) == (7, "org/repo")


def test_query_runs_with_where(tmp_path: Path):
    upsert_run(tmp_path, make_manifest(session_id="s1", repo_slug="org/a"))
    upsert_run(tmp_path, make_manifest(session_id="s2", repo_slug="org/b", archive_path="/tmp/s2"))
    upsert_run(tmp_path, make_manifest(session_id="s3", repo_slug="org/a", archive_path="/tmp/s3"))

    rows = query_runs(tmp_path, where="repo_slug = ?", params=("org/a",))
    assert len(rows) == 2
    ids = {r["session_id"] for r in rows}
    assert ids == {"s1", "s3"}


def test_upsert_run_persists_erosion_verbosity(tmp_path: Path):
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


def test_runs_erosion_verbosity_columns_migrate_existing_db(tmp_path: Path):
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
        make_manifest(session_id="s-psr-mig", per_stack_review_backend="codex",
                      per_stack_review_model="gpt-psr"),
    )
    legacy = query_runs(tmp_path, where="session_id = ?", params=("legacy-psr",))[0]
    assert legacy["per_stack_review_backend"] is None   # pre-existing row preserved, nullable
    row = query_runs(tmp_path, where="session_id = ?", params=("s-psr-mig",))[0]
    assert row["per_stack_review_backend"] == "codex"
    assert row["per_stack_review_model"] == "gpt-psr"


def test_build_manifest_carries_fix_quality_gate(tmp_path: Path):
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


def test_manifest_fix_quality_gate_none_when_absent(tmp_path: Path):
    """No gate artifact => the manifest field stays null (additive, never invented)."""
    m = _build(tmp_path)
    assert m.fix_quality_gate is None
    assert m.to_dict()["fix_quality_gate"] is None


def test_upsert_run_persists_fix_quality_gate(tmp_path: Path):
    """Issue #315: fix_quality_gate JSON round-trips through upsert_run -> query_runs."""
    gate = {"enabled": True, "rounds": [{"round": 1, "per_file": {"api.py": {"flagged": True}}}]}
    upsert_run(tmp_path, make_manifest(session_id="s-gate", fix_quality_gate=gate))
    row = query_runs(tmp_path, where="session_id = ?", params=("s-gate",))[0]
    assert json.loads(row["fix_quality_gate"]) == gate


def test_runs_fix_quality_gate_column_migrates_existing_db(tmp_path: Path):
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
        ("legacy-gate-run", "2026-01-01T00:00:00Z", "normal", str(tmp_path / "legacy-gate-run")),
    )
    conn.commit()
    conn.close()

    gate = {"enabled": True, "rounds": []}
    upsert_run(tmp_path, make_manifest(session_id="s-mig-gate", fix_quality_gate=gate))

    legacy = query_runs(tmp_path, where="session_id = ?", params=("legacy-gate-run",))[0]
    assert legacy["fix_quality_gate"] is None  # pre-existing row preserved, column nullable
    row = query_runs(tmp_path, where="session_id = ?", params=("s-mig-gate",))[0]
    assert json.loads(row["fix_quality_gate"]) == gate


def test_read_fix_quality_gate_requires_matching_session(tmp_path: Path):
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


def test_read_fix_quality_gate_unbound_artifact_is_none(tmp_path: Path):
    """#329: an artifact with no session_id key cannot be attributed to this run."""
    gate_p = tmp_path / ".daydream" / "deep" / "fix-quality-gate.json"
    gate_p.parent.mkdir(parents=True)
    gate_p.write_text(json.dumps({"enabled": True, "rounds": [{"round": 1, "per_file": {}}]}))

    assert _read_fix_quality_gate(tmp_path, "sess-42") is None


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
    _copy_bundle(target, run_dir, recorder, RunConfig())

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

    _copy_bundle(target, run_dir, recorder, RunConfig())

    assert json.loads((run_dir / "trajectory.json.partial").read_text())["partial"] is True


@pytest.mark.parametrize(
    ("relative_path", "expected"),
    [
        pytest.param("review-output.md", "review findings", id="review-output"),
        pytest.param("diff.patch", "diff content", id="diff-patch"),
    ],
)
def test_copy_bundle_file_path(tmp_path: Path, relative_path: str, expected: str):
    """Verify bundle copying preserves parameterized file contents and paths."""
    target, run_dir, recorder = _setup_bundle(tmp_path)
    _copy_bundle(target, run_dir, recorder, RunConfig())

    assert (run_dir / relative_path).read_text() == expected


def test_copy_bundle_deep_directory(tmp_path: Path):
    target, run_dir, recorder = _setup_bundle(tmp_path)
    _copy_bundle(target, run_dir, recorder, RunConfig())

    assert (run_dir / "deep" / "intent.md").read_text() == "intent"


def test_copy_bundle_sub_trajectories_copied(tmp_path: Path):
    """Sibling trajectories under the live run dir copy verbatim — no prefix filtering."""
    target, run_dir, recorder = _setup_bundle(tmp_path)
    _copy_bundle(target, run_dir, recorder, RunConfig())

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
    _copy_bundle(target, run_dir, recorder, RunConfig())

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
    _copy_bundle(target, run_dir, recorder, RunConfig())

    assert not (run_dir / "trajectory.json").exists()
    assert not (run_dir / "review-output.md").exists()
    assert not (run_dir / "deep").exists()
    assert not (run_dir / "diff.patch").exists()


def test_copy_bundle_archives_findings_artifact(tmp_path: Path):
    """findings.json from --findings-out is archived so harvest's per-finding join has a source."""
    target, run_dir, recorder = _setup_bundle(tmp_path)
    # Review-bot workflow writes findings.json under the repo root (CWD at run time).
    findings_src = target / "findings" / "findings.json"
    findings_src.parent.mkdir(parents=True)
    findings_src.write_text('{"findings": [{"fingerprint": "abc"}]}')

    config = RunConfig(findings_out="findings/findings.json")
    _copy_bundle(target, run_dir, recorder, config)

    archived = run_dir / "findings.json"
    assert archived.exists()
    assert json.loads(archived.read_text())["findings"][0]["fingerprint"] == "abc"


def test_copy_bundle_findings_artifact_skipped_without_findings_out(tmp_path: Path):
    """No findings_out means no findings.json is archived (no source to copy)."""
    target, run_dir, recorder = _setup_bundle(tmp_path)
    _copy_bundle(target, run_dir, recorder, RunConfig())

    assert not (run_dir / "findings.json").exists()


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
    upsert_run(tmp_path, make_manifest())  # forces _get_connection to build schema
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
    upsert_run(tmp_path, make_manifest(session_id="s-mig"))
    _seed_legacy_label_observation(tmp_path, "s-mig")
    assert "source" not in _label_obs_columns(tmp_path)  # precondition: legacy shape

    # The production connection path must ALTER-ADD `source`.
    upsert_run(tmp_path, make_manifest(session_id="s-mig2"))

    cols = _label_obs_columns(tmp_path)
    assert "source" in cols
    rows = label_observation_history(tmp_path, "s-mig")
    assert rows and rows[0]["source"] == "auto"  # existing row defaulted, non-destructive


def test_human_label_wins_over_newer_auto_in_projection(tmp_path: Path):
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


def test_append_cache_reflects_winning_human_label(tmp_path: Path):
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


def test_auto_append_dedups_on_unchanged_evidence(tmp_path: Path):
    upsert_run(tmp_path, make_manifest(session_id="s-dedup"))
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


def test_auto_append_appends_when_only_has_posterior_changes(tmp_path: Path):
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


def test_human_append_never_dedups(tmp_path: Path):
    upsert_run(tmp_path, make_manifest(session_id="s-h"))
    append_label_observation(tmp_path, "s-h", labels=["accepted"], pr_state=None,
                             labeler_version="human", evidence_sha=None, source="human")
    append_label_observation(tmp_path, "s-h", labels=["accepted"], pr_state=None,
                             labeler_version="human", evidence_sha=None, source="human")
    assert len(label_observation_history(tmp_path, "s-h")) == 2


def test_append_observation_persists_valid_at_and_reward(tmp_path: Path):
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


def test_append_observation_defaults_valid_at_to_observed_at(tmp_path: Path):
    upsert_run(tmp_path, make_manifest(session_id="s2"))
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
    assert user_version == SCHEMA_VERSION == 6

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


# Canonical UTC timestamp contract: one spelling at write time, strict as_of
# validation at the entry boundary, and legacy "Z" rows purged at bootstrap.


def test_canonical_utc_iso_converts_and_rejects():
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


def test_normalize_as_of_is_strict_utc_only():
    assert normalize_as_of("2026-04-01T00:00:00Z") == "2026-04-01T00:00:00+00:00"
    assert normalize_as_of("2026-04-01T00:00:00+00:00") == "2026-04-01T00:00:00+00:00"
    with pytest.raises(ValueError, match="must be a UTC timestamp"):
        normalize_as_of("2026-04-01T05:00:00+05:00")
    with pytest.raises(ValueError, match="must be a UTC timestamp"):
        normalize_as_of("2026-04-01T00:00:00")
    with pytest.raises(ValueError, match="not a valid ISO-8601"):
        normalize_as_of("yesterday")


def test_append_label_observation_canonicalizes_valid_at_spelling(tmp_path: Path):
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


def test_append_label_observation_rejects_naive_valid_at(tmp_path: Path):
    _seed_one_run(tmp_path, "sess-naive")
    with pytest.raises(ValueError, match="naive"):
        append_label_observation(
            tmp_path, "sess-naive", labels=["accepted"], pr_state="merged",
            labeler_version="v1", evidence_sha=None,
            valid_at="2026-02-01T00:00:00",
        )


def test_reviewer_prior_bound_spelling_cannot_misorder(tmp_path: Path):
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
        tmp_path, ["alice"], before_valid_at="2026-03-01T00:00:00Z", exclude_session="cur"
    )
    assert (prior, n) == (None, 0)
    # And a bound safely after the row still pools it, regardless of spelling.
    prior2, n2 = reviewer_set_penalty_prior(
        tmp_path, ["alice"], before_valid_at="2026-03-01T00:00:01Z", exclude_session="cur"
    )
    assert prior2 == pytest.approx(1.0) and n2 == 1


def test_legacy_z_valid_at_rows_are_left_untouched(tmp_path: Path):
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


def test_manifest_backend_is_general_default_not_review_override(tmp_path: Path) -> None:
    """#647: backend records the general default even when review differs."""
    m = _build(tmp_path, config=_MockConfig(backend="claude", review_backend="codex"))
    assert m.backend == "claude"
    assert m.review_backend == "codex"


def test_manifest_review_backend_none_when_no_override(tmp_path: Path) -> None:
    """#647: only a general backend -> review_backend stays None."""
    m = _build(tmp_path, config=_MockConfig(backend="codex"))
    assert m.backend == "codex"
    assert m.review_backend is None


def test_manifest_backend_falls_back_to_file_config_global(tmp_path: Path) -> None:
    """#647: no CLI backend -> backend resolves from the file-config global."""
    m = _build(
        tmp_path,
        config=_MockConfig(backend=None, file_config=DaydreamFileConfig(backend="file-backend")),
    )
    assert m.backend == "file-backend"
    assert m.review_backend is None


def test_manifest_review_backend_from_file_config_phase(tmp_path: Path) -> None:
    """#647: a file-config review-phase override stamps review_backend only."""
    m = _build(
        tmp_path,
        config=_MockConfig(backend="claude", file_config=DaydreamFileConfig(phases={"review": {"backend": "codex"}})),
    )
    assert m.backend == "claude"
    assert m.review_backend == "codex"


def test_archive_run_records_general_backend_and_override(tmp_path: Path, archive_dir: Path) -> None:
    """#647: archive_run persists the general backend + nullable review override."""
    session_id = "abcd1234-0000-0000-0000-000000000000"
    config = _MockConfig(backend="claude", review_backend="codex")
    target, _, _ = _setup_bundle(tmp_path, session_id)
    recorder = _MockRecorder(session_id=session_id)
    archive_run(
        recorder=cast(TrajectoryRecorder, recorder),
        target_dir=target,
        config=cast(RunConfig, config),
        status="complete",
    )
    manifest_data = json.loads((archive_dir / "runs" / session_id / "manifest.json").read_text())
    assert manifest_data["run"]["backend"] == "claude"
    assert manifest_data["run"]["review_backend"] == "codex"
    rows = query_runs(archive_dir)
    assert len(rows) == 1
    assert rows[0]["backend"] == "claude"
    assert rows[0]["review_backend"] == "codex"


async def test_build_manifest_totals_include_fork_trajectories(tmp_path: Path):
    """Manifest totals are whole-run: the fork's tokens/cost are folded in."""
    from daydream.backends import MetricsEvent, ResultEvent, TextEvent
    from daydream.trajectory import DaydreamPhase, DaydreamRunFlow

    recorder = TrajectoryRecorder(
        path=tmp_path / ".daydream" / "runs" / "sess-fold" / "trajectory.json",
        run_flow=DaydreamRunFlow.NORMAL,
        target_dir=tmp_path,
        agent_model_name="opus",
        session_id="sess-fold",
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

    m = build_manifest(
        recorder=recorder,
        config=cast(RunConfig, _MockConfig()),
        git_ctx=GitContext(),
        status="complete",
        archive_path=tmp_path,
    )

    assert m.total_prompt_tokens == 500  # 100 main + 400 fork
    assert m.total_completion_tokens == 75
    assert m.total_cached_tokens == 25
    assert m.total_cost_usd == pytest.approx(0.25)


def test_build_manifest_pi_records_cwd_configured_default_model(tmp_path: Path) -> None:
    """Issue #646 finding 1: a Pi deep run with no explicit override records the
    model PiBackend actually resolved — the cwd-configured default from
    ``<repo>/.pi/settings.json``, not DEFAULT_PI_MODEL — so the archived
    'who reviewed' identity matches what ran."""
    from daydream.backends.pi import _configured_pi_model
    from daydream.config import DEFAULT_PI_MODEL

    # Configure a Pi default in the repo cwd (the same seam PiBackend reads).
    pi_dir = tmp_path / ".pi"
    pi_dir.mkdir()
    (pi_dir / "settings.json").write_text(
        json.dumps({"defaultModel": "gpt-psr-configured"}), encoding="utf-8"
    )
    assert _configured_pi_model(tmp_path) == "gpt-psr-configured"

    m = build_manifest(
        recorder=cast(TrajectoryRecorder, _MockRecorder()),
        config=RunConfig(target=str(tmp_path), backend="pi", model=None),
        git_ctx=GitContext(),
        status="complete", archive_path=tmp_path,
        cwd=str(tmp_path),
    )
    assert m.per_stack_review_backend == "pi"
    assert m.per_stack_review_model == "gpt-psr-configured"
    assert m.per_stack_review_model != DEFAULT_PI_MODEL
    assert m.to_dict()["run"]["per_stack_review_model"] == "gpt-psr-configured"


def test_build_manifest_pi_falls_back_to_default_when_no_cwd_config(tmp_path: Path) -> None:
    """Issue #646 finding 1: with no cwd-configured Pi default (and no cwd passed),
    the manifest records DEFAULT_PI_MODEL as before — the fallback path is intact."""
    from daydream.config import DEFAULT_PI_MODEL

    m = build_manifest(
        recorder=cast(TrajectoryRecorder, _MockRecorder()),
        config=RunConfig(target=str(tmp_path), backend="pi", model=None),
        git_ctx=GitContext(),
        status="complete", archive_path=tmp_path,
    )
    assert m.per_stack_review_backend == "pi"
    assert m.per_stack_review_model == DEFAULT_PI_MODEL


def test_build_manifest_omits_per_stack_review_on_merge_fix_resume(tmp_path: Path) -> None:
    """Issue #646 finding 2: a --start-at merge/fix resume skips
    phase_per_stack_reviews (orchestrator.py:1132), so the manifest must not
    attribute the resume config's per-stack tier to the prior run's artifacts."""
    for start_at in ("merge", "fix"):
        config = RunConfig(
            target=str(tmp_path), backend=None, model=None,
            flow_name="deep", start_at=start_at,
        )
        m = build_manifest(
            recorder=cast(TrajectoryRecorder, _MockRecorder()),
            config=config, git_ctx=GitContext(),
            status="complete", archive_path=tmp_path,
        )
        run = m.to_dict()["run"]
        assert m.per_stack_review_backend is None, f"start_at={start_at}"
        assert m.per_stack_review_model is None, f"start_at={start_at}"
        assert "per_stack_review_backend" not in run, f"start_at={start_at}"
        assert "per_stack_review_model" not in run, f"start_at={start_at}"


def test_manifest_splits_status_from_pipeline():
    from daydream.archive.provenance import ExecutableProvenance
    m = Manifest(
        session_id="s-1", status="complete", archive_status="complete",
        pipeline_status="failed", phase_states={
            "merge": {"ran": True, "status": "failed"},
            "fix": {"ran": False, "status": "absent"},
            "test": {"ran": False, "status": "absent"},
        },
        daydream=ExecutableProvenance(version="0.27.0", install_source="git",
                                      commit="abc", dirty=False,
                                      container_digest="unknown"),
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


def test_legacy_manifest_reads_new_fields_as_unknown():
    legacy = {k: v for k, v in Manifest().to_dict().items()
              if k not in ("archive_status", "pipeline_status", "phase_states", "daydream")}
    # A raw manifest.json dict without the new keys still yields explicit
    # unknown for consumers, never a KeyError and never a fabricated value.
    assert legacy.get("pipeline_status", "unknown") == "unknown"
    assert legacy.get("archive_status", "unknown") == "unknown"
    assert legacy.get("daydream", "unknown") == "unknown"


def _write_deep(target: Path, name: str, data):
    deep = target / ".daydream" / "deep"
    deep.mkdir(parents=True, exist_ok=True)
    (deep / name).write_text(json.dumps(data), encoding="utf-8")


def test_merge_failed_discriminates_on_merge_key_not_merged_items(tmp_path):
    from daydream.archive import pipeline
    _write_deep(tmp_path, "merged-items.json", {"items": []})
    _write_deep(tmp_path, "per-stack-failures.json", {"__merge__": {"message": "x"}})
    states = pipeline.derive_phase_states(tmp_path, phase_events=[])
    assert states["merge"]["ran"] is True
    assert states["merge"]["status"] == "failed"   # merged-items present is NOT sufficient


def test_merge_succeeded_when_items_and_no_merge_key(tmp_path):
    from daydream.archive import pipeline
    _write_deep(tmp_path, "merged-items.json", {"items": []})
    states = pipeline.derive_phase_states(tmp_path, phase_events=[])
    assert states["merge"]["status"] == "succeeded"


def test_test_failed_from_verdict(tmp_path):
    from daydream.archive import pipeline
    _write_deep(tmp_path, "test-verdict.json", {"passed": False, "retries": 1, "ignored": False})
    states = pipeline.derive_phase_states(tmp_path, phase_events=[])
    assert states["test"]["ran"] is True
    assert states["test"]["status"] == "failed"


def test_test_absent_when_no_verdict(tmp_path):
    from daydream.archive import pipeline
    states = pipeline.derive_phase_states(tmp_path, phase_events=[])
    assert states["test"]["ran"] is False
    assert states["test"]["status"] == "absent"


def test_fix_partial_from_failures(tmp_path):
    from daydream.archive import pipeline
    _write_deep(tmp_path, "fix-failures.json", {"src/a.py": "reverted"})
    states = pipeline.derive_phase_states(tmp_path, phase_events=[])
    assert states["fix"]["status"] == "partial"


def test_pipeline_status_precedence():
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


def _deep(tmp_path: Path, name: str, data):
    d = tmp_path / ".daydream" / "deep"
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(data), encoding="utf-8")


def test_merge_failed_archives_failed_pipeline(tmp_path: Path, make_config: MakeConfig):
    from daydream.archive import _archive_run_inner
    from tests.harness.trajectory import make_recorder
    _deep(tmp_path, "merged-items.json", {"items": []})
    _deep(tmp_path, "per-stack-failures.json", {"__merge__": {"message": "x"}})
    _deep(tmp_path, "test-verdict.json", {"passed": False, "retries": 0, "ignored": False})
    recorder = make_recorder(tmp_path)  # run_flow NORMAL; fake config with archive=False
    config = make_config(tmp_path, archive=False)
    _archive_run_inner(recorder=recorder, target_dir=tmp_path, config=config,
                       status="complete", run_eval=False, work=None, upload=False)
    manifest_path = sorted(get_archive_dir().glob("runs/*/manifest.json"))[-1]
    m = json.loads(manifest_path.read_text())
    assert m["archive_status"] == "complete"   # cleanly archived...
    assert m["pipeline_status"] == "failed"    # ...but the pipeline failed
    assert m["phase_states"]["merge"]["status"] == "failed"
    assert m["daydream"]["version"]  # executable provenance recorded
    assert m["daydream"]["commit"] in {"unknown"} or m["daydream"]["commit"]


def test_schema_additive_columns_and_migration(tmp_path):
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


def test_upsert_run_persists_pipeline_fields(tmp_path):
    from daydream.archive import index
    from daydream.archive.manifest import Manifest
    from daydream.archive.provenance import ExecutableProvenance
    m = Manifest(session_id="s-2", status="complete", archive_status="complete",
                 pipeline_status="failed", phase_states={"merge": {"ran": True, "status": "failed"}},
                 daydream=ExecutableProvenance(version="0.27.0", install_source="git",
                                               commit="abc", dirty=False, container_digest="unknown"))
    index.upsert_run(tmp_path, m)
    row = index.query_runs(tmp_path, "session_id = ?", ("s-2",))[0]
    assert row["archive_status"] == "complete"
    assert row["pipeline_status"] == "failed"
    assert row["daydream_version"] == "0.27.0"
    assert row["daydream_dirty"] == 0
