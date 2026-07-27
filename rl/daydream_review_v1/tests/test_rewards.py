"""Phase 3: scoring, driven through ``Task.score`` with a real runtime.

Every test here stages real files on the host and scores through the production
entrypoint — ``await task.score(trace, runtime)`` — then asserts on what the
trainer would actually consume (``trace.rewards`` / ``trace.metrics`` /
``trace.info``). The runtime is the real ``SubprocessRuntime``: it shares the
host filesystem, so an absolute ``daydream_archive_root`` / ``daydream_repo_path``
in ``trace.info`` makes the reward path run real ``sh`` and real reads.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import verifiers.v1 as vf
from daydream.training.harvest import assemble_scoring_inputs
from daydream.training.reward import score_trajectory

from daydream_review_v1.fixture import build_fixture_repo
from daydream_review_v1.taskset import (
    DaydreamReviewConfig,
    DaydreamReviewTask,
    DaydreamReviewTaskset,
)

SESSION_ID = "9b36227a-9f80-41e5-a419-5cfed5a34b5b"

_CALC_BROKEN = '''"""A deliberately wrong calc.py: `add` is off by one, so test_add fails."""


def add(a: int, b: int) -> int:
    return a + b + 1


def divide(a: int, b: int) -> float:
    return a / b


def mean(values: list[int]) -> float:
    return sum(values) / len(values)
'''


def _task(corpus_mini_dir: Path, fixture_manifest_path: Path, *, pr_number: int = 1) -> DaydreamReviewTask:
    taskset = DaydreamReviewTaskset(
        DaydreamReviewConfig(
            id="daydream-review-v1",
            corpus_dir=corpus_mini_dir,
            manifest_path=fixture_manifest_path,
            use_images=False,
        )
    )
    return next(task for task in taskset.load() if task.data.pr_number == pr_number)


def _trace(task: DaydreamReviewTask, *, archive_root: Path, repo_path: Path) -> vf.Trace:
    trace: vf.Trace = vf.Trace(task=vf.TraceTask(type=type(task).__name__, data=task.data))
    trace.info["daydream_archive_root"] = str(archive_root)
    trace.info["daydream_repo_path"] = str(repo_path)
    return trace


def _stage_run(archive_root: Path, source: Path, *, session_id: str = SESSION_ID) -> Path:
    """Copy an archived run dir to ``<archive_root>/runs/<session_id>``."""
    dest = archive_root / "runs" / session_id
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, dest)
    return dest


def _manifest_row_like_production(run_dir: Path) -> dict[str, object]:
    """Flatten ``manifest.json`` exactly as ``taskset._manifest_row`` does."""
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    return {**manifest, **(manifest.get("metrics") or {})}


def _stage_repo(repo_path: Path, *, patch: str | None, red: bool = False) -> Path:
    """Build the fixture repo at *repo_path*, optionally red, with a fix marker."""
    build_fixture_repo(repo_path)
    if red:
        (repo_path / "calc.py").write_text(_CALC_BROKEN, encoding="utf-8")
    if patch is not None:
        daydream_dir = repo_path / ".daydream"
        daydream_dir.mkdir(parents=True, exist_ok=True)
        (daydream_dir / "recommended.patch").write_text(patch, encoding="utf-8")
    return repo_path


_REAL_PATCH = "diff --git a/tests/test_calc.py b/tests/test_calc.py\n@@ -1 +1 @@\n-old\n+new\n"


async def test_intrinsic_composite_parity(
    tmp_path: Path, runtime, rundir_golden: Path, corpus_mini_dir: Path, fixture_manifest_path: Path
) -> None:
    """The online reward is byte-equal to the offline pipeline's own scorer."""
    archive_root = tmp_path / "archive"
    _stage_run(archive_root, rundir_golden)
    task = _task(corpus_mini_dir, fixture_manifest_path)
    trace = _trace(task, archive_root=archive_root, repo_path=tmp_path / "repo")

    await task.score(trace, runtime)

    expected = score_trajectory(
        assemble_scoring_inputs(rundir_golden, _manifest_row_like_production(rundir_golden))
    ).composite
    assert expected is not None
    assert trace.rewards["intrinsic_composite"] == expected
    assert trace.info["reward_breakdown"]["composite"] == expected


async def test_intrinsic_composite_carries_the_grounding_axis(
    tmp_path: Path, runtime, rundir_golden: Path, corpus_mini_dir: Path, fixture_manifest_path: Path
) -> None:
    """grounding_rate lives under ``metrics`` in the archive; reading the manifest
    verbatim would null the axis. The expectation comes from evaluation.json, so
    that regression cannot hide behind a manifest-shaped fixture."""
    evaluation = json.loads((rundir_golden / "evaluation.json").read_text(encoding="utf-8"))
    expected_grounding = evaluation["grounding"]["grounding_rate"]
    assert expected_grounding == 1.0, "fixture drift: the golden run is fully grounded"

    archive_root = tmp_path / "archive"
    _stage_run(archive_root, rundir_golden)
    task = _task(corpus_mini_dir, fixture_manifest_path)
    trace = _trace(task, archive_root=archive_root, repo_path=tmp_path / "repo")

    await task.score(trace, runtime)

    breakdown = trace.info["reward_breakdown"]
    assert breakdown["axes_present"]["grounding"] is True
    assert breakdown["grounding"] is not None
    assert breakdown["grounding"] == expected_grounding


async def test_missing_run_dir_scores_zero(
    tmp_path: Path, runtime, corpus_mini_dir: Path, fixture_manifest_path: Path
) -> None:
    """A crashed daydream still gets a gradient — zero, not an exception."""
    archive_root = tmp_path / "archive"
    (archive_root / "runs").mkdir(parents=True)
    task = _task(corpus_mini_dir, fixture_manifest_path)
    trace = _trace(task, archive_root=archive_root, repo_path=tmp_path / "repo")

    await task.score(trace, runtime)

    assert trace.rewards["intrinsic_composite"] == 0.0
    assert trace.info["reward_breakdown"] == {"error": "no archived run dir"}
    assert trace.metrics["n_findings"] == 0.0


async def test_fix_tests_pass_green(
    tmp_path: Path, runtime, corpus_mini_dir: Path, fixture_manifest_path: Path
) -> None:
    archive_root = tmp_path / "archive"
    (archive_root / "runs").mkdir(parents=True)
    repo = _stage_repo(tmp_path / "repo", patch=_REAL_PATCH)
    task = _task(corpus_mini_dir, fixture_manifest_path)
    assert task.data.test_command == "python -m unittest discover -q"
    trace = _trace(task, archive_root=archive_root, repo_path=repo)

    await task.score(trace, runtime)

    assert trace.metrics["fixes_applied"] == 1.0
    assert trace.rewards["fix_tests_pass"] == 1.0


async def test_fix_tests_pass_red(
    tmp_path: Path, runtime, corpus_mini_dir: Path, fixture_manifest_path: Path
) -> None:
    archive_root = tmp_path / "archive"
    (archive_root / "runs").mkdir(parents=True)
    repo = _stage_repo(tmp_path / "repo", patch=_REAL_PATCH, red=True)
    task = _task(corpus_mini_dir, fixture_manifest_path)
    trace = _trace(task, archive_root=archive_root, repo_path=repo)

    await task.score(trace, runtime)

    assert trace.metrics["fixes_applied"] == 1.0
    assert trace.rewards["fix_tests_pass"] == 0.0


@pytest.mark.parametrize("patch", [None, ""], ids=["absent", "empty"])
async def test_no_fixes_returns_no_fix_reward(
    patch: str | None, tmp_path: Path, runtime, corpus_mini_dir: Path, fixture_manifest_path: Path
) -> None:
    """daydream writes an EMPTY recommended.patch when no fix landed; a green tree
    it never touched is not evidence the suite was fixed."""
    archive_root = tmp_path / "archive"
    (archive_root / "runs").mkdir(parents=True)
    repo = _stage_repo(tmp_path / "repo", patch=patch)
    task = _task(corpus_mini_dir, fixture_manifest_path)
    trace = _trace(task, archive_root=archive_root, repo_path=repo)

    await task.score(trace, runtime)

    assert trace.metrics["fixes_applied"] == 0.0
    assert trace.rewards["fix_tests_pass"] == task.config.no_fix_reward
    assert "test_claim_mismatch" not in trace.metrics


@pytest.mark.parametrize("red, expected", [(True, 1.0), (False, 0.0)], ids=["mismatch", "agrees"])
async def test_metric_claim_mismatch_fires(
    red: bool,
    expected: float,
    tmp_path: Path,
    runtime,
    rundir_golden: Path,
    corpus_mini_dir: Path,
    fixture_manifest_path: Path,
) -> None:
    """daydream's prose-derived verdict is a claim, graded against the real re-run."""
    archive_root = tmp_path / "archive"
    run_dir = _stage_run(archive_root, rundir_golden)
    claim = json.loads((run_dir / "deep" / "test-verdict.json").read_text(encoding="utf-8"))
    assert claim == {"passed": True, "retries": 0}

    repo = _stage_repo(tmp_path / "repo", patch=_REAL_PATCH, red=red)
    task = _task(corpus_mini_dir, fixture_manifest_path)
    trace = _trace(task, archive_root=archive_root, repo_path=repo)

    await task.score(trace, runtime)

    assert trace.metrics["fixes_applied"] == 1.0
    assert trace.rewards["fix_tests_pass"] == (0.0 if red else 1.0)
    assert trace.metrics["test_claim_mismatch"] == expected


async def test_review_shape_metrics(
    tmp_path: Path, runtime, rundir_golden: Path, corpus_mini_dir: Path, fixture_manifest_path: Path
) -> None:
    """n_findings mirrors merged-items.json; golden_overlap is a path fraction."""
    archive_root = tmp_path / "archive"
    _stage_run(archive_root, rundir_golden)
    task = _task(corpus_mini_dir, fixture_manifest_path)
    trace = _trace(task, archive_root=archive_root, repo_path=tmp_path / "repo")

    await task.score(trace, runtime)

    items = json.loads((rundir_golden / "deep" / "merged-items.json").read_text(encoding="utf-8"))["items"]
    found_files = {item["file"] for item in items}
    golden_paths = [comment.path for comment in task.data.golden_comments if comment.path]
    assert golden_paths, "the corpus fixture must carry at least one golden path"

    assert trace.metrics["n_findings"] == float(len(items))
    assert trace.metrics["n_golden_comments"] == float(len(golden_paths))
    assert trace.metrics["golden_overlap"] == sum(1 for p in golden_paths if p in found_files) / len(golden_paths)
    # The golden run found tests/test_calc.py; the bot commented on calc.py.
    assert found_files.isdisjoint(golden_paths) and trace.metrics["golden_overlap"] == 0.0

    # Same task, a run that DID land on the bot's file: the fraction must move.
    hit_root = tmp_path / "archive-hit"
    hit_run = _stage_run(hit_root, rundir_golden, session_id="session-hit")
    (hit_run / "deep" / "merged-items.json").write_text(
        json.dumps({"items": [{"id": 1, "file": golden_paths[0]}, {"id": 2, "file": "README.md"}]}),
        encoding="utf-8",
    )
    hit_trace = _trace(task, archive_root=hit_root, repo_path=tmp_path / "repo")

    await task.score(hit_trace, runtime)

    assert hit_trace.metrics["n_findings"] == 2.0
    assert hit_trace.metrics["golden_overlap"] == 1.0
