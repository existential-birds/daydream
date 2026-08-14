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
import subprocess
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


def _stage_repo(
    repo_path: Path,
    head_sha: str,
    *,
    edit: str | None = None,
    patch: str | None = None,
    commit: bool = False,
    commit_patch: bool = False,
) -> Path:
    """Build the fixture repo detached at *head_sha*, as the rollout image does.

    Args:
        edit: New ``calc.py`` contents, standing in for a fix the agent applied.
        patch: Contents to plant at ``.daydream/recommended.patch``. Planting one
            WITHOUT an edit is the contamination case: daydream writes that file
            into the repository under review and its own untracked artifacts leak
            into it, so it must never be read as "a fix landed".
        commit: Commit, moving HEAD past the baked snapshot — what the deep
            flow does once the suite goes green. Without an ``edit`` this is an
            ``--allow-empty`` commit: HEAD advances but the committed tree is
            byte-identical to the snapshot (the empty-commit regression).
        commit_patch: ALSO force-add and commit the planted ``.daydream/``
            directory. Real targets don't gitignore ``.daydream/`` (only the
            fixture does), so committing it faithfully reproduces a no-fix
            rollout whose agent committed daydream's own artifacts — a tree
            that differs from the snapshot for no reason other than the
            artifacts. Without this option, ``commit=True`` stages only real
            edits.
    """
    build_fixture_repo(repo_path)
    subprocess.run(["git", "-C", str(repo_path), "checkout", "--quiet", "--detach", head_sha], check=True)
    if edit is not None:
        (repo_path / "calc.py").write_text(edit, encoding="utf-8")
    if patch is not None:
        daydream_dir = repo_path / ".daydream"
        daydream_dir.mkdir(parents=True, exist_ok=True)
        (daydream_dir / "recommended.patch").write_text(patch, encoding="utf-8")
    if commit:
        subprocess.run(["git", "-C", str(repo_path), "add", "-A"], check=True)
        if commit_patch:
            # The fixture gitignores .daydream/ (fixture.py _GITIGNORE), unlike
            # real targets, so force-add it to stage the committed-artifacts case.
            subprocess.run(["git", "-C", str(repo_path), "add", "-f", ".daydream"], check=True)
        subprocess.run(["git", "-C", str(repo_path), "commit", "--quiet", "--allow-empty", "-m", "fix"], check=True)
    return repo_path


_REAL_PATCH = "diff --git a/tests/test_calc.py b/tests/test_calc.py\n@@ -1 +1 @@\n-old\n+new\n"

_CALC_FIXED = _CALC_BROKEN.replace("return a + b + 1", "return a + b")


def test_rundir_golden_step_11_message_is_inert(rundir_golden: Path) -> None:
    """No user-authored step in the committed golden trajectory may carry
    instruction-bearing text an automated reviewer could mistake for trusted
    direction. This is a clean-pass fixture, so every user slot must be inert;
    a directive reappearing in any user slot fails the build. The step-11
    message is the canonical case and must also keep its structural shape."""
    trajectory = json.loads((rundir_golden / "trajectory.json").read_text(encoding="utf-8"))
    user_steps = [s for s in trajectory["steps"] if s.get("source") == "user"]
    assert user_steps, "expected at least one user-authored step"
    for step in user_steps:
        assert step["message"] == ""
    step_11 = [s for s in trajectory["steps"] if s.get("step_id") == 11]
    assert len(step_11) == 1
    step = step_11[0]
    assert set(step) == {"extra", "message", "source", "step_id", "timestamp"}
    assert step["source"] == "user"
    assert step["message"] == ""


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


async def test_zero_finding_rollout_scores_no_intrinsic_reward(
    tmp_path: Path, runtime, rundir_golden: Path, corpus_mini_dir: Path, fixture_manifest_path: Path
) -> None:
    """Saying nothing must not be the cheapest path to a perfect reward.

    A review that reports ZERO findings has an UNDEFINED grounding rate — the
    ratio has no denominator — and no verified recommendations, so it has no
    credit axis at all. Scoring the empty ratio as a vacuous 1.0 handed such a
    rollout the maximum ``intrinsic_composite``, making silence the degenerate
    optimum the policy would learn first. With ``grounding_rate`` undefined the
    whole credit side must be absent, ``composite`` must be ``None``, and the
    reward the trainer sums must be 0.0.
    """
    archive_root = tmp_path / "archive"
    run_dir = _stage_run(archive_root, rundir_golden)

    (run_dir / "deep" / "merged-items.json").write_text(json.dumps({"items": []}), encoding="utf-8")
    # The correctness axis comes from deep/recommendation-verdicts.json
    # (daydream/training/harvest.py:196-201 -> daydream/training/reward.py:369-374);
    # the golden run ships one, and leaving it would keep the composite non-None.
    (run_dir / "deep" / "recommendation-verdicts.json").unlink()

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["metrics"]["grounding_rate"] = None
    manifest["metrics"]["total_findings"] = 0
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    evaluation = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
    evaluation["grounding"]["grounding_rate"] = None
    evaluation["findings"]["total"] = 0
    (run_dir / "evaluation.json").write_text(json.dumps(evaluation), encoding="utf-8")

    task = _task(corpus_mini_dir, fixture_manifest_path)
    trace = _trace(task, archive_root=archive_root, repo_path=tmp_path / "repo")

    await task.score(trace, runtime)

    breakdown = trace.info["reward_breakdown"]
    assert trace.rewards["intrinsic_composite"] == 0.0
    assert breakdown["axes_present"]["grounding"] is False
    assert breakdown["composite"] is None
    # Not merely "grounding went away": no credit axis survives, so the
    # composite is uncomputable rather than carried by a stale verdict file.
    assert breakdown["axes_present"]["correctness"] is False


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
    task = _task(corpus_mini_dir, fixture_manifest_path)
    repo = _stage_repo(tmp_path / "repo", task.data.head_sha, edit=_CALC_FIXED, patch=_REAL_PATCH)
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
    task = _task(corpus_mini_dir, fixture_manifest_path)
    repo = _stage_repo(tmp_path / "repo", task.data.head_sha, edit=_CALC_BROKEN, patch=_REAL_PATCH)
    trace = _trace(task, archive_root=archive_root, repo_path=repo)

    await task.score(trace, runtime)

    assert trace.metrics["fixes_applied"] == 1.0
    assert trace.rewards["fix_tests_pass"] == 0.0


@pytest.mark.parametrize(
    "patch, commit",
    [
        (None, False),
        ("", False),
        (_REAL_PATCH, False),
        (None, True),  # empty commit: HEAD advances, committed tree unchanged
    ],
    ids=["no-patch-file", "empty-patch", "non-empty-patch-but-untouched-tree", "empty-commit"],
)
async def test_no_fixes_returns_no_fix_reward(
    patch: str | None, commit: bool, tmp_path: Path, runtime, corpus_mini_dir: Path, fixture_manifest_path: Path
) -> None:
    """An untouched tree earns no_fix_reward however recommended.patch looks.

    The third case is the one that matters. daydream writes `.daydream/` INTO the
    repository under review, and `capture_recommended_patch` appends a creation
    hunk for every untracked non-ignored file — so on any repository that does not
    gitignore that directory (i.e. every real repository), the patch is non-empty
    after a rollout that changed nothing. Reading it as "a fix landed" would hand
    out the full w_tests off the still-green baseline, for free, forever.
    """
    archive_root = tmp_path / "archive"
    (archive_root / "runs").mkdir(parents=True)
    task = _task(corpus_mini_dir, fixture_manifest_path)
    repo = _stage_repo(tmp_path / "repo", task.data.head_sha, patch=patch, commit=commit)
    trace = _trace(task, archive_root=archive_root, repo_path=repo)

    await task.score(trace, runtime)

    assert trace.metrics["fixes_applied"] == 0.0
    assert trace.rewards["fix_tests_pass"] == task.config.no_fix_reward
    assert "test_claim_mismatch" not in trace.metrics
    # No archived run at all, so there is no claim to record either.
    assert "test_claim_passed_without_fix" not in trace.metrics


@pytest.mark.parametrize("head_sha", ["0" * 40], ids=["unresolvable-head"])
async def test_unresolvable_head_sha_scores_no_fix(
    head_sha: str, tmp_path: Path, runtime, corpus_mini_dir: Path, fixture_manifest_path: Path
) -> None:
    """A baked snapshot object that no longer resolves must read as no fix.

    `git diff --quiet <head_sha> HEAD` exits 128 when the head object is absent
    from the object store (the documented `exit_code != 1` branch in
    `_fixes_applied`). Unlike a string comparison on `rev-parse`, that must not
    be scored as a fix: every non-1 exit is `no_fix_reward`, honoring the
    deliberate false-negative bias.
    """
    archive_root = tmp_path / "archive"
    (archive_root / "runs").mkdir(parents=True)
    task = _task(corpus_mini_dir, fixture_manifest_path)
    # Stage the real, resolvable snapshot first so the checkout succeeds, then
    # simulate snapshot/object-store drift: the baked head object is gone.
    repo = _stage_repo(tmp_path / "repo", task.data.head_sha)
    task.data = task.data.model_copy(update={"head_sha": head_sha})
    trace = _trace(task, archive_root=archive_root, repo_path=repo)

    await task.score(trace, runtime)

    assert trace.metrics["fixes_applied"] == 0.0
    assert trace.rewards["fix_tests_pass"] == task.config.no_fix_reward


@pytest.mark.parametrize("claimed", [True, False], ids=["claimed-green", "claimed-red"])
async def test_no_fixes_still_records_the_test_claim(
    claimed: bool,
    tmp_path: Path,
    runtime,
    rundir_golden: Path,
    corpus_mini_dir: Path,
    fixture_manifest_path: Path,
) -> None:
    """Changing nothing while claiming the suite is green must stay visible.

    The no-fix path returns before the re-run, so `test_claim_mismatch` can never
    fire there — and the rollout that touched nothing yet archived a green
    `deep/test-verdict.json` is the sharpest hack shape of all. The claim is
    recorded on its own, without buying a second suite run.
    """
    archive_root = tmp_path / "archive"
    run_dir = _stage_run(archive_root, rundir_golden)
    (run_dir / "deep" / "test-verdict.json").write_text(
        json.dumps({"passed": claimed, "retries": 0}), encoding="utf-8"
    )

    task = _task(corpus_mini_dir, fixture_manifest_path)
    repo = _stage_repo(tmp_path / "repo", task.data.head_sha, patch=_REAL_PATCH)
    trace = _trace(task, archive_root=archive_root, repo_path=repo)

    await task.score(trace, runtime)

    assert trace.metrics["fixes_applied"] == 0.0
    assert trace.rewards["fix_tests_pass"] == task.config.no_fix_reward
    assert trace.metrics["test_claim_passed_without_fix"] == float(claimed)
    # Observability only: recording the claim must not invent a verdict comparison.
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

    task = _task(corpus_mini_dir, fixture_manifest_path)
    repo = _stage_repo(
        tmp_path / "repo",
        task.data.head_sha,
        edit=_CALC_BROKEN if red else _CALC_FIXED,
        patch=_REAL_PATCH,
    )
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


async def test_review_shape_survives_a_non_object_merged_items(
    tmp_path: Path, runtime, rundir_golden: Path, corpus_mini_dir: Path, fixture_manifest_path: Path
) -> None:
    """merged-items.json is written inside the rollout, so a corrupt one must not crash scoring."""
    archive_root = tmp_path / "archive"
    run_dir = _stage_run(archive_root, rundir_golden)
    (run_dir / "deep" / "merged-items.json").write_text(json.dumps([{"file": "calc.py"}]), encoding="utf-8")
    task = _task(corpus_mini_dir, fixture_manifest_path)
    trace = _trace(task, archive_root=archive_root, repo_path=tmp_path / "repo")

    await task.score(trace, runtime)

    assert trace.metrics["n_findings"] == 0.0
    assert trace.metrics["golden_overlap"] == 0.0


async def test_committed_fix_counts_even_with_a_clean_tree(
    tmp_path: Path, runtime, corpus_mini_dir: Path, fixture_manifest_path: Path
) -> None:
    """The deep flow commits once the suite is green, leaving nothing to `git diff`.

    A fix-detection rule that only looked at working-tree changes would score
    every successful rollout as "no fix" — the exact inverse mistake.
    """
    archive_root = tmp_path / "archive"
    (archive_root / "runs").mkdir(parents=True)
    task = _task(corpus_mini_dir, fixture_manifest_path)
    repo = _stage_repo(tmp_path / "repo", task.data.head_sha, edit=_CALC_FIXED, commit=True)
    trace = _trace(task, archive_root=archive_root, repo_path=repo)

    await task.score(trace, runtime)

    assert trace.metrics["fixes_applied"] == 1.0
    assert trace.rewards["fix_tests_pass"] == 1.0


async def test_committed_daydream_artifacts_not_a_fix(
    tmp_path: Path, runtime, corpus_mini_dir: Path, fixture_manifest_path: Path
) -> None:
    """Committing daydream's own .daydream/ artifacts must not read as a fix.

    The pathspec exclusion owns the commit path. Real targets don't gitignore
    ``.daydream/`` (only the fixture does), so a no-fix rollout whose agent
    commits daydream's own untracked artifacts produces a tree that differs from
    the baked snapshot — which ``git diff --quiet <head_sha> HEAD`` would read as
    a fix and pay the full ``w_tests`` off a still-green baseline. Committing the
    artifacts (force-added, since the fixture ignores them) must score
    ``no_fix_reward``.
    """
    archive_root = tmp_path / "archive"
    (archive_root / "runs").mkdir(parents=True)
    task = _task(corpus_mini_dir, fixture_manifest_path)
    repo = _stage_repo(
        tmp_path / "repo", task.data.head_sha, patch=_REAL_PATCH, commit=True, commit_patch=True
    )
    trace = _trace(task, archive_root=archive_root, repo_path=repo)

    await task.score(trace, runtime)

    assert trace.metrics["fixes_applied"] == 0.0
    assert trace.rewards["fix_tests_pass"] == task.config.no_fix_reward
    assert "test_claim_mismatch" not in trace.metrics


async def test_unresolvable_snapshot_sha_reads_as_no_fix(
    tmp_path: Path, runtime, corpus_mini_dir: Path, fixture_manifest_path: Path
) -> None:
    """A fix signal that cannot be evaluated reads as no-fix, not a free win.

    ``fix_tests_pass`` stays deliberately false-negative biased: any ``git diff
    --quiet`` exit other than 1 (0 = identical trees, 128 = unresolvable baked
    SHA, 127 = missing sh/git) means "no fix found". Here the baked snapshot SHA
    is not present in the repository at all, so the diff exits 128 — the reward
    must be ``no_fix_reward``, never the full ``w_tests`` for nothing.
    """
    archive_root = tmp_path / "archive"
    (archive_root / "runs").mkdir(parents=True)
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "fix@fixture.invalid"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "fixture"], check=True)
    (repo / "calc.py").write_text(_CALC_BROKEN, encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "--quiet", "-m", "snapshot"], check=True)

    task = _task(corpus_mini_dir, fixture_manifest_path)
    # task.data.head_sha is the baked snapshot SHA from the corpus; this fresh
    # repo has never contained it, so the fix-signal diff cannot resolve it.
    trace = _trace(task, archive_root=archive_root, repo_path=repo)

    await task.score(trace, runtime)

    assert trace.metrics["fixes_applied"] == 0.0
    assert trace.rewards["fix_tests_pass"] == task.config.no_fix_reward


async def test_reward_version_is_pinned(
    tmp_path: Path, runtime, rundir_golden: Path, corpus_mini_dir: Path, fixture_manifest_path: Path
) -> None:
    """AC-3: pin the scorer version the parity test cannot see move.

    `test_intrinsic_composite_parity` runs the same `score_trajectory` on both
    sides, so a REWARD_VERSION bump moves expectation and actual together and the
    parity assertion stays green through a semantic change. This is the assertion
    that actually fails when the offline scorer changes under us.
    """
    from daydream.training.reward import REWARD_VERSION

    assert REWARD_VERSION == "2026.05.28-2", (
        f"the training pipeline's reward version moved to {REWARD_VERSION!r}. Re-derive the "
        "rollout reward's expected values before trusting any run scored across the boundary."
    )

    archive_root = tmp_path / "archive"
    _stage_run(archive_root, rundir_golden)
    task = _task(corpus_mini_dir, fixture_manifest_path)
    trace = _trace(task, archive_root=archive_root, repo_path=tmp_path / "repo")

    await task.score(trace, runtime)

    assert trace.info["reward_breakdown"]["reward_version"] == REWARD_VERSION
