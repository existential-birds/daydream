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
import re
import shutil
import subprocess
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest
import verifiers.v1 as vf
from daydream.atif import validate
from daydream.training.harvest import assemble_scoring_inputs
from daydream.training.reward import score_trajectory

from daydream_review_v1.fixture import build_fixture_repo
from daydream_review_v1.taskset import (
    DaydreamReviewConfig,
    DaydreamReviewState,
    DaydreamReviewTask,
    DaydreamReviewTaskset,
    _review_state,
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
    trace: vf.Trace = vf.Trace(
        task=vf.TraceTask(type=type(task).__name__, data=task.data), state=DaydreamReviewState()
    )
    trace.info["daydream_archive_root"] = str(archive_root)
    trace.info["daydream_repo_path"] = str(repo_path)
    return trace


def test_review_state_guard_rejects_base_state(
    corpus_mini_dir: Path, fixture_manifest_path: Path
) -> None:
    """Scoring state must be a DaydreamReviewState, never the base State.

    Production traces carry a DaydreamReviewState automatically (the rollout
    resolves the task's StateT through the MRO); test helpers that score must
    pass one explicitly. A base State must fail loudly rather than silently
    dereference a missing run_dir.
    """
    task = _task(corpus_mini_dir, fixture_manifest_path)
    base_trace = vf.Trace(task=vf.TraceTask(type=type(task).__name__, data=task.data))
    with pytest.raises(TypeError):
        _review_state(base_trace)
    good_trace = vf.Trace(
        task=vf.TraceTask(type=type(task).__name__, data=task.data),
        state=DaydreamReviewState(),
    )
    assert _review_state(good_trace).run_dir is None


async def test_score_without_runtime_records_nothing(
    corpus_mini_dir: Path, fixture_manifest_path: Path
) -> None:
    """The offline replay path — ``score(trace, None)`` — completes and records nothing.

    The verifiers replay CLI scores archived traces with no runtime; the base
    then skips every runtime-dependent signal (all three handlers here require
    ``runtime``) and this task stages no run dir. The trace deliberately carries
    the base ``State`` — the load-bearing shape from the replay path — so
    reaching ``_review_state`` would raise TypeError. Completing, with no
    rewards/metrics recorded, is the proof the offline branch never touches the
    state guard, the run-dir fetch, or the runtime.
    """
    task = _task(corpus_mini_dir, fixture_manifest_path)
    trace = vf.Trace(task=vf.TraceTask(type=type(task).__name__, data=task.data))
    trace.info["daydream_archive_root"] = "/does/not/exist"
    trace.info["daydream_repo_path"] = "/does/not/exist"

    await task.score(trace)  # runtime defaults to None — the offline replay path

    assert trace.rewards == {}
    assert trace.metrics == {}


def _stage_run(archive_root: Path, source: Path, *, session_id: str = SESSION_ID) -> Path:
    """Copy an archived run dir to ``<archive_root>/runs/<session_id>``."""
    dest = archive_root / "runs" / session_id
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, dest)
    return dest


# Absolute Unix path shape (``/Users/...``, ``/private/tmp/...``, ``/home/...``):
# a ``/`` after a string boundary, followed by a letter. The original capture
# host's ``/private/tmp/`` prefix was one instance of this shape; the guard
# rejects any match in either fixture blob, not just that one prefix.
#
# NOTE: this runs over ``json.dumps(blob)``, where embedded newlines are escaped
# to ``\n``, so a path whose leading ``/`` sits at the start of a line *inside* a
# content string (rather than at a JSON value boundary) is invisible to it. That
# shape is not present in today's fixtures -- every real machine path sat at a
# JSON value boundary -- so the guard is a first-line check, not a proof that no
# absolute-path shape exists anywhere.
_ABS_PATH_RE = re.compile(r'(?:^|[\s"\'])/[A-Za-z]')


def _walk_json_keys(node: object, targets: set[str], prefix: str = "") -> list[str]:
    """Collect every JSON key path whose key is in *targets* (recursive).

    Each entry is a dot-joined path from the JSON root (e.g.
    ``steps.0.reasoning_content``), so a failure message locates the offending
    key instead of naming a bare key that may occur at many depths.
    """
    found: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            path = f"{prefix}.{k}" if prefix else k
            if k in targets:
                found.append(path)
            found.extend(_walk_json_keys(v, targets, path))
    elif isinstance(node, list):
        for index, item in enumerate(node):
            found.extend(_walk_json_keys(item, targets, f"{prefix}.{index}"))
    return found


def test_rundir_golden_fixture_is_clean(rundir_golden: Path) -> None:
    """Static fixture guard: the rundir-golden fixture ships no dangling per-fork
    trajectory refs, no machine-specific absolute paths, and no embedded
    model-directed prompt copies in agent-step internals.

    The per-fork transcripts under ``trajectories/`` are gone; the retained root
    ``trajectory.json`` and ``manifest.json`` must not carry operational dirt
    from the real run that produced them. This guard enforces the pruned state:
    (1) no ``trajectory_path``/``sibling_trajectory_ref`` key references a
    non-shipped transcript; (2) no machine-specific absolute path shape
    appears anywhere; (3) no ``reasoning_content``/``tool_calls`` field
    carries model-directed prompt text. The runtime projection exclusion in
    ``test_rundir.py`` remains the boundary that keeps this data out of model
    context.
    """
    trajectories = rundir_golden / "trajectories"
    assert not trajectories.is_dir() or not next(trajectories.iterdir(), None)

    trajectory = json.loads((rundir_golden / "trajectory.json").read_text(encoding="utf-8"))
    manifest = json.loads((rundir_golden / "manifest.json").read_text(encoding="utf-8"))

    # 1. no dangling per-fork transcript refs
    dangling = _walk_json_keys(trajectory, {"trajectory_path", "sibling_trajectory_ref"})
    assert not dangling, f"dangling per-fork trajectory refs: {dangling}"

    # 2. no machine-specific absolute paths (all shipped golden fixtures)
    evaluation = json.loads((rundir_golden / "evaluation.json").read_text(encoding="utf-8"))
    for name, blob in (
        ("trajectory.json", trajectory),
        ("manifest.json", manifest),
        ("evaluation.json", evaluation),
    ):
        assert not _ABS_PATH_RE.search(json.dumps(blob)), (
            f"{name} carries a machine-specific absolute path"
        )

    # 2b. the shipped trajectory must satisfy the codebase's own ATIF validator
    #     (daydream.atif.validate, the primary guard): a prune must not leave
    #     dangling observation source_call_id refs that hard-fail validation.
    assert validate(trajectory) is True, (
        "trajectory.json fails daydream.atif.validate (dangling tool-call refs)"
    )

    # 3. no embedded model-directed prompt copies in agent-step internals
    for step in trajectory["steps"]:
        rc = step.get("reasoning_content")
        assert not rc, f"step {step.get('step_id')} carries reasoning_content prompt text"
        tcs = step.get("tool_calls")
        assert not tcs, f"step {step.get('step_id')} carries tool_calls prompt text"


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

# A gutted suite that PASSES if run: a vulnerable impl that executes the
# tampered oracle gets exit 0 -> reward 1.0. The gate must return 0.0 instead.
_TAMPER_PASSING = (
    "import unittest\n\n"
    "class TestCalc(unittest.TestCase):\n"
    "    def test_placeholder(self) -> None:\n"
    "        pass\n"
)

# Overwriting the tracked tests/__init__.py to skip discovery: suite still exits 0.
_TAMPER_SKIPALL = 'import unittest\nraise unittest.SkipTest("gutted")\n'


def test_rundir_golden_user_messages_are_inert(rundir_golden: Path) -> None:
    """Defensive regression guard only: every user-authored step in the
    committed golden trajectory must carry an empty message. The
    reward/verification path derives its signals from ``deep/`` verdicts and
    review-output length, never from this trajectory's step messages, so this
    does not reduce harm on that path — it is fixture hygiene for a clean-pass
    reference, not a security control. Every user slot of the fixture must be
    inert; a directive reappearing in any user step fails the build. The
    step-11 verify slot is the canonical case and must still keep its
    structural shape."""
    trajectory = json.loads((rundir_golden / "trajectory.json").read_text(encoding="utf-8"))
    user_steps = [s for s in trajectory["steps"] if s.get("source") == "user"]
    assert user_steps, "expected at least one user-authored step"
    for step in user_steps:
        assert step["message"] == ""
    step_11 = [s for s in trajectory["steps"] if s.get("step_id") == 11]
    assert len(step_11) == 1
    step = step_11[0]
    # Structural shape check on the required fields only: exact key-set equality
    # would fail the build on any message-neutral ATIF metadata field added to
    # the record, decoupling unrelated schema drift from the inert-message guard.
    assert {"extra", "message", "source", "step_id", "timestamp"} <= set(step)
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
    assert trace.metrics["test_oracle_unchanged"] == 1.0


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
    ("edit", "tamper_rel", "tamper_content"),
    [
        # test-source-tamper: overwrite a tracked test with a passing suite
        (None, "tests/test_calc.py", _TAMPER_PASSING),
        # test-package-config: overwrite tracked tests/__init__.py to skip discovery
        (None, "tests/__init__.py", _TAMPER_SKIPALL),
        # untracked oracle file: a real fix (calc.py) plus a NEW non-ignored
        # tests/pytest.ini — ls-files --others must catch it
        (_CALC_FIXED, "tests/pytest.ini", "[pytest]\n"),
    ],
    ids=["test-source-tamper", "test-package-config-tamper", "untracked-oracle-file"],
)
async def test_fix_tests_pass_rejects_protected_test_path_changes(
    edit: str | None,
    tamper_rel: str,
    tamper_content: str,
    tmp_path: Path,
    runtime,
    rundir_golden: Path,
    corpus_mini_dir: Path,
    fixture_manifest_path: Path,
) -> None:
    """A changed test oracle earns a literal zero, never the w_tests reward.

    Every tamper row is deliberately green-if-run: a vulnerable impl that
    executes the tampered oracle gets exit 0 -> reward 1.0. The gate must
    return 0.0 without running test_command. Staging a real
    ``deep/test-verdict.json`` claim makes the tripwire load-bearing: any
    execution of test_command records ``test_claim_mismatch`` (0.0 on this
    green tampered suite, 1.0 if it ran red), so its absence below proves the
    gate held rather than passing vacuously over an empty archive.
    """
    archive_root = tmp_path / "archive"
    _stage_run(archive_root, rundir_golden)
    task = _task(corpus_mini_dir, fixture_manifest_path)
    repo = _stage_repo(tmp_path / "repo", task.data.head_sha, edit=edit)
    (repo / tamper_rel).write_text(tamper_content, encoding="utf-8")
    trace = _trace(task, archive_root=archive_root, repo_path=repo)

    await task.score(trace, runtime)

    assert trace.metrics["fixes_applied"] == 1.0
    assert trace.metrics["test_oracle_unchanged"] == 0.0
    assert trace.rewards["fix_tests_pass"] == 0.0
    # The staged claim (passed=True) means ANY test_command execution would
    # record test_claim_mismatch; its absence proves the gate held.
    assert "test_claim_mismatch" not in trace.metrics


async def test_oracle_gate_fails_closed_on_git_error(
    tmp_path: Path,
    runtime,
    rundir_golden: Path,
    corpus_mini_dir: Path,
    fixture_manifest_path: Path,
) -> None:
    """A Git error in the oracle comparison must read as 'changed' (zero reward).

    The staged ``deep/test-verdict.json`` claim makes the tripwire assertion
    load-bearing: a gate that ran test_command (on this green fixed repo) would
    record ``test_claim_mismatch``, so its absence proves the fail-closed branch
    returned 0.0 without running the suite.
    """
    archive_root = tmp_path / "archive"
    _stage_run(archive_root, rundir_golden)
    task = _task(corpus_mini_dir, fixture_manifest_path)
    repo = _stage_repo(tmp_path / "repo", task.data.head_sha, edit=_CALC_FIXED)
    # The uncommitted calc.py fix makes _fixes_applied return True via the
    # dirty-tree check (no head_sha resolution); the gate then diff's an
    # unresolvable baked SHA -> exit 128 -> fail closed.
    task.data = task.data.model_copy(update={"head_sha": "0" * 40})
    trace = _trace(task, archive_root=archive_root, repo_path=repo)

    await task.score(trace, runtime)

    assert trace.metrics["fixes_applied"] == 1.0
    assert trace.metrics["test_oracle_unchanged"] == 0.0
    assert trace.rewards["fix_tests_pass"] == 0.0
    assert "test_claim_mismatch" not in trace.metrics


@pytest.mark.parametrize(
    "flag",
    ["--skip-worktree", "--assume-unchanged"],
    ids=["skip-worktree", "assume-unchanged"],
)
async def test_oracle_gate_rejects_flag_tampered_tracked_file(
    flag: str,
    tmp_path: Path,
    runtime,
    rundir_golden: Path,
    corpus_mini_dir: Path,
    fixture_manifest_path: Path,
) -> None:
    """A skip-worktree/assume-unchanged flag hides tamper from the diff probe.

    git reads a file carrying either flag from the INDEX, not the worktree, so a
    tampered tracked protected file reads as clean to ``git diff`` (exit 0). The
    flag probe (``git ls-files -v``) must fail closed: setting the flag on any
    protected file is itself tampering with the oracle's verifiability.
    """
    archive_root = tmp_path / "archive"
    _stage_run(archive_root, rundir_golden)
    task = _task(corpus_mini_dir, fixture_manifest_path)
    # A real fix (calc.py) plus a flagged, gutted tracked test: diff is fooled,
    # so only the flag probe stands between this and a free w_tests.
    repo = _stage_repo(tmp_path / "repo", task.data.head_sha, edit=_CALC_FIXED)
    (repo / "tests/test_calc.py").write_text(_TAMPER_PASSING, encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(repo), "update-index", flag, "tests/test_calc.py"],
        check=True,
    )
    trace = _trace(task, archive_root=archive_root, repo_path=repo)

    await task.score(trace, runtime)

    assert trace.metrics["fixes_applied"] == 1.0
    assert trace.metrics["test_oracle_unchanged"] == 0.0
    assert trace.rewards["fix_tests_pass"] == 0.0
    assert "test_claim_mismatch" not in trace.metrics


async def test_oracle_gate_rejects_tracked_gitignore_edit(
    tmp_path: Path,
    runtime,
    rundir_golden: Path,
    corpus_mini_dir: Path,
    fixture_manifest_path: Path,
) -> None:
    """A tracked .gitignore edit masking a new oracle file must fail closed.

    ``git ls-files --exclude-standard`` honors the repo's own ignore rules, so a
    rollout that edits the tracked ``.gitignore`` to ignore a new untracked
    tests/pytest.ini hides that file from the untracked probe. The diff probe
    covers those ignore files (``:(glob)**/.gitignore``), so the edit itself
    must read as an oracle change.
    """
    archive_root = tmp_path / "archive"
    _stage_run(archive_root, rundir_golden)
    task = _task(corpus_mini_dir, fixture_manifest_path)
    repo = _stage_repo(tmp_path / "repo", task.data.head_sha, edit=_CALC_FIXED)
    gitignore = repo / ".gitignore"
    gitignore.write_text(gitignore.read_text(encoding="utf-8") + "tests/pytest.ini\n", encoding="utf-8")
    (repo / "tests/pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    trace = _trace(task, archive_root=archive_root, repo_path=repo)

    await task.score(trace, runtime)

    assert trace.metrics["fixes_applied"] == 1.0
    assert trace.metrics["test_oracle_unchanged"] == 0.0
    assert trace.rewards["fix_tests_pass"] == 0.0
    assert "test_claim_mismatch" not in trace.metrics


async def test_oracle_gate_rejects_info_exclude_rule(
    tmp_path: Path,
    runtime,
    rundir_golden: Path,
    corpus_mini_dir: Path,
    fixture_manifest_path: Path,
) -> None:
    """A rule written to .git/info/exclude must fail closed.

    ``--exclude-standard`` reads the rollout-controlled ``.git/info/exclude``,
    so a rule written there can mask an untracked oracle file with zero tracked
    changes. A fresh clone's file is comments-only; any real rule means the
    oracle changed.
    """
    archive_root = tmp_path / "archive"
    _stage_run(archive_root, rundir_golden)
    task = _task(corpus_mini_dir, fixture_manifest_path)
    repo = _stage_repo(tmp_path / "repo", task.data.head_sha, edit=_CALC_FIXED)
    info_exclude = repo / ".git/info/exclude"
    info_exclude.write_text(info_exclude.read_text(encoding="utf-8") + "tests/conftest.py\n", encoding="utf-8")
    (repo / "tests/conftest.py").write_text("", encoding="utf-8")
    trace = _trace(task, archive_root=archive_root, repo_path=repo)

    await task.score(trace, runtime)

    assert trace.metrics["fixes_applied"] == 1.0
    assert trace.metrics["test_oracle_unchanged"] == 0.0
    assert trace.rewards["fix_tests_pass"] == 0.0
    assert "test_claim_mismatch" not in trace.metrics


async def test_oracle_gate_rejects_untracked_hidden_by_core_excludesfile(
    tmp_path: Path,
    runtime,
    rundir_golden: Path,
    corpus_mini_dir: Path,
    fixture_manifest_path: Path,
) -> None:
    """A core.excludesFile rule masking an untracked oracle file must fail closed.

    ``git ls-files --exclude-standard`` honors the repo-local ``core.excludesFile``
    set in the untracked ``.git/config`` — a file the gate never probes — so a
    rollout could otherwise hide an untracked tests/pytest.ini behind it and pass
    every probe. The untracked probe runs with ``-c core.excludesFile=``, which
    also neutralizes the global excludes file (``$HOME/.config/git/ignore``), so
    the file is listed and the oracle reads as changed.
    """
    archive_root = tmp_path / "archive"
    _stage_run(archive_root, rundir_golden)
    task = _task(corpus_mini_dir, fixture_manifest_path)
    repo = _stage_repo(tmp_path / "repo", task.data.head_sha, edit=_CALC_FIXED)
    ignores = repo.parent / "excludes"
    ignores.write_text("tests/pytest.ini\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(repo), "config", "core.excludesFile", str(ignores)],
        check=True,
    )
    (repo / "tests/pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    trace = _trace(task, archive_root=archive_root, repo_path=repo)

    await task.score(trace, runtime)

    assert trace.metrics["fixes_applied"] == 1.0
    assert trace.metrics["test_oracle_unchanged"] == 0.0
    assert trace.rewards["fix_tests_pass"] == 0.0
    assert "test_claim_mismatch" not in trace.metrics


async def test_oracle_gate_green_despite_suite_bytecode_artifacts(
    tmp_path: Path,
    runtime,
    rundir_golden: Path,
    corpus_mini_dir: Path,
    fixture_manifest_path: Path,
) -> None:
    """A green suite's own bytecode under protected paths must not trip the gate.

    The untracked probe lists every untracked file under a protected path (no
    ``--exclude-standard``), so the ``__pycache__/`` and ``*.py[cod]`` files a
    legitimate test run drops while importing ``tests/`` would otherwise read as
    an oracle change and withhold ``w_tests`` from a genuinely fixed tree. Those
    artifacts are excluded explicitly via ``ORACLE_BENIGN_PATHSPECS`` — never
    loaded by the runner, so excluding them cannot hide a real oracle file.
    """
    archive_root = tmp_path / "archive"
    _stage_run(archive_root, rundir_golden)
    task = _task(corpus_mini_dir, fixture_manifest_path)
    repo = _stage_repo(tmp_path / "repo", task.data.head_sha, edit=_CALC_FIXED)
    pycache = repo / "tests" / "__pycache__"
    pycache.mkdir()
    (pycache / "test_calc.cpython-312.pyc").write_bytes(b"x")
    (repo / "tests" / "test_calc.pyc").write_bytes(b"x")
    trace = _trace(task, archive_root=archive_root, repo_path=repo)

    await task.score(trace, runtime)

    assert trace.metrics["fixes_applied"] == 1.0
    assert trace.metrics["test_oracle_unchanged"] == 1.0
    assert trace.rewards["fix_tests_pass"] == 1.0


async def test_oracle_gate_rejects_root_sitecustomize(
    tmp_path: Path,
    runtime,
    rundir_golden: Path,
    corpus_mini_dir: Path,
    fixture_manifest_path: Path,
) -> None:
    """An untracked root sitecustomize.py that exits 0 must fail closed.

    ``sitecustomize.py`` is imported from the repository root by every ``python``
    invocation ``test_command`` runs (cwd is on ``sys.path``), so one that calls
    ``sys.exit(0)`` makes a suite that never ran look green. It sits outside the
    declared protected paths, so the untracked probe must cover it explicitly.
    """
    archive_root = tmp_path / "archive"
    _stage_run(archive_root, rundir_golden)
    task = _task(corpus_mini_dir, fixture_manifest_path)
    repo = _stage_repo(tmp_path / "repo", task.data.head_sha, edit=_CALC_FIXED)
    (repo / "sitecustomize.py").write_text("import sys\nsys.exit(0)\n", encoding="utf-8")
    trace = _trace(task, archive_root=archive_root, repo_path=repo)

    await task.score(trace, runtime)

    assert trace.metrics["fixes_applied"] == 1.0
    assert trace.metrics["test_oracle_unchanged"] == 0.0
    assert trace.rewards["fix_tests_pass"] == 0.0
    assert "test_claim_mismatch" not in trace.metrics


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


async def test_score_reuses_one_archived_run_snapshot(
    tmp_path: Path,
    runtime,
    rundir_golden: Path,
    corpus_mini_dir: Path,
    fixture_manifest_path: Path,
    monkeypatch,
) -> None:
    """One score call fetches the archived run dir exactly once; all consumers share it.

    A read-once wrapper over the real subprocess runtime rejects any repeated
    ``runtime.read`` of an artifact path — the exact seam the refactor
    consolidates. After the single score-level fetch, the three consumers must
    read the shared host snapshot (via ``_read_json``), never re-enter the
    runtime; ``trace.state.run_dir`` must be cleared once scoring returns.
    """
    archive_root = tmp_path / "archive"
    _stage_run(archive_root, rundir_golden)
    task = _task(corpus_mini_dir, fixture_manifest_path)
    repo = _stage_repo(tmp_path / "repo", task.data.head_sha, patch=_REAL_PATCH)
    trace = _trace(task, archive_root=archive_root, repo_path=repo)

    class ReadOnce:
        def __init__(self, real: Callable[[str], Awaitable[bytes]]) -> None:
            self._real = real
            self._seen: set[str] = set()

        async def __call__(self, path: str) -> bytes:
            if path in self._seen:
                raise AssertionError(
                    f"archived artifact {path} read more than once in a single score call"
                )
            self._seen.add(path)
            return await self._real(path)

    real_read = runtime.read
    monkeypatch.setattr(runtime, "read", ReadOnce(real_read))

    await task.score(trace, runtime)

    expected = score_trajectory(
        assemble_scoring_inputs(rundir_golden, _manifest_row_like_production(rundir_golden))
    ).composite
    assert expected is not None
    assert trace.rewards["intrinsic_composite"] == expected
    assert trace.metrics["test_claim_passed_without_fix"] == 1.0
    assert trace.metrics["n_findings"] == 1.0
    assert trace.state.run_dir is None


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
    assert trace.metrics["test_oracle_unchanged"] == 1.0


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
