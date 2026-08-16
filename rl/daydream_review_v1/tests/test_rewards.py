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
import os
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
    DaydreamReviewData,
    DaydreamReviewState,
    DaydreamReviewTask,
    DaydreamReviewTaskset,
    _archive_root,
    _claimed_test_verdict,
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


def _assert_gate_held(trace: vf.Trace) -> None:
    """A test-oracle change must zero suite_non_regression, never pay a reward.

    The helper reconstructs the archived run dir and enforces that its
    ``deep/test-verdict.json`` claim is parseable before checking the tripwire.
    This makes absence of ``test_claim_mismatch`` prove the gate held rather
    than pass vacuously over an empty claim.

    NOTE: the ``all(...)`` precondition below is deliberately stricter than
    production's ``rundir._session_dir`` attribution rule, which returns
    ``None`` — attributing no claim, so the tripwire never fires — whenever
    more than one run dir exists under the archive root. This helper instead
    demands a parseable claim in EVERY run dir (test-side defense in depth), so
    a test that stages multiple run dirs must keep each one claim-bearing or
    the tripwire-absence assertion fails loudly by design.
    """
    # NOTE: production's tripwire (taskset.py) consumes the fetch_run_dir
    # staging copy filtered by RUN_DIR_FILES (rundir.py:32-40), not this host
    # archive read; keep deep/test-verdict.json on that allowlist or this
    # precondition can pass while the staged copy is claim-less.
    runs_root = Path(_archive_root(trace)) / "runs"
    assert runs_root.is_dir(), f"expected archive runs dir under {runs_root}"
    run_dirs = [entry for entry in runs_root.iterdir() if entry.is_dir()]
    assert run_dirs, f"expected at least one archived run dir under {runs_root}"
    assert all(
        _claimed_test_verdict(run_dir) is not None for run_dir in run_dirs
    ), (
        "missing or malformed deep/test-verdict.json claim; without it, the test_claim_mismatch-absence "
        "assertion would pass vacuously"
    )
    assert trace.metrics["fixes_applied"] == 1.0
    assert trace.metrics["test_oracle_unchanged"] == 0.0
    assert trace.metrics["suite_non_regression"] == 0.0
    assert "fix_tests_pass" not in trace.rewards
    assert "test_claim_mismatch" not in trace.metrics


def test_review_state_guard_rejects_base_state() -> None:
    """Scoring state must be a DaydreamReviewState, never the base State.

    Production traces carry a DaydreamReviewState automatically (the rollout
    resolves the task's StateT through the MRO); test helpers that score must
    pass one explicitly. A base State must fail loudly rather than silently
    dereference a missing run_dir.
    """
    data = DaydreamReviewData(
        idx=0,
        name="org/repo#1",
        prompt="Deep-review PR #1 of org/repo @ 111111111111",
        repo_slug="org/repo",
        clone_url="https://example.com/repo.git",
        pr_number=1,
        base_sha="0" * 40,
        head_sha="1" * 40,
        test_command="true",
        protected_test_paths=["tests/"],
    )
    base_trace = vf.Trace(task=vf.TraceTask(type=DaydreamReviewTask.__name__, data=data))
    with pytest.raises(TypeError):
        _review_state(base_trace)
    good_trace = vf.Trace(
        task=vf.TraceTask(type=DaydreamReviewTask.__name__, data=data),
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


def _assert_checkout_pinned_at(
    verify_dir: Path,
    head_sha: str,
    *,
    exists_msg: str = "the checkout must exist",
    pinned_msg: str = "the checkout must be pinned at the baked head",
    clean_msg: str = "the checkout tree must equal the baked head",
) -> None:
    """Assert the verify checkout is a real repo pinned at *head_sha* whose tree
    is identical to it (no partial candidate diff applied).

    Shared by the failed-diff and empty-diff verify-checkout tests: both must
    prove clone + checkout --detach ran and the candidate diff was either never
    applied (failed diff) or applied as a genuine no-op (empty diff), leaving
    the tree byte-identical to the baked head.
    """
    assert verify_dir.exists(), exists_msg
    head = subprocess.run(
        ["git", "-C", str(verify_dir), "rev-parse", "HEAD"],
        capture_output=True, check=True,
    ).stdout.decode().strip()
    assert head == head_sha, pinned_msg
    clean = subprocess.run(
        ["git", "-C", str(verify_dir), "diff", "--quiet", "HEAD", "--"],
        capture_output=True,
    )
    assert clean.returncode == 0, clean_msg


_REAL_PATCH = "diff --git a/tests/test_calc.py b/tests/test_calc.py\n@@ -1 +1 @@\n-old\n+new\n"

_CALC_FIXED = _CALC_BROKEN.replace("return a + b + 1", "return a + b")


def _seal_run(run_dir: Path, task: DaydreamReviewTask, repo_path: Path) -> Path:
    """Harness-side seal production over a staged committed-fix repo.

    Stages the fixture repo with the fix COMMITTED (the seal binds the committed
    diff the verifier re-derives at scoring time, never the working tree),
    hashes the archive members the harness recorded, and writes ``seal.json``
    into *run_dir*. Returns the staged repo path.
    """
    from daydream_review_v1.rundir import RUN_DIR_FILES
    from daydream_review_v1.verifier import seal_artifacts

    repo = _stage_repo(repo_path, task.data.head_sha, edit=_CALC_FIXED, commit=True)
    diff = subprocess.run(
        ["git", "-C", str(repo), "diff", task.data.head_sha, "HEAD"],
        capture_output=True,
        check=True,
    ).stdout
    present = [
        run_dir / rel for rel in RUN_DIR_FILES if (run_dir / rel).is_file()
    ] + sorted(run_dir.glob("deep/stack-*-records.json"))
    seal = seal_artifacts(present, candidate_diff=diff)
    (run_dir / "seal.json").write_text(seal.model_dump_json(), encoding="utf-8")
    return repo


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


async def test_green_suite_records_non_regression(
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
    assert trace.metrics["suite_non_regression"] == 1.0
    assert trace.metrics["test_oracle_unchanged"] == 1.0


async def test_verifier_identity_branch_executes_and_fails_closed(
    tmp_path: Path,
    runtime,
    corpus_mini_dir: Path,
    fixture_manifest_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The verifier identity branch is reachable and never falls back to the mutable tree.

    The image provisions the distinct non-root verifier identity
    (``base.Dockerfile``); the host subprocess smoke path does not, so force the
    identity probe to simulate the container. The real checkout construction
    then executes (git clone/detach/apply against the staged repo). On an
    unprivileged host the root-owned ``chown`` cannot succeed, so construction
    fails and the re-run must be an explicit zero — fail-closed, never a
    fallback to the agent-mutable tree, whose suite here is green (contrast
    ``test_green_suite_records_non_regression``, which scores 1.0 on exactly
    the mutable-tree fallback shape). On a root host with the verifier identity
    (the production container shape) the ``chown`` succeeds and the suite
    re-runs green under the verifier identity instead; the expected reading
    follows the host shape either way.
    """
    from daydream_review_v1 import taskset

    archive_root = tmp_path / "archive"
    (archive_root / "runs").mkdir(parents=True)
    task = _task(corpus_mini_dir, fixture_manifest_path)
    # The fix is COMMITTED: the verifier checkout's candidate diff is
    # ``git diff <head_sha> HEAD``, i.e. the committed contents, never the
    # working tree.
    repo = _stage_repo(tmp_path / "repo", task.data.head_sha, edit=_CALC_FIXED, commit=True)
    trace = _trace(task, archive_root=archive_root, repo_path=repo)

    # Keep the real host probe (setpriv + verifier user) before it is replaced:
    # the assertion below must know whether the host can actually complete the
    # verifier re-run, not just whether the branch was forced on.
    real_identity_available = taskset._verifier_identity_available

    async def _identity_available(_runtime: vf.Runtime) -> bool:
        return True  # container shape: setpriv + verifier user provisioned

    monkeypatch.setattr(taskset, "_verifier_identity_available", _identity_available)

    await task.score(trace, runtime)

    # The oracle gate passed: the reading below comes from the verifier branch,
    # not from a changed test oracle.
    assert trace.metrics["test_oracle_unchanged"] == 1.0
    # The expected suite reading follows the host shape: the root-owned chown
    # only fails on an unprivileged host, so there construction fails closed to
    # an explicit zero; on a root host with the verifier identity the checkout
    # is built and the suite re-runs green under the verifier identity. Both
    # shapes prove the re-run never fell back to the agent-mutable tree.
    verifier_rerun_succeeds = os.geteuid() == 0 and await real_identity_available(runtime)
    assert trace.metrics["suite_non_regression"] == (1.0 if verifier_rerun_succeeds else 0.0)
    # _prepare_verify_checkout really executed its git steps: the verify clone
    # exists and carries the applied candidate diff (it is left on disk by the
    # failed construction rather than silently skipped).
    verify_dir = tmp_path / "repo-verify"
    assert (verify_dir / "calc.py").read_text(encoding="utf-8") == _CALC_FIXED


async def test_verify_checkout_derives_diff_from_shared_helper_with_empty_guard(
    corpus_mini_dir, fixture_manifest_path,
) -> None:
    """_prepare_verify_checkout must derive its candidate diff through
    rundir.candidate_diff_cmd (the single source) and apply it behind an
    empty-guard, so an empty diff is a clean no-op and a failed diff never
    pipes raw/partial output into git apply.
    """
    import shlex

    from conftest import FakeRuntime

    from daydream_review_v1 import taskset
    from daydream_review_v1.rundir import candidate_diff_cmd

    rt = FakeRuntime(exit_code=0)
    repo, head_sha = "/work/repo", "deadbeef"
    await taskset._prepare_verify_checkout(rt, repo, head_sha)

    script = rt.commands[0][2]  # the single sh -c script
    # (a) single derivation site: the script embeds the helper's argv
    assert shlex.join(candidate_diff_cmd(repo, head_sha)) in script
    # (b) empty-guard: the apply is skipped when the diff is empty
    assert "[ ! -s " in script and "apply" in script


async def test_green_unrelated_edit_gets_no_suite_reward(
    tmp_path: Path, runtime, corpus_mini_dir: Path, fixture_manifest_path: Path,
) -> None:
    """Starting green and making an unrelated edit earns no suite credit."""
    archive_root = tmp_path / "archive"
    (archive_root / "runs").mkdir(parents=True)
    task = _task(corpus_mini_dir, fixture_manifest_path)
    # The baked head is already green; the agent only touches README (unrelated)
    # — the suite stays green.
    repo = _stage_repo(tmp_path / "repo", task.data.head_sha, patch=_REAL_PATCH)
    (repo / "README.md").write_text("# changed\n", encoding="utf-8")
    trace = _trace(task, archive_root=archive_root, repo_path=repo)

    await task.score(trace, runtime)

    # Suite green is telemetry, never a reward axis.
    assert set(trace.rewards) == {"intrinsic_composite"}
    assert trace.metrics["fixes_applied"] == 1.0
    assert trace.metrics["test_oracle_unchanged"] == 1.0
    assert trace.metrics["suite_non_regression"] == 1.0


async def test_red_suite_records_no_non_regression(
    tmp_path: Path, runtime, corpus_mini_dir: Path, fixture_manifest_path: Path
) -> None:
    archive_root = tmp_path / "archive"
    (archive_root / "runs").mkdir(parents=True)
    task = _task(corpus_mini_dir, fixture_manifest_path)
    repo = _stage_repo(tmp_path / "repo", task.data.head_sha, edit=_CALC_BROKEN, patch=_REAL_PATCH)
    trace = _trace(task, archive_root=archive_root, repo_path=repo)

    await task.score(trace, runtime)

    assert trace.metrics["fixes_applied"] == 1.0
    assert trace.metrics["suite_non_regression"] == 0.0


async def test_suite_result_is_telemetry_not_reward(
    tmp_path: Path, runtime, corpus_mini_dir: Path, fixture_manifest_path: Path,
) -> None:
    """A green suite no longer sums into the reward: only intrinsic_composite remains."""
    archive_root = tmp_path / "archive"
    (archive_root / "runs").mkdir(parents=True)
    task = _task(corpus_mini_dir, fixture_manifest_path)
    repo = _stage_repo(tmp_path / "repo", task.data.head_sha, edit=_CALC_FIXED, patch=_REAL_PATCH)
    trace = _trace(task, archive_root=archive_root, repo_path=repo)

    await task.score(trace, runtime)

    assert set(trace.rewards) == {"intrinsic_composite"}
    assert "fix_tests_pass" not in trace.rewards
    assert trace.metrics["fixes_applied"] == 1.0
    assert trace.metrics["test_oracle_unchanged"] == 1.0
    assert trace.metrics["suite_non_regression"] == 1.0


async def test_tampered_suite_never_records_honest_non_regression(
    tmp_path: Path, runtime, rundir_golden: Path, corpus_mini_dir: Path, fixture_manifest_path: Path,
) -> None:
    """A gutted test oracle records suite_non_regression 0.0 and no suite reward."""
    archive_root = tmp_path / "archive"
    _stage_run(archive_root, rundir_golden)
    task = _task(corpus_mini_dir, fixture_manifest_path)
    repo = _stage_repo(tmp_path / "repo", task.data.head_sha, edit=_CALC_FIXED)
    (repo / "tests/test_calc.py").write_text(_TAMPER_PASSING, encoding="utf-8")
    trace = _trace(task, archive_root=archive_root, repo_path=repo)

    await task.score(trace, runtime)

    assert "fix_tests_pass" not in trace.rewards
    assert trace.metrics["test_oracle_unchanged"] == 0.0
    assert trace.metrics["suite_non_regression"] == 0.0


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
async def test_suite_rejects_protected_test_path_changes(
    edit: str | None,
    tamper_rel: str,
    tamper_content: str,
    tmp_path: Path,
    runtime,
    rundir_golden: Path,
    corpus_mini_dir: Path,
    fixture_manifest_path: Path,
) -> None:
    """A changed test oracle records suite_non_regression 0.0, never an honest reading.

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

    _assert_gate_held(trace)


async def _score_fail_closed(
    tmp_path: Path,
    runtime,
    archive_root: Path,
    rundir_golden: Path,
    corpus_mini_dir: Path,
    fixture_manifest_path: Path,
    *,
    unlink_claim: bool = False,
) -> vf.Trace:
    """Stage a single run dir and score it down the fail-closed git-error path.

    Shared skeleton of the oracle-gate tests: stage -> task -> dirty-fixed repo
    -> unresolvable baked SHA override -> score. ``unlink_claim`` deletes the
    archived ``deep/test-verdict.json`` before scoring, staging the negative
    control: production records no ``test_claim_mismatch`` (behavior unchanged)
    while ``_assert_gate_held``'s claim precondition must fail loudly.
    """
    run_dir = _stage_run(archive_root, rundir_golden)
    if unlink_claim:
        (run_dir / "deep" / "test-verdict.json").unlink()
    task = _task(corpus_mini_dir, fixture_manifest_path)
    repo = _stage_repo(tmp_path / "repo", task.data.head_sha, edit=_CALC_FIXED)
    # The uncommitted calc.py fix makes _fixes_applied return True via the
    # dirty-tree check (no head_sha resolution); the gate then diff's an
    # unresolvable baked SHA -> exit 128 -> fail closed.
    task.data = task.data.model_copy(update={"head_sha": "0" * 40})
    trace = _trace(task, archive_root=archive_root, repo_path=repo)

    await task.score(trace, runtime)

    return trace


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
    trace = await _score_fail_closed(
        tmp_path, runtime, archive_root, rundir_golden, corpus_mini_dir, fixture_manifest_path
    )

    _assert_gate_held(trace)


async def test_assert_gate_held_raises_when_claim_absent(
    tmp_path: Path,
    runtime,
    rundir_golden: Path,
    corpus_mini_dir: Path,
    fixture_manifest_path: Path,
) -> None:
    """A claim-less staged run must not let the gate oracle pass vacuously.

    Production behavior is unchanged: no claim -> no test_claim_mismatch recorded.
    But _assert_gate_held must now fail loudly on that shape, because without
    the claim its test_claim_mismatch-absence assertion proves nothing.
    """
    archive_root = tmp_path / "archive"
    trace = await _score_fail_closed(
        tmp_path,
        runtime,
        archive_root,
        rundir_golden,
        corpus_mini_dir,
        fixture_manifest_path,
        unlink_claim=True,
    )

    assert "test_claim_mismatch" not in trace.metrics
    with pytest.raises(AssertionError):
        _assert_gate_held(trace)


async def test_gate_held_raises_when_a_second_run_dir_is_claim_less(
    tmp_path: Path,
    runtime,
    rundir_golden: Path,
    corpus_mini_dir: Path,
    fixture_manifest_path: Path,
) -> None:
    """The claim precondition covers EVERY archived run dir, not just one.

    Production's ``rundir._session_dir`` attributes nothing when more than one
    run dir exists (the archive is not this rollout's alone), so the tripwire
    never fires in that shape — ``_assert_gate_held``'s ``all(...)`` claim
    precondition is what stops a claim-less second run dir from passing
    vacuously. A two-dir archive with one claim deleted must fail loudly.
    """
    archive_root = tmp_path / "archive"
    _stage_run(archive_root, rundir_golden, session_id=SESSION_ID)
    second = _stage_run(archive_root, rundir_golden, session_id="session-second")
    (second / "deep" / "test-verdict.json").unlink()
    task = _task(corpus_mini_dir, fixture_manifest_path)
    repo = _stage_repo(tmp_path / "repo", task.data.head_sha, edit=_CALC_FIXED)
    trace = _trace(task, archive_root=archive_root, repo_path=repo)

    await task.score(trace, runtime)

    # Two run dirs -> _session_dir attributes nothing -> the tripwire never
    # fires even though the gate is intact here; only the helper's claim
    # precondition stands between this shape and a vacuous pass.
    assert "test_claim_mismatch" not in trace.metrics
    with pytest.raises(AssertionError, match="deep/test-verdict.json claim"):
        _assert_gate_held(trace)


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
    # so only the flag probe stands between this and a free green reading.
    repo = _stage_repo(tmp_path / "repo", task.data.head_sha, edit=_CALC_FIXED)
    (repo / "tests/test_calc.py").write_text(_TAMPER_PASSING, encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(repo), "update-index", flag, "tests/test_calc.py"],
        check=True,
    )
    trace = _trace(task, archive_root=archive_root, repo_path=repo)

    await task.score(trace, runtime)

    _assert_gate_held(trace)


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

    _assert_gate_held(trace)


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

    _assert_gate_held(trace)


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

    _assert_gate_held(trace)


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
    an oracle change and withhold a green non-regression reading from a genuinely fixed tree. Those
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
    assert trace.metrics["suite_non_regression"] == 1.0


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

    _assert_gate_held(trace)


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
async def test_no_fixes_records_no_non_regression(
    patch: str | None, commit: bool, tmp_path: Path, runtime, corpus_mini_dir: Path, fixture_manifest_path: Path
) -> None:
    """An untouched tree records suite_non_regression 0.0 however recommended.patch looks.

    The third case is the one that matters. daydream writes `.daydream/` INTO the
    repository under review, and `capture_recommended_patch` appends a creation
    hunk for every untracked non-ignored file — so on any repository that does not
    gitignore that directory (i.e. every real repository), the patch is non-empty
    after a rollout that changed nothing. Reading it as "a fix landed" would hand
    out a free green non-regression reading off the still-green baseline, for free, forever.
    """
    archive_root = tmp_path / "archive"
    (archive_root / "runs").mkdir(parents=True)
    task = _task(corpus_mini_dir, fixture_manifest_path)
    repo = _stage_repo(tmp_path / "repo", task.data.head_sha, patch=patch, commit=commit)
    trace = _trace(task, archive_root=archive_root, repo_path=repo)

    await task.score(trace, runtime)

    assert trace.metrics["fixes_applied"] == 0.0
    assert trace.metrics["suite_non_regression"] == 0.0
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
    be scored as a fix: every non-1 exit records suite_non_regression 0.0, honoring the
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
    assert trace.metrics["suite_non_regression"] == 0.0


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
    assert trace.metrics["suite_non_regression"] == 0.0
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
    assert trace.metrics["suite_non_regression"] == (0.0 if red else 1.0)
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
    assert trace.metrics["suite_non_regression"] == 1.0
    assert trace.metrics["test_oracle_unchanged"] == 1.0


async def test_committed_daydream_artifacts_not_a_fix(
    tmp_path: Path, runtime, corpus_mini_dir: Path, fixture_manifest_path: Path
) -> None:
    """Committing daydream's own .daydream/ artifacts must not read as a fix.

    The pathspec exclusion owns the commit path. Real targets don't gitignore
    ``.daydream/`` (only the fixture does), so a no-fix rollout whose agent
    commits daydream's own untracked artifacts produces a tree that differs from
    the baked snapshot — which ``git diff --quiet <head_sha> HEAD`` would read as
    a fix and pay a free green non-regression reading off a still-green baseline. Committing the
    artifacts (force-added, since the fixture ignores them) must record
    ``suite_non_regression`` 0.0.
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
    assert trace.metrics["suite_non_regression"] == 0.0
    assert "test_claim_mismatch" not in trace.metrics


async def test_unresolvable_snapshot_sha_reads_as_no_fix(
    tmp_path: Path, runtime, corpus_mini_dir: Path, fixture_manifest_path: Path
) -> None:
    """A fix signal that cannot be evaluated reads as no-fix, not a free win.

    ``suite_non_regression`` stays deliberately false-negative biased: any ``git diff
    --quiet`` exit other than 1 (0 = identical trees, 128 = unresolvable baked
    SHA, 127 = missing sh/git) means "no fix found". Here the baked snapshot SHA
    is not present in the repository at all, so the diff exits 128 — the reward
    must record ``suite_non_regression`` 0.0, never a free green reading for nothing.
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
    assert trace.metrics["suite_non_regression"] == 0.0


async def test_reward_version_is_pinned(
    tmp_path: Path, runtime, rundir_golden: Path, corpus_mini_dir: Path, fixture_manifest_path: Path
) -> None:
    """AC-3: pin the scorer version the parity test cannot see move — both stamps.

    `test_intrinsic_composite_parity` runs the same `score_trajectory` on both
    sides, so a REWARD_VERSION bump moves expectation and actual together and the
    parity assertion stays green through a semantic change. This is the assertion
    that actually fails when the offline scorer changes under us. Since the
    demotion, the aggregate rollout contract carries its own version: the
    breakdown stamps both the rollout boundary (``reward_version``) and the
    intrinsic scorer it was evaluated against (``intrinsic_reward_version``).
    """
    from daydream.training.reward import REWARD_VERSION

    from daydream_review_v1.taskset import ROLLOUT_REWARD_VERSION

    assert REWARD_VERSION == "2026.05.28-2", (
        f"the training pipeline's reward version moved to {REWARD_VERSION!r}. Re-derive the "
        "rollout reward's expected values before trusting any run scored across the boundary."
    )
    assert ROLLOUT_REWARD_VERSION == "2026.08.15-1", (
        f"the rollout reward contract version moved to {ROLLOUT_REWARD_VERSION!r}"
    )

    archive_root = tmp_path / "archive"
    _stage_run(archive_root, rundir_golden)
    task = _task(corpus_mini_dir, fixture_manifest_path)
    trace = _trace(task, archive_root=archive_root, repo_path=tmp_path / "repo")

    await task.score(trace, runtime)

    breakdown = trace.info["reward_breakdown"]
    assert breakdown["reward_version"] == ROLLOUT_REWARD_VERSION
    assert breakdown["intrinsic_reward_version"] == REWARD_VERSION


async def test_tampered_sealed_artifact_zeroes_intrinsic_and_non_regression(
    tmp_path, runtime, rundir_golden, corpus_mini_dir, fixture_manifest_path,
) -> None:
    """A tampered sealed artifact makes the only reward zero and non-regression dishonest."""
    archive_root = tmp_path / "archive"
    run_dir = _stage_run(archive_root, rundir_golden)
    task = _task(corpus_mini_dir, fixture_manifest_path)
    # Harness produced the seal; the agent then rewrote an archived artifact.
    repo = _seal_run(run_dir, task, tmp_path / "repo")
    (run_dir / "deep" / "merged-items.json").write_text(
        json.dumps({"items": []}), encoding="utf-8"
    )  # tamper after sealing

    trace = _trace(task, archive_root=archive_root, repo_path=repo)

    await task.score(trace, runtime)

    assert trace.metrics["seal_verified"] == 0.0
    assert trace.rewards["intrinsic_composite"] == 0.0
    assert trace.metrics["suite_non_regression"] == 0.0


async def test_untampered_sealed_run_scores_normally(
    tmp_path, runtime, rundir_golden, corpus_mini_dir, fixture_manifest_path,
) -> None:
    """An intact seal leaves scoring unchanged."""
    archive_root = tmp_path / "archive"
    run_dir = _stage_run(archive_root, rundir_golden)
    task = _task(corpus_mini_dir, fixture_manifest_path)
    repo = _seal_run(run_dir, task, tmp_path / "repo")

    trace = _trace(task, archive_root=archive_root, repo_path=repo)

    await task.score(trace, runtime)

    assert trace.metrics["seal_verified"] == 1.0
    assert trace.rewards["intrinsic_composite"] == (
        score_trajectory(
            assemble_scoring_inputs(rundir_golden, _manifest_row_like_production(rundir_golden))
        ).composite
    )


async def test_seal_detects_committed_diff_changed_after_sealing(
    tmp_path, runtime, rundir_golden, corpus_mini_dir, fixture_manifest_path,
) -> None:
    """The seal binds the diff applied at scoring time, not a self-consistent record.

    verify_seal re-derives the candidate diff from the live repo, so a commit
    made after sealing (a tampered committed diff — exactly what the verifier
    checkout would apply) fails verification. The old tautology — re-hashing
    the seal's own embedded copy of the diff — could never fail on this.
    """
    archive_root = tmp_path / "archive"
    run_dir = _stage_run(archive_root, rundir_golden)
    task = _task(corpus_mini_dir, fixture_manifest_path)
    repo = _seal_run(run_dir, task, tmp_path / "repo")

    # The committed diff changes after sealing: a second commit past the fix.
    (repo / "README.md").write_text("# changed\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "--quiet", "-m", "tamper"], check=True)

    trace = _trace(task, archive_root=archive_root, repo_path=repo)

    await task.score(trace, runtime)

    assert trace.metrics["seal_verified"] == 0.0
    assert trace.rewards["intrinsic_composite"] == 0.0


async def test_vanished_seal_on_a_harness_sealed_run_is_a_tamper(
    tmp_path, runtime, rundir_golden, corpus_mini_dir, fixture_manifest_path,
) -> None:
    """A harness-sealed run whose seal vanished is a tamper, never legacy unsealed.

    verify_seal's ``None`` is the legacy path only when no seal was expected:
    the harness recorded ``daydream_seal_ok=True``, so a missing ``seal.json``
    at scoring time is an internal contradiction and must fail closed
    (``seal_verified`` 0.0, zero intrinsic, no honest non-regression) instead
    of scoring the run at full trust.
    """
    archive_root = tmp_path / "archive"
    _stage_run(archive_root, rundir_golden)  # staged copy carries no seal.json
    task = _task(corpus_mini_dir, fixture_manifest_path)
    repo = _stage_repo(tmp_path / "repo", task.data.head_sha, edit=_CALC_FIXED, commit=True)
    trace = _trace(task, archive_root=archive_root, repo_path=repo)
    trace.info["daydream_seal_ok"] = True  # the harness claims it sealed the run

    await task.score(trace, runtime)

    assert trace.metrics["seal_verified"] == 0.0
    assert trace.rewards["intrinsic_composite"] == 0.0
    assert trace.metrics["suite_non_regression"] == 0.0


async def test_git_failure_at_verify_time_fails_closed(
    tmp_path, runtime, rundir_golden, corpus_mini_dir, fixture_manifest_path,
) -> None:
    """A scoring-time diff re-derivation failure must not hash as the empty diff.

    seal_archived_run records ``b""`` when git fails at seal time; if
    verify_seal mapped a scoring-time git failure to ``b""`` too, a git failure
    at BOTH times would yield matching sha256(b"") digests and verify True
    (``seal_verified`` 1.0) on a run whose diff was never re-derived. A failed
    re-derivation must fail closed like any other unverifiable seal.
    """
    archive_root = tmp_path / "archive"
    run_dir = _stage_run(archive_root, rundir_golden)
    task = _task(corpus_mini_dir, fixture_manifest_path)
    from daydream_review_v1.rundir import RUN_DIR_FILES
    from daydream_review_v1.verifier import seal_artifacts

    present = [
        run_dir / rel for rel in RUN_DIR_FILES if (run_dir / rel).is_file()
    ] + sorted(run_dir.glob("deep/stack-*-records.json"))
    # A seal produced while git failed at seal time seals the empty diff.
    seal = seal_artifacts(present, candidate_diff=b"")
    (run_dir / "seal.json").write_text(seal.model_dump_json(), encoding="utf-8")

    # The repo under review is not a git repository: ``git diff`` fails at
    # scoring time with a non-zero exit, exactly the empty-diff collision.
    trace = _trace(task, archive_root=archive_root, repo_path=tmp_path / "not-a-repo")
    trace.info["daydream_seal_ok"] = True

    await task.score(trace, runtime)

    assert trace.metrics["seal_verified"] == 0.0
    assert trace.rewards["intrinsic_composite"] == 0.0
    assert trace.metrics["suite_non_regression"] == 0.0


async def test_verify_checkout_failed_diff_fails_closed(
    tmp_path, runtime, corpus_mini_dir, fixture_manifest_path,
) -> None:
    """A failed candidate-diff derivation must not pipe raw/partial output into
    git apply: _prepare_verify_checkout returns None, never a partially-built
    checkout. Mirrors the rundir fail-closed contract on the unified path.
    """
    from daydream_review_v1 import taskset

    task = _task(corpus_mini_dir, fixture_manifest_path)
    repo = _stage_repo(tmp_path / "repo", task.data.head_sha, edit=_CALC_FIXED, commit=True)
    # Make the diff-derivation step itself fail (not the clone): a diff.external
    # tool that cannot run makes ``git diff`` exit non-zero while clone and
    # checkout --detach both succeed.
    subprocess.run(
        ["git", "-C", str(repo), "config", "diff.external", "/nonexistent-diff-tool"],
        check=True,
    )
    result = await taskset._prepare_verify_checkout(runtime, str(repo), task.data.head_sha)
    assert result is None, "a failed diff derivation must not return a checkout"
    # The chain died at the diff step, after clone + checkout: the verify dir
    # exists and is pinned at the baked head (proving clone and checkout ran),
    # but the candidate diff was never applied -- the tree is still exactly
    # the baked head, not the fix.
    verify_dir = tmp_path / "repo-verify"
    _assert_checkout_pinned_at(
        verify_dir,
        task.data.head_sha,
        exists_msg="the chain must reach the diff step (clone + checkout ran)",
        pinned_msg="the chain must reach the diff step (checkout detached at the baked head)",
        clean_msg="a failed diff must never apply a partial candidate diff",
    )


async def test_verify_checkout_empty_diff_is_clean_noop(
    tmp_path, runtime, corpus_mini_dir, fixture_manifest_path,
) -> None:
    """A review-only rollout (no committed fix) has an empty candidate diff; it
    must apply cleanly as a no-op, never failing _prepare_verify_checkout.
    """
    from daydream_review_v1 import taskset

    task = _task(corpus_mini_dir, fixture_manifest_path)
    # --allow-empty commit: HEAD advances, tree identical -> genuinely empty diff
    repo = _stage_repo(tmp_path / "repo", task.data.head_sha, commit=True)
    result = await taskset._prepare_verify_checkout(runtime, str(repo), task.data.head_sha)
    if os.geteuid() == 0:
        assert result == str(tmp_path / "repo-verify"), "an empty diff must not fail construction"
    else:
        # non-root host: the trailing chown root:root fails, so construction
        # returns None regardless of the diff; pin the on-disk invariant that
        # the empty-guard made the apply a clean no-op: the checkout is a real
        # repo pinned at the baked head whose tree is identical to it.
        verify_dir = tmp_path / "repo-verify"
        _assert_checkout_pinned_at(
            verify_dir,
            task.data.head_sha,
            exists_msg="the empty diff must not abort the checkout build",
            clean_msg="the empty diff must apply as a clean no-op",
        )


async def test_verify_checkout_applies_exactly_the_candidate_diff(
    tmp_path, runtime, corpus_mini_dir, fixture_manifest_path,
) -> None:
    """The verify-checkout and the seal bind the same candidate diff.

    _prepare_verify_checkout applies the diff derived from rundir.candidate_diff_cmd
    onto the detached head; this asserts the applied result is exactly that
    diff (no drift between the two sites), as the verifier re-runs the suite
    against the same contract the seal binds.
    """
    from daydream_review_v1 import taskset
    from daydream_review_v1.rundir import candidate_diff_cmd

    task = _task(corpus_mini_dir, fixture_manifest_path)
    repo = _stage_repo(tmp_path / "repo", task.data.head_sha, edit=_CALC_FIXED, commit=True)
    head_sha = task.data.head_sha

    # The contract the seal binds: the shared helper's own output.
    expected = subprocess.run(
        candidate_diff_cmd(str(repo), head_sha), capture_output=True, check=True,
    ).stdout
    assert expected, "staged committed fix must yield a non-empty candidate diff"

    result = await taskset._prepare_verify_checkout(runtime, str(repo), head_sha)
    verify_dir = tmp_path / "repo-verify"
    # The checkout is built (root host) or at least its git steps ran (non-root
    # leaves it on disk before the trailing chown fails). The applied result
    # must equal the helper's derivation.
    assert verify_dir.is_dir(), "the verify checkout was not constructed"
    if os.geteuid() == 0:
        assert result == str(verify_dir), "construction must return the built checkout path"
    applied = subprocess.run(
        ["git", "-C", str(verify_dir), "diff", head_sha, "--", "calc.py"],
        capture_output=True, check=True,
    ).stdout
    # Drift guard: the tree after apply is head + exactly the candidate diff
    # the seal binds -- the checkout and the seal can never disagree.
    assert applied == expected, "the verify checkout drifted from the candidate diff the seal binds"
    assert (verify_dir / "calc.py").read_text(encoding="utf-8") == _CALC_FIXED, (
        "the verify checkout did not carry the candidate diff the seal binds"
    )
