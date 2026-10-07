"""Score real staged files through Task.score and SubprocessRuntime.

Assert trainer-visible rewards, metrics, and info after real shell/read operations.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
import verifiers.v1 as vf
from conftest import FakeRuntime, passed_gate_report
from daydream.atif import validate
from daydream.dataset_scoring import assemble_scoring_inputs
from daydream.training.reward import REWARD_VERSION, score_trajectory
from verifiers.v1.runtimes.subprocess import SubprocessRuntime

from daydream_review import rundir as rundir_mod, taskset
from daydream_review.fixture import build_fixture_repo
from daydream_review.rundir import DAYDREAM_EXCLUDE, RUN_DIR_FILES, candidate_diff_cmd
from daydream_review.taskset import (
    ROLLOUT_REWARD_VERSION,
    DaydreamReviewConfig,
    DaydreamReviewState,
    DaydreamReviewTask,
    DaydreamReviewTaskset,
    _archive_root,
    _claimed_test_verdict,
)
from daydream_review.verifier import seal_artifacts

MODEL = "some-org/some-policy-model"

SESSION_ID = "9b36227a-9f80-41e5-a419-5cfed5a34b5b"

_CALC_BROKEN = '''"""A deliberately wrong calc.py: `add` is off by one, so test_add fails."""


def add(a: int, b: int) -> int:
    return a + b + 1


def divide(a: int, b: int) -> float:
    return a / b


def mean(values: list[int]) -> float:
    return sum(values) / len(values)
'''


def _task(fixture_manifest_path: Path, *, pr_number: int = 1,) -> DaydreamReviewTask:
    # Provide a passed gate so the test reaches scoring.
    with passed_gate_report() as gate_path:
        taskset = DaydreamReviewTaskset(DaydreamReviewConfig(
                id="daydream-review", manifest_path=fixture_manifest_path, gate_report_path=gate_path, use_images=False,
            )
        )
        return next(task for task in taskset.load() if task.data.pr_number == pr_number)


def _trace(task: DaydreamReviewTask, *, archive_root: Path, repo_path: Path) -> vf.Trace:
    trace: vf.Trace = vf.Trace(
        task=vf.TraceTask(type=type(task).__name__, data=task.data), agent=vf.AgentInfo(model=MODEL),
        state=DaydreamReviewState(),
    )
    trace.info["daydream_archive_root"] = str(archive_root)
    trace.info["daydream_repo_path"] = str(repo_path)
    return trace


def _assert_gate_held(trace: vf.Trace) -> None:
    """Require zero suite telemetry and no claim mismatch after an oracle change.

    First require a parseable claim in every archived run so absence cannot pass
    vacuously. This is stricter than production attribution, which declines to
    choose a run when multiple directories exist.
    """
    # Production reads filtered staging. Keep the test-verdict allowlist so the host-claim probe
    # exercises the scoring path.
    runs_root = Path(_archive_root(trace)) / "runs"
    assert runs_root.is_dir(), f"expected archive runs dir under {runs_root}"
    run_dirs = [entry for entry in runs_root.iterdir() if entry.is_dir()]
    assert run_dirs, f"expected at least one archived run dir under {runs_root}"
    assert all(_claimed_test_verdict(run_dir) is not None for run_dir in run_dirs), (
        "missing or malformed deep/test-verdict.json claim; without it, the test_claim_mismatch-absence "
        "assertion would pass vacuously"
    )
    assert trace.metrics["fixes_applied"] == 1.0
    assert trace.metrics["test_oracle_unchanged"] == 0.0
    assert trace.metrics["suite_non_regression"] == 0.0
    assert "fix_tests_pass" not in trace.rewards
    assert "test_claim_mismatch" not in trace.metrics


async def test_review_state_guard_rejects_base_state(
    runtime: SubprocessRuntime, fixture_manifest_path: Path,
) -> None:
    task = _task(fixture_manifest_path)
    base_trace = vf.Trace(
        task=vf.TraceTask(type=DaydreamReviewTask.__name__, data=task.data), agent=vf.AgentInfo(model=MODEL),
    )
    with pytest.raises(TypeError, match="scoring state must be a DaydreamReviewState"):
        await task.score(base_trace, runtime)
    assert base_trace.rewards == {} and base_trace.metrics == {}

async def test_score_without_runtime_records_nothing(fixture_manifest_path: Path) -> None:
    """Offline replay with runtime=None leaves rewards/metrics empty, even with base State.

    It must bypass state validation, artifact fetching, and runtime-dependent scoring.
    """
    task = _task(fixture_manifest_path)
    trace = vf.Trace(task=vf.TraceTask(type=type(task).__name__, data=task.data), agent=vf.AgentInfo(model=MODEL))
    trace.info["daydream_archive_root"] = "/does/not/exist"
    trace.info["daydream_repo_path"] = "/does/not/exist"

    await task.score(trace)  # runtime defaults to None — the offline replay path

    assert trace.rewards == {}
    assert trace.metrics == {}


def _stage_run(archive_root: Path, source: Path, *, session_id: str = SESSION_ID) -> Path:
    dest = archive_root / "runs" / session_id
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, dest)
    return dest


def _golden_task(tmp_path: Path, fixture_manifest_path: Path, rundir_golden: Path) -> tuple[Path, DaydreamReviewTask]:
    archive_root = tmp_path / "archive"
    _stage_run(archive_root, rundir_golden)
    return archive_root, _task(fixture_manifest_path)

# This first-line guard matches Unix paths at JSON string boundaries. Escaped embedded newlines can
# hide a leading slash, so it is not a proof that every path is safe.
_ABS_PATH_RE = re.compile(r'(?:^|[\s"\'])/[A-Za-z]')


def _walk_json_keys(node: object, targets: set[str], prefix: str = "") -> list[str]:
    """Return matching JSON key paths, including list indices, for precise fixture failure locations."""
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
    """Reject dangling transcript refs, machine paths, and embedded operational prompts.

    The retained root trajectory is test data; fetch_run_dir exclusion remains the
    boundary preventing model-context ingestion.
    """
    trajectories = rundir_golden / "trajectories"
    assert not trajectories.is_dir() or not next(trajectories.iterdir(), None)

    trajectory = json.loads((rundir_golden / "trajectory.json").read_text(encoding="utf-8"))
    manifest = json.loads((rundir_golden / "manifest.json").read_text(encoding="utf-8"))

    dangling = _walk_json_keys(trajectory, {"trajectory_path", "sibling_trajectory_ref"})
    assert not dangling, f"dangling per-fork trajectory refs: {dangling}"

    evaluation = json.loads((rundir_golden / "evaluation.json").read_text(encoding="utf-8"))
    for name, blob in (("trajectory.json", trajectory), ("manifest.json", manifest), ("evaluation.json", evaluation)):
        assert not _ABS_PATH_RE.search(json.dumps(blob)), (f"{name} carries a machine-specific absolute path"
        )

    # ATIF validation catches dangling source_call_id references after fixture pruning.
    assert validate(trajectory) is True, ("trajectory.json fails daydream.atif.validate (dangling tool-call refs)"
    )

    for step in trajectory["steps"]:
        rc = step.get("reasoning_content")
        assert not rc, f"step {step.get('step_id')} carries reasoning_content prompt text"
        tcs = step.get("tool_calls")
        assert not tcs, f"step {step.get('step_id')} carries tool_calls prompt text"


def _stage_repo(
    repo_path: Path, head_sha: str, *, edit: str | None = None, patch: str | None = None, commit: bool = False,
    commit_patch: bool = False,
) -> Path:
    """Stage the image-style detached fixture; optionally edit, plant a patch, or commit.

    A planted patch without edits models artifact contamination. commit alone makes
    an empty commit; commit_patch also force-adds .daydream to model real targets
    that do not ignore generated artifacts. Ordinary commits stage only real edits.
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
            # Force-add the ignored .daydream fixture to simulate a target that tracks it.
            subprocess.run(["git", "-C", str(repo_path), "add", "-f", ".daydream"], check=True)
        subprocess.run(["git", "-C", str(repo_path), "commit", "--quiet", "--allow-empty", "-m", "fix"], check=True)
    return repo_path


def _assert_checkout_pinned_at(verify_dir: Path, head_sha: str, *, exists_msg: str = "the checkout must exist",
    pinned_msg: str = "the checkout must be pinned at the baked head",
    clean_msg: str = "the checkout tree must equal the baked head",
) -> None:
    """Assert a real detached verification repo has exactly the baked committed/index/worktree contents."""
    assert verify_dir.exists(), exists_msg
    head = subprocess.run(["git", "-C", str(verify_dir), "rev-parse", "HEAD"], capture_output=True, check=True,
    ).stdout.decode().strip()
    assert head == head_sha, pinned_msg
    clean = subprocess.run(["git", "-C", str(verify_dir), "diff", "--quiet", "HEAD", "--"], capture_output=True)
    assert clean.returncode == 0, clean_msg

_REAL_PATCH = "diff --git a/tests/test_calc.py b/tests/test_calc.py\n@@ -1 +1 @@\n-old\n+new\n"

_CALC_FIXED = _CALC_BROKEN.replace("return a + b + 1", "return a + b")


def _seal_run(run_dir: Path, task: DaydreamReviewTask, repo_path: Path) -> Path:
    """Stage a committed fix, derive its shared candidate diff, and seal the archive; return the repo."""

    repo = _stage_repo(repo_path, task.data.head_sha, edit=_CALC_FIXED, commit=True)
    diff = subprocess.run(candidate_diff_cmd(str(repo), task.data.head_sha), capture_output=True, check=True).stdout
    present = [run_dir / rel for rel in RUN_DIR_FILES if (run_dir / rel).is_file()
    ] + sorted(run_dir.glob("deep/stack-*-records.json"))
    seal = seal_artifacts(present, candidate_diff=diff)
    (run_dir / "seal.json").write_text(seal.model_dump_json(), encoding="utf-8")
    return repo

# A tampered suite that exits zero must still earn zero suite reward.
_TAMPER_PASSING = (
    "import unittest\n\n"
    "class TestCalc(unittest.TestCase):\n"
    "    def test_placeholder(self) -> None:\n"
    "        pass\n"
)

# Overwriting the tracked tests/__init__.py to skip discovery: suite still exits 0.
_TAMPER_SKIPALL = 'import unittest\nraise unittest.SkipTest("gutted")\n'


def test_rundir_golden_user_messages_are_inert(rundir_golden: Path) -> None:
    """Every golden user message stays empty, including the structurally retained verify slot.

    This is fixture hygiene: scoring reads verdicts/output length and excludes the
    trajectory, so empty messages are not a scoring security boundary.
    """
    trajectory = json.loads((rundir_golden / "trajectory.json").read_text(encoding="utf-8"))
    user_steps = [s for s in trajectory["steps"] if s.get("source") == "user"]
    assert user_steps, "expected at least one user-authored step"
    for step in user_steps:
        assert step["message"] == ""
    step_11 = [s for s in trajectory["steps"] if s.get("step_id") == 11]
    assert len(step_11) == 1
    step = step_11[0]
    # Allow extra metadata fields while requiring the message fields used by this boundary.
    assert {"extra", "message", "source", "step_id", "timestamp"} <= set(step)
    assert step["source"] == "user"
    assert step["message"] == ""

async def test_intrinsic_composite_parity(
    tmp_path: Path, runtime: SubprocessRuntime, rundir_golden: Path, fixture_manifest_path: Path
) -> None:
    archive_root, task = _golden_task(tmp_path, fixture_manifest_path, rundir_golden)
    trace = _trace(task, archive_root=archive_root, repo_path=tmp_path / "repo")

    await task.score(trace, runtime)

    expected = score_trajectory(assemble_scoring_inputs(rundir_golden)).composite
    assert expected is not None
    assert trace.rewards["intrinsic_composite"] == expected
    assert trace.info["reward_breakdown"]["composite"] == expected

async def test_intrinsic_composite_ignores_historical_read_metrics(
    tmp_path: Path, runtime: SubprocessRuntime, rundir_golden: Path, fixture_manifest_path: Path
) -> None:
    archive_root, task = _golden_task(tmp_path, fixture_manifest_path, rundir_golden)
    trace = _trace(task, archive_root=archive_root, repo_path=tmp_path / "repo")

    await task.score(trace, runtime)

    breakdown = trace.info["reward_breakdown"]
    assert "grounding" not in breakdown["axes_present"]
    assert "grounding" not in breakdown

async def test_zero_finding_rollout_scores_no_intrinsic_reward(
    tmp_path: Path, runtime: SubprocessRuntime, rundir_golden: Path, fixture_manifest_path: Path
) -> None:
    archive_root = tmp_path / "archive"
    run_dir = _stage_run(archive_root, rundir_golden)

    (run_dir / "deep" / "merged-items.json").write_text(json.dumps({"items": []}), encoding="utf-8")
    # Remove recommendation verdicts so correctness cannot keep the composite non-None.
    (run_dir / "deep" / "recommendation-verdicts.json").unlink()

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["metrics"]["total_findings"] = 0
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    evaluation = json.loads((run_dir / "evaluation.json").read_text(encoding="utf-8"))
    evaluation["findings"]["total"] = 0
    (run_dir / "evaluation.json").write_text(json.dumps(evaluation), encoding="utf-8")

    task = _task(fixture_manifest_path)
    trace = _trace(task, archive_root=archive_root, repo_path=tmp_path / "repo")

    await task.score(trace, runtime)

    breakdown = trace.info["reward_breakdown"]
    assert trace.rewards["intrinsic_composite"] == 0.0
    assert "grounding" not in breakdown["axes_present"]
    assert breakdown["composite"] is None
    assert breakdown["axes_present"]["correctness"] is False

async def test_missing_run_dir_scores_zero(tmp_path: Path, runtime: SubprocessRuntime, fixture_manifest_path: Path
) -> None:
    """A crashed daydream still gets a gradient — zero, not an exception."""
    archive_root = tmp_path / "archive"
    (archive_root / "runs").mkdir(parents=True)
    task = _task(fixture_manifest_path)
    trace = _trace(task, archive_root=archive_root, repo_path=tmp_path / "repo")

    await task.score(trace, runtime)

    assert trace.rewards["intrinsic_composite"] == 0.0
    assert trace.info["reward_breakdown"] == {"error": "no archived run dir"}
    assert trace.metrics["n_findings"] == 0.0

@pytest.mark.parametrize("stage_edit", [False, True], ids=["unstaged", "staged"])
async def test_verifier_identity_branch_executes_and_fails_closed(
    stage_edit: bool, tmp_path: Path, runtime: SubprocessRuntime, fixture_manifest_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Force the container verifier branch through real clone/detach/apply setup.

    An unprivileged host cannot chown the verification checkout and must record zero
    without falling back to the mutable green tree. Root with the verifier identity
    can construct the protected checkout and obtain a green reading.
    """

    archive_root = tmp_path / "archive"
    (archive_root / "runs").mkdir(parents=True)
    task = _task(fixture_manifest_path)
    repo = _stage_repo(tmp_path / "repo", task.data.head_sha, edit=_CALC_FIXED)
    if stage_edit:
        subprocess.run(["git", "-C", str(repo), "add", "calc.py"], check=True)
    assert subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip() == task.data.head_sha
    expected_status = "M  calc.py\n" if stage_edit else " M calc.py\n"
    assert subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                          capture_output=True, text=True, check=True).stdout == expected_status
    trace = _trace(task, archive_root=archive_root, repo_path=repo)

    # Confirm host verifier capability before forcing the unprivileged branch.
    real_identity_available = taskset._verifier_identity_available

    async def _identity_available(_runtime: vf.Runtime) -> bool:
        return True  # container shape: setpriv + verifier user provisioned

    monkeypatch.setattr(taskset, "_verifier_identity_available", _identity_available)

    await task.score(trace, runtime)

    assert trace.metrics["test_oracle_unchanged"] == 1.0
    # Root with the verifier identity can rebuild and rerun the green suite. Unprivileged hosts fail
    # chown and return zero, without falling back to the mutable tree.
    verifier_rerun_succeeds = os.geteuid() == 0 and await real_identity_available(runtime)
    assert trace.metrics["suite_non_regression"] == (1.0 if verifier_rerun_succeeds else 0.0)
    # The retained clone and diff prove setup ran before verification refused.
    verify_dir = tmp_path / "repo-verify"
    assert (verify_dir / "calc.py").read_text(encoding="utf-8") == _CALC_FIXED

async def test_red_suite_records_no_non_regression(
    tmp_path: Path, runtime: SubprocessRuntime, fixture_manifest_path: Path
) -> None:
    trace = await _score(tmp_path, fixture_manifest_path, runtime, edit=_CALC_BROKEN, patch=_REAL_PATCH)

    assert trace.metrics["fixes_applied"] == 1.0
    assert trace.metrics["suite_non_regression"] == 0.0

async def test_suite_result_is_telemetry_not_reward(
    tmp_path: Path, runtime: SubprocessRuntime, fixture_manifest_path: Path,
) -> None:
    task = _task(fixture_manifest_path)
    assert task.data.test_command == "python -m unittest discover -q"
    trace = await _score(tmp_path, fixture_manifest_path, runtime, task=task, edit=_CALC_FIXED, patch=_REAL_PATCH)

    assert set(trace.rewards) == {"intrinsic_composite"}
    assert "fix_tests_pass" not in trace.rewards
    assert trace.metrics["fixes_applied"] == 1.0
    assert trace.metrics["test_oracle_unchanged"] == 1.0
    assert trace.metrics["suite_non_regression"] == 1.0

async def test_tampered_suite_never_records_honest_non_regression(
    tmp_path: Path, runtime: SubprocessRuntime, rundir_golden: Path, fixture_manifest_path: Path,
) -> None:
    """A gutted test oracle records suite_non_regression 0.0 and no suite reward."""
    archive_root, task = _golden_task(tmp_path, fixture_manifest_path, rundir_golden)
    repo = _stage_repo(tmp_path / "repo", task.data.head_sha, edit=_CALC_FIXED)
    (repo / "tests/test_calc.py").write_text(_TAMPER_PASSING, encoding="utf-8")
    trace = _trace(task, archive_root=archive_root, repo_path=repo)

    await task.score(trace, runtime)

    assert "fix_tests_pass" not in trace.rewards
    assert trace.metrics["test_oracle_unchanged"] == 0.0
    assert trace.metrics["suite_non_regression"] == 0.0

@pytest.mark.parametrize(("edit", "tamper_rel", "tamper_content"),
    [
        (None, "tests/test_calc.py", _TAMPER_PASSING),
        (None, "tests/__init__.py", _TAMPER_SKIPALL),
        # The new untracked oracle must be discovered through ls-files.
        (_CALC_FIXED, "tests/pytest.ini", "[pytest]\n"),
    ], ids=["test-source-tamper", "test-package-config-tamper", "untracked-oracle-file"],
)
async def test_suite_rejects_protected_test_path_changes(
    edit: str | None, tamper_rel: str, tamper_content: str, tmp_path: Path, runtime: SubprocessRuntime,
    rundir_golden: Path, fixture_manifest_path: Path,
) -> None:
    """Green-if-run oracle tampering must produce zero telemetry without executing the suite.

    A real archived claim makes any execution emit test_claim_mismatch, so its
    absence proves the gate held rather than merely lacking a claim.
    """
    archive_root, task = _golden_task(tmp_path, fixture_manifest_path, rundir_golden)
    repo = _stage_repo(tmp_path / "repo", task.data.head_sha, edit=edit)
    (repo / tamper_rel).write_text(tamper_content, encoding="utf-8")
    trace = _trace(task, archive_root=archive_root, repo_path=repo)

    await task.score(trace, runtime)

    _assert_gate_held(trace)


async def _score(tmp_path: Path, fixture_manifest_path: Path, runtime: SubprocessRuntime, *, edit: str | None = None,
    patch: str | None = None, commit: bool = False, commit_patch: bool = False, task: DaydreamReviewTask | None = None,
    archive_root: Path | None = None, seal_ok: bool = False,
) -> vf.Trace:
    """Stage a standard repo/archive task and run its real ``task.score``.

    ``task``/``archive_root`` let a caller pre-arrange state; otherwise both are built.
    """
    archive_root = archive_root or tmp_path / "archive"
    (archive_root / "runs").mkdir(parents=True, exist_ok=True)
    task = task or _task(fixture_manifest_path)
    repo = _stage_repo(
        tmp_path / "repo", task.data.head_sha, edit=edit, patch=patch, commit=commit, commit_patch=commit_patch
    )
    trace = _trace(task, archive_root=archive_root, repo_path=repo)
    if seal_ok:
        trace.info["daydream_seal_ok"] = True
    await task.score(trace, runtime)
    return trace


async def _score_fail_closed(
    tmp_path: Path, runtime: SubprocessRuntime, archive_root: Path, rundir_golden: Path, fixture_manifest_path: Path, *,
    unlink_claim: bool = False,
) -> vf.Trace:
    """Score a dirty-fixed repo against an unresolvable baked SHA.

    unlink_claim creates the negative control: production omits mismatch telemetry,
    but the test helper must reject the missing claim instead of passing vacuously.
    """
    run_dir = _stage_run(archive_root, rundir_golden)
    if unlink_claim:
        (run_dir / "deep" / "test-verdict.json").unlink()
    task = _task(fixture_manifest_path)
    repo = _stage_repo(tmp_path / "repo", task.data.head_sha, edit=_CALC_FIXED)
    # Apply a dirty fix first so fix_applied is true before the baked-SHA diff fails with exit 128.
    task.data = task.data.model_copy(update={"head_sha": "0" * 40})
    trace = _trace(task, archive_root=archive_root, repo_path=repo)

    await task.score(trace, runtime)

    return trace


async def test_oracle_gate_fails_closed_on_git_error(
    tmp_path: Path, runtime: SubprocessRuntime, rundir_golden: Path, fixture_manifest_path: Path,
) -> None:
    """A failed Git comparison returns zero without running the green suite; its claim keeps the probe nonvacuous."""
    archive_root = tmp_path / "archive"
    trace = await _score_fail_closed(tmp_path, runtime, archive_root, rundir_golden, fixture_manifest_path)

    _assert_gate_held(trace)

async def test_assert_gate_held_raises_when_claim_absent(
    tmp_path: Path, runtime: SubprocessRuntime, rundir_golden: Path, fixture_manifest_path: Path,
) -> None:
    """The test gate helper rejects absent claims even though production legitimately omits mismatch telemetry."""
    archive_root = tmp_path / "archive"
    trace = await _score_fail_closed(
        tmp_path, runtime, archive_root, rundir_golden, fixture_manifest_path, unlink_claim=True,
    )

    assert "test_claim_mismatch" not in trace.metrics
    with pytest.raises(AssertionError):
        _assert_gate_held(trace)

async def test_gate_held_raises_when_a_second_run_dir_is_claim_less(
    tmp_path: Path, runtime: SubprocessRuntime, rundir_golden: Path, fixture_manifest_path: Path,
) -> None:
    """Require claims in every test run directory; production declines attribution for multiple runs.

    A claim-less sibling must therefore fail the helper before mismatch absence is accepted.
    """
    archive_root = tmp_path / "archive"
    _stage_run(archive_root, rundir_golden, session_id=SESSION_ID)
    second = _stage_run(archive_root, rundir_golden, session_id="session-second")
    (second / "deep" / "test-verdict.json").unlink()
    task = _task(fixture_manifest_path)
    repo = _stage_repo(tmp_path / "repo", task.data.head_sha, edit=_CALC_FIXED)
    trace = _trace(task, archive_root=archive_root, repo_path=repo)

    await task.score(trace, runtime)

    # Multiple run directories suppress production attribution; the claims helper must still expose
    # the injected verdict.
    assert "test_claim_mismatch" not in trace.metrics
    with pytest.raises(AssertionError, match="deep/test-verdict.json claim"):
        _assert_gate_held(trace)

@pytest.mark.parametrize("flag", ["--skip-worktree", "--assume-unchanged"], ids=["skip-worktree", "assume-unchanged"])
async def test_oracle_gate_rejects_flag_tampered_tracked_file(
    flag: str, tmp_path: Path, runtime: SubprocessRuntime, rundir_golden: Path, fixture_manifest_path: Path,
) -> None:
    """Reject skip-worktree/assume-unchanged flags on protected files.

    These make Git diff consult index content and hide worktree tampering; ls-files
    must detect the loss of oracle verifiability.
    """
    archive_root, task = _golden_task(tmp_path, fixture_manifest_path, rundir_golden)
    # A real fix (calc.py) plus a flagged, gutted tracked test: diff is fooled,
    # so only the flag probe stands between this and a free green reading.
    repo = _stage_repo(tmp_path / "repo", task.data.head_sha, edit=_CALC_FIXED)
    (repo / "tests/test_calc.py").write_text(_TAMPER_PASSING, encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "update-index", flag, "tests/test_calc.py"], check=True)
    trace = _trace(task, archive_root=archive_root, repo_path=repo)

    await task.score(trace, runtime)

    _assert_gate_held(trace)

def _tracked_gitignore_edit(repo: Path) -> None:
    """Changing a tracked ignore rule is itself an oracle change."""
    gitignore = repo / ".gitignore"
    gitignore.write_text(gitignore.read_text(encoding="utf-8") + "tests/pytest.ini\n", encoding="utf-8")
    (repo / "tests/pytest.ini").write_text("[pytest]\n", encoding="utf-8")


def _info_exclude_rule(repo: Path) -> None:
    """A fresh clone's info/exclude is comments-only; any real rule is tampering."""
    info_exclude = repo / ".git/info/exclude"
    info_exclude.write_text(info_exclude.read_text(encoding="utf-8") + "tests/conftest.py\n", encoding="utf-8")
    (repo / "tests/conftest.py").write_text("", encoding="utf-8")


def _core_excludesfile(repo: Path) -> None:
    """The probe must disable core.excludesFile even when .git/config looks untracked."""
    ignores = repo.parent / "excludes"
    ignores.write_text("tests/pytest.ini\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "config", "core.excludesFile", str(ignores)], check=True)
    (repo / "tests/pytest.ini").write_text("[pytest]\n", encoding="utf-8")


@pytest.mark.parametrize("tamper", [_tracked_gitignore_edit, _info_exclude_rule, _core_excludesfile], ids=[
    "tracked_gitignore_edit", "info_exclude_rule", "untracked_hidden_by_core_excludesfile",
])
async def test_oracle_gate_rejects_ignore_tampering(
    tamper: Callable[[Path], None], tmp_path: Path, runtime: SubprocessRuntime,
    rundir_golden: Path, fixture_manifest_path: Path,
) -> None:
    """Agent-controlled ignore rules cannot hide a new oracle file from the gate."""
    archive_root, task = _golden_task(tmp_path, fixture_manifest_path, rundir_golden)
    repo = _stage_repo(tmp_path / "repo", task.data.head_sha, edit=_CALC_FIXED)
    tamper(repo)
    trace = _trace(task, archive_root=archive_root, repo_path=repo)

    await task.score(trace, runtime)

    _assert_gate_held(trace)

async def test_oracle_gate_green_despite_suite_bytecode_artifacts(
    tmp_path: Path, runtime: SubprocessRuntime, rundir_golden: Path, fixture_manifest_path: Path,
) -> None:
    """Exclude ordinary Python bytecode artifacts explicitly from untracked-oracle checks.

    The probe uses no --exclude-standard; benign exclusions must not hide test sources.
    """
    archive_root, task = _golden_task(tmp_path, fixture_manifest_path, rundir_golden)
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
    tmp_path: Path, runtime: SubprocessRuntime, rundir_golden: Path, fixture_manifest_path: Path,
) -> None:
    """Reject an untracked root sitecustomize.py even outside declared protected paths.

    Python startup could execute it and exit zero without running the suite.
    """
    archive_root, task = _golden_task(tmp_path, fixture_manifest_path, rundir_golden)
    repo = _stage_repo(tmp_path / "repo", task.data.head_sha, edit=_CALC_FIXED)
    (repo / "sitecustomize.py").write_text("import sys\nsys.exit(0)\n", encoding="utf-8")
    trace = _trace(task, archive_root=archive_root, repo_path=repo)

    await task.score(trace, runtime)

    _assert_gate_held(trace)

@pytest.mark.parametrize("patch, commit",
    [(None, False), ("", False), (_REAL_PATCH, False),
        (None, True),  # empty commit: HEAD advances, committed tree unchanged
    ], ids=["no-patch-file", "empty-patch", "non-empty-patch-but-untouched-tree", "empty-commit"],
)
async def test_no_fixes_records_no_non_regression(
    patch: str | None, commit: bool, tmp_path: Path, runtime: SubprocessRuntime, fixture_manifest_path: Path,
) -> None:
    """An unchanged target records zero even when recommended.patch contains generated artifacts.

    Real repositories may not ignore .daydream, so patch existence/size cannot prove a fix.
    """
    trace = await _score(tmp_path, fixture_manifest_path, runtime, patch=patch, commit=commit)

    assert trace.metrics["fixes_applied"] == 0.0
    assert trace.metrics["suite_non_regression"] == 0.0
    assert "test_claim_mismatch" not in trace.metrics
    assert "test_claim_passed_without_fix" not in trace.metrics

async def test_unresolvable_head_sha_scores_no_fix(
    tmp_path: Path, runtime: SubprocessRuntime, fixture_manifest_path: Path
) -> None:
    """Only diff exit 1 proves a change; an unresolved baked object (128) must record zero."""
    archive_root = tmp_path / "archive"
    (archive_root / "runs").mkdir(parents=True)
    task = _task(fixture_manifest_path)
    # Take a real snapshot before changing the object store.
    repo = _stage_repo(tmp_path / "repo", task.data.head_sha)
    task.data = task.data.model_copy(update={"head_sha": "0" * 40})
    trace = _trace(task, archive_root=archive_root, repo_path=repo)

    await task.score(trace, runtime)

    assert trace.metrics["fixes_applied"] == 0.0
    assert trace.metrics["suite_non_regression"] == 0.0

@pytest.mark.parametrize("claimed", [True, False], ids=["claimed-green", "claimed-red"])
async def test_no_fixes_still_records_the_test_claim(
    claimed: bool, tmp_path: Path, runtime: SubprocessRuntime, rundir_golden: Path, fixture_manifest_path: Path,
) -> None:
    """Preserve claimed suite status even when no fix means no verification rerun.

    This keeps unsupported green claims visible without buying a second suite run.
    """
    archive_root = tmp_path / "archive"
    run_dir = _stage_run(archive_root, rundir_golden)
    (run_dir / "deep" / "test-verdict.json").write_text(json.dumps({"passed": claimed, "retries": 0}), encoding="utf-8")

    task = _task(fixture_manifest_path)
    repo = _stage_repo(tmp_path / "repo", task.data.head_sha, patch=_REAL_PATCH)
    trace = _trace(task, archive_root=archive_root, repo_path=repo)

    await task.score(trace, runtime)

    assert trace.metrics["fixes_applied"] == 0.0
    assert trace.metrics["suite_non_regression"] == 0.0
    assert trace.metrics["test_claim_passed_without_fix"] == float(claimed)
    # Observability only: recording the claim must not invent a verdict comparison.
    assert "test_claim_mismatch" not in trace.metrics

async def test_score_reuses_one_archived_run_snapshot(
    tmp_path: Path, runtime: SubprocessRuntime, rundir_golden: Path, fixture_manifest_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A read-once runtime enforces one shared artifact snapshot across all three scoring consumers; clear
    run_dir afterward.
    """
    archive_root, task = _golden_task(tmp_path, fixture_manifest_path, rundir_golden)
    repo = _stage_repo(tmp_path / "repo", task.data.head_sha, patch=_REAL_PATCH)
    trace = _trace(task, archive_root=archive_root, repo_path=repo)

    class ReadOnce:
        def __init__(self, real: Callable[[str], Awaitable[bytes]]) -> None:
            self._real = real
            self._seen: set[str] = set()

        async def __call__(self, path: str) -> bytes:
            if path in self._seen:
                raise AssertionError(f"archived artifact {path} read more than once in a single score call")
            self._seen.add(path)
            return await self._real(path)

    real_read = runtime.read
    monkeypatch.setattr(runtime, "read", ReadOnce(real_read))

    await task.score(trace, runtime)

    expected = score_trajectory(assemble_scoring_inputs(rundir_golden)).composite
    assert expected is not None
    assert trace.rewards["intrinsic_composite"] == expected
    assert trace.metrics["test_claim_passed_without_fix"] == 1.0
    assert trace.metrics["n_findings"] == 1.0
    assert trace.state.run_dir is None

@pytest.mark.parametrize("red, expected", [(True, 1.0), (False, 0.0)], ids=["mismatch", "agrees"])
async def test_metric_claim_mismatch_fires(
    red: bool, expected: float, tmp_path: Path, runtime: SubprocessRuntime, rundir_golden: Path,
    fixture_manifest_path: Path,
) -> None:
    """daydream's prose-derived verdict is a claim, graded against the real re-run."""
    archive_root = tmp_path / "archive"
    run_dir = _stage_run(archive_root, rundir_golden)
    claim = json.loads((run_dir / "deep" / "test-verdict.json").read_text(encoding="utf-8"))
    assert claim == {"passed": True, "retries": 0}

    task = _task(fixture_manifest_path)
    repo = _stage_repo(
        tmp_path / "repo", task.data.head_sha, edit=_CALC_BROKEN if red else _CALC_FIXED, patch=_REAL_PATCH,
    )
    trace = _trace(task, archive_root=archive_root, repo_path=repo)

    await task.score(trace, runtime)

    assert trace.metrics["fixes_applied"] == 1.0
    assert trace.metrics["suite_non_regression"] == (0.0 if red else 1.0)
    assert trace.metrics["test_claim_mismatch"] == expected

async def test_review_shape_metrics(
    tmp_path: Path, runtime: SubprocessRuntime, rundir_golden: Path, fixture_manifest_path: Path
) -> None:
    """n_findings mirrors merged-items.json; golden_overlap is a path fraction."""
    archive_root, task = _golden_task(tmp_path, fixture_manifest_path, rundir_golden)
    trace = _trace(task, archive_root=archive_root, repo_path=tmp_path / "repo")

    await task.score(trace, runtime)

    items = json.loads((rundir_golden / "deep" / "merged-items.json").read_text(encoding="utf-8"))["items"]
    found_files = {item["file"] for item in items}
    golden_paths = [comment.path for comment in task.data.golden_comments if comment.path]
    assert golden_paths, "the manifest fixture must carry at least one golden path"

    assert trace.metrics["n_findings"] == float(len(items))
    assert trace.metrics["n_golden_comments"] == float(len(golden_paths))
    assert trace.metrics["golden_overlap"] == sum(1 for p in golden_paths if p in found_files) / len(golden_paths)
    assert found_files.isdisjoint(golden_paths) and trace.metrics["golden_overlap"] == 0.0

    hit_root = tmp_path / "archive-hit"
    hit_run = _stage_run(hit_root, rundir_golden, session_id="session-hit")
    (hit_run / "deep" / "merged-items.json").write_text(
        json.dumps({"items": [{"id": 1, "file": golden_paths[0]}, {"id": 2, "file": "README.md"}]}), encoding="utf-8",
    )
    hit_trace = _trace(task, archive_root=hit_root, repo_path=tmp_path / "repo")

    await task.score(hit_trace, runtime)

    assert hit_trace.metrics["n_findings"] == 2.0
    assert hit_trace.metrics["golden_overlap"] == 1.0

async def test_review_shape_survives_a_non_object_merged_items(
    tmp_path: Path, runtime: SubprocessRuntime, rundir_golden: Path, fixture_manifest_path: Path
) -> None:
    """merged-items.json is written inside the rollout, so a corrupt one must not crash scoring."""
    archive_root = tmp_path / "archive"
    run_dir = _stage_run(archive_root, rundir_golden)
    (run_dir / "deep" / "merged-items.json").write_text(json.dumps([{"file": "calc.py"}]), encoding="utf-8")
    task = _task(fixture_manifest_path)
    trace = _trace(task, archive_root=archive_root, repo_path=tmp_path / "repo")

    await task.score(trace, runtime)

    assert trace.metrics["n_findings"] == 0.0
    assert trace.metrics["golden_overlap"] == 0.0

async def test_committed_fix_counts_even_with_a_clean_tree(
    tmp_path: Path, runtime: SubprocessRuntime, fixture_manifest_path: Path
) -> None:
    trace = await _score(tmp_path, fixture_manifest_path, runtime, edit=_CALC_FIXED, commit=True)

    assert trace.metrics["fixes_applied"] == 1.0
    assert trace.metrics["suite_non_regression"] == 1.0
    assert trace.metrics["test_oracle_unchanged"] == 1.0

async def test_committed_daydream_artifacts_not_a_fix(
    tmp_path: Path, runtime: SubprocessRuntime, fixture_manifest_path: Path
) -> None:
    """Force-committed .daydream artifacts alone must yield zero non-regression telemetry.

    Exclude generated paths from committed-tree comparison as well as working changes.
    """
    trace = await _score(tmp_path, fixture_manifest_path, runtime, patch=_REAL_PATCH, commit=True, commit_patch=True)

    assert trace.metrics["fixes_applied"] == 0.0
    assert trace.metrics["suite_non_regression"] == 0.0
    assert "test_claim_mismatch" not in trace.metrics

async def test_reward_version_is_pinned(
    tmp_path: Path, runtime: SubprocessRuntime, rundir_golden: Path, fixture_manifest_path: Path
) -> None:
    """Pin both rollout and intrinsic versions independently of scorer parity.

    Calling the same scorer on both sides would let a semantic/version change pass unnoticed.
    """

    assert REWARD_VERSION == "2026.10.01-1", (
        f"the training pipeline's reward version moved to {REWARD_VERSION!r}. Re-derive the "
        "rollout reward's expected values before trusting any run scored across the boundary."
    )
    assert ROLLOUT_REWARD_VERSION == "2026.10.01-1", (
        f"the rollout reward contract version moved to {ROLLOUT_REWARD_VERSION!r}"
    )

    archive_root, task = _golden_task(tmp_path, fixture_manifest_path, rundir_golden)
    trace = _trace(task, archive_root=archive_root, repo_path=tmp_path / "repo")

    await task.score(trace, runtime)

    breakdown = trace.info["reward_breakdown"]
    assert breakdown["reward_version"] == ROLLOUT_REWARD_VERSION
    assert breakdown["intrinsic_reward_version"] == REWARD_VERSION

@pytest.mark.parametrize("tamper", ["artifact", "none", "committed-diff"], ids=[
    "tampered_sealed_artifact_zeroes_intrinsic_and_non_regression",
    "untampered_sealed_run_scores_normally",
    "seal_detects_committed_diff_changed_after_sealing",
])
async def test_sealed_run_scoring(
    tamper: str, tmp_path: Path, runtime: SubprocessRuntime, rundir_golden: Path, fixture_manifest_path: Path,
) -> None:
    """Bind reward to sealed artifacts and the live candidate diff.

    Re-hashing the seal's embedded diff would miss a subsequent commit; the
    verifier must derive it again from the repository before trusting reward.
    """
    archive_root = tmp_path / "archive"
    run_dir = _stage_run(archive_root, rundir_golden)
    task = _task(fixture_manifest_path)
    repo = _seal_run(run_dir, task, tmp_path / "repo")
    if tamper == "artifact":
        (run_dir / "deep" / "merged-items.json").write_text(json.dumps({"items": []}), encoding="utf-8")
    elif tamper == "committed-diff":
        (repo / "README.md").write_text("# changed\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
        subprocess.run(["git", "-C", str(repo), "commit", "--quiet", "-m", "tamper"], check=True)
    trace = _trace(task, archive_root=archive_root, repo_path=repo)

    await task.score(trace, runtime)

    assert trace.metrics["seal_verified"] == float(tamper == "none")
    expected = score_trajectory(assemble_scoring_inputs(rundir_golden)).composite if tamper == "none" else 0.0
    assert trace.rewards["intrinsic_composite"] == expected
    if tamper != "none":
        assert trace.metrics["suite_non_regression"] == 0.0

async def test_vanished_seal_on_a_harness_sealed_run_is_a_tamper(
    tmp_path: Path, runtime: SubprocessRuntime, rundir_golden: Path, fixture_manifest_path: Path,
) -> None:
    """An expected seal that vanishes must fail, never enter legacy unsealed scoring.

    The harness's successful-seal marker makes absence a contradiction requiring zero reward.
    """
    archive_root, task = _golden_task(tmp_path, fixture_manifest_path, rundir_golden)
    trace = await _score(tmp_path, fixture_manifest_path, runtime,
        task=task, archive_root=archive_root, edit=_CALC_FIXED, commit=True, seal_ok=True,
    )

    assert trace.metrics["seal_verified"] == 0.0
    assert trace.rewards["intrinsic_composite"] == 0.0
    assert trace.metrics["suite_non_regression"] == 0.0

async def test_git_failure_at_verify_time_fails_closed(
    tmp_path: Path, runtime: SubprocessRuntime, rundir_golden: Path, fixture_manifest_path: Path,
) -> None:
    """A failed scoring-time diff cannot verify as an empty diff.

    Seal-time failure may store empty bytes; mapping verification failure likewise
    would falsely validate matching hashes without ever deriving a diff.
    """
    archive_root = tmp_path / "archive"
    run_dir = _stage_run(archive_root, rundir_golden)
    task = _task(fixture_manifest_path)

    present = [run_dir / rel for rel in RUN_DIR_FILES if (run_dir / rel).is_file()
    ] + sorted(run_dir.glob("deep/stack-*-records.json"))
    # A seal produced while git failed at seal time seals the empty diff.
    seal = seal_artifacts(present, candidate_diff=b"")
    (run_dir / "seal.json").write_text(seal.model_dump_json(), encoding="utf-8")

    # A failed re-derivation must not collide with the digest of a genuinely empty sealed diff.
    trace = _trace(task, archive_root=archive_root, repo_path=tmp_path / "not-a-repo")
    trace.info["daydream_seal_ok"] = True

    await task.score(trace, runtime)

    assert trace.metrics["seal_verified"] == 0.0
    assert trace.rewards["intrinsic_composite"] == 0.0
    assert trace.metrics["suite_non_regression"] == 0.0

async def test_verify_checkout_failed_diff_fails_closed(
    tmp_path: Path, runtime: SubprocessRuntime, fixture_manifest_path: Path,
) -> None:

    task = _task(fixture_manifest_path)
    repo = _stage_repo(tmp_path / "repo", task.data.head_sha, edit=_CALC_FIXED, commit=True)
    # An invalid diff.algorithm makes diff fail after clone succeeds. The hardened argv ignores
    # diff.external, so it cannot trigger this failure.
    subprocess.run(["git", "-C", str(repo), "config", "diff.algorithm", "not-a-real-algorithm"], check=True)
    result = await taskset._prepare_verify_checkout(runtime, str(repo), task.data.head_sha)
    assert result is None, "a failed diff derivation must not return a checkout"
    # The retained clone at the pinned head proves refusal happened before applying the patch.
    verify_dir = tmp_path / "repo-verify"
    _assert_checkout_pinned_at(
        verify_dir, task.data.head_sha, exists_msg="the chain must reach the diff step (clone + checkout ran)",
        pinned_msg="the chain must reach the diff step (checkout detached at the baked head)",
        clean_msg="a failed diff must never apply a partial candidate diff",
    )

async def test_verify_checkout_empty_diff_is_clean_noop(
    tmp_path: Path, runtime: SubprocessRuntime, fixture_manifest_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:

    task = _task(fixture_manifest_path)
    # --allow-empty commit: HEAD advances, committed tree identical -> genuinely empty diff
    repo = _stage_repo(tmp_path / "repo", task.data.head_sha, commit=True)
    chown_marker = tmp_path / "chown-reached"
    if os.geteuid() != 0:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        chown = bin_dir / "chown"
        chown.write_text(f"#!/bin/sh\ntouch {shlex.quote(str(chown_marker))}\nexit 1\n", encoding="utf-8")
        chown.chmod(0o755)
        monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    result = await taskset._prepare_verify_checkout(runtime, str(repo), task.data.head_sha)
    if os.geteuid() == 0:
        assert result == str(tmp_path / "repo-verify"), "an empty diff must not fail construction"
    else:
        assert result is None
        assert chown_marker.is_file(), "empty diff must pass the apply guard before chown refuses"
        verify_dir = tmp_path / "repo-verify"
        _assert_checkout_pinned_at(
            verify_dir, task.data.head_sha, exists_msg="the empty diff must not abort the checkout build",
            clean_msg="the empty diff must apply as a clean no-op",
        )

async def test_verify_checkout_applies_exactly_the_candidate_diff(
    tmp_path: Path, runtime: SubprocessRuntime, fixture_manifest_path: Path,
) -> None:

    task = _task(fixture_manifest_path)
    repo = _stage_repo(tmp_path / "repo", task.data.head_sha, edit=_CALC_FIXED, commit=True)
    head_sha = task.data.head_sha

    expected = subprocess.run(candidate_diff_cmd(str(repo), head_sha), capture_output=True, check=True).stdout
    assert expected, "staged committed fix must yield a non-empty candidate diff"

    result = await taskset._prepare_verify_checkout(runtime, str(repo), head_sha)
    verify_dir = tmp_path / "repo-verify"
    # Both root and non-root paths must execute Git setup before chown.
    assert verify_dir.is_dir(), "the verify checkout was not constructed"
    if os.geteuid() == 0:
        assert result == str(verify_dir), "construction must return the built checkout path"
    applied = subprocess.run(
        ["git", "-C", str(verify_dir), "diff", head_sha, "--", "calc.py"], capture_output=True, check=True,
    ).stdout
    assert applied == expected, "the verify checkout drifted from the candidate diff the seal binds"
    assert (verify_dir / "calc.py").read_text(encoding="utf-8") == _CALC_FIXED, (
        "the verify checkout did not carry the candidate diff the seal binds"
    )

def test_candidate_diff_cmd_carries_hardening_flags() -> None:
    argv = candidate_diff_cmd("/work/repo", "deadbeef")
    assert argv == [
        "git", "-C", "/work/repo", "diff", "--no-ext-diff", "--no-textconv", "deadbeef", "--", DAYDREAM_EXCLUDE,
    ]

@pytest.mark.parametrize("attack", ["external-diff", "textconv"])
async def test_verify_checkout_repo_helper_ignored(
    tmp_path: Path, runtime: SubprocessRuntime, fixture_manifest_path: Path, attack: str,
) -> None:
    """A repo-local helper that cannot run must not abort verifier-checkout."""

    task = _task(fixture_manifest_path)
    repo = _stage_repo(tmp_path / "repo", task.data.head_sha, edit=_CALC_FIXED, commit=True)
    if attack == "external-diff":
        subprocess.run(["git", "-C", str(repo), "config", "diff.external", "/nonexistent-diff-tool"], check=True)
    else:
        # Repository-controlled attribute selecting a driver whose textconv cannot run.
        (repo / ".gitattributes").write_text("*.py diff=evil\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(repo), "add", ".gitattributes"], check=True)
        subprocess.run(["git", "-C", str(repo), "commit", "-qm", "attr"], check=True)
        subprocess.run(
            ["git", "-C", str(repo), "config", "diff.evil.textconv", "/nonexistent-textconv-tool"], check=True,
        )
    result = await taskset._prepare_verify_checkout(runtime, str(repo), task.data.head_sha)
    verify_dir = tmp_path / "repo-verify"
    assert verify_dir.is_dir(), "the verify checkout must still be constructed"
    if os.geteuid() == 0:
        assert result == str(verify_dir), "construction must succeed on a root host"
    assert (verify_dir / "calc.py").read_text(encoding="utf-8") == _CALC_FIXED, (
        "the candidate patch must reach repo-verify despite the repo helper"
    )

async def test_fixes_applied_quiet_probe_carries_hardening_flags() -> None:
    rt = FakeRuntime(exit_code=0)
    await taskset._fixes_applied(rt, "/work/repo", "deadbeef")
    assert rt.commands[1] == ["git", "-C", "/work/repo", "diff", "--no-ext-diff", "--no-textconv", "--quiet",
        "deadbeef", "HEAD", "--", taskset.DAYDREAM_EXCLUDE,
    ]

async def test_protected_test_paths_unchanged_quiet_probe_carries_hardening_flags() -> None:
    rt = FakeRuntime(exit_code=0)
    await taskset._protected_test_paths_unchanged(rt, "/work/repo", "deadbeef", ["tests"])
    assert rt.commands[0] == ["git", "-C", "/work/repo", "diff", "--no-ext-diff", "--no-textconv", "--quiet",
        "deadbeef", "--", *["tests"], *taskset.ORACLE_IGNORE_PATHSPECS,
    ]

@pytest.mark.parametrize(("stage_kwargs", "expected"),
    [pytest.param({}, False, id="clean-tree-at-baked-head"),
        pytest.param({"edit": _CALC_FIXED}, True, id="uncommitted-edit"),
        pytest.param({"edit": _CALC_FIXED, "commit": True}, True, id="committed-fix"),
        pytest.param({"commit": True}, False, id="empty-commit-moves-head-only"),
        pytest.param({"patch": "not-a-fix-signal", "commit": True, "commit_patch": True}, False,
            id="committed-daydream-artifacts-only",
        ), pytest.param({"edit": _CALC_FIXED, "patch": "not-a-fix-signal", "commit": True, "commit_patch": True}, True,
            id="committed-fix-plus-daydream-artifacts",
        ),
    ],
)
async def test_oracle_acceptance_matches_candidate_diff_semantics(
    tmp_path: Any, runtime: Any, fixture_manifest_path: Any, stage_kwargs: dict[str, Any], expected: bool,
) -> None:
    """Oracle change detection and candidate diff must agree for every canonical repository state.

    Committed/staged/unstaged tracked edits count; untracked and excluded .daydream
    artifacts do not. Keep these independently implemented probes from drifting.
    """

    task = _task(fixture_manifest_path)
    repo = _stage_repo(tmp_path / "repo", task.data.head_sha, **stage_kwargs)

    accepted = await taskset._fixes_applied(runtime, str(repo), task.data.head_sha)
    assert accepted == expected, (f"oracle verdict {accepted!r} contradicts canonical state "
        f"{stage_kwargs!r}: the acceptance probe drifted from the "
        f"tracked-tree contract"
    )

    proc = subprocess.run(rundir_mod.candidate_diff_cmd(str(repo), task.data.head_sha), capture_output=True)
    produced_fix = bool(proc.stdout.strip())
    assert produced_fix == expected, (f"derived candidate diff {'showed changes' if produced_fix else 'was empty'} "
        f"against canonical state {stage_kwargs!r}: the deriv site drifted "
        f"from the tracked-tree contract"
    )
