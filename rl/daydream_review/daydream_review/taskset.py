"""Taskset: one task per manifest-declared pull-request snapshot.

Tasks come from ``images/manifest.toml`` — never from
the pinned Martian-5 held-out benchmark, whose five repositories are exactly the
SPEC C5 exclusion list. :meth:`DaydreamReviewTaskset.load` enforces that
unconditionally: there is no bypass parameter and no split exception. Train and
eval are two different manifests, not a flag.

Each manifest entry carries everything one task needs: a container image name, a
test command, a clone URL, protected test-oracle paths, and the PR snapshots
(``base_sha``/``head_sha``, plus golden comments) the entry trains against. The
manifest is the single source of truth for the rollout set: a repo or PR absent
from it is not a task.
"""

from __future__ import annotations

import json
import logging
import shlex
import tempfile
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import verifiers.v1 as vf
from daydream.dataset_scoring import assemble_scoring_inputs
from daydream.training.exclusion import load_exclusion_list
from daydream.training.reward import score_trajectory
from daydream.training.reward_model import OutcomeModel, score_comment as _score_outcome_comment
from daydream.training.rubric import RubricV2Breakdown, score_review as _score_rubric_review
from pydantic import BaseModel, ConfigDict, Field, field_validator
from verifiers.v1.errors import boundary

from daydream_review.gate_refusal import (
    Stage0GateRefused,
    require_outcome_model_bound,
    require_stage0_gate,
)
from daydream_review.rundir import (
    DAYDREAM_EXCLUDE as DAYDREAM_EXCLUDE,
    DEFAULT_ARCHIVE_ROOT,
    candidate_diff_cmd,
    candidate_quiet_diff_cmd,
    fetch_run_dir,
    verify_seal,
)

logger = logging.getLogger(__name__)

DEFAULT_REPO_PATH = "/work/repo"

#: The taskset/harness id this package registers under (verifiers resolves the
#: id by importing the hyphen-to-underscore module name). rl.toml and the docs
#: cite this constant; renaming the environment means changing it here and
#: everywhere it appears.
DEFAULT_TASKSET_ID = "daydream-review"

#: The aggregate rollout reward contract version, distinct from the intrinsic
#: scorer's ``REWARD_VERSION`` (``daydream/training/reward.py``, unchanged and
#: the intrinsic parity pin). ``reward_breakdown`` stamps both: ``reward_version``
#: is this rollout contract, and ``intrinsic_reward_version`` is the offline
#: scorer it was evaluated against. Archives scored before this boundary was
#: introduced carry only the intrinsic version and need no migration tag.
ROLLOUT_REWARD_VERSION = "2026.10.01-1"


class _OutcomeScorer:
    """Adapt the frozen outcome model to the rubric scoring protocol."""

    def __init__(self, model: OutcomeModel) -> None:
        self._model = model

    def score_comment(self, text: str) -> float:
        return float(_score_outcome_comment(self._model, text))


_outcome_model_cache: dict[Path, _OutcomeScorer] = {}


def _load_outcome_model(path: Path) -> _OutcomeScorer:
    """Cache a Stage-0 checkpoint already bound to the passed gate report.

    A missing or corrupt checkpoint still refuses scoring; never fall back to
    intrinsic-only scoring after validation."""
    cached = _outcome_model_cache.get(path)
    if cached is not None:
        return cached
    if not path.is_file():
        raise Stage0GateRefused(
            f"Stage-0 outcome model missing at {path}: the load path validated this checkpoint "
            "against the gate report, so a vanished file is a refusal, never a silent "
            "intrinsic-only fallback."
        )
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage0GateRefused(
            f"Stage-0 outcome model at {path} is unreadable: {exc}. "
            "A corrupt checkpoint is a refusal, never an implicit pass."
        ) from exc
    scorer = _OutcomeScorer(OutcomeModel(**state))
    _outcome_model_cache[path] = scorer
    return scorer


def stage0_composite_terms(outcome_model_path: Path, run_dir: Path) -> dict[str, Any] | None:
    """Score merged finding descriptions with the validated Stage-0 rubric.

    No model or no described findings yields None. There is no live FP judge,
    so fp_count remains zero; the returned breakdown records the rubric terms."""
    if outcome_model_path == Path(""):
        return None
    findings: list[dict[str, Any]] = [
        {"text": str(item["description"]), "verdict": None, "tools": []}
        for item in _merged_items(run_dir)
        if isinstance(item, dict) and item.get("description")
    ]
    if not findings:
        return None
    result = _score_rubric_review(
        _load_outcome_model(outcome_model_path),
        findings=findings,
        fp_count=0,
        total_findings=len(findings),
        breakdown=True,
    )
    assert isinstance(result, RubricV2Breakdown)
    return result.to_dict()


def _archive_root(trace: vf.Trace) -> str:
    """Archive root the harness told daydream to use, for this rollout."""
    return str(trace.info.get("daydream_archive_root") or DEFAULT_ARCHIVE_ROOT)


def _repo_path(trace: vf.Trace) -> str:
    """Path of the repository under review inside the sandbox."""
    return str(trace.info.get("daydream_repo_path") or DEFAULT_REPO_PATH)


def _read_json(path: Path, *, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default



def _merged_items(run_dir: Path | None) -> list[Any]:
    """Read the shared finding list without filtering shape-metric inputs."""
    if run_dir is None:
        return []
    merged = _read_json(run_dir / "deep" / "merged-items.json", default={})
    return (merged.get("items") or []) if isinstance(merged, dict) else []




#: Ignore-rule files and Python's root startup hook are part of the oracle.
#: A new sitecustomize.py can exit before tests run; ignore rules can hide files.
ORACLE_IGNORE_PATHSPECS = ["sitecustomize.py", ":(glob)**/.gitignore"]

#: The suite's own bytecode is benign. Explicit exclusions keep agent-controlled
#: ignore rules out of the decision when listing all untracked protected files.
ORACLE_BENIGN_PATHSPECS = [":(exclude,glob)**/__pycache__/**", ":(exclude,glob)**/*.py[cod]"]


async def _probe(
    runtime: vf.Runtime,
    argv: list[str],
    changed: Callable[[vf.ProgramResult], bool],
) -> bool:
    """Return whether one oracle probe passes its fail-closed predicate."""
    return not changed(await runtime.run(argv, {}))


async def _fixes_applied(runtime: vf.Runtime, repo: str, head_sha: str) -> bool:
    """Detect tracked product changes, excluding .daydream artifacts.

    A dirty tracked tree or a committed tree differing from the baked snapshot
    counts. An empty commit does not. New uncommitted files deliberately do not
    count, and Git errors read as no fix. recommended.patch is not evidence:
    it can include Daydream artifacts even when the product tree is unchanged."""
    dirty = await runtime.run(
        [
            "git",
            "-C",
            repo,
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--",
            DAYDREAM_EXCLUDE,
        ],
        {},
    )
    if dirty.exit_code == 0 and dirty.stdout.strip():
        return True
    # A clean tree can contain a committed fix. Compare contents, not ref movement:
    # an empty commit and every Git exit code except 1 still mean no fix.
    diff = await runtime.run(
        candidate_quiet_diff_cmd(repo, head_sha, [DAYDREAM_EXCLUDE], include_head=True),
        {},
    )
    # ``runtime`` is the untyped ``vf.Runtime`` boundary, so exit codes are
    # Any; coerce to int so the fail-closed boolean stays concrete.
    return int(diff.exit_code) == 1


async def _protected_test_paths_unchanged(
    runtime: vf.Runtime, repo: str, head_sha: str, protected_test_paths: list[str]
) -> bool:
    """Verify the test oracle against the green baked snapshot before running it.

    Every probe fails closed, including Git errors. The declared paths also
    cover .gitignore files and root sitecustomize.py (a Python startup hook).

    - Diff the working tree against head_sha without HEAD, so committed, staged
      and unstaged tampering all count. Disable external diff and textconv as
      defense in depth even on Git versions where --quiet already ignores them.
    - Reject skip-worktree/assume-unchanged flags: they hide worktree tampering
      by making Git read the index instead.
    - List every untracked protected file without --exclude-standard. Ignore
      rules are agent-controlled; only explicit bytecode exclusions are trusted.
      core.excludesFile= also disables repository-local and global ignore files.
    - Separately catch ignored .gitignore files, including self-ignoring rules.
    - Require .git/info/exclude to remain comments-only.

    A gutted or unverifiable oracle must never report honest non-regression."""
    oracle_pathspecs = [*protected_test_paths, *ORACLE_IGNORE_PATHSPECS]

    def diff_changed(result: vf.ProgramResult) -> bool:
        return int(result.exit_code) != 0

    def flags_changed(result: vf.ProgramResult) -> bool:
        return result.exit_code != 0 or any(
            line[:1] == "S" or line[:1].islower()
            for line in result.stdout.splitlines()
        )

    def nonempty_changed(result: vf.ProgramResult) -> bool:
        # Any listed file — or any probe error — means the oracle changed.
        return result.exit_code != 0 or bool(result.stdout.strip())

    def info_exclude_changed(result: vf.ProgramResult) -> bool:
        return result.exit_code != 0 or any(
            line.strip() and not line.lstrip().startswith("#")
            for line in result.stdout.splitlines()
        )

    probes = [
        (
            candidate_quiet_diff_cmd(repo, head_sha, oracle_pathspecs),
            diff_changed,
        ),
        (
            ["git", "-C", repo, "ls-files", "-v", "--", *oracle_pathspecs],
            flags_changed,
        ),
        (
            [
                "git",
                "-C",
                repo,
                "-c",
                "core.excludesFile=",
                "ls-files",
                "--others",
                "--",
                *oracle_pathspecs,
                *ORACLE_BENIGN_PATHSPECS,
            ],
            nonempty_changed,
        ),
        (
            [
                "git",
                "-C",
                repo,
                "-c",
                "core.excludesFile=",
                "ls-files",
                "--others",
                "--ignored",
                "--exclude-standard",
                "--",
                ":(glob)**/.gitignore",
            ],
            nonempty_changed,
        ),
        (
            ["cat", f"{repo}/.git/info/exclude"],
            info_exclude_changed,
        ),
    ]
    for argv, changed in probes:
        if not await _probe(runtime, argv, changed):
            return False
    return True


def _claimed_test_verdict(run_dir: Path | None) -> bool | None:
    """daydream's own ``deep/test-verdict.json`` claim, or ``None`` if absent."""
    if run_dir is None:
        return None
    verdict = _read_json(run_dir / "deep" / "test-verdict.json", default=None)
    if not isinstance(verdict, dict) or not isinstance(verdict.get("passed"), bool):
        return None
    return bool(verdict["passed"])


async def _verifier_identity_available(runtime: vf.Runtime) -> bool:
    """Check for the container's setpriv and distinct non-root verifier user.

    The local subprocess smoke path has no such identity and runs in its staged
    repository instead."""
    result = await runtime.run(
        ["sh", "-c", "command -v setpriv >/dev/null 2>&1 && id verifier >/dev/null 2>&1"], {}
    )
    return int(result.exit_code) == 0


async def _prepare_verify_checkout(runtime: vf.Runtime, repo: str, head_sha: str) -> str | None:
    """Apply the candidate diff to a root-owned, read-only baked-head checkout.

    Return None on construction failure. The container must never fall back to
    running the suite in the agent-mutable repository."""
    verify_dir = f"{repo}-verify"
    # Single derivation site: the candidate diff is derived by the same helper
    # the seal binds (rundir.candidate_diff_cmd), spliced into the atomic sh -c
    # chain because Runtime.run has no stdin. It is applied behind an empty-guard
    # so a genuinely empty diff is a clean no-op (git apply - exits 128 on empty
    # input), while a failed diff short-circuits the && chain to None -- never
    # piping raw/partial output into git apply.
    #
    # The patch file is written into a private mktemp directory (root-only 700)
    # under /tmp, not the agent-writable workspace, so a pre-planted symlink at
    # a predictable path cannot be followed with O_TRUNC as root. A trap ensures
    # the patch directory is removed on every exit path -- success or failure.
    diff_cmd = shlex.join(candidate_diff_cmd(repo, head_sha))
    script = (
        f"patch_dir=$(mktemp -d) || exit 1; "
        f"trap 'rm -rf \"$patch_dir\"' EXIT; "
        f"rm -rf {shlex.quote(verify_dir)} && "
        f"git clone -q {shlex.quote(repo)} {shlex.quote(verify_dir)} && "
        f"git -C {shlex.quote(verify_dir)} checkout -q --detach {shlex.quote(head_sha)} && "
        f"{diff_cmd} > \"$patch_dir/candidate.patch\" && "
        f"( [ ! -s \"$patch_dir/candidate.patch\" ] || "
        f"git -C {shlex.quote(verify_dir)} apply \"$patch_dir/candidate.patch\" ) && "
        f"chown -R root:root {shlex.quote(verify_dir)} && "
        f"chmod -R a-w {shlex.quote(verify_dir)}"
    )
    result = await runtime.run(["sh", "-c", script], {})
    if result.exit_code != 0:
        return None
    return verify_dir

#: Wall-clock ceilings per rollout stage, in seconds. ``harness`` bounds the whole
#: deep loop (daydream's own per-phase wall budget is 1800s); ``scoring`` must fit a
#: full re-run of the repository's test suite.
DEFAULT_TIMEOUT = vf.TaskTimeout(setup=900, harness=5400, scoring=1800)


class GoldenComment(BaseModel):
    """One review comment the upstream bot actually posted on the PR.

    Parsed from the manifest entry's per-PR ``golden_comments`` tables.
    Used only for the non-summed ``golden_overlap`` metric — never a reward.
    """

    model_config = ConfigDict(extra="forbid")
    comment: str
    path: str | None = None
    line: int | None = None
    resolved: bool | None = None
    severity: str | None = None


class DaydreamReviewData(vf.TaskData):
    """One reviewable PR snapshot."""

    repo_slug: str

    clone_url: str
    """Upstream provenance URL, straight from the manifest entry.

    Nothing clones at rollout time — the repository is baked into the task's
    image (D6). The URL the image build mirror-clones is the manifest entry's
    own ``clone_url``, which may differ (the fixture repo uses a sentinel).
    """

    pr_number: int
    base_sha: str
    head_sha: str
    base_ref: str | None = None
    test_command: str
    protected_test_paths: list[str]
    golden_comments: list[GoldenComment] = []


class DaydreamReviewTaskConfig(vf.TaskConfig):
    """Reward weights, overridable as ``--taskset.task.*``."""

    w_composite: float = 1.0

    outcome_model_path: Path = Path("")
    """Stage-0 outcome model checkpoint (M13); stamped from the taskset config at load."""


class DaydreamReviewState(vf.State):
    """Per-rollout state holding the single staged snapshot shared by all signals.

    Verifiers constructs this through the task's StateT; manual traces must
    supply it explicitly."""

    run_dir: Path | None = None
    seal_ok: bool | None = None
    """Whether the staged run dir's supervisor seal verified; ``None`` = no seal.

    ``False`` means a seal existed but did not verify — a tamper — and both the
    intrinsic reward and the non-regression metric must score zero."""


def _review_state(trace: vf.Trace) -> DaydreamReviewState:
    """Require the typed scoring state; bare vf.State cannot hold a staged run."""
    state = trace.state
    if not isinstance(state, DaydreamReviewState):
        raise TypeError(
            f"scoring state must be a DaydreamReviewState, got {type(state).__name__}"
        )
    return state


class DaydreamReviewTask(vf.Task[DaydreamReviewData, DaydreamReviewState, DaydreamReviewTaskConfig]):
    """Score the archived run with the offline intrinsic scorer or validated rubric.

    The supervisor seals artifacts after the agent's write window. score()
    verifies one staged copy plus the current candidate diff before signals run;
    a failed seal zeros reward and non-regression telemetry.

    The suite is a metric, never a reward: a green suite proves non-regression,
    not defect repair. Its protected oracle must match the baked head, and the
    container runs it in a distinct root-owned read-only checkout. The agent's
    prose-derived test verdict is only a claim to compare against that result.

    Correctness requires verifier verdicts from the accepted fix gate. Missing
    verdicts earn no correctness credit; zero findings have no composite and
    score zero. Watch n_findings for a policy learning to say nothing. A clean
    review has no positive floor under this reward contract.

    With a gate-bound outcome model, the Stage-0 rubric replaces the intrinsic
    composite (which remains one rubric term). Golden-comment overlap stays
    telemetry in both modes."""

    async def score(self, trace: vf.Trace, runtime: vf.Runtime | None = None) -> None:
        """Stage and verify the archive once, then share it across scoring signals.

        Without a runtime, defer to the base class's offline replay behavior without
        accessing typed scoring state."""
        if runtime is None:
            await super().score(trace, None)
            return
        state = _review_state(trace)
        with tempfile.TemporaryDirectory(prefix="daydream-rundir-") as staging:
            # The run-dir fetch is scoring work: it runs inside the same
            # TaskError boundary the base class draws around signal evaluation,
            # so a fetch failure (e.g. a missing artifact) is attributed to the
            # task-scoring boundary rather than escaping as a raw OSError.
            async with boundary(vf.TaskError, f"task {type(self).__name__} scoring"):
                state.run_dir = await fetch_run_dir(runtime, Path(staging), _archive_root(trace))
                # Seal verification is host-side over the staged copy; a seal
                # failure is a tamper signal (explicit zero), never a crash.
                # The candidate diff is re-derived from the sandbox here so the
                # seal binds the diff the verifier checkout will actually apply.
                if state.run_dir is not None:
                    if trace.info.get("daydream_seal_ok") is False:
                        # The harness could not produce a seal at all; score the
                        # run as a failed seal, never as an unsealed full-trust
                        # run (rundir.seal_archived_run also writes a fail-closed
                        # marker; this covers even a marker-write failure).
                        state.seal_ok = False
                    else:
                        state.seal_ok = await verify_seal(
                            state.run_dir,
                            runtime,
                            _repo_path(trace),
                            self.data.head_sha,
                            # The harness claims to have sealed this run: a
                            # seal that did not survive to scoring is a vanished
                            # seal (tamper), never the legacy unsealed path.
                            seal_expected=trace.info.get("daydream_seal_ok") is True,
                        )
            if state.seal_ok is not None:
                trace.record_metric("seal_verified", float(state.seal_ok))
            try:
                await super().score(trace, runtime)
            finally:
                state.run_dir = None

    @vf.reward(weight=1.0)
    async def intrinsic_composite(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        """daydream's own trajectory composite over the archived run."""
        state = _review_state(trace)
        # The staged copy must verify against the supervisor's seal before any
        # value is trusted: a tampered archive zeroes the only remaining reward.
        if state.seal_ok is False:
            trace.info["reward_breakdown"] = {"error": "seal verification failed"}
            return 0.0
        run_dir = state.run_dir
        if run_dir is None:
            trace.info["reward_breakdown"] = {"error": "no archived run dir"}
            return 0.0
        breakdown = score_trajectory(assemble_scoring_inputs(run_dir))

        # M13: when a validated Stage-0 outcome model is configured, the reward
        # becomes the rubric composite (which itself carries the intrinsic
        # composite as a term — no double counting); otherwise intrinsic-only.
        stage0 = stage0_composite_terms(self.config.outcome_model_path, run_dir)

        reward_breakdown = breakdown.to_dict() | {
            "reward_version": ROLLOUT_REWARD_VERSION,
            "intrinsic_reward_version": breakdown.reward_version,
        }
        if stage0 is not None:
            reward_breakdown["stage0"] = stage0
        trace.info["reward_breakdown"] = reward_breakdown
        # ``self.config`` rides the untyped ``vf.Task`` boundary; coerce both
        # factors so the composite reward stays a concrete float.
        reward_composite = (
            float(stage0["composite"])
            if stage0 is not None and stage0.get("composite") is not None
            else float(breakdown.composite or 0.0)
        )
        return float(self.config.w_composite) * reward_composite

    @vf.metric
    async def suite_non_regression(self, trace: vf.Trace, runtime: vf.Runtime) -> dict[str, float]:
        """Report whether a real fix preserves the baked green suite; never a reward.

        No fix, a failed seal, a changed oracle, or a Git error yields zero without
        running the mutable test command. Compare any archived prose claim only
        when the suite actually runs; otherwise record a no-fix claim separately."""
        repo = _repo_path(trace)
        if _review_state(trace).seal_ok is False:
            # The archived run cannot be trusted: the oracle is unverifiable, so
            # the tampered result is never recorded as honest non-regression.
            trace.record_metric("test_oracle_unchanged", 0.0)
            return {"suite_non_regression": 0.0}
        if not await _fixes_applied(runtime, repo, self.data.head_sha):
            trace.record_metric("fixes_applied", 0.0)
            # There is no re-run to compare against on this path, but a rollout
            # that changed nothing and still wrote a green test-verdict is the
            # sharpest hack shape there is, so record the bare claim.
            claimed = _claimed_test_verdict(_review_state(trace).run_dir)
            if claimed is not None:
                trace.record_metric("test_claim_passed_without_fix", float(claimed))
            return {"suite_non_regression": 0.0}

        trace.record_metric("fixes_applied", 1.0)
        # Do not execute an agent-modified oracle, even if its suite would pass.
        unchanged = await _protected_test_paths_unchanged(
            runtime, repo, self.data.head_sha, self.data.protected_test_paths
        )
        trace.record_metric("test_oracle_unchanged", float(unchanged))
        if not unchanged:
            return {"suite_non_regression": 0.0}

        result = await self._run_test_command(runtime, repo)
        passed = result.exit_code == 0

        # Compare the claim here: metrics precede rewards, so a separate handler
        # could only obtain this result by rerunning the suite.
        claimed = _claimed_test_verdict(_review_state(trace).run_dir)
        if claimed is not None:
            trace.record_metric("test_claim_mismatch", float(claimed != passed))

        return {"suite_non_regression": float(passed)}

    async def _run_test_command(self, runtime: vf.Runtime, repo: str) -> vf.ProgramResult:
        """Run the suite under the verifier identity in a read-only candidate checkout.

        Failed construction returns nonzero, never a mutable-tree fallback. Only
        the local smoke runtime, lacking the verifier identity, uses its staged repo."""
        if not await _verifier_identity_available(runtime):
            return await runtime.run(
                ["sh", "-c", f"cd {shlex.quote(repo)} && {self.data.test_command}"], {}
            )
        verify_dir = await _prepare_verify_checkout(runtime, repo, self.data.head_sha)
        if verify_dir is None:
            return vf.ProgramResult(exit_code=1, stdout="", stderr="verify checkout construction failed")
        command = f"cd {shlex.quote(verify_dir)} && {self.data.test_command}"
        runner = f"setpriv --no-new-privs --reuid=verifier --regid=verifier --clear-groups sh -c {shlex.quote(command)}"
        return await runtime.run(["sh", "-c", runner], {})

    @vf.metric
    async def review_shape(self, trace: vf.Trace, runtime: vf.Runtime) -> dict[str, float]:
        """Record finding counts and golden-file overlap as telemetry only.

        Overlap is the share of golden comments whose file appears among findings;
        it is a crude localization measure, never a reward."""
        run_dir = _review_state(trace).run_dir
        items = _merged_items(run_dir)

        found_files = {item.get("file") for item in items if isinstance(item, dict)}
        golden_paths = [c.path for c in self.data.golden_comments if c.path]
        overlap = (
            sum(1 for path in golden_paths if path in found_files) / len(golden_paths)
            if golden_paths
            else 0.0
        )
        return {
            "n_findings": float(len(items)),
            "golden_overlap": overlap,
            "n_golden_comments": float(len(golden_paths)),
            "daydream_exit_code": float(trace.info.get("daydream_exit_code", -1)),
        }


class DaydreamReviewConfig(vf.TasksetConfig):
    manifest_path: Path = Path("")
    task: DaydreamReviewTaskConfig = DaydreamReviewTaskConfig()

    gate_report_path: Path = Path("")
    """Path to the Stage-0 gate report (``GateReport.to_dict()`` payload).

    M4: the load path re-validates this report unconditionally, before any
    rollout can be scheduled — a missing, corrupt, or failed report refuses
    the load with :class:`Stage0GateRefused`. Leaving it empty is itself a
    refusal: there is no configuration under which a taskset loads without
    gate evidence.
    """

    outcome_model_path: Path = Path("")
    """Path to the Stage-0 trained outcome model checkpoint (``OutcomeModel.state_dict()``).

    M13: when set, the ``intrinsic_composite`` reward composes the validated
    Stage-0 rubric (``daydream.training.rubric``) over the archived run's
    merged findings, and the reward becomes the rubric composite. When unset,
    scoring stays intrinsic-only (the offline-parity shape).
    """

    use_images: bool = True
    """Stamp the manifest image onto each task.

    A task carrying an ``image`` may only run in a container — verifiers refuses
    the subprocess runtime outright (``verifiers/v1/env.py:189-195``). Set this
    false ONLY for the local subprocess smoke path (``configs/eval-stub.toml``),
    where the repository under review is staged into the runtime workdir instead
    of being baked into an image. Real train/eval runs leave it true; without the
    image there is no green-baseline guarantee and the fix reward is noise.
    """


class _ManifestPR(BaseModel):
    """Pin one PR snapshot; golden comments supply overlap telemetry only."""

    model_config = ConfigDict(extra="forbid")
    pr_number: int
    base_sha: str
    head_sha: str
    base_ref: str | None = None
    golden_comments: list[GoldenComment] = []


class _ManifestEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    clone_url: str
    image: str
    test_command: str
    protected_test_paths: list[str] = Field(min_length=1)
    setup_cmds: list[str] = []
    prs: list[_ManifestPR] = Field(min_length=1)

    @field_validator("protected_test_paths")
    @classmethod
    def _require_literal_paths(cls, paths: list[str]) -> list[str]:
        """Require literal repository-relative paths; reject rather than normalize.

        Git interprets a leading colon as magic, *?[ as glob characters, and . / ..
        components as traversal. Absolute paths also violate the manifest contract."""
        for path in paths:
            if (
                not path
                or path[0] == ":"
                or path.startswith("/")
                or any(component in {".", ".."} for component in path.split("/"))
                or any(ch in path for ch in "*?[")
            ):
                raise ValueError(
                    "protected_test_paths must be LITERAL repository-relative paths "
                    "(nonempty, no leading ':', '/', '.', or '..' components, "
                    "no '*', '?' or '['); got "
                    f"{path!r}"
                )
        return paths


def load_manifest(path: Path) -> dict[str, _ManifestEntry]:
    """Read ``images/manifest.toml`` into ``{repo_slug: entry}``.

    Raises:
        ValueError: If an ``image`` carries an explicit tag. The tag is reserved
            for the task's head SHA so one image is exactly one PR snapshot.
    """
    raw: dict[str, Any] = tomllib.loads(path.read_text(encoding="utf-8"))
    entries = {slug: _ManifestEntry(**body) for slug, body in raw.get("repos", {}).items()}
    # Only the final path segment can carry a tag; `registry:5000/img` is a host:port.
    tagged = sorted(slug for slug, entry in entries.items() if ":" in entry.image.rsplit("/", 1)[-1])
    if tagged:
        raise ValueError(
            f"{path}: image must be a repository name with no tag (the tag is the head SHA); "
            f"tagged entries: {', '.join(tagged)}"
        )
    return entries


def _repo_slug(clone_url: str) -> str:
    """``https://github.com/owner/name`` -> ``owner/name``."""
    parts = clone_url.rstrip("/").removesuffix(".git").split("/")
    return "/".join(parts[-2:])


class DaydreamReviewTaskset(vf.Taskset[DaydreamReviewTask, DaydreamReviewConfig]):
    def load(self) -> list[DaydreamReviewTask]:
        config = self.config
        if config.manifest_path == Path(""):
            raise ValueError("no image manifest: pass --taskset.manifest-path <images/manifest.toml>")

        if not config.use_images:
            logger.warning(
                "use_images is off: tasks carry no image, so nothing guarantees a green baseline "
                "and suite_non_regression is not deterministic. This is the local smoke path only."
            )

        # Stage-0 gate first and unconditionally (M4): no rollout may be
        # scheduled until the offline gate has passed on the trained outcome
        # model. An unconfigured path is itself a refusal — there is no
        # default-to-allowed branch. Empty manifest flags above fail
        # first only because they are argument errors, not gate decisions.
        if config.gate_report_path == Path(""):
            raise Stage0GateRefused(
                "no Stage-0 gate report configured: pass --taskset.gate-report-path <gate.json>. "
                "A Stage-3 run may not schedule rollouts until the offline gate has passed (M4)."
            )
        gate_report = require_stage0_gate(config.gate_report_path)
        # M4 binding: a passed report alone is not enough. When an outcome model
        # is configured, its checkpoint must re-derive the report's
        # evidence_digest (split_digest / model_fingerprint + the report's
        # clear-text measurements) — any-checkpoint-plus-any-report must not
        # cross the Stage-3 boundary. Intrinsic-only runs (no model) have
        # nothing to bind and stay as designed.
        if config.outcome_model_path != Path(""):
            require_outcome_model_bound(gate_report, config.outcome_model_path)

        manifest = load_manifest(config.manifest_path)
        prs = sorted(
            ((slug, entry, pr) for slug, entry in manifest.items() for pr in entry.prs),
            key=lambda item: (item[0], item[2].pr_number),
        )

        # C5 first and unconditionally: an excluded repo must fail the load before
        # any per-PR check can mask it. Slugs are compared
        # case-insensitively — GitHub treats `GetSentry/Sentry` and
        # `getsentry/sentry` as the same repository, and so must this gate.
        excluded = {slug.casefold() for slug in load_exclusion_list()}
        offenders = sorted({slug for slug, _entry, _pr in prs if slug.casefold() in excluded})
        if offenders:
            raise ValueError(
                f"C5 violation: excluded repo(s) in manifest {config.manifest_path}: {', '.join(offenders)}. "
                "These repositories are the held-out benchmark and must never appear in a training or "
                "eval rollout set."
            )

        unbased = sorted(f"{slug}#{pr.pr_number}" for slug, _entry, pr in prs if not pr.base_sha)
        if unbased:
            raise ValueError(
                f"manifest {config.manifest_path} has entry(ies) with no base_sha: {', '.join(unbased)}. "
                "A PR with no pinned base has no reviewable diff and no image to build; fix "
                "the manifest so base_sha is captured."
            )

        tasks: list[DaydreamReviewTask] = []
        for idx, (slug, entry, pr) in enumerate(prs):
            assert pr.base_sha is not None  # narrowed by the `unbased` guard above
            data = DaydreamReviewData(
                idx=idx,
                name=f"{slug}#{pr.pr_number}",
                # Informational only: the daydream CLI takes no prompt. It must still be
                # non-None or the interception server opens a user simulator instead
                # (verifiers 0.2.1 interception/server.py:346-356).
                prompt=f"Deep-review PR #{pr.pr_number} of {slug} @ {pr.head_sha[:12]}",
                image=f"{entry.image}:{pr.head_sha[:12]}" if config.use_images else None,
                timeout=DEFAULT_TIMEOUT,
                repo_slug=slug,
                clone_url=entry.clone_url,
                pr_number=pr.pr_number,
                base_sha=pr.base_sha,
                head_sha=pr.head_sha,
                base_ref=pr.base_ref,
                test_command=entry.test_command,
                protected_test_paths=entry.protected_test_paths,
                golden_comments=pr.golden_comments,
            )
            tasks.append(
                DaydreamReviewTask(
                    data,
                    # The Stage-0 outcome model is taskset-level; stamp it onto
                    # the per-task config the reward functions actually read.
                    config.task.model_copy(update={"outcome_model_path": config.outcome_model_path}),
                )
            )
        return tasks
