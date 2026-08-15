"""Taskset: one task per harvested-corpus pull request.

Tasks come from a ``daydream bench harvest``-format corpus directory — never from
the pinned Martian-5 held-out benchmark, whose five repositories are exactly the
SPEC C5 exclusion list. :meth:`DaydreamReviewTaskset.load` enforces that
unconditionally: there is no bypass parameter and no split exception. Train and
eval are two different corpus directories, not a flag.

Each task also needs a container image, a test command and a clone URL, which
come from ``images/manifest.toml`` keyed by repo slug. A corpus PR whose repo has
no manifest entry is an error, not a silent skip: a rollout set that quietly
shrinks is a rollout set nobody can reproduce.
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
from daydream.benchmark.corpus import harvested_corpus
from daydream.training.exclusion import load_exclusion_list
from daydream.training.harvest import assemble_scoring_inputs
from daydream.training.reward import score_trajectory
from pydantic import BaseModel, ConfigDict, Field, field_validator
from verifiers.v1.errors import boundary

from daydream_review_v1.rundir import DEFAULT_ARCHIVE_ROOT, fetch_run_dir

logger = logging.getLogger(__name__)

DEFAULT_REPO_PATH = "/work/repo"


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


def _manifest_row(run_dir: Path) -> dict[str, Any]:
    """Flatten ``manifest.json`` into the flat row shape the scorer expects.

    ``assemble_scoring_inputs(run_dir, row)`` reads ``row["grounding_rate"]``
    (``daydream/training/harvest.py:223``). In the archive that value is nested
    under ``metrics`` (``daydream/archive/manifest.py:213``); it only becomes a
    top-level column when the run is indexed into SQLite
    (``daydream/archive/_schema.py:128``). Reading the manifest verbatim would
    silently null the grounding axis on every rollout.
    """
    manifest = _read_json(run_dir / "manifest.json", default={})
    if not isinstance(manifest, dict):
        return {}
    metrics = manifest.get("metrics") or {}
    return {**manifest, **metrics}


#: Git pathspec (passed as a bare argv element, never shell-interpolated)
#: excluding daydream's own ``.daydream/`` artifacts from the fix signal.
#: Shared by both dirty-tree and moved-HEAD probes so the exclusion set only
#: drifts by intentional edit, never by one string falling out of sync.
DAYDREAM_EXCLUDE = ":(exclude).daydream"

#: Extra pathspecs the oracle probes treat as part of the oracle itself.
#: ``git ls-files --exclude-standard`` honors ignore rules, so a rollout that
#: edits the tracked ``.gitignore`` (or drops a new untracked one) can mask a
#: tampered untracked oracle file from the probes — the ignore files are
#: therefore probed too. ``sitecustomize.py`` is imported from the repository
#: root by every ``python`` invocation ``test_command`` runs (cwd is on
#: ``sys.path``), so an untracked one that ``sys.exit(0)``s makes a suite that
#: never ran look green.
ORACLE_IGNORE_PATHSPECS = ["sitecustomize.py", ":(glob)**/.gitignore"]

#: Pathspecs excluding the suite's own bytecode artifacts from the untracked
#: probe. That probe deliberately runs ``git ls-files --others`` WITHOUT
#: ``--exclude-standard`` (see ``_protected_test_paths_unchanged``), so it lists
#: every untracked file under a protected path — including the ``__pycache__/``
#: and ``*.py[cod]`` files a green suite itself drops while importing the test
#: modules. Without this explicit exclusion a genuinely fixed tree would trip
#: the probe on its own legitimate test runs and be withheld ``w_tests``. The
#: benign exclusions are baked into the probe rather than delegated to the
#: repo's ignore rules, whose decisions this gate deliberately no longer trusts.
ORACLE_BENIGN_PATHSPECS = [":(exclude,glob)**/__pycache__/**", ":(exclude,glob)**/*.py[cod]"]


async def _probe(
    runtime: vf.Runtime,
    argv: list[str],
    changed: Callable[[vf.ProgramResult], bool],
) -> bool:
    """Run one oracle probe; return True iff the oracle is unchanged.

    Every probe shares the same fail-closed run -> check shape: run one command
    against the mutable tree and let the ``changed`` predicate decide — anything
    it flags, including an unusual exit code, reads as an oracle change, never
    as a pass. The predicate is the single place each probe's semantics live, so
    the probe list in :func:`_protected_test_paths_unchanged` reads as a table
    of ``(argv, changed-verdict)`` pairs.
    """
    return not changed(await runtime.run(argv, {}))


async def _fixes_applied(runtime: vf.Runtime, repo: str, head_sha: str) -> bool:
    """Whether the rollout actually changed the code under review.

    Answered from the TRACKED tree — modified files, or a ``HEAD`` that has moved
    past the snapshot the image baked — and never from
    ``.daydream/recommended.patch``. That file is not a fix signal:
    ``capture_recommended_patch`` appends a creation hunk for every untracked
    non-ignored file (``daydream/git_ops.py:842-846``), and daydream writes its
    own ``.daydream/`` directory inside the repository under review. On any
    repository that does not gitignore that directory — which is every repository
    except our own fixture — the patch is non-empty after a rollout that changed
    nothing, and the still-green baseline would hand out a free ``w_tests``.

    Deliberately biased toward false negatives: a fix consisting ONLY of new,
    never-committed files reads as "no fix" and scores ``no_fix_reward``. That
    direction costs a gradient; the other direction corrupts one.

    A clean tree at a moved HEAD counts as a fix only when the *committed
    contents differ* from the snapshot. HEAD advancing on its own — e.g. an
    `--allow-empty` commit that leaves the tree byte-identical — is not a fix
    and scores ``no_fix_reward``. Either way the decision is read from the
    tracked tree, never from daydream's own ``.daydream/`` directory.
    """
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
    # The deep flow commits and pushes once the suite is green, so a clean tree
    # at a moved HEAD is the successful-fix case, not the untouched one. But
    # "moved" is not enough — an empty commit advances HEAD while leaving the
    # committed tree identical to the baked snapshot, so compare the committed
    # contents, not the ref. `git diff --quiet` exits 1 when the trees differ
    # (a fix) and 0 when they are identical (no fix); any other exit (e.g. 128
    # for an unresolvable baked SHA) is treated as no-fix, preserving the
    # deliberate false-negative bias. Both checks exclude `.daydream/` — the
    # agent may commit daydream's own artifacts into the tree, but they are
    # never a fix signal.
    diff = await runtime.run(
        [
            "git",
            "-C",
            repo,
            "diff",
            "--quiet",
            head_sha,
            "HEAD",
            "--",
            DAYDREAM_EXCLUDE,
        ],
        {},
    )
    return diff.exit_code == 1


async def _protected_test_paths_unchanged(
    runtime: vf.Runtime, repo: str, head_sha: str, protected_test_paths: list[str]
) -> bool:
    """Whether the declared test-oracle paths still match the baked head.

    The oracle is the repository's own mutable test infrastructure — test
    sources, runner config, package config — so a rollout could otherwise earn
    ``w_tests`` by rewriting it into a trivial suite. The baked head SHA is the
    trustworthy baseline: the image build proved the suite green at exactly that
    commit before any agent touched the tree.

    Fail-closed by design, with every ambiguity reading as "changed". The probe
    pathspecs are the declared paths plus ``ORACLE_IGNORE_PATHSPECS``: the repo's
    own ignore files and the interpreter-startup hook. The five probes share one
    run -> fail-closed-check shape, expressed as a table of ``(argv, changed)``
    pairs over :func:`_probe`.

    - ``git diff --quiet <head_sha> -- <paths>`` compares the baked head against
      the WORKING TREE (no ``HEAD`` argument — that form would miss uncommitted
      tampering, the exact attack). Exit 0 means no tracked difference
      (committed, staged, unstaged, deleted, or renamed); any other exit — a
      tracked diff (1) or a Git error such as 128 for an unresolvable baked SHA —
      means the oracle changed. A tracked ``.gitignore`` edit is itself a
      tracked difference here: without it, the edit could mask a new untracked
      oracle file from the untracked probe.
    - ``git ls-files -v`` rejects any protected or ignore file carrying the
      ``skip-worktree``/``assume-unchanged`` flag (tag ``S`` or a lowercase
      letter): git then reads the INDEX instead of the worktree, so the diff
      probe above would stay clean against a tampered tracked file with the
      flag set.
    - ``git -c core.excludesFile= ls-files --others -- <paths>`` lists EVERY
      untracked file under a protected path — e.g. a new ``pytest.ini`` or
      ``conftest.py``, an untracked root ``sitecustomize.py`` (imported at
      startup by every ``python`` ``test_command`` runs, so one that
      ``sys.exit(0)``s makes a suite that never ran look green), or a new
      untracked ``.gitignore`` (the ignore-rule attack). It deliberately does
      NOT pass ``--exclude-standard``: a protected path the baked tree's own
      ignore rules already cover would otherwise ship inert protection, and a
      freshly planted ignore rule could hide the file. Listing everything makes
      the baked tree's ignore decisions irrelevant to the probe; the suite's
      own runtime byproducts (``__pycache__/`` and ``*.py[cod]``) are excluded
      explicitly via ``ORACLE_BENIGN_PATHSPECS`` instead.
    - The ``core.excludesFile=`` override above also neutralizes the two ignore
      sources ``--exclude-standard`` would otherwise honor with no probe here to
      vouch for them: the repo-local ``core.excludesFile`` set in ``.git/config``
      and the global excludes file (default ``$HOME/.config/git/ignore``). Both
      are agent-writable, so without this a rollout could mask an untracked
      oracle file behind either and pass every probe. An empty value loads
      neither — the default-path fallback only applies when the key is unset —
      so the same override rides on the ``--ignored`` probe below.
    - ``git -c core.excludesFile= ls-files --others --ignored ... --
      ':(glob)**/.gitignore'`` catches a ``.gitignore`` that ignores itself
      (e.g. a bare ``*``), which hides from the listing above.
    - ``cat .git/info/exclude`` verifies the rollout-controlled per-repo ignore
      file. A fresh clone's file is comments-only; any real rule means the
      oracle changed.

    There is deliberately no case that defaults to pass on an error — an
    unverifiable oracle never earns the test reward.
    """
    oracle_pathspecs = [*protected_test_paths, *ORACLE_IGNORE_PATHSPECS]

    def diff_changed(result: vf.ProgramResult) -> bool:
        return result.exit_code != 0

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
            ["git", "-C", repo, "diff", "--quiet", head_sha, "--", *oracle_pathspecs],
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

#: Wall-clock ceilings per rollout stage, in seconds. ``harness`` bounds the whole
#: deep loop (daydream's own per-phase wall budget is 1800s); ``scoring`` must fit a
#: full re-run of the repository's test suite.
DEFAULT_TIMEOUT = vf.TaskTimeout(setup=900, harness=5400, scoring=1800)


class GoldenComment(BaseModel):
    """One review comment the upstream bot actually posted on the PR.

    Shape mirrors ``daydream/benchmark/harvest.py`` ``build_harvested_corpus``.
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
    """Upstream provenance URL, straight from the corpus record.

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
    w_tests: float = 1.0
    no_fix_reward: float = 0.0


class DaydreamReviewState(vf.State):
    """Mutable per-rollout scoring state, living on the trace.

    Production traces carry one automatically — the rollout resolves the task's
    ``StateT`` through the MRO (``verifiers/v1/rollout.py:112-116``) — and the
    overridden :meth:`DaydreamReviewTask.score` holds the single host-side
    snapshot of the archived run dir here, so every reward/metric reads the
    same copy instead of re-fetching one.
    """

    run_dir: Path | None = None


def _review_state(trace: vf.Trace) -> DaydreamReviewState:
    """The trace's scoring state, or a loud error for a base ``State``.

    A test helper or consumer that scores without a ``DaydreamReviewState``
    (e.g. by constructing a bare ``vf.Trace``) would otherwise silently
    dereference a missing ``run_dir``. Fail loudly instead.
    """
    state = trace.state
    if not isinstance(state, DaydreamReviewState):
        raise TypeError(
            f"scoring state must be a DaydreamReviewState, got {type(state).__name__}"
        )
    return state


class DaydreamReviewTask(vf.Task[DaydreamReviewData, DaydreamReviewState, DaydreamReviewTaskConfig]):
    """Two reward axes: daydream's own intrinsic composite, and the test suite.

    ``intrinsic_composite`` replays the archived run through
    :func:`daydream.training.reward.score_trajectory` — the exact scorer the
    offline training pipeline uses, imported rather than reimplemented, so an
    online reward and an offline label can never disagree about the same run.

    ``fix_tests_pass`` is the ground truth the intrinsic composite cannot supply:
    the repository's own suite, re-run inside the still-live sandbox. daydream's
    own test verdict is a regex over agent prose
    (``daydream/agent.py:252-300``), so it is recorded as a claim and compared
    against the re-run — never trusted as reward.

    Important: ``verifier_verdicts`` exist only when the fix gate was accepted
    (``deep/recommendation-verdicts.json`` is written at
    ``daydream/deep/orchestrator.py:1213-1229``). A review-only rollout therefore
    scores on grounding and format alone. That is the designed behaviour, and
    ``trace.info["reward_breakdown"]["axes_present"]`` records it per rollout.

    The #91 Stage-0 preference rubric is deliberately NOT a reward here. Its seam
    is ``TaskConfig.judges`` (or an additional ``@vf.reward``); until it passes
    #91's offline ranking gate, golden-comment agreement is exposed only as the
    non-summed ``golden_overlap`` metric.

    A rollout that reports ZERO findings scores ``intrinsic_composite`` 0.0, not
    1.0. ``analyze_grounding`` returns ``grounding_rate = None`` over an empty
    finding set (undefined, not perfect), so no credit axis is present and
    ``score_trajectory`` returns ``composite = None``, mapped to 0.0 below. This
    was a live defect — a codex rollout with 27 captured turns and 0 findings
    scored 1.0 — fixed at the write chokepoint in ``daydream/eval/analyzer.py``
    so the offline corpus and this reward agree. Archived runs scored before that
    fix keep their 1.0 and were not migrated.

    A correct "nothing wrong here" therefore scores the same as a broken run.
    Any positive floor for a genuinely clean review is reward design and belongs
    to the #91 Stage-0 rubric, not here. Keep watching ``n_findings`` alongside
    the reward regardless: reward climbing while ``n_findings`` falls is still
    the signal that the policy is learning to say nothing.
    """

    async def score(self, trace: vf.Trace, runtime: vf.Runtime | None = None) -> None:
        """Score *trace*, staging the archived run dir into the state exactly once.

        The single run-dir fetch happens here, at the entrypoint; the consumers
        read the staged snapshot off ``state.run_dir`` and never re-enter the
        runtime. With no runtime the base class simply skips the
        runtime-dependent signals, and there is nothing to stage.
        """
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
            try:
                await super().score(trace, runtime)
            finally:
                state.run_dir = None

    @vf.reward(weight=1.0)
    async def intrinsic_composite(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        """daydream's own trajectory composite over the archived run."""
        run_dir = _review_state(trace).run_dir
        if run_dir is None:
            trace.info["reward_breakdown"] = {"error": "no archived run dir"}
            return 0.0
        breakdown = score_trajectory(assemble_scoring_inputs(run_dir, _manifest_row(run_dir)))

        trace.info["reward_breakdown"] = {
            "correctness_per_finding": breakdown.correctness_per_finding,
            "grounding": breakdown.grounding,
            "format_valid": breakdown.format_valid,
            "length_penalty": breakdown.length_penalty,
            "composite": breakdown.composite,
            "axes_present": breakdown.axes_present,
            "reward_version": breakdown.reward_version,
        }
        return self.config.w_composite * (breakdown.composite or 0.0)

    @vf.reward(weight=1.0)
    async def fix_tests_pass(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        """Re-run the repository's pinned suite against the fixed tree.

        Deterministic because the image build proved the same command green at
        the same commit before any agent touched it (D6). A rollout that applied
        no fix is not evidence either way, so it takes ``no_fix_reward`` rather
        than a free 1.0 for leaving the tree alone.

        The declared ``protected_test_paths`` oracle must still match the baked
        head before ``test_command`` is trusted: any oracle change — a tracked
        difference, an untracked file under a protected path or a root
        ``sitecustomize.py``, a ``skip-worktree``/``assume-unchanged`` flag, an
        ignore-rule change (``.gitignore`` files or ``.git/info/exclude``), or a
        Git error — returns a literal ``0.0`` without running the repository's
        mutable ``test_command``, so a rollout can never pay itself the test
        reward by gutting its own suite.
        """
        repo = _repo_path(trace)
        if not await _fixes_applied(runtime, repo, self.data.head_sha):
            trace.record_metric("fixes_applied", 0.0)
            # There is no re-run to compare against on this path, but a rollout
            # that changed nothing and still wrote a green test-verdict is the
            # sharpest hack shape there is, so record the bare claim.
            claimed = _claimed_test_verdict(_review_state(trace).run_dir)
            if claimed is not None:
                trace.record_metric("test_claim_passed_without_fix", float(claimed))
            return self.config.no_fix_reward

        trace.record_metric("fixes_applied", 1.0)
        # Security boundary: the test oracle must match the baked head before the
        # repository's own mutable test_command is trusted. A changed oracle
        # (committed/staged/unstaged/deleted/renamed, an untracked protected
        # file or root sitecustomize.py, a skip-worktree/assume-unchanged flag,
        # an ignore-rule change, or a Git error) earns a literal zero WITHOUT
        # running test_command.
        unchanged = await _protected_test_paths_unchanged(
            runtime, repo, self.data.head_sha, self.data.protected_test_paths
        )
        trace.record_metric("test_oracle_unchanged", float(unchanged))
        if not unchanged:
            return 0.0

        result = await runtime.run(
            ["sh", "-c", f"cd {shlex.quote(repo)} && {self.data.test_command}"], {}
        )
        passed = result.exit_code == 0

        # Reward-hack tripwire: daydream's own prose-derived verdict versus what
        # the suite actually does. Recorded here rather than as a @vf.metric
        # because metrics run BEFORE rewards (verifiers task.py:299-306), so a
        # metric cannot see this re-run without paying for a second one.
        claimed = _claimed_test_verdict(_review_state(trace).run_dir)
        if claimed is not None:
            trace.record_metric("test_claim_mismatch", float(claimed != passed))

        return self.config.w_tests * float(passed)

    @vf.metric
    async def review_shape(self, trace: vf.Trace, runtime: vf.Runtime) -> dict[str, float]:
        """Observability only — never summed into the reward.

        ``golden_overlap`` is an explicitly crude localisation proxy: the share of
        the bot's golden comments whose file appears among daydream's merged
        findings. It exists to inform #91's rubric design, not to grade a rollout.
        """
        run_dir = _review_state(trace).run_dir
        items: list[dict[str, Any]] = []
        if run_dir is not None:
            merged = _read_json(run_dir / "deep" / "merged-items.json", default={})
            if isinstance(merged, dict):
                items = merged.get("items") or []

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
    corpus_dir: Path = Path("")
    manifest_path: Path = Path("")
    task: DaydreamReviewTaskConfig = DaydreamReviewTaskConfig()

    use_images: bool = True
    """Stamp the manifest image onto each task.

    A task carrying an ``image`` may only run in a container — verifiers refuses
    the subprocess runtime outright (``verifiers/v1/env.py:189-195``). Set this
    false ONLY for the local subprocess smoke path (``configs/eval-stub.toml``),
    where the repository under review is staged into the runtime workdir instead
    of being baked into an image. Real train/eval runs leave it true; without the
    image there is no green-baseline guarantee and the fix reward is noise.
    """


class _ManifestEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    clone_url: str
    image: str
    test_command: str
    protected_test_paths: list[str] = Field(min_length=1)
    setup_cmds: list[str] = []

    @field_validator("protected_test_paths")
    @classmethod
    def _require_literal_paths(cls, paths: list[str]) -> list[str]:
        """Reject entries git would re-interpret instead of matching byte-for-byte.

        The manifest promises LITERAL repository-relative paths, but the scoring
        gate passes each entry to git as a bare pathspec (``_protected_test_paths_unchanged``),
        where ``*``, ``?`` and ``[`` are glob metacharacters and a leading ``:``
        is pathspec magic. A glob-shaped entry that matches nothing would read as
        a clean diff and an empty ``ls-files`` list, letting ``test_command`` run
        against an unprotected oracle — so such entries must fail the load rather
        than ship a silently-inert protection.
        """
        for path in paths:
            if not path or path[0] == ":" or any(ch in path for ch in "*?["):
                raise ValueError(
                    "protected_test_paths must be LITERAL repository-relative paths "
                    "(nonempty, no leading ':', no '*', '?' or '['); got "
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


def _load_golden_comments(corpus_dir: Path) -> dict[str, list[GoldenComment]]:
    """Read ``results/benchmark_data.json``, keyed by golden URL.

    Raises:
        ValueError: If the file is absent. ``daydream bench harvest`` always
            writes it (``daydream/benchmark/harvest.py:379``), so its absence
            means a truncated or hand-rolled corpus — defaulting to no golden
            comments would silently zero the ``golden_overlap`` metric instead.
    """
    path = corpus_dir / "results" / "benchmark_data.json"
    if not path.exists():
        raise ValueError(
            f"corpus {corpus_dir} has no results/benchmark_data.json; "
            "re-run `daydream bench harvest` to produce a complete corpus"
        )
    corpus: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return {
        url: [GoldenComment(**comment) for comment in entry.get("golden_comments", [])]
        for url, entry in corpus.items()
    }


class DaydreamReviewTaskset(vf.Taskset[DaydreamReviewTask, DaydreamReviewConfig]):
    def load(self) -> list[DaydreamReviewTask]:
        config = self.config
        if config.corpus_dir == Path(""):
            raise ValueError("no corpus directory: pass --taskset.corpus-dir <harvested corpus dir>")
        if config.manifest_path == Path(""):
            raise ValueError("no image manifest: pass --taskset.manifest-path <images/manifest.toml>")

        if not config.use_images:
            logger.warning(
                "use_images is off: tasks carry no image, so nothing guarantees a green baseline "
                "and fix_tests_pass is not deterministic. This is the local smoke path only."
            )

        source = harvested_corpus(config.corpus_dir)
        prs = sorted(source.prs, key=lambda pr: (_repo_slug(pr.clone_url), pr.pr_number))

        # harvested_corpus() drops records with no review_commit_id (daydream
        # benchmark/corpus.py:73) — there is no snapshot to replay. Say so out loud
        # rather than letting a corpus quietly shrink between harvest and rollout.
        indexed = len(json.loads((config.corpus_dir / "index.json").read_text(encoding="utf-8")).get("prs", []))
        if indexed > len(prs):
            logger.warning(
                "corpus %s indexes %d PR(s) but only %d have a review snapshot commit; "
                "the rest have no head SHA to replay and are not rollout tasks",
                config.corpus_dir,
                indexed,
                len(prs),
            )

        # C5 first and unconditionally: an excluded repo must fail the load before
        # any manifest or per-record check can mask it. Slugs are compared
        # case-insensitively — GitHub treats `GetSentry/Sentry` and
        # `getsentry/sentry` as the same repository, and so must this gate.
        excluded = {slug.casefold() for slug in load_exclusion_list()}
        offenders = sorted({slug for pr in prs if (slug := _repo_slug(pr.clone_url)).casefold() in excluded})
        if offenders:
            raise ValueError(
                f"C5 violation: excluded repo(s) in corpus {config.corpus_dir}: {', '.join(offenders)}. "
                "These repositories are the held-out benchmark and must never appear in a training or "
                "eval rollout set."
            )

        unbased = sorted(f"{_repo_slug(pr.clone_url)}#{pr.pr_number}" for pr in prs if not pr.base_sha)
        if unbased:
            raise ValueError(
                f"corpus {config.corpus_dir} has record(s) with no base_sha: {', '.join(unbased)}. "
                "A PR with no pinned base has no reviewable diff and no image to build; re-harvest "
                "the corpus so base_sha is captured (daydream/benchmark/harvest.py:355)."
            )

        manifest = load_manifest(config.manifest_path)
        missing = sorted({slug for pr in prs if (slug := _repo_slug(pr.clone_url)) not in manifest})
        if missing:
            raise ValueError(
                f"repo(s) absent from image manifest {config.manifest_path}: {', '.join(missing)}. "
                "Add an entry (or remove the PRs from the corpus) — tasks are never silently skipped."
            )

        golden = _load_golden_comments(config.corpus_dir)
        tasks: list[DaydreamReviewTask] = []
        for idx, pr in enumerate(prs):
            slug = _repo_slug(pr.clone_url)
            entry = manifest[slug]
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
                clone_url=pr.clone_url,
                pr_number=pr.pr_number,
                base_sha=pr.base_sha,
                head_sha=pr.head_sha,
                base_ref=pr.base_ref,
                test_command=entry.test_command,
                protected_test_paths=entry.protected_test_paths,
                golden_comments=golden.get(pr.golden_url, []),
            )
            tasks.append(DaydreamReviewTask(data, config.task))
        return tasks
