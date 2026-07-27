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
from pathlib import Path
from typing import Any

import verifiers.v1 as vf
from daydream.benchmark.corpus import harvested_corpus
from daydream.training.exclusion import load_exclusion_list
from daydream.training.harvest import assemble_scoring_inputs
from daydream.training.reward import score_trajectory
from pydantic import BaseModel

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


async def _claimed_test_verdict(runtime: vf.Runtime, archive_root: str) -> bool | None:
    """daydream's own ``deep/test-verdict.json`` claim, or ``None`` if absent."""
    with tempfile.TemporaryDirectory(prefix="daydream-verdict-") as staging:
        run_dir = await fetch_run_dir(runtime, Path(staging), archive_root)
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
    golden_comments: list[GoldenComment] = []


class DaydreamReviewTaskConfig(vf.TaskConfig):
    """Reward weights, overridable as ``--taskset.task.*``."""

    w_composite: float = 1.0
    w_tests: float = 1.0
    no_fix_reward: float = 0.0


class DaydreamReviewTask(vf.Task[DaydreamReviewData, vf.State, DaydreamReviewTaskConfig]):
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

    KNOWN DEGENERATE OPTIMUM — input for #91, not a bug in this file. A rollout
    that reports ZERO findings scores ``intrinsic_composite`` 1.0: the grounding
    axis is vacuously perfect over an empty finding set, correctness is absent,
    and the length penalty is nil. Observed on a live codex rollout (27 captured
    turns, 0 findings, composite 1.0). ``score_trajectory`` is reused verbatim on
    purpose — an online reward and an offline label must not disagree about the
    same run — so the fix belongs in the rubric #91 owns, not here. Until then,
    watch the ``n_findings`` metric alongside the reward: a training curve where
    reward climbs while ``n_findings`` falls is the policy learning to say
    nothing.
    """

    @vf.reward(weight=1.0)
    async def intrinsic_composite(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        """daydream's own trajectory composite over the archived run."""
        with tempfile.TemporaryDirectory(prefix="daydream-rundir-") as staging:
            run_dir = await fetch_run_dir(runtime, Path(staging), _archive_root(trace))
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
        """
        repo = _repo_path(trace)
        applied = await runtime.run(["sh", "-c", f"test -s {shlex.quote(repo)}/.daydream/recommended.patch"], {})
        if applied.exit_code != 0:
            trace.record_metric("fixes_applied", 0.0)
            return self.config.no_fix_reward

        trace.record_metric("fixes_applied", 1.0)
        result = await runtime.run(
            ["sh", "-c", f"cd {shlex.quote(repo)} && {self.data.test_command}"], {}
        )
        passed = result.exit_code == 0

        # Reward-hack tripwire: daydream's own prose-derived verdict versus what
        # the suite actually does. Recorded here rather than as a @vf.metric
        # because metrics run BEFORE rewards (verifiers task.py:299-306), so a
        # metric cannot see this re-run without paying for a second one.
        claimed = await _claimed_test_verdict(runtime, _archive_root(trace))
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
        with tempfile.TemporaryDirectory(prefix="daydream-shape-") as staging:
            run_dir = await fetch_run_dir(runtime, Path(staging), _archive_root(trace))
            items: list[dict[str, Any]] = []
            if run_dir is not None:
                items = _read_json(run_dir / "deep" / "merged-items.json", default={}).get("items") or []

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
    clone_url: str
    image: str
    test_command: str
    setup_cmds: list[str] = []


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
                golden_comments=golden.get(pr.golden_url, []),
            )
            tasks.append(DaydreamReviewTask(data, config.task))
        return tasks
