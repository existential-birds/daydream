"""End-to-end tests for a harvested bot-review corpus through the bench CLI.

Real-path: a real local bare git upstream (a filesystem path is a valid git
remote, so there is NO network), a real harvest dir on disk, and entry through
the production entrypoint :func:`daydream.benchmark.cli._handle_bench_command`.
The only mocked seam is ``run_daydream_review`` — the reviewer-subprocess
boundary, the bench harness's equivalent of the ``Backend`` seam.

Acquisition is NOT mocked: the per-PR clone cache is pre-seeded from the local
bare upstream (which is all a prior clone would have produced), so the real
``acquire_checkout`` fetch / detach / merge-base path runs against real git.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from daydream.benchmark.acquire import _cache_subdir_name
from daydream.benchmark.anthropic_score import _DEDUP_SYSTEM, _EXTRACTION_SYSTEM, AnthropicJsonClient
from daydream.benchmark.cli import _handle_bench_command
from daydream.benchmark.judge import _JUDGE_SYSTEM
from daydream.benchmark.score import ANTHROPIC_JUDGE_API_KEY_ENV, JUDGE_BASE_URL_ENV, model_results_dir
from tests.harness.git_helpers import git as _git

#: Slug used for the fixture corpus. The derived ``clone_url``
#: (``https://github.com/<slug>``) is never dialled: the clone cache is
#: pre-seeded, so acquisition finds an existing checkout and skips the clone.
REPO_SLUG = "acme/widgets"
BOT = "coderabbitai[bot]"


@dataclass(frozen=True)
class HarvestedUpstream:
    """A bare upstream with two PRs, one reviewed mid-PR by the bot.

    ``main`` is ``m1 -> m2``. PR 1 forks at ``m1`` with ``a1 -> a2`` and is
    published at ``refs/pull/1/head = a2``, but the bot's review snapshot is
    ``a1`` — so replaying it must derive the merge-base ``m1``, not the base
    branch tip and not the PR head. PR 2 forks at ``m2`` with a single commit
    that is also its review snapshot.
    """

    url: str
    m1: str
    m2: str
    pr1_review_sha: str
    pr1_head_sha: str
    pr2_review_sha: str


def _build_upstream(tmp_path: Path) -> HarvestedUpstream:
    work = tmp_path / "upstream-work"
    work.mkdir(parents=True, exist_ok=True)
    _git(work, "init", "-b", "main")
    _git(work, "config", "user.email", "test@example.com")
    _git(work, "config", "user.name", "Tester")

    def _commit(name: str) -> str:
        (work / name).write_text(f"{name}\n")
        _git(work, "add", name)
        _git(work, "commit", "-m", f"add {name}")
        return _git(work, "rev-parse", "HEAD")

    m1 = _commit("m1.txt")
    _git(work, "checkout", "-b", "pr1")
    a1 = _commit("a1.txt")
    a2 = _commit("a2.txt")
    _git(work, "checkout", "main")
    m2 = _commit("m2.txt")
    _git(work, "checkout", "-b", "pr2")
    b1 = _commit("b1.txt")
    _git(work, "checkout", "main")

    _git(work, "update-ref", "refs/pull/1/head", a2)
    _git(work, "update-ref", "refs/pull/2/head", b1)

    bare = tmp_path / "upstream.git"
    _git(work, "clone", "--bare", str(work), str(bare))
    _git(bare, "update-ref", "refs/pull/1/head", a2)
    _git(bare, "update-ref", "refs/pull/2/head", b1)

    return HarvestedUpstream(
        url=str(bare), m1=m1, m2=m2, pr1_review_sha=a1, pr1_head_sha=a2, pr2_review_sha=b1
    )


def _golden_url(pr_number: int) -> str:
    return f"https://github.com/{REPO_SLUG}/pull/{pr_number}"


def _seed_harvest_dir(harvest_dir: Path, upstream: HarvestedUpstream) -> None:
    """Write the on-disk shape ``run_harvest`` emits (index + corpus), no gh."""
    index = {
        "repo": REPO_SLUG,
        "bot": BOT,
        "n_prs_with_bot_activity": 2,
        "prs": [
            {
                "pr_number": 1,
                "title": "first",
                "state": "closed",
                "merged": True,
                "base_ref": "main",
                "review_commit_id": upstream.pr1_review_sha,
                "n_inline_comments": 1,
                "n_review_summaries": 1,
                "n_resolved_threads": 1,
                "threads_complete": True,
            },
            {
                "pr_number": 2,
                "title": "second",
                "state": "open",
                "merged": False,
                "base_ref": "main",
                "review_commit_id": upstream.pr2_review_sha,
                "n_inline_comments": 1,
                "n_review_summaries": 1,
                "n_resolved_threads": 0,
                "threads_complete": True,
            },
        ],
    }
    harvest_dir.mkdir(parents=True, exist_ok=True)
    (harvest_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")

    corpus: dict[str, Any] = {}
    for pr_number, resolved in ((1, True), (2, False)):
        url = _golden_url(pr_number)
        corpus[url] = {
            "pr_url": url,
            "repo_name": REPO_SLUG,
            "golden_comments": [
                {
                    "comment": f"bot finding on PR {pr_number}",
                    "path": "m1.txt",
                    "line": 1,
                    "resolved": resolved,
                    "severity": None,
                }
            ],
            "reviews": [
                {
                    "tool": "coderabbitai",
                    "repo_name": REPO_SLUG,
                    "pr_url": url,
                    "review_comments": [
                        {
                            "path": "m1.txt",
                            "line": 1,
                            "body": f"bot finding on PR {pr_number}",
                            "created_at": "2026-01-01T00:00:00Z",
                        }
                    ],
                }
            ],
        }
    results = harvest_dir / "results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "benchmark_data.json").write_text(json.dumps(corpus, indent=2), encoding="utf-8")


def _seed_clone_cache(cache_dir: Path, upstream: HarvestedUpstream, pr_numbers: tuple[int, ...]) -> None:
    """Pre-populate the per-PR clone cache from the local bare upstream.

    ``acquire_checkout`` clones only when ``<checkout>/.git`` is absent, so
    seeding these directories replaces the network clone and leaves every other
    acquisition step (fetch of ``refs/pull/<N>/head``, detach, merge-base)
    running for real.
    """
    clone_url = f"https://github.com/{REPO_SLUG}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for pr_number in pr_numbers:
        target = cache_dir / _cache_subdir_name(clone_url, pr_number)
        _git(cache_dir, "clone", upstream.url, str(target))


def _merged_item(file: str, line: int, description: str) -> dict[str, Any]:
    return {
        "id": f"{file}:{line}",
        "description": description,
        "file": file,
        "line": line,
        "confidence": "high",
        "rationale": "Because reasons",
        "lens": "correctness",
        "severity": "medium",
    }


@pytest.fixture
def harvested_run(tmp_path, monkeypatch):
    """Build the upstream + harvest dir + seeded cache and capture review calls.

    Returns ``(upstream, harvest_dir, cache_dir, calls)`` where *calls* accrues
    one ``{"checkout", "base_sha"}`` dict per mocked reviewer invocation.
    """
    upstream = _build_upstream(tmp_path)
    harvest_dir = tmp_path / "harvest-corpus"
    _seed_harvest_dir(harvest_dir, upstream)
    cache_dir = tmp_path / "cache"
    _seed_clone_cache(cache_dir, upstream, (1, 2))

    calls: list[dict[str, Any]] = []

    def _fake_review(checkout: Path, **kwargs: Any) -> Path:
        calls.append({"checkout": Path(checkout), "base_sha": kwargs["base_sha"]})
        artifact = Path(checkout) / ".daydream" / "deep" / "merged-items.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        head = _git(Path(checkout), "rev-parse", "HEAD")
        items = [_merged_item("m1.txt", 1, f"daydream finding at {head[:7]}")]
        artifact.write_text(json.dumps({"items": items}), encoding="utf-8")
        return artifact

    monkeypatch.setattr("daydream.benchmark.orchestrator.run_daydream_review", _fake_review)
    # Keep [tool.daydream.bench] in the daydream repo's own pyproject.toml out of
    # the parse (a benchmark-repo key there would collide with --harvest-dir).
    monkeypatch.chdir(tmp_path)
    return upstream, harvest_dir, cache_dir, calls


def test_harvested_corpus_2pr_run_through_cli_injects_reviews(harvested_run):
    upstream, harvest_dir, cache_dir, calls = harvested_run

    rc = _handle_bench_command(
        ["--harvest-dir", str(harvest_dir), "--no-score", "--cache-dir", str(cache_dir)]
    )

    assert rc == 0

    corpus = json.loads((harvest_dir / "results" / "benchmark_data.json").read_text(encoding="utf-8"))
    for pr_number in (1, 2):
        reviews = corpus[_golden_url(pr_number)]["reviews"]
        daydream_reviews = [r for r in reviews if r["tool"] == "daydream"]
        assert len(daydream_reviews) == 1, f"PR {pr_number} has no injected daydream review"
        comments = daydream_reviews[0]["review_comments"]
        assert comments and all(c["body"].strip() for c in comments)
        # The bot's own review survives alongside the daydream arm.
        assert any(r["tool"] == "coderabbitai" for r in reviews)

    # Acquisition really ran: each PR's cached checkout is detached onto the
    # harvested review snapshot, and the review got the derived merge-base.
    assert len(calls) == 2
    by_head = {_git(c["checkout"], "rev-parse", "HEAD"): c["base_sha"] for c in calls}
    assert set(by_head) == {upstream.pr1_review_sha, upstream.pr2_review_sha}
    assert upstream.pr1_head_sha not in by_head  # snapshot, not the PR head

    bare = Path(upstream.url)
    for head, base_sha in by_head.items():
        assert base_sha == _git(bare, "merge-base", "main", head)
    # PR 1's base is the fork point, not the base-branch tip.
    assert by_head[upstream.pr1_review_sha] == upstream.m1
    assert by_head[upstream.pr2_review_sha] == upstream.m2

    # Every run emits the unified JSON report, same shape as a withmartian run.
    report = json.loads(
        (harvest_dir / ".daydream-bench" / "report-daydream.json").read_text(encoding="utf-8")
    )
    assert report["schema_version"] == 1
    assert report["corpus"] == "harvested"
    assert report["corpus_root"] == str(harvest_dir)
    assert report["tool_label"] == "daydream"
    assert [entry["golden_url"] for entry in report["prs"]] == [_golden_url(1), _golden_url(2)]
    assert all(entry["injected_comments"] == 1 for entry in report["prs"])
    assert report["aggregate"] is None  # --no-score


def test_harvested_corpus_scored_run_calls_judge_per_pair(harvested_run, monkeypatch):
    """A scored harvested run reaches the same `FindingJudge` the golden corpus uses.

    The network is mocked at the Anthropic client seam only; extraction, dedup
    and every per-pair ``same_issue`` call run through the production
    ``anthropic-direct`` scoring path.
    """
    _upstream, harvest_dir, cache_dir, calls = harvested_run

    monkeypatch.setenv(ANTHROPIC_JUDGE_API_KEY_ENV, "sk-ant-test")
    monkeypatch.delenv(JUDGE_BASE_URL_ENV, raising=False)

    judged_pairs: list[str] = []

    async def _canned(self, *, system, user, max_tokens):
        if system == _EXTRACTION_SYSTEM:
            return {"issues": ["daydream finding"]}
        if system == _DEDUP_SYSTEM:
            return {"groups": [[0]]}
        assert system == _JUDGE_SYSTEM, f"unexpected judge-client system prompt: {system}"
        judged_pairs.append(user)
        return {"reasoning": "same underlying issue", "match": True, "confidence": 0.95}

    monkeypatch.setattr(AnthropicJsonClient, "complete_json", _canned)

    rc = _handle_bench_command(
        [
            "--harvest-dir",
            str(harvest_dir),
            "--cache-dir",
            str(cache_dir),
            "--score",
            "--judge-route",
            "anthropic-direct",
            "--model",
            "judge-x",
        ]
    )

    assert rc == 0
    assert len(calls) == 2  # both PRs were reviewed before scoring

    evals = json.loads(
        (model_results_dir(harvest_dir, "judge-x") / "evaluations.json").read_text(encoding="utf-8")
    )
    for pr_number in (1, 2):
        leaf = evals[_golden_url(pr_number)]["daydream"]
        assert leaf["judge_route"] == "anthropic-direct"
        assert leaf["judge_model"] == "judge-x"
        assert leaf["errors_count"] == 0
        assert leaf["tp"] == 1 and leaf["fp"] == 0 and leaf["fn"] == 0
        assert leaf["precision"] == 1.0 and leaf["recall"] == 1.0

    # The emitted report carries the aggregate the canned judge produced.
    report = json.loads(
        (harvest_dir / ".daydream-bench" / "report-daydream.json").read_text(encoding="utf-8")
    )
    assert report["schema_version"] == 1 and report["corpus"] == "harvested"
    assert len(report["prs"]) == 2
    assert report["judge_route"] == "anthropic-direct" and report["judge_model"] == "judge-x"
    assert report["aggregate"] == {
        "scored_pr_count": 2,
        "tp": 2,
        "fp": 0,
        "fn": 0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
    }
    assert all(entry["tp"] == 1 and entry["fp"] == 0 and entry["fn"] == 0 for entry in report["prs"])

    # One judge call per (golden, candidate) pair, each carrying both texts.
    assert len(judged_pairs) == 2
    for pr_number, prompt in zip((1, 2), judged_pairs, strict=True):
        assert f"bot finding on PR {pr_number}" in prompt
        assert "daydream finding" in prompt


def test_harvested_rerun_is_idempotent(harvested_run):
    _upstream, harvest_dir, cache_dir, calls = harvested_run
    argv = ["--harvest-dir", str(harvest_dir), "--no-score", "--cache-dir", str(cache_dir)]

    assert _handle_bench_command(argv) == 0
    first = json.loads((harvest_dir / "results" / "benchmark_data.json").read_text(encoding="utf-8"))
    assert len(calls) == 2

    assert _handle_bench_command(argv) == 0
    second = json.loads((harvest_dir / "results" / "benchmark_data.json").read_text(encoding="utf-8"))

    assert second == first
    assert len(calls) == 2  # already-injected PRs are skipped, not re-reviewed
