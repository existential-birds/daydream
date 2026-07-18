"""Pluggable benchmark corpus sources.

A benchmark run needs exactly two things from its corpus: the directory holding
``results/benchmark_data.json`` and the set of PRs to review. Two sources supply
them:

- ``withmartian`` — the external ``code-review-benchmark`` checkout with the 26
  pinned evaluable PRs (:mod:`daydream.benchmark.prs`).
- ``harvested`` — a directory produced by ``daydream bench harvest``, holding one
  review bot's historic PR reviews as golden comments.

Both share the same on-disk corpus shape, so the injection, scoring, stats, and
trial-isolation layers are corpus-agnostic.

Exports:
    CorpusSource: Resolved corpus root + PR set.
    withmartian_corpus: Build a source from a benchmark-repo checkout.
    harvested_corpus: Build a source from a harvest dir's ``index.json``.
    resolve_corpus: Pick the source implied by a :class:`BenchConfig`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from daydream.benchmark.prs import EvaluablePR, load_evaluable_prs

if TYPE_CHECKING:
    from daydream.benchmark.config import BenchConfig


@dataclass(frozen=True)
class CorpusSource:
    """One resolved benchmark corpus.

    Attributes:
        kind: Which source produced it (``"withmartian"`` / ``"harvested"``).
        root: Directory containing ``results/benchmark_data.json``.
        prs: The PRs to review, before ``--only``/``--limit`` filtering.
    """

    kind: Literal["withmartian", "harvested"]
    root: Path
    prs: tuple[EvaluablePR, ...]


def withmartian_corpus(benchmark_repo: Path) -> CorpusSource:
    """Build the pinned withmartian corpus source rooted at *benchmark_repo*."""
    return CorpusSource(kind="withmartian", root=benchmark_repo, prs=load_evaluable_prs())


def harvested_corpus(harvest_dir: Path) -> CorpusSource:
    """Build a corpus source from a harvest dir's ``index.json``.

    Each indexed PR becomes an :class:`EvaluablePR` whose head is the bot's
    review snapshot commit (``review_commit_id``) and whose base is the PR's
    recorded ``base_sha``. Corpora harvested before ``base_sha`` was captured
    fall back to deriving the base from ``base_ref`` at acquisition time.
    Records without a ``review_commit_id`` have no snapshot to replay and are
    skipped.

    Raises:
        FileNotFoundError: If ``index.json`` is absent.
        KeyError: If the index lacks the ``repo`` slug.
        ValueError: If a replayable record has no ``base_ref``; guessing one
            would silently replay the PR against the wrong base branch.
    """
    index = json.loads((harvest_dir / "index.json").read_text(encoding="utf-8"))
    repo = index["repo"]
    records = [record for record in index.get("prs", []) if record.get("review_commit_id")]
    for record in records:
        if not record.get("base_ref"):
            raise ValueError(f"harvested record for {repo} PR #{record['pr_number']} has no base_ref")
    prs = tuple(
        EvaluablePR(
            golden_url=f"https://github.com/{repo}/pull/{record['pr_number']}",
            clone_url=f"https://github.com/{repo}",
            source_repo=repo,
            pr_number=record["pr_number"],
            base_sha=record.get("base_sha") or None,
            head_sha=record["review_commit_id"],
            base_ref=record["base_ref"],
        )
        for record in records
    )
    return CorpusSource(kind="harvested", root=harvest_dir, prs=prs)


def resolve_corpus(config: BenchConfig) -> CorpusSource:
    """Resolve the corpus source implied by *config*.

    Raises:
        ValueError: If neither ``harvest_dir`` nor ``benchmark_repo`` is set.
    """
    if config.harvest_dir is not None:
        return harvested_corpus(config.harvest_dir)
    if config.benchmark_repo is not None:
        return withmartian_corpus(config.benchmark_repo)
    raise ValueError("BenchConfig needs benchmark_repo or harvest_dir")
