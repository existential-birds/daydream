"""Harvested-PR corpus loading, vendored for this standalone subproject.

The legacy ``daydream bench harvest`` surface was removed from the ``daydream``
package (issue-785), which deleted ``daydream/benchmark/corpus.py`` and
``daydream/benchmark/prs.py``. This subproject is the sole remaining consumer of
the harvested-corpus shape (one review bot's historic PR reviews as golden
comments, produced by ``daydream bench harvest``), so it owns a self-contained
copy of the types and loader it needs rather than importing from the removed
benchmark modules.

Exports:
    EvaluablePR: One harvested PR (mirrors the removed ``EvaluablePR``).
    CorpusSource: Resolved corpus root + PR set.
    harvested_corpus: Build a source from a harvest dir's ``index.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvaluablePR:
    """One harvested PR to review.

    Attributes:
        golden_url: Upstream GitHub PR URL; the corpus's dict key for goldens.
        clone_url: HTTPS clone URL of the upstream repository.
        source_repo: Logical repo slug (e.g. ``"sentry"``, ``"grafana"``).
        pr_number: Upstream pull-request number.
        base_sha: Full hex base commit SHA (the bot-review diff base), or
            ``None`` when the base must be derived from ``base_ref``.
        head_sha: Full hex head commit SHA (the bot's review snapshot commit,
            which may be an ancestor of the PR head).
        base_ref: Base branch name (e.g. ``"main"``) used to derive the
            merge-base when ``base_sha`` is ``None``.
    """

    golden_url: str
    clone_url: str
    source_repo: str
    pr_number: int
    base_sha: str | None
    head_sha: str
    base_ref: str | None = None


@dataclass(frozen=True)
class CorpusSource:
    """One resolved harvested corpus."""

    kind: str
    root: Path
    prs: tuple[EvaluablePR, ...]


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
