"""Build the committed manifest for a harvested bot-review corpus.

A harvest dir holds the full payloads (``harvest/pr-*.json`` and
``results/``) that are too heavy to track, plus a small ``index.json``
inventory. Neither alone gives a reviewer a quick sense of the corpus, and the
payloads are gitignored, so ``daydream bench manifest`` folds them into one
compact, git-tracked ``manifest.json``:

- corpus-level counts (PRs, golden comments, resolved threads) so the corpus
  can be sanity-checked and trended across re-harvests,
- per-PR snapshot commit plus comment/resolved counts, and
- the golden comment identifiers (id + path + line) — the exact set scoring is
  measured against. Only the identifiers are committed, keeping the manifest
  compact rather than duplicating the gitignored payload bodies.

The manifest is fully derivable from ``index.json`` + the per-PR harvest
records and is regenerable on demand::

    daydream bench manifest --harvest-dir benchmark/corpora/osprey-coderabbit

``state`` reports the single PR state shared by every indexed PR (``closed``
for a ``--state closed`` harvest), or ``"mixed"`` when the corpus spans states.

Exports:
    build_corpus_manifest: Pure index + harvest-records -> manifest dict.
    write_corpus_manifest: Build and persist ``manifest.json`` atomically.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from daydream.benchmark.benchmark_data import save_benchmark_data

MANIFEST_NAME = "manifest.json"


def build_corpus_manifest(corpus_root: Path, *, harvested_at: str | None = None) -> dict[str, Any]:
    """Build the manifest dict for a harvested corpus rooted at *corpus_root*.

    Reads ``index.json`` for the repo/bot/pr inventory and each indexed PR's
    ``harvest/pr-<N>.json`` record for the golden comment set. ``comment_count``
    and per-PR ``comment_count`` are the golden standalone inline comments (the
    recall denominator); ``resolved_count`` sums the index's review-thread
    resolution flags (the "acted upon" metadata proxy). ``golden_comments``
    keeps each comment's ``id``/``path``/``line`` — enough to pin the golden set
    — not the full body, which stays in the gitignored harvest payloads.

    Args:
        corpus_root: Directory containing ``index.json`` and ``harvest/``.
        harvested_at: ISO date for the ``harvested_at`` field; defaults to the
            current UTC date.

    Returns:
        The manifest dict.

    Raises:
        FileNotFoundError: If ``index.json`` or an indexed PR record is missing;
            a manifest is only meaningful for a complete corpus.
    """
    index = json.loads((corpus_root / "index.json").read_text(encoding="utf-8"))
    prs = index.get("prs", [])

    pr_entries: list[dict[str, Any]] = []
    comment_count = 0
    for pr in prs:
        number = pr["pr_number"]
        record = json.loads((corpus_root / "harvest" / f"pr-{number}.json").read_text(encoding="utf-8"))
        golden = [
            {"id": comment.get("id"), "path": comment.get("path"), "line": comment.get("line")}
            for comment in record.get("comments", [])
        ]
        pr_entries.append(
            {
                "number": number,
                "snapshot_commit": record.get("review_commit_id"),
                "comment_count": len(golden),
                "resolved_count": pr.get("n_resolved_threads", 0),
                "golden_comments": golden,
            }
        )
        comment_count += len(golden)

    states = {pr.get("state") for pr in prs}
    return {
        "harvested_at": harvested_at or datetime.now(UTC).date().isoformat(),
        "repo": index["repo"],
        "bot": index["bot"],
        "state": states.pop() if len(states) == 1 else "mixed",
        "pr_count": len(prs),
        "comment_count": comment_count,
        "resolved_count": sum(pr.get("n_resolved_threads", 0) for pr in prs),
        "prs": pr_entries,
    }


def write_corpus_manifest(corpus_root: Path, *, harvested_at: str | None = None) -> Path:
    """Build the manifest for *corpus_root* and write ``manifest.json`` next to it.

    Writes atomically (sibling temp + ``os.replace``, matching the rest of the
    benchmark package) so a crash mid-write cannot truncate the committed
    manifest.

    Returns:
        The path of the written manifest.
    """
    path = corpus_root / MANIFEST_NAME
    save_benchmark_data(path, build_corpus_manifest(corpus_root, harvested_at=harvested_at))
    return path
