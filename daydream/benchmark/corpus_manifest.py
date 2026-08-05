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

Determinism: ``harvested_at`` is projected from the ``harvested_at`` the
harvest stamped into ``index.json`` (see :func:`~daydream.benchmark.harvest.run_harvest`),
never regenerated at manifest-build time. An older index without that key falls
back to an already-committed ``manifest.json``'s date, so re-running the command
on an unchanged corpus is byte-stable (identical bytes in, identical bytes out).

Validation: a corpora that is incomplete or internally inconsistent is rejected
rather than silently projected — a missing ``prs`` inventory, a harvest record
missing its ``comments`` key, a declared ``n_inline_comments`` or
``review_commit_id`` that disagrees with the record, or a missing per-PR record
all raise before anything is written.

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
            timestamp the harvest stamped into ``index.json``. When that too is
            absent (a legacy index), an already-committed ``manifest.json``'s
            date is reused so a regeneration is byte-stable; only as a last
            resort is the current UTC date used.

    Returns:
        The manifest dict.

    Raises:
        FileNotFoundError: If ``index.json`` or an indexed PR record is missing;
            a manifest is only meaningful for a complete corpus.
        ValueError: If the corpus is internally inconsistent — ``index.json``
            lacks the ``prs`` inventory, a harvest record lacks its ``comments``
            key, or an index-declared ``n_inline_comments``/``review_commit_id``
            disagrees with the record. An interrupted or stale re-harvest must
            fail instead of being projected as hybrid metadata.
    """
    index_path = corpus_root / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if "prs" not in index:
        raise ValueError(
            f"{index_path} is missing the 'prs' inventory; "
            "refusing to fabricate an empty corpus"
        )
    prs = index["prs"]

    pr_entries: list[dict[str, Any]] = []
    comment_count = 0
    for pr in prs:
        number = pr["pr_number"]
        record_path = corpus_root / "harvest" / f"pr-{number}.json"
        record = json.loads(record_path.read_text(encoding="utf-8"))
        if "comments" not in record:
            raise ValueError(
                f"{record_path} is missing the 'comments' key; a corrupt record "
                "must fail, not project as zero golden comments"
            )
        golden = [
            {"id": comment.get("id"), "path": comment.get("path"), "line": comment.get("line")}
            for comment in record["comments"]
        ]
        declared_comments = pr.get("n_inline_comments", 0)
        if len(golden) != declared_comments:
            raise ValueError(
                f"index.json declares n_inline_comments={declared_comments} for PR #{number} "
                f"but {record_path} has {len(golden)} golden comments"
            )
        snapshot_commit = record.get("review_commit_id")
        if snapshot_commit != pr.get("review_commit_id"):
            raise ValueError(
                f"index.json declares review_commit_id={pr.get('review_commit_id')} for PR #{number} "
                f"but {record_path} has {snapshot_commit}"
            )
        pr_entries.append(
            {
                "number": number,
                "snapshot_commit": snapshot_commit,
                "comment_count": len(golden),
                "resolved_count": pr.get("n_resolved_threads", 0),
                "golden_comments": golden,
            }
        )
        comment_count += len(golden)

    states = {pr.get("state") for pr in prs}
    return {
        "harvested_at": _resolve_harvested_at(corpus_root, index, harvested_at),
        "repo": index["repo"],
        "bot": index["bot"],
        "state": states.pop() if len(states) == 1 else "mixed",
        "pr_count": len(prs),
        "comment_count": comment_count,
        "resolved_count": sum(pr.get("n_resolved_threads", 0) for pr in prs),
        "prs": pr_entries,
    }


def _resolve_harvested_at(
    corpus_root: Path, index: dict[str, Any], harvested_at: str | None
) -> str:
    """Pick the ``harvested_at`` date for the manifest, most-specific first.

    Explicit arg > the timestamp the harvest stamped into ``index.json`` > the
    already-committed ``manifest.json``'s date (so a regeneration of a legacy
    corpus stays byte-stable) > the current UTC date.
    """
    if harvested_at is not None:
        return harvested_at
    if index.get("harvested_at"):
        return index["harvested_at"]
    manifest_path = corpus_root / MANIFEST_NAME
    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if existing.get("harvested_at"):
                return existing["harvested_at"]
        except (json.JSONDecodeError, OSError):
            pass
    return datetime.now(UTC).date().isoformat()


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
