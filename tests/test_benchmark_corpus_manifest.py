"""Tests for the harvested-corpus manifest.

Covers the pure ``build_corpus_manifest`` projection (counts, golden comments,
resolved flags, mixed-state derivation), the atomic ``write_corpus_manifest``
persistence, and a real-path check against this worktree's committed
``benchmark/corpora/osprey-coderabbit/`` corpus (skipped until the orchestrator
harvest lands).
"""

import json
from pathlib import Path

import pytest

from daydream.benchmark.cli import _handle_bench_command
from daydream.benchmark.corpus_manifest import build_corpus_manifest, write_corpus_manifest

HARVESTED_AT = "2026-08-05"


def _write_fixture(root: Path) -> Path:
    """Write a three-PR harvested corpus fixture under *root* and return it."""
    (root / "harvest").mkdir(parents=True)
    (root / "index.json").write_text(
        json.dumps(
            {
                "repo": "acme/widgets",
                "bot": "cr[bot]",
                "harvested_at": HARVESTED_AT,
                "n_prs_with_bot_activity": 3,
                "prs": [
                    {
                        "pr_number": 1,
                        "state": "closed",
                        "review_commit_id": "a" * 40,
                        "n_inline_comments": 2,
                        "n_resolved_threads": 1,
                    },
                    {
                        "pr_number": 2,
                        "state": "closed",
                        "review_commit_id": "b" * 40,
                        "n_inline_comments": 1,
                        "n_resolved_threads": 0,
                    },
                    # Review-summary-only PR: no inline comments, still indexed.
                    {
                        "pr_number": 3,
                        "state": "closed",
                        "review_commit_id": "c" * 40,
                        "n_inline_comments": 0,
                        "n_resolved_threads": 0,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (root / "harvest" / "pr-1.json").write_text(
        json.dumps(
            {
                "review_commit_id": "a" * 40,
                "comments": [
                    {"id": 10, "path": "a.py", "line": 12},
                    {"id": 11, "path": "b.py", "line": 3},
                ],
            }
        ),
        encoding="utf-8",
    )
    (root / "harvest" / "pr-2.json").write_text(
        json.dumps(
            {
                "review_commit_id": "b" * 40,
                "comments": [{"id": 20, "path": "c.py", "line": 7}],
            }
        ),
        encoding="utf-8",
    )
    (root / "harvest" / "pr-3.json").write_text(
        json.dumps({"review_commit_id": "c" * 40, "comments": []}),
        encoding="utf-8",
    )
    return root


def test_build_corpus_manifest_projects_counts_and_golden_comments(tmp_path):
    manifest = build_corpus_manifest(_write_fixture(tmp_path / "corpus"), harvested_at=HARVESTED_AT)

    assert manifest["harvested_at"] == HARVESTED_AT
    assert manifest["repo"] == "acme/widgets"
    assert manifest["bot"] == "cr[bot]"
    assert manifest["state"] == "closed"
    assert manifest["pr_count"] == 3
    assert manifest["comment_count"] == 3
    assert manifest["resolved_count"] == 1

    pr1, pr2, pr3 = manifest["prs"]
    assert pr1["number"] == 1
    assert pr1["snapshot_commit"] == "a" * 40
    assert pr1["comment_count"] == 2
    assert pr1["resolved_count"] == 1
    # Golden comments are compact identifiers (id/path/line), not full bodies:
    # the bodies live in the gitignored harvest payloads.
    assert pr1["golden_comments"] == [
        {"id": 10, "path": "a.py", "line": 12},
        {"id": 11, "path": "b.py", "line": 3},
    ]
    assert pr2["golden_comments"] == [{"id": 20, "path": "c.py", "line": 7}]
    # Review-summary-only PR keeps its index entry with an empty golden set.
    assert pr3["number"] == 3
    assert pr3["comment_count"] == 0
    assert pr3["golden_comments"] == []


def test_build_corpus_manifest_reports_mixed_state(tmp_path):
    corpus = _write_fixture(tmp_path / "corpus")
    index = json.loads((corpus / "index.json").read_text(encoding="utf-8"))
    index["prs"][1]["state"] = "open"
    (corpus / "index.json").write_text(json.dumps(index), encoding="utf-8")

    manifest = build_corpus_manifest(corpus, harvested_at=HARVESTED_AT)
    assert manifest["state"] == "mixed"


def test_build_corpus_manifest_tolerates_missing_anchor(tmp_path):
    corpus = _write_fixture(tmp_path / "corpus")
    record = json.loads((corpus / "harvest" / "pr-2.json").read_text(encoding="utf-8"))
    record["comments"][0]["line"] = None
    record["comments"][0].pop("path")
    (corpus / "harvest" / "pr-2.json").write_text(json.dumps(record), encoding="utf-8")

    manifest = build_corpus_manifest(corpus, harvested_at=HARVESTED_AT)
    assert manifest["prs"][1]["golden_comments"] == [{"id": 20, "path": None, "line": None}]


def test_build_corpus_manifest_requires_complete_corpus(tmp_path):
    corpus = _write_fixture(tmp_path / "corpus")
    (corpus / "harvest" / "pr-1.json").unlink()
    with pytest.raises(FileNotFoundError):
        build_corpus_manifest(corpus, harvested_at=HARVESTED_AT)


def test_write_corpus_manifest_persists_manifest(tmp_path):
    corpus = _write_fixture(tmp_path / "corpus")
    path = write_corpus_manifest(corpus, harvested_at=HARVESTED_AT)
    assert path == corpus / "manifest.json"
    assert path.exists()
    assert json.loads(path.read_text(encoding="utf-8")) == build_corpus_manifest(
        corpus, harvested_at=HARVESTED_AT
    )


def test_manifest_subcommand_writes_manifest(tmp_path):
    corpus = _write_fixture(tmp_path / "corpus")
    assert _handle_bench_command(["manifest", "--harvest-dir", str(corpus)]) == 0
    assert (corpus / "manifest.json").exists()


def test_manifest_subcommand_fails_on_incomplete_corpus(tmp_path):
    corpus = _write_fixture(tmp_path / "corpus")
    (corpus / "index.json").unlink()
    assert _handle_bench_command(["manifest", "--harvest-dir", str(corpus)]) == 2
    assert not (corpus / "manifest.json").exists()


def test_build_corpus_manifest_rejects_missing_prs_inventory(tmp_path):
    corpus = _write_fixture(tmp_path / "corpus")
    index = json.loads((corpus / "index.json").read_text(encoding="utf-8"))
    index.pop("prs")
    (corpus / "index.json").write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(ValueError, match="missing the 'prs' inventory"):
        build_corpus_manifest(corpus, harvested_at=HARVESTED_AT)


def test_build_corpus_manifest_rejects_missing_comments_key(tmp_path):
    corpus = _write_fixture(tmp_path / "corpus")
    record = json.loads((corpus / "harvest" / "pr-2.json").read_text(encoding="utf-8"))
    record.pop("comments")
    (corpus / "harvest" / "pr-2.json").write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="missing the 'comments' key"):
        build_corpus_manifest(corpus, harvested_at=HARVESTED_AT)


def test_build_corpus_manifest_rejects_inline_count_mismatch(tmp_path):
    corpus = _write_fixture(tmp_path / "corpus")
    index = json.loads((corpus / "index.json").read_text(encoding="utf-8"))
    index["prs"][0]["n_inline_comments"] = 5
    (corpus / "index.json").write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(ValueError, match="n_inline_comments=5"):
        build_corpus_manifest(corpus, harvested_at=HARVESTED_AT)


def test_build_corpus_manifest_rejects_snapshot_commit_mismatch(tmp_path):
    corpus = _write_fixture(tmp_path / "corpus")
    index = json.loads((corpus / "index.json").read_text(encoding="utf-8"))
    index["prs"][0]["review_commit_id"] = "d" * 40
    (corpus / "index.json").write_text(json.dumps(index), encoding="utf-8")

    with pytest.raises(ValueError, match="review_commit_id"):
        build_corpus_manifest(corpus, harvested_at=HARVESTED_AT)


def test_manifest_subcommand_fails_on_inconsistent_corpus(tmp_path):
    corpus = _write_fixture(tmp_path / "corpus")
    index = json.loads((corpus / "index.json").read_text(encoding="utf-8"))
    index["prs"][0]["n_inline_comments"] = 5
    (corpus / "index.json").write_text(json.dumps(index), encoding="utf-8")

    assert _handle_bench_command(["manifest", "--harvest-dir", str(corpus)]) == 2
    assert not (corpus / "manifest.json").exists()


def test_write_corpus_manifest_is_deterministic(tmp_path):
    corpus = _write_fixture(tmp_path / "corpus")
    first = write_corpus_manifest(corpus, harvested_at=HARVESTED_AT)
    second = write_corpus_manifest(corpus, harvested_at=HARVESTED_AT)
    assert first.read_bytes() == second.read_bytes()


def test_manifest_projects_harvested_at_from_index(tmp_path):
    corpus = _write_fixture(tmp_path / "corpus")
    manifest = build_corpus_manifest(corpus)
    assert manifest["harvested_at"] == HARVESTED_AT


def test_manifest_reuses_committed_date_for_legacy_index(tmp_path):
    corpus = _write_fixture(tmp_path / "corpus")
    index = json.loads((corpus / "index.json").read_text(encoding="utf-8"))
    index.pop("harvested_at")
    (corpus / "index.json").write_text(json.dumps(index), encoding="utf-8")
    write_corpus_manifest(corpus, harvested_at="2026-08-04")

    assert build_corpus_manifest(corpus)["harvested_at"] == "2026-08-04"
    # Regenerating a legacy index must not churn the committed date.
    assert write_corpus_manifest(corpus).read_bytes() == write_corpus_manifest(corpus).read_bytes()


_CORPUS_ROOT = Path(__file__).resolve().parents[1] / "benchmark" / "corpora" / "osprey-coderabbit"


def test_real_osprey_coderabbit_manifest_loads():
    """Real-path check against the committed CodeRabbit-parity corpus.

    The committed ``manifest.json`` must be a faithful projection of the
    git-tracked ``index.json`` inventory — PR order and numbers, per-PR
    ``comment_count``/``resolved_count``, corpus ``pr_count``/``comment_count``
    — so removing a PR (while adjusting totals), swapping snapshot commits, or
    replacing golden IDs while preserving counts would fail this check in CI.
    The harvest payloads themselves are gitignored; when they are present
    locally, the manifest is additionally required to be identical to a full
    rebuild, pinning snapshot commits and golden comment id/path/line.
    """
    manifest_path = _CORPUS_ROOT / "manifest.json"
    index_path = _CORPUS_ROOT / "index.json"
    if not manifest_path.exists() or not index_path.exists():
        pytest.skip("osprey-coderabbit manifest is not committed yet")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    indexed_prs = index["prs"]

    assert manifest["repo"] == index["repo"]
    assert manifest["bot"] == index["bot"]
    assert manifest["state"] == "closed"
    assert manifest["pr_count"] == index["n_prs_with_bot_activity"]
    assert manifest["pr_count"] >= 100  # real harvest: 142 PRs (of 400 scanned)
    assert manifest["comment_count"] == sum(pr["n_inline_comments"] for pr in indexed_prs)
    assert manifest["comment_count"] > 0
    assert manifest["resolved_count"] == sum(pr["n_resolved_threads"] for pr in indexed_prs)
    assert manifest["resolved_count"] == sum(pr["resolved_count"] for pr in manifest["prs"])
    assert [pr["number"] for pr in manifest["prs"]] == [pr["pr_number"] for pr in indexed_prs]
    for pr, indexed in zip(manifest["prs"], indexed_prs, strict=True):
        assert pr["number"] == indexed["pr_number"]
        assert pr["comment_count"] == indexed["n_inline_comments"]
        assert pr["resolved_count"] == indexed["n_resolved_threads"]
        assert pr["snapshot_commit"] == indexed.get("review_commit_id")
    assert all(pr["golden_comments"] for pr in manifest["prs"] if pr["comment_count"] > 0)

    if not (_CORPUS_ROOT / "harvest").is_dir():
        pytest.skip("gitignored harvest/ payloads absent on CI; skipping payload cross-check")

    rebuilt = build_corpus_manifest(_CORPUS_ROOT, harvested_at=manifest["harvested_at"])
    assert rebuilt == manifest
    assert rebuilt["prs"] == manifest["prs"]
