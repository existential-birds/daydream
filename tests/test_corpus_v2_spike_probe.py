"""Spike probe (task 0, issue #1081): force-check the two load-bearing
assumptions of the corpus-v2 enrichment plan against a real-shaped curated
bundle before any task writes code that depends on them:

1. Extraction point — the curated bundle's batch directory
   ``batches/<session_id>/`` carries ``findings.json`` (findings keyed by the
   same 64-hex ``fingerprint`` as the annotation resolution rows, each with a
   ``body`` string), ``diff.patch``, and a git-bearing ``manifest.json``.

2. SHA coverage — all three files are covered by the curation bundle's
   ``SHA256SUMS`` (content-addressed).

Producer-realistic shapes confirmed against the archive writer
(``daydream/archive/__init__.py`` copies the run's ``findings.json`` artifact
into the batch dir; ``daydream/archive/manifest.py`` serializes the run
manifest with ``git.head_sha`` and ``code_context.{base_sha,head_sha}``;
``daydream/training/harvest.py:_row_recorded_fingerprints`` reads
``findings.json`` as ``{"findings": [{"fingerprint": ..., "body": ...}]}``).
"""

import json
from pathlib import Path

from daydream.training.corpus_v2.projector import (
    read_batch_artifacts,
    run_build_corpus_v2,
)
from tests.test_corpus_v2 import (
    _cfg,
    _write_annotations_snapshot,
    _write_bundle,
    _write_sumsums,
)


def test_spike_probe_curated_batch_layout(tmp_path: Path) -> None:
    bundle_dir = _write_bundle(tmp_path)
    batch_dir = bundle_dir / "batches" / "sess-a"
    # The batch finding's fingerprint must match an annotation resolution row's
    # fingerprint — that is the join key the projector will use. The snapshot
    # helper uses the canonical fingerprints for a lone session ("a1"*32 is the
    # first, accepted), and the snapshot must be (re)harvested against the
    # final batch fileset, so the artifacts land before the snapshot is
    # written — mirroring the real curation order.
    accepted_fp = "a1" * 32
    finding_body = "Foo() leaks the pooled connection when the retry path raises."
    (batch_dir / "findings.json").write_text(
        json.dumps({"findings": [
            {"fingerprint": accepted_fp, "body": finding_body,
             "file": "pool.py", "line": 3, "placement": "inline"},
        ]}) + "\n"
    )
    diff_body = "--- a/pool.py\n+++ b/pool.py\n@@ -1,2 +1,3 @@\n import ctx\n+pool.release()\n"
    (batch_dir / "diff.patch").write_text(diff_body)
    # Producer-realistic manifest shape: head SHA under ``git``, base SHA under
    # ``code_context`` (archive/manifest.py:374-387).
    batch_manifest = {
        "git": {"head_sha": "h" * 40},
        "code_context": {"base_sha": "b" * 40, "head_sha": "h" * 40},
    }
    (batch_dir / "manifest.json").write_text(json.dumps(batch_manifest) + "\n")
    _write_sumsums(bundle_dir)
    snap = _write_annotations_snapshot(bundle_dir, session_id="sess-a")
    accepted_fp = "a1" * 32

    # The projection pipeline accepts a bundle whose batches carry the three
    # artifacts (no shape validation rejects them).
    run_build_corpus_v2(_cfg(tmp_path / "out", bundle_dir, snap))

    # THE RELATION TO PROVE: a helper read from the batch path resolves a
    # resolution's finding text and diff exactly.
    artifacts = read_batch_artifacts(bundle_dir, "sess-a")
    assert artifacts.findings_by_fingerprint == {accepted_fp: finding_body}
    assert artifacts.diff == diff_body
    assert artifacts.manifest_git == batch_manifest["git"]
    assert artifacts.manifest_code_context == batch_manifest["code_context"]

    # SHA coverage: each of the three artifacts is covered by the curation
    # bundle's SHA256SUMS (content-addressed).
    sums = (bundle_dir / "SHA256SUMS").read_text().splitlines()
    covered = [
        line for line in sums
        if any(f"batches/sess-a/{name}" in line
               for name in ("findings.json", "diff.patch", "manifest.json"))
    ]
    assert len(covered) == 3, sums
