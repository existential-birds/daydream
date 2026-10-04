"""Build exactly 50 frozen records through the real projector: both gold classes on training and
holdout sides, plus silver process traces and task-only rows.

Every admitted batch supplies findings.json, diff.patch, and a manifest with git.head_sha and
code_context base/head SHAs. These exercise finding enrichment and carry full repo/base/head/diff
identity into RFT without post-processing.

Identical inputs produce byte-identical projection directories and stable corpus digests.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from daydream.training.corpus_projection.identity import record_id
from daydream.training.corpus_projection.projector import (
    BuildFrozenCorpusConfig,
    build_frozen_corpus,
)
from daydream.training.corpus_projection.splits import assign_split
from tests.test_corpus_projection import (
    _policy_file,
    _write_annotations_snapshot,
    _write_bundle,
    _write_sumsums,
)

SALT = "issue-1081-fixture-salt"
HOLDOUT_RATE = 0.2
VAL_RATE = 0.2

ACCEPTED_TEXT = "exact localized finding body"
REJECTED_TEXT = "rejected finding body"
AMBIGUOUS_TEXT = "ambiguous finding body"

_GOLD_SESSIONS = 23  # 2 gold findings each -> 46 gold outcome-finding records
# 46 gold + 2 process-trace + 2 task-only = 50 records.

_SESSION_ORDER = ["sess-a"] + [
    *(f"sess-gold-{i:02d}" for i in range(_GOLD_SESSIONS - 1)), *(f"sess-amb-{c}" for c in "ab"),
]


def _fingerprints(session_id: str, *, prefixed: bool) -> list[str]:
    """Use the snapshot helper's canonical fingerprints for the first session and sha256(sid)[:2]
    prefixes afterward to avoid globally keyed collisions.
    """
    prefix = hashlib.sha256(session_id.encode()).hexdigest()[:2] if prefixed else ""
    return [prefix + fp for fp in ("a1" * 32, "b2" * 32, "c3" * 32)]


def _plan_dispositions() -> dict[str, list[str]]:
    """Assign dispositions using the projector's record IDs and assign_split so both gold classes
    appear on both sides of the frozen boundary.
    """
    gold_pairs: list[tuple[str, str, str]] = []  # (session_id, fingerprint, split)
    for index, sid in enumerate(_SESSION_ORDER):
        if sid.startswith("sess-amb-"):
            continue
        fps = _fingerprints(sid, prefixed=index > 0)
        for fp in fps[:2]:
            rid = record_id(sid, f"{sid}:root", "seg-0", fp)
            split = assign_split(rid, salt=SALT, holdout_rate=HOLDOUT_RATE, val_rate=VAL_RATE)
            gold_pairs.append((sid, fp, split))

    holdout = [pair for pair in gold_pairs if pair[2] == "holdout"]
    assert len(holdout) >= 2, "fixture design requires >=2 holdout gold findings"
    label_of: dict[tuple[str, str], str] = {}
    # Pin both classes in holdout, then alternate labels to retain both in training.
    for position, pair in enumerate(holdout):
        label_of[(pair[0], pair[1])] = "accepted" if position % 2 == 0 else "rejected"
    for position, pair in enumerate(p for p in gold_pairs if p[2] != "holdout"):
        label_of[(pair[0], pair[1])] = "accepted" if position % 2 == 0 else "rejected"

    dispositions: dict[str, list[str]] = {}
    for index, sid in enumerate(_SESSION_ORDER):
        if sid.startswith("sess-amb-"):
            dispositions[sid] = ["ambiguous"]
        else:
            fps = _fingerprints(sid, prefixed=index > 0)
            dispositions[sid] = [label_of[(sid, fp)] for fp in fps[:2]]
    return dispositions


def _body_for(label: str) -> str:
    return {"accepted": ACCEPTED_TEXT, "rejected": REJECTED_TEXT, "ambiguous": AMBIGUOUS_TEXT}[label]


def _add_batch(bundle_dir: Path, manifest: dict[str, Any], session_id: str, dispositions: list[str],) -> None:
    """Write a batch manifest with git.head_sha and code_context base/head SHAs, fingerprint-keyed
    findings, diff, and its curation-manifest row.
    """
    batch_dir = bundle_dir / "batches" / session_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    fps = _fingerprints(session_id, prefixed=_SESSION_ORDER.index(session_id) > 0)
    head_sha = hashlib.sha256(f"{session_id}-head".encode()).hexdigest()[:40]
    base_sha = hashlib.sha256(f"{session_id}-base".encode()).hexdigest()[:40]
    (batch_dir / "manifest.json").write_text(json.dumps(
            {"git": {"head_sha": head_sha}, "code_context": {"base_sha": base_sha, "head_sha": head_sha}},
            sort_keys=True,
        )
        + "\n"
    )
    (batch_dir / "findings.json").write_text(json.dumps(
            {"findings": [{"fingerprint": fp, "body": _body_for(label)} for fp, label in zip(fps, dispositions)]},
            sort_keys=True,
        )
        + "\n"
    )
    (batch_dir / "diff.patch").write_text(
        f"diff --git a/{session_id}.py b/{session_id}.py\n"
        f"--- a/{session_id}.py\n+++ b/{session_id}.py\n"
        f"@@ -1 +1 @@\n-pass\n+fixed-{session_id}\n"
    )
    batch_row = {"session_id": session_id, "content_digest": hashlib.sha256(session_id.encode()).hexdigest(),
        "status": "admitted", "reason_code": None, "artifact_relpath": f"batches/{session_id}", "artifact_digest": None,
        "manifest_relpath": f"batches/{session_id}/manifest.json",
        "repo_slug": f"owner/repo-{hashlib.sha256(session_id.encode()).hexdigest()[:6]}",
        "license_evidence": {"spdx_id": "MIT", "source": "manifest"},
    }
    existing = {b["session_id"] for b in manifest["batches"]}
    if session_id in existing:
        manifest["batches"] = [
            {**b, "repo_slug": batch_row["repo_slug"], "license_evidence": batch_row["license_evidence"]}
            if b["session_id"] == session_id else b
            for b in manifest["batches"]
        ]
    else:
        manifest["batches"].append(batch_row)


def build_projection_50(tmp_path: Path) -> Path:
    """Build the train --projection input directory. Assert the required population: 50 records, both
    gold classes, silver traces, and task-only records. A broken fixture must fail at construction.
    """
    work = tmp_path / "projection-fixture"
    bundle_dir = _write_bundle(work)
    manifest = json.loads((bundle_dir / "curation-manifest.json").read_text())
    dispositions = _plan_dispositions()
    for sid in _SESSION_ORDER:
        _add_batch(bundle_dir, manifest, sid, dispositions[sid])
    (bundle_dir / "curation-manifest.json").write_text(json.dumps(manifest))
    _write_sumsums(bundle_dir)

    for sid in _SESSION_ORDER:
        _write_annotations_snapshot(bundle_dir, session_id=sid, dispositions=dispositions[sid])

    proj_dir = work / "proj"
    build_frozen_corpus(BuildFrozenCorpusConfig(out_dir=proj_dir, bundle_dir=bundle_dir,
            annotation_bundle_dir=bundle_dir.parent / f"{bundle_dir.name}-annotations",
            license_policy_path=_policy_file(work), salt=SALT, holdout_rate=HOLDOUT_RATE, val_rate=VAL_RATE,
            emit_process_traces=True,
        )
    )
    # The projection directory holds exactly what build_frozen_corpus writes. Nothing downstream
    # reads a SHA256SUMS or curation-manifest.json here: load_v2_projection consumes only _SUCCESS,
    # the split JSONL files and lineage.json, and the directory digest it reports is recomputed from
    # whatever is present.
    return proj_dir


def main() -> None:
    """Build in scratch space and copy content to --out; the loader digest depends on bytes, not
    mtimes.
    """
    import argparse
    import tempfile

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True,
        help="Destination projection directory (e.g. tests/fixtures/training/projection-50)",
    )
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        proj_dir = build_projection_50(Path(tmp))
        if args.out.exists():
            shutil.rmtree(args.out)
        shutil.copytree(proj_dir, args.out)
    print(f"committed projection fixture written to {args.out}")


if __name__ == "__main__":
    main()


