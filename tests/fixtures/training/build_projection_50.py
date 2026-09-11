"""50-record frozen-corpus projection integration fixture.

Builds a real frozen-corpus projection directory with :func:`build_frozen_corpus`
over a curated bundle + annotation snapshot — the same staging helpers
``tests.test_corpus_projection`` uses — sized so that:

- exactly 50 records are emitted,
- both gold classes (accepted + rejected) are present, including on the
  frozen holdout side (Stage 0's gate evaluates there),
- silver ``process-trace`` and ``task-only`` records are present
  (``emit_process_traces=True``),
- every admitted batch carries ``findings.json`` (localized finding text),
  ``diff.patch``, and a ``manifest.json`` with ``git.head_sha`` plus
  ``code_context.{base_sha, head_sha}`` (the producer-realistic namespaces),
  so the per-finding record enrichment is exercised end-to-end.

The build is fully deterministic: the same inputs produce byte-identical
projection directories, so the loader's directory-level digest — and the
pipeline run's ``run_identity.corpus_digest`` — is stable across runs.

The projector embeds the raw diff body on every record (training record schema
``diff``) directly from the bundle's ``batches/<sid>/diff.patch``, so the
fixture needs no post-processing: the real projector -> Stage-2 journey
(``coordinator._rft_rows`` over ``build_frozen_corpus`` output) carries the
full RFT identity (repo_slug/base_sha/head_sha/diff) on its own.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from daydream.training.corpus_projection.identity import record_id
from daydream.training.corpus_projection.projector import build_frozen_corpus
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
_AMBIGUOUS_SESSIONS = 2  # 1 non-decisive finding each -> 2 derived records
# 46 gold + 2 process-trace + 2 task-only = 50 records.

_SESSION_ORDER = ["sess-a"] + [
    *(f"sess-gold-{i:02d}" for i in range(_GOLD_SESSIONS - 1)),
    *(f"sess-amb-{c}" for c in "ab"),
]


def _fingerprints(session_id: str, *, prefixed: bool) -> list[str]:
    """The fingerprint triple ``_write_annotations_snapshot`` derives for one
    session: the first snapshot call writes unprefixed canonical fingerprints;
    every later session's fingerprints are prefixed with ``sha256(sid)[:2]``
    so globally keyed snapshots never collide."""
    prefix = hashlib.sha256(session_id.encode()).hexdigest()[:2] if prefixed else ""
    return [prefix + fp for fp in ("a1" * 32, "b2" * 32, "c3" * 32)]


def _plan_dispositions() -> dict[str, list[str]]:
    """Deterministic per-session dispositions guaranteeing both gold classes
    on both sides of the frozen boundary. Labels are assigned over the
    record ids the snapshot helper will derive, using the same
    ``assign_split`` call the projector makes, so the plan matches the build."""
    gold_pairs: list[tuple[str, str, str]] = []  # (session_id, fingerprint, split)
    for index, sid in enumerate(_SESSION_ORDER):
        if sid.startswith("sess-amb-"):
            continue
        fps = _fingerprints(sid, prefixed=index > 0)
        for fp in fps[:2]:
            rid = record_id(sid, f"{sid}:root", "seg-0", fp)
            split = assign_split(
                rid, salt=SALT, holdout_rate=HOLDOUT_RATE, val_rate=VAL_RATE
            )
            gold_pairs.append((sid, fp, split))

    holdout = [pair for pair in gold_pairs if pair[2] == "holdout"]
    assert len(holdout) >= 2, "fixture design requires >=2 holdout gold findings"
    label_of: dict[tuple[str, str], str] = {}
    # First two holdout findings pin the two classes on the evaluated side;
    # everything else alternates, so the training side carries both too.
    for position, pair in enumerate(holdout):
        label_of[(pair[0], pair[1])] = "accepted" if position == 0 else (
            "rejected" if position == 1 else ("accepted" if position % 2 == 0 else "rejected")
        )
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
    return {
        "accepted": ACCEPTED_TEXT,
        "rejected": REJECTED_TEXT,
        "ambiguous": AMBIGUOUS_TEXT,
    }[label]


def _add_batch(
    bundle_dir: Path,
    manifest: dict[str, Any],
    session_id: str,
    dispositions: list[str],
) -> None:
    """One admitted batch directory: producer-realistic ``manifest.json``
    (``git.head_sha`` plus ``code_context.{base_sha, head_sha}``),
    ``findings.json`` (fingerprint-keyed bodies), and ``diff.patch``, plus
    its curation-manifest row."""
    batch_dir = bundle_dir / "batches" / session_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    fps = _fingerprints(
        session_id, prefixed=_SESSION_ORDER.index(session_id) > 0
    )
    (batch_dir / "manifest.json").write_text(
        json.dumps(
            {
                "git": {
                    "head_sha": hashlib.sha256(
                        f"{session_id}-head".encode()
                    ).hexdigest()[:40],
                },
                "code_context": {
                    "base_sha": hashlib.sha256(
                        f"{session_id}-base".encode()
                    ).hexdigest()[:40],
                    "head_sha": hashlib.sha256(
                        f"{session_id}-head".encode()
                    ).hexdigest()[:40],
                },
            },
            sort_keys=True,
        )
        + "\n"
    )
    (batch_dir / "findings.json").write_text(
        json.dumps(
            {
                "findings": [
                    {"fingerprint": fp, "body": _body_for(label)}
                    for fp, label in zip(fps, dispositions)
                ]
            },
            sort_keys=True,
        )
        + "\n"
    )
    (batch_dir / "diff.patch").write_text(
        f"diff --git a/{session_id}.py b/{session_id}.py\n"
        f"--- a/{session_id}.py\n+++ b/{session_id}.py\n"
        f"@@ -1 +1 @@\n-pass\n+fixed-{session_id}\n"
    )
    batch_row = {
        "session_id": session_id,
        "content_digest": hashlib.sha256(session_id.encode()).hexdigest(),
        "status": "admitted",
        "reason_code": None,
        "artifact_relpath": f"batches/{session_id}",
        "artifact_digest": None,
        "manifest_relpath": f"batches/{session_id}/manifest.json",
        "repo_slug": f"owner/repo-{hashlib.sha256(session_id.encode()).hexdigest()[:6]}",
        "license_evidence": {"spdx_id": "MIT", "source": "manifest"},
    }
    existing = {b["session_id"] for b in manifest["batches"]}
    if session_id in existing:
        # e.g. sess-a from the shared bundle helper: refresh identity in place
        manifest["batches"] = [
            {**b, "repo_slug": batch_row["repo_slug"], "license_evidence": batch_row["license_evidence"]}
            if b["session_id"] == session_id else b
            for b in manifest["batches"]
        ]
    else:
        manifest["batches"].append(batch_row)


def build_projection_50(tmp_path: Path) -> Path:
    """Materialize the 50-record frozen-corpus projection under ``tmp_path``.

    Returns:
        The projection directory (the ``train --projection`` input).

    Raises:
        AssertionError: When the deterministic build does not produce the
            contracted population (50 records, both gold classes, silver +
            task-only records) — a broken fixture is a test-authoring bug,
            never a silently accepted projection.
    """
    from daydream.training.corpus_projection.projector import BuildFrozenCorpusConfig

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
    build_frozen_corpus(
        BuildFrozenCorpusConfig(
            out_dir=proj_dir,
            bundle_dir=bundle_dir,
            annotation_bundle_dir=bundle_dir.parent / f"{bundle_dir.name}-annotations",
            license_policy_path=_policy_file(work),
            salt=SALT,
            holdout_rate=HOLDOUT_RATE,
            val_rate=VAL_RATE,
            emit_process_traces=True,
        )
    )
    # Reproduce the committed projection-dir shape: SHA256SUMS over the
    # payload files (mirroring the bundle's own manifest) and the bundle's
    # curation-manifest.json, both before nothing depends on ordering — the
    # loader's digest is computed over whatever the directory holds.
    sums_lines = []
    for path in sorted(proj_dir.rglob("*")):
        if not path.is_file() or path.name in {"SHA256SUMS", "_SUCCESS"}:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        sums_lines.append(f"{digest}  {path.relative_to(proj_dir).as_posix()}\n")
    (proj_dir / "SHA256SUMS").write_text("".join(sums_lines))
    shutil.copyfile(bundle_dir / "curation-manifest.json", proj_dir / "curation-manifest.json")
    return proj_dir


def main() -> None:
    """Commit the fixture: build into a scratch dir, copy the projection
    directory to ``--out`` (content-only — the loader's directory digest is
    computed over file bytes, never mtimes)."""
    import argparse
    import shutil
    import tempfile

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
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


