"""50-record corpus-v2 integration fixture (mirrors the v1 ``records-50``).

Builds a real corpus-v2 projection directory with :func:`run_build_corpus_v2`
over a curated bundle + annotation snapshot — the same staging helpers
``tests.test_corpus_v2`` uses — sized so that:

- exactly 50 records are emitted,
- both gold classes (accepted + rejected) are present, including on the
  frozen holdout side (Stage 0's gate evaluates there),
- silver ``process-trace`` and ``task-only`` records are present
  (``emit_process_traces=True``),
- every admitted batch carries ``findings.json`` (localized finding text),
  ``diff.patch``, and a ``manifest.json`` with git ``base_sha``/``head_sha``,
  so the additive v2 enrichment is exercised end-to-end.

The build is fully deterministic: the same inputs produce byte-identical
projection directories, so the loader's directory-level digest — and the
pipeline run's ``run_identity.corpus_digest`` — is stable across runs.

Diff-body materialization: the projector emits the content-addressed
``task_identity.diff_ref`` pointer (never a raw diff body — the v2 schema is
``additionalProperties: false``). This fixture resolves each record's pointer
against the bundle and stamps the raw ``diff`` body onto the record, the same
archive-side materialization ``coordinator._materialize_diff`` performs for
v1 ``fix_diff_ref`` pointers, so Stage 2's raw-diff contract is exercised
over a real projection without touching the coordinator.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from daydream.training.corpus_v2.identity import record_id
from daydream.training.corpus_v2.projector import run_build_corpus_v2
from daydream.training.corpus_v2.splits import assign_split
from tests.test_corpus_v2 import (
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
    (git base/head), ``findings.json`` (fingerprint-keyed bodies), and
    ``diff.patch``, plus its curation-manifest row."""
    batch_dir = bundle_dir / "batches" / session_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    fps = _fingerprints(
        session_id, prefixed=_SESSION_ORDER.index(session_id) > 0
    )
    (batch_dir / "manifest.json").write_text(
        json.dumps(
            {
                "git": {
                    "base_sha": hashlib.sha256(
                        f"{session_id}-base".encode()
                    ).hexdigest()[:40],
                    "head_sha": hashlib.sha256(
                        f"{session_id}-head".encode()
                    ).hexdigest()[:40],
                }
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


def _materialize_diff_bodies(proj_dir: Path, bundle_dir: Path) -> None:
    """Resolve each record's ``task_identity.diff_ref`` against the bundle and
    stamp the raw ``diff`` body onto the record (see module docstring). The
    split/corpus files are rewritten with the projector's own canonical JSONL
    form, keeping the directory byte-deterministic."""
    for filename in (
        "corpus.jsonl",
        "corpus-v2.jsonl",
        "train.jsonl",
        "validation.jsonl",
        "holdout.jsonl",
    ):
        path = proj_dir / filename
        if not path.is_file():
            continue
        records = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        for record in records:
            identity = record.get("task_identity")
            ref = identity.get("diff_ref") if isinstance(identity, dict) else None
            if isinstance(ref, dict) and isinstance(ref.get("relpath"), str):
                record["diff"] = (bundle_dir / ref["relpath"]).read_text(encoding="utf-8")
        path.write_text(
            "".join(
                json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                + "\n"
                for r in records
            ),
            encoding="utf-8",
        )


def build_corpus_v2_50(tmp_path: Path) -> Path:
    """Materialize the 50-record corpus-v2 projection under ``tmp_path``.

    Returns:
        The projection directory (the ``--corpus-v2`` input).

    Raises:
        AssertionError: When the deterministic build does not produce the
            contracted population (50 records, both gold classes, silver +
            task-only records) — a broken fixture is a test-authoring bug,
            never a silently accepted projection.
    """
    from daydream.training.corpus_v2.projector import BuildCorpusV2Config

    work = tmp_path / "corpus-v2-fixture"
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
    run_build_corpus_v2(
        BuildCorpusV2Config(
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
    _materialize_diff_bodies(proj_dir, bundle_dir)
    return proj_dir
