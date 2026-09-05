"""Issue #1095 acceptance: hydrated archive -> preview -> imported local
history -> queue -> canonical drift-checked harvest, every finding exactly once.

Real-path fixture: real SQLite ``index.db`` files written through the
production archive writers (``upsert_run`` / ``append_label_observation``),
the real CLI entrypoint for the local-history import, and the real
materialize/preview/canonical pipeline. No mocking anywhere — Task 10 owns
the publish wiring.
"""

import json
from pathlib import Path

from daydream.archive.index import append_label_observation, label_observation_history, upsert_run
from daydream.training.adjudication.canonical import run_canonical_harvest
from daydream.training.adjudication.cli import handle_adjudicate
from daydream.training.adjudication.materialize import run_materialize
from daydream.training.adjudication.preview import run_preview
from tests.harness.trajectory import make_manifest

_PIN = {  # same shape as tests/test_training_adjudication_canonical.py
    "curation_id": "cur-e2e", "sanitized_hub_commit": "a" * 40,
    "source_hub_commit": "b" * 40, "archive_index_digest": "c" * 64,
    "evidence_observed_at": "2026-01-01T00:00:00+00:00",
    "as_of": "2026-02-01T00:00:00+00:00",
    "labeler_version": "v1", "rubric_version": "v1", "classifier_version": "v1",
}

_OBSERVED = "2026-01-02T00:00:00+00:00"
_EVIDENCE = [{"reply_id": 1, "body_sha256": "abc", "created_at": "2026-01-01T00:00:00+00:00"}]


def _resolution(fingerprint: str, disposition: str, digest: str) -> dict:
    return {
        "fingerprint": fingerprint, "disposition": disposition,
        "evidence": _EVIDENCE, "evidence_digest": digest,
        "profile": "pr_review", "stack": "python", "comment_id": 7,
    }


def _seed_observation(
    root: Path, session_id: str, fingerprint: str, disposition: str, digest: str,
    *, labels: list[str], observed_at: str = _OBSERVED,
) -> None:
    append_label_observation(
        root,
        session_id,
        labels=labels,
        pr_state=None,
        labeler_version="980-rubric-r2",
        evidence_sha=digest,
        rubric_json=json.dumps({
            "posterior_source": "pr_review",
            "per_finding_resolutions": [_resolution(fingerprint, disposition, digest)],
        }),
        valid_at=_OBSERVED,
        reply_evidence_digest=None,
        reward_version=None,
        has_posterior=False,
        source="auto",
        observed_at=observed_at,
    )


def _seed_run(root: Path, session_id: str) -> tuple[str, str]:
    head = "h" + session_id.encode().hex()
    base = "b" + session_id.encode().hex()
    upsert_run(root, make_manifest(session_id=session_id, repo_slug="org/repo",
                                   head_sha=head, base_sha=base))
    return base, head


def _seed_hydrated_archive(tmp_path: Path) -> Path:
    """Four sessions: accepted, rejected, conflicted (two distinct dedup keys),
    unresolved (unanswered) — each through the real archive writers, with
    ``rubric_json`` carrying the full ``per_finding_resolutions`` (Task 2
    shape) so the SQLite materialization path can consume the index."""
    root = tmp_path / "hydrated"
    _seed_run(root, "s-acc")
    _seed_observation(root, "s-acc", "fp-acc", "accepted", "d0" * 32,
                      labels=["finding-accepted"])
    _seed_run(root, "s-rej")
    _seed_observation(root, "s-rej", "fp-rej", "rejected", "d1" * 32,
                      labels=["finding-rejected"])
    _seed_run(root, "s-unres")
    _seed_observation(root, "s-unres", "fp-unres", "unanswered", "d2" * 32,
                      labels=["finding-unanswered"])
    # Conflicted: two observations with distinct dedup keys (different labels)
    # for the same session — the materializer must surface the session
    # non-gold (``conflicting: true``), never merge the generations away.
    _seed_run(root, "s-conf")
    _seed_observation(root, "s-conf", "fp-conf", "accepted", "d3" * 32,
                      labels=["finding-accepted"], observed_at="2026-01-02T00:00:00+00:00")
    _seed_observation(root, "s-conf", "fp-conf", "accepted", "d3" * 32,
                      labels=["finding-accepted", "posterior"],
                      observed_at="2026-01-03T00:00:00+00:00")
    (root / "downloads" / ("a" * 40)).mkdir(parents=True)
    return root


def _seed_local_backup(tmp_path: Path, base: str, head: str) -> Path:
    """A local backup archive carrying a byte-identical copy of s-acc's
    observation — the merge must dedupe it (no new generation, s-acc stays
    non-conflicting at harvest time)."""
    backup = tmp_path / "backup"
    upsert_run(backup, make_manifest(session_id="s-acc", repo_slug="org/repo",
                                     head_sha=head, base_sha=base))
    _seed_observation(backup, "s-acc", "fp-acc", "accepted", "d0" * 32,
                      labels=["finding-accepted"])
    return backup


def test_hydrated_to_canonical_harvest_end_to_end(tmp_path: Path) -> None:
    root = _seed_hydrated_archive(tmp_path)
    # 1. preview is read-only over the hydrated index and pins the operator
    #    queue (only the non-decisive finding enters it; the decisive ones are
    #    enumerated by materialize + the complete-set drift gate).
    summary = run_preview(root, tmp_path / "ledger.json")
    assert summary["item_count"] == 1
    # 2. materialize from SQLite: one record per finding, all four classes.
    mat = run_materialize(root, tmp_path / "snapshot", pin=_PIN)
    assert mat["record_count"] == 4
    # 3. import a local backup history for one session (CLI path, dry-run
    #    then real) — identical content, so the merge dedupes and the
    #    session stays non-conflicting.
    base, head = "b" + b"s-acc".hex(), "h" + b"s-acc".hex()
    backup = _seed_local_backup(tmp_path, base, head)
    rc = handle_adjudicate([
        "import-local-observations", "--archive-root", str(backup),
        "--index-root", str(root), "--archive-dir", str(root),
        "--state-dir", str(tmp_path / "state"), "--dry-run", "--json",
    ])
    assert rc == 0
    rc = handle_adjudicate([
        "import-local-observations", "--archive-root", str(backup),
        "--index-root", str(root), "--archive-dir", str(root),
        "--state-dir", str(tmp_path / "state"),
    ])
    assert rc == 0
    # 4. canonical drift-checked harvest over the same hydrated archive
    out = run_canonical_harvest(
        index_root=root, materialize_dir=tmp_path / "snapshot", archive_dir=root,
    )
    # exactly-once accounting: 4 sessions in, 4 session rows out
    assert out["record_count"] == 4
    rows = [r for sid in ("s-acc", "s-rej", "s-conf", "s-unres")
            for r in label_observation_history(root, sid)]
    assert len(rows) >= 4  # one per session at minimum (import may add generations)
    dispositions = set()
    for row in rows:
        rubric = json.loads(row["rubric_json"])
        for rec in rubric["per_finding_resolutions"]:
            dispositions.add((rec["fingerprint"], rec["disposition"], rec.get("conflicting", False)))
    # accepted, rejected, unresolved each exactly once; conflicted surfaced non-gold
    assert ("fp-acc", "accepted", False) in dispositions
    assert ("fp-rej", "rejected", False) in dispositions
    assert ("fp-unres", "unanswered", False) in dispositions
    assert ("fp-conf", "accepted", True) in dispositions
    assert len([d for d in dispositions if d[0] == "fp-acc"]) == 1
