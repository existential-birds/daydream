"""Exercise SQLite archive writers and preview/materialize/canonical harvest together.
Every finding appears once; no boundaries are mocked.
"""

import json
from pathlib import Path

from daydream.archive.index import append_label_observation, label_observation_history, upsert_run
from daydream.training.adjudication.canonical import run_canonical_harvest
from daydream.training.adjudication.materialize import run_materialize
from daydream.training.adjudication.preview import run_preview
from daydream.training.adjudication.queue import build_queue
from tests.harness.trajectory import make_manifest
from tests.test_training_adjudication_canonical import _PIN as _CANONICAL_PIN

_PIN = {**_CANONICAL_PIN, "curation_id": "cur-e2e"}

_OBSERVED = "2026-01-02T00:00:00+00:00"
_EVIDENCE = [{"reply_id": 1, "body_sha256": "abc", "created_at": "2026-01-01T00:00:00+00:00"}]


def _resolution(fingerprint: str, disposition: str, digest: str) -> dict[str, object]:
    return {"fingerprint": fingerprint, "disposition": disposition, "evidence": _EVIDENCE, "evidence_digest": digest,
        "profile": "pr_review", "stack": "python", "comment_id": 7,
    }


def _seed_observation(root: Path, session_id: str, fingerprint: str, disposition: str, digest: str,
    *, labels: list[str], observed_at: str = _OBSERVED,
) -> None:
    append_label_observation(
        root, session_id, labels=labels, pr_state=None, labeler_version="980-rubric-r2", evidence_sha=digest,
        rubric_json=json.dumps({
            "posterior_source": "pr_review", "per_finding_resolutions": [_resolution(fingerprint, disposition, digest)],
        }), valid_at=_OBSERVED, reply_evidence_digest=None, reward_version=None, has_posterior=False, source="auto",
        observed_at=observed_at,
    )


def _seed_run(root: Path, session_id: str) -> tuple[str, str]:
    head = "h" + session_id.encode().hex()
    base = "b" + session_id.encode().hex()
    upsert_run(root, make_manifest(session_id=session_id, repo_slug="org/repo", head_sha=head, base_sha=base))
    return base, head


def _seed_hydrated_archive(tmp_path: Path) -> Path:
    """Seed accepted, rejected, conflicted, unanswered, and legacy labels-only sessions through the real
    archive writers. Only the legacy row lacks a trajectory and per-finding rubric.
    """
    root = tmp_path / "hydrated"
    _seed_run(root, "s-acc")
    _seed_observation(root, "s-acc", "fp-acc", "accepted", "d0" * 32, labels=["finding-accepted"])
    _seed_run(root, "s-rej")
    _seed_observation(root, "s-rej", "fp-rej", "rejected", "d1" * 32, labels=["finding-rejected"])
    _seed_run(root, "s-unres")
    _seed_observation(root, "s-unres", "fp-unres", "unanswered", "d2" * 32, labels=["finding-unanswered"])
    # Disagreeing generations with distinct dedup keys remain conflicting instead of being merged
    # away.
    _seed_run(root, "s-conf")
    _seed_observation(root, "s-conf", "fp-conf", "accepted", "d3" * 32,
                      labels=["finding-accepted"], observed_at="2026-01-02T00:00:00+00:00")
    _seed_observation(root, "s-conf", "fp-conf", "accepted", "d3" * 32,
                      labels=["finding-accepted", "posterior"], observed_at="2026-01-03T00:00:00+00:00")
    # Legacy labels-only rows have neither trajectories nor per-finding resolutions; they must not
    # abort curation.
    _seed_run(root, "s-legacy")
    append_label_observation(root, "s-legacy", labels=["finding-accepted"], pr_state=None,
        labeler_version="980-rubric-r2", evidence_sha=None,
        rubric_json=json.dumps({"per_finding_outcomes": ["accepted"]}), valid_at=_OBSERVED, reply_evidence_digest=None,
        reward_version=None, has_posterior=False, source="auto", observed_at="2026-01-02T00:00:00+00:00",
    )
    (root / "downloads" / ("a" * 40)).mkdir(parents=True)
    return root


def test_hydrated_to_canonical_harvest_end_to_end(tmp_path: Path) -> None:
    root = _seed_hydrated_archive(tmp_path)
    # The queue contains only nondecisive findings, while materialization and drift checks include
    # decisive findings.
    summary = run_preview(root, tmp_path / "ledger.json")
    assert summary["item_count"] == 1
    # Materialize one record per finding; the legacy labels-only session contributes none.
    mat = run_materialize(root, tmp_path / "snapshot", pin=_PIN)
    assert mat["record_count"] == 4
    # A conflicting ambiguous finding is task-only and must never become gold.
    rows = [json.loads(line) for line in (tmp_path / "snapshot" / "sessions.jsonl").read_text().splitlines() if line]
    assert len(rows) == len({row["record_id"] for row in rows}) == 4
    snapshot_records = {row["fingerprint"]: row for row in rows}
    assert snapshot_records["fp-conf"]["conflicting"] is True
    assert snapshot_records["fp-conf"]["disposition"] == "ambiguous"
    queue = build_queue(rows)
    assert snapshot_records["fp-conf"]["record_id"] in {str(item["record_id"]) for item in queue}
    out = run_canonical_harvest(index_root=root, materialize_dir=tmp_path / "snapshot", archive_dir=root)
    assert out["record_count"] == 4
    rows = [r for sid in ("s-acc", "s-rej", "s-conf", "s-unres") for r in label_observation_history(root, sid)]
    assert len(rows) >= 4  # one per session at minimum
    dispositions = set()
    for row in rows:
        rubric = json.loads(row["rubric_json"])
        for rec in rubric["per_finding_resolutions"]:
            dispositions.add((rec["fingerprint"], rec["disposition"], rec.get("conflicting", False)))
    assert ("fp-acc", "accepted", False) in dispositions
    assert ("fp-rej", "rejected", False) in dispositions
    assert ("fp-unres", "unanswered", False) in dispositions
    assert ("fp-conf", "accepted", True) in dispositions
    assert len([d for d in dispositions if d[0] == "fp-acc"]) == 1
