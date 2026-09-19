"""Shared adjudication test fixtures: hydrated ``index.db`` staging archives."""

from __future__ import annotations

import json
from pathlib import Path

from daydream.archive.index import _get_connection


def make_hydrated_sqlite_index(
    tmp_path: Path,
    generations: list[tuple[str, str, str, str, str]],
    *,
    session_id: str = "s1",
) -> Path:
    """Build a hydrated staging archive whose per-finding data lives ONLY in
    ``label_observations.rubric_json`` — no ``trajectory.json`` resolutions key.

    Each generation is ``(observed_at, labels, evidence_sha, policy_version,
    disposition)``.
    """
    root = tmp_path / "hydrated"
    conn = _get_connection(root)
    conn.execute(
        "INSERT INTO runs (session_id, archived_at, run_flow, archive_path) "
        "VALUES (?, '2026-01-01T00:00:00+00:00', 'deep', 'archive/s1')",
        (session_id,),
    )
    for observed_at, labels, evidence_sha, policy_version, disposition in generations:
        rubric = {"posterior_source": "pr_review",
                  "per_finding_resolutions": [{
                      "fingerprint": "fp-1", "comment_id": 7, "disposition": disposition,
                      "evidence": [{"reply_id": 1, "body_sha256": "abc"}],
                      "evidence_digest": "d" * 32}]}
        conn.execute(
            "INSERT INTO label_observations (session_id, observed_at, labels, labeler_version, "
            "evidence_sha, rubric_json, has_posterior, source, labeler_policy_version) "
            "VALUES (?, ?, ?, 'v1', ?, ?, 0, 'auto', ?)",
            (session_id, observed_at, labels, evidence_sha, json.dumps(rubric), policy_version),
        )
    conn.commit()
    conn.close()
    (root / "downloads" / ("a" * 40)).mkdir(parents=True)
    return root
