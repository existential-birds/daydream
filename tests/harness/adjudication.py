"""Shared adjudication test fixtures: hydrated ``index.db`` staging archives."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from daydream.archive.hydrate_rules import derive_curation_id
from daydream.archive.index import _get_connection


def policy_binding(source: str) -> tuple[dict[str, Any], str]:
    """The v2 policy-binding dict and its derived curation id for ``source``."""
    binding: dict[str, Any] = {
        "schema_version": "2",
        "policy_digest": "1" * 64,
        "policy_version": "production-v1",
        "allow_copyleft": ["owner/repo"],
        "exclusions_digest": "2" * 64,
        "resolved_decisions_digest": "3" * 64,
        "distribution_digest": "4" * 64,
    }
    curation_id = derive_curation_id(
        source,
        binding["policy_digest"],
        binding["policy_version"],
        frozenset(binding["allow_copyleft"]),
        binding["exclusions_digest"],
        binding["resolved_decisions_digest"],
        binding["distribution_digest"],
    )
    return binding, curation_id


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


def write_sessions_jsonl(root: Path, sessions: list[dict[str, Any]]) -> None:
    """Write ``sessions`` as the canonical sort-keys newline-delimited JSON the index reader expects."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "sessions.jsonl").write_text(
        "".join(json.dumps(s, sort_keys=True) + "\n" for s in sessions), encoding="utf-8"
    )


def write_sessions_index(root: Path, *, profiles: list[str] | None = None) -> Path:
    """Write a two-session hydrated ``sessions.jsonl`` under *root* and return *root*.

    One ambiguous + one unanswered finding across two sessions, in the shape
    the adjudication queue builder consumes. ``profiles`` overrides the
    per-session profile (default: both ``pr_review``).
    """
    session_profiles = profiles or ["pr_review", "pr_review"]
    sessions = [
        {
            "session_id": "s1", "trajectory_id": "s1-traj", "segment_id": "s1-seg",
            "resolutions": [{
                "fingerprint": "fp-b", "disposition": "unanswered",
                "evidence": [{"reply_id": "r1", "body_sha256": "abc"}],
                "evidence_digest": "d2" * 32, "profile": session_profiles[0], "stack": "python",
            }],
        },
        {
            "session_id": "s2", "trajectory_id": "s2-traj", "segment_id": "s2-seg",
            "resolutions": [{
                "fingerprint": "fp-a", "disposition": "ambiguous",
                "evidence": [{"reply_id": "r2", "body_sha256": "abd"}],
                "evidence_digest": "d1" * 32, "profile": session_profiles[1], "stack": "python",
            }],
        },
    ]
    write_sessions_jsonl(root, sessions)
    return root


def seed_index_dispositions(root: Path) -> None:
    """Write the accepted/rejected/unanswered finding index both decisive fixtures re-derive."""
    resolutions = [
        {
            "fingerprint": f"fp-{n}", "disposition": disposition,
            "evidence": [{"reply_id": n, "body_sha256": "abc",
                          "created_at": "2026-01-01T00:00:00+00:00"}],
            "evidence_digest": "d" * 32, "profile": "pr_review", "stack": "python",
            "comment_id": 7,
        }
        for n, disposition in enumerate(("accepted", "rejected", "unanswered"), start=1)
    ]
    sessions = [
        {
            "session_id": f"s{n}", "trajectory_id": f"s{n}-t", "segment_id": f"s{n}-seg",
            "resolutions": [resolution],
        }
        for n, resolution in enumerate(resolutions, start=1)
    ]
    write_sessions_jsonl(root, sessions)
    (root / "index-revision.txt").write_text("a" * 40, encoding="utf-8")
