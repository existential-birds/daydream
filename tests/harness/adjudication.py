"""Shared adjudication test fixtures: hydrated ``index.db`` staging archives.

Also owns the ``SHA256SUMS`` bundle-envelope writer every corpus, annotation, and
final-bundle fixture has to produce byte-identically.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from daydream.archive.hydrate_rules import derive_curation_id
from daydream.archive.index import _get_connection
from daydream.training.corpus_projection.identity import record_id


def sha256sums_text(entries: Iterable[tuple[str, bytes]], *, prefix: str = "") -> str:
    """``<sha256>  <name>\\n`` lines for ``(name, bytes)`` pairs, in the given order.

    ``prefix`` prepends a bundle-relative directory (e.g. the producer-realistic
    ``curated/<id>/`` a hub checkout records); it is part of the hashed name.
    """
    return "".join(f"{hashlib.sha256(data).hexdigest()}  {prefix}{name}\n" for name, data in entries)


def write_sha256sums(root: Path, *, skip: frozenset[str], prefix: str = "") -> None:
    """Write ``root/SHA256SUMS`` over every file under *root*, in relpath order.

    ``skip`` names the files that must never appear in the listing — always the
    listing itself, and ``_SUCCESS`` wherever the producer excludes it.
    """
    entries = sorted(
        (path.relative_to(root).as_posix(), path)
        for path in root.rglob("*")
        if path.is_file() and path.name not in skip
    )
    (root / "SHA256SUMS").write_text(
        sha256sums_text(((rel, path.read_bytes()) for rel, path in entries), prefix=prefix),
        encoding="utf-8",
    )


def accepted_observation() -> dict[str, Any]:
    """The accepted ``s1`` human observation the final-bundle fixtures materialize."""
    return {
        "record_id": record_id("s1", "s1-t", "s1-seg", "fp-1"), "disposition": "accepted", "evidence_digest": "d" * 32,
        "evidence": [{"reply_id": 1, "body_sha256": "abc", "created_at": "2026-01-01T00:00:00+00:00"}],
        "labeler": "alice", "role": "rater", "rationale": "clear maintainer approval",
        "valid_at": "2026-02-02T00:00:00+00:00", "observed_at": "2026-02-02T00:00:00+00:00", "rubric_version": "v1",
    }


def policy_binding(source: str) -> tuple[dict[str, Any], str]:
    """The v2 policy-binding dict and its derived curation id for ``source``."""
    binding: dict[str, Any] = {"schema_version": "2", "policy_digest": "1" * 64, "policy_version": "production-v1",
        "allow_copyleft": ["owner/repo"], "exclusions_digest": "2" * 64, "resolved_decisions_digest": "3" * 64,
        "distribution_digest": "4" * 64,
    }
    curation_id = derive_curation_id(
        source, binding["policy_digest"], binding["policy_version"], frozenset(binding["allow_copyleft"]),
        binding["exclusions_digest"], binding["resolved_decisions_digest"], binding["distribution_digest"],
    )
    return binding, curation_id


def make_hydrated_sqlite_index(
    tmp_path: Path, generations: list[tuple[str, str, str, str, str]], *, session_id: str = "s1",
) -> Path:
    """Build a hydrated staging archive whose per-finding data lives ONLY in
    ``label_observations.rubric_json`` — no ``trajectory.json`` resolutions key.

    Each generation is ``(observed_at, labels, evidence_sha, policy_version,
    disposition)``.
    """
    root = tmp_path / "hydrated"
    conn = _get_connection(root)
    conn.execute("INSERT INTO runs (session_id, archived_at, run_flow, archive_path) "
        "VALUES (?, '2026-01-01T00:00:00+00:00', 'deep', 'archive/s1')", (session_id,),
    )
    for observed_at, labels, evidence_sha, policy_version, disposition in generations:
        rubric = {"posterior_source": "pr_review",
                  "per_finding_resolutions": [{"fingerprint": "fp-1", "comment_id": 7, "disposition": disposition,
                      "evidence": [{"reply_id": 1, "body_sha256": "abc"}],
                      "evidence_digest": "d" * 32}]}
        conn.execute("INSERT INTO label_observations (session_id, observed_at, labels, labeler_version, "
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
    sessions = [{"session_id": "s1", "trajectory_id": "s1-traj", "segment_id": "s1-seg",
            "resolutions": [{"fingerprint": "fp-b", "disposition": "unanswered",
                "evidence": [{"reply_id": "r1", "body_sha256": "abc"}],
                "evidence_digest": "d2" * 32, "profile": session_profiles[0], "stack": "python",
            }],
        }, {"session_id": "s2", "trajectory_id": "s2-traj", "segment_id": "s2-seg",
            "resolutions": [{"fingerprint": "fp-a", "disposition": "ambiguous",
                "evidence": [{"reply_id": "r2", "body_sha256": "abd"}],
                "evidence_digest": "d1" * 32, "profile": session_profiles[1], "stack": "python",
            }],
        },
    ]
    write_sessions_jsonl(root, sessions)
    return root


def write_checkpoint_inputs(root: Path, *, curation_id: str = "cur-1") -> tuple[Path, Path]:
    """Create the state dir and preview manifest the adjudicate publish verbs consume."""
    state = root / "state"
    state.mkdir()
    (state / "queue.json").write_text("[]\n", encoding="utf-8")
    (state / "observations.jsonl").write_text("", encoding="utf-8")
    (state / "preview-ledger.json").write_text("{}\n", encoding="utf-8")
    manifest = root / "preview-manifest.json"
    manifest.write_text(json.dumps({"curation_id": curation_id, "snapshot_id": "e" * 64}) + "\n", encoding="utf-8",)
    return state, manifest


def seed_index_dispositions(root: Path) -> None:
    """Write the accepted/rejected/unanswered finding index both decisive fixtures re-derive."""
    resolutions = [{"fingerprint": f"fp-{n}", "disposition": disposition,
            "evidence": [{"reply_id": n, "body_sha256": "abc", "created_at": "2026-01-01T00:00:00+00:00"}],
            "evidence_digest": "d" * 32, "profile": "pr_review", "stack": "python", "comment_id": 7,
        }
        for n, disposition in enumerate(("accepted", "rejected", "unanswered"), start=1)
    ]
    sessions = [
        {"session_id": f"s{n}", "trajectory_id": f"s{n}-t", "segment_id": f"s{n}-seg", "resolutions": [resolution]}
        for n, resolution in enumerate(resolutions, start=1)
    ]
    write_sessions_jsonl(root, sessions)
    (root / "index-revision.txt").write_text("a" * 40, encoding="utf-8")
