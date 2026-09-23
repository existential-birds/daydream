"""Project identity-linked SQLite history into the finding observation store."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from daydream.archive.importer import gold_eligible
from daydream.training.adjudication.observations import validate_observation
from daydream.training.adjudication.queue import build_queue
from daydream.training.labeler_versions import reply_evidence_digest


def project_local_history(
    rows: Sequence[Mapping[str, Any]], sessions: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep every judgment, admitting only exact fingerprint/identity/evidence joins.

    Run-level labels cannot supply a finding judgment. Unmappable resolutions
    stay in the import report and the immutable SQLite source history.
    """
    current = {
        (str(item["session_id"]), str(item["fingerprint"])): item
        for item in build_queue(sessions, include_decisive=True)
    }
    observations: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for row in rows:
        rubric = json.loads(row.get("rubric_json") or "{}")
        resolutions = rubric.get("per_finding_resolutions") if isinstance(rubric, dict) else None
        if not isinstance(resolutions, list) or not resolutions:
            decisions.append({
                "session_id": row["session_id"], "observed_at": row["observed_at"],
                "reason": "run_level_only",
            })
            continue
        counts = Counter(str(r.get("fingerprint")) for r in resolutions if isinstance(r, dict))
        for resolution in resolutions:
            if not isinstance(resolution, dict):
                raise ValueError(f"session {row['session_id']!r}: non-object imported resolution")
            fingerprint = str(resolution.get("fingerprint", ""))
            finding = current.get((str(row["session_id"]), fingerprint))
            reason = "matched"
            if counts[fingerprint] != 1:
                reason = "ambiguous_fingerprint"
            elif finding is None:
                reason = "unknown_fingerprint"
            elif resolution.get("record_id", finding["record_id"]) != finding["record_id"]:
                reason = "record_identity_mismatch"
            elif (
                resolution.get("evidence_digest") != finding["evidence_digest"]
                or not isinstance(resolution.get("evidence"), list)
                or reply_evidence_digest(resolution["evidence"]) != finding["evidence_digest"]
            ):
                reason = "evidence_digest_mismatch"
            elif resolution.get("disposition") not in {"accepted", "rejected", "ambiguous", "unknown"}:
                reason = "non_decisive_history"
            decisions.append({
                "session_id": row["session_id"], "observed_at": row["observed_at"],
                "fingerprint": fingerprint, "reason": reason,
            })
            if reason != "matched" or finding is None:
                continue
            human = row.get("source") == "human"
            role = resolution.get("human_role", "rater") if human else "model-suggested"
            labeler = resolution.get("human_labeler") if human else row.get("labeler_version")
            observation = {
                "record_id": finding["record_id"], "disposition": resolution["disposition"],
                "evidence": resolution["evidence"], "evidence_digest": resolution["evidence_digest"],
                "role": role, "labeler": labeler or "imported-human",
                "rationale": resolution.get("rationale") or "Imported local finding judgment",
                "observed_at": row["observed_at"], "valid_at": row.get("valid_at") or row["observed_at"],
                "rubric_version": resolution.get("rubric_version") or row.get("labeler_version") or "legacy",
                "review_required": not gold_eligible(dict(row)) or (human and not labeler),
                "import_provenance": {
                    key: row.get(key) for key in (
                        "session_id", "source", "payload_digest", "evidence_sha",
                        "labeler_version", "labeler_policy_version", "reply_classifier_version",
                        "reply_evidence_digest", "observed_at", "valid_at",
                    )
                },
            }
            validate_observation(observation)
            observations.append(observation)
    return observations, decisions
