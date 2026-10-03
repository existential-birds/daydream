"""Export adjudications against the preview ledger's pinned finding identities and evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from daydream.training.adjudication.canonical import AnnotationDriftError
from daydream.training.adjudication.export import EXPORT_KEYS
from daydream.training.adjudication.observations import (
    load_observations,
    prior_adjudications,
)
from daydream.training.adjudication.preview import _load_sessions
from daydream.training.adjudication.queue import build_queue
from daydream.training.adjudication.report import adjudicated_items

__all__ = ["build_export_entries"]


def build_export_entries(
    index_root: Path,
    ledger_path: Path,
    *,
    observations_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Build corpus adjudicate export rows sorted by record_id. Verify preview digests against a fresh
    hydrated-index queue before applying effective_adjudication precedence; never re-pin drifted
    evidence. Missing ledgers, unknown IDs, and digest drift fail before writing. Rows use the
    projector adjudication shape plus record_id and evidence_digest.
    """
    observations = load_observations(observations_path) if observations_path is not None else []
    items = build_queue(
        _load_sessions(index_root)[0], prior_observations=prior_adjudications(observations),
    )
    by_record_id = {str(item["record_id"]): item for item in items}

    if not ledger_path.is_file():
        raise FileNotFoundError(
            f"preview ledger not found (run `corpus adjudicate preview` first): {ledger_path}"
        )
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        from daydream.archive.hydrate import HubUnavailableError

        raise HubUnavailableError(
            f"unreadable preview ledger at {ledger_path}: {exc}"
        ) from exc

    drifted: list[str] = []
    for ledger_item in ledger["items"]:
        record_id = str(ledger_item["record_id"])
        fresh = by_record_id.get(record_id)
        if fresh is None:
            raise ValueError(
                f"preview ledger record_id {record_id!r} is absent from the "
                "freshly built adjudication queue over the index"
            )
        if str(fresh["evidence_digest"]) != str(ledger_item["evidence_digest"]):
            drifted.append(record_id)
    if drifted:
        raise AnnotationDriftError(
            f"evidence digests drifted from the preview ledger for "
            f"{len(drifted)} finding(s); re-run `corpus adjudicate preview` and re-adjudicate. "
            f"Requeued record_ids: {drifted}",
            drifted,
        )

    exported: list[dict[str, Any]] = []
    for item in adjudicated_items(items, observations):
        record_id = str(item["record_id"])
        disposition = str(item["disposition"])
        evidence = item["evidence"]
        profile = str(item["profile"])
        entry: dict[str, Any] = {
            "record_id": record_id,
            "evidence_digest": str(item["evidence_digest"]),
            "fingerprint": str(item["fingerprint"]),
            "disposition": disposition,
            "evidence": evidence,
            "exclusion_reason": None,
            "profile": profile,
            "stack": item["stack"],
            "session_id": item["session_id"],
            "trajectory_id": item["trajectory_id"],
            "segment_id": item["segment_id"],
            "tier": item["tier"],
            "posterior_eligible": item["posterior_eligible"],
            "rubric_version": item["rubric_version"],
        }
        tier = item["tier"]
        if tier == "task-only":
            entry["exclusion_reason"] = (
                f"non-decisive disposition {disposition!r} — missing decisive human verdict "
                "(evidence carried for the adjudication pass)"
            )
        assert set(entry) == set(EXPORT_KEYS), "export key drift"
        exported.append(entry)
    exported.sort(key=lambda e: str(e["record_id"]))
    return exported
