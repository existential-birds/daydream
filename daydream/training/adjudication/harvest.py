"""Export adjudications against the preview ledger's pinned finding identities and evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from daydream.dataset import LocalRecordStore
from daydream.training.adjudication.canonical import AnnotationDriftError
from daydream.training.adjudication.export import EXPORT_KEYS
from daydream.training.adjudication.observations import group_observations_by_record, prior_adjudications
from daydream.training.adjudication.precedence import DECISIVE_DISPOSITIONS, effective_adjudication
from daydream.training.adjudication.queue import build_queue
from daydream.training.corpus_projection.tiers import classify_tier
from daydream.training.record_evidence import finding_observations, sessions_from_snapshot

__all__ = ["build_export_entries"]


def build_export_entries(
    store_dir: Path,
    snapshot_id: str,
    ledger_path: Path,
) -> list[dict[str, Any]]:
    """Build corpus adjudicate export rows sorted by record_id. Verify preview digests against a fresh
    frozen-record queue before applying effective_adjudication precedence; never re-pin drifted
    evidence. Missing ledgers, unknown IDs, and digest drift fail before writing. Rows use the
    projector adjudication shape plus record_id and evidence_digest.
    """
    records = LocalRecordStore(store_dir).read_snapshot(snapshot_id)
    observations = finding_observations(records)
    items = build_queue(
        sessions_from_snapshot(records, overlay_judgments=False),
        prior_observations=prior_adjudications([o for o in observations if o["role"] != "automatic"]),
    )
    by_record_id = {str(item["record_id"]): item for item in items}

    if not ledger_path.is_file():
        raise FileNotFoundError(f"preview ledger not found (run `corpus adjudicate preview` first): {ledger_path}")
    try:
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("unreadable preview ledger") from exc

    drifted: list[str] = []
    for ledger_item in ledger["items"]:
        record_id = str(ledger_item["record_id"])
        fresh = by_record_id.get(record_id)
        if fresh is None:
            raise ValueError(
                f"preview ledger record_id {record_id!r} is absent from the "
                "freshly built adjudication queue over the selected snapshot"
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

    grouped = group_observations_by_record(
        observations, {str(item["record_id"]) for item in items}, "adjudicate export"
    )

    from daydream.training.record_identity import record_finding_id

    hosts = {
        record_finding_id(s["session_id"], s["trajectory_id"], s["segment_id"], r["item_uid"]): r["item_uid"]
        for s in sessions_from_snapshot(records)
        for r in s["resolutions"]
    }
    exported: list[dict[str, Any]] = []
    for item in items:
        record_id = str(item["record_id"])
        disposition = str(item["disposition"])
        evidence = item["evidence"]
        role: str = "automatic"
        gold_eligible = False
        if record_id in grouped:
            resolved = effective_adjudication(grouped[record_id])
            role = resolved["role"]
            gold_eligible = resolved["gold_eligible"]
            if (
                role in ("rater", "adjudicator")
                and resolved["evidence_digest"] == str(item["evidence_digest"])
                and resolved["disposition"] in DECISIVE_DISPOSITIONS
            ):
                disposition = resolved["disposition"]

        profile = str(item["profile"])
        entry: dict[str, Any] = {
            "record_id": record_id,
            "item_uid": hosts.get(record_id),
            "evidence_digest": str(item["evidence_digest"]),
            "fingerprint": str(item["fingerprint"]),
            "disposition": disposition,
            "evidence": evidence,
            "reply_captures": item["reply_captures"],
            "correction": item["correction"],
            "exclusion_reason": None,
            "profile": profile,
            "stack": item["stack"],
            "session_id": item["session_id"],
            "trajectory_id": item["trajectory_id"],
            "segment_id": item["segment_id"],
            "tier": None,
            "posterior_eligible": False,
            "rubric_version": item["rubric_version"],
        }
        tier = classify_tier(entry)
        if tier == "gold" and not gold_eligible:
            # A failed human gate (conflict or review-required) withholds structurally eligible
            # gold.
            tier = "task-only"
        entry["tier"] = tier
        if tier == "task-only":
            entry["exclusion_reason"] = (
                f"non-decisive disposition {disposition!r} — missing decisive human verdict "
                "(evidence carried for the adjudication pass)"
            )
        entry["posterior_eligible"] = tier == "gold" and profile == "pr_review"
        assert set(entry) == set(EXPORT_KEYS), "export key drift"
        exported.append(entry)
    exported.sort(key=lambda e: str(e["record_id"]))
    return exported
