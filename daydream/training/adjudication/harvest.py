"""Canonical fail-closed adjudication harvest (issue #984, Task 10).

Verifies the preview ledger's per-finding evidence digests against a freshly
built queue over the hydrated index and merges human judgments from the
observation store under three-tier precedence. Digest drift raises
:class:`AdjudicationDriftError` before anything is written (delta on
``corpus_projection.projector.build_frozen_corpus``'s digest-pinned snapshot flow:
harvest verifies the *preview ledger's* digests rather than re-pinning its
own, so preview identities and digests are stable into the export by
construction).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from daydream.training.adjudication.canonical import AnnotationDriftError
from daydream.training.adjudication.export import EXPORT_KEYS
from daydream.training.adjudication.observations import (
    group_observations_by_record,
    load_observations,
    prior_adjudications,
)
from daydream.training.adjudication.precedence import DECISIVE_DISPOSITIONS, effective_adjudication
from daydream.training.adjudication.preview import _load_sessions
from daydream.training.adjudication.queue import build_queue
from daydream.training.corpus_projection.tiers import classify_tier

__all__ = ["AdjudicationDriftError", "build_export_entries"]

AdjudicationDriftError = AnnotationDriftError


def build_export_entries(
    index_root: Path,
    ledger_path: Path,
    *,
    observations_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Build the projector-shape export rows over a hydrated index.

    Builder for the ``corpus adjudicate export`` CLI verb:
    verifies every preview-ledger ``record_id``'s evidence digest
    against the fresh queue, merges human judgments from the observation
    store under three-tier precedence, and returns the rows in the
    ``project_findings`` adjudication entry shape (plus ``record_id`` and
    ``evidence_digest``), sorted by ``record_id``. Raises digest drift, a
    missing ledger, and unknown record ids.
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
        raise AdjudicationDriftError(
            f"evidence digests drifted from the preview ledger for "
            f"{len(drifted)} finding(s); re-run `corpus adjudicate preview` and re-adjudicate. "
            f"Requeued record_ids: {drifted}",
            drifted,
        )

    grouped = group_observations_by_record(
        observations, {str(item["record_id"]) for item in items}, "adjudicate export"
    )

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
            "tier": None,
            "posterior_eligible": False,
            "rubric_version": item["rubric_version"],
        }
        tier = classify_tier(entry)
        if tier == "gold" and not gold_eligible:
            # Structural gate passed but the human gate did not (rater
            # conflict or review-required): the finding stays out of gold.
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
