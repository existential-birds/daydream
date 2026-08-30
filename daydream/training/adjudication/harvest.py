"""Canonical fail-closed adjudication harvest (issue #984, Task 10).

Verifies the preview ledger's per-finding evidence digests against a freshly
built queue over the hydrated index, merges human judgments from the
observation store under three-tier precedence, and writes the
``adjudication.jsonl`` export atomically. Digest drift raises
:class:`AdjudicationDriftError` before anything is written — the export is
never produced in a drifted state (delta on
``corpus_v2.projector.run_build_corpus_v2``'s digest-pinned snapshot flow:
harvest verifies the *preview ledger's* digests rather than re-pinning its
own, so preview identities and digests are stable into the export by
construction).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from daydream.training.adjudication.observations import load_observations
from daydream.training.adjudication.precedence import DECISIVE_DISPOSITIONS, effective_adjudication
from daydream.training.adjudication.preview import _load_sessions
from daydream.training.adjudication.queue import build_queue
from daydream.training.corpus_v2.tiers import classify_tier

__all__ = ["AdjudicationDriftError", "run_harvest"]

_EXPORT_FILENAME = "adjudication.jsonl"

_EXPORT_KEYS = (
    "record_id",
    "evidence_digest",
    "fingerprint",
    "disposition",
    "evidence",
    "exclusion_reason",
    "profile",
    "stack",
    "session_id",
    "trajectory_id",
    "segment_id",
    "tier",
    "posterior_eligible",
    "rubric_version",
)


class AdjudicationDriftError(ValueError):
    """Raised when a queue item's evidence digest differs from the preview ledger.

    Fail-closed: the harvest's export is never written in a drifted state; the
    affected findings are requeued (reported via ``requeued_record_ids``).
    """

    def __init__(self, message: str, requeued_record_ids: list[str]) -> None:
        super().__init__(message)
        self.requeued_record_ids = requeued_record_ids


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _load_observations_grouped(
    observations_path: Path | None,
    queue_record_ids: set[str],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    if observations_path is None or not observations_path.is_file():
        return grouped
    for obs in load_observations(observations_path):
        record_id = str(obs["record_id"])
        if record_id not in queue_record_ids:
            raise ValueError(
                f"run_harvest: observation references record_id {record_id!r} which is not "
                f"in the adjudication queue over the hydrated index (observation evidence "
                f"digest {obs.get('evidence_digest')!r})"
            )
        grouped.setdefault(record_id, []).append(obs)
    return grouped


def run_harvest(
    index_root: Path,
    ledger_path: Path,
    out_dir: Path,
    *,
    observations_path: Path | None = None,
) -> dict[str, Any]:
    """Run the canonical adjudication harvest over a hydrated index.

    Re-derives the fresh queue over the index, verifies every preview-ledger
    ``record_id``'s evidence digest against the fresh evidence, then writes
    ``out_dir/adjudication.jsonl`` in the ``project_findings`` entry shape
    (``{fingerprint, disposition, evidence, exclusion_reason}`` plus
    ``record_id`` and ``evidence_digest``) — atomically via temp-file rename,
    so an interrupted harvest leaves no partial export.

    - Digest drift ⇒ :class:`AdjudicationDriftError` listing
      ``requeued_record_ids``; the export is never written.
    - Human judgments from ``observations_path`` (when supplied) merge under
      three-tier precedence only when their pinned digest matches the fresh
      evidence and the resolution is gold-eligible (no rater conflict, no
      review-required flag); the final tier is classified by
      ``corpus_v2.tiers.classify_tier`` so the gold gate has one
      implementation. Conflicted/review-required decisive judgments stay out
      of the gold tier.
    - ``posterior_eligible`` is true only for gold-tier findings with
      ``profile == "pr_review"`` (finding-level twin of the run-level C5
      posterior feed rule).

    Failure policy: a missing/unreadable preview ledger or sessions file
    raises the ``HydrationError`` family; a ledger entry whose ``record_id``
    is absent from the fresh queue, or an observation referencing a record
    absent from the queue, raises ``ValueError`` naming the record; drift
    raises :class:`AdjudicationDriftError`. No fallback digest, no silent
    skip.
    """
    sessions, index_revision = _load_sessions(index_root)
    items = build_queue(sessions)
    by_record_id = {str(item["record_id"]): item for item in items}

    if not ledger_path.is_file():
        raise FileNotFoundError(
            f"run_harvest: preview ledger not found (run `corpus adjudicate preview` first): "
            f"{ledger_path}"
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
                f"run_harvest: preview ledger record_id {record_id!r} is absent from the "
                "freshly built adjudication queue over the index"
            )
        if str(fresh["evidence_digest"]) != str(ledger_item["evidence_digest"]):
            drifted.append(record_id)
    if drifted:
        raise AdjudicationDriftError(
            f"run_harvest: evidence digests drifted from the preview ledger for "
            f"{len(drifted)} finding(s); re-run `corpus adjudicate preview` and re-adjudicate. "
            f"Requeued record_ids: {drifted}",
            drifted,
        )

    grouped = _load_observations_grouped(
        observations_path, {str(item["record_id"]) for item in items}
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
        assert set(entry) == set(_EXPORT_KEYS), "export key drift"
        exported.append(entry)
    exported.sort(key=lambda e: str(e["record_id"]))

    out_dir.mkdir(parents=True, exist_ok=True)
    export_path = out_dir / _EXPORT_FILENAME
    payload = "".join(_canonical(e) + "\n" for e in exported)
    tmp_path = out_dir / (_EXPORT_FILENAME + ".tmp")
    tmp_path.write_text(payload, encoding="utf-8")
    os.replace(tmp_path, export_path)

    return {
        "index_revision": index_revision,
        "export_sha256": hashlib.sha256(export_path.read_bytes()).hexdigest(),
        "item_count": len(exported),
        "exported": [
            {
                "record_id": e["record_id"],
                "profile": e["profile"],
                "tier": e["tier"],
                "disposition": e["disposition"],
                "posterior_eligible": e["posterior_eligible"],
            }
            for e in exported
        ],
        "drifted_record_ids": [],
    }
