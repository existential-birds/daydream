"""Pure annotation views over validated, frozen record evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from daydream.dataset import SnapshotRecords
from daydream.json_utils import canonical_json
from daydream.training.adjudication.precedence import effective_adjudication, retained_reply_content
from daydream.training.labeler_versions import reply_evidence_digest
from daydream.training.record_identity import record_finding_id


def section_value(run: Mapping[str, Any], name: str) -> Any:
    section = run[name]
    return section["value"] if section["status"] == "available" else None


def latest_annotation(records: SnapshotRecords, run_id: str) -> dict[str, Any] | None:
    """Resolve eligible run outcome labels with human precedence over automation."""
    observations = [o for o in records.eligible_observations if o["run_id"] == run_id]
    automatic = [o for o in observations if o["payload"]["type"] == "harvest-annotation"]

    def recency(o: dict[str, Any]) -> tuple[datetime, str, str]:
        return datetime.fromisoformat(o["observed_at"]), o["author"], o["observation_id"]

    annotation = dict(max(automatic, key=recency)["payload"]["annotation"]) if automatic else None
    labels = [o for o in observations if o["payload"]["type"] == "run-label"]
    if labels:
        winner = max(labels, key=lambda o: (o["role"] in {"rater", "adjudicator"}, *recency(o)))
        if winner["role"] in {"rater", "adjudicator"} or annotation is None:
            annotation = annotation or {}
            annotation["labels"] = [] if winner["payload"]["label"] == "unknown" else [winner["payload"]["label"]]
            annotation["human_labeler"] = winner["author"] if winner["role"] in {"rater", "adjudicator"} else None
    return annotation


def finding_observations(records: SnapshotRecords) -> list[dict[str, Any]]:
    """Translate eligible typed judgments without reducing their retained evidence."""
    items = {
        run["run_id"]: {i["item_uid"]: i for i in (section_value(run, "findings") or {}).get("items", [])}
        for run in records.runs
    }
    rows = []
    for observation in records.eligible_observations:
        if observation["payload"]["type"] != "finding-judgment":
            continue
        run_id = observation["run_id"]
        item = items[run_id][observation["item_uid"]]
        rows.append(
            {
                **observation,
                **observation["payload"],
                "record_id": record_finding_id(run_id, run_id, run_id, item["item_uid"]),
                "labeler": observation["author"],
                "evidence": observation["semantic_evidence"],
            }
        )
    return rows


def sessions_from_snapshot(records: SnapshotRecords, *, overlay_judgments: bool = True) -> list[dict[str, Any]]:
    """Enumerate the complete host finding population, including unanswered evidence.

    Identity is run + host item_uid; fingerprints remain the external comment join.
    Missing judgments never drop a captured finding. Only eligible snapshot evidence
    participates, and human precedence is delegated to the existing reducer.
    """
    sessions = []
    for run in records.runs:
        run_id = run["run_id"]
        finding_payload = section_value(run, "findings") or {}
        annotation = latest_annotation(records, run_id)
        annotation_history = [
            o for o in records.eligible_observations
            if o["run_id"] == run_id and o["payload"]["type"] == "harvest-annotation"
        ]
        latest_harvest = max(
            annotation_history,
            key=lambda o: (datetime.fromisoformat(o["observed_at"]), o["author"], o["observation_id"]),
            default=None,
        )
        rubric = None
        if annotation and annotation.get("rubric_json"):
            rubric = json.loads(annotation["rubric_json"])
        by_fingerprint = {r["fingerprint"]: r for r in (rubric or {}).get("per_finding_resolutions") or []}
        resolutions = []
        for item in finding_payload.get("items", []):
            raw = by_fingerprint.get(item["fingerprint"], {})
            evidence = raw.get("evidence") or []
            disposition = raw.get("disposition", "unanswered")
            history = [
                o
                for o in records.eligible_observations
                if o["run_id"] == run_id
                and o.get("item_uid") == item["item_uid"]
                and o["payload"]["type"] == "finding-judgment"
            ]
            resolved = None
            # Automatic history is the current evidence authority. Human judgments
            # apply only to the same digest, so edited replies reopen prior labels.
            automatic = [o for o in history if o["role"] == "automatic"]
            if automatic:
                current = max(automatic, key=lambda o: (datetime.fromisoformat(o["observed_at"]), o["observation_id"]))
                evidence = current["semantic_evidence"]
                disposition = current["payload"]["disposition"]
                digest = current["evidence_digest"]
                scheme = current.get("evidence_digest_scheme", "canonical-json-v1")
            elif raw:
                digest = raw.get("evidence_digest") or reply_evidence_digest(evidence)
                scheme = "reply-evidence-v1"
            elif history:
                current = max(history, key=lambda o: (datetime.fromisoformat(o["observed_at"]), o["observation_id"]))
                evidence = current["semantic_evidence"]
                digest = current["evidence_digest"]
                scheme = current.get("evidence_digest_scheme", "canonical-json-v1")
            else:
                digest = reply_evidence_digest(evidence)
                scheme = "reply-evidence-v1"
            if not digest:
                digest = hashlib.sha256(canonical_json(evidence).encode()).hexdigest()
            if overlay_judgments and history:
                matching = [
                    {
                        **o,
                        **o["payload"],
                        "record_id": item["item_uid"],
                        "labeler": o["author"],
                        "evidence": o["semantic_evidence"],
                    }
                    for o in history
                    if o["evidence_digest"] == digest
                ]
                if matching:
                    resolved = effective_adjudication(matching)
                    disposition = resolved["disposition"]
            auto_conflict = (
                len(
                    {
                        o["payload"]["disposition"]
                        for o in automatic
                        if o["payload"]["disposition"] in {"accepted", "rejected"} and o["evidence_digest"] == digest
                    }
                )
                > 1
            )
            if auto_conflict and (resolved is None or resolved["role"] not in {"rater", "adjudicator"}):
                disposition = "ambiguous"
            if disposition == "unknown":
                disposition = "ambiguous"
            source = next(
                (
                    {"stack": stack["stack"], **c}
                    for stack in finding_payload.get("claims", [])
                    for c in (
                        stack["records"].get("issues", []) if isinstance(stack["records"], dict) else stack["records"]
                    )
                    if c.get("uid") in item["source_uids"]
                ),
                {},
            )
            content_history = [o for o in history if o["evidence_digest"] == digest]
            raw_digest = raw.get("evidence_digest") or reply_evidence_digest(raw.get("evidence") or [])
            if latest_harvest and raw and raw_digest == digest:
                content_history.append({**latest_harvest, **raw})
            resolution = {
                **run.get("provenance", {}),
                **source,
                **raw,
                "item_uid": item["item_uid"],
                "fingerprint": item["fingerprint"],
                "disposition": disposition,
                "evidence": evidence,
                "evidence_digest": digest,
                "evidence_digest_scheme": scheme,
                **retained_reply_content(content_history),
                "comment_id": raw.get("comment_id"),
                "profile": "pr_review"
                if (rubric or {}).get("posterior_source") == "pr_review"
                else raw.get("profile", run.get("provenance", {}).get("profile")),
            }
            if resolved:
                resolution.update({key: resolved[key] for key in ("gold_eligible", "conflict", "review_required")})
                if resolved["conflict"] or resolved["review_required"]:
                    resolution["disposition"] = "ambiguous"
            if auto_conflict and (resolved is None or resolved["role"] not in {"rater", "adjudicator"}):
                resolution.update(conflict=True, gold_eligible=False)
            cutoff = records.snapshot["valid_before"]
            resolution["evidence_after_as_of"] = bool(
                cutoff
                and any(
                    e.get("created_at") and datetime.fromisoformat(e["created_at"]) > datetime.fromisoformat(cutoff)
                    for e in evidence
                )
            )
            if resolved:
                resolution["valid_at"] = next(
                    (o["valid_at"] for o in history if o["observation_id"] == resolved.get("observation_id")), None
                )
            resolutions.append(resolution)
        task = section_value(run, "original_task") or {}
        license_evidence = None
        for observation in sorted(
            records.eligible_observations, key=lambda o: (datetime.fromisoformat(o["observed_at"]), o["observation_id"])
        ):
            payload = observation["payload"]
            if observation["run_id"] == run_id and payload["type"] == "enrichment" and payload["kind"] == "license":
                license_evidence = (
                    payload["evidence"]["value"] if payload["evidence"]["status"] == "available" else None
                )
        sessions.append(
            {
                "session_id": run_id,
                "trajectory_id": run_id,
                "segment_id": run_id,
                "repo_slug": (task.get("repository") or {}).get("repo_slug"),
                "license_evidence": license_evidence,
                "annotation": annotation,
                "resolutions": resolutions,
            }
        )
    return sessions


def validate_output_path(store_dir: Path, output: Path) -> None:
    """Refuse derived output in or over the authoritative record namespace."""
    source = store_dir.expanduser().resolve()
    destination = output.expanduser().resolve()
    if source.is_relative_to(destination) or destination.is_relative_to(source):
        raise ValueError("annotation output overlaps the authoritative record store")
