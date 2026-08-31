"""Canonical harvest of per-finding annotations into the archive (issue #1055).

Re-verifies the materialized preview snapshot against a freshly built queue
over the hydrated index, merges human observations under three-tier
precedence, appends exactly one ``label_observations`` row per session via
``archive.index.append_label_observation`` (the existing auto dedup key makes
re-runs no-ops ⇒ exactly-once, M8), and emits ``annotations.jsonl`` from the
*same* in-memory merged records used for the append — no second serialization
(M5). Digest drift raises :class:`AnnotationDriftError` **before any write**
(fail-closed-then-requeue, M5).

Unlike ``adjudication/harvest.py`` this path does not touch
``training/harvest.py``'s GitHub-facing signals: the #980 semantic evidence
was already collected into the index at materialization; canonical harvest
only re-verifies it against the preview pin.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from daydream.archive.index import append_label_observation
from daydream.training.adjudication.materialize import _SESSIONS_OUT_FILENAME
from daydream.training.adjudication.observations import load_observations
from daydream.training.adjudication.precedence import (
    DECISIVE_DISPOSITIONS,
    effective_adjudication,
)
from daydream.training.adjudication.preview import _load_sessions
from daydream.training.adjudication.queue import build_queue
from daydream.training.adjudication.snapshot import record_evidence_digest
from daydream.training.labeler_versions import REPLY_CLASSIFIER_VERSION

__all__ = ["AnnotationDriftError", "run_canonical_harvest"]

_ANNOTATIONS_FILENAME = "annotations.jsonl"
_MANIFEST_FILENAME = "preview-manifest.json"

_HUMAN_ROLES = frozenset({"rater", "adjudicator"})


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class AnnotationDriftError(ValueError):
    """Raised when a materialized record's evidence digest differs from the fresh queue.

    Fail-closed: the canonical append and ``annotations.jsonl`` are never
    written in a drifted state; the affected findings are requeued (reported
    via ``requeued_record_ids``).
    """

    def __init__(self, message: str, requeued_record_ids: list[str]) -> None:
        super().__init__(message)
        self.requeued_record_ids = requeued_record_ids


def _load_materialized_records(materialize_dir: Path) -> list[dict[str, Any]]:
    records_path = materialize_dir / _SESSIONS_OUT_FILENAME
    if not records_path.is_file():
        raise FileNotFoundError(
            f"materialized preview snapshot not found (run `corpus adjudicate materialize` "
            f"first): {records_path}"
        )
    try:
        return [
            json.loads(line)
            for line in records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable materialized snapshot at {records_path}: {exc}") from exc


def _load_pin(materialize_dir: Path) -> dict[str, Any]:
    manifest_path = materialize_dir / _MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"preview manifest not found (run `corpus adjudicate materialize` first): "
            f"{manifest_path}"
        )
    try:
        pin: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable preview manifest at {manifest_path}: {exc}") from exc
    labeler_version = pin.get("labeler_version")
    if not isinstance(labeler_version, str) or not labeler_version:
        raise ValueError(f"preview manifest at {manifest_path} is missing 'labeler_version'")
    return pin


def run_canonical_harvest(
    index_root: Path,
    materialize_dir: Path,
    archive_dir: Path,
    *,
    observations_path: Path | None = None,
) -> dict[str, Any]:
    """Harvest the materialized preview snapshot into canonical annotation storage.

    1. Re-derives the fresh queue over the hydrated index and verifies every
       materialized record's ``evidence_digest`` against the fresh queue
       **before any write**; drift raises :class:`AnnotationDriftError`
       listing the drifted ``record_id``s.
    2. Merges human observations under three-tier precedence: a human
       (rater/adjudicator) observation with a decisive disposition matching
       the current evidence digest overrides the automatic disposition in the
       stored per-finding record. An observation referencing an unknown
       ``record_id`` raises ``ValueError`` naming it.
    3. Groups records by ``session_id`` and appends one ``label_observations``
       row per session (``rubric_json`` carries the merged per-finding records,
       ``reply_evidence_digest`` is the shared serializer's session digest,
       ``labeler_version`` is the pin's, ``reply_classifier_version`` is
       ``REPLY_CLASSIFIER_VERSION``). The existing auto dedup key makes
       unchanged re-runs no-ops — exactly-once.
    4. Emits ``materialize_dir/annotations.jsonl`` (one canonical-JSON record
       per finding, sorted by ``record_id``) from the same merged records.

    Returns ``{"appended_sessions", "skipped_sessions", "human_adjudicated",
    "record_count"}``.
    """
    pin = _load_pin(materialize_dir)
    materialized = _load_materialized_records(materialize_dir)
    sessions, _index_revision = _load_sessions(index_root)
    fresh_by_record_id = {str(item["record_id"]): item for item in build_queue(sessions)}

    # Fail-closed drift gate: verify BEFORE any write.
    drifted: list[str] = []
    for record in materialized:
        record_id = str(record["record_id"])
        fresh = fresh_by_record_id.get(record_id)
        if fresh is None:
            raise ValueError(
                f"materialized snapshot record_id {record_id!r} is absent from the "
                "freshly built adjudication queue over the index"
            )
        if str(fresh["evidence_digest"]) != str(record.get("evidence_digest")):
            drifted.append(record_id)
    if drifted:
        raise AnnotationDriftError(
            f"evidence digests drifted from the materialized preview snapshot for "
            f"{len(drifted)} finding(s); re-run `corpus adjudicate materialize` and "
            f"re-adjudicate. Requeued record_ids: {drifted}",
            drifted,
        )

    # Merge human observations under three-tier precedence (M4/M5).
    known_record_ids = {str(record["record_id"]) for record in materialized}
    grouped: dict[str, list[dict[str, Any]]] = {}
    if observations_path is not None and observations_path.is_file():
        for obs in load_observations(observations_path):
            record_id = str(obs["record_id"])
            if record_id not in known_record_ids:
                raise ValueError(
                    f"run_canonical_harvest: observation references record_id {record_id!r} "
                    f"which is not in the materialized snapshot over the hydrated index "
                    f"(observation evidence digest {obs.get('evidence_digest')!r})"
                )
            grouped.setdefault(record_id, []).append(obs)

    human_adjudicated = 0
    merged_records: list[dict[str, Any]] = []
    for record in materialized:
        record = dict(record)
        record_id = str(record["record_id"])
        if record_id in grouped:
            resolved = effective_adjudication(grouped[record_id])
            if (
                resolved["role"] in _HUMAN_ROLES
                and resolved["evidence_digest"] == str(record["evidence_digest"])
                and resolved["disposition"] in DECISIVE_DISPOSITIONS
            ):
                record["disposition"] = resolved["disposition"]
                record["human_labeler"] = resolved["labeler"]
                record["human_role"] = resolved["role"]
                human_adjudicated += 1
        merged_records.append(record)
    merged_records.sort(key=lambda r: str(r["record_id"]))

    # One AnnotationPayload-shaped row per session, appended exactly once.
    by_session: dict[str, list[dict[str, Any]]] = {}
    for record in merged_records:
        by_session.setdefault(str(record["session_id"]), []).append(record)

    appended_sessions = 0
    skipped_sessions = 0
    for session_id, session_records in sorted(by_session.items()):
        rubric = {
            "per_finding_resolutions": session_records,
            "rubric_version": pin["rubric_version"],
        }
        labels = sorted(
            {
                f"finding-{record['disposition']}"
                for record in session_records
                if record["disposition"] in DECISIVE_DISPOSITIONS
            }
        )
        inserted = append_label_observation(
            archive_dir,
            session_id,
            labels=labels,
            pr_state=None,
            labeler_version=str(pin["labeler_version"]),
            evidence_sha=None,
            rubric_json=_canonical(rubric),
            valid_at=None,
            has_posterior=False,
            reply_classifier_version=REPLY_CLASSIFIER_VERSION,
            reply_evidence_digest=record_evidence_digest(
                [list(record.get("evidence") or []) for record in session_records]
            ),
        )
        if inserted:
            appended_sessions += 1
        else:
            skipped_sessions += 1

    # annotations.jsonl from the same in-memory merged records — no second shape.
    records_path = materialize_dir / _ANNOTATIONS_FILENAME
    tmp_path = records_path.with_name(records_path.name + ".tmp")
    tmp_path.write_text(
        "".join(_canonical(record) + "\n" for record in merged_records), encoding="utf-8"
    )
    tmp_path.replace(records_path)

    return {
        "appended_sessions": appended_sessions,
        "skipped_sessions": skipped_sessions,
        "human_adjudicated": human_adjudicated,
        "record_count": len(merged_records),
    }
