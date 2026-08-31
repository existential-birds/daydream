"""Canonical harvest of per-finding annotations into the archive (issue #1055).

Re-verifies the materialized preview snapshot (every disposition — automatic
decisive, human-decisive, and non-decisive) against a freshly built
*complete* queue — ``build_queue(..., include_decisive=True)`` — over the
hydrated index, so the automatic decisive records drift-check too. It merges
human observations under three-tier
precedence, appends exactly one ``label_observations`` row per session via
``archive.index.append_label_observation`` (the auto dedup key keys on the
pin's ``snapshot_id`` plus the archived ``rubric_json`` — fed through
``evidence_sha`` — so unchanged-pin/unchanged-rubric re-runs are no-ops ⇒
exactly-once while a changed pin or a label-preserving observation-overlay
change appends a fresh generation, M8), and emits ``annotations.jsonl`` from the
*same* in-memory merged records used for the append — no second serialization
(M5). Digest drift raises :class:`AnnotationDriftError` **before any write**
(fail-closed-then-requeue, M5).

Unlike ``adjudication/harvest.py`` this path does not touch
``training/harvest.py``'s GitHub-facing signals: the #980 semantic evidence
was already collected into the index at materialization; canonical harvest
only re-verifies it against the preview pin.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from daydream.archive.index import append_label_observation
from daydream.training.adjudication.materialize import (
    _SESSIONS_OUT_FILENAME,
    _sessions_from_hydrated_stage,
)
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


def _evidence_after_as_of(record: Mapping[str, Any], as_of: str | None) -> bool:
    """Recorded-and-flagged ``as_of`` edge policy: True when any evidence entry's
    ``created_at`` is after the pin's ``as_of`` (both parsed with
    :func:`datetime.fromisoformat`, mirroring ``projector.py:_max_valid_at`` so
    ``Z`` vs ``+00:00`` spelling can never mis-order the comparison). Such
    records keep their evidence but are never gold-eligible."""
    if not as_of:
        return False
    pin_dt = datetime.fromisoformat(as_of)
    for entry in record.get("evidence") or []:
        if not isinstance(entry, Mapping):
            continue
        created_at = entry.get("created_at")
        if created_at and datetime.fromisoformat(str(created_at)) > pin_dt:
            return True
    return False


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
    # Dereferenced with ``pin["rubric_version"]`` in the session loop, so it must
    # be validated here (like ``labeler_version``) to raise the documented
    # ValueError naming the missing component, never an uncaught KeyError.
    rubric_version = pin.get("rubric_version")
    if not isinstance(rubric_version, str) or not rubric_version:
        raise ValueError(f"preview manifest at {manifest_path} is missing 'rubric_version'")
    return pin


def run_canonical_harvest(
    index_root: Path,
    materialize_dir: Path,
    archive_dir: Path,
    *,
    observations_path: Path | None = None,
) -> dict[str, Any]:
    """Harvest the materialized preview snapshot into canonical annotation storage.

    1. Re-derives the fresh complete queue over the hydrated index
       (``build_queue(..., include_decisive=True)``) and verifies every
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
       ``REPLY_CLASSIFIER_VERSION``). ``evidence_sha`` is a digest over the
       pin's content-addressed ``snapshot_id`` *plus* the archived
       ``rubric_json``, so the dedup key changes with the pin or with any
       rubric-content (observation-overlay) change: unchanged-pin re-runs are
       no-ops — exactly-once — while a changed pin (new snapshot id) or a
       label-preserving overlay edit under an unchanged pin appends a fresh
       generation carrying the new pin's flags/rubric.
    4. Emits ``materialize_dir/annotations.jsonl`` (one canonical-JSON record
       per finding, sorted by ``record_id``) from the same merged records.

    Returns ``{"appended_sessions", "skipped_sessions", "human_adjudicated",
    "record_count"}``.
    """
    pin = _load_pin(materialize_dir)
    materialized = _load_materialized_records(materialize_dir)
    if (index_root / _SESSIONS_OUT_FILENAME).is_file():
        sessions, _index_revision = _load_sessions(index_root)
    else:
        # Hydrated staging archive (materialize's primary flow): no
        # sessions.jsonl — derive the sessions from the SQLite index plus the
        # sanitized per-run trajectories, so the drift gate re-derives the
        # fresh queue over the same hydrated index the materialized preview
        # was built from instead of feeding the materialize dir back and
        # comparing each digest against itself (tautological).
        sessions, _index_revision = _sessions_from_hydrated_stage(index_root)
    # The complete set is the drift authority: widened materialization emits a
    # record for every disposition, so the fresh queue must include the
    # automatic decisive records too — an unresolved-only queue would
    # fail-closed on every decisive finding. Unchanged automatic decisive
    # records are preserved verbatim by the merge loop below: no observation
    # group ⇒ disposition untouched; ``human_labeler``/``human_role`` are only
    # set by the three-tier precedence branch.
    fresh_by_record_id = {
        str(item["record_id"]): item for item in build_queue(sessions, include_decisive=True)
    }

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
    flagged_after_as_of: list[str] = []
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
        record["evidence_after_as_of"] = _evidence_after_as_of(record, pin.get("as_of"))
        if record["evidence_after_as_of"]:
            flagged_after_as_of.append(record_id)
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
        rubric_json = _canonical(rubric)
        labels = sorted(
            {
                f"finding-{record['disposition']}"
                for record in session_records
                if record["disposition"] in DECISIVE_DISPOSITIONS
            }
        )
        # Pin member of the auto dedup key. The dedup tuple (M14) omits
        # rubric_json, so the digest fed through evidence_sha must cover the
        # rubric content itself: the content-addressed snapshot_id changes
        # with any pin change (AC 8), and the rubric_json digest turns on any
        # label-preserving observation-overlay change under an unchanged pin,
        # so a fresh generation carrying the updated rubric_json is appended
        # and the archived rubric always matches the emitted bundle's
        # pin/flags. Unchanged pin + unchanged rubric stays a deduped no-op
        # (exactly-once). Absent snapshot_id (legacy manifest) falls back to
        # None, the pre-change dedup behavior.
        snapshot_id = pin.get("snapshot_id")
        if snapshot_id is not None:
            generation_sha = hashlib.sha256(
                (str(snapshot_id) + ":" + rubric_json).encode("utf-8")
            ).hexdigest()
        else:
            generation_sha = None
        inserted = append_label_observation(
            archive_dir,
            session_id,
            labels=labels,
            pr_state=None,
            labeler_version=str(pin["labeler_version"]),
            evidence_sha=generation_sha,
            rubric_json=rubric_json,
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
        "evidence_after_as_of": sorted(flagged_after_as_of),
    }
