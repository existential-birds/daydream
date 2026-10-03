"""Drift-gated harvest of annotations into the archive.

Rebuild the complete queue, including automatic decisive records, through the
read-only semantic builder before writing. Merge human precedence once, then
use those records for both session observations and annotations.jsonl. The
append identity includes the snapshot pin and rubric: identical re-runs are
no-ops, while label-preserving evidence changes append a fresh generation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from daydream.archive.index import append_label_observation
from daydream.json_utils import atomic_write_bytes, canonical_json as _canonical, umask_derived_mode
from daydream.training.adjudication.materialize import (
    _ANNOTATIONS_FILENAME,
    _MANIFEST_FILENAME,
    _SESSIONS_OUT_FILENAME,
    index_sessions,
)
from daydream.training.adjudication.observations import (
    group_observations_by_record,
    load_observations,
)
from daydream.training.adjudication.precedence import (
    DECISIVE_DISPOSITIONS,
)
from daydream.training.adjudication.queue import build_queue
from daydream.training.adjudication.snapshot import FindingRecord, record_evidence_digest
from daydream.training.labeler_versions import REPLY_CLASSIFIER_VERSION

__all__ = ["AnnotationDriftError", "run_canonical_harvest"]


class AnnotationDriftError(ValueError):
    """Raised when a materialized record's evidence digest differs from the fresh queue.

    Fail-closed: the canonical append and ``annotations.jsonl`` are never
    written in a drifted state; the affected findings are requeued (reported
    via ``requeued_record_ids``).
    """

    def __init__(self, message: str, requeued_record_ids: list[str]) -> None:
        super().__init__(message)
        self.requeued_record_ids = requeued_record_ids


def read_jsonl(path: Path, *, missing: str, invalid: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(missing)
    try:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{invalid} at {path}: {exc}") from exc


def _load_materialized_records(materialize_dir: Path) -> list[dict[str, Any]]:
    records_path = materialize_dir / _SESSIONS_OUT_FILENAME
    return read_jsonl(
        records_path,
        missing=(
            f"materialized preview snapshot not found (run `corpus adjudicate materialize` "
            f"first): {records_path}"
        ),
        invalid="unreadable materialized snapshot",
    )


def _read_manifest(manifest_path: Path) -> dict[str, Any]:
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"preview manifest not found (run `corpus adjudicate materialize` first): "
            f"{manifest_path}"
        )
    try:
        pin: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable preview manifest at {manifest_path}: {exc}") from exc
    return pin


def _load_pin(materialize_dir: Path) -> dict[str, Any]:
    manifest_path = materialize_dir / _MANIFEST_FILENAME
    pin = _read_manifest(manifest_path)
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
    """Verify all materialized digests against the fresh complete queue before any write.

    Drift raises AnnotationDriftError with record ids. Merge human precedence only
    when a decisive observation matches current evidence; unknown record ids raise.
    Append one observation per session using the merged rubric, shared reply digest,
    pinned labeler version, and current reply-classifier version. evidence_sha binds
    snapshot_id plus rubric_json: identical reruns deduplicate, while changed pins
    or label-preserving overlays append a new generation.

    Write annotations.jsonl from those same merged records in record_id order.
    Return appended/skipped sessions, human_adjudicated, and record_count.
    """
    pin = _load_pin(materialize_dir)
    materialized = _load_materialized_records(materialize_dir)
    sessions, _index_revision, _runs = index_sessions(index_root)
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
    # The fresh session derivation is the conflict authority, never the
    # materialized snapshot's flags: a session turned conflicting *after*
    # materialize (runbook step-3b import appending a disagreeing generation
    # to the same index.db) passes the evidence-digest drift gate below unless
    # the stack re-derives its ``conflicting`` verdict here and stamps it onto
    # the merged records before the decisive-label projection.
    fresh_conflicting_sessions = {
        str(session.get("session_id"))
        for session in sessions
        if session.get("conflicting")
    }

    # Fail-closed drift gate: verify BEFORE any write.
    materialized_ids = {str(record["record_id"]) for record in materialized}
    if len(materialized_ids) != len(materialized):
        raise ValueError("materialized snapshot contains duplicate finding identities")
    drifted = sorted(set(fresh_by_record_id) - materialized_ids)
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
    observations = load_observations(observations_path) if observations_path is not None else []
    grouped = group_observations_by_record(
        observations, materialized_ids, "run_canonical_harvest"
    )

    human_adjudicated = 0
    flagged_after_as_of: list[str] = []
    findings: list[FindingRecord] = []
    for record in materialized:
        finding, human = FindingRecord.from_annotation(record).adjudicate(
            grouped.get(str(record["record_id"]), []), fresh_by_record_id.get(str(record["record_id"])),
            conflicting=str(record.get("session_id")) in fresh_conflicting_sessions,
            as_of=pin.get("as_of"),
        )
        human_adjudicated += int(human)
        if finding.metadata["evidence_after_as_of"]:
            flagged_after_as_of.append(str(record["record_id"]))
        findings.append(finding)
    findings.sort(key=lambda finding: str(finding.metadata["record_id"]))
    merged_records = [finding.canonical() for finding in findings]

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
        # Decisive labels come from every decisive record EXCEPT conflicted
        # ones: a session whose harvester generations disagree is non-gold per
        # existing precedence — its dispositions never project into
        # ``finding-<disposition>`` labels. The full record (``conflicting``
        # flag included) still lands in the archive ``rubric_json`` —
        # provenance preserved, and acceptance is never inferred from merge
        # state (the flag is set by the materializer and merely carried
        # through the merge loop above). The ``annotations.jsonl`` projection
        # row (written below) neutralizes the conflicted disposition so the
        # projection schema gate never classifies it gold.
        labels = sorted(
            {
                f"finding-{record['disposition']}"
                for record in session_records
                if record["disposition"] in DECISIVE_DISPOSITIONS
                and not record.get("conflicting")
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

    records_path = materialize_dir / _ANNOTATIONS_FILENAME
    atomic_write_bytes(
        records_path,
        "".join(_canonical(finding.canonical(project_conflict=True)) + "\n" for finding in findings).encode("utf-8"),
        fsync=False,
        dir_fsync=False,
        mode=umask_derived_mode(),
    )

    return {
        "appended_sessions": appended_sessions,
        "skipped_sessions": skipped_sessions,
        "human_adjudicated": human_adjudicated,
        "record_count": len(merged_records),
        "evidence_after_as_of": sorted(flagged_after_as_of),
    }
