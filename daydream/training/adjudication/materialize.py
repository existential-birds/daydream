"""Materialize deterministic sessions.jsonl and a pinned preview-manifest.json.

Acquire fresh production evidence through read-only harvest services; legacy
and index-only history can supply stored resolutions. Serialize all findings
through the immutable FindingRecord owner. Never append observations, update
resume/completion markers, or write the hydrated SQLite index.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import closing
from pathlib import Path
from typing import Any

from daydream.archive.hydrate import HubUnavailableError
from daydream.archive.index import readonly_connection
from daydream.json_utils import atomic_write_bytes, canonical_json as _canonical, umask_derived_mode
from daydream.training.adjudication.preview import (
    _SESSIONS_OUT_FILENAME as _SESSIONS_OUT_FILENAME,
    _load_sessions,
)
from daydream.training.adjudication.snapshot import FindingRecord, snapshot_id
from daydream.training.dispositions import DECISIVE_DISPOSITIONS
from daydream.training.labeler_signals import resolution_from_dict

__all__ = ["run_materialize"]

_MANIFEST_FILENAME = "preview-manifest.json"
_ANNOTATIONS_FILENAME = "annotations.jsonl"

def index_sessions(
    index_root: Path,
) -> tuple[list[dict[str, Any]], str, dict[str, dict[str, Any]] | None]:
    """Read source sessions and revision, retaining the hydrated run inventory.

    JSONL takes precedence and supplies no acquired SQL authority (None). Import
    may independently read its co-located archive; preview/materialize need not.
    """
    if (index_root / _SESSIONS_OUT_FILENAME).is_file():
        sessions, revision = _load_sessions(index_root)
        return sessions, revision, None
    return _sessions_from_hydrated_stage(index_root)


def _sessions_from_hydrated_stage(
    index_root: Path,
) -> tuple[list[dict[str, Any]], str, dict[str, dict[str, Any]]]:
    """Materialize fresh bronze evidence, falling back to stored resolutions for legacy history.

    Acquire production evidence even when annotations exist, so drift checks see
    current replies. DB-only histories and embedded legacy resolutions use the
    winning observation adapter; evidence-only sessions emit no records.

    _winning_observation selects deterministic human-first precedence and marks
    conflicting non-human decisive generations. The conflict flag travels with
    all session records; run_materialize neutralizes their dispositions to keep
    them out of gold while canonical harvest retains original provenance.

    Read SQLite without sidecars or writes. Require exactly one downloads/
    revision directory as the pinned source commit."""
    if not (index_root / "index.db").is_file():
        raise HubUnavailableError(
            f"hydrated index sessions file not found: {index_root / 'sessions.jsonl'}"
        )
    try:
        conn = readonly_connection(index_root)
    except ValueError as exc:
        raise HubUnavailableError(str(exc)) from exc
    with closing(conn):
        runs = {str(row["session_id"]): dict(row) for row in conn.execute("SELECT * FROM runs")}
        if not runs:
            raise HubUnavailableError(f"hydrated index at {index_root} has no runs")
        sessions: list[dict[str, Any]] = []
        for row in runs.values():
            session_id = str(row["session_id"])
            observations = [dict(observation) for observation in conn.execute(
                "SELECT * FROM label_observations WHERE session_id = ?", (session_id,),
            )]
            conflicting = False
            resolutions, trajectory = _source_resolutions_readonly(index_root, row)
            session: dict[str, Any]
            if observations:
                winner, conflicting = _winning_observation(observations)
                rubric_raw = winner.get("rubric_json")
                if rubric_raw is not None:
                    try:
                        rubric = json.loads(rubric_raw)
                    except (KeyError, TypeError, ValueError) as exc:
                        raise HubUnavailableError(
                            f"session {session_id!r}: unreadable winning rubric_json: {exc}"
                        ) from exc
                    if not isinstance(rubric, dict):
                        raise HubUnavailableError(
                            f"session {session_id!r}: winning rubric_json is not an object"
                        )
                    per_finding = rubric.get("per_finding_resolutions")
                    if resolutions is None and isinstance(per_finding, list) and per_finding:
                        resolutions = per_finding
            # DB-only history and legacy embedded resolutions remain supported.
            # Production bronze was already acquired above, independently of the
            # winner's dispositions and without changing the pinned source tree.
            if resolutions is None:
                if trajectory is None:
                    continue
                resolutions = trajectory.get("resolutions")
                if not isinstance(resolutions, list) or not resolutions:
                    raise HubUnavailableError(
                        f"hydrated trajectory for session {session_id!r} carries no "
                        "per-finding resolutions to materialize"
                    )
            if not resolutions:
                continue  # An explicitly empty bronze finding inventory has no records.
            session = {
                "session_id": session_id,
                "trajectory_id": session_id,
                "segment_id": session_id,
                "resolutions": resolutions,
            }
            if conflicting:
                session["conflicting"] = True
            sessions.append(session)
    downloads = index_root / "downloads"
    if not downloads.is_dir():
        raise HubUnavailableError(f"hydrated index at {index_root} has no downloads/ revision pin")
    revisions = sorted(p.name for p in downloads.iterdir() if p.is_dir())
    if len(revisions) != 1:
        raise HubUnavailableError(
            f"hydrated index at {index_root} has {len(revisions)} downloaded revisions; "
            "expected exactly one pinned source commit"
        )
    return sessions, revisions[0], runs


def _source_resolutions_readonly(
    index_root: Path, row: dict[str, Any],
) -> tuple[list[dict[str, Any]] | None, dict[str, Any] | None]:
    """Acquire live evidence for production bronze, including on re-harvest.

    Legacy snapshots with embedded resolutions and DB-only imported histories
    retain their stored-evidence adapter. Production trajectories never need
    an annotation field or a prior canonical write.
    """
    from daydream.training.harvest import HarvestConfig, HarvestPass
    from daydream.training.harvest_types import HarvestRow
    from daydream.trajectory import run_directory, run_document_path
    from daydream.ui import create_console

    run_dir = run_directory(index_root, str(row["session_id"]))
    path = run_document_path(run_dir)
    if not path.is_file():
        return None, None
    try:
        trajectory = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(trajectory, dict):
            raise ValueError("trajectory must be an object")
        if "resolutions" in trajectory:
            return None, trajectory
        findings_path = run_dir / "findings.json"
        findings = json.loads(findings_path.read_text()).get("findings") if findings_path.is_file() else None
        if findings == [] or (findings is None and row.get("total_findings") == 0):
            return [], trajectory
        # Hydration owns the bronze path; never follow an archived producer's
        # absolute archive_path into a different tree.
        harvest_row = HarvestRow.from_mapping(
            {**row, "archive_path": str(run_dir.resolve())}, row_number=1,
        )
        config = HarvestConfig(archive_dir=index_root, dry_run=True)
        _linked_row, payload = HarvestPass(config).collect_annotation(harvest_row, console=create_console())
        rubric = json.loads(payload.rubric_json or "{}")
        resolutions = rubric.get("per_finding_resolutions")
        if not isinstance(resolutions, list) or not resolutions:
            raise ValueError("no per-finding resolutions; missing recorded finding identities")
        by_fingerprint = {str(finding["fingerprint"]): finding for finding in findings or []}
        provenance_keys = (
            "profile_schema_version", "profile_name", "profile_source_kind", "profile_digest", "stack",
        )
        return [
            {
                **{key: row.get(key) for key in provenance_keys},
                **{key: value for key, value in by_fingerprint.get(resolution["fingerprint"], {}).items()
                   if key in provenance_keys},
                **resolution,
            }
            for resolution in resolutions
        ], trajectory
    except Exception as exc:
        raise HubUnavailableError(
            f"semantic preview for session {row['session_id']!r} failed: {exc}"
        ) from exc


def _winning_observation(
    observations: list[dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    """Select the latest row per archive dedup key, then human-first/latest overall.

    Conflict requires distinct decisive label sets among non-human winners.
    Policy bumps, edited evidence, and label-preserving overlays agree; human
    overrides are authoritative, and a non-decisive generation resolving to a
    decisive one is an evolution rather than a conflict."""
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for obs in observations:
        key = (
            obs.get("evidence_sha"),
            obs.get("labeler_policy_version"),
            obs.get("reply_evidence_digest"),
            obs.get("labels"),
            obs.get("has_posterior"),
        )
        existing = groups.get(key)
        if existing is None or str(obs.get("observed_at", "")) > str(existing.get("observed_at", "")):
            groups[key] = obs
    winners = list(groups.values())
    winner = max(
        winners,
        key=lambda o: (
            1 if o.get("source") == "human" else 0,
            str(o.get("observed_at", "")),
        ),
    )
    # Human overrides and non-decisive generations do not compete with
    # automatic decisive labels; agreeing evidence generations remain gold-eligible.
    decisive_sets = {
        o.get("labels")
        for o in winners
        if o.get("source") != "human" and _labels_claim_decisive(o.get("labels"))
    }
    return winner, len(decisive_sets) > 1


def _labels_claim_decisive(labels: Any) -> bool:
    """Recognize decisive finding-* labels in a list or JSON array string.

    Malformed data, non-finding labels and non-decisive labels claim nothing."""
    if isinstance(labels, str):
        try:
            labels = json.loads(labels)
        except ValueError:
            return False
    if not isinstance(labels, list):
        return False
    return any(
        isinstance(label, str) and label.startswith("finding-") and label[len("finding-"):] in DECISIVE_DISPOSITIONS
        for label in labels
    )


def run_materialize(
    index_root: Path,
    out_dir: Path,
    *,
    pin: dict[str, str],
    dry_run: bool = False,
) -> dict[str, Any]:
    """Atomically write canonical sessions sorted by record_id and their preview manifest.

    Include every disposition and all pin components. snapshot_id binds the pin
    and evidence digests, so changed evidence changes identity. Missing/empty pin
    fields raise ValueError; missing/unreadable indexes raise HydrationError and
    symbolic revisions raise MovingBranchError. dry_run validates and returns the
    same summary without writing. Identical inputs produce identical bytes.
    """
    # Validate the pin before touching its components in the loop body:
    # ``snapshot_id`` raises the documented ValueError naming the missing
    # component, never a KeyError from ``pin["evidence_observed_at"]``.
    pin_id = snapshot_id(pin)
    sessions, index_revision, _runs = index_sessions(index_root)

    records: list[dict[str, Any]] = []
    for session in sessions:
        resolutions = session.get("resolutions")
        if not isinstance(resolutions, list):
            continue
        for row in resolutions:
            if not isinstance(row, dict):
                raise ValueError(f"materialize: non-object resolution row in session data: {row!r}")
            resolution = resolution_from_dict(row)
            finding = FindingRecord.snapshot(
                session, resolution, evidence_observed_at=pin["evidence_observed_at"],
                as_of=pin.get("as_of"), conflicting=bool(session.get("conflicting")),
            )
            records.append(finding.canonical(project_conflict=True))
    records.sort(key=lambda r: str(r["record_id"]))

    id_digest = hashlib.sha256(
        (
            pin_id
            + ":"
            + hashlib.sha256(
                "".join(str(r["evidence_digest"]) for r in records).encode("utf-8")
            ).hexdigest()
        ).encode("utf-8")
    ).hexdigest()

    summary: dict[str, Any] = {
        "snapshot_id": id_digest,
        "index_revision": index_revision,
        "record_count": len(records),
    }
    if dry_run:
        return summary

    atomic_write_bytes(
        out_dir / _SESSIONS_OUT_FILENAME,
        "".join(_canonical(r) + "\n" for r in records).encode("utf-8"),
        fsync=False,
        dir_fsync=False,
        mode=umask_derived_mode(),
    )
    manifest: dict[str, Any] = dict(pin)
    manifest["snapshot_id"] = id_digest
    manifest["index_revision"] = index_revision
    atomic_write_bytes(
        out_dir / _MANIFEST_FILENAME,
        _canonical(manifest).encode("utf-8"),
        fsync=False,
        dir_fsync=False,
        mode=umask_derived_mode(),
    )
    return summary
