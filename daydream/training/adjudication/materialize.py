"""Preview materialization of per-finding annotation snapshots (issue #1055).

Runs the #980 semantic resolutions stored in a hydrated index's
``label_observations.rubric_json`` through the shared serializer
(``snapshot.build_canonical_record``) and emits a deterministic
``sessions.jsonl`` plus a pin-pinned ``preview-manifest.json``.

Preview mode guarantees (AC 4 / M2): never appends ``label_observations``,
never writes any resume-cache or harvest-complete marker. When the input is a
hydrated staging archive the SQLite index is only ever **read** (via
``archive.index.query_runs``) — never written.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, cast, get_args

from daydream.archive.hydrate import HubUnavailableError
from daydream.training.adjudication.preview import _load_sessions
from daydream.training.adjudication.snapshot import build_canonical_record, snapshot_id
from daydream.training.dispositions import DECISIVE_DISPOSITIONS
from daydream.training.labeler_signals import (
    PerFindingDisposition,
    PerFindingResolution,
    resolution_from_dict,
)

__all__ = ["run_materialize"]

_SESSIONS_OUT_FILENAME = "sessions.jsonl"
_MANIFEST_FILENAME = "preview-manifest.json"

# Disposition written for a conflicted generation's materialized records
# (sessions.jsonl). The operator queue (``queue.build_queue``'s default
# non-decisive set) and the final bundle's sessions.jsonl must route the
# finding to task-only adjudication -- never gold -- and the archive
# `rubric_json` keeps the real decisive disposition for provenance (the
# canonical harvest restores it from the fresh queue). Corpus-v2's gold gate
# (``tiers.classify_tier``) keys solely on disposition/evidence and never
# reads the ``conflicting`` flag, so a decisive disposition here would still
# classify gold; a non-decisive disposition forces ``task-only``.
_CONFLICTED_DISPOSITION = "ambiguous"


def _canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _write_atomic(out_path: Path, payload: str) -> None:
    """Temp-file + ``os.replace`` write, mirroring ``export.write_export_rows``."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    tmp_path.write_text(payload, encoding="utf-8")
    os.replace(tmp_path, out_path)


def _sessions_from_hydrated_stage(index_root: Path) -> tuple[list[dict[str, Any]], str]:
    """Build queue-consumable session records from a hydrated staging archive.

    A hydrated staging archive (``archive.hydrate.run_hydrate_hub``) has no
    ``sessions.jsonl``: its per-finding resolutions live in the SQLite
    index's ``label_observations.rubric_json`` (the canonical dict shape
    emitted by ``Rubric.to_dict``). This adapter joins ``query_runs`` rows
    to their observations into the same session shape ``_load_sessions``
    returns — the observations connection is opened read-only, fail-closed
    on any missing or adjudication-empty run.

    Latest-observation selection (deterministic): a session's rows are
    grouped by the harvester dedup key ``(evidence_sha, labeler_policy_version,
    reply_evidence_digest, labels, has_posterior)``; within an identical key
    the latest ``observed_at`` wins; across distinct keys the winner follows
    the archive's ``_PRECEDENCE_ORDER`` (human-first ``source='human'``, then
    ``observed_at DESC``). The session is **conflicting** only when its
    generations disagree in disposition-relevant content — more than one
    distinct decision-bearing ``labels`` set across the dedup-key groups,
    scoped to non-human rows (``_winning_observation``): the archive appends
    fresh generations with identical labels on policy-version bumps,
    edited-reply digest changes, and label-preserving observation overlays
    (``index.append_label_observation``), so a dedup-key split alone never
    marks a session non-gold, and neither a human override row (authoritative
    under the archive's precedence) nor a non-decisive-only generation (a
    pre-adjudication evolution) does. For a conflicted session the winner
    still supplies the resolutions and every emitted record for that session
    carries ``"conflicting": true`` with the disposition neutralized to
    ``_CONFLICTED_DISPOSITION`` (surfaced non-gold downstream, never merged
    away). A session whose rows carry no materializable per-finding
    resolutions (``rubric_json`` NULL — a human ``daydream label`` row — or
    a legacy labels-only row) and no sanitized per-run trajectory contributes
    no records at all: such sessions are evidence-only (e.g. rows a runbook
    step-3b import admitted from a backup root outside the curation) and must
    not fail the whole curation.

    The index revision is the pinned source commit (the single revision
    directory under ``downloads/``) — a full 40-hex SHA, exactly what the
    publication machinery's pinned-revision resolver accepts.
    """
    if not (index_root / "index.db").is_file():
        raise HubUnavailableError(
            f"hydrated index sessions file not found: {index_root / 'sessions.jsonl'}"
        )
    _raise_on_uncheckpointed_wal(index_root / "index.db")
    rows = _query_runs_readonly(index_root / "index.db")
    if not rows:
        raise HubUnavailableError(f"hydrated index at {index_root} has no runs")
    sessions: list[dict[str, Any]] = []
    for row in rows:
        session_id = str(row["session_id"])
        observations = _label_observations_readonly(index_root / "index.db", session_id)
        if observations:
            winner, conflicting = _winning_observation(observations)
            resolutions: list[dict[str, Any]] | None = None
            session: dict[str, Any]
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
                if isinstance(per_finding, list) and per_finding:
                    resolutions = per_finding
            if resolutions is None:
                # NULL rubric_json (a human-sourced ``daydream label`` row --
                # ``index.update_labels`` appends with no rubric) or a legacy
                # labels-only row (pre-#1095 ``Rubric.to_dict`` emitted only
                # ``per_finding_outcomes``) carries no per-finding semantics to
                # materialize, and the import path (runbook step 3b) appends
                # such rows verbatim — an imported archive therefore reaches
                # this branch with observations present, and failing closed
                # here would brick the session for every later
                # preview/materialize/harvest. Serve the session from the
                # sanitized per-run trajectory instead (the pre-#1095
                # materialization source), exactly like a session with no
                # observation rows at all; a session with no trajectory either
                # has no materializable content at all and contributes no
                # records (evidence-only rows, e.g. sessions an import admitted
                # from a backup root outside the curation), never failing the
                # whole stage over one such session.
                resolutions = _trajectory_resolutions_readonly(index_root, session_id)
                if resolutions is None:
                    continue
            session = {
                "session_id": session_id,
                "trajectory_id": session_id,
                "segment_id": session_id,
                "resolutions": resolutions,
            }
            if conflicting:
                session["conflicting"] = True
        else:
            # Freshly hydrated staging archive: no label_observations history
            # yet (canonical harvest appends the first rows, runbook step 5).
            # Fall back to the sanitized per-run trajectory the hydration gate
            # guarantees (_REQUIRED_SESSION_ARTIFACTS) -- the pre-#1095
            # materialization source -- so the first preview pass over a new
            # stage still works. Once any observation row exists for the
            # session it wins; the two sources are never mixed. A session
            # without either contributes no records (the one-session-fails-all
            # blast radius is reserved for corrupt data, not absence).
            resolutions = _trajectory_resolutions_readonly(index_root, session_id)
            if resolutions is None:
                continue
            session = {
                "session_id": session_id,
                "trajectory_id": session_id,
                "segment_id": session_id,
                "resolutions": resolutions,
            }
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
    return sessions, revisions[0]


def _trajectory_resolutions_readonly(
    index_root: Path, session_id: str
) -> list[dict[str, Any]] | None:
    """Read a session's per-finding resolutions from the sanitized per-run
    trajectory (the pre-#1095 materialization source). Used when the hydrated
    staging archive has no materializable ``label_observations`` history for
    the session: no observation rows yet (freshly hydrated stage; canonical
    harvest appends the first rows), a NULL ``rubric_json`` (human-sourced
    row), or only legacy labels-only rows whose rubric_json carries no
    ``per_finding_resolutions`` (pre-#1095 ``Rubric.to_dict``; such rows are
    appended verbatim by the import path, runbook step 3b).

    Returns ``None`` when the trajectory is absent -- the session has no
    materializable content at all (e.g. a session a runbook step-3b import
    admitted from a backup root outside the curation: DB-only ``runs`` row,
    evidence-only observation rows, no ``runs/<sid>`` files) and contributes
    no records; the caller skips it. Anything *present* but unreadable,
    malformed, or empty still raises ``HubUnavailableError`` naming the
    session -- corrupt data is never silently skipped.
    """
    trajectory_path = index_root / "runs" / session_id / "trajectory.json"
    if not trajectory_path.is_file():
        return None
    try:
        trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HubUnavailableError(
            f"unreadable hydrated trajectory at {trajectory_path}: {exc}"
        ) from exc
    resolutions = trajectory.get("resolutions") if isinstance(trajectory, dict) else None
    if not isinstance(resolutions, list) or not resolutions:
        raise HubUnavailableError(
            f"hydrated trajectory for session {session_id!r} carries no "
            "per-finding resolutions to materialize"
        )
    return resolutions


def _raise_on_uncheckpointed_wal(db_path: Path) -> None:
    """Fail loudly when the hydrated index has an uncheckpointed WAL.

    The archive's writer (``index._get_connection``) runs in persistent WAL
    mode, so a crashed/interrupted writer between commit and close leaves
    committed rows in ``index.db-wal``. The read-only adapters below open with
    ``immutable=1``, which by design skips ``-wal``/``-shm`` entirely — such
    rows would then be silently dropped and preview/materialize would serve
    fewer sessions with no error and no sidecar. A surviving ``index.db-wal``
    is therefore a loud error: the operator must checkpoint/recover the index
    (or let an active writer finish) before the read-only guarantee holds.
    """
    if (db_path.with_name(db_path.name + "-wal")).is_file():
        raise HubUnavailableError(
            f"hydrated index {db_path} has an uncheckpointed WAL "
            f"({db_path.name}-wal): committed rows may live only in the WAL; "
            "checkpoint or recover the index before previewing/materializing"
        )


def _query_runs_readonly(db_path: Path) -> list[dict[str, Any]]:
    """Read all ``runs`` rows over a **read-only** connection (``mode=ro``
    URI — ``query_runs`` goes through ``_get_connection``, which opens
    read-write, runs ``PRAGMA journal_mode=WAL`` against the hydrated
    staging index, and leaves ``-wal``/``-shm`` sidecars behind). Callers
    must reject an uncheckpointed ``index.db-wal`` first
    (``_raise_on_uncheckpointed_wal``): ``immutable=1`` skips it entirely.
    """
    import sqlite3

    conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(r) for r in conn.execute("SELECT * FROM runs").fetchall()]
    finally:
        conn.close()


def _label_observations_readonly(db_path: Path, session_id: str) -> list[dict[str, Any]]:
    """Read one session's ``label_observations`` rows over a **read-only**
    connection (``mode=ro&immutable=1`` URI — never ``_get_connection``,
    which opens read-write and runs WAL pragmas against the hydrated
    staging index; ``immutable=1`` also keeps a WAL-mode db from
    materializing ``-shm``/``-wal`` sidecars on read).
    """
    import sqlite3

    conn = sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(
            "SELECT * FROM label_observations WHERE session_id = ?", (session_id,)
        )
        return [dict(r) for r in cursor.fetchall()]
    finally:
        conn.close()


def _winning_observation(
    observations: list[dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    """Deterministic latest-observation selection (see the module docstring for
    the rule). Returns the winning row and whether the session is conflicting
    (more than one distinct disposition-relevant ``labels`` set across the
    dedup-key groups — see ``index.append_label_observation``; a dedup-key
    split that preserves the labels — policy-version bump, edited-reply
    digest change, label-preserving overlay — is agreeing generations, not a
    conflict).
    """
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
    # Conflict is decided from disposition-relevant content (the archived
    # labels projection), never from the full dedup tuple: two rows agreeing
    # on the disposition but split on evidence_sha / policy version / reply
    # digest / has_posterior are agreeing generations that stay gold-eligible.
    # The comparison is additionally scoped (issue #336): a human override row
    # is authoritative under the archive's precedence (``_PRECEDENCE_ORDER``,
    # ``index.update_labels``' human-wins contract) -- never a disagreeing
    # generation -- so it cannot contribute a distinct label set; and only
    # decision-bearing label sets (labels claiming a decisive
    # ``finding-<disposition>``) count, so a pre-adjudication generation
    # (``[]`` labels, an ``unanswered``-only snapshot) that a later decisive
    # generation resolves is an evolution -- resolved-unanswered -> accepted --
    # not a harvester disagreement, and stays gold-eligible after
    # re-materialization.
    decisive_sets = {
        o.get("labels")
        for o in winners
        if o.get("source") != "human" and _labels_claim_decisive(o.get("labels"))
    }
    return winner, len(decisive_sets) > 1


def _labels_claim_decisive(labels: Any) -> bool:
    """True when the archived labels projection claims at least one decisive
    ``finding-<disposition>`` label (e.g. ``finding-accepted``). Rows store the
    labels column as a JSON array string; non-string / non-list / unparsable
    values claim nothing (a session with no decisive claim cannot disagree
    about a gold disposition). Non-decisive finding labels (``finding-unanswered``)
    and non-finding labels (``posterior``) are not claims.
    """
    if isinstance(labels, str):
        try:
            labels = json.loads(labels)
        except ValueError:
            return False
    if not isinstance(labels, list):
        return False
    for label in labels:
        if not isinstance(label, str):
            continue
        disposition = label[len("finding-"):] if label.startswith("finding-") else None
        if disposition in DECISIVE_DISPOSITIONS:
            return True
    return False


def _resolution_from_row(row: dict[str, Any]) -> PerFindingResolution:
    """Rebuild the #980 semantic-resolution object from a stored resolution entry.

    Delegates to the canonical ``labeler_signals.resolution_from_dict``;
    fail-closed: a stored entry missing a required field raises ``ValueError``
    naming the field and fingerprint — never ``None``-coerced into a record.
    """
    fingerprint = row.get("fingerprint")
    disposition = row.get("disposition")
    evidence_digest = row.get("evidence_digest")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError(f"materialize: stored resolution is missing 'fingerprint': {row!r}")
    if not isinstance(disposition, str) or not disposition:
        raise ValueError(
            f"materialize: resolution for fingerprint {fingerprint!r} is missing 'disposition'"
        )
    if not isinstance(evidence_digest, str) or not evidence_digest:
        raise ValueError(
            f"materialize: resolution for fingerprint {fingerprint!r} is missing 'evidence_digest'"
        )
    if disposition not in get_args(PerFindingDisposition):
        raise ValueError(
            f"materialize: resolution for fingerprint {fingerprint!r} has unknown "
            f"disposition {disposition!r}"
        )
    typed_disposition = cast(PerFindingDisposition, disposition)
    return resolution_from_dict(
        {
            "fingerprint": fingerprint,
            "comment_id": row.get("comment_id"),
            "disposition": typed_disposition,
            "evidence": list(row.get("evidence") or []),
            "evidence_digest": evidence_digest,
        }
    )


def run_materialize(
    index_root: Path,
    out_dir: Path,
    *,
    pin: dict[str, str],
    dry_run: bool = False,
) -> dict[str, Any]:
    """Materialize the preview snapshot for one curation pin.

    Loads the hydrated index's sessions (via ``preview._load_sessions`` —
    raises the ``HydrationError`` family on a missing/unreadable index and
    ``MovingBranchError`` on a symbolic index revision), builds one canonical
    record per finding (every disposition — automatic decisive, human-decisive,
    and non-decisive), and emits:

    - ``out_dir/sessions.jsonl``: canonical-JSON records sorted by
      ``record_id``, written atomically. Identical index + pin ⇒
      byte-identical file (C4).
    - ``out_dir/preview-manifest.json``: all K2 pin components plus the
      content-addressed ``snapshot_id`` and ``index_revision``, canonical JSON.

    The ``snapshot_id`` is derived from the pin + the emitted records'
    evidence digests, so any evidence change yields a new id (AC 8) — a stale
    id is never reused. Missing/empty pin components raise ``ValueError``
    naming the field (propagated from ``snapshot.snapshot_id``).

    ``dry_run=True`` validates everything and returns the summary without
    writing any file.
    """
    if (index_root / _SESSIONS_OUT_FILENAME).is_file():
        sessions, index_revision = _load_sessions(index_root)
    else:
        # Hydrated staging archive: derive the sessions from the SQLite
        # index's label_observations (read-only).
        sessions, index_revision = _sessions_from_hydrated_stage(index_root)

    # Validate the pin before touching its components in the loop body:
    # ``snapshot_id`` raises the documented ValueError naming the missing
    # component, never a KeyError from ``pin["evidence_observed_at"]``.
    pin_id = snapshot_id(pin)

    records: list[dict[str, Any]] = []
    for session in sessions:
        resolutions = session.get("resolutions")
        if not isinstance(resolutions, list):
            continue
        for row in resolutions:
            if not isinstance(row, dict):
                raise ValueError(f"materialize: non-object resolution row in session data: {row!r}")
            resolution = _resolution_from_row(row)
            record = build_canonical_record(
                session,
                resolution,
                evidence_observed_at=pin["evidence_observed_at"],
                as_of=pin.get("as_of"),
            )
            if session.get("conflicting"):
                # The session-level conflict flag rides on every emitted
                # per-finding record so downstream consumers (canonical
                # harvest) can exclude the disposition from decisive labels
                # while the full record — flag included — lands in rubric_json.
                # The winner's decisive disposition is neutralized to
                # ``_CONFLICTED_DISPOSITION``: the operator queue (build_queue's
                # default non-decisive set) and the final bundle's sessions.jsonl
                # then route the finding to task-only adjudication — never gold,
                # one disposition in the bundle — while the canonical harvest
                # restores the real decisive disposition for the archive
                # rubric_json provenance from the freshly re-derived queue.
                record["conflicting"] = True
                record["disposition"] = _CONFLICTED_DISPOSITION
                # The record embeds the session-shape view (``resolutions``)
                # that ``project_findings``/``build_queue`` consume; neutralize
                # its disposition too, or the operator queue would still
                # classify the finding gold (``tiers.classify_tier`` reads the
                # resolution, never the record's top-level disposition).
                for nested in record.get("resolutions") or []:
                    if isinstance(nested, dict):
                        nested["disposition"] = _CONFLICTED_DISPOSITION
            records.append(record)
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

    _write_atomic(
        out_dir / _SESSIONS_OUT_FILENAME,
        "".join(_canonical(r) + "\n" for r in records),
    )
    manifest: dict[str, Any] = dict(pin)
    manifest["snapshot_id"] = id_digest
    manifest["index_revision"] = index_revision
    _write_atomic(out_dir / _MANIFEST_FILENAME, _canonical(manifest))
    return summary
