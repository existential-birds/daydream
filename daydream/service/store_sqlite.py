"""Production transactional store backed by SQLite (stdlib ``sqlite3``).

The controller's durable state machine needs crash-safe persistence across
process restarts *and* correct compare-and-set semantics under concurrent
claimers. This is the production implementation of :class:`ServiceStore`.

Selected/ pinned DB stack
-------------------------
- Client: Python stdlib ``sqlite3`` (bundled SQLite; Python >= 3.12.13).
  Install command: *none* — it ships with CPython. No extra dependency, no
  pip install, no background service; keeps the public core hermetic.
- Migration tool: a tiny versioned schema migrator driven by ``PRAGMA
  user_version`` (SQLite's own metadata slot). ``SCHEMA_VERSION`` is the
  current schema revision; ``_migrate`` advances the file in idempotent
  transactions and sets ``user_version`` so a re-open is a no-op.

Error / idempotency mapping
---------------------------
- ``sqlite3.IntegrityError`` on ``create_job`` is mapped to
  :class:`IdempotencyError` (unique ``idempotency_key``) — never a silent dup.
- ``JobNotFoundError`` is raised when a claim/transaction references a missing
  row (the controller treats this as a stale/superseded job, not a crash).
- ``StateConflictError`` for owned transition mismatches.
- ``sqlite3.OperationalError`` (locked/busy) is surfaced as
  :class:`StoreError` with the driver message, so the controller's retry
  classifier can treat lock contention as transient infra error.
- Writes are serialized through a process-local lock plus ``BEGIN IMMEDIATE``
  transactions; readers do not block writers (WAL).

Hermetic: uses only local files / stdout-free stdlib. No network.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

from daydream.service.store import (
    NON_RECOVERABLE_STATES,
    AttemptRecord,
    ClaimStatus,
    IdempotencyError,
    JobNotFoundError,
    JobRecord,
    RecoverableAttempt,
    ServiceState,
    ServiceStore,
    StateConflictError,
    StoreError,
)

#: Current durable schema revision. Bump when altering the schema; the migrator
#: applies each step idempotently via ``PRAGMA user_version``.
SCHEMA_VERSION = 1


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


class SqliteServiceStore(ServiceStore):
    """Crash-safe, concurrent ServiceStore persisted to a SQLite file.

    Args:
        path: Database file path. Use ``:memory:`` at your own risk — it does
            not survive a process restart, which defeats the store's purpose.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=10000")
        self._lock = threading.RLock()
        self._migrate()

    # ------------------------------------------------------------- migration
    def _migrate(self) -> None:
        with self._conn:  # transaction
            version = self._conn.execute("PRAGMA user_version").fetchone()[0]
            if version < 1:
                self._conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS jobs (
                        job_id        TEXT PRIMARY KEY,
                        idempotency_key TEXT NOT NULL UNIQUE,
                        target_key    TEXT NOT NULL,
                        round         INTEGER NOT NULL,
                        state         TEXT NOT NULL,
                        version       INTEGER NOT NULL,
                        current_attempt_id TEXT,
                        owner         TEXT,
                        lease_expires_at    TEXT,
                        created_at    TEXT NOT NULL,
                        updated_at    TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS attempts (
                        job_id        TEXT NOT NULL,
                        attempt_id    TEXT NOT NULL,
                        owner         TEXT NOT NULL,
                        state         TEXT NOT NULL,
                        execution_ref TEXT NOT NULL,
                        artifact_refs_json TEXT NOT NULL DEFAULT '[]',
                        externalized  INTEGER NOT NULL DEFAULT 0,
                        created_at    TEXT NOT NULL,
                        PRIMARY KEY (job_id, attempt_id),
                        FOREIGN KEY (job_id) REFERENCES jobs (job_id)
                    );
                    CREATE INDEX IF NOT EXISTS idx_attempts_job ON attempts (job_id);
                    """
                )
                self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")

    # ---------------------------------------------------------------- helpers
    def _row_to_job(self, row: sqlite3.Row) -> JobRecord:
        lease_raw = row["lease_expires_at"]
        return JobRecord(
            job_id=row["job_id"],
            idempotency_key=row["idempotency_key"],
            target_key=row["target_key"],
            round=row["round"],
            state=ServiceState(row["state"]),
            version=row["version"],
            current_attempt_id=row["current_attempt_id"],
            owner=row["owner"],
            lease_expires_at=_parse_iso(lease_raw) if lease_raw else None,
            created_at=_parse_iso(row["created_at"]),
            updated_at=_parse_iso(row["updated_at"]),
        )

    def _row_to_attempt(self, row: sqlite3.Row) -> AttemptRecord:
        raw_refs: list[list[str]] = json.loads(row["artifact_refs_json"] or "[]")
        refs: tuple[tuple[str, str], ...] = tuple((a, b) for a, b in raw_refs)
        return AttemptRecord(
            job_id=row["job_id"],
            attempt_id=row["attempt_id"],
            owner=row["owner"],
            state=ServiceState(row["state"]),
            execution_ref=row["execution_ref"],
            artifact_refs=refs,
            externalized=bool(row["externalized"]),
            created_at=_parse_iso(row["created_at"]),
        )

    def _job_not_found(self, job_id: str) -> None:
        raise JobNotFoundError(job_id)

    # --------------------------------------------------------------- public: jobs
    def create_job(self, job: JobRecord) -> JobRecord:
        with self._lock, self._conn:
            existing = self._conn.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job.job_id,)
            ).fetchone()
            if existing is not None:
                if existing["idempotency_key"] != job.idempotency_key:
                    raise IdempotencyError(
                        f"job {job.job_id!r} re-created with a different idempotency key"
                    )
                return self._row_to_job(existing)
            dup = self._conn.execute(
                "SELECT job_id FROM jobs WHERE idempotency_key = ?", (job.idempotency_key,)
            ).fetchone()
            if dup is not None:
                raise IdempotencyError(
                    f"idempotency key {job.idempotency_key!r} already bound to job {dup['job_id']!r}"
                )
            try:
                self._conn.execute(
                    "INSERT INTO jobs (job_id, idempotency_key, target_key, round, state, version,"
                    " created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
                    (
                        job.job_id,
                        job.idempotency_key,
                        job.target_key,
                        job.round,
                        job.state.value,
                        job.version,
                        _iso(job.created_at),
                        _iso(job.updated_at),
                    ),
                )
            except sqlite3.IntegrityError as exc:  # concurrent duplicate create
                raise IdempotencyError(str(exc)) from exc
        return job

    def get_job(self, job_id: str) -> JobRecord | None:
        row = self._conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        return self._row_to_job(row) if row is not None else None

    # ------------------------------------------------------- public: claims
    def claim(
        self,
        job_id: str,
        attempt_id: str,
        *,
        expected: frozenset[ServiceState],
        new_state: ServiceState,
        owner: str,
        execution_ref: str,
        now: datetime,
        ttl_seconds: float,
    ) -> ClaimStatus:
        with self._lock:
            job = self.get_job(job_id)
            if job is None:
                self._job_not_found(job_id)
            assert job is not None
            status = _derive_claim_status(job, attempt_id, expected, new_state, owner, now)
            if status is not ClaimStatus.OK:
                return status
            self._begin()
            try:
                self._conn.execute(
                    "UPDATE jobs SET state = ?, current_attempt_id = ?, owner = ?,"
                    " lease_expires_at = ?, version = version + 1, updated_at = ? WHERE job_id = ?",
                    (
                        new_state.value,
                        attempt_id,
                        owner,
                        _iso(_after(now, ttl_seconds)),
                        _iso(now),
                        job_id,
                    ),
                )
                self._conn.execute(
                    "INSERT OR IGNORE INTO attempts (job_id, attempt_id, owner, state, execution_ref,"
                    " created_at) VALUES (?,?,?,?,?,?)",
                    (job_id, attempt_id, owner, new_state.value, execution_ref, _iso(now)),
                )
                self._conn.execute(
                    "UPDATE attempts SET state = ?, owner = ? WHERE job_id = ? AND attempt_id = ?",
                    (new_state.value, owner, job_id, attempt_id),
                )
                self._commit()
            except sqlite3.IntegrityError as exc:
                self._rollback()
                raise StoreError(f"claim failed: {exc}") from exc
            except sqlite3.OperationalError as exc:
                self._rollback()
                raise StoreError(f"claim lock contended: {exc}") from exc
            return ClaimStatus.OK

    def update_state(
        self,
        job_id: str,
        from_state: ServiceState,
        to_state: ServiceState,
        *,
        attempt_id: str,
        owner: str,
        now: datetime,
    ) -> None:
        with self._lock:
            job = self.get_job(job_id)
            if job is None:
                self._job_not_found(job_id)
            assert job is not None
            if job.state != from_state or job.current_attempt_id != attempt_id or job.owner != owner:
                raise StateConflictError(
                    f"job {job_id} not in {from_state.value}/{attempt_id}/{owner} "
                    f"(state={job.state.value}, attempt={job.current_attempt_id}, owner={job.owner})"
                )
            self._begin()
            try:
                self._conn.execute(
                    "UPDATE jobs SET state = ?, version = version + 1, updated_at = ? WHERE job_id = ?",
                    (to_state.value, _iso(now), job_id),
                )
                self._conn.execute(
                    "UPDATE attempts SET state = ? WHERE job_id = ? AND attempt_id = ?",
                    (to_state.value, job_id, attempt_id),
                )
                self._commit()
            except sqlite3.OperationalError as exc:
                self._rollback()
                raise StoreError(f"update_state lock contended: {exc}") from exc

    # --------------------------------------------------------- public: hb
    def heartbeat(
        self,
        job_id: str,
        attempt_id: str,
        *,
        owner: str,
        now: datetime,
        ttl_seconds: float,
    ) -> bool:
        with self._lock:
            job = self.get_job(job_id)
            if job is None:
                raise JobNotFoundError(job_id)
            if job.owner != owner or job.current_attempt_id != attempt_id:
                return False
            if job.lease_expires_at is None or job.lease_expires_at <= now:
                return False
            self._begin()
            try:
                self._conn.execute(
                    "UPDATE jobs SET lease_expires_at = ?, version = version + 1, updated_at = ?"
                    " WHERE job_id = ? AND owner = ? AND current_attempt_id = ?",
                    (_iso(_after(now, ttl_seconds)), _iso(now), job_id, owner, attempt_id),
                )
                self._commit()
            except sqlite3.OperationalError as exc:
                self._rollback()
                raise StoreError(f"heartbeat lock contended: {exc}") from exc
            return True

    # --------------------------------------------------------- public: artifacts
    def bind_artifacts(
        self,
        job_id: str,
        attempt_id: str,
        *,
        owner: str,
        artifact_refs: Mapping[str, str],
    ) -> None:
        with self._lock:
            self._begin()
            try:
                row = self._conn.execute(
                    "SELECT state, owner FROM attempts WHERE job_id = ? AND attempt_id = ?",
                    (job_id, attempt_id),
                ).fetchone()
                if row is None:
                    self._rollback()
                    raise JobNotFoundError(f"attempt {attempt_id} for job {job_id}")
                if row["owner"] != owner:
                    self._rollback()
                    raise StateConflictError(f"attempt {attempt_id} for job {job_id} not owned by {owner}")
                payload = json.dumps(sorted(artifact_refs.items()))
                self._conn.execute(
                    "UPDATE attempts SET artifact_refs_json = ? WHERE job_id = ? AND attempt_id = ?",
                    (payload, job_id, attempt_id),
                )
                self._commit()
            except sqlite3.OperationalError as exc:
                self._rollback()
                raise StoreError(f"bind_artifacts lock contended: {exc}") from exc

    def externalize(self, job_id: str, attempt_id: str, *, owner: str) -> None:
        with self._lock:
            self._begin()
            try:
                row = self._conn.execute(
                    "SELECT owner FROM attempts WHERE job_id = ? AND attempt_id = ?", (job_id, attempt_id)
                ).fetchone()
                if row is None:
                    self._rollback()
                    raise JobNotFoundError(f"attempt {attempt_id} for job {job_id}")
                if row["owner"] != owner:
                    self._rollback()
                    raise StateConflictError(f"attempt {attempt_id} for job {job_id} not owned by {owner}")
                self._conn.execute(
                    "UPDATE attempts SET externalized = 1 WHERE job_id = ? AND attempt_id = ?",
                    (job_id, attempt_id),
                )
                self._commit()
            except sqlite3.OperationalError as exc:
                self._rollback()
                raise StoreError(f"externalize lock contended: {exc}") from exc

    # -------------------------------------------------------------- public: reads
    def execution_ref(self, job_id: str, attempt_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT execution_ref FROM attempts WHERE job_id = ? AND attempt_id = ?", (job_id, attempt_id)
        ).fetchone()
        return row["execution_ref"] if row is not None else None

    def attempt_history(self, job_id: str) -> list[AttemptRecord]:
        rows = self._conn.execute(
            "SELECT * FROM attempts WHERE job_id = ? ORDER BY rowid", (job_id,)
        ).fetchall()
        return [self._row_to_attempt(r) for r in rows]

    def recoverable(self, *, now: datetime) -> list[RecoverableAttempt]:
        states = [s.value for s in NON_RECOVERABLE_STATES]
        placeholders = ",".join("?" for _ in states)
        rows = self._conn.execute(
            f"SELECT * FROM jobs WHERE state NOT IN ({placeholders})", states
        ).fetchall()
        out: list[RecoverableAttempt] = []
        for row in rows:
            job = self._row_to_job(row)
            ref: str | None = None
            if job.current_attempt_id is not None:
                arow = self._conn.execute(
                    "SELECT execution_ref FROM attempts WHERE job_id = ? AND attempt_id = ?",
                    (job.job_id, job.current_attempt_id),
                ).fetchone()
                ref = arow["execution_ref"] if arow is not None else None
            lease_expired = job.lease_expires_at is not None and job.lease_expires_at <= now
            out.append(
                RecoverableAttempt(
                    job_id=job.job_id,
                    attempt_id=job.current_attempt_id,
                    state=job.state,
                    execution_ref=ref,
                    owner=job.owner,
                    lease_expired=lease_expired,
                )
            )
        return out

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.ProgrammingError:
                pass

    # ------------------------------------------------------------- tx helpers
    def _begin(self) -> None:
        self._conn.execute("BEGIN IMMEDIATE")

    def _commit(self) -> None:
        self._conn.commit()

    def _rollback(self) -> None:
        self._conn.rollback()


def _derive_claim_status(
    job: JobRecord,
    attempt_id: str,
    expected: frozenset[ServiceState],
    new_state: ServiceState,
    owner: str,
    now: datetime,
) -> ClaimStatus:
    """Shared claim-outcome predicate (kept in lockfile parity with store_memory)."""
    if job.state == new_state and job.current_attempt_id == attempt_id and job.owner == owner:
        return ClaimStatus.OK
    if job.state not in expected:
        return ClaimStatus.CONFLICT
    if job.owner is not None and job.owner != owner and job.lease_expires_at is not None:
        if job.lease_expires_at > now:
            return ClaimStatus.LEASED
    return ClaimStatus.OK


def _after(now: datetime, ttl_seconds: float) -> datetime:
    from datetime import timedelta

    return now + timedelta(seconds=ttl_seconds)
