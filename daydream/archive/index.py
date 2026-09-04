"""SQLite index for cross-project querying of archived daydream runs.

Manages a SQLite database at ``~/.daydream/archive/index.db`` that indexes
all archived runs by their manifest metadata. The schema is created
idempotently on every connection open, so the database is self-bootstrapping.

Exports:
    SCHEMA_VERSION: Current schema version integer.
    upsert_run: Insert or replace a run from a Manifest.
    update_labels: Update outcome labels for a session (supports prefix matching).
    query_runs: Query runs with optional WHERE clause.
    count_runs: Count rows matching an optional WHERE clause.
    append_label_observation: Append a row to the immutable bitemporal
        label_observations history (``observed_at`` transaction time,
        ``valid_at`` valid time, reward columns, plus ``reviewer_logins`` and
        the ``has_posterior`` population discriminator) and refresh the
        denormalized runs cache (including the ``has_posterior`` mirror).
    latest_label_observation: Return the highest-precedence (human-first, then
        recency) label_observations row for a session, optionally constrained by
        an ``as_of`` cutoff timestamp.
    bulk_latest_label_observations: Return the highest-precedence (human-first,
        then recency) label_observations row for each session in a collection —
        single round-trip alternative to calling ``latest_label_observation`` in
        a loop.
    delete_runs: Delete ``runs`` rows whose ``session_id`` matches any member of
        a collection (exact match, parameterized ``IN``); return the ``int``
        count of rows deleted. ``label_observations`` is untouched.
    reviewer_set_penalty_prior: Pooled mean false-positive penalty over prior
        runs sharing a reviewer (strict ``valid_at`` cutoff), for the posterior
        outcome prior (C4).
    label_observation_history: Return the full label_observations history for
        a session in chronological order.
    label_count_summary: Return label counts for all runs in a single aggregate
        query (replaces N+1 per-session lookups).
    canonical_utc_iso: Convert an ISO-8601 timestamp to the canonical UTC
        spelling this index stores and compares (``+00:00`` suffix).
    normalize_as_of: Validate and canonicalize a user-supplied ``as_of`` pin
        (strict: UTC-only input) for lexical comparison against ``observed_at``.

Timestamp canonicalization contract
-----------------------------------

The bitemporal columns are TEXT and every cutoff (``observed_at <= as_of``,
``valid_at < before_valid_at``) is a lexical string comparison, which matches
chronological order only when both sides share one spelling. The canonical
spelling is ``datetime.isoformat()`` in UTC — ``YYYY-MM-DDTHH:MM:SS[.ffffff]+00:00``
(fractional seconds absent or exactly six digits, never a ``Z`` suffix).

- ``observed_at`` has a single writer (:func:`append_label_observation` stamps
  ``datetime.now(timezone.utc).isoformat()``), so the stored column is uniformly
  canonical and ``observed_at <= as_of`` / ``ORDER BY observed_at`` are safe once
  ``as_of`` is canonical (enforced at its entry boundary via
  :func:`normalize_as_of`).
- ``valid_at`` historically mixed spellings: caller-supplied values (GitHub
  merge timestamps, harvest fallbacks) arrived ``Z``-suffixed while the
  ``None``→``observed_at`` collapse stored ``+00:00``. All rows are now
  canonicalized at write time (:func:`canonical_utc_iso` in
  :func:`append_label_observation`), so the column converges on the canonical
  spelling and the lexical ``valid_at < before_valid_at`` cutoff compares
  chronologically. Rows written by pre-convergence versions may still carry a
  ``Z`` suffix; they are deliberately NOT rewritten or deleted (destructive
  bootstrap migrations are off the table). A stray legacy row sorts after any
  ``+00:00`` string sharing its second prefix, so the reviewer-prior cutoff
  can only over-exclude it (a smaller pool, never posterior leakage); the
  corpus leakage guard parses datetimes and is spelling-immune. A re-harvest
  appends canonical generations that supersede legacy rows in every winner
  projection.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import warnings
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from daydream.archive._schema import (
    _CREATE_INDEXES,
    _CREATE_LABEL_OBSERVATIONS_TABLE,
    _CREATE_TABLE,
    _PRECEDENCE_ORDER,
    _REVIEWER_PENALTY_MAP,
    _UPSERT_SQL,
    SCHEMA_VERSION,
    _migrate_label_observations_schema,
    _migrate_schema,
    _recreate_label_observations_if_stale,
)
from daydream.archive.git_safe import normalize_remote_url
from daydream.archive.known_versions import STALE_LEGACY
from daydream.archive.manifest import Manifest

# Re-export for callers (including tests) that import these names from this module.
__all__ = [
    "SCHEMA_VERSION",
    "_CREATE_TABLE",
    "upsert_run",
    "update_labels",
    "query_runs",
    "count_runs",
    "append_label_observation",
    "latest_label_observation",
    "bulk_latest_label_observations",
    "delete_runs",
    "reviewer_set_penalty_prior",
    "label_observation_history",
    "label_count_summary",
    "pr_attached_label_coverage",
    "set_run_pr_link",
    "canonical_utc_iso",
    "normalize_as_of",
]


def canonical_utc_iso(ts: str) -> str:
    """Return *ts* in the canonical UTC spelling stored by this index.

    Parses any valid ISO-8601 timestamp (``Z`` or numeric offset, any
    sub-second precision) and re-emits ``datetime.isoformat()`` in UTC:
    ``YYYY-MM-DDTHH:MM:SS[.ffffff]+00:00``. Aware non-UTC offsets are
    *converted* to UTC — a data timestamp in a foreign zone is an unambiguous
    instant, so conversion is always chronologically correct. Idempotent for
    already-canonical input.

    Raises:
        ValueError: When *ts* is not parseable ISO-8601, or is naive (no
            offset) — a naive timestamp names no single instant, so it cannot
            be canonicalized.
    """
    dt = datetime.fromisoformat(ts)
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"naive timestamp {ts!r}: an explicit UTC offset is required")
    return dt.astimezone(timezone.utc).isoformat()


def normalize_as_of(value: str) -> str:
    """Validate and canonicalize a user-supplied ``as_of`` pin.

    The single entry-boundary normalizer for ``as_of``: call it once where the
    pin enters the system (``BuildCorpusConfig``); downstream consumers — the
    ``observed_at <= as_of`` SQL cutoffs here and the valid-time leakage guard
    in ``daydream.training.corpus`` — receive the canonical spelling and never
    re-normalize.

    Stricter than :func:`canonical_utc_iso`: a non-UTC offset is *rejected*,
    not converted. An operator writing ``+05:00`` on a reproducibility pin is
    almost certainly thinking in local time; silently shifting the pin five
    hours invites irreproducible corpora, so the input must already be UTC
    (``Z`` or ``+00:00``, any sub-second precision).

    Raises:
        ValueError: When *value* is not parseable ISO-8601, is naive, or
            carries a non-UTC offset.
    """
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError(f"as_of {value!r} is not a valid ISO-8601 timestamp") from None
    if dt.tzinfo is None or dt.utcoffset() != timedelta(0):
        raise ValueError(f"as_of {value!r} must be a UTC timestamp (ending in Z or +00:00)")
    return dt.astimezone(timezone.utc).isoformat()


def _get_connection(archive_dir: Path) -> sqlite3.Connection:
    """Open the index database, creating schema if needed.

    Enables WAL mode for concurrent read access and sets a busy timeout
    to handle contention from parallel daydream runs.

    Returns:
        An open sqlite3.Connection with row_factory set to sqlite3.Row.
    """
    archive_dir.mkdir(parents=True, exist_ok=True)
    db_path = archive_dir / "index.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version != SCHEMA_VERSION:
        conn.execute(_CREATE_TABLE)
        conn.execute(_CREATE_LABEL_OBSERVATIONS_TABLE)
    _recreate_label_observations_if_stale(conn)
    _migrate_label_observations_schema(conn)
    _migrate_schema(conn)
    if version != SCHEMA_VERSION:
        for idx_sql in _CREATE_INDEXES:
            conn.execute(idx_sql)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    conn.commit()
    return conn


def _project_daydream(daydream: Any) -> dict[str, Any]:
    """Project the ``manifest.daydream`` provenance onto per-column values.

    Collapses the guarded-ternary projection ladder into one place so the five
    ``daydream_*`` projections stay in sync with the ``_UPSERT_SQL`` column
    list. ``None`` (no executable provenance captured) projects every field to
    ``None``; ``daydream_dirty`` is stored as an int only when it is a real
    bool so a sentinel (non-bool) dirty state persists as ``NULL``.
    """
    if daydream is None:
        return {
            "daydream_version": None,
            "daydream_install_source": None,
            "daydream_commit": None,
            "daydream_dirty": None,
            "daydream_container_digest": None,
        }
    dirty = daydream.dirty
    return {
        "daydream_version": daydream.version,
        "daydream_install_source": daydream.install_source,
        "daydream_commit": daydream.commit,
        "daydream_dirty": int(dirty) if isinstance(dirty, bool) else None,
        "daydream_container_digest": daydream.container_digest,
    }


def upsert_run(archive_dir: Path, manifest: Manifest) -> None:
    """Insert or replace a run entry from a Manifest.

    Bool fields (review_only, deep) are normalized to integers (0/1)
    for SQLite storage.
    """
    conn = _get_connection(archive_dir)
    daydream = manifest.daydream
    # Defense-in-depth: never persist a credential-bearing remote URL, even if
    # upstream capture bypassed the normalizer. None identity stores None.
    if manifest.remote_url is None:
        # No remote URL to normalize; keep the manifest's slug as-is.
        normalized_slug, normalized_url = manifest.repo_slug, None
    else:
        normalized_slug, normalized_url = normalize_remote_url(manifest.remote_url)
    try:
        conn.execute(
            _UPSERT_SQL,
            {
                "session_id": manifest.session_id,
                "archived_at": manifest.archived_at,
                "status": manifest.status,
                "archive_status": manifest.archive_status,
                "pipeline_status": manifest.pipeline_status,
                "phase_states": json.dumps(manifest.phase_states)
                if manifest.phase_states is not None
                else None,
                **_project_daydream(daydream),
                "run_flow": manifest.run_flow,
                "skill": manifest.skill,
                "model": manifest.model,
                "backend": manifest.backend,
                "review_backend": manifest.review_backend,
                "fix_backend": manifest.fix_backend,
                "test_backend": manifest.test_backend,
                "per_stack_review_backend": manifest.per_stack_review_backend,
                "per_stack_review_model": manifest.per_stack_review_model,
                "review_only": int(manifest.review_only),
                "deep": int(manifest.deep),
                "remote_url": normalized_url,
                "repo_slug": normalized_slug,
                "source_path": manifest.source_path,
                "branch": manifest.branch,
                "base_branch": manifest.base_branch,
                "head_sha": manifest.head_sha,
                "base_sha": manifest.base_sha,
                "changed_files": json.dumps(manifest.changed_files),
                "pr_number": manifest.pr_number,
                "pr_repo": manifest.pr_repo,
                "total_cost_usd": manifest.total_cost_usd,
                "total_findings": manifest.total_findings,
                "grounding_rate": manifest.grounding_rate,
                "coverage_ratio": manifest.coverage_ratio,
                "cost_per_finding_usd": manifest.cost_per_finding_usd,
                "wall_clock_seconds": manifest.wall_clock_seconds,
                "erosion": manifest.erosion,
                "verbosity": manifest.verbosity,
                "location_in_hunk_rate": manifest.location_in_hunk_rate,
                "shipped_duplicate_pairs": manifest.shipped_duplicate_pairs,
                "fix_quality_gate": json.dumps(manifest.fix_quality_gate)
                if manifest.fix_quality_gate is not None
                else None,
                "recommended_patch_capture": manifest.recommended_patch_capture,
                "total_prompt_tokens": manifest.total_prompt_tokens,
                "total_completion_tokens": manifest.total_completion_tokens,
                "total_cached_tokens": manifest.total_cached_tokens,
                "outcome_labels": manifest.outcome_labels,
                "labeled_at": manifest.labeled_at,
                "composite_reward": manifest.composite_reward,
                "archive_path": manifest.archive_path,
                "schema_version": SCHEMA_VERSION,
                "profile_schema_version": manifest.profile_schema_version,
                "profile_name": manifest.profile_name,
                "profile_source_kind": manifest.profile_source_kind,
                "profile_digest": manifest.profile_digest,
            },
        )
        conn.commit()
    finally:
        conn.close()


def append_label_observation(
    archive_dir: Path,
    session_id: str,
    *,
    labels: list[str],
    pr_state: str | None,
    labeler_version: str,
    evidence_sha: str | None,
    rubric_json: str | None = None,
    valid_at: str | None = None,
    reward_version: str | None = None,
    reward_json: str | None = None,
    composite_reward: float | None = None,
    reviewer_logins: list[str] | None = None,
    has_posterior: bool = False,
    source: str = "auto",
    reply_classifier_version: str | None = None,
    reply_evidence_digest: str | None = None,
    labeler_policy_version: str | None = None,
    legacy: str = "auto",
    observed_at: str | None = None,
) -> bool:
    """Append a row to the immutable ``label_observations`` history.

    Writes a single ``(session_id, observed_at)`` row capturing the current
    label decision plus the bitemporal valid time and reward breakdown, and in
    the same transaction refreshes the denormalized
    ``runs.outcome_labels`` / ``runs.labeled_at`` / ``runs.rubric_json`` /
    ``runs.composite_reward`` / ``runs.has_posterior`` cache.

    The cache is recomputed from the *winning* observation under the
    precedence projection (human-first, then most recent) — **not** necessarily
    the row just inserted. A newer automated append therefore cannot dethrone an
    existing human label in the denormalized cache.

    Args:
        archive_dir: Path to the archive root.
        session_id: Full session UUID — must already exist in ``runs``.
        labels: List of label strings; serialised as a JSON array.
        pr_state: One of ``open``/``merged``/``closed``/``reverted`` or
            ``None`` when not applicable (e.g. local-branch runs).
        labeler_version: Free-form version tag of the labeler that produced
            this observation (e.g. ``2026.05.22`` for an automated rubric, or
            ``human`` for a maintainer override).
        evidence_sha: Optional commit SHA / artifact hash that grounds the
            decision; ``None`` when no concrete evidence applies.
        rubric_json: Optional JSON-serialised rubric (``Rubric.to_dict()``).
        valid_at: ISO 8601 valid time — when the outcome the annotation
            describes became true (e.g. a PR merge timestamp). Canonicalized
            via :func:`canonical_utc_iso` before storage so the column
            converges on one spelling regardless of the caller's (GitHub emits
            ``Z``; the collapse path emits ``+00:00``). ``None`` for
            non-PR/local runs, in which case it collapses to ``observed_at``
            so an ``as_of``-pinned corpus never spuriously drops the run (Q2).
        reward_version: Version tag of the reward reducer that produced
            ``reward_json`` (``RewardBreakdown.reward_version``); ``None`` when
            no reward was scored.
        reward_json: Full ``RewardBreakdown.to_dict()`` serialised as JSON so a
            corpus re-projection has every axis; ``None`` when unscored.
        composite_reward: The cached composite reward scalar. Persisted on the
            ``label_observations`` row (so each annotation generation is
            self-describing) and mirrored onto ``runs.composite_reward`` for
            SQL thresholding; ``None`` when uncomputable.
        reviewer_logins: Human GitHub accounts whose review/reply outcomes
            seeded the posterior axis. Serialised as a JSON array on the
            ``label_observations`` row; ``None`` (stored as SQL ``NULL``) for
            non-PR/local runs with no reviewer set.
        has_posterior: Population discriminator. ``True`` when the row carries a
            ``PosteriorBreakdown`` (a mapped PR-outcome label was scored).
            Coerced to ``int`` and written to ``label_observations.has_posterior``
            and mirrored onto ``runs.has_posterior`` so SQL consumers can split
            labeled/unlabeled populations without parsing ``reward_json``.
        reply_classifier_version: Version of the reply classifier that produced
            the per-finding dispositions (version axis, M13); persisted on the
            row so each annotation generation is self-describing.
        reply_evidence_digest: Stable digest over the combined reply evidence
            (versioned dedup input, M14); persisted and compared in the auto
            dedup key so an edited reply appends a new generation.
        labeler_policy_version: Versioned policy axis (M13); persisted on the
            row and part of the auto dedup key so a policy bump appends a new
            generation. ``None`` (the default, keeping pre-policy callers
            unchanged) mirrors ``labeler_version`` into the policy column; the
            ``STALE_LEGACY`` sentinel — the inventory marker for legacy-schema
            source rows with no policy axis — is stored as SQL ``NULL`` with
            the ``legacy`` marker stamped, the representation the corpus gold
            gate (``labeler_policy_version IS NOT NULL``) already treats as
            non-gold.
        legacy: Legacy marker for the row — ``"auto"`` (the default) or
            ``"legacy"`` (a row whose policy axis predates
            ``label_observations`` versioning; never gold-eligible). Mirrors
            the schema's legacy stamping so imported legacy rows persist the
            same representation a migrated in-place row has.
        source: Provenance of this observation — ``"auto"`` (automated rubric
            labeler; the default that keeps existing harvest callers
            unchanged) or ``"human"`` (operator override). Human-sourced rows
            take precedence over automated ones in every projection regardless
            of timing, which is why the cache is written from the winning row
            rather than the inserted one.

    Returns:
        ``True`` when a new observation row was inserted; ``False`` when the
        append was a deduped no-op. Dedup is **auto-only**: an automated
        (``source="auto"``) append matching the latest existing *auto*
        observation for the session on the versioned evidence tuple
        ``(evidence_sha, labeler_policy_version, reply_evidence_digest,
        labels, has_posterior, reward_version)`` is
        skipped without inserting and without touching the cache. A
        policy-version bump, a reward-version bump, or an edited-reply digest
        change on otherwise-identical evidence appends
        a fresh generation. A ``None`` digest is not coerced to ``""`` — ``None``
        never equals a present digest, so digest-less legacy callers dedupe on
        the remaining tuple exactly as before (deliberate default). Human
        (``source != "auto"``) appends are never deduped by the evidence
        tuple; only a byte-identical row already occupying the same
        ``(session_id, observed_at)`` primary key is a no-op re-import (so a
        re-merged source never grows microsecond-shifted duplicates).

        An explicit ``observed_at`` (aware ISO-8601) is preserved bitemporally
        as the row's observation time instead of the wall clock; ``None``
        keeps the existing now() behavior.

    Raises:
        ValueError: When ``session_id`` is not present in the ``runs`` table,
            when ``valid_at`` is not a parseable aware ISO-8601 timestamp, or
            when ``observed_at`` is set and is not a parseable aware ISO-8601
            timestamp.
    """
    if valid_at is not None:
        valid_at = canonical_utc_iso(valid_at)
    if observed_at is not None:
        # Bitemporal preservation: an explicit data timestamp (e.g. imported
        # from a surviving local archive) is stored in place of the wall clock.
        # Fails closed on non-ISO-8601 or naive input before any write.
        try:
            parsed = datetime.fromisoformat(observed_at)
        except ValueError:
            raise ValueError(
                f"observed_at {observed_at!r} is not a parseable ISO-8601 timestamp"
            ) from None
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError(
                f"observed_at {observed_at!r} is naive: an explicit UTC offset is required"
            )
        observed_dt = parsed.astimezone(timezone.utc)
    else:
        observed_dt = datetime.now(timezone.utc)
    # Policy axis resolution: versioned callers (the import merge) pass the
    # source's labeler_policy_version verbatim; callers that predate the axis
    # leave it None and the free-form labeler_version mirrors into the policy
    # column (inherited behavior). The STALE_LEGACY sentinel — the inventory-
    # time marker for a legacy-schema row with no policy axis — is stored as
    # the canonical legacy representation: NULL policy + legacy='legacy', so
    # the corpus gold gate (which rejects rows by labeler_policy_version IS
    # NULL) can never admit a row the importer's version gate excluded.
    if labeler_policy_version == STALE_LEGACY:
        labeler_policy_version = None
        legacy = "legacy"
    elif labeler_policy_version is None:
        labeler_policy_version = labeler_version
    labels_json = json.dumps(labels)
    reviewer_logins_json = json.dumps(reviewer_logins) if reviewer_logins is not None else None
    has_posterior_int = int(has_posterior)
    conn = _get_connection(archive_dir)
    try:
        cursor = conn.execute(
            "SELECT session_id FROM runs WHERE session_id = ?",
            (session_id,),
        )
        if cursor.fetchone() is None:
            msg = f"Unknown session {session_id!r}"
            raise ValueError(msg)
        # Idempotency: an automated re-score with identical evidence is a no-op.
        # Compare against the latest *auto* row specifically so a human override
        # appended in between cannot mask a genuine automated re-score.
        #
        # The dedup key is the versioned evidence tuple (M14):
        # ``(evidence_sha, labeler_policy_version, reply_evidence_digest, labels,
        # has_posterior, reward_version)``. A policy-version bump, a
        # reward-version bump, or an edited-reply digest change therefore
        # appends a new generation rather than deduping. A
        # ``None`` digest is deliberately NOT coerced to ``""``: ``None`` never
        # equals a present digest, so digest-less legacy callers dedupe on the
        # remaining tuple exactly as before.
        if source == "auto":
            latest_auto = conn.execute(
                "SELECT evidence_sha, labeler_policy_version, reply_evidence_digest, labels, has_posterior, "
                "reward_version "
                "FROM label_observations "
                "WHERE session_id = ? AND source = 'auto' "
                "ORDER BY observed_at DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            if (
                latest_auto is not None
                and latest_auto["evidence_sha"] == evidence_sha
                and latest_auto["labeler_policy_version"] == labeler_policy_version
                and latest_auto["reply_evidence_digest"] == reply_evidence_digest
                and latest_auto["reward_version"] == reward_version
                and latest_auto["labels"] == labels_json
                # Population membership varies independently of the label (a
                # local_branch outcome is labeled but not posterior evidence).
                and latest_auto["has_posterior"] == has_posterior_int
            ):
                return False
        # Bump observed_at by a microsecond and retry on a same-microsecond
        # primary-key collision so the column stays a clean ISO 8601 timestamp.
        while True:
            observed_at = observed_dt.isoformat()
            valid_at_value = valid_at if valid_at is not None else observed_at
            try:
                conn.execute(
                    "INSERT INTO label_observations "
                    "(session_id, observed_at, labels, pr_state, labeler_version, evidence_sha, rubric_json, "
                    "valid_at, reward_version, reward_json, composite_reward, reviewer_logins, has_posterior, source, "
                    "labeler_policy_version, reply_classifier_version, reply_evidence_digest, legacy) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        session_id,
                        observed_at,
                        labels_json,
                        pr_state,
                        labeler_version,
                        evidence_sha,
                        rubric_json,
                        valid_at_value,
                        reward_version,
                        reward_json,
                        composite_reward,
                        reviewer_logins_json,
                        has_posterior_int,
                        source,
                        labeler_policy_version,
                        reply_classifier_version,
                        reply_evidence_digest,
                        legacy,
                    ),
                )
                break
            except sqlite3.IntegrityError:
                # Primary-key collision on (session_id, observed_at): a
                # byte-identical row already occupying this exact stamp is a
                # no-op re-import — the microsecond bump must never fabricate
                # a duplicate generation for it (idempotent re-merge of a
                # surviving source, human or auto). A genuinely distinct
                # generation at the same stamp keeps the pre-existing bump.
                existing = conn.execute(
                    "SELECT labels, pr_state, labeler_version, evidence_sha, rubric_json, "
                    "valid_at, reward_version, reward_json, composite_reward, reviewer_logins, "
                    "has_posterior, source, labeler_policy_version, reply_classifier_version, "
                    "reply_evidence_digest, legacy "
                    "FROM label_observations WHERE session_id = ? AND observed_at = ?",
                    (session_id, observed_at),
                ).fetchone()
                if existing is not None and tuple(existing) == (
                    labels_json,
                    pr_state,
                    labeler_version,
                    evidence_sha,
                    rubric_json,
                    valid_at_value,
                    reward_version,
                    reward_json,
                    composite_reward,
                    reviewer_logins_json,
                    has_posterior_int,
                    source,
                    labeler_policy_version,
                    reply_classifier_version,
                    reply_evidence_digest,
                    legacy,
                ):
                    return False
                observed_dt += timedelta(microseconds=1)
        # Recompute the winning observation (human-first, then recency) so the
        # denormalized runs cache mirrors the precedence projection — not
        # necessarily the row just inserted (a newer auto must not dethrone a
        # human label).
        winner = conn.execute(
            f"SELECT labels, observed_at, rubric_json, composite_reward, has_posterior "
            f"FROM label_observations WHERE session_id = ? "
            f"ORDER BY {_PRECEDENCE_ORDER} LIMIT 1",
            (session_id,),
        ).fetchone()
        conn.execute(
            "UPDATE runs SET outcome_labels = ?, labeled_at = ?, rubric_json = ?, composite_reward = ?, "
            "has_posterior = ? "
            "WHERE session_id = ?",
            (
                winner["labels"],
                winner["observed_at"],
                winner["rubric_json"],
                winner["composite_reward"],
                winner["has_posterior"],
                session_id,
            ),
        )
        conn.commit()
        return True
    finally:
        conn.close()


def latest_label_observation(
    archive_dir: Path,
    session_id: str,
    *,
    as_of: str | None = None,
) -> dict[str, Any] | None:
    """Return the highest-precedence (human-first, then most recent) label observation for ``session_id``.

    Human-sourced observations win over automated ones regardless of timing;
    ties broken by recency. When ``as_of`` is provided, the result is the
    highest-precedence observation whose ``observed_at <= as_of`` — enabling
    reproducible corpus pinning.

    Args:
        as_of: Optional ISO 8601 cutoff timestamp in the canonical UTC
            spelling (see :func:`normalize_as_of` — the entry boundary
            normalizes once; this lexical cutoff assumes canonical input).
    """
    cutoff = "AND observed_at <= ? " if as_of is not None else ""
    params: tuple[Any, ...] = (session_id,) if as_of is None else (session_id, as_of)
    conn = _get_connection(archive_dir)
    try:
        cursor = conn.execute(
            f"SELECT * FROM label_observations WHERE session_id = ? "
            f"{cutoff}ORDER BY {_PRECEDENCE_ORDER} LIMIT 1",
            params,
        )
        row = cursor.fetchone()
        return dict(row) if row is not None else None
    finally:
        conn.close()


def bulk_latest_label_observations(
    archive_dir: Path,
    session_ids: list[str],
    *,
    as_of: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Return the highest-precedence (human-first, then most recent) label observation for each session.

    Human-sourced observations win over automated ones regardless of timing;
    ties broken by recency. Fetches all matching rows in a single SQL query
    instead of one query per session, eliminating the N+1 pattern when building
    a corpus.

    When ``as_of`` is provided, only observations whose ``observed_at <= as_of``
    are considered — the same temporal constraint applied by
    :func:`latest_label_observation`.

    Args:
        as_of: Optional ISO 8601 cutoff timestamp in the canonical UTC
            spelling (see :func:`normalize_as_of` — the entry boundary
            normalizes once; this lexical cutoff assumes canonical input).

    Returns:
        Mapping of ``session_id`` → row dict for every session that has at
        least one qualifying observation.  Sessions with no observation are
        absent from the returned dict (callers should treat them as ``None``).
    """
    if not session_ids:
        return {}
    placeholders = ",".join("?" * len(session_ids))
    cutoff = "\n                      AND observed_at <= ?" if as_of is not None else ""
    params = [*session_ids, as_of] if as_of is not None else list(session_ids)
    conn = _get_connection(archive_dir)
    try:
        cursor = conn.execute(
            f"""
                SELECT *
                FROM (
                    SELECT *,
                           ROW_NUMBER() OVER (
                               PARTITION BY session_id
                               ORDER BY {_PRECEDENCE_ORDER}
                           ) AS _rn
                    FROM label_observations
                    WHERE session_id IN ({placeholders}){cutoff}
                )
                WHERE _rn = 1
                """,
            params,
        )
        return {row["session_id"]: dict(row) for row in cursor.fetchall()}
    finally:
        conn.close()


def delete_runs(archive_dir: Path, session_ids: Iterable[object]) -> int:
    """Destructively prune ``runs`` rows for the given session ids.

    Deletes rows from the ``runs`` table and removes the on-disk bundle
    directory each row points at (``archive_path``); the append-only
    ``label_observations`` history is never touched, so label provenance
    survives hydration reruns that prune rejected sessions.

    The bundle removal matters for consistency: ``rebuild_index`` re-upserts
    every directory under ``runs/``, so an index-only deletion would be
    silently resurrected by the next filesystem-driven hydrate/sanitize
    pass.  Only bundles located inside ``<archive_dir>/runs/`` are removed;
    rows pointing elsewhere are pruned from the index but their directories
    are left untouched.

    Every member is coerced via ``str()`` during normalization, so ints,
    UUIDs, and Path-like objects are accepted.

    Session ids missing from the index are silent no-ops. The deletion runs as
    a single ``DELETE ... WHERE session_id IN (?, ...)`` statement, which is
    bounded by SQLite's host-parameter limit; per-stage run counts sit far
    below that limit, so no chunking is performed today.

    Returns:
        The number of rows deleted (``0`` when ``session_ids`` is empty —
        the database is never opened in that case).
    """
    ids = [str(item) for item in session_ids]
    if not ids:
        return 0
    placeholders = ",".join("?" * len(ids))
    conn = _get_connection(archive_dir)
    try:
        cursor = conn.execute(
            f"DELETE FROM runs WHERE session_id IN ({placeholders})",
            ids,
        )
        deleted = int(cursor.rowcount)
        conn.commit()
        _remove_bundles(archive_dir, ids)
        return deleted
    finally:
        conn.close()


def _remove_bundles(archive_dir: Path, ids: list[str]) -> None:
    """Remove on-disk bundles for deleted sessions so rebuild cannot resurrect them.

    Mirrors the sibling hydrate/sanitize contract where ``rebuild_index``
    reflects the on-disk ``runs/`` tree: a surviving bundle directory would
    be re-upserted by the next filesystem-driven pass.  Only bare session-id
    directories under ``<archive_dir>/runs/`` are removed; anything else
    (including traversal-shaped ids or archive paths outside ``runs/``) is
    left untouched.
    """
    runs_root = (archive_dir / "runs").resolve()
    for sid in ids:
        if not sid or Path(sid).name != sid:
            continue
        bundle = runs_root / sid
        if bundle.is_dir():
            shutil.rmtree(bundle, ignore_errors=True)


def reviewer_set_penalty_prior(
    archive_dir: Path,
    logins: list[str],
    *,
    before_valid_at: str,
    exclude_session: str,
    repo_slug: str | None = None,
) -> tuple[float | None, int]:
    """Return the pooled mean penalty over prior runs sharing a reviewer (C4).

    Pools ``label_observations`` rows whose ``reviewer_logins`` JSON intersects
    *logins*, restricted to ``session_id != exclude_session`` and
    ``valid_at < before_valid_at`` (strict). When *repo_slug* is provided the
    pool is further restricted to rows whose parent ``runs.repo_slug`` matches —
    preventing cross-repo reviewer history from inflating or deflating the prior
    (C4 per-repo scoping). One outcome is taken per session (latest
    ``observed_at``); its first label is mapped to a false-positive penalty via
    ``_REVIEWER_PENALTY_MAP`` (``accepted→0.0``, ``contested→0.5``,
    ``rejected→1.0``). The raw pooled mean and count are returned — the ``>=10``
    sufficiency threshold and the ``0.5`` default fallback are the caller's
    responsibility.

    Rows with malformed ``reviewer_logins`` / ``labels`` JSON are skipped with a
    :func:`warnings.warn` (mirroring ``corpus._annotation_reward``) so a single
    bad row never crashes the aggregate.

    Args:
        logins: The current run's reviewer set. Empty → no pool.
        before_valid_at: ISO 8601 strict upper bound on ``valid_at``.
            Canonicalized via :func:`canonical_utc_iso` so the lexical ``<``
            against the (uniformly canonical) stored column compares
            chronologically regardless of the caller's spelling.
        exclude_session: Session id to exclude (the current run).
        repo_slug: When provided, restrict the pool to observations whose
            parent run shares this ``repo_slug`` (joined via ``runs``).
            ``None`` disables per-repo filtering (backward-compatible).

    Returns:
        ``(mean_penalty, count)`` over the pooled sessions, or ``(None, 0)``
        when *logins* is empty or the pool is empty.
    """
    if not logins:
        return None, 0
    # Canonicalize the bound so the lexical < against the canonical stored
    # column stays chronological regardless of the caller's spelling.
    before_valid_at = canonical_utc_iso(before_valid_at)
    login_set = set(logins)
    penalty_map = _REVIEWER_PENALTY_MAP

    # Build an IN-list so SQLite's json_each() can filter reviewer intersection
    # inside the query, avoiding a full-table fetch followed by Python-side
    # isdisjoint() for every archived row.
    placeholders = ",".join("?" * len(logins))

    p = "lo." if repo_slug is not None else ""
    alias = " lo" if repo_slug is not None else ""
    join = "\n                JOIN runs r ON r.session_id = lo.session_id" if repo_slug is not None else ""
    repo_filter = "\n                  AND r.repo_slug = ?" if repo_slug is not None else ""
    params: tuple[Any, ...] = (exclude_session, before_valid_at, *logins)
    if repo_slug is not None:
        params = (*params, repo_slug)
    sql = f"""
            SELECT reviewer_logins, labels
            FROM (
                SELECT {p}reviewer_logins, {p}labels,
                       ROW_NUMBER() OVER (
                           PARTITION BY {p}session_id
                           ORDER BY {p}observed_at DESC
                       ) AS _rn
                FROM label_observations{alias}{join}
                WHERE {p}session_id != ?
                  AND {p}valid_at < ?
                  AND {p}reviewer_logins IS NOT NULL
                  AND EXISTS (
                      SELECT 1 FROM json_each({p}reviewer_logins)
                      WHERE value IN ({placeholders})
                  ){repo_filter}
            )
            WHERE _rn = 1
            """

    conn = _get_connection(archive_dir)
    try:
        cursor = conn.execute(sql, params)
        rows = cursor.fetchall()
    finally:
        conn.close()

    penalties: list[float] = []
    for row in rows:
        raw_logins = row["reviewer_logins"]
        # reviewer_logins IS NOT NULL is enforced in SQL; guard retained for
        # safety in case the column somehow carries an empty string.
        if not raw_logins:
            continue
        try:
            row_logins = json.loads(raw_logins)
        except (json.JSONDecodeError, TypeError) as exc:
            warnings.warn(f"Invalid reviewer_logins payload {raw_logins!r}: {exc}", stacklevel=2)
            continue
        if not isinstance(row_logins, list) or login_set.isdisjoint(row_logins):
            continue
        try:
            row_labels = json.loads(row["labels"])
        except (json.JSONDecodeError, TypeError) as exc:
            warnings.warn(f"Invalid labels payload {row['labels']!r}: {exc}", stacklevel=2)
            continue
        if not isinstance(row_labels, list) or not row_labels:
            continue
        penalty = penalty_map.get(str(row_labels[0]))
        if penalty is None:
            continue
        penalties.append(penalty)

    if not penalties:
        return None, 0
    return sum(penalties) / len(penalties), len(penalties)


def label_observation_history(archive_dir: Path, session_id: str) -> list[dict[str, Any]]:
    """Return the full label history for ``session_id`` in chronological order.

    Returns:
        List of row dicts ordered by ``observed_at`` ascending.
    """
    conn = _get_connection(archive_dir)
    try:
        cursor = conn.execute(
            "SELECT * FROM label_observations WHERE session_id = ? ORDER BY observed_at ASC",
            (session_id,),
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def update_labels(archive_dir: Path, session_id: str, labels: list[str]) -> bool:
    """Update outcome labels for a session, supporting prefix matching.

    Thin wrapper around :func:`append_label_observation` that records a
    **human-sourced** observation (``source="human"``, ``labeler_version="human"``).
    Human labels win over automated ones in every precedence projection and are
    never deduped, so this is the authoritative override surface backing
    ``daydream label``. The session_id can be a prefix (e.g. first 8 chars of
    the UUID). If the prefix matches exactly one row, that row is updated. If it
    matches multiple rows, a ValueError is raised asking for a longer prefix.

    Returns:
        True if a row was updated, False if no matching session was found.

    Raises:
        ValueError: If the prefix matches more than one session.
    """
    conn = _get_connection(archive_dir)
    try:
        cursor = conn.execute(
            "SELECT session_id FROM runs WHERE session_id LIKE ? || '%'",
            (session_id,),
        )
        matches = cursor.fetchall()
    finally:
        conn.close()

    if not matches:
        return False

    if len(matches) > 1:
        matched_ids = [row["session_id"] for row in matches]
        msg = f"Prefix '{session_id}' matches {len(matches)} sessions: {matched_ids}. Provide a longer prefix."
        raise ValueError(msg)

    full_id = matches[0]["session_id"]
    append_label_observation(
        archive_dir,
        full_id,
        labels=labels,
        pr_state=None,
        labeler_version="human",
        evidence_sha=None,
        source="human",
    )
    return True


def set_run_pr_link(archive_dir: Path, session_id: str, pr_number: int, pr_repo: str) -> None:
    """Backfill the PR linkage columns on a run row.

    Used by harvest to durably record a PR resolved for an orphan run (a run
    launched before its PR existed, so ``pr_number`` was frozen as ``None``).
    Persisting the linkage keeps subsequent harvest passes from re-querying
    GitHub for the same row and makes the resolution auditable.

    This is a pure linkage backfill: it touches only the ``pr_number`` and
    ``pr_repo`` columns on the ``runs`` table and never writes to
    ``label_observations`` or any cache column. A zero-row match (no such
    ``session_id``) is a silent no-op; the caller guarantees the row exists.
    """
    conn = _get_connection(archive_dir)
    try:
        conn.execute(
            "UPDATE runs SET pr_number = ?, pr_repo = ? WHERE session_id = ?",
            (pr_number, pr_repo, session_id),
        )
        conn.commit()
    finally:
        conn.close()


def query_runs(archive_dir: Path, where: str = "", params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    """Query the runs index with an optional WHERE clause.

    Args:
        where: Optional SQL WHERE clause (without the ``WHERE`` keyword).
            Example: ``"repo_slug = ? AND status = ?"``.
        params: Parameter tuple to bind to the WHERE clause placeholders.
    """
    conn = _get_connection(archive_dir)
    try:
        sql = "SELECT * FROM runs"
        if where:
            sql += f" WHERE {where}"  # noqa: S608 - caller-supplied SQL fragment with bound params
        cursor = conn.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def pr_attached_label_coverage(
    archive_dir: Path,
    *,
    as_of: str | None = None,
) -> dict[str, float | int]:
    """Return the fraction of PR-attached runs with a decisive automated label.

    "PR-attached" means the run carries a ``pr_number`` (``pr_number IS NOT
    NULL``). A run is "decisive" when its winning label — under the same
    human-first, then-recency precedence projection used by
    :func:`bulk_latest_label_observations` — is one of ``"accepted"``,
    ``"contested"``, or ``"rejected"``. Runs whose winning label is
    ``"unknown"`` (or that have no qualifying observation at all) are not
    decisive.

    This is a pure read; it never writes. An archive with zero PR-attached runs
    yields ``coverage`` ``0.0`` rather than raising ``ZeroDivisionError``.

    Args:
        as_of: Optional ISO 8601 cutoff; threaded through to
            :func:`bulk_latest_label_observations` so only observations whose
            ``observed_at <= as_of`` are considered (reproducible pinning).

    Returns:
        ``{"pr_attached": N, "decisive": M, "coverage": M / N,
        "malformed_labels": K}`` with ``coverage`` as ``0.0`` when ``N == 0``.
    """
    rows = query_runs(archive_dir, where="pr_number IS NOT NULL")
    pr_attached = len(rows)
    if pr_attached == 0:
        return {"pr_attached": 0, "decisive": 0, "coverage": 0.0, "malformed_labels": 0}

    session_ids = [row["session_id"] for row in rows]
    winners = bulk_latest_label_observations(archive_dir, session_ids, as_of=as_of)

    decisive_labels = {"accepted", "contested", "rejected"}
    decisive = 0
    malformed = 0
    for session_id in session_ids:
        observation = winners.get(session_id)
        if observation is None:
            continue
        try:
            labels = json.loads(observation["labels"])
        except (json.JSONDecodeError, TypeError):
            malformed += 1
            continue
        if not isinstance(labels, list):
            malformed += 1
            continue
        if labels and str(labels[0]) in decisive_labels:
            decisive += 1

    return {
        "pr_attached": pr_attached,
        "decisive": decisive,
        "coverage": decisive / pr_attached,
        "malformed_labels": malformed,
    }


def label_count_summary(
    archive_dir: Path,
    as_of: str | None = None,
) -> dict[str, int]:
    """Return label counts for all runs in a single aggregate query.

    For each run in ``runs``, finds the highest-precedence (human-first, then
    most recent) ``label_observations`` row whose ``observed_at <= as_of`` (or
    overall when ``as_of`` is ``None``), extracts the first label, and tallies
    counts.  Runs with no qualifying observation are counted under
    ``"unlabeled"``.

    This replaces the N+1 pattern of calling
    :func:`latest_label_observation` once per run.

    Args:
        as_of: Optional ISO 8601 cutoff timestamp. When ``None``, the
            most recent observation for each session is used regardless of
            ``observed_at``.

    Returns:
        Dict mapping label string → count.  Always includes at least one key
        when the archive is non-empty.
    """
    cutoff = "   WHERE observed_at <= ?" if as_of is not None else ""
    params: tuple[Any, ...] = (as_of,) if as_of is not None else ()
    best_sql = (
        f"SELECT session_id, labels FROM ("
        f"  SELECT session_id, labels, "
        f"         ROW_NUMBER() OVER ("
        f"             PARTITION BY session_id "
        f"             ORDER BY {_PRECEDENCE_ORDER}"
        f"         ) AS _rn "
        f"  FROM label_observations{cutoff}"
        f") WHERE _rn = 1"
    )
    conn = _get_connection(archive_dir)
    try:
        cursor = conn.execute(
            f"SELECT best.labels "  # noqa: S608
            f"FROM runs r "
            f"LEFT JOIN ({best_sql}) best ON r.session_id = best.session_id",
            params,
        )
        counts: dict[str, int] = {}
        for (labels_raw,) in cursor.fetchall():
            label = "unlabeled"
            if labels_raw:
                try:
                    parsed = json.loads(labels_raw) if isinstance(labels_raw, str) else labels_raw
                    if isinstance(parsed, list) and parsed and parsed[0]:
                        label = str(parsed[0])
                except (json.JSONDecodeError, TypeError) as exc:
                    warnings.warn(
                        f"Invalid labels payload {labels_raw!r}: {exc}",
                        stacklevel=2,
                    )
            counts[label] = counts.get(label, 0) + 1
        return counts
    finally:
        conn.close()


def count_runs(archive_dir: Path, where: str = "", params: tuple[Any, ...] = ()) -> int:
    """Return the number of runs matching an optional WHERE clause.

    Uses ``SELECT COUNT(*)`` so no rows are materialised.

    Args:
        where: Optional SQL WHERE clause (without the ``WHERE`` keyword).
        params: Parameter tuple to bind to the WHERE clause placeholders.
    """
    conn = _get_connection(archive_dir)
    try:
        sql = "SELECT COUNT(*) FROM runs"
        if where:
            sql += f" WHERE {where}"  # noqa: S608 - caller-supplied SQL fragment with bound params
        cursor = conn.execute(sql, params)
        return int(cursor.fetchone()[0])
    finally:
        conn.close()
