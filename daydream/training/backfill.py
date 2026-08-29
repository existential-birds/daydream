"""Backfill pass — re-annotate indexed runs and append a fresh generation.

Walks the archive index and, for every PR-linked run, rebuilds the annotation
through the *same* :func:`~daydream.training.harvest.build_annotation` path the
harvest pass uses (imported, never forked), then appends it to
``label_observations`` with the current
:data:`~daydream.training.labeler_versions.LABELER_POLICY_VERSION`. The write
layer's versioned dedup key makes an unchanged re-run a no-op (M18); a policy
or reply-evidence change appends a fresh generation so older ``as_of`` pins
still resolve their original generation. Legacy rows are never mutated or
deleted (M17) — backfill only ever appends.

Failure policy:

* A *benign* PR absence (fork/deleted PR 404, unpushed-SHA 422, detected via
  :func:`~daydream.training.harvest._is_benign_pr_absence`) degrades inside
  ``build_annotation`` to a local-branch rubric whose outcome is ``unknown``;
  the backfill stores that as ``["unknown"]`` — fail closed, never decisive
  (M19/M22).
* Transient per-row failures count in ``summary["errors"]`` and never derail
  subsequent rows (harvest's isolation pattern). ``RateLimitError`` aborts the
  sweep cleanly, preserving the remaining queue for a later re-run.
* Every fallible step (fetch, observation append, report write) propagates real
  errors — a report write failure raises *after* observations are committed, and
  the summary reports what landed.

The optional machine-readable report (M20) is written with ``sort_keys=True``
for deterministic diffing: per-session run-label transitions, disposition
counts, the classifier's parser rule count, the ambiguous-manual-review queue
seed count, bot/self-reply exclusions, PR-state counts, and class balance.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from rich.console import Console

from daydream.archive.index import (
    append_label_observation,
    latest_label_observation,
    query_runs,
)
from daydream.git_ops import GitError, RateLimitError
from daydream.training import labeler_versions
from daydream.training.harvest import (
    _gh_api,
    _is_benign_pr_absence,
    _materialize_base_sha_if_missing,
    _resolve_repo_for_row,
    build_annotation,
)
from daydream.training.reply_classifier import (
    _ACCEPT_RULES,
    _DAYDREAM_AGENT_LOGINS,
    _DISPUTE_RULES,
    _FACTUAL_DISAGREEMENT_RULES,
    _REJECT_RULES,
)
from daydream.ui import create_console, print_warning

__all__ = ["run_backfill"]


def _parser_rule_count() -> int:
    """Size of the reply classifier's exported rule table (M20 report)."""
    return (
        len(_ACCEPT_RULES)
        + len(_REJECT_RULES)
        + len(_DISPUTE_RULES)
        + len(_FACTUAL_DISAGREEMENT_RULES)
    )


def _latest_labels(archive_dir: Path, session_id: str) -> list[str] | None:
    """Latest existing observation's labels for a session, or ``None``.

    Read *before* the append so the report's ``run_label_transitions`` records
    the old winning generation, not the one backfill just wrote. Uses the
    archive's winner projection (human-first precedence, then recency) so a
    human override coexisting with a newer auto row still reads as the human
    generation.
    """
    row = latest_label_observation(archive_dir, session_id)
    if row is None or row["labels"] is None:
        return None
    try:
        labels = json.loads(row["labels"])
    except ValueError:
        return None
    return labels if isinstance(labels, list) else None


def _count_bot_self_reply_exclusions(comments: list[Any]) -> int:
    """Count fetched replies authored by daydream's own accounts (M20 report).

    Scans the raw ``/comments`` payloads captured through the ``gh_api`` seam:
    a thread reply whose author login is one of daydream's agent logins (or is
    flagged ``is_self_reply`` by the API shape) is a self-reply exclusion.
    """
    exclusions = 0
    for comment in comments:
        if not isinstance(comment, dict) or comment.get("in_reply_to_id") is None:
            continue
        if comment.get("is_self_reply"):
            exclusions += 1
            continue
        user = comment.get("user")
        login = user.get("login") if isinstance(user, dict) else None
        if isinstance(login, str) and login in _DAYDREAM_AGENT_LOGINS:
            exclusions += 1
    return exclusions


class _RecordingGH:
    """``gh_api`` proxy that captures ``/comments`` payloads for the report."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.comments: list[Any] = []

    def __call__(self, repo: str, endpoint: str, **kwargs: Any) -> Any:
        response = self._inner(repo, endpoint, **kwargs)
        if endpoint.endswith("/comments") and isinstance(response, list):
            self.comments.extend(response)
        return response


def run_backfill(
    archive_dir: Path,
    *,
    dry_run: bool = False,
    session_filter: str | None = None,
    report_path: Path | None = None,
    valid_at_override: str | None = None,
) -> dict[str, Any]:
    """Re-annotate every PR-linked indexed run and append a fresh generation.

    Args:
        archive_dir: Archive root containing ``index.db``.
        dry_run: Build annotations but suppress writes (and the report's
            post-write transitions stay prospective). Walked PR-linked rows
            still count in ``sessions_reprocessed`` but never in ``skipped``,
            so the summary distinguishes 'nothing changed' from 'dry run
            suppressed writes'.
        session_filter: Optional ``session_id`` prefix restricting the queue.
        report_path: When given, write the M20 machine-readable report JSON
            (``sort_keys=True``) after the observation loop; a write failure
            raises after observations are committed.
        valid_at_override: Explicit valid-time passed through to the
            annotation builder, beating the derived decisive-evidence stamp.

    Returns:
        Summary dict with ``sessions_reprocessed`` (PR-linked rows walked),
        ``appended``, ``skipped`` (deduped no-ops), ``non_pr_skipped``,
        ``errors``, and ``aborted``.
    """
    if not archive_dir.exists():
        msg = f"archive_dir does not exist: {archive_dir}"
        raise FileNotFoundError(msg)

    if session_filter:
        queue = query_runs(
            archive_dir,
            "session_id LIKE ? || '%'",
            (session_filter,),
        )
    else:
        queue = query_runs(archive_dir)
    # Pinned session order for determinism (M18): identical archives must
    # produce identical report and append orderings.
    queue = sorted(queue, key=lambda row: row["session_id"])

    console: Console = create_console()
    recording_gh = _RecordingGH(_gh_api)

    summary: dict[str, Any] = {
        "sessions_reprocessed": 0,
        "appended": 0,
        "skipped": 0,
        "non_pr_skipped": 0,
        "errors": 0,
        "aborted": 0,
    }
    transitions: dict[str, dict[str, Any]] = {}
    disposition_counts: Counter[str] = Counter()
    pr_state_counts: Counter[str] = Counter()
    class_balance: Counter[str] = Counter()
    ambiguous_manual_review = 0
    fetched_repos: set[Path] = set()

    for row in queue:
        if not row.get("pr_repo") or row.get("pr_number") is None:
            summary["non_pr_skipped"] += 1
            continue
        try:
            run_dir = Path(row["archive_path"])
            row_repo_clone = _resolve_repo_for_row(
                row, clone_cache=None, fetched_repos=fetched_repos, console=console
            )
            _materialize_base_sha_if_missing(row, run_dir, repo_clone=row_repo_clone, console=console)
            old_labels = _latest_labels(archive_dir, row["session_id"])
            try:
                payload = build_annotation(
                    row,
                    run_dir=run_dir,
                    archive_dir=archive_dir,
                    gh_api=recording_gh,
                    repo_clone=row_repo_clone or archive_dir,
                    clone_resolved=row_repo_clone is not None,
                    valid_at_override=valid_at_override,
                )
            except RateLimitError:
                raise
            except GitError as exc:
                # Benign absence (deleted repo/PR/comments 404, unpushed-SHA
                # 422) that escaped build_annotation's internal degrade — fail
                # closed (M19/M22): store unknown with the current policy
                # version, never a decisive label. Transient failures re-raise
                # into the per-row isolation handler below.
                if not _is_benign_pr_absence(exc):
                    raise
                print_warning(
                    console,
                    f"backfill: PR evidence absent for session {row['session_id']} "
                    f"({type(exc).__name__}: {exc}); labeling unknown (fail closed)",
                )
                labels = ["unknown"]
                transitions[row["session_id"]] = {
                    "old_labels": old_labels,
                    "new_labels": labels,
                }
                pr_state_counts["unknown"] += 1
                class_balance["unknown"] += 1
                if dry_run:
                    summary["sessions_reprocessed"] += 1
                    continue
                appended = append_label_observation(
                    archive_dir,
                    row["session_id"],
                    labels=labels,
                    pr_state=None,
                    labeler_version=labeler_versions.LABELER_POLICY_VERSION,
                    evidence_sha=row.get("head_sha"),
                )
                summary["appended" if appended else "skipped"] += 1
                summary["sessions_reprocessed"] += 1
                continue
            # Fail closed (M19): an unknown outcome is stored as ["unknown"],
            # never as an empty/decisive label set.
            labels = payload.labels or ["unknown"]
            transitions[row["session_id"]] = {
                "old_labels": old_labels,
                "new_labels": labels,
            }
            rubric = json.loads(payload.rubric_json) if payload.rubric_json else {}
            outcomes = rubric.get("per_finding_outcomes") or []
            for outcome in outcomes:
                disposition_counts[str(outcome)] += 1
                if outcome == "ambiguous":
                    ambiguous_manual_review += 1
            pr_state_counts[str(payload.pr_state)] += 1
            for label in labels:
                class_balance[str(label)] += 1
            if dry_run:
                # Suppressed write, not a deduped no-op: count the row as
                # walked; ``skipped`` stays reserved for deduped no-ops (M18)
                # so the summary distinguishes 'nothing changed' from 'dry run
                # suppressed writes'.
                summary["sessions_reprocessed"] += 1
                continue
            appended = append_label_observation(
                archive_dir,
                row["session_id"],
                labels=labels,
                pr_state=payload.pr_state,
                labeler_version=labeler_versions.LABELER_POLICY_VERSION,
                evidence_sha=payload.evidence_sha,
                rubric_json=payload.rubric_json,
                valid_at=payload.valid_at,
                reward_version=payload.reward_version,
                reward_json=payload.reward_json,
                composite_reward=payload.composite_reward,
                reviewer_logins=payload.reviewer_logins,
                has_posterior=payload.has_posterior,
                reply_classifier_version=payload.reply_classifier_version,
                reply_evidence_digest=payload.reply_evidence_digest,
                source="auto",
            )
            if appended:
                summary["appended"] += 1
            else:
                # Deduped: unchanged evidence + unchanged versions is a no-op (M18).
                summary["skipped"] += 1
            summary["sessions_reprocessed"] += 1
        except RateLimitError:
            summary["aborted"] = 1
            print_warning(
                console,
                "backfill: GitHub rate limit exhausted; aborting cleanly.",
            )
            break
        except Exception as exc:  # noqa: BLE001 - per-row isolation by design
            summary["errors"] += 1
            print_warning(
                console,
                f"backfill: session {row.get('session_id', '<unknown>')} failed: "
                f"{type(exc).__name__}: {exc}",
            )
            continue

    if report_path is not None:
        report = {
            "dry_run": dry_run,
            "run_label_transitions": transitions,
            "disposition_counts": dict(sorted(disposition_counts.items())),
            "parser_rule_count": _parser_rule_count(),
            "ambiguous_manual_review": ambiguous_manual_review,
            "bot_self_reply_exclusions": _count_bot_self_reply_exclusions(recording_gh.comments),
            "pr_state_counts": dict(sorted(pr_state_counts.items())),
            "class_balance": dict(sorted(class_balance.items())),
            "summary": dict(summary),
        }
        report_path.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    return summary
